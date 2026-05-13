# 04_workspace/run.py — 技术文档

## 文件调用关系图

```
04_workspace/run.py
  ├─ common/model_loader.py    ← 加载 MJCF 模型、注入运行时元素
  ├─ common/ik_solver.py        ← 逆运动学求解、轨迹规划
  ├─ common/motion.py           ← 夹爪限位检测、即时指令
  ├─ common/force_sensor.py     ← 六维力/力矩传感器读取与可视化
  ├─ common/camera.py           ← 腕部 RGB 相机渲染窗口
  └─ synriard (外部库)          ← get_model_path() 定位 MJCF 文件
```

---

## 1. 入口文件：`examples/04_workspace/run.py`

### 1.1 总体架构

脚本分为 **离线准备** 和 **在线循环** 两个阶段：

```
离线准备（一次性）:
  load_and_inject() → 构建 MuJoCo 模型
  build_gripper_limits() → 检测夹爪开/闭限位
  build_ik_plan() → 预计算 4 个关键路径点的 IK 解
  compute_workspace() → 3D 网格采样可达空间

在线循环（每帧）:
  1. demo_tick()       → 状态机推进
  2. arm_ctrl.step()   → 手臂余弦插值一步
  3. gripper_ctrl.step() → 夹爪速率限制一步
  4. [5× mj_step]      → 物理仿真子步 (250 Hz)
  5. 渲染              → 力曲线/关节面板/RGB 相机/viewer 同步
  6. 帧计时            → spin-wait 保持 50 Hz
```

### 1.2 定时常量

| 常量 | 值 | 含义 |
|---|---|---|
| `TARGET_DISPLAY_HZ` | 50 | 显示刷新率 |
| `PHYSICS_SUBSTEPS` | 5 | 每帧物理步数 → 有效 250 Hz |
| `FRAME_DT` | 0.020 s | 帧预算 |
| `FT_EVERY_N` | 2 | 力曲线更新间隔 → 25 Hz |
| `JOINT_EVERY_N` | 4 | 关节面板更新 → 12.5 Hz |
| `CAMERA_EVERY_N` | 2 | RGB 相机更新 → 25 Hz |
| `SPEED_NORMAL` | 1.4 | 正常速度 |
| `SPEED_SLOW` | 0.80 | 慢速（下降抓取） |
| `WORKSPACE_MAX_SPHERES` | 250 | 可达球体渲染上限 |

### 1.3 Windows 高精度定时器

```python
ctypes.windll.winmm.timeBeginPeriod(1)
```

Windows 默认 `time.sleep()` 精度约 15.6 ms。调用此 API 将系统定时器分辨率提升到 1 ms，使帧计时 spin-wait 能够精确控制 20 ms 帧预算。

### 1.4 SmoothArmController — 非阻塞手臂运动

**设计目标**：替代原始阻塞式 `move_arm_to_angles`（每个动作在函数内部循环直到完成，阻断主循环），实现逐帧增量插值。

**核心状态**：

| 字段 | 类型 | 作用 |
|---|---|---|
| `_start` | `ndarray[6]` | 插值起点（当前实际关节角） |
| `_target` | `ndarray[6]` | 插值终点（目标关节角） |
| `_progress` | `float` | 插值进度 0.0 → 1.0 |
| `_total_frames` | `int` | 总插值帧数 |
| `_done` | `bool` | 运动是否完成 |

**帧数公式**（line 112-116）：

```
原始阻塞版: steps = max(int(max_diff / (speed × 0.008)) + 1, 80)
本版适配:   frames = max(int(max_diff / (speed × 0.008 × SUBSTEPS)) + 1, 80 / SUBSTEPS)
           = max(int(max_diff / (speed × 0.04)) + 1, 16)
```

推导过程：原始版每步调用 1 次 `mj_step`，本版每帧调用 `PHYSICS_SUBSTEPS=5` 次。为保持等量物理仿真时间，帧数需为原始步数的 1/5。

**插值函数**（line 132）：

```
t = 0.5 − 0.5 × cos(π × progress)    # 余弦平滑：t=0 和 t=1 处导数为 0
angles[i] = start[i] + (target[i] − start[i]) × t
```

**完成后的保持机制**（line 123-126）：插值完成后 `step()` 不做 `return`，而是**每帧持续写入 `_target` 到 `data.ctrl`**。这使得 PD 位置执行器始终有一个稳定的参考信号，从而避免残余动量引起的震荡。

### 1.5 SmoothGripperController — 非阻塞夹爪控制

**与手臂控制器的关键区别**：
- 支持**力阈值停止**：当执行器力超过阈值时提前终止闭合
- 闭合/张开使用**线性插值**（非余弦），因为行程短
- 完成时重放 `_last_ctrl` 字典中的**最后实际位置**，而非始终写入终点（避免压碎物体）

**力检测防误触**（line 221）：

```python
if self._mode == "close" and self._progress > 0.30 and max_force >= force_threshold:
```

两个保护条件：
1. `_progress > 0.30`：跳过前 30% 行程（PD 加速阶段），此时即使没有接触物体，执行器力也会因加速度而偏高
2. `force_threshold = 12.0 N`：远高于 PD 空载加速力（约 2-8 N），只有真正夹到物体才会触发

**`_write_final()` 逻辑**（line 225-236）：
- 优先重放 `_last_ctrl`（逐步建立的实际位置）
- `_last_ctrl` 为空时回退到写入开/闭终点（仅首次调用时可能发生）

### 1.6 compute_workspace — 工作空间采样

**算法**：

```
输入: model, data, 关节列表, 手指 body ID
参数: resolution=0.04 m  (4 cm 网格)

1. 定义 3D 网格:
   X: 0.08 → 0.48 m  (前伸范围)
   Y: -0.30 → 0.31 m (左右范围)
   Z: 0.02 → 0.42 m  (高度范围)
   共 10×16×10 = 1600 个采样点

2. 对每一点:
   a. 在临时 MjData 上用阻尼最小二乘 IK 求解
   b. 若 IK 误差 ≤ 6 mm → 标记为"可达"
   c. 将 scratch data 更新到该 IK 解（作为下一点的初始猜测）

3. 渲染优化: 若可达点 > 250，用固定种子随机抽样至 250
```

**优化细节**：使用 `scratch = mujoco.MjData(model)` 作为独立物理数据副本，不干扰主 `data`。每求解完一点后把 scratch 更新到该 IK 解，使下一点的初始猜测更接近目标——这显著减少了连续点的 IK 迭代次数。

### 1.7 render_workspace_spheres — 渲染可达空间

```python
with viewer.lock():           # 获取 viewer 锁（viewer 在另一线程渲染）
    viewer.user_scn.ngeom = 0  # 清空旧几何
    for pt in reachable:
        mjv_initGeom(SPHERE, r=0.007m, pos=pt, rgba=绿半透明)
```

使用 MuJoCo 的 **user scene overlay** 机制，在物理场景之上叠加渲染。球体为半透明绿色（rgba α=0.38），半径 7 mm，不会影响物理仿真。

### 1.8 JointStatePanel — 关节状态面板

独立的 OpenCV 窗口（340×260 px），显示：
- **6 个臂关节**：名称、角度 (°)、弧度 (rad)，橙色文字
- **2 个手指关节**：同上，绿色文字

更新频率 12.5 Hz（`JOINT_EVERY_N=4`），因为文本渲染相对较重且数值变化缓慢。

注意：`show()` 方法只调用 `cv2.imshow()`，**不调用** `cv2.waitKey()`。所有 OpenCV 窗口的事件处理由主循环统一在帧末尾执行一次 `cv2.waitKey(1)`——这消除了多个独立 `waitKey` 调用的累积延迟。

### 1.9 主函数 main() 流程

```
步骤 1: load_and_inject(ball=True, sensors=True, camera=True)
        → 获得 model, data, 关节列表, 手指 body ID, 执行器映射

步骤 2: 初始设置
        - build_gripper_limits() → 检测夹爪开/闭位置
        - command_gripper("open") → 发送张开指令
        - 同步所有执行器 ctrl = qpos → 手臂锁定在初始姿态

步骤 3: build_ik_plan() → 对 approach/grasp/lift/place 四点求解 IK

步骤 4: compute_workspace() → 3D 可达性采样

步骤 5: 初始化各子系统
        - SmoothArmController / SmoothGripperController
        - ForceTorqueSensor / FTDisplay
        - RGBCameraWindow / JointStatePanel

步骤 6: 创建 MuJoCo viewer (被动模式 launch_passive)

步骤 7: 预热 30 帧 (纯物理步进，不做显示)

步骤 8: 主循环 ← 见 1.10

步骤 9: 清理 — 关闭所有窗口
```

### 1.10 主循环详细逻辑

每帧执行 8 个步骤：

```
while viewer.is_running():
    frame_start = perf_counter()

    (1) demo_tick()                  状态机：检查当前 phase 是否完成，
                                      推进到下一 phase 并设置新目标

    (2) if not finished:             仅在 demo 未完成时执行
            arm_ctrl.step()          手臂插值一步
            gripper_ctrl.step()      夹爪插值一步
        (finished 时跳过 → MuJoCo 控制面板滑块可用)

    (3) for _ in range(5):           5 次物理子步 (250 Hz 有效)
            mj_step(model, data)

    (4) if not drawn:                首次渲染工作空间球体
            render_workspace_spheres()

    (5) 条件渲染 (节流):
        frame % 2 == 0 → FTDisplay   (25 Hz)
        frame % 4 == 0 → JointPanel  (12.5 Hz)
        frame % 2 == 0 → RGB Camera  (25 Hz)

    (6) cv2.waitKey(1)               统一处理所有 OpenCV 窗口事件

    (7) viewer.sync()                将状态同步到 MuJoCo viewer

    (8) spin-wait                    精确帧计时: 剩余时间 > 3ms 则 sleep，
                                      最后 1.5ms 用自旋等待消耗
```

### 1.11 状态机（demo_tick）

采用 **二阶段状态机**：`phase` 表示当前动作阶段，`sub` 区分"发起动作"与"等待完成"：

| Phase | 动作 | 发起 (sub=0) | 完成条件 |
|---|---|---|---|
| 0 | 回零 + 张夹爪 | `set_target(zeros)` + `gripper.open()` | `arm_ctrl.done AND gripper_ctrl.done` |
| 1 | 接近（球体上方 15cm） | `set_target(ik["approach"])` | `arm_ctrl.done` |
| 2 | 下降（慢速） | `set_target(ik["grasp"], slow)` | `arm_ctrl.done` |
| 3 | 闭合夹爪 | `gripper.close(100)` | `gripper_ctrl.done` |
| 4 | 举起 | `set_target(ik["lift"])` | `arm_ctrl.done` |
| 5 | 搬运至放置点 | `set_target(ik["place"])` | `arm_ctrl.done` |
| 6 | 释放 | `gripper.open(25)` | `gripper_ctrl.done` |
| 7 | 完成 | — | 打印信息，设置 `finished=True` |

**关键设计决策**：使用 `arm_ctrl.done` / `gripper_ctrl.done` 而非固定帧数等待。这确保每个动作真正完成后才推进，不受运动距离变化影响。

---

## 2. 依赖文件：`examples/common/model_loader.py`

### 2.1 职责

加载 Alicia_D v5.6 机械臂的 MJCF 模型，并向 XML 字符串中**注入运行时所需元素**。所有注入均在内存中的 XML 字符串上完成，不修改源 MJCF 文件。

### 2.2 注入函数列表

| 函数 | 注入内容 | 用途 |
|---|---|---|
| `inject_options()` | `<option integrator="implicitfast" cone="elliptic"/>` | 更稳定的隐式积分器 + 椭圆摩擦锥 |
| `inject_overview_camera()` | `<camera name="overview" .../>` | MuJoCo viewer 的概览视角 |
| `inject_soft_ball()` | `<body name="target_ball">` + `<freejoint/>` + `<geom sphere>` | 直径 5cm 的柔性目标球体（25g，半透明红） |
| `inject_wrist_camera()` | `<camera name="wrist_rgb" .../>` | 腕部 RGB 相机（安装在机械臂末端） |
| `inject_force_sensor()` | `<site>` + `<sensor><force/><torque/></sensor>` | 腕部六维力/力矩传感器 |
| `inject_actuators()` | 8 个 `<position>` 执行器 | 6 个臂关节 + 2 个手指关节的 PD 位置控制器 |

### 2.3 执行器参数

| 执行器 | 关节 | kp | kv | ctrlrange (rad) | forcerange (N) |
|---|---|---|---|---|---|
| Joint1_act | Joint1 | 90 | 12 | [-2.16, 2.16] | ±25 |
| Joint2_act | Joint2 | 90 | 12 | [-1.57, 1.57] | ±25 |
| Joint3_act | Joint3 | 90 | 12 | [-0.5, 2.36] | ±25 |
| Joint4_act | Joint4 | 70 | 10 | [-2.79, 2.79] | ±20 |
| Joint5_act | Joint5 | 70 | 10 | [-1.57, 1.57] | ±20 |
| Joint6_act | Joint6 | 45 | 7 | [-3.14, 3.14] | ±15 |
| left_finger_act | left_finger | 55 | 7 | 无限制 | ±8 |
| right_finger_act | right_finger | 55 | 7 | 无限制 | ±8 |

PD 控制律：`force = kp × (ctrl − q) − kv × q̇`

### 2.4 load_and_inject() 返回值

```python
return (model, data, arm_joints, gripper_joints,
        left_body_id, right_body_id, joint_to_actuator)
```

| 返回值 | 类型 | 说明 |
|---|---|---|
| `model` | `MjModel` | MuJoCo 模型 |
| `data` | `MjData` | MuJoCo 数据（含 qpos, ctrl, sensordata 等） |
| `arm_joints` | `list[int]` | 6 个臂关节的 ID |
| `gripper_joints` | `list[int]` | 2 个手指关节的 ID |
| `left_body_id` | `int` | Link7（左手指）body ID |
| `right_body_id` | `int` | Link8（右手指）body ID |
| `joint_to_actuator` | `dict[int,int]` | 关节 ID → 执行器 ID 映射 |

### 2.5 关节检测逻辑

遍历模型中所有关节（`model.njnt`），跳过 `mjJNT_FREE` 类型。对铰链关节（`mjJNT_HINGE`）和滑动关节（`mjJNT_SLIDE`），
根据名称是否包含 "finger" 分类为臂关节或手指关节。

---

## 3. 依赖文件：`examples/common/ik_solver.py`

### 3.1 数据结构

```python
@dataclass
class IKResult:
    angles: ndarray      # 解得的关节角 (rad)
    target: ndarray      # 目标笛卡尔位置
    final_pos: ndarray   # 实际到达的位置
    error_norm: float    # 最终误差范数 (m)
    iterations: int      # 使用的迭代次数
    success: bool        # 是否收敛 (error ≤ 6 mm)
```

### 3.2 solve_gripper_center_ik — 阻尼最小二乘 IK

**目标**：求解使**两指中点**（gripper center）到达目标笛卡尔位置的关节角度。

**算法流程**：

```
1. 复制 seed_data 到临时 tmp_data
2. 读取各关节的角度范围 [lower, upper]

3. 迭代 (max 500 次):
   a. forward kinematics → 计算当前手指中点位置
   b. error = target − current_position
   c. 若 ‖error‖ ≤ tol (6 mm) → 收敛，退出
   d. 计算雅可比矩阵:
      - mj_jacBody(left_body) → J_L
      - mj_jacBody(right_body) → J_R
      - J = (J_L + J_R) / 2    (两指中点的雅可比)
   e. 阻尼最小二乘:
      - Δq = Jᵀ (J Jᵀ + λ²I)⁻¹ error
      - λ = 0.025 (阻尼因子)
   f. 更新: q ← clip(q + 0.45 × Δq, lower, upper)

4. 返回 IKResult
```

**关键参数**：

| 参数 | 值 | 含义 |
|---|---|---|
| `max_iter` | 500 | 最大迭代次数 |
| `tol` | 0.006 m (6 mm) | 收敛阈值 |
| `λ` (阻尼) | 0.025 | 防止奇异位形附近 Δq 过大 |
| `step_size` | 0.45 | 每次迭代的步长因子 |
| `Δq clip` | ±0.08 rad | 单次迭代关节角最大变化 |

### 3.3 build_ik_plan — 轨迹规划

预计算从 home 姿态（全零关节角）出发的 4 个连续 IK 目标：

| 名称 | 目标位置 | 物理含义 |
|---|---|---|
| `approach` | `BALL_POS + (0, 0, 0.15)` | 球体正上方 15 cm |
| `grasp` | `BALL_POS + (0, 0, 0.025×0.6)` | 球心上方 1.5 cm（手指可合拢的位置） |
| `lift` | `BALL_POS + (0, 0, 0.025×0.6+0.20)` | 抓起后抬高 20 cm |
| `place` | `(0.45, 0.15, 0.20)` | 放置点（右前方高处） |

每个目标求解后，将 scratch data 更新到该 IK 解，作为下一个目标的初始猜测。这种**链式初始化**确保后续目标在连续可达的关节空间中求解，减少迭代次数。

---

## 4. 依赖文件：`examples/common/motion.py`

### 4.1 build_gripper_limits — 夹爪限位检测

**算法**：对每个手指关节，遍历其角度范围的上下限，通过正运动学计算手指 body 在笛卡尔空间的位置，根据距离远近判断"张开"和"闭合"分别对应哪个角度极值。

```python
if "left" in name:
    open_value, closed_value = hi, lo     # 左指：上限=张开
elif "right" in name:
    open_value, closed_value = lo, hi     # 右指：下限=张开
```

**返回值**：

```python
[
    {"joint": joint_id, "open": open_rad, "closed": closed_rad},
    ...
]
```

### 4.2 command_gripper — 即时夹爪指令

直接将夹爪执行器设置为指定的位置键（"open" 或 "closed"）。用于初始化阶段和 demo 完成后的快速指令。

### 4.3 move_arm_to_angles — 阻塞式手臂运动（本 demo 未使用）

原始版本的手臂控制函数。采用余弦插值 + 内部 `mj_step` 循环。本 `run.py` 使用 `SmoothArmController` 替代此函数（非阻塞优势）。保留在公共模块中供其他示例使用。

### 4.4 close_gripper — 阻塞式夹爪闭合（本 demo 未使用）

原始版本的力控夹爪函数。本 `run.py` 使用 `SmoothGripperController` 替代。

---

## 5. 依赖文件：`examples/common/force_sensor.py`

### 5.1 ForceTorqueSensor — 传感器读取

```python
class ForceTorqueSensor:
    def __init__(self, model):
        # 查找 "wrist_force" 和 "wrist_torque" 传感器 ID
        # 记录它们在 sensordata 数组中的起始地址

    def read(self, data) -> FTReading:
        # 从 data.sensordata[adr:adr+3] 提取力/力矩分量
        return FTReading(force=[Fx,Fy,Fz], torque=[Mx,My,Mz], timestamp=t)
```

MuJoCo 的 `sensordata` 是一维数组，每个传感器的数据是连续的 3 个浮点数（三维力或力矩）。`sensor_adr` 记录了每个传感器在此数组中的起始偏移。

### 5.2 FTDisplay — 实时力/力矩曲线

**窗口**：OpenCV `"Force/Torque Sensor"`，520×380 px。

**数据流**：
```
FTReading → update() 推入环形历史缓冲 → show() 绘制曲线
```

**环形缓冲**：`force_history` 和 `torque_history` 各为 `(history_len, 3)` 的数组，`write_idx` 循环写入。

**绘制逻辑**（`show()`）：

1. 清空画布为深灰色背景
2. 上层 (top 50px)：数值文本 — `Fx:+1.23 Fy:-0.45 ... Mx:+0.012 ...`
3. 中层 (plot_h)：力曲线 — 红(Fx) 绿(Fy) 蓝(Fz)，自动缩放 Y 轴
4. 下层 (plot_h)：力矩曲线 — 青(Mx) 品红(My) 黄(Mz)

**自动缩放**：`update()` 计算历史数据的最大绝对值，动态调整 Y 轴范围。初始力范围 5 N，力矩范围 1 Nm，随数据增长自动扩展。

**重要修改**：`show()` 内**不调用** `cv2.waitKey(1)`。主循环统一调用一次 `cv2.waitKey(1)` 处理所有 OpenCV 窗口。这避免了 3 个窗口各调用一次导致的累计延迟。

---

## 6. 依赖文件：`examples/common/camera.py`

### 6.1 RGBCameraWindow — 眼在手相机

**窗口**：OpenCV `"Eye-in-Hand RGB Camera"`，480×360 px。

**渲染管线**：

```
mjRenderer.update_scene(data, camera=wrist_rgb)  ← 从腕部相机视角更新场景
mjRenderer.render()                               ← 离屏渲染为 RGB 数组
cv2.cvtColor(RGB→BGR)                             ← 颜色空间转换
cv2.putText(overlay)                               ← 叠加力数值
cv2.imshow(window_name, bgr)                       ← 显示
```

**帧率控制**：`should_update(step)` 返回 `step % render_every_n == 0`。本 demo 中 `render_every_n = CAMERA_EVERY_N = 2`，即每 2 帧渲染一次（25 Hz）。

**离线渲染器**：`mujoco.Renderer` 是一个独立的 OpenGL 离屏渲染上下文。它使用 GPU 但不依赖 MuJoCo viewer 窗口。每次 `render()` 返回 numpy 数组 (height, width, 3)，无需与 viewer 同步。

**重要修改**：`update()` 内**不调用** `cv2.waitKey(1)`（从原始版本中移除），由主循环统一处理。

---

## 7. 依赖文件：`examples/common/__init__.py`

聚合导出，使使用者可以写：
```python
from common import load_and_inject, IKResult, ForceTorqueSensor, ...
```

导出清单：
- `load_and_inject`
- `IKResult`, `solve_gripper_center_ik`, `build_ik_plan`, `gripper_center`, `set_joint_positions`
- `move_arm_to_angles`, `command_gripper`, `close_gripper`, `sync_position_actuators`, `build_gripper_limits`
- `ForceTorqueSensor`, `FTReading`, `FTDisplay`
- `RGBCameraWindow`

---

## 附录 A：完整数据流

```
                    ┌─────────────────────────┐
                    │   synriard.get_model()  │
                    │   → Alicia_D MJCF XML   │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   model_loader.py       │
                    │   load_and_inject()     │
                    │   + ball + camera       │
                    │   + force sensor        │
                    │   + position actuators  │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   ik_solver.py          │
                    │   build_ik_plan()       │
                    │   → approach/grasp/     │
                    │     lift/place angles   │
                    └───────────┬─────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                  ▼
    ┌─────────────┐   ┌─────────────┐   ┌──────────────┐
    │SmoothArm    │   │SmoothGripper│   │FTDisplay     │
    │Controller   │   │Controller   │   │RGBCameraWin  │
    │             │   │             │   │JointPanel    │
    │ step()→ctrl │   │step()→ctrl  │   │              │
    └──────┬──────┘   └──────┬──────┘   └──────┬───────┘
           │                 │                  │
           └─────────┬───────┘                  │
                     ▼                          ▼
           ┌─────────────────┐       ┌──────────────────┐
           │  mj_step() ×5   │       │ cv2.imshow() ×3  │
           │  (physics 250Hz)│       │ cv2.waitKey(1)×1 │
           └────────┬────────┘       └──────────────────┘
                    ▼
           ┌─────────────────┐
           │  viewer.sync()  │
           │  (GPU render)   │
           └─────────────────┘
```

## 附录 B：帧循环时序图

```
时间轴 →

帧 N-1                 帧 N (20ms)                  帧 N+1
  │                        │                            │
  ├─ demo_tick()           ├─ demo_tick()               ├─ ...
  ├─ arm_ctrl.step()       ├─ arm_ctrl.step()
  ├─ gripper_ctrl.step()   ├─ gripper_ctrl.step()
  ├─ [mj_step ×5]          ├─ [mj_step ×5]
  ├─ render_ws (仅首次)     ├─ (skip)
  ├─ FT (if N%2=0)         ├─ FT (if N%2≠0 → skip)
  ├─ Joints (if N%4=0)      ├─ (skip)
  ├─ Camera (if N%2=0)     ├─ Camera (if N%2≠0 → skip)
  ├─ cv2.waitKey(1)        ├─ cv2.waitKey(1)
  ├─ viewer.sync()         ├─ viewer.sync()
  └─ spin-wait → 20ms      └─ spin-wait → 20ms
```

## 附录 C：关键参数速查表

| 参数 | 默认值 | 位置 | 说明 |
|---|---|---|---|
| 仿真步长 | 1/240 s | `model_loader.SIM_HZ` | MuJoCo 物理步长 |
| 显示帧率 | 50 Hz | `TARGET_DISPLAY_HZ` | 渲染循环目标 |
| 物理子步 | 5 | `PHYSICS_SUBSTEPS` | 每帧 mj_step 次数 |
| IK 容差 | 6 mm | `ik_solver.tol` | 收敛判定阈值 |
| IK 阻尼 | 0.025 | `ik_solver.lam` | 阻尼最小二乘系数 |
| 球体重量 | 25 g | `model_loader.BALL_RADIUS` | 柔性小球物理参数 |
| 球体位置 | (0.30, 0.00, 0.05) | `model_loader.BALL_POS` | 球体世界坐标 |
| 力阈值 | 12.0 N | `SmoothGripper.step()` | 夹爪接触判定 |
| 力检测延迟 | 30% 行程 | `SmoothGripper.step()` | 防 PD 加速误触 |
| 工作空间分辨率 | 4 cm | `compute_workspace()` | IK 采样网格间距 |
| 工作空间球体上限 | 250 | `WORKSPACE_MAX_SPHERES` | GPU 渲染优化 |
| 力曲线刷新率 | 25 Hz | `FT_EVERY_N` | 环形缓冲写入频率 |
| 相机刷新率 | 25 Hz | `CAMERA_EVERY_N` | 离屏渲染频率 |
| 关节面板刷新 | 12.5 Hz | `JOINT_EVERY_N` | 文本叠加刷新率 |
