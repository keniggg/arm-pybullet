# SynriaRD: Synria Robotics 玄雅科技 | 机器人标准模型文件

[English](README_EN.md) | 中文

本仓库包含 Synria Robotics 玄雅科技机器人平台的 URDF（统一机器人描述格式）和 MJCF（MuJoCo 建模格式）模型。

## 仓库结构

```
├── synriard
│   ├── meshes
│   │   ├── Alicia_D_v5_5
│   │   ├── Alicia_D_v5_6
│   │   ├── Alicia_M_v1_0
│   │   └── Bessica_D_v1_0
│   ├── mjcf
│   │   ├── Alicia_D_v5_5
│   │   ├── Alicia_D_v5_6
│   │   ├── Alicia_M_v1_0
│   │   └── Bessica_D_v1_0
│   └── urdf
│       ├── Alicia_D_v5_5
│       ├── Alicia_D_v5_6
│       ├── Alicia_M_v1_0
│       └── Bessica_D_v1_0
```

## 命名规范

所有模型文件遵循统一的命名格式：`{name}_{version}_{variant}.{ext}`

- **name**: 机器人名称（如 `Alicia_D`, `Alicia_M`, `Bessica_D`）
- **version**: 版本号（如 `v5_5`, `v5_6`, `v1_0`）
- **variant**: 变体标识
  - 对于带夹爪的机器人（Alicia_D, Alicia_M）：`gripper_{size}`（如 `gripper_50mm`, `gripper_100mm`）
  - 对于其他变体（Bessica_D）：直接使用变体名（如 `covered`, `skeleton`, `covered_interactive`）
- **ext**: 文件扩展名（`.urdf` 或 `.xml`）

### 示例

- `Alicia_D_v5_6_gripper_50mm.urdf` - Alicia_D v5.6 版本，50mm 夹爪
- `Alicia_M_v1_0_gripper_100mm.urdf` - Alicia_M v1.0 版本，100mm 夹爪
- `Bessica_D_v1_0_covered.urdf` - Bessica_D v1.0 版本，带外壳变体
- `Bessica_D_v1_0_covered_interactive.xml` - Bessica_D v1.0 版本，带外壳交互式变体（MJCF）

### 使用 API

```python
from synriard import get_model_path, list_available_models

# 获取模型路径
urdf_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_50mm")
mjcf_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_50mm", model_format="mjcf")

# 列出所有可用模型
print(list_available_models(model_format="urdf"))
print(list_available_models(model_format="urdf", show_path=True))
```

## 产品

### Alicia-D 
- **描述**: 灵动操作臂
- **自由度**： 6
- **夹爪配置**：50mm 和 100mm
- **位置**：[`Alicia_D_v5_5`](synriard/urdf/Alicia_D_v5_5)和[`Alicia_D_v5_6`](synriard/urdf/Alicia_D_v5_6)



### Alicia-M 

- **描述**: 云擎操作臂
- **自由度**： 6
- **位置**: [`Alicia_M_v1_0_gripper_100mm.urdf`](synriard/urdf/Alicia_M_v1_0/Alicia_M_v1_0_gripper_100mm.urdf)

### Bessica-D 
- **描述**: 灵越双臂人形机器人
- **自由度**：14 （双 7 自由度机械臂）
- **外观**：无外壳版和外壳版

#### 无外壳版
- **URDF**: [`Bessica_D_v1_0_skeleton.urdf`](synriard/urdf/Bessica_D_v1_0/Bessica_D_v1_0_skeleton.urdf)
- **MuJoCo XML**: [`Bessica_D_v1_0_skeleton.xml`](synriard/mjcf/Bessica_D_v1_0/Bessica_D_v1_0_skeleton.xml)


#### 带外壳版
- **URDF**: [`Bessica_D_v1_0_covered.urdf`](synriard/urdf/Bessica_D_v1_0/Bessica_D_v1_0_covered.urdf)
- **MuJoCo XML**: [`Bessica_D_v1_0_covered.xml`](synriard/mjcf/Bessica_D_v1_0/Bessica_D_v1_0_covered.xml)
- **MuJoCo XML (交互式)**: [`Bessica_D_v1_0_covered_interactive.xml`](synriard/mjcf/Bessica_D_v1_0/Bessica_D_v1_0_covered_interactive.xml)


## 添加新机器人模型

添加新机器人模型后，运行自动化脚本自动生成所需的 `__init__.py` 文件：

```bash
# 1. 添加模型文件到对应目录
mkdir -p synriard/mjcf/RobotName_v1_0
cp RobotName_v1_0_gripper_100mm.xml synriard/mjcf/RobotName_v1_0/

# 2. 运行自动化脚本
python3 auto_generate_init.py

# 脚本会自动：
# - 生成每个机器人目录的 __init__.py
# - 更新父目录的 __init__.py 注册所有机器人
```

脚本选项：
- `--format mjcf|urdf|all`: 指定处理的格式（默认：all）
- `--synriard-path PATH`: 指定 synriard 目录路径（默认：自动检测）

## 支持的仿真环境

- ROS/ROS2 (通过 URDF)
- MuJoCo (通过 MJCF)
- Gazebo (通过 URDF)
- PyBullet (通过 URDF)
- Isaac Sim (通过 URDF、MJCF)


