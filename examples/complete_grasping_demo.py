#!/usr/bin/env python3
"""
完整的机械臂抓取演示程序
功能：
1. 自动移动机械臂到小球位置
2. 抓取小球
3. 移动小球到新位置
4. RGB摄像头视角实时更新
5. 机械臂关节参数实时更新
6. 力反馈参数实时更新
"""

import os
import time
import math
import numpy as np
import pybullet as p
import pybullet_data

SIM_HZ = 240.0
# 运行流畅度控制：sleep 越小越“快/顺滑”（占用 CPU 更高）；越大越“慢/稳”
SLEEP_SCALE = 0.15  # 原来等同于 1.0（每步都 sleep 1/240），会显得很慢

try:
    import cv2  # type: ignore[import-not-found]  # optional, for RGB preview window
except Exception:
    cv2 = None


def get_joint_angles_for_position(robot_id, target_pos, target_orn=None):
    """
    计算逆运动学，获取到达目标位置所需的关节角度
    """

    # 找到末端执行器link
    num_joints = p.getNumJoints(robot_id)
    end_effector_link = None
    for j in range(num_joints):
        info = p.getJointInfo(robot_id, j)
        link_name = info[12].decode("utf-8") if info[12] else ""
        if "tool" in link_name.lower() or "end" in link_name.lower():
            end_effector_link = j
            break
    if end_effector_link is None:
        end_effector_link = num_joints - 1

    if target_orn is None:
        link_state = p.getLinkState(robot_id, end_effector_link)
        target_orn = link_state[1]


    # 计算逆运动学
    joint_angles = p.calculateInverseKinematics(
        bodyUniqueId=robot_id,
        endEffectorLinkIndex=end_effector_link,
        targetPosition=target_pos,
        targetOrientation=target_orn,
        maxNumIterations=100,
        residualThreshold=1e-5
    )

    return joint_angles


def map_ik_to_controllable(robot_id, ik_solution, controllable_joints):
    """Map IK results to controllable joints safely."""
    joint_count = p.getNumJoints(robot_id)
    padded = list(ik_solution)
    if len(padded) < joint_count:
        for j in range(len(padded), joint_count):
            padded.append(p.getJointState(robot_id, j)[0])
    return [padded[j] for j in controllable_joints]


def move_arm_to_position(robot_id, target_angles, controllable_joints, speed=1.0, update_fn=None):
    """
    平滑移动机械臂到指定关节角度
    """
    if not p.isConnected():
        return
    
    # 获取当前关节角度
    current_angles = []
    for joint_idx in controllable_joints:
        if not p.isConnected():
            return
        joint_state = p.getJointState(robot_id, joint_idx)
        current_angles.append(joint_state[0])

    # 计算角度差
    angle_diffs = [target - current for target, current in zip(target_angles, current_angles)]

    # 计算移动步数（基于最大角度差）
    max_diff = max(abs(diff) for diff in angle_diffs)
    if max_diff < 1e-6:
        return  # 已经在目标位置

    steps = int(max_diff / (speed * 0.01)) + 1  # 每步0.01弧度
    steps = min(steps, 200)  # 最大200步

    for step in range(steps):
        if not p.isConnected():
            return
        
        t = (step + 1) / steps
        # 使用平滑插值
        t_smooth = 0.5 - 0.5 * math.cos(t * math.pi)  # 余弦插值

        intermediate_angles = []
        for current, diff in zip(current_angles, angle_diffs):
            intermediate_angles.append(current + diff * t_smooth)

        # 设置关节角度
        for i, joint_idx in enumerate(controllable_joints):
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=intermediate_angles[i],
                force=200,
                positionGain=0.3
            )

        if update_fn:
            update_fn()
        p.stepSimulation()
        if SLEEP_SCALE > 0:
            time.sleep((1.0 / SIM_HZ) * SLEEP_SCALE)


def step_simulation(steps, update_fn=None):
    """运行指定步数的仿真并刷新显示。"""
    for _ in range(steps):
        if not p.isConnected():
            return
        if update_fn:
            update_fn()
        if not p.isConnected():
            return
        p.stepSimulation()
        if SLEEP_SCALE > 0:
            time.sleep((1.0 / SIM_HZ) * SLEEP_SCALE)


def open_gripper(robot_id, gripper_limits):
    """Open gripper to joint-specific limits."""
    if not p.isConnected():
        return
    for item in gripper_limits:
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=item["joint"],
            controlMode=p.POSITION_CONTROL,
            targetPosition=item["open"],
            force=50
        )


def close_gripper(robot_id, gripper_limits, force_threshold=2.0, update_fn=None):
    """Close gripper until a contact force threshold is reached."""
    if not p.isConnected():
        return False
    
    max_force = 0
    max_steps = 240

    for step in range(max_steps):
        if not p.isConnected():
            return False
            
        t = (step + 1) / max_steps
        for item in gripper_limits:
            target_pos = item["open"] + (item["closed"] - item["open"]) * t
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=item["joint"],
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=120,
                positionGain=0.4,
                velocityGain=1.0
            )

        max_force = 0
        for item in gripper_limits:
            if not p.isConnected():
                return False
            joint_state = p.getJointState(robot_id, item["joint"])
            forces = joint_state[2]
            if forces:
                fx, fy, fz, mx, my, mz = forces
                total_force = math.sqrt(fx**2 + fy**2 + fz**2)
                max_force = max(max_force, total_force)

        if update_fn:
            update_fn()
        p.stepSimulation()
        if SLEEP_SCALE > 0:
            time.sleep((1.0 / SIM_HZ) * SLEEP_SCALE)

        if max_force >= force_threshold:
            break

    return max_force >= force_threshold


def has_gripper_contact(robot_id, ball_id, gripper_links, distance_threshold=0.0):
    """Check whether any gripper link is in contact (or close) with the ball."""
    if not p.isConnected():
        return False
    
    for link_idx in gripper_links:
        if p.getContactPoints(bodyA=robot_id, bodyB=ball_id, linkIndexA=link_idx):
            return True
    if distance_threshold and distance_threshold > 0:
        for link_idx in gripper_links:
            if p.getClosestPoints(bodyA=robot_id, bodyB=ball_id, distance=distance_threshold, linkIndexA=link_idx):
                return True
    return False


def attempt_attach_ball(robot_id, ball_id, end_effector_link, gripper_joints, distance_threshold=0.003):
    """检测夹爪与小球接触后，创建约束辅助抓取（保持当前相对位姿，避免“瞬移/甩飞”）。"""
    if not p.isConnected():
        return None
    
    contacts = []
    for joint_idx in gripper_joints:
        if not p.isConnected():
            return None
        contacts.extend(p.getContactPoints(bodyA=robot_id, bodyB=ball_id, linkIndexA=joint_idx))

    if not contacts and distance_threshold > 0:
        for joint_idx in gripper_joints:
            if not p.isConnected():
                return None
            if p.getClosestPoints(bodyA=robot_id, bodyB=ball_id,
                                  distance=distance_threshold, linkIndexA=joint_idx):
                contacts.append(True)
                break

    if not contacts:
        return None

    # 计算 ball 在 tool(link) 坐标系下的相对位姿，保证创建约束时不发生位置跳变
    link_state = p.getLinkState(robot_id, end_effector_link, computeForwardKinematics=True)
    parent_pos, parent_orn = link_state[0], link_state[1]
    ball_pos, ball_orn = p.getBasePositionAndOrientation(ball_id)
    inv_parent_pos, inv_parent_orn = p.invertTransform(parent_pos, parent_orn)
    rel_pos, rel_orn = p.multiplyTransforms(inv_parent_pos, inv_parent_orn, ball_pos, ball_orn)

    constraint_id = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=end_effector_link,
        childBodyUniqueId=ball_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=rel_pos,
        childFramePosition=[0, 0, 0],
        parentFrameOrientation=rel_orn,
        childFrameOrientation=[0, 0, 0, 1]
    )
    return constraint_id


def compute_gripper_center(robot_id, gripper_joints):
    """Compute midpoint between left/right gripper links."""
    if not p.isConnected():
        return None
    if len(gripper_joints) < 2:
        return None
    left_state = p.getLinkState(robot_id, gripper_joints[0], computeForwardKinematics=True)
    right_state = p.getLinkState(robot_id, gripper_joints[1], computeForwardKinematics=True)
    return [
        (left_state[0][0] + right_state[0][0]) * 0.5,
        (left_state[0][1] + right_state[0][1]) * 0.5,
        (left_state[0][2] + right_state[0][2]) * 0.5,
    ]


def compute_tool_offset(robot_id, end_effector_link, gripper_joints):
    """Compute world-space offset from gripper center to tool link."""
    if not p.isConnected():
        return [0.0, 0.0, 0.0]
    center = compute_gripper_center(robot_id, gripper_joints)
    if center is None:
        return [0.0, 0.0, 0.0]
    tool_state = p.getLinkState(robot_id, end_effector_link, computeForwardKinematics=True)
    tool_pos = tool_state[0]
    return [
        tool_pos[0] - center[0],
        tool_pos[1] - center[1],
        tool_pos[2] - center[2],
    ]


def apply_tool_offset(target_pos, tool_offset, extra_z=0.0):
    """Apply tool offset so the gripper center reaches target_pos."""
    return [
        target_pos[0] + tool_offset[0],
        target_pos[1] + tool_offset[1],
        target_pos[2] + tool_offset[2] + extra_z,
    ]


def normalize_vector(vec, fallback=(0.0, 0.0, 1.0)):
    norm = math.sqrt(sum(v * v for v in vec))
    if norm < 1e-9:
        return [fallback[0], fallback[1], fallback[2]]
    return [v / norm for v in vec]


def create_ball_stand(base_pos, height, radius=0.035, color=(0.6, 0.6, 0.6, 1.0)):
    """Create a simple pedestal so the ball rests at a reachable height."""
    if height <= 0:
        return None
    collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
    visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
    stand_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=[base_pos[0], base_pos[1], height / 2],
    )
    return stand_id


def create_camera_display():
    """创建摄像头显示窗口"""
    # 创建一个小的纹理显示区域
    camera_display = p.addUserDebugText(
        "RGB Camera View",
        [0.5, 0.8, 0.9],
        textColorRGB=[0, 1, 0],
        textSize=1.2
    )
    return camera_display


def _show_rgb_preview(rgb_rgba, width, height, window_name="RGB Camera"):
    """Show RGB image in an OpenCV window (optional)."""
    if cv2 is None or rgb_rgba is None:
        return
    try:
        rgba = np.reshape(rgb_rgba, (height, width, 4))
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        cv2.imshow(window_name, bgr)
        cv2.waitKey(1)
    except Exception:
        # Don't crash the simulation if preview fails
        return


def _close_rgb_preview(window_name="RGB Camera"):
    if cv2 is None:
        return
    try:
        cv2.destroyWindow(window_name)
    except Exception:
        pass


def update_camera_view(robot_id, end_effector_link, camera_state=None,
                       camera_width=320, camera_height=240, smoothing=0.05):
    """更新眼在手摄像头视图"""
    try:
        # 获取末端link状态
        link_state = p.getLinkState(robot_id, end_effector_link, computeForwardKinematics=True)
        camera_pos = list(link_state[0])
        camera_orn = link_state[1]

        # 计算摄像头朝向（用工具坐标系的轴向量；并对 forward 做平滑，减少抖动）
        camera_rot_matrix = p.getMatrixFromQuaternion(camera_orn)
        forward = normalize_vector([camera_rot_matrix[0], camera_rot_matrix[3], camera_rot_matrix[6]])
        if camera_state is not None:
            prev_forward = camera_state.get("forward")
            if prev_forward is not None:
                alpha = smoothing
                forward = normalize_vector([
                    prev_forward[0] * (1 - alpha) + forward[0] * alpha,
                    prev_forward[1] * (1 - alpha) + forward[1] * alpha,
                    prev_forward[2] * (1 - alpha) + forward[2] * alpha,
                ])
            camera_state["forward"] = forward
        camera_target = [
            camera_pos[0] + forward[0] * 0.3,
            camera_pos[1] + forward[1] * 0.3,
            camera_pos[2] + forward[2] * 0.3
        ]

        if camera_state is not None:
            prev_pos = camera_state.get("pos")
            prev_target = camera_state.get("target")
            if prev_pos is not None and prev_target is not None:
                alpha = smoothing
                camera_pos = [
                    prev_pos[0] * (1 - alpha) + camera_pos[0] * alpha,
                    prev_pos[1] * (1 - alpha) + camera_pos[1] * alpha,
                    prev_pos[2] * (1 - alpha) + camera_pos[2] * alpha,
                ]
                camera_target = [
                    prev_target[0] * (1 - alpha) + camera_target[0] * alpha,
                    prev_target[1] * (1 - alpha) + camera_target[1] * alpha,
                    prev_target[2] * (1 - alpha) + camera_target[2] * alpha,
                ]
            camera_state["pos"] = camera_pos
            camera_state["target"] = camera_target

        # 视角稳定化：锁定地平线（horizon lock）
        # - 直接用 world_up 会让画面“水平”，不随末端 roll 抖动
        # - 但要把 up 从 forward 中正交化，否则接近竖直时会数值不稳
        world_up = [0.0, 0.0, 1.0]
        # up_proj = up - (up·f) f
        dot_uf = world_up[0] * forward[0] + world_up[1] * forward[1] + world_up[2] * forward[2]
        up_proj = [
            world_up[0] - dot_uf * forward[0],
            world_up[1] - dot_uf * forward[1],
            world_up[2] - dot_uf * forward[2],
        ]
        camera_up = normalize_vector(up_proj, fallback=(0.0, 1.0, 0.0))
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=camera_target,
            cameraUpVector=camera_up
        )

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=camera_width/camera_height,
            nearVal=0.001,  # 更近距离，提高近景清晰度
            farVal=5.0  # 更远距离，增加场景深度
        )

        # 获取RGB图像，提高渲染质量
        images = p.getCameraImage(
            width=camera_width,
            height=camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        return images[2]  # RGB图像数据

    except Exception as e:
        print(f"Camera update failed: {e}")
        return None


def update_joint_display(robot_id, controllable_joints, display_ids, anchor_pos):
    """Update joint angle display."""
    if not p.isConnected():
        return
    for i, joint_idx in enumerate(controllable_joints):
        try:
            joint_state = p.getJointState(robot_id, joint_idx)
            angle = joint_state[0]
            velocity = joint_state[1]

            info = p.getJointInfo(robot_id, joint_idx)
            joint_name = info[1].decode("utf-8")

            p.addUserDebugText(
                f"{joint_name}: {angle:.3f} rad ({math.degrees(angle):.1f} deg)",
                [anchor_pos[0], anchor_pos[1], anchor_pos[2] - i * 0.05],
                textColorRGB=[0, 0.8, 0.8],
                textSize=1.0,
                replaceItemUniqueId=display_ids[i]
            )
        except Exception as e:
            # GUI 关闭/断连时这里会报错：避免刷屏
            if p.isConnected():
                print(f"Joint display update failed: {e}")


def update_force_display(robot_id, gripper_joints, force_display_ids):
    """更新力反馈显示"""
    for i, joint_idx in enumerate(gripper_joints):
        try:
            if not p.isConnected():
                return
            joint_state = p.getJointState(robot_id, joint_idx)
            forces = joint_state[2]

            if forces:
                fx, fy, fz, mx, my, mz = forces
                total_force = math.sqrt(fx**2 + fy**2 + fz**2)

                # 根据力的大小改变颜色
                if total_force > 5.0:
                    color = [0, 1, 0]  # 绿色 - 成功抓取
                    status = "GRABBED"
                elif total_force > 1.0:
                    color = [1, 0.5, 0]  # 橙色 - 接触
                    status = "CONTACT"
                else:
                    color = [1, 0, 0]  # 红色 - 无接触
                    status = "NO CONTACT"

                p.addUserDebugText(
                    f"Gripper {i+1}: {total_force:.2f}N ({status})",
                    [0, 0.1 + i*0.05, 0.8],
                    textColorRGB=color,
                    textSize=1.2,
                    replaceItemUniqueId=force_display_ids[i]
                )
            else:
                p.addUserDebugText(
                    f"Gripper {i+1}: No sensor data",
                    [0, 0.1 + i*0.05, 0.8],
                    textColorRGB=[0.5, 0.5, 0.5],
                    textSize=1.2,
                    replaceItemUniqueId=force_display_ids[i]
                )
        except Exception as e:
            if p.isConnected():
                p.addUserDebugText(
                    f"Gripper {i+1}: Error",
                    [0, 0.1 + i*0.05, 0.8],
                    textColorRGB=[1, 0, 0],
                    textSize=1.2,
                    replaceItemUniqueId=force_display_ids[i]
                ) 


def create_open_box(base_pos, inner_size=(0.18, 0.18), wall_height=0.12, wall_thickness=0.01,
                    color=(0.7, 0.7, 0.9, 1.0)):
    """创建一个开口箱子（底板+四面墙）。"""
    half_x = inner_size[0] / 2
    half_y = inner_size[1] / 2
    half_t = wall_thickness / 2
    half_h = wall_height / 2

    shapes = [
        # 底板
        (p.GEOM_BOX, [half_x, half_y, half_t], [0, 0, wall_thickness / 2]),
        # 前后墙
        (p.GEOM_BOX, [half_x, half_t, half_h], [0, half_y + half_t, wall_thickness + half_h]),
        (p.GEOM_BOX, [half_x, half_t, half_h], [0, -(half_y + half_t), wall_thickness + half_h]),
        # 左右墙
        (p.GEOM_BOX, [half_t, half_y, half_h], [half_x + half_t, 0, wall_thickness + half_h]),
        (p.GEOM_BOX, [half_t, half_y, half_h], [-(half_x + half_t), 0, wall_thickness + half_h]),
    ]

    body_ids = []
    for geom_type, half_extents, local_offset in shapes:
        collision_shape = p.createCollisionShape(geom_type, halfExtents=half_extents)
        visual_shape = p.createVisualShape(geom_type, halfExtents=half_extents, rgbaColor=color)
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[base_pos[0] + local_offset[0],
                          base_pos[1] + local_offset[1],
                          base_pos[2] + local_offset[2]]
        )
        body_ids.append(body_id)

    return body_ids


def main():
    """主演示程序"""
    # 改变工作目录到项目根目录
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(repo_root)

    # 连接PyBullet GUI，设置更高的渲染分辨率
    physics_client = p.connect(p.GUI, options="--width=1280 --height=720 --mp4=" + "/tmp/grasping_demo.mp4")
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / SIM_HZ)
    p.setPhysicsEngineParameter(numSolverIterations=200, solverResidualThreshold=1e-7)
    
    # 提高渲染质量
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # 启用阴影
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)  # 启用RGB缓冲区预览
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)  # 启用深度缓冲区预览
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)  # 禁用分割标记预览
    p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)  # 禁用线框模式
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 0)  # 禁用单步渲染
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)  # 启用渲染
    
    # 设置相机视角，提供更好的初始视图
    p.resetDebugVisualizerCamera(cameraDistance=1.8, cameraYaw=50, cameraPitch=-35,
                               cameraTargetPosition=[0.3, 0, 0.3])

    # 创建地面
    plane_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[10, 10, 0.01])
    plane_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[10, 10, 0.01], rgbaColor=[0.8, 0.8, 0.8, 1])
    plane_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=plane_collision_shape,
                                baseVisualShapeIndex=plane_visual_shape, basePosition=[0, 0, -0.01])

    # 加载机械臂
    urdf_path = "synriard/urdf/Alicia_D_v5_6/Alicia_D_v5_6_gripper_50mm.urdf"
    p.setAdditionalSearchPath("synriard")
    p.setAdditionalSearchPath("synriard/meshes")

    # 添加meshes子目录
    meshes_root = "synriard/meshes"
    if os.path.isdir(meshes_root):
        for root, dirs, files in os.walk(meshes_root):
            p.setAdditionalSearchPath(root)

    flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_MAINTAIN_LINK_ORDER
    robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True, flags=flags)

    # 获取关节信息
    num_joints = p.getNumJoints(robot_id)
    controllable_joints = []
    gripper_joints = []
    end_effector_link = None

    for j in range(num_joints):
        info = p.getJointInfo(robot_id, j)
        joint_name = info[1].decode("utf-8")
        joint_type = info[2]
        link_name = info[12].decode("utf-8") if info[12] else ""

        if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            controllable_joints.append(j)

        if joint_name in ["left_finger", "right_finger"]:
            gripper_joints.append(j)

        if "tool" in link_name.lower() or "end" in link_name.lower():
            end_effector_link = j

    if end_effector_link is None:
        end_effector_link = num_joints - 1

    arm_joints = [j for j in controllable_joints if j not in gripper_joints]

    gripper_limits = []
    for joint_idx in gripper_joints:
        info = p.getJointInfo(robot_id, joint_idx)
        lower, upper = info[8], info[9]
        if lower == upper:
            lower, upper = 0.0, 0.04
        open_pos = max(lower, upper)
        closed_pos = min(lower, upper)
        gripper_limits.append({"joint": joint_idx, "open": open_pos, "closed": closed_pos})

    # 启用力传感器
    for joint_idx in gripper_joints:
        p.enableJointForceTorqueSensor(robot_id, joint_idx, enableSensor=True)
        print(f"Enabled force sensor for joint {joint_idx}")

    for joint_idx in gripper_joints:
        p.changeDynamics(robot_id, joint_idx,
                        lateralFriction=2.0,
                        spinningFriction=0.1,
                        rollingFriction=0.1)

    # 设置相机视角
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-30,
                               cameraTargetPosition=[0.2, 0, 0.2])

    # 创建显示元素
    status_id = p.addUserDebugText("Initializing...", [-0.5, 0, 0.95],
                                 textColorRGB=[1, 1, 0], textSize=1.5)

    camera_display = create_camera_display()

    # 创建关节角度显示
    joint_display_anchor = [0.2, -0.4, 0.9]
    joint_display_ids = []
    for i, joint_idx in enumerate(arm_joints):
        info = p.getJointInfo(robot_id, joint_idx)
        joint_name = info[1].decode("utf-8")
        text_id = p.addUserDebugText(
            f"{joint_name}: 0.000rad",
            [joint_display_anchor[0], joint_display_anchor[1], joint_display_anchor[2] - i * 0.05],
            textColorRGB=[0, 0.8, 0.8],
            textSize=1.0
        )
        joint_display_ids.append(text_id)

    # 创建力反馈显示
    force_display_ids = []
    for i, joint_idx in enumerate(gripper_joints):
        text_id = p.addUserDebugText(f"Gripper {i+1}: 0.00N",
                                   [0, 0.1 + i*0.05, 0.8],
                                   textColorRGB=[1, 0, 0], textSize=1.2)
        force_display_ids.append(text_id)

    display_frame = {"count": 0}
    camera_state = {"pos": None, "target": None, "forward": None}
    # 改进相机设置：默认开启，提高更新频率
    camera_enabled_id = p.addUserDebugParameter("Enable RGB Camera (0/1)", 0, 1, 1)
    camera_quality_id = p.addUserDebugParameter("RGB Quality (0=fast,1=clear)", 0, 1, 1)
    assist_constraint_id = p.addUserDebugParameter("Assist Grasp Constraint (0/1)", 0, 1, 1)
    camera_update_interval = 10  # 提高相机更新频率，从60降低到10
    status_update_interval = 6    # 提高状态更新频率
    camera_preview_window_open = {"open": False}

    def update_all_displays(force=False):
        if not p.isConnected():
            return
        display_frame["count"] += 1
        frame = display_frame["count"]
        # 只有在用户开启时才渲染相机，否则不调用 getCameraImage（最影响流畅度）
        camera_enabled = False
        camera_quality = 0.0
        try:
            camera_enabled = p.readUserDebugParameter(camera_enabled_id) > 0.5
            camera_quality = p.readUserDebugParameter(camera_quality_id)
        except Exception:
            camera_enabled = False

        if camera_enabled and (force or frame % camera_update_interval == 0):
            if camera_quality > 0.5:
                cam_w, cam_h = 320, 240
            else:
                cam_w, cam_h = 160, 120
            rgb_image = update_camera_view(
                robot_id,
                end_effector_link,
                camera_state=camera_state,
                camera_width=cam_w,
                camera_height=cam_h,
                smoothing=0.10,
            )
            if rgb_image is not None and p.isConnected():
                p.addUserDebugText("Camera: Active", [0.5, 0.8, 0.9],
                                 textColorRGB=[0, 1, 0], textSize=1.2,
                                 replaceItemUniqueId=camera_display)
                if cv2 is not None:
                    camera_preview_window_open["open"] = True
                    _show_rgb_preview(rgb_image, cam_w, cam_h)
        else:
            # camera disabled: close preview window if it was opened
            if camera_preview_window_open["open"]:
                _close_rgb_preview()
                camera_preview_window_open["open"] = False
        if force or frame % status_update_interval == 0:
            update_joint_display(robot_id, arm_joints, joint_display_ids, joint_display_anchor)
            update_force_display(robot_id, gripper_joints, force_display_ids)

    # 启动时先刷新一次显示
    update_all_displays(force=True)

    print("=== 完整的机械臂抓取演示开始 ===")

    # 步骤0: 回到安全初始位置，确保未接触小球
    print("步骤0: 回到安全初始位置...")
    p.addUserDebugText("Step 0: Moving to safe start", [-0.5, 0, 0.95],
                     textColorRGB=[1, 1, 0], textSize=1.5, replaceItemUniqueId=status_id)
    home_angles = [0.0] * len(arm_joints)
    open_gripper(robot_id, gripper_limits)
    move_arm_to_position(robot_id, home_angles, arm_joints, update_fn=update_all_displays)
    step_simulation(120, update_fn=update_all_displays)

    # 对抓取任务来说，强行锁定末端姿态经常会导致 IK 在接近球时“姿态优先”，从而错过球
    # 因此这里允许 IK 自由选择姿态（target_orn=None）以提高抓取成功率
    fixed_orn = None
    # 注意：tool_offset 目前按“世界坐标”计算一次，姿态变化时不会随工具旋转，会导致目标点偏移
    # 为了让抓取任务稳定跑通，这里先直接用 tool link 作为控制目标（不再对目标点加 tool_offset）
    tool_offset = [0.0, 0.0, 0.0]

    ball_radius = 0.02
    # 放到一个稳定、可达的固定位置（避免“根据当前夹爪位置生成球”，导致球在不理想位置/穿模/太靠近夹爪）
    ball_pos = [0.30, 0.00, ball_radius + 0.06]

    ball_collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
    ball_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=ball_radius, rgbaColor=[0.8, 0.2, 0.2, 1])
    ball_id = p.createMultiBody(
        baseMass=0.03,
        baseCollisionShapeIndex=ball_collision_shape,
        baseVisualShapeIndex=ball_visual_shape,
        basePosition=ball_pos
    )

    p.changeDynamics(ball_id, -1,
                    lateralFriction=1.8,
                    spinningFriction=0.1,
                    rollingFriction=0.1,
                    restitution=0.05,
                    contactDamping=1.0,
                    contactStiffness=300)

    stand_height = max(ball_pos[2] - ball_radius, 0.01)
    if stand_height > 0.01:
        create_ball_stand(ball_pos, stand_height)

    # 两个箱子：箱1（初始放球）+ 箱2（可达范围内的放置目标）
    box1_inner = 0.18
    box2_inner = 0.24  # 放大一点，放置更容易成功
    box1_base_pos = [ball_pos[0], ball_pos[1], 0.0]
    # 放置箱放近一点，保证在可达范围内（原 0.48 对部分初始姿态/IK 容易不可达）
    box2_base_pos = [0.42, 0.12, 0.0]
    create_open_box(base_pos=box1_base_pos, inner_size=(box1_inner, box1_inner), color=(0.7, 0.7, 0.9, 1.0))
    create_open_box(base_pos=box2_base_pos, inner_size=(box2_inner, box2_inner), color=(0.7, 0.9, 0.7, 1.0))
    # GUI 可视化：标记箱1/箱2中心
    p.addUserDebugLine([box1_base_pos[0], box1_base_pos[1], 0.02], [box1_base_pos[0], box1_base_pos[1], 0.25], [0.2, 0.2, 1.0], 2.0)
    p.addUserDebugLine([box2_base_pos[0], box2_base_pos[1], 0.02], [box2_base_pos[0], box2_base_pos[1], 0.25], [0.2, 1.0, 0.2], 2.0)
    step_simulation(120, update_fn=update_all_displays)

    # 分段接近 + 更低的抓取高度，有助于让球进入两指之间
    approach_height = 0.06
    pregrasp_height = 0.025
    grasp_height_offset = -ball_radius * 0.10
    lift_height = 0.08
    box_clearance_height = ball_pos[2] + 0.12

    # 步骤1: 移动到球的上方准备位置
    print("步骤1: 移动到球的上方...")
    p.addUserDebugText("Step 1: Moving above ball", [-0.5, 0, 0.95],
                     textColorRGB=[1, 1, 0], textSize=1.5, replaceItemUniqueId=status_id)

    ball_world_pos, _ = p.getBasePositionAndOrientation(ball_id)
    approach_pos = apply_tool_offset(ball_world_pos, tool_offset, extra_z=approach_height)
    approach_angles = get_joint_angles_for_position(robot_id, approach_pos, target_orn=fixed_orn)
    if approach_angles:
        approach_targets = map_ik_to_controllable(robot_id, approach_angles, arm_joints)
        move_arm_to_position(robot_id, approach_targets,
                             arm_joints, update_fn=update_all_displays)

    step_simulation(120, update_fn=update_all_displays)

    # 步骤2: 张开夹爪
    print("步骤2: 张开夹爪...")
    p.addUserDebugText("Step 2: Opening gripper", [-0.5, 0, 0.95],
                     textColorRGB=[1, 1, 0], textSize=1.5, replaceItemUniqueId=status_id)
    open_gripper(robot_id, gripper_limits)
    step_simulation(120, update_fn=update_all_displays)

    # 步骤3: 分两段下降到球的位置（先到 pregrasp，再到 grasp）
    print("步骤3: 下降到球的位置...")
    p.addUserDebugText("Step 3: Descending to ball", [-0.5, 0, 0.95],
                     textColorRGB=[1, 1, 0], textSize=1.5, replaceItemUniqueId=status_id)

    ball_world_pos, _ = p.getBasePositionAndOrientation(ball_id)
    pregrasp_pos = apply_tool_offset(ball_world_pos, tool_offset, extra_z=pregrasp_height)
    pregrasp_angles = get_joint_angles_for_position(robot_id, pregrasp_pos, target_orn=fixed_orn)
    if pregrasp_angles:
        pregrasp_targets = map_ik_to_controllable(robot_id, pregrasp_angles, arm_joints)
        move_arm_to_position(robot_id, pregrasp_targets, arm_joints, update_fn=update_all_displays)

    step_simulation(90, update_fn=update_all_displays)

    ball_world_pos, _ = p.getBasePositionAndOrientation(ball_id)
    grasp_pos = apply_tool_offset(ball_world_pos, tool_offset, extra_z=grasp_height_offset)
    grasp_angles = get_joint_angles_for_position(robot_id, grasp_pos, target_orn=fixed_orn)
    if grasp_angles:
        grasp_targets = map_ik_to_controllable(robot_id, grasp_angles, arm_joints)
        move_arm_to_position(robot_id, grasp_targets, arm_joints, speed=0.7, update_fn=update_all_displays)

    step_simulation(60, update_fn=update_all_displays)

    # 步骤4: 闭合夹爪抓取
    print("步骤4: 闭合夹爪抓取...")
    p.addUserDebugText("Step 4: Closing gripper to grasp", [-0.5, 0, 0.95],
                     textColorRGB=[1, 1, 0], textSize=1.5, replaceItemUniqueId=status_id)

    attach_constraint = None
    # 1) 先尝试闭合夹爪
    _ = close_gripper(
        robot_id,
        gripper_limits,
        force_threshold=0.8,
        update_fn=update_all_displays,
    )
    # 2) 用接触/最近距离做“是否夹住”的判定（比关节反力可靠）
    grasp_success = has_gripper_contact(
        robot_id,
        ball_id,
        gripper_links=gripper_joints,
        distance_threshold=max(0.01, ball_radius * 0.6),
    )
    # 3) 可选：创建“辅助抓取约束”（关闭可体验纯物理抓取；打开能显著提高放置成功率）
    assist_on = True
    try:
        assist_on = p.readUserDebugParameter(assist_constraint_id) > 0.5
    except Exception:
        assist_on = True
    if assist_on:
        attach_constraint = attempt_attach_ball(
            robot_id, ball_id, end_effector_link, gripper_joints,
            distance_threshold=max(0.02, ball_radius * 1.2)
        )
    if attach_constraint is not None:
        grasp_success = True

    if grasp_success:
        print("抓取成功！")
        p.addUserDebugText("Step 4: GRASP SUCCESSFUL!", [-0.5, 0, 0.95],
                         textColorRGB=[0, 1, 0], textSize=1.5, replaceItemUniqueId=status_id)
    else:
        print("抓取失败 - 继续演示流程")
        p.addUserDebugText("Step 4: Continuing demo...", [-0.5, 0, 0.95],
                         textColorRGB=[1, 0.5, 0], textSize=1.5, replaceItemUniqueId=status_id)

    step_simulation(240, update_fn=update_all_displays)

    # 步骤5: 抬起球
    print("步骤5: 抬起球...")
    p.addUserDebugText("Step 5: Lifting ball", [-0.5, 0, 0.95],
                     textColorRGB=[1, 1, 0], textSize=1.5, replaceItemUniqueId=status_id)

    # 抬升用“当前末端位置 + 向上”更稳，避免用球位置反算导致抬不起来
    tool_state = p.getLinkState(robot_id, end_effector_link, computeForwardKinematics=True)
    tool_pos = tool_state[0]
    lift_pos = [tool_pos[0], tool_pos[1], tool_pos[2] + lift_height]
    lift_angles = get_joint_angles_for_position(robot_id, lift_pos, target_orn=fixed_orn)
    if lift_angles:
        lift_targets = map_ik_to_controllable(robot_id, lift_angles, arm_joints)
        move_arm_to_position(robot_id, lift_targets,
                             arm_joints, update_fn=update_all_displays)

    step_simulation(120, update_fn=update_all_displays)

    # 步骤6: 移动到箱子上方
    print("步骤6: 移动到箱子上方...")
    p.addUserDebugText("Step 6: Moving above box", [-0.5, 0, 0.95],
                     textColorRGB=[1, 1, 0], textSize=1.5, replaceItemUniqueId=status_id)

    box_above_center = [box2_base_pos[0], box2_base_pos[1], box_clearance_height]
    box_above_pos = apply_tool_offset(box_above_center, tool_offset)
    new_angles = get_joint_angles_for_position(robot_id, box_above_pos, target_orn=fixed_orn)
    if new_angles:
        new_targets = map_ik_to_controllable(robot_id, new_angles, arm_joints)
        move_arm_to_position(robot_id, new_targets,
                             arm_joints, update_fn=update_all_displays)
    else:
        print(f"[WARN] IK failed for box_above_pos={box_above_pos}")

    step_simulation(120, update_fn=update_all_displays)

    # 步骤7: 下降到箱子内部
    print("步骤7: 下降到箱子内部...")
    p.addUserDebugText("Step 7: Lowering into box", [-0.5, 0, 0.95],
                     textColorRGB=[1, 1, 0], textSize=1.5, replaceItemUniqueId=status_id)

    # 放置采用两段下降：先到箱口附近，再到箱内
    box_place_center_hi = [box2_base_pos[0], box2_base_pos[1], 0.20]
    box_place_pos_hi = apply_tool_offset(box_place_center_hi, tool_offset)
    place_hi_angles = get_joint_angles_for_position(robot_id, box_place_pos_hi, target_orn=fixed_orn)
    if place_hi_angles:
        place_hi_targets = map_ik_to_controllable(robot_id, place_hi_angles, arm_joints)
        move_arm_to_position(robot_id, place_hi_targets, arm_joints, speed=0.8, update_fn=update_all_displays)
        step_simulation(60, update_fn=update_all_displays)
    else:
        print(f"[WARN] IK failed for box_place_pos_hi={box_place_pos_hi}")

    # 箱内放置高度：箱底上方一点，确保球会掉进箱内
    box_place_center_lo = [box2_base_pos[0], box2_base_pos[1], ball_radius + 0.05]
    box_place_pos_lo = apply_tool_offset(box_place_center_lo, tool_offset)
    place_lo_angles = get_joint_angles_for_position(robot_id, box_place_pos_lo, target_orn=fixed_orn)
    if place_lo_angles:
        place_lo_targets = map_ik_to_controllable(robot_id, place_lo_angles, arm_joints)
        move_arm_to_position(robot_id, place_lo_targets, arm_joints, speed=0.6, update_fn=update_all_displays)
    else:
        print(f"[WARN] IK failed for box_place_pos_lo={box_place_pos_lo}")

    step_simulation(60, update_fn=update_all_displays)

    # 步骤8: 放下球
    print("步骤8: 放下球...")
    p.addUserDebugText("Step 8: Releasing ball", [-0.5, 0, 0.95],
                     textColorRGB=[1, 1, 0], textSize=1.5, replaceItemUniqueId=status_id)

    def ball_in_box2():
        pos, _ = p.getBasePositionAndOrientation(ball_id)
        dx = pos[0] - box2_base_pos[0]
        dy = pos[1] - box2_base_pos[1]
        half_inner = box2_inner / 2.0
        return (abs(dx) < half_inner) and (abs(dy) < half_inner), pos

    # 纯物理放置：不允许 teleport。若没放进箱2，则尝试“再抓一次再放一次”（最多 2 次）
    max_place_attempts = 2
    placed = False
    for attempt in range(max_place_attempts):
        open_gripper(robot_id, gripper_limits)
        step_simulation(60, update_fn=update_all_displays)
        if attach_constraint is not None:
            p.removeConstraint(attach_constraint)
            attach_constraint = None

        # 松开后先抬高，避免末端/夹爪碰撞把球打飞
        tool_state = p.getLinkState(robot_id, end_effector_link, computeForwardKinematics=True)
        tool_pos = tool_state[0]
        retreat_pos = [tool_pos[0], tool_pos[1], tool_pos[2] + 0.12]
        retreat_angles = get_joint_angles_for_position(robot_id, retreat_pos, target_orn=fixed_orn)
        if retreat_angles:
            retreat_targets = map_ik_to_controllable(robot_id, retreat_angles, arm_joints)
            move_arm_to_position(robot_id, retreat_targets, arm_joints, speed=0.9, update_fn=update_all_displays)

        step_simulation(240, update_fn=update_all_displays)
        in_box2, ball_after_pos = ball_in_box2()
        if in_box2:
            placed = True
            print("放置成功：小球已进入箱2。")
            break

        print(f"[WARN] Place attempt {attempt+1} failed. ball={ball_after_pos}, retrying...")

        # 重试：回到球上方 -> 下探 -> 夹紧 -> (可选) 约束 -> 再回到箱2放置点
        ball_world_pos, _ = p.getBasePositionAndOrientation(ball_id)
        re_approach = apply_tool_offset(ball_world_pos, tool_offset, extra_z=approach_height)
        re_angles = get_joint_angles_for_position(robot_id, re_approach, target_orn=fixed_orn)
        if re_angles:
            re_targets = map_ik_to_controllable(robot_id, re_angles, arm_joints)
            move_arm_to_position(robot_id, re_targets, arm_joints, speed=0.9, update_fn=update_all_displays)
        open_gripper(robot_id, gripper_limits)
        step_simulation(60, update_fn=update_all_displays)

        re_grasp = apply_tool_offset(ball_world_pos, tool_offset, extra_z=grasp_height_offset)
        re_grasp_angles = get_joint_angles_for_position(robot_id, re_grasp, target_orn=fixed_orn)
        if re_grasp_angles:
            re_grasp_targets = map_ik_to_controllable(robot_id, re_grasp_angles, arm_joints)
            move_arm_to_position(robot_id, re_grasp_targets, arm_joints, speed=0.7, update_fn=update_all_displays)
        step_simulation(60, update_fn=update_all_displays)

        _ = close_gripper(robot_id, gripper_limits, force_threshold=0.8, update_fn=update_all_displays)
        if assist_on:
            attach_constraint = attempt_attach_ball(
                robot_id, ball_id, end_effector_link, gripper_joints,
                distance_threshold=max(0.02, ball_radius * 1.2)
            )

        # 回到箱2上方与箱内位置（复用已计算的 box_above_pos/box_place_pos_hi/box_place_pos_lo）
        new_angles = get_joint_angles_for_position(robot_id, box_above_pos, target_orn=fixed_orn)
        if new_angles:
            new_targets = map_ik_to_controllable(robot_id, new_angles, arm_joints)
            move_arm_to_position(robot_id, new_targets, arm_joints, speed=0.9, update_fn=update_all_displays)
        place_hi_angles = get_joint_angles_for_position(robot_id, box_place_pos_hi, target_orn=fixed_orn)
        if place_hi_angles:
            place_hi_targets = map_ik_to_controllable(robot_id, place_hi_angles, arm_joints)
            move_arm_to_position(robot_id, place_hi_targets, arm_joints, speed=0.8, update_fn=update_all_displays)
        place_lo_angles = get_joint_angles_for_position(robot_id, box_place_pos_lo, target_orn=fixed_orn)
        if place_lo_angles:
            place_lo_targets = map_ik_to_controllable(robot_id, place_lo_angles, arm_joints)
            move_arm_to_position(robot_id, place_lo_targets, arm_joints, speed=0.6, update_fn=update_all_displays)

    if not placed:
        print("[WARN] Place did not succeed after retries (no teleport). Consider enabling Assist Grasp Constraint.")

    # 步骤9: 返回初始位置
    print("步骤9: 返回初始位置...")
    p.addUserDebugText("Step 9: Returning to home", [-0.5, 0, 0.95],
                     textColorRGB=[1, 1, 0], textSize=1.5, replaceItemUniqueId=status_id)

    move_arm_to_position(robot_id, home_angles, arm_joints, update_fn=update_all_displays)

    p.addUserDebugText("Demo Complete! Close window to exit.", [-0.5, 0, 0.95],
                     textColorRGB=[0, 1, 0], textSize=1.5, replaceItemUniqueId=status_id)

    print("演示完成！现在进入实时监控模式...")

    # 实时监控模式
    frame_count = 0
    while p.isConnected():
        frame_count += 1
        update_all_displays()
        try:
            if not p.isConnected():
                break
            p.stepSimulation()
        except Exception:
            break
        if SLEEP_SCALE > 0:
            time.sleep((1.0 / SIM_HZ) * SLEEP_SCALE)

    try:
        if p.isConnected():
            p.disconnect()
    except Exception:
        pass
    _close_rgb_preview()


if __name__ == "__main__":
    main()
