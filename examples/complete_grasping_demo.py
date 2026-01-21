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


def get_joint_angles_for_position(robot_id, target_pos, target_orn=None):
    """
    计算逆运动学，获取到达目标位置所需的关节角度
    """
    if target_orn is None:
        target_orn = p.getQuaternionFromEuler([0, math.pi/2, 0])  # 默认姿态

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


def move_arm_to_position(robot_id, target_angles, controllable_joints, speed=1.0, update_fn=None):
    """
    平滑移动机械臂到指定关节角度
    """
    # 获取当前关节角度
    current_angles = []
    for joint_idx in controllable_joints:
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
        time.sleep(1.0 / 240.0)


def step_simulation(steps, update_fn=None):
    """运行指定步数的仿真并刷新显示。"""
    for _ in range(steps):
        if update_fn:
            update_fn()
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


def open_gripper(robot_id, gripper_joints, target_opening=0.08):
    """张开夹爪"""
    for joint_idx in gripper_joints:
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_opening,
            force=50
        )


def close_gripper(robot_id, gripper_joints, force_threshold=2.0, update_fn=None):
    """闭合夹爪直到检测到足够大的力 - 更保守的阈值"""
    max_force = 0
    step = 0
    max_steps = 200  # 增加最大步数

    # 首先快速闭合到接近球的位置
    for quick_step in range(30):
        target_pos = max(0.02, 0.08 - quick_step * 0.002)  # 快速接近
        for joint_idx in gripper_joints:
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=30  # 较小的力
            )
        if update_fn:
            update_fn()
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    # 然后慢慢增加力进行抓取
    while max_force < force_threshold and step < max_steps:
        # 逐渐减小夹爪开口
        target_pos = max(0, 0.02 - step * 0.0002)

        for joint_idx in gripper_joints:
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=80  # 增加抓取力
            )

        # 检查力传感器
        max_force = 0
        for joint_idx in gripper_joints:
            joint_state = p.getJointState(robot_id, joint_idx)
            forces = joint_state[2]
            if forces:
                fx, fy, fz, mx, my, mz = forces
                total_force = math.sqrt(fx**2 + fy**2 + fz**2)
                max_force = max(max_force, total_force)

        if update_fn:
            update_fn()
        p.stepSimulation()
        time.sleep(1.0 / 240.0)
        step += 1

    return max_force >= force_threshold


def attempt_attach_ball(robot_id, ball_id, end_effector_link, gripper_joints):
    """检测夹爪与小球接触后，创建约束辅助抓取。"""
    contacts = []
    for joint_idx in gripper_joints:
        contacts.extend(p.getContactPoints(bodyA=robot_id, bodyB=ball_id, linkIndexA=joint_idx))

    if not contacts:
        return None

    constraint_id = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=end_effector_link,
        childBodyUniqueId=ball_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0]
    )
    return constraint_id


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


def update_camera_view(robot_id, end_effector_link, camera_width=320, camera_height=240):
    """更新眼在手摄像头视图"""
    try:
        # 获取末端link状态
        link_state = p.getLinkState(robot_id, end_effector_link)
        camera_pos = link_state[0]
        camera_orn = link_state[1]

        # 计算摄像头朝向
        camera_rot_matrix = p.getMatrixFromQuaternion(camera_orn)
        camera_target = [
            camera_pos[0] + camera_rot_matrix[0] * 0.3,
            camera_pos[1] + camera_rot_matrix[3] * 0.3,
            camera_pos[2] + camera_rot_matrix[6] * 0.3
        ]

        # 设置摄像头视图
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=camera_target,
            cameraUpVector=[0, 0, 1]
        )

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=camera_width/camera_height,
            nearVal=0.01,
            farVal=2.0
        )

        # 获取RGB图像
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


def update_joint_display(robot_id, controllable_joints, display_ids):
    """更新关节角度显示"""
    for i, joint_idx in enumerate(controllable_joints):
        try:
            joint_state = p.getJointState(robot_id, joint_idx)
            angle = joint_state[0]
            velocity = joint_state[1]

            info = p.getJointInfo(robot_id, joint_idx)
            joint_name = info[1].decode("utf-8")

            p.addUserDebugText(
                f"{joint_name}: {angle:.3f}rad ({math.degrees(angle):.1f}°)",
                [-0.8, 0.8 - i*0.05, 0.8],
                textColorRGB=[0, 0.8, 0.8],
                textSize=1.0,
                replaceItemUniqueId=display_ids[i]
            )
        except Exception as e:
            print(f"Joint display update failed: {e}")


def update_force_display(robot_id, gripper_joints, force_display_ids):
    """更新力反馈显示"""
    for i, joint_idx in enumerate(gripper_joints):
        try:
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

    # 连接PyBullet GUI
    physics_client = p.connect(p.GUI)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)

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

    # 创建柔软小球
    ball_radius = 0.05
    ball_collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
    ball_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=ball_radius, rgbaColor=[0.8, 0.2, 0.2, 1])

    ball_pos = [0.4, 0, 0.15]  # 球的位置
    ball_id = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=ball_collision_shape,
        baseVisualShapeIndex=ball_visual_shape,
        basePosition=ball_pos
    )

    # 设置球的柔软物理属性
    p.changeDynamics(ball_id, -1,
                    lateralFriction=1.5,
                    spinningFriction=0.1,
                    rollingFriction=0.1,
                    restitution=0.1,
                    contactDamping=1.0,
                    contactStiffness=300)

    # 创建目标箱子
    box_base_pos = [0.1, 0.3, 0.0]
    create_open_box(base_pos=box_base_pos)

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

    # 启用力传感器
    for joint_idx in gripper_joints:
        p.enableJointForceTorqueSensor(robot_id, joint_idx, enableSensor=True)
        print(f"Enabled force sensor for joint {joint_idx}")

    # 设置相机视角
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-30,
                               cameraTargetPosition=[0.2, 0, 0.2])

    # 创建显示元素
    status_id = p.addUserDebugText("Initializing...", [-0.5, 0, 0.95],
                                 textColorRGB=[1, 1, 0], textSize=1.5)

    camera_display = create_camera_display()

    # 创建关节角度显示
    joint_display_ids = []
    for i, joint_idx in enumerate(controllable_joints):
        info = p.getJointInfo(robot_id, joint_idx)
        joint_name = info[1].decode("utf-8")
        text_id = p.addUserDebugText(f"{joint_name}: 0.000rad",
                                   [-0.8, 0.8 - i*0.05, 0.8],
                                   textColorRGB=[0, 0.8, 0.8], textSize=1.0)
        joint_display_ids.append(text_id)

    # 创建力反馈显示
    force_display_ids = []
    for i, joint_idx in enumerate(gripper_joints):
        text_id = p.addUserDebugText(f"Gripper {i+1}: 0.00N",
                                   [0, 0.1 + i*0.05, 0.8],
                                   textColorRGB=[1, 0, 0], textSize=1.2)
        force_display_ids.append(text_id)

    display_frame = {"count": 0}

    def update_all_displays():
        if not p.isConnected():
            return
        display_frame["count"] += 1
        if display_frame["count"] % 5 == 0:
            rgb_image = update_camera_view(robot_id, end_effector_link)
            if rgb_image is not None:
                p.addUserDebugText("Camera: Active", [0.5, 0.8, 0.9],
                                 textColorRGB=[0, 1, 0], textSize=1.2,
                                 replaceItemUniqueId=camera_display)
            update_joint_display(robot_id, controllable_joints, joint_display_ids)
            update_force_display(robot_id, gripper_joints, force_display_ids)

    # 启动时先刷新一次显示
    update_all_displays()

    print("=== 完整的机械臂抓取演示开始 ===")

    # 步骤0: 回到安全初始位置，确保未接触小球
    print("步骤0: 回到安全初始位置...")
    p.addUserDebugText("Step 0: Moving to safe start", [-0.5, 0, 0.95],
                     textColorRGB=[1, 1, 0], textSize=1.5, replaceItemUniqueId=status_id)
    home_angles = [0.0] * len(controllable_joints)
    open_gripper(robot_id, gripper_joints)
    move_arm_to_position(robot_id, home_angles, controllable_joints)
    time.sleep(1)

    # 步骤1: 移动到球的上方准备位置
    print("步骤1: 移动到球的上方...")
    p.addUserDebugText("Step 1: Moving above ball", [-0.5, 0, 0.95],
                     textColorRGB=[1, 1, 0], textSize=1.5, replaceItemUniqueId=status_id)

    approach_pos = [ball_pos[0], ball_pos[1], ball_pos[2] + 0.1]  # 在球上方10cm
    approach_angles = get_joint_angles_for_position(robot_id, approach_pos)
    if approach_angles:
        move_arm_to_position(robot_id, approach_angles[:len(controllable_joints)],
                             controllable_joints, update_fn=update_all_displays)

    step_simulation(120, update_fn=update_all_displays)

    # 步骤2: 张开夹爪
    print("步骤2: 张开夹爪...")
    p.addUserDebugText("Step 2: Opening gripper", [-0.5, 0, 0.95],
                     textColorRGB=[1, 1, 0], textSize=1.5, replaceItemUniqueId=status_id)
    open_gripper(robot_id, gripper_joints)
    step_simulation(120, update_fn=update_all_displays)

    # 步骤3: 下降到球的位置
    print("步骤3: 下降到球的位置...")
    p.addUserDebugText("Step 3: Descending to ball", [-0.5, 0, 0.95],
                     textColorRGB=[1, 1, 0], textSize=1.5, replaceItemUniqueId=status_id)

    # 稍微高于球的位置，让夹爪更容易接触
    grasp_pos = [ball_pos[0], ball_pos[1], ball_pos[2] + 0.03]  # 3cm above ball
    grasp_angles = get_joint_angles_for_position(robot_id, grasp_pos)
    if grasp_angles:
        move_arm_to_position(robot_id, grasp_angles[:len(controllable_joints)],
                             controllable_joints, update_fn=update_all_displays)

    step_simulation(60, update_fn=update_all_displays)

    # 步骤4: 闭合夹爪抓取
    print("步骤4: 闭合夹爪抓取...")
    p.addUserDebugText("Step 4: Closing gripper to grasp", [-0.5, 0, 0.95],
                     textColorRGB=[1, 1, 0], textSize=1.5, replaceItemUniqueId=status_id)

    attach_constraint = None
    grasp_success = close_gripper(robot_id, gripper_joints, force_threshold=1.5,
                                  update_fn=update_all_displays)  # 降低阈值
    attach_constraint = attempt_attach_ball(robot_id, ball_id, end_effector_link, gripper_joints)
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

    lift_pos = [ball_pos[0], ball_pos[1], ball_pos[2] + 0.15]  # 抬起到15cm高度
    lift_angles = get_joint_angles_for_position(robot_id, lift_pos)
    if lift_angles:
        move_arm_to_position(robot_id, lift_angles[:len(controllable_joints)],
                             controllable_joints, update_fn=update_all_displays)

    step_simulation(120, update_fn=update_all_displays)

    # 步骤6: 移动到箱子上方
    print("步骤6: 移动到箱子上方...")
    p.addUserDebugText("Step 6: Moving above box", [-0.5, 0, 0.95],
                     textColorRGB=[1, 1, 0], textSize=1.5, replaceItemUniqueId=status_id)

    box_above_pos = [box_base_pos[0], box_base_pos[1], 0.25]
    new_angles = get_joint_angles_for_position(robot_id, box_above_pos)
    if new_angles:
        move_arm_to_position(robot_id, new_angles[:len(controllable_joints)],
                             controllable_joints, update_fn=update_all_displays)

    step_simulation(120, update_fn=update_all_displays)

    # 步骤7: 下降到箱子内部
    print("步骤7: 下降到箱子内部...")
    p.addUserDebugText("Step 7: Lowering into box", [-0.5, 0, 0.95],
                     textColorRGB=[1, 1, 0], textSize=1.5, replaceItemUniqueId=status_id)

    box_place_pos = [box_base_pos[0], box_base_pos[1], ball_radius + 0.02]
    place_angles = get_joint_angles_for_position(robot_id, box_place_pos)
    if place_angles:
        move_arm_to_position(robot_id, place_angles[:len(controllable_joints)],
                             controllable_joints, update_fn=update_all_displays)

    step_simulation(60, update_fn=update_all_displays)

    # 步骤8: 放下球
    print("步骤8: 放下球...")
    p.addUserDebugText("Step 8: Releasing ball", [-0.5, 0, 0.95],
                     textColorRGB=[1, 1, 0], textSize=1.5, replaceItemUniqueId=status_id)

    open_gripper(robot_id, gripper_joints)
    if attach_constraint is not None:
        p.removeConstraint(attach_constraint)
    step_simulation(240, update_fn=update_all_displays)

    # 步骤9: 返回初始位置
    print("步骤9: 返回初始位置...")
    p.addUserDebugText("Step 9: Returning to home", [-0.5, 0, 0.95],
                     textColorRGB=[1, 1, 0], textSize=1.5, replaceItemUniqueId=status_id)

    move_arm_to_position(robot_id, home_angles, controllable_joints, update_fn=update_all_displays)

    p.addUserDebugText("Demo Complete! Close window to exit.", [-0.5, 0, 0.95],
                     textColorRGB=[0, 1, 0], textSize=1.5, replaceItemUniqueId=status_id)

    print("演示完成！现在进入实时监控模式...")

    # 实时监控模式
    frame_count = 0
    while p.isConnected():
        frame_count += 1
        if frame_count % 5 == 0:
            update_all_displays()
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    p.disconnect()


if __name__ == "__main__":
    main()
