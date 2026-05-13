#!/usr/bin/env python3
"""
简化的机械臂抓取演示 - 专注于力反馈展示
"""

import os
import time
import math
import pybullet as p
import pybullet_data


def create_force_feedback_demo():
    """创建力反馈演示程序"""
    # 改变工作目录
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

    meshes_root = "synriard/meshes"
    if os.path.isdir(meshes_root):
        for root, dirs, files in os.walk(meshes_root):
            p.setAdditionalSearchPath(root)

    flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_MAINTAIN_LINK_ORDER
    robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True, flags=flags)

    # 创建测试对象 - 一个可以抓取的方块
    box_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.03])
    box_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.03], rgbaColor=[0.2, 0.8, 0.2, 1])

    box_pos = [0.35, 0, 0.08]  # 放在机械臂前方
    box_id = p.createMultiBody(
        baseMass=0.05,  # 轻质量便于抓取
        baseCollisionShapeIndex=box_collision_shape,
        baseVisualShapeIndex=box_visual_shape,
        basePosition=box_pos
    )

    # 设置物体的物理属性
    p.changeDynamics(box_id, -1,
                    lateralFriction=2.0,
                    spinningFriction=0.1,
                    rollingFriction=0.1,
                    restitution=0.1)

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
    p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=45, cameraPitch=-25,
                               cameraTargetPosition=[0.2, 0, 0.15])

    # 创建状态显示
    status_id = p.addUserDebugText("Force Feedback Demo Ready", [-0.5, 0, 0.95],
                                 textColorRGB=[0, 1, 0], textSize=1.5)

    # 创建关节角度显示
    joint_display_ids = []
    for i, joint_idx in enumerate(controllable_joints[:6]):  # 只显示前6个关节
        info = p.getJointInfo(robot_id, joint_idx)
        joint_name = info[1].decode("utf-8")[:8]  # 截断长名称
        text_id = p.addUserDebugText(f"{joint_name}: 0.00°",
                                   [-0.9, 0.8 - i*0.06, 0.7],
                                   textColorRGB=[0, 0.8, 0.8], textSize=1.0)
        joint_display_ids.append(text_id)

    # 创建力反馈显示
    force_display_ids = []
    for i, joint_idx in enumerate(gripper_joints):
        text_id = p.addUserDebugText(f"Gripper {i+1}: 0.00N (NO CONTACT)",
                                   [-0.2, 0.2 + i*0.08, 0.8],
                                   textColorRGB=[1, 0, 0], textSize=1.4)
        force_display_ids.append(text_id)

    # 创建控制滑块
    slider_ids = []
    for i, joint_idx in enumerate(controllable_joints[:6]):  # 只为前6个关节创建滑块
        info = p.getJointInfo(robot_id, joint_idx)
        joint_name = info[1].decode("utf-8")
        low, high = info[8], info[9]

        if low >= high or (abs(low) < 1e-6 and abs(high) < 1e-6):
            low, high = -3.14159, 3.14159

        sid = p.addUserDebugParameter(f"{joint_name[:10]}", low, high, 0.0)
        slider_ids.append((joint_idx, sid))

    print("=== 力反馈演示程序启动 ===")
    print("使用滑块控制机械臂关节，观察力反馈传感器读数变化")
    print("绿色物体可以被夹爪抓取，观察力值的实时变化")

    frame_count = 0
    while p.isConnected():
        frame_count += 1

        # 读取滑块值并控制关节
        try:
            for joint_idx, sid in slider_ids:
                val = p.readUserDebugParameter(sid)
                p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL,
                                      targetPosition=val, force=150)
        except:
            break

        # 每10帧更新关节角度显示
        if frame_count % 10 == 0:
            for i, joint_idx in enumerate(controllable_joints[:6]):
                try:
                    joint_state = p.getJointState(robot_id, joint_idx)
                    angle = joint_state[0]
                    angle_deg = math.degrees(angle)

                    info = p.getJointInfo(robot_id, joint_idx)
                    joint_name = info[1].decode("utf-8")[:8]

                    color = [0, 0.8, 0.8]  # 青色
                    if abs(angle_deg) > 45:  # 大角度显示为黄色
                        color = [1, 1, 0]

                    p.addUserDebugText(f"{joint_name}: {angle_deg:.1f}°",
                                     [-0.9, 0.8 - i*0.06, 0.7],
                                     textColorRGB=color,
                                     textSize=1.0,
                                     replaceItemUniqueId=joint_display_ids[i])
                except Exception as e:
                    print(f"Joint display error: {e}")

        # 每5帧更新力反馈显示
        if frame_count % 5 == 0:
            for i, joint_idx in enumerate(gripper_joints):
                try:
                    joint_state = p.getJointState(robot_id, joint_idx)
                    forces = joint_state[2]

                    if forces:
                        fx, fy, fz, mx, my, mz = forces
                        total_force = math.sqrt(fx**2 + fy**2 + fz**2)

                        # 根据力的大小改变颜色和状态
                        if total_force > 3.0:
                            color = [0, 1, 0]  # 绿色 - 强抓取
                            status = "STRONG GRASP"
                        elif total_force > 1.0:
                            color = [1, 0.5, 0]  # 橙色 - 中等接触
                            status = "CONTACT"
                        elif total_force > 0.1:
                            color = [1, 1, 0]  # 黄色 - 轻微接触
                            status = "LIGHT TOUCH"
                        else:
                            color = [1, 0, 0]  # 红色 - 无接触
                            status = "NO CONTACT"

                        p.addUserDebugText(f"Gripper {i+1}: {total_force:.2f}N ({status})",
                                         [-0.2, 0.2 + i*0.08, 0.8],
                                         textColorRGB=color,
                                         textSize=1.4,
                                         replaceItemUniqueId=force_display_ids[i])
                    else:
                        p.addUserDebugText(f"Gripper {i+1}: No sensor data",
                                         [-0.2, 0.2 + i*0.08, 0.8],
                                         textColorRGB=[0.5, 0.5, 0.5],
                                         textSize=1.4,
                                         replaceItemUniqueId=force_display_ids[i])
                except Exception as e:
                    print(f"Force display error: {e}")

        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    p.disconnect()


if __name__ == "__main__":
    create_force_feedback_demo()