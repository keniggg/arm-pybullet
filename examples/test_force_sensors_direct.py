#!/usr/bin/env python3
"""
直接测试力传感器的简单脚本 - 让夹爪接触固定物体
"""

import pybullet as p
import time
import os

def test_force_sensors_direct():
    # 连接PyBullet (DIRECT模式，无GUI)
    physics_client = p.connect(p.DIRECT)
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)

    # 改变工作目录到项目根目录
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(repo_root)

    # 加载机械臂
    urdf_path = "synriard/urdf/Alicia_D_v5_6/Alicia_D_v5_6_gripper_50mm.urdf"
    p.setAdditionalSearchPath("synriard")
    p.setAdditionalSearchPath("synriard/meshes")

    flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_MAINTAIN_LINK_ORDER
    robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True, flags=flags)

    # 创建一个固定的障碍物让夹爪接触
    box_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.1, 0.02])
    box_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.1, 0.02], rgbaColor=[0.2, 0.8, 0.2, 1])

    box_pos = [0.285, 0, 0.14]  # 进一步靠近夹爪
    box_id = p.createMultiBody(
        baseMass=0,  # 固定物体
        baseCollisionShapeIndex=box_collision_shape,
        baseVisualShapeIndex=box_visual_shape,
        basePosition=box_pos
    )

    # 启用夹爪力传感器
    gripper_joints = []
    num_joints = p.getNumJoints(robot_id)
    for j in range(num_joints):
        info = p.getJointInfo(robot_id, j)
        joint_name = info[1].decode("utf-8")
        if joint_name in ["left_finger", "right_finger"]:
            gripper_joints.append(j)
            p.enableJointForceTorqueSensor(robot_id, j, enableSensor=True)
            print(f"Enabled force sensor for joint {j}: {joint_name}")

    # 移动机械臂到障碍物位置
    print("移动机械臂到障碍物位置...")
    arm_joint_angles = [0, 0.2, 0.5, 0, 1.0, 0]
    for i, angle in enumerate(arm_joint_angles):
        p.setJointMotorControl2(robot_id, i+1, p.POSITION_CONTROL, targetPosition=angle, force=200)

    # 检查夹爪末端位置
    print("检查夹爪位置...")
    link_state = p.getLinkState(robot_id, gripper_joints[0])  # 检查左夹爪的位置
    left_finger_pos = link_state[0]
    print(f"左夹爪位置: {left_finger_pos}")

    link_state = p.getLinkState(robot_id, gripper_joints[1])  # 检查右夹爪的位置
    right_finger_pos = link_state[0]
    print(f"右夹爪位置: {right_finger_pos}")
    print(f"障碍物位置: {box_pos}")

    # 计算距离
    import math
    dist_left = math.sqrt(sum((a-b)**2 for a, b in zip(left_finger_pos, box_pos)))
    dist_right = math.sqrt(sum((a-b)**2 for a, b in zip(right_finger_pos, box_pos)))
    print(f"左夹爪到障碍物距离: {dist_left:.3f}")
    print(f"右夹爪到障碍物距离: {dist_right:.3f}")

    # 测试1: 夹爪打开状态
    print("\n=== 测试1: 夹爪打开 (无接触) ===")
    for joint_idx in gripper_joints:
        p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL, targetPosition=0.05, force=50)

    for i in range(30):  # 3秒
        p.stepSimulation()
        time.sleep(0.1)

    # 读取力传感器数据
    print("夹爪打开时的力:")
    for j, joint_idx in enumerate(gripper_joints):
        joint_state = p.getJointState(robot_id, joint_idx)
        forces = joint_state[2]
        if forces:
            fx, fy, fz, mx, my, mz = forces
            total_force = (fx**2 + fy**2 + fz**2)**0.5
            print(f"  Gripper {j+1}: 合力={total_force:.3f}N (Fx={fx:.3f}, Fy={fy:.3f}, Fz={fz:.3f})")

    # 测试2: 慢慢关闭夹爪直到接触
    print("\n=== 测试2: 慢慢关闭夹爪 (接触测试) ===")
    contact_detected = False
    force_threshold = 2.0  # 检测接触的力阈值

    for close_step in range(0, 50, 2):  # 逐渐关闭
        target_pos = 0.05 - (close_step * 0.001)  # 逐渐减小开口
        for joint_idx in gripper_joints:
            p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL, targetPosition=max(0, target_pos), force=50)

        # 等待运动
        for _ in range(10):
            p.stepSimulation()
            time.sleep(0.01)

        # 检查力
        max_force = 0
        for j, joint_idx in enumerate(gripper_joints):
            joint_state = p.getJointState(robot_id, joint_idx)
            forces = joint_state[2]
            if forces:
                fx, fy, fz, mx, my, mz = forces
                total_force = (fx**2 + fy**2 + fz**2)**0.5
                max_force = max(max_force, total_force)

        if max_force > force_threshold and not contact_detected:
            print(f"检测到接触！步数: {close_step}, 最大力: {max_force:.3f}N")
            contact_detected = True

        if close_step % 10 == 0:
            print(f"步数 {close_step}: 最大力 = {max_force:.3f}N")

    # 测试3: 完全关闭夹爪
    print("\n=== 测试3: 完全关闭夹爪 ===")
    for joint_idx in gripper_joints:
        p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL, targetPosition=0.0, force=50)

    for i in range(50):  # 5秒
        p.stepSimulation()
        time.sleep(0.1)

    print("完全关闭后的力:")
    for j, joint_idx in enumerate(gripper_joints):
        joint_state = p.getJointState(robot_id, joint_idx)
        forces = joint_state[2]
        if forces:
            fx, fy, fz, mx, my, mz = forces
            total_force = (fx**2 + fy**2 + fz**2)**0.5
            print(f"  Gripper {j+1}: 合力={total_force:.3f}N (Fx={fx:.3f}, Fy={fy:.3f}, Fz={fz:.3f})")

    p.disconnect()
    print("\n测试完成！")

if __name__ == "__main__":
    test_force_sensors_direct()