#!/usr/bin/env python3
"""
测试力反馈传感器的简单脚本
"""

import pybullet as p
import time
import os

def test_force_sensors():
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

    # 创建测试球 - 放在更容易抓取的位置
    ball_radius = 0.05
    ball_collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
    ball_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=ball_radius, rgbaColor=[0.8, 0.2, 0.2, 1])

    ball_pos = [0.4, 0, 0.15]  # 调整位置让球更容易被抓取
    ball_id = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=ball_collision_shape,
        baseVisualShapeIndex=ball_visual_shape,
        basePosition=ball_pos
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

    # 首先移动机械臂到球的位置
    print("移动机械臂到球的位置...")
    # 设置关节角度让夹爪靠近球 - 更精确的位置
    arm_joint_angles = [0, 0.2, 0.5, 0, 1.0, 0]  # 调整角度让夹爪更靠近球
    for i, angle in enumerate(arm_joint_angles):
        p.setJointMotorControl2(robot_id, i+1, p.POSITION_CONTROL, targetPosition=angle, force=200)

    # 等待机械臂移动到位
    for _ in range(100):  # 更长时间等待
        p.stepSimulation()
        time.sleep(0.02)

    # 检查当前位置
    print("机械臂位置就绪，开始测试...")

    # 打开夹爪
    for joint_idx in gripper_joints:
        p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL, targetPosition=0.05, force=50)

    # 测试10秒 - 夹爪打开状态
    for i in range(100):  # 100 * 0.1s = 10s
        p.stepSimulation()

        if i % 10 == 0:  # 每秒打印一次
            print(f"\n时间: {i/10:.1f}s (夹爪打开)")
            for j, joint_idx in enumerate(gripper_joints):
                try:
                    joint_state = p.getJointState(robot_id, joint_idx)
                    forces = joint_state[2]
                    if forces:
                        fx, fy, fz, mx, my, mz = forces
                        total_force = (fx**2 + fy**2 + fz**2)**0.5
                        print(f"  Gripper {j+1}: 合力={total_force:.2f}N (Fx={fx:.2f}, Fy={fy:.2f}, Fz={fz:.2f})")
                    else:
                        print(f"  Gripper {j+1}: 无传感器数据")
                except Exception as e:
                    print(f"  Gripper {j+1}: 错误 - {e}")

        time.sleep(0.1)

    # 关闭夹爪 - 尝试抓取
    print("\n关闭夹爪尝试抓取...")
    for joint_idx in gripper_joints:
        p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL, targetPosition=0.0, force=50)

    for i in range(50):  # 5秒
        p.stepSimulation()

        if i % 10 == 0:
            print(f"\n时间: {(i+100)/10:.1f}s (夹爪关闭 - 抓取中)")
            for j, joint_idx in enumerate(gripper_joints):
                try:
                    joint_state = p.getJointState(robot_id, joint_idx)
                    forces = joint_state[2]
                    if forces:
                        fx, fy, fz, mx, my, mz = forces
                        total_force = (fx**2 + fy**2 + fz**2)**0.5
                        print(f"  Gripper {j+1}: 合力={total_force:.2f}N (Fx={fx:.2f}, Fy={fy:.2f}, Fz={fz:.2f})")
                    else:
                        print(f"  Gripper {j+1}: 无传感器数据")
                except Exception as e:
                    print(f"  Gripper {j+1}: 错误 - {e}")

        time.sleep(0.1)

    p.disconnect()
    print("\n测试完成！")

if __name__ == "__main__":
    test_force_sensors()