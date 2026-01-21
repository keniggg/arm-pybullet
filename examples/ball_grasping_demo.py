import os
import time
import pybullet as p
import pybullet_data


def main():
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
    plane_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=plane_collision_shape, baseVisualShapeIndex=plane_visual_shape, basePosition=[0, 0, -0.01])

    # 加载机械臂
    urdf_path = "synriard/urdf/Alicia_D_v5_6/Alicia_D_v5_6_gripper_50mm.urdf"
    p.setAdditionalSearchPath("synriard")
    p.setAdditionalSearchPath("synriard/meshes")

    flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_MAINTAIN_LINK_ORDER
    robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True, flags=flags)

    # 创建柔软的可抓取小球
    ball_radius = 0.05
    ball_collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
    ball_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=ball_radius, rgbaColor=[0.8, 0.2, 0.2, 1])

    ball_pos = [0.3, 0, 0.1]  # 在机械臂前方
    ball_id = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=ball_collision_shape,
        baseVisualShapeIndex=ball_visual_shape,
        basePosition=ball_pos
    )

    # 设置球的柔软物理属性
    p.changeDynamics(ball_id, -1,
                    lateralFriction=1.5,    # 高摩擦力便于抓取
                    spinningFriction=0.1,
                    rollingFriction=0.1,
                    restitution=0.1,        # 低弹性
                    contactDamping=1.0,     # 高阻尼
                    contactStiffness=300)   # 较低刚度

    # 获取关节信息
    num_joints = p.getNumJoints(robot_id)
    controllable_joints = []

    for j in range(num_joints):
        info = p.getJointInfo(robot_id, j)
        joint_name = info[1].decode("utf-8")
        joint_type = info[2]

        if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            controllable_joints.append(j)

    # 启用夹爪力传感器
    gripper_joints = []
    for j in range(num_joints):
        info = p.getJointInfo(robot_id, j)
        joint_name = info[1].decode("utf-8")
        if joint_name in ["left_finger", "right_finger"]:
            gripper_joints.append(j)
            p.enableJointForceTorqueSensor(robot_id, j, enableSensor=True)
            print(f"Enabled force sensor for joint {j}: {joint_name}")

    # 设置相机视角
    p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=50, cameraPitch=-25, cameraTargetPosition=[0, 0, 0.3])

    # 创建控制滑块
    slider_ids = []
    for i, joint_idx in enumerate(controllable_joints):
        info = p.getJointInfo(robot_id, joint_idx)
        joint_name = info[1].decode("utf-8")
        low, high = info[8], info[9]

        if low >= high:
            low, high = -3.14159, 3.14159

        sid = p.addUserDebugParameter(f"{joint_name}", low, high, 0.0)
        slider_ids.append((joint_idx, sid))

    # 添加状态显示
    ball_status_id = p.addUserDebugText("Soft Ball: Ready for grasping", [-0.5, 0, 0.9], textColorRGB=[0.8, 0.2, 0.2], textSize=1.3)

    force_display_ids = []
    for i, joint_idx in enumerate(gripper_joints):
        text_id = p.addUserDebugText(f"Gripper {i+1}: 0.00 N", [0, 0.1 + i*0.05, 0.8], textColorRGB=[1, 0, 0], textSize=1.2)
        force_display_ids.append(text_id)

    print("柔软小球抓取演示已启动！")
    print("使用滑块控制机械臂关节，尝试抓取红色的柔软小球")
    print("力反馈传感器已启用 - 观察夹爪力显示的变化")
    print("绿色: 成功抓取 (>5N) | 橙色: 接触检测 (1-5N) | 红色: 无接触 (<1N)")
    print("关闭窗口退出程序")

    frame_count = 0
    while p.isConnected():
        frame_count += 1

        # 读取滑块值并控制关节
        try:
            for joint_idx, sid in slider_ids:
                val = p.readUserDebugParameter(sid)
                p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL, targetPosition=val, force=200)
        except:
            # 如果GUI关闭，退出循环
            break

        # 每10帧更新力传感器显示
        if frame_count % 10 == 0:
            for i, joint_idx in enumerate(gripper_joints):
                try:
                    joint_state = p.getJointState(robot_id, joint_idx)
                    # jointReactionForces 返回 [Fx, Fy, Fz, Mx, My, Mz]
                    forces = joint_state[2]
                    if forces:
                        fx, fy, fz, mx, my, mz = forces
                        # 计算合力和主要方向的力
                        total_force = (fx**2 + fy**2 + fz**2)**0.5
                        
                        # 根据力的大小改变颜色
                        if total_force > 5.0:  # 检测到较大抓取力
                            color = [0, 1, 0]  # 绿色表示成功抓取
                        elif total_force > 1.0:  # 检测到接触力
                            color = [1, 0.5, 0]  # 橙色表示接触
                        else:
                            color = [1, 0, 0]  # 红色表示无接触
                        
                        p.addUserDebugText(f"Gripper {i+1}: {total_force:.2f} N (Fx:{fx:.1f}, Fy:{fy:.1f}, Fz:{fz:.1f})",
                                         [0, 0.1 + i*0.05, 0.8],
                                         textColorRGB=color,
                                         textSize=1.2,
                                         replaceItemUniqueId=force_display_ids[i])
                    else:
                        p.addUserDebugText(f"Gripper {i+1}: No sensor data",
                                         [0, 0.1 + i*0.05, 0.8],
                                         textColorRGB=[0.5, 0.5, 0.5],
                                         textSize=1.2,
                                         replaceItemUniqueId=force_display_ids[i])
                except Exception as e:
                    p.addUserDebugText(f"Gripper {i+1}: Error",
                                     [0, 0.1 + i*0.05, 0.8],
                                     textColorRGB=[1, 0, 0],
                                     textSize=1.2,
                                     replaceItemUniqueId=force_display_ids[i])
                    print(f"Error reading force sensor for joint {joint_idx}: {e}")

        p.stepSimulation()
        time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    main()