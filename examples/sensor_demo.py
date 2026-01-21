import os
import time
import pybullet as p
import pybullet_data
from synriard import get_model_path


def main():
    # 改变工作目录到项目根目录，避免中文路径问题
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(repo_root)

    # 1) 连接 PyBullet（GUI）
    physics_client = p.connect(p.GUI)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 用于加载 plane.urdf
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)

    # 2) 加载地面
    plane_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[10, 10, 0.01])
    plane_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[10, 10, 0.01], rgbaColor=[0.8, 0.8, 0.8, 1])
    plane_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=plane_collision_shape, baseVisualShapeIndex=plane_visual_shape, basePosition=[0, 0, -0.01])

    # 3) 加载机械臂 URDF
    urdf_path = "synriard/urdf/Alicia_D_v5_6/Alicia_D_v5_6_gripper_50mm.urdf"
    print(f"Loading URDF: {urdf_path}")
    assert os.path.exists(urdf_path), f"URDF 不存在: {urdf_path}"

    # 设置搜索路径以找到mesh文件
    p.setAdditionalSearchPath("synriard")
    p.setAdditionalSearchPath("synriard/meshes")

    # 添加meshes下的所有子目录
    meshes_root = "synriard/meshes"
    if os.path.isdir(meshes_root):
        for root, dirs, files in os.walk(meshes_root):
            p.setAdditionalSearchPath(root)

    # 推荐 flags
    flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_MAINTAIN_LINK_ORDER

    base_pos = [0, 0, 0]
    base_orn = p.getQuaternionFromEuler([0, 0, 0])

    robot_id = p.loadURDF(
        urdf_path,
        basePosition=base_pos,
        baseOrientation=base_orn,
        useFixedBase=True,
        flags=flags
    )

    # 4) 获取关节信息
    num_joints = p.getNumJoints(robot_id)
    print(f"Loaded robot id={robot_id}, joints={num_joints}")

    # 找出末端执行器和夹爪关节
    end_effector_link = None
    gripper_joints = []

    for j in range(num_joints):
        info = p.getJointInfo(robot_id, j)
        joint_name = info[1].decode("utf-8")
        link_name = info[12].decode("utf-8") if info[12] else ""

        if "tool" in link_name.lower():
            end_effector_link = j
        if "finger" in joint_name.lower():
            gripper_joints.append(j)

    if end_effector_link is None:
        end_effector_link = num_joints - 1

    print(f"End effector link: {end_effector_link}")
    print(f"Gripper joints: {gripper_joints}")

    # 5) 启用力反馈传感器
    for joint_idx in gripper_joints:
        p.enableJointForceTorqueSensor(robot_id, joint_idx, enableSensor=True)
        print(f"Enabled force sensor for joint {joint_idx}")

    # 6) 摄像头参数
    camera_width = 320
    camera_height = 240
    camera_fov = 60

    # 7) 设置相机视角
    p.resetDebugVisualizerCamera(
        cameraDistance=1.2,
        cameraYaw=50,
        cameraPitch=-25,
        cameraTargetPosition=[0, 0, 0.3]
    )

    print("眼在手上摄像头和力反馈传感器已启用！")
    print("绿色文本：摄像头状态")
    print("红色文本：夹爪力传感器读数")
    print("关闭窗口退出程序。")

    # 8) 显示状态文本
    camera_status_id = p.addUserDebugText("Eye-in-Hand Camera: Active", [0, 0, 0.9], textColorRGB=[0, 1, 0], textSize=1.5)
    force_display_ids = []

    for i, joint_idx in enumerate(gripper_joints):
        text_id = p.addUserDebugText(f"Gripper {i+1} Force: 0.00 N", [0, 0.1 + i*0.05, 0.8], textColorRGB=[1, 0, 0], textSize=1.2)
        force_display_ids.append(text_id)

    frame_count = 0
    while True:
        frame_count += 1

        # 每10帧更新传感器数据
        if frame_count % 10 == 0:
            # 更新摄像头（模拟眼在手上）
            try:
                link_state = p.getLinkState(robot_id, end_effector_link)
                camera_pos = link_state[0]
                camera_orn = link_state[1]

                # 计算摄像头目标位置（末端前方）
                rot_matrix = p.getMatrixFromQuaternion(camera_orn)
                target_pos = [
                    camera_pos[0] + rot_matrix[0] * 0.3,
                    camera_pos[1] + rot_matrix[3] * 0.3,
                    camera_pos[2] + rot_matrix[6] * 0.3
                ]

                # 这里可以获取摄像头图像，但为了性能只显示状态
                # images = p.getCameraImage(width=camera_width, height=camera_height, ...)

            except Exception as e:
                p.addUserDebugText("Camera: Error", [0, 0, 0.9], textColorRGB=[1, 0, 0], textSize=1.5, replaceItemUniqueId=camera_status_id)

            # 更新力传感器读数
            for i, joint_idx in enumerate(gripper_joints):
                try:
                    joint_state = p.getJointState(robot_id, joint_idx)
                    force = joint_state[2]  # applied motor torque

                    p.addUserDebugText(f"Gripper {i+1} Force: {force:.2f} N",
                                     [0, 0.1 + i*0.05, 0.8],
                                     textColorRGB=[1, 0, 0],
                                     textSize=1.2,
                                     replaceItemUniqueId=force_display_ids[i])
                except Exception as e:
                    p.addUserDebugText(f"Gripper {i+1}: Error",
                                     [0, 0.1 + i*0.05, 0.8],
                                     textColorRGB=[1, 0, 0],
                                     textSize=1.2,
                                     replaceItemUniqueId=force_display_ids[i])

        p.stepSimulation()
        time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    main()