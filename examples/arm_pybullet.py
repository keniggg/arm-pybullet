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

    # 2) 加载地面（简化方式）
    # plane_id = p.loadURDF("plane.urdf")
    # 如果plane.urdf找不到，用createMultiBody创建简单地面
    plane_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[10, 10, 0.01])
    plane_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[10, 10, 0.01], rgbaColor=[0.8, 0.8, 0.8, 1])
    plane_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=plane_collision_shape, baseVisualShapeIndex=plane_visual_shape, basePosition=[0, 0, -0.01])

    # 3) 加载你的机械臂 URDF
    # 使用相对路径避免中文路径编码问题
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

    # 推荐 flags：保持惯性、读取材质颜色等
    flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_MAINTAIN_LINK_ORDER

    # 机械臂基座位置/姿态（你可以改高度、朝向）
    base_pos = [0, 0, 0]        # 如果模型悬空可把 z 改成 0.1 或 0.2
    base_orn = p.getQuaternionFromEuler([0, 0, 0])

    robot_id = p.loadURDF(
        urdf_path,
        basePosition=base_pos,
        baseOrientation=base_orn,
        useFixedBase=True,   # 机械臂通常固定在地面
        flags=flags
    )

    # 3.5) 创建柔软的可抓取小球
    # 创建软体球
    ball_radius = 0.05  # 5cm半径
    
    # 将球放置在机械臂前方
    ball_pos = [0.3, 0, 0.1]  # 在机械臂前方30cm，高度10cm
    
    # 创建软体球（真正的柔软物体）
    ball_file = None  # 我们将使用程序生成的球体
    
    # 或者，如果软体创建失败，使用刚体球但设置柔软属性
    try:
        # 尝试创建软体球
        ball_id = p.createSoftBody(
            mass=0.1,
            collisionShape=p.GEOM_SPHERE,
            visualShape=p.GEOM_SPHERE,
            radius=ball_radius,
            position=ball_pos,
            color=[0.8, 0.2, 0.2, 1],  # 红色
            useMassSpring=1,            # 使用质量-弹簧系统
            useSelfCollision=0,         # 禁用自碰撞
            springElasticStiffness=100, # 弹簧弹性刚度
            springDampingStiffness=0.1, # 弹簧阻尼
            springBendingStiffness=10,  # 弯曲刚度
            useFaceContact=1            # 启用面接触
        )
        print(f"Created soft body ball with id={ball_id}")
    except Exception as e:
        print(f"Soft body creation failed: {e}, using rigid body instead")
        # 回退到刚体球
        ball_collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
        ball_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=ball_radius, rgbaColor=[0.8, 0.2, 0.2, 1])
        
        ball_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=ball_collision_shape,
            baseVisualShapeIndex=ball_visual_shape,
            basePosition=ball_pos,
            baseOrientation=[0, 0, 0, 1]
        )
        
        # 设置柔软的物理属性
        p.changeDynamics(ball_id, -1, 
                        lateralFriction=1.0,    # 高摩擦力便于抓取
                        spinningFriction=0.1,
                        rollingFriction=0.1,
                        restitution=0.1,        # 低弹性
                        contactDamping=0.5,     # 高阻尼使之柔软
                        contactStiffness=500)   # 较低的接触刚度
        
        print(f"Created rigid body ball with soft properties, id={ball_id}")
    
    print(f"Ball positioned at {ball_pos} for grasping")

    # 4) 输出关节信息，找出可控关节和传感器配置
    num_joints = p.getNumJoints(robot_id)
    print(f"Loaded robot id={robot_id}, joints={num_joints}\n")

    controllable = []
    end_effector_link = None
    gripper_joints = []
    link_names = []
    
    for j in range(num_joints):
        info = p.getJointInfo(robot_id, j)
        joint_index = info[0]
        joint_name = info[1].decode("utf-8")
        joint_type = info[2]
        link_name = info[12].decode("utf-8") if info[12] else ""
        lower = info[8]
        upper = info[9]
        max_force = info[10]
        max_vel = info[11]

        link_names.append(link_name)
        
        # 找出末端执行器link
        if "tool" in link_name.lower() or "end" in link_name.lower() or "ee" in link_name.lower():
            end_effector_link = j
        
        # 找出夹爪关节
        if "finger" in joint_name.lower() or "gripper" in joint_name.lower():
            gripper_joints.append(j)

        # 关节类型：REVOLUTE=0, PRISMATIC=1, FIXED=4 等
        is_ctrl = joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC)
        print(
            f"[{joint_index:2d}] {joint_name:35s} "
            f"type={joint_type} limit=({lower:.3f}, {upper:.3f}) "
            f"F={max_force:.1f} V={max_vel:.2f} ctrl={is_ctrl}"
        )
        if is_ctrl:
            controllable.append((joint_index, joint_name, lower, upper))
    
    # 如果没找到末端link，使用最后一个关节
    if end_effector_link is None:
        end_effector_link = num_joints - 1
    
    print(f"End effector link index: {end_effector_link}")
    print(f"Gripper joints: {gripper_joints}")
    print(f"Link names: {link_names}")

    # 5) 添加眼在手上的摄像头参数
    camera_width = 320
    camera_height = 240
    camera_fov = 60
    camera_near = 0.01
    camera_far = 5.0
    
    # 6) 启用夹爪力反馈传感器
    for joint_idx in gripper_joints:
        p.enableJointForceTorqueSensor(robot_id, joint_idx, enableSensor=True)
        print(f"Enabled force/torque sensor for joint {joint_idx}")

    # 5) 创建滑块（对每个可控关节）
    slider_ids = []
    for (j, name, low, high) in controllable:
        # 有些 URDF 可能没写限制（low/high 会是 0/ -1/ 1e+something），这里做个兜底
        if low > high or (abs(low) < 1e-6 and abs(high) < 1e-6):
            low, high = -3.14159, 3.14159  # 默认给转动关节 ±pi
        init_pos = 0.0
        sid = p.addUserDebugParameter(name, low, high, init_pos)
        slider_ids.append((j, sid))

    # 6) 相机视角（可选）
    p.resetDebugVisualizerCamera(
        cameraDistance=1.2,
        cameraYaw=50,
        cameraPitch=-25,
        cameraTargetPosition=[0, 0, 0.3]
    )

    print("模型加载完成！使用滑块控制关节，或关闭窗口退出。")
    print("眼在手上摄像头已启用，力反馈传感器已启用。")
    print("红色柔软小球已放置在机械臂前方，可以使用夹爪抓取！")

    # 7) 主循环：读取滑块 -> 位置控制 + 传感器数据
    camera_image_window = None
    force_display_ids = []
    ball_status_id = p.addUserDebugText("Soft Ball: Ready for grasping", [-0.5, 0, 0.9], textColorRGB=[0.8, 0.2, 0.2], textSize=1.3)
    
    # 创建力显示文本
    for i, joint_idx in enumerate(gripper_joints):
        text_id = p.addUserDebugText(f"Gripper Force {i+1}: 0.00 N", [0, 0.1 + i*0.05, 0.8], textColorRGB=[1, 0, 0], textSize=1.2)
        force_display_ids.append(text_id)

    while True:
        targets = []
        joint_indices = []
        for (j, sid) in slider_ids:
            val = p.readUserDebugParameter(sid)
            joint_indices.append(j)
            targets.append(val)

        # 位置控制（P增益可按需要调）
        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=targets,
            positionGains=[0.1] * len(joint_indices),
            forces=[200] * len(joint_indices)  # 力限制：按模型大小可调
        )

        # 获取眼在手上摄像头图像
        try:
            # 获取末端link状态
            link_state = p.getLinkState(robot_id, end_effector_link)
            camera_pos = link_state[0]
            camera_orn = link_state[1]
            
            # 计算摄像头朝向（朝向末端前方，稍微向下倾斜）
            camera_rot_matrix = p.getMatrixFromQuaternion(camera_orn)
            # 摄像头朝向是link的前方方向
            camera_target = [
                camera_pos[0] + camera_rot_matrix[0] * 0.5,
                camera_pos[1] + camera_rot_matrix[3] * 0.5, 
                camera_pos[2] + camera_rot_matrix[6] * 0.5 - 0.1  # 稍微向下
            ]
            
            # 设置摄像头视图
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=camera_pos,
                cameraTargetPosition=camera_target,
                cameraUpVector=[0, 0, 1]
            )
            
            projection_matrix = p.computeProjectionMatrixFOV(
                fov=camera_fov,
                aspect=camera_width/camera_height,
                nearVal=camera_near,
                farVal=camera_far
            )
            
            # 获取图像（只获取RGB，不显示以节省性能）
            images = p.getCameraImage(
                width=camera_width,
                height=camera_height,
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            
            # 显示摄像头状态
            if camera_image_window is None:
                camera_image_window = p.addUserDebugText("Eye-in-Hand Camera Active", [0, 0, 0.9], textColorRGB=[0, 1, 0], textSize=1.5)
                
        except Exception as e:
            # 如果摄像头设置失败，继续运行
            if camera_image_window is None:
                camera_image_window = p.addUserDebugText("Camera Setup Failed", [0, 0, 0.9], textColorRGB=[1, 0, 0], textSize=1.5)
            pass
        
        # 读取力反馈传感器数据
        for i, joint_idx in enumerate(gripper_joints):
            try:
                joint_state = p.getJointState(robot_id, joint_idx)
                applied_force = joint_state[2]  # appliedJointMotorTorque
                
                # 更新力显示
                p.addUserDebugText(f"Gripper Force {i+1}: {applied_force:.2f} N", 
                                 [0, 0.1 + i*0.05, 0.8], 
                                 textColorRGB=[1, 0, 0], 
                                 textSize=1.2,
                                 replaceItemUniqueId=force_display_ids[i])
            except Exception as e:
                # 如果传感器读取失败，显示错误
                p.addUserDebugText(f"Gripper Force {i+1}: Error", 
                                 [0, 0.1 + i*0.05, 0.8], 
                                 textColorRGB=[1, 0, 0], 
                                 textSize=1.2,
                                 replaceItemUniqueId=force_display_ids[i])

        p.stepSimulation()
        time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    main()
