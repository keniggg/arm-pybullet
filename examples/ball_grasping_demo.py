import os
import time
import pybullet as p
import pybullet_data


def build_basket(base_position, inner_size=0.16, wall_thickness=0.01, wall_height=0.08):
    half_inner = inner_size / 2.0
    base_half = [half_inner + wall_thickness, half_inner + wall_thickness, wall_thickness / 2.0]
    wall_half_x = [wall_thickness / 2.0, half_inner + wall_thickness, wall_height / 2.0]
    wall_half_y = [half_inner + wall_thickness, wall_thickness / 2.0, wall_height / 2.0]

    collision_shape = p.createCollisionShapeArray(
        shapeTypes=[p.GEOM_BOX] * 5,
        halfExtents=[
            base_half,
            wall_half_x,
            wall_half_x,
            wall_half_y,
            wall_half_y,
        ],
        collisionFramePositions=[
            [0, 0, wall_thickness / 2.0],
            [half_inner + wall_thickness / 2.0, 0, wall_height / 2.0 + wall_thickness],
            [-(half_inner + wall_thickness / 2.0), 0, wall_height / 2.0 + wall_thickness],
            [0, half_inner + wall_thickness / 2.0, wall_height / 2.0 + wall_thickness],
            [0, -(half_inner + wall_thickness / 2.0), wall_height / 2.0 + wall_thickness],
        ],
    )

    visual_shape = p.createVisualShapeArray(
        shapeTypes=[p.GEOM_BOX] * 5,
        halfExtents=[
            base_half,
            wall_half_x,
            wall_half_x,
            wall_half_y,
            wall_half_y,
        ],
        visualFramePositions=[
            [0, 0, wall_thickness / 2.0],
            [half_inner + wall_thickness / 2.0, 0, wall_height / 2.0 + wall_thickness],
            [-(half_inner + wall_thickness / 2.0), 0, wall_height / 2.0 + wall_thickness],
            [0, half_inner + wall_thickness / 2.0, wall_height / 2.0 + wall_thickness],
            [0, -(half_inner + wall_thickness / 2.0), wall_height / 2.0 + wall_thickness],
        ],
        rgbaColors=[
            [0.4, 0.3, 0.2, 1],
            [0.6, 0.5, 0.4, 1],
            [0.6, 0.5, 0.4, 1],
            [0.6, 0.5, 0.4, 1],
            [0.6, 0.5, 0.4, 1],
        ],
    )

    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=base_position,
    )


def smooth_targets(previous, target, alpha=0.15):
    return [prev + alpha * (tgt - prev) for prev, tgt in zip(previous, target)]


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
    p.setPhysicsEngineParameter(numSolverIterations=150, solverResidualThreshold=1e-7)

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

    # 创建柔软的可抓取小球（缩小直径便于夹爪抓取）
    ball_radius = 0.02
    ball_collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
    ball_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=ball_radius, rgbaColor=[0.8, 0.2, 0.2, 1])

    ball_pos = [0.3, 0, 0.1]  # 在机械臂前方
    ball_id = p.createMultiBody(
        baseMass=0.03,
        baseCollisionShapeIndex=ball_collision_shape,
        baseVisualShapeIndex=ball_visual_shape,
        basePosition=ball_pos
    )

    # 设置球的柔软物理属性
    p.changeDynamics(ball_id, -1,
                    lateralFriction=2.0,    # 高摩擦力便于抓取
                    spinningFriction=0.1,
                    rollingFriction=0.1,
                    restitution=0.1,        # 低弹性
                    contactDamping=1.0,     # 高阻尼
                    contactStiffness=300)   # 较低刚度

    # 添加可放置小球的篮子
    basket_id = build_basket(base_position=[0.5, 0.0, 0.0], inner_size=0.16, wall_thickness=0.01, wall_height=0.08)

    # 获取关节信息
    num_joints = p.getNumJoints(robot_id)
    controllable_joints = []
    joint_lower_limits = []
    joint_upper_limits = []
    joint_ranges = []
    joint_rest_positions = []
    joint_name_map = {}

    for j in range(num_joints):
        info = p.getJointInfo(robot_id, j)
        joint_name = info[1].decode("utf-8")
        joint_type = info[2]

        if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            controllable_joints.append(j)
            joint_name_map[j] = joint_name

    for j in range(num_joints):
        info = p.getJointInfo(robot_id, j)
        joint_type = info[2]
        if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            if info[8] < info[9]:
                lower, upper = info[8], info[9]
            else:
                lower, upper = -3.14159, 3.14159
            rest = p.getJointState(robot_id, j)[0]
        else:
            lower, upper = 0.0, 0.0
            rest = 0.0
        joint_lower_limits.append(lower)
        joint_upper_limits.append(upper)
        joint_ranges.append(upper - lower)
        joint_rest_positions.append(rest)

    # 找出末端执行器
    end_effector_link = None
    for j in range(num_joints):
        info = p.getJointInfo(robot_id, j)
        link_name = info[12].decode("utf-8") if info[12] else ""
        if "tool" in link_name.lower():
            end_effector_link = j
            break
    if end_effector_link is None:
        end_effector_link = num_joints - 1

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
    control_mode_id = p.addUserDebugParameter("Control Mode (0=Joint, 1=EE)", 0, 1, 0)
    slider_ids = []
    for i, joint_idx in enumerate(controllable_joints):
        joint_name = joint_name_map.get(joint_idx, f"joint_{joint_idx}")
        low, high = joint_lower_limits[joint_idx], joint_upper_limits[joint_idx]
        sid = p.addUserDebugParameter(f"Joint: {joint_name}", low, high, p.getJointState(robot_id, joint_idx)[0])
        slider_ids.append((joint_idx, sid))

    link_state = p.getLinkState(robot_id, end_effector_link)
    ee_pos = link_state[0]
    ee_orn = link_state[1]
    ee_euler = p.getEulerFromQuaternion(ee_orn)

    ee_x_id = p.addUserDebugParameter("EE X", ee_pos[0] - 0.2, ee_pos[0] + 0.2, ee_pos[0])
    ee_y_id = p.addUserDebugParameter("EE Y", ee_pos[1] - 0.2, ee_pos[1] + 0.2, ee_pos[1])
    ee_z_id = p.addUserDebugParameter("EE Z", max(0.05, ee_pos[2] - 0.2), ee_pos[2] + 0.2, ee_pos[2])
    ee_roll_id = p.addUserDebugParameter("EE Roll", -3.14159, 3.14159, ee_euler[0])
    ee_pitch_id = p.addUserDebugParameter("EE Pitch", -3.14159, 3.14159, ee_euler[1])
    ee_yaw_id = p.addUserDebugParameter("EE Yaw", -3.14159, 3.14159, ee_euler[2])

    gripper_low = 0.0
    gripper_high = 0.04
    for joint_idx in gripper_joints:
        info = p.getJointInfo(robot_id, joint_idx)
        if info[8] < info[9]:
            gripper_low = min(gripper_low, info[8])
            gripper_high = max(gripper_high, info[9])
    gripper_id = p.addUserDebugParameter("Gripper Open", gripper_low, gripper_high, gripper_high)

    # 添加状态显示
    ball_status_id = p.addUserDebugText("Soft Ball: Ready for grasping", [-0.5, 0, 0.9], textColorRGB=[0.8, 0.2, 0.2], textSize=1.3)
    camera_status_id = p.addUserDebugText("Eye-in-Hand RGB: Initializing", [0.1, -0.3, 0.9], textColorRGB=[0, 1, 1], textSize=1.2)

    force_display_ids = []
    for i, joint_idx in enumerate(gripper_joints):
        text_id = p.addUserDebugText(f"Gripper {i+1}: 0.00 N", [0, 0.1 + i*0.05, 0.8], textColorRGB=[1, 0, 0], textSize=1.2)
        force_display_ids.append(text_id)

    print("柔软小球抓取演示已启动！")
    print("使用滑块控制机械臂关节，尝试抓取红色的柔软小球")
    print("Control Mode 0 = 关节控制, 1 = 末端执行器 (IK) 控制")
    print("力反馈传感器已启用 - 观察夹爪力显示的变化")
    print("绿色: 成功抓取 (>5N) | 橙色: 接触检测 (1-5N) | 红色: 无接触 (<1N)")
    print("关闭窗口退出程序")

    camera_width = 160
    camera_height = 120
    camera_fov = 60

    joint_targets = [p.getJointState(robot_id, j)[0] for j in controllable_joints]
    ee_target = list(ee_pos)
    ee_target_euler = list(ee_euler)
    gripper_target = gripper_high

    frame_count = 0
    while p.isConnected():
        frame_count += 1

        # 读取滑块值并控制关节/末端执行器
        try:
            control_mode = p.readUserDebugParameter(control_mode_id)
            if control_mode < 0.5:
                current_targets = []
                for idx, (joint_idx, sid) in enumerate(slider_ids):
                    val = p.readUserDebugParameter(sid)
                    current_targets.append(val)

                joint_targets = smooth_targets(joint_targets, current_targets)
                for joint_idx, target in zip(controllable_joints, joint_targets):
                    p.setJointMotorControl2(
                        robot_id,
                        joint_idx,
                        p.POSITION_CONTROL,
                        targetPosition=target,
                        force=200,
                        positionGain=0.25,
                        velocityGain=1.0,
                    )
            else:
                ee_target = smooth_targets(
                    ee_target,
                    [
                        p.readUserDebugParameter(ee_x_id),
                        p.readUserDebugParameter(ee_y_id),
                        p.readUserDebugParameter(ee_z_id),
                    ],
                )
                ee_target_euler = smooth_targets(
                    ee_target_euler,
                    [
                        p.readUserDebugParameter(ee_roll_id),
                        p.readUserDebugParameter(ee_pitch_id),
                        p.readUserDebugParameter(ee_yaw_id),
                    ],
                )
                gripper_target = p.readUserDebugParameter(gripper_id)

                ee_target_orn = p.getQuaternionFromEuler(ee_target_euler)
                ik_solution = p.calculateInverseKinematics(
                    robot_id,
                    end_effector_link,
                    ee_target,
                    ee_target_orn,
                    lowerLimits=joint_lower_limits,
                    upperLimits=joint_upper_limits,
                    jointRanges=joint_ranges,
                    restPoses=joint_rest_positions,
                )

                ik_targets = [ik_solution[j] for j in controllable_joints]
                joint_targets = smooth_targets(joint_targets, ik_targets)
                for joint_idx, target in zip(controllable_joints, joint_targets):
                    p.setJointMotorControl2(
                        robot_id,
                        joint_idx,
                        p.POSITION_CONTROL,
                        targetPosition=target,
                        force=200,
                        positionGain=0.3,
                        velocityGain=1.0,
                    )

                for joint_idx in gripper_joints:
                    p.setJointMotorControl2(
                        robot_id,
                        joint_idx,
                        p.POSITION_CONTROL,
                        targetPosition=gripper_target,
                        force=80,
                        positionGain=0.4,
                        velocityGain=1.0,
                    )
        except Exception:
            # 如果GUI关闭，退出循环
            break

        # 实时更新摄像头与力传感器显示
        try:
            link_state = p.getLinkState(robot_id, end_effector_link, computeForwardKinematics=True)
            camera_pos = link_state[0]
            camera_orn = link_state[1]
            rot_matrix = p.getMatrixFromQuaternion(camera_orn)
            camera_dir = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]
            camera_target = [
                camera_pos[0] + camera_dir[0] * 0.25,
                camera_pos[1] + camera_dir[1] * 0.25,
                camera_pos[2] + camera_dir[2] * 0.25,
            ]

            view_matrix = p.computeViewMatrix(camera_pos, camera_target, [0, 0, 1])
            proj_matrix = p.computeProjectionMatrixFOV(camera_fov, camera_width / camera_height, 0.02, 2.0)
            camera_img = p.getCameraImage(
                width=camera_width,
                height=camera_height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )
            rgb = camera_img[2]
            mean_r = sum(rgb[0::4]) / (camera_width * camera_height)
            mean_g = sum(rgb[1::4]) / (camera_width * camera_height)
            mean_b = sum(rgb[2::4]) / (camera_width * camera_height)
            p.addUserDebugText(
                f"Eye-in-Hand RGB: ({mean_r:.0f}, {mean_g:.0f}, {mean_b:.0f})",
                [0.1, -0.3, 0.9],
                textColorRGB=[0, 1, 1],
                textSize=1.2,
                replaceItemUniqueId=camera_status_id,
            )
        except Exception:
            p.addUserDebugText(
                "Eye-in-Hand RGB: Error",
                [0.1, -0.3, 0.9],
                textColorRGB=[1, 0, 0],
                textSize=1.2,
                replaceItemUniqueId=camera_status_id,
            )

        for i, joint_idx in enumerate(gripper_joints):
            try:
                joint_state = p.getJointState(robot_id, joint_idx)
                # jointReactionForces 返回 [Fx, Fy, Fz, Mx, My, Mz]
                forces = joint_state[2]
                if forces:
                    fx, fy, fz, mx, my, mz = forces
                    total_force = (fx**2 + fy**2 + fz**2)**0.5

                    if total_force > 5.0:
                        color = [0, 1, 0]
                    elif total_force > 1.0:
                        color = [1, 0.5, 0]
                    else:
                        color = [1, 0, 0]

                    p.addUserDebugText(
                        f"Gripper {i+1}: {total_force:.2f} N (Fx:{fx:.1f}, Fy:{fy:.1f}, Fz:{fz:.1f})",
                        [0, 0.1 + i * 0.05, 0.8],
                        textColorRGB=color,
                        textSize=1.2,
                        replaceItemUniqueId=force_display_ids[i],
                    )
                else:
                    p.addUserDebugText(
                        f"Gripper {i+1}: No sensor data",
                        [0, 0.1 + i * 0.05, 0.8],
                        textColorRGB=[0.5, 0.5, 0.5],
                        textSize=1.2,
                        replaceItemUniqueId=force_display_ids[i],
                    )
            except Exception as e:
                p.addUserDebugText(
                    f"Gripper {i+1}: Error",
                    [0, 0.1 + i * 0.05, 0.8],
                    textColorRGB=[1, 0, 0],
                    textSize=1.2,
                    replaceItemUniqueId=force_display_ids[i],
                )
                print(f"Error reading force sensor for joint {joint_idx}: {e}")

        p.stepSimulation()
        time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    main()
