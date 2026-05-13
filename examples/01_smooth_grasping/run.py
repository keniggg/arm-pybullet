#!/usr/bin/env python3
"""Smooth grasping demo with 6-axis force feedback and eye-in-hand camera.

Integrates: smooth arm control, soft ball grasping, real-time F/T display,
wrist RGB camera, and workspace reachability check.
"""
from __future__ import annotations

import sys
import os
import time

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("MUJOCO_GL", "glfw")
import mujoco
import mujoco.viewer

from common.model_loader import load_and_inject, RGB_CAMERA_NAME, SIM_HZ
from common.ik_solver import build_ik_plan, gripper_center
from common.motion import (
    build_gripper_limits,
    command_gripper,
    close_gripper,
    move_arm_to_angles,
    sync_position_actuators,
    SLEEP_SCALE,
)
from common.force_sensor import ForceTorqueSensor, FTDisplay
from common.camera import RGBCameraWindow


def main() -> None:
    print("=== Smooth Grasping Demo (Force Feedback + Eye-in-Hand) ===")

    model, data, arm_joints, gripper_joints, left_body_id, right_body_id, joint_to_actuator = (
        load_and_inject(include_ball=True, include_sensors=True, include_camera=True)
    )

    gripper_limits = build_gripper_limits(model, data, gripper_joints)
    command_gripper(model, data, gripper_limits, joint_to_actuator, "open")
    sync_position_actuators(model, data, joint_to_actuator)
    mujoco.mj_forward(model, data)

    print(f"Arm joints: {len(arm_joints)}, Gripper joints: {len(gripper_joints)}")
    print(f"Gripper center: {np.round(gripper_center(data, left_body_id, right_body_id), 4)}")

    plan_ok, ik_plan = build_ik_plan(
        model, data, arm_joints, left_body_id, right_body_id, gripper_limits
    )
    if not plan_ok:
        print("ERROR: IK plan failed — targets unreachable. Exiting.")
        return

    ft_sensor = ForceTorqueSensor(model)
    ft_display = FTDisplay(width=480, height=320, history_len=200)

    rgb_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, RGB_CAMERA_NAME)
    rgb_window = None
    if rgb_camera_id >= 0:
        try:
            rgb_window = RGBCameraWindow(model, rgb_camera_id, width=480, height=360, render_every_n=8)
            print("Eye-in-hand camera: active (480x360 @ 30fps)")
        except Exception as exc:
            print(f"Camera init failed: {exc}")

    demo_step = 0
    demo_wait_frames = 0
    step_count = 0

    def sync_and_update() -> None:
        nonlocal step_count
        try:
            if not viewer.is_running():
                return
            # Force sensor (every 4 steps = 60Hz)
            if step_count % 4 == 0:
                reading = ft_sensor.read(data)
                ft_display.update(reading)
            # Force display (offset from camera)
            if step_count % 4 == 2:
                ft_display.show()
            # RGB camera (every 8 steps = 30Hz)
            if rgb_window and rgb_window.should_update(step_count):
                reading = ft_sensor.read(data)
                force_mag = float(np.linalg.norm(reading.force))
                rgb_window.update(data, overlay_text=f"|F|={force_mag:.2f}N")
            viewer.sync()
        except Exception:
            pass

    def execute_demo_step() -> None:
        nonlocal demo_step, demo_wait_frames

        if demo_wait_frames > 0:
            demo_wait_frames -= 1
            return

        if demo_step == 0:
            print(">>> Step 0: Home position")
            command_gripper(model, data, gripper_limits, joint_to_actuator, "open")
            move_arm_to_angles(
                model, data, np.zeros(len(arm_joints)), arm_joints,
                joint_to_actuator, update_fn=sync_and_update,
            )
            demo_step = 1
            demo_wait_frames = 60

        elif demo_step == 1:
            print(">>> Step 1: Approach (above ball)")
            move_arm_to_angles(
                model, data, ik_plan["approach"].angles, arm_joints,
                joint_to_actuator, update_fn=sync_and_update,
            )
            demo_step = 2
            demo_wait_frames = 60

        elif demo_step == 2:
            print(">>> Step 2: Descend to grasp pose")
            move_arm_to_angles(
                model, data, ik_plan["grasp"].angles, arm_joints,
                joint_to_actuator, speed=0.55, update_fn=sync_and_update,
            )
            demo_step = 3
            demo_wait_frames = 45

        elif demo_step == 3:
            print(">>> Step 3: Close gripper (force-controlled)")
            contact = close_gripper(
                model, data, gripper_limits, joint_to_actuator, update_fn=sync_and_update,
            )
            reading = ft_sensor.read(data)
            print(f"    Contact: {contact}, Wrist force: {np.round(reading.force, 3)} N")
            demo_step = 4
            demo_wait_frames = 90

        elif demo_step == 4:
            print(">>> Step 4: Lift object")
            move_arm_to_angles(
                model, data, ik_plan["lift"].angles, arm_joints,
                joint_to_actuator, update_fn=sync_and_update,
            )
            demo_step = 5
            demo_wait_frames = 60

        elif demo_step == 5:
            print(">>> Step 5: Move to place position")
            move_arm_to_angles(
                model, data, ik_plan["place"].angles, arm_joints,
                joint_to_actuator, update_fn=sync_and_update,
            )
            demo_step = 6
            demo_wait_frames = 60

        elif demo_step == 6:
            print(">>> Step 6: Release object")
            command_gripper(model, data, gripper_limits, joint_to_actuator, "open")
            demo_step = 7
            demo_wait_frames = 120

        elif demo_step == 7:
            sync_position_actuators(model, data, joint_to_actuator)
            print(">>> Demo complete. Manual control available via MuJoCo viewer.")
            demo_step = 99

    print("Launching viewer...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.distance = 1.2
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.lookat[:] = [0.26, 0.0, 0.16]

        while viewer.is_running():
            step_start = time.perf_counter()

            execute_demo_step()
            mujoco.mj_step(model, data)
            step_count += 1
            sync_and_update()
            if CV2_AVAILABLE:
                cv2.waitKey(1)

            elapsed = time.perf_counter() - step_start
            target_dt = (1.0 / SIM_HZ) * SLEEP_SCALE
            remaining = target_dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

    ft_display.close()
    if rgb_window:
        rgb_window.close()
    print("Done.")


if __name__ == "__main__":
    main()
