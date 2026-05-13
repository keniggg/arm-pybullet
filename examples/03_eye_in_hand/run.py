#!/usr/bin/env python3
"""Eye-in-hand RGB camera demo.

Demonstrates the wrist-mounted camera with smooth arm motion.
The camera view updates at 30fps while the arm moves through various poses.
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
from common.ik_solver import solve_gripper_center_ik, gripper_center
from common.motion import (
    build_gripper_limits,
    command_gripper,
    move_arm_to_angles,
    sync_position_actuators,
    SLEEP_SCALE,
)
from common.camera import RGBCameraWindow


def main() -> None:
    print("=== Eye-in-Hand RGB Camera Demo ===")

    model, data, arm_joints, gripper_joints, left_body_id, right_body_id, joint_to_actuator = (
        load_and_inject(include_ball=True, include_sensors=False, include_camera=True)
    )

    gripper_limits = build_gripper_limits(model, data, gripper_joints)
    command_gripper(model, data, gripper_limits, joint_to_actuator, "open")
    sync_position_actuators(model, data, joint_to_actuator)
    mujoco.mj_forward(model, data)

    rgb_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, RGB_CAMERA_NAME)
    if rgb_camera_id < 0:
        print("ERROR: Wrist camera not found.")
        return

    rgb_window = RGBCameraWindow(
        model, rgb_camera_id, width=640, height=480, render_every_n=8,
        window_name="Eye-in-Hand Camera (640x480)"
    )
    print("Camera window opened: 640x480 @ 30fps")

    scan_targets = [
        np.array([0.30, 0.00, 0.20]),
        np.array([0.25, -0.15, 0.15]),
        np.array([0.35, 0.10, 0.10]),
        np.array([0.20, 0.00, 0.25]),
        np.array([0.30, 0.00, 0.05]),
    ]

    scan_angles = []
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos[:]
    for target in scan_targets:
        result = solve_gripper_center_ik(
            model, scratch, target, left_body_id, right_body_id, arm_joints
        )
        if result.success:
            scan_angles.append(result.angles)
            from common.ik_solver import set_joint_positions
            set_joint_positions(model, scratch, arm_joints, result.angles)
            mujoco.mj_forward(model, scratch)
        else:
            print(f"  Skipping unreachable target: {target}")

    if not scan_angles:
        print("ERROR: No reachable scan targets.")
        rgb_window.close()
        return

    print(f"Scanning {len(scan_angles)} viewpoints...")

    demo_idx = 0
    demo_wait_frames = 0
    step_count = 0
    looping = True

    def sync_and_update() -> None:
        nonlocal step_count
        try:
            if not viewer.is_running():
                return
            if rgb_window.should_update(step_count):
                rgb_window.update(data, overlay_text=f"View {demo_idx + 1}/{len(scan_angles)}")
            viewer.sync()
        except Exception:
            pass

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.distance = 1.2
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.lookat[:] = [0.26, 0.0, 0.16]

        while viewer.is_running():
            step_start = time.perf_counter()

            if demo_wait_frames > 0:
                demo_wait_frames -= 1
            elif looping:
                move_arm_to_angles(
                    model, data, scan_angles[demo_idx], arm_joints,
                    joint_to_actuator, speed=0.6, update_fn=sync_and_update,
                )
                demo_wait_frames = 90
                demo_idx = (demo_idx + 1) % len(scan_angles)

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

    rgb_window.close()
    print("Done.")


if __name__ == "__main__":
    main()
