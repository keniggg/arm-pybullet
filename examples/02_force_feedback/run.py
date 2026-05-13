#!/usr/bin/env python3
"""Standalone 6-axis force/torque visualization demo.

Moves the arm to contact the ball and displays real-time F/T sensor readings.
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

from common.model_loader import load_and_inject, SIM_HZ
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


def main() -> None:
    print("=== 6-Axis Force/Torque Feedback Demo ===")

    model, data, arm_joints, gripper_joints, left_body_id, right_body_id, joint_to_actuator = (
        load_and_inject(include_ball=True, include_sensors=True, include_camera=False)
    )

    gripper_limits = build_gripper_limits(model, data, gripper_joints)
    command_gripper(model, data, gripper_limits, joint_to_actuator, "open")
    sync_position_actuators(model, data, joint_to_actuator)
    mujoco.mj_forward(model, data)

    plan_ok, ik_plan = build_ik_plan(
        model, data, arm_joints, left_body_id, right_body_id, gripper_limits
    )
    if not plan_ok:
        print("ERROR: IK plan failed. Exiting.")
        return

    ft_sensor = ForceTorqueSensor(model)
    ft_display = FTDisplay(width=520, height=380, history_len=300)
    print("Force/Torque display window opened.")

    demo_step = 0
    demo_wait_frames = 0
    step_count = 0

    def sync_and_update() -> None:
        nonlocal step_count
        try:
            if not viewer.is_running():
                return
            if step_count % 4 == 0:
                reading = ft_sensor.read(data)
                ft_display.update(reading)
            if step_count % 4 == 2:
                ft_display.show()
            viewer.sync()
        except Exception:
            pass

    def execute_demo() -> None:
        nonlocal demo_step, demo_wait_frames

        if demo_wait_frames > 0:
            demo_wait_frames -= 1
            return

        if demo_step == 0:
            print(">>> Moving to home...")
            move_arm_to_angles(
                model, data, np.zeros(len(arm_joints)), arm_joints,
                joint_to_actuator, update_fn=sync_and_update,
            )
            demo_step = 1
            demo_wait_frames = 30

        elif demo_step == 1:
            print(">>> Approaching ball...")
            move_arm_to_angles(
                model, data, ik_plan["approach"].angles, arm_joints,
                joint_to_actuator, update_fn=sync_and_update,
            )
            demo_step = 2
            demo_wait_frames = 30

        elif demo_step == 2:
            print(">>> Descending to grasp...")
            move_arm_to_angles(
                model, data, ik_plan["grasp"].angles, arm_joints,
                joint_to_actuator, speed=0.4, update_fn=sync_and_update,
            )
            demo_step = 3
            demo_wait_frames = 20

        elif demo_step == 3:
            print(">>> Closing gripper — watch force display...")
            close_gripper(model, data, gripper_limits, joint_to_actuator, update_fn=sync_and_update)
            reading = ft_sensor.read(data)
            print(f"    Grasp force: Fx={reading.force[0]:.3f} Fy={reading.force[1]:.3f} Fz={reading.force[2]:.3f} N")
            print(f"    Grasp torque: Mx={reading.torque[0]:.4f} My={reading.torque[1]:.4f} Mz={reading.torque[2]:.4f} Nm")
            demo_step = 4
            demo_wait_frames = 120

        elif demo_step == 4:
            print(">>> Lifting — observe weight on Fz...")
            move_arm_to_angles(
                model, data, ik_plan["lift"].angles, arm_joints,
                joint_to_actuator, update_fn=sync_and_update,
            )
            demo_step = 5
            demo_wait_frames = 180

        elif demo_step == 5:
            print(">>> Releasing...")
            command_gripper(model, data, gripper_limits, joint_to_actuator, "open")
            demo_step = 6
            demo_wait_frames = 120

        elif demo_step == 6:
            sync_position_actuators(model, data, joint_to_actuator)
            print(">>> Demo complete. Force display continues in real-time.")
            demo_step = 99

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.distance = 1.1
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -20
        viewer.cam.lookat[:] = [0.25, 0.0, 0.15]

        while viewer.is_running():
            step_start = time.perf_counter()
            execute_demo()
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
    print("Done.")


if __name__ == "__main__":
    main()
