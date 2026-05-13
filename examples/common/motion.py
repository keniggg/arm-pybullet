"""Smooth motion control and gripper utilities."""
from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np
import mujoco

from .model_loader import SIM_HZ

SLEEP_SCALE = 0.15


def sync_position_actuators(model, data, joint_to_actuator: dict[int, int]) -> None:
    for joint_id, act_id in joint_to_actuator.items():
        data.ctrl[act_id] = data.qpos[model.jnt_qposadr[joint_id]]


def build_gripper_limits(model, data, gripper_joints: list[int]) -> list[dict[str, float]]:
    """Detect open/closed values for each gripper joint."""
    limits = []
    base_qpos = data.qpos.copy()

    for joint_id in gripper_joints:
        lo, hi = float(model.jnt_range[joint_id][0]), float(model.jnt_range[joint_id][1])
        if not hi > lo:
            continue

        distances = []
        for value in (lo, hi):
            data.qpos[:] = base_qpos
            data.qpos[model.jnt_qposadr[joint_id]] = value
            mujoco.mj_forward(model, data)
            joint_body = int(model.jnt_bodyid[joint_id])
            distances.append(float(np.linalg.norm(data.xpos[joint_body])))

        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or "").lower()
        if "left" in name:
            open_value, closed_value = hi, lo
        elif "right" in name:
            open_value, closed_value = lo, hi
        else:
            open_value, closed_value = (lo, hi) if distances[0] > distances[1] else (hi, lo)

        limits.append({"joint": joint_id, "open": open_value, "closed": closed_value})

    data.qpos[:] = base_qpos
    mujoco.mj_forward(model, data)
    return limits

def command_gripper(
    model,
    data,
    gripper_limits: list[dict[str, float]],
    joint_to_actuator: dict[int, int],
    key: str,
) -> None:
    for item in gripper_limits:
        act_id = joint_to_actuator.get(item["joint"])
        if act_id is not None:
            data.ctrl[act_id] = item[key]


def move_arm_to_angles(
    model,
    data,
    target_angles: np.ndarray | None,
    arm_joints: list[int],
    joint_to_actuator: dict[int, int],
    speed: float = 1.0,
    update_fn: Callable[[], None] | None = None,
) -> None:
    """Move arm smoothly using cosine interpolation."""
    if target_angles is None:
        return

    current_angles = np.array(
        [data.qpos[model.jnt_qposadr[j]] for j in arm_joints], dtype=np.float64
    )
    angle_diffs = target_angles - current_angles
    max_diff = float(np.max(np.abs(angle_diffs))) if len(angle_diffs) else 0.0
    if max_diff < 1e-6:
        return

    steps = max(int(max_diff / (speed * 0.008)) + 1, 80)
    steps = min(steps, 520)
    target_dt = (1.0 / SIM_HZ) * SLEEP_SCALE

    for step in range(steps):
        step_start = time.perf_counter()
        t = (step + 1) / steps
        t_smooth = 0.5 - 0.5 * math.cos(t * math.pi)
        interp = current_angles + angle_diffs * t_smooth

        for idx, joint_id in enumerate(arm_joints):
            act_id = joint_to_actuator.get(joint_id)
            if act_id is not None:
                data.ctrl[act_id] = interp[idx]

        mujoco.mj_step(model, data)
        if update_fn:
            update_fn()

        elapsed = time.perf_counter() - step_start
        remaining = target_dt - elapsed
        if remaining > 0:
            time.sleep(remaining)


def close_gripper(
    model,
    data,
    gripper_limits: list[dict[str, float]],
    joint_to_actuator: dict[int, int],
    force_threshold: float = 1.8,
    update_fn: Callable[[], None] | None = None,
) -> bool:
    """Close gripper gradually, stopping when force threshold is reached."""
    max_steps = 360
    target_dt = (1.0 / SIM_HZ) * SLEEP_SCALE

    for step in range(max_steps):
        step_start = time.perf_counter()
        t = (step + 1) / max_steps
        max_force = 0.0
        for item in gripper_limits:
            act_id = joint_to_actuator.get(item["joint"])
            if act_id is None:
                continue
            data.ctrl[act_id] = item["open"] + (item["closed"] - item["open"]) * t
            if hasattr(data, "actuator_force"):
                max_force = max(max_force, abs(float(data.actuator_force[act_id])))

        mujoco.mj_step(model, data)
        if update_fn:
            update_fn()
        if max_force >= force_threshold:
            return True

        elapsed = time.perf_counter() - step_start
        remaining = target_dt - elapsed
        if remaining > 0:
            time.sleep(remaining)
    return False
