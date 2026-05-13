"""Inverse kinematics solver for the Alicia_D gripper center."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco


@dataclass
class IKResult:
    angles: np.ndarray
    target: np.ndarray
    final_pos: np.ndarray
    error_norm: float
    iterations: int
    success: bool


def gripper_center(data, left_body_id: int, right_body_id: int) -> np.ndarray:
    return (data.xpos[left_body_id] + data.xpos[right_body_id]) * 0.5


def set_joint_positions(model, data, joints: list[int], values: np.ndarray) -> None:
    for joint_id, value in zip(joints, values):
        data.qpos[model.jnt_qposadr[joint_id]] = value


def solve_gripper_center_ik(
    model,
    seed_data,
    target_pos: np.ndarray,
    left_body_id: int,
    right_body_id: int,
    joint_indices: list[int],
    max_iter: int = 500,
    tol: float = 0.006,
) -> IKResult:
    """Damped least-squares IK for the midpoint between the two gripper bodies."""
    tmp_data = mujoco.MjData(model)
    tmp_data.qpos[:] = seed_data.qpos[:]
    tmp_data.qvel[:] = 0.0
    mujoco.mj_forward(model, tmp_data)

    jacp_l = np.zeros((3, model.nv))
    jacr_l = np.zeros((3, model.nv))
    jacp_r = np.zeros((3, model.nv))
    jacr_r = np.zeros((3, model.nv))

    dof_indices = [model.jnt_dofadr[j] for j in joint_indices]
    lower = np.array([model.jnt_range[j][0] for j in joint_indices], dtype=np.float64)
    upper = np.array([model.jnt_range[j][1] for j in joint_indices], dtype=np.float64)

    lam = 0.025
    step_size = 0.45
    final_error = float("inf")
    final_pos = gripper_center(tmp_data, left_body_id, right_body_id)
    iteration = 0

    for iteration in range(1, max_iter + 1):
        mujoco.mj_forward(model, tmp_data)
        final_pos = gripper_center(tmp_data, left_body_id, right_body_id)
        error = target_pos - final_pos
        final_error = float(np.linalg.norm(error))
        if final_error <= tol:
            break

        mujoco.mj_jacBody(model, tmp_data, jacp_l, jacr_l, left_body_id)
        mujoco.mj_jacBody(model, tmp_data, jacp_r, jacr_r, right_body_id)
        jac = ((jacp_l + jacp_r) * 0.5)[:, dof_indices]

        lhs = jac @ jac.T + lam * np.eye(3)
        delta_q = jac.T @ np.linalg.solve(lhs, error)
        delta_q = np.clip(delta_q, -0.08, 0.08)

        q = np.array([tmp_data.qpos[model.jnt_qposadr[j]] for j in joint_indices])
        q = np.clip(q + step_size * delta_q, lower, upper)
        set_joint_positions(model, tmp_data, joint_indices, q)

    angles = np.array([tmp_data.qpos[model.jnt_qposadr[j]] for j in joint_indices])
    return IKResult(
        angles=angles,
        target=target_pos.copy(),
        final_pos=final_pos.copy(),
        error_norm=final_error,
        iterations=iteration,
        success=final_error <= tol,
    )


def build_ik_plan(
    model,
    data,
    arm_joints: list[int],
    left_body_id: int,
    right_body_id: int,
    gripper_limits: list[dict],
    targets: dict[str, np.ndarray] | None = None,
) -> tuple[bool, dict[str, IKResult]]:
    """Pre-compute IK for a set of named target positions."""
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos[:]
    scratch.qvel[:] = 0.0

    home_angles = np.zeros(len(arm_joints), dtype=np.float64)
    set_joint_positions(model, scratch, arm_joints, home_angles)
    for item in gripper_limits:
        scratch.qpos[model.jnt_qposadr[item["joint"]]] = item["open"]
    mujoco.mj_forward(model, scratch)

    if targets is None:
        from .model_loader import BALL_POS, BALL_RADIUS
        targets = {
            "approach": BALL_POS + np.array([0.0, 0.0, 0.15]),
            "grasp": BALL_POS + np.array([0.0, 0.0, BALL_RADIUS * 0.6]),
            "lift": BALL_POS + np.array([0.0, 0.0, BALL_RADIUS * 0.6 + 0.20]),
            "place": np.array([0.45, 0.15, 0.20]),
        }

    plan: dict[str, IKResult] = {}
    ok = True
    for name, target in targets.items():
        result = solve_gripper_center_ik(
            model, scratch, target, left_body_id, right_body_id, arm_joints
        )
        plan[name] = result
        status = "reachable" if result.success else "unreachable"
        print(
            f"IK {name:8s}: {status}, error={result.error_norm:.4f} m, "
            f"target={np.round(target, 4)}, reached={np.round(result.final_pos, 4)}"
        )
        if not result.success:
            ok = False
            break
        set_joint_positions(model, scratch, arm_joints, result.angles)
        mujoco.mj_forward(model, scratch)

    return ok, plan
