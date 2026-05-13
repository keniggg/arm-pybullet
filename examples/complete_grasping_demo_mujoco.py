#!/usr/bin/env python3
"""Complete MuJoCo grasping demo for Alicia_D with a 50 mm gripper.

This example keeps all demo-only objects in the generated XML string:
target ball, fixed cameras, and position actuators.  The source MJCF under
synriard/ is not modified.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import math
import os
import time
from typing import Callable

import numpy as np

os.environ.setdefault("MUJOCO_GL", "glfw")

try:
    import mujoco
    import mujoco.viewer

    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("Warning: mujoco is not available. Install it with: pip install mujoco")

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import tkinter as tk

    TK_AVAILABLE = True
except ImportError:
    tk = None
    TK_AVAILABLE = False

from synriard import get_model_path


SIM_HZ = 240.0
SLEEP_SCALE = 0.15

BALL_POS = np.array([0.30, 0.00, 0.05], dtype=np.float64)
BALL_RADIUS = 0.025
APPROACH_OFFSET = np.array([0.0, 0.0, 0.15], dtype=np.float64)
GRASP_OFFSET = np.array([0.0, 0.0, BALL_RADIUS * 0.6], dtype=np.float64)
LIFT_OFFSET = np.array([0.0, 0.0, 0.20], dtype=np.float64)
PLACE_POS = np.array([0.45, 0.15, 0.20], dtype=np.float64)

IK_TOL = 0.006
RGB_CAMERA_NAME = "wrist_rgb"
RGB_CAMERA_WIDTH = 320
RGB_CAMERA_HEIGHT = 240
AUTO_RUN_DEMO = True


@dataclass
class IKResult:
    angles: np.ndarray
    target: np.ndarray
    final_pos: np.ndarray
    error_norm: float
    iterations: int
    success: bool


def _first_tag_end(xml_content: str, tag_name: str) -> int:
    start = xml_content.find(f"<{tag_name}")
    if start < 0:
        return -1
    return xml_content.find(">", start)


def inject_demo_xml(xml_content: str) -> str:
    """Add demo-only objects to the MJCF string."""
    if "<option" not in xml_content:
        tag_end = _first_tag_end(xml_content, "mujoco")
        if tag_end >= 0:
            xml_content = (
                xml_content[: tag_end + 1]
                + '\n  <option integrator="implicitfast" cone="elliptic"/>'
                + xml_content[tag_end + 1 :]
            )
    elif "integrator=" not in xml_content:
        xml_content = xml_content.replace("<option", '<option integrator="implicitfast"', 1)

    if 'name="overview"' not in xml_content:
        overview_xml = (
            '    <camera name="overview" pos="0.85 -0.75 0.55" '
            'xyaxes="0.66 0.75 0 -0.25 0.22 0.94" fovy="45"/>\n'
        )
        worldbody_end = xml_content.find("<worldbody>")
        if worldbody_end >= 0:
            insert_at = xml_content.find(">", worldbody_end) + 1
            xml_content = xml_content[:insert_at] + "\n" + overview_xml + xml_content[insert_at:]

    if 'name="target_ball"' not in xml_content:
        ball_xml = f"""
        <body name="target_ball" pos="{BALL_POS[0]} {BALL_POS[1]} {BALL_POS[2]}">
            <freejoint/>
            <geom name="ball_geom" type="sphere" size="{BALL_RADIUS}"
                  rgba="1 0 0 1" mass="0.035" condim="4"
                  friction="1.5 0.08 0.02" solimp="0.95 0.99 0.001"
                  solref="0.01 1"/>
        </body>"""
        xml_content = xml_content.replace("</worldbody>", ball_xml + "\n  </worldbody>")

    if f'name="{RGB_CAMERA_NAME}"' not in xml_content:
        wrist_camera_xml = (
            f'                  <camera name="{RGB_CAMERA_NAME}" pos="0 0 0.085" '
            'xyaxes="1 0 0 0 -1 0" fovy="65"/>\n'
        )
        marker = '<geom size="0.005" pos="-0.0002 -0.0003 0.13118"'
        marker_pos = xml_content.find(marker)
        if marker_pos >= 0:
            insert_at = xml_content.find("/>", marker_pos) + 2
            xml_content = xml_content[:insert_at] + "\n" + wrist_camera_xml + xml_content[insert_at:]

    if "<actuator>" not in xml_content:
        actuator_section = """
  <actuator>
    <position name="Joint1_act" joint="Joint1" kp="90" kv="12"
              ctrlrange="-2.16 2.16" forcerange="-25 25"/>
    <position name="Joint2_act" joint="Joint2" kp="90" kv="12"
              ctrlrange="-1.57 1.57" forcerange="-25 25"/>
    <position name="Joint3_act" joint="Joint3" kp="90" kv="12"
              ctrlrange="-0.5 2.35619" forcerange="-25 25"/>
    <position name="Joint4_act" joint="Joint4" kp="70" kv="10"
              ctrlrange="-2.7925 2.7925" forcerange="-20 20"/>
    <position name="Joint5_act" joint="Joint5" kp="70" kv="10"
              ctrlrange="-1.57 1.57" forcerange="-20 20"/>
    <position name="Joint6_act" joint="Joint6" kp="45" kv="7"
              ctrlrange="-3.14 3.14" forcerange="-15 15"/>
    <position name="left_finger_act" joint="left_finger" kp="55" kv="7"
              forcerange="-8 8"/>
    <position name="right_finger_act" joint="right_finger" kp="55" kv="7"
              forcerange="-8 8"/>
  </actuator>"""
        xml_content = xml_content.replace("</mujoco>", actuator_section + "\n</mujoco>")

    return xml_content


def gripper_center(data, left_body_id: int, right_body_id: int) -> np.ndarray:
    return (data.xpos[left_body_id] + data.xpos[right_body_id]) * 0.5


def set_joint_positions(model, data, joints: list[int], values: np.ndarray) -> None:
    for joint_id, value in zip(joints, values):
        qadr = model.jnt_qposadr[joint_id]
        data.qpos[qadr] = value


def sync_position_actuators(model, data, joint_to_actuator: dict[int, int]) -> None:
    for joint_id, act_id in joint_to_actuator.items():
        data.ctrl[act_id] = data.qpos[model.jnt_qposadr[joint_id]]


def solve_gripper_center_ik(
    model,
    seed_data,
    target_pos: np.ndarray,
    left_body_id: int,
    right_body_id: int,
    joint_indices: list[int],
    max_iter: int = 500,
    tol: float = IK_TOL,
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
    gripper_limits: list[dict[str, float]],
) -> tuple[bool, dict[str, IKResult]]:
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos[:]
    scratch.qvel[:] = 0.0

    home_angles = np.zeros(len(arm_joints), dtype=np.float64)
    set_joint_positions(model, scratch, arm_joints, home_angles)
    for item in gripper_limits:
        scratch.qpos[model.jnt_qposadr[item["joint"]]] = item["open"]
    mujoco.mj_forward(model, scratch)

    targets = {
        "approach": BALL_POS + APPROACH_OFFSET,
        "grasp": BALL_POS + GRASP_OFFSET,
        "lift": BALL_POS + GRASP_OFFSET + LIFT_OFFSET,
        "place": PLACE_POS,
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


def move_arm_to_angles(
    model,
    data,
    target_angles: np.ndarray | None,
    arm_joints: list[int],
    joint_to_actuator: dict[int, int],
    speed: float = 1.0,
    update_fn: Callable[[], None] | None = None,
) -> None:
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

    for step in range(steps):
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
        if SLEEP_SCALE > 0:
            time.sleep((1.0 / SIM_HZ) * SLEEP_SCALE)


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


def close_gripper(
    model,
    data,
    gripper_limits: list[dict[str, float]],
    joint_to_actuator: dict[int, int],
    force_threshold: float = 1.8,
    update_fn: Callable[[], None] | None = None,
) -> bool:
    max_steps = 360
    for step in range(max_steps):
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
        if SLEEP_SCALE > 0:
            time.sleep((1.0 / SIM_HZ) * SLEEP_SCALE)
    return False


def build_gripper_limits(model, data, gripper_joints: list[int]) -> list[dict[str, float]]:
    """Choose open/closed values by measuring finger distance at each limit."""
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


class RGBCameraWindow:
    def __init__(self, model, camera_id: int):
        self.camera_id = camera_id
        self.renderer = mujoco.Renderer(
            model, height=RGB_CAMERA_HEIGHT, width=RGB_CAMERA_WIDTH
        )
        self.backend = "cv2" if CV2_AVAILABLE else "tk"
        self.enabled = True
        self.root = None
        self.label = None
        self.photo = None
        self.error_reported = False

        if self.backend == "cv2":
            cv2.namedWindow("Wrist RGB Camera", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Wrist RGB Camera", RGB_CAMERA_WIDTH, RGB_CAMERA_HEIGHT)
            cv2.moveWindow("Wrist RGB Camera", 40, 40)
        else:
            if not TK_AVAILABLE:
                raise RuntimeError("neither OpenCV nor tkinter is available")
            self.root = tk.Tk()
            self.root.title("Wrist RGB Camera")
            self.root.resizable(False, False)
            self.root.geometry(f"{RGB_CAMERA_WIDTH}x{RGB_CAMERA_HEIGHT}+40+40")
            self.root.attributes("-topmost", True)
            self.root.after(1500, lambda: self.root.attributes("-topmost", False))
            self.label = tk.Label(
                self.root,
                text="Waiting for RGB camera frame...",
                bg="black",
                fg="white",
                bd=0,
            )
            self.label.pack(fill="both", expand=True)
            self.root.update_idletasks()
            self.root.update()

    def update(self, data) -> None:
        if not self.enabled:
            return

        try:
            self.renderer.update_scene(data, camera=self.camera_id)
            rgb = self.renderer.render()

            if self.backend == "cv2":
                cv2.imshow("Wrist RGB Camera", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
                return

            ppm = (
                f"P6\n{RGB_CAMERA_WIDTH} {RGB_CAMERA_HEIGHT}\n255\n".encode("ascii")
                + rgb.tobytes()
            )
            encoded = base64.b64encode(ppm).decode("ascii")
            try:
                self.photo = tk.PhotoImage(data=encoded, format="PPM")
            except Exception:
                self.photo = tk.PhotoImage(data=ppm, format="PPM")
            self.label.configure(image=self.photo)
            self.root.update_idletasks()
            self.root.update()
        except Exception as exc:
            if not self.error_reported:
                print(f"RGB camera: frame update failed ({exc}).")
                self.error_reported = True
            self.enabled = False

    def close(self) -> None:
        self.enabled = False
        if self.renderer is not None:
            self.renderer.close()
        if self.backend == "cv2" and CV2_AVAILABLE:
            cv2.destroyWindow("Wrist RGB Camera")
        elif self.root is not None:
            try:
                self.root.destroy()
            except Exception:
                pass


def main() -> None:
    if not MUJOCO_AVAILABLE:
        print("Error: MuJoCo is not available.")
        return

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(repo_root)

    model_path = get_model_path(
        "Alicia_D", version="v5_6", variant="gripper_50mm", model_format="mjcf"
    )
    with open(model_path, "r", encoding="utf-8") as file:
        xml_content = inject_demo_xml(file.read())

    xml_dir = os.path.dirname(model_path)
    os.chdir(xml_dir)
    model = mujoco.MjModel.from_xml_string(xml_content)
    os.chdir(repo_root)

    data = mujoco.MjData(model)
    model.opt.timestep = 1.0 / SIM_HZ

    arm_joints: list[int] = []
    gripper_joints: list[int] = []
    for joint_id in range(model.njnt):
        joint_type = model.jnt_type[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        if joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
            if "finger" in name.lower():
                gripper_joints.append(joint_id)
            else:
                arm_joints.append(joint_id)

    left_finger_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Link7")
    right_finger_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Link8")
    if left_finger_body_id < 0 or right_finger_body_id < 0:
        raise RuntimeError("Could not find Link7/Link8 finger bodies in the model.")

    joint_to_actuator: dict[int, int] = {}
    for act_id in range(model.nu):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id) or ""
        joint_name = act_name[:-4] if act_name.endswith("_act") else act_name
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            joint_to_actuator[int(joint_id)] = int(act_id)

    mujoco.mj_forward(model, data)
    gripper_limits = build_gripper_limits(model, data, gripper_joints)
    command_gripper(model, data, gripper_limits, joint_to_actuator, "open")
    sync_position_actuators(model, data, joint_to_actuator)
    mujoco.mj_forward(model, data)

    print(f"Info: arm_joints={arm_joints}, gripper_joints={gripper_joints}")
    print(f"Info: initial gripper center={np.round(gripper_center(data, left_finger_body_id, right_finger_body_id), 4)}")
    print(f"Info: ball position={BALL_POS}, radius={BALL_RADIUS}")
    print(f"Info: gripper limits={gripper_limits}")

    plan_ok, ik_plan = build_ik_plan(
        model, data, arm_joints, left_finger_body_id, right_finger_body_id, gripper_limits
    )
    if plan_ok:
        print("Reachability: target ball and planned place point are inside the IK workspace.")
    else:
        print("Reachability: at least one planned target is outside the IK workspace; demo will not auto-run.")

    rgb_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, RGB_CAMERA_NAME)
    rgb_window = None
    if rgb_camera_id >= 0:
        try:
            rgb_window = RGBCameraWindow(model, rgb_camera_id)
            rgb_window.update(data)
            print(f"RGB camera: showing separate {rgb_window.backend} window 'Wrist RGB Camera'.")
        except Exception as exc:
            print(f"RGB camera: separate window failed to start ({exc}).")

    demo_step = 0 if plan_ok and AUTO_RUN_DEMO else 99
    if demo_step == 99:
        print("Manual mode: use the MuJoCo Control panel sliders to command the arm.")
    demo_substep = 0
    demo_wait_frames = 0
    demo_complete_sync_done = False

    def execute_demo_step(update_fn: Callable[[], None] | None = None) -> None:
        nonlocal demo_step, demo_substep, demo_wait_frames, demo_complete_sync_done

        if demo_wait_frames > 0:
            demo_wait_frames -= 1
            return

        if demo_step == 0:
            print(">>> Step 0: open gripper and move to home.")
            command_gripper(model, data, gripper_limits, joint_to_actuator, "open")
            move_arm_to_angles(
                model,
                data,
                np.zeros(len(arm_joints)),
                arm_joints,
                joint_to_actuator,
                update_fn=update_fn,
            )
            demo_step = 1
            demo_wait_frames = 60

        elif demo_step == 1:
            print(">>> Step 1: move above the ball.")
            move_arm_to_angles(
                model, data, ik_plan["approach"].angles, arm_joints, joint_to_actuator, update_fn=update_fn
            )
            demo_step = 2
            demo_wait_frames = 60

        elif demo_step == 2:
            print(">>> Step 2: descend to grasp pose.")
            move_arm_to_angles(
                model,
                data,
                ik_plan["grasp"].angles,
                arm_joints,
                joint_to_actuator,
                speed=0.55,
                update_fn=update_fn,
            )
            demo_step = 3
            demo_wait_frames = 45

        elif demo_step == 3:
            print(">>> Step 3: close gripper.")
            contact_seen = close_gripper(
                model, data, gripper_limits, joint_to_actuator, update_fn=update_fn
            )
            print(f"Gripper contact/force threshold reached: {contact_seen}")
            demo_step = 4
            demo_wait_frames = 90

        elif demo_step == 4:
            print(">>> Step 4: lift object.")
            move_arm_to_angles(
                model, data, ik_plan["lift"].angles, arm_joints, joint_to_actuator, update_fn=update_fn
            )
            demo_step = 5
            demo_wait_frames = 60

        elif demo_step == 5:
            if demo_substep == 0:
                print(">>> Step 5: move to place pose.")
                move_arm_to_angles(
                    model,
                    data,
                    ik_plan["place"].angles,
                    arm_joints,
                    joint_to_actuator,
                    update_fn=update_fn,
                )
                demo_substep = 1
                demo_wait_frames = 60
            else:
                print(">>> Step 5b: open gripper.")
                command_gripper(model, data, gripper_limits, joint_to_actuator, "open")
                demo_step = 6
                demo_wait_frames = 120

        elif demo_step == 6:
            sync_position_actuators(model, data, joint_to_actuator)
            demo_complete_sync_done = True
            print(">>> Demo complete. Use the MuJoCo Control panel sliders for manual control.")
            demo_step = 7

    print("=== MuJoCo grasping demo start ===")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.distance = 1.2
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.lookat[:] = [0.26, 0.0, 0.16]

        def sync_and_update() -> None:
            try:
                if viewer.is_running():
                    if rgb_window is not None:
                        rgb_window.update(data)
                    viewer.sync()
            except Exception:
                pass

        while viewer.is_running():
            execute_demo_step(update_fn=sync_and_update)

            if demo_step == 99 and not demo_complete_sync_done:
                sync_position_actuators(model, data, joint_to_actuator)
                demo_complete_sync_done = True

            mujoco.mj_step(model, data)
            sync_and_update()
            if SLEEP_SCALE > 0:
                time.sleep((1.0 / SIM_HZ) * SLEEP_SCALE)

    if rgb_window is not None:
        rgb_window.close()


if __name__ == "__main__":
    main()
