#!/usr/bin/env python3
"""Interactive grasping demo — drag the ball, press SPACE, watch the arm find & fetch it.

Features:
  - Ball is draggable (Ctrl + drag in MuJoCo viewer) anywhere in the workspace
  - A small target box sits inside the reachable region
  - Press SPACE to trigger the autonomous sequence:
      1. Camera scanning — find the red ball via OpenCV colour detection
      2. Dynamic IK planning — compute approach / grasp / lift / place targets
      3. Smooth arm motion with cosine interpolation (non-blocking)
      4. Force-controlled gripper close
      5. Place the ball into the box
  - Real-time 6-axis F/T display, eye-in-hand camera, joint-state panel
  - Demo complete → manual control via MuJoCo sliders restored
"""

from __future__ import annotations

import ctypes
import math
import os
import random
import sys
import time
import numpy as np

# ── Windows high-resolution timer ──────────────────────────────────────
try:
    ctypes.windll.winmm.timeBeginPeriod(1)
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("MUJOCO_GL", "glfw")
import mujoco
import mujoco.viewer

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from common.model_loader import load_and_inject, RGB_CAMERA_NAME, BALL_POS
from common.ik_solver import (
    solve_gripper_center_ik, set_joint_positions, IKResult,
)
from common.motion import build_gripper_limits, command_gripper
from common.force_sensor import ForceTorqueSensor, FTDisplay
from common.camera import RGBCameraWindow

# ────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────
TARGET_DISPLAY_HZ = 50
PHYSICS_SUBSTEPS = 5
FRAME_DT = 1.0 / TARGET_DISPLAY_HZ

FT_EVERY_N = 2         # 25 Hz
JOINT_EVERY_N = 4      # 12.5 Hz
CAMERA_EVERY_N = 2     # 25 Hz

SPEED_NORMAL = 1.4
SPEED_SLOW = 0.55  # very slow for precise grasp positioning
SPEED_SCAN = 0.6

WORKSPACE_MAX_SPHERES = 250

# Colour ranges for red ball detection (HSV) — lenient to catch varying lighting
RED_LOWER_1 = (0, 60, 50)
RED_UPPER_1 = (18, 255, 255)
RED_LOWER_2 = (158, 60, 50)
RED_UPPER_2 = (180, 255, 255)

MIN_BALL_AREA_PX = 12       # minimum contour area (small ball at distance ≈ 15-30 px)
CUSTOM_BALL_RADIUS = 0.020  # 40 mm diameter — good grip depth in 50 mm gripper

# ────────────────────────────────────────────────────────────────────────
# XML injection — target box
# ────────────────────────────────────────────────────────────────────────
BOX_POS = np.array([0.42, -0.15, 0.06], dtype=np.float64)
BOX_SIZE = np.array([0.07, 0.07, 0.04], dtype=np.float64)  # half-extents
BOX_WALL = 0.005  # wall thickness


def inject_control_actuators(xml_content: str) -> str:
    """Add dummy actuators that appear as sliders in the MuJoCo viewer.

    These produce zero force (gainprm=0) so they don't affect physics.
    The sliders are used purely as UI controls for:
      - start_demo:  slide to 1 to trigger the autonomous sequence
      - ball_x/y/z:  reposition the target ball (only during idle)
    """
    if 'name="start_demo"' in xml_content:
        return xml_content

    ctrl_xml = """
    <general name="start_demo" joint="Joint1" biastype="affine" gaintype="fixed"
             ctrlrange="0 1" dyntype="none" gainprm="0" biasprm="0 0 0"/>
    <general name="ball_x" joint="Joint1" biastype="affine" gaintype="fixed"
             ctrlrange="0.10 0.46" dyntype="none" gainprm="0" biasprm="0 0 0"/>
    <general name="ball_y" joint="Joint1" biastype="affine" gaintype="fixed"
             ctrlrange="-0.30 0.30" dyntype="none" gainprm="0" biasprm="0 0 0"/>
    <general name="ball_z" joint="Joint1" biastype="affine" gaintype="fixed"
             ctrlrange="0.03 0.35" dyntype="none" gainprm="0" biasprm="0 0 0"/>
  """
    # Insert into the existing <actuator> section, right before </actuator>
    act_end = xml_content.find("</actuator>")
    if act_end >= 0:
        xml_content = xml_content[:act_end] + ctrl_xml + xml_content[act_end:]
    return xml_content


def inject_target_box(xml_content: str) -> str:
    if 'name="target_box"' in xml_content:
        return xml_content

    bx, by, bz = BOX_POS
    sx, sy, sz = BOX_SIZE
    t = BOX_WALL
    rgba = (0.3, 0.6, 1.0, 0.7)  # blueish semi-transparent

    box_body = f"""
        <body name="target_box" pos="{bx} {by} {bz}">
            <geom name="box_bottom" type="box" size="{sx} {sy} {t/2}"
                  pos="0 0 {-sz}" rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>
            <geom name="box_front" type="box" size="{sx} {t/2} {sz}"
                  pos="0 {sy} 0" rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>
            <geom name="box_back" type="box" size="{sx} {t/2} {sz}"
                  pos="0 {-sy} 0" rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>
            <geom name="box_left" type="box" size="{t/2} {sy} {sz}"
                  pos="{-sx} 0 0" rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>
            <geom name="box_right" type="box" size="{t/2} {sy} {sz}"
                  pos="{sx} 0 0" rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>
        </body>"""
    xml_content = xml_content.replace("</worldbody>", box_body + "\n  </worldbody>")
    return xml_content


# ────────────────────────────────────────────────────────────────────────
# Ball detector — camera-based red-ball localisation
# ────────────────────────────────────────────────────────────────────────
class BallDetector:
    """Find the red ball in an RGB image and estimate its 3-D position."""

    def __init__(self, model, camera_id: int, ball_body_id: int):
        self._model = model
        self._camera_id = camera_id
        self._ball_body_id = ball_body_id
        self._last_detection_uv: tuple[int, int] | None = None
        self._last_estimated_xyz: np.ndarray | None = None

    def detect(self, rgb: np.ndarray, data) -> tuple[tuple[int, int] | None, int]:
        """Return ((cx, cy), radius_px) or (None, 0) if not found."""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
        mask2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
        mask = mask1 | mask2
        # Morphological clean-up: remove small noise, merge nearby blobs
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < MIN_BALL_AREA_PX:
            return None, 0
        (cx, cy), radius = cv2.minEnclosingCircle(largest)
        return (int(cx), int(cy)), int(radius)

    def estimate_3d(self, center_uv: tuple[int, int], data,
                    img_w: int, img_h: int,
                    ball_z: float | None = None) -> np.ndarray:
        """Ray-plane intersection: camera ray × horizontal plane at given Z."""
        cam_pos = data.cam_xpos[self._camera_id].copy()
        cam_mat = data.cam_xmat[self._camera_id].reshape(3, 3)

        fovy = float(self._model.cam_fovy[self._camera_id])
        f_px = (img_h / 2.0) / math.tan(math.radians(fovy) / 2.0)

        px = center_uv[0] - img_w / 2.0
        py = center_uv[1] - img_h / 2.0
        ray_cam = np.array([px / f_px, py / f_px, 1.0], dtype=np.float64)
        ray_cam /= np.linalg.norm(ray_cam)
        ray_world = cam_mat @ ray_cam

        z_plane = ball_z if ball_z is not None else CUSTOM_BALL_RADIUS
        if abs(ray_world[2]) < 1e-9:
            return data.xpos[self._ball_body_id].copy()
        t = (z_plane - cam_pos[2]) / ray_world[2]
        if t < 0:
            return data.xpos[self._ball_body_id].copy()
        est = cam_pos + t * ray_world
        est[2] = z_plane
        return est

    @property
    def last_estimated_xyz(self) -> np.ndarray | None:
        return self._last_estimated_xyz


# ────────────────────────────────────────────────────────────────────────
# Non-blocking smooth arm controller
# ────────────────────────────────────────────────────────────────────────
class SmoothArmController:
    """Cosine-interpolated joint-space motion  (non-blocking).

    After interpolation finishes the controller enters a *settling* phase:
    it keeps writing the final target for SETTLE_FRAMES more frames so
    the PD actuators have time to physically reach the target despite
    gravity / dynamics.
    """

    SETTLE_FRAMES = 30  # frames — higher kp means faster convergence

    def __init__(self, model, data, arm_joints: list[int],
                 joint_to_actuator: dict[int, int]):
        self._model = model
        self._data = data
        self._joints = arm_joints
        self._j2a = joint_to_actuator
        cur = np.array([data.qpos[model.jnt_qposadr[j]] for j in arm_joints],
                       dtype=np.float64)
        self._start = cur.copy()
        self._target = cur.copy()
        self._progress = 1.0
        self._total_frames = 1
        self._done = True
        self._settle_left = 0

    @property
    def done(self) -> bool:
        """True only after interpolation + settling have both finished."""
        return self._done and self._settle_left <= 0

    def current(self) -> np.ndarray:
        return np.array([self._data.qpos[self._model.jnt_qposadr[j]]
                         for j in self._joints], dtype=np.float64)

    def set_target(self, angles: np.ndarray, speed: float = 1.0) -> None:
        cur = self.current()
        diffs = angles - cur
        max_diff = float(np.max(np.abs(diffs)))
        divisor = 0.008 * PHYSICS_SUBSTEPS
        min_f = max(80 // PHYSICS_SUBSTEPS, 10)
        raw = int(max_diff / (speed * divisor)) + 1
        self._total_frames = max(raw, min_f)
        self._total_frames = min(self._total_frames, 200)
        self._start = cur.copy()
        self._target = angles.copy()
        self._progress = 0.0
        self._done = False
        self._settle_left = 0

    def step(self) -> None:
        if self._done:
            if self._settle_left > 0:
                self._settle_left -= 1
            # Always hold the final target so the PD actuators can converge
            self._write_angles(self._target)
            return
        self._progress += 1.0 / self._total_frames
        if self._progress >= 1.0:
            self._progress = 1.0
            self._done = True
            self._settle_left = self.SETTLE_FRAMES
        t = 0.5 - 0.5 * math.cos(self._progress * math.pi)
        angles = self._start + (self._target - self._start) * t
        self._write_angles(angles)

    def _write_angles(self, angles: np.ndarray) -> None:
        for idx, jid in enumerate(self._joints):
            act = self._j2a.get(jid)
            if act is not None:
                self._data.ctrl[act] = float(angles[idx])


# ────────────────────────────────────────────────────────────────────────
# Non-blocking gripper controller
# ────────────────────────────────────────────────────────────────────────
class SmoothGripperController:
    """Rate-limited gripper with force-threshold stop  (non-blocking)."""

    def __init__(self, model, data, limits: list[dict],
                 joint_to_actuator: dict[int, int]):
        self._model = model
        self._data = data
        self._limits = limits
        self._j2a = joint_to_actuator
        self._progress = 1.0
        self._total_frames = 1
        self._mode: str | None = None
        self._done = True
        self._contact = False
        self._last_ctrl: dict[int, float] = {}

    @property
    def done(self) -> bool:
        return self._done

    @property
    def contact_triggered(self) -> bool:
        return self._contact

    def open(self, duration_frames: int = 25) -> None:
        self._mode = "open"
        self._progress = 0.0
        self._total_frames = max(duration_frames, 1)
        self._done = False
        self._contact = False

    def close(self, duration_frames: int = 100) -> None:
        self._mode = "close"
        self._progress = 0.0
        self._total_frames = max(duration_frames, 1)
        self._done = False
        self._contact = False

    def step(self, force_threshold: float = 2.0) -> None:
        if self._done:
            self._write_final()
            return
        self._progress += 1.0 / self._total_frames
        if self._progress >= 1.0:
            self._progress = 1.0
            self._done = True
            return
        max_force = 0.0
        for item in self._limits:
            act = self._j2a.get(item["joint"])
            if act is None:
                continue
            if self._mode == "close":
                val = item["open"] + (item["closed"] - item["open"]) * self._progress
            else:
                val = item["closed"] + (item["open"] - item["closed"]) * self._progress
            self._data.ctrl[act] = val
            self._last_ctrl[act] = val
            if hasattr(self._data, "actuator_force"):
                max_force = max(max_force, abs(float(self._data.actuator_force[act])))
        if self._mode == "close" and self._progress > 0.30 and max_force >= force_threshold:
            self._contact = True
            self._done = True

    def _write_final(self) -> None:
        if self._last_ctrl:
            for act, val in self._last_ctrl.items():
                self._data.ctrl[act] = val
        else:
            key = "open" if self._mode == "open" else "closed"
            for item in self._limits:
                act = self._j2a.get(item["joint"])
                if act is not None:
                    self._data.ctrl[act] = item[key]


# ────────────────────────────────────────────────────────────────────────
# Workspace sampling
# ────────────────────────────────────────────────────────────────────────
def compute_workspace(
    model, data, arm_joints: list[int],
    left_body_id: int, right_body_id: int,
    resolution: float = 0.04,
) -> list[np.ndarray]:
    xs = np.arange(0.08, 0.48, resolution)
    ys = np.arange(-0.30, 0.31, resolution)
    zs = np.arange(0.02, 0.42, resolution)
    total = len(xs) * len(ys) * len(zs)
    print(f"Sampling workspace: {len(xs)}x{len(ys)}x{len(zs)} = {total} pts "
          f"(res={resolution}m) ...")
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos[:]
    scratch.qvel[:] = 0.0
    home = np.zeros(len(arm_joints), dtype=np.float64)
    set_joint_positions(model, scratch, arm_joints, home)
    mujoco.mj_forward(model, scratch)
    reachable: list[np.ndarray] = []
    cnt = 0
    last_pct = -1
    for x in xs:
        for y in ys:
            for z in zs:
                cnt += 1
                pct = cnt * 100 // total
                if pct > last_pct:
                    last_pct = pct
                    if pct % 10 == 0:
                        print(f"  {pct}% ...")
                result = solve_gripper_center_ik(
                    model, scratch, np.array([x, y, z], dtype=np.float64),
                    left_body_id, right_body_id, arm_joints,
                )
                if result.success:
                    reachable.append(np.array([x, y, z], dtype=np.float64))
                    set_joint_positions(model, scratch, arm_joints, result.angles)
                    mujoco.mj_forward(model, scratch)
    print(f"Workspace done: {len(reachable)} reachable ({len(reachable)*100/total:.1f}%)")
    if len(reachable) > WORKSPACE_MAX_SPHERES:
        rng = random.Random(42)
        reachable = rng.sample(reachable, WORKSPACE_MAX_SPHERES)
        print(f"  →  subsampled to {len(reachable)} for smooth rendering")
    return reachable


def render_workspace_spheres(viewer, reachable: list[np.ndarray]) -> None:
    if not reachable:
        return
    with viewer.lock():
        viewer.user_scn.ngeom = 0
        r = 0.007
        rgba = (0.0, 0.85, 0.25, 0.38)
        for pt in reachable:
            g = viewer.user_scn.ngeom
            viewer.user_scn.ngeom += 1
            if g >= viewer.user_scn.maxgeom:
                break
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[g],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([r, 0, 0]), pt,
                np.eye(3, 1).flatten(), rgba,
            )


# ────────────────────────────────────────────────────────────────────────
# Joint-state panel  (OpenCV)
# ────────────────────────────────────────────────────────────────────────
class JointStatePanel:
    def __init__(self) -> None:
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV required.")
        self._w, self._h = 340, 260
        self._canvas = np.zeros((self._h, self._w, 3), dtype=np.uint8)
        cv2.namedWindow("Joint States", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Joint States", self._w, self._h)
        cv2.moveWindow("Joint States", 880, 40)

    def update(self, model, data,
               arm_joints: list[int], gripper_joints: list[int],
               status_text: str = "") -> None:
        self._canvas[:] = (25, 25, 30)
        y = 20
        if status_text:
            cv2.putText(self._canvas, status_text, (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
            y += 22
        cv2.putText(self._canvas, "--- ARM JOINTS ---", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
        y += 18
        for jid in arm_joints:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"J{jid}"
            rad = float(data.qpos[model.jnt_qposadr[jid]])
            deg = float(np.degrees(rad))
            cv2.putText(self._canvas,
                        f"  {name:12s} {deg:+8.2f} deg  ({rad:+.4f} rad)",
                        (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.37,
                        (0, 220, 255), 1, cv2.LINE_AA)
            y += 18
        y += 4
        cv2.putText(self._canvas, "--- FINGER JOINTS ---", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
        y += 18
        for jid in gripper_joints:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"F{jid}"
            rad = float(data.qpos[model.jnt_qposadr[jid]])
            deg = float(np.degrees(rad))
            cv2.putText(self._canvas,
                        f"  {name:12s} {deg:+8.2f} deg  ({rad:+.4f} rad)",
                        (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.37,
                        (100, 255, 100), 1, cv2.LINE_AA)
            y += 18

    def show(self) -> None:
        cv2.imshow("Joint States", self._canvas)

    def close(self) -> None:
        if CV2_AVAILABLE:
            cv2.destroyWindow("Joint States")


# ────────────────────────────────────────────────────────────────────────
# Scanning poses for ball search
# ────────────────────────────────────────────────────────────────────────
def generate_scan_targets(ball_z: float = 0.05) -> list[np.ndarray]:
    """Return a grid of 3-D points covering the workspace.

    The camera looks down from ~18 cm above each grid point, giving a
    wide-area search.  9 positions cover X ∈ [0.18, 0.38], Y ∈ [-0.20, 0.20].
    """
    targets: list[np.ndarray] = []
    z_look = ball_z + 0.18
    for x in [0.38, 0.28, 0.18]:
        for y in [-0.20, 0.0, 0.20]:
            targets.append(np.array([x, y, z_look], dtype=np.float64))
    return targets


def draw_detection_overlay(rgb: np.ndarray, found: bool,
                           xyz: np.ndarray | None = None) -> np.ndarray:
    """Draw ✓ (found) or ✗ (not found) + position on an RGB image."""
    h, w = rgb.shape[:2]
    overlay = rgb.copy()
    if found and xyz is not None:
        cv2.putText(overlay, "V", (w - 50, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, f"X:{xyz[0]:.3f}", (10, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(overlay, f"Y:{xyz[1]:.3f}", (10, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
    else:
        cv2.drawMarker(overlay, (w - 35, 25), (0, 0, 255),
                       cv2.MARKER_TILTED_CROSS, 20, 2, cv2.LINE_AA)
    return overlay


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 62)
    print("Interactive Grasping Demo")
    print("- Use MuJoCo sliders to pose the arm freely before starting")
    print("- Use 'ball_x / ball_y / ball_z' sliders to move the target ball")
    print("- Slide 'start_demo' to 1  →  autonomous detection & grasping")
    print("=" * 62)

    # ── Model loading with box injection ──────────────────────────────
    import synriard
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(repo_root)
    model_path = synriard.get_model_path(
        "Alicia_D", version="v5_6", variant="gripper_50mm", model_format="mjcf",
    )
    with open(model_path, "r", encoding="utf-8") as f:
        xml_content = f.read()

    xml_content = inject_target_box(xml_content)

    # Re-use model_loader injections, but bypass load_and_inject to add box
    from common.model_loader import (
        inject_options, inject_overview_camera,
        inject_wrist_camera, inject_force_sensor, inject_actuators,
        SIM_HZ,
    )
    xml_content = inject_options(xml_content)
    xml_content = inject_overview_camera(xml_content)
    # ── Large high-friction pad covering the workspace ───────────────
    if 'name="friction_pad"' not in xml_content:
        pad_xml = """
        <body name="friction_pad" pos="0.28 0.0 0.0">
            <geom name="pad_geom" type="box" size="0.22 0.30 0.003"
                  rgba="0.4 0.4 0.4 0.5" friction="8.0 5.0 0.8"/>
        </body>"""
        xml_content = xml_content.replace("</worldbody>", pad_xml + "\n  </worldbody>", 1)

    # ── Workspace boundary walls (keep ball in reachable area) ───────
    if 'name="wall_left"' not in xml_content:
        wall_thick = 0.002; wall_h = 0.08; x_c = 0.28
        walls = f"""
        <body name="wall_left"  pos="{x_c - 0.22} 0.0 {wall_h/2}">
            <geom type="box" size="{wall_thick} 0.30 {wall_h}" rgba="0.5 0.5 0.5 0.3"/>
        </body>
        <body name="wall_right" pos="{x_c + 0.22} 0.0 {wall_h/2}">
            <geom type="box" size="{wall_thick} 0.30 {wall_h}" rgba="0.5 0.5 0.5 0.3"/>
        </body>
        <body name="wall_front" pos="{x_c} 0.30 {wall_h/2}">
            <geom type="box" size="0.22 {wall_thick} {wall_h}" rgba="0.5 0.5 0.5 0.3"/>
        </body>
        <body name="wall_back"  pos="{x_c} -0.30 {wall_h/2}">
            <geom type="box" size="0.22 {wall_thick} {wall_h}" rgba="0.5 0.5 0.5 0.3"/>
        </body>"""
        xml_content = xml_content.replace("</worldbody>", walls + "\n  </worldbody>", 1)

    # Custom ball: 40 mm diameter, 150 g, with high grip friction
    if 'name="target_ball"' not in xml_content:
        custom_ball_xml = f"""
        <body name="target_ball" pos="{BALL_POS[0]} {BALL_POS[1]} {BALL_POS[2]}">
            <freejoint/>
            <geom name="ball_geom" type="sphere" size="{CUSTOM_BALL_RADIUS}"
                  rgba="1 0.3 0.3 0.9" mass="0.150" condim="4"
                  friction="2.0 1.0 0.1" solimp="0.95 0.99 0.001"
                  solref="0.015 1"/>
        </body>"""
        xml_content = xml_content.replace("</worldbody>", custom_ball_xml + "\n  </worldbody>")
    # (inject_soft_ball is skipped because target_ball already exists)
    xml_content = inject_wrist_camera(xml_content)
    xml_content = inject_force_sensor(xml_content)
    xml_content = inject_actuators(xml_content)
    # Must be AFTER inject_actuators — inserts into the existing <actuator>
    xml_content = inject_control_actuators(xml_content)

    # ── Boost PD gains for precise positioning under gravity ──────────
    xml_content = xml_content.replace('kp="90"', 'kp="350"')
    xml_content = xml_content.replace('kp="70"', 'kp="250"')
    xml_content = xml_content.replace('kp="45"', 'kp="180"')

    xml_dir = os.path.dirname(model_path)
    os.chdir(xml_dir)
    model = mujoco.MjModel.from_xml_string(xml_content)
    os.chdir(repo_root)

    data = mujoco.MjData(model)
    model.opt.timestep = 1.0 / SIM_HZ

    # Joint classification
    arm_joints: list[int] = []
    gripper_joints: list[int] = []
    for jid in range(model.njnt):
        jt = model.jnt_type[jid]
        if jt == mujoco.mjtJoint.mjJNT_FREE:
            continue
        if jt in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
            if "finger" in name.lower():
                gripper_joints.append(jid)
            else:
                arm_joints.append(jid)

    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Link7")
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Link8")
    if left_id < 0 or right_id < 0:
        raise RuntimeError("Could not find Link7/Link8 finger bodies.")

    joint_to_actuator: dict[int, int] = {}
    for act_id in range(model.nu):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id) or ""
        joint_name = act_name[:-4] if act_name.endswith("_act") else act_name
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jnt_id >= 0:
            joint_to_actuator[int(jnt_id)] = int(act_id)

    mujoco.mj_forward(model, data)

    # ── Initial setup ─────────────────────────────────────────────────
    gripper_limits = build_gripper_limits(model, data, gripper_joints)
    command_gripper(model, data, gripper_limits, joint_to_actuator, "open")
    for jid, act in joint_to_actuator.items():
        data.ctrl[act] = data.qpos[model.jnt_qposadr[jid]]
    mujoco.mj_forward(model, data)

    # ── Workspace ─────────────────────────────────────────────────────
    workspace_pts = compute_workspace(model, data, arm_joints, left_id, right_id,
                                      resolution=0.04)

    # ── Controllers ───────────────────────────────────────────────────
    arm_ctrl = SmoothArmController(model, data, arm_joints, joint_to_actuator)
    gripper_ctrl = SmoothGripperController(model, data, gripper_limits, joint_to_actuator)

    # ── Ball detector + interactive placement ─────────────────────────
    rgb_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, RGB_CAMERA_NAME)
    ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_ball")
    detector = BallDetector(model, rgb_cam_id, ball_body_id)

    # Find the ball's freejoint qpos address (for slider placement)
    ball_qpos_adr = -1
    for jid in range(model.njnt):
        if model.jnt_bodyid[jid] == ball_body_id:
            ball_qpos_adr = model.jnt_qposadr[jid]
            break

    # Find control actuator IDs (dummy sliders in the viewer's control panel)
    start_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "start_demo")
    ball_x_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ball_x")
    ball_y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ball_y")
    ball_z_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ball_z")
    # Initialise ball sliders to the default ball position
    _last_bx = float(BALL_POS[0])
    _last_by = float(BALL_POS[1])
    _last_bz = float(BALL_POS[2])
    if ball_x_act_id >= 0:
        data.ctrl[ball_x_act_id] = _last_bx
        data.ctrl[ball_y_act_id] = _last_by
        data.ctrl[ball_z_act_id] = _last_bz
    # Freejoint qvel  ≠  qpos address  (qpos=7, qvel=6 elements)
    ball_dof_adr = -1
    for jid in range(model.njnt):
        if model.jnt_bodyid[jid] == ball_body_id:
            ball_dof_adr = model.jnt_dofadr[jid]
            break

    # ── Force / torque ────────────────────────────────────────────────
    ft_sensor = ForceTorqueSensor(model)
    ft_display = FTDisplay(width=500, height=350, history_len=180)

    # ── Eye-in-hand camera ────────────────────────────────────────────
    rgb_window = None
    if rgb_cam_id >= 0:
        try:
            rgb_window = RGBCameraWindow(
                model, rgb_cam_id, width=480, height=360,
                render_every_n=CAMERA_EVERY_N,
                window_name="Eye-in-Hand RGB Camera")
            print("Eye-in-hand camera: active (480x360)")
        except Exception as exc:
            print(f"Camera init: {exc}")

    # ── Joint state panel ─────────────────────────────────────────────
    joint_panel = None
    if CV2_AVAILABLE:
        try:
            joint_panel = JointStatePanel()
            print("Joint-state panel: active")
        except Exception as exc:
            print(f"Joint panel init: {exc}")

    # ── State machine ─────────────────────────────────────────────────
    # -1 = idle      0 = scan   1 = home    2 = approach   3 = descend
    #  4 = close     5 = verify 6 = move    7 = release    8 = re-grasp
    #  9 = done
    phase = -1
    sub = 0
    finished = False
    scan_idx = 0
    scan_targets: list[np.ndarray] = []
    detected_ball_pos: np.ndarray | None = None
    dynamic_ik_plan: dict[str, IKResult] = {}
    regrasp_count = 0
    MAX_REGRASP = 5
    ball_z_before_lift = 0.0
    status_msg = "IDLE — use ball_x/y/z & start_demo sliders"

    def compute_dynamic_ik(ball_xyz: np.ndarray,
                           grasp_z_offs: float = -0.010) -> dict[str, IKResult]:
        """Compute IK plan for a detected ball position.

        grasp_z_offs: offset from ball centre for the grasp target.
        Negative = below centre = deeper grip.
        """
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = data.qpos[:]
        scratch.qvel[:] = 0.0
        for item in gripper_limits:
            scratch.qpos[model.jnt_qposadr[item["joint"]]] = item["open"]
        mujoco.mj_forward(model, scratch)

        place_above_box = BOX_POS + np.array([0.0, 0.0, 0.10])

        targets = {
            "approach": ball_xyz + np.array([0.0, 0.0, 0.15]),
            "grasp":    ball_xyz + np.array([0.0, 0.0, grasp_z_offs]),
            "lift":     ball_xyz + np.array([0.0, 0.0, 0.20]),
            "place":    place_above_box,
        }

        plan: dict[str, IKResult] = {}
        ok = True
        for name, target in targets.items():
            result = solve_gripper_center_ik(
                model, scratch, target, left_id, right_id, arm_joints,
            )
            plan[name] = result
            status = "reachable" if result.success else "UNREACHABLE"
            print(f"  IK {name:8s}: {status}, err={result.error_norm:.4f}m, "
                  f"target={np.round(target, 3)}")
            if not result.success:
                ok = False
                break
            set_joint_positions(model, scratch, arm_joints, result.angles)
            mujoco.mj_forward(model, scratch)

        if not ok:
            actual_xyz = data.xpos[ball_body_id].copy()
            print(f"    Retrying IK at actual ball pos {np.round(actual_xyz, 3)} ...")
            targets2 = {
                "approach": actual_xyz + np.array([0.0, 0.0, 0.15]),
                "grasp":    actual_xyz + np.array([0.0, 0.0, grasp_z_offs]),
                "lift":     actual_xyz + np.array([0.0, 0.0, 0.20]),
                "place":    BOX_POS + np.array([0.0, 0.0, 0.10]),
            }
            scratch2 = mujoco.MjData(model)
            scratch2.qpos[:] = data.qpos[:]
            scratch2.qvel[:] = 0.0
            for item in gripper_limits:
                scratch2.qpos[model.jnt_qposadr[item["joint"]]] = item["open"]
            mujoco.mj_forward(model, scratch2)
            for name, target in targets2.items():
                result = solve_gripper_center_ik(
                    model, scratch2, target, left_id, right_id, arm_joints,
                )
                plan[name] = result
                if result.success:
                    set_joint_positions(model, scratch2, arm_joints, result.angles)
                    mujoco.mj_forward(model, scratch2)
            return plan
        return plan

    def demo_tick() -> None:
        nonlocal phase, sub, finished, scan_idx, scan_targets, regrasp_count
        nonlocal detected_ball_pos, dynamic_ik_plan, status_msg, ball_z_before_lift

        if finished:
            return

        # ── Phase -1:  idle — wait for SPACE ──────────────────────────
        if phase == -1:
            status_msg = "IDLE — use ball_x/y/z & start_demo sliders"
            return  # do nothing, wait for keyboard

        # ── Phase 0:  scanning ─────────────────────────────────────────
        elif phase == 0:
            if sub == 0:
                print(">>> Phase 0 : Camera scanning for ball ...")
                scan_targets = generate_scan_targets(_last_bz)
                scan_idx = 0
                result = solve_gripper_center_ik(
                    model, data, scan_targets[0], left_id, right_id, arm_joints,
                )
                if result.success:
                    arm_ctrl.set_target(result.angles, speed=SPEED_SCAN)
                    status_msg = f"Scanning 1/{len(scan_targets)} ..."
                else:
                    scan_idx = 1
                sub = 1

            elif sub == 1:
                if not arm_ctrl.done:
                    return

                # ── Capture and detect ─────────────────────────────────
                if rgb_window is not None:
                    renderer = rgb_window.renderer
                    renderer.update_scene(data, camera=rgb_cam_id)
                    rgb_img = renderer.render()
                    
                    # ---- [FIX APPLIED HERE] ----
                    center_uv, radius = detector.detect(rgb_img, data)
                    if center_uv is not None:
                        cu, cv = center_uv
                        est_xyz = detector.estimate_3d(
                            (cu, cv), data,
                            rgb_window.width, rgb_window.height,
                            ball_z=_last_bz,
                        )
                        # Sanity check: reject estimates far from workspace
                        err_vs_actual = np.linalg.norm(
                            est_xyz[:2] - data.xpos[ball_body_id][:2])
                        in_workspace = (0.05 < est_xyz[0] < 0.50 and
                                        abs(est_xyz[1]) < 0.35)
                        if not in_workspace or err_vs_actual > 0.30:
                            print(f"    False positive rejected — est {np.round(est_xyz, 3)} "
                                  f"outside workspace or too far from ball")
                            # Fall through to next scan pose
                        else:
                            detector._last_estimated_xyz = est_xyz
                            detected_ball_pos = est_xyz
                            actual = data.xpos[ball_body_id]
                            print(f"    Ball DETECTED at ({cu},{cv}) r={radius}px")
                            print(f"    Estimated: {np.round(est_xyz, 3)}")
                            print(f"    Actual:    {np.round(actual, 3)}")
                            print(f"    Error: {np.linalg.norm(est_xyz - actual):.4f} m")
                            dynamic_ik_plan = compute_dynamic_ik(detected_ball_pos)
                            status_msg = (f"Ball found! "
                                          f"X={est_xyz[0]:.3f} Y={est_xyz[1]:.3f} Z={est_xyz[2]:.3f}")
                            annotated = draw_detection_overlay(rgb_img, True, est_xyz)
                            cv2.imshow("Eye-in-Hand RGB Camera",
                                       cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                            if CV2_AVAILABLE:
                                cv2.waitKey(1)
                            phase = 1; sub = 0
                            return
                    else:
                        # Show X on camera
                        annotated = draw_detection_overlay(rgb_img, False)
                        cv2.imshow("Eye-in-Hand RGB Camera",
                                   cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                        if CV2_AVAILABLE:
                            cv2.waitKey(1)
                        print(f"    Scan {scan_idx+1}/{len(scan_targets)}: not in view")

                # ── Next scan pose ─────────────────────────────────────
                scan_idx += 1
                while scan_idx < len(scan_targets):
                    target = scan_targets[scan_idx]
                    result = solve_gripper_center_ik(
                        model, data, target, left_id, right_id, arm_joints,
                    )
                    if result.success:
                        arm_ctrl.set_target(result.angles, speed=SPEED_SCAN)
                        status_msg = f"Scanning {scan_idx+1}/{len(scan_targets)} ..."
                        return
                    scan_idx += 1

                # ── Fallback: use actual ball position from sim state ──
                print("    Ball not found visually — using sim ground-truth")
                detected_ball_pos = data.xpos[ball_body_id].copy()
                print(f"    Actual ball at: {np.round(detected_ball_pos, 3)}")
                dynamic_ik_plan = compute_dynamic_ik(detected_ball_pos)
                status_msg = "Using ground-truth plan ..."
                phase = 1; sub = 0

        # ── Phase 1:  home ────────────────────────────────────────────
        elif phase == 1:
            if sub == 0:
                print(">>> Phase 1 : Home")
                gripper_ctrl.open(duration_frames=25)
                arm_ctrl.set_target(np.zeros(len(arm_joints)), speed=SPEED_NORMAL)
                sub = 1
                status_msg = "Moving to home ..."
            if arm_ctrl.done and gripper_ctrl.done:
                phase = 2; sub = 0

        # ── Phase 2:  approach ─────────────────────────────────────────
        elif phase == 2:
            if sub == 0:
                print(">>> Phase 2 : Approach")
                arm_ctrl.set_target(dynamic_ik_plan["approach"].angles, speed=SPEED_NORMAL)
                sub = 1
                status_msg = "Approaching ball ..."
            if arm_ctrl.done:
                phase = 3; sub = 0

        # ── Phase 3:  descend ──────────────────────────────────────────
        elif phase == 3:
            if sub == 0:
                # Start at ball centre, go slightly lower on retries
                offsets = [0.0, -0.005, -0.010, -0.015, -0.020]
                z_offs = offsets[min(regrasp_count, len(offsets) - 1)]
                print(f">>> Phase 3 : Descend  (grasp Z offset = {z_offs*1000:.0f} mm)")
                # Recompute IK with deeper grasp on each retry
                dynamic_ik_plan = compute_dynamic_ik(
                    detected_ball_pos if detected_ball_pos is not None
                    else data.xpos[ball_body_id].copy(),
                    grasp_z_offs=z_offs,
                )
                arm_ctrl.set_target(dynamic_ik_plan["grasp"].angles, speed=SPEED_SLOW)
                sub = 1
                status_msg = f"Descending (retry {regrasp_count}) ..."
            if arm_ctrl.done:
                ball_z_before_lift = float(data.xpos[ball_body_id][2])
                phase = 4; sub = 0

        # ── Phase 4:  close gripper ────────────────────────────────────
        elif phase == 4:
            if sub == 0:
                gc = (data.xpos[left_id] + data.xpos[right_id]) * 0.5
                ball = data.xpos[ball_body_id]
                err = np.linalg.norm(ball - gc)
                print(f"    Gripper→ball: {err*100:.1f}cm  "
                      f"(grip Z={gc[2]:.3f}, ball Z={ball[2]:.3f})")
                gripper_ctrl.close(duration_frames=100)
                sub = 1
                status_msg = "Closing gripper ..."
            if gripper_ctrl.done:
                phase = 5; sub = 0

        # ── Phase 5:  lift + verify ────────────────────────────────────
        elif phase == 5:
            if sub == 0:
                print(">>> Phase 5 : Lift & verify")
                arm_ctrl.set_target(dynamic_ik_plan["lift"].angles, speed=SPEED_NORMAL)
                sub = 1
                status_msg = "Lifting ..."
            if arm_ctrl.done:
                # ── Verify: did the ball actually rise? ────────────────
                ball_z_now = float(data.xpos[ball_body_id][2])
                lifted = ball_z_now - ball_z_before_lift
                print(f"    Ball Z: before={ball_z_before_lift:.3f}  "
                      f"after={ball_z_now:.3f}  Δ={lifted*100:.1f}cm")
                if lifted > 0.015:
                    # Ball was lifted >1.5 cm → grasp succeeded
                    print(f"    Grasp SUCCESS after {regrasp_count} retries")
                    phase = 6; sub = 0; regrasp_count = 0
                else:
                    # Ball still on ground → grasp failed
                    regrasp_count += 1
                    print(f"    Grasp FAILED (attempt {regrasp_count}/{MAX_REGRASP})")
                    if regrasp_count >= MAX_REGRASP:
                        print("    Max retries exhausted — proceeding anyway")
                        phase = 6; sub = 0; regrasp_count = 0
                    else:
                        # Re-grasp: release, wait for ball to settle, retry
                        phase = 8; sub = 0

        # ── Phase 6:  move to box ──────────────────────────────────────
        elif phase == 6:
            if sub == 0:
                print(">>> Phase 6 : Move to box")
                arm_ctrl.set_target(dynamic_ik_plan["place"].angles, speed=SPEED_NORMAL)
                sub = 1
                status_msg = "Moving to box ..."
            if arm_ctrl.done:
                phase = 7; sub = 0

        # ── Phase 7:  release ──────────────────────────────────────────
        elif phase == 7:
            if sub == 0:
                print(">>> Phase 7 : Release into box")
                gripper_ctrl.open(duration_frames=25)
                sub = 1
                status_msg = "Releasing ..."
            if gripper_ctrl.done:
                phase = 9; sub = 0

        # ── Phase 8:  re-grasp — release, settle, retry ────────────────
        elif phase == 8:
            if sub == 0:
                print(f">>> Phase 8 : Re-grasp ({regrasp_count}/{MAX_REGRASP})")
                gripper_ctrl.open(duration_frames=20)
                sub = 1
            elif sub == 1:
                if not gripper_ctrl.done:
                    return
                # Wait for ball to settle (velocity below threshold)
                bv = float(np.linalg.norm(data.qvel[ball_dof_adr:ball_dof_adr+3]))
                if bv < 0.02:
                    print(f"    Ball settled (|v|={bv:.3f} m/s)")
                    # Update ball position for next attempt
                    detected_ball_pos = data.xpos[ball_body_id].copy()
                    phase = 3; sub = 0  # go back to descend with deeper offset
                else:
                    status_msg = f"Waiting for ball to settle (|v|={bv:.3f}) ..."

        # ── Phase 9:  done ─────────────────────────────────────────────
        elif phase == 9:
            print(">>> Demo complete — manual control restored.")
            status_msg = "DONE — manual control"
            finished = True

    # ── Main loop ─────────────────────────────────────────────────────
    print("\nLaunching viewer + windows ...")
    print("  Sliders appear in the MuJoCo viewer Control Panel →")
    print()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.distance = 1.2
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.lookat[:] = [0.26, 0.0, 0.16]

        workspace_drawn = False
        frame = 0
        last_ft_reading = ft_sensor.read(data)

        # Warmup — settle physics
        for _ in range(30):
            for _ in range(PHYSICS_SUBSTEPS):
                mujoco.mj_step(model, data)
            frame += 1

        while viewer.is_running():
            frame_start = time.perf_counter()

            # ── 1. Input: slider-based ball placement + START ───────────
            if phase == -1 and ball_x_act_id >= 0:
                bx = data.ctrl[ball_x_act_id]
                by = data.ctrl[ball_y_act_id]
                bz = data.ctrl[ball_z_act_id]
                # Only move the ball when a slider actually changed
                if (abs(bx - _last_bx) > 0.0005 or
                    abs(by - _last_by) > 0.0005 or
                    abs(bz - _last_bz) > 0.0005):
                    data.qpos[ball_qpos_adr]     = bx
                    data.qpos[ball_qpos_adr + 1] = by
                    data.qpos[ball_qpos_adr + 2] = bz
                    # Zero velocity so the ball stays at the new spot
                    data.qvel[ball_dof_adr]     = 0.0
                    data.qvel[ball_dof_adr + 1] = 0.0
                    data.qvel[ball_dof_adr + 2] = 0.0
                    _last_bx, _last_by, _last_bz = bx, by, bz

            # start_demo slider → trigger
            if phase == -1 and start_act_id >= 0 and data.ctrl[start_act_id] > 0.5:
                print("\n*** START — beginning autonomous sequence ***\n")
                # Freeze ball at its current slider position with zero velocity
                if ball_qpos_adr >= 0:
                    data.qpos[ball_qpos_adr]     = _last_bx
                    data.qpos[ball_qpos_adr + 1] = _last_by
                    data.qpos[ball_qpos_adr + 2] = max(_last_bz, CUSTOM_BALL_RADIUS + 0.005)
                    for i in range(6):
                        data.qvel[ball_dof_adr + i] = 0.0
                phase = 0; sub = 0
                status_msg = "Starting ball search ..."
                # Reset trigger slider
                data.ctrl[start_act_id] = 0.0

            # OpenCV event pump
            if CV2_AVAILABLE:
                cv2.waitKey(1)

            # ── 2. State machine ────────────────────────────────────────
            try:
                demo_tick()
            except Exception as exc:
                print(f"ERROR in demo_tick: {exc}")
                import traceback; traceback.print_exc()
                finished = True
                status_msg = f"ERROR: {exc}"

            # ── 3. Control (skip during idle → manual sliders work) ────
            try:
                if not finished and phase != -1:
                    arm_ctrl.step()
                    gripper_ctrl.step()
            except Exception as exc:
                print(f"ERROR in controller step: {exc}")
                import traceback; traceback.print_exc()
                finished = True
                status_msg = f"ERROR: {exc}"

            # ── 4. Physics substeps ─────────────────────────────────────
            for _ in range(PHYSICS_SUBSTEPS):
                mujoco.mj_step(model, data)

            # ── 4b. Clamp ball inside workspace ─────────────────────────
            if ball_qpos_adr >= 0:
                bx, by, bz = (data.qpos[ball_qpos_adr],
                              data.qpos[ball_qpos_adr + 1],
                              data.qpos[ball_qpos_adr + 2])
                clamped = False
                if bx < 0.10:  bx = 0.10; clamped = True
                if bx > 0.44:  bx = 0.44; clamped = True
                if by < -0.28: by = -0.28; clamped = True
                if by > 0.28:  by = 0.28; clamped = True
                if bz < CUSTOM_BALL_RADIUS + 0.003: bz = CUSTOM_BALL_RADIUS + 0.003; clamped = True
                if bz > 0.30:  bz = 0.30; clamped = True
                if clamped:
                    data.qpos[ball_qpos_adr] = bx
                    data.qpos[ball_qpos_adr + 1] = by
                    data.qpos[ball_qpos_adr + 2] = bz
                    data.qvel[ball_dof_adr] = 0.0
                    data.qvel[ball_dof_adr + 1] = 0.0
                    data.qvel[ball_dof_adr + 2] = 0.0

            # ── 5. Workspace spheres (once) ────────────────────────────
            if not workspace_drawn and workspace_pts:
                try:
                    render_workspace_spheres(viewer, workspace_pts)
                    workspace_drawn = True
                except Exception:
                    pass

            # ── 6. Displays (throttled) ────────────────────────────────
            if frame % FT_EVERY_N == 0:
                last_ft_reading = ft_sensor.read(data)
                ft_display.update(last_ft_reading)
                ft_display.show()

            if joint_panel and frame % JOINT_EVERY_N == 0:
                idle_hint = "" if phase != -1 else " [slide start_demo to 1]"
                joint_panel.update(model, data, arm_joints, gripper_joints,
                                   status_text=status_msg + idle_hint)
                joint_panel.show()

            if rgb_window and rgb_window.should_update(frame):
                force_mag = float(np.linalg.norm(last_ft_reading.force))
                hint = " | [slide start_demo to 1]" if phase == -1 else ""
                rgb_window.update(data, overlay_text=f"|F|={force_mag:.2f}N{hint}")

            # ── 7. Viewer sync ─────────────────────────────────────────
            try:
                viewer.sync()
            except Exception:
                pass

            frame += 1

            # ── 8. Frame pacing ────────────────────────────────────────
            elapsed = time.perf_counter() - frame_start
            if elapsed < FRAME_DT:
                remaining = FRAME_DT - elapsed
                if remaining > 0.003:
                    time.sleep(remaining - 0.0015)
                while time.perf_counter() - frame_start < FRAME_DT:
                    pass

    ft_display.close()
    if rgb_window:
        rgb_window.close()
    if joint_panel:
        joint_panel.close()
    print("Done.")


if __name__ == "__main__":
    main()