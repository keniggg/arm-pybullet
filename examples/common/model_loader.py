"""Model loading and XML injection for MuJoCo demos."""
from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco

from synriard import get_model_path

SIM_HZ = 240.0

BALL_POS = np.array([0.30, 0.00, 0.05], dtype=np.float64)
BALL_RADIUS = 0.025

RGB_CAMERA_NAME = "wrist_rgb"


def _first_tag_end(xml_content: str, tag_name: str) -> int:
    start = xml_content.find(f"<{tag_name}")
    if start < 0:
        return -1
    return xml_content.find(">", start)


def inject_options(xml_content: str) -> str:
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
    return xml_content


def inject_overview_camera(xml_content: str) -> str:
    if 'name="overview"' not in xml_content:
        overview_xml = (
            '    <camera name="overview" pos="0.85 -0.75 0.55" '
            'xyaxes="0.66 0.75 0 -0.25 0.22 0.94" fovy="45"/>\n'
        )
        worldbody_end = xml_content.find("<worldbody>")
        if worldbody_end >= 0:
            insert_at = xml_content.find(">", worldbody_end) + 1
            xml_content = xml_content[:insert_at] + "\n" + overview_xml + xml_content[insert_at:]
    return xml_content


def inject_soft_ball(xml_content: str, pos: np.ndarray = None, radius: float = None) -> str:
    if pos is None:
        pos = BALL_POS
    if radius is None:
        radius = BALL_RADIUS
    if 'name="target_ball"' not in xml_content:
        ball_xml = f"""
        <body name="target_ball" pos="{pos[0]} {pos[1]} {pos[2]}">
            <freejoint/>
            <geom name="ball_geom" type="sphere" size="{radius}"
                  rgba="1 0.3 0.3 0.9" mass="0.025" condim="4"
                  friction="1.5 0.08 0.02" solimp="0.92 0.98 0.001 0.5 2"
                  solref="0.005 0.8"/>
        </body>"""
        xml_content = xml_content.replace("</worldbody>", ball_xml + "\n  </worldbody>")
    return xml_content

def inject_wrist_camera(xml_content: str) -> str:
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
    return xml_content


def inject_force_sensor(xml_content: str) -> str:
    if 'name="wrist_ft_site"' in xml_content:
        return xml_content

    marker = 'rgba="0 1 0 1" />'
    marker_pos = xml_content.find(marker)
    if marker_pos < 0:
        marker = 'rgba="0 1 0 1"/>'
        marker_pos = xml_content.find(marker)
    if marker_pos >= 0:
        insert_at = marker_pos + len(marker)
        site_xml = '\n                  <site name="wrist_ft_site" pos="0 0 0.131" size="0.005" rgba="1 1 0 0.3"/>'
        xml_content = xml_content[:insert_at] + site_xml + xml_content[insert_at:]

    sensor_xml = """
  <sensor>
    <force name="wrist_force" site="wrist_ft_site"/>
    <torque name="wrist_torque" site="wrist_ft_site"/>
  </sensor>"""
    xml_content = xml_content.replace("</mujoco>", sensor_xml + "\n</mujoco>")
    return xml_content


def inject_actuators(xml_content: str) -> str:
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

def load_and_inject(
    include_ball: bool = True,
    include_sensors: bool = True,
    include_camera: bool = True,
    ball_pos: np.ndarray = None,
) -> tuple:
    """Load the Alicia_D MJCF and inject runtime elements.

    Returns (model, data, arm_joints, gripper_joints, left_body_id, right_body_id, joint_to_actuator).
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    os.chdir(repo_root)

    model_path = get_model_path(
        "Alicia_D", version="v5_6", variant="gripper_50mm", model_format="mjcf"
    )
    with open(model_path, "r", encoding="utf-8") as f:
        xml_content = f.read()

    xml_content = inject_options(xml_content)
    xml_content = inject_overview_camera(xml_content)
    if include_ball:
        xml_content = inject_soft_ball(xml_content, pos=ball_pos)
    if include_camera:
        xml_content = inject_wrist_camera(xml_content)
    if include_sensors:
        xml_content = inject_force_sensor(xml_content)
    xml_content = inject_actuators(xml_content)

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

    left_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Link7")
    right_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Link8")
    if left_body_id < 0 or right_body_id < 0:
        raise RuntimeError("Could not find Link7/Link8 finger bodies in the model.")

    joint_to_actuator: dict[int, int] = {}
    for act_id in range(model.nu):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id) or ""
        joint_name = act_name[:-4] if act_name.endswith("_act") else act_name
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            joint_to_actuator[int(joint_id)] = int(act_id)

    mujoco.mj_forward(model, data)

    return model, data, arm_joints, gripper_joints, left_body_id, right_body_id, joint_to_actuator
