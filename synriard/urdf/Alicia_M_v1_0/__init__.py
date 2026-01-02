import os
from types import SimpleNamespace

# Get the absolute path to THIS directory
_MODULE_PATH = os.path.dirname(os.path.abspath(__file__))

Alicia_M_v1_0_gripper_100mm = SimpleNamespace()
Alicia_M_v1_0_gripper_100mm.urdf = os.path.join(_MODULE_PATH, "Alicia_M_v1_0_gripper_100mm.urdf")
