import os
from types import SimpleNamespace

# Get the absolute path to THIS directory
_MODULE_PATH = os.path.dirname(os.path.abspath(__file__))

Alicia_D_v5_5_gripper_100mm = SimpleNamespace()
Alicia_D_v5_5_gripper_100mm.xml = os.path.join(_MODULE_PATH, "Alicia_D_v5_5_gripper_100mm.xml")
