# This line makes the sub-folders available as attributes of this module
from . import Alicia_D_v5_5
from . import Alicia_D_v5_6
from . import Alicia_M_v1_0
from . import Bessica_D_v1_0

# Note: Do not import specific URDF namespace objects here (e.g., Bessica_D_v1_0_Covered),
# as this may trigger "partially initialized module" circular import errors in some environments.
# Correct usage: Access through subpackages, e.g.,
#   from synriard import urdf
#   urdf_path = urdf.Bessica_D_v1_0.Bessica_D_v1_0_Covered.urdf

__all__ = [
    "Alicia_D_v5_5",
    "Alicia_D_v5_6",
    "Alicia_M_v1_0",
    "Bessica_D_v1_0",
]
