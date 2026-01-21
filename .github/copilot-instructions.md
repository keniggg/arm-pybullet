# Synria Robot Descriptions - AI Agent Instructions

## Project Overview
This is a Python package providing URDF and MJCF robot description files for Synria Robotics platforms (Alicia-D, Alicia-M, Bessica-D, Bessica-M). The package enables simulation in environments like PyBullet, MuJoCo, ROS/Gazebo.

## Architecture
- **synriard/**: Main package with robot models
  - **urdf/**: URDF files for ROS/Gazebo/PyBullet
  - **mjcf/**: MJCF files for MuJoCo
  - **meshes/**: 3D mesh files referenced by models
- **examples/**: Usage examples, especially PyBullet integration, force feedback, and grasping demos
- **utils/**: Utilities like convex decomposition for collision shapes
- **test/**: Path validation tests

## Key Patterns

### Model Organization
Models follow `{name}_{version}_{variant}.{ext}` naming:
- `name`: Robot name (Alicia_D, Alicia_M, Bessica_D, Bessica_M)
- `version`: Version (v5_5, v1_0, v1_1)
- `variant`: For grippers: `gripper_{size}mm`, for Bessica: `covered`, `skeleton`

Directory structure: `synriard/{format}/{name}_{version}/{files}`

### API Usage
```python
from synriard import get_model_path, list_available_models

# Get specific model
urdf_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm")
mjcf_path = get_model_path("Alicia_D", version="v5_6", variant="gripper_100mm", model_format="mjcf")

# List all models
print(list_available_models(model_format="urdf", show_path=True))
```

### Auto-Generated __init__.py Files
The `__init__.py` files are auto-generated using `SimpleNamespace` objects:
```python
# Example: synriard/urdf/Alicia_D_v5_6/__init__.py
Alicia_D_v5_6_gripper_50mm = SimpleNamespace()
Alicia_D_v5_6_gripper_50mm.urdf = os.path.join(_MODULE_PATH, "Alicia_D_v5_6_gripper_50mm.urdf")
```

### Adding New Models
1. Place model files in appropriate `synriard/{urdf|mjcf}/{name}_{version}/` directory
2. Run `python auto_generate_init.py` to update `__init__.py` files
3. Use `--format urdf|mjcf|all` to specify formats

### PyBullet Integration
When loading URDFs in PyBullet, set search paths for mesh resolution:
```python
import pybullet as p
p.setAdditionalSearchPath(repo_root)
p.setAdditionalSearchPath(os.path.join(repo_root, "synriard"))
p.setAdditionalSearchPath(os.path.join(repo_root, "synriard", "meshes"))
# Add all subdirs under meshes
for root, dirs, files in os.walk(meshes_dir):
    p.setAdditionalSearchPath(root)
```

Use flags: `p.URDF_USE_INERTIA_FROM_FILE | p.URDF_MAINTAIN_LINK_ORDER`

For Windows/non-ASCII paths, examples include fallback loading strategies including temporary directory copying.

### Convex Decomposition
Use `utils/convex_decompositon.py` with coacd and trimesh for collision shape optimization:
```bash
python utils/convex_decompositon.py --input mesh.stl --output decomposed.obj --threshold 0.05
```

### Testing
Run path validation: `python test/test_paths.py`

## Development Workflow
- Install: `pip install -e .`
- Run examples: `python examples/load_synriard_pybullet.py --model Alicia_D_v5_6_gripper_100mm.urdf`
- Test paths: `python test/test_paths.py`
- Virtual env: Use `.venv` (already configured)

## Dependencies
- Core: coacd, trimesh (for convex decomposition)
- Simulation: pybullet (not in requirements.txt, install separately)

## Conventions
- All paths resolved relative to package root
- UTF-8 encoding for cross-platform compatibility
- GPL-3.0 license
- Version modules auto-generated, do not edit manually
- Use SimpleNamespace objects for model file access (e.g., `Alicia_D_v5_6_gripper_100mm.urdf`)
- MJCF files use `.xml` extension but accessed via `.xml` attribute</content>
<parameter name="filePath">d:/成信研究生院/机械臂力反馈项目/Synria-Robot-Descriptions-main/.github/copilot-instructions.md