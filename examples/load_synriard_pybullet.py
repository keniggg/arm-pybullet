"""
Quick loader for models in synriard/urdf using PyBullet GUI.

Usage examples:
    python examples/load_synriard_pybullet.py                 # interactive selection
    python examples/load_synriard_pybullet.py --model Alicia_D_v5_6_gripper_50mm.urdf
    python examples/load_synriard_pybullet.py --model "Alicia_D_v5_6/Alicia_D_v5_6_gripper_50mm.urdf"

Notes:
- This script expects to be run from the repository root or anywhere; it resolves the repo root
  relative to this file and scans `synriard/urdf` recursively for .urdf files.
- PyBullet is not listed in `requirements.txt`; install it if needed: `pip install pybullet`.
"""

import os
import argparse
import glob
import time
import sys

try:
    import pybullet as p
    import pybullet_data
except Exception as e:
    print("PyBullet not available. Install with: pip install pybullet")
    raise


def discover_urdfs(urdf_root):
    pattern = os.path.join(urdf_root, "**", "*.urdf")
    files = glob.glob(pattern, recursive=True)
    files = sorted(files)
    return files


def choose_model(choices, preferred=None):
    if not choices:
        return None
    if preferred:
        # try exact match
        for c in choices:
            if os.path.basename(c) == preferred or c.endswith(preferred):
                return c
    # if only one choice, return it
    if len(choices) == 1:
        return choices[0]
    # otherwise interactive selection
    print("Available URDF models:")
    for i, c in enumerate(choices):
        print(f"  [{i}] {os.path.relpath(c)}")
    try:
        idx = int(input("Select model index: "))
        return choices[idx]
    except Exception:
        print("Invalid selection")
        return None


def main():
    parser = argparse.ArgumentParser(description="Load a synriard URDF in PyBullet GUI")
    parser.add_argument("--model", "-m", help="model filename or relative path under synriard/urdf")
    parser.add_argument("--nogui", action="store_true", help="use DIRECT mode (no GUI)")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    urdf_root = os.path.join(repo_root, "synriard", "urdf")

    if not os.path.isdir(urdf_root):
        print(f"Could not find urdf folder at: {urdf_root}")
        sys.exit(1)

    urdfs = discover_urdfs(urdf_root)
    if not urdfs:
        print(f"No URDF files found under: {urdf_root}")
        sys.exit(1)

    model_path = choose_model(urdfs, preferred=args.model)
    if not model_path:
        print("No model selected. Exiting.")
        sys.exit(1)

    print(f"Using model: {model_path}")

    connection_mode = p.DIRECT if args.nogui else p.GUI
    p.connect(connection_mode)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Add the URDF directory and relevant repo search paths so meshes/textures can be found
    urdf_dir = os.path.dirname(os.path.abspath(model_path))
    p.setAdditionalSearchPath(urdf_dir)
    # add repo root, synriard root and synriard/meshes (and their subdirs) to help resolve ../../meshes/... references
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    synriard_root = os.path.join(repo_root, "synriard")
    synriard_meshes = os.path.join(synriard_root, "meshes")
    p.setAdditionalSearchPath(repo_root)
    p.setAdditionalSearchPath(synriard_root)
    if os.path.isdir(synriard_meshes):
        p.setAdditionalSearchPath(synriard_meshes)
        # also add all subdirectories under meshes
        for root, dirs, files in os.walk(synriard_meshes):
            p.setAdditionalSearchPath(root)

    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)

    # load a simple plane for reference
    try:
        plane = p.loadURDF("plane.urdf")
    except Exception:
        plane = None

    flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_MAINTAIN_LINK_ORDER

    # some URDFs expect to be fixed to ground
    # Try several fallbacks to handle Windows paths and non-ASCII characters.
    from pathlib import Path
    import tempfile
    import shutil

    model_abs = os.path.abspath(model_path)
    if not os.path.exists(model_abs):
        print(f"Model file does not exist: {model_abs}")
        sys.exit(1)

    robot_id = None
    last_err = None

    # Attempt 1: load using absolute path
    try:
        robot_id = p.loadURDF(model_abs, basePosition=[0, 0, 0], useFixedBase=True, flags=flags)
    except Exception as e:
        last_err = e

    # Attempt 2: load by basename (search paths include the URDF dir)
    if robot_id is None:
        try:
            basename = os.path.basename(model_abs)
            robot_id = p.loadURDF(basename, basePosition=[0, 0, 0], useFixedBase=True, flags=flags)
        except Exception as e:
            last_err = e

    # Attempt 3: convert to POSIX-style path (forward slashes)
    if robot_id is None:
        try:
            posix_path = Path(model_abs).as_posix()
            robot_id = p.loadURDF(posix_path, basePosition=[0, 0, 0], useFixedBase=True, flags=flags)
        except Exception as e:
            last_err = e

    # Attempt 4: copy the entire `synriard` tree to a temporary ASCII-only
    # directory and load the URDF from there. This preserves relative
    # references (e.g. ../../meshes/...) so PyBullet can find meshes even
    # when the original repo path contains non-ASCII characters.
    if robot_id is None:
        try:
            # original synriard root in repo
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            synriard_root = os.path.join(repo_root, "synriard")
            if os.path.isdir(synriard_root):
                # create a fresh temp root and copy the entire synriard tree there
                temp_root = tempfile.mkdtemp(prefix="synriard_")
                temp_synriard = os.path.join(temp_root, "synriard")
                try:
                    shutil.copytree(synriard_root, temp_synriard, dirs_exist_ok=True)
                except Exception:
                    # fallback: attempt a best-effort copy of files
                    for root, dirs, files in os.walk(synriard_root):
                        rel = os.path.relpath(root, synriard_root)
                        dest_root = os.path.join(temp_synriard, rel)
                        os.makedirs(dest_root, exist_ok=True)
                        for fn in files:
                            srcf = os.path.join(root, fn)
                            dstf = os.path.join(dest_root, fn)
                            try:
                                shutil.copy2(srcf, dstf)
                            except Exception:
                                pass

                # compute the path to the copied URDF inside the temp synriard tree
                rel_to_syn = os.path.relpath(model_abs, synriard_root)
                temp_model = os.path.join(temp_synriard, rel_to_syn)

                # add temp synriard root and its subdirs to pybullet search paths
                p.setAdditionalSearchPath(temp_synriard)
                for root, dirs, files in os.walk(temp_synriard):
                    p.setAdditionalSearchPath(root)

                # try loading from the temp copy
                robot_id = p.loadURDF(temp_model, basePosition=[0, 0, 0], useFixedBase=True, flags=flags)
            else:
                # no synriard tree found; leave last_err as-is
                pass
        except Exception as e:
            last_err = e

    if robot_id is None:
        print("Failed to load URDF after multiple attempts.")
        print("Last error:", last_err)
        sys.exit(1)

    print(f"Loaded robot id={robot_id}")

    # print joints summary
    n = p.getNumJoints(robot_id)
    print(f"Joints: {n}")
    for i in range(n):
        info = p.getJointInfo(robot_id, i)
        name = info[1].decode("utf-8")
        jtype = info[2]
        lower = info[8]
        upper = info[9]
        print(f"  [{i}] {name} type={jtype} limits=({lower}, {upper})")

    print("Close the GUI window or press Ctrl+C to exit.")
    try:
        while p.isConnected():
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()
