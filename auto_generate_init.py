#!/usr/bin/env python3
"""
Auto-generate __init__.py files for robot models in synriard.

Usage:
    python auto_generate_init.py [--format mjcf|urdf|all]

This script will:
1. Scan mjcf/ and urdf/ directories
2. Auto-generate __init__.py files for each robot directory
3. Update parent __init__.py files to register all robots

After adding new robot models (meshes, mjcf, urdf files), run this script
to automatically generate the necessary __init__.py files.
"""

import os
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


def find_model_files(directory: Path, extensions: List[str]) -> List[Path]:
    """Find all model files with given extensions in directory."""
    model_files = []
    for ext in extensions:
        model_files.extend(directory.glob(f"*.{ext}"))
    return sorted(model_files)


def extract_object_name(file_path: Path, robot_dir_name: str) -> str:
    """
    Extract object name from file path for Synria naming convention.
    
    For example:
    - Alicia_D_v5_5/Alicia_D_v5_5_gripper_100mm.xml -> Alicia_D_v5_5_gripper_100mm
    - Bessica_D_v1_0/Bessica_D_v1_0_covered.xml -> Bessica_D_v1_0_covered
    """
    stem = file_path.stem  # filename without extension
    
    # For Synria, object name is typically the full filename stem
    # which includes robot name, version, and variant
    return stem


def generate_init_py(robot_dir: Path, format_type: str) -> str:
    """
    Generate __init__.py content for a robot directory.
    
    :param robot_dir: Path to robot directory (e.g., synriard/mjcf/Alicia_D_v5_5)
    :param format_type: 'mjcf' or 'urdf'
    :return: Generated __init__.py content
    """
    robot_name = robot_dir.name
    
    # Find model files
    if format_type == "mjcf":
        extensions = ["xml"]
        attr_name = "xml"
    else:  # urdf
        extensions = ["urdf"]
        attr_name = "urdf"
    
    model_files = find_model_files(robot_dir, extensions)
    
    if not model_files:
        return None
    
    lines = [
        "import os",
        "from types import SimpleNamespace",
        "",
        "# Get the absolute path to THIS directory",
        "_MODULE_PATH = os.path.dirname(os.path.abspath(__file__))",
        "",
    ]
    
    # Generate SimpleNamespace objects for each model file
    for model_file in model_files:
        object_name = extract_object_name(model_file, robot_name)
        file_name = model_file.name
        
        lines.append(f"{object_name} = SimpleNamespace()")
        lines.append(
            f'{object_name}.{attr_name} = os.path.join(_MODULE_PATH, "{file_name}")'
        )
        lines.append("")
    
    return "\n".join(lines)


def update_parent_init(parent_dir: Path, format_type: str) -> None:
    """
    Update parent __init__.py to include all robot directories.
    
    :param parent_dir: Path to parent directory (e.g., synriard/mjcf)
    :param format_type: 'mjcf' or 'urdf'
    """
    init_file = parent_dir / "__init__.py"
    
    # Find all robot directories (subdirectories with __init__.py or model files)
    robot_dirs = []
    for item in sorted(parent_dir.iterdir()):
        if item.is_dir() and not item.name.startswith("__"):
            # Check if it has model files
            if find_model_files(item, ["xml", "urdf"]) or (item / "__init__.py").exists():
                robot_dirs.append(item.name)
    
    # Generate new __init__.py content
    import_lines = []
    all_lines = []
    
    for robot_dir in robot_dirs:
        # Convert to valid Python identifier
        import_name = robot_dir
        import_lines.append(f"from . import {import_name}")
        all_lines.append(f'    "{import_name}",')
    
    # Check if there's a note about not importing specific objects
    note_text = (
        "# Note: Do not import specific URDF namespace objects here (e.g., "
        "Bessica_D_v1_0_Covered),\n"
        "# as this may trigger \"partially initialized module\" circular import "
        "errors in some environments.\n"
        "# Correct usage: Access through subpackages, e.g.,\n"
        "#   from synriard import urdf\n"
        "#   urdf_path = urdf.Bessica_D_v1_0.Bessica_D_v1_0_Covered.urdf"
    ) if format_type == "urdf" else ""
    
    content = [
        "# This line makes the sub-folders available as attributes of this module",
    ] + import_lines
    
    if note_text:
        content.append("")
        content.append(note_text)
    
    content.extend([
        "",
        "__all__ = [",
    ] + all_lines + [
        "]",
        "",
    ])
    
    # Write to file
    init_file.write_text("\n".join(content))
    print(f"  ✓ Updated {init_file.name}/__init__.py")


def process_library(library_path: Path, format_types: List[str]) -> None:
    """
    Process the synriard library.
    
    :param library_path: Path to library root (e.g., Synria-Robot-Descriptions/synriard)
    :param format_types: List of formats to process ['mjcf', 'urdf']
    """
    library_name = library_path.name
    
    for format_type in format_types:
        format_dir = library_path / format_type
        if not format_dir.exists():
            print(f"⚠ Skipping {format_dir} (does not exist)")
            continue
        
        print(f"\n📁 Processing {library_name}/{format_type}/")
        
        # Process each robot directory
        robot_dirs = [d for d in format_dir.iterdir() 
                     if d.is_dir() and not d.name.startswith("__")]
        
        for robot_dir in sorted(robot_dirs):
            init_file = robot_dir / "__init__.py"
            
            # Generate __init__.py
            init_content = generate_init_py(robot_dir, format_type)
            
            if init_content is None:
                print(f"  ⚠ Skipping {robot_dir.name} (no model files found)")
                continue
            
            # Write __init__.py
            init_file.write_text(init_content)
            print(f"  ✓ Generated {robot_dir.name}/__init__.py")
        
        # Update parent __init__.py
        update_parent_init(format_dir, format_type)


def main():
    parser = argparse.ArgumentParser(
        description="Auto-generate __init__.py files for robot models in synriard"
    )
    parser.add_argument(
        "--format",
        choices=["mjcf", "urdf", "all"],
        default="all",
        help="Which format to process (default: all)"
    )
    parser.add_argument(
        "--synriard-path",
        type=str,
        default=None,
        help="Path to synriard directory (auto-detected if not specified)"
    )
    
    args = parser.parse_args()
    
    # Determine format types
    format_types = ["mjcf", "urdf"] if args.format == "all" else [args.format]
    
    # Find library path
    script_dir = Path(__file__).parent.absolute()
    
    if args.synriard_path:
        synriard_path = Path(args.synriard_path)
    else:
        # Auto-detect: script should be in Synria-Robot-Descriptions root
        synriard_path = script_dir / "synriard"
    
    if not synriard_path.exists():
        print(f"❌ Error: synriard directory not found at {synriard_path}")
        print(f"   Please specify --synriard-path or run from Synria-Robot-Descriptions root")
        return
    
    print(f"🔧 Processing Synria-Robot-Descriptions: {synriard_path}")
    process_library(synriard_path, format_types)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()

