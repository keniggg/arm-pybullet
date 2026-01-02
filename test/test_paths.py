#!/usr/bin/env python3
"""Test script to verify all model paths are correct."""

import os
from synriard import get_model_path, list_available_models


def test_model_paths():
    """Test that all model paths are correct and files exist."""
    print("=" * 80)
    print("Testing URDF model paths...")
    print("=" * 80)
    
    # Get all URDF models
    urdf_table = list_available_models(model_format="urdf", show_path=True)
    urdf_lines = urdf_table.split('\n')[2:]  # Skip header and separator
    
    urdf_errors = []
    urdf_count = 0
    urdf_passed = 0
    
    for line in urdf_lines:
        if not line.strip():
            continue
        
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 4:
            continue
            
        name = parts[0]
        version = parts[1]
        variant = parts[2]
        path = parts[3]
        
        urdf_count += 1
        
        # Verify file exists
        if not os.path.exists(path):
            urdf_errors.append(
                f"❌ {name} {version} {variant}: File not found\n"
                f"   Path: {path}"
            )
            continue
        
        # Verify get_model_path returns the same path
        try:
            # Use variant as-is (should be full variant like "gripper_50mm")
            retrieved_path = get_model_path(name, version=version, variant=variant)
            if retrieved_path != path:
                urdf_errors.append(
                    f"❌ {name} {version} {variant}: Path mismatch\n"
                    f"   Expected: {path}\n"
                    f"   Got:      {retrieved_path}"
                )
            else:
                print(f"✅ {name} {version} {variant}")
                urdf_passed += 1
        except Exception as e:
            urdf_errors.append(
                f"❌ {name} {version} {variant}: Error getting path\n"
                f"   Error: {e}"
            )
    
    print(f"\nURDF tests: {urdf_passed}/{urdf_count} passed")
    
    print("\n" + "=" * 80)
    print("Testing MJCF model paths...")
    print("=" * 80)
    
    # Get all MJCF models
    mjcf_table = list_available_models(model_format="mjcf", show_path=True)
    mjcf_lines = mjcf_table.split('\n')[2:]  # Skip header and separator
    
    mjcf_errors = []
    mjcf_count = 0
    mjcf_passed = 0
    
    for line in mjcf_lines:
        if not line.strip():
            continue
        
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 4:
            continue
            
        name = parts[0]
        version = parts[1]
        variant = parts[2]
        path = parts[3]
        
        mjcf_count += 1
        
        # Verify file exists
        if not os.path.exists(path):
            mjcf_errors.append(
                f"❌ {name} {version} {variant}: File not found\n"
                f"   Path: {path}"
            )
            continue
        
        # Verify get_model_path returns the same path
        try:
            # Use variant as-is (should be full variant like "gripper_50mm")
            retrieved_path = get_model_path(
                name, version=version, variant=variant, model_format="mjcf"
            )
            if retrieved_path != path:
                mjcf_errors.append(
                    f"❌ {name} {version} {variant}: Path mismatch\n"
                    f"   Expected: {path}\n"
                    f"   Got:      {retrieved_path}"
                )
            else:
                print(f"✅ {name} {version} {variant}")
                mjcf_passed += 1
        except Exception as e:
            mjcf_errors.append(
                f"❌ {name} {version} {variant}: Error getting path\n"
                f"   Error: {e}"
            )
    
    print(f"\nMJCF tests: {mjcf_passed}/{mjcf_count} passed")
    
    # Print errors if any
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    total_tests = urdf_count + mjcf_count
    total_passed = urdf_passed + mjcf_passed
    total_errors = len(urdf_errors) + len(mjcf_errors)
    
    if urdf_errors:
        print("\nURDF Errors:")
        for error in urdf_errors:
            print(f"  {error}")
    
    if mjcf_errors:
        print("\nMJCF Errors:")
        for error in mjcf_errors:
            print(f"  {error}")
    
    if total_errors == 0:
        print(f"\n✅ All {total_tests} tests passed!")
        return 0
    else:
        print(f"\n❌ {total_errors}/{total_tests} tests failed ({total_passed} passed)")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(test_model_paths())

