#!/usr/bin/env python3
"""
Dead Code Cleanup Script for CT Metal Artifact Characterization Project

This script removes:
1. Unused UI sections from main.py
2. Unused functions from contour_operations.py
3. Unused discrimination methods from core/discrimination.py
4. Unused metal detection methods from core/metal_detection.py

Creates backups before modifying files.
"""

import os
import shutil
from datetime import datetime

def create_backup(filepath):
    """Create a timestamped backup of a file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{filepath}.backup_{timestamp}"
    shutil.copy2(filepath, backup_path)
    print(f"[OK] Created backup: {backup_path}")
    return backup_path

def remove_lines(filepath, start_marker, end_marker, description):
    """Remove lines between two markers (inclusive)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find start and end indices
    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if start_marker in line and start_idx is None:
            start_idx = i
        if end_marker in line and start_idx is not None and end_idx is None:
            end_idx = i
            break

    if start_idx is not None and end_idx is not None:
        removed_count = end_idx - start_idx + 1
        del lines[start_idx:end_idx + 1]
        print(f"  [OK] Removed {removed_count} lines: {description}")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        return True
    else:
        print(f"  [FAIL] Could not find markers for: {description}")
        return False

def remove_function(filepath, function_signature):
    """Remove an entire function by its signature."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find function start
    func_start = None
    for i, line in enumerate(lines):
        if function_signature in line and line.strip().startswith('def '):
            func_start = i
            break

    if func_start is None:
        print(f"  [FAIL] Could not find function: {function_signature}")
        return False

    # Find function end (next def or class at same indentation, or end of file)
    func_indent = len(lines[func_start]) - len(lines[func_start].lstrip())
    func_end = len(lines)

    for i in range(func_start + 1, len(lines)):
        line = lines[i]
        if line.strip() == '':
            continue
        current_indent = len(line) - len(line.lstrip())
        if current_indent <= func_indent and (line.strip().startswith('def ') or line.strip().startswith('class ')):
            func_end = i
            break

    removed_count = func_end - func_start
    function_name = function_signature.split('(')[0].replace('def ', '').strip()
    del lines[func_start:func_end]
    print(f"  [OK] Removed function {function_name} ({removed_count} lines)")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    return True

def remove_method(filepath, class_name, method_signature):
    """Remove a method from a class."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find class
    class_start = None
    for i, line in enumerate(lines):
        if f"class {class_name}" in line:
            class_start = i
            break

    if class_start is None:
        print(f"  [FAIL] Could not find class: {class_name}")
        return False

    # Find method within class
    method_start = None
    for i in range(class_start, len(lines)):
        if method_signature in lines[i] and '    def ' in lines[i]:
            method_start = i
            break

    if method_start is None:
        print(f"  [FAIL] Could not find method: {method_signature}")
        return False

    # Find method end
    method_indent = len(lines[method_start]) - len(lines[method_start].lstrip())
    method_end = len(lines)

    for i in range(method_start + 1, len(lines)):
        line = lines[i]
        if line.strip() == '':
            continue
        current_indent = len(line) - len(line.lstrip())
        if current_indent <= method_indent:
            method_end = i
            break

    removed_count = method_end - method_start
    method_name = method_signature.split('(')[0].replace('def ', '').strip()
    del lines[method_start:method_end]
    print(f"  [OK] Removed method {class_name}.{method_name} ({removed_count} lines)")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    return True

def remove_enum_value(filepath, enum_name, value_name):
    """Remove an enum value."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find and remove the enum value
    for i, line in enumerate(lines):
        if value_name in line and '=' in line and not line.strip().startswith('#'):
            del lines[i]
            print(f"  [OK] Removed enum value {enum_name}.{value_name}")
            break

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    return True

def cleanup_main_py():
    """Clean up dead code in main.py"""
    filepath = "app/main.py"
    print(f"\n{'='*60}")
    print(f"Cleaning up {filepath}")
    print(f"{'='*60}")

    create_backup(filepath)

    # Remove "Russian Doll with Enhanced Edge Analysis" UI section
    remove_lines(
        filepath,
        'elif segmentation_method == "Russian Doll with Enhanced Edge Analysis":',
        'st.markdown("- Multi-scale edge persistence")',
        "Enhanced Edge Analysis UI section"
    )

    # Remove "Russian Doll with Advanced Texture/Gradient Analysis" UI section
    remove_lines(
        filepath,
        'elif segmentation_method == "Russian Doll with Advanced Texture/Gradient Analysis (Best Accuracy)":',
        'st.session_state.thresholds[\'russian_doll\'][\'max_distance\'] = artifact_distance_cm_adv',
        "Advanced Texture/Gradient Analysis UI section"
    )

    # Remove success message check for Enhanced Edge Analysis
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    content = content.replace(
        '''if segmentation_method == "Russian Doll with Enhanced Edge Analysis":
                                    st.success("Enhanced edge-based segmentation complete!")
                                else:
                                    st.success("Smart artifact segmentation complete!")''',
        '''st.success("Smart artifact segmentation complete!")'''
    )

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print("  [OK] Removed Enhanced Edge Analysis success message check")

def cleanup_contour_operations():
    """Clean up dead code in contour_operations.py"""
    filepath = "app/contour_operations.py"
    print(f"\n{'='*60}")
    print(f"Cleaning up {filepath}")
    print(f"{'='*60}")

    create_backup(filepath)

    # Remove unused wrapper functions
    remove_function(filepath, "def create_fast_russian_doll_segmentation")
    remove_function(filepath, "def create_enhanced_russian_doll_segmentation")
    remove_function(filepath, "def combine_masks_multilabel")
    remove_function(filepath, "def load_nifti_mask")

def cleanup_discrimination():
    """Clean up dead code in core/discrimination.py"""
    filepath = "app/core/discrimination.py"
    print(f"\n{'='*60}")
    print(f"Cleaning up {filepath}")
    print(f"{'='*60}")

    create_backup(filepath)

    # Remove unused discrimination method
    remove_method(filepath, "ArtifactDiscriminator", "def _discriminate_star")

    # Remove convenience functions at the end
    remove_function(filepath, "def discriminate_fast")
    remove_function(filepath, "def discriminate_enhanced")
    remove_function(filepath, "def discriminate_advanced")

    # Remove STAR_PROFILE enum reference (update to remove alias)
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        # Remove the PROFILE_BASED alias line
        if 'PROFILE_BASED = "star_profile"' in line and '# Alias' in line:
            print("  [OK] Removed PROFILE_BASED enum alias")
            continue
        # Update discriminators dict to remove STAR_PROFILE
        if 'DiscriminationMethod.STAR_PROFILE:' in line:
            continue
        new_lines.append(line)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

def cleanup_metal_detection():
    """Clean up dead code in core/metal_detection.py"""
    filepath = "app/core/metal_detection.py"
    print(f"\n{'='*60}")
    print(f"Cleaning up {filepath}")
    print(f"{'='*60}")

    create_backup(filepath)

    # Remove ADAPTIVE_2D enum value
    remove_enum_value(filepath, "MetalDetectionMethod", "ADAPTIVE_2D")

    # Remove unused detection method
    remove_method(filepath, "MetalDetector", "def _detect_adaptive_2d")

    # Remove ADAPTIVE_2D from detectors dict
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if 'MetalDetectionMethod.ADAPTIVE_2D:' in line:
            print("  [OK] Removed ADAPTIVE_2D from detectors dict")
            continue
        new_lines.append(line)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

def main():
    """Run all cleanup operations."""
    import sys

    print("\n" + "="*60)
    print("CT Metal Artifact Characterization - Dead Code Cleanup")
    print("="*60)
    print("\nThis script will remove unused code from the following files:")
    print("  - app/main.py")
    print("  - app/contour_operations.py")
    print("  - app/core/discrimination.py")
    print("  - app/core/metal_detection.py")
    print("\nBackups will be created before any modifications.")

    # Check for --yes flag
    if '--yes' not in sys.argv:
        response = input("\nProceed with cleanup? (y/n): ")
        if response.lower() != 'y':
            print("Cleanup cancelled.")
            return

    try:
        cleanup_main_py()
        cleanup_contour_operations()
        cleanup_discrimination()
        cleanup_metal_detection()

        print(f"\n{'='*60}")
        print("[SUCCESS] Cleanup completed successfully!")
        print(f"{'='*60}")
        print("\nSummary:")
        print("  - Removed unused UI sections from main.py")
        print("  - Removed 4 unused functions from contour_operations.py")
        print("  - Removed 4 unused discrimination functions/methods")
        print("  - Removed 1 unused metal detection method")
        print("\nBackup files have been created with .backup_TIMESTAMP extension")
        print("If everything works correctly, you can delete the backup files.")

    except Exception as e:
        print(f"\n[ERROR] Error during cleanup: {str(e)}")
        print("Please check the backup files and restore if needed.")
        raise

if __name__ == "__main__":
    main()
