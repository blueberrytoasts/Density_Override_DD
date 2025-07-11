#!/usr/bin/env python3
"""Test single patient quickly."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

import numpy as np
from pathlib import Path
from dicom_utils import load_dicom_series_to_hu
from metal_detection import detect_metal_volume
from contour_operations import (create_bright_artifact_mask, create_dark_artifact_mask,
                               create_bone_mask, refine_mask)
from visualization import create_overlay_image

# Test with HIP1
patient_name = "HIP1 Patient"
data_dir = Path("data")
patient_path = data_dir / patient_name
ct_dirs = [d for d in patient_path.iterdir() if d.is_dir() and "CT" in d.name]
ct_dir = ct_dirs[0]

print(f"Loading {patient_name}...")
ct_volume, ct_metadata = load_dicom_series_to_hu(str(ct_dir))
print(f"✓ Loaded volume: {ct_volume.shape}")

print("\nDetecting metal...")
result = detect_metal_volume(ct_volume, ct_metadata['spacing'], margin_cm=3.0)
metal_mask = result['mask']
roi_bounds = result['roi_bounds']
print(f"✓ Metal voxels: {np.sum(metal_mask):,}")
print(f"  ROI: {roi_bounds}")

print("\nSegmenting artifacts...")
bright_low, bright_high = 800, 3000
dark_high = -200
bone_low, bone_high = 150, 1500

bright_mask = create_bright_artifact_mask(ct_volume, metal_mask, roi_bounds, bright_low, bright_high)
dark_mask = create_dark_artifact_mask(ct_volume, metal_mask, roi_bounds, dark_high)
bone_mask = create_bone_mask(ct_volume, metal_mask, bright_mask, dark_mask, roi_bounds, bone_low, bone_high)

print(f"  Raw bright: {np.sum(bright_mask):,} voxels")
print(f"  Raw dark: {np.sum(dark_mask):,} voxels")
print(f"  Raw bone: {np.sum(bone_mask):,} voxels")

bright_mask = refine_mask(bright_mask)
dark_mask = refine_mask(dark_mask)
bone_mask = refine_mask(bone_mask)

print(f"✓ Refined results:")
print(f"  Bright: {np.sum(bright_mask):,} voxels")
print(f"  Dark: {np.sum(dark_mask):,} voxels")
print(f"  Bone: {np.sum(bone_mask):,} voxels")

print("\nTesting visualization...")
current_slice = ct_volume.shape[0] // 2
ct_slice = ct_volume[current_slice]

slice_masks = {
    'metal': metal_mask[current_slice],
    'bright_artifacts': bright_mask[current_slice],
    'dark_artifacts': dark_mask[current_slice],
    'bone': bone_mask[current_slice]
}

roi_boundaries_tuple = (
    roi_bounds['y_min'],
    roi_bounds['y_max'],
    roi_bounds['x_min'],
    roi_bounds['x_max']
)

fig = create_overlay_image(ct_slice, slice_masks, roi_boundaries_tuple, current_slice)
print("✓ Visualization created successfully")

print("\nAll tests passed!")