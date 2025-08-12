#!/usr/bin/env python3
"""Debug why artifact segmentation returns 0 voxels."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

import numpy as np
from pathlib import Path
from dicom_utils import load_dicom_series_to_hu
from metal_detection import detect_metal_volume
from contour_operations import create_bright_artifact_mask

# Test with HIP1
patient_name = "HIP1 Patient"
data_dir = Path("data")
patient_path = data_dir / patient_name
ct_dirs = [d for d in patient_path.iterdir() if d.is_dir() and "CT" in d.name]
ct_dir = ct_dirs[0]

print(f"Loading {patient_name}...")
ct_volume, ct_metadata = load_dicom_series_to_hu(str(ct_dir))

print("Detecting metal...")
result = detect_metal_volume(ct_volume, ct_metadata['spacing'], margin_cm=3.0)
metal_mask = result['mask']
roi_bounds = result['roi_bounds']

print(f"Metal mask shape: {metal_mask.shape}")
print(f"Metal voxels: {np.sum(metal_mask)}")
print(f"ROI bounds: {roi_bounds}")

# Check ROI volume
roi_volume = ct_volume[
    roi_bounds['z_min']:roi_bounds['z_max'],
    roi_bounds['y_min']:roi_bounds['y_max'],
    roi_bounds['x_min']:roi_bounds['x_max']
]
print(f"ROI volume shape: {roi_volume.shape}")
print(f"ROI HU range: [{roi_volume.min()}, {roi_volume.max()}]")

# Check values within ROI that meet bright artifact criteria
bright_low, bright_high = 800, 3000
bright_voxels = np.sum((roi_volume >= bright_low) & (roi_volume <= bright_high))
print(f"Voxels in bright range ({bright_low}-{bright_high} HU): {bright_voxels}")

# Try creating bright mask
print("\nCreating bright artifact mask...")
bright_mask = create_bright_artifact_mask(
    ct_volume, metal_mask, roi_bounds, bright_low, bright_high
)
print(f"Bright artifact voxels: {np.sum(bright_mask)}")

# Check the contour_operations to see if there's an issue
print("\nChecking bright mask creation logic...")
roi_mask = np.zeros_like(ct_volume, dtype=bool)
roi_mask[
    roi_bounds['z_min']:roi_bounds['z_max'],
    roi_bounds['y_min']:roi_bounds['y_max'],
    roi_bounds['x_min']:roi_bounds['x_max']
] = True

bright_mask_manual = ((ct_volume >= bright_low) & 
                     (ct_volume <= bright_high) & 
                     roi_mask & 
                     ~metal_mask)
print(f"Manual bright calculation: {np.sum(bright_mask_manual)} voxels")