#!/usr/bin/env python3
"""Debug ROI calculation issues."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

import numpy as np
from pathlib import Path
from dicom_utils import load_dicom_series_to_hu
from metal_detection import find_metal_cluster, create_roi_box

def debug_roi_calculation(patient_name):
    """Debug ROI calculation for a patient."""
    print(f"\nDebugging ROI for: {patient_name}")
    print("-" * 50)
    
    # Load CT data
    data_dir = Path("data")
    patient_path = data_dir / patient_name
    ct_dirs = [d for d in patient_path.iterdir() if d.is_dir() and "CT" in d.name]
    ct_dir = ct_dirs[0]
    
    ct_volume, ct_metadata = load_dicom_series_to_hu(str(ct_dir))
    print(f"CT volume shape: {ct_volume.shape}")
    print(f"Spacing: {ct_metadata['spacing']}")
    
    # Find metal cluster
    center_coords, metal_cluster_mask = find_metal_cluster(ct_volume)
    print(f"Metal center: {center_coords}")
    
    # Create ROI
    roi_bounds = create_roi_box(center_coords, ct_volume.shape, ct_metadata['spacing'], margin_cm=3.0)
    print(f"ROI bounds: {roi_bounds}")
    
    # Check if bounds are valid
    for axis in ['x', 'y', 'z']:
        min_key = f'{axis}_min'
        max_key = f'{axis}_max'
        if roi_bounds[min_key] >= roi_bounds[max_key]:
            print(f"ERROR: {min_key}={roi_bounds[min_key]} >= {max_key}={roi_bounds[max_key]}")
    
    # Calculate margin in voxels
    margin_mm = 30  # 3 cm
    margin_voxels = [int(margin_mm / abs(s)) for s in ct_metadata['spacing']]
    print(f"Margin voxels: {margin_voxels}")
    
    # Show detailed calculation
    for i, (axis, center, margin, size) in enumerate(zip(['z', 'y', 'x'], center_coords, margin_voxels, ct_volume.shape)):
        print(f"\n{axis}-axis:")
        print(f"  Center: {center}")
        print(f"  Margin: {margin} voxels")
        print(f"  Volume size: {size}")
        print(f"  Min bound: max(0, {center} - {margin}) = {max(0, center - margin)}")
        print(f"  Max bound: min({size}, {center} + {margin}) = {min(size, center + margin)}")


if __name__ == "__main__":
    # Debug all patients
    for patient in ["HIP1 Patient", "HIP2 Patient", "HIP3 Patient", "HIP4 Patient"]:
        debug_roi_calculation(patient)