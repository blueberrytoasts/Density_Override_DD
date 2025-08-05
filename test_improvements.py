#!/usr/bin/env python3
"""Test the improved metal detection"""

import sys
sys.path.append('app')

from pathlib import Path
import numpy as np
from dicom_utils import load_dicom_series_to_hu
from metal_detection import detect_metal_volume

# Test with HIP4 (from the screenshot)
data_dir = Path("data")
patient_path = data_dir / "HIP4 Patient"
ct_dirs = [d for d in patient_path.iterdir() if d.is_dir() and "CT" in d.name]
ct_dir = ct_dirs[0]

print("Loading HIP4 data...")
ct_volume, ct_metadata = load_dicom_series_to_hu(str(ct_dir))

# Test with different parameters
configs = [
    {"margin_cm": 3.0, "fw_percentage": 75, "min_metal_hu": 2500},  # Original
    {"margin_cm": 5.0, "fw_percentage": 75, "min_metal_hu": 2000},  # Improved defaults
    {"margin_cm": 5.0, "fw_percentage": 60, "min_metal_hu": 2000},  # More inclusive
    {"margin_cm": 7.0, "fw_percentage": 50, "min_metal_hu": 1500},  # Very inclusive
]

for i, config in enumerate(configs):
    print(f"\nTest {i+1}: {config}")
    result = detect_metal_volume(ct_volume, ct_metadata['spacing'], **config)
    
    if result['mask'] is not None:
        metal_voxels = np.sum(result['mask'])
        roi = result['roi_bounds']
        roi_size_cm = {
            'z': (roi['z_max'] - roi['z_min']) * ct_metadata['spacing'][2] / 10,
            'y': (roi['y_max'] - roi['y_min']) * ct_metadata['spacing'][0] / 10,
            'x': (roi['x_max'] - roi['x_min']) * ct_metadata['spacing'][1] / 10,
        }
        print(f"  Metal voxels: {metal_voxels:,}")
        print(f"  ROI size (cm): Z={roi_size_cm['z']:.1f}, Y={roi_size_cm['y']:.1f}, X={roi_size_cm['x']:.1f}")
        
        # Check slice 124 (from screenshot)
        if 124 < result['mask'].shape[0]:
            slice_metal = np.sum(result['mask'][124])
            print(f"  Metal pixels in slice 124: {slice_metal}")