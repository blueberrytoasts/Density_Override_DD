#!/usr/bin/env python3
"""Test the pure adaptive metal detection"""

import sys
sys.path.append('app')

from pathlib import Path
import numpy as np
from dicom_utils import load_dicom_series_to_hu
from metal_detection_v2 import detect_metal_volume_pure_adaptive

# Test with HIP4 (from the screenshot)
data_dir = Path("data")
patient_path = data_dir / "HIP4 Patient"
ct_dirs = [d for d in patient_path.iterdir() if d.is_dir() and "CT" in d.name]
ct_dir = ct_dirs[0]

print("Loading HIP4 data...")
ct_volume, ct_metadata = load_dicom_series_to_hu(str(ct_dir))

print("\nTesting pure adaptive detection...")
result = detect_metal_volume_pure_adaptive(
    ct_volume, 
    ct_metadata['spacing'],
    search_box_cm=8.0,
    fw_percentage=75
)

if result['mask'] is not None:
    metal_voxels = np.sum(result['mask'])
    print(f"Total metal voxels: {metal_voxels:,}")
    print(f"Initial threshold used: {result['initial_threshold']:.0f} HU")
    
    # Check slice 124 specifically
    if 124 < result['mask'].shape[0]:
        slice_metal = np.sum(result['mask'][124])
        print(f"Metal pixels in slice 124: {slice_metal}")
        
        # Show some threshold info for slice 124
        slice_thresh = [t for t in result['slice_thresholds'] if t['slice'] == 124]
        if slice_thresh and slice_thresh[0]['thresholds']:
            lower, upper = slice_thresh[0]['thresholds']
            print(f"Slice 124 thresholds: {lower:.0f} - {upper:.0f} HU")
else:
    print("No metal detected")