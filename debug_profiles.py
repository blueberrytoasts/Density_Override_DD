#!/usr/bin/env python3
"""Debug the star profiles"""

import sys
sys.path.append('app')

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from dicom_utils import load_dicom_series_to_hu
from metal_detection_v2 import find_high_intensity_region, create_search_box, get_star_profile_lines

# Load HIP4 data
data_dir = Path("data")
patient_path = data_dir / "HIP4 Patient"
ct_dirs = [d for d in patient_path.iterdir() if d.is_dir() and "CT" in d.name]
ct_dir = ct_dirs[0]

ct_volume, ct_metadata = load_dicom_series_to_hu(str(ct_dir))

# Get the high intensity region
center_coords, initial_region, auto_threshold = find_high_intensity_region(ct_volume)
print(f"Center: {center_coords}, Auto threshold: {auto_threshold}")

# Create search box
box_bounds = create_search_box(center_coords, ct_volume.shape, ct_metadata['spacing'], 8.0)
print(f"Box bounds: {box_bounds}")

# Look at slice 124
slice_124 = ct_volume[124]
search_bounds_2d = {
    'y_min': box_bounds['y_min'],
    'y_max': box_bounds['y_max'],
    'x_min': box_bounds['x_min'],
    'x_max': box_bounds['x_max']
}

# Get star profiles
profiles = get_star_profile_lines(slice_124, center_coords[1], center_coords[2], search_bounds_2d)

# Analyze profiles
for i, (distances, hu_values) in enumerate(profiles[:3]):  # Look at first 3 profiles
    print(f"\nProfile {i}:")
    print(f"  HU range: {np.min(hu_values):.0f} - {np.max(hu_values):.0f}")
    print(f"  HU values > 2000: {np.sum(hu_values > 2000)}")
    print(f"  HU values > 1000: {np.sum(hu_values > 1000)}")
    print(f"  HU values > 500: {np.sum(hu_values > 500)}")

# Check the actual HU values in the metal region
metal_region = slice_124 > 2000
if np.any(metal_region):
    metal_hu = slice_124[metal_region]
    print(f"\nActual metal HU values in slice 124:")
    print(f"  Min: {np.min(metal_hu):.0f}")
    print(f"  Max: {np.max(metal_hu):.0f}")
    print(f"  Mean: {np.mean(metal_hu):.0f}")
    print(f"  75% value: {np.percentile(metal_hu, 75):.0f}")
    print(f"  50% value: {np.percentile(metal_hu, 50):.0f}")