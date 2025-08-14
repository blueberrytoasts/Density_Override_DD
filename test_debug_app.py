#!/usr/bin/env python3
"""
Test script to debug the CT Metal Artifact Characterization application
Tests each HIP patient data systematically
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from dicom_utils import load_dicom_series_to_hu
from metal_detection import detect_metal_volume, create_affine_from_dicom_meta
from contour_operations import (create_bright_artifact_mask, create_dark_artifact_mask,
                               create_bone_mask, refine_mask)
from visualization import create_overlay_image, fig_to_base64

def test_patient_data(patient_name, data_dir):
    """Test loading and processing a single patient's data"""
    print(f"\n{'='*60}")
    print(f"Testing {patient_name}")
    print(f"{'='*60}")
    
    patient_path = data_dir / patient_name
    
    # Find CT directory
    ct_dirs = [d for d in patient_path.iterdir() if d.is_dir() and "CT" in d.name]
    if not ct_dirs:
        print(f"ERROR: No CT directory found for {patient_name}")
        return False
    
    ct_dir = ct_dirs[0]
    print(f"Found CT directory: {ct_dir.name}")
    
    # Test 1: Load DICOM data
    print("\nTest 1: Loading DICOM data...")
    try:
        ct_volume, ct_metadata = load_dicom_series_to_hu(str(ct_dir))
        if ct_volume is not None:
            print(f"✓ Loaded {ct_volume.shape[0]} slices successfully")
            print(f"  Volume shape: {ct_volume.shape}")
            print(f"  Spacing: {ct_metadata['spacing']}")
            print(f"  HU range: [{np.min(ct_volume):.1f}, {np.max(ct_volume):.1f}]")
        else:
            print("✗ Failed to load CT data")
            return False
    except Exception as e:
        print(f"✗ Error loading DICOM: {str(e)}")
        return False
    
    # Test 2: Metal detection
    print("\nTest 2: Running metal detection...")
    try:
        result = detect_metal_volume(
            ct_volume,
            ct_metadata['spacing'],
            margin_cm=3.0
        )
        
        if result['mask'] is not None:
            metal_voxels = np.sum(result['mask'])
            print(f"✓ Metal detection complete")
            print(f"  Metal voxels: {metal_voxels:,}")
            print(f"  Center: {result['center_coords']}")
            roi_bounds = result['roi_bounds']
            print(f"  ROI: Z[{roi_bounds['z_min']}-{roi_bounds['z_max']}], "
                  f"Y[{roi_bounds['y_min']}-{roi_bounds['y_max']}], "
                  f"X[{roi_bounds['x_min']}-{roi_bounds['x_max']}]")
        else:
            print("✗ No metal implant detected")
            return False
    except Exception as e:
        print(f"✗ Error in metal detection: {str(e)}")
        return False
    
    # Test 3: Artifact segmentation
    print("\nTest 3: Segmenting artifacts...")
    try:
        metal_mask = result['mask']
        roi_bounds = result['roi_bounds']
        
        # Create artifact masks
        bright_mask = create_bright_artifact_mask(
            ct_volume, metal_mask, roi_bounds, 800, 3000
        )
        dark_mask = create_dark_artifact_mask(
            ct_volume, metal_mask, roi_bounds, -200
        )
        bone_mask = create_bone_mask(
            ct_volume, metal_mask, bright_mask, dark_mask, 
            roi_bounds, 150, 1500
        )
        
        # Refine masks
        bright_mask = refine_mask(bright_mask)
        dark_mask = refine_mask(dark_mask)
        bone_mask = refine_mask(bone_mask)
        
        print(f"✓ Artifact segmentation complete")
        print(f"  Bright artifacts: {np.sum(bright_mask):,} voxels")
        print(f"  Dark artifacts: {np.sum(dark_mask):,} voxels")
        print(f"  Bone: {np.sum(bone_mask):,} voxels")
    except Exception as e:
        print(f"✗ Error in artifact segmentation: {str(e)}")
        return False
    
    # Test 4: Visualization
    print("\nTest 4: Testing visualization...")
    try:
        # Test on middle slice
        slice_idx = ct_volume.shape[0] // 2
        ct_slice = ct_volume[slice_idx]
        
        # Create masks for current slice
        slice_masks = {
            'metal': metal_mask[slice_idx],
            'bright_artifacts': bright_mask[slice_idx],
            'dark_artifacts': dark_mask[slice_idx],
            'bone': bone_mask[slice_idx]
        }
        
        # Convert ROI bounds to 2D tuple format
        roi_boundaries_tuple = (
            roi_bounds['y_min'],
            roi_bounds['y_max'],
            roi_bounds['x_min'],
            roi_bounds['x_max']
        )
        
        # Create overlay image
        fig = create_overlay_image(
            ct_slice,
            slice_masks,
            roi_boundaries_tuple,
            slice_idx
        )
        
        # Convert to base64
        img_base64 = fig_to_base64(fig)
        
        print(f"✓ Visualization created successfully")
        print(f"  Image size: {len(img_base64)} characters (base64)")
    except Exception as e:
        print(f"✗ Error in visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n✓ All tests passed for {patient_name}")
    return True


def main():
    """Main test function"""
    print("CT Metal Artifact Characterization - Debug Test")
    print("=" * 60)
    
    # Find data directory
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir = Path("../data")
    
    if not data_dir.exists():
        print("ERROR: Data directory not found")
        return
    
    # Get list of HIP patients
    patient_dirs = sorted([d for d in data_dir.iterdir() 
                          if d.is_dir() and "HIP" in d.name])
    
    print(f"Found {len(patient_dirs)} patient directories")
    
    # Test each patient
    results = {}
    for patient_dir in patient_dirs:
        success = test_patient_data(patient_dir.name, data_dir)
        results[patient_dir.name] = success
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for patient, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{patient}: {status}")
    
    total_passed = sum(results.values())
    print(f"\nTotal: {total_passed}/{len(results)} patients passed all tests")


if __name__ == "__main__":
    main()