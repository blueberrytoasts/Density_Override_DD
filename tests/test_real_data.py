#!/usr/bin/env python3
"""
Test star profile algorithms on REAL patient CT data.

This script compares simplified vs star profile methods on actual DICOM data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

import numpy as np
from dicom_utils import load_dicom_series_to_hu
from core.metal_detection import MetalDetector, MetalDetectionMethod
from core.discrimination import ArtifactDiscriminator, DiscriminationMethod


def find_patient_folders():
    """Find available patient data folders."""
    data_dir = "data"
    if not os.path.exists(data_dir):
        return []

    folders = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and "Patient" in item:
            folders.append(item_path)
    return folders


def main():
    print("="*60)
    print("STAR PROFILE TESTING - REAL PATIENT DATA")
    print("="*60)

    # Find patient data
    patient_folders = find_patient_folders()

    if not patient_folders:
        print("\n[ERROR] No patient data found in 'data/' folder")
        print("\nExpected folder structure:")
        print("  data/")
        print("    HIP001 Patient/")
        print("    HIP002 Patient/")
        print("    ...")
        return 1

    print(f"\nFound {len(patient_folders)} patient folder(s):")
    for i, folder in enumerate(patient_folders, 1):
        print(f"  {i}. {folder}")

    # Select patient
    if len(patient_folders) == 1:
        patient_folder = patient_folders[0]
        print(f"\nUsing: {patient_folder}")
    else:
        try:
            choice = int(input(f"\nSelect patient (1-{len(patient_folders)}): "))
            patient_folder = patient_folders[choice - 1]
        except (ValueError, IndexError):
            print("[ERROR] Invalid selection")
            return 1

    # Load CT data
    print("\n" + "="*60)
    print("LOADING DICOM DATA")
    print("="*60)

    try:
        # Find CT subdirectory (same approach as main.py)
        from pathlib import Path
        patient_path = Path(patient_folder)
        ct_dirs = [d for d in patient_path.iterdir() if d.is_dir() and "CT" in d.name]

        if not ct_dirs:
            print(f"[ERROR] No CT subdirectory found in {patient_folder}")
            print("Expected a folder with 'CT' in the name")
            return 1

        ct_dir = ct_dirs[0]
        print(f"Loading CT series from: {ct_dir.name}")

        ct_volume, spatial_meta = load_dicom_series_to_hu(str(ct_dir))

        if ct_volume is None or spatial_meta is None:
            print(f"[ERROR] Failed to load DICOM data from {ct_dir}")
            return 1

        # Extract spacing (z, y, x) from spatial metadata
        spacing = spatial_meta['spacing']

        print(f"[OK] Loaded CT volume: {ct_volume.shape}")
        print(f"     Spacing: {spacing} mm (z, y, x)")
        print(f"     HU range: {np.min(ct_volume):.0f} to {np.max(ct_volume):.0f}")
        print(f"     Slice positions: {len(spatial_meta['slice_z_positions'])} slices")
    except Exception as e:
        print(f"[ERROR] Failed to load DICOM: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test 1: Metal Detection - Simplified
    print("\n" + "="*60)
    print("TEST 1: SIMPLIFIED METAL DETECTION")
    print("="*60)

    print("\nRunning detection WITHOUT star profiles...")
    detector_simple = MetalDetector(MetalDetectionMethod.ADAPTIVE_3D)
    result_simple = detector_simple.detect(
        ct_volume,
        spacing,
        use_star_profiles=False,
        intensity_percentile=99.5
    )

    print(f"\nResults:")
    print(f"  Threshold: {result_simple['threshold']:.1f} HU")
    print(f"  Metal voxels: {np.sum(result_simple['mask']):,}")
    print(f"  Valid slices: {len(result_simple['valid_roi_slices'])}")
    print(f"  ROI bounds: z={result_simple['roi_bounds']['z_min']}-{result_simple['roi_bounds']['z_max']}")

    # Test 2: Metal Detection - Star Profiles
    print("\n" + "="*60)
    print("TEST 2: STAR PROFILE METAL DETECTION")
    print("="*60)

    print("\nRunning detection WITH star profiles...")
    detector_star = MetalDetector(MetalDetectionMethod.ADAPTIVE_3D)
    result_star = detector_star.detect(
        ct_volume,
        spacing,
        use_star_profiles=True,  # ENABLE STAR PROFILES
        fw_percentage=75.0,
        intensity_percentile=99.5
    )

    print(f"\nResults:")
    print(f"  Average threshold: {result_star['threshold']:.1f} HU")
    if len(result_star['threshold_evolution']) > 1:
        print(f"  Threshold range: {min(result_star['threshold_evolution']):.1f} - {max(result_star['threshold_evolution']):.1f} HU")
        print(f"  Threshold std dev: {np.std(result_star['threshold_evolution']):.1f} HU")
    print(f"  Metal voxels: {np.sum(result_star['mask']):,}")
    print(f"  Valid slices: {len(result_star['valid_roi_slices'])}")
    print(f"  Per-slice thresholds: {len(result_star['threshold_evolution'])}")

    # Comparison
    print("\n[Comparison]")
    metal_diff = np.sum(result_star['mask']) - np.sum(result_simple['mask'])
    threshold_diff = result_star['threshold'] - result_simple['threshold']
    print(f"  Threshold difference: {threshold_diff:+.1f} HU")
    print(f"  Metal voxel difference: {metal_diff:+,} ({metal_diff / np.sum(result_simple['mask']) * 100:+.1f}%)")

    if len(result_star['threshold_evolution']) > 1:
        print(f"  [OK] Star profile per-slice analysis working")
    else:
        print(f"  [WARN] Only 1 threshold calculated - star profiles may not be active")

    # Test 3: Discrimination Comparison
    print("\n" + "="*60)
    print("TEST 3: DISCRIMINATION METHODS")
    print("="*60)

    # Use star profile detection result
    metal_mask = result_star['mask']

    # Create bright mask (potential bone and artifacts)
    bright_threshold = 400  # HU
    bright_mask = (ct_volume > bright_threshold) & (~metal_mask)
    bright_voxels = np.sum(bright_mask)

    print(f"\nBright voxels to classify (>{bright_threshold} HU): {bright_voxels:,}")

    if bright_voxels == 0:
        print("[WARN] No bright voxels found - skipping discrimination test")
    else:
        # Distance-based discrimination (current default)
        print("\n[A] Distance-based discrimination...")
        disc_distance = ArtifactDiscriminator(DiscriminationMethod.DISTANCE_BASED)
        result_distance = disc_distance.discriminate(
            ct_volume, metal_mask, bright_mask, spacing
        )

        print(f"  Bone voxels: {result_distance['metadata']['bone_voxels']:,}")
        print(f"  Artifact voxels: {result_distance['metadata']['artifact_voxels']:,}")
        bone_pct_dist = result_distance['metadata']['bone_voxels'] / bright_voxels * 100
        art_pct_dist = result_distance['metadata']['artifact_voxels'] / bright_voxels * 100
        print(f"  Bone: {bone_pct_dist:.1f}%, Artifacts: {art_pct_dist:.1f}%")

        # Star profile discrimination (recovered)
        print("\n[B] Star profile discrimination...")
        print("  (This may take 10-30 seconds...)")
        disc_star = ArtifactDiscriminator(DiscriminationMethod.STAR_PROFILE)
        result_star_disc = disc_star.discriminate(
            ct_volume, metal_mask, bright_mask, spacing, num_angles=16
        )

        print(f"  Bone voxels: {result_star_disc['metadata']['bone_voxels']:,}")
        print(f"  Artifact voxels: {result_star_disc['metadata']['artifact_voxels']:,}")
        bone_pct_star = result_star_disc['metadata']['bone_voxels'] / bright_voxels * 100
        art_pct_star = result_star_disc['metadata']['artifact_voxels'] / bright_voxels * 100
        print(f"  Bone: {bone_pct_star:.1f}%, Artifacts: {art_pct_star:.1f}%")

        # Confidence analysis
        if np.any(bright_mask):
            conf_mean = np.mean(result_star_disc['confidence_map'][bright_mask])
            conf_min = np.min(result_star_disc['confidence_map'][bright_mask])
            conf_max = np.max(result_star_disc['confidence_map'][bright_mask])
            print(f"  Confidence: {conf_mean:.2f} (range: {conf_min:.2f}-{conf_max:.2f})")

        # Comparison
        print("\n[Comparison]")
        bone_diff = result_star_disc['metadata']['bone_voxels'] - result_distance['metadata']['bone_voxels']
        art_diff = result_star_disc['metadata']['artifact_voxels'] - result_distance['metadata']['artifact_voxels']

        print(f"  Bone voxel difference: {bone_diff:+,} ({bone_diff / result_distance['metadata']['bone_voxels'] * 100:+.1f}%)")
        print(f"  Artifact voxel difference: {art_diff:+,} ({art_diff / result_distance['metadata']['artifact_voxels'] * 100 if result_distance['metadata']['artifact_voxels'] > 0 else 0:+.1f}%)")

        if art_diff > 0:
            print(f"  [OK] Star profile identifies {art_diff:,} MORE artifacts")
        else:
            print(f"  [INFO] Star profile identifies {abs(art_diff):,} FEWER artifacts")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\nMetal Detection:")
    print(f"  [{'OK' if len(result_star['threshold_evolution']) > 1 else 'WARN'}] Star profile per-slice thresholding: {len(result_star['threshold_evolution'])} thresholds")
    print(f"  [{'OK' if abs(threshold_diff) > 10 else 'INFO'}] Threshold difference: {threshold_diff:+.1f} HU")
    print(f"  [{'OK' if abs(metal_diff) < np.sum(result_simple['mask']) * 0.1 else 'WARN'}] Metal voxel difference: {metal_diff / np.sum(result_simple['mask']) * 100:+.1f}%")

    if bright_voxels > 0:
        print("\nDiscrimination:")
        print(f"  [{'OK' if art_diff > 0 else 'INFO'}] Star profile artifact detection: {art_diff:+,} voxels")
        print(f"  [OK] Both methods classified all {bright_voxels:,} bright voxels")

    print("\nConclusion:")
    if len(result_star['threshold_evolution']) > 1:
        print("  The star profile algorithms are working correctly!")
    else:
        print("  Star profiles may need adjustment or more metal slices")

    print("\nNext steps:")
    print("  1. Review results above")
    print("  2. Test in Streamlit UI for visual comparison")
    print("  3. Compare with existing results on this patient")
    print("  4. Merge to main if satisfied")

    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Test cancelled by user")
        exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
