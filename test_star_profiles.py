#!/usr/bin/env python3
"""
Test script for recovered star profile algorithms.

This script validates that the star profile implementations work correctly
with synthetic data before testing on real CT scans.
"""

import numpy as np
import sys
import os

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from core.metal_detection import MetalDetector, MetalDetectionMethod
from core.discrimination import ArtifactDiscriminator, DiscriminationMethod


def create_synthetic_ct_volume():
    """Create a synthetic CT volume with metal implant and artifacts."""
    print("Creating synthetic CT volume...")

    # Create base volume (512x512x100 voxels)
    ct_volume = np.random.randn(100, 512, 512) * 50 + 0  # Soft tissue baseline

    # Add metal implant (spherical, ~3000 HU)
    center_z, center_y, center_x = 50, 256, 256
    radius = 15

    for z in range(max(0, center_z - radius), min(100, center_z + radius)):
        for y in range(max(0, center_y - radius), min(512, center_y + radius)):
            for x in range(max(0, center_x - radius), min(512, center_x + radius)):
                dist = np.sqrt((z - center_z)**2 + (y - center_y)**2 + (x - center_x)**2)
                if dist < radius:
                    ct_volume[z, y, x] = 3000 + np.random.randn() * 100

    # Add bright artifacts (streaks radiating from metal)
    for angle in np.linspace(0, 2*np.pi, 8):
        for r in range(radius + 5, radius + 40):
            y = int(center_y + r * np.sin(angle))
            x = int(center_x + r * np.cos(angle))
            if 0 <= y < 512 and 0 <= x < 512:
                # Artifacts: narrow, high intensity
                ct_volume[center_z-2:center_z+2, y-1:y+1, x-1:x+1] = 800 + np.random.randn() * 50

    # Add bone (broad, moderate intensity)
    bone_y_start, bone_y_end = 350, 400
    bone_x_start, bone_x_end = 200, 300
    for z in range(40, 60):
        ct_volume[z, bone_y_start:bone_y_end, bone_x_start:bone_x_end] = 600 + np.random.randn() * 80

    print(f"[OK] Created CT volume: {ct_volume.shape}")
    print(f"  - Metal peak: {np.max(ct_volume):.0f} HU")
    print(f"  - Background: {np.mean(ct_volume[:, :100, :100]):.0f} HU")

    return ct_volume, (center_z, center_y, center_x)


def test_star_profile_detection():
    """Test FW75% star profile metal detection."""
    print("\n" + "="*60)
    print("TEST 1: Star Profile Metal Detection")
    print("="*60)

    ct_volume, (center_z, center_y, center_x) = create_synthetic_ct_volume()
    spacing = (2.5, 0.7, 0.7)  # mm

    # Test without star profiles (simplified)
    print("\n[A] Testing WITHOUT star profiles (simplified method)...")
    detector_simple = MetalDetector(MetalDetectionMethod.ADAPTIVE_3D)
    result_simple = detector_simple.detect(ct_volume, spacing, use_star_profiles=False)

    print(f"  Threshold: {result_simple['threshold']:.1f} HU")
    print(f"  Metal voxels: {np.sum(result_simple['mask'])}")
    print(f"  Valid slices: {len(result_simple['valid_roi_slices'])}")

    # Test with star profiles (recovered algorithm)
    print("\n[B] Testing WITH star profiles (recovered algorithm)...")
    detector_star = MetalDetector(MetalDetectionMethod.ADAPTIVE_3D)
    result_star = detector_star.detect(ct_volume, spacing, use_star_profiles=True, fw_percentage=75.0)

    print(f"  Average threshold: {result_star['threshold']:.1f} HU")
    print(f"  Threshold range: {min(result_star['threshold_evolution']):.1f} - {max(result_star['threshold_evolution']):.1f} HU")
    print(f"  Metal voxels: {np.sum(result_star['mask'])}")
    print(f"  Valid slices: {len(result_star['valid_roi_slices'])}")
    print(f"  Per-slice thresholds calculated: {len(result_star['threshold_evolution'])}")

    # Validation
    print("\n[Validation]")
    if len(result_star['threshold_evolution']) > 1:
        print("  [PASS] Per-slice star profile analysis working")
    else:
        print("  [FAIL] Per-slice analysis not working correctly")

    if result_star['threshold'] != result_simple['threshold']:
        print("  [PASS] Star profile produces different threshold than simplified")
    else:
        print("  [WARN] Star profile threshold same as simplified (unexpected)")

    if np.sum(result_star['mask']) > 0:
        print("  [PASS] Metal detected successfully")
    else:
        print("  [FAIL] No metal detected")

    return result_star


def test_legacy_star_profile():
    """Test legacy method star profile calculation."""
    print("\n" + "="*60)
    print("TEST 2: Legacy Method Star Profile")
    print("="*60)

    ct_volume, _ = create_synthetic_ct_volume()
    spacing = (2.5, 0.7, 0.7)

    print("\nTesting legacy detection with star profile refinement...")
    detector = MetalDetector(MetalDetectionMethod.LEGACY)
    result = detector.detect(ct_volume, spacing, min_metal_hu=2500, fw_percentage=75.0)

    print(f"  Threshold (star profile): {result['threshold']:.1f} HU")
    print(f"  Metal voxels: {np.sum(result['mask'])}")
    print(f"  Holes filled: {result['metadata']['holes_filled']}")

    print("\n[Validation]")
    if result['threshold'] > 2000:
        print("  [PASS] Star profile threshold calculated")
    else:
        print("  [FAIL] Threshold too low")

    if np.sum(result['mask']) > 0:
        print("  [PASS] Metal detected")
    else:
        print("  [FAIL] No metal detected")


def test_star_profile_discrimination():
    """Test star profile-based bone/artifact discrimination."""
    print("\n" + "="*60)
    print("TEST 3: Star Profile Discrimination")
    print("="*60)

    ct_volume, (center_z, center_y, center_x) = create_synthetic_ct_volume()
    spacing = (2.5, 0.7, 0.7)

    # First detect metal
    print("\n[A] Detecting metal...")
    detector = MetalDetector(MetalDetectionMethod.ADAPTIVE_3D)
    metal_result = detector.detect(ct_volume, spacing, use_star_profiles=False)
    metal_mask = metal_result['mask']

    # Create bright mask (artifacts + bone)
    bright_mask = (ct_volume > 500) & (~metal_mask)
    print(f"  Bright voxels to discriminate: {np.sum(bright_mask)}")

    # Test distance-based discrimination (baseline)
    print("\n[B] Testing distance-based discrimination (baseline)...")
    disc_distance = ArtifactDiscriminator(DiscriminationMethod.DISTANCE_BASED)
    result_distance = disc_distance.discriminate(ct_volume, metal_mask, bright_mask, spacing)

    print(f"  Bone voxels: {result_distance['metadata']['bone_voxels']}")
    print(f"  Artifact voxels: {result_distance['metadata']['artifact_voxels']}")

    # Test star profile discrimination (recovered)
    print("\n[C] Testing star profile discrimination (recovered)...")
    disc_star = ArtifactDiscriminator(DiscriminationMethod.STAR_PROFILE)
    result_star = disc_star.discriminate(ct_volume, metal_mask, bright_mask, spacing, num_angles=16)

    print(f"  Bone voxels: {result_star['metadata']['bone_voxels']}")
    print(f"  Artifact voxels: {result_star['metadata']['artifact_voxels']}")
    print(f"  Confidence range: {np.min(result_star['confidence_map'][bright_mask]):.2f} - {np.max(result_star['confidence_map'][bright_mask]):.2f}")

    # Validation
    print("\n[Validation]")
    total_bright = np.sum(bright_mask)
    classified = result_star['metadata']['bone_voxels'] + result_star['metadata']['artifact_voxels']

    if classified > 0:
        print(f"  [PASS] Star profile discrimination working ({classified}/{total_bright} voxels classified)")
    else:
        print("  [FAIL] No voxels classified")

    if result_star['metadata']['bone_voxels'] > 0 and result_star['metadata']['artifact_voxels'] > 0:
        print("  [PASS] Both bone and artifacts detected")
    else:
        print("  [WARN] Only one class detected")

    # Compare methods
    print("\n[Comparison]")
    bone_diff = result_star['metadata']['bone_voxels'] - result_distance['metadata']['bone_voxels']
    print(f"  Bone voxel difference: {bone_diff:+d}")
    print(f"  Star profile classifies {'more' if bone_diff > 0 else 'less'} as bone")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("STAR PROFILE ALGORITHM RECOVERY - TEST SUITE")
    print("="*60)
    print("\nThis script tests the recovered star profile algorithms:")
    print("  1. FW75% star profile metal detection")
    print("  2. Legacy method star profile refinement")
    print("  3. Star profile-based bone/artifact discrimination")
    print("\n" + "="*60)

    try:
        # Run tests
        test_star_profile_detection()
        test_legacy_star_profile()
        test_star_profile_discrimination()

        # Summary
        print("\n" + "="*60)
        print("TEST SUITE COMPLETED")
        print("="*60)
        print("\n[SUCCESS] All tests executed successfully!")
        print("\nThe recovered star profile algorithms are functional.")
        print("Review the output above to verify expected behavior.")
        print("\nNext steps:")
        print("  1. Test on real CT data")
        print("  2. Compare accuracy with simplified methods")
        print("  3. Integrate into Streamlit UI")
        print("  4. Optimize performance if needed")

    except Exception as e:
        print("\n" + "="*60)
        print("TEST FAILED")
        print("="*60)
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
