"""
Analyze slice 124 in HIP4 patient - ADAPTIVE_3D method only.
Reports the thresholds used for metal detection.
"""
import numpy as np
import pydicom
import os
from pathlib import Path

# Import the metal detection module
import sys
sys.path.insert(0, str(Path(__file__).parent / 'app' / 'core'))
from metal_detection import MetalDetector, MetalDetectionMethod

def load_ct_volume(ct_dir):
    """Load CT DICOM files and convert to HU volume."""
    ct_files = sorted([f for f in os.listdir(ct_dir) if f.endswith('.dcm')])
    slices = []
    for f in ct_files:
        ds = pydicom.dcmread(os.path.join(ct_dir, f))
        slices.append(ds)

    # Sort by Z coordinate
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    # Convert to HU
    volume = []
    for ds in slices:
        pixel_array = ds.pixel_array.astype(np.float32)
        slope = float(ds.RescaleSlope) if hasattr(ds, 'RescaleSlope') else 1.0
        intercept = float(ds.RescaleIntercept) if hasattr(ds, 'RescaleIntercept') else 0.0
        hu_array = pixel_array * slope + intercept
        volume.append(hu_array)

    volume = np.array(volume)

    # Get spacing
    spacing_z = float(slices[0].SliceThickness) if hasattr(slices[0], 'SliceThickness') else 1.0
    spacing_y, spacing_x = slices[0].PixelSpacing
    spacing = (spacing_z, float(spacing_y), float(spacing_x))

    return volume, spacing

if __name__ == "__main__":
    ct_dir = r"data\HIP4 Patient\HIP4 CT"

    print("Loading HIP4 CT volume...")
    ct_volume, spacing = load_ct_volume(ct_dir)
    print(f"Volume shape: {ct_volume.shape}")

    # Get slice 124
    slice_124 = ct_volume[124]

    print("\n" + "="*80)
    print("SLICE 124 - HU STATISTICS")
    print("="*80)
    print(f"Max HU on slice 124: {np.max(slice_124):.1f}")
    print(f"99.5th percentile (whole volume): {np.percentile(ct_volume, 99.5):.1f} HU")

    # Run ADAPTIVE_3D detection
    print("\n" + "="*80)
    print("ADAPTIVE_3D METAL DETECTION")
    print("="*80)
    detector = MetalDetector(MetalDetectionMethod.ADAPTIVE_3D)
    result = detector.detect(ct_volume, spacing)

    if result['mask'] is not None:
        print(f"\nTHRESHOLDS DETERMINED:")
        print(f"  Initial threshold (99.5th percentile): {result['threshold']:.1f} HU")

        metadata = result.get('metadata', {})
        if 'median_metal_hu' in metadata and metadata['median_metal_hu'] is not None:
            print(f"  Median metal HU (ROI restriction): {metadata['median_metal_hu']:.1f} HU")
            print(f"\n  -> Initial detection: anything > {result['threshold']:.1f} HU")
            print(f"  -> ROI restriction: only voxels >= {metadata['median_metal_hu']:.1f} HU included in ROI")

        # Slice 124 info
        metal_mask_124 = result['mask'][124]
        metal_count = np.sum(metal_mask_124)

        print(f"\n" + "-"*80)
        print(f"SLICE 124 RESULTS")
        print("-"*80)
        print(f"Metal voxels detected: {metal_count}")

        if metal_count > 0:
            metal_hu_values = slice_124[metal_mask_124]
            print(f"Metal HU range: [{np.min(metal_hu_values):.1f}, {np.max(metal_hu_values):.1f}]")
            print(f"Metal mean HU: {np.mean(metal_hu_values):.1f}")
            print(f"Metal median HU: {np.median(metal_hu_values):.1f}")
    else:
        print("\nNo metal detected")
