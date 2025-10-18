"""
Example script to export poster figures.
Run this after processing your CT data to generate all poster visualizations.
"""

import numpy as np
from pathlib import Path
from poster_figures import (
    create_pipeline_progression,
    create_before_after_comparison,
    create_multi_slice_montage,
    create_multi_patient_comparison,
    create_individual_thumbnails,
    export_all_poster_figures
)
from dicom_utils import load_dicom_series


def export_single_patient_figures(patient_folder, output_folder):
    """
    Export all figures for a single patient.

    Args:
        patient_folder: Path to patient DICOM folder (e.g., "data/HIP1 Patient")
        output_folder: Where to save figures (e.g., "output/poster_figures")
    """
    print(f"Processing {patient_folder}...")

    # 1. Load CT data
    ct_volume, spacing, origin, slice_positions = load_dicom_series(patient_folder)
    print(f"Loaded CT volume: {ct_volume.shape}")

    # 2. Load or generate masks
    # Option A: Load from saved NIFTI masks
    # import nibabel as nib
    # metal_img = nib.load(f"output/{patient_name}_metal.nii.gz")
    # metal_mask = metal_img.get_fdata().astype(bool)

    # Option B: Run segmentation (assuming you have these functions available)
    from metal_detection_v3 import detect_metal_3d_adaptive
    from contour_operations import create_sequential_masks

    # Detect metal
    metal_result = detect_metal_3d_adaptive(ct_volume, spacing)
    metal_mask = metal_result['metal_mask']

    # Create tissue masks
    masks_dict = create_sequential_masks(
        ct_volume,
        metal_mask,
        spacing,
        discrimination_method='edge',  # or 'distance', 'profile'
        bright_range=[800, 3500],
        bone_range=[500, 1500],
        dark_range=[-1024, -150]
    )

    # 3. Choose a representative slice
    # Find slice with most metal
    metal_counts = np.sum(metal_mask, axis=(1, 2))
    representative_slice = np.argmax(metal_counts)
    print(f"Using slice {representative_slice} as representative")

    # 4. Export all figures
    export_all_poster_figures(
        ct_volume=ct_volume,
        masks_dict=masks_dict,
        slice_idx=representative_slice,
        output_dir=output_folder,
        patient_label="Patient A",
        additional_slices=[
            representative_slice - 10,
            representative_slice - 5,
            representative_slice + 5,
            representative_slice + 10,
            representative_slice + 15
        ]
    )

    print(f"✓ All figures saved to {output_folder}")


def export_multi_patient_comparison_example():
    """
    Example: Create a multi-patient comparison figure.
    """
    # Load data for 3 patients
    patient_data_list = []

    for patient_name in ["HIP1 Patient", "HIP2 Patient", "HIP3 Patient"]:
        # Load CT
        ct_volume, spacing, _, _ = load_dicom_series(f"data/{patient_name}")

        # Load masks (from saved files or generate)
        # ... load or generate masks_dict ...

        # Find representative slice
        # metal_mask = masks_dict['metal']
        # metal_counts = np.sum(metal_mask, axis=(1, 2))
        # slice_idx = np.argmax(metal_counts)

        # For this example, use middle slice
        slice_idx = ct_volume.shape[0] // 2

        # Calculate stats
        stats = {
            'Metal': f"{np.sum(masks_dict['metal']) * 0.001:.1f} cm³",
            'Bone': f"{np.sum(masks_dict['bone']) * 0.001:.1f} cm³",
            'Artifacts': f"{np.sum(masks_dict['bright_artifacts']) * 0.001:.1f} cm³"
        }

        patient_data_list.append({
            'ct_volume': ct_volume,
            'masks_dict': masks_dict,
            'slice_idx': slice_idx,
            'label': patient_name.replace(" Patient", ""),
            'stats': stats
        })

    # Create comparison figure
    create_multi_patient_comparison(
        patient_data_list=patient_data_list,
        output_path="output/poster_figures/multi_patient_comparison.png",
        title="Segmentation Results Across Patients",
        dpi=300
    )


def create_custom_figure_example(ct_volume, masks_dict, slice_idx):
    """
    Example: Create a custom figure with specific layout.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    ct_slice = ct_volume[slice_idx]

    # Top-left: Raw CT
    axes[0, 0].imshow(ct_slice, cmap='gray', vmin=-200, vmax=1500)
    axes[0, 0].set_title('Raw CT', fontweight='bold')
    axes[0, 0].axis('off')

    # Top-right: Metal only
    axes[0, 1].imshow(ct_slice, cmap='gray', vmin=-200, vmax=1500)
    metal_overlay = np.ma.masked_where(~masks_dict['metal'][slice_idx],
                                       masks_dict['metal'][slice_idx])
    axes[0, 1].imshow(metal_overlay, cmap='Reds', alpha=0.7)
    axes[0, 1].set_title('Metal Detection', fontweight='bold')
    axes[0, 1].axis('off')

    # Bottom-left: Bone only
    axes[1, 0].imshow(ct_slice, cmap='gray', vmin=-200, vmax=1500)
    bone_overlay = np.ma.masked_where(~masks_dict['bone'][slice_idx],
                                      masks_dict['bone'][slice_idx])
    axes[1, 0].imshow(bone_overlay, cmap='Blues', alpha=0.5)
    axes[1, 0].set_title('Bone Segmentation', fontweight='bold')
    axes[1, 0].axis('off')

    # Bottom-right: All overlays
    axes[1, 1].imshow(ct_slice, cmap='gray', vmin=-200, vmax=1500)
    axes[1, 1].imshow(metal_overlay, cmap='Reds', alpha=0.7)
    axes[1, 1].imshow(bone_overlay, cmap='Blues', alpha=0.5)
    bright_overlay = np.ma.masked_where(~masks_dict['bright_artifacts'][slice_idx],
                                        masks_dict['bright_artifacts'][slice_idx])
    axes[1, 1].imshow(bright_overlay, cmap='YlOrBr', alpha=0.6)
    axes[1, 1].set_title('Final Segmentation', fontweight='bold')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('output/poster_figures/custom_figure.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Custom figure saved")


if __name__ == "__main__":
    # Example 1: Export all figures for a single patient
    export_single_patient_figures(
        patient_folder="data/HIP1 Patient",
        output_folder="output/poster_figures/patient_a"
    )

    # Example 2: Export just thumbnails for flowchart
    # ct_volume, spacing, _, _ = load_dicom_series("data/HIP1 Patient")
    # # ... generate masks_dict ...
    # slice_idx = 50
    # create_individual_thumbnails(
    #     ct_volume, masks_dict, slice_idx,
    #     output_dir="output/poster_figures/thumbnails",
    #     prefix="patient_a",
    #     dpi=150
    # )

    # Example 3: Create before/after comparison only
    # create_before_after_comparison(
    #     ct_volume, masks_dict, slice_idx,
    #     output_path="output/poster_figures/before_after_patient_a.png",
    #     patient_label="Patient A - Bilateral Hip Prostheses",
    #     dpi=300
    # )
