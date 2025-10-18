"""
Generate publication-quality figures for poster presentation.
Export pipeline visualizations, before/after comparisons, and multi-slice montages.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path


def create_pipeline_progression(ct_volume, masks_dict, slice_idx, output_path,
                                title="Segmentation Pipeline Progression", dpi=300):
    """
    Create a 4-panel figure showing pipeline progression.

    Args:
        ct_volume: 3D CT data in HU
        masks_dict: Dictionary with 'metal', 'bone', 'bright_artifacts', 'dark_artifacts'
        slice_idx: Slice index to visualize
        output_path: Path to save figure (PNG/JPG)
        title: Figure title
        dpi: Resolution for export
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Get the slice
    ct_slice = ct_volume[slice_idx]

    # Panel 1: Raw CT
    axes[0].imshow(ct_slice, cmap='gray', vmin=-200, vmax=1500)
    axes[0].set_title('1. Input: CT Scan', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Metal Detection
    axes[1].imshow(ct_slice, cmap='gray', vmin=-200, vmax=1500)
    if 'metal' in masks_dict:
        metal_overlay = np.ma.masked_where(~masks_dict['metal'][slice_idx],
                                           masks_dict['metal'][slice_idx])
        axes[1].imshow(metal_overlay, cmap='Reds', alpha=0.7, vmin=0, vmax=1)
    axes[1].set_title('2. Detect Metal', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Panel 3: Artifact Detection
    axes[2].imshow(ct_slice, cmap='gray', vmin=-200, vmax=1500)
    if 'metal' in masks_dict:
        metal_overlay = np.ma.masked_where(~masks_dict['metal'][slice_idx],
                                           masks_dict['metal'][slice_idx])
        axes[2].imshow(metal_overlay, cmap='Reds', alpha=0.7, vmin=0, vmax=1)
    if 'bright_artifacts' in masks_dict:
        bright_overlay = np.ma.masked_where(~masks_dict['bright_artifacts'][slice_idx],
                                            masks_dict['bright_artifacts'][slice_idx])
        axes[2].imshow(bright_overlay, cmap='YlOrBr', alpha=0.6, vmin=0, vmax=1)
    if 'dark_artifacts' in masks_dict:
        dark_overlay = np.ma.masked_where(~masks_dict['dark_artifacts'][slice_idx],
                                          masks_dict['dark_artifacts'][slice_idx])
        axes[2].imshow(dark_overlay, cmap='RdPu', alpha=0.6, vmin=0, vmax=1)
    axes[2].set_title('3. Find Artifacts', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    # Panel 4: Bone Discrimination
    axes[3].imshow(ct_slice, cmap='gray', vmin=-200, vmax=1500)
    if 'metal' in masks_dict:
        metal_overlay = np.ma.masked_where(~masks_dict['metal'][slice_idx],
                                           masks_dict['metal'][slice_idx])
        axes[3].imshow(metal_overlay, cmap='Reds', alpha=0.7, vmin=0, vmax=1)
    if 'bone' in masks_dict:
        bone_overlay = np.ma.masked_where(~masks_dict['bone'][slice_idx],
                                          masks_dict['bone'][slice_idx])
        axes[3].imshow(bone_overlay, cmap='Blues', alpha=0.5, vmin=0, vmax=1)
    if 'bright_artifacts' in masks_dict:
        bright_overlay = np.ma.masked_where(~masks_dict['bright_artifacts'][slice_idx],
                                            masks_dict['bright_artifacts'][slice_idx])
        axes[3].imshow(bright_overlay, cmap='YlOrBr', alpha=0.6, vmin=0, vmax=1)
    if 'dark_artifacts' in masks_dict:
        dark_overlay = np.ma.masked_where(~masks_dict['dark_artifacts'][slice_idx],
                                          masks_dict['dark_artifacts'][slice_idx])
        axes[3].imshow(dark_overlay, cmap='RdPu', alpha=0.6, vmin=0, vmax=1)
    axes[3].set_title('4. Final Segmentation', fontsize=14, fontweight='bold')
    axes[3].axis('off')

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved pipeline progression to {output_path}")


def create_before_after_comparison(ct_volume, masks_dict, slice_idx, output_path,
                                   patient_label="Patient", dpi=300):
    """
    Create a before/after comparison figure.

    Args:
        ct_volume: 3D CT data in HU
        masks_dict: Dictionary with segmentation masks
        slice_idx: Slice index to visualize
        output_path: Path to save figure
        patient_label: Label for the patient
        dpi: Resolution for export
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    ct_slice = ct_volume[slice_idx]

    # Before: Raw CT
    axes[0].imshow(ct_slice, cmap='gray', vmin=-200, vmax=1500)
    axes[0].set_title('BEFORE\n(Uncorrected)', fontsize=14, fontweight='bold')
    axes[0].text(0.5, -0.1, 'Artifacts obscure bone boundaries',
                ha='center', va='top', transform=axes[0].transAxes,
                fontsize=11, style='italic')
    axes[0].axis('off')

    # After: Full color overlay
    axes[1].imshow(ct_slice, cmap='gray', vmin=-200, vmax=1500)

    # Overlay all masks with custom colors
    if 'metal' in masks_dict:
        metal_overlay = np.ma.masked_where(~masks_dict['metal'][slice_idx],
                                           masks_dict['metal'][slice_idx])
        axes[1].imshow(metal_overlay, cmap='Reds', alpha=0.7, vmin=0, vmax=1)

    if 'bone' in masks_dict:
        bone_overlay = np.ma.masked_where(~masks_dict['bone'][slice_idx],
                                          masks_dict['bone'][slice_idx])
        axes[1].imshow(bone_overlay, cmap='Blues', alpha=0.5, vmin=0, vmax=1)

    if 'bright_artifacts' in masks_dict:
        bright_overlay = np.ma.masked_where(~masks_dict['bright_artifacts'][slice_idx],
                                            masks_dict['bright_artifacts'][slice_idx])
        axes[1].imshow(bright_overlay, cmap='YlOrBr', alpha=0.6, vmin=0, vmax=1)

    if 'dark_artifacts' in masks_dict:
        dark_overlay = np.ma.masked_where(~masks_dict['dark_artifacts'][slice_idx],
                                          masks_dict['dark_artifacts'][slice_idx])
        axes[1].imshow(dark_overlay, cmap='RdPu', alpha=0.6, vmin=0, vmax=1)

    axes[1].set_title('AFTER\n(Segmented)', fontsize=14, fontweight='bold')
    axes[1].text(0.5, -0.1, 'Tissue types clearly identified',
                ha='center', va='top', transform=axes[1].transAxes,
                fontsize=11, style='italic')
    axes[1].axis('off')

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='red', alpha=0.7, label='Metal'),
        mpatches.Patch(facecolor='blue', alpha=0.5, label='Bone'),
        mpatches.Patch(facecolor='yellow', alpha=0.6, label='Bright Artifacts'),
        mpatches.Patch(facecolor='magenta', alpha=0.6, label='Dark Artifacts')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
              fontsize=11, frameon=True, fancybox=True)

    plt.suptitle(f'{patient_label}', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved before/after comparison to {output_path}")


def create_multi_slice_montage(ct_volume, masks_dict, slice_indices, output_path,
                               title="Multi-Slice Segmentation Results", dpi=300):
    """
    Create a grid of multiple slices showing segmentation consistency.

    Args:
        ct_volume: 3D CT data in HU
        masks_dict: Dictionary with segmentation masks
        slice_indices: List of slice indices to show (e.g., [20, 25, 30, 35, 40, 45])
        output_path: Path to save figure
        title: Figure title
        dpi: Resolution for export
    """
    n_slices = len(slice_indices)
    n_cols = 3
    n_rows = (n_slices + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for idx, slice_idx in enumerate(slice_indices):
        ax = axes[idx]
        ct_slice = ct_volume[slice_idx]

        # Display CT
        ax.imshow(ct_slice, cmap='gray', vmin=-200, vmax=1500)

        # Overlay masks
        if 'metal' in masks_dict:
            metal_overlay = np.ma.masked_where(~masks_dict['metal'][slice_idx],
                                               masks_dict['metal'][slice_idx])
            ax.imshow(metal_overlay, cmap='Reds', alpha=0.7, vmin=0, vmax=1)

        if 'bone' in masks_dict:
            bone_overlay = np.ma.masked_where(~masks_dict['bone'][slice_idx],
                                              masks_dict['bone'][slice_idx])
            ax.imshow(bone_overlay, cmap='Blues', alpha=0.5, vmin=0, vmax=1)

        if 'bright_artifacts' in masks_dict:
            bright_overlay = np.ma.masked_where(~masks_dict['bright_artifacts'][slice_idx],
                                                masks_dict['bright_artifacts'][slice_idx])
            ax.imshow(bright_overlay, cmap='YlOrBr', alpha=0.6, vmin=0, vmax=1)

        if 'dark_artifacts' in masks_dict:
            dark_overlay = np.ma.masked_where(~masks_dict['dark_artifacts'][slice_idx],
                                              masks_dict['dark_artifacts'][slice_idx])
            ax.imshow(dark_overlay, cmap='RdPu', alpha=0.6, vmin=0, vmax=1)

        ax.set_title(f'Slice {slice_idx}', fontsize=12, fontweight='bold')
        ax.axis('off')

    # Hide extra subplots
    for idx in range(len(slice_indices), len(axes)):
        axes[idx].axis('off')

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='red', alpha=0.7, label='Metal'),
        mpatches.Patch(facecolor='blue', alpha=0.5, label='Bone'),
        mpatches.Patch(facecolor='yellow', alpha=0.6, label='Bright Artifacts'),
        mpatches.Patch(facecolor='magenta', alpha=0.6, label='Dark Artifacts')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
              fontsize=12, frameon=True, fancybox=True)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved multi-slice montage to {output_path}")


def create_multi_patient_comparison(patient_data_list, output_path,
                                   title="Multi-Patient Results", dpi=300):
    """
    Create a comparison of multiple patients side by side.

    Args:
        patient_data_list: List of dicts with keys:
            - 'ct_volume': 3D CT data
            - 'masks_dict': Segmentation masks
            - 'slice_idx': Slice to show
            - 'label': Patient label (e.g., "Patient A - Unilateral")
            - 'stats': Optional dict with volume stats
        output_path: Path to save figure
        title: Figure title
        dpi: Resolution for export
    """
    n_patients = len(patient_data_list)
    fig, axes = plt.subplots(1, n_patients, figsize=(6*n_patients, 6))
    axes = [axes] if n_patients == 1 else axes

    for idx, patient_data in enumerate(patient_data_list):
        ax = axes[idx]
        ct_volume = patient_data['ct_volume']
        masks_dict = patient_data['masks_dict']
        slice_idx = patient_data['slice_idx']
        label = patient_data.get('label', f'Patient {idx+1}')

        ct_slice = ct_volume[slice_idx]

        # Display CT
        ax.imshow(ct_slice, cmap='gray', vmin=-200, vmax=1500)

        # Overlay masks
        if 'metal' in masks_dict:
            metal_overlay = np.ma.masked_where(~masks_dict['metal'][slice_idx],
                                               masks_dict['metal'][slice_idx])
            ax.imshow(metal_overlay, cmap='Reds', alpha=0.7, vmin=0, vmax=1)

        if 'bone' in masks_dict:
            bone_overlay = np.ma.masked_where(~masks_dict['bone'][slice_idx],
                                              masks_dict['bone'][slice_idx])
            ax.imshow(bone_overlay, cmap='Blues', alpha=0.5, vmin=0, vmax=1)

        if 'bright_artifacts' in masks_dict:
            bright_overlay = np.ma.masked_where(~masks_dict['bright_artifacts'][slice_idx],
                                                masks_dict['bright_artifacts'][slice_idx])
            ax.imshow(bright_overlay, cmap='YlOrBr', alpha=0.6, vmin=0, vmax=1)

        if 'dark_artifacts' in masks_dict:
            dark_overlay = np.ma.masked_where(~masks_dict['dark_artifacts'][slice_idx],
                                              masks_dict['dark_artifacts'][slice_idx])
            ax.imshow(dark_overlay, cmap='RdPu', alpha=0.6, vmin=0, vmax=1)

        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.axis('off')

        # Add stats if provided
        if 'stats' in patient_data:
            stats = patient_data['stats']
            stats_text = '\n'.join([f"{k}: {v}" for k, v in stats.items()])
            ax.text(0.5, -0.05, stats_text, ha='center', va='top',
                   transform=ax.transAxes, fontsize=9, family='monospace')

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='red', alpha=0.7, label='Metal'),
        mpatches.Patch(facecolor='blue', alpha=0.5, label='Bone'),
        mpatches.Patch(facecolor='yellow', alpha=0.6, label='Bright Artifacts'),
        mpatches.Patch(facecolor='magenta', alpha=0.6, label='Dark Artifacts')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
              fontsize=11, frameon=True, fancybox=True)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved multi-patient comparison to {output_path}")


def create_individual_thumbnails(ct_volume, masks_dict, slice_idx, output_dir,
                                prefix="thumbnail", dpi=150):
    """
    Create individual thumbnail images for each pipeline step.
    Useful for inserting into flowcharts.

    Args:
        ct_volume: 3D CT data in HU
        masks_dict: Dictionary with segmentation masks
        slice_idx: Slice index to visualize
        output_dir: Directory to save thumbnails
        prefix: Filename prefix
        dpi: Resolution for export

    Returns:
        List of paths to saved thumbnails
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ct_slice = ct_volume[slice_idx]
    saved_paths = []

    # Thumbnail 1: Raw CT
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(ct_slice, cmap='gray', vmin=-200, vmax=1500)
    ax.axis('off')
    plt.tight_layout(pad=0)
    path1 = output_dir / f"{prefix}_1_input.png"
    plt.savefig(path1, dpi=dpi, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close()
    saved_paths.append(path1)

    # Thumbnail 2: Metal Detection
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(ct_slice, cmap='gray', vmin=-200, vmax=1500)
    if 'metal' in masks_dict:
        metal_overlay = np.ma.masked_where(~masks_dict['metal'][slice_idx],
                                           masks_dict['metal'][slice_idx])
        ax.imshow(metal_overlay, cmap='Reds', alpha=0.7, vmin=0, vmax=1)
    ax.axis('off')
    plt.tight_layout(pad=0)
    path2 = output_dir / f"{prefix}_2_metal.png"
    plt.savefig(path2, dpi=dpi, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close()
    saved_paths.append(path2)

    # Thumbnail 3: Artifacts
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(ct_slice, cmap='gray', vmin=-200, vmax=1500)
    if 'metal' in masks_dict:
        metal_overlay = np.ma.masked_where(~masks_dict['metal'][slice_idx],
                                           masks_dict['metal'][slice_idx])
        ax.imshow(metal_overlay, cmap='Reds', alpha=0.7, vmin=0, vmax=1)
    if 'bright_artifacts' in masks_dict:
        bright_overlay = np.ma.masked_where(~masks_dict['bright_artifacts'][slice_idx],
                                            masks_dict['bright_artifacts'][slice_idx])
        ax.imshow(bright_overlay, cmap='YlOrBr', alpha=0.6, vmin=0, vmax=1)
    if 'dark_artifacts' in masks_dict:
        dark_overlay = np.ma.masked_where(~masks_dict['dark_artifacts'][slice_idx],
                                          masks_dict['dark_artifacts'][slice_idx])
        ax.imshow(dark_overlay, cmap='RdPu', alpha=0.6, vmin=0, vmax=1)
    ax.axis('off')
    plt.tight_layout(pad=0)
    path3 = output_dir / f"{prefix}_3_artifacts.png"
    plt.savefig(path3, dpi=dpi, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close()
    saved_paths.append(path3)

    # Thumbnail 4: Bone Discrimination
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(ct_slice, cmap='gray', vmin=-200, vmax=1500)
    if 'metal' in masks_dict:
        metal_overlay = np.ma.masked_where(~masks_dict['metal'][slice_idx],
                                           masks_dict['metal'][slice_idx])
        ax.imshow(metal_overlay, cmap='Reds', alpha=0.7, vmin=0, vmax=1)
    if 'bone' in masks_dict:
        bone_overlay = np.ma.masked_where(~masks_dict['bone'][slice_idx],
                                          masks_dict['bone'][slice_idx])
        ax.imshow(bone_overlay, cmap='Blues', alpha=0.5, vmin=0, vmax=1)
    if 'bright_artifacts' in masks_dict:
        bright_overlay = np.ma.masked_where(~masks_dict['bright_artifacts'][slice_idx],
                                            masks_dict['bright_artifacts'][slice_idx])
        ax.imshow(bright_overlay, cmap='YlOrBr', alpha=0.6, vmin=0, vmax=1)
    ax.axis('off')
    plt.tight_layout(pad=0)
    path4 = output_dir / f"{prefix}_4_bone_separated.png"
    plt.savefig(path4, dpi=dpi, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close()
    saved_paths.append(path4)

    # Thumbnail 5: Final Color Map
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(ct_slice, cmap='gray', vmin=-200, vmax=1500)
    if 'metal' in masks_dict:
        metal_overlay = np.ma.masked_where(~masks_dict['metal'][slice_idx],
                                           masks_dict['metal'][slice_idx])
        ax.imshow(metal_overlay, cmap='Reds', alpha=0.7, vmin=0, vmax=1)
    if 'bone' in masks_dict:
        bone_overlay = np.ma.masked_where(~masks_dict['bone'][slice_idx],
                                          masks_dict['bone'][slice_idx])
        ax.imshow(bone_overlay, cmap='Blues', alpha=0.5, vmin=0, vmax=1)
    if 'bright_artifacts' in masks_dict:
        bright_overlay = np.ma.masked_where(~masks_dict['bright_artifacts'][slice_idx],
                                            masks_dict['bright_artifacts'][slice_idx])
        ax.imshow(bright_overlay, cmap='YlOrBr', alpha=0.6, vmin=0, vmax=1)
    if 'dark_artifacts' in masks_dict:
        dark_overlay = np.ma.masked_where(~masks_dict['dark_artifacts'][slice_idx],
                                          masks_dict['dark_artifacts'][slice_idx])
        ax.imshow(dark_overlay, cmap='RdPu', alpha=0.6, vmin=0, vmax=1)
    ax.axis('off')
    plt.tight_layout(pad=0)
    path5 = output_dir / f"{prefix}_5_final.png"
    plt.savefig(path5, dpi=dpi, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close()
    saved_paths.append(path5)

    print(f"Saved {len(saved_paths)} thumbnails to {output_dir}")
    return saved_paths


def export_all_poster_figures(ct_volume, masks_dict, slice_idx, output_dir,
                              patient_label="Patient", additional_slices=None):
    """
    Convenience function to export all poster figures at once.

    Args:
        ct_volume: 3D CT data in HU
        masks_dict: Dictionary with segmentation masks
        slice_idx: Primary slice index
        output_dir: Directory to save all figures
        patient_label: Label for the patient
        additional_slices: List of additional slice indices for montage
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting all poster figures to {output_dir}...")

    # Pipeline progression
    create_pipeline_progression(
        ct_volume, masks_dict, slice_idx,
        output_dir / f"{patient_label}_pipeline_progression.png"
    )

    # Before/After comparison
    create_before_after_comparison(
        ct_volume, masks_dict, slice_idx,
        output_dir / f"{patient_label}_before_after.png",
        patient_label=patient_label
    )

    # Individual thumbnails
    create_individual_thumbnails(
        ct_volume, masks_dict, slice_idx,
        output_dir / "thumbnails",
        prefix=patient_label.lower().replace(" ", "_")
    )

    # Multi-slice montage if additional slices provided
    if additional_slices:
        all_slices = [slice_idx] + additional_slices
        create_multi_slice_montage(
            ct_volume, masks_dict, all_slices,
            output_dir / f"{patient_label}_multi_slice.png",
            title=f"{patient_label} - Multi-Slice Results"
        )

    print(f"✓ All figures exported successfully!")
