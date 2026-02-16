import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from PIL import Image
import io
import base64


def fast_render_slice(ct_slice, window_center=50, window_width=400):
    """
    Render a CT slice to PNG bytes using numpy windowing + PIL.
    Much faster than matplotlib (~10ms vs ~300ms).

    Args:
        ct_slice: 2D numpy array of HU values
        window_center: center of display window (HU)
        window_width: width of display window (HU)

    Returns:
        bytes: PNG image data ready for st.image()
    """
    # Apply windowing (convert HU to 0-255 grayscale)
    vmin = window_center - window_width / 2
    vmax = window_center + window_width / 2
    windowed = np.clip((ct_slice - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)

    # Convert to PIL Image and encode as PNG
    img = Image.fromarray(windowed, mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf.getvalue()


def create_overlay_image(ct_slice, masks, roi_boundaries=None, slice_index=None, individual_regions=None, custom_names=None, spacing=None):
    """
    Create an overlay visualization of CT slice with segmented regions.
    
    Args:
        ct_slice: 2D numpy array of CT data
        masks: dict containing masks for different tissue types
        roi_boundaries: tuple of (r_min, r_max, c_min, c_max) for ROI (legacy)
        slice_index: optional slice number for title
        individual_regions: list of individual ROI regions for this slice
        custom_names: dict of custom names for each mask type
        spacing: tuple of (z,y,x) pixel spacing in mm
        
    Returns:
        matplotlib figure object
    """
    fig = plt.figure(figsize=(10, 10), facecolor='black')
    ax = fig.add_axes([0, 0, 1, 1])  # fill entire figure, no padding

    # Display base CT image
    ax.imshow(ct_slice, cmap='gray', vmin=-150, vmax=250)
    ax.axis('off')

    # Scale bar disabled for now
    # if spacing is not None:
    #     # spacing is (z, y, x) in mm/pixel. Use y for row spacing.
    #     pixel_spacing_y = spacing[1]
    #     scale_bar_length_pixels = 10 / pixel_spacing_y # 10mm = 1cm
    #
    #     # Position in the bottom-left area.
    #     # Y position is for the text, to have the bar above it.
    #     y_pos = ct_slice.shape[0] * 0.85
    #     x_pos = ct_slice.shape[1] * 0.1
    #
    #     # Draw the text "1 cm"
    #     ax.text(x_pos + scale_bar_length_pixels / 2, y_pos, '1 cm', color='white',
    #             ha='center', va='center', fontsize=10)
    #
    #     # Draw the scale bar above the text
    #     bar_y_pos = y_pos + 10
    #     bar_height = 4
    #     ax.add_patch(Rectangle((x_pos, bar_y_pos), scale_bar_length_pixels, bar_height, color='white'))

    # Define colors for each category - high contrast for visibility
    colors = {
        'metal': (1.0, 0.0, 0.0, 0.9),                    # Bright Red
        'bright_artifacts': (1.0, 1.0, 0.0, 0.8),         # Bright Yellow (legacy/general)
        'bright_artifact_bone': (1.0, 0.5, 0.0, 0.8),     # Bright Orange (bright artifacts over bone)
        'bright_artifact_tissue': (0.0, 1.0, 0.0, 0.8),   # Bright Green (bright artifacts over tissue)
        'bright_artifacts_mild': (0.5, 1.0, 0.5, 0.8),    # Light Green (mild elevation)
        'bright_artifacts_moderate': (1.0, 1.0, 0.0, 0.8), # Bright Yellow (moderate elevation)
        'bright_artifacts_severe': (1.0, 0.5, 0.0, 0.8),  # Orange-Red (severe elevation)
        'dark_artifacts': (1.0, 0.0, 1.0, 0.8),           # Bright Magenta (legacy/general)
        'dark_artifact_bone': (0.8, 0.0, 0.8, 0.8),       # Dark Magenta (dark artifacts over bone)
        'dark_artifact_tissue': (1.0, 0.4, 1.0, 0.8),     # Light Magenta (dark artifacts over tissue)
        'bone': (0.0, 0.8, 1.0, 0.8)                      # Bright Cyan (distinct from others)
    }
    
    # Overlay masks
    for mask_name, mask in masks.items():
        if mask_name in colors and np.any(mask):
            overlay = np.zeros((*ct_slice.shape, 4))
            overlay[mask] = colors[mask_name]
            ax.imshow(overlay)

    # Draw individual ROI regions if provided (preferred)
    roi_drawn = False
    if individual_regions and len(individual_regions) > 0:
        for i, region in enumerate(individual_regions):
            r_min = int(region['y_min'])
            r_max = int(region['y_max'])
            c_min = int(region['x_min'])
            c_max = int(region['x_max'])
            
            # Use different linestyle for multiple regions
            label = 'Auto ROI' if i == 0 else None  # Only label the first one
            ax.add_patch(plt.Rectangle((c_min, r_min), c_max-c_min, r_max-r_min,
                                      edgecolor='lime', facecolor='none', linewidth=2, 
                                      linestyle='--', label=label))
        roi_drawn = True
    
    # Fall back to legacy ROI boundary if no individual regions provided or empty
    elif roi_boundaries:
        # Handle both tuple and dict formats
        if isinstance(roi_boundaries, dict):
            r_min = int(roi_boundaries.get('y_min', 0))
            r_max = int(roi_boundaries.get('y_max', 0))
            c_min = int(roi_boundaries.get('x_min', 0))
            c_max = int(roi_boundaries.get('x_max', 0))
        else:
            r_min, r_max, c_min, c_max = roi_boundaries
            # Ensure values are integers to avoid string subtraction errors
            r_min, r_max = int(r_min), int(r_max)
            c_min, c_max = int(c_min), int(c_max)
        ax.add_patch(plt.Rectangle((c_min, r_min), c_max-c_min, r_max-r_min,
                                  edgecolor='lime', facecolor='none', linewidth=2, 
                                  linestyle='--', label='Auto ROI'))
        roi_drawn = True
    
    return fig


def create_histogram(hu_values, region_name, color='blue'):
    """
    Create a histogram for HU values of a specific region.
    
    Args:
        hu_values: numpy array of HU values
        region_name: name of the region for title
        color: color for histogram bars
        
    Returns:
        matplotlib figure object
    """
    if hu_values.size == 0:
        return None
    
    fig = plt.figure(figsize=(7, 5))
    
    # Define bin range
    min_hu_plot = -1024
    max_hu_plot = 5000
    bins = np.linspace(min_hu_plot, max_hu_plot, 100)
    
    plt.hist(hu_values, bins=bins, color=color, edgecolor='black', alpha=0.7)
    plt.title(f"IVH for {region_name}", fontsize=14)
    plt.xlabel('Hounsfield Unit (HU)', fontsize=12)
    plt.ylabel('Frequency (Number of Pixels)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    return fig


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for web display."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def fig_to_png_bytes(fig, dpi=300):
    """
    Convert matplotlib figure to PNG bytes for high-quality export.

    Args:
        fig: matplotlib figure object
        dpi: resolution for export (default 300 for publication quality)

    Returns:
        bytes: PNG image data
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    return buf.getvalue()


def visualize_star_profiles(ct_slice, profiles, center_coords, roi_bounds, thresholds=None):
    """
    Visualize the star profile lines on the CT slice and HU vs distance curves.
    
    Args:
        ct_slice: 2D CT slice
        profiles: List of (distances, hu_values) tuples
        center_coords: (y, x) center coordinates
        roi_bounds: Dictionary with ROI boundaries
        thresholds: Tuple of (lower, upper) threshold values
        
    Returns:
        matplotlib figure with subplots
    """
    fig = plt.figure(figsize=(15, 6))
    
    # Left: CT slice with star lines
    ax1 = plt.subplot(1, 3, (1, 2))
    ax1.imshow(ct_slice, cmap='gray', vmin=-150, vmax=250)
    
    # Draw ROI box
    roi_rect = Rectangle(
        (roi_bounds['x_min'], roi_bounds['y_min']),
        roi_bounds['x_max'] - roi_bounds['x_min'],
        roi_bounds['y_max'] - roi_bounds['y_min'],
        fill=False, edgecolor='lime', linewidth=2, linestyle='--'
    )
    ax1.add_patch(roi_rect)
    
    # Draw center point
    ax1.plot(center_coords[1], center_coords[0], 'r+', markersize=15, markeredgewidth=3)
    
    # Draw all 16 star lines
    y_min, y_max = roi_bounds['y_min'], roi_bounds['y_max']
    x_min, x_max = roi_bounds['x_min'], roi_bounds['x_max']
    
    # Calculate intermediate points for 16-point star
    y_mid = (y_min + y_max) // 2
    x_mid = (x_min + x_max) // 2
    
    y_q1 = (y_min + y_mid) // 2
    y_q3 = (y_mid + y_max) // 2
    x_q1 = (x_min + x_mid) // 2
    x_q3 = (x_mid + x_max) // 2
    
    # Define all 16 endpoints (same as in metal detection)
    endpoints = [
        # Cardinals (N, S, E, W)
        (y_min, x_mid), (y_max, x_mid), (y_mid, x_max), (y_mid, x_min),
        # Primary diagonals
        (y_min, x_min), (y_min, x_max), (y_max, x_min), (y_max, x_max),
        # Secondary points
        (y_min, x_q1), (y_min, x_q3),
        (y_max, x_q1), (y_max, x_q3),
        (y_q1, x_min), (y_q3, x_min),
        (y_q1, x_max), (y_q3, x_max)
    ]
    
    # Draw all 16 lines
    for end_y, end_x in endpoints:
        # Plot from center to endpoint (note: x is second coordinate, y is first)
        ax1.plot([center_coords[1], end_x], [center_coords[0], end_y], 
                'y-', linewidth=1, alpha=0.7)
    
    ax1.set_title('Star Profile Analysis')
    ax1.axis('off')
    
    # Right: HU vs Distance curves
    ax2 = plt.subplot(1, 3, 3)
    
    # Plot all profiles
    for i, (distances, hu_values) in enumerate(profiles):
        ax2.plot(distances, hu_values, alpha=0.3, color='gray')
    
    # Highlight one representative profile
    if profiles:
        dist, hu = profiles[0]
        ax2.plot(dist, hu, 'b-', linewidth=2, label='Sample Profile')
    
    # Show thresholds if available
    if thresholds:
        ax2.axhline(y=thresholds[0], color='r', linestyle='--', 
                   label=f'Lower: {thresholds[0]:.0f} HU')
        ax2.axhline(y=thresholds[1], color='r', linestyle='-', 
                   label=f'Upper: {thresholds[1]:.0f} HU')
        
        # Show 75% line
        if profiles:
            max_hu = np.max([np.max(p[1]) for p in profiles])
            ax2.axhline(y=0.75*max_hu, color='orange', linestyle=':', 
                       label=f'75% Max: {0.75*max_hu:.0f} HU')
    
    ax2.set_xlabel('Distance from Center (pixels)')
    ax2.set_ylabel('HU Value')
    ax2.set_title('HU vs Distance Profiles')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1000, 5000)
    
    plt.tight_layout()
    return fig


def create_multi_slice_view(ct_volume, masks_dict, slice_indices, roi_bounds=None, individual_regions=None, valid_roi_slices=None):
    """
    Create a grid view of multiple slices with contour overlays.
    
    Args:
        ct_volume: 3D CT data
        masks_dict: Dictionary of mask arrays
        slice_indices: List of slice indices to display
        roi_bounds: Optional ROI boundaries (legacy)
        individual_regions: Dict of individual ROI regions per slice
        
    Returns:
        matplotlib figure
    """
    n_slices = len(slice_indices)
    n_cols = min(4, n_slices)
    n_rows = (n_slices + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(4*n_cols, 4*n_rows))
    
    colors = {
        'metal': (1.0, 0.0, 0.0, 0.9),
        'bright_artifacts': (1.0, 1.0, 0.0, 0.8),
        'dark_artifacts': (1.0, 0.0, 1.0, 0.8),
        'bone': (0.0, 0.8, 1.0, 0.8)
    }
    
    for idx, slice_idx in enumerate(slice_indices):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        
        # Display CT slice
        ax.imshow(ct_volume[slice_idx], cmap='gray', vmin=-150, vmax=250)
        
        # Overlay masks
        for mask_name, mask in masks_dict.items():
            if mask_name in colors and isinstance(mask, np.ndarray):
                if mask.ndim == 3 and np.any(mask[slice_idx]):
                    overlay = np.zeros((*ct_volume[slice_idx].shape, 4))
                    overlay[mask[slice_idx]] = colors[mask_name]
                    ax.imshow(overlay)
        
        # Draw individual ROI regions if provided (preferred)
        if individual_regions and slice_idx in individual_regions:
            for region in individual_regions[slice_idx]:
                roi_rect = Rectangle(
                    (region['x_min'], region['y_min']),
                    region['x_max'] - region['x_min'],
                    region['y_max'] - region['y_min'],
                    fill=False, edgecolor='lime', linewidth=1, linestyle='--'
                )
                ax.add_patch(roi_rect)
        # Fall back to legacy ROI bounds - but only show on valid slices
        elif roi_bounds and (valid_roi_slices is None or slice_idx in valid_roi_slices):
            roi_rect = Rectangle(
                (roi_bounds['x_min'], roi_bounds['y_min']),
                roi_bounds['x_max'] - roi_bounds['x_min'],
                roi_bounds['y_max'] - roi_bounds['y_min'],
                fill=False, edgecolor='lime', linewidth=1, linestyle='--'
            )
            ax.add_patch(roi_rect)
        
        ax.set_title(f'Slice {slice_idx}')
        ax.axis('off')
    
    # Add legend
    if n_slices > 0:
        legend_elements = [
            Patch(facecolor=colors['metal'], label='Metal'),
            Patch(facecolor=colors['bright_artifacts'], label='Bright'),
            Patch(facecolor=colors['dark_artifacts'], label='Dark'),
            Patch(facecolor=colors['bone'], label='Bone'),
            Line2D([0], [0], color='lime', linestyle='--', label='ROI')
        ]
        fig.legend(handles=legend_elements, loc='upper right', 
                  bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    return fig


def plot_threshold_evolution(slice_thresholds):
    """
    Plot how thresholds change across slices.
    
    Args:
        slice_thresholds: List of dictionaries with slice and threshold info
        
    Returns:
        matplotlib figure
    """
    fig = plt.figure(figsize=(10, 6))
    
    slices = []
    lower_thresholds = []
    upper_thresholds = []
    
    for item in slice_thresholds:
        if item['thresholds']:
            slices.append(item['slice'])
            lower_thresholds.append(item['thresholds'][0])
            upper_thresholds.append(item['thresholds'][1])
    
    if slices:
        plt.plot(slices, lower_thresholds, 'b-', label='Lower Threshold', marker='o')
        plt.plot(slices, upper_thresholds, 'r-', label='Upper Threshold', marker='s')
        
        plt.fill_between(slices, lower_thresholds, upper_thresholds, 
                        alpha=0.3, color='gray', label='Metal HU Range')
    
    plt.xlabel('Slice Number')
    plt.ylabel('HU Value')
    plt.title('Adaptive Threshold Values Across Slices')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


def visualize_discrimination_slice(ct_slice, bone_mask, artifact_mask, confidence_map, slice_idx):
    """
    Visualize the bone vs artifact discrimination results for a single slice.
    
    Args:
        ct_slice: 2D CT slice
        bone_mask: 2D bone mask
        artifact_mask: 2D bright artifact mask
        confidence_map: 2D confidence map
        slice_idx: Slice index
        
    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # CT with overlay
    ax = axes[0, 0]
    ax.imshow(ct_slice, cmap='gray', vmin=-150, vmax=250)
    
    # Create overlay
    overlay = np.zeros((*ct_slice.shape, 4))
    if np.any(bone_mask):
        overlay[bone_mask] = [0, 0.2, 0.8, 0.5]  # Blue for bone
    if np.any(artifact_mask):
        overlay[artifact_mask] = [1, 0.7, 0, 0.5]  # Orange for artifacts
    
    ax.imshow(overlay)
    ax.set_title(f'CT Slice {slice_idx} with Discrimination')
    ax.axis('off')
    
    # Bone mask
    ax = axes[0, 1]
    ax.imshow(bone_mask, cmap='Blues')
    ax.set_title(f'Bone (n={np.sum(bone_mask):,} pixels)')
    ax.axis('off')
    
    # Artifact mask
    ax = axes[1, 0]
    ax.imshow(artifact_mask, cmap='Oranges')
    ax.set_title(f'Bright Artifacts (n={np.sum(artifact_mask):,} pixels)')
    ax.axis('off')
    
    # Confidence map
    ax = axes[1, 1]
    confident_regions = confidence_map > 0
    if np.any(confident_regions):
        im = ax.imshow(confidence_map, cmap='viridis', vmin=0, vmax=1)
        avg_conf = np.mean(confidence_map[confident_regions])
        ax.set_title(f'Discrimination Confidence (avg={avg_conf:.2%})')
        plt.colorbar(im, ax=ax, fraction=0.046)
    else:
        ax.text(0.5, 0.5, 'No discrimination performed', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Discrimination Confidence')
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def create_histogram_with_thresholds(ct_volume, thresholds, method='russian_doll', slice_index=None):
    """
    Create a histogram of HU values with threshold overlays for real-time preview.
    
    Args:
        ct_volume: 3D numpy array of CT data in HU or 2D slice
        thresholds: dict containing threshold values
        method: 'russian_doll' or 'legacy' to determine which thresholds to show
        slice_index: if provided, only use that slice; otherwise use whole volume
        
    Returns:
        matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get HU values
    if slice_index is not None and len(ct_volume.shape) == 3:
        hu_values = ct_volume[slice_index].flatten()
        title = f"HU Distribution - Slice {slice_index}"
    elif len(ct_volume.shape) == 2:
        hu_values = ct_volume.flatten()
        title = "HU Distribution - Current Slice"
    else:
        # Sample every 10th voxel for performance
        hu_values = ct_volume[::10, ::10, ::10].flatten()
        title = "HU Distribution - Full Volume (Sampled)"
    
    # Filter out air values for better visualization
    hu_values = hu_values[hu_values > -1000]
    
    # Create histogram
    counts, bins, patches = ax.hist(hu_values, bins=100, alpha=0.7, color='gray', edgecolor='none')
    
    # Get threshold values based on method
    if method == 'russian_doll':
        dark_min = thresholds['russian_doll']['dark_min']
        dark_max = thresholds['russian_doll']['dark_max']
        bright_min = thresholds['russian_doll']['bright_min']
        bright_max = thresholds['russian_doll']['bright_max']
        bone_min = thresholds['russian_doll'].get('bone_min', bright_min)
        bone_max = thresholds['russian_doll'].get('bone_max', bright_max)
    else:  # legacy
        dark_min = -1024
        dark_max = thresholds['legacy']['dark_max']
        bright_min = thresholds['legacy']['bright_min']
        bright_max = thresholds['legacy']['bright_max']
        bone_min = thresholds['legacy']['bone_min']
        bone_max = thresholds['legacy']['bone_max']
    
    # Add metal threshold if available
    metal_threshold = thresholds.get('metal_detection', {}).get('metal_threshold', 2500)
    
    # Add vertical lines for thresholds
    ax.axvspan(dark_min, dark_max, alpha=0.3, color='magenta', label=f'Dark Artifacts [{dark_min:.0f}, {dark_max:.0f}]')
    ax.axvspan(bright_min, bright_max, alpha=0.3, color='yellow', label=f'Bright Artifacts [{bright_min:.0f}, {bright_max:.0f}]')
    ax.axvspan(bone_min, bone_max, alpha=0.2, color='blue', label=f'Bone [{bone_min:.0f}, {bone_max:.0f}]')
    ax.axvline(x=metal_threshold, color='red', linestyle='--', linewidth=2, label=f'Metal Threshold ({metal_threshold:.0f})')
    
    # Color histogram bars based on thresholds
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        if dark_min <= bin_center <= dark_max:
            patch.set_facecolor('magenta')
            patch.set_alpha(0.7)
        elif bright_min <= bin_center <= bright_max:
            if bone_min <= bin_center <= bone_max:
                patch.set_facecolor('blue')
                patch.set_alpha(0.6)
            else:
                patch.set_facecolor('yellow')
                patch.set_alpha(0.7)
        elif bin_center >= metal_threshold:
            patch.set_facecolor('red')
            patch.set_alpha(0.8)
    
    ax.set_xlabel('Hounsfield Units (HU)')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.set_xlim(-500, 3500)
    ax.set_yscale('log')  # Log scale for better visualization
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig


def create_threshold_preview(ct_slice, thresholds, method='russian_doll'):
    """
    Create a preview of how thresholds will segment the current slice.
    
    Args:
        ct_slice: 2D numpy array of CT data in HU
        thresholds: dict containing threshold values
        method: 'russian_doll' or 'legacy'
        
    Returns:
        matplotlib figure object showing segmentation preview
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Get threshold values
    if method == 'russian_doll':
        dark_min = thresholds['russian_doll']['dark_min']
        dark_max = thresholds['russian_doll']['dark_max']
        bright_min = thresholds['russian_doll']['bright_min']
        bright_max = thresholds['russian_doll']['bright_max']
    else:
        dark_min = -1024
        dark_max = thresholds['legacy']['dark_max']
        bright_min = thresholds['legacy']['bright_min']
        bright_max = thresholds['legacy']['bright_max']
    
    metal_threshold = thresholds.get('metal_detection', {}).get('metal_threshold', 2500)
    
    # Original slice
    axes[0].imshow(ct_slice, cmap='gray', vmin=-150, vmax=250)
    axes[0].set_title('Original CT Slice')
    axes[0].axis('off')
    
    # Threshold preview
    preview = np.zeros((*ct_slice.shape, 3))
    
    # Apply thresholds
    dark_mask = (ct_slice >= dark_min) & (ct_slice <= dark_max)
    bright_mask = (ct_slice >= bright_min) & (ct_slice <= bright_max)
    metal_mask = ct_slice >= metal_threshold
    
    # Color the regions
    preview[dark_mask] = [1.0, 0.0, 1.0]  # Magenta for dark
    preview[bright_mask] = [1.0, 1.0, 0.0]  # Yellow for bright
    preview[metal_mask] = [1.0, 0.0, 0.0]  # Red for metal
    
    axes[1].imshow(preview)
    axes[1].set_title('Threshold Preview')
    axes[1].axis('off')
    
    # Overlay on original
    axes[2].imshow(ct_slice, cmap='gray', vmin=-150, vmax=250)
    overlay = np.zeros((*ct_slice.shape, 4))
    overlay[dark_mask] = [1.0, 0.0, 1.0, 0.5]
    overlay[bright_mask] = [1.0, 1.0, 0.0, 0.5]
    overlay[metal_mask] = [1.0, 0.0, 0.0, 0.7]
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay on Original')
    axes[2].axis('off')
    
    plt.suptitle('Real-time Threshold Preview')
    plt.tight_layout()
    return fig