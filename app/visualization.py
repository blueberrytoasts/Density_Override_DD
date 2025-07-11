import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
import io
import base64


def create_overlay_image(ct_slice, masks, roi_boundaries=None, slice_index=None, individual_regions=None, custom_names=None):
    """
    Create an overlay visualization of CT slice with segmented regions.
    
    Args:
        ct_slice: 2D numpy array of CT data
        masks: dict containing masks for different tissue types
        roi_boundaries: tuple of (r_min, r_max, c_min, c_max) for ROI (legacy)
        slice_index: optional slice number for title
        individual_regions: list of individual ROI regions for this slice
        custom_names: dict of custom names for each mask type
        
    Returns:
        matplotlib figure object
    """
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    # Display base CT image
    ax.imshow(ct_slice, cmap='gray', vmin=-150, vmax=250)
    
    title = f"CT Slice {slice_index} with Characterized Regions" if slice_index else "CT Slice with Characterized Regions"
    ax.set_title(title)
    ax.axis('off')

    # Define colors for each category
    colors = {
        'metal': (1.0, 0.0, 0.0, 0.7),          # Red
        'bright_artifacts': (1.0, 1.0, 0.0, 0.6), # Yellow
        'dark_artifacts': (1.0, 0.0, 1.0, 0.6),   # Magenta
        'bone': (0.0, 0.2, 0.8, 0.5)             # Blue
    }
    
    # Overlay masks
    for mask_name, mask in masks.items():
        if mask_name in colors and np.any(mask):
            overlay = np.zeros((*ct_slice.shape, 4))
            overlay[mask] = colors[mask_name]
            ax.imshow(overlay)

    # Draw individual ROI regions if provided (preferred)
    roi_drawn = False
    if individual_regions:
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
    
    # Fall back to legacy ROI boundary if no individual regions provided
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
    
    # Create legend using custom names if provided
    if custom_names is None:
        custom_names = {
            'metal': 'Metal Implant',
            'bright_artifacts': 'Bright Artifacts',
            'bone': 'Bone',
            'dark_artifacts': 'Dark Artifacts'
        }
    
    # Only add legend items for masks that are being displayed
    legend_elements = []
    for mask_name in masks.keys():
        if mask_name in colors:
            label = custom_names.get(mask_name, mask_name.replace('_', ' ').title())
            legend_elements.append(
                Patch(facecolor=colors[mask_name], edgecolor='black', label=label)
            )
    if roi_drawn:
        legend_elements.append(Patch(facecolor='none', edgecolor='lime', lw=2, ls='--', label='Auto ROI'))
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
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


def create_slice_preview(ct_slice, slice_index):
    """Create a simple preview of a CT slice."""
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(ct_slice, cmap='gray', vmin=-150, vmax=250)
    plt.title(f"CT Slice {slice_index}")
    plt.axis('off')
    return fig


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


def create_multi_slice_view(ct_volume, masks_dict, slice_indices, roi_bounds=None, individual_regions=None):
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
        'metal': (1.0, 0.0, 0.0, 0.7),
        'bright_artifacts': (1.0, 1.0, 0.0, 0.6),
        'dark_artifacts': (1.0, 0.0, 1.0, 0.6),
        'bone': (0.0, 0.2, 0.8, 0.5)
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
        # Fall back to legacy ROI bounds
        elif roi_bounds and roi_bounds['z_min'] <= slice_idx < roi_bounds['z_max']:
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


def visualize_edge_analysis(ct_slice, edge_features, slice_idx):
    """
    Visualize the edge-based analysis features for enhanced discrimination.
    
    Args:
        ct_slice: 2D CT slice
        edge_features: Dictionary containing edge analysis results:
            - coherence_map: Edge coherence scores
            - grad_mag: Gradient magnitude
            - sharp_edges: Sharp edge locations
            - radial_alignment: Radial alignment scores
            - persistent_edges: Multi-scale persistent edges
            - continuity_score: 3D continuity scores
        slice_idx: Slice index
        
    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original CT slice
    ax = axes[0, 0]
    ax.imshow(ct_slice, cmap='gray', vmin=-150, vmax=250)
    ax.set_title(f'CT Slice {slice_idx}')
    ax.axis('off')
    
    # Edge coherence
    ax = axes[0, 1]
    if 'coherence_map' in edge_features and np.any(edge_features['coherence_map'] > 0):
        im = ax.imshow(edge_features['coherence_map'], cmap='jet', vmin=0, vmax=1)
        ax.set_title('Edge Coherence\n(High=Bone-like)')
        plt.colorbar(im, ax=ax, fraction=0.046)
    else:
        ax.text(0.5, 0.5, 'No coherence data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Edge Coherence')
    ax.axis('off')
    
    # Gradient magnitude
    ax = axes[0, 2]
    if 'grad_mag' in edge_features:
        im = ax.imshow(edge_features['grad_mag'], cmap='hot')
        ax.set_title('Gradient Magnitude\n(Edge Strength)')
        plt.colorbar(im, ax=ax, fraction=0.046)
    else:
        ax.text(0.5, 0.5, 'No gradient data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Gradient Magnitude')
    ax.axis('off')
    
    # Sharp edges
    ax = axes[1, 0]
    if 'sharp_edges' in edge_features:
        ax.imshow(edge_features['sharp_edges'], cmap='binary')
        ax.set_title('Sharp Edges\n(Bone Characteristic)')
    else:
        ax.text(0.5, 0.5, 'No sharp edge data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Sharp Edges')
    ax.axis('off')
    
    # Radial alignment
    ax = axes[1, 1]
    if 'radial_alignment' in edge_features and np.any(edge_features['radial_alignment'] > 0):
        im = ax.imshow(edge_features['radial_alignment'], cmap='RdYlBu_r', vmin=0, vmax=1)
        ax.set_title('Radial Alignment\n(High=Artifact-like)')
        plt.colorbar(im, ax=ax, fraction=0.046)
    else:
        ax.text(0.5, 0.5, 'No radial data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Radial Alignment')
    ax.axis('off')
    
    # 3D continuity
    ax = axes[1, 2]
    if 'continuity_score' in edge_features and np.any(edge_features['continuity_score'] > 0):
        im = ax.imshow(edge_features['continuity_score'], cmap='viridis', vmin=0, vmax=1)
        ax.set_title('3D Continuity\n(High=Bone-like)')
        plt.colorbar(im, ax=ax, fraction=0.046)
    else:
        ax.text(0.5, 0.5, 'No continuity data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('3D Continuity')
    ax.axis('off')
    
    plt.tight_layout()
    return fig