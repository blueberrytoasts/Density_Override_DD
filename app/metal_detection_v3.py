import numpy as np
from scipy.ndimage import label, center_of_mass, binary_dilation, binary_erosion
from scipy.signal import find_peaks
from skimage.draw import line
import nibabel as nib


def analyze_3d_metal_distribution(ct_volume, percentile=99.5):
    """
    Analyze the 3D distribution of high-intensity regions to understand
    the full extent of metal implants across all anatomical planes.
    
    Args:
        ct_volume: 3D numpy array of CT data in HU
        percentile: Percentile for initial high-intensity detection
        
    Returns:
        dict: Contains 3D analysis results and bounding information
    """
    # Use a much higher threshold to focus on actual metal, not bone
    # Start with definitive metal-level intensities
    threshold = max(np.percentile(ct_volume, percentile), 2500)
    print(f"Using metal-focused threshold: {threshold:.0f} HU (min 2500 HU)")
    
    # Create high-intensity mask
    high_intensity_mask = ct_volume > threshold
    
    if not np.any(high_intensity_mask):
        # Fallback to lower threshold but still well above bone
        threshold = max(np.percentile(ct_volume, 99.0), 2000)
        high_intensity_mask = ct_volume > threshold
        print(f"Fallback to threshold: {threshold:.0f} HU")
        
    if not np.any(high_intensity_mask):
        return None
    
    # Focus on the largest connected components (actual implants)
    labeled_array, num_features = label(high_intensity_mask)
    
    if num_features == 0:
        return None
    
    # Find the largest components (likely the actual implants)
    component_sizes = []
    for i in range(1, num_features + 1):
        size = np.sum(labeled_array == i)
        if size > 100:  # Minimum meaningful size
            component_sizes.append((i, size))
    
    if not component_sizes:
        return None
    
    # Keep only the largest component (actual implant) to avoid bilateral detection
    component_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Only keep the largest component, and only if it's significantly larger than others
    if len(component_sizes) > 1:
        largest_size = component_sizes[0][1]
        second_size = component_sizes[1][1]
        
        # If the largest is at least 3x bigger than the second, only keep the largest
        if largest_size >= 3 * second_size:
            keep_components = component_sizes[:1]
        else:
            # If sizes are similar, might be bilateral implants - keep top 2 max
            keep_components = component_sizes[:2]
    else:
        keep_components = component_sizes[:1]  # Keep only the largest
    
    # Create refined mask with only the largest components
    refined_mask = np.zeros_like(high_intensity_mask)
    for comp_id, _ in keep_components:
        refined_mask |= (labeled_array == comp_id)
    
    high_intensity_mask = refined_mask
    print(f"Focused on {len(keep_components)} largest metal components")
    
    # Analyze projections in all three planes
    projections = {}
    
    # Axial projection (sum along Z axis) - traditional view
    axial_proj = np.sum(high_intensity_mask, axis=0)
    projections['axial'] = axial_proj
    
    # Coronal projection (sum along Y axis) - front/back view
    coronal_proj = np.sum(high_intensity_mask, axis=1)
    projections['coronal'] = coronal_proj
    
    # Sagittal projection (sum along X axis) - side view
    sagittal_proj = np.sum(high_intensity_mask, axis=2)
    projections['sagittal'] = sagittal_proj
    
    # Find bounds in each projection
    bounds_3d = {}
    
    # Axial bounds (Y, X from axial projection)
    y_coords, x_coords = np.where(axial_proj > 0)
    if len(y_coords) > 0:
        bounds_3d['y_min'] = int(np.min(y_coords))
        bounds_3d['y_max'] = int(np.max(y_coords)) + 1
        bounds_3d['x_min'] = int(np.min(x_coords))
        bounds_3d['x_max'] = int(np.max(x_coords)) + 1
    
    # Coronal bounds (Z, X from coronal projection)
    z_coords, x_coords = np.where(coronal_proj > 0)
    if len(z_coords) > 0:
        z_min_coronal = int(np.min(z_coords))
        z_max_coronal = int(np.max(z_coords)) + 1
        x_min_coronal = int(np.min(x_coords))
        x_max_coronal = int(np.max(x_coords)) + 1
        
        # Update bounds with coronal info
        bounds_3d['z_min'] = z_min_coronal
        bounds_3d['z_max'] = z_max_coronal
        if 'x_min' in bounds_3d:
            bounds_3d['x_min'] = min(bounds_3d['x_min'], x_min_coronal)
            bounds_3d['x_max'] = max(bounds_3d['x_max'], x_max_coronal)
        else:
            bounds_3d['x_min'] = x_min_coronal
            bounds_3d['x_max'] = x_max_coronal
    
    # Sagittal bounds (Z, Y from sagittal projection)
    z_coords, y_coords = np.where(sagittal_proj > 0)
    if len(z_coords) > 0:
        z_min_sagittal = int(np.min(z_coords))
        z_max_sagittal = int(np.max(z_coords)) + 1
        y_min_sagittal = int(np.min(y_coords))
        y_max_sagittal = int(np.max(y_coords)) + 1
        
        # Update bounds with sagittal info
        if 'z_min' in bounds_3d:
            bounds_3d['z_min'] = min(bounds_3d['z_min'], z_min_sagittal)
            bounds_3d['z_max'] = max(bounds_3d['z_max'], z_max_sagittal)
        else:
            bounds_3d['z_min'] = z_min_sagittal
            bounds_3d['z_max'] = z_max_sagittal
            
        if 'y_min' in bounds_3d:
            bounds_3d['y_min'] = min(bounds_3d['y_min'], y_min_sagittal)
            bounds_3d['y_max'] = max(bounds_3d['y_max'], y_max_sagittal)
        else:
            bounds_3d['y_min'] = y_min_sagittal
            bounds_3d['y_max'] = y_max_sagittal
    
    # Ensure bounds are within volume
    bounds_3d['z_min'] = max(0, bounds_3d.get('z_min', 0))
    bounds_3d['z_max'] = min(ct_volume.shape[0], bounds_3d.get('z_max', ct_volume.shape[0]))
    bounds_3d['y_min'] = max(0, bounds_3d.get('y_min', 0))
    bounds_3d['y_max'] = min(ct_volume.shape[1], bounds_3d.get('y_max', ct_volume.shape[1]))
    bounds_3d['x_min'] = max(0, bounds_3d.get('x_min', 0))
    bounds_3d['x_max'] = min(ct_volume.shape[2], bounds_3d.get('x_max', ct_volume.shape[2]))
    
    # Calculate center of mass of all high-intensity regions
    center_coords = center_of_mass(high_intensity_mask)
    center_coords = tuple(int(c) for c in center_coords)
    
    # Calculate sizes in each dimension
    extent_voxels = {
        'z': bounds_3d['z_max'] - bounds_3d['z_min'],
        'y': bounds_3d['y_max'] - bounds_3d['y_min'],
        'x': bounds_3d['x_max'] - bounds_3d['x_min']
    }
    
    return {
        'threshold_used': threshold,
        'bounds_3d': bounds_3d,
        'center_coords': center_coords,
        'extent_voxels': extent_voxels,
        'projections': projections,
        'high_intensity_mask': high_intensity_mask
    }


def create_adaptive_search_regions(analysis_result, spacing, margin_cm=2.0, volume_shape=None):
    """
    Create adaptive search regions based on 3D analysis.
    Instead of a fixed box, create regions that adapt to the metal shape.
    
    Args:
        analysis_result: Result from analyze_3d_metal_distribution
        spacing: Voxel spacing (z, y, x) in mm
        margin_cm: Additional margin around detected regions in cm
        volume_shape: Shape of the CT volume for bounds checking
        
    Returns:
        dict: Adaptive search regions
    """
    bounds = analysis_result['bounds_3d']
    margin_mm = margin_cm * 10
    
    # Calculate margin in voxels for each axis
    # Use uniform margins for Y and X axes to avoid rectangular ROIs
    avg_pixel_spacing = (abs(spacing[1]) + abs(spacing[2])) / 2.0
    uniform_margin_pixels = int(margin_mm / avg_pixel_spacing)
    
    # Z margin can be different as it's the slice direction
    z_margin = int(margin_mm / abs(spacing[0]))
    margin_voxels = [z_margin, uniform_margin_pixels, uniform_margin_pixels]
    
    # Create expanded bounds with margins, ensuring they stay within volume
    expanded_bounds = {}
    axes = ['z', 'y', 'x']
    for i, (axis, margin) in enumerate(zip(axes, margin_voxels)):
        min_key = f'{axis}_min'
        max_key = f'{axis}_max'
        
        expanded_bounds[min_key] = max(0, bounds[min_key] - margin)
        
        # Ensure max bound doesn't exceed volume dimensions
        if volume_shape is not None:
            expanded_bounds[max_key] = min(volume_shape[i], bounds[max_key] + margin)
        else:
            expanded_bounds[max_key] = bounds[max_key] + margin
    
    # Create multiple search regions for different parts of the implant
    regions = []
    
    # Full region
    regions.append({
        'name': 'full_implant',
        'bounds': expanded_bounds,
        'priority': 1
    })
    
    # Superior region (upper part - femoral head/cup area)
    z_range = bounds['z_max'] - bounds['z_min']
    superior_z_min = bounds['z_min']
    superior_z_max = min(bounds['z_max'], bounds['z_min'] + int(z_range * 0.6))
    
    regions.append({
        'name': 'superior_region',
        'bounds': {
            'z_min': superior_z_min,
            'z_max': superior_z_max,
            'y_min': expanded_bounds['y_min'],
            'y_max': expanded_bounds['y_max'],
            'x_min': expanded_bounds['x_min'],
            'x_max': expanded_bounds['x_max']
        },
        'priority': 2
    })
    
    # Inferior region (lower part - femoral stem area)
    inferior_z_min = max(bounds['z_min'], bounds['z_min'] + int(z_range * 0.4))
    inferior_z_max = bounds['z_max']
    
    regions.append({
        'name': 'inferior_region',
        'bounds': {
            'z_min': inferior_z_min,
            'z_max': inferior_z_max,
            'y_min': expanded_bounds['y_min'],
            'y_max': expanded_bounds['y_max'],
            'x_min': expanded_bounds['x_min'],
            'x_max': expanded_bounds['x_max']
        },
        'priority': 2
    })
    
    return regions


def identify_slices_with_metal(ct_volume, high_intensity_mask, min_metal_voxels=100):
    """
    Identify which slices actually contain metal implants.
    
    Args:
        ct_volume: 3D numpy array of CT data
        high_intensity_mask: 3D mask of high-intensity regions
        min_metal_voxels: Minimum number of metal voxels to consider a slice as having metal
        
    Returns:
        list: Slice indices that contain metal
    """
    slices_with_metal = []
    
    for z in range(ct_volume.shape[0]):
        slice_metal = high_intensity_mask[z]
        metal_count = np.sum(slice_metal)
        
        # Additional check: ensure the slice has actual high-intensity values
        if metal_count >= min_metal_voxels:
            slice_data = ct_volume[z]
            max_hu_in_slice = np.max(slice_data[slice_metal]) if np.any(slice_metal) else 0
            
            # Only consider it metal if there are truly high HU values
            if max_hu_in_slice >= 2500:
                slices_with_metal.append(z)
    
    return slices_with_metal


def create_individual_metal_regions(ct_volume, high_intensity_mask, slices_with_metal, spacing, margin_cm):
    """
    Create individual ROI regions for each metal component, focusing on individual
    femoral heads rather than spanning the full patient width.
    
    Args:
        ct_volume: 3D numpy array of CT data
        high_intensity_mask: 3D mask of high-intensity regions
        slices_with_metal: List of slice indices with metal
        spacing: Voxel spacing (z, y, x) in mm
        margin_cm: Margin around each component in cm
        
    Returns:
        dict: Individual regions per slice
    """
    margin_mm = margin_cm * 10
    
    # Use uniform margin based on average pixel spacing to avoid rectangular ROIs
    # Most CT scans have similar Y and X spacing, but use average to be safe
    avg_pixel_spacing = (abs(spacing[1]) + abs(spacing[2])) / 2.0
    uniform_margin_pixels = int(margin_mm / avg_pixel_spacing)
    
    # Use the same margin for both Y and X to create square margins
    margin_voxels = [uniform_margin_pixels, uniform_margin_pixels]
    
    slice_regions = {}
    
    for z in slices_with_metal:
        slice_metal = high_intensity_mask[z]
        
        if not np.any(slice_metal):
            continue
            
        # Find connected components in this slice
        labeled_slice, num_components = label(slice_metal)
        
        if num_components == 0:
            continue
            
        # Create individual ROI for each component
        components = []
        for comp_id in range(1, num_components + 1):
            comp_mask = labeled_slice == comp_id
            comp_size = np.sum(comp_mask)
            
            if comp_size < 20:  # Skip very small components
                continue
                
            # Get bounding box for this component
            y_coords, x_coords = np.where(comp_mask)
            
            if len(y_coords) == 0:
                continue
                
            # Create focused ROI around this component
            y_min = max(0, np.min(y_coords) - margin_voxels[0])
            y_max = min(ct_volume.shape[1], np.max(y_coords) + margin_voxels[0] + 1)
            x_min = max(0, np.min(x_coords) - margin_voxels[1])
            x_max = min(ct_volume.shape[2], np.max(x_coords) + margin_voxels[1] + 1)
            
            # Only create ROI if it's reasonable size (not too large)
            roi_width = x_max - x_min
            roi_height = y_max - y_min
            
            # Skip if ROI is too large (likely capturing both sides)
            # Be much more conservative - max 25% of image width/height
            max_roi_width = ct_volume.shape[2] // 4  # 25% of image width
            max_roi_height = ct_volume.shape[1] // 4  # 25% of image height
            
            if roi_width > max_roi_width or roi_height > max_roi_height:
                print(f"Skipping oversized ROI in slice {z}: {roi_width}x{roi_height} (max {max_roi_width}x{max_roi_height})")
                continue
            
            components.append({
                'y_min': y_min,
                'y_max': y_max,
                'x_min': x_min,
                'x_max': x_max,
                'size': comp_size,
                'center_y': int(np.mean(y_coords)),
                'center_x': int(np.mean(x_coords))
            })
        
        if components:
            slice_regions[z] = components
    
    return slice_regions


def detect_metal_adaptive_3d(ct_volume, spacing, fw_percentage=75, margin_cm=2.0, intensity_percentile=99.5):
    """
    Advanced metal detection using 3D analysis and adaptive search regions.
    
    Args:
        ct_volume: 3D numpy array of CT data
        spacing: Voxel spacing (z, y, x) in mm
        fw_percentage: Percentage for Full Width threshold
        margin_cm: Margin around detected regions
        intensity_percentile: Percentile for initial detection
        
    Returns:
        dict: Comprehensive detection results
    """
    print("Analyzing 3D metal distribution...")
    
    # Step 1: Analyze 3D distribution
    analysis = analyze_3d_metal_distribution(ct_volume, intensity_percentile)
    
    if analysis is None:
        return {
            'mask': np.zeros_like(ct_volume, dtype=bool),
            'analysis': None,
            'regions': None
        }
    
    # Step 2: Global pass - identify slices with actual metal
    print("Performing global slice analysis...")
    slices_with_metal = identify_slices_with_metal(ct_volume, analysis['high_intensity_mask'])
    print(f"Found metal in {len(slices_with_metal)} slices: {slices_with_metal[:10]}{'...' if len(slices_with_metal) > 10 else ''}")
    
    # Step 3: Create individual ROI regions for each metal component per slice
    print("Creating focused ROI regions...")
    individual_regions = create_individual_metal_regions(
        ct_volume, 
        analysis['high_intensity_mask'], 
        slices_with_metal,
        spacing, 
        margin_cm
    )
    
    # Step 4: Process only slices with metal using individual focused regions
    metal_mask_3d = np.zeros_like(ct_volume, dtype=bool)
    slice_results = []
    
    # Process only slices that actually have metal
    for z in slices_with_metal:
        if z not in individual_regions:
            continue
            
        # Process each metal component in this slice separately
        slice_mask = np.zeros_like(ct_volume[z], dtype=bool)
        slice_thresholds = []
        
        for component in individual_regions[z]:
            # Create focused search bounds for this component
            search_bounds_2d = {
                'y_min': component['y_min'],
                'y_max': component['y_max'],
                'x_min': component['x_min'],
                'x_max': component['x_max']
            }
            
            # Use component center
            center_y = component['center_y']
            center_x = component['center_x']
            
            # Analyze this component with star profiles
            result = analyze_slice_with_adaptive_thresholds(
                ct_volume[z], 
                center_y, 
                center_x,
                search_bounds_2d,
                fw_percentage
            )
            
            if result and result['mask'] is not None:
                # Combine this component's mask with the slice mask
                slice_mask |= result['mask']
                slice_thresholds.append(result['thresholds'])
            else:
                # Fallback: use simple intensity thresholding for this component
                comp_region = ct_volume[z][search_bounds_2d['y_min']:search_bounds_2d['y_max'],
                                          search_bounds_2d['x_min']:search_bounds_2d['x_max']]
                
                if np.any(comp_region > 1500):
                    # There's metal here, use conservative thresholding
                    comp_mask = np.zeros_like(ct_volume[z], dtype=bool)
                    region_mask = comp_region > 1500
                    comp_mask[search_bounds_2d['y_min']:search_bounds_2d['y_max'],
                             search_bounds_2d['x_min']:search_bounds_2d['x_max']] = region_mask
                    
                    slice_mask |= comp_mask
                    slice_thresholds.append((1500, np.max(comp_region)))
        
        # Store results for this slice
        if np.any(slice_mask):
            metal_mask_3d[z] = slice_mask
            
            # Use the average threshold for the slice to be more representative
            if slice_thresholds:
                avg_thresh = np.mean([t[0] for t in slice_thresholds])
                max_thresh = max(t[1] for t in slice_thresholds)
                final_thresholds = (avg_thresh, max_thresh)
            else:
                final_thresholds = (1500, 3000)
                
            slice_results.append({
                'slice': z,
                'thresholds': final_thresholds,
                'method': 'focused_adaptive',
                'num_components': len(individual_regions[z])
            })
    
    # Step 4: Post-processing to improve connectivity
    print("Applying 3D post-processing...")
    
    # Apply 3D morphological operations to connect nearby regions
    metal_mask_3d = binary_dilation(metal_mask_3d, iterations=1)
    metal_mask_3d = binary_erosion(metal_mask_3d, iterations=1)
    
    # Fill small holes
    from scipy.ndimage import binary_fill_holes
    for z in range(metal_mask_3d.shape[0]):
        metal_mask_3d[z] = binary_fill_holes(metal_mask_3d[z])
    
    total_voxels = np.sum(metal_mask_3d)
    print(f"Final metal detection: {total_voxels:,} voxels in {len(slices_with_metal)} slices")
    
    # Create overall ROI bounds for compatibility
    if slices_with_metal and individual_regions:
        # Find overall bounds from all individual regions
        all_y_min = min(min(comp['y_min'] for comp in regions) for regions in individual_regions.values())
        all_y_max = max(max(comp['y_max'] for comp in regions) for regions in individual_regions.values())
        all_x_min = min(min(comp['x_min'] for comp in regions) for regions in individual_regions.values())
        all_x_max = max(max(comp['x_max'] for comp in regions) for regions in individual_regions.values())
        all_z_min = min(slices_with_metal)
        all_z_max = max(slices_with_metal) + 1
        
        overall_roi_bounds = {
            'z_min': all_z_min,
            'z_max': all_z_max,
            'y_min': all_y_min,
            'y_max': all_y_max,
            'x_min': all_x_min,
            'x_max': all_x_max
        }
    else:
        # Fallback to analysis bounds
        overall_roi_bounds = analysis['bounds_3d']
    
    return {
        'mask': metal_mask_3d,
        'analysis': analysis,
        'individual_regions': individual_regions,
        'slices_with_metal': slices_with_metal,
        'roi_bounds': overall_roi_bounds,  # For compatibility
        'slice_thresholds': slice_results,
        'center_coords': analysis['center_coords']
    }


def analyze_slice_with_adaptive_thresholds(ct_slice, center_y, center_x, search_bounds, fw_percentage=75):
    """
    Analyze a single slice using star profiles with adaptive thresholding.
    """
    # Get star profile lines
    profiles = get_star_profile_lines(ct_slice, center_y, center_x, search_bounds)
    
    # Analyze profiles
    all_metal_thresholds = []
    
    for distances, hu_values in profiles:
        # Find metal peaks
        peaks, properties = find_peaks(hu_values, prominence=150, height=800)
        
        for peak_idx in peaks:
            peak_value = hu_values[peak_idx]
            if peak_value > 1200:  # Metal candidate
                # Calculate FW threshold
                fw_threshold = (fw_percentage / 100.0) * peak_value
                all_metal_thresholds.append(fw_threshold)
    
    if not all_metal_thresholds:
        # No metal peaks found
        return None
    
    # Use the most conservative (highest) threshold to avoid including non-metal
    # For multiple peaks, use the average threshold
    final_threshold = np.mean(all_metal_thresholds)
    
    # Create mask
    metal_mask = ct_slice >= final_threshold
    
    # Constrain to search bounds
    search_mask = np.zeros_like(ct_slice, dtype=bool)
    search_mask[search_bounds['y_min']:search_bounds['y_max'],
                search_bounds['x_min']:search_bounds['x_max']] = True
    
    metal_mask = metal_mask & search_mask
    
    # Add morphological processing to connect nearby regions
    metal_mask = binary_dilation(metal_mask, iterations=2)
    metal_mask = binary_erosion(metal_mask, iterations=1)
    
    return {
        'mask': metal_mask,
        'thresholds': (final_threshold, np.max(ct_slice[metal_mask]) if np.any(metal_mask) else final_threshold),
        'profiles': profiles
    }


def get_star_profile_lines(slice_2d, center_y, center_x, bounds):
    """
    Generate 16 star profile lines from center to boundaries.
    """
    y_min, y_max = bounds['y_min'], bounds['y_max']
    x_min, x_max = bounds['x_min'], bounds['x_max']
    
    # Calculate intermediate points for 16-point star
    y_mid = (y_min + y_max) // 2
    x_mid = (x_min + x_max) // 2
    
    y_q1 = (y_min + y_mid) // 2
    y_q3 = (y_mid + y_max) // 2
    x_q1 = (x_min + x_mid) // 2
    x_q3 = (x_mid + x_max) // 2
    
    # Define 16 endpoints
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
    
    profiles = []
    
    for end_y, end_x in endpoints:
        # Ensure endpoints are within bounds
        end_y = max(0, min(slice_2d.shape[0] - 1, end_y))
        end_x = max(0, min(slice_2d.shape[1] - 1, end_x))
        
        # Get line coordinates
        rr, cc = line(center_y, center_x, end_y, end_x)
        
        # Calculate distances from center
        distances = np.sqrt((rr - center_y)**2 + (cc - center_x)**2)
        
        # Get HU values along the line
        hu_values = slice_2d[rr, cc]
        
        profiles.append((distances, hu_values))
    
    return profiles