import numpy as np
from scipy.ndimage import label, center_of_mass, binary_dilation, maximum_filter
from scipy.signal import find_peaks
from skimage.draw import line
import nibabel as nib


def find_high_intensity_region(ct_volume, percentile=99.5):
    """
    Find regions of very high intensity without using a fixed threshold.
    Uses percentile-based detection to adapt to each scan.
    
    Args:
        ct_volume: 3D numpy array of CT data in HU
        percentile: Percentile to use for initial detection (99.5 = top 0.5% of voxels)
        
    Returns:
        tuple: (center_coords, initial_region_mask, threshold_used)
    """
    # Calculate percentile threshold from the volume
    threshold = np.percentile(ct_volume, percentile)
    print(f"Using percentile-based threshold: {threshold:.0f} HU (top {100-percentile}% of voxels)")
    
    # Create mask of high-intensity voxels
    high_intensity_mask = ct_volume > threshold
    
    if not np.any(high_intensity_mask):
        # If nothing found, try a lower percentile
        threshold = np.percentile(ct_volume, 99.0)
        high_intensity_mask = ct_volume > threshold
        
    if not np.any(high_intensity_mask):
        print("No high-intensity regions found")
        return None, None, None
    
    # Find the brightest connected region
    labeled_array, num_features = label(high_intensity_mask)
    
    if num_features == 0:
        return None, None, None
    
    # Find all regions and their mean intensities
    regions = []
    for i in range(1, num_features + 1):
        mask = labeled_array == i
        if np.sum(mask) > 50:  # Minimum size
            mean_hu = np.mean(ct_volume[mask])
            max_hu = np.max(ct_volume[mask])
            regions.append((i, np.sum(mask), mean_hu, max_hu))
    
    if not regions:
        return None, None, None
    
    # Sort by maximum HU value (brightest first)
    regions.sort(key=lambda x: x[3], reverse=True)
    
    # Get the brightest region
    brightest_id = regions[0][0]
    brightest_mask = labeled_array == brightest_id
    
    # Find center of the brightest region
    center_coords = center_of_mass(brightest_mask)
    center_coords = tuple(int(c) for c in center_coords)
    
    return center_coords, brightest_mask, threshold


def create_search_box(center_coords, volume_shape, spacing, box_size_cm=8.0):
    """
    Create a search box around the high-intensity region.
    This box will be used for star profile analysis.
    
    Args:
        center_coords: (z, y, x) coordinates of the center
        volume_shape: Shape of the CT volume
        spacing: Voxel spacing in mm (z, y, x)
        box_size_cm: Size of the search box in centimeters
        
    Returns:
        dict: Box boundaries for each axis
    """
    box_size_mm = box_size_cm * 10  # Convert cm to mm
    
    # Calculate box size in voxels for each axis
    # Handle negative spacing (common in DICOM)
    box_voxels = [int(box_size_mm / abs(s)) for s in spacing]
    
    box_bounds = {}
    for i, (center, box_size, vol_size) in enumerate(zip(center_coords, box_voxels, volume_shape)):
        min_bound = max(0, center - box_size // 2)
        max_bound = min(vol_size, center + box_size // 2)
        
        # Ensure min < max
        if min_bound >= max_bound:
            min_bound = max(0, center - 20)  # Fallback to 20 voxels
            max_bound = min(vol_size, center + 20)
        
        axis_name = ['z', 'y', 'x'][i]
        box_bounds[f'{axis_name}_min'] = int(min_bound)
        box_bounds[f'{axis_name}_max'] = int(max_bound)
    
    return box_bounds


def analyze_slice_with_star_profiles(ct_slice, center_y, center_x, search_bounds, fw_percentage=75):
    """
    Analyze a single slice using star profiles to determine metal thresholds.
    
    Args:
        ct_slice: 2D CT slice
        center_y, center_x: Approximate center of metal region
        search_bounds: Dictionary with y_min, y_max, x_min, x_max
        fw_percentage: Percentage for Full Width threshold
        
    Returns:
        dict: Contains threshold range and metal mask for this slice
    """
    # Get star profile lines
    profiles = get_star_profile_lines(ct_slice, center_y, center_x, search_bounds)
    
    # Analyze all profiles to find threshold ranges
    all_metal_peaks = []
    all_fw_thresholds = []
    
    for distances, hu_values in profiles:
        # Find peaks that could be metal - use more sensitive detection
        peaks, properties = find_peaks(hu_values, prominence=200, height=1000)
        
        if len(peaks) > 0:
            # Find all significant peaks (potential metal)
            for peak_idx in peaks:
                peak_value = hu_values[peak_idx]
                if peak_value > 1500:  # Only consider high-intensity peaks
                    all_metal_peaks.append(peak_value)
                    
                    # Calculate FW threshold for this peak
                    threshold_value = (fw_percentage / 100.0) * peak_value
                    all_fw_thresholds.append(threshold_value)
    
    if not all_metal_peaks:
        # No metal peaks found - fall back to simple intensity analysis
        search_region = ct_slice[search_bounds['y_min']:search_bounds['y_max'],
                               search_bounds['x_min']:search_bounds['x_max']]
        if np.any(search_region > 2000):
            # There's clearly metal here, use a conservative threshold
            final_lower = 1500
            final_upper = np.max(search_region)
        else:
            return None
    else:
        # Use the FW thresholds but be more inclusive
        # Take the minimum FW threshold to capture transition zones
        final_lower = min(np.min(all_fw_thresholds), 1500)  # Cap at reasonable minimum
        final_upper = np.max(all_metal_peaks)
    
    # Create metal mask for this slice
    # Use a stepped approach: strict metal core + transition zones
    core_metal = ct_slice >= final_lower
    
    # Add transition zones around the core metal
    from scipy.ndimage import binary_dilation
    expanded_metal = binary_dilation(core_metal, iterations=2)
    
    # But only include voxels above bone density in the expansion
    bone_threshold = max(800, final_lower * 0.6)  # Adaptive bone threshold
    transition_mask = (ct_slice >= bone_threshold) & expanded_metal
    
    # Combine core metal and transitions
    metal_mask = core_metal | transition_mask
    
    # Constrain to search region
    search_mask = np.zeros_like(ct_slice, dtype=bool)
    search_mask[search_bounds['y_min']:search_bounds['y_max'],
                search_bounds['x_min']:search_bounds['x_max']] = True
    
    metal_mask = metal_mask & search_mask
    
    return {
        'thresholds': (final_lower, final_upper),
        'mask': metal_mask,
        'profiles': profiles,
        'core_threshold': final_lower,
        'transition_threshold': bone_threshold
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
        # Get line coordinates
        rr, cc = line(center_y, center_x, end_y, end_x)
        
        # Calculate distances from center
        distances = np.sqrt((rr - center_y)**2 + (cc - center_x)**2)
        
        # Get HU values along the line
        hu_values = slice_2d[rr, cc]
        
        profiles.append((distances, hu_values))
    
    return profiles


def detect_metal_volume_pure_adaptive(ct_volume, spacing, search_box_cm=8.0, fw_percentage=75):
    """
    Detect metal using pure adaptive thresholding based on star profiles.
    No initial HU threshold required - the algorithm finds metal automatically.
    
    Args:
        ct_volume: 3D numpy array of CT data
        spacing: Voxel spacing (z, y, x) in mm
        search_box_cm: Size of search box in cm (larger = more area to search)
        fw_percentage: Percentage for Full Width threshold
        
    Returns:
        dict: Contains 3D metal mask, box bounds, and per-slice thresholds
    """
    # Step 1: Find high-intensity region using percentile-based detection
    center_coords, initial_region, auto_threshold = find_high_intensity_region(ct_volume)
    
    if center_coords is None:
        return {
            'mask': np.zeros_like(ct_volume, dtype=bool),
            'box_bounds': None,
            'slice_thresholds': None,
            'initial_threshold': None
        }
    
    print(f"Found high-intensity region at {center_coords}")
    
    # Step 2: Create search box around the high-intensity region
    box_bounds = create_search_box(center_coords, ct_volume.shape, spacing, search_box_cm)
    
    # Step 3: Process each slice with star profile analysis
    metal_mask_3d = np.zeros_like(ct_volume, dtype=bool)
    slice_thresholds = []
    
    for z in range(box_bounds['z_min'], box_bounds['z_max']):
        # Get 2D bounds for this slice
        search_bounds_2d = {
            'y_min': box_bounds['y_min'],
            'y_max': box_bounds['y_max'],
            'x_min': box_bounds['x_min'],
            'x_max': box_bounds['x_max']
        }
        
        # Find approximate center for this slice
        # Use the high-intensity region if it exists in this slice
        if np.any(initial_region[z]):
            y_coords, x_coords = np.where(initial_region[z])
            slice_center_y = int(np.mean(y_coords))
            slice_center_x = int(np.mean(x_coords))
        else:
            # Use the global center
            slice_center_y = center_coords[1]
            slice_center_x = center_coords[2]
        
        # Analyze this slice
        result = analyze_slice_with_star_profiles(
            ct_volume[z], 
            slice_center_y, 
            slice_center_x,
            search_bounds_2d,
            fw_percentage
        )
        
        if result:
            metal_mask_3d[z] = result['mask']
            slice_thresholds.append({
                'slice': z,
                'thresholds': result['thresholds']
            })
        else:
            slice_thresholds.append({
                'slice': z,
                'thresholds': None
            })
    
    # Step 4: Post-process to fill gaps
    # Apply some morphological operations to connect regions
    metal_mask_3d = binary_dilation(metal_mask_3d, iterations=2)
    from scipy.ndimage import binary_erosion
    metal_mask_3d = binary_erosion(metal_mask_3d, iterations=1)
    
    return {
        'mask': metal_mask_3d,
        'box_bounds': box_bounds,
        'slice_thresholds': slice_thresholds,
        'center_coords': center_coords,
        'initial_threshold': auto_threshold
    }