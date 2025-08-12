import numpy as np
from scipy.ndimage import label, center_of_mass, maximum_filter
from scipy.signal import find_peaks
from skimage.draw import line
import nibabel as nib


def find_metal_cluster(ct_volume, min_hu_threshold=2500, dilation_iterations=3):
    """
    Find the most intense cluster of voxels in the CT volume.
    
    Args:
        ct_volume: 3D numpy array of CT data in HU
        min_hu_threshold: Minimum HU value to consider as potential metal
        dilation_iterations: Number of dilation iterations to connect regions
        
    Returns:
        tuple: (center_coords, cluster_mask) - 3D coordinates of cluster center and boolean mask
    """
    # Create binary mask of high-intensity voxels
    high_intensity_mask = ct_volume > min_hu_threshold
    
    if not np.any(high_intensity_mask):
        print(f"No voxels found above {min_hu_threshold} HU")
        return None, None
    
    # Apply dilation to connect nearby metal regions
    from scipy.ndimage import binary_dilation
    dilated_mask = binary_dilation(high_intensity_mask, iterations=dilation_iterations)
    
    # Label connected components on dilated mask
    labeled_array, num_features = label(dilated_mask)
    
    if num_features == 0:
        return None, None
    
    # Find all significant components (not just the largest)
    component_sizes = []
    for i in range(1, num_features + 1):
        component_mask = labeled_array == i
        # Check original mask to get actual metal voxels
        actual_metal = component_mask & high_intensity_mask
        size = np.sum(actual_metal)
        if size > 100:  # Minimum size threshold
            mean_hu = np.mean(ct_volume[actual_metal])
            component_sizes.append((i, size, mean_hu))
    
    if not component_sizes:
        return None, None
    
    # Sort by size and intensity
    component_sizes.sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    # Combine top components if multiple significant ones exist
    combined_mask = np.zeros_like(high_intensity_mask)
    for i, (comp_id, size, _) in enumerate(component_sizes[:3]):  # Top 3 components
        component_mask = labeled_array == comp_id
        combined_mask |= (component_mask & high_intensity_mask)
    
    # Find center of mass of the combined metal regions
    center_coords = center_of_mass(combined_mask)
    center_coords = tuple(int(c) for c in center_coords)
    
    return center_coords, combined_mask


def create_roi_box(center_coords, volume_shape, spacing, margin_cm=3.0):
    """
    Create a 3D ROI box around the center coordinates.
    
    Args:
        center_coords: (z, y, x) coordinates of the center
        volume_shape: Shape of the CT volume
        spacing: Voxel spacing in mm (z, y, x)
        margin_cm: Margin in centimeters
        
    Returns:
        dict: ROI boundaries for each axis
    """
    margin_mm = margin_cm * 10  # Convert cm to mm
    
    # Calculate margin in voxels for each axis
    # Use absolute value of spacing to handle negative z-spacing in DICOM
    margin_voxels = [int(margin_mm / abs(s)) for s in spacing]
    
    roi_bounds = {}
    for i, (center, margin, size) in enumerate(zip(center_coords, margin_voxels, volume_shape)):
        min_bound = max(0, center - margin)
        max_bound = min(size, center + margin)
        axis_name = ['z', 'y', 'x'][i]
        roi_bounds[f'{axis_name}_min'] = min_bound
        roi_bounds[f'{axis_name}_max'] = max_bound
    
    return roi_bounds


def get_star_profile_lines(slice_2d, center_y, center_x, roi_bounds_2d):
    """
    Generate 16 star profile lines from center to ROI boundaries.
    
    Args:
        slice_2d: 2D CT slice
        center_y, center_x: Center coordinates in the slice
        roi_bounds_2d: Dictionary with y_min, y_max, x_min, x_max
        
    Returns:
        list: List of (distances, hu_values) tuples for each profile line
    """
    y_min, y_max = roi_bounds_2d['y_min'], roi_bounds_2d['y_max']
    x_min, x_max = roi_bounds_2d['x_min'], roi_bounds_2d['x_max']
    
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


def find_fw_threshold(distances, hu_values, percentage=75, min_peak_prominence=500):
    """
    Find the Full Width at specified percentage Maximum threshold from a profile.
    
    Args:
        distances: Array of distances from center
        hu_values: Array of HU values corresponding to distances
        percentage: Percentage of peak value for threshold (default 75)
        min_peak_prominence: Minimum prominence for peak detection
        
    Returns:
        tuple: (lower_threshold, upper_threshold) or None if no valid peak
    """
    # Find peaks in the HU profile
    peaks, properties = find_peaks(hu_values, 
                                  prominence=min_peak_prominence,
                                  height=1000)  # Minimum height for metal
    
    if len(peaks) == 0:
        return None
    
    # Find the highest peak
    highest_peak_idx = peaks[np.argmax(hu_values[peaks])]
    peak_value = hu_values[highest_peak_idx]
    
    # Calculate percentage of peak value
    threshold_value = (percentage / 100.0) * peak_value
    
    # Find where the profile crosses the threshold
    # Search backward from peak
    lower_idx = highest_peak_idx
    while lower_idx > 0 and hu_values[lower_idx] > threshold_value:
        lower_idx -= 1
    
    # Search forward from peak
    upper_idx = highest_peak_idx
    while upper_idx < len(hu_values) - 1 and hu_values[upper_idx] > threshold_value:
        upper_idx += 1
    
    # Get the HU values at the threshold crossings
    lower_threshold = hu_values[lower_idx] if lower_idx > 0 else threshold_value
    upper_threshold = peak_value
    
    return (lower_threshold, upper_threshold)


def detect_metal_adaptive(ct_slice, roi_bounds_2d, spacing_2d, fw_percentage=75):
    """
    Detect metal in a single slice using adaptive thresholding.
    
    Args:
        ct_slice: 2D CT slice
        roi_bounds_2d: ROI boundaries for this slice
        spacing_2d: Pixel spacing (y, x) in mm
        fw_percentage: Percentage for Full Width threshold (default 75)
        
    Returns:
        dict: Contains metal mask and threshold values
    """
    # Find high-intensity region to get initial center
    high_intensity_mask = ct_slice > 2500
    
    if not np.any(high_intensity_mask):
        return {'mask': np.zeros_like(ct_slice, dtype=bool), 'thresholds': None}
    
    # Find center of high-intensity region
    y_coords, x_coords = np.where(high_intensity_mask)
    center_y = int(np.mean(y_coords))
    center_x = int(np.mean(x_coords))
    
    # Get star profiles
    profiles = get_star_profile_lines(ct_slice, center_y, center_x, roi_bounds_2d)
    
    # Collect all threshold ranges
    all_thresholds = []
    
    for distances, hu_values in profiles:
        threshold_range = find_fw_threshold(distances, hu_values, percentage=fw_percentage)
        if threshold_range:
            all_thresholds.append(threshold_range)
    
    if not all_thresholds:
        # Fallback to default threshold
        return {
            'mask': ct_slice > 2500,
            'thresholds': (2500, np.max(ct_slice))
        }
    
    # Combine thresholds - use the most inclusive range
    all_thresholds = np.array(all_thresholds)
    final_lower = np.percentile(all_thresholds[:, 0], 25)  # 25th percentile of lower bounds
    final_upper = np.max(all_thresholds[:, 1])  # Maximum of upper bounds
    
    # Create metal mask using adaptive thresholds
    metal_mask = (ct_slice >= final_lower) & (ct_slice <= final_upper)
    
    # Constrain to ROI
    roi_mask = np.zeros_like(ct_slice, dtype=bool)
    roi_mask[roi_bounds_2d['y_min']:roi_bounds_2d['y_max'],
             roi_bounds_2d['x_min']:roi_bounds_2d['x_max']] = True
    
    metal_mask = metal_mask & roi_mask
    
    return {
        'mask': metal_mask,
        'thresholds': (final_lower, final_upper),
        'profiles': profiles
    }


def detect_metal_volume(ct_volume, spacing, margin_cm=3.0, fw_percentage=75, min_metal_hu=2500, dilation_iterations=3):
    """
    Detect metal throughout the entire CT volume using slice-by-slice adaptive thresholding.
    
    Args:
        ct_volume: 3D numpy array of CT data
        spacing: Voxel spacing (z, y, x) in mm
        margin_cm: ROI margin in centimeters
        fw_percentage: Percentage for Full Width threshold (default 75)
        min_metal_hu: Minimum HU threshold for initial metal detection
        dilation_iterations: Number of dilation iterations for metal connection
        
    Returns:
        dict: Contains 3D metal mask, ROI bounds, and per-slice thresholds
    """
    # Find initial metal cluster
    center_coords, initial_mask = find_metal_cluster(ct_volume, min_hu_threshold=min_metal_hu, dilation_iterations=dilation_iterations)
    
    if center_coords is None:
        return {
            'mask': np.zeros_like(ct_volume, dtype=bool),
            'roi_bounds': None,
            'slice_thresholds': None
        }
    
    # Create ROI box
    roi_bounds = create_roi_box(center_coords, ct_volume.shape, spacing, margin_cm)
    
    # Process each slice
    metal_mask_3d = np.zeros_like(ct_volume, dtype=bool)
    slice_thresholds = []
    
    for z in range(roi_bounds['z_min'], roi_bounds['z_max']):
        # Get 2D ROI bounds for this slice
        roi_bounds_2d = {
            'y_min': roi_bounds['y_min'],
            'y_max': roi_bounds['y_max'],
            'x_min': roi_bounds['x_min'],
            'x_max': roi_bounds['x_max']
        }
        
        # Detect metal in this slice
        result = detect_metal_adaptive(ct_volume[z], roi_bounds_2d, spacing[1:], fw_percentage)
        
        metal_mask_3d[z] = result['mask']
        slice_thresholds.append({
            'slice': z,
            'thresholds': result['thresholds']
        })
    
    return {
        'mask': metal_mask_3d,
        'roi_bounds': roi_bounds,
        'slice_thresholds': slice_thresholds,
        'center_coords': center_coords
    }


def save_mask_as_nifti(mask, affine, output_path):
    """
    Save a binary mask as a NIFTI file.
    
    Args:
        mask: 3D boolean numpy array
        affine: 4x4 affine transformation matrix
        output_path: Path to save the NIFTI file
    """
    # Convert boolean to int
    mask_int = mask.astype(np.uint8)
    
    # Create NIFTI image
    nifti_img = nib.Nifti1Image(mask_int, affine)
    
    # Save to file
    nib.save(nifti_img, output_path)
    
    
def create_affine_from_dicom_meta(ct_metadata):
    """
    Create a NIFTI affine matrix from DICOM metadata.
    
    Args:
        ct_metadata: Dictionary containing DICOM spatial metadata
        
    Returns:
        4x4 numpy array representing the affine transformation
    """
    origin = ct_metadata['origin']
    spacing = ct_metadata['spacing']
    
    # Create affine matrix
    affine = np.eye(4)
    affine[0, 0] = spacing[1]  # x spacing
    affine[1, 1] = spacing[0]  # y spacing  
    affine[2, 2] = spacing[2]  # z spacing
    affine[:3, 3] = origin
    
    return affine