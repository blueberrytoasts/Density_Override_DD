import numpy as np
from scipy.ndimage import center_of_mass, binary_dilation
from skimage.draw import line


def get_star_profile_values(current_slice, metal_mask, roi_boundaries):
    """
    Performs the 16-point star profile analysis and returns all HU values,
    ignoring any pixels that are part of the metal mask itself.
    """
    # Find the center of the metal implant using center of mass
    try:
        r_center, c_center = center_of_mass(metal_mask)
        r_center, c_center = int(r_center), int(c_center)
    except Exception as e:
        print(f"Warning: Center of mass calculation failed ({e}). Falling back to bounding box center.")
        r_coords, c_coords = np.where(metal_mask)
        r_center = int(np.mean([np.min(r_coords), np.max(r_coords)]))
        c_center = int(np.mean([np.min(c_coords), np.max(c_coords)]))

    # Use the provided ROI boundaries
    r_min_roi, r_max_roi, c_min_roi, c_max_roi = roi_boundaries

    # Define the 16 endpoints for the star
    r_mid = int((r_min_roi + r_max_roi) / 2)
    c_mid = int((c_min_roi + c_max_roi) / 2)
    endpoints = [
        (r_min_roi, c_min_roi), (r_min_roi, c_max_roi), (r_max_roi, c_min_roi), (r_max_roi, c_max_roi),
        (r_min_roi, c_mid), (r_max_roi, c_mid), (r_mid, c_min_roi), (r_mid, c_max_roi),
        (r_min_roi, int((c_min_roi + c_mid) / 2)), (r_min_roi, int((c_max_roi + c_mid) / 2)),
        (r_max_roi, int((c_min_roi + c_mid) / 2)), (r_max_roi, int((c_max_roi + c_mid) / 2)),
        (int((r_min_roi + r_mid) / 2), c_min_roi), (int((r_max_roi + r_mid) / 2), c_min_roi),
        (int((r_min_roi + r_mid) / 2), c_max_roi), (int((r_max_roi + r_mid) / 2), c_max_roi)
    ]

    # Extract HU values along each of the 16 lines
    all_profile_values = []
    for r_end, c_end in endpoints:
        rr, cc = line(r_center, c_center, r_end, c_end)

        # Filter out pixels that are part of the metal mask
        is_metal = metal_mask[rr, cc]
        non_metal_rr = rr[~is_metal]
        non_metal_cc = cc[~is_metal]
        
        # Get the HU values for only the non-metal pixels
        all_profile_values.append(current_slice[non_metal_rr, non_metal_cc])

    return np.concatenate(all_profile_values)


def find_dark_artifact_range_automatically(current_slice, metal_mask, roi_boundaries):
    """
    Analyzes star-profile values to automatically determine the HU range for dark artifacts.
    """
    profile_values = get_star_profile_values(current_slice, metal_mask, roi_boundaries)
    potential_dark_values = profile_values[profile_values < -100]

    if potential_dark_values.size < 20:
        print("Warning: Not enough dark pixel samples found to determine range automatically.")
        return None

    min_hu = np.min(potential_dark_values)
    max_hu = np.percentile(potential_dark_values, 75)

    if min_hu >= max_hu:
        return None

    return (int(min_hu), int(max_hu))


def find_bright_artifact_range_automatically(current_slice, metal_mask, roi_boundaries):
    """
    Analyzes star-profile values to automatically determine the HU range for
    high-HU bright artifacts around the metal implant.
    """
    profile_values = get_star_profile_values(current_slice, metal_mask, roi_boundaries)
    potential_bright_values = profile_values[profile_values > 800]

    if potential_bright_values.size < 20:
        print("Warning: Not enough bright pixel samples found to determine range automatically.")
        return None

    min_hu = np.min(potential_bright_values)
    max_hu = np.percentile(potential_bright_values, 75)

    if min_hu >= max_hu or min_hu > 3000:
        return None

    return [(int(min_hu), int(max_hu))]

def find_bone_range_automatically(ct_slice, metal_mask, dark_artifact_mask, bright_artifact_mask, roi_boundaries):
    """
    Automatically determines the HU range for bone by analyzing the histogram
    of non-artifact, non-metal regions within the ROI.
    """
    # Create a mask of non-metal, non-artifact regions within the ROI
    roi_mask = np.zeros_like(ct_slice, dtype=bool)
    r_min_roi, r_max_roi, c_min_roi, c_max_roi = roi_boundaries
    roi_mask[r_min_roi:r_max_roi, c_min_roi:c_max_roi] = True

    non_artifact_mask = roi_mask & ~metal_mask & ~dark_artifact_mask & ~bright_artifact_mask

    potential_bone_values = ct_slice[non_artifact_mask]

    if potential_bone_values.size < 100: # Need a reasonable number of samples
        print("Warning: Not enough non-artifact pixels to determine bone range automatically. Using default.")
        return None

    # Analyze histogram to find bone peak
    hist, bin_edges = np.histogram(potential_bone_values, bins=np.arange(-1024, 3000, 10))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Look for a peak in the typical bone range (e.g., 150-1500 HU)
    bone_search_min = 150
    bone_search_max = 1500
    search_indices = np.where((bin_centers >= bone_search_min) & (bin_centers <= bone_search_max))

    if len(search_indices[0]) == 0:
        print("Warning: No significant bone-like HU values found in histogram. Using default.")
        return None

    # Find the peak within the bone search range
    peak_index = search_indices[0][np.argmax(hist[search_indices])]
    peak_hu = bin_centers[peak_index]

    # Define range around the peak
    # This is a heuristic and can be refined
    min_hu = max(bone_search_min, int(peak_hu - 150)) # Lower bound, not going below search min
    max_hu = min(bone_search_max, int(peak_hu + 300)) # Upper bound, not going above search max

    # Ensure min_hu is less than max_hu
    if min_hu >= max_hu:
        min_hu = bone_search_min
        max_hu = bone_search_max

    return (min_hu, max_hu)

def segment_artifacts_and_tissues(ct_slice, metal_mask, roi_boundaries, 
                                  dark_artifact_range=None, bright_artifact_ranges=None,
                                  bone_range=(300, 1300), artifact_margin=30,
                                  auto_find_ranges=True):
    """
    Segments the CT slice into metal, bright artifacts, dark artifacts, and bone.
    
    Returns:
        dict: Dictionary containing masks for each tissue type
    """
    rows, cols = ct_slice.shape
    
    # Define ROI around metal
    r_coords, c_coords = np.where(metal_mask)
    r_min_metal, r_max_metal = np.min(r_coords), np.max(r_coords)
    c_min_metal, c_max_metal = np.min(c_coords), np.max(c_coords)
    
    r_min_roi = max(0, r_min_metal - artifact_margin)
    r_max_roi = min(rows, r_max_metal + artifact_margin)
    c_min_roi = max(0, c_min_metal - artifact_margin)
    c_max_roi = min(cols, c_max_metal - artifact_margin)
    
    roi_boundaries = (r_min_roi, r_max_roi, c_min_roi, c_max_roi)
    
    # Auto-find ranges if requested
    if auto_find_ranges:
        if dark_artifact_range is None:
            dark_artifact_range = find_dark_artifact_range_automatically(
                ct_slice, metal_mask, roi_boundaries
            )
            if dark_artifact_range is None:
                dark_artifact_range = (-1000, -250)  # Default
                
        if bright_artifact_ranges is None:
            bright_artifact_ranges = find_bright_artifact_range_automatically(
                ct_slice, metal_mask, roi_boundaries
            )
            if bright_artifact_ranges is None:
                bright_artifact_ranges = [(1000, 3000)]  # Default
    
    # Create ROI mask
    artifact_roi_mask = np.zeros(ct_slice.shape, dtype=bool)
    artifact_roi_mask[r_min_roi:r_max_roi, c_min_roi:c_max_roi] = True
    
    # Create provisional bone seed
    provisional_bone_range = (300, 800)
    provisional_bone_mask = (ct_slice >= provisional_bone_range[0]) & \
                           (ct_slice <= provisional_bone_range[1]) & \
                           artifact_roi_mask
    
    # Create bright artifact mask from ranges
    bright_artifact_mask = np.zeros_like(ct_slice, dtype=bool)
    for min_hu, max_hu in bright_artifact_ranges:
        bright_artifact_mask |= (ct_slice >= min_hu) & (ct_slice <= max_hu)
    
    provisional_bone_mask &= ~bright_artifact_mask
    
    # Create search ring around bone
    dilated_bone_mask = binary_dilation(provisional_bone_mask, iterations=5)
    bone_search_ring_mask = dilated_bone_mask & ~provisional_bone_mask
    
    # Find bone-adjacent artifacts
    bone_adjacent_artifact_range = (800, 1200)
    bone_adjacent_artifact_mask = (ct_slice >= bone_adjacent_artifact_range[0]) & \
                                 (ct_slice <= bone_adjacent_artifact_range[1]) & \
                                 bone_search_ring_mask
    
    # Combine all bright artifacts
    combined_bright_artifact_mask = bright_artifact_mask | bone_adjacent_artifact_mask
    final_bright_artifact_mask = combined_bright_artifact_mask & ~metal_mask
    
    # Create dark artifact mask
    min_hu_dark, max_hu_dark = dark_artifact_range
    final_dark_artifact_mask = (ct_slice >= min_hu_dark) & \
                              (ct_slice <= max_hu_dark) & \
                              artifact_roi_mask
    
    # Create final bone mask
    min_hu_bone, max_hu_bone = bone_range
    bone_mask_raw = (ct_slice >= min_hu_bone) & \
                    (ct_slice <= max_hu_bone) & \
                    artifact_roi_mask
    
    final_bone_mask = bone_mask_raw & ~metal_mask & ~final_bright_artifact_mask & ~final_dark_artifact_mask
    
    return {
        'metal': metal_mask,
        'bright_artifacts': final_bright_artifact_mask,
        'dark_artifacts': final_dark_artifact_mask,
        'bone': final_bone_mask,
        'roi_boundaries': roi_boundaries,
        'dark_artifact_range': dark_artifact_range,
        'bright_artifact_ranges': bright_artifact_ranges
    }
