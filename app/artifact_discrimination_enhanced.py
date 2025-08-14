import numpy as np
from scipy.ndimage import gaussian_filter, sobel, generic_gradient_magnitude
from scipy.ndimage import label, binary_dilation, binary_erosion, distance_transform_edt
from skimage.feature import canny
from skimage.measure import regionprops
import time


def compute_edge_coherence(ct_slice, mask, window_size=5):
    """
    Analyze edge coherence - bone has continuous edges, artifacts are chaotic.
    
    Args:
        ct_slice: 2D CT slice
        mask: Binary mask of candidate regions
        window_size: Size of local window for coherence analysis
        
    Returns:
        coherence_map: Map of edge coherence scores
    """
    # Compute gradients
    grad_y = sobel(ct_slice.astype(float), axis=0)
    grad_x = sobel(ct_slice.astype(float), axis=1)
    
    # Gradient magnitude and direction
    grad_mag = np.sqrt(grad_y**2 + grad_x**2)
    grad_dir = np.arctan2(grad_y, grad_x)
    
    # Only analyze where we have significant gradients
    edge_mask = (grad_mag > np.percentile(grad_mag[mask], 75)) & mask
    
    coherence_map = np.zeros_like(ct_slice, dtype=float)
    half_window = window_size // 2
    
    # Compute local coherence
    for y in range(half_window, ct_slice.shape[0] - half_window):
        for x in range(half_window, ct_slice.shape[1] - half_window):
            if edge_mask[y, x]:
                # Get local window
                local_window = grad_dir[y-half_window:y+half_window+1,
                                      x-half_window:x+half_window+1]
                local_mask = edge_mask[y-half_window:y+half_window+1,
                                     x-half_window:x+half_window+1]
                
                if np.sum(local_mask) > 3:  # Need enough edge pixels
                    # Calculate circular variance of directions
                    angles = local_window[local_mask]
                    # Convert to unit vectors and average
                    mean_x = np.mean(np.cos(angles))
                    mean_y = np.mean(np.sin(angles))
                    # Coherence is the magnitude of average vector
                    coherence = np.sqrt(mean_x**2 + mean_y**2)
                    coherence_map[y, x] = coherence
    
    return coherence_map, grad_mag, grad_dir


def analyze_gradient_jumps(ct_slice, mask):
    """
    Detect rigid jumps characteristic of bone edges.
    
    Args:
        ct_slice: 2D CT slice
        mask: Binary mask of candidate regions
        
    Returns:
        jump_characteristics: Map of gradient jump features
    """
    # Smooth slightly to reduce noise
    smoothed = gaussian_filter(ct_slice.astype(float), sigma=0.5)
    
    # First derivative (gradient magnitude)
    grad_mag = generic_gradient_magnitude(smoothed, sobel)
    
    # Second derivative to find inflection points
    grad_y = sobel(smoothed, axis=0)
    grad_x = sobel(smoothed, axis=1)
    
    # Second derivatives
    grad_yy = sobel(grad_y, axis=0)
    grad_xx = sobel(grad_x, axis=1)
    grad_xy = sobel(grad_x, axis=0)
    
    # Hessian determinant (indicates edges)
    hessian_det = grad_yy * grad_xx - grad_xy**2
    
    # Bone edges have high gradient AND high second derivative
    sharp_edges = (grad_mag > np.percentile(grad_mag[mask], 80)) & \
                  (np.abs(hessian_det) > np.percentile(np.abs(hessian_det[mask]), 80))
    
    return sharp_edges, grad_mag, hessian_det


def analyze_radial_features(ct_slice, mask, metal_center):
    """
    Analyze features relative to metal center - artifacts are radial, bone is tangential.
    
    Args:
        ct_slice: 2D CT slice
        mask: Binary mask of candidate regions
        metal_center: (y, x) coordinates of metal center
        
    Returns:
        radial_score: How radial vs tangential the features are
    """
    # Compute gradients
    grad_y = sobel(ct_slice.astype(float), axis=0)
    grad_x = sobel(ct_slice.astype(float), axis=1)
    
    # Create coordinate grids
    y_coords, x_coords = np.ogrid[:ct_slice.shape[0], :ct_slice.shape[1]]
    
    # Radial vectors from metal center
    dy = y_coords - metal_center[0]
    dx = x_coords - metal_center[1]
    
    # Normalize radial vectors
    radial_dist = np.sqrt(dy**2 + dx**2) + 1e-6
    radial_y = dy / radial_dist
    radial_x = dx / radial_dist
    
    # Compute alignment between gradient and radial direction
    # Dot product: 1 = parallel (radial), 0 = perpendicular (tangential)
    gradient_norm = np.sqrt(grad_y**2 + grad_x**2) + 1e-6
    alignment = np.abs(grad_y * radial_y + grad_x * radial_x) / gradient_norm
    
    # Apply mask
    alignment_masked = alignment * mask
    
    return alignment_masked, radial_dist


def analyze_3d_continuity(ct_volume, z, mask_slice, dz_range=2):
    """
    Check structural continuity across slices - bone is continuous, artifacts jump.
    
    Args:
        ct_volume: 3D CT volume
        z: Current slice index
        mask_slice: 2D mask for current slice
        dz_range: Number of slices to check above/below
        
    Returns:
        continuity_score: Map of continuity scores
    """
    continuity_score = np.zeros_like(mask_slice, dtype=float)
    
    # For each pixel in the mask
    y_coords, x_coords = np.where(mask_slice)
    
    for y, x in zip(y_coords, x_coords):
        # Check continuity in adjacent slices
        continuous_count = 0
        hu_variance = []
        
        for dz in range(-dz_range, dz_range + 1):
            if dz == 0:
                continue
            z_check = z + dz
            if 0 <= z_check < ct_volume.shape[0]:
                # Check small neighborhood
                y_min = max(0, y-2)
                y_max = min(ct_volume.shape[1], y+3)
                x_min = max(0, x-2)
                x_max = min(ct_volume.shape[2], x+3)
                
                neighborhood = ct_volume[z_check, y_min:y_max, x_min:x_max]
                
                # Check if similar HU values exist
                current_hu = ct_volume[z, y, x]
                similar_hu = np.abs(neighborhood - current_hu) < 200
                
                if np.any(similar_hu):
                    continuous_count += 1
                    hu_variance.append(np.min(np.abs(neighborhood - current_hu)))
        
        # Continuity score based on how many slices show continuation
        if continuous_count > 0:
            continuity_score[y, x] = continuous_count / (2 * dz_range)
            # Bonus for consistent HU values
            if hu_variance:
                continuity_score[y, x] *= np.exp(-np.std(hu_variance) / 100)
    
    return continuity_score


def analyze_component_shape(binary_mask):
    """
    Analyze connected component shapes - bone is compact, artifacts are scattered.
    
    Args:
        binary_mask: Binary mask to analyze
        
    Returns:
        shape_features: Dictionary of shape features per component
    """
    labeled_mask, num_features = label(binary_mask)
    shape_features = {}
    
    if num_features == 0:
        print("WARNING: No connected components found in mask")
        return labeled_mask, shape_features
    
    for region in regionprops(labeled_mask):
        features = {
            'label': region.label,
            'area': region.area,
            'eccentricity': region.eccentricity,  # 0=circle, 1=line
            'solidity': region.solidity,  # Ratio of pixels to convex hull
            'extent': region.extent,  # Ratio of pixels to bounding box
            'major_axis': region.major_axis_length,
            'minor_axis': region.minor_axis_length,
            'orientation': region.orientation,
            'centroid': region.centroid
        }
        
        # Bone tends to be more solid and less eccentric than artifacts
        features['bone_likelihood'] = (
            features['solidity'] * (1 - features['eccentricity'] * 0.5) * 
            min(1.0, features['area'] / 100)  # Favor larger regions
        )
        
        shape_features[region.label] = features
    
    return labeled_mask, shape_features


def multi_scale_edge_detection(ct_slice, mask):
    """
    Detect edges at multiple scales - bone edges persist across scales.
    
    Args:
        ct_slice: 2D CT slice
        mask: Binary mask of candidate regions
        
    Returns:
        persistent_edges: Edges that appear at multiple scales
    """
    # Detect edges at different scales
    edges_fine = canny(ct_slice, sigma=0.5, low_threshold=100, high_threshold=200)
    edges_medium = canny(ct_slice, sigma=1.0, low_threshold=100, high_threshold=200)
    edges_coarse = canny(ct_slice, sigma=2.0, low_threshold=100, high_threshold=200)
    
    # Edges that persist across scales are more likely bone
    persistent_edges = edges_fine & edges_medium
    very_persistent = persistent_edges & edges_coarse
    
    # Apply mask
    persistent_edges = persistent_edges & mask
    very_persistent = very_persistent & mask
    
    return persistent_edges, very_persistent


def enhanced_bone_artifact_discrimination(ct_volume, metal_mask, spacing,
                                        bone_range=(300, 1500),
                                        max_distance_cm=10.0,
                                        progress_callback=None):
    """
    Enhanced discrimination using edge coherence and structural analysis.
    
    Args:
        ct_volume: 3D CT volume
        metal_mask: 3D binary mask of metal regions
        spacing: Voxel spacing (z, y, x) in mm
        bone_range: HU range for bone/bright artifacts
        max_distance_cm: Maximum distance from metal
        progress_callback: Optional callback function(progress, message) for progress updates
        
    Returns:
        dict: Enhanced discrimination results
    """
    start_time = time.time()
    print("Starting enhanced bone/artifact discrimination...")
    
    # Get distance map from metal
    if progress_callback:
        progress_callback(0.05, "Computing distance map from metal...")
    
    # Create distance map (inline to match artifact_discrimination_fast.py)
    inverted = np.logical_not(metal_mask)
    distances = distance_transform_edt(inverted, sampling=spacing)
    distance_map = distances / 10.0  # Convert to cm
    
    # Identify candidate voxels
    candidates_mask = (ct_volume >= bone_range[0]) & (ct_volume <= bone_range[1]) & \
                     (distance_map <= max_distance_cm) & (~metal_mask)
    
    print(f"Found {np.sum(candidates_mask):,} candidate voxels")
    
    # Initialize result masks
    bone_mask = np.zeros_like(ct_volume, dtype=bool)
    artifact_mask = np.zeros_like(ct_volume, dtype=bool)
    confidence_map = np.zeros_like(ct_volume, dtype=float)
    
    # Find metal centers for radial analysis
    if progress_callback:
        progress_callback(0.1, "Finding metal centers...")
    
    metal_centers = {}
    for z in range(metal_mask.shape[0]):
        if np.any(metal_mask[z]):
            y_coords, x_coords = np.where(metal_mask[z])
            metal_centers[z] = (np.mean(y_coords), np.mean(x_coords))
    
    # Check if we found any metal centers
    if not metal_centers:
        print("WARNING: No metal centers found in the volume!")
        # Return empty masks
        return {
            'bone_mask': bone_mask,
            'artifact_mask': artifact_mask,
            'confidence_map': confidence_map,
            'distance_map': distance_map
        }
    
    # Process each slice
    total_slices = ct_volume.shape[0]
    slices_with_candidates = sum(1 for z in range(total_slices) if np.any(candidates_mask[z]))
    processed_slices = 0
    
    for z in range(total_slices):
        slice_candidates = candidates_mask[z]
        
        if not np.any(slice_candidates):
            continue
        
        processed_slices += 1
            
        ct_slice = ct_volume[z]
        
        # Get metal center for this slice
        try:
            if z in metal_centers:
                metal_center = metal_centers[z]
            else:
                # Find nearest slice with metal
                if metal_centers:  # Check if we have any centers
                    nearest_z = min(metal_centers.keys(), key=lambda k: abs(k - z))
                    metal_center = metal_centers[nearest_z]
                else:
                    # Use center of slice as fallback
                    metal_center = (ct_slice.shape[0] // 2, ct_slice.shape[1] // 2)
        except Exception as e:
            print(f"WARNING: Error finding metal center for slice {z}: {e}")
            # Use center of slice as fallback
            metal_center = (ct_slice.shape[0] // 2, ct_slice.shape[1] // 2)
        
        # Update progress
        base_progress = 0.15
        slice_progress = 0.75  # 75% of progress for slice processing
        current_progress = base_progress + (z / total_slices) * slice_progress
        
        if progress_callback:
            progress_callback(current_progress, f"Processing slice {z+1}/{total_slices}...")
        else:
            print(f"Processing slice {z+1}/{total_slices}")
        
        # Initialize feature maps with defaults
        coherence_map = np.zeros_like(ct_slice, dtype=float)
        grad_mag = np.zeros_like(ct_slice, dtype=float)
        sharp_edges = np.zeros_like(ct_slice, dtype=bool)
        radial_alignment = np.zeros_like(ct_slice, dtype=float)
        persistent_edges = np.zeros_like(ct_slice, dtype=bool)
        continuity_score = np.zeros_like(ct_slice, dtype=float)
        
        try:
            # 1. Edge coherence analysis
            coherence_map, grad_mag, grad_dir = compute_edge_coherence(
                ct_slice, slice_candidates, window_size=7
            )
        except Exception as e:
            print(f"WARNING: Edge coherence failed for slice {z}: {e}")
        
        try:
            # 2. Gradient jump analysis
            sharp_edges, _, hessian = analyze_gradient_jumps(ct_slice, slice_candidates)
        except Exception as e:
            print(f"WARNING: Gradient jump analysis failed for slice {z}: {e}")
        
        try:
            # 3. Radial vs tangential features
            radial_alignment, radial_dist = analyze_radial_features(
                ct_slice, slice_candidates, metal_center
            )
        except Exception as e:
            print(f"WARNING: Radial feature analysis failed for slice {z}: {e}")
        
        try:
            # 4. Multi-scale edge persistence
            persistent_edges, very_persistent = multi_scale_edge_detection(
                ct_slice, slice_candidates
            )
        except Exception as e:
            print(f"WARNING: Multi-scale edge detection failed for slice {z}: {e}")
        
        try:
            # 5. 3D continuity (only check every few slices for speed)
            if z % 3 == 0:  # Check every 3rd slice
                continuity_score = analyze_3d_continuity(ct_volume, z, slice_candidates)
        except Exception as e:
            print(f"WARNING: 3D continuity analysis failed for slice {z}: {e}")
        
        # 6. Component shape analysis
        try:
            labeled_mask, shape_features = analyze_component_shape(slice_candidates)
        except Exception as e:
            print(f"WARNING: Error analyzing component shapes for slice {z}: {e}")
            continue
        
        # Combine features for discrimination
        for label_id, features in shape_features.items():
            component_mask = labeled_mask == label_id
            
            # Calculate feature scores with error handling
            try:
                coherence_score = np.mean(coherence_map[component_mask]) if np.any(component_mask) else 0
                edge_persistence = np.sum(persistent_edges & component_mask) / features['area'] if features['area'] > 0 else 0
                radial_score = np.mean(radial_alignment[component_mask]) if np.any(component_mask) else 0.5
                continuity_mean = np.mean(continuity_score[component_mask]) if np.any(component_mask) else 0
                has_sharp_edges = np.sum(sharp_edges & component_mask) / features['area'] if features['area'] > 0 else 0
            except Exception as e:
                print(f"WARNING: Error calculating features for component {label_id} on slice {z}: {e}")
                continue
            
            # Bone characteristics:
            # - High edge coherence (>0.7)
            # - Edge persistence across scales
            # - Low radial alignment (<0.4)
            # - High 3D continuity
            # - Compact shape (high solidity)
            # - Sharp, well-defined edges
            
            bone_score = (
                coherence_score * 2.0 +  # Weight coherence heavily
                edge_persistence * 1.5 +
                (1 - radial_score) * 1.5 +  # Tangential features
                continuity_mean * 1.0 +
                features['solidity'] * 1.0 +
                has_sharp_edges * 0.5
            )
            
            # Artifact characteristics:
            # - Low edge coherence
            # - Radial alignment
            # - Poor 3D continuity
            # - Scattered shape
            
            artifact_score = (
                (1 - coherence_score) * 2.0 +
                radial_score * 2.0 +
                (1 - continuity_mean) * 1.0 +
                (1 - features['solidity']) * 1.0 +
                features['eccentricity'] * 0.5
            )
            
            # Normalize scores
            total_score = bone_score + artifact_score
            if total_score > 0:
                bone_prob = bone_score / total_score
                
                # Classify based on probability
                if bone_prob > 0.65:
                    bone_mask[z][component_mask] = True
                    confidence_map[z][component_mask] = bone_prob
                elif bone_prob < 0.35:
                    artifact_mask[z][component_mask] = True
                    confidence_map[z][component_mask] = 1 - bone_prob
                else:
                    # Ambiguous - use distance as tiebreaker
                    avg_distance = np.mean(distance_map[z][component_mask])
                    if avg_distance < 2.5:  # Close to metal
                        bone_mask[z][component_mask] = True
                        confidence_map[z][component_mask] = 0.5
                    else:
                        artifact_mask[z][component_mask] = True
                        confidence_map[z][component_mask] = 0.5
    
    # Post-processing
    if progress_callback:
        progress_callback(0.9, "Applying post-processing...")
    else:
        print("Applying post-processing...")
    
    # Clean up small isolated regions - but be conservative
    # Only apply if we have sufficient voxels to work with
    bone_voxels_before = np.sum(bone_mask)
    artifact_voxels_before = np.sum(artifact_mask)
    
    if bone_voxels_before > 1000:  # Only clean up if we have enough voxels
        bone_mask = binary_erosion(bone_mask, iterations=1)
        bone_mask = binary_dilation(bone_mask, iterations=2)
        bone_mask = binary_erosion(bone_mask, iterations=1)
    
    print(f"Post-processing: Bone {bone_voxels_before:,} -> {np.sum(bone_mask):,}")
    print(f"Post-processing: Artifacts {artifact_voxels_before:,} -> {np.sum(artifact_mask):,}")
    
    # Validation check - if we have no results, try a fallback
    total_results = np.sum(bone_mask) + np.sum(artifact_mask)
    if total_results == 0 and np.sum(candidates_mask) > 0:
        print("WARNING: No voxels classified! Using distance-based fallback...")
        # Simple distance-based fallback
        bone_mask = candidates_mask & (distance_map < 2.0)
        artifact_mask = candidates_mask & (distance_map >= 2.0) & (distance_map <= max_distance_cm)
        print(f"Fallback: {np.sum(bone_mask):,} bone, {np.sum(artifact_mask):,} artifact voxels")
    
    elapsed_time = time.time() - start_time
    
    if progress_callback:
        progress_callback(1.0, f"Completed in {elapsed_time:.1f}s - Found {np.sum(bone_mask):,} bone, {np.sum(artifact_mask):,} artifact voxels")
    else:
        print(f"Enhanced discrimination completed in {elapsed_time:.1f} seconds")
        print(f"Found {np.sum(bone_mask):,} bone voxels and {np.sum(artifact_mask):,} artifact voxels")
    
    return {
        'bone_mask': bone_mask,
        'artifact_mask': artifact_mask,
        'confidence_map': confidence_map,
        'distance_map': distance_map
    }


def create_enhanced_russian_doll_segmentation(ct_volume, metal_mask, spacing,
                                            dark_range=(-1024, -150),
                                            bone_range=(300, 1500),
                                            max_distance_cm=10.0,
                                            progress_callback=None):
    """
    Russian doll segmentation with enhanced bone/artifact discrimination.
    
    Args:
        ct_volume: 3D CT volume
        metal_mask: Already segmented metal mask
        spacing: Voxel spacing
        dark_range: HU range for dark artifacts
        bone_range: HU range for bone/bright artifacts
        max_distance_cm: Max distance from metal
        progress_callback: Optional callback function(progress, message) for progress updates
        
    Returns:
        dict: All segmentation masks
    """
    print("Starting enhanced Russian doll segmentation...")
    
    # Step 1: Dark artifacts (simple threshold)
    if progress_callback:
        progress_callback(0, "Step 1: Segmenting dark artifacts...")
    else:
        print("\nStep 1: Segmenting dark artifacts...")
    
    dark_mask = (ct_volume >= dark_range[0]) & (ct_volume <= dark_range[1]) & (~metal_mask)
    
    # Step 2: Enhanced bone/bright discrimination
    if progress_callback:
        progress_callback(0.02, "Step 2: Enhanced bone/bright artifact discrimination...")
    else:
        print("\nStep 2: Enhanced bone/bright artifact discrimination...")
    
    discrimination_result = enhanced_bone_artifact_discrimination(
        ct_volume, metal_mask, spacing,
        bone_range=bone_range,
        max_distance_cm=max_distance_cm,
        progress_callback=progress_callback
    )
    
    bone_mask = discrimination_result['bone_mask']
    bright_artifact_mask = discrimination_result['artifact_mask']
    
    # Step 3: Ensure mutual exclusivity
    bone_mask = bone_mask & (~metal_mask) & (~dark_mask)
    bright_artifact_mask = bright_artifact_mask & (~metal_mask) & (~dark_mask) & (~bone_mask)
    
    # Handle any remaining candidates
    remaining_candidates = (ct_volume >= bone_range[0]) & (ct_volume <= bone_range[1]) & \
                          (~metal_mask) & (~dark_mask) & (~bone_mask) & (~bright_artifact_mask)
    
    if np.any(remaining_candidates):
        print(f"Assigning {np.sum(remaining_candidates):,} remaining voxels...")
        # Use simple distance criterion for remaining
        distances_cm = discrimination_result['distance_map']
        bone_mask |= remaining_candidates & (distances_cm < 2.0)
        bright_artifact_mask |= remaining_candidates & (distances_cm >= 2.0)
    
    return {
        'metal': metal_mask,
        'dark_artifacts': dark_mask,
        'bone': bone_mask,
        'bright_artifacts': bright_artifact_mask,
        'confidence_map': discrimination_result['confidence_map'],
        'distance_map': discrimination_result['distance_map']
    }