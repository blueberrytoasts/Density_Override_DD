import numpy as np
try:
    import cupy as cp  # GPU acceleration
    from cupyx.scipy import ndimage as cp_ndimage
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    cp_ndimage = None
    GPU_AVAILABLE = False
    # Silently fall back to CPU processing
    
from scipy.signal import find_peaks
from scipy.ndimage import distance_transform_edt
import time


def create_distance_from_metal_mask(metal_mask, max_distance_cm=10.0, spacing=(1.0, 1.0, 1.0)):
    """
    Create a distance map from metal regions using GPU acceleration if available.
    
    Args:
        metal_mask: 3D binary mask of metal regions
        max_distance_cm: Maximum distance to consider (cm)
        spacing: Voxel spacing in mm
        
    Returns:
        distance_mask: 3D array with distances from metal
    """
    print("Computing distance from metal regions...")
    
    if GPU_AVAILABLE:
        # Convert to GPU array
        metal_gpu = cp.asarray(metal_mask)
        
        # Compute distance transform on GPU
        # Invert mask so metal is 0 and non-metal is 1
        inverted = cp.logical_not(metal_gpu)
        
        # Distance transform
        distances_gpu = cp_ndimage.distance_transform_edt(inverted, sampling=spacing)
        
        # Convert back to CPU
        distances = cp.asnumpy(distances_gpu)
    else:
        # CPU fallback
        inverted = np.logical_not(metal_mask)
        distances = distance_transform_edt(inverted, sampling=spacing)
    
    # Convert to cm
    distances_cm = distances / 10.0
    
    return distances_cm


def analyze_profile_characteristics(distances, hu_values):
    """
    Analyze profile characteristics to distinguish bone from artifacts.
    
    Args:
        distances: Distance array along profile
        hu_values: HU values along profile
        
    Returns:
        dict: Profile characteristics
    """
    if len(hu_values) < 5:
        return None
    
    # Find peaks in the profile
    peaks, properties = find_peaks(hu_values, height=300, prominence=100)
    
    if len(peaks) == 0:
        return None
    
    characteristics = {}
    
    # 1. Peak characteristics
    peak_widths = []
    peak_symmetries = []
    
    for peak_idx in peaks:
        # Find FWHM (Full Width Half Maximum)
        peak_val = hu_values[peak_idx]
        half_max = peak_val / 2
        
        # Find left and right indices at half maximum
        left_idx = peak_idx
        right_idx = peak_idx
        
        while left_idx > 0 and hu_values[left_idx] > half_max:
            left_idx -= 1
        while right_idx < len(hu_values) - 1 and hu_values[right_idx] > half_max:
            right_idx += 1
            
        width = distances[right_idx] - distances[left_idx] if right_idx > left_idx else 0
        peak_widths.append(width)
        
        # Calculate symmetry (ratio of rise to fall slopes)
        if peak_idx > 0 and peak_idx < len(hu_values) - 1:
            left_slope = (hu_values[peak_idx] - hu_values[left_idx]) / max(peak_idx - left_idx, 1)
            right_slope = (hu_values[peak_idx] - hu_values[right_idx]) / max(right_idx - peak_idx, 1)
            symmetry = min(abs(left_slope), abs(right_slope)) / max(abs(left_slope), abs(right_slope), 1e-6)
            peak_symmetries.append(symmetry)
    
    # 2. Smoothness (variance of first derivative)
    if len(hu_values) > 1:
        first_derivative = np.diff(hu_values)
        smoothness = 1.0 / (1.0 + np.var(first_derivative))
    else:
        smoothness = 0
    
    # 3. Edge sharpness (maximum gradient)
    max_gradient = np.max(np.abs(first_derivative)) if len(hu_values) > 1 else 0
    
    characteristics['peak_widths'] = peak_widths
    characteristics['avg_width'] = np.mean(peak_widths) if peak_widths else 0
    characteristics['symmetry'] = np.mean(peak_symmetries) if peak_symmetries else 0
    characteristics['smoothness'] = smoothness
    characteristics['max_gradient'] = max_gradient
    characteristics['num_peaks'] = len(peaks)
    
    return characteristics


def classify_bone_vs_artifact(profile_characteristics_list):
    """
    Classify whether profiles indicate bone or artifact.
    
    Args:
        profile_characteristics_list: List of profile characteristics from multiple directions
        
    Returns:
        dict: Classification result with confidence
    """
    # Filter out None results
    valid_profiles = [p for p in profile_characteristics_list if p is not None]
    
    if len(valid_profiles) < 4:  # Need at least 4 valid profiles
        return {'classification': 'uncertain', 'confidence': 0.0}
    
    # Calculate directional variance (key discriminator)
    widths = [p['avg_width'] for p in valid_profiles if p['avg_width'] > 0]
    smoothnesses = [p['smoothness'] for p in valid_profiles]
    gradients = [p['max_gradient'] for p in valid_profiles]
    
    # High variance in characteristics across directions indicates artifact
    width_variance = np.var(widths) if len(widths) > 1 else 0
    smoothness_variance = np.var(smoothnesses) if smoothnesses else 0
    gradient_variance = np.var(gradients) if gradients else 0
    
    # Calculate scores
    bone_score = 0
    artifact_score = 0
    
    # Width analysis (bone has consistent, broader peaks)
    avg_width = np.mean(widths) if widths else 0
    if avg_width > 3.0:  # Broad peaks (>3mm)
        bone_score += 2
    elif avg_width < 1.5:  # Narrow peaks (<1.5mm)
        artifact_score += 2
        
    # Consistency analysis (bone is consistent across directions)
    if width_variance < 2.0:
        bone_score += 2
    else:
        artifact_score += 2
        
    # Smoothness analysis (bone has smoother transitions)
    avg_smoothness = np.mean(smoothnesses)
    if avg_smoothness > 0.7:
        bone_score += 1
    else:
        artifact_score += 1
        
    # Gradient analysis (artifacts have sharper edges)
    avg_gradient = np.mean(gradients)
    if avg_gradient < 500:
        bone_score += 1
    else:
        artifact_score += 1
    
    # Final classification
    total_score = bone_score + artifact_score
    if total_score == 0:
        return {'classification': 'uncertain', 'confidence': 0.0}
        
    bone_confidence = bone_score / total_score
    
    if bone_confidence > 0.65:
        classification = 'bone'
    elif bone_confidence < 0.35:
        classification = 'artifact'
    else:
        classification = 'uncertain'
    
    return {
        'classification': classification,
        'confidence': max(bone_confidence, 1 - bone_confidence),
        'bone_score': bone_score,
        'artifact_score': artifact_score,
        'metrics': {
            'avg_width': avg_width,
            'width_variance': width_variance,
            'avg_smoothness': avg_smoothness,
            'avg_gradient': avg_gradient
        }
    }


def discriminate_bone_from_artifacts_gpu(ct_volume, metal_mask, spacing, 
                                         hu_range=(300, 3000), 
                                         max_distance_cm=10.0,
                                         batch_size=10000):
    """
    Main function to discriminate bone from bright artifacts using GPU acceleration.
    
    Args:
        ct_volume: 3D CT volume
        metal_mask: 3D binary mask of metal regions
        spacing: Voxel spacing (z, y, x) in mm
        hu_range: HU range to analyze
        max_distance_cm: Maximum distance from metal to consider
        batch_size: Number of voxels to process in parallel
        
    Returns:
        dict: Contains bone_mask and artifact_mask
    """
    start_time = time.time()
    
    # Step 1: Create distance map from metal
    distance_map = create_distance_from_metal_mask(metal_mask, max_distance_cm, spacing)
    
    # Step 2: Identify candidate voxels (in HU range and near metal)
    candidates_mask = (ct_volume >= hu_range[0]) & (ct_volume <= hu_range[1]) & \
                     (distance_map <= max_distance_cm) & (~metal_mask)
    
    candidate_coords = np.argwhere(candidates_mask)
    print(f"Found {len(candidate_coords):,} candidate voxels to analyze")
    
    # Step 3: Initialize result masks
    bone_mask = np.zeros_like(ct_volume, dtype=bool)
    artifact_mask = np.zeros_like(ct_volume, dtype=bool)
    confidence_map = np.zeros_like(ct_volume, dtype=float)
    
    # Step 4: Process candidates in batches
    from metal_detection_v3 import get_star_profile_lines
    
    for batch_start in range(0, len(candidate_coords), batch_size):
        batch_end = min(batch_start + batch_size, len(candidate_coords))
        batch_coords = candidate_coords[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(len(candidate_coords)-1)//batch_size + 1}")
        
        for coord in batch_coords:
            z, y, x = coord
            
            # Define search bounds for this voxel
            search_size = 20  # pixels
            bounds = {
                'y_min': max(0, y - search_size),
                'y_max': min(ct_volume.shape[1], y + search_size),
                'x_min': max(0, x - search_size),
                'x_max': min(ct_volume.shape[2], x + search_size)
            }
            
            # Get star profiles
            profiles = get_star_profile_lines(ct_volume[z], y, x, bounds)
            
            # Analyze each profile
            profile_characteristics = []
            for distances, hu_values in profiles:
                chars = analyze_profile_characteristics(distances, hu_values)
                if chars:
                    profile_characteristics.append(chars)
            
            # Classify based on profiles
            classification = classify_bone_vs_artifact(profile_characteristics)
            
            # Update masks based on classification
            if classification['classification'] == 'bone':
                bone_mask[z, y, x] = True
            elif classification['classification'] == 'artifact':
                artifact_mask[z, y, x] = True
            
            confidence_map[z, y, x] = classification['confidence']
    
    # Step 5: Post-processing - fill small holes and remove isolated pixels
    from scipy.ndimage import binary_closing, binary_opening
    
    bone_mask = binary_closing(bone_mask, iterations=2)
    bone_mask = binary_opening(bone_mask, iterations=1)
    
    artifact_mask = binary_closing(artifact_mask, iterations=2)
    artifact_mask = binary_opening(artifact_mask, iterations=1)
    
    elapsed_time = time.time() - start_time
    print(f"Discrimination completed in {elapsed_time:.1f} seconds")
    print(f"Identified {np.sum(bone_mask):,} bone voxels and {np.sum(artifact_mask):,} artifact voxels")
    
    return {
        'bone_mask': bone_mask,
        'artifact_mask': artifact_mask,
        'confidence_map': confidence_map,
        'distance_map': distance_map
    }


def create_sequential_masks(ct_volume, metal_mask, spacing, 
                           dark_range=(-1024, -150),
                           bone_range=(300, 1500),
                           bright_artifact_max_distance_cm=10.0):
    """
    Create masks using the Russian doll approach with proper exclusions.
    
    Args:
        ct_volume: 3D CT volume
        metal_mask: Already segmented metal mask
        spacing: Voxel spacing
        dark_range: HU range for dark artifacts
        bone_range: HU range for potential bone
        bright_artifact_max_distance_cm: Max distance from metal for artifacts
        
    Returns:
        dict: All segmentation masks
    """
    print("Starting Russian doll segmentation...")
    
    # Step 1: Dark artifacts (excluding metal)
    print("\nStep 1: Segmenting dark artifacts...")
    dark_mask = (ct_volume >= dark_range[0]) & (ct_volume <= dark_range[1]) & (~metal_mask)
    
    # Step 2: Discriminate bone from bright artifacts
    print("\nStep 2: Discriminating bone from bright artifacts...")
    discrimination_result = discriminate_bone_from_artifacts_gpu(
        ct_volume, metal_mask, spacing,
        hu_range=bone_range,
        max_distance_cm=bright_artifact_max_distance_cm
    )
    
    bone_mask = discrimination_result['bone_mask']
    bright_artifact_mask = discrimination_result['artifact_mask']
    
    # Step 3: Ensure mutual exclusivity
    bone_mask = bone_mask & (~metal_mask) & (~dark_mask)
    bright_artifact_mask = bright_artifact_mask & (~metal_mask) & (~dark_mask) & (~bone_mask)
    
    return {
        'metal': metal_mask,
        'dark_artifacts': dark_mask,
        'bone': bone_mask,
        'bright_artifacts': bright_artifact_mask,
        'confidence_map': discrimination_result['confidence_map'],
        'distance_map': discrimination_result['distance_map']
    }