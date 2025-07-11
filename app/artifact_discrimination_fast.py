import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.ndimage import label, binary_dilation
import time


def create_simplified_discrimination(ct_volume, metal_mask, spacing, 
                                   bone_range=(300, 1500),
                                   max_distance_cm=10.0):
    """
    Simplified but faster discrimination between bone and bright artifacts.
    Uses distance from metal and smoothness as primary discriminators.
    
    Args:
        ct_volume: 3D CT volume
        metal_mask: 3D binary mask of metal regions
        spacing: Voxel spacing (z, y, x) in mm
        bone_range: HU range for bone/bright artifacts
        max_distance_cm: Maximum distance from metal to consider
        
    Returns:
        dict: Contains bone_mask and artifact_mask
    """
    start_time = time.time()
    print("Starting simplified bone/artifact discrimination...")
    
    # Step 1: Create distance map from metal
    print("Computing distance from metal...")
    inverted = np.logical_not(metal_mask)
    distances = distance_transform_edt(inverted, sampling=spacing)
    distances_cm = distances / 10.0
    
    # Step 2: Get candidate regions
    candidates_mask = (ct_volume >= bone_range[0]) & (ct_volume <= bone_range[1]) & \
                     (distances_cm <= max_distance_cm) & (~metal_mask)
    
    print(f"Found {np.sum(candidates_mask):,} candidate voxels")
    
    # Step 3: Apply smoothness-based discrimination
    print("Applying smoothness discrimination...")
    
    # Smooth the CT volume to identify bone regions
    # Bone will remain relatively high after smoothing, artifacts will be reduced
    smoothed = gaussian_filter(ct_volume.astype(float), sigma=2.0)
    
    # Calculate local variance (artifacts have high variance)
    local_std = np.zeros_like(ct_volume, dtype=float)
    for z in range(1, ct_volume.shape[0]-1):
        slice_std = np.std([ct_volume[z-1], ct_volume[z], ct_volume[z+1]], axis=0)
        local_std[z] = slice_std
    
    # Step 4: Classification rules (CORRECTED)
    # Bone: adjacent to metal, high HU after smoothing, low local variance
    # Artifacts: extend outward from bone, high variance, variable HU
    
    # Distance-based regions (ANATOMICALLY CORRECT)
    # Bone (femur/acetabulum) is RIGHT NEXT to metal (0.5-2.5cm)
    # Bright artifacts are the STREAKS extending outward (variable distance)
    bone_zone = (distances_cm >= 0.5) & (distances_cm < 2.5)
    artifact_zone = (distances_cm >= 0.3) & (distances_cm <= max_distance_cm)
    
    # Initialize masks
    bone_mask = np.zeros_like(ct_volume, dtype=bool)
    artifact_mask = np.zeros_like(ct_volume, dtype=bool)
    
    # TRUE BONE characteristics (femur/acetabulum):
    # - Adjacent to metal implant
    # - CONSISTENT high HU (true cortical/trabecular bone)
    # - Maintains structure after smoothing
    # - Low variance between slices
    # - Anatomically expected location
    true_bone = candidates_mask & bone_zone & \
                (smoothed > 600) & (local_std < 150) & \
                (ct_volume > 500)  # True bone has consistently high HU
    
    # TRUE BRIGHT ARTIFACT characteristics (beam hardening streaks):
    # - Can be anywhere in the FOV
    # - HIGH VARIANCE (streaking pattern)
    # - Variable/inconsistent HU values
    # - Changes dramatically between slices
    # - Often radial/directional patterns
    true_artifacts = candidates_mask & \
                    ((local_std > 200) |  # High variance
                     ((smoothed < 0.8 * ct_volume) & (ct_volume > 400)) |  # HU drops with smoothing
                     (artifact_zone & (local_std > 100)))  # Moderate variance in artifact zone
    
    # Initial assignment with CORRECT anatomy
    bone_mask = true_bone
    artifact_mask = true_artifacts & (~bone_mask)  # Artifacts exclude bone
    
    # Handle unclassified candidates
    unclassified = candidates_mask & (~bone_mask) & (~artifact_mask)
    
    if np.any(unclassified):
        # Additional criteria for unclassified regions
        # Near metal + consistent HU = bone
        # High variance anywhere = artifact
        
        # Bone criteria: close to metal, high persistent HU, low variance
        additional_bone = unclassified & bone_zone & \
                         (smoothed > 500) & (local_std < 180)
        
        # Artifact criteria: high variance or inconsistent HU
        additional_artifacts = unclassified & \
                             ((local_std > 150) | \
                              (np.abs(smoothed - ct_volume) > 200))
        
        bone_mask |= additional_bone
        artifact_mask |= additional_artifacts & (~bone_mask)
        
        # Final assignment for remaining: use distance
        still_unclassified = unclassified & (~additional_bone) & (~additional_artifacts)
        if np.any(still_unclassified):
            # Very close to metal -> probably bone
            # Far from metal -> probably artifact
            bone_mask |= still_unclassified & (distances_cm < 2.0)
            artifact_mask |= still_unclassified & (distances_cm >= 2.0)
    
    # Post-processing to clean up
    bone_mask = binary_dilation(bone_mask, iterations=1)
    
    # Create confidence map based on corrected distance logic
    confidence_map = np.zeros_like(ct_volume, dtype=float)
    # Bone confidence: highest when 1-2cm from metal
    if np.any(bone_mask):
        bone_distances = distances_cm[bone_mask]
        # Peak confidence at 1.5cm, decreasing outward
        bone_confidence = np.exp(-((bone_distances - 1.5) ** 2) / 2.0)
        confidence_map[bone_mask] = np.clip(bone_confidence, 0.5, 1.0)
    
    # Artifact confidence: increases with distance from metal (2-10cm)
    if np.any(artifact_mask):
        artifact_distances = distances_cm[artifact_mask]
        # Higher confidence for artifacts farther from metal
        artifact_confidence = (artifact_distances - 2.0) / 8.0
        confidence_map[artifact_mask] = np.clip(artifact_confidence, 0.5, 1.0)
    
    elapsed = time.time() - start_time
    print(f"Discrimination completed in {elapsed:.1f} seconds")
    print(f"Found {np.sum(bone_mask):,} bone voxels and {np.sum(artifact_mask):,} artifact voxels")
    
    return {
        'bone_mask': bone_mask,
        'artifact_mask': artifact_mask,
        'confidence_map': confidence_map,
        'distance_map': distances_cm
    }


def create_fast_russian_doll_segmentation(ct_volume, metal_mask, spacing, 
                                        dark_range=(-1024, -150),
                                        bone_range=(300, 1500),
                                        max_distance_cm=10.0):
    """
    Fast Russian doll segmentation using simplified discrimination.
    
    Args:
        ct_volume: 3D CT volume
        metal_mask: Already segmented metal mask
        spacing: Voxel spacing
        dark_range: HU range for dark artifacts
        bone_range: HU range for potential bone/bright artifacts
        max_distance_cm: Max distance from metal for artifacts
        
    Returns:
        dict: All segmentation masks
    """
    print("Starting fast Russian doll segmentation...")
    
    # Step 1: Dark artifacts (simple threshold)
    print("\nStep 1: Segmenting dark artifacts...")
    dark_mask = (ct_volume >= dark_range[0]) & (ct_volume <= dark_range[1]) & (~metal_mask)
    
    # Step 2: Fast bone/bright discrimination
    print("\nStep 2: Fast bone/bright artifact discrimination...")
    discrimination_result = create_simplified_discrimination(
        ct_volume, metal_mask, spacing,
        bone_range=bone_range,
        max_distance_cm=max_distance_cm
    )
    
    bone_mask = discrimination_result['bone_mask']
    bright_artifact_mask = discrimination_result['artifact_mask']
    
    # Step 3: Ensure mutual exclusivity
    bone_mask = bone_mask & (~metal_mask) & (~dark_mask)
    bright_artifact_mask = bright_artifact_mask & (~metal_mask) & (~dark_mask) & (~bone_mask)
    
    # Additional cleanup for any remaining candidates
    remaining_candidates = (ct_volume >= bone_range[0]) & (ct_volume <= bone_range[1]) & \
                          (~metal_mask) & (~dark_mask) & (~bone_mask) & (~bright_artifact_mask)
    
    if np.any(remaining_candidates):
        # Assign remaining to bone if far from metal, otherwise to artifacts
        distances_cm = discrimination_result['distance_map']
        far_remaining = remaining_candidates & (distances_cm > 5.0)
        close_remaining = remaining_candidates & (distances_cm <= 5.0)
        
        bone_mask |= far_remaining
        bright_artifact_mask |= close_remaining
    
    return {
        'metal': metal_mask,
        'dark_artifacts': dark_mask,
        'bone': bone_mask,
        'bright_artifacts': bright_artifact_mask,
        'confidence_map': discrimination_result['confidence_map'],
        'distance_map': discrimination_result['distance_map']
    }