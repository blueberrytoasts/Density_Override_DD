import numpy as np
from scipy.ndimage import label, binary_dilation, gaussian_filter
from scipy.signal import convolve2d
from skimage.measure import regionprops
import cv2


def analyze_contour_characteristics(mask_slice, ct_slice):
    """
    Analyze characteristics of segmented regions to determine if they're bone or artifact.
    
    Returns dict of features for each connected component.
    """
    labeled_mask, num_features = label(mask_slice)
    characteristics = {}
    
    for region in regionprops(labeled_mask, intensity_image=ct_slice):
        # Get region properties
        coords = region.coords
        
        # 1. Intensity statistics
        intensities = ct_slice[coords[:, 0], coords[:, 1]]
        intensity_mean = np.mean(intensities)
        intensity_std = np.std(intensities)
        intensity_cv = intensity_std / (intensity_mean + 1e-6)  # Coefficient of variation
        
        # 2. Edge characteristics - bone has sharp edges, artifacts are diffuse
        edge_mask = np.zeros_like(mask_slice)
        edge_mask[coords[:, 0], coords[:, 1]] = 1
        edges = cv2.Canny(edge_mask.astype(np.uint8) * 255, 50, 150)
        edge_ratio = np.sum(edges > 0) / region.area
        
        # 3. Shape regularity - bone is more regular
        solidity = region.solidity
        eccentricity = region.eccentricity
        
        # 4. Gradient analysis along principal axis
        # Bone has consistent gradients, artifacts have chaotic gradients
        y0, x0 = region.centroid
        orientation = region.orientation
        
        # Sample along major axis
        major_length = int(region.major_axis_length)
        if major_length > 10:
            # Create line along major axis
            cos_angle = np.cos(orientation)
            sin_angle = np.sin(orientation)
            
            # Sample points along major axis
            t = np.linspace(-major_length/2, major_length/2, major_length)
            x_line = x0 + t * cos_angle
            y_line = y0 - t * sin_angle  # Negative because y increases downward
            
            # Get valid points within image bounds
            valid_mask = (x_line >= 0) & (x_line < ct_slice.shape[1]) & \
                        (y_line >= 0) & (y_line < ct_slice.shape[0])
            
            if np.sum(valid_mask) > 5:
                x_valid = x_line[valid_mask].astype(int)
                y_valid = y_line[valid_mask].astype(int)
                
                # Sample HU values along line
                line_values = ct_slice[y_valid, x_valid]
                
                # Calculate gradient consistency
                if len(line_values) > 3:
                    gradients = np.diff(line_values)
                    gradient_consistency = 1.0 - (np.std(gradients) / (np.mean(np.abs(gradients)) + 1e-6))
                else:
                    gradient_consistency = 0.5
            else:
                gradient_consistency = 0.5
        else:
            gradient_consistency = 0.5
        
        # 5. Radial pattern detection (artifacts often radiate from metal)
        # This will be computed globally later
        
        characteristics[region.label] = {
            'area': region.area,
            'centroid': region.centroid,
            'intensity_mean': intensity_mean,
            'intensity_std': intensity_std,
            'intensity_cv': intensity_cv,
            'edge_ratio': edge_ratio,
            'solidity': solidity,
            'eccentricity': eccentricity,
            'gradient_consistency': gradient_consistency,
            'bbox': region.bbox
        }
    
    return labeled_mask, characteristics


def detect_radial_patterns(mask_slice, metal_mask_slice):
    """
    Detect if regions follow radial patterns from metal (characteristic of artifacts).
    """
    # Find metal center
    metal_coords = np.where(metal_mask_slice)
    if len(metal_coords[0]) == 0:
        return None
    
    metal_center_y = np.mean(metal_coords[0])
    metal_center_x = np.mean(metal_coords[1])
    
    # For each region, check if it's aligned radially
    labeled_mask, _ = label(mask_slice)
    radial_scores = {}
    
    for region in regionprops(labeled_mask):
        y, x = region.centroid
        
        # Vector from metal center to region
        dy = y - metal_center_y
        dx = x - metal_center_x
        
        # Normalize
        length = np.sqrt(dy**2 + dx**2)
        if length > 0:
            dy /= length
            dx /= length
            
            # Check if region is elongated along radial direction
            if region.major_axis_length > region.minor_axis_length * 1.5:
                # Get orientation vector
                orientation = region.orientation
                orient_dy = -np.sin(orientation)
                orient_dx = np.cos(orientation)
                
                # Dot product gives alignment (1 = perfectly aligned)
                alignment = abs(dy * orient_dy + dx * orient_dx)
                radial_scores[region.label] = alignment
            else:
                radial_scores[region.label] = 0.0
        else:
            radial_scores[region.label] = 0.0
    
    return radial_scores, (metal_center_y, metal_center_x)


def compute_3d_consistency(mask_3d, z_slice, label_id, bbox):
    """
    Check if a region is consistent across slices (bone) or jumps around (artifact).
    """
    # Get bbox for efficiency
    min_row, min_col, max_row, max_col = bbox
    
    # Check slices above and below
    consistency_score = 0
    overlap_count = 0
    
    for dz in [-2, -1, 1, 2]:
        z_check = z_slice + dz
        if 0 <= z_check < mask_3d.shape[0]:
            # Get slice region
            slice_region = mask_3d[z_check, min_row:max_row, min_col:max_col]
            current_region = mask_3d[z_slice, min_row:max_row, min_col:max_col] == label_id
            
            if np.any(slice_region) and np.any(current_region):
                # Calculate overlap
                overlap = np.sum(slice_region & current_region)
                union = np.sum(slice_region | current_region)
                if union > 0:
                    consistency_score += overlap / union
                    overlap_count += 1
    
    if overlap_count > 0:
        consistency_score /= overlap_count
    
    return consistency_score


def refine_bone_artifact_discrimination(ct_volume, masks_dict, metal_mask, spacing):
    """
    Second-pass refinement of bone vs artifact discrimination.
    
    Args:
        ct_volume: 3D CT data
        masks_dict: Dictionary with 'bone' and 'bright_artifacts' masks
        metal_mask: 3D metal mask
        spacing: Voxel spacing
        
    Returns:
        dict: Refined masks
    """
    print("\nStarting second-pass refinement of bone/artifact discrimination...")
    
    bone_mask = masks_dict['bone'].copy()
    artifact_mask = masks_dict['bright_artifacts'].copy()
    
    # Track changes
    bone_to_artifact = np.zeros_like(bone_mask, dtype=bool)
    artifact_to_bone = np.zeros_like(artifact_mask, dtype=bool)
    
    # Process each slice
    for z in range(ct_volume.shape[0]):
        if not (np.any(bone_mask[z]) or np.any(artifact_mask[z])):
            continue
        
        ct_slice = ct_volume[z]
        bone_slice = bone_mask[z]
        artifact_slice = artifact_mask[z]
        metal_slice = metal_mask[z]
        
        # Analyze bone regions
        if np.any(bone_slice):
            labeled_bone, bone_chars = analyze_contour_characteristics(bone_slice, ct_slice)
            
            # Get radial patterns if metal present
            radial_scores = None
            if np.any(metal_slice):
                radial_scores, metal_center = detect_radial_patterns(bone_slice, metal_slice)
            
            # Check each bone region
            for label_id, chars in bone_chars.items():
                # Criteria for reclassifying bone -> artifact
                is_artifact = False
                
                # 1. High intensity variation (artifacts are inconsistent)
                if chars['intensity_cv'] > 0.4:  # 40% coefficient of variation
                    is_artifact = True
                
                # 2. Low edge ratio (artifacts are diffuse)
                if chars['edge_ratio'] < 0.1:
                    is_artifact = True
                
                # 3. High eccentricity + radial alignment (streak pattern)
                if radial_scores and label_id in radial_scores:
                    if chars['eccentricity'] > 0.8 and radial_scores[label_id] > 0.7:
                        is_artifact = True
                
                # 4. Poor gradient consistency
                if chars['gradient_consistency'] < 0.3:
                    is_artifact = True
                
                # 5. Check 3D consistency
                consistency = compute_3d_consistency(bone_mask, z, label_id, chars['bbox'])
                if consistency < 0.2:  # Poor 3D consistency
                    is_artifact = True
                
                # Mark for reclassification
                if is_artifact:
                    bone_to_artifact[z][labeled_bone == label_id] = True
        
        # Analyze artifact regions
        if np.any(artifact_slice):
            labeled_artifact, artifact_chars = analyze_contour_characteristics(artifact_slice, ct_slice)
            
            # Check each artifact region
            for label_id, chars in artifact_chars.items():
                # Criteria for reclassifying artifact -> bone
                is_bone = False
                
                # 1. Low intensity variation (bone is consistent)
                if chars['intensity_cv'] < 0.15:  # 15% coefficient of variation
                    is_bone = True
                
                # 2. High edge ratio (bone has sharp edges)
                if chars['edge_ratio'] > 0.3:
                    is_bone = True
                
                # 3. High solidity and reasonable size (bone is solid)
                if chars['solidity'] > 0.8 and chars['area'] > 50:
                    is_bone = True
                
                # 4. Good gradient consistency
                if chars['gradient_consistency'] > 0.7:
                    is_bone = True
                
                # 5. High mean HU typical of bone
                if chars['intensity_mean'] > 600:
                    is_bone = True
                
                # 6. Check 3D consistency
                consistency = compute_3d_consistency(artifact_mask, z, label_id, chars['bbox'])
                if consistency > 0.6:  # Good 3D consistency
                    is_bone = True
                
                # But override if it's clearly a streak pattern
                if chars['eccentricity'] > 0.9 and chars['intensity_cv'] > 0.3:
                    is_bone = False
                
                # Mark for reclassification
                if is_bone:
                    artifact_to_bone[z][labeled_artifact == label_id] = True
    
    # Apply refinements
    refined_bone = bone_mask.copy()
    refined_artifact = artifact_mask.copy()
    
    # Remove from bone what goes to artifact
    refined_bone[bone_to_artifact] = False
    # Add to artifact what comes from bone
    refined_artifact[bone_to_artifact] = True
    
    # Remove from artifact what goes to bone
    refined_artifact[artifact_to_bone] = False
    # Add to bone what comes from artifact
    refined_bone[artifact_to_bone] = True
    
    # Report changes
    bone_to_artifact_count = np.sum(bone_to_artifact)
    artifact_to_bone_count = np.sum(artifact_to_bone)
    
    print(f"Refinement complete:")
    print(f"  Reclassified {bone_to_artifact_count:,} voxels from bone to artifact")
    print(f"  Reclassified {artifact_to_bone_count:,} voxels from artifact to bone")
    print(f"  Final bone: {np.sum(refined_bone):,} voxels")
    print(f"  Final artifacts: {np.sum(refined_artifact):,} voxels")
    
    return {
        'bone': refined_bone,
        'bright_artifacts': refined_artifact,
        'bone_to_artifact': bone_to_artifact,
        'artifact_to_bone': artifact_to_bone
    }