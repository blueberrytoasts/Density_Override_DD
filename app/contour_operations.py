import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes, generate_binary_structure
<<<<<<< HEAD
from core.discrimination import ArtifactDiscriminator, DiscriminationMethod
from body_mask import create_body_mask, constrain_to_body


# Wrapper functions for backward compatibility
def create_sequential_masks(ct_volume, metal_mask, spacing, **kwargs):
    """Legacy wrapper for discrimination."""
    discriminator = ArtifactDiscriminator(DiscriminationMethod.DISTANCE_BASED)
    
    # Get threshold ranges
    bright_range = kwargs.get('bright_range', [800, 2000])
    bone_range = kwargs.get('bone_range', [300, 1500])
    dark_range = kwargs.get('dark_range', [-1024, -150])
    
    # Create body mask to exclude air outside patient
    body_mask = create_body_mask(ct_volume, air_threshold=-300)
    print(f"Debug - Body mask voxels: {np.sum(body_mask)}")
    
    # Create combined bright mask (includes both bone and bright artifact ranges for discrimination)
    bright_mask = ((ct_volume >= bright_range[0]) & (ct_volume <= bright_range[1])) | \
                  ((ct_volume >= bone_range[0]) & (ct_volume <= bone_range[1]))
    bright_mask = bright_mask & (~metal_mask) & body_mask  # Exclude metal and constrain to body
    
    print(f"Debug - Bright mask voxels before discrimination: {np.sum(bright_mask)}")
    print(f"Debug - Bright range: {bright_range}, Bone range: {bone_range}")
    
    # Discriminate bone from artifacts
    result = discriminator.discriminate(ct_volume, metal_mask, bright_mask, spacing)
    
    print(f"Debug - After discrimination - Bone: {np.sum(result['bone_mask'])}, Artifacts: {np.sum(result['artifact_mask'])}")
    
    # Create dark mask - constrained to body to avoid air outside patient
    dark_mask = (ct_volume >= dark_range[0]) & (ct_volume <= dark_range[1]) & (~metal_mask) & body_mask
    print(f"Debug - Dark mask voxels: {np.sum(dark_mask)}, Dark range: {dark_range}")
    
    return {
        'metal': metal_mask,
        'dark_artifacts': dark_mask,
        'bone': result['bone_mask'],
        'bright_artifacts': result['artifact_mask']
    }


def create_fast_russian_doll_segmentation(ct_volume, metal_mask, spacing, **kwargs):
    """Fast Russian doll segmentation using distance-based discrimination."""
    return create_sequential_masks(ct_volume, metal_mask, spacing, **kwargs)


def create_enhanced_russian_doll_segmentation(ct_volume, metal_mask, spacing, **kwargs):
    """Enhanced Russian doll segmentation using edge-based discrimination."""
    discriminator = ArtifactDiscriminator(DiscriminationMethod.EDGE_BASED)
    
    # Get threshold ranges
    bright_range = kwargs.get('bright_range', [800, 2000])
    bone_range = kwargs.get('bone_range', [300, 1500])
    dark_range = kwargs.get('dark_range', [-1024, -150])
    
    # Create body mask to exclude air outside patient
    body_mask = create_body_mask(ct_volume, air_threshold=-300)
    
    # Create combined bright mask (includes both bone and bright artifact ranges for discrimination)
    bright_mask = ((ct_volume >= bright_range[0]) & (ct_volume <= bright_range[1])) | \
                  ((ct_volume >= bone_range[0]) & (ct_volume <= bone_range[1]))
    bright_mask = bright_mask & (~metal_mask) & body_mask  # Exclude metal and constrain to body
    
    # Discriminate bone from artifacts
    result = discriminator.discriminate(ct_volume, metal_mask, bright_mask, spacing)
    
    # Create dark mask - constrained to body to avoid air outside patient
    dark_mask = (ct_volume >= dark_range[0]) & (ct_volume <= dark_range[1]) & (~metal_mask) & body_mask
    
    return {
        'metal': metal_mask,
        'dark_artifacts': dark_mask,
        'bone': result['bone_mask'],
        'bright_artifacts': result['artifact_mask']
    }


def refine_bone_artifact_discrimination(bone_mask, artifact_mask, ct_volume, spacing, **kwargs):
    """Refine bone/artifact discrimination using morphological operations."""
    # Apply morphological operations
    struct = generate_binary_structure(3, 1)
    bone_refined = binary_fill_holes(bone_mask)
    bone_refined = binary_erosion(bone_refined, struct, iterations=1)
    bone_refined = binary_dilation(bone_refined, struct, iterations=1)
    
    artifact_refined = binary_fill_holes(artifact_mask)
    artifact_refined = binary_erosion(artifact_refined, struct, iterations=1)
    artifact_refined = binary_dilation(artifact_refined, struct, iterations=1)
    
    return {
        'bone_mask': bone_refined,
        'artifact_mask': artifact_refined,
        'success': True
    }
=======
from artifact_discrimination import create_sequential_masks
from artifact_discrimination_fast import create_fast_russian_doll_segmentation
from artifact_discrimination_enhanced import create_enhanced_russian_doll_segmentation
from artifact_discrimination_refinement import refine_bone_artifact_discrimination
>>>>>>> main


def boolean_subtract(mask1, mask2):
    """
    Subtract mask2 from mask1 (mask1 AND NOT mask2).
    
    Args:
        mask1: First binary mask (numpy array)
        mask2: Second binary mask (numpy array)
        
    Returns:
        Binary mask with mask2 subtracted from mask1
    """
    return mask1 & ~mask2


def boolean_union(mask1, mask2):
    """
    Union of two masks (mask1 OR mask2).
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
        
    Returns:
        Binary mask representing the union
    """
    return mask1 | mask2


def boolean_intersection(mask1, mask2):
    """
    Intersection of two masks (mask1 AND mask2).
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
        
    Returns:
        Binary mask representing the intersection
    """
    return mask1 & mask2


def create_bright_artifact_mask(ct_volume, metal_mask, roi_bounds, 
                               bright_threshold_low=800, bright_threshold_high=3000):
    """
    Create bright artifact mask by thresholding and subtracting metal.
    
    Args:
        ct_volume: 3D CT data in HU
        metal_mask: 3D binary mask of metal
        roi_bounds: Dictionary with ROI boundaries
        bright_threshold_low: Lower HU threshold for bright artifacts
        bright_threshold_high: Upper HU threshold for bright artifacts
        
    Returns:
        3D binary mask of bright artifacts (excluding metal)
    """
    # Create initial bright region mask
    bright_mask = (ct_volume >= bright_threshold_low) & (ct_volume <= bright_threshold_high)
    
    # Constrain to ROI
    roi_mask = np.zeros_like(ct_volume, dtype=bool)
    roi_mask[roi_bounds['z_min']:roi_bounds['z_max'],
             roi_bounds['y_min']:roi_bounds['y_max'],
             roi_bounds['x_min']:roi_bounds['x_max']] = True
    
    bright_mask = bright_mask & roi_mask
    
    # Subtract metal to get isolated bright artifacts
    bright_artifacts_mask = boolean_subtract(bright_mask, metal_mask)
    
    return bright_artifacts_mask


def create_dark_artifact_mask(ct_volume, metal_mask, roi_bounds,
<<<<<<< HEAD
                             dark_threshold_high=-200):
=======
                             dark_threshold_low=-1000, dark_threshold_high=-150):
>>>>>>> main
    """
    Create dark artifact mask for low HU streaking artifacts.
    
    Args:
        ct_volume: 3D CT data in HU
        metal_mask: 3D binary mask of metal
        roi_bounds: Dictionary with ROI boundaries
<<<<<<< HEAD
=======
        dark_threshold_low: Lower HU threshold for dark artifacts
>>>>>>> main
        dark_threshold_high: Upper HU threshold for dark artifacts
        
    Returns:
        3D binary mask of dark artifacts
    """
    # Create dark region mask
<<<<<<< HEAD
    dark_mask = ct_volume <= dark_threshold_high
=======
    dark_mask = (ct_volume >= dark_threshold_low) & (ct_volume <= dark_threshold_high)
>>>>>>> main
    
    # Constrain to ROI
    roi_mask = np.zeros_like(ct_volume, dtype=bool)
    roi_mask[roi_bounds['z_min']:roi_bounds['z_max'],
             roi_bounds['y_min']:roi_bounds['y_max'],
             roi_bounds['x_min']:roi_bounds['x_max']] = True
    
    dark_mask = dark_mask & roi_mask
    
    # Dark artifacts shouldn't overlap with metal
    dark_artifacts_mask = boolean_subtract(dark_mask, metal_mask)
    
    return dark_artifacts_mask


def refine_mask(mask, min_size=10, fill_holes=True, smooth_iterations=1):
    """
    Refine a binary mask by removing small components and optionally filling holes.
    
    Args:
        mask: Binary mask to refine
        min_size: Minimum component size in voxels
        fill_holes: Whether to fill holes in the mask
        smooth_iterations: Number of morphological smoothing iterations
        
    Returns:
        Refined binary mask
    """
    from scipy.ndimage import label
    
    refined_mask = mask.copy()
    
    # Determine if this is a 3D mask
    is_3d = mask.ndim == 3
    
    # Fill holes if requested
    if fill_holes and np.any(refined_mask):
        try:
            refined_mask = binary_fill_holes(refined_mask)
        except:
            print(f"  Warning: hole filling failed for shape {mask.shape}")
    
    # Use appropriate connectivity for labeling
    if is_3d:
        # 3D connectivity structure (26-connected)
        struct = generate_binary_structure(3, 3)
    else:
        # 2D connectivity structure (8-connected)
        struct = generate_binary_structure(2, 2)
    
    # Remove small components
    labeled_array, num_features = label(refined_mask, structure=struct)
    print(f"  Found {num_features} connected components (min_size={min_size})")
    
    if num_features > 0:
        # Count component sizes
        component_sizes = []
        for i in range(1, num_features + 1):
            component_mask = labeled_array == i
            size = np.sum(component_mask)
            component_sizes.append(size)
            if size < min_size:
                refined_mask[component_mask] = False
        
        if component_sizes:
            print(f"  Component sizes: min={min(component_sizes)}, max={max(component_sizes)}, mean={np.mean(component_sizes):.1f}")
    
    # Smooth with morphological operations
    if smooth_iterations > 0 and np.any(refined_mask):
        # Use 3D structuring element for 3D data
        if is_3d:
            struct_erode = generate_binary_structure(3, 1)  # 6-connected for erosion
            struct_dilate = generate_binary_structure(3, 2)  # 18-connected for dilation
        else:
            struct_erode = struct_dilate = None
            
        for _ in range(smooth_iterations):
            refined_mask = binary_dilation(refined_mask, iterations=1, structure=struct_dilate)
            refined_mask = binary_erosion(refined_mask, iterations=1, structure=struct_erode)
    
    return refined_mask


def create_bone_mask(ct_volume, metal_mask, bright_mask, dark_mask, roi_bounds,
                    bone_threshold_low=150, bone_threshold_high=1500):
    """
    Create bone mask by thresholding and subtracting other masks.
    
    Args:
        ct_volume: 3D CT data in HU
        metal_mask: 3D binary mask of metal
        bright_mask: 3D binary mask of bright artifacts
        dark_mask: 3D binary mask of dark artifacts
        roi_bounds: Dictionary with ROI boundaries
        bone_threshold_low: Lower HU threshold for bone
        bone_threshold_high: Upper HU threshold for bone
        
    Returns:
        3D binary mask of bone tissue
    """
    # Create initial bone region mask
    bone_mask = (ct_volume >= bone_threshold_low) & (ct_volume <= bone_threshold_high)
    
    # Constrain to ROI
    roi_mask = np.zeros_like(ct_volume, dtype=bool)
    roi_mask[roi_bounds['z_min']:roi_bounds['z_max'],
             roi_bounds['y_min']:roi_bounds['y_max'],
             roi_bounds['x_min']:roi_bounds['x_max']] = True
    
    bone_mask = bone_mask & roi_mask
    
    # Subtract all other tissue types
    bone_mask = boolean_subtract(bone_mask, metal_mask)
    bone_mask = boolean_subtract(bone_mask, bright_mask)
    bone_mask = boolean_subtract(bone_mask, dark_mask)
    
    return bone_mask


def create_russian_doll_segmentation(ct_volume, metal_mask, spacing, roi_bounds=None,
<<<<<<< HEAD
                                   dark_threshold_high=-150,
                                   dark_threshold_low=-1024,  # Add parameter for lower bound
                                   bone_threshold_low=300, bone_threshold_high=1500,
                                   bright_threshold_low=None, bright_threshold_high=None,
                                   bright_artifact_max_distance_cm=10.0,
                                   use_fast_mode=True,
                                   use_enhanced_mode=False,
                                   use_advanced_mode=False,
=======
                                   dark_threshold_low=-1000, dark_threshold_high=-150,
                                   bone_threshold_low=300, bone_threshold_high=1500,
                                   bright_artifact_max_distance_cm=10.0,
                                   use_fast_mode=True,
                                   use_enhanced_mode=False,
>>>>>>> main
                                   use_refinement=True,
                                   progress_callback=None):
    """
    Create segmentation using Russian doll approach with smart bone/artifact discrimination.
    
    Args:
        ct_volume: 3D CT data in HU
        metal_mask: Already segmented metal mask
        spacing: Voxel spacing (z, y, x) in mm
        roi_bounds: Optional ROI bounds to constrain analysis
<<<<<<< HEAD
        dark_threshold_high: Upper threshold for dark artifacts
        bone_threshold_low: Lower threshold for bone tissue
        bone_threshold_high: Upper threshold for bone tissue
        bright_threshold_low: Lower threshold for bright artifacts (optional, defaults to bone_threshold_low)
        bright_threshold_high: Upper threshold for bright artifacts (optional, defaults to higher value)
        bright_artifact_max_distance_cm: Max distance from metal for artifacts
        use_fast_mode: Use fast discrimination (distance-based) instead of profile analysis
        use_enhanced_mode: Use enhanced edge-based discrimination
        use_advanced_mode: Use advanced texture/gradient-based discrimination
=======
        dark_threshold_low: Lower threshold for dark artifacts
        dark_threshold_high: Upper threshold for dark artifacts
        bone_threshold_low: Lower threshold for bone/bright artifacts
        bone_threshold_high: Upper threshold for bone/bright artifacts
        bright_artifact_max_distance_cm: Max distance from metal for artifacts
        use_fast_mode: Use fast discrimination (distance-based) instead of profile analysis
        use_enhanced_mode: Use enhanced edge-based discrimination
>>>>>>> main
        use_refinement: Apply second-pass refinement to improve bone/artifact discrimination
        progress_callback: Optional callback function(progress, message) for progress updates
        
    Returns:
        dict: All segmentation masks including discrimination results
    """
<<<<<<< HEAD
    # Set bright thresholds if not provided (backward compatibility)
    if bright_threshold_low is None:
        bright_threshold_low = bone_threshold_low
    if bright_threshold_high is None:
        bright_threshold_high = max(bone_threshold_high, 2000)  # Bright artifacts can go higher than bone
    # Choose discrimination method
    if use_advanced_mode:
        # Use new advanced texture/gradient-based discrimination
        # Use texture-based discrimination from consolidated module
        discriminator = ArtifactDiscriminator(DiscriminationMethod.TEXTURE_BASED)
        
        def classify_bone_vs_artifact(ct_volume, bright_mask, metal_mask=None, use_ml=False):
            result = discriminator.discriminate(ct_volume, metal_mask, bright_mask, spacing)
            confidence_map = result.get('confidence_map', np.ones_like(bright_mask, dtype=float))
            return result['bone_mask'], result['artifact_mask'], confidence_map
        
        # Create body mask to exclude air outside patient
        body_mask = create_body_mask(ct_volume, air_threshold=-400)
        
        # First segment dark artifacts (excluding metal and constrained to body)
        dark_mask = (ct_volume >= dark_threshold_low) & (ct_volume <= dark_threshold_high)
        dark_mask = boolean_subtract(dark_mask, metal_mask) & body_mask
        
        # Get combined bright regions that need discrimination (union of bright artifact and bone ranges)
        bright_artifact_mask = (ct_volume >= bright_threshold_low) & (ct_volume <= bright_threshold_high)
        bone_range_mask = (ct_volume >= bone_threshold_low) & (ct_volume <= bone_threshold_high)
        bright_mask = (bright_artifact_mask | bone_range_mask) & body_mask  # Union of both ranges, constrained to body
        bright_mask = boolean_subtract(bright_mask, metal_mask)
        bright_mask = boolean_subtract(bright_mask, dark_mask)
        
        # Apply advanced discrimination
        bone_mask, bright_artifacts, confidence_map = classify_bone_vs_artifact(
            ct_volume, bright_mask, metal_mask
        )
        
        segmentation_result = {
            'metal': metal_mask,
            'dark_artifacts': dark_mask,
            'bright_artifacts': bright_artifacts,
            'bone': bone_mask,
            'confidence_map': confidence_map,
            'method': 'advanced_texture_gradient'
        }
        
    elif use_enhanced_mode:
=======
    # Choose discrimination method
    if use_enhanced_mode:
>>>>>>> main
        segmentation_result = create_enhanced_russian_doll_segmentation(
            ct_volume,
            metal_mask,
            spacing,
            dark_range=(dark_threshold_low, dark_threshold_high),
            bone_range=(bone_threshold_low, bone_threshold_high),
<<<<<<< HEAD
            bright_range=(bright_threshold_low, bright_threshold_high),
=======
>>>>>>> main
            max_distance_cm=bright_artifact_max_distance_cm,
            progress_callback=progress_callback
        )
    elif use_fast_mode:
        segmentation_result = create_fast_russian_doll_segmentation(
            ct_volume,
            metal_mask,
            spacing,
            dark_range=(dark_threshold_low, dark_threshold_high),
            bone_range=(bone_threshold_low, bone_threshold_high),
<<<<<<< HEAD
            bright_range=(bright_threshold_low, bright_threshold_high),
=======
>>>>>>> main
            max_distance_cm=bright_artifact_max_distance_cm
        )
    else:
        segmentation_result = create_sequential_masks(
            ct_volume, 
            metal_mask, 
            spacing,
            dark_range=(dark_threshold_low, dark_threshold_high),
            bone_range=(bone_threshold_low, bone_threshold_high),
<<<<<<< HEAD
            bright_range=(bright_threshold_low, bright_threshold_high),
=======
>>>>>>> main
            bright_artifact_max_distance_cm=bright_artifact_max_distance_cm
        )
    
    # If ROI bounds provided, constrain results
    if roi_bounds is not None:
        roi_mask = np.zeros_like(ct_volume, dtype=bool)
        roi_mask[roi_bounds['z_min']:roi_bounds['z_max'],
                 roi_bounds['y_min']:roi_bounds['y_max'],
                 roi_bounds['x_min']:roi_bounds['x_max']] = True
        
        for mask_name in ['dark_artifacts', 'bone', 'bright_artifacts']:
            if mask_name in segmentation_result:
                segmentation_result[mask_name] = segmentation_result[mask_name] & roi_mask
    
    # Apply refinement to all masks
    print("\nApplying refinement to masks...")
    # Apply second-pass refinement if requested
    if use_refinement and 'bone' in segmentation_result and 'bright_artifacts' in segmentation_result:
        print("\nApplying second-pass refinement...")
        refinement_result = refine_bone_artifact_discrimination(
<<<<<<< HEAD
            segmentation_result['bone'],
            segmentation_result['bright_artifacts'],
            ct_volume,
            spacing
        )
        # Update the masks
        segmentation_result['bone'] = refinement_result['bone_mask']
        segmentation_result['bright_artifacts'] = refinement_result['artifact_mask']
=======
            ct_volume, 
            segmentation_result,
            metal_mask,
            spacing
        )
        # Update the masks
        segmentation_result['bone'] = refinement_result['bone']
        segmentation_result['bright_artifacts'] = refinement_result['bright_artifacts']
>>>>>>> main
    
    # Store originals in case refinement fails
    originals = {}
    for mask_name in ['dark_artifacts', 'bone', 'bright_artifacts']:
        if mask_name in segmentation_result:
            originals[mask_name] = segmentation_result[mask_name].copy()
            original_count = np.sum(segmentation_result[mask_name])
            
            # Use different refinement parameters based on the mode
            if use_enhanced_mode:
                # For enhanced mode, be much more conservative
                # The edge analysis already produces clean results
                min_component_size = 5 if mask_name == 'bone' else 3
                smooth_iters = 0  # No smoothing - preserve edge details
            else:
                # Original parameters for other modes
                min_component_size = 10
                smooth_iters = 1
            
            segmentation_result[mask_name] = refine_mask(
                segmentation_result[mask_name], 
                min_size=min_component_size,
                fill_holes=True,
                smooth_iterations=smooth_iters
            )
            refined_count = np.sum(segmentation_result[mask_name])
            print(f"  {mask_name}: {original_count:,} -> {refined_count:,} voxels after refinement")
            
            # Safety check - if refinement removed everything, restore original
            if refined_count == 0 and original_count > 0:
                print(f"  WARNING: Refinement removed all {mask_name} voxels! Restoring original.")
                segmentation_result[mask_name] = originals[mask_name]
    
    print(f"\nFinal segmentation results:")
    for mask_name, mask in segmentation_result.items():
        if isinstance(mask, np.ndarray):
            print(f"  {mask_name}: shape={mask.shape}, voxels={np.sum(mask):,}")
    
    return segmentation_result


def save_all_contours_as_nifti(masks_dict, affine, output_prefix):
    """
    Save all contour masks as separate NIFTI files.
    
    Args:
        masks_dict: Dictionary of mask_name: mask_array pairs
        affine: 4x4 affine transformation matrix
        output_prefix: Prefix for output filenames
    """
    for mask_name, mask in masks_dict.items():
        if isinstance(mask, np.ndarray):
            output_path = f"{output_prefix}_{mask_name}.nii.gz"
            mask_int = mask.astype(np.uint8)
            nifti_img = nib.Nifti1Image(mask_int, affine)
            nib.save(nifti_img, output_path)
            print(f"Saved {mask_name} to {output_path}")


def load_nifti_mask(filepath):
    """
    Load a binary mask from a NIFTI file.
    
    Args:
        filepath: Path to the NIFTI file
        
    Returns:
        Binary mask as numpy array
    """
    nifti_img = nib.load(filepath)
    mask = nifti_img.get_fdata().astype(bool)
    return mask, nifti_img.affine


def combine_masks_multilabel(masks_dict):
    """
    Combine multiple binary masks into a single multi-label mask.
    
    Args:
        masks_dict: Dictionary of {label_name: (mask, label_value)}
        
    Returns:
        Multi-label mask where each tissue type has a unique integer value
    """
    # Get shape from first mask
    first_mask = next(iter(masks_dict.values()))[0]
    multilabel = np.zeros_like(first_mask, dtype=np.uint8)
    
    # Assign labels in order of priority (metal should override others)
    priority_order = ['metal', 'bright_artifacts', 'dark_artifacts', 'bone']
    
    label_map = {}
    current_label = 1
    
    for tissue_type in priority_order:
        if tissue_type in masks_dict:
            mask, _ = masks_dict[tissue_type]
            multilabel[mask] = current_label
            label_map[tissue_type] = current_label
            current_label += 1
    
    # Handle any remaining masks
    for tissue_type, (mask, _) in masks_dict.items():
        if tissue_type not in priority_order:
            multilabel[mask] = current_label
            label_map[tissue_type] = current_label
            current_label += 1
    
    return multilabel, label_map