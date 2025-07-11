import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes
from artifact_discrimination import create_sequential_masks
from artifact_discrimination_fast import create_fast_russian_doll_segmentation
from artifact_discrimination_enhanced import create_enhanced_russian_doll_segmentation


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
                             dark_threshold_high=-200):
    """
    Create dark artifact mask for low HU streaking artifacts.
    
    Args:
        ct_volume: 3D CT data in HU
        metal_mask: 3D binary mask of metal
        roi_bounds: Dictionary with ROI boundaries
        dark_threshold_high: Upper HU threshold for dark artifacts
        
    Returns:
        3D binary mask of dark artifacts
    """
    # Create dark region mask
    dark_mask = ct_volume <= dark_threshold_high
    
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
    
    # Fill holes if requested
    if fill_holes:
        refined_mask = binary_fill_holes(refined_mask)
    
    # Remove small components
    labeled_array, num_features = label(refined_mask)
    for i in range(1, num_features + 1):
        component_mask = labeled_array == i
        if np.sum(component_mask) < min_size:
            refined_mask[component_mask] = False
    
    # Smooth with morphological operations
    if smooth_iterations > 0:
        for _ in range(smooth_iterations):
            refined_mask = binary_dilation(refined_mask, iterations=1)
            refined_mask = binary_erosion(refined_mask, iterations=1)
    
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
                                   dark_threshold_high=-150,
                                   bone_threshold_low=300, bone_threshold_high=1500,
                                   bright_artifact_max_distance_cm=10.0,
                                   use_fast_mode=True,
                                   use_enhanced_mode=False,
                                   progress_callback=None):
    """
    Create segmentation using Russian doll approach with smart bone/artifact discrimination.
    
    Args:
        ct_volume: 3D CT data in HU
        metal_mask: Already segmented metal mask
        spacing: Voxel spacing (z, y, x) in mm
        roi_bounds: Optional ROI bounds to constrain analysis
        dark_threshold_high: Upper threshold for dark artifacts
        bone_threshold_low: Lower threshold for bone/bright artifacts
        bone_threshold_high: Upper threshold for bone/bright artifacts
        bright_artifact_max_distance_cm: Max distance from metal for artifacts
        use_fast_mode: Use fast discrimination (distance-based) instead of profile analysis
        use_enhanced_mode: Use enhanced edge-based discrimination
        progress_callback: Optional callback function(progress, message) for progress updates
        
    Returns:
        dict: All segmentation masks including discrimination results
    """
    # Choose discrimination method
    if use_enhanced_mode:
        segmentation_result = create_enhanced_russian_doll_segmentation(
            ct_volume,
            metal_mask,
            spacing,
            dark_range=(-1024, dark_threshold_high),
            bone_range=(bone_threshold_low, bone_threshold_high),
            max_distance_cm=bright_artifact_max_distance_cm,
            progress_callback=progress_callback
        )
    elif use_fast_mode:
        segmentation_result = create_fast_russian_doll_segmentation(
            ct_volume,
            metal_mask,
            spacing,
            dark_range=(-1024, dark_threshold_high),
            bone_range=(bone_threshold_low, bone_threshold_high),
            max_distance_cm=bright_artifact_max_distance_cm
        )
    else:
        segmentation_result = create_sequential_masks(
            ct_volume, 
            metal_mask, 
            spacing,
            dark_range=(-1024, dark_threshold_high),
            bone_range=(bone_threshold_low, bone_threshold_high),
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
    for mask_name in ['dark_artifacts', 'bone', 'bright_artifacts']:
        if mask_name in segmentation_result:
            segmentation_result[mask_name] = refine_mask(
                segmentation_result[mask_name], 
                min_size=10,
                fill_holes=True,
                smooth_iterations=1
            )
    
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