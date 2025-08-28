"""
Legacy utility functions for backward compatibility.
Main metal detection logic is now in core/metal_detection.py
"""

import numpy as np
import nibabel as nib
from scipy.ndimage import label, binary_dilation, generate_binary_structure
from typing import Dict, Tuple, Optional
from core.metal_detection import MetalDetector, MetalDetectionMethod


def detect_metal_volume(ct_volume: np.ndarray, spacing: Tuple[float, float, float],
                        margin_cm: float = 2.0, fw_percentage: float = 75.0,
                        min_metal_hu: float = 2500, dilation_iterations: int = 2) -> Dict:
    """
    Legacy function for detecting metal volume.
    Wrapper around the new MetalDetector class.
    """
    detector = MetalDetector(MetalDetectionMethod.LEGACY)
    return detector.detect(
        ct_volume, spacing,
        min_metal_hu=min_metal_hu,
        margin_cm=margin_cm,
        fw_percentage=fw_percentage,
        dilation_iterations=dilation_iterations
    )


def create_affine_from_dicom_meta(metadata: dict) -> np.ndarray:
    """
    Create affine transformation matrix from DICOM metadata.
    
    Args:
        metadata: Dictionary containing DICOM metadata with:
            - origin: (x, y, z) coordinates of the first voxel
            - spacing: (z, y, x) voxel spacing in mm
            
    Returns:
        4x4 affine transformation matrix
    """
    origin = metadata.get('origin', [0, 0, 0])
    spacing = metadata.get('spacing', [1, 1, 1])
    
    # Create affine matrix
    affine = np.eye(4)
    affine[0, 0] = spacing[2]  # x spacing
    affine[1, 1] = spacing[1]  # y spacing
    affine[2, 2] = spacing[0]  # z spacing
    affine[0, 3] = origin[0]    # x origin
    affine[1, 3] = origin[1]    # y origin
    affine[2, 3] = origin[2]    # z origin
    
    return affine


def save_mask_as_nifti(mask: np.ndarray, affine: np.ndarray, output_path: str):
    """
    Save a binary mask as a NIfTI file.
    
    Args:
        mask: 3D binary mask array
        affine: 4x4 affine transformation matrix
        output_path: Path to save the NIfTI file
    """
    # Ensure mask is the right data type
    mask_data = mask.astype(np.uint8)
    
    # Create NIfTI image
    nifti_img = nib.Nifti1Image(mask_data, affine)
    
    # Save to file
    nib.save(nifti_img, output_path)
    
    return output_path