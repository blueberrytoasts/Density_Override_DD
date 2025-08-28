"""
Unified Metal Detection Module
Consolidates all metal detection algorithms into a single interface.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Callable
from scipy.ndimage import label, binary_dilation, generate_binary_structure
from scipy.ndimage import gaussian_filter, distance_transform_edt
from enum import Enum


class MetalDetectionMethod(Enum):
    """Available metal detection methods."""
    LEGACY = "legacy"
    ADAPTIVE_2D = "adaptive_2d"
    ADAPTIVE_3D = "adaptive_3d"


class MetalDetector:
    """
    Unified metal detection interface supporting multiple algorithms.
    
    Methods:
        LEGACY: Initial threshold + star profile refinement
        ADAPTIVE_2D: Percentile-based adaptive thresholding
        ADAPTIVE_3D: Multi-planar analysis with FW75% thresholding
    """
    
    def __init__(self, method: MetalDetectionMethod = MetalDetectionMethod.ADAPTIVE_3D):
        """
        Initialize metal detector with specified method.
        
        Args:
            method: Detection method to use
        """
        self.method = method
        self.detectors = {
            MetalDetectionMethod.LEGACY: self._detect_legacy,
            MetalDetectionMethod.ADAPTIVE_2D: self._detect_adaptive_2d,
            MetalDetectionMethod.ADAPTIVE_3D: self._detect_adaptive_3d
        }
    
    def detect(self, ct_volume: np.ndarray, spacing: Tuple[float, float, float],
               **kwargs) -> Dict:
        """
        Detect metal in CT volume using selected method.
        
        Args:
            ct_volume: 3D CT volume in HU
            spacing: Voxel spacing (z, y, x) in mm
            **kwargs: Method-specific parameters
            
        Returns:
            Dictionary containing:
                - mask: 3D binary mask of detected metal
                - roi_bounds: ROI boundaries
                - threshold: Detected/used threshold
                - metadata: Method-specific metadata
        """
        detector = self.detectors.get(self.method)
        if not detector:
            raise ValueError(f"Unknown detection method: {self.method}")
        
        return detector(ct_volume, spacing, **kwargs)
    
    def _detect_legacy(self, ct_volume: np.ndarray, spacing: Tuple[float, float, float],
                      min_metal_hu: float = 2500,
                      margin_cm: float = 1.0,
                      fw_percentage: float = 75.0,
                      dilation_iterations: int = 2) -> Dict:
        """
        Legacy metal detection using initial threshold + star profiles.
        
        This method:
        1. Applies initial HU threshold
        2. Finds largest connected component
        3. Uses star profiles to refine threshold
        4. Creates ROI with specified margin
        """
        # Initial thresholding - use higher threshold for faster processing
        metal_mask = ct_volume > min_metal_hu
        
        if not np.any(metal_mask):
            return self._empty_result()
        
        # Find largest connected component more efficiently
        labeled, num_features = label(metal_mask)
        if num_features == 0:
            return self._empty_result()
        
        # More efficient size calculation - only process if reasonable number of components
        if num_features > 1000:
            # Too many components, likely noise - increase threshold
            metal_mask = ct_volume > (min_metal_hu + 500)
            labeled, num_features = label(metal_mask)
            if num_features == 0:
                return self._empty_result()
        
        # Use bincount for faster size calculation
        sizes = np.bincount(labeled.ravel())[1:]  # Skip background (0)
        largest_label = np.argmax(sizes) + 1
        metal_mask = labeled == largest_label
        
        # Apply dilation if requested
        if dilation_iterations > 0:
            struct = generate_binary_structure(3, 3)
            metal_mask = binary_dilation(metal_mask, structure=struct, 
                                        iterations=dilation_iterations)
        
        # Find center of mass
        z_coords, y_coords, x_coords = np.where(metal_mask)
        center_z = int(np.mean(z_coords))
        center_y = int(np.mean(y_coords))
        center_x = int(np.mean(x_coords))
        
        # Create ROI bounds
        # Convert margin from cm to voxels
        # spacing is already in mm, so cm * 10 = mm, then divide by spacing
        spacing_mm = np.mean(spacing)  # spacing is already in mm
        margin_mm = margin_cm * 10  # Convert cm to mm
        margin_voxels = int(margin_mm / spacing_mm)
        roi_bounds = {
            'z_min': max(0, np.min(z_coords) - margin_voxels),
            'z_max': min(ct_volume.shape[0], np.max(z_coords) + margin_voxels + 1),
            'y_min': max(0, np.min(y_coords) - margin_voxels),
            'y_max': min(ct_volume.shape[1], np.max(y_coords) + margin_voxels + 1),
            'x_min': max(0, np.min(x_coords) - margin_voxels),
            'x_max': min(ct_volume.shape[2], np.max(x_coords) + margin_voxels + 1)
        }
        
        # Use star profiles to refine threshold (simplified)
        threshold = self._calculate_star_threshold(
            ct_volume[center_z], center_y, center_x, roi_bounds, fw_percentage
        )
        
        # Refine mask with new threshold
        if threshold and threshold > min_metal_hu:
            metal_mask = ct_volume > threshold
        
        return {
            'mask': metal_mask,
            'roi_bounds': roi_bounds,
            'threshold': threshold if threshold else min_metal_hu,
            'center_coords': (center_z, center_y, center_x),
            'method': 'legacy',
            'metadata': {
                'dilation_iterations': dilation_iterations,
                'margin_cm': margin_cm,
                'fw_percentage': fw_percentage
            }
        }
    
    def _detect_adaptive_2d(self, ct_volume: np.ndarray, spacing: Tuple[float, float, float],
                           intensity_percentile: float = 99.5,
                           margin_cm: float = 1.0,
                           min_component_size: int = 100) -> Dict:
        """
        Adaptive 2D metal detection using percentile-based thresholding.
        
        This method:
        1. Uses high percentile of intensities as initial threshold
        2. Applies connected component analysis
        3. Filters small components
        4. Creates adaptive ROI
        """
        # Calculate adaptive threshold
        threshold = np.percentile(ct_volume, intensity_percentile)
        metal_mask = ct_volume > threshold
        
        if not np.any(metal_mask):
            return self._empty_result()
        
        # Connected component filtering
        labeled, num_features = label(metal_mask)
        metal_mask = np.zeros_like(metal_mask)
        
        for i in range(1, num_features + 1):
            component = labeled == i
            if np.sum(component) >= min_component_size:
                metal_mask |= component
        
        if not np.any(metal_mask):
            return self._empty_result()
        
        # Find bounds
        z_coords, y_coords, x_coords = np.where(metal_mask)
        center_z = int(np.mean(z_coords))
        center_y = int(np.mean(y_coords))
        center_x = int(np.mean(x_coords))
        
        # Create ROI
        # Convert margin from cm to voxels
        # spacing is already in mm, so cm * 10 = mm, then divide by spacing
        spacing_mm = np.mean(spacing)  # spacing is already in mm
        margin_mm = margin_cm * 10  # Convert cm to mm
        margin_voxels = int(margin_mm / spacing_mm)
        roi_bounds = {
            'z_min': max(0, np.min(z_coords) - margin_voxels),
            'z_max': min(ct_volume.shape[0], np.max(z_coords) + margin_voxels + 1),
            'y_min': max(0, np.min(y_coords) - margin_voxels),
            'y_max': min(ct_volume.shape[1], np.max(y_coords) + margin_voxels + 1),
            'x_min': max(0, np.min(x_coords) - margin_voxels),
            'x_max': min(ct_volume.shape[2], np.max(x_coords) + margin_voxels + 1)
        }
        
        return {
            'mask': metal_mask,
            'roi_bounds': roi_bounds,
            'threshold': threshold,
            'center_coords': (center_z, center_y, center_x),
            'method': 'adaptive_2d',
            'metadata': {
                'intensity_percentile': intensity_percentile,
                'margin_cm': margin_cm,
                'min_component_size': min_component_size
            }
        }
    
    def _detect_adaptive_3d(self, ct_volume: np.ndarray, spacing: Tuple[float, float, float],
                           fw_percentage: float = 75.0,
                           margin_cm: float = 1.0,
                           intensity_percentile: float = 99.5) -> Dict:
        """
        Advanced 3D adaptive metal detection with multi-planar analysis.
        
        This method:
        1. Analyzes axial, coronal, and sagittal projections
        2. Uses FW75% (Full Width at 75% Maximum) for thresholding
        3. Creates per-slice adaptive thresholds
        4. Generates individual ROIs for each metal component
        """
        # Initial detection using high percentile - more aggressive for speed
        initial_threshold = np.percentile(ct_volume, intensity_percentile)
        # Ensure minimum threshold for metal
        initial_threshold = max(initial_threshold, 1500)  
        initial_mask = ct_volume > initial_threshold
        
        if not np.any(initial_mask):
            return self._empty_result()
        
        # Multi-planar analysis
        z_coords, y_coords, x_coords = np.where(initial_mask)
        
        # Analyze extent in each plane
        z_extent = np.max(z_coords) - np.min(z_coords)
        y_extent = np.max(y_coords) - np.min(y_coords)
        x_extent = np.max(x_coords) - np.min(x_coords)
        
        # Simplified processing - use global threshold instead of per-slice for speed
        # Calculate a single robust threshold based on the high-intensity regions
        metal_region = ct_volume[initial_mask]
        if len(metal_region) > 0:
            # Use a robust threshold based on the metal region statistics
            robust_threshold = np.percentile(metal_region, 25)  # 25th percentile of metal region
            refined_mask = ct_volume > robust_threshold
            slice_thresholds = [robust_threshold]  # Single threshold for all slices
        else:
            refined_mask = initial_mask
            slice_thresholds = [initial_threshold]
        
        # Create individual ROIs for components
        labeled, num_components = label(refined_mask)
        individual_regions = {}
        
        # Get coordinates of the refined mask for overall ROI
        z_coords, y_coords, x_coords = np.where(refined_mask)
        
        if len(z_coords) > 0:
            # Create simple per-slice ROIs based on the metal mask extent
            # Convert margin from cm to voxels
            # spacing is already in mm, so cm * 10 = mm, then divide by spacing
            spacing_mm = np.mean(spacing)  # spacing is already in mm
            margin_mm = margin_cm * 10  # Convert cm to mm
            margin_voxels = int(margin_mm / spacing_mm)
            
            for z in np.unique(z_coords):
                slice_mask = refined_mask[z]
                if np.any(slice_mask):
                    y_slice, x_slice = np.where(slice_mask)
                    if len(y_slice) > 0:
                        if z not in individual_regions:
                            individual_regions[z] = []
                        
                        individual_regions[z].append({
                            'component_id': 1,
                            'y_min': max(0, np.min(y_slice) - margin_voxels),
                            'y_max': min(ct_volume.shape[1], np.max(y_slice) + margin_voxels),
                            'x_min': max(0, np.min(x_slice) - margin_voxels),
                            'x_max': min(ct_volume.shape[2], np.max(x_slice) + margin_voxels),
                            'center_y': int(np.mean(y_slice)),
                            'center_x': int(np.mean(x_slice))
                        })
        
        # Overall ROI - ensure we have valid coordinates
        if len(z_coords) > 0:
            # Convert margin from cm to voxels
            # spacing is already in mm, so cm * 10 = mm, then divide by spacing
            spacing_mm = np.mean(spacing)  # spacing is already in mm
            margin_mm = margin_cm * 10  # Convert cm to mm
            margin_voxels = int(margin_mm / spacing_mm)
            roi_bounds = {
                'z_min': max(0, np.min(z_coords) - margin_voxels),
                'z_max': min(ct_volume.shape[0], np.max(z_coords) + margin_voxels + 1),
                'y_min': max(0, np.min(y_coords) - margin_voxels),
                'y_max': min(ct_volume.shape[1], np.max(y_coords) + margin_voxels + 1),
                'x_min': max(0, np.min(x_coords) - margin_voxels),
                'x_max': min(ct_volume.shape[2], np.max(x_coords) + margin_voxels + 1)
            }
        else:
            # No metal found, return empty result
            return self._empty_result()
        
        # Calculate average threshold
        avg_threshold = np.mean(slice_thresholds) if slice_thresholds else initial_threshold
        
        return {
            'mask': refined_mask,
            'roi_bounds': roi_bounds,
            'threshold': avg_threshold,
            'threshold_evolution': slice_thresholds,
            'individual_regions': individual_regions,
            'center_coords': (int(np.mean(z_coords)), int(np.mean(y_coords)), int(np.mean(x_coords))),
            'method': 'adaptive_3d',
            'metadata': {
                'fw_percentage': fw_percentage,
                'margin_cm': margin_cm,
                'intensity_percentile': intensity_percentile,
                'num_components': num_components,
                'extent_voxels': {'z': z_extent, 'y': y_extent, 'x': x_extent}
            }
        }
    
    def _calculate_star_threshold(self, slice_data: np.ndarray, center_y: int, center_x: int,
                                 roi_bounds: Dict, fw_percentage: float) -> Optional[float]:
        """Calculate threshold using star profile analysis."""
        # Simplified star profile calculation
        # In production, this would use the full star profile algorithm
        roi_data = slice_data[
            roi_bounds.get('y_min', 0):roi_bounds.get('y_max', slice_data.shape[0]),
            roi_bounds.get('x_min', 0):roi_bounds.get('x_max', slice_data.shape[1])
        ]
        
        if roi_data.size > 0:
            max_val = np.max(roi_data)
            threshold = max_val * (fw_percentage / 100.0)
            return threshold
        return None
    
    def _calculate_fw75_threshold(self, slice_data: np.ndarray, center_y: int, center_x: int,
                                 fw_percentage: float) -> Optional[float]:
        """Calculate Full Width at X% Maximum threshold."""
        # Sample around the center point
        window_size = 20
        y_min = max(0, center_y - window_size)
        y_max = min(slice_data.shape[0], center_y + window_size)
        x_min = max(0, center_x - window_size)
        x_max = min(slice_data.shape[1], center_x + window_size)
        
        local_region = slice_data[y_min:y_max, x_min:x_max]
        
        if local_region.size > 0:
            max_val = np.max(local_region)
            threshold = max_val * (fw_percentage / 100.0)
            return threshold
        return None
    
    def _empty_result(self) -> Dict:
        """Return empty result when no metal is detected."""
        return {
            'mask': None,
            'roi_bounds': None,
            'threshold': None,
            'center_coords': None,
            'method': self.method.value,
            'metadata': {}
        }


def get_star_profile_lines(slice_2d: np.ndarray, center_y: int, center_x: int, bounds: Dict) -> List:
    """
    Generate 16 star profile lines from center to boundaries.
    Used for visualization of the star profile algorithm.
    """
    from skimage.draw import line
    
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
        # Ensure endpoints are within bounds
        end_y = max(0, min(slice_2d.shape[0] - 1, end_y))
        end_x = max(0, min(slice_2d.shape[1] - 1, end_x))
        
        # Get line coordinates
        rr, cc = line(center_y, center_x, end_y, end_x)
        
        # Calculate distances from center
        distances = np.sqrt((rr - center_y)**2 + (cc - center_x)**2)
        
        # Get HU values along the line
        hu_values = slice_2d[rr, cc]
        
        profiles.append((distances, hu_values))
    
    return profiles


# Convenience functions for backward compatibility
def detect_metal_legacy(ct_volume: np.ndarray, spacing: Tuple[float, float, float], **kwargs) -> Dict:
    """Legacy metal detection for backward compatibility."""
    detector = MetalDetector(MetalDetectionMethod.LEGACY)
    return detector.detect(ct_volume, spacing, **kwargs)


def detect_metal_adaptive(ct_volume: np.ndarray, spacing: Tuple[float, float, float], **kwargs) -> Dict:
    """Adaptive 2D metal detection for backward compatibility."""
    detector = MetalDetector(MetalDetectionMethod.ADAPTIVE_2D)
    return detector.detect(ct_volume, spacing, **kwargs)


def detect_metal_adaptive_3d(ct_volume: np.ndarray, spacing: Tuple[float, float, float], **kwargs) -> Dict:
    """Adaptive 3D metal detection for backward compatibility."""
    detector = MetalDetector(MetalDetectionMethod.ADAPTIVE_3D)
    return detector.detect(ct_volume, spacing, **kwargs)