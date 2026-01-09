"""
Unified Metal Detection Module
Consolidates all metal detection algorithms into a single interface.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Callable
from scipy.ndimage import label, binary_dilation, generate_binary_structure, binary_fill_holes
from scipy.ndimage import gaussian_filter, distance_transform_edt
from enum import Enum


class MetalDetectionMethod(Enum):
    """Available metal detection methods."""
    LEGACY = "legacy"
    ADAPTIVE_3D = "adaptive_3d"


class MetalDetector:
    """
    Unified metal detection interface supporting multiple algorithms.

    Methods:
        LEGACY: Initial threshold + star profile refinement
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
        
        # Implement intelligent Z-bounds using 50% peak metal HU cutoff
        peak_metal_hu = np.max(ct_volume[metal_mask])
        cutoff_hu = peak_metal_hu * 0.5
        
        # Find slices that contain substantial metal (above 50% of peak)
        valid_z_slices = []
        for z in np.unique(z_coords):
            slice_mask = metal_mask[z]
            if np.any(slice_mask):
                slice_metal_values = ct_volume[z][slice_mask]
                if np.any(slice_metal_values >= cutoff_hu):
                    valid_z_slices.append(z)
        
        if valid_z_slices:
            z_min_intelligent = min(valid_z_slices)
            z_max_intelligent = max(valid_z_slices) + 1
        else:
            z_min_intelligent = int(np.min(z_coords))
            z_max_intelligent = int(np.max(z_coords)) + 1
        
        # Create ROI bounds
        # Convert margin from cm to voxels
        # spacing is already in mm, so cm * 10 = mm, then divide by spacing
        spacing_mm = np.mean(spacing)  # spacing is already in mm
        margin_mm = margin_cm * 10  # Convert cm to mm
        margin_voxels = int(margin_mm / spacing_mm)
        roi_bounds = {
            'z_min': z_min_intelligent,  # Intelligent Z bounds using 50% peak cutoff
            'z_max': z_max_intelligent,  # Intelligent Z bounds using 50% peak cutoff
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
        
        # Apply hole filling to the metal mask
        filled_metal_mask, holes_filled_mask = self._fill_metal_holes(metal_mask)
        
        return {
            'mask': filled_metal_mask,  # Return filled mask
            'original_mask': metal_mask,  # Keep original for reference
            'holes_filled_mask': holes_filled_mask,  # Mask of filled holes
            'roi_bounds': roi_bounds,
            'threshold': threshold if threshold else min_metal_hu,
            'center_coords': (center_z, center_y, center_x),
            'method': 'legacy',
            'valid_roi_slices': valid_z_slices if 'valid_z_slices' in locals() and valid_z_slices else list(range(roi_bounds['z_min'], roi_bounds['z_max'])),
            'metadata': {
                'dilation_iterations': dilation_iterations,
                'margin_cm': margin_cm,
                'fw_percentage': fw_percentage,
                'holes_filled': np.sum(holes_filled_mask),
                'hole_filling_enabled': True
            }
        }
    
    def _detect_adaptive_3d(self, ct_volume: np.ndarray, spacing: Tuple[float, float, float],
                           fw_percentage: float = 75.0,
                           margin_cm: float = 1.0,
                           intensity_percentile: float = 99.5,
                           use_star_profiles: bool = True) -> Dict:
        """
        Advanced 3D adaptive metal detection with multi-planar analysis and star profiles.

        This method:
        1. Detects metal using 99.5th percentile threshold (min 2500 HU)
        2. Analyzes axial, coronal, and sagittal projections for spatial extent
        3. Optionally refines threshold using star profile analysis per slice
        4. Restricts ROI to 50th percentile (median) of detected metal HU values
        5. Creates focused ROI centered on core metal implant
        6. Constrains ROI depth to only slices containing substantial metal

        Args:
            use_star_profiles: If True, uses star profile FW% analysis per slice
                             for adaptive thresholding (recovered algorithm)
        """
        # Initial detection using high percentile - more aggressive for speed
        initial_threshold = np.percentile(ct_volume, intensity_percentile)
        # Ensure minimum threshold for metal - increase to focus on true metal
        initial_threshold = max(initial_threshold, 2500)  # Increased from 1500 to 2000
        initial_mask = ct_volume > initial_threshold

        if not np.any(initial_mask):
            return self._empty_result()

        # Multi-planar analysis
        z_coords, y_coords, x_coords = np.where(initial_mask)

        # Analyze extent in each plane
        z_extent = np.max(z_coords) - np.min(z_coords)
        y_extent = np.max(y_coords) - np.min(y_coords)
        x_extent = np.max(x_coords) - np.min(x_coords)

        # STAR PROFILE ENHANCEMENT: Per-slice adaptive thresholding
        if use_star_profiles:
            refined_mask = np.zeros_like(initial_mask, dtype=bool)
            slice_thresholds = []

            # Find center of initial detection for star profile analysis
            center_z = int(np.mean(z_coords))
            center_y = int(np.mean(y_coords))
            center_x = int(np.mean(x_coords))

            # Create temporary ROI bounds for star profile calculation
            temp_roi = {
                'y_min': max(0, center_y - 100),
                'y_max': min(ct_volume.shape[1], center_y + 100),
                'x_min': max(0, center_x - 100),
                'x_max': min(ct_volume.shape[2], center_x + 100)
            }

            # Process slices with detected metal
            for z in np.unique(z_coords):
                slice_mask = initial_mask[z]
                if not np.any(slice_mask):
                    continue

                # Find center of metal on this slice
                y_indices, x_indices = np.where(slice_mask)
                slice_center_y = int(np.mean(y_indices))
                slice_center_x = int(np.mean(x_indices))

                # Calculate star profile threshold for this slice
                slice_threshold = self._calculate_star_threshold(
                    ct_volume[z], slice_center_y, slice_center_x,
                    temp_roi, fw_percentage
                )

                if slice_threshold is not None and slice_threshold > 0:
                    # Apply adaptive threshold to this slice
                    refined_mask[z] = ct_volume[z] > slice_threshold
                    slice_thresholds.append(slice_threshold)
                else:
                    # Fallback to initial threshold
                    refined_mask[z] = slice_mask
                    slice_thresholds.append(initial_threshold)

            # If no valid thresholds were calculated, fall back to initial detection
            if not np.any(refined_mask):
                refined_mask = initial_mask
                slice_thresholds = [initial_threshold]
        else:
            # Use initial threshold directly - anything above 2500 HU is metal
            refined_mask = initial_mask
            slice_thresholds = [initial_threshold]

        # Create individual ROIs for components
        labeled, num_components = label(refined_mask)
        individual_regions = {}

        # Get coordinates of the refined mask for overall ROI
        z_coords, y_coords, x_coords = np.where(refined_mask)
        
        if len(z_coords) > 0:
            # ROI restricted to 50th percentile (median) of detected metal HU values
            # This focuses the ROI on the core metal implant, excluding dimmer edges
            metal_hu_values = ct_volume[refined_mask]
            median_metal_hu = np.percentile(metal_hu_values, 50)

            # Create ROI mask: only include voxels with HU >= median of detected metal
            roi_metal_mask = refined_mask & (ct_volume >= median_metal_hu)

            # Find slices that contain metal above the median threshold
            z_coords_roi, y_coords_roi, x_coords_roi = np.where(roi_metal_mask)
            valid_z_slices = list(np.unique(z_coords_roi))
            
            if valid_z_slices:
                # Use only slices with substantial metal (above median)
                z_min_intelligent = min(valid_z_slices)
                z_max_intelligent = max(valid_z_slices) + 1
            else:
                # Fallback to original bounds if no slices meet criteria
                z_min_intelligent = int(np.min(z_coords))
                z_max_intelligent = int(np.max(z_coords)) + 1
                z_coords_roi, y_coords_roi, x_coords_roi = z_coords, y_coords, x_coords
                print(f"ROI: No voxels met median threshold (≥{median_metal_hu:.1f} HU), using all detected metal")
            # Convert margin from cm to voxels
            # Use only x,y spacing (first two elements), not z spacing
            spacing_mm = np.mean(spacing[:2])  # Use x,y spacing for lateral margin (exclude z)
            margin_mm = margin_cm * 10  # Convert cm to mm
            margin_voxels = int(margin_mm / spacing_mm)
            
            # Create adaptive ROI boxes per slice and per component
            # Limit processing to reasonable number of slices for performance
            from scipy.ndimage import label as scipy_label
            
            max_slices_to_process = min(len(valid_z_slices), 100)  # Increased limit for better coverage
            processed_slices = valid_z_slices[:max_slices_to_process] if len(valid_z_slices) > max_slices_to_process else valid_z_slices
            
            
            for z in processed_slices:
                if z not in individual_regions:
                    individual_regions[z] = []

                slice_mask = roi_metal_mask[z]  # Use ROI-restricted mask (median and above)
                if np.any(slice_mask):
                    # Find connected components on this slice to handle bilateral implants
                    # Use 4-connectivity to better separate bilateral implants
                    slice_components, n_components = scipy_label(slice_mask, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
                    
                    # Sort components by size and process only the largest ones
                    component_sizes = []
                    for comp_id in range(1, n_components + 1):
                        size = np.sum(slice_components == comp_id)
                        component_sizes.append((size, comp_id))
                    
                    # Sort by size (largest first) and limit to top components
                    component_sizes.sort(reverse=True)
                    max_components = min(len(component_sizes), 3)  # Process up to 3 largest components
                    
                    
                    for size, comp_id in component_sizes[:max_components]:
                        # Skip very small components (likely noise)
                        if size < 10:  # Minimum size threshold
                            continue
                            
                        component_mask = slice_components == comp_id
                        
                        # Fast bounding box calculation
                        y_indices, x_indices = np.where(component_mask)
                        
                        if len(y_indices) > 0:
                            # Create ROI box sized specifically for this component on this slice
                            y_min_comp = max(0, np.min(y_indices) - margin_voxels)
                            y_max_comp = min(ct_volume.shape[1], np.max(y_indices) + margin_voxels + 1)
                            x_min_comp = max(0, np.min(x_indices) - margin_voxels)
                            x_max_comp = min(ct_volume.shape[2], np.max(x_indices) + margin_voxels + 1)
                            
                            center_y = int(np.mean(y_indices))
                            center_x = int(np.mean(x_indices))
                            
                            individual_regions[z].append({
                                'component_id': comp_id,
                                'y_min': y_min_comp,
                                'y_max': y_max_comp,
                                'x_min': x_min_comp,
                                'x_max': x_max_comp,
                                'center_y': center_y,
                                'center_x': center_x
                            })
                
                # Merge overlapping ROI boxes on this slice immediately
                if z in individual_regions and len(individual_regions[z]) > 1:
                    individual_regions[z] = self._merge_overlapping_boxes(individual_regions[z])
        
        # Create Conservative ROI - calculate bounds from ROI-restricted metal voxels (median and above)
        if len(z_coords_roi) > 0:
            # Use the ROI-restricted mask (median HU and above) to create a focused ROI
            # This ensures the ROI captures the core metal implant

            # Add extra margin for artifacts that extend beyond metal
            conservative_margin_voxels = margin_voxels + 5  # Extra margin for artifacts

            roi_bounds = {
                'z_min': z_min_intelligent,
                'z_max': z_max_intelligent,
                'y_min': max(0, int(np.min(y_coords_roi)) - conservative_margin_voxels),
                'y_max': min(ct_volume.shape[1], int(np.max(y_coords_roi)) + conservative_margin_voxels + 1),
                'x_min': max(0, int(np.min(x_coords_roi)) - conservative_margin_voxels),
                'x_max': min(ct_volume.shape[2], int(np.max(x_coords_roi)) + conservative_margin_voxels + 1)
            }


            # Override individual regions to use the same conservative ROI for all slices
            # This ensures consistent ROI display and artifact containment
            conservative_region = {
                'component_id': 0,
                'y_min': roi_bounds['y_min'],
                'y_max': roi_bounds['y_max'],
                'x_min': roi_bounds['x_min'],
                'x_max': roi_bounds['x_max'],
                'center_y': int(np.mean(y_coords_roi)),
                'center_x': int(np.mean(x_coords_roi))
            }

            # Apply this conservative ROI to all valid slices
            individual_regions = {}
            for z in valid_z_slices:
                individual_regions[z] = [conservative_region.copy()]
        else:
            # No metal found, return empty result
            return self._empty_result()
        
        # Apply hole filling to the refined mask
        filled_metal_mask, holes_filled_mask = self._fill_metal_holes(refined_mask)
        
        # Update coordinates and ROI based on filled mask
        if np.any(filled_metal_mask):
            z_coords_filled, y_coords_filled, x_coords_filled = np.where(filled_metal_mask)
            
            # Update ROI bounds if hole filling expanded the region
            if len(z_coords_filled) > 0:
                # Recalculate median threshold with filled mask
                # Only calculate from original metal (not filled holes)
                metal_hu_values_filled = ct_volume[refined_mask]
                median_metal_hu_filled = np.percentile(metal_hu_values_filled, 50)
                
                # Create ROI mask from filled mask using median threshold
                roi_metal_mask_filled = filled_metal_mask & (ct_volume >= median_metal_hu_filled)
                z_coords_roi_filled, y_coords_roi_filled, x_coords_roi_filled = np.where(roi_metal_mask_filled)
                valid_z_slices_filled = list(np.unique(z_coords_roi_filled))
                
                if valid_z_slices_filled:
                    z_min_filled = min(valid_z_slices_filled)
                    z_max_filled = max(valid_z_slices_filled) + 1
                else:
                    z_min_filled = int(np.min(z_coords_filled))
                    z_max_filled = int(np.max(z_coords_filled)) + 1
                
                # Update ROI bounds with potential expansion from hole filling
                spacing_mm = np.mean(spacing[:2])
                margin_mm = margin_cm * 10
                margin_voxels = int(margin_mm / spacing_mm)
                conservative_margin_voxels = margin_voxels + 5
                
                roi_bounds = {
                    'z_min': z_min_filled,
                    'z_max': z_max_filled,
                    'y_min': max(0, int(np.min(y_coords_roi_filled)) - conservative_margin_voxels),
                    'y_max': min(ct_volume.shape[1], int(np.max(y_coords_roi_filled)) + conservative_margin_voxels + 1),
                    'x_min': max(0, int(np.min(x_coords_roi_filled)) - conservative_margin_voxels),
                    'x_max': min(ct_volume.shape[2], int(np.max(x_coords_roi_filled)) + conservative_margin_voxels + 1)
                }

                # Update individual regions to use conservative ROI for all valid slices
                conservative_region = {
                    'component_id': 0,
                    'y_min': roi_bounds['y_min'],
                    'y_max': roi_bounds['y_max'],
                    'x_min': roi_bounds['x_min'],
                    'x_max': roi_bounds['x_max'],
                    'center_y': int(np.mean(y_coords_roi_filled)),
                    'center_x': int(np.mean(x_coords_roi_filled))
                }
                
                individual_regions = {}
                for z in valid_z_slices_filled:
                    individual_regions[z] = [conservative_region.copy()]
        
        # Calculate average threshold
        avg_threshold = np.mean(slice_thresholds) if slice_thresholds else initial_threshold
        
        return {
            'mask': filled_metal_mask,  # Return filled mask instead of original
            'original_mask': refined_mask,  # Keep original mask for reference
            'holes_filled_mask': holes_filled_mask,  # Mask of filled holes only
            'roi_bounds': roi_bounds,
            'threshold': avg_threshold,
            'threshold_evolution': slice_thresholds,
            'individual_regions': individual_regions,
            'center_coords': (int(np.mean(z_coords_roi_filled)), int(np.mean(y_coords_roi_filled)), int(np.mean(x_coords_roi_filled))) if 'z_coords_roi_filled' in locals() and len(z_coords_roi_filled) > 0 else (int(np.mean(z_coords_roi)), int(np.mean(y_coords_roi)), int(np.mean(x_coords_roi))),
            'method': 'adaptive_3d',
            'valid_roi_slices': valid_z_slices_filled if 'valid_z_slices_filled' in locals() and valid_z_slices_filled else (valid_z_slices if 'valid_z_slices' in locals() and valid_z_slices else list(range(roi_bounds['z_min'], roi_bounds['z_max']))),
            'metadata': {
                'fw_percentage': fw_percentage,
                'margin_cm': margin_cm,
                'intensity_percentile': intensity_percentile,
                'num_components': num_components,
                'extent_voxels': {'z': z_extent, 'y': y_extent, 'x': x_extent},
                'holes_filled': np.sum(holes_filled_mask),
                'hole_filling_enabled': True,
                'median_metal_hu': median_metal_hu_filled if 'median_metal_hu_filled' in locals() else (median_metal_hu if 'median_metal_hu' in locals() else None),
                'roi_restriction': '50th percentile (median) of detected metal HU values'
            }
        }
    
    def _merge_overlapping_boxes(self, boxes: List[Dict]) -> List[Dict]:
        """Fast merge of overlapping ROI boxes on the same slice."""
        if len(boxes) <= 1:
            return boxes
        
        # Quick optimization: if only 2 boxes, just check once
        if len(boxes) == 2:
            box1, box2 = boxes[0], boxes[1]
            
            # Calculate overlap area
            x_overlap = max(0, min(box1['x_max'], box2['x_max']) - max(box1['x_min'], box2['x_min']))
            y_overlap = max(0, min(box1['y_max'], box2['y_max']) - max(box1['y_min'], box2['y_min']))
            overlap_area = x_overlap * y_overlap
            
            # Calculate individual box areas
            box1_area = (box1['x_max'] - box1['x_min']) * (box1['y_max'] - box1['y_min'])
            box2_area = (box2['x_max'] - box2['x_min']) * (box2['y_max'] - box2['y_min'])
            min_area = min(box1_area, box2_area)
            
            # Only merge if overlap is significant (>20% of smaller box)
            if overlap_area > 0 and overlap_area > 0.2 * min_area:
                # Significant overlap, merge them
                return [{
                    'component_id': min(box1['component_id'], box2['component_id']),
                    'y_min': min(box1['y_min'], box2['y_min']),
                    'y_max': max(box1['y_max'], box2['y_max']),
                    'x_min': min(box1['x_min'], box2['x_min']),
                    'x_max': max(box1['x_max'], box2['x_max']),
                    'center_y': (box1['center_y'] + box2['center_y']) // 2,
                    'center_x': (box1['center_x'] + box2['center_x']) // 2
                }]
            else:
                return boxes  # No overlap, keep separate
        
        # For more than 2 boxes, use simple greedy approach
        # Most cases will be <= 2 boxes per slice anyway
        merged = []
        unprocessed = boxes.copy()
        
        while unprocessed:
            current = unprocessed.pop(0)
            
            # Find first significantly overlapping box (if any)
            merged_any = False
            for i, other in enumerate(unprocessed):
                # Calculate overlap area
                x_overlap = max(0, min(current['x_max'], other['x_max']) - max(current['x_min'], other['x_min']))
                y_overlap = max(0, min(current['y_max'], other['y_max']) - max(current['y_min'], other['y_min']))
                overlap_area = x_overlap * y_overlap
                
                if overlap_area > 0:
                    # Calculate box areas
                    current_area = (current['x_max'] - current['x_min']) * (current['y_max'] - current['y_min'])
                    other_area = (other['x_max'] - other['x_min']) * (other['y_max'] - other['y_min'])
                    min_area = min(current_area, other_area)
                    
                    # Only merge if overlap is significant (>20% of smaller box)
                    if overlap_area > 0.2 * min_area:
                        # Merge and remove the other box
                        current = {
                            'component_id': min(current['component_id'], other['component_id']),
                            'y_min': min(current['y_min'], other['y_min']),
                            'y_max': max(current['y_max'], other['y_max']),
                            'x_min': min(current['x_min'], other['x_min']),
                            'x_max': max(current['x_max'], other['x_max']),
                            'center_y': (current['center_y'] + other['center_y']) // 2,
                            'center_x': (current['center_x'] + other['center_x']) // 2
                        }
                        unprocessed.pop(i)
                        merged_any = True
                        break
            
            if not merged_any:
                merged.append(current)
        
        return merged
    
    def _calculate_star_threshold(self, slice_data: np.ndarray, center_y: int, center_x: int,
                                 roi_bounds: Dict, fw_percentage: float) -> Optional[float]:
        """
        Calculate threshold using FULL star profile analysis.

        This is the recovered FW75% algorithm that:
        1. Shoots 16 radial lines from center to ROI boundaries
        2. Finds peak HU value along each line
        3. Calculates FW% threshold for each profile
        4. Returns average threshold across all profiles

        Args:
            slice_data: 2D CT slice
            center_y, center_x: Metal center coordinates
            roi_bounds: ROI boundaries
            fw_percentage: Full Width percentage (typically 75%)

        Returns:
            Average threshold across all star profiles, or None if calculation fails
        """
        from skimage.draw import line

        y_min, y_max = roi_bounds.get('y_min', 0), roi_bounds.get('y_max', slice_data.shape[0])
        x_min, x_max = roi_bounds.get('x_min', 0), roi_bounds.get('x_max', slice_data.shape[1])

        # Calculate intermediate points for 16-point star
        y_mid = (y_min + y_max) // 2
        x_mid = (x_min + x_max) // 2

        y_q1 = (y_min + y_mid) // 2
        y_q3 = (y_mid + y_max) // 2
        x_q1 = (x_min + x_mid) // 2
        x_q3 = (x_mid + x_max) // 2

        # Define 16 endpoints (same as visualization)
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

        thresholds = []

        for end_y, end_x in endpoints:
            # Ensure endpoints are within bounds
            end_y = max(0, min(slice_data.shape[0] - 1, end_y))
            end_x = max(0, min(slice_data.shape[1] - 1, end_x))

            # Get line coordinates
            rr, cc = line(center_y, center_x, end_y, end_x)

            # Ensure coordinates are within image bounds
            valid_mask = (rr >= 0) & (rr < slice_data.shape[0]) & (cc >= 0) & (cc < slice_data.shape[1])
            rr = rr[valid_mask]
            cc = cc[valid_mask]

            if len(rr) == 0:
                continue

            # Get HU values along this profile
            hu_values = slice_data[rr, cc]

            # Find peak HU value along this line
            peak_hu = np.max(hu_values)

            # Calculate FW% threshold for this profile
            profile_threshold = peak_hu * (fw_percentage / 100.0)
            thresholds.append(profile_threshold)

        # Return average threshold across all profiles
        if thresholds:
            avg_threshold = np.mean(thresholds)
            return avg_threshold

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
    
    def _fill_metal_holes(self, metal_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fill holes in metal mask to capture hollow implants (like hip prostheses).
        
        Args:
            metal_mask: 3D binary mask of detected metal
            
        Returns:
            Tuple of (filled_mask, holes_filled_mask)
                - filled_mask: Metal mask with holes filled
                - holes_filled_mask: Mask of only the filled holes
        """
        filled_mask = np.zeros_like(metal_mask, dtype=bool)
        holes_filled_mask = np.zeros_like(metal_mask, dtype=bool)
        
        # Process each slice individually for better hole detection
        for z in range(metal_mask.shape[0]):
            slice_mask = metal_mask[z]
            
            if np.any(slice_mask):
                # Fill holes in this slice
                slice_filled = binary_fill_holes(slice_mask)
                filled_mask[z] = slice_filled
                
                # Track which pixels were filled (the holes)
                holes_filled_mask[z] = slice_filled & ~slice_mask
        
        return filled_mask, holes_filled_mask
    
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


def detect_metal_adaptive_3d(ct_volume: np.ndarray, spacing: Tuple[float, float, float], **kwargs) -> Dict:
    """Adaptive 3D metal detection for backward compatibility."""
    detector = MetalDetector(MetalDetectionMethod.ADAPTIVE_3D)
    return detector.detect(ct_volume, spacing, **kwargs)


def detect_metal_multi_component(ct_volume: np.ndarray, spacing: Tuple[float, float, float], **kwargs) -> Dict:
    """Multi-component metal detection for backward compatibility."""
    detector = MetalDetector(MetalDetectionMethod.MULTI_COMPONENT)
    return detector.detect(ct_volume, spacing, **kwargs)