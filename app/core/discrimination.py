"""
Unified Artifact Discrimination Module
Consolidates all bone/artifact discrimination algorithms into a single interface.
Supports GPU acceleration via CuPy when available.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from enum import Enum
from scipy.ndimage import distance_transform_edt, gaussian_filter, sobel
from scipy.ndimage import binary_opening, binary_closing, label

# Import GPU utilities
from core.gpu_utils import (
    is_gpu_available, distance_transform_edt_gpu, gaussian_filter_gpu,
    calculate_local_variance_gpu, batch_neighborhood_analysis_gpu,
    GPU_AVAILABLE
)


class DiscriminationMethod(Enum):
    """Available discrimination methods."""
    DISTANCE_BASED = "distance_based"      # Fast, distance from metal
    EDGE_BASED = "edge_based"              # Enhanced edge coherence
    TEXTURE_BASED = "texture_based"        # Advanced texture/gradient
    STAR_PROFILE = "star_profile"          # Star profile-based discrimination (recovered)


class ArtifactDiscriminator:
    """
    Unified interface for bone vs bright artifact discrimination.

    Methods:
        DISTANCE_BASED: Fast discrimination using distance from metal
        EDGE_BASED: Enhanced edge coherence analysis
        TEXTURE_BASED: Advanced texture and gradient features
    """
    
    def __init__(self, method: DiscriminationMethod = DiscriminationMethod.DISTANCE_BASED):
        """
        Initialize discriminator with specified method.
        
        Args:
            method: Discrimination method to use
        """
        self.method = method
        self.discriminators = {
            DiscriminationMethod.DISTANCE_BASED: self._discriminate_distance,
            DiscriminationMethod.EDGE_BASED: self._discriminate_edge,
            DiscriminationMethod.TEXTURE_BASED: self._discriminate_texture,
            DiscriminationMethod.STAR_PROFILE: self._discriminate_star_profile,
        }
    
    def discriminate(self, ct_volume: np.ndarray, metal_mask: np.ndarray,
                    bright_mask: np.ndarray, spacing: Tuple[float, float, float],
                    **kwargs) -> Dict:
        """
        Discriminate between bone and bright artifacts.
        
        Args:
            ct_volume: 3D CT volume in HU
            metal_mask: Binary mask of metal regions
            bright_mask: Binary mask of bright regions to discriminate
            spacing: Voxel spacing (z, y, x) in mm
            **kwargs: Method-specific parameters
            
        Returns:
            Dictionary containing:
                - bone_mask: Binary mask of bone tissue
                - artifact_mask: Binary mask of bright artifacts
                - confidence_map: Confidence scores for discrimination
                - metadata: Method-specific metadata
        """
        discriminator = self.discriminators.get(self.method)
        if not discriminator:
            raise ValueError(f"Unknown discrimination method: {self.method}")
        
        return discriminator(ct_volume, metal_mask, bright_mask, spacing, **kwargs)
    
    def _discriminate_distance(self, ct_volume: np.ndarray, metal_mask: np.ndarray,
                              bright_mask: np.ndarray, spacing: Tuple[float, float, float],
                              max_distance_cm: float = 10.0,
                              bone_hu_low: float = 150.0,
                              bone_hu_high: float = 1500.0,
                              tissue_hu_low: float = -100.0,
                              tissue_hu_high: float = 300.0,
                              use_gpu: bool = True) -> Dict:
        """
        Distance-based discrimination with configurable HU thresholds.
        Supports GPU acceleration when CuPy is available.

        Principle: Bright artifacts are typically closer to metal than bone.

        Algorithm:
        1. Calculate distance from metal (GPU accelerated)
        2. Apply distance-based classification using user-specified HU ranges
        3. Use local smoothness as secondary criterion (GPU accelerated)
        4. Classify artifacts as overlying bone or tissue (GPU batch processing)

        Args:
            max_distance_cm: Maximum distance from metal to consider
            bone_hu_low: Lower bound of bone HU range (from slider)
            bone_hu_high: Upper bound of bone HU range (from slider)
            tissue_hu_low: Lower bound of soft tissue HU range
            tissue_hu_high: Upper bound of soft tissue HU range
            use_gpu: Whether to use GPU acceleration if available
        """
        # Calculate distance from metal (GPU accelerated)
        inverted_metal = np.logical_not(metal_mask)
        distances = distance_transform_edt_gpu(inverted_metal, spacing, use_gpu=use_gpu)
        distances_cm = distances / 10.0

        # Calculate local variance (GPU accelerated)
        local_variance = calculate_local_variance_gpu(ct_volume, use_gpu=use_gpu)

        # Classification based on distance and smoothness
        bone_mask = np.zeros_like(bright_mask)
        artifact_bone_mask = np.zeros_like(bright_mask)
        artifact_tissue_mask = np.zeros_like(bright_mask)
        confidence_map = np.zeros_like(ct_volume, dtype=float)

        # Bone criteria using user-specified HU range:
        # - At moderate distance from metal (not too close)
        # - Has HU values within user-specified bone range
        # - Has relatively low variance (smooth structure)
        bone_criteria = bright_mask & \
                       (distances_cm > 1.0) & \
                       (ct_volume >= bone_hu_low) & (ct_volume <= bone_hu_high) & \
                       (local_variance < 300)

        # Artifact characteristics: very close to metal OR high variance OR outside bone HU range
        artifact_criteria = bright_mask & \
                          ((distances_cm < 1.0) |  # Very close to metal
                           (local_variance > 400) |  # High variance
                           (ct_volume > bone_hu_high))  # Above bone range (likely artifact)

        bone_mask = bone_criteria
        artifact_mask_temp = artifact_criteria & (~bone_mask)

        # Handle unclassified regions
        unclassified = bright_mask & (~bone_mask) & (~artifact_mask_temp)
        if np.any(unclassified):
            # Use HU value as primary criterion for unclassified
            # Use middle of bone range as typical bone
            bone_mid = (bone_hu_low + bone_hu_high) / 2
            bone_hu_range = unclassified & (ct_volume >= bone_hu_low) & (ct_volume <= bone_mid)
            artifact_hu_range = unclassified & (ct_volume > bone_mid)

            # Secondary criterion: distance
            very_close = unclassified & (distances_cm < 0.5)
            moderate_dist = unclassified & (distances_cm >= 0.5) & (distances_cm < 3.0)

            # Combine criteria
            bone_mask |= (bone_hu_range & moderate_dist)
            artifact_mask_temp |= (artifact_hu_range | very_close)

            # Remaining unclassified: use distance threshold
            still_unclassified = unclassified & (~bone_mask) & (~artifact_mask_temp)
            near_metal = still_unclassified & (distances_cm < 2.0)
            far_from_metal = still_unclassified & (distances_cm >= 2.0)
            artifact_mask_temp |= near_metal
            bone_mask |= far_from_metal

        # Now classify artifacts as overlying bone or tissue based on neighborhood
        # Use GPU batch processing for speed
        artifact_coords = np.where(artifact_mask_temp)
        n_artifacts = len(artifact_coords[0])

        if n_artifacts > 0:
            # Batch neighborhood analysis (GPU accelerated)
            bone_ratios, tissue_ratios = batch_neighborhood_analysis_gpu(
                ct_volume, artifact_coords,
                bone_hu_low, bone_hu_high,
                tissue_hu_low, tissue_hu_high,
                window_size=5, use_gpu=use_gpu
            )

            # Classify based on ratios
            for i in range(n_artifacts):
                z, y, x = artifact_coords[0][i], artifact_coords[1][i], artifact_coords[2][i]

                if bone_ratios[i] > tissue_ratios[i]:
                    artifact_bone_mask[z, y, x] = True
                elif tissue_ratios[i] > bone_ratios[i]:
                    artifact_tissue_mask[z, y, x] = True
                else:
                    # Tie: use distance
                    if distances_cm[z, y, x] < 1.5:
                        artifact_tissue_mask[z, y, x] = True
                    else:
                        artifact_bone_mask[z, y, x] = True

        # Combined artifact mask for backward compatibility
        artifact_mask = artifact_bone_mask | artifact_tissue_mask

        # Calculate confidence based on distance and variance
        confidence_map[bright_mask] = np.clip(
            1.0 - (local_variance[bright_mask] / 500.0), 0, 1
        )

        return {
            'bone_mask': bone_mask,
            'artifact_mask': artifact_mask,
            'artifact_bone_mask': artifact_bone_mask,
            'artifact_tissue_mask': artifact_tissue_mask,
            'confidence_map': confidence_map,
            'distance_map': distances_cm,
            'method': 'distance_based',
            'metadata': {
                'max_distance_cm': max_distance_cm,
                'bone_hu_low': bone_hu_low,
                'bone_hu_high': bone_hu_high,
                'bone_voxels': np.sum(bone_mask),
                'artifact_bone_voxels': np.sum(artifact_bone_mask),
                'artifact_tissue_voxels': np.sum(artifact_tissue_mask),
                'artifact_voxels': np.sum(artifact_mask),
                'gpu_used': use_gpu and GPU_AVAILABLE
            }
        }
    
    def _discriminate_edge(self, ct_volume: np.ndarray, metal_mask: np.ndarray,
                          bright_mask: np.ndarray, spacing: Tuple[float, float, float],
                          **kwargs) -> Dict:
        """
        Enhanced edge-based discrimination.
        
        Principle: Bone has coherent, continuous edges while artifacts have chaotic edges.
        
        Algorithm:
        1. Compute edge coherence using structure tensor
        2. Analyze edge continuity across slices
        3. Measure radial vs tangential edge alignment
        """
        # Compute gradients
        grad_z = sobel(ct_volume, axis=0)
        grad_y = sobel(ct_volume, axis=1)
        grad_x = sobel(ct_volume, axis=2)
        
        # Gradient magnitude
        grad_mag = np.sqrt(grad_z**2 + grad_y**2 + grad_x**2)
        
        # Edge coherence (simplified structure tensor analysis)
        smoothed_grad_mag = gaussian_filter(grad_mag, sigma=1.0)
        
        # Initialize masks
        bone_mask = np.zeros_like(bright_mask)
        artifact_mask = np.zeros_like(bright_mask)
        confidence_map = np.zeros_like(ct_volume, dtype=float)
        
        # Bone: strong, coherent edges
        strong_edges = (smoothed_grad_mag > np.percentile(smoothed_grad_mag[bright_mask], 75))
        coherent_regions = bright_mask & strong_edges
        
        # Analyze edge continuity
        for z in range(ct_volume.shape[0]):
            if not np.any(bright_mask[z]):
                continue
            
            slice_bright = bright_mask[z]
            slice_edges = coherent_regions[z]
            
            # Bone has continuous edges
            labeled_edges, num_features = label(slice_edges)
            for i in range(1, num_features + 1):
                component = labeled_edges == i
                if np.sum(component) > 50:  # Significant edge component
                    bone_mask[z] |= component & slice_bright
        
        # Remaining bright regions are artifacts
        artifact_mask = bright_mask & (~bone_mask)
        
        # Confidence based on edge strength
        confidence_map[bright_mask] = np.clip(
            smoothed_grad_mag[bright_mask] / np.max(smoothed_grad_mag[bright_mask]), 0, 1
        )
        
        return {
            'bone_mask': bone_mask,
            'artifact_mask': artifact_mask,
            'confidence_map': confidence_map,
            'edge_magnitude': grad_mag,
            'method': 'edge_based',
            'metadata': {
                'bone_voxels': np.sum(bone_mask),
                'artifact_voxels': np.sum(artifact_mask)
            }
        }
    
    def _discriminate_texture(self, ct_volume: np.ndarray, metal_mask: np.ndarray,
                             bright_mask: np.ndarray, spacing: Tuple[float, float, float],
                             **kwargs) -> Dict:
        """
        Advanced texture-based discrimination using multiple features.
        
        Principle: Bone and artifacts have different textural characteristics.
        
        Features:
        - Local variance
        - Gradient direction variance
        - Smoothness measures
        - Distance from metal
        """
        # Calculate texture features
        local_variance = self._calculate_local_variance(ct_volume, window_size=5)
        gradient_variance = self._calculate_gradient_variance(ct_volume)
        smoothness = self._calculate_smoothness(ct_volume)
        
        # Distance from metal
        inverted_metal = np.logical_not(metal_mask)
        distances = distance_transform_edt(inverted_metal, sampling=spacing)
        distances_cm = distances / 10.0
        
        # Combine features for classification
        # Normalize features
        features = {
            'variance': local_variance / (np.max(local_variance) + 1e-10),
            'gradient': gradient_variance / (np.max(gradient_variance) + 1e-10),
            'smoothness': smoothness / (np.max(smoothness) + 1e-10),
            'distance': np.clip(distances_cm / 10.0, 0, 1)
        }
        
        # Artifact score (higher = more likely artifact)
        artifact_score = np.zeros_like(ct_volume, dtype=float)
        
        # Weights for features
        weights = {
            'variance': 0.3,      # High variance = artifact
            'gradient': 0.3,      # High gradient variance = artifact
            'smoothness': -0.2,   # Low smoothness = artifact
            'distance': -0.2      # Close to metal = artifact
        }
        
        for feature_name, weight in weights.items():
            artifact_score += weight * features[feature_name]
        
        # Normalize scores
        artifact_score = np.clip(artifact_score, 0, 1)
        
        # Threshold to create masks (using Otsu-like approach)
        bright_scores = artifact_score[bright_mask]
        if len(bright_scores) > 0:
            threshold = np.median(bright_scores)
        else:
            threshold = 0.5
        
        # Create masks
        artifact_mask = bright_mask & (artifact_score > threshold)
        bone_mask = bright_mask & (artifact_score <= threshold)
        
        # Post-processing
        artifact_mask = binary_opening(artifact_mask, iterations=1)
        artifact_mask = binary_closing(artifact_mask, iterations=1)
        bone_mask = binary_opening(bone_mask, iterations=1)
        bone_mask = binary_closing(bone_mask, iterations=1)
        
        # Confidence map
        confidence_map = np.abs(artifact_score - threshold) * 2
        confidence_map = np.clip(confidence_map, 0, 1)
        
        return {
            'bone_mask': bone_mask,
            'artifact_mask': artifact_mask,
            'confidence_map': confidence_map,
            'artifact_score': artifact_score,
            'method': 'texture_based',
            'metadata': {
                'threshold': threshold,
                'bone_voxels': np.sum(bone_mask),
                'artifact_voxels': np.sum(artifact_mask)
            }
        }
    
    def _calculate_local_variance(self, volume: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Calculate local variance in a sliding window."""
        from scipy.ndimage import uniform_filter
        
        mean = uniform_filter(volume.astype(float), size=window_size)
        sqr_mean = uniform_filter(volume.astype(float)**2, size=window_size)
        variance = sqr_mean - mean**2
        
        return np.maximum(variance, 0)
    
    def _calculate_gradient_variance(self, volume: np.ndarray) -> np.ndarray:
        """Calculate gradient direction variance."""
        grad_z = sobel(volume, axis=0)
        grad_y = sobel(volume, axis=1)
        grad_x = sobel(volume, axis=2)
        
        # Calculate angles
        with np.errstate(divide='ignore', invalid='ignore'):
            angles_xy = np.arctan2(grad_y, grad_x)
            angles_xz = np.arctan2(grad_z, grad_x)
        
        # Local variance of angles
        angle_variance = self._calculate_local_variance(angles_xy, window_size=3)
        angle_variance += self._calculate_local_variance(angles_xz, window_size=3)
        
        return angle_variance
    
    def _calculate_smoothness(self, volume: np.ndarray) -> np.ndarray:
        """Calculate local smoothness measure."""
        smoothed = gaussian_filter(volume.astype(float), sigma=2.0)
        difference = np.abs(volume - smoothed)
        smoothness = 1.0 / (1.0 + difference)
        
        return smoothness
    
    def _discriminate_star_profile(self, ct_volume: np.ndarray, metal_mask: np.ndarray,
                                   bright_mask: np.ndarray, spacing: Tuple[float, float, float],
                                   num_angles: int = 16,
                                   bone_hu_low: float = 150.0,
                                   bone_hu_high: float = 1500.0,
                                   tissue_hu_low: float = -100.0,
                                   tissue_hu_high: float = 300.0,
                                   w_hu: float = 0.45,
                                   w_width: float = 0.35,
                                   w_smooth: float = 0.25,
                                   w_gradient: float = 0.25,
                                   use_gpu: bool = True) -> Dict:
        """
        Star profile-based discrimination with tissue classification.
        Supports GPU acceleration for distance transform and neighborhood analysis.

        Principle: Analyzes radial HU profiles to distinguish bone from artifacts,
        then further classifies artifacts as overlying bone or soft tissue.

        Bone characteristics:
        - Broad peaks (3-5mm width)
        - Smooth transitions (low gradient variance)
        - Consistent across angles (low directional variance)
        - Gradual gradients
        - HU values within typical bone range (uses bone_hu_low/bone_hu_high)

        Artifact characteristics:
        - Narrow peaks (<2mm width)
        - Sharp edges (high gradient variance)
        - Variable across angles (high directional variance)
        - Steep gradients
        - HU values often outside typical bone range

        Args:
            num_angles: Number of radial profiles to analyze (default: 16)
            bone_hu_low: Lower bound of expected bone HU range (default: 150)
            bone_hu_high: Upper bound of expected bone HU range (default: 1500)
            tissue_hu_low: Lower bound of soft tissue HU range (default: -100)
            tissue_hu_high: Upper bound of soft tissue HU range (default: 300)
        """
        from scipy.ndimage import gaussian_filter1d
        from skimage.draw import line

        bone_mask = np.zeros_like(bright_mask)
        artifact_bone_mask = np.zeros_like(bright_mask)  # Artifacts over bone
        artifact_tissue_mask = np.zeros_like(bright_mask)  # Artifacts over soft tissue
        confidence_map = np.zeros_like(ct_volume, dtype=float)

        # Calculate distance from metal for tie-breaking (GPU accelerated)
        inverted_metal = np.logical_not(metal_mask)
        distance_from_metal = distance_transform_edt_gpu(inverted_metal, spacing, use_gpu=use_gpu) / 10.0  # cm

        # Process each slice
        for z in range(ct_volume.shape[0]):
            if not np.any(bright_mask[z]) or not np.any(metal_mask[z]):
                continue

            # One star per connected metal component (bilateral-safe): with two
            # implants, the all-metal centroid lands between them and the rays
            # sample anatomy unrelated to either implant.
            stars = self._get_slice_stars(ct_volume[z], metal_mask[z], num_angles)
            if not stars:
                continue

            star_centers = np.array([[s['center_y'], s['center_x']] for s in stars])

            # Analyze each bright pixel on this slice
            bright_coords = np.where(bright_mask[z])

            for i in range(len(bright_coords[0])):
                pixel_y = bright_coords[0][i]
                pixel_x = bright_coords[1][i]

                # Judge this pixel using the star of its nearest metal component
                d2 = (star_centers[:, 0] - pixel_y) ** 2 + (star_centers[:, 1] - pixel_x) ** 2
                star = stars[int(np.argmin(d2))]
                center_y = star['center_y']
                center_x = star['center_x']

                # Calculate angle to this pixel from its component's center
                dy = pixel_y - center_y
                dx = pixel_x - center_x
                angle = np.arctan2(dy, dx)
                if angle < 0:
                    angle += 2 * np.pi

                # Find nearest profile
                profile_idx = int((angle / (2 * np.pi)) * num_angles) % num_angles
                if profile_idx >= len(star['profiles']):
                    continue

                profile = star['profiles'][profile_idx]

                # Calculate distance from metal center
                distance_voxels = np.sqrt(dy**2 + dx**2)
                distance_mm = distance_voxels * np.mean(spacing[:2])

                # Get the actual HU value at this pixel
                pixel_hu = ct_volume[z, pixel_y, pixel_x]

                # Analyze profile characteristics at this distance (now includes HU)
                characteristics = self._analyze_profile_characteristics(
                    profile, distance_mm, spacing,
                    pixel_hu=pixel_hu,
                    bone_hu_low=bone_hu_low,
                    bone_hu_high=bone_hu_high,
                    w_hu=w_hu, w_width=w_width,
                    w_smooth=w_smooth, w_gradient=w_gradient
                )

                # Classify based on characteristics
                is_bone = self._classify_from_profile(characteristics)

                if is_bone:
                    bone_mask[z, pixel_y, pixel_x] = True
                    confidence_map[z, pixel_y, pixel_x] = characteristics['confidence']
                else:
                    # This is an artifact - classify what tissue it's overlying
                    # Use neighborhood analysis to determine context
                    context = self._analyze_neighborhood_context(
                        ct_volume, z, pixel_y, pixel_x,
                        bone_hu_low, bone_hu_high,
                        tissue_hu_low, tissue_hu_high
                    )

                    dist_cm = distance_from_metal[z, pixel_y, pixel_x]

                    if context['bone_ratio'] > context['tissue_ratio']:
                        artifact_bone_mask[z, pixel_y, pixel_x] = True
                    elif context['tissue_ratio'] > context['bone_ratio']:
                        artifact_tissue_mask[z, pixel_y, pixel_x] = True
                    else:
                        # Tie: use distance as decider
                        # Close to metal (<1.5cm) → likely tissue (muscle around implant)
                        # Far from metal (>=1.5cm) → likely bone
                        if dist_cm < 1.5:
                            artifact_tissue_mask[z, pixel_y, pixel_x] = True
                        else:
                            artifact_bone_mask[z, pixel_y, pixel_x] = True

                    confidence_map[z, pixel_y, pixel_x] = characteristics['confidence']

        # Create combined artifact mask for backward compatibility
        artifact_mask = artifact_bone_mask | artifact_tissue_mask

        return {
            'bone_mask': bone_mask,
            'artifact_mask': artifact_mask,  # Combined for backward compatibility
            'artifact_bone_mask': artifact_bone_mask,
            'artifact_tissue_mask': artifact_tissue_mask,
            'confidence_map': confidence_map,
            'method': 'star_profile',
            'metadata': {
                'num_angles': num_angles,
                'bone_hu_low': bone_hu_low,
                'bone_hu_high': bone_hu_high,
                'tissue_hu_low': tissue_hu_low,
                'tissue_hu_high': tissue_hu_high,
                'bone_voxels': np.sum(bone_mask),
                'artifact_bone_voxels': np.sum(artifact_bone_mask),
                'artifact_tissue_voxels': np.sum(artifact_tissue_mask),
                'artifact_voxels': np.sum(artifact_mask),
                'gpu_used': use_gpu and GPU_AVAILABLE
            }
        }

    def _analyze_neighborhood_context(self, ct_volume: np.ndarray, z: int, y: int, x: int,
                                      bone_hu_low: float, bone_hu_high: float,
                                      tissue_hu_low: float, tissue_hu_high: float,
                                      window_size: int = 5) -> Dict:
        """
        Analyze the neighborhood around a voxel to determine tissue context.

        Returns ratio of bone-like vs tissue-like neighbors.
        """
        half_w = window_size // 2
        z_min = max(0, z - 1)
        z_max = min(ct_volume.shape[0], z + 2)
        y_min = max(0, y - half_w)
        y_max = min(ct_volume.shape[1], y + half_w + 1)
        x_min = max(0, x - half_w)
        x_max = min(ct_volume.shape[2], x + half_w + 1)

        neighborhood = ct_volume[z_min:z_max, y_min:y_max, x_min:x_max]
        total_voxels = neighborhood.size

        if total_voxels == 0:
            return {'bone_ratio': 0.0, 'tissue_ratio': 0.0}

        # Count voxels in bone and tissue HU ranges
        bone_count = np.sum((neighborhood >= bone_hu_low) & (neighborhood <= bone_hu_high))
        tissue_count = np.sum((neighborhood >= tissue_hu_low) & (neighborhood <= tissue_hu_high))

        return {
            'bone_ratio': bone_count / total_voxels,
            'tissue_ratio': tissue_count / total_voxels
        }

    def _get_slice_stars(self, slice_data: np.ndarray, metal_slice: np.ndarray,
                         num_angles: int, min_component_size: int = 10,
                         merge_distance_px: int = 10) -> list:
        """
        Build one star (center + radial profiles) per metal implant on this
        slice, so bilateral implants each get their own star instead of a
        single star centered between them.

        A single implant often fragments into several connected components
        (cup, screws, satellite bits), so components within merge_distance_px
        of each other are grouped into one star; bilateral implants sit far
        apart and stay separate. Groups with fewer than min_component_size
        metal pixels are ignored as noise; if all groups are tiny, falls back
        to a single star at the overall metal centroid.

        Returns list of dicts: {'center_y', 'center_x', 'profiles'}.
        """
        from scipy.ndimage import label as ndi_label, binary_dilation

        merged = binary_dilation(metal_slice, iterations=merge_distance_px)
        labeled, n_components = ndi_label(
            merged, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        )

        stars = []
        for comp_id in range(1, n_components + 1):
            # Centroid from the actual metal pixels, not the dilated blob
            component = metal_slice & (labeled == comp_id)
            if np.sum(component) < min_component_size:
                continue
            ys, xs = np.where(component)
            center_y, center_x = int(np.mean(ys)), int(np.mean(xs))
            stars.append({
                'center_y': center_y,
                'center_x': center_x,
                'profiles': self._get_star_profiles_detailed(
                    slice_data, center_y, center_x, num_angles
                ),
            })

        if not stars and np.any(metal_slice):
            ys, xs = np.where(metal_slice)
            center_y, center_x = int(np.mean(ys)), int(np.mean(xs))
            stars.append({
                'center_y': center_y,
                'center_x': center_x,
                'profiles': self._get_star_profiles_detailed(
                    slice_data, center_y, center_x, num_angles
                ),
            })

        return stars

    def _get_star_profiles_detailed(self, slice_data: np.ndarray, center_y: int,
                                     center_x: int, num_angles: int) -> list:
        """
        Get detailed radial profiles from center point.

        Returns list of profile dictionaries with HU values, distances, and gradients.
        """
        from scipy.ndimage import gaussian_filter1d
        from skimage.draw import line

        profiles = []
        max_radius = int(max(slice_data.shape) * 1.5)

        for i in range(num_angles):
            angle = 2 * np.pi * i / num_angles

            # Calculate endpoint
            end_y = int(center_y + max_radius * np.sin(angle))
            end_x = int(center_x + max_radius * np.cos(angle))

            # Clip to image bounds
            end_y = max(0, min(slice_data.shape[0] - 1, end_y))
            end_x = max(0, min(slice_data.shape[1] - 1, end_x))

            # Get line coordinates
            rr, cc = line(center_y, center_x, end_y, end_x)

            # Get HU values and distances
            hu_values = slice_data[rr, cc]
            distances = np.sqrt((rr - center_y)**2 + (cc - center_x)**2)

            # Smooth the profile
            if len(hu_values) > 5:
                hu_values_smooth = gaussian_filter1d(hu_values.astype(float), sigma=2.0)
            else:
                hu_values_smooth = hu_values.astype(float)

            # Calculate gradient
            gradient = np.gradient(hu_values_smooth)

            profiles.append({
                'hu_values': hu_values_smooth,
                'distances': distances,
                'gradient': gradient,
                'angle': angle
            })

        return profiles

    def _analyze_profile_characteristics(self, profile: Dict, distance_mm: float,
                                         spacing: Tuple[float, float, float],
                                         pixel_hu: float = None,
                                         bone_hu_low: float = 150.0,
                                         bone_hu_high: float = 1500.0,
                                         w_hu: float = 0.45,
                                         w_width: float = 0.35,
                                         w_smooth: float = 0.25,
                                         w_gradient: float = 0.25) -> Dict:
        """
        Analyze profile characteristics at a given distance.

        Returns dictionary with characteristics and classification confidence.

        Args:
            profile: Profile dictionary with HU values, distances, gradients
            distance_mm: Distance from metal center in mm
            spacing: Voxel spacing (z, y, x) in mm
            pixel_hu: Actual HU value at the pixel being classified
            bone_hu_low: Lower bound of expected bone HU range
            bone_hu_high: Upper bound of expected bone HU range
        """
        # Find index closest to desired distance
        distances = profile['distances'] * np.mean(spacing[:2])  # Convert to mm
        idx = np.argmin(np.abs(distances - distance_mm))

        # Get local window around this point
        window_size = 5
        idx_min = max(0, idx - window_size)
        idx_max = min(len(profile['hu_values']), idx + window_size + 1)

        hu_window = profile['hu_values'][idx_min:idx_max]
        grad_window = profile['gradient'][idx_min:idx_max]

        # Calculate characteristics
        peak_width_mm = self._calculate_peak_width(hu_window, spacing)
        smoothness = self._calculate_smoothness_score(grad_window)
        gradient_magnitude = np.abs(np.mean(grad_window))

        # Bone scoring: broad peaks, smooth, gradual gradients, appropriate HU.
        # Each feature votes ±its weight; weights are tunable from the UI so the
        # relative influence of shape vs. HU can be explored. Score range is
        # ±(w_width + w_smooth + w_gradient + w_hu).
        bone_score = 0.0

        # Peak width criterion (bone: 3-5mm, artifact: <2mm)
        if peak_width_mm > 3.0:
            bone_score += w_width
        elif peak_width_mm < 2.0:
            bone_score -= w_width

        # Smoothness criterion (higher = more bone-like)
        if smoothness > 0.7:
            bone_score += w_smooth
        elif smoothness < 0.3:
            bone_score -= w_smooth

        # Gradient criterion (lower = more bone-like)
        if gradient_magnitude < 50:
            bone_score += w_gradient
        elif gradient_magnitude > 150:
            bone_score -= w_gradient

        # HU range criterion (within bone range = bone-like)
        # This gives significant influence to user-specified HU range
        hu_score = 0.0
        if pixel_hu is not None:
            # Calculate how well the HU fits within the bone range
            bone_hu_center = (bone_hu_low + bone_hu_high) / 2.0
            bone_hu_range = bone_hu_high - bone_hu_low

            if bone_hu_low <= pixel_hu <= bone_hu_high:
                # Inside bone range: score based on how centered it is
                # Typical cortical bone: 700-1500 HU, cancellous: 300-700 HU
                # Favor mid-range values as most bone-like
                distance_from_center = abs(pixel_hu - bone_hu_center)
                normalized_distance = distance_from_center / (bone_hu_range / 2.0) if bone_hu_range > 0 else 0
                hu_score = w_hu * (1.0 - 0.5 * normalized_distance)  # 0.5*w_hu to w_hu
            else:
                # Outside bone range: negative score proportional to how far outside
                if pixel_hu < bone_hu_low:
                    distance_outside = bone_hu_low - pixel_hu
                else:
                    distance_outside = pixel_hu - bone_hu_high
                # Penalize more heavily as we get further from the range
                penalty = min(1.0, distance_outside / 500.0)  # Max penalty at 500 HU outside
                hu_score = -w_hu * penalty

            bone_score += hu_score

        # Convert to confidence (0 to 1). Normalize by the total possible
        # magnitude so confidence stays in [0,1] as weights change.
        max_score = w_width + w_smooth + w_gradient + (w_hu if pixel_hu is not None else 0.0)
        if max_score > 0:
            confidence = (bone_score + max_score) / (2.0 * max_score)
        else:
            confidence = 0.5
        confidence = np.clip(confidence, 0.0, 1.0)

        return {
            'peak_width_mm': peak_width_mm,
            'smoothness': smoothness,
            'gradient_magnitude': gradient_magnitude,
            'pixel_hu': pixel_hu,
            'hu_score': hu_score if pixel_hu is not None else None,
            'bone_score': bone_score,
            'confidence': confidence
        }

    def _classify_from_profile(self, characteristics: Dict) -> bool:
        """
        Classify as bone (True) or artifact (False) based on profile characteristics.
        """
        return characteristics['bone_score'] > 0.0

    def _calculate_peak_width(self, hu_values: np.ndarray, spacing: Tuple[float, float, float]) -> float:
        """
        Calculate peak width in mm using Full Width at Half Maximum (FWHM).
        """
        if len(hu_values) < 3:
            return 0.0

        peak_val = np.max(hu_values)
        half_max = peak_val / 2.0

        # Find points above half maximum
        above_half = hu_values > half_max
        if not np.any(above_half):
            return 0.0

        # Width in voxels
        width_voxels = np.sum(above_half)

        # Convert to mm
        width_mm = width_voxels * np.mean(spacing[:2])

        return width_mm

    def _calculate_smoothness_score(self, gradient: np.ndarray) -> float:
        """
        Calculate smoothness score (0 to 1, higher = smoother).
        """
        if len(gradient) < 2:
            return 1.0

        # Variance of gradient (lower = smoother)
        grad_variance = np.var(gradient)

        # Convert to score (inverse relationship)
        smoothness = 1.0 / (1.0 + grad_variance / 100.0)

        return smoothness

    def _get_star_profiles(self, slice_data: np.ndarray, center_y: int, center_x: int,
                          num_angles: int) -> list:
        """Get radial profiles from center point (legacy method for compatibility)."""
        profiles = []
        max_radius = max(slice_data.shape)

        for i in range(num_angles):
            angle = 2 * np.pi * i / num_angles
            profile = []

            for r in range(max_radius):
                y = int(center_y + r * np.sin(angle))
                x = int(center_x + r * np.cos(angle))

                if 0 <= y < slice_data.shape[0] and 0 <= x < slice_data.shape[1]:
                    profile.append(slice_data[y, x])
                else:
                    break

            if profile:
                # Smooth the profile
                from scipy.ndimage import gaussian_filter1d
                profile = gaussian_filter1d(profile, sigma=2.0)
                profiles.append(profile)

        return profiles
    
    def _empty_result(self) -> Dict:
        """Return empty result when discrimination fails."""
        return {
            'bone_mask': np.array([]),
            'artifact_mask': np.array([]),
            'confidence_map': np.array([]),
            'method': self.method.value,
            'metadata': {}
        }


# Convenience functions for backward compatibility
