"""
Unified Artifact Discrimination Module
Consolidates all bone/artifact discrimination algorithms into a single interface.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from enum import Enum
from scipy.ndimage import distance_transform_edt, gaussian_filter, sobel
from scipy.ndimage import binary_opening, binary_closing, label


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
                              max_distance_cm: float = 10.0) -> Dict:
        """
        Fast distance-based discrimination.
        
        Principle: Bright artifacts are typically closer to metal than bone.
        
        Algorithm:
        1. Calculate distance from metal
        2. Apply distance-based classification
        3. Use local smoothness as secondary criterion
        """
        # Calculate distance from metal
        inverted_metal = np.logical_not(metal_mask)
        distances = distance_transform_edt(inverted_metal, sampling=spacing)
        distances_cm = distances / 10.0
        
        # Smooth CT for texture analysis
        smoothed = gaussian_filter(ct_volume.astype(float), sigma=2.0)
        
        # Calculate local variance
        local_variance = np.zeros_like(ct_volume, dtype=float)
        for z in range(1, ct_volume.shape[0]-1):
            slice_std = np.std([ct_volume[z-1], ct_volume[z], ct_volume[z+1]], axis=0)
            local_variance[z] = slice_std
        
        # Classification based on distance and smoothness
        bone_mask = np.zeros_like(bright_mask)
        artifact_mask = np.zeros_like(bright_mask)
        confidence_map = np.zeros_like(ct_volume, dtype=float)
        
        # More relaxed bone criteria - bone is typically:
        # - At moderate distance from metal (not too close, not too far)
        # - Has moderate HU values (400-1500)
        # - Has relatively low variance (smooth structure)
        bone_criteria = bright_mask & \
                       (distances_cm > 1.0) & \
                       (ct_volume >= 400) & (ct_volume <= 1500) & \
                       (local_variance < 300)  # Increased variance threshold
        
        # Artifact characteristics: very close to metal OR high variance OR very high HU
        artifact_criteria = bright_mask & \
                          ((distances_cm < 1.0) |  # Very close to metal
                           (local_variance > 400) |  # High variance
                           (ct_volume > 1500))  # Very high HU (likely artifact)
        
        bone_mask = bone_criteria
        artifact_mask = artifact_criteria & (~bone_mask)
        
        # Handle unclassified regions with better heuristics
        unclassified = bright_mask & (~bone_mask) & (~artifact_mask)
        if np.any(unclassified):
            # Use HU value as primary criterion for unclassified
            # Bone is typically 400-1000 HU, artifacts can be higher
            bone_hu_range = unclassified & (ct_volume >= 400) & (ct_volume <= 1000)
            artifact_hu_range = unclassified & (ct_volume > 1000)
            
            # Secondary criterion: distance
            # Very close = artifact, moderate distance = bone
            very_close = unclassified & (distances_cm < 0.5)
            moderate_dist = unclassified & (distances_cm >= 0.5) & (distances_cm < 3.0)
            
            # Combine criteria
            bone_mask |= (bone_hu_range & moderate_dist)
            artifact_mask |= (artifact_hu_range | very_close)
            
            # Remaining unclassified: use distance threshold
            still_unclassified = unclassified & (~bone_mask) & (~artifact_mask)
            near_metal = still_unclassified & (distances_cm < 2.0)
            far_from_metal = still_unclassified & (distances_cm >= 2.0)
            artifact_mask |= near_metal
            bone_mask |= far_from_metal
        
        # Calculate confidence based on distance and variance
        confidence_map[bright_mask] = np.clip(
            1.0 - (local_variance[bright_mask] / 500.0), 0, 1
        )
        
        return {
            'bone_mask': bone_mask,
            'artifact_mask': artifact_mask,
            'confidence_map': confidence_map,
            'distance_map': distances_cm,
            'method': 'distance_based',
            'metadata': {
                'max_distance_cm': max_distance_cm,
                'bone_voxels': np.sum(bone_mask),
                'artifact_voxels': np.sum(artifact_mask)
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
                                   num_angles: int = 16) -> Dict:
        """
        Star profile-based discrimination (RECOVERED ALGORITHM).

        Principle: Analyzes radial HU profiles to distinguish bone from artifacts.

        Bone characteristics:
        - Broad peaks (3-5mm width)
        - Smooth transitions (low gradient variance)
        - Consistent across angles (low directional variance)
        - Gradual gradients

        Artifact characteristics:
        - Narrow peaks (<2mm width)
        - Sharp edges (high gradient variance)
        - Variable across angles (high directional variance)
        - Steep gradients

        Args:
            num_angles: Number of radial profiles to analyze (default: 16)
        """
        from scipy.ndimage import gaussian_filter1d
        from skimage.draw import line

        bone_mask = np.zeros_like(bright_mask)
        artifact_mask = np.zeros_like(bright_mask)
        confidence_map = np.zeros_like(ct_volume, dtype=float)

        # Process each slice
        for z in range(ct_volume.shape[0]):
            if not np.any(bright_mask[z]) or not np.any(metal_mask[z]):
                continue

            # Find metal center on this slice
            metal_coords = np.where(metal_mask[z])
            if len(metal_coords[0]) == 0:
                continue

            center_y = int(np.mean(metal_coords[0]))
            center_x = int(np.mean(metal_coords[1]))

            # Get star profiles for this slice
            profiles = self._get_star_profiles_detailed(
                ct_volume[z], center_y, center_x, num_angles
            )

            # Analyze each bright pixel on this slice
            bright_coords = np.where(bright_mask[z])

            for i in range(len(bright_coords[0])):
                pixel_y = bright_coords[0][i]
                pixel_x = bright_coords[1][i]

                # Calculate angle to this pixel from metal center
                dy = pixel_y - center_y
                dx = pixel_x - center_x
                angle = np.arctan2(dy, dx)
                if angle < 0:
                    angle += 2 * np.pi

                # Find nearest profile
                profile_idx = int((angle / (2 * np.pi)) * num_angles) % num_angles
                if profile_idx >= len(profiles):
                    continue

                profile = profiles[profile_idx]

                # Calculate distance from metal center
                distance_voxels = np.sqrt(dy**2 + dx**2)
                distance_mm = distance_voxels * np.mean(spacing[:2])

                # Analyze profile characteristics at this distance
                characteristics = self._analyze_profile_characteristics(
                    profile, distance_mm, spacing
                )

                # Classify based on characteristics
                is_bone = self._classify_from_profile(characteristics)

                if is_bone:
                    bone_mask[z, pixel_y, pixel_x] = True
                    confidence_map[z, pixel_y, pixel_x] = characteristics['confidence']
                else:
                    artifact_mask[z, pixel_y, pixel_x] = True
                    confidence_map[z, pixel_y, pixel_x] = characteristics['confidence']

        return {
            'bone_mask': bone_mask,
            'artifact_mask': artifact_mask,
            'confidence_map': confidence_map,
            'method': 'star_profile',
            'metadata': {
                'num_angles': num_angles,
                'bone_voxels': np.sum(bone_mask),
                'artifact_voxels': np.sum(artifact_mask)
            }
        }

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
                                         spacing: Tuple[float, float, float]) -> Dict:
        """
        Analyze profile characteristics at a given distance.

        Returns dictionary with characteristics and classification confidence.
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

        # Bone scoring: broad peaks, smooth, gradual gradients
        bone_score = 0.0

        # Peak width criterion (bone: 3-5mm, artifact: <2mm)
        if peak_width_mm > 3.0:
            bone_score += 0.4
        elif peak_width_mm < 2.0:
            bone_score -= 0.4

        # Smoothness criterion (higher = more bone-like)
        if smoothness > 0.7:
            bone_score += 0.3
        elif smoothness < 0.3:
            bone_score -= 0.3

        # Gradient criterion (lower = more bone-like)
        if gradient_magnitude < 50:
            bone_score += 0.3
        elif gradient_magnitude > 150:
            bone_score -= 0.3

        # Convert to confidence (0 to 1)
        confidence = (bone_score + 1.0) / 2.0  # Normalize to [0, 1]

        return {
            'peak_width_mm': peak_width_mm,
            'smoothness': smoothness,
            'gradient_magnitude': gradient_magnitude,
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
