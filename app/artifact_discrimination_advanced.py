"""
Advanced Bone vs Artifact Discrimination using Texture and Gradient Analysis
Based on multi-faceted algorithmic strategy combining:
- Texture analysis (LBP, GLCM, local variance)
- Gradient analysis (LoG, gradient direction variance)
- Machine learning classification
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, sobel
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import laplace, gaussian
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


def extract_texture_features(ct_volume: np.ndarray, mask: np.ndarray, 
                            window_size: int = 5) -> Dict[str, np.ndarray]:
    """
    Extract texture features for each voxel in the mask.
    
    Args:
        ct_volume: 3D CT volume in HU
        mask: Binary mask of regions to analyze
        window_size: Size of local neighborhood for feature extraction
        
    Returns:
        Dictionary of feature arrays
    """
    features = {
        'local_variance': np.zeros_like(mask, dtype=np.float32),
        'local_std': np.zeros_like(mask, dtype=np.float32),
        'contrast': np.zeros_like(mask, dtype=np.float32),
        'homogeneity': np.zeros_like(mask, dtype=np.float32),
        'energy': np.zeros_like(mask, dtype=np.float32),
        'lbp_variance': np.zeros_like(mask, dtype=np.float32)
    }
    
    # Pad volume for window operations
    pad_width = window_size // 2
    padded_volume = np.pad(ct_volume, pad_width, mode='edge')
    
    # Get coordinates of mask voxels
    coords = np.where(mask)
    
    for i, (z, y, x) in enumerate(zip(*coords)):
        # Extract local neighborhood
        z_pad, y_pad, x_pad = z + pad_width, y + pad_width, x + pad_width
        local_patch = padded_volume[
            z_pad - pad_width:z_pad + pad_width + 1,
            y_pad - pad_width:y_pad + pad_width + 1,
            x_pad - pad_width:x_pad + pad_width + 1
        ]
        
        # Local variance and standard deviation
        features['local_variance'][z, y, x] = np.var(local_patch)
        features['local_std'][z, y, x] = np.std(local_patch)
        
        # GLCM features for center slice of patch
        center_slice = local_patch[pad_width, :, :]
        
        # Normalize to 8-bit for GLCM
        if center_slice.max() > center_slice.min():
            normalized = ((center_slice - center_slice.min()) / 
                         (center_slice.max() - center_slice.min()) * 255).astype(np.uint8)
            
            # Compute GLCM
            glcm = graycomatrix(normalized, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                               levels=256, symmetric=True, normed=True)
            
            # Extract GLCM properties
            features['contrast'][z, y, x] = graycoprops(glcm, 'contrast').mean()
            features['homogeneity'][z, y, x] = graycoprops(glcm, 'homogeneity').mean()
            features['energy'][z, y, x] = graycoprops(glcm, 'energy').mean()
            
            # LBP for texture pattern
            lbp = local_binary_pattern(normalized.astype(np.float64), P=8, R=1, method='uniform')
            features['lbp_variance'][z, y, x] = np.var(lbp)
    
    return features


def extract_gradient_features(ct_volume: np.ndarray, mask: np.ndarray,
                             sigma: float = 1.0) -> Dict[str, np.ndarray]:
    """
    Extract gradient-based features for discrimination.
    
    Args:
        ct_volume: 3D CT volume in HU
        mask: Binary mask of regions to analyze
        sigma: Gaussian smoothing parameter for LoG
        
    Returns:
        Dictionary of gradient feature arrays
    """
    features = {
        'log_magnitude': np.zeros_like(mask, dtype=np.float32),
        'gradient_magnitude': np.zeros_like(mask, dtype=np.float32),
        'gradient_direction_variance': np.zeros_like(mask, dtype=np.float32),
        'edge_density': np.zeros_like(mask, dtype=np.float32)
    }
    
    # Apply Gaussian smoothing for noise reduction
    smoothed = gaussian_filter(ct_volume.astype(np.float32), sigma=0.5)
    
    # Laplacian of Gaussian (LoG)
    log_response = ndimage.gaussian_laplace(smoothed, sigma=sigma)
    features['log_magnitude'][mask] = np.abs(log_response[mask])
    
    # Compute gradients using Sobel operators
    grad_z = sobel(smoothed, axis=0)
    grad_y = sobel(smoothed, axis=1)
    grad_x = sobel(smoothed, axis=2)
    
    # Gradient magnitude
    grad_mag = np.sqrt(grad_z**2 + grad_y**2 + grad_x**2)
    features['gradient_magnitude'][mask] = grad_mag[mask]
    
    # Gradient direction in each plane
    # Compute local gradient direction variance
    window_size = 5
    pad_width = window_size // 2
    
    # Pad arrays
    padded_grad_z = np.pad(grad_z, pad_width, mode='edge')
    padded_grad_y = np.pad(grad_y, pad_width, mode='edge')
    padded_grad_x = np.pad(grad_x, pad_width, mode='edge')
    
    coords = np.where(mask)
    for z, y, x in zip(*coords):
        z_pad, y_pad, x_pad = z + pad_width, y + pad_width, x + pad_width
        
        # Extract local gradient patches
        local_gz = padded_grad_z[
            z_pad - pad_width:z_pad + pad_width + 1,
            y_pad - pad_width:y_pad + pad_width + 1,
            x_pad - pad_width:x_pad + pad_width + 1
        ]
        local_gy = padded_grad_y[
            z_pad - pad_width:z_pad + pad_width + 1,
            y_pad - pad_width:y_pad + pad_width + 1,
            x_pad - pad_width:x_pad + pad_width + 1
        ]
        local_gx = padded_grad_x[
            z_pad - pad_width:z_pad + pad_width + 1,
            y_pad - pad_width:y_pad + pad_width + 1,
            x_pad - pad_width:x_pad + pad_width + 1
        ]
        
        # Compute gradient directions (angles)
        with np.errstate(divide='ignore', invalid='ignore'):
            # XY plane angles
            angles_xy = np.arctan2(local_gy.flatten(), local_gx.flatten())
            angles_xy = angles_xy[~np.isnan(angles_xy)]
            
            if len(angles_xy) > 1:
                # Circular variance for gradient directions
                features['gradient_direction_variance'][z, y, x] = np.var(angles_xy)
        
        # Edge density: proportion of high gradient magnitude voxels in neighborhood
        local_mag = grad_mag[
            max(0, z-pad_width):min(grad_mag.shape[0], z+pad_width+1),
            max(0, y-pad_width):min(grad_mag.shape[1], y+pad_width+1),
            max(0, x-pad_width):min(grad_mag.shape[2], x+pad_width+1)
        ]
        threshold = np.percentile(grad_mag[mask], 75)
        features['edge_density'][z, y, x] = np.mean(local_mag > threshold)
    
    return features


def compute_structure_tensor_features(ct_volume: np.ndarray, mask: np.ndarray,
                                     sigma: float = 1.0) -> Dict[str, np.ndarray]:
    """
    Compute structure tensor features for analyzing local structure coherence.
    
    Args:
        ct_volume: 3D CT volume
        mask: Binary mask
        sigma: Gaussian smoothing parameter
        
    Returns:
        Dictionary of structure tensor features
    """
    from scipy.ndimage import gaussian_filter
    
    features = {
        'coherence': np.zeros_like(mask, dtype=np.float32),
        'anisotropy': np.zeros_like(mask, dtype=np.float32)
    }
    
    # Compute gradients
    grad_z = sobel(ct_volume, axis=0)
    grad_y = sobel(ct_volume, axis=1)
    grad_x = sobel(ct_volume, axis=2)
    
    # Structure tensor components
    Szz = gaussian_filter(grad_z * grad_z, sigma)
    Syy = gaussian_filter(grad_y * grad_y, sigma)
    Sxx = gaussian_filter(grad_x * grad_x, sigma)
    Szy = gaussian_filter(grad_z * grad_y, sigma)
    Szx = gaussian_filter(grad_z * grad_x, sigma)
    Syx = gaussian_filter(grad_y * grad_x, sigma)
    
    coords = np.where(mask)
    for z, y, x in zip(*coords):
        # Build structure tensor for this voxel
        S = np.array([
            [Szz[z,y,x], Szy[z,y,x], Szx[z,y,x]],
            [Szy[z,y,x], Syy[z,y,x], Syx[z,y,x]],
            [Szx[z,y,x], Syx[z,y,x], Sxx[z,y,x]]
        ])
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(S)
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
        
        # Coherence: how aligned the local structure is
        if eigenvalues[0] > 0:
            features['coherence'][z, y, x] = (eigenvalues[0] - eigenvalues[2]) / eigenvalues[0]
            
        # Anisotropy: measure of directional dependence
        if np.sum(eigenvalues) > 0:
            features['anisotropy'][z, y, x] = np.std(eigenvalues) / np.mean(eigenvalues)
    
    return features


def classify_bone_vs_artifact(ct_volume: np.ndarray, bright_mask: np.ndarray,
                             metal_mask: np.ndarray = None,
                             use_ml: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Advanced classification of bright regions into bone vs artifact.
    
    Args:
        ct_volume: 3D CT volume in HU
        bright_mask: Binary mask of bright regions to classify
        metal_mask: Optional metal mask for distance weighting
        use_ml: Whether to use machine learning (requires training data)
        
    Returns:
        Tuple of (bone_mask, artifact_mask, confidence_map)
    """
    if not np.any(bright_mask):
        return np.zeros_like(bright_mask), np.zeros_like(bright_mask), np.zeros_like(bright_mask, dtype=np.float32)
    
    print("Extracting texture features...")
    texture_features = extract_texture_features(ct_volume, bright_mask)
    
    print("Extracting gradient features...")
    gradient_features = extract_gradient_features(ct_volume, bright_mask)
    
    print("Computing structure tensor features...")
    structure_features = compute_structure_tensor_features(ct_volume, bright_mask)
    
    # Combine all features
    all_features = {**texture_features, **gradient_features, **structure_features}
    
    # Normalize features to [0, 1] range for scoring
    normalized_features = {}
    for name, feature_map in all_features.items():
        mask_values = feature_map[bright_mask]
        if mask_values.max() > mask_values.min():
            normalized = np.zeros_like(feature_map)
            normalized[bright_mask] = (mask_values - mask_values.min()) / (mask_values.max() - mask_values.min())
            normalized_features[name] = normalized
        else:
            normalized_features[name] = feature_map
    
    # Heuristic scoring system for bone vs artifact
    # Higher scores indicate artifact, lower scores indicate bone
    artifact_score = np.zeros_like(bright_mask, dtype=np.float32)
    
    # Artifacts have:
    # - High local variance (noisy)
    # - High contrast (GLCM)
    # - Low homogeneity (GLCM)
    # - Low energy (GLCM)
    # - High LoG response (edges/noise)
    # - High gradient direction variance (chaotic)
    # - Low coherence (less structured)
    
    # Weight factors for each feature
    weights = {
        'local_variance': 0.15,        # Higher = more artifact-like
        'contrast': 0.15,               # Higher = more artifact-like
        'homogeneity': -0.15,           # Lower = more artifact-like (negative weight)
        'energy': -0.10,                # Lower = more artifact-like (negative weight)
        'log_magnitude': 0.20,          # Higher = more artifact-like
        'gradient_direction_variance': 0.15,  # Higher = more artifact-like
        'coherence': -0.10,             # Lower = more artifact-like (negative weight)
        'edge_density': 0.10,           # Higher = more artifact-like
    }
    
    # Compute weighted artifact score
    for feature_name, weight in weights.items():
        if feature_name in normalized_features:
            artifact_score += weight * normalized_features[feature_name]
    
    # Add distance weighting if metal mask is provided
    if metal_mask is not None:
        from scipy.ndimage import distance_transform_edt
        metal_distance = distance_transform_edt(~metal_mask)
        # Normalize distance to [0, 1], closer to metal = higher artifact probability
        max_dist = 50  # mm, adjust based on your data
        distance_weight = np.clip(1 - metal_distance / max_dist, 0, 1)
        artifact_score += 0.2 * distance_weight
    
    # Normalize final scores to [0, 1]
    artifact_score_masked = artifact_score[bright_mask]
    if artifact_score_masked.max() > artifact_score_masked.min():
        artifact_score[bright_mask] = (artifact_score_masked - artifact_score_masked.min()) / \
                                     (artifact_score_masked.max() - artifact_score_masked.min())
    
    # Threshold to create binary masks
    # Use Otsu's method or fixed threshold
    from skimage.filters import threshold_otsu
    threshold = 0.5  # Default threshold
    
    try:
        # Try Otsu's method for automatic thresholding
        artifact_values = artifact_score[bright_mask]
        if len(np.unique(artifact_values)) > 1:
            threshold = threshold_otsu(artifact_values)
    except:
        pass
    
    # Create binary masks
    artifact_mask = np.zeros_like(bright_mask)
    bone_mask = np.zeros_like(bright_mask)
    
    artifact_mask[bright_mask] = artifact_score[bright_mask] > threshold
    bone_mask[bright_mask] = artifact_score[bright_mask] <= threshold
    
    # Post-processing: morphological operations
    from scipy.ndimage import binary_opening, binary_closing
    
    # Clean up small isolated regions
    artifact_mask = binary_opening(artifact_mask, iterations=1)
    artifact_mask = binary_closing(artifact_mask, iterations=1)
    
    bone_mask = binary_opening(bone_mask, iterations=1)
    bone_mask = binary_closing(bone_mask, iterations=1)
    
    # Create confidence map (distance from threshold)
    confidence_map = np.zeros_like(bright_mask, dtype=np.float32)
    confidence_map[bright_mask] = np.abs(artifact_score[bright_mask] - threshold) * 2
    confidence_map = np.clip(confidence_map, 0, 1)
    
    print(f"Classification complete. Threshold: {threshold:.3f}")
    print(f"Bone voxels: {np.sum(bone_mask):,}, Artifact voxels: {np.sum(artifact_mask):,}")
    
    return bone_mask.astype(bool), artifact_mask.astype(bool), confidence_map


def analyze_discrimination_quality(bone_mask: np.ndarray, artifact_mask: np.ndarray,
                                  confidence_map: np.ndarray) -> Dict:
    """
    Analyze the quality of bone/artifact discrimination.
    
    Returns:
        Dictionary with quality metrics
    """
    metrics = {
        'bone_voxels': int(np.sum(bone_mask)),
        'artifact_voxels': int(np.sum(artifact_mask)),
        'mean_confidence': float(np.mean(confidence_map[confidence_map > 0])) if np.any(confidence_map > 0) else 0,
        'high_confidence_ratio': float(np.sum(confidence_map > 0.7) / np.sum(confidence_map > 0)) if np.any(confidence_map > 0) else 0,
        'low_confidence_ratio': float(np.sum((confidence_map > 0) & (confidence_map < 0.3)) / np.sum(confidence_map > 0)) if np.any(confidence_map > 0) else 0,
    }
    
    # Analyze spatial distribution
    if np.any(bone_mask):
        from scipy.ndimage import label
        bone_labels, num_bone_regions = label(bone_mask)
        metrics['num_bone_regions'] = int(num_bone_regions)
        
        # Get size of largest bone region
        if num_bone_regions > 0:
            region_sizes = [np.sum(bone_labels == i) for i in range(1, num_bone_regions + 1)]
            metrics['largest_bone_region'] = int(max(region_sizes))
    else:
        metrics['num_bone_regions'] = 0
        metrics['largest_bone_region'] = 0
    
    if np.any(artifact_mask):
        artifact_labels, num_artifact_regions = label(artifact_mask)
        metrics['num_artifact_regions'] = int(num_artifact_regions)
        
        if num_artifact_regions > 0:
            region_sizes = [np.sum(artifact_labels == i) for i in range(1, num_artifact_regions + 1)]
            metrics['largest_artifact_region'] = int(max(region_sizes))
    else:
        metrics['num_artifact_regions'] = 0
        metrics['largest_artifact_region'] = 0
    
    return metrics