"""
GPU Utilities Module
Provides GPU-accelerated operations using CuPy with automatic fallback to CPU.
"""

import numpy as np
from typing import Tuple, Optional

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cp_ndimage
    GPU_AVAILABLE = True
    print("GPU acceleration available (CuPy detected)")
except ImportError:
    cp = None
    cp_ndimage = None
    GPU_AVAILABLE = False
    print("GPU acceleration not available (CuPy not installed)")
except Exception as e:
    # Catch CUDA configuration errors (FileNotFoundError, OSError, etc.)
    cp = None
    cp_ndimage = None
    GPU_AVAILABLE = False
    print(f"GPU acceleration not available (CUDA error: {e})")


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return GPU_AVAILABLE


def to_gpu(array: np.ndarray) -> 'cp.ndarray':
    """Transfer numpy array to GPU memory."""
    if GPU_AVAILABLE:
        return cp.asarray(array)
    return array


def to_cpu(array) -> np.ndarray:
    """Transfer array from GPU to CPU memory."""
    if GPU_AVAILABLE and hasattr(array, 'get'):
        return array.get()
    return np.asarray(array)


def get_array_module(array):
    """Get the appropriate array module (cupy or numpy) for the given array."""
    if GPU_AVAILABLE and hasattr(array, 'device'):
        return cp
    return np


def distance_transform_edt_gpu(binary_mask: np.ndarray,
                                spacing: Tuple[float, float, float],
                                use_gpu: bool = True) -> np.ndarray:
    """
    GPU-accelerated Euclidean distance transform.

    Args:
        binary_mask: 3D binary mask (True = background for distance calculation)
        spacing: Voxel spacing (z, y, x) in mm
        use_gpu: Whether to use GPU if available

    Returns:
        Distance transform array (on CPU)
    """
    if use_gpu and GPU_AVAILABLE:
        # CuPy's distance_transform_edt
        mask_gpu = cp.asarray(binary_mask)
        # Note: cupyx.scipy.ndimage.distance_transform_edt supports sampling
        distances_gpu = cp_ndimage.distance_transform_edt(mask_gpu, sampling=spacing)
        return distances_gpu.get()
    else:
        # CPU fallback
        from scipy.ndimage import distance_transform_edt
        return distance_transform_edt(binary_mask, sampling=spacing)


def gaussian_filter_gpu(volume: np.ndarray,
                        sigma: float,
                        use_gpu: bool = True) -> np.ndarray:
    """
    GPU-accelerated Gaussian filter.

    Args:
        volume: Input volume
        sigma: Standard deviation for Gaussian kernel
        use_gpu: Whether to use GPU if available

    Returns:
        Filtered volume (on CPU)
    """
    if use_gpu and GPU_AVAILABLE:
        volume_gpu = cp.asarray(volume.astype(np.float32))
        filtered_gpu = cp_ndimage.gaussian_filter(volume_gpu, sigma=sigma)
        return filtered_gpu.get()
    else:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(volume.astype(float), sigma=sigma)


def calculate_local_variance_gpu(ct_volume: np.ndarray,
                                  use_gpu: bool = True) -> np.ndarray:
    """
    GPU-accelerated local variance calculation across slices.

    Args:
        ct_volume: 3D CT volume
        use_gpu: Whether to use GPU if available

    Returns:
        Local variance volume (on CPU)
    """
    if use_gpu and GPU_AVAILABLE:
        volume_gpu = cp.asarray(ct_volume.astype(cp.float32))
        local_variance = cp.zeros_like(volume_gpu, dtype=cp.float32)

        # Vectorized variance calculation across slices
        for z in range(1, ct_volume.shape[0] - 1):
            # Stack 3 adjacent slices and compute std
            stack = cp.stack([volume_gpu[z-1], volume_gpu[z], volume_gpu[z+1]], axis=0)
            local_variance[z] = cp.std(stack, axis=0)

        return local_variance.get()
    else:
        # CPU version
        local_variance = np.zeros_like(ct_volume, dtype=float)
        for z in range(1, ct_volume.shape[0] - 1):
            slice_std = np.std([ct_volume[z-1], ct_volume[z], ct_volume[z+1]], axis=0)
            local_variance[z] = slice_std
        return local_variance


def batch_neighborhood_analysis_gpu(ct_volume: np.ndarray,
                                     coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                     bone_hu_low: float,
                                     bone_hu_high: float,
                                     tissue_hu_low: float,
                                     tissue_hu_high: float,
                                     window_size: int = 5,
                                     use_gpu: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated batch neighborhood analysis for artifact classification.

    Analyzes neighborhoods around multiple voxels simultaneously on GPU.

    Args:
        ct_volume: 3D CT volume
        coords: Tuple of (z, y, x) coordinate arrays for voxels to analyze
        bone_hu_low/high: Bone HU range
        tissue_hu_low/high: Tissue HU range
        window_size: Neighborhood window size
        use_gpu: Whether to use GPU if available

    Returns:
        Tuple of (bone_ratios, tissue_ratios) arrays
    """
    n_voxels = len(coords[0])

    if use_gpu and GPU_AVAILABLE and n_voxels > 1000:
        # GPU version for large numbers of voxels
        volume_gpu = cp.asarray(ct_volume.astype(cp.float32))
        z_coords = cp.asarray(coords[0])
        y_coords = cp.asarray(coords[1])
        x_coords = cp.asarray(coords[2])

        bone_ratios = cp.zeros(n_voxels, dtype=cp.float32)
        tissue_ratios = cp.zeros(n_voxels, dtype=cp.float32)

        half_w = window_size // 2
        shape = volume_gpu.shape

        # Process in batches on GPU
        batch_size = 10000
        for start in range(0, n_voxels, batch_size):
            end = min(start + batch_size, n_voxels)

            for i in range(start, end):
                z, y, x = int(z_coords[i]), int(y_coords[i]), int(x_coords[i])

                z_min = max(0, z - 1)
                z_max = min(shape[0], z + 2)
                y_min = max(0, y - half_w)
                y_max = min(shape[1], y + half_w + 1)
                x_min = max(0, x - half_w)
                x_max = min(shape[2], x + half_w + 1)

                neighborhood = volume_gpu[z_min:z_max, y_min:y_max, x_min:x_max]
                total = neighborhood.size

                if total > 0:
                    bone_count = cp.sum((neighborhood >= bone_hu_low) & (neighborhood <= bone_hu_high))
                    tissue_count = cp.sum((neighborhood >= tissue_hu_low) & (neighborhood <= tissue_hu_high))
                    bone_ratios[i] = bone_count / total
                    tissue_ratios[i] = tissue_count / total

        return bone_ratios.get(), tissue_ratios.get()
    else:
        # CPU version
        bone_ratios = np.zeros(n_voxels, dtype=np.float32)
        tissue_ratios = np.zeros(n_voxels, dtype=np.float32)

        half_w = window_size // 2
        shape = ct_volume.shape

        for i in range(n_voxels):
            z, y, x = coords[0][i], coords[1][i], coords[2][i]

            z_min = max(0, z - 1)
            z_max = min(shape[0], z + 2)
            y_min = max(0, y - half_w)
            y_max = min(shape[1], y + half_w + 1)
            x_min = max(0, x - half_w)
            x_max = min(shape[2], x + half_w + 1)

            neighborhood = ct_volume[z_min:z_max, y_min:y_max, x_min:x_max]
            total = neighborhood.size

            if total > 0:
                bone_count = np.sum((neighborhood >= bone_hu_low) & (neighborhood <= bone_hu_high))
                tissue_count = np.sum((neighborhood >= tissue_hu_low) & (neighborhood <= tissue_hu_high))
                bone_ratios[i] = bone_count / total
                tissue_ratios[i] = tissue_count / total

        return bone_ratios, tissue_ratios


def vectorized_threshold_gpu(volume: np.ndarray,
                              low: float,
                              high: float,
                              use_gpu: bool = True) -> np.ndarray:
    """
    GPU-accelerated threshold operation.

    Args:
        volume: Input volume
        low: Lower threshold
        high: Upper threshold
        use_gpu: Whether to use GPU if available

    Returns:
        Binary mask (on CPU)
    """
    if use_gpu and GPU_AVAILABLE:
        volume_gpu = cp.asarray(volume)
        mask_gpu = (volume_gpu >= low) & (volume_gpu <= high)
        return mask_gpu.get()
    else:
        return (volume >= low) & (volume <= high)
