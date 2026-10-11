"""Overlap and surface-distance metrics for comparing two 3D masks.

Used to compare app contours against reference (e.g. physician-approved)
contours on the same CT grid. All distances are in mm and respect
anisotropic voxel spacing.
"""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


def _surface(mask: np.ndarray) -> np.ndarray:
    """Boundary voxels of a 3D mask (6-connected erosion residue)."""
    if not mask.any():
        return mask
    eroded = binary_erosion(mask, structure=generate_binary_structure(3, 1),
                            border_value=0)
    return mask & ~eroded


def surface_distances(test: np.ndarray, ref: np.ndarray, spacing_zyx) -> tuple:
    """Distances (mm) from each surface voxel of one mask to the other's surface.

    Returns:
        (test->ref distances, ref->test distances) as 1D arrays.
    """
    s_test, s_ref = _surface(test), _surface(ref)
    dt_ref = distance_transform_edt(~s_ref, sampling=spacing_zyx)
    dt_test = distance_transform_edt(~s_test, sampling=spacing_zyx)
    return dt_ref[s_test], dt_test[s_ref]


def compare_masks(test: np.ndarray, ref: np.ndarray, spacing_zyx) -> dict:
    """Compare a test mask against a reference mask.

    Args:
        test, ref: 3D bool arrays of the same shape (volume order).
        spacing_zyx: (slice, row, col) voxel spacing in mm.

    Returns:
        dict with dice, jaccard, volumes (cc), volume difference, surface
        distance stats (mm) and sensitivity / precision. Distance metrics are
        NaN when either mask is empty.
    """
    test = np.asarray(test, dtype=bool)
    ref = np.asarray(ref, dtype=bool)
    if test.shape != ref.shape:
        raise ValueError(f"Shape mismatch: {test.shape} vs {ref.shape}")

    voxel_cc = float(np.prod(spacing_zyx)) / 1000.0
    n_test, n_ref = int(test.sum()), int(ref.sum())
    n_both = int((test & ref).sum())
    n_union = n_test + n_ref - n_both

    nan = float('nan')
    out = {
        'test_cc': n_test * voxel_cc,
        'ref_cc': n_ref * voxel_cc,
        'volume_diff_cc': (n_test - n_ref) * voxel_cc,
        'volume_diff_pct': (100.0 * (n_test - n_ref) / n_ref) if n_ref else nan,
        'dice': (2.0 * n_both / (n_test + n_ref)) if (n_test + n_ref) else nan,
        'jaccard': (n_both / n_union) if n_union else nan,
        'sensitivity': (n_both / n_ref) if n_ref else nan,   # ref covered by test
        'precision': (n_both / n_test) if n_test else nan,   # test inside ref
        'hd95_mm': nan, 'hd_max_mm': nan, 'msd_mm': nan,
    }
    if n_test and n_ref:
        d_tr, d_rt = surface_distances(test, ref, spacing_zyx)
        both = np.concatenate([d_tr, d_rt])
        # HD95 as the max of the two directed 95th percentiles (the common
        # definition in RT contour-comparison literature).
        out['hd95_mm'] = float(max(np.percentile(d_tr, 95), np.percentile(d_rt, 95)))
        out['hd_max_mm'] = float(both.max())
        out['msd_mm'] = float(both.mean())
    return out
