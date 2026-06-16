"""Fast conversion of a Hounsfield-Unit slice to a displayable QImage.

This is the hot path during slice scrubbing and window/level dragging, so it is
pure NumPy + a single QImage wrap (no matplotlib, no PIL). A typical 512x512
slice converts in well under a millisecond.
"""
import numpy as np
from PySide6.QtGui import QImage


def hu_to_qimage(slice_hu: np.ndarray, center: float, width: float) -> QImage:
    """Map a 2D HU array to an 8-bit grayscale QImage using window/level.

    Args:
        slice_hu: 2D array of Hounsfield Units (any integer/float dtype).
        center: Window center (level) in HU.
        width: Window width in HU. Clamped to a minimum of 1.

    Returns:
        A ``QImage`` (Format_Grayscale8) that owns its pixel buffer.
    """
    width = max(float(width), 1.0)
    low = center - width / 2.0
    high = center + width / 2.0

    # Normalize to [0, 1] within the window, then to 8-bit.
    normalized = (slice_hu.astype(np.float32) - low) / (high - low)
    np.clip(normalized, 0.0, 1.0, out=normalized)
    buf = np.ascontiguousarray((normalized * 255.0).astype(np.uint8))

    h, w = buf.shape
    image = QImage(buf.data, w, h, w, QImage.Format.Format_Grayscale8)
    # .copy() detaches the QImage from the temporary NumPy buffer so it stays
    # valid after this function returns.
    return image.copy()


def hu_to_qimage_with_overlay(
    slice_hu: np.ndarray,
    center: float,
    width: float,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.7,
) -> QImage:
    """Like ``hu_to_qimage`` but tints ``mask`` pixels with a translucent color.

    The CT slice is windowed to grayscale exactly as in ``hu_to_qimage``, then
    promoted to RGB so masked pixels can be alpha-blended toward ``color``.
    Used to show what metal detection found (red, per the project color code).

    Args:
        slice_hu: 2D HU array.
        center, width: window/level, same meaning as ``hu_to_qimage``.
        mask: 2D boolean array, same shape as ``slice_hu``; True = highlight.
        color: RGB tint for masked pixels (default red = metal).
        alpha: blend weight, 0 (invisible) .. 1 (opaque tint).

    Returns:
        A ``QImage`` (Format_RGB888) that owns its pixel buffer.
    """
    width = max(float(width), 1.0)
    low = center - width / 2.0
    high = center + width / 2.0

    normalized = (slice_hu.astype(np.float32) - low) / (high - low)
    np.clip(normalized, 0.0, 1.0, out=normalized)
    gray = normalized * 255.0  # float, (h, w)

    # Promote grayscale to RGB by repeating the channel, then blend the mask.
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    m = mask.astype(bool)
    tint = np.asarray(color, dtype=np.float32)
    rgb[m] = (1.0 - alpha) * rgb[m] + alpha * tint

    buf = np.ascontiguousarray(rgb.astype(np.uint8))
    h, w, _ = buf.shape
    # Bytes-per-line = 3 * width for 8-bit RGB.
    image = QImage(buf.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return image.copy()


def auto_window_level(volume: np.ndarray) -> tuple[float, float]:
    """Pick a reasonable initial window/level for a CT volume.

    Percentiles are computed over *body* voxels only (HU > -500), so the air
    surrounding the patient doesn't drag the window toward black, and the huge
    HU of metal doesn't wash out the soft-tissue/bone range. The user can
    fine-tune interactively (right-drag) afterward.

    Returns:
        ``(center, width)`` in HU.
    """
    body = volume[volume > -500]
    if body.size == 0:
        body = volume
    lo, hi = np.percentile(body, [2.0, 98.0])
    if hi <= lo:
        lo, hi = float(body.min()), float(body.max())
    center = (lo + hi) / 2.0
    width = max(hi - lo, 1.0)
    return float(center), float(width)
