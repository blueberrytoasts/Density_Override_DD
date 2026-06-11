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
