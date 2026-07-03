"""Background metal detection for the PySide app.

Metal detection runs star-profile analysis across the whole volume and can take
several seconds, so -- exactly like ``DicomLoadWorker`` -- it runs on a worker
thread to keep the UI responsive. This file is pure glue: the actual algorithm
lives in ``app/core/metal_detection.py`` and is shared with the Streamlit app.
We do not reimplement it here; we only call it off the UI thread and report the
result back through signals.
"""
import numpy as np
from PySide6.QtCore import QObject, Signal

from pyside_app import bootstrap  # noqa: F401  (side effect: puts app/ on sys.path)
from core.metal_detection import MetalDetector, MetalDetectionMethod


class MetalDetectionWorker(QObject):
    """Runs 3D adaptive metal detection off the UI thread.

    Signals:
        finished(result): emitted with the detection result dict on success.
            The dict contains at least ``mask`` (3D binary np.ndarray),
            ``roi_bounds``, ``threshold``, and ``individual_regions``.
        failed(message): emitted with a human-readable error string.
    """

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, volume: np.ndarray, spacing, fw_percentage: float = 75.0):
        super().__init__()
        self._volume = volume
        # Some DICOM series report a negative z-spacing depending on slice
        # ordering; the detector expects positive magnitudes.
        self._spacing = np.abs(spacing)
        # Full-Width % of each star line's peak used as that line's threshold.
        # Lower => lower per-slice cutoff => larger mask.
        self._fw_percentage = fw_percentage

    def run(self) -> None:
        try:
            detector = MetalDetector(MetalDetectionMethod.ADAPTIVE_3D)
            result = detector.detect(
                self._volume,
                self._spacing,
                fw_percentage=self._fw_percentage,
                use_star_profiles=True,
            )
            if result.get("mask") is None or not np.any(result["mask"]):
                self.failed.emit("No metal implant detected.")
                return
            self.finished.emit(result)
        except Exception as exc:  # surface any detector error to the UI
            self.failed.emit(f"{type(exc).__name__}: {exc}")
