"""Background artifact segmentation for the PySide app.

Two workers are provided:
    SegmentationWorker      — star profile discrimination (slow, smarter)
    LegacySegmentationWorker — pure boolean HU thresholds (fast, no ML)

Result dict keys (both workers):
    dark_artifacts   : 3D bool mask
    bright_artifacts : 3D bool mask
    bone             : 3D bool mask
"""
import numpy as np
from PySide6.QtCore import QObject, Signal

from pyside_app import bootstrap  # noqa: F401  (puts app/ on sys.path)
from body_mask import create_body_mask
from core.discrimination import ArtifactDiscriminator, DiscriminationMethod

# Default HU thresholds matching ThresholdConfig in app/config.py
_DARK_LOW = -1024
_DARK_HIGH = -150
_BRIGHT_LOW = 200
_BRIGHT_HIGH = 2500
_BONE_LOW = 400
_BONE_HIGH = 1800


class LegacySegmentationWorker(QObject):
    """Pure boolean HU threshold segmentation — no discriminator.

    dark_artifacts  : HU in [_DARK_LOW, _DARK_HIGH], excluding metal
    bone            : HU in [_BONE_LOW, _BONE_HIGH], excluding metal + dark
    bright_artifacts: HU in [_BRIGHT_LOW, _BRIGHT_HIGH], excluding metal + dark + bone

    Signals:
        finished(result): dict with keys dark_artifacts, bright_artifacts, bone.
        failed(message): human-readable error string.
    """

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, volume: np.ndarray, metal_mask: np.ndarray,
                 roi_mask: np.ndarray | None = None):
        super().__init__()
        self._volume = volume
        self._metal_mask = metal_mask
        self._roi_mask = roi_mask

    def run(self) -> None:
        try:
            body_mask = create_body_mask(self._volume, air_threshold=-400)
            constraint = body_mask & ~self._metal_mask
            if self._roi_mask is not None:
                # Per-component ROI boxes from metal detection: keeps analysis
                # local to each implant (bilateral-safe)
                constraint &= self._roi_mask

            dark_mask = (
                (self._volume >= _DARK_LOW)
                & (self._volume <= _DARK_HIGH)
                & constraint
            )

            bone_mask = (
                (self._volume >= _BONE_LOW)
                & (self._volume <= _BONE_HIGH)
                & ~dark_mask
                & constraint
            )

            bright_mask = (
                (self._volume >= _BRIGHT_LOW)
                & (self._volume <= _BRIGHT_HIGH)
                & ~dark_mask
                & ~bone_mask
                & constraint
            )

            self.finished.emit({
                "dark_artifacts":  dark_mask,
                "bright_artifacts": bright_mask,
                "bone":            bone_mask,
            })
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class SegmentationWorker(QObject):
    """Runs star-profile Russian Doll segmentation off the UI thread.

    Signals:
        finished(result): dict with keys dark_artifacts, bright_artifacts, bone.
        failed(message): human-readable error string.
    """

    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        volume: np.ndarray,
        spacing,
        metal_mask: np.ndarray,
        num_angles: int = 32,
        roi_mask: np.ndarray | None = None,
        bright_low: float = _BRIGHT_LOW,
        bright_high: float = _BRIGHT_HIGH,
        bone_low: float = _BONE_LOW,
        bone_high: float = _BONE_HIGH,
        w_hu: float = 0.45,
        w_width: float = 0.35,
        w_smooth: float = 0.25,
        w_gradient: float = 0.25,
    ):
        super().__init__()
        self._volume = volume
        self._spacing = np.abs(spacing)
        self._metal_mask = metal_mask
        self._num_angles = num_angles
        self._roi_mask = roi_mask
        self._bright_low = bright_low
        self._bright_high = bright_high
        self._bone_low = bone_low
        self._bone_high = bone_high
        self._w_hu = w_hu
        self._w_width = w_width
        self._w_smooth = w_smooth
        self._w_gradient = w_gradient

    def run(self) -> None:
        try:
            body_mask = create_body_mask(self._volume, air_threshold=-400)
            constraint = body_mask & ~self._metal_mask
            if self._roi_mask is not None:
                # Per-component ROI boxes from metal detection: keeps analysis
                # local to each implant (bilateral-safe)
                constraint &= self._roi_mask

            dark_mask = (
                (self._volume >= _DARK_LOW)
                & (self._volume <= _DARK_HIGH)
                & constraint
            )

            bright_mask = (
                (self._volume >= self._bright_low)
                & (self._volume <= self._bright_high)
                & ~dark_mask
                & constraint
            )

            discriminator = ArtifactDiscriminator(DiscriminationMethod.STAR_PROFILE)
            disc = discriminator.discriminate(
                self._volume,
                self._metal_mask,
                bright_mask,
                self._spacing,
                num_angles=self._num_angles,
                bone_hu_low=self._bone_low,
                bone_hu_high=self._bone_high,
                w_hu=self._w_hu,
                w_width=self._w_width,
                w_smooth=self._w_smooth,
                w_gradient=self._w_gradient,
                use_gpu=False,
            )

            self.finished.emit({
                "dark_artifacts":  dark_mask,
                "bright_artifacts": disc["artifact_mask"],
                "bone":            disc["bone_mask"],
            })
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
