"""Background artifact segmentation for the PySide app.

Two workers are provided:
    SegmentationWorker      — star profile discrimination (slow, smarter)
    LegacySegmentationWorker — pure boolean HU thresholds (fast, no ML)

Result dict keys (both workers):
    dark_artifacts   : 3D bool mask
    bright_artifacts : 3D bool mask
    bone             : 3D bool mask

SegmentationWorker additionally returns the *bright* over-bone/over-tissue
split (what tissue the artifact is corrupting, for density override), which the
discriminator produces for free:
    bright_artifact_bone / bright_artifact_tissue : split of bright_artifacts

The remaining splits (dark for both methods, bright for legacy) are expensive
(per-voxel neighborhood loop) and only feed opt-in overlays, so they are NOT
computed here — ContextSplitWorker computes each lazily on demand:
    dark_artifact_bone   / dark_artifact_tissue   : split of dark_artifacts
    bright_artifact_bone / bright_artifact_tissue : split of bright_artifacts
"""
import numpy as np
from PySide6.QtCore import QObject, Signal

from pyside_app import bootstrap  # noqa: F401  (puts app/ on sys.path)
from body_mask import create_body_mask
from contour_operations import classify_artifacts_contextually
from core.discrimination import (
    ArtifactDiscriminator, DiscriminationMethod,
    build_discriminator_star_overlay,
)

# Default HU thresholds matching ThresholdConfig in app/config.py
_DARK_LOW = -1024
_DARK_HIGH = -150
_BRIGHT_LOW = 200
_BRIGHT_HIGH = 2500
_BONE_LOW = 400
_BONE_HIGH = 1800
# Context bands for the over-bone/over-tissue sub-typing (Decision 2). These
# decide what surrounding HU counts as "bone" vs "soft tissue" when judging
# what an artifact is corrupting; distinct from the bone vote band above.
_CTX_BONE_LOW = 500
_CTX_BONE_HIGH = 1500
_CTX_TISSUE_LOW = -100
_CTX_TISSUE_HIGH = 300
# In-plane neighborhood width (px) for the over-bone/over-tissue vote. Larger =
# each artifact pixel "sees" farther, so boundary pixels next to bone can be
# pulled toward bone. Drives both the bright and dark splits (z fixed at ±1).
_CTX_WINDOW = 5


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
                 roi_mask: np.ndarray | None = None,
                 dark_low: float = _DARK_LOW,
                 dark_high: float = _DARK_HIGH):
        super().__init__()
        self._volume = volume
        self._metal_mask = metal_mask
        self._roi_mask = roi_mask
        self._dark_low = dark_low
        self._dark_high = dark_high

    def run(self) -> None:
        try:
            body_mask = create_body_mask(self._volume, air_threshold=-400)
            constraint = body_mask & ~self._metal_mask
            if self._roi_mask is not None:
                # Per-component ROI boxes from metal detection: keeps analysis
                # local to each implant (bilateral-safe)
                constraint &= self._roi_mask

            dark_mask = (
                (self._volume >= self._dark_low)
                & (self._volume <= self._dark_high)
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
        dark_low: float = _DARK_LOW,
        dark_high: float = _DARK_HIGH,
        bright_low: float = _BRIGHT_LOW,
        bright_high: float = _BRIGHT_HIGH,
        bone_low: float = _BONE_LOW,
        bone_high: float = _BONE_HIGH,
        w_hu: float = 0.45,
        w_width: float = 0.35,
        w_smooth: float = 0.25,
        w_gradient: float = 0.25,
        ctx_bone_low: float = _CTX_BONE_LOW,
        ctx_bone_high: float = _CTX_BONE_HIGH,
        ctx_tissue_low: float = _CTX_TISSUE_LOW,
        ctx_tissue_high: float = _CTX_TISSUE_HIGH,
        ctx_window: int = _CTX_WINDOW,
    ):
        super().__init__()
        self._volume = volume
        self._spacing = np.abs(spacing)
        self._metal_mask = metal_mask
        self._num_angles = num_angles
        self._roi_mask = roi_mask
        self._dark_low = dark_low
        self._dark_high = dark_high
        self._bright_low = bright_low
        self._bright_high = bright_high
        self._bone_low = bone_low
        self._bone_high = bone_high
        self._w_hu = w_hu
        self._w_width = w_width
        self._w_smooth = w_smooth
        self._w_gradient = w_gradient
        self._ctx_bone_low = ctx_bone_low
        self._ctx_bone_high = ctx_bone_high
        self._ctx_tissue_low = ctx_tissue_low
        self._ctx_tissue_high = ctx_tissue_high
        self._ctx_window = ctx_window

    def run(self) -> None:
        try:
            body_mask = create_body_mask(self._volume, air_threshold=-400)
            constraint = body_mask & ~self._metal_mask
            if self._roi_mask is not None:
                # Per-component ROI boxes from metal detection: keeps analysis
                # local to each implant (bilateral-safe)
                constraint &= self._roi_mask

            dark_mask = (
                (self._volume >= self._dark_low)
                & (self._volume <= self._dark_high)
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
                tissue_hu_low=self._ctx_tissue_low,
                tissue_hu_high=self._ctx_tissue_high,
                ctx_bone_low=self._ctx_bone_low,
                ctx_bone_high=self._ctx_bone_high,
                ctx_window=self._ctx_window,
                w_hu=self._w_hu,
                w_width=self._w_width,
                w_smooth=self._w_smooth,
                w_gradient=self._w_gradient,
                use_gpu=False,
            )

            # Rasterize the discriminator's star placements (still off the UI
            # thread) so the window can toggle the overlay instantly.
            star_mask = build_discriminator_star_overlay(
                self._volume.shape,
                disc.get("star_centers", {}),
                self._num_angles,
            )

            # NB: the bright over-bone/over-tissue split comes free with the
            # discriminator result. The *dark* split is expensive (per-voxel
            # neighborhood loop) and only feeds opt-in overlays, so it is NOT
            # computed here — it runs lazily via ContextSplitWorker the first
            # time the user views a dark-split overlay.
            self.finished.emit({
                "dark_artifacts":  dark_mask,
                "bright_artifacts": disc["artifact_mask"],
                "bone":            disc["bone_mask"],
                "disc_star_mask":  star_mask,
                "bright_artifact_bone":   disc.get("artifact_bone_mask"),
                "bright_artifact_tissue": disc.get("artifact_tissue_mask"),
            })
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ContextSplitWorker(QObject):
    """Lazily computes an artifact mask's over-bone/over-tissue split.

    Works for either parent mask: ``kind="dark"`` (both segmentation methods)
    or ``kind="bright"`` (legacy method — the star-profile worker already gets
    its bright split from the discriminator). This is the expensive per-voxel
    neighborhood classification, deferred out of the segmentation workers so it
    only runs when the user actually views a split overlay. The context HU
    bands are captured from the Segment run the split belongs to (not the live
    spin boxes), so the result stays consistent with that segmentation.

    Signals:
        finished(kind, result): result dict with <kind>_artifact_bone /
            <kind>_artifact_tissue. The kind is emitted so the window can
            connect a plain bound method (queued back to the GUI thread);
            connecting a lambda to capture the kind would run the slot in
            *this* thread and touch the GUI from it.
        failed(kind, message): human-readable error string.
    """

    finished = Signal(str, object)
    failed = Signal(str, str)

    def __init__(self, volume, parent_mask, metal_mask, spacing, kind,
                 ctx_bone_low=_CTX_BONE_LOW, ctx_bone_high=_CTX_BONE_HIGH,
                 ctx_tissue_low=_CTX_TISSUE_LOW, ctx_tissue_high=_CTX_TISSUE_HIGH,
                 ctx_window=_CTX_WINDOW):
        super().__init__()
        self._volume = volume
        self._parent_mask = parent_mask
        self._metal_mask = metal_mask
        self._spacing = np.abs(spacing)
        self._kind = kind
        self._ctx_bone_low = ctx_bone_low
        self._ctx_bone_high = ctx_bone_high
        self._ctx_tissue_low = ctx_tissue_low
        self._ctx_tissue_high = ctx_tissue_high
        self._ctx_window = ctx_window

    def run(self) -> None:
        try:
            n = int(self._ctx_window)
            ctx = classify_artifacts_contextually(
                self._volume, self._parent_mask, self._metal_mask, self._spacing,
                bone_range=(self._ctx_bone_low, self._ctx_bone_high),
                tissue_range=(self._ctx_tissue_low, self._ctx_tissue_high),
                # (z, y, x): match the discriminator's neighborhood — ±1 slice,
                # n×n in-plane — so all splits use the same window.
                window_size=(3, n, n),
                artifact_type=self._kind,
            )
            self.finished.emit(self._kind, {
                f"{self._kind}_artifact_bone":   ctx["artifact_bone"],
                f"{self._kind}_artifact_tissue": ctx["artifact_tissue"],
            })
        except Exception as exc:
            self.failed.emit(self._kind, f"{type(exc).__name__}: {exc}")
