"""Background body-mask computation for the PySide app.

The body mask (largest connected component per slice, holes filled) is what
lets the viewer blank out the CT couch: the couch is separated from the
patient by an air gap, so it is never part of the largest component.
Computing it runs 2D+3D morphology over the whole volume (a few seconds),
so — like the other workers — it runs off the UI thread and is cached by the
window for the lifetime of the loaded patient.
"""
import numpy as np
from PySide6.QtCore import QObject, Signal

from pyside_app import bootstrap  # noqa: F401  (puts app/ on sys.path)
from body_mask import create_body_mask


class BodyMaskWorker(QObject):
    """Computes the 3D body mask off the UI thread.

    Signals:
        finished(mask): 3D bool array, True = inside the patient's body.
        failed(message): human-readable error string.
    """

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, volume: np.ndarray):
        super().__init__()
        self._volume = volume

    def run(self) -> None:
        try:
            # Same air threshold the segmentation workers use.
            mask = create_body_mask(self._volume, air_threshold=-400)
            self.finished.emit(mask)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
