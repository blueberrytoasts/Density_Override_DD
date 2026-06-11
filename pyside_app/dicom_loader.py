"""Background DICOM loading for the PySide app.

Loading a CT series reads hundreds of files and can take a few seconds, so it
runs on a worker thread (``DicomLoadWorker`` moved onto a ``QThread``) to keep
the UI responsive. This is the canonical Qt pattern: a plain ``QObject`` worker
with signals, moved to a thread, never touching widgets directly.
"""
import os

from PySide6.QtCore import QObject, Signal

from pyside_app import bootstrap  # noqa: F401  (side effect: puts app/ on sys.path)
from dicom_utils import load_dicom_series_to_hu


def resolve_ct_dir(root: str) -> str | None:
    """Find the directory that holds the CT slices under ``root``.

    Patient folders nest the CT series in a subfolder alongside an RTSTRUCT
    folder. The CT series has hundreds of files while RTSTRUCT has ~1, so we
    pick the directory (at any depth, including ``root`` itself) that directly
    contains the most files. This is fast (no DICOM parsing) and robust across
    the differing folder-naming conventions in ``data/``.

    Returns:
        Absolute path to the best candidate directory, or ``None`` if ``root``
        contains no files at all.
    """
    best_dir = None
    best_count = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        n = len(filenames)
        if n > best_count:
            best_count = n
            best_dir = dirpath
    return best_dir


class DicomLoadWorker(QObject):
    """Loads a CT series off the UI thread.

    Signals:
        finished(volume, meta): emitted with the HU volume (np.ndarray) and the
            spatial metadata dict on success.
        failed(message): emitted with a human-readable error string.
    """

    finished = Signal(object, object)
    failed = Signal(str)

    def __init__(self, patient_dir: str):
        super().__init__()
        self._patient_dir = patient_dir

    def run(self) -> None:
        try:
            ct_dir = resolve_ct_dir(self._patient_dir)
            if ct_dir is None:
                self.failed.emit(f"No files found under:\n{self._patient_dir}")
                return

            volume, meta = load_dicom_series_to_hu(ct_dir)
            if volume is None:
                self.failed.emit(f"No CT DICOM images found in:\n{ct_dir}")
                return

            meta = dict(meta)
            meta["ct_dir"] = ct_dir
            self.finished.emit(volume, meta)
        except Exception as exc:  # surface any loader error to the UI
            self.failed.emit(f"{type(exc).__name__}: {exc}")
