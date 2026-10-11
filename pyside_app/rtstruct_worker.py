"""Background RTSTRUCT export for the PySide app.

Tracing contours on every slice of several structures takes a few seconds, so
-- like the other workers -- it runs on a worker thread. The DICOM building
lives in ``algorithms/rtstruct.py``; this file only calls it off the UI thread
and reports back through signals.
"""
from PySide6.QtCore import QObject, Signal

from pyside_app import bootstrap  # noqa: F401  (side effect: puts algorithms/ on sys.path)
from rtstruct import create_rtstruct, save_rtstruct


class RtstructExportWorker(QObject):
    """Writes the given structures to an RTSTRUCT file.

    Signals:
        finished(path, names): output path and the structure names written
            (empty masks are skipped).
        failed(message): human-readable error string.
    """

    finished = Signal(str, object)
    failed = Signal(str)

    def __init__(self, structures, meta, shape, path):
        super().__init__()
        self._structures = structures  # [(name, mask, (r, g, b)), ...]
        self._meta = meta
        self._shape = shape
        self._path = path

    def run(self) -> None:
        try:
            ds, names = create_rtstruct(self._structures, self._meta, self._shape)
            if not names:
                self.failed.emit("All selected structures are empty — nothing to export.")
                return
            save_rtstruct(ds, self._path)
            self.finished.emit(self._path, names)
        except Exception as exc:  # surface any export error to the UI
            self.failed.emit(f"{type(exc).__name__}: {exc}")
