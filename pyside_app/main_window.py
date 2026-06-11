"""Main application window for the PySide CT viewer.

Layout:
    [ toolbar: Load Patient ]
    [        SliceView        ]   <- central, fills the window
    [ slice slider | W/L readout ]
    [ status bar: messages + HU under cursor ]

DICOM loading runs on a worker thread (``DicomLoadWorker``) so the UI stays
responsive; the window owns the thread/worker references for their lifetime.
"""
from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QMainWindow, QPushButton, QSlider,
    QVBoxLayout, QWidget,
)

from pyside_app.slice_view import SliceView
from pyside_app.dicom_loader import DicomLoadWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CT Metal Artifact Viewer (PySide)")
        self.resize(1000, 800)

        self._thread: QThread | None = None
        self._worker: DicomLoadWorker | None = None

        # --- central viewer ------------------------------------------------
        self._view = SliceView()

        # --- bottom control row -------------------------------------------
        self._slice_slider = QSlider(Qt.Orientation.Horizontal)
        self._slice_slider.setEnabled(False)
        self._slice_label = QLabel("Slice -/-")
        self._slice_label.setMinimumWidth(110)
        self._wl_label = QLabel("W/L: -/-")
        self._wl_label.setMinimumWidth(160)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Slice:"))
        controls.addWidget(self._slice_slider, stretch=1)
        controls.addWidget(self._slice_label)
        controls.addWidget(self._wl_label)

        layout = QVBoxLayout()
        layout.addWidget(self._view, stretch=1)
        layout.addLayout(controls)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # --- toolbar -------------------------------------------------------
        toolbar = self.addToolBar("Main")
        self._load_btn = QPushButton("Load Patient…")
        self._load_btn.clicked.connect(self._on_load_clicked)
        toolbar.addWidget(self._load_btn)

        # --- status bar ----------------------------------------------------
        self._hu_label = QLabel("HU: -")
        self.statusBar().addPermanentWidget(self._hu_label)
        self.statusBar().showMessage("Load a patient folder to begin.")

        # --- wiring --------------------------------------------------------
        self._view.slice_changed.connect(self._on_slice_changed)
        self._view.window_changed.connect(self._on_window_changed)
        self._view.cursor_hu.connect(self._on_cursor_hu)
        self._slice_slider.valueChanged.connect(self._view.set_slice)

    # ---- loading ---------------------------------------------------------
    def _on_load_clicked(self):
        directory = QFileDialog.getExistingDirectory(self, "Select patient folder")
        if not directory:
            return

        self._load_btn.setEnabled(False)
        self.statusBar().showMessage(f"Loading {directory} …")

        self._thread = QThread(self)
        self._worker = DicomLoadWorker(directory)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_load_finished)
        self._worker.failed.connect(self._on_load_failed)
        # Tear down the thread once the worker reports either outcome.
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.start()

    def _on_load_finished(self, volume, meta):
        self._view.set_volume(volume)
        self._slice_slider.setEnabled(True)
        self._slice_slider.setMaximum(volume.shape[0] - 1)
        self._slice_slider.setValue(volume.shape[0] // 2)
        self._load_btn.setEnabled(True)
        ct_dir = meta.get("ct_dir", "")
        self.statusBar().showMessage(
            f"Loaded {volume.shape[0]} slices "
            f"({volume.shape[2]}x{volume.shape[1]}) from {ct_dir}"
        )

    def _on_load_failed(self, message):
        self._load_btn.setEnabled(True)
        self.statusBar().showMessage(f"Load failed: {message.splitlines()[0]}")

    # ---- view callbacks --------------------------------------------------
    def _on_slice_changed(self, index, total):
        self._slice_label.setText(f"Slice {index + 1}/{total}")
        # Keep the slider in sync without re-triggering set_slice.
        self._slice_slider.blockSignals(True)
        self._slice_slider.setValue(index)
        self._slice_slider.blockSignals(False)

    def _on_window_changed(self, center, width):
        self._wl_label.setText(f"W/L: {width:.0f}/{center:.0f}")

    def _on_cursor_hu(self, hu):
        self._hu_label.setText("HU: -" if hu is None else f"HU: {hu}")
