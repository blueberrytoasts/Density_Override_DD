"""Main application window for the PySide CT viewer.

Layout:
    [ toolbar: Load Patient ]
    [        SliceView        ]   <- central, fills the window
    [ slice slider | W/L readout ]
    [ status bar: messages + HU under cursor ]

DICOM loading runs on a worker thread (``DicomLoadWorker``) so the UI stays
responsive; the window owns the thread/worker references for their lifetime.
"""
import numpy as np
from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from pyside_app.slice_view import SliceView
from pyside_app.hu_render import auto_window_level
from pyside_app.dicom_loader import DicomLoadWorker
from pyside_app.metal_worker import MetalDetectionWorker
from pyside_app.segment_worker import SegmentationWorker, LegacySegmentationWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CT Metal Artifact Viewer (PySide)")
        self.resize(1000, 800)

        self._thread: QThread | None = None
        self._worker: DicomLoadWorker | None = None

        # Separate thread/worker refs for detection so a detect run can't
        # clobber an in-flight load (and vice versa).
        self._detect_thread: QThread | None = None
        self._detect_worker: MetalDetectionWorker | None = None

        self._segment_thread: QThread | None = None
        self._segment_worker: SegmentationWorker | None = None

        # Held so detection (a separate user action) can reuse them after load.
        self._volume = None
        self._spacing = None

        # {slice_index: threshold_HU} from the last detection; empty until then.
        self._slice_thresholds: dict[int, float] = {}

        # Named window/level presets as (center, width) in HU. "Metal" is very
        # wide so 3000+ HU implants don't bloom to solid white and internal
        # structure/streaks stay visible.
        self._wl_presets: dict[str, tuple[float, float]] = {
            "Soft Tissue": (40.0, 400.0),
            "Bone": (400.0, 1800.0),
            "Lung": (-600.0, 1500.0),
            "Metal": (1000.0, 4000.0),
        }
        # True while a preset is being applied, so _on_window_changed can tell
        # preset-driven changes from manual right-drag adjustments.
        self._applying_wl_preset = False

        # Metal mask retained so segmentation can use it after detection.
        self._metal_mask = None
        # Per-component ROI mask (one box per implant) from detection.
        self._roi_mask = None
        # Thin outline of _roi_mask (per-slice box edges) for display.
        self._roi_outline = None
        # Star-profile placement overlay (lines + center dots) from detection.
        self._star_mask = None
        # Last segmentation masks so overlays can be recomposed on toggle.
        self._seg_masks: dict | None = None

        # --- central viewer ------------------------------------------------
        self._view = SliceView()

        # --- bottom control row -------------------------------------------
        self._slice_slider = QSlider(Qt.Orientation.Horizontal)
        self._slice_slider.setEnabled(False)
        self._slice_label = QLabel("Slice -/-")
        self._slice_label.setMinimumWidth(110)
        self._wl_label = QLabel("W/L: -/-")
        self._wl_label.setMinimumWidth(160)
        # Metal threshold for the slice currently shown (per-slice, not the avg).
        self._thr_label = QLabel("Metal thr: -")
        self._thr_label.setMinimumWidth(150)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Slice:"))
        controls.addWidget(self._slice_slider, stretch=1)
        controls.addWidget(self._slice_label)
        controls.addWidget(self._wl_label)
        controls.addWidget(self._thr_label)

        # Permanent legend — each entry is a toggle so overlays can be shown
        # or hidden individually (before/after comparisons for figures).
        legend = QHBoxLayout()
        legend.addWidget(QLabel("Overlays:"))
        # Master toggle: one click flips every overlay for before/after views.
        self._all_check = QCheckBox("All")
        self._all_check.setChecked(True)
        self._all_check.setToolTip("Show/hide all overlays at once.")
        self._all_check.toggled.connect(self._on_all_overlays_toggled)
        legend.addWidget(self._all_check)
        self._overlay_checks: dict[str, QCheckBox] = {}
        for key, color, name in (
            ("metal",  "#ff0000", "Metal"),
            ("dark",   "#ff00ff", "Dark artifact"),
            ("bright", "#ffff00", "Bright artifact"),
            ("bone",   "#0033cc", "Bone"),
            ("roi",    "#32ff32", "ROI"),
            ("star",   "#00ffff", "Star profiles"),
        ):
            check = QCheckBox(name)
            swatch = QPixmap(12, 12)
            swatch.fill(QColor(color))
            check.setIcon(QIcon(swatch))
            # ROI outline and star-profile placement are diagnostic — opt-in.
            check.setChecked(key not in ("roi", "star"))
            check.toggled.connect(self._refresh_overlays)
            self._overlay_checks[key] = check
            legend.addWidget(check)
        legend.addStretch(1)

        layout = QVBoxLayout()
        layout.addWidget(self._view, stretch=1)
        layout.addLayout(legend)
        layout.addLayout(controls)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # --- toolbar -------------------------------------------------------
        toolbar = self.addToolBar("Main")
        self._load_btn = QPushButton("Load Patient…")
        self._load_btn.clicked.connect(self._on_load_clicked)
        toolbar.addWidget(self._load_btn)

        self._detect_btn = QPushButton("Detect Metal")
        self._detect_btn.setEnabled(False)  # enabled once a volume is loaded
        self._detect_btn.clicked.connect(self._on_detect_clicked)
        toolbar.addWidget(self._detect_btn)

        self._segment_btn = QPushButton("Segment Artifacts")
        self._segment_btn.setEnabled(False)  # enabled once metal is detected
        self._segment_btn.clicked.connect(self._on_segment_clicked)
        toolbar.addWidget(self._segment_btn)

        self._seg_method_combo = QComboBox()
        self._seg_method_combo.addItem("Legacy (HU thresholds)", userData="legacy")
        self._seg_method_combo.addItem("Star Profile", userData="star")
        self._seg_method_combo.setToolTip(
            "Legacy: pure boolean HU ranges — fast, no discriminator.\n"
            "Star Profile: radial profile analysis to separate bone from artifact streaks."
        )
        # Star Profile is the method actually used for analysis; default to it
        # so each session doesn't start on Legacy.
        self._seg_method_combo.setCurrentIndex(self._seg_method_combo.findData("star"))
        toolbar.addWidget(self._seg_method_combo)

        self._export_btn = QPushButton("Export Slice…")
        self._export_btn.setEnabled(False)  # enabled once a volume is loaded
        self._export_btn.setToolTip(
            "Save the current slice (with visible overlays) as a lossless\n"
            "PNG/TIFF at 4x resolution for posters and figures."
        )
        self._export_btn.clicked.connect(self._on_export_clicked)
        toolbar.addWidget(self._export_btn)

        # FW% control: each star line's threshold is peak x (FW%/100). Lower
        # widens the mask, higher tightens it. Re-run detection to apply.
        toolbar.addWidget(QLabel(" FW%:"))
        self._fw_spin = QSpinBox()
        self._fw_spin.setRange(50, 95)
        self._fw_spin.setValue(75)
        self._fw_spin.setSuffix("%")
        self._fw_spin.setToolTip(
            "Full-Width %: per-line threshold = peak HU x (FW%/100).\n"
            "Lower = larger mask, higher = tighter. Re-run Detect Metal to apply."
        )
        toolbar.addWidget(self._fw_spin)

        # W/L presets: Auto re-derives from the volume, named entries are fixed
        # (center, width) pairs, Custom reflects a manual right-drag.
        toolbar.addWidget(QLabel(" W/L:"))
        self._wl_combo = QComboBox()
        self._wl_combo.addItem("Auto", userData="auto")
        for name, (center, width) in self._wl_presets.items():
            self._wl_combo.addItem(f"{name} ({width:.0f}/{center:.0f})", userData=name)
        self._wl_combo.addItem("Custom", userData="custom")
        self._wl_combo.setToolTip(
            "Window/level preset (width/center in HU).\n"
            "Right-drag on the image fine-tunes and switches to Custom."
        )
        self._wl_combo.currentIndexChanged.connect(self._on_wl_preset_changed)
        toolbar.addWidget(self._wl_combo)

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
        self._volume = volume
        self._spacing = meta.get("spacing")
        self._metal_mask = None
        self._roi_mask = None
        self._roi_outline = None
        self._star_mask = None
        self._seg_masks = None
        self._slice_thresholds = {}  # previous patient's thresholds no longer apply
        self._thr_label.setText("Metal thr: -")
        self._segment_btn.setEnabled(False)
        self._view.set_volume(volume)
        # set_volume applied auto W/L; reflect that in the preset combo
        # (its window_changed just flipped the combo to Custom).
        self._wl_combo.blockSignals(True)
        self._wl_combo.setCurrentIndex(self._wl_combo.findData("auto"))
        self._wl_combo.blockSignals(False)
        self._slice_slider.setEnabled(True)
        self._slice_slider.setMaximum(volume.shape[0] - 1)
        self._slice_slider.setValue(volume.shape[0] // 2)
        self._load_btn.setEnabled(True)
        self._detect_btn.setEnabled(True)
        self._export_btn.setEnabled(True)
        ct_dir = meta.get("ct_dir", "")
        self.statusBar().showMessage(
            f"Loaded {volume.shape[0]} slices "
            f"({volume.shape[2]}x{volume.shape[1]}) from {ct_dir}"
        )

    def _on_load_failed(self, message):
        self._load_btn.setEnabled(True)
        self.statusBar().showMessage(f"Load failed: {message.splitlines()[0]}")

    # ---- metal detection -------------------------------------------------
    def _on_detect_clicked(self):
        if self._volume is None:
            return

        self._detect_btn.setEnabled(False)
        self.statusBar().showMessage("Detecting metal…")

        self._detect_thread = QThread(self)
        self._detect_worker = MetalDetectionWorker(
            self._volume, self._spacing, fw_percentage=float(self._fw_spin.value())
        )
        self._detect_worker.moveToThread(self._detect_thread)
        self._detect_thread.started.connect(self._detect_worker.run)
        self._detect_worker.finished.connect(self._on_detect_finished)
        self._detect_worker.failed.connect(self._on_detect_failed)
        # Tear down the thread once the worker reports either outcome.
        self._detect_worker.finished.connect(self._detect_thread.quit)
        self._detect_worker.failed.connect(self._detect_thread.quit)
        self._detect_thread.start()

    def _on_detect_finished(self, result):
        self._detect_btn.setEnabled(True)
        self._metal_mask = result["mask"]
        self._roi_mask = result.get("roi_mask")
        self._roi_outline = self._compute_roi_outline(self._roi_mask)
        self._star_mask = result.get("star_mask")
        self._seg_masks = None  # previous segmentation no longer matches
        voxels = int(self._metal_mask.sum())
        # Per-slice thresholds (keyed by slice index); keys may be numpy ints.
        self._slice_thresholds = {
            int(z): float(t) for z, t in result.get("threshold_evolution", {}).items()
        }
        avg = result.get("threshold")
        avg_txt = f" (avg ~{avg:.0f} HU over {len(self._slice_thresholds)} slices)" if avg else ""
        self._refresh_overlays()
        self._segment_btn.setEnabled(True)
        self.statusBar().showMessage(f"Metal detected: {voxels:,} voxels{avg_txt}")
        # Refresh the per-slice readout for whatever slice is showing now.
        self._update_threshold_label(self._slice_slider.value())

    def _update_threshold_label(self, index: int) -> None:
        """Show the metal threshold for slice ``index`` (or '-' if none)."""
        thr = self._slice_thresholds.get(index)
        if thr is None:
            self._thr_label.setText("Metal thr: -")
        else:
            self._thr_label.setText(f"Metal thr: >{thr:.0f} HU")

    def _on_detect_failed(self, message):
        self._detect_btn.setEnabled(True)
        self.statusBar().showMessage(f"Detection: {message.splitlines()[0]}")

    # ---- artifact segmentation -------------------------------------------
    def _on_segment_clicked(self):
        if self._volume is None or self._metal_mask is None:
            return

        method = self._seg_method_combo.currentData()
        self._segment_btn.setEnabled(False)

        self._segment_thread = QThread(self)
        if method == "legacy":
            self.statusBar().showMessage("Segmenting artifacts (legacy HU thresholds)…")
            self._segment_worker = LegacySegmentationWorker(
                self._volume, self._metal_mask, roi_mask=self._roi_mask
            )
        else:
            self.statusBar().showMessage("Segmenting artifacts (star profile)…")
            self._segment_worker = SegmentationWorker(
                self._volume, self._spacing, self._metal_mask,
                roi_mask=self._roi_mask,
            )
        self._segment_worker.moveToThread(self._segment_thread)
        self._segment_thread.started.connect(self._segment_worker.run)
        self._segment_worker.finished.connect(self._on_segment_finished)
        self._segment_worker.failed.connect(self._on_segment_failed)
        self._segment_worker.finished.connect(self._segment_thread.quit)
        self._segment_worker.failed.connect(self._segment_thread.quit)
        self._segment_thread.start()

    def _on_segment_finished(self, result):
        self._segment_btn.setEnabled(True)
        dark = result["dark_artifacts"]
        bright = result["bright_artifacts"]
        bone = result["bone"]
        self._seg_masks = {"dark": dark, "bright": bright, "bone": bone}
        self._refresh_overlays()
        self.statusBar().showMessage(
            f"Segmentation done — dark: {int(dark.sum()):,}  "
            f"bright artifact: {int(bright.sum()):,}  bone: {int(bone.sum()):,}"
        )

    # ---- export ------------------------------------------------------------
    def _on_export_clicked(self):
        image = self._view.current_image()
        if image is None:
            return
        default_name = f"slice_{self._slice_slider.value() + 1:03d}.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export slice", default_name,
            "PNG image (*.png);;TIFF image (*.tif)",
        )
        if not path:
            return
        # 4x nearest-neighbour upscale: lossless, keeps mask edges razor sharp
        # at poster print sizes (512 px native is too soft at 300 DPI).
        scaled = image.scaled(
            image.width() * 4, image.height() * 4,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        if scaled.save(path):
            self.statusBar().showMessage(
                f"Exported {scaled.width()}x{scaled.height()} image to {path}"
            )
        else:
            self.statusBar().showMessage(f"Export failed: could not write {path}")

    # ---- overlay composition ---------------------------------------------
    @staticmethod
    def _compute_roi_outline(roi_mask) -> "np.ndarray | None":
        """2-pixel in-plane edge of the ROI mask, for drawing box outlines."""
        if roi_mask is None:
            return None
        from scipy.ndimage import binary_erosion
        # (1, 5, 5) structure erodes each slice in 2D only, leaving a
        # 2-px rim; boxes on adjacent slices don't bleed into each other.
        eroded = binary_erosion(roi_mask, structure=np.ones((1, 5, 5), dtype=bool))
        return roi_mask & ~eroded

    def _on_all_overlays_toggled(self, checked: bool) -> None:
        """Flip every overlay checkbox in one click (before/after shots)."""
        for check in self._overlay_checks.values():
            check.blockSignals(True)  # one redraw at the end, not one per box
            check.setChecked(checked)
            check.blockSignals(False)
        self._refresh_overlays()

    def _refresh_overlays(self) -> None:
        """Recompose viewer overlays from current masks + legend toggles."""
        show = {k: c.isChecked() for k, c in self._overlay_checks.items()}
        overlays = []
        if self._metal_mask is not None and show["metal"]:
            overlays.append((self._metal_mask, (255, 0, 0), 0.7))     # red
        if self._seg_masks is not None:
            if show["dark"]:
                overlays.append((self._seg_masks["dark"],   (255,   0, 255), 0.6))  # magenta
            if show["bright"]:
                overlays.append((self._seg_masks["bright"], (255, 255,   0), 0.6))  # yellow
            if show["bone"]:
                overlays.append((self._seg_masks["bone"],   (  0,  51, 204), 0.5))  # blue
        if self._roi_outline is not None and show["roi"]:
            overlays.append((self._roi_outline, (50, 255, 50), 0.9))  # lime
        # Last so the thin lines stay visible on top of the other masks.
        if self._star_mask is not None and show["star"]:
            overlays.append((self._star_mask, (0, 255, 255), 0.9))    # cyan
        self._view.set_overlays(overlays)

    def _on_segment_failed(self, message):
        self._segment_btn.setEnabled(True)
        self.statusBar().showMessage(f"Segmentation: {message.splitlines()[0]}")

    # ---- view callbacks --------------------------------------------------
    def _on_slice_changed(self, index, total):
        self._slice_label.setText(f"Slice {index + 1}/{total}")
        # Keep the slider in sync without re-triggering set_slice.
        self._slice_slider.blockSignals(True)
        self._slice_slider.setValue(index)
        self._slice_slider.blockSignals(False)
        self._update_threshold_label(index)

    def _on_wl_preset_changed(self, _index: int) -> None:
        """Apply the selected window/level preset to the viewer."""
        key = self._wl_combo.currentData()
        if key == "custom":
            return  # placeholder entry; the current window is already custom
        if key == "auto":
            if self._volume is None:
                return
            center, width = auto_window_level(self._volume)
        else:
            center, width = self._wl_presets[key]
        self._applying_wl_preset = True
        try:
            self._view.set_window(center, width)
        finally:
            self._applying_wl_preset = False

    def _on_window_changed(self, center, width):
        self._wl_label.setText(f"W/L: {width:.0f}/{center:.0f}")
        if not self._applying_wl_preset:
            # Manual right-drag (or volume load) — the window no longer
            # matches the selected preset, so show Custom.
            self._wl_combo.blockSignals(True)
            self._wl_combo.setCurrentIndex(self._wl_combo.findData("custom"))
            self._wl_combo.blockSignals(False)

    def _on_cursor_hu(self, hu):
        self._hu_label.setText("HU: -" if hu is None else f"HU: {hu}")
