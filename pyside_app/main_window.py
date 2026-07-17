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
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel,
    QMainWindow, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from pyside_app.slice_view import SliceView
from pyside_app.hu_render import auto_window_level
from pyside_app.dicom_loader import DicomLoadWorker
from pyside_app.metal_worker import MetalDetectionWorker
from pyside_app.segment_worker import (
    SegmentationWorker, LegacySegmentationWorker, ContextSplitWorker,
)
from pyside_app.body_mask_worker import BodyMaskWorker


def _autocrop_to_content(image: QImage, threshold: int = 10,
                         margin: int = 12) -> QImage:
    """Crop the black surround off a rendered slice, keeping a margin.

    Air renders as (near-)black at any sane window/level, so the content
    bounding box is simply "pixels brighter than threshold". Returns the
    image unchanged if it is entirely black.
    """
    gray = image.convertToFormat(QImage.Format.Format_Grayscale8)
    h, w = gray.height(), gray.width()
    buf = np.frombuffer(gray.constBits(), dtype=np.uint8)
    buf = buf.reshape(h, gray.bytesPerLine())[:, :w]
    rows = np.any(buf > threshold, axis=1)
    cols = np.any(buf > threshold, axis=0)
    if not rows.any():
        return image
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    top = max(int(top) - margin, 0)
    left = max(int(left) - margin, 0)
    bottom = min(int(bottom) + margin, h - 1)
    right = min(int(right) + margin, w - 1)
    return image.copy(left, top, right - left + 1, bottom - top + 1)


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

        # Lazy over-bone/over-tissue splits, keyed by parent mask kind
        # ("bright"/"dark"): computed on demand the first time a split overlay
        # is viewed (expensive per-voxel loop). Star-profile runs get the
        # bright split eagerly from the discriminator; legacy runs get both
        # lazily.
        self._ctx_threads: dict[str, QThread] = {}
        self._ctx_workers: dict[str, ContextSplitWorker] = {}
        # The parent mask an in-flight split is being computed against
        # (identity check so a stale result isn't painted onto a newer
        # segmentation).
        self._ctx_for: dict[str, object] = {}
        # Context HU bands captured at the last Segment run, so the lazy
        # splits match that segmentation rather than the live spin boxes.
        self._seg_ctx_bands: tuple | None = None

        self._bed_thread: QThread | None = None
        self._bed_worker: BodyMaskWorker | None = None
        # Cached body mask for the loaded patient (computed on first toggle).
        self._body_mask = None

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
        # Discriminator star-profile placement overlay (from segmentation).
        self._disc_star_mask = None
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
        # Single source of truth for legend key → (swatch color, label). Reused
        # by the on-screen toggles and by Export Legend.
        self._legend_entries = (
            ("metal",         "#ff0000", "Metal"),
            ("dark",          "#ff00ff", "Dark artifact"),
            ("bright",        "#ffff00", "Bright artifact"),
            ("bone",          "#0033cc", "Bone"),
            ("bright_bone",   "#ff8000", "Bright→bone"),
            ("bright_tissue", "#ff69b4", "Bright→tissue"),
            ("dark_bone",     "#8a2be2", "Dark→bone"),
            ("dark_tissue",   "#00ced1", "Dark→tissue"),
            ("roi",           "#32ff32", "ROI"),
            ("star",          "#00ffff", "Metal stars"),
            ("disc_star",     "#ffffff", "Disc stars"),
        )
        # Diagnostic overlays excluded from an exported poster legend.
        self._legend_export_exclude = frozenset({"roi", "star", "disc_star"})
        self._overlay_checks: dict[str, QCheckBox] = {}
        for key, color, name in self._legend_entries:
            check = QCheckBox(name)
            swatch = QPixmap(12, 12)
            swatch.fill(QColor(color))
            check.setIcon(QIcon(swatch))
            # Contextual splits, ROI outline, and star placements are
            # diagnostic — opt-in.
            check.setChecked(key not in (
                "bright_bone", "bright_tissue", "dark_bone", "dark_tissue",
                "roi", "star", "disc_star"))
            check.toggled.connect(self._refresh_overlays)
            self._overlay_checks[key] = check
            legend.addWidget(check)
        # Viewing a split overlay triggers its lazy classification when the
        # segmentation didn't already provide it (dark always; bright when the
        # legacy method was used).
        for _k in ("bright_bone", "bright_tissue", "dark_bone", "dark_tissue"):
            self._overlay_checks[_k].toggled.connect(self._ensure_context_splits)
        _ctx_tip = (
            "Contextual split of the {kind} artifact mask: what tissue the\n"
            "artifact is corrupting underneath (decides the HU it should be\n"
            "restored to). Subset of '{parent}'; computed on first view when\n"
            "the segmentation didn't already provide it.")
        self._overlay_checks["bright_bone"].setToolTip(
            _ctx_tip.format(kind="bright", parent="Bright artifact"))
        self._overlay_checks["bright_tissue"].setToolTip(
            _ctx_tip.format(kind="bright", parent="Bright artifact"))
        self._overlay_checks["dark_bone"].setToolTip(
            _ctx_tip.format(kind="dark", parent="Dark artifact"))
        self._overlay_checks["dark_tissue"].setToolTip(
            _ctx_tip.format(kind="dark", parent="Dark artifact"))
        self._overlay_checks["star"].setToolTip(
            "Metal-detection stars: the FW% rays shot from each metal blob to\n"
            "set that slice's metal threshold. Appears after Detect Metal.")
        self._overlay_checks["disc_star"].setToolTip(
            "Discriminator stars: the radial profiles shot from each implant's\n"
            "centroid to judge bone vs. bright artifact. Appears after Segment\n"
            "Artifacts (Star Profile method).")
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
            "PNG/TIFF at 4x resolution for posters and figures.\n"
            "The black air border is cropped off automatically."
        )
        self._export_btn.clicked.connect(self._on_export_clicked)
        toolbar.addWidget(self._export_btn)

        self._export_legend_btn = QPushButton("Export Legend…")
        self._export_legend_btn.setToolTip(
            "Save a standalone color-key PNG for poster figures. Includes the\n"
            "tissue/artifact classes currently toggled on, so the key matches\n"
            "your exported slice; ROI, metal stars, and disc stars are never\n"
            "included. Toggle the classes you want, then export."
        )
        self._export_legend_btn.clicked.connect(self._on_export_legend_clicked)
        toolbar.addWidget(self._export_legend_btn)

        # Hide the CT couch: renders everything outside the body mask as air.
        # Applies to the live view AND exports (export saves the rendered view).
        self._hide_bed_check = QCheckBox("Hide bed")
        self._hide_bed_check.setEnabled(False)  # needs a loaded volume
        self._hide_bed_check.setToolTip(
            "Blank everything outside the patient's body (CT couch, blankets).\n"
            "First use per patient computes the body mask (a few seconds).\n"
            "Exported images honor this setting."
        )
        self._hide_bed_check.toggled.connect(self._on_hide_bed_toggled)
        toolbar.addWidget(self._hide_bed_check)

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

        # --- second toolbar: segmentation/discrimination tuning ------------
        # All of these apply on the next "Segment Artifacts" run; none require
        # re-detecting metal. Dark HU applies to both methods; the bright/bone
        # ranges and weights only affect the Star Profile method.
        self.addToolBarBreak()
        disc_bar = self.addToolBar("Discrimination")

        disc_bar.addWidget(QLabel("Dark HU:"))
        # Floor set well below the usual -1024 air line: reconstruction padding
        # and severe dark streaks can read down to ~-3000 HU in this data.
        self._dark_low_spin = self._make_hu_spin(
            -4000, 500, -1024,
            "Lower bound of the dark artifact range (plain threshold, applies\n"
            "to both segmentation methods). Can go below -1024 to reach\n"
            "reconstruction padding / very dark streaks (down to ~-3000 HU).")
        self._dark_high_spin = self._make_hu_spin(
            -4000, 500, -150,
            "Upper bound of the dark artifact range. Raising it toward 0 grabs\n"
            "more of the gray streak shadows (and eventually fat, ~-100 HU).")
        disc_bar.addWidget(self._dark_low_spin)
        disc_bar.addWidget(QLabel("–"))
        disc_bar.addWidget(self._dark_high_spin)

        disc_bar.addWidget(QLabel("   Bright HU (gate):"))
        self._bright_low_spin = self._make_hu_spin(
            -1024, 5000, 200,
            "Lower bound of the bright candidate pool. Pixels outside the\n"
            "bright range are never judged (not bone, not bright artifact).")
        self._bright_high_spin = self._make_hu_spin(
            -1024, 5000, 2500,
            "Upper bound of the bright candidate pool.")
        disc_bar.addWidget(self._bright_low_spin)
        disc_bar.addWidget(QLabel("–"))
        disc_bar.addWidget(self._bright_high_spin)

        disc_bar.addWidget(QLabel("   Bone HU (vote band):"))
        self._bone_low_spin = self._make_hu_spin(
            -1024, 5000, 400,
            "Bone HU band used by the HU vote (not a hard gate). Pixels inside\n"
            "this band are pushed toward 'bone'; the shape votes can override it.")
        self._bone_high_spin = self._make_hu_spin(
            -1024, 5000, 1800,
            "Upper bound of the bone HU vote band.")
        disc_bar.addWidget(self._bone_low_spin)
        disc_bar.addWidget(QLabel("–"))
        disc_bar.addWidget(self._bone_high_spin)

        # Weights: each feature votes ±weight toward bone; bone_score>0 => bone.
        weights_label = QLabel("   Vote weights →  HU:")
        weights_label.setToolTip(
            "How loudly each of the 4 clues votes for 'bone'. A pixel's votes are\n"
            "summed; if the total is positive it's called bone, else bright artifact.\n"
            "Bigger weight = that clue matters more. Set to 0 to ignore a clue.")
        disc_bar.addWidget(weights_label)
        self._w_hu_spin = self._make_weight_spin(
            0.45, "HU: is the pixel's brightness inside the bone HU band above?\n"
                  "Default 0.45 — the loudest vote.")
        disc_bar.addWidget(self._w_hu_spin)
        disc_bar.addWidget(QLabel("width:"))
        self._w_width_spin = self._make_weight_spin(
            0.35, "width: is the intensity peak broad (bone, 3-5mm) or\n"
                  "narrow (streak artifact, <2mm)? Default 0.35.")
        disc_bar.addWidget(self._w_width_spin)
        disc_bar.addWidget(QLabel("smooth:"))
        self._w_smooth_spin = self._make_weight_spin(
            0.25, "smooth: does brightness change gradually (bone) or\n"
                  "with a sharp edge (artifact)? Default 0.25.")
        disc_bar.addWidget(self._w_smooth_spin)
        disc_bar.addWidget(QLabel("grad:"))
        self._w_gradient_spin = self._make_weight_spin(
            0.25, "grad (gradient): is the slope from the pixel into surrounding\n"
                  "tissue gentle (bone) or steep (artifact)? Default 0.25.")
        disc_bar.addWidget(self._w_gradient_spin)

        # Context bands: what surrounding HU counts as bone vs. soft tissue when
        # deciding what an artifact is corrupting (over-bone / over-tissue split,
        # "Decision 2"). Separate from the bone vote band above; drives both the
        # bright and dark contextual splits.
        disc_bar.addWidget(QLabel("   Context bone HU:"))
        self._ctx_bone_low_spin = self._make_hu_spin(
            -1024, 5000, 500,
            "Neighbors in this HU band count as 'bone' when judging what an\n"
            "artifact overlies. Used by both the bright and dark over-bone/\n"
            "over-tissue splits (not the bone-vs-artifact vote).")
        self._ctx_bone_high_spin = self._make_hu_spin(
            -1024, 5000, 1500, "Upper bound of the context bone band.")
        disc_bar.addWidget(self._ctx_bone_low_spin)
        disc_bar.addWidget(QLabel("–"))
        disc_bar.addWidget(self._ctx_bone_high_spin)

        disc_bar.addWidget(QLabel("   Context tissue HU:"))
        self._ctx_tissue_low_spin = self._make_hu_spin(
            -1024, 5000, -100,
            "Neighbors in this HU band count as 'soft tissue' when judging what\n"
            "an artifact overlies.")
        self._ctx_tissue_high_spin = self._make_hu_spin(
            -1024, 5000, 300, "Upper bound of the context soft-tissue band.")
        disc_bar.addWidget(self._ctx_tissue_low_spin)
        disc_bar.addWidget(QLabel("–"))
        disc_bar.addWidget(self._ctx_tissue_high_spin)

        # Context window: in-plane neighborhood size (px) for the over-bone/
        # over-tissue vote. Larger lets a boundary artifact pixel "see" bone a
        # few mm away and be pulled toward bone.
        disc_bar.addWidget(QLabel("   Context window:"))
        self._ctx_window_spin = QSpinBox()
        self._ctx_window_spin.setRange(3, 25)
        self._ctx_window_spin.setValue(5)
        self._ctx_window_spin.setSingleStep(2)
        self._ctx_window_spin.setSuffix(" px")
        self._ctx_window_spin.setMinimumWidth(72)
        self._ctx_window_spin.setToolTip(
            "In-plane size of the neighborhood the over-bone/over-tissue vote\n"
            "looks at (odd values; default 5). Bigger = an artifact pixel counts\n"
            "bone/tissue farther away, so pixels bordering bone lean bone. Drives\n"
            "both the bright and dark splits; z is fixed at ±1 slice.")
        disc_bar.addWidget(self._ctx_window_spin)

        self._reset_tuning_btn = QPushButton("Reset")
        self._reset_tuning_btn.setToolTip("Restore all discrimination values to defaults.")
        self._reset_tuning_btn.clicked.connect(self._on_reset_tuning)
        disc_bar.addWidget(self._reset_tuning_btn)

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
        self._disc_star_mask = None
        self._seg_masks = None
        self._body_mask = None  # stale for the new patient
        self._hide_bed_check.blockSignals(True)
        self._hide_bed_check.setChecked(False)
        self._hide_bed_check.blockSignals(False)
        self._hide_bed_check.setEnabled(True)
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
        self._disc_star_mask = None  # stale until segmentation reruns
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
    @staticmethod
    def _make_hu_spin(low, high, value, tooltip):
        spin = QSpinBox()
        spin.setRange(low, high)
        spin.setValue(value)
        spin.setSingleStep(50)
        spin.setSuffix(" HU")
        spin.setToolTip(tooltip)
        spin.setMinimumWidth(96)
        return spin

    @staticmethod
    def _make_weight_spin(value, tooltip):
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 2.0)
        spin.setValue(value)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        spin.setToolTip(tooltip)
        spin.setMinimumWidth(72)
        return spin

    def _on_reset_tuning(self):
        for spin, default in (
            (self._dark_low_spin, -1024), (self._dark_high_spin, -150),
            (self._bright_low_spin, 200), (self._bright_high_spin, 2500),
            (self._bone_low_spin, 400), (self._bone_high_spin, 1800),
            (self._w_hu_spin, 0.45), (self._w_width_spin, 0.35),
            (self._w_smooth_spin, 0.25), (self._w_gradient_spin, 0.25),
            (self._ctx_bone_low_spin, 500), (self._ctx_bone_high_spin, 1500),
            (self._ctx_tissue_low_spin, -100), (self._ctx_tissue_high_spin, 300),
            (self._ctx_window_spin, 5),
        ):
            spin.setValue(default)

    def _on_segment_clicked(self):
        if self._volume is None or self._metal_mask is None:
            return

        method = self._seg_method_combo.currentData()
        self._segment_btn.setEnabled(False)

        # Capture the context bands now so the lazily-computed splits match
        # this segmentation even if the spin boxes change afterward.
        self._seg_ctx_bands = (
            self._ctx_bone_low_spin.value(), self._ctx_bone_high_spin.value(),
            self._ctx_tissue_low_spin.value(), self._ctx_tissue_high_spin.value(),
            self._ctx_window_spin.value(),
        )

        self._segment_thread = QThread(self)
        if method == "legacy":
            self.statusBar().showMessage("Segmenting artifacts (legacy HU thresholds)…")
            self._segment_worker = LegacySegmentationWorker(
                self._volume, self._metal_mask, roi_mask=self._roi_mask,
                dark_low=self._dark_low_spin.value(),
                dark_high=self._dark_high_spin.value(),
            )
        else:
            self.statusBar().showMessage("Segmenting artifacts (star profile)…")
            self._segment_worker = SegmentationWorker(
                self._volume, self._spacing, self._metal_mask,
                roi_mask=self._roi_mask,
                dark_low=self._dark_low_spin.value(),
                dark_high=self._dark_high_spin.value(),
                bright_low=self._bright_low_spin.value(),
                bright_high=self._bright_high_spin.value(),
                bone_low=self._bone_low_spin.value(),
                bone_high=self._bone_high_spin.value(),
                w_hu=self._w_hu_spin.value(),
                w_width=self._w_width_spin.value(),
                w_smooth=self._w_smooth_spin.value(),
                w_gradient=self._w_gradient_spin.value(),
                ctx_bone_low=self._ctx_bone_low_spin.value(),
                ctx_bone_high=self._ctx_bone_high_spin.value(),
                ctx_tissue_low=self._ctx_tissue_low_spin.value(),
                ctx_tissue_high=self._ctx_tissue_high_spin.value(),
                ctx_window=self._ctx_window_spin.value(),
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
        self._seg_masks = {
            "dark": dark, "bright": bright, "bone": bone,
            # Contextual over-bone/over-tissue splits. The star-profile worker
            # provides the bright split eagerly; anything left None here is
            # computed lazily (ContextSplitWorker) when its overlay is viewed.
            "bright_bone":   result.get("bright_artifact_bone"),
            "bright_tissue": result.get("bright_artifact_tissue"),
            "dark_bone":     result.get("dark_artifact_bone"),
            "dark_tissue":   result.get("dark_artifact_tissue"),
        }
        # Only the star-profile worker returns this; legacy leaves it None.
        self._disc_star_mask = result.get("disc_star_mask")
        self._refresh_overlays()
        self.statusBar().showMessage(
            f"Segmentation done — dark: {int(dark.sum()):,}  "
            f"bright artifact: {int(bright.sum()):,}  bone: {int(bone.sum()):,}"
        )
        # If a split overlay is already on, compute what it needs now.
        self._ensure_context_splits()

    # ---- lazy over-bone/over-tissue splits ---------------------------------
    def _ensure_context_splits(self, *args) -> None:
        """Compute any over-bone/over-tissue split that is viewed but missing.

        Triggered when a split overlay is toggled on (or a segmentation
        finishes with one already on). Each kind is a no-op unless it's
        actually needed and not already computed or in flight.
        """
        for kind in ("bright", "dark"):
            self._ensure_context_split(kind)

    def _ensure_context_split(self, kind: str) -> None:
        want = (self._overlay_checks[f"{kind}_bone"].isChecked()
                or self._overlay_checks[f"{kind}_tissue"].isChecked())
        if not want:
            return
        if self._seg_masks is None or self._seg_masks.get(kind) is None:
            return
        if self._seg_masks.get(f"{kind}_bone") is not None:
            return  # already computed for this segmentation
        if kind in self._ctx_threads:
            return  # a computation is already in flight

        bands = self._seg_ctx_bands or ()
        # Remember which parent mask this run is for, so a result arriving
        # after a re-segmentation is discarded instead of painted onto the
        # new masks.
        self._ctx_for[kind] = self._seg_masks[kind]
        self.statusBar().showMessage(
            f"Classifying {kind} artifacts (over bone/tissue)…")
        thread = QThread(self)
        worker = ContextSplitWorker(
            self._volume, self._seg_masks[kind], self._metal_mask,
            self._spacing, kind, *bands,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        # Bound methods only: a lambda here would run in the WORKER thread
        # (no receiver QObject to queue onto) and repaint the GUI from it —
        # black viewport + "QBasicTimer" warnings. The worker emits its kind
        # precisely so these can stay plain bound methods.
        worker.finished.connect(self._on_context_split_finished)
        worker.failed.connect(self._on_context_split_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        self._ctx_threads[kind] = thread
        self._ctx_workers[kind] = worker
        thread.start()

    def _on_context_split_finished(self, kind: str, result) -> None:
        self._ctx_threads.pop(kind, None)
        self._ctx_workers.pop(kind, None)
        # Only apply if the masks are still the ones we computed against; a
        # re-segmentation while we were running replaces them.
        if self._seg_masks is not None and \
                self._seg_masks.get(kind) is self._ctx_for.get(kind):
            self._seg_masks[f"{kind}_bone"] = result[f"{kind}_artifact_bone"]
            self._seg_masks[f"{kind}_tissue"] = result[f"{kind}_artifact_tissue"]
            self._refresh_overlays()
            self.statusBar().showMessage(f"{kind.capitalize()} artifact split ready.")
        self._ctx_for.pop(kind, None)
        # A segmentation may have superseded this run; recompute if still wanted.
        self._ensure_context_split(kind)

    def _on_context_split_failed(self, kind: str, message) -> None:
        self._ctx_threads.pop(kind, None)
        self._ctx_workers.pop(kind, None)
        self._ctx_for.pop(kind, None)
        self.statusBar().showMessage(
            f"{kind.capitalize()} split: {message.splitlines()[0]}")

    # ---- hide bed ----------------------------------------------------------
    def _on_hide_bed_toggled(self, checked: bool) -> None:
        if not checked:
            self._view.set_body_mask(None)
            return
        if self._body_mask is not None:
            self._view.set_body_mask(self._body_mask)
            return
        if self._volume is None:
            return
        # First use for this patient: compute the mask on a worker thread.
        self._hide_bed_check.setEnabled(False)
        self.statusBar().showMessage("Computing body mask…")

        self._bed_thread = QThread(self)
        self._bed_worker = BodyMaskWorker(self._volume)
        self._bed_worker.moveToThread(self._bed_thread)
        self._bed_thread.started.connect(self._bed_worker.run)
        self._bed_worker.finished.connect(self._on_body_mask_ready)
        self._bed_worker.failed.connect(self._on_body_mask_failed)
        self._bed_worker.finished.connect(self._bed_thread.quit)
        self._bed_worker.failed.connect(self._bed_thread.quit)
        self._bed_thread.start()

    def _on_body_mask_ready(self, mask) -> None:
        self._body_mask = mask
        self._hide_bed_check.setEnabled(True)
        # Only apply if the box is still checked (user may have changed
        # their mind while the mask was computing).
        if self._hide_bed_check.isChecked():
            self._view.set_body_mask(mask)
        self.statusBar().showMessage("Body mask ready — bed hidden.")

    def _on_body_mask_failed(self, message) -> None:
        self._hide_bed_check.setEnabled(True)
        self._hide_bed_check.blockSignals(True)
        self._hide_bed_check.setChecked(False)
        self._hide_bed_check.blockSignals(False)
        self.statusBar().showMessage(f"Body mask: {message.splitlines()[0]}")

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
        # Trim the black air surround so figures don't need manual cropping.
        image = _autocrop_to_content(image)
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

    def _on_export_legend_clicked(self):
        """Save a standalone color-key PNG of the tissue/artifact classes.

        Excludes the diagnostic overlays (ROI, metal stars, disc stars). Only
        the classes currently toggled on are included, so the exported key
        matches the figure you exported from Export Slice.
        """
        entries = [
            (color, name) for key, color, name in self._legend_entries
            if key not in self._legend_export_exclude
            and self._overlay_checks[key].isChecked()
        ]
        if not entries:
            self.statusBar().showMessage(
                "Export legend: no class overlays are toggled on.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export legend", "legend.png", "PNG image (*.png)")
        if not path:
            return

        # Render at 4x for crisp poster print, matching Export Slice.
        scale = 4
        pad, sw, gap, row_h = 12 * scale, 20 * scale, 10 * scale, 28 * scale
        font = QFont()
        font.setPointSize(11 * scale)
        # Measure the widest label so the canvas fits the text.
        probe = QPixmap(1, 1)
        p = QPainter(probe)
        p.setFont(font)
        text_w = max(p.fontMetrics().horizontalAdvance(name) for _, name in entries)
        p.end()

        width = pad + sw + gap + text_w + pad
        height = pad * 2 + row_h * len(entries)
        canvas = QPixmap(width, height)
        canvas.fill(QColor("white"))

        painter = QPainter(canvas)
        painter.setFont(font)
        border = QColor("#555555")
        text_color = QColor("black")
        for i, (color, name) in enumerate(entries):
            y = pad + i * row_h
            painter.setPen(border)
            painter.setBrush(QColor(color))
            painter.drawRect(pad, y + (row_h - sw) // 2, sw, sw)
            painter.setPen(text_color)
            painter.drawText(pad + sw + gap, y, text_w, row_h,
                             int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                             name)
        painter.end()

        if canvas.save(path):
            self.statusBar().showMessage(
                f"Exported legend ({len(entries)} classes) to {path}")
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
        # blockSignals suppressed the split toggles' lazy-compute hook; run it
        # once here so "All" can still trigger a missing split.
        self._ensure_context_splits()

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
            # Contextual splits sit on top of their parent masks so they stay
            # visible when both are toggled on.
            for key, color in (
                ("bright_bone",   (255, 128,   0)),   # orange
                ("bright_tissue", (255, 105, 180)),   # hot pink
                ("dark_bone",     (138,  43, 226)),   # blue-violet
                ("dark_tissue",   (  0, 206, 209)),   # teal
            ):
                mask = self._seg_masks.get(key)
                if mask is not None and show[key]:
                    overlays.append((mask, color, 0.75))
        if self._roi_outline is not None and show["roi"]:
            overlays.append((self._roi_outline, (50, 255, 50), 0.9))  # lime
        # Last so the thin lines stay visible on top of the other masks.
        if self._star_mask is not None and show["star"]:
            overlays.append((self._star_mask, (0, 255, 255), 0.9))    # cyan
        if self._disc_star_mask is not None and show["disc_star"]:
            overlays.append((self._disc_star_mask, (255, 255, 255), 0.9))  # white
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
