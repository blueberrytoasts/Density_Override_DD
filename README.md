# CT Metal Artifact Characterization

A PySide6 desktop app for detecting metal implants in hip CT scans and
segmenting the artifacts they cause: dark streaks, bright streaks, and
bright streaks split by the tissue they overlay (bone vs soft tissue), for
density override work.

## Installation

### Conda (recommended)
```bash
conda env create -f environment.yml
conda activate IVH
```

### pip
```bash
pip install -r requirements.txt
```

Optional GPU acceleration for profile analysis: install the CuPy wheel that
matches your CUDA version (see the comments in `requirements.txt`).

## Running

From the repository root:
```bash
python -m pyside_app
```

## Workflow

1. **Load Patient…**: pick a CT series folder. DICOM is read on a worker
   thread and converted to Hounsfield Units.
2. **Detect Metal**: 3D adaptive detection with FW% star profiles (the FW%
   is set in the toolbar). The per-slice metal threshold shows in the
   status area.
3. **Segment Artifacts**: star-profile discrimination of dark artifacts,
   bright artifacts and bone. The HU bands, vote weights (HU, peak width,
   smoothness, gradient) and the bone/tissue split of each artifact can all be tuned
   from the discrimination toolbar. **Reset** restores the defaults.
4. **Review**: toggle the overlays, switch window/level presets (Soft Tissue,
   Bone, Lung, Metal), hide the CT couch (body mask) and read the HU under the
   cursor in the status bar.
5. **Export Slice… / Export Legend…**: save the current view (auto-cropped,
   upscaled 4x, PNG/TIFF) or the color legend (PNG) for figures.

### Viewer controls

| Input | Action |
|---|---|
| Mouse wheel / arrow keys | Change slice |
| Ctrl + wheel | Zoom at cursor |
| Left-drag | Pan |
| Right-drag | Window/level |
| `F` | Fit to view |

### Overlay colors

| Color | Meaning |
|---|---|
| Red | Metal |
| Magenta | Dark artifact |
| Yellow | Bright artifact |
| Blue | Bone |
| Orange / Pink | Bright artifact over bone / over tissue |
| Violet / Turquoise | Dark artifact over bone / over tissue |
| Lime | ROI |
| Cyan | Metal-detection star rays |

## Project structure

```
├── pyside_app/              # Desktop UI (Qt)
│   ├── __main__.py          # Entry point: python -m pyside_app
│   ├── main_window.py       # Window, toolbars, workflow wiring
│   ├── slice_view.py        # QGraphicsView slice viewer (zoom/pan/W-L)
│   ├── hu_render.py         # Fast HU -> QImage conversion
│   ├── dicom_loader.py      # Threaded DICOM loading
│   ├── metal_worker.py      # Threaded metal detection
│   ├── segment_worker.py    # Threaded artifact segmentation
│   ├── body_mask_worker.py  # Threaded body mask (couch removal)
│   └── bootstrap.py         # Puts algorithms/ on sys.path
├── algorithms/              # UI-independent algorithm code
│   ├── dicom_utils.py       # DICOM loading, HU conversion
│   ├── dicom_export.py      # RT Structure Set export (not yet in the UI)
│   ├── contour_operations.py# Mask operations, Russian Doll segmentation
│   ├── body_mask.py         # Body / air masking
│   ├── visualization.py     # Matplotlib/PIL rendering helpers (used by tests)
│   └── core/
│       ├── metal_detection.py  # 3D adaptive + star-profile detection
│       ├── discrimination.py   # Bone vs artifact discrimination
│       └── gpu_utils.py        # Optional CuPy acceleration
├── tools/                   # Stand-alone figure/debug scripts (incl. poster figures)
├── tests/                   # Test scripts
├── docs/                    # Algorithm notes and changelogs
├── archive/                 # Original Jupyter notebooks (project origins)
├── requirements.txt
└── environment.yml
```

## Patient data

`data/` and `output/` are where the app reads DICOM and writes results
locally. Both are **gitignored and must never be committed**, along with any
CT screenshots. Put non-patient figures (diagrams, legends) in `docs/images/`.

## Algorithms

See [docs/ALGORITHM.md](docs/ALGORITHM.md) and
[docs/algorithms/metal_detection_star_profiles.md](docs/algorithms/metal_detection_star_profiles.md).

In brief:
- **FW% metal thresholding**: radial star profiles from each metal center.
  Lines whose peak is above 75% of the slice max set the threshold at FW% of
  their peak.
- **Discrimination**: each bright candidate pixel is scored by HU, peak width,
  profile smoothness and gradient. Bone tends to give broad, smooth peaks and
  artifacts narrow, sharp ones.
- **Russian Doll segmentation**: tissue classes are segmented in sequence and
  kept mutually exclusive.

## Tests

```bash
python tests/final_test_all_patients.py
python tests/test_3d_adaptive.py
python tests/test_russian_doll.py
python tests/test_segmentation.py
```
These run against the local `data/` folder.

## License

For research and educational use in medical imaging.
