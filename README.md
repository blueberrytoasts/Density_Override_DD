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
6. **Export RTSTRUCT…**: save the structures currently toggled on as a DICOM
   RT Structure Set that references the loaded CT, ready to import into MIM
   or a planning system. ROI names: `Metal`, `Dark_Artifact`,
   `Bright_Artifact`, `Bone`, `Bright_Over_Bone`, `Bright_Over_Tissue`,
   `Dark_Over_Bone`, `Dark_Over_Tissue`. Empty structures are skipped.

## Comparing against reference contours

Export the reference contours (e.g. physician-approved, from MIM) as an
RTSTRUCT on the same CT, then:

```bash
# list structure names in both files
python tools/compare_contours.py --ct "data/HIP3 Patient" --test RS_app.dcm --ref RS_mim.dcm --list

# compare; same-named structures pair automatically, others via --match TEST=REF
python tools/compare_contours.py --ct "data/HIP3 Patient" --test RS_app.dcm --ref RS_mim.dcm \
    --match "Bright_Over_Bone=MD_BrightArt_Bone" --out output/compare_hip3.csv
```

Reports Dice, Jaccard, volumes (cc), volume difference, HD95, max Hausdorff
and mean surface distance (mm), sensitivity and precision per structure pair.
Both files are rasterized onto the CT grid the same way, so differences
reflect the contours themselves.

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
│   ├── rtstruct_worker.py   # Threaded RTSTRUCT export
│   └── bootstrap.py         # Puts algorithms/ on sys.path
├── algorithms/              # UI-independent algorithm code
│   ├── dicom_utils.py       # DICOM loading, HU conversion
│   ├── rtstruct.py          # RTSTRUCT export/import + voxel<->patient geometry
│   ├── contour_metrics.py   # Dice, HD95, surface distance, volumes
│   ├── contour_operations.py# Mask operations, Russian Doll segmentation
│   ├── body_mask.py         # Body / air masking
│   └── core/
│       ├── metal_detection.py  # 3D adaptive + star-profile detection
│       ├── discrimination.py   # Bone vs artifact discrimination
│       └── gpu_utils.py        # Optional CuPy acceleration
├── tools/                   # compare_contours.py, figure/debug/poster scripts
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
python tests/test_rtstruct_roundtrip.py  # synthetic CT, no patient data needed
python tests/test_star_profiles.py   # synthetic phantom, no patient data needed
python tests/test_segmentation.py    # needs local data/
python tests/test_real_data.py       # needs local data/
```

## License

For research and educational use in medical imaging.
