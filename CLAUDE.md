# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## CURRENT WORK STATUS

_Last reviewed: 2026-06-11. `feature/fast-slice-viewer` was fast-forward merged into `main`; `main` is the current tip and is pushed to `origin/main`._

### HU Range Sliders — RESOLVED (now weighted, was reported broken)

The old note here claimed the `bone_low`/`bone_high` sliders had no effect. That is **no longer true** as of the `star_profile_upgrade` work now on `main`. HU is integrated as a weighted feature (Option B):
- `main.py:~943` passes `bone_hu_low=bone_low, bone_hu_high=bone_high` into discrimination.
- `discrimination.py:_analyze_profile_characteristics` (~704-729) scores HU with weight **±0.45** — the single largest weight (peak width ±0.35, smoothness ±0.25, gradient ±0.25). Final decision: `bone_score > 0.0` (`_classify_from_profile`, ~750).

**Remaining design question (not a bug):** HU is a weighted vote, not a hard gate. A pixel inside the HU range can still be classified as artifact if the shape features outvote it. If a strict cutoff is desired, switch to **Option A**: `bone = (bone_score > 0) AND (bone_low <= HU <= bone_high)`.

### Per-Component Metal ROIs — FIXED (2026-07-09, was Priority: HIGH)

The old bug: `metal_detection.py` found separate connected metal components per slice, then overwrote `individual_regions` with a single averaged "conservative" box — for bilateral implants (HIP3) the averaged centroid landed *between* the hips and the box spanned both.

**Fix (on `feature/pyside-app`):**
- `app/core/metal_detection.py`: per-component region building extracted into `_build_individual_regions()`; both overwrite blocks removed; regions rebuilt from the hole-filled mask. Works for any number of implants. `roi_bounds` remains as overall display/extent only.
- New result key `roi_mask`: 3D bool union of per-component boxes (+5 voxel conservative margin) via `_regions_to_roi_mask()`. Use this to constrain segmentation, not `roi_bounds`.
- `app/contour_operations.py`: `create_russian_doll_segmentation(..., roi_mask=None)` — takes precedence over `roi_bounds` box for constraining.
- `app/main.py`: both segmentation paths pass the detection `roi_mask`.
- `pyside_app/segment_worker.py` + `main_window.py`: both workers accept `roi_mask`; wired from the detection result (previously PySide segmentation had no ROI constraint at all).

Verified with a synthetic bilateral phantom (2 regions per slice, centers on each implant, gap between implants excluded from ROI mask) and a unilateral phantom (1 region, unchanged behavior).

### Per-Implant Discriminator Stars — FIXED (2026-07-15)

The star-profile discriminator had the same averaged-centroid flaw as the old ROI bug: it shot one star per slice from the mean of *all* metal pixels, which for bilateral implants lands between the hips — half the rays sampled empty midline tissue and pixels were judged against profiles from the wrong place.

**Fix (on `feature/pyside-app`):**
- `app/core/discrimination.py`: new `_get_slice_stars()` — one star per metal implant per slice. Components within 10 px (dilation-merge) are grouped so a fragmented implant (cup + screws) gets ONE star; groups <10 metal px skipped as noise; falls back to overall centroid if all tiny. `_discriminate_star_profile` assigns each bright pixel to its nearest star's rays.
- Benefits both apps automatically (Streamlit `main.py:944` and `pyside_app/segment_worker.py:137` both use `DiscriminationMethod.STAR_PROFILE`).
- Debug/poster tool: `tools/visualize_discriminator.py "data/HIP3 Patient" [--slice N] [--angles N]` renders the exact stars + per-ray profiles + per-pixel verdicts (blue=bone, yellow=artifact). Verified: HIP3 slice 161 → 2 stars (one per hip); HIP1 slice 90 → 1 star.
- Expect classification shifts on bilateral patients (HIP3 slice 161: bone 1000→618, artifact 543→925 of 1543 candidates) — rays now start inside metal so near-metal streaks are judged on real local geometry.

### Discrimination Tuning Controls in PySide — ADDED (2026-07-15)

The star-profile discriminator's HU ranges and the four feature weights were hardcoded. Now exposed on a second PySide toolbar ("Discrimination") for poster experimentation:
- **Bright HU range** (gate) and **Bone HU range** (vote band) as HU spin boxes.
- **Weights** `w_hu / w_width / w_smooth / w_gradient` (defaults 0.45/0.35/0.25/0.25) as 0–2 spin boxes, plus a **Reset** button.
- Wiring: `discrimination.py:_discriminate_star_profile` + `_analyze_profile_characteristics` take `w_hu/w_width/w_smooth/w_gradient`; confidence is normalized by total weight so it stays in [0,1]. `segment_worker.py:SegmentationWorker` accepts `bright_low/high`, `bone_low/high`, and the four weights; `main_window.py` passes the spin-box values on Segment. All apply on the next Segment run (no re-detect needed).
- **Behavior-preserving at defaults:** verified HIP3 slice 161 still gives bone=618, artifact=925 with default weights. Doc: `docs/ALGORITHM.md`.

### What's Working
- Fast slice viewer: PIL-based rendering (~10ms) via `fast_render_slice()` in `app/visualization.py`
- `@st.fragment` isolated viewer — slice navigation doesn't trigger full page rerun
- Overlay view: matplotlib figure rendered to PNG bytes, displayed via `st.image()` (no legend in figure)
- Static HTML legend + styled analysis panel in right column (col2)
- Auto-switch to Overlays mode when Detect Metal or Segment Artifacts completes
- Full FW75% star profile metal detection (`app/core/metal_detection.py:479-610`)
- Per-slice adaptive thresholding (`app/core/metal_detection.py:216-267`)
- Adaptive metal filter: 75% of max HU in slice (not hardcoded 2500)
- 32-angle configurable star profiles
- ROI bounds and body mask constraints
- Profile-based discrimination (just needs HU integration)

### Scroll Wheel Navigation — Not Working (Parked)
Attempted JS injection to capture wheel events on the image and step the slider. Does not work reliably with Streamlit 1.54 + `@st.fragment` because React destroys/recreates slider DOM on each fragment rerun. The JS injection code has been removed. Arrow keys work natively when the slider has focus.

---

## Project Overview

This is a medical imaging research project focused on characterizing metal artifacts in CT scans of patients with hip implants. The project has been refactored from Jupyter notebooks into a Streamlit web application for better usability and deployment.

## Project Structure

```
├── app/                       # Application source code
│   ├── main.py               # Streamlit web app (fragment viewer, two view modes)
│   ├── dicom_utils.py        # DICOM loading and RTSTRUCT handling
│   ├── dicom_export.py       # DICOM RT Structure export
│   ├── contour_operations.py # Boolean operations and mask refinement
│   ├── visualization.py      # Visualization (fast_render_slice, create_overlay_image)
│   ├── config.py             # Configuration management
│   ├── body_mask.py          # Body masking utilities
│   └── core/                  # Core algorithms
│       ├── metal_detection.py   # Metal detection (3D adaptive + star profiles)
│       └── discrimination.py    # Bone/artifact discrimination
├── algorithm detailed descriptions/  # Algorithm documentation
├── data/                      # Patient DICOM data (HIP* Patient folders)
├── output/                    # Generated masks and exports
├── docs/                      # Changelogs and plans
├── requirements.txt           # Python dependencies (streamlit>=1.37.0)
├── run.sh                     # Launch script
├── CLAUDE.md                  # THIS FILE - read by Claude at session start
└── README.md                  # User documentation
```

## Key Architecture & Concepts

### Core Analysis Pipeline
1. **DICOM Loading**: Read CT series and convert pixel values to Hounsfield Units (HU)
2. **Metal Detection**: Two methods available:
   - Legacy: Initial HU threshold + star profile refinement
   - 3D Adaptive: Multi-planar analysis with automatic thresholding
3. **ROI Creation**: Individual regions per metal component (avoids bilateral capture)
4. **Artifact Segmentation**: Two approaches:
   - Legacy: Simple threshold-based with boolean operations
   - Russian Doll: Smart discrimination using star profile analysis
5. **Export**: NIFTI masks or DICOM RT Structure Sets

### Key HU Ranges (Hounsfield Units)
- Metal: adaptive (75% of slice max HU, refined by FW75% star profiles)
- Bright artifacts/Bone: 300-1500 HU (discriminated by profile analysis)
- Dark artifacts: <-150 HU
- Soft tissue: -100 to 300 HU

### Algorithm Features
- **Star-profile analysis**: 16-point radial sampling for both metal detection and tissue discrimination
- **FW75% Thresholding**: Full Width at 75% Maximum for adaptive metal thresholding
- **Russian Doll Segmentation**: Sequential exclusion approach ensuring no tissue overlap
- **Profile-based Discrimination**: Analyzes peak width, smoothness, and directional variance
- **3D Analysis**: Considers coronal and sagittal projections for complete metal extent
- **GPU Acceleration**: Optional CuPy support for faster profile analysis

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application (production mode)
./run.sh

# Run in development mode
cd app
streamlit run main.py

# Run on custom address/port
streamlit run app/main.py --server.address localhost --server.port 8501
```

## Important Libraries

- `streamlit`: Web application framework
- `pydicom`: DICOM file handling and RT Structure creation
- `numpy`: Array operations and mask manipulation
- `matplotlib`: Visualization and plotting
- `scipy`: Image processing (ndimage, morphology, distance transforms)
- `scikit-image`: Advanced image operations (measure, draw)
- `nibabel`: NIFTI file I/O for mask export
- `cupy` (optional): GPU acceleration for profile analysis

## Code Standards

### Module Organization
- `main.py`: Streamlit UI, `@st.fragment` slice viewer, two view modes, workflow coordination
- `dicom_utils.py`: DICOM I/O operations, HU conversion
- `dicom_export.py`: DICOM RT Structure Set creation
- `contour_operations.py`: Boolean operations, mask refinement, Russian doll segmentation
- `visualization.py`: `fast_render_slice()` (PIL ~10ms), `create_overlay_image()` (matplotlib), `fig_to_png_bytes()`
- `config.py`: Configuration management
- `body_mask.py`: Body masking utilities (air exclusion)
- `core/metal_detection.py`: 3D adaptive metal detection with star profiles
- `core/discrimination.py`: Star profile-based bone vs artifact discrimination

### Error Handling
- Graceful handling of missing DICOM files
- Validation of RTSTRUCT contours
- Fallback to manual ranges if auto-detection fails

### Performance Considerations
- Lazy loading of DICOM data
- Efficient numpy operations for large 3D volumes
- Matplotlib figure cleanup to prevent memory leaks

## Working with DICOM Data

Always handle DICOM metadata carefully:
- Preserve spatial information (origin, spacing, slice positions)
- Apply rescale slope/intercept for accurate HU conversion
- Sort slices by ImagePositionPatient[2] for correct ordering
- Handle both CT and RTSTRUCT DICOM types

## Visualization Standards

The project uses consistent color coding:
- Red (rgba: 1,0,0,0.7): Metal implant
- Yellow (rgba: 1,1,0,0.6): Bright artifacts
- Magenta (rgba: 1,0,1,0.6): Dark artifacts
- Blue (rgba: 0,0.2,0.8,0.5): Bone tissue
- Lime: ROI boundary indicator

## Deployment Notes

The application is configured to run on:
- Address: 192.168.1.11
- Port: 4224
- Use `run.sh` for consistent deployment settings

## Testing Commands

```bash
# Run the Streamlit app
cd app && streamlit run main.py

# Test star profile detection (load patient, enable star profiles, run detection)
# Test discrimination (select "Russian Doll with Star Profile Discrimination")
```

## Key Algorithms

### FW75% Metal Detection
The star profile algorithm automatically determines metal thresholds by:
1. Shooting 16 radial lines from detected high-intensity centers
2. Finding peaks along each profile
3. Filtering: only lines where peak > 75% of slice max HU (adaptive, not hardcoded)
4. Calculating 75% of peak value as the threshold per valid line
5. Using minimum threshold across valid profiles for the slice

### Russian Doll Segmentation
Sequential tissue segmentation with mutual exclusion:
1. Segment dark artifacts (excluding metal)
2. Discriminate bone from bright artifacts using profile analysis
3. Ensure all masks are mutually exclusive
4. Apply morphological refinement

### Profile-Based Discrimination
Distinguishes bone from bright artifacts by analyzing:
- Peak width (bone: broad ~3-5mm, artifacts: narrow <2mm)
- Smoothness (bone: smooth transitions, artifacts: sharp edges)
- Directional consistency (bone: consistent, artifacts: variable)
- Gradient magnitude (bone: gradual, artifacts: steep)