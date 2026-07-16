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
- **Dark HU range** (plain threshold, applies to both segmentation methods), **Bright HU range** (gate), and **Bone HU range** (vote band) as HU spin boxes.
- **Weights** `w_hu / w_width / w_smooth / w_gradient` (defaults 0.45/0.35/0.25/0.25) as 0–2 spin boxes, plus a **Reset** button.
- Wiring: `discrimination.py:_discriminate_star_profile` + `_analyze_profile_characteristics` take `w_hu/w_width/w_smooth/w_gradient`; confidence is normalized by total weight so it stays in [0,1]. `segment_worker.py:SegmentationWorker` accepts `dark_low/high`, `bright_low/high`, `bone_low/high`, and the four weights; `LegacySegmentationWorker` accepts `dark_low/high`; `main_window.py` passes the spin-box values on Segment. All apply on the next Segment run (no re-detect needed).
- **Behavior-preserving at defaults:** verified HIP3 slice 161 still gives bone=618, artifact=925 with default weights. Doc: `docs/ALGORITHM.md`.

### Contextual Artifact Sub-typing in PySide — ADDED (2026-07-15)

Both apps already answer "is this bright pixel bone or artifact." The *second* question — for each pixel called *artifact*, what tissue is it corrupting underneath (for density override) — existed only in Streamlit. Now in PySide too:
- **Bright split** comes free from the discriminator (`disc["artifact_bone_mask"]` / `artifact_tissue_mask`) — the worker previously discarded these two keys.
- **Dark split** via `classify_dark_artifacts_contextually` (`app/contour_operations.py`): 7×7×3 neighborhood bone-HU vs tissue-HU vote, distance-from-metal tie-break. Per-voxel loop, so it runs **lazily** (`segment_worker.py:DarkContextWorker`) the first time a dark-split overlay is toggled on, not on every Segment; cached until re-segment (`main_window.py:_ensure_dark_split`, identity-guarded so a stale result can't paint onto a newer segmentation). Disc-stars overlay recolored orange→**white** to disambiguate from the bright→bone orange.
- `segment_worker.py:SegmentationWorker` now returns `bright_artifact_bone/tissue` + `dark_artifact_bone/tissue`; `main_window.py` stores them and adds four opt-in overlay toggles (orange=bright→bone, green=bright→tissue, blue-violet=dark→bone, teal=dark→tissue). Star-profile worker only; legacy returns none.
- **Verified** on HIP1: each split partitions its parent exactly (bright 38,745 → bone + tissue; dark 5,303 → bone 309 + tissue 4,994 at the original hardcoded bands), no bone/tissue overlap, correct shape/dtype. (Exact bright bone/tissue counts depend on the context band — see next entry.) Doc: `docs/ALGORITHM.md` §3c.

### Context (Decision-2) HU Ranges in PySide — ADDED (2026-07-15)

The over-bone/over-tissue sub-typing (Decision 2) had hardcoded HU bands, and the two paths disagreed (bright reused the Decision-1 vote band 400–1800; dark hardcoded 500–1500 / −100–300). Now unified and tunable:
- Two new Discrimination-toolbar ranges: **Context bone HU** (default 500–1500) and **Context tissue HU** (default −100–300). Drive *both* the bright and dark splits, so the two paths stay consistent.
- Wiring: `discrimination.py:_discriminate_star_profile` gains `ctx_bone_low/high` (used by `_analyze_neighborhood_context` instead of the vote band; default to the vote band when omitted, so non-PySide callers are unchanged) and now receives `tissue_hu_low/high` from the worker. `segment_worker.py:SegmentationWorker` accepts `ctx_bone_low/high` + `ctx_tissue_low/high` and passes them to both the discriminator and `classify_dark_artifacts_contextually`. `main_window.py` adds the four spin boxes (+Reset).
- **Note:** this slightly changes the *bright* split at defaults — its bone context band moves from the vote band (400–1800) to 500–1500 — in exchange for consistency with the dark split and matching the documented bone band. Decision 1 (bone vs. artifact) is unchanged.
- **Verified** on HIP1: at defaults both splits still partition their parent exactly; widening context bone to 300–3000 pushes more artifact voxels to "over bone" (bright and dark both increase) while partitions stay exact. Doc: `docs/ALGORITHM.md` §3c.

### Context Window Knob in PySide — ADDED (2026-07-16)

Decision-2 sub-typing (over-bone/over-tissue) is a neighborhood vote; the neighborhood *size* was hardcoded (bright 5×5×3, dark 7×7×3). Now a single **Context window** spin box (in-plane px, default 5; z fixed at ±1 slice) drives both splits, unifying them. It's the lever when a green region sits *next to* bone rather than *over* bone-HU (widening lets a boundary pixel tally the adjacent bone). Wiring: `discrimination.py:_discriminate_star_profile` gains `ctx_window` → `_analyze_neighborhood_context(window_size=…)`; `segment_worker.py` `SegmentationWorker` + `DarkContextWorker` accept `ctx_window` (dark maps it to `window_size=(3, n, n)`, replacing the old (7,7,3)); `main_window.py` adds the spin box, captures it in `_seg_ctx_bands` for the lazy dark path, and passes it on Segment. Default 5 preserves the bright split exactly; the dark neighborhood changes (7×7×3 → 3×5×5, now matching bright). Measured caveat (HIP4/123): the window does *not* rescue a green region that sits on the tissue side of the bone ring — bigger window pulls in *more* surrounding soft tissue (bright→bone 321→294→238 as window 5→11→17), so it moves toward tissue. That region genuinely sits over ~260 HU tissue; no Decision-2 knob (bands or window) flips it to bone without mislabeling real tissue.

### Contextual Splits for Legacy Method — ADDED (2026-07-16)

The over-bone/over-tissue splits (Decision 2) previously existed only for the star-profile method; after a Legacy segmentation the four split toggles silently showed nothing. Now both methods support all four splits:
- `segment_worker.py`: `DarkContextWorker` generalized to `ContextSplitWorker(volume, parent_mask, metal_mask, spacing, kind, …)` — `kind` is `"bright"` or `"dark"`; calls `classify_artifacts_contextually` directly and emits `finished(kind, result)` / `failed(kind, message)`. The kind is IN the signal so the window connects plain bound methods — connecting a lambda to capture the kind runs the slot in the worker thread and repaints the GUI from it (black viewport + `QBasicTimer` warnings; this bug shipped briefly on 2026-07-16).
- `main_window.py`: lazy-split state is per-kind dicts (`_ctx_threads/_ctx_workers/_ctx_for`); `_ensure_dark_split` → `_ensure_context_splits`/`_ensure_context_split(kind)`. All four split toggles trigger it; the star path is unaffected (bright split arrives eagerly from the discriminator, so its ensure is a no-op). The "All" checkbox now also triggers missing splits (previously suppressed by blockSignals). Tooltips no longer claim "Star Profile method only".
- **Consistency note:** legacy's bright split uses the banded 5-3-1 contextual score (same as dark), not the discriminator's simple majority vote — the two agree except in mixed neighborhoods.
- Verified offscreen with a synthetic phantom: legacy segment → toggle splits → both lazy workers run, each split partitions its parent exactly (bright 1040 → 8+1032, dark 160 → 8+152), overlays render, identity guard still discards stale results. Doc: `docs/ALGORITHM.md` §3c.

### Export Legend Button in PySide — ADDED (2026-07-16)

`main_window.py:_on_export_legend_clicked` renders a standalone color-key PNG (4× for print) of the mask classes, for placing beside poster figures. Includes only the classes currently toggled on (so the key matches an Export Slice image); the diagnostic overlays (`roi`, `star`, `disc_star`) are always excluded via `_legend_export_exclude`. Legend key→color→label pulled from the single-source `self._legend_entries` tuple (also drives the on-screen toggles). Refuses with a status message if no class overlay is on. Verified offscreen: correct classes/order/exclusions, height scales with class count, empty and diagnostics-only selections write nothing.

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