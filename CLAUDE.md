# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## CURRENT WORK STATUS

_Last reviewed: 2026-10-11. `feature/pyside-app` was fast-forward merged into `main` and the Streamlit UI was removed (last Streamlit version: tag `streamlit-final`). PySide6 is the only front-end. The algorithm folder `app/` was then renamed to `algorithms/` (status entries below use the new paths). Entries below dated 2026-07 were written on `feature/pyside-app`._

### Bug Review 2026-10-11 — OPEN ITEMS (fix before trusting MIM comparisons)

Found by a four-part code review; each item was confirmed on synthetic data unless marked *plausible*. Line numbers are as of commit `6b89c3b` and will drift. Suggested order: 1, 3, 4, 5, 10, then 2 (2 is partly a design decision — discuss before changing).

1. **Star rays point the wrong way** — `discrimination.py:_get_star_profiles` (~709-714) clamps `end_y`/`end_x` to the image *separately*, which changes the ray direction. With 16 angles, 22.5°/45°/67.5° all become 45°; only ~8 distinct directions survive, and off-centre stars skew further. Pixels are judged on anatomy up to ~3 cm away. `build_discriminator_star_overlay` draws ideal rays, which hides this. *Fix:* shorten each ray to its first border hit (scale the radius) instead of clamping each axis.
2. **Shape features don't separate streaks from bone** — `discrimination.py` (~724-726, 778, 856-877).
   - Width: profiles are smoothed (sigma=2) before FWHM, so a 1-px streak measures ~4.9 mm and votes bone; the "<2 mm artifact" branch can't fire.
   - Gradient: `abs(mean(grad_window))` ≈ 0 for any centred peak, so it always votes bone.
   - Smoothness: `1/(1+var/100)` ≈ 0.01-0.05, so it always votes artifact.
   - Net effect: only HU discriminates. Synthetic: a 2000 HU radial streak over 50 HU tissue → 648/720 voxels "bone".
   - Also: width and gradient are per *sample*, not per mm (diagonal rays under-report width by up to 41%).
3. **Over-bone/over-tissue vote counts the artifact's own voxels** — `discrimination.py:_analyze_neighborhood_context` (~624-632) and `contour_operations.py:analyze_local_neighborhood` (~59-66). Bright artifact HU (500-1500) falls in the context bone band, so a streak over pure muscle labels itself "over bone" (synthetic: 810/900). That gives it the wrong density override. *Fix:* exclude the parent artifact mask and metal from the counts and the denominator. The two split implementations also use different rules (raw ratio + 1.5 cm tie-break vs 5/3/1 bins + >30 mm bone bonus + 15 mm tie-break); unify them.
4. **Body mask drops tissue** — `body_mask.py:create_body_mask`.
   - (a) Keeps only the largest component per slice (~35-42) and in 3D (~52-58), so a separate thigh is dropped entirely: no segmentation there, and "Hide bed" blanks it.
   - (b) 3D erosion with the default border treats outside-the-volume as background, so the first/last slice is always empty and the next ones shrink.
   - (c) Dark streaks below −300 HU that reach the skin open a channel to outside air, so the worst dark artifacts are cut out of the body (*plausible clinically*).
   - *Fix:* keep all large components, pad in z before morphology, reconsider the air threshold / use 3D hole filling.
5. **Metal detection misses implants** — `metal_detection.py`.
   - (a) ~82-88: `initial_threshold = percentile(99.5)` then a strict `>`. If >0.5% of voxels sit at the clip value (3071 HU), the mask is empty → "No metal implant detected". Likely with bilateral implants, few slices or a small FOV.
   - (b) ~496-497, 174-177: a star line only counts if its peak > 75% of the *slice* max, so on extended-HU scans the dimmer of two implants gets no threshold. The brighter implant's threshold then replaces the slice mask and the dimmer implant (and its ROI box) disappears.
   - (c) ~176: the per-slice threshold is applied to the whole slice. At FW% 50 (UI minimum), dense cortical bone on the other femur becomes "metal" and gets its own ROI box.
   - (d) ~82-85, 122: only slices above the volume-wide percentile are refined, so a stem tip slice at 2900 HU can be dropped.
10. **Distance map uses mean spacing** — `contour_operations.py:calculate_metal_distance_map` (~20-24) multiplies voxel distance by `mean(spacing)`. With 0.8/0.8/3.0 mm spacing, 8 mm in-plane reads as 15.3 mm. This flips the dark split's 15 mm tie-break and >30 mm bone bonus. *Fix:* `distance_transform_edt(~metal, sampling=_sampling_zyx(spacing))`.

Smaller:
- Nearest-ray lookup floors instead of rounding (`discrimination.py` ~523).
- Small (<10 px) implant fragments are judged with the other implant's star (~667-668, *plausible*).
- A 1-px ray crashes `np.gradient` (~730, unlikely in practice).
- `meta['spacing']` z comes from SliceThickness, not the real slice step (`dicom_utils.py`).
- An empty SliceThickness (Type 2) crashes the load.
- `rasterize_structure` XORs same-slice contours, so *overlapping islands* from other software become holes.
- `compare_contours.py` keys structures by name, so duplicate ROI names overwrite each other.
- metal_detection comments contradict the code (noise cut-off 10 vs 4 px; "average" vs minimum threshold).
- ~900 lines of `contour_operations.py` are unreachable from the app (Russian-doll / sequential / refine paths); its `refine_mask` step breaks mutual exclusivity if revived.

**FIXED 2026-10-11** (items 6-9):
- **6. Stale results / wrong-patient export** — `main_window.py`: `_busy` set + `_refresh_actions()` lock Load/Detect/Segment/Export RTSTRUCT while any job runs; detect/segment/body-mask results are identity-checked against `_detect_for`/`_segment_for`/`_bed_for` and dropped if stale.
- **7. Long contours corrupted the RTSTRUCT** — now written as Implicit VR Little Endian (32-bit lengths). Test: `test_long_contour_roundtrip` (>3000-point contour).
- **8. Loader merged every CT series in a folder** — `load_dicom_series_to_hu(dir, series_instance_uid=None)` loads one series (the requested one, else the largest), drops duplicate positions, and reports `other_series_ignored` / `duplicate_slices_dropped` (shown in the PySide status bar). `compare_contours.py` loads the series the `--test` RTSTRUCT references (`rtstruct.referenced_series_uid`).
- **9. Closing mid-job aborted the process** — `closeEvent` offers to close automatically once running threads finish.
- Verified with the synthetic round-trip suite (8 tests) and a headless GUI test of every race (detect→load, segment→re-detect, body mask→load, close mid-job).

### HU Range Sliders — RESOLVED (now weighted, was reported broken)

The old note here claimed the `bone_low`/`bone_high` sliders had no effect. That is **no longer true** as of the `star_profile_upgrade` work now on `main`. HU is integrated as a weighted feature (Option B):
- `pyside_app/segment_worker.py:SegmentationWorker` passes the Bone HU band from the Discrimination toolbar into discrimination.
- `discrimination.py:_analyze_profile_characteristics` (~704-729) scores HU with weight **±0.45** — the single largest weight (peak width ±0.35, smoothness ±0.25, gradient ±0.25). Final decision: `bone_score > 0.0` (`_classify_from_profile`, ~750).

**Remaining design question (not a bug):** HU is a weighted vote, not a hard gate. A pixel inside the HU range can still be classified as artifact if the shape features outvote it. If a strict cutoff is desired, switch to **Option A**: `bone = (bone_score > 0) AND (bone_low <= HU <= bone_high)`.

### Per-Component Metal ROIs — FIXED (2026-07-09, was Priority: HIGH)

The old bug: `metal_detection.py` found separate connected metal components per slice, then overwrote `individual_regions` with a single averaged "conservative" box — for bilateral implants (HIP3) the averaged centroid landed *between* the hips and the box spanned both.

**Fix (on `feature/pyside-app`):**
- `algorithms/core/metal_detection.py`: per-component region building extracted into `_build_individual_regions()`; both overwrite blocks removed; regions rebuilt from the hole-filled mask. Works for any number of implants. `roi_bounds` remains as overall display/extent only.
- New result key `roi_mask`: 3D bool union of per-component boxes (+5 voxel conservative margin) via `_regions_to_roi_mask()`. Use this to constrain segmentation, not `roi_bounds`.
- `algorithms/contour_operations.py`: `create_russian_doll_segmentation(..., roi_mask=None)` — takes precedence over `roi_bounds` box for constraining.
- `pyside_app/segment_worker.py` + `main_window.py`: both workers accept `roi_mask`; wired from the detection result (previously PySide segmentation had no ROI constraint at all).

Verified with a synthetic bilateral phantom (2 regions per slice, centers on each implant, gap between implants excluded from ROI mask) and a unilateral phantom (1 region, unchanged behavior).

### Per-Implant Discriminator Stars — FIXED (2026-07-15)

The star-profile discriminator had the same averaged-centroid flaw as the old ROI bug: it shot one star per slice from the mean of *all* metal pixels, which for bilateral implants lands between the hips — half the rays sampled empty midline tissue and pixels were judged against profiles from the wrong place.

**Fix (on `feature/pyside-app`):**
- `algorithms/core/discrimination.py`: new `_get_slice_stars()` — one star per metal implant per slice. Components within 10 px (dilation-merge) are grouped so a fragmented implant (cup + screws) gets ONE star; groups <10 metal px skipped as noise; falls back to overall centroid if all tiny. `_discriminate_star_profile` assigns each bright pixel to its nearest star's rays.
- Used by `pyside_app/segment_worker.py` via `DiscriminationMethod.STAR_PROFILE`.
- Debug/poster tool: `tools/visualize_discriminator.py "data/HIP3 Patient" [--slice N] [--angles N]` renders the exact stars + per-ray profiles + per-pixel verdicts (blue=bone, yellow=artifact). Verified: HIP3 slice 161 → 2 stars (one per hip); HIP1 slice 90 → 1 star.
- Expect classification shifts on bilateral patients (HIP3 slice 161: bone 1000→618, artifact 543→925 of 1543 candidates) — rays now start inside metal so near-metal streaks are judged on real local geometry.

### Discrimination Tuning Controls in PySide — ADDED (2026-07-15)

The star-profile discriminator's HU ranges and the four feature weights were hardcoded. Now exposed on a second PySide toolbar ("Discrimination") for poster experimentation:
- **Dark HU range** (plain threshold, applies to both segmentation methods), **Bright HU range** (gate), and **Bone HU range** (vote band) as HU spin boxes.
- **Weights** `w_hu / w_width / w_smooth / w_gradient` (defaults 0.45/0.35/0.25/0.25) as 0–2 spin boxes, plus a **Reset** button.
- Wiring: `discrimination.py:_discriminate_star_profile` + `_analyze_profile_characteristics` take `w_hu/w_width/w_smooth/w_gradient`; confidence is normalized by total weight so it stays in [0,1]. `segment_worker.py:SegmentationWorker` accepts `dark_low/high`, `bright_low/high`, `bone_low/high`, and the four weights; `LegacySegmentationWorker` accepts `dark_low/high`; `main_window.py` passes the spin-box values on Segment. All apply on the next Segment run (no re-detect needed).
- **Behavior-preserving at defaults:** verified HIP3 slice 161 still gives bone=618, artifact=925 with default weights. Doc: `docs/ALGORITHM.md`.

### Contextual Artifact Sub-typing in PySide — ADDED (2026-07-15)

The discriminator answers "is this bright pixel bone or artifact." The *second* question — for each pixel called *artifact*, what tissue is it corrupting underneath (for density override) — was originally Streamlit-only and is now in PySide:
- **Bright split** comes free from the discriminator (`disc["artifact_bone_mask"]` / `artifact_tissue_mask`) — the worker previously discarded these two keys.
- **Dark split** via `classify_dark_artifacts_contextually` (`algorithms/contour_operations.py`): 7×7×3 neighborhood bone-HU vs tissue-HU vote, distance-from-metal tie-break. Per-voxel loop, so it runs **lazily** (`segment_worker.py:DarkContextWorker`) the first time a dark-split overlay is toggled on, not on every Segment; cached until re-segment (`main_window.py:_ensure_dark_split`, identity-guarded so a stale result can't paint onto a newer segmentation). Disc-stars overlay recolored orange→**white** to disambiguate from the bright→bone orange.
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

### RTSTRUCT Export/Import + Contour Comparison — ADDED (2026-10-11)

For comparing app contours with physician-approved MIM contours. Replaced the old `dicom_export.py`, which (a) never added the image origin, (b) used `spacing[2]` (slice spacing) for x because loader spacing is (x, y, z), and (c) made up new Frame of Reference / series / SOP UIDs because the loader discarded them — so MIM would have misplaced or orphaned every contour.
- `algorithms/dicom_utils.py:load_dicom_series_to_hu` now also returns per-slice `image_position_patient`, `image_orientation_patient`, `pixel_spacing_rc`, `sop_instance_uids`, `sop_class_uid`, `series_instance_uid`, `frame_of_reference_uid`, and `patient_study` (patient/study attributes copied into the RTSTRUCT so it files under the same patient/study; in memory only, never log it). Old keys unchanged. `resolve_ct_dir` moved here from `pyside_app/dicom_loader.py`.
- `algorithms/rtstruct.py`: `CTGeometry` (DICOM plane equation, any orientation), `create_rtstruct`/`save_rtstruct` (padded `find_contours` at 0.5 → outline on voxel boundaries; holes = separate contours), `read_rtstruct`, `rasterize_structure` (voxel centre in polygon; same-slice contours XOR → holes).
- `algorithms/contour_metrics.py:compare_masks` — Dice, Jaccard, cc volumes, HD95 (max of directed 95th percentiles), max HD, MSD, sensitivity, precision; anisotropic spacing.
- `tools/compare_contours.py` — CLI: `--ct`, `--test`, `--ref`, `--match TEST=REF`, `--list`, `--out csv`; warns on Frame-of-Reference mismatch or contours off the CT grid.
- PySide: **Export RTSTRUCT…** button (enabled after Detect Metal) exports the toggled-on classes via `pyside_app/rtstruct_worker.py`; names in `main_window.py:_rtstruct_names`, colors from `_legend_entries`.
- `discrimination.py`: three distance transforms passed (x, y, z) spacing to (z, y, x) arrays — now `_sampling_zyx(spacing)`. Affects the star-profile over-bone/over-tissue **tie-break** (distance < 1.5 cm) and the distance-based method; expect small count shifts vs earlier runs. Docstrings across `algorithms/` corrected to say spacing is (x, y, z).
- **Verified:** `tests/test_rtstruct_roundtrip.py` (synthetic CT; HFS/HFP/FFS; hand-written expected coordinates; exact voxel round trip; mutation-checked to catch missing-origin and swapped-spacing bugs) and a headless GUI run on a phantom (load → detect → segment → export: 7 structures round-tripped with 0 voxel differences, toggled-off classes excluded). **Not yet verified:** import into MIM on a real patient.

### What's Working
- PySide6 viewer (`pyside_app/slice_view.py`): wheel/arrow slice scrubbing, Ctrl+wheel zoom, pan, right-drag W/L, W/L presets, HU under cursor
- All heavy work on `QThread` workers (load, metal detection, segmentation, body mask, lazy context splits)
- Full FW75% star profile metal detection, per-slice adaptive thresholding, adaptive metal filter (75% of slice max HU)
- Per-component ROIs (`roi_mask`) and per-implant discriminator stars — bilateral implants handled
- Star-profile and Legacy segmentation, with bright/dark over-bone/over-tissue splits for both
- Tunable discrimination bands, weights and context window from the Discrimination toolbar
- Export Slice (auto-cropped, 4×) and Export Legend for poster figures
- Export RTSTRUCT (references the loaded CT) and `tools/compare_contours.py` for comparison against reference contours

---

## PHI — Read First

This repo's `data/`, `output/` and `tests/test_output/` hold patient CT data and derived masks. They are gitignored and **must never be committed, read into an AI session, or pushed**. Earlier history on `origin` still contains them — do not push until that history has been rewritten with sign-off from Ara Alexandrian / Jason Stephens. Put non-patient figures in `docs/images/`, never CT screenshots.

## Project Overview

Medical imaging research project characterizing metal artifacts in CT scans of patients with hip implants, for density override. It began as Jupyter notebooks (`archive/`), became a Streamlit app (tag `streamlit-final`), and is now a PySide6 desktop app.

## Project Structure

```
├── pyside_app/                # Desktop UI (Qt) — python -m pyside_app
│   ├── main_window.py         # Window, toolbars, overlay toggles, workflow wiring
│   ├── slice_view.py          # QGraphicsView viewer (zoom/pan/W-L/scroll)
│   ├── hu_render.py           # HU -> QImage hot path (pure NumPy)
│   ├── dicom_loader.py        # DicomLoadWorker
│   ├── metal_worker.py        # MetalDetectionWorker
│   ├── segment_worker.py      # SegmentationWorker, LegacySegmentationWorker, ContextSplitWorker
│   ├── body_mask_worker.py    # Body mask (couch removal)
│   ├── rtstruct_worker.py     # RtstructExportWorker
│   └── bootstrap.py           # Puts algorithms/ on sys.path (flat imports)
├── algorithms/                # UI-independent algorithm code (formerly app/)
│   ├── dicom_utils.py         # DICOM I/O, HU conversion
│   ├── rtstruct.py            # CTGeometry, RTSTRUCT export/import, rasterization
│   ├── contour_metrics.py     # compare_masks: Dice, HD95, MSD, volumes
│   ├── contour_operations.py  # Mask ops, Russian Doll, contextual artifact splits
│   ├── body_mask.py           # Body / air masking
│   └── core/
│       ├── metal_detection.py # 3D adaptive + star-profile detection
│       ├── discrimination.py  # Bone vs artifact discrimination
│       └── gpu_utils.py       # Optional CuPy
├── tools/                     # compare_contours.py + figure/debug/poster scripts (add algorithms/ to sys.path themselves)
├── tests/                     # test_rtstruct_roundtrip.py + test_star_profiles.py (synthetic, safe anywhere); test_segmentation/test_real_data need local data/
├── docs/                      # ALGORITHM.md, algorithms/, changelogs/
├── archive/                   # Original notebooks — keep (project origins)
├── data/, output/             # Local only, gitignored (PHI)
├── requirements.txt / environment.yml
└── README.md
```

## Key Architecture & Concepts

### Core Analysis Pipeline
1. **DICOM Loading**: Read CT series, convert to HU (worker thread)
2. **Metal Detection**: 3D adaptive + FW% star profiles; per-component ROIs → `roi_mask`
3. **Artifact Segmentation**: Star-profile discrimination (Russian Doll) or Legacy HU thresholds, constrained by `roi_mask`
4. **Context splits (Decision 2)**: bright/dark artifacts split into over-bone / over-tissue
5. **Export**: slice/legend images and RTSTRUCT from the UI (`algorithms/rtstruct.py`); NIFTI helper (`save_all_contours_as_nifti`) exists but isn't in the UI
6. **Compare**: `tools/compare_contours.py` against reference RTSTRUCTs

### Qt conventions
- Workers are plain `QObject`s moved onto a `QThread`; they never touch widgets and report via signals.
- Connect worker signals to bound methods, not lambdas — a lambda slot runs in the worker thread and repaints the GUI from it (see the 2026-07-16 entry above).
- Import `pyside_app.bootstrap` before any `algorithms/` module. Scripts in `tests/` and `tools/` insert `<repo>/algorithms` on `sys.path` themselves.

### Key HU Ranges (defaults; tunable in the Discrimination toolbar)
- Metal: adaptive (75% of slice max HU, refined by FW75% star profiles)
- Bright gate 200–2500; bone vote band 400–1800; context bone 500–1500; context tissue −100–300
- Dark artifacts: −1024 to −150 HU

## Development Commands

```bash
pip install -r requirements.txt     # or: conda env create -f environment.yml
python -m pyside_app                # run from the repo root
```

## Important Libraries

`PySide6` (UI), `numpy`, `scipy`, `scikit-image`, `pydicom`, `nibabel`, `matplotlib`/`Pillow` (figures), optional `cupy`.

## Working with DICOM Data

- Preserve spatial information (origin, spacing, slice positions)
- Apply rescale slope/intercept for accurate HU conversion
- Sort slices by ImagePositionPatient[2]
- **Spacing convention:** `meta['spacing']` is (x, y, z) = (col, row, slice). Arrays are indexed [slice, row, col], so anything taking `sampling=` needs it reversed (`_sampling_zyx`, or `CTGeometry.voxel_spacing_zyx()`).
- Voxel ↔ patient mm: always go through `rtstruct.CTGeometry` (uses per-slice IPP + IOP); never hand-roll `index * spacing`.

## Visualization Standards

Single source: `main_window.py:self._legend_entries`. Metal red `#ff0000`, dark magenta `#ff00ff`, bright yellow `#ffff00`, bone blue `#0033cc`, bright→bone orange `#ff8000`, bright→tissue pink `#ff69b4`, dark→bone violet `#8a2be2`, dark→tissue turquoise `#00ced1`, ROI lime `#32ff32`, metal stars cyan `#00ffff`.

## Key Algorithms

### FW75% Metal Detection
1. Shoot radial lines (configurable, default 32 angles) from detected high-intensity centers
2. Find peaks along each profile
3. Keep only lines whose peak > 75% of slice max HU
4. Threshold per valid line = FW% of its peak
5. Slice threshold = minimum across valid lines

### Russian Doll Segmentation
1. Segment dark artifacts (excluding metal)
2. Discriminate bone from bright artifacts using profile analysis
3. Keep all masks mutually exclusive
4. Morphological refinement

### Profile-Based Discrimination
Weighted vote (defaults HU 0.45, width 0.35, smoothness 0.25, gradient 0.25); bone if score > 0. One star per implant per slice. See `docs/ALGORITHM.md`.
