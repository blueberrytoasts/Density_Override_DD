# Session Work - February 14, 2026

**Branch:** `main` (merged PR #4 from `star_profile_upgrade`)
**Sessions:** 2 (first session compacted, continued in second)

---

## Committed Changes (PR #4 merge + uncommitted work from sessions)

### 1. GPU Acceleration Support (committed as `7f2c1dc`)
**Files:** `app/core/gpu_utils.py` (new), `app/core/discrimination.py`, `requirements.txt`

- Added CuPy GPU utilities for accelerated distance transforms and variance calculations
- New `gpu_utils.py` module with optional CuPy dependency
- Updated requirements.txt with optional GPU dependencies
- Created `changelogs/GPU_SETUP_TODO.md` for future GPU setup

### 2. Artifact Discrimination Improvements (committed as `7f2c1dc`)
**Files:** `app/core/discrimination.py`, `app/contour_operations.py`

- Removed 'mixed' artifact category - now forces bone/tissue classification using distance tie-breaker
- Passed bone HU range parameters through to discrimination algorithms

### 3. UI/Visualization Fixes (committed as `7f2c1dc`)
**Files:** `app/visualization.py`, `app/main.py`

- Added `fig_to_png_bytes()` for high-quality image export
- Disabled scale bar temporarily in visualization
- Fixed Streamlit deprecation: `use_container_width` -> `width='stretch'`

---

## Uncommitted Changes (from today's sessions)

### 4. Per-Component Star Profile Thresholding
**File:** `app/core/metal_detection.py` (lines ~116-170)

Changed star profile threshold calculation from single-centroid to per-component:
- **Before:** Found one center of ALL metal on each slice, shot star profiles from there
- **After:** Uses `scipy.ndimage.label` with 4-connectivity to find connected components per slice, calculates star profile threshold independently for each component, then uses the minimum threshold (most inclusive)
- Skips components smaller than 10 voxels (noise)
- Each component gets its own 200x200 ROI for star profile calculation

### 5. Adaptive Metal Filter in Star Profiles
**File:** `app/core/metal_detection.py` (lines ~501-511, ~552-556)

- Replaced hardcoded `2500 HU` metal filter with adaptive `50% of max HU in slice`
- Generalizes to any metal type (titanium, steel, tantalum, etc.)

### 6. Removed Dead Code
**File:** `app/core/metal_detection.py`

- Removed unused `_calculate_fw75_threshold()` method (superseded by `_calculate_star_threshold()`)
- Removed unused `detect_metal_adaptive_3d()` convenience function

### 7. Sidebar Visibility Rework
**File:** `app/main.py` (lines ~567-620)

- Changed from 2-column to 3-column layout for primary contours (Metal, Bone, Bright Legacy)
- Contextual artifact classifications (Bright/Dark Over Bone/Tissue) now only appear when those masks exist
- Legacy artifact checkboxes only show when those specific masks exist
- Cleaner grouping with "Artifact Classifications" subheader

### 8. Intensity Percentile Slider Tooltip Clarification
**File:** `app/main.py` (lines ~178-183)

- Renamed slider to "Initial Detection Percentile (for centroid/ROI)"
- Updated help text to clarify it only affects centroid/ROI, NOT final threshold

### 9. Removed Unused Visualization Functions
**File:** `app/visualization.py`

- Removed `create_slice_preview()` - unused
- Removed `visualize_edge_analysis()` - unused (~80 lines)

---

## TODO / Known Issues

### Centroid Calculation Needs Rework
**Priority: High**
**File:** `app/core/metal_detection.py` (lines ~299-314, ~361-373)

The conservative ROI override is still active. After detecting individual metal components per slice (with proper per-component centers), the code **overrides everything** with a single conservative ROI:

```python
# Lines 299-314: OVERRIDES per-component work
conservative_region = {
    'center_y': int(np.mean(y_coords_roi)),   # Average of ALL metal
    'center_x': int(np.mean(x_coords_roi))    # Average of ALL metal
}
individual_regions = {}  # Wipes out per-component regions!
for z in valid_z_slices:
    individual_regions[z] = [conservative_region.copy()]  # One big ROI
```

Same override happens again after hole filling (lines ~361-373).

**Problem for HIP3 (bilateral implant):**
```
Left Hip    Centroid    Right Hip
   XX          *           XX
  XXXX    (middle of      XXXX
   XX      nowhere!)       XX
```

- The centroid is placed between the two implants, hitting neither
- Star profiles radiate from this wrong centroid
- ROI size is dictated by the initial mask + this centroid, so it encompasses both hips in one giant box instead of two focused boxes
- Individual component detection code EXISTS (lines 226-279) but its output gets thrown away

**Attempted fix this session:** Removed both overrides to keep per-component regions. **Reverted** at user request to avoid breaking things before testing. The per-component detection infrastructure is in place, just needs the overrides removed and tested.

**What needs to happen:**
1. Remove the two conservative ROI overrides (lines ~299-314 and ~361-373)
2. After hole filling, recompute per-component regions from the filled mask
3. Update star profile visualization in `main.py` to show profiles for ALL components (currently only shows `[0]`)
4. Test with HIP3 bilateral patient to verify two separate ROIs appear
5. Verify unilateral patients (HIP1, HIP2, HIP4) still work correctly with single ROI
