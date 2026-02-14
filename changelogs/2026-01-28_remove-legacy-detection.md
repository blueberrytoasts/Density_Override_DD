# Remove Legacy Metal Detection Method

**Date:** 2026-01-28
**Branch:** `star_profile_upgrade`
**Main Idea:** Removed legacy fixed-threshold metal detection in favor of adaptive star profile algorithm for better generalization across different prosthesis metals

---

## Why This Change

Per advisor feedback:
- Fixed 2500 HU threshold doesn't generalize well across different metal types (titanium, stainless steel, cobalt-chrome, tantalum)
- Adaptive star profile algorithm automatically finds the appropriate threshold based on actual peak intensity
- Different scanners have different calibrations, so adaptive is more robust

---

## Changes Made

### 1. Removed Legacy Detection Method from UI
**File:** `app/main.py`

- Removed radio button choosing between "3D Adaptive + Star Algorithm" and "Legacy with Initial Threshold"
- Now only 3D Adaptive method is available
- Simplified UI header from "Advanced Metal Detection" to "3D Adaptive Metal Detection"
- Removed legacy-specific sliders (ROI Margin, Initial Metal Threshold, Metal Region Connection)

### 2. Cleaned Up Star Profile Checkbox Label
**File:** `app/main.py`

```python
# Before
"Enable Per-Slice Star Profile Analysis (Recovered Algorithm)"

# After
"Enable Per-Slice Star Profile Analysis"
```

### 3. Removed Legacy Detection Code
**File:** `app/core/metal_detection.py`

- Removed `MetalDetectionMethod.LEGACY` enum value
- Removed `_detect_legacy()` method (~115 lines)
- Removed `detect_metal_legacy()` convenience function
- Removed `detect_metal_multi_component()` (referenced non-existent method)

### 4. Removed Legacy Wrapper from main.py
**File:** `app/main.py`

- Removed `detect_metal_volume()` wrapper function
- Removed conditional branches checking `detection_method` variable

---

## Code Retrieval

If legacy code is needed in the future, retrieve from git:

```bash
# View the legacy detection method
git show f0052b1:app/core/metal_detection.py | grep -A 120 "_detect_legacy"

# View the full file at that commit
git show f0052b1:app/core/metal_detection.py

# Restore entire file from that commit (if needed)
git checkout f0052b1 -- app/core/metal_detection.py
```

---

## Metal Types and Expected HU Ranges

For reference, why adaptive is important:

| Metal | Typical HU Range |
|-------|------------------|
| Titanium | 3000-4000 HU |
| Stainless Steel | 5000-8000 HU |
| Cobalt-Chrome | 4000-6000 HU |
| Tantalum | >10000 HU |

Star profiles adapt to whatever peak is present, rather than assuming 2500 HU.

---

## Testing

- Load any patient with hip implant
- Click "Detect Metal Automatically"
- Verify adaptive threshold is calculated and displayed
- Star profiles should refine threshold per-slice
