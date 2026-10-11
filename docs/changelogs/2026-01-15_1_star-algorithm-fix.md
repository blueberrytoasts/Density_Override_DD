# Star Profile Algorithm Fix

**Date:** 2026-01-15
**Branch:** `metal-star-algorithm-fix`
**Main Idea:** Fixed star profile threshold calculation to properly detect metal by filtering out non-metal peaks

---

## Changes Made

### 1. Filter non-metal peaks from threshold calculation
**File:** `app/core/metal_detection.py` (lines 637-642)

**Problem:** The star profile algorithm was including ALL 16 lines in the threshold calculation, even lines that hit bone/tissue instead of metal. This dragged the threshold down to 500-600 HU.

**Fix:** Only include lines where peak HU > 2500 (actually hit metal):
```python
# Before: included all lines
peak_hu = np.max(hu_values)
profile_threshold = peak_hu * (fw_percentage / 100.0)
thresholds.append(profile_threshold)

# After: only include if line hit metal
peak_hu = np.max(hu_values)
if peak_hu > 2500:  # Only if this line actually hit metal
    profile_threshold = peak_hu * (fw_percentage / 100.0)
    thresholds.append(profile_threshold)
```

### 2. Use minimum threshold instead of average
**File:** `app/core/metal_detection.py` (lines 644-648)

**Problem:** Averaging thresholds could still be too high (e.g., 4431 HU), excluding legitimate metal edges.

**Fix:** Use minimum valid threshold to capture all metal:
```python
# Before
avg_threshold = np.mean(thresholds)

# After
min_threshold = np.min(thresholds)
```

### 3. Display per-slice threshold in UI
**File:** `app/main.py` (lines 1121-1137)

**Problem:** UI showed only the average threshold across all slices, which was meaningless since each slice uses its own threshold.

**Fix:** Show current slice's threshold (updates when navigating):
```
Metal (this slice): >3200 HU      <- Updates per slice
Metal (avg all slices): >3750 HU  <- Reference
```

---

## Technical Notes

### How star profile threshold works:
1. Draw 16 lines from center of metal outward
2. Find peak HU along each line
3. If peak > 2500 HU (hit metal), calculate threshold = peak × FW%
4. Use minimum of all valid thresholds

### Fallback behavior:
- If no lines hit metal on a slice, falls back to 2500 HU default
- This explains why some slices show exactly 2500 HU threshold

---

## Testing Notes
- Tested on HIP4 patient
- Threshold now more reasonable (was 4431 HU with average, lower with minimum)
- Slice 95 shows 2500 HU because no lines hit metal (fallback)
