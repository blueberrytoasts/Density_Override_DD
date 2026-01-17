# Metal Detection with Per-Slice Star Profile Analysis

## Overview

This document describes the adaptive metal detection algorithm used to identify metal implants (e.g., hip prostheses) in CT scans. The algorithm uses a star profile analysis technique to automatically determine optimal HU (Hounsfield Unit) thresholds on a per-slice basis, rather than relying on a single fixed threshold.

---

## Two Detection Modes

### 1. Adaptive Star Profile Detection (Recommended)

Uses 16-angle radial profiles emanating from detected metal centers to calculate slice-specific thresholds based on the Full Width at X% Maximum (FW%) principle.

**When to use:** Most cases - provides accurate metal boundary detection that adapts to varying implant densities and scanner calibrations.

### 2. Fixed Threshold Detection (Fallback)

Simple thresholding where any voxel >2500 HU is classified as metal.

**When to use:**
- Quick analysis where precision isn't critical
- When star profile analysis produces unexpected results
- Debugging or comparison purposes

**To enable:** Disable "Use Star Profiles" checkbox in the UI sidebar.

---

## Star Profile Algorithm: Step-by-Step

### Step 1: Initial Metal Detection

First, a coarse detection identifies candidate metal regions:

```
initial_mask = ct_volume > 2500 HU
```

This captures high-intensity voxels that are almost certainly metal.

### Step 2: Find Metal Centers Per Slice

For each axial slice containing detected metal:

```python
for z in slices_with_metal:
    y_indices, x_indices = np.where(initial_mask[z])
    center_y = mean(y_indices)
    center_x = mean(x_indices)
```

### Step 3: Cast 16 Radial Lines (Star Pattern)

From each slice's metal center, 16 lines are projected outward at equal angular intervals (22.5° apart):

```
Angles: 0°, 22.5°, 45°, 67.5°, 90°, 112.5°, 135°, 157.5°,
        180°, 202.5°, 225°, 247.5°, 270°, 292.5°, 315°, 337.5°
```

Each line extends from the center toward the ROI boundary, sampling HU values along its path.

```
        Line 2 (22.5°)
             \
    Line 1    \      Line 3
    (0°) ------●------ (45°)
              /\
             /  \
        Line 16  Line 4
        (337.5°) (67.5°)
            ...etc
```

### Step 4: Extract Peak HU Along Each Line

For each of the 16 lines, find the maximum HU value:

```python
for angle in range(16):
    hu_values = sample_along_line(slice_data, center, angle, length)
    peak_hu = max(hu_values)
```

### Step 5: Filter Lines That Actually Hit Metal

**Critical filtering step:** Only include lines where the peak HU > 2500.

```python
if peak_hu > 2500:  # This line actually hit metal
    valid_thresholds.append(peak_hu * fw_percentage)
```

**Why this matters:** Some lines may miss the metal entirely and hit bone (~1000-1500 HU) or soft tissue. Including these would drag the threshold down incorrectly.

### Step 6: Calculate FW% Threshold

For each valid line (peak > 2500 HU), calculate the threshold as a percentage of the peak:

```python
threshold = peak_hu * (fw_percentage / 100.0)
```

Default `fw_percentage = 75%`, meaning the threshold is set at 75% of the peak metal intensity.

**Example:**
- Peak HU = 4000
- FW75% threshold = 4000 × 0.75 = 3000 HU

### Step 7: Select Minimum Threshold

Use the **minimum** of all valid line thresholds for this slice:

```python
slice_threshold = min(valid_thresholds)
```

**Rationale:** Using the minimum ensures we capture the full extent of the metal, including edges where intensity drops off. Using average or maximum could exclude legitimate metal voxels.

### Step 8: Apply Threshold to Slice

```python
refined_mask[z] = ct_volume[z] > slice_threshold
```

### Step 9: Store Per-Slice Threshold

Thresholds are stored in a dictionary for display and debugging:

```python
threshold_evolution = {
    90: 3200,  # Slice 90: threshold 3200 HU
    91: 3150,  # Slice 91: threshold 3150 HU
    92: 2500,  # Slice 92: fallback (no valid lines hit metal)
    ...
}
```

---

## Fallback Behavior

If no lines on a slice hit metal (all peaks < 2500 HU), the algorithm falls back to the initial 2500 HU threshold for that slice:

```python
if no valid thresholds:
    slice_threshold = 2500  # Fallback
```

This can happen on slices at the edge of the implant where the metal cross-section is small.

---

## Algorithm Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `fw_percentage` | 75% | 50-90% | Percentage of peak HU for threshold calculation |
| `use_star_profiles` | True | True/False | Enable/disable adaptive detection |
| `num_angles` | 16 | - | Number of radial lines in star pattern |
| `min_metal_hu` | 2500 | - | Minimum HU to consider a line as hitting metal |

---

## Output Data Structure

The detection returns:

```python
{
    'mask': np.ndarray,           # Final 3D binary metal mask
    'threshold': float,           # Average threshold across all slices
    'threshold_evolution': dict,  # {slice_index: threshold_value}
    'roi_bounds': dict,           # Bounding box of detected metal
    'individual_regions': dict,   # Per-slice region information
    'metadata': {
        'fw_percentage': 75,
        'num_components': 2,      # e.g., bilateral hip implants
        ...
    }
}
```

---

## UI Display

When viewing results, the UI shows:

**For slices with metal:**
```
Metal (this slice): >3200 HU
Metal (avg all slices): >3450 HU
```

**For slices without metal:**
```
Metal: >3450 HU (no metal on this slice)
```

---

## Advantages Over Fixed Threshold

| Aspect | Fixed (2500 HU) | Star Profile Adaptive |
|--------|-----------------|----------------------|
| Scanner variability | Sensitive | Robust |
| Different implant materials | May miss some | Adapts automatically |
| Partial volume effects | Over/under-segment | Better edge detection |
| Per-slice accuracy | Same for all | Optimized per slice |
| Computational cost | Minimal | Slightly higher |

---

## Code References

- Star profile threshold calculation: `app/core/metal_detection.py:590-650`
- Per-slice adaptive loop: `app/core/metal_detection.py:216-267`
- UI threshold display: `app/main.py:1121-1137`

---

## Related Documentation

- [Star Algorithm Fix Changelog](../changelogs/2026-01-15_1_star-algorithm-fix.md) - Recent bug fixes to threshold calculation
