# Technical Guide: CT Metal Artifact Characterization

## Overview

This document provides a technical deep-dive into the metal artifact characterization algorithms, focusing on the star profile analysis and Russian doll segmentation approach.

## Core Concepts

### 1. Hounsfield Units (HU)
CT scanners measure X-ray attenuation and convert it to Hounsfield Units:
- Air: -1000 HU
- Water: 0 HU
- Soft tissue: -100 to 300 HU
- Bone: 300-1500 HU
- Metal: >2500 HU

Metal implants cause beam hardening artifacts that appear as bright streaks (high HU) and dark streaks (low HU) in the CT image.

### 2. Star Profile Analysis
The star profile algorithm shoots 16 radial lines from a center point, sampling HU values along each line:

```
        N
    NW  |  NE
      \ | /
   W ---*--- E
      / | \
    SW  |  SE
        S
```

Each profile captures HU values as a function of distance from center, revealing characteristic patterns for different tissue types.

## Metal Detection Algorithm (3D Adaptive)

### Step 1: Initial High-Intensity Detection
```python
# Find voxels likely to be metal
threshold = max(np.percentile(ct_volume, 99.5), 2500)
high_intensity_mask = ct_volume > threshold
```

### Step 2: 3D Analysis
The algorithm analyzes metal distribution across three anatomical planes:
- **Axial**: Traditional slice-by-slice view
- **Coronal**: Front-to-back view
- **Sagittal**: Side-to-side view

This ensures complete capture of elongated implants (like hip stems).

### Step 3: Component Identification
```python
# Find connected components
labeled_array, num_features = label(high_intensity_mask)

# Keep only significant components
for i in range(1, num_features + 1):
    size = np.sum(labeled_array == i)
    if size > 100:  # Minimum size threshold
        components.append(i)
```

### Step 4: Individual ROI Creation
Instead of one large ROI spanning both hips, create focused regions:
```python
# For each metal component
for component in components:
    # Get bounding box
    y_coords, x_coords = np.where(component_mask)
    
    # Add uniform margin (avoiding rectangular ROIs)
    margin_pixels = int(margin_cm * 10 / pixel_spacing)
    
    roi = {
        'y_min': np.min(y_coords) - margin_pixels,
        'y_max': np.max(y_coords) + margin_pixels,
        'x_min': np.min(x_coords) - margin_pixels,
        'x_max': np.max(x_coords) + margin_pixels
    }
```

### Step 5: FW75% Thresholding
For each slice with metal:
1. Shoot 16 star profiles from metal center
2. Find peaks along each profile
3. Calculate 75% of peak value
4. Average thresholds across all profiles

```python
for peak_idx in peaks:
    peak_value = hu_values[peak_idx]
    if peak_value > 1200:  # Metal candidate
        fw_threshold = 0.75 * peak_value
        thresholds.append(fw_threshold)

final_threshold = np.mean(thresholds)
```

## Russian Doll Segmentation

Named after nested Russian dolls, this approach segments tissues in order of decreasing HU values, with each subsequent segmentation excluding previous ones.

### Step 1: Dark Artifacts
```python
# Simple thresholding for dark streaks
dark_mask = (ct_volume < -150) & (~metal_mask)
```

### Step 2: Bone vs Bright Artifact Discrimination

This is the key innovation. Instead of using fixed HU ranges, we analyze profile characteristics:

#### Profile Analysis for Each Candidate Voxel
```python
# For each voxel in the 300-1500 HU range
for z, y, x in candidate_voxels:
    # Get 16 star profiles
    profiles = get_star_profiles(ct_volume[z], y, x)
    
    # Analyze each profile
    characteristics = []
    for distances, hu_values in profiles:
        char = analyze_profile(distances, hu_values)
        characteristics.append(char)
    
    # Classify based on characteristics
    result = classify_bone_vs_artifact(characteristics)
```

#### Key Discriminating Features

1. **Peak Width (FWHM)**
   - Bone: Broad peaks (3-5mm)
   - Artifacts: Narrow peaks (<2mm)

2. **Smoothness**
   - Bone: Smooth transitions (low derivative variance)
   - Artifacts: Sharp edges (high derivative variance)

3. **Directional Consistency**
   - Bone: Similar profiles in all directions
   - Artifacts: Highly variable across directions

4. **Edge Gradient**
   - Bone: Gradual transitions
   - Artifacts: Steep transitions

#### Classification Logic
```python
def classify_bone_vs_artifact(profiles):
    # Calculate variance across directions
    width_variance = np.var([p['avg_width'] for p in profiles])
    
    # High variance indicates artifact
    if width_variance > 2.0:
        return 'artifact'
    
    # Consistent broad peaks indicate bone
    avg_width = np.mean([p['avg_width'] for p in profiles])
    if avg_width > 3.0 and width_variance < 1.0:
        return 'bone'
    
    # Additional checks...
```

### Step 3: Mutual Exclusion
```python
# Ensure no overlap
bone_mask = bone_mask & (~metal_mask) & (~dark_mask)
bright_mask = bright_mask & (~metal_mask) & (~dark_mask) & (~bone_mask)
```

## Confidence Mapping

The discrimination algorithm assigns confidence scores:
```python
confidence = bone_score / (bone_score + artifact_score)
```

High confidence (>0.8) indicates clear discrimination, while low confidence (<0.5) suggests ambiguous regions.

## GPU Acceleration (Optional)

When available, CuPy accelerates distance calculations:
```python
if GPU_AVAILABLE:
    # GPU version
    metal_gpu = cp.asarray(metal_mask)
    distances_gpu = cp_ndimage.distance_transform_edt(~metal_gpu)
    distances = cp.asnumpy(distances_gpu)
else:
    # CPU fallback
    distances = distance_transform_edt(~metal_mask)
```

## Export Formats

### NIFTI (.nii.gz)
Binary masks with affine transformation matrix preserving spatial information:
```python
nifti_img = nib.Nifti1Image(mask.astype(np.uint8), affine)
nib.save(nifti_img, output_path)
```

### DICOM RT Structure Set
Clinical standard for radiotherapy contouring:
```python
# Convert mask to contours
contours = measure.find_contours(mask_slice, 0.5)

# Convert to patient coordinates
for point in contour:
    x_mm = x_idx * spacing[2]
    y_mm = y_idx * spacing[1] 
    z_mm = slice_positions[z_idx]
    contour_data.extend([x_mm, y_mm, z_mm])
```

## Key Advantages

1. **No Initial HU Threshold**: FW75% algorithm adapts to each patient
2. **3D Awareness**: Captures full implant extent across all planes
3. **Smart Discrimination**: Distinguishes bone from artifacts using physics
4. **Clinical Compatibility**: Exports to standard formats
5. **Confidence Metrics**: Transparency about algorithm decisions

## Common Pitfalls and Solutions

### Problem: Bilateral ROI Capture
**Solution**: Individual component analysis with size constraints

### Problem: Bone Misclassified as Artifact
**Solution**: Profile-based discrimination considering width and smoothness

### Problem: Incomplete Metal Detection
**Solution**: 3D analysis across all anatomical planes

### Problem: Variable Artifact Intensity
**Solution**: Distance-based constraints from metal source

## Testing and Validation

To test the discrimination algorithm:
```bash
python3 test_integration.py
```

To run full pipeline test:
```bash
python3 test_russian_doll.py
```

## Further Reading

- Star profile algorithm: Based on radiological physics of beam hardening
- Russian doll approach: Sequential segmentation with mutual exclusion
- Profile discrimination: Leverages characteristic differences in HU transitions
- DICOM standards: See DICOM PS3.3 for RT Structure Set specification