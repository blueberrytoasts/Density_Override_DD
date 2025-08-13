# Technical Documentation: CT Metal Artifact Characterization System

## Table of Contents
1. [System Overview](#system-overview)
2. [Metal Detection Algorithms](#metal-detection-algorithms)
3. [Artifact Discrimination Methods](#artifact-discrimination-methods)
4. [Edge Detection Techniques](#edge-detection-techniques)
5. [Segmentation Workflows](#segmentation-workflows)
6. [Visualization Components](#visualization-components)
7. [Configuration Management](#configuration-management)
8. [API Reference](#api-reference)

---

## System Overview

The CT Metal Artifact Characterization System is a comprehensive medical imaging analysis tool designed to identify and characterize metal artifacts in CT scans of patients with hip implants. The system employs multiple advanced algorithms for metal detection, tissue discrimination, and artifact segmentation.

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface (Streamlit)            │
├─────────────────────────────────────────────────────────┤
│                    Workflow Management                   │
├──────────────┬──────────────┬──────────────┬───────────┤
│Metal Detection│ Discrimination│ Segmentation │ Export    │
├──────────────┴──────────────┴──────────────┴───────────┤
│                    Core Algorithms                       │
├─────────────────────────────────────────────────────────┤
│                    DICOM I/O Layer                       │
└─────────────────────────────────────────────────────────┘
```

---

## Metal Detection Algorithms

### 1. Legacy Detection Method
**Function**: `detect_metal_legacy()`
**Location**: `core/metal_detection.py`

**Algorithm**:
1. Initial HU thresholding (default: >2500 HU)
2. Connected component analysis to find largest metal region
3. Star profile refinement using FW75% (Full Width at 75% Maximum)
4. ROI generation with configurable margin

**Parameters**:
- `min_metal_hu`: Minimum HU value for metal (default: 2500)
- `margin_cm`: ROI margin in cm (default: 2.0)
- `fw_percentage`: Percentage for FW threshold (default: 75.0)
- `dilation_iterations`: Morphological dilation iterations (default: 2)

**Mathematical Foundation**:
```
Threshold = max(HU) × (fw_percentage / 100)
ROI_margin = margin_cm × 10 / voxel_spacing
```

### 2. Adaptive 2D Detection
**Function**: `detect_metal_adaptive()`
**Location**: `core/metal_detection.py`

**Algorithm**:
1. Percentile-based adaptive thresholding
2. Component size filtering
3. Automatic ROI generation

**Key Innovation**: Uses statistical distribution of HU values rather than fixed threshold

**Parameters**:
- `intensity_percentile`: Percentile for threshold (default: 99.5)
- `min_component_size`: Minimum voxels for valid component (default: 100)

### 3. Adaptive 3D Multi-Planar Detection
**Function**: `detect_metal_adaptive_3d()`
**Location**: `core/metal_detection.py`

**Algorithm**:
1. Multi-planar analysis (axial, coronal, sagittal)
2. Per-slice adaptive thresholding using FW75%
3. Individual ROI generation for each metal component
4. 3D extent analysis

**Advanced Features**:
- Handles bilateral implants separately
- Creates per-slice threshold evolution
- Generates component-specific ROIs

**Star Profile Algorithm (FW75%)**:
```python
for angle in [0, 2π]:
    profile = radial_sample(center, angle, max_radius)
    peak = max(profile)
    threshold = peak × 0.75
    thresholds.append(threshold)
final_threshold = mean(thresholds)
```

---

## Artifact Discrimination Methods

### 1. Distance-Based Discrimination (Fast)
**Function**: `discriminate_fast()`
**Location**: `core/discrimination.py`

**Principle**: Bright artifacts are typically closer to metal than bone tissue.

**Algorithm**:
```python
distance_map = distance_transform_edt(~metal_mask)
bone = bright_mask & (distance > 0.5cm) & (distance < 5cm) & (smoothness > threshold)
artifact = bright_mask & ((variance > 250) | (distance > 5cm))
```

**Features Analyzed**:
- Distance from metal implant
- Local variance (texture)
- Smoothness after Gaussian filtering

### 2. Edge-Based Discrimination (Enhanced)
**Function**: `discriminate_enhanced()`
**Location**: `core/discrimination.py`

**Principle**: Bone has coherent, continuous edges while artifacts have chaotic edges.

**Edge Coherence Analysis**:
1. **Structure Tensor Computation**:
   ```
   S = [Gxx Gxy Gxz]
       [Gxy Gyy Gyz]
       [Gxz Gyz Gzz]
   ```
   Where G represents gradient components

2. **Eigenvalue Analysis**:
   - λ₁, λ₂, λ₃ = eigenvalues of S
   - Coherence = (λ₁ - λ₃) / λ₁
   - Anisotropy = std(λ) / mean(λ)

3. **Edge Continuity**:
   - Tracks edge persistence across slices
   - Measures radial vs tangential alignment
   - Computes multi-scale edge responses

**Classification Rules**:
- **Bone**: High coherence (>0.7), continuous edges, low radial alignment
- **Artifact**: Low coherence (<0.3), discontinuous edges, high radial alignment

### 3. Texture-Based Discrimination (Advanced)
**Function**: `discriminate_advanced()`
**Location**: `core/discrimination.py`

**Principle**: Bone and artifacts have distinct textural signatures.

**Texture Features**:

1. **Local Binary Patterns (LBP)**:
   ```python
   LBP(x,y) = Σ s(gₚ - gᵧ) × 2ᵖ
   where s(x) = 1 if x ≥ 0, else 0
   ```
   - Captures local texture patterns
   - Rotation-invariant uniform patterns

2. **Gray-Level Co-occurrence Matrix (GLCM)**:
   ```
   GLCM(i,j) = frequency of pixel pairs with values (i,j)
   ```
   - **Contrast**: Σᵢⱼ (i-j)² × GLCM(i,j)
   - **Homogeneity**: Σᵢⱼ GLCM(i,j) / (1 + |i-j|)
   - **Energy**: Σᵢⱼ GLCM(i,j)²
   - **Correlation**: Σᵢⱼ (i-μᵢ)(j-μⱼ) × GLCM(i,j) / (σᵢσⱼ)

3. **Gradient Features**:
   - **Laplacian of Gaussian (LoG)**: ∇²(G * I)
   - **Gradient Direction Variance**: var(arctan(Gy/Gx))
   - **Gradient Magnitude**: |∇I| = √(Gx² + Gy² + Gz²)

**Machine Learning Classification**:
```python
artifact_score = Σ wᵢ × fᵢ
where:
  w = feature weights
  f = normalized features
```

### 4. Star Profile Discrimination (Original)
**Function**: `discriminate_star()`
**Location**: `core/discrimination.py`

**Algorithm**:
1. Shoot 16 radial lines from metal center
2. Analyze HU profiles along each line
3. Classify based on profile smoothness and consistency

**Profile Analysis**:
- Peak detection and width measurement
- Smoothness calculation using gradient
- Comparison with expected anatomical profiles

---

## Edge Detection Techniques

### Enhanced Edge Analysis Components

#### 1. Sobel Edge Detection
```python
Gx = [[-1, 0, 1],    Gy = [[-1, -2, -1],
      [-2, 0, 2],          [ 0,  0,  0],
      [-1, 0, 1]]          [ 1,  2,  1]]
```

#### 2. Canny Edge Detection (Alternative)
- Gaussian smoothing
- Gradient calculation
- Non-maximum suppression
- Double thresholding
- Edge tracking by hysteresis

#### 3. Structure Tensor Analysis
**Purpose**: Analyze local image structure and edge coherence

**Computation**:
```python
# Structure tensor components
Ixx = Gaussian(Ix * Ix, σ)
Iyy = Gaussian(Iy * Iy, σ)
Ixy = Gaussian(Ix * Iy, σ)

# Eigenvalues represent edge strength and direction
λ₁ = 0.5 × (Ixx + Iyy + √((Ixx-Iyy)² + 4×Ixy²))
λ₂ = 0.5 × (Ixx + Iyy - √((Ixx-Iyy)² + 4×Ixy²))
```

#### 4. Multi-Scale Edge Persistence
**Algorithm**:
```python
scales = [0.5, 1.0, 2.0, 4.0]
persistent_edges = intersection(edges_at_scale(s) for s in scales)
```

#### 5. Radial vs Tangential Edge Analysis
**Purpose**: Distinguish streaking artifacts (radial) from anatomical edges (tangential)

```python
radial_vector = (x - metal_center) / |x - metal_center|
edge_direction = gradient_direction(x)
alignment = dot(radial_vector, edge_direction)
```

---

## Segmentation Workflows

### Russian Doll Segmentation
**Principle**: Sequential segmentation with mutual exclusion

**Workflow**:
```
1. Segment Metal (highest priority)
   ↓ Exclude from remaining
2. Segment Dark Artifacts
   ↓ Exclude from remaining
3. Discriminate Bright/Bone
   ↓ Apply discrimination
4. Segment Bright Artifacts
   ↓ Exclude from remaining
5. Segment Bone (lowest priority)
```

**Mathematical Formulation**:
```
Metal = {x | HU(x) > T_metal}
Dark = {x | HU(x) ∈ [T_dark_min, T_dark_max]} \ Metal
Bright_candidates = {x | HU(x) ∈ [T_bright_min, T_bright_max]} \ (Metal ∪ Dark)
Bone = Discriminator(Bright_candidates, features)
Bright = Bright_candidates \ Bone
```

### Refinement Pipeline
**Function**: `refine_bone_artifact_discrimination()`

**Steps**:
1. Morphological operations (opening/closing)
2. Connected component filtering
3. Hole filling
4. Smoothing with structure preservation

---

## Visualization Components

### 1. Overlay Generation
**Function**: `create_overlay_image()`

**Color Scheme**:
- Red (rgba: 1,0,0,0.7): Metal implant
- Yellow (rgba: 1,1,0,0.6): Bright artifacts
- Magenta (rgba: 1,0,1,0.6): Dark artifacts
- Blue (rgba: 0,0.2,0.8,0.5): Bone tissue
- Lime: ROI boundary

### 2. Star Profile Visualization
**Function**: `visualize_star_profiles()`

**Components**:
- CT slice with radial lines
- HU vs distance plots
- Threshold indicators
- 75% maximum line

### 3. Multi-Slice View
**Function**: `create_multi_slice_view()`

**Layout**: Grid of N×M slices with contour overlays

### 4. Histogram Analysis
**Function**: `create_histogram_with_thresholds()`

**Features**:
- Real-time threshold overlay
- Log-scale option for better visualization
- Color-coded regions

---

## Configuration Management

### Threshold Configuration
**Location**: `config.py`

**Structure**:
```python
ThresholdRange = {
    min_bound: float,    # Absolute minimum
    max_bound: float,    # Absolute maximum
    default_min: float,  # Default lower threshold
    default_max: float,  # Default upper threshold
    step: float,         # UI slider step
    label: str,          # Display label
    help_text: str       # Help tooltip
}
```

### Dynamic Threshold Adjustment
**Algorithm**:
```python
if metal_detected:
    metal_threshold = star_profile_result
    bright_min = metal_threshold × 0.75  # 75% rule
    bright_max = metal_threshold - 500
else:
    use_defaults()
```

---

## API Reference

### Core Functions

#### Metal Detection
```python
detector = MetalDetector(method=MetalDetectionMethod.ADAPTIVE_3D)
result = detector.detect(
    ct_volume=volume,
    spacing=(z_spacing, y_spacing, x_spacing),
    fw_percentage=75.0,
    margin_cm=2.0
)
```

#### Discrimination
```python
discriminator = ArtifactDiscriminator(method=DiscriminationMethod.TEXTURE_BASED)
result = discriminator.discriminate(
    ct_volume=volume,
    metal_mask=metal,
    bright_mask=bright_candidates,
    spacing=spacing
)
```

#### Segmentation
```python
segmentation = create_russian_doll_segmentation(
    ct_volume=volume,
    metal_mask=metal,
    spacing=spacing,
    dark_threshold_high=-150,
    bone_threshold_low=300,
    bone_threshold_high=1200,
    bright_threshold_low=800,
    bright_threshold_high=2000,
    use_enhanced_mode=True
)
```

### Result Structures

#### Metal Detection Result
```python
{
    'mask': ndarray,           # 3D binary mask
    'roi_bounds': dict,        # ROI boundaries
    'threshold': float,        # Used/detected threshold
    'threshold_evolution': list,  # Per-slice thresholds
    'individual_regions': dict,   # Component-specific ROIs
    'center_coords': tuple,    # Metal center (z,y,x)
    'method': str,             # Method used
    'metadata': dict           # Method-specific data
}
```

#### Discrimination Result
```python
{
    'bone_mask': ndarray,      # Binary bone mask
    'artifact_mask': ndarray,  # Binary artifact mask
    'confidence_map': ndarray, # Confidence scores [0,1]
    'distance_map': ndarray,   # Distance from metal
    'method': str,             # Method used
    'metadata': dict           # Additional data
}
```

---

## Performance Considerations

### Optimization Strategies
1. **GPU Acceleration**: Available for texture analysis via CuPy
2. **Multi-threading**: Profile analysis can be parallelized
3. **Caching**: Reuse distance maps and gradients
4. **Lazy Evaluation**: Compute features only where needed

### Memory Management
- Use float32 instead of float64 where possible
- Process slice-by-slice for large volumes
- Clear matplotlib figures after rendering

### Typical Performance Metrics
- Metal detection: 2-5 seconds
- Discrimination (fast): 3-5 seconds
- Discrimination (advanced): 10-20 seconds
- Full pipeline: 15-30 seconds

---

## Testing and Validation

### Unit Tests
- Test each algorithm independently
- Validate threshold ranges
- Check mask mutual exclusivity

### Integration Tests
- Full pipeline execution
- DICOM I/O verification
- Export format validation

### Performance Tests
- Memory usage monitoring
- Execution time benchmarks
- Scalability with volume size

---

## Future Enhancements

### Planned Features
1. **Deep Learning Integration**: CNN-based discrimination
2. **Automatic Parameter Tuning**: ML-based optimization
3. **Multi-Modal Support**: MRI/CT fusion
4. **Real-time Processing**: Stream processing for large datasets
5. **Cloud Deployment**: Distributed processing capabilities

### Research Directions
1. Advanced texture descriptors (Gabor, wavelets)
2. Graph-based segmentation methods
3. Physics-based artifact simulation
4. Uncertainty quantification

---

## References

1. **Star Profile Algorithm**: Based on "Adaptive Metal Artifact Reduction" (Wang et al., 2020)
2. **Structure Tensor**: "Analyzing Oriented Patterns" (Bigun & Granlund, 1987)
3. **LBP**: "Multiresolution Gray-Scale and Rotation Invariant Texture Classification" (Ojala et al., 2002)
4. **GLCM**: "Textural Features for Image Classification" (Haralick et al., 1973)
5. **Russian Doll Segmentation**: Novel approach developed for this project

---

## Appendix: Mathematical Notation

- `HU`: Hounsfield Units
- `T`: Threshold
- `σ`: Standard deviation / Gaussian sigma
- `λ`: Eigenvalue
- `∇`: Gradient operator
- `∇²`: Laplacian operator
- `G`: Gaussian kernel
- `*`: Convolution operation
- `\`: Set difference
- `∪`: Set union
- `∩`: Set intersection