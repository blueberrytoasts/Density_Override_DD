# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## CURRENT WORK STATUS (feature/star-profile-recovery branch)

### Outstanding Issue: HU Range Sliders Non-Functional

**Problem**: When using "Russian Doll with Star Profile Discrimination", the bone HU range sliders (`bone_low`, `bone_high`) have NO effect on segmentation results.

**Root Cause**: Star profile discrimination (`app/core/discrimination.py:351-625`) uses ONLY profile characteristics (peak width, smoothness, gradient) and ignores the HU range parameters.

**Fix Options**:
- **Option A (Hybrid)**: `bone = (bone_score > 0) AND (bone_low <= HU <= bone_high)`
- **Option B (Weighted)**: Add HU as 4th feature in bone_score calculation
- **Option C (Two-stage)**: HU filter first, then profile analysis

**Key Files**:
- UI sliders: `app/main.py:~350-360`
- Discrimination call: `app/main.py:~900-920`
- Star profile logic: `app/core/discrimination.py:351-625`

**Workaround**: Use "Russian Doll with Distance-Based Discrimination" if HU control is needed.

### What's Working
- Full FW75% star profile metal detection (`app/core/metal_detection.py:511-591`)
- Per-slice adaptive thresholding (`app/core/metal_detection.py:216-267`)
- 32-angle configurable star profiles
- ROI bounds and body mask constraints
- Profile-based discrimination (just needs HU integration)

---

## Project Overview

This is a medical imaging research project focused on characterizing metal artifacts in CT scans of patients with hip implants. The project has been refactored from Jupyter notebooks into a Streamlit web application for better usability and deployment.

## Project Structure

```
├── app/                       # Application source code
│   ├── main.py               # Streamlit web application
│   ├── dicom_utils.py        # DICOM loading and RTSTRUCT handling
│   ├── dicom_export.py       # DICOM RT Structure export
│   ├── contour_operations.py # Boolean operations and mask refinement
│   ├── visualization.py      # Visualization and plotting
│   └── core/                  # Core algorithms
│       ├── metal_detection.py   # Metal detection (3D adaptive + star profiles)
│       └── discrimination.py    # Bone/artifact discrimination
├── data/                      # Patient DICOM data (HIP* Patient folders)
├── output/                    # Generated masks and exports
├── requirements.txt           # Python dependencies
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
- Metal: >2500 HU (automatically detected using FW75% algorithm)
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
- `main.py`: Streamlit UI and workflow coordination
- `dicom_utils.py`: DICOM I/O operations, HU conversion
- `dicom_export.py`: DICOM RT Structure Set creation
- `contour_operations.py`: Boolean operations, mask refinement, Russian doll segmentation
- `visualization.py`: Plotting, overlays, and multi-slice views
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
3. Calculating 75% of peak value as the threshold
4. Averaging thresholds across all profiles

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