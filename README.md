# CT Metal Artifact Characterization

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.29.0-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Medical Imaging](https://img.shields.io/badge/domain-medical--imaging-brightgreen.svg)](#)

An advanced Streamlit web application for automatic detection and characterization of metal artifacts in CT scans of patients with hip implants, designed for building machine learning datasets.

## ✨ Key Features

### 🎯 Automatic Metal Detection
- **3D Adaptive Detection**: Multi-planar analysis across axial, coronal, and sagittal views
- **16-point star profile analysis** for precise boundary detection
- **FW75% Maximum thresholding** - adaptive thresholds without initial HU values
- **Individual ROI regions** per metal component (avoids bilateral capture)
- **Legacy mode** available with initial HU thresholds

### 🔬 Advanced Segmentation
- **Russian Doll Segmentation**: Smart discrimination between bone and bright artifacts
- **Star profile-based discrimination** using peak characteristics:
  - Peak width analysis (bone: broad, artifacts: narrow)
  - Smoothness metrics (bone: smooth, artifacts: sharp)
  - Directional consistency (bone: uniform, artifacts: variable)
- **Sequential exclusion** ensuring no tissue overlap
- **Legacy threshold-based** option with boolean operations
- **GPU acceleration** support (optional with CuPy)

### 📊 Visualization & Analysis
- **Multi-slice grid view** with individual ROI indicators
- **Star profile visualization** showing all 16 analysis lines
- **Discrimination confidence maps** for bone/artifact separation
- **Threshold evolution plots** across slices
- **Volume statistics** with confidence metrics
- **Interactive contour visibility** toggles
- **Custom contour naming** for exports

### 💾 Export Capabilities
- **NIFTI format** (.nii.gz) for ML pipelines
- **DICOM RT Structure Sets** for clinical compatibility
- **Separate binary masks** for each tissue type
- **Multi-label NIFTI** option
- **Confidence maps** for discrimination results
- **Automatic affine matrix generation** from DICOM metadata

## 🛠️ Installation

### Using Conda (Recommended)
```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate IVH
```

**Note**: The environment name is `IVH` as specified in `environment.yml`.

### Using pip
```bash
# Install required dependencies
pip install -r requirements.txt

# Additional dependencies for full functionality
pip install nibabel>=4.0.0 plotly>=5.15.0
```

### Dependencies Overview
**Core Libraries:**
- `streamlit` - Web application framework
- `numpy` - Array operations and numerical computing
- `matplotlib` - Visualization and plotting
- `pydicom` - DICOM file handling
- `scipy` - Scientific computing and image processing
- `scikit-image` - Advanced image processing
- `nibabel` - NIFTI file I/O for ML pipelines
- `pandas` - Data manipulation
- `plotly` - Interactive plotting
- `Pillow` - Image processing support

## Usage

### 🚀 Quick Start
```bash
# Production deployment
./run.sh
```
Application will be available at `http://192.168.1.11:4224`

### 🔧 Development Mode
```bash
# Local development (default: localhost:8501)
cd app
streamlit run main.py

# Custom address/port
streamlit run main.py --server.address localhost --server.port 8501
```

### 🏥 Medical Network Deployment
```bash
# For medical network deployment
./run_medical.sh
```

## Project Structure

```
├── app/                      # Application source code
│   ├── main.py              # Streamlit web application
│   ├── dicom_utils.py       # DICOM loading and metadata handling
│   ├── dicom_export.py      # DICOM RT Structure Set creation
│   ├── contour_operations.py # Boolean operations and Russian doll segmentation
│   ├── visualization.py     # Advanced plotting and visualization
│   ├── config.py            # Configuration management
│   ├── body_mask.py         # Body masking utilities
│   ├── cornerstone_viewer.py # Medical image viewer integration
│   └── core/                # Core algorithms
│       ├── metal_detection.py # Metal detection algorithms
│       └── discrimination.py  # Artifact discrimination
├── data/                    # Patient DICOM data
│   └── [Patient directories] # Individual patient datasets
├── output/                  # Exported masks and RT structures
├── tests/                   # Test suite
│   ├── test_*.py           # Individual test modules
│   └── test_output/        # Test output directory
├── archive/                 # Original Jupyter notebooks
├── .streamlit/             # Streamlit configuration
├── requirements.txt         # Python dependencies
├── environment.yml          # Conda environment file
├── run.sh                  # Launch script
├── run_medical.sh          # Medical deployment script
├── verify_config.py        # Configuration verification
├── CLAUDE.md               # AI assistant guidance
└── README.md               # This file
```

## 📊 Workflow

### 1. 📂 Load Patient Data
1. Select patient from sidebar dropdown
2. Click **"Load Patient Data"** button
3. System automatically loads all DICOM slices and converts to Hounsfield Units
4. Displays basic volume information and slice count

### 2. 🎯 Automatic Metal Detection
1. Choose detection method in sidebar:
   - **✅ 3D Adaptive + Star Algorithm** (Recommended)
   - Legacy with Manual Threshold
2. Configure parameters (optional):
   - Search margin: 1.0-5.0 cm
   - FW percentage: 50-90%
   - Intensity percentile: 99.0-99.9%
3. Click **"🎯 Detect Metal Automatically"**
4. Algorithm automatically:
   - Analyzes axial, coronal, and sagittal planes
   - Identifies metal components >2500 HU
   - Creates individual ROIs per component
   - Performs 16-point star profile analysis
   - Applies FW75% adaptive thresholding

### 3. 🔍 Artifact Segmentation  
1. Select segmentation approach:
   - **✅ Russian Doll with Smart Discrimination** (Recommended)
   - Legacy Threshold-Based
2. Configure artifact detection distance (2-15 cm from metal)
3. Click **"🔍 Segment All Artifacts"**
4. Russian Doll process:
   - Segments dark artifacts (<-150 HU) excluding metal
   - Analyzes bone/artifact candidates (300-1500 HU)  
   - Uses star profile discrimination (peak width, smoothness)
   - Ensures mutual tissue exclusion
   - Applies morphological refinement

### 4. 📊 Analysis & Visualization
Navigate through multiple analysis tabs:
- **Single Slice Analysis**: Interactive slice viewer with overlays
- **Multi-Slice View**: Grid display with ROI indicators  
- **Metal Detection Details**: Star profile visualizations
- **Volume Statistics**: Quantitative metrics and confidence scores

### 5. 💾 Export Results
- **NIFTI Format**: Individual masks or multi-label volumes for ML
- **DICOM RT Structures**: Clinical-compatible format with custom naming
- **Confidence Maps**: Discrimination confidence for quality assessment

## 🧠 Algorithm Details

### FW75% Maximum Thresholding
The Full Width at 75% Maximum method:
1. Generates HU vs distance profiles along 16 radial lines
2. Finds peak HU values (metal centers)
3. Calculates 75% of peak intensity as threshold
4. Averages thresholds across all profiles for robustness
5. No initial HU threshold required

### Russian Doll Segmentation
Sequential tissue segmentation with mutual exclusion:
1. **Dark artifacts**: Simple thresholding (< -150 HU) excluding metal
2. **Bone/Bright discrimination**: 
   - Analyzes star profiles for each candidate voxel
   - Bone characteristics: broad peaks (~3-5mm), smooth transitions, consistent across directions
   - Artifact characteristics: narrow peaks (<2mm), sharp edges, variable across directions
3. **Mutual exclusion**: Each voxel belongs to exactly one tissue type

### Profile-Based Discrimination Metrics
- **Peak Width**: FWHM (Full Width Half Maximum) in mm
- **Smoothness**: 1/(1 + variance of first derivative)
- **Directional Variance**: Consistency of characteristics across 16 directions
- **Edge Sharpness**: Maximum gradient magnitude
- **Confidence Score**: Weighted combination of all metrics

### 3D Adaptive Analysis
- **Multi-planar projections**: Analyzes axial, coronal, and sagittal views
- **Individual component tracking**: Separate ROIs for each metal region
- **Extent calculation**: Full 3D bounding box from all projections
- **Focused processing**: Only analyzes slices with confirmed metal

## ⚙️ Configuration

### Metal Detection Parameters
- **3D Search Margin**: 1.0-5.0 cm (default: 2.0 cm)
- **FW Percentage**: 50-90% (default: 75%)
- **Intensity Percentile**: 99.0-99.9% (default: 99.5%)

### Segmentation Parameters (Russian Doll)
- **Dark Artifacts**: < -150 HU (adjustable)
- **Bone/Bright Range**: 300-1500 HU (discriminated by profiles)
- **Max Artifact Distance**: 2.0-15.0 cm from metal (default: 10.0 cm)

### Legacy Parameters
- **Initial Metal Threshold**: 1000-4000 HU (default: 2000)
- **Bright Artifacts**: 800-3000 HU
- **Bone**: 150-1500 HU

## 🧪 Testing

The project includes a comprehensive test suite to validate functionality:

```bash
# Run all tests
python tests/final_test_all_patients.py

# Test specific features
python tests/test_3d_adaptive.py      # 3D adaptive metal detection
python tests/test_russian_doll.py     # Russian doll segmentation
python tests/test_segmentation.py     # General segmentation
python tests/test_app.py             # Application functionality
```

## 💻 Requirements

### Minimum Requirements
- Python 3.8+
- 8GB RAM
- Modern web browser
- DICOM CT series with metal implants

### Recommended
- Python 3.12+ (as specified in environment.yml)
- 16GB RAM
- NVIDIA GPU with CUDA (for optional acceleration)
- CuPy for GPU processing (optional)

## Development Status

### ✅ Implemented Features
- Bilateral implant support (individual ROIs)
- DICOM RT Structure export
- Smart bone/artifact discrimination using star profiles
- GPU acceleration support (optional CuPy)
- Confidence mapping for discrimination results
- Custom contour naming for exports
- 3D adaptive metal detection with FW75% thresholding
- Russian doll segmentation with mutual exclusion
- Multi-format export (NIFTI, DICOM RT)

### 🚧 Future Enhancements
- Batch processing for multiple patients
- Machine learning model integration
- 3D visualization capabilities
- Automated quality assessment metrics
- Cloud deployment options
- REST API endpoints for integration

---

## Contributing

This is a medical imaging research project. When contributing:
1. Follow existing code patterns and documentation standards
2. Test changes with multiple patient datasets
3. Ensure DICOM compliance for clinical compatibility
4. Validate HU ranges and spatial accuracy

## License

This project is intended for research and educational purposes in medical imaging.