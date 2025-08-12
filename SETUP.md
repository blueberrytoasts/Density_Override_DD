# Setup Instructions

## Quick Setup with Conda

### 1. Install Miniconda (if not already installed)
```bash
# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Make executable and run
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh

# Follow prompts and restart terminal
```

### 2. Create IVH Environment
```bash
# Navigate to project directory
cd /path/to/IVH-DO_DD

# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate IVH
```

### 3. Run the Application
```bash
# Make run script executable
chmod +x run.sh

# Launch application
./run.sh
```

The application will be available at: `http://192.168.1.11:4224`

## Alternative: Manual Environment Setup

```bash
# Create new environment with Python 3.12
conda create -n IVH python=3.12

# Activate environment
conda activate IVH

# Install packages
pip install -r requirements.txt
```

## Troubleshooting

### Port Already in Use
If port 4224 is already in use:
```bash
# Find process using port
lsof -i :4224

# Or run on different port
streamlit run app/main.py --server.port 8501
```

### Module Import Errors
```bash
# Ensure environment is activated
conda activate IVH

# Reinstall requirements
pip install -r requirements.txt --upgrade
```

### DICOM Loading Issues
- Ensure DICOM files have .dcm extension
- Check that CT folders contain "CT" in the name
- Verify RTSTRUCT files are in separate folders

## Verify Installation

```bash
# Check Python version
python --version  # Should show 3.12.x

# Test imports
python -c "import streamlit, nibabel, pydicom; print('All packages imported successfully')"
```