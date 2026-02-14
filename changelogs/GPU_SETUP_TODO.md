# GPU Acceleration Setup - In Progress

**Date:** 2026-01-24
**Branch:** star_profile_upgrade

## Current Status
GPU acceleration is partially set up but not yet working.

## What's Done
- Created `app/core/gpu_utils.py` with CuPy GPU functions
- Added GPU toggle checkbox in UI (only shows when CuPy is detected)
- Installed `cupy-cuda13x` package (required for RTX 5060 Ti Blackwell GPU)
- Installed CUDA Toolkit 13.1

## What's Left
1. **Verify CUDA 13.1 installation** - Check that `nvrtc64_130_0.dll` exists in:
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\
   ```

2. **Add CUDA 13.1 to PATH** - Run in PowerShell (Admin):
   ```powershell
   $newPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin"
   [Environment]::SetEnvironmentVariable("Path", $env:Path + ";" + $newPath, "User")
   ```

3. **Set CUDA_PATH environment variable**:
   ```powershell
   [Environment]::SetEnvironmentVariable("CUDA_PATH", "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1", "User")
   ```

4. **Restart terminal** and run Streamlit to test

## Hardware Info
- **GPU:** NVIDIA RTX 5060 Ti (Blackwell architecture, compute capability 12.0)
- **CUDA Driver:** 13.1
- **Required:** CUDA Toolkit 13.x (not 12.x - Blackwell is too new)

## Test Command
```bash
python -c "import cupy; print(cupy.cuda.runtime.getDeviceProperties(0)['name'])"
```

## If It Still Fails
- Check if nvrtc DLL exists: `dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvrtc*.dll"`
- May need to reinstall CUDA Toolkit 13.1 with full components (Runtime, Libraries, Compiler)
- Uncheck "GPU Acceleration" in the app to use CPU fallback for now
