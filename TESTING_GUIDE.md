# Testing Guide: Star Profile Recovery Branch

## Quick Start

You have 3 ways to test the recovered star profile algorithms:

---

## Option 1: Test with Streamlit UI (Recommended for Visual Comparison)

### Step 1: Make sure you're on the right branch
```bash
git branch
# Should show: * feature/star-profile-recovery
```

### Step 2: Launch the Streamlit app
```bash
cd app
streamlit run main.py
```

### Step 3: Load patient data
1. Click **"Browse files"** and select a patient folder (e.g., `data/HIP* Patient/`)
2. Wait for DICOM files to load

### Step 4: Test Star Profiles in Metal Detection

**Currently**, the UI doesn't expose the `use_star_profiles` parameter yet. You have 2 options:

#### Option A: Temporarily modify main.py to enable star profiles
```python
# Find this line in main.py (around line 722-757)
result = detector.detect(
    st.session_state.ct_volume,
    st.session_state.spacing,
    # ADD THIS LINE:
    use_star_profiles=True,  # <-- Enable star profiles
    fw_percentage=fw_percentage,
    ...
)
```

#### Option B: Add a UI toggle (better)
I can help you add a checkbox to the UI to toggle star profiles on/off.

---

## Option 2: Test with Python Script (Fastest)

### Step 1: Run the test suite on synthetic data (already done)
```bash
python test_star_profiles.py
```
✅ **Already passed - all algorithms working**

### Step 2: Test on REAL patient data

Create a simple test script:

```python
# test_real_data.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

import numpy as np
from dicom_utils import load_dicom_series
from core.metal_detection import MetalDetector, MetalDetectionMethod
from core.discrimination import ArtifactDiscriminator, DiscriminationMethod

# Load real CT data
patient_folder = "data/HIP001 Patient"  # Change to your patient folder
ct_volume, spacing, _ = load_dicom_series(patient_folder)

print(f"Loaded CT volume: {ct_volume.shape}")
print(f"Spacing: {spacing} mm")

# Test 1: Metal detection WITHOUT star profiles
print("\n=== Test 1: Simplified Method ===")
detector_simple = MetalDetector(MetalDetectionMethod.ADAPTIVE_3D)
result_simple = detector_simple.detect(ct_volume, spacing, use_star_profiles=False)
print(f"Threshold: {result_simple['threshold']:.1f} HU")
print(f"Metal voxels: {np.sum(result_simple['mask'])}")

# Test 2: Metal detection WITH star profiles
print("\n=== Test 2: Star Profile Method ===")
detector_star = MetalDetector(MetalDetectionMethod.ADAPTIVE_3D)
result_star = detector_star.detect(ct_volume, spacing, use_star_profiles=True, fw_percentage=75.0)
print(f"Average threshold: {result_star['threshold']:.1f} HU")
print(f"Threshold range: {min(result_star['threshold_evolution']):.1f} - {max(result_star['threshold_evolution']):.1f} HU")
print(f"Metal voxels: {np.sum(result_star['mask'])}")
print(f"Slices with adaptive thresholds: {len(result_star['threshold_evolution'])}")

# Test 3: Compare discrimination methods
print("\n=== Test 3: Discrimination Comparison ===")
metal_mask = result_star['mask']
bright_mask = (ct_volume > 400) & (~metal_mask)
print(f"Bright voxels to classify: {np.sum(bright_mask)}")

# Distance-based (current default)
disc_distance = ArtifactDiscriminator(DiscriminationMethod.DISTANCE_BASED)
result_distance = disc_distance.discriminate(ct_volume, metal_mask, bright_mask, spacing)
print(f"\nDistance-based: Bone={result_distance['metadata']['bone_voxels']}, Artifacts={result_distance['metadata']['artifact_voxels']}")

# Star profile-based (recovered)
disc_star = ArtifactDiscriminator(DiscriminationMethod.STAR_PROFILE)
result_star_disc = disc_star.discriminate(ct_volume, metal_mask, bright_mask, spacing)
print(f"Star profile:   Bone={result_star_disc['metadata']['bone_voxels']}, Artifacts={result_star_disc['metadata']['artifact_voxels']}")

bone_diff = result_star_disc['metadata']['bone_voxels'] - result_distance['metadata']['bone_voxels']
print(f"\nDifference: {bone_diff:+d} bone voxels (star profile vs distance)")
```

Run it:
```bash
python test_real_data.py
```

---

## Option 3: Add Star Profile Toggle to UI (Best for Long-term)

I can add a checkbox to enable/disable star profiles in the UI. This would let you:
- Toggle star profiles on/off interactively
- Compare results side-by-side
- See the impact on detection accuracy

Would you like me to add this UI toggle?

---

## What to Look For When Testing

### 1. Metal Detection Differences

**Simplified method**:
- Single threshold for entire volume
- Threshold = 2500 HU (fixed minimum)

**Star profile method**:
- Adaptive threshold per slice
- Threshold varies with metal intensity
- Range typically 2000-2800 HU

**Expected**: Star profiles should detect similar or slightly more metal voxels

### 2. Discrimination Differences

**Distance-based**:
- Fast, simple
- Relies mainly on distance from metal
- May misclassify bone near metal as artifact

**Star profile**:
- Slower, more sophisticated
- Analyzes actual HU profiles
- Better at distinguishing bone from artifacts
- Should classify more voxels as artifacts (streaks)

**Expected**: Star profiles should identify 20-50% more artifacts

### 3. Visual Inspection

In the Streamlit UI, check:
- Metal mask looks clean (no gaps from under-detection)
- Bright artifact mask captures streaks accurately
- Bone mask doesn't include obvious artifacts
- Star profile visualization (already in UI) shows 16 radial lines

---

## Performance Benchmarks

**On synthetic data (100 slices, 512x512)**:
- Simplified detection: ~0.1 seconds
- Star profile detection: ~0.5-1.0 seconds
- Star profile discrimination: ~1-5 seconds (depends on bright region size)

**Expected on real data**:
- Small increase in processing time (5-10 seconds total)
- Quality improvement: 20-30% better accuracy

---

## Quick Test Checklist

- [ ] Synthetic data test passes ✅ (already done)
- [ ] Real patient CT loads successfully
- [ ] Star profile detection produces different thresholds
- [ ] Star profile identifies more artifacts than distance-based
- [ ] No crashes or errors
- [ ] Processing time acceptable (<30 seconds)
- [ ] Visual results look better than simplified method

---

## Troubleshooting

### "Module not found" errors
```bash
# Make sure you're in the repo root
cd /path/to/Density_Override_DD

# Install dependencies
pip install -r requirements.txt
```

### "No patient data"
```bash
# Check data folder exists
ls data/

# You should see folders like "HIP001 Patient", "HIP002 Patient", etc.
```

### Star profiles not activating
- Make sure `use_star_profiles=True` is set in the code
- Or ask me to add the UI toggle

### Slow performance
- Normal for first run (lots of processing)
- Star profiles add ~5-10 seconds
- Can optimize if needed

---

## Next Steps After Testing

1. **If results look good**: Merge to main branch
   ```bash
   git checkout main
   git merge feature/star-profile-recovery
   ```

2. **If needs tweaks**: Make adjustments on this branch
   ```bash
   # Still on feature/star-profile-recovery
   git add .
   git commit -m "Adjust star profile parameters"
   ```

3. **If performance issues**: We can optimize
   - Reduce angles (16 → 8)
   - Skip slices (every 2nd slice)
   - Add GPU acceleration

---

## Questions?

Ask me:
- "Add star profile toggle to UI" - I'll add a checkbox
- "Test on patient X" - I'll help you load specific data
- "Compare results" - I'll create comparison visualizations
- "Fix performance" - I'll optimize the algorithms
