---
name: validate-pipeline
description: Run a comprehensive validation of the analysis pipeline. Checks that UI controls connect to algorithms, function signatures match, masks are consistent, and no parameters are orphaned.
---

# Validate Pipeline

Perform a static analysis of the full pipeline from DICOM loading through export. This does NOT run the app — it reads the code and checks for integration issues.

## Validation Checks

### 1. Function Signature Matching
For every function call from `main.py` into core modules, verify:
- [ ] All arguments passed exist in the function signature
- [ ] All required parameters (no default) are provided at the call site
- [ ] Argument types are consistent (not passing a string where int expected)
- [ ] No deprecated parameter names from old versions

Check these call chains:
- `main.py` -> `dicom_utils.py` (loading)
- `main.py` -> `core/metal_detection.py` (detection)
- `main.py` -> `core/discrimination.py` (discrimination)
- `main.py` -> `contour_operations.py` (segmentation)
- `main.py` -> `dicom_export.py` (export)
- `main.py` -> `visualization.py` (display)

### 2. UI Parameter Coverage
Scan all Streamlit widgets in `main.py` (sliders, checkboxes, selectboxes, number_inputs). For each:
- [ ] Is the value actually used downstream? (not just displayed)
- [ ] Does changing the value affect the output? (trace to computation)
- [ ] Is it gated behind the correct conditional (e.g., only shown when star profiles enabled)?

Flag any widget whose value is never passed to a core function.

### 3. Mask Shape Consistency
Trace mask creation and usage:
- [ ] Metal mask shape matches volume shape `(num_slices, rows, cols)`
- [ ] ROI masks are per-component, not global
- [ ] Boolean operations (AND, OR, NOT) are between same-shaped arrays
- [ ] No 2D mask applied to 3D volume without proper broadcasting
- [ ] Russian Doll exclusion: `dark AND metal = empty`, `bright AND metal = empty`, `bone AND artifact = empty`

### 4. Pipeline Stage Dependencies
Verify execution order makes sense:
- [ ] Metal detection runs before discrimination (discrimination needs metal mask)
- [ ] ROI creation runs before discrimination (discrimination is per-ROI)
- [ ] Body mask available before discrimination (used for bounds)
- [ ] All required session_state keys are set before they're read

### 5. Import Consistency
- [ ] All imports resolve (no importing removed functions)
- [ ] No circular imports between modules
- [ ] Optional imports (cupy) have proper try/except fallbacks

### 6. Known Issue Check
Cross-reference against known issues in CLAUDE.md:
- [ ] bone_low/bone_high slider connectivity status
- [ ] Star profile discrimination HU integration status
- [ ] Any workarounds documented still match current code

## Output Format

```
PIPELINE VALIDATION REPORT
==========================

1. FUNCTION SIGNATURES
   [PASS] main.py:423 -> metal_detection.detect_metal_3d() -- all args match
   [ISSUE] main.py:910 -> discrimination.discriminate() -- bone_low not passed

2. UI PARAMETERS
   [PASS] metal_threshold (slider) -> metal_detection.py:45
   [ORPHAN] bone_low (slider) -- defined at main.py:355, never reaches computation

3. MASK CONSISTENCY
   [PASS] All masks are (num_slices, H, W)
   [WARN] Line 892: 2D operation on potentially 3D mask

4. PIPELINE ORDER
   [PASS] All stage dependencies satisfied

5. IMPORTS
   [PASS] All imports resolve

6. KNOWN ISSUES
   [CONFIRMED] bone_low/bone_high still disconnected (CLAUDE.md matches code)

SUMMARY: X passes, Y issues, Z warnings
```

## Important Notes
- This is a READ-ONLY analysis. Do not modify any files.
- Focus on actionable issues -- don't flag style preferences or minor warnings.
- If an issue was already documented in CLAUDE.md, mark it as [CONFIRMED] rather than [ISSUE].
- Prioritize issues that would cause silent incorrect results (wrong segmentation) over issues that would cause visible errors (crashes).
