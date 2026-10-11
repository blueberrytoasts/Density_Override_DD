---
name: validate-pipeline
description: Run a comprehensive validation of the analysis pipeline. Checks that UI controls connect to algorithms, function signatures match, masks are consistent, and no parameters are orphaned.
---

# Validate Pipeline

Perform a static analysis of the full pipeline from DICOM loading through export. This does NOT run the app — it reads the code and checks for integration issues.

## Validation Checks

### 1. Function Signature Matching
For every call from `pyside_app/` (window + workers) into `app/` modules, verify:
- [ ] All arguments passed exist in the function signature
- [ ] All required parameters (no default) are provided at the call site
- [ ] Argument types are consistent (not passing a string where int expected)
- [ ] No deprecated parameter names from old versions

Check these call chains:
- `dicom_loader.py` -> `dicom_utils.py` (loading)
- `metal_worker.py` -> `core/metal_detection.py` (detection)
- `segment_worker.py` -> `core/discrimination.py` (discrimination)
- `segment_worker.py` -> `contour_operations.py` (segmentation, context splits)
- `body_mask_worker.py` -> `body_mask.py`
- `main_window.py` -> worker constructors (argument names and order)

### 2. UI Parameter Coverage
Scan all Qt input widgets in `pyside_app/main_window.py` (spin boxes, checkboxes, combo boxes). For each:
- [ ] Is the value actually used downstream? (not just displayed)
- [ ] Does changing the value affect the output? (trace to computation)
- [ ] Is it applied on the right action (Detect vs Segment vs lazy split), and reset by **Reset**?

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
- [ ] Window state (`_metal_mask`, `_seg_masks`, cached splits) is set before it's read, and stale worker results are discarded

### 5. Import Consistency
- [ ] All imports resolve (no importing removed functions)
- [ ] No circular imports between modules
- [ ] Optional imports (cupy) have proper try/except fallbacks

### 6. Known Issue Check
Cross-reference against known issues in CLAUDE.md:
- [ ] HU is a weighted vote, not a hard gate (design note)
- [ ] Worker signals connect to bound methods, not lambdas (thread-safety)
- [ ] Any workarounds documented still match current code

## Output Format

```
PIPELINE VALIDATION REPORT
==========================

1. FUNCTION SIGNATURES
   [PASS] metal_worker.py:NN -> metal_detection.detect_metal_3d() -- all args match
   [ISSUE] segment_worker.py:NN -> discrimination.discriminate() -- bone_low not passed

2. UI PARAMETERS
   [PASS] fw_percentage (spin box) -> metal_detection.py:NN
   [ORPHAN] bone_low (spin box) -- defined at main_window.py:NN, never reaches computation

3. MASK CONSISTENCY
   [PASS] All masks are (num_slices, H, W)
   [WARN] Line 892: 2D operation on potentially 3D mask

4. PIPELINE ORDER
   [PASS] All stage dependencies satisfied

5. IMPORTS
   [PASS] All imports resolve

6. KNOWN ISSUES
   [CONFIRMED] HU is a weighted vote (CLAUDE.md matches code)

SUMMARY: X passes, Y issues, Z warnings
```

## Important Notes
- This is a READ-ONLY analysis. Do not modify any files.
- Focus on actionable issues -- don't flag style preferences or minor warnings.
- If an issue was already documented in CLAUDE.md, mark it as [CONFIRMED] rather than [ISSUE].
- Prioritize issues that would cause silent incorrect results (wrong segmentation) over issues that would cause visible errors (crashes).
