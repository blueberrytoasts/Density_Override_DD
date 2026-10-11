---
name: trace-parameter
description: Trace a UI parameter from its PySide widget (spin box, checkbox, combo box) all the way through to where it's actually used in computation. Finds orphaned parameters that are defined but never consumed.
argument-hint: "[parameter-name]"
---

# Trace Parameter

Given a parameter name (provided as the argument, e.g. `/trace-parameter bone_low`), trace it through the entire codebase end-to-end. The argument is: $ARGUMENTS

## Steps

### 1. Find UI Definition
Search `pyside_app/main_window.py` for where this parameter is defined as a Qt widget (`_make_hu_spin`, `_make_weight_spin`, `QSpinBox`, `QDoubleSpinBox`, `QCheckBox`, `QComboBox`, etc.). Record:
- The widget type, range and default value
- The `self._…` attribute it's stored on
- Where its `.value()` / `.isChecked()` is read (usually in an `_on_…_clicked` handler)
- File and line number

### 2. Find Where It's Passed
Search for where the variable is passed to function calls. Trace through:
- The worker constructor call in `main_window.py` (e.g. `SegmentationWorker(...)`)
- The worker's `__init__` and `run()` in `pyside_app/*_worker.py`
- Any cached state on the window (e.g. `_seg_ctx_bands`) used by lazy workers
- Record each handoff as `file:line`

### 3. Find Where It's Consumed
In the target core module (`app/core/metal_detection.py`, `app/core/discrimination.py`, `app/contour_operations.py`, etc.), find where the parameter is actually used in a computation (comparison, multiplication, array indexing, conditional, etc.). Not just received as a function argument — actually *used*.

### 4. Report the Chain

Format the trace as a chain:

```
PARAMETER: <name>

UI DEFINITION:
  pyside_app/main_window.py:<line> — <widget_type>(min=..., max=..., default=...)

PASSED THROUGH:
  pyside_app/main_window.py:<line> → WorkerClass(<name>=spin.value())
  pyside_app/<worker>.py:<line> → function_name(<name>=self._<name>)
  app/<module>.py:<line> → function_name(<name>=variable)

CONSUMED IN COMPUTATION:
  app/<module>.py:<line> — <how it's used, e.g. "mask = (hu_volume >= bone_low)">

STATUS: CONNECTED | BROKEN (parameter defined but never reaches computation)
```

### 5. If BROKEN
- Identify exactly where the chain breaks
- Show the function signature that should accept it but doesn't
- Suggest the minimal fix (add parameter to function signature, add it to the call site, add the computation)
- Check if there's a similar parameter that IS connected — the broken one may be a duplicate or renamed version

### 6. Check for Shadow Parameters
Look for other variables with similar names (e.g. `bone_low` vs `bone_threshold_low` vs `bone_hu_min`) that might cause confusion or indicate a rename that wasn't completed everywhere.

## Important Notes
- If no argument is provided, scan ALL spin boxes / checkboxes / combo boxes in main_window.py and trace each one. Report a summary table.
- Pay special attention to parameters in the discrimination pipeline — these have known connectivity issues (see CLAUDE.md).
- Check both `SegmentationWorker` (star profile) and `LegacySegmentationWorker`, plus the lazy `ContextSplitWorker` path, as parameters may be connected in one but not the other.
