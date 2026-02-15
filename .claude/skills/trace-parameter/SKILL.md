---
name: trace-parameter
description: Trace a UI parameter from its Streamlit slider/widget all the way through to where it's actually used in computation. Finds orphaned parameters that are defined but never consumed.
argument-hint: "[parameter-name]"
---

# Trace Parameter

Given a parameter name (provided as the argument, e.g. `/trace-parameter bone_low`), trace it through the entire codebase end-to-end. The argument is: $ARGUMENTS

## Steps

### 1. Find UI Definition
Search `app/main.py` for where this parameter is defined as a Streamlit widget (slider, number_input, selectbox, checkbox, etc.). Record:
- The widget type and default value
- The `st.session_state` key if used
- The variable name it's assigned to
- File and line number

### 2. Find Where It's Passed
Search for where the variable is passed to function calls. Trace through:
- Direct function arguments in `main.py`
- Any intermediate wrapper functions
- Session state lookups that use it indirectly
- Record each handoff as `file:line`

### 3. Find Where It's Consumed
In the target core module (`app/core/metal_detection.py`, `app/core/discrimination.py`, `app/contour_operations.py`, etc.), find where the parameter is actually used in a computation (comparison, multiplication, array indexing, conditional, etc.). Not just received as a function argument — actually *used*.

### 4. Report the Chain

Format the trace as a chain:

```
PARAMETER: <name>

UI DEFINITION:
  app/main.py:<line> — <widget_type>(label="...", min=..., max=..., default=...)

PASSED THROUGH:
  app/main.py:<line> → function_name(<name>=variable)
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
- If no argument is provided, scan ALL slider/number_input widgets in main.py and trace each one. Report a summary table.
- Pay special attention to parameters in the discrimination pipeline — these have known connectivity issues (see CLAUDE.md).
- Check both the "Russian Doll with Star Profile Discrimination" and "Russian Doll with Distance-Based Discrimination" code paths, as parameters may be connected in one but not the other.
