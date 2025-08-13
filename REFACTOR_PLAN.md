# Refactoring Plan for IVH-DO_DD Codebase

## Current Issues
1. **main.py** is 1,349 lines - too large and monolithic
2. **5 artifact discrimination files** with overlapping functionality
3. **3 metal detection versions** that should be consolidated
4. **visualization.py** is 675 lines - needs splitting
5. Poor code organization making it hard for students to work on specific features

## New Architecture

```
app/
├── __init__.py
├── main.py                    # Entry point (minimal, ~100 lines)
├── config.py                   # Keep as-is
│
├── core/                       # Core algorithms
│   ├── __init__.py
│   ├── metal_detection.py     # Unified metal detection
│   ├── discrimination.py      # Unified artifact discrimination
│   ├── segmentation.py        # Russian doll and other segmentation
│   └── refinement.py          # Post-processing and refinement
│
├── ui/                         # UI components
│   ├── __init__.py
│   ├── sidebar.py             # Sidebar controls
│   ├── tabs/
│   │   ├── __init__.py
│   │   ├── single_slice.py   # Single slice analysis tab
│   │   ├── multi_slice.py    # Multi-slice view tab
│   │   ├── metal_analysis.py # Metal detection details tab
│   │   ├── statistics.py     # Volume statistics tab
│   │   └── preview.py        # Threshold preview tab
│   └── components/
│       ├── __init__.py
│       ├── file_selector.py  # Patient/file selection
│       ├── threshold_controls.py # Threshold sliders
│       └── export_controls.py # Export options
│
├── visualization/              # Visualization modules
│   ├── __init__.py
│   ├── overlays.py           # CT overlays and masks
│   ├── histograms.py         # Intensity histograms
│   ├── profiles.py           # Star profile visualization
│   ├── multi_slice.py        # Multi-slice views
│   └── utils.py              # Visualization utilities
│
├── io/                        # Input/Output operations
│   ├── __init__.py
│   ├── dicom.py              # DICOM loading/saving (merge dicom_utils + dicom_export)
│   └── nifti.py              # NIFTI operations
│
├── algorithms/                # Algorithm implementations
│   ├── __init__.py
│   ├── metal/
│   │   ├── __init__.py
│   │   ├── legacy.py         # Legacy metal detection
│   │   ├── adaptive.py       # Adaptive percentile-based
│   │   └── adaptive_3d.py    # 3D multi-planar
│   ├── discrimination/
│   │   ├── __init__.py
│   │   ├── distance_based.py # Fast distance-based
│   │   ├── edge_based.py     # Enhanced edge analysis
│   │   ├── texture_based.py  # Advanced texture/gradient
│   │   └── star_profile.py   # Original star profile
│   └── operations/
│       ├── __init__.py
│       ├── boolean.py         # Boolean mask operations
│       └── morphology.py      # Morphological operations
│
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── state.py              # Session state management
│   ├── validation.py         # Input validation
│   └── helpers.py            # General helper functions
│
└── workflows/                  # Processing workflows
    ├── __init__.py
    ├── metal_workflow.py      # Metal detection workflow
    ├── segmentation_workflow.py # Segmentation workflow
    └── export_workflow.py     # Export workflow
```

## Implementation Steps

### Phase 1: Create Directory Structure
- Create all new directories
- Add __init__.py files
- Set up proper Python packages

### Phase 2: Break Down main.py
- Extract UI components to ui/ directory
- Move workflow logic to workflows/
- Keep only app initialization in main.py

### Phase 3: Consolidate Algorithms
- Merge 3 metal detection files into core/metal_detection.py with strategy pattern
- Merge 5 discrimination files into core/discrimination.py with pluggable algorithms
- Move algorithm implementations to algorithms/ directory

### Phase 4: Split Visualization
- Break visualization.py into focused modules
- Create reusable components
- Improve plot generation efficiency

### Phase 5: Clean Up
- Remove redundant files
- Update all imports
- Add proper documentation
- Test everything

## Benefits
1. **Easier to navigate** - Clear separation of concerns
2. **Easier to modify** - Each module has single responsibility
3. **Better for collaboration** - Students can work on specific modules
4. **More maintainable** - Less code duplication
5. **Better testing** - Modules can be tested independently

## Migration Strategy
1. Create new structure alongside old
2. Gradually move functionality
3. Test each module as migrated
4. Remove old files when complete
5. Update documentation