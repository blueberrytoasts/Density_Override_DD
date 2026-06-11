# PySide desktop app (work in progress)

A native (Qt/PySide6) front-end for CT metal-artifact characterization. It
**reuses the existing algorithm code** under `app/core/` and `app/` — only the
UI layer is being rebuilt. The goal is the responsiveness Streamlit couldn't
give us: instant slice scrubbing, real window/level, pan/zoom, and background
computation that never freezes the UI.

## Run it

```bash
# from the repo root, with the project's Python environment active
python -m pyside_app
```

Then click **Load Patient…** and pick a patient folder (e.g. `data/HIP4 Patient`).
The loader automatically finds the CT series subfolder (it's the one with the
most files).

## Controls

| Action | Result |
|---|---|
| Mouse wheel | scrub through slices |
| Ctrl + wheel | zoom at cursor |
| Left-drag | pan |
| Right-drag | window/level (x = level, y = width) |
| ↑/↓ or ←/→ | step one slice |
| `F` | fit image to view |
| Slider (bottom) | jump to slice |

The status bar shows the HU value under the cursor.

## Current status (v0.1)

- [x] Background DICOM loading (worker thread, responsive UI)
- [x] Fast slice viewer (NumPy → QImage, sub-ms per slice)
- [x] Window/level, zoom/pan, slice scrubbing, HU readout
- [ ] Metal detection (wire up `core/metal_detection.py`)
- [ ] Artifact segmentation + discrimination
- [ ] Mask overlays (color-coded, toggleable)
- [ ] Export (NIFTI / DICOM RT — reuse `dicom_export.py`)

## Architecture

```
pyside_app/
  __main__.py     # entry point: python -m pyside_app
  main_window.py  # QMainWindow: toolbar, viewer, slider, status bar
  slice_view.py   # QGraphicsView viewer (scrub / W-L / zoom / pan)
  dicom_loader.py # CT-folder resolver + threaded load worker
  hu_render.py    # HU + window/level -> QImage (the hot path)
  bootstrap.py    # puts app/ on sys.path to reuse existing modules
```

The next step is a **controller/pipeline layer** so detection/segmentation can
be triggered from here exactly as the Streamlit app does, without duplicating
logic. See the migration plan in the project docs.
