# Changelog — 2026-02-15

## Session 1: Planning & Branch Setup

**UI Text Fix (main branch)**
- Removed misleading "Combines coronal/sagittal analysis with star profile algorithm" message from the 3D Adaptive Metal Detection info box (`app/main.py:158`)
- Updated matching comment at line 780

**Cleanup**
- Deleted stale `kind-tesla` branch and its orphaned git worktree

**New Branch:** `feature/fast-slice-viewer` created with plan at `docs/plans/2026-02-15-fast-slice-viewer.md`

---

## Session 2: Implementation (`feature/fast-slice-viewer`)

### Completed (6 commits)

**1. Streamlit upgrade** (`5016ba3`)
- `requirements.txt`: `streamlit==1.29.0` → `streamlit>=1.37.0` (installed 1.54.0)
- Enables `@st.fragment` for partial page reruns

**2. Fast slice renderer** (`b909a7e`)
- New `fast_render_slice()` in `app/visualization.py`
- numpy HU windowing → PIL grayscale → PNG bytes (~10ms vs ~300ms matplotlib)
- Added `from PIL import Image` import

**3. Fast/Overlay view toggle** (`2ac4c38`)
- Added `fast_render_slice` import to `main.py`
- Two rendering paths: "Fast (CT Only)" uses `st.image()`, "Overlays" uses existing `create_overlay_image()` + `st.pyplot()`
- Old matplotlib simple preview path removed (replaced by fast renderer)

**4. @st.fragment isolation** (`9f3c998`)
- Extracted slice navigation + rendering into `slice_viewer()` decorated with `@st.fragment`
- Analysis buttons (Detect Metal, Segment Artifacts) stay outside fragment — trigger full page rerun
- `st.rerun()` calls changed to `st.rerun(scope="fragment")` inside viewer
- Fixed `current_slice` references in col2 to use `st.session_state.current_slice`

**5. Keyboard arrow navigation** (`67d68ed`)
- JavaScript injected via `st.components.v1.html`
- Arrow keys step through slices
- Added mouse wheel listener and debounced live slider drag

**6. UI cleanup** (`c02c16f`)
- Removed all nav buttons (⬅️➡️⏮️⏪⏩⏭️) — were buggy, caused view mode resets
- Removed redundant "Slice X of X" info text
- Moved View Mode radio to sidebar (top of Settings)
- JS rewritten to inject into parent page context (not iframe)
- Throttled keyboard at 120ms for held-key scrolling
- Image uses `width="stretch"`, deprecated `use_container_width` removed

### Outstanding Issues

**Scroll wheel on image not working**
- JS `e.preventDefault()` fires (page scroll blocked when hovering image) but `stepSlider()` doesn't change the slice
- Root cause: likely the `getSlider()` DOM query can't find the `input[type="range"]` element, OR the native value setter + event dispatch doesn't trigger Streamlit's React state update in v1.54
- User provided SO link for reference: https://stackoverflow.com/questions/74626851/python-streamlit-feature-for-interacting-with-displayed-images
- **Next session:** investigate Streamlit 1.54 slider DOM structure, consider custom component approach

**Arrow keys require slider focus**
- Custom JS keyboard handler may not be executing; the working behavior is native browser range-input keyboard support
- Holding key down doesn't scroll continuously (fragment rerun may cause focus loss)

### Files Modified
- `requirements.txt` — Streamlit version bump
- `app/visualization.py` — added `fast_render_slice()`, PIL import
- `app/main.py` — fragment viewer, view mode in sidebar, JS navigation, removed buttons

### Notes
- `app/cornerstone_viewer.py` is an orphaned file from a previous JS viewer attempt. Never integrated. Can be deleted.
- The bone_low/bone_high slider disconnection issue (documented in CLAUDE.md) is still outstanding.
- Data pipeline (numpy arrays, masks, DICOM/NIFTI export) completely unaffected — all changes are display-only.
