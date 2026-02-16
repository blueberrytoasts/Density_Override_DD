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

**Scroll wheel — parked (not fixable with current approach)**
- JS injection cannot reliably control Streamlit's React slider after `@st.fragment` reruns destroy/recreate the DOM element
- `_valueTracker` trick (React 16/17) doesn't work with React 18 in Streamlit 1.54
- All scroll wheel JS code has been removed; arrow keys work natively when slider has focus

### Files Modified (Session 2)
- `requirements.txt` — Streamlit version bump
- `app/visualization.py` — added `fast_render_slice()`, PIL import
- `app/main.py` — fragment viewer, view mode in sidebar, JS navigation, removed buttons

### Notes
- The bone_low/bone_high slider disconnection issue (documented in CLAUDE.md) is still outstanding.
- Data pipeline (numpy arrays, masks, DICOM/NIFTI export) completely unaffected — all changes are display-only.

---

## Session 3: UI Cleanup & Overlay Improvements (`feature/fast-slice-viewer`)

### Completed (1 commit: `60a71bb`)

**1. Remove broken scroll wheel JS**
- Removed entire JS injection block (scroll wheel + keyboard hack + DOM manipulation)
- Root cause confirmed: `@st.fragment` reruns destroy/recreate slider DOM, making cached JS references stale

**2. Delete orphaned cornerstone_viewer.py**
- Removed `app/cornerstone_viewer.py` (354 lines, never imported)
- Removed from README project structure

**3. Overlay rendering overhaul**
- Removed legend from matplotlib figure — now rendered as static HTML in col2
- Removed title from overlay figure ("CT Slice X with Characterized Regions" → just "Slice X" caption)
- Changed `create_overlay_image()` to use `fig.add_axes([0,0,1,1])` with black background — no white border
- Overlay now renders via `fig_to_png_bytes()` → `st.image()` instead of `st.pyplot()` — same sizing as Fast mode
- `fig_to_png_bytes()` now uses `fig.get_facecolor()` instead of hardcoded white

**4. Status messages → toast popups**
- All `st.success()`/`st.info()` boxes from detection and segmentation replaced with `st.toast()`
- Eliminates the stack of blue/green boxes that pushed the image down

**5. Auto-switch to Overlays mode**
- Detection and segmentation set `_switch_to_overlays` flag
- Sidebar radio widget consumes flag before rendering (avoids StreamlitAPIException)

**6. Restyled col2 (right panel)**
- Removed "Analysis Results" subheader
- "Adaptive Thresholds" and "Segmentation Statistics" now blue-colored smaller headings with grey data text
- Legend rendered as vertical column below statistics with a divider line
- Fixed `ct_slice` scope for histogram checkbox in col2

**7. Metal detection threshold change**
- Star profile metal filter: 50% → 75% of max HU in slice (comments updated)

### Files Modified (Session 3)
- `app/main.py` — toast messages, view mode auto-switch, col2 restyling, legend placement, removed JS
- `app/visualization.py` — removed legend/title from figure, black background, edge-to-edge axes
- `app/cornerstone_viewer.py` — deleted
- `app/core/metal_detection.py` — metal filter threshold 50% → 75%
- `README.md` — removed cornerstone_viewer from structure

---

## Session 4: Auto-switch fix + held arrow key attempts (`feature/fast-slice-viewer`)

### Completed

**1. Fix auto-switch to Overlays on first press**
- Added `st.rerun()` after metal detection and segmentation complete
- Previously required two presses because the `_switch_to_overlays` flag was set after the sidebar radio had already rendered in the same script run
- `st.rerun()` forces an immediate new run where the sidebar consumes the flag

### Attempted & Reverted

**Held arrow key continuous scrubbing**
- Attempted 3 approaches, all failed:
  1. **In-fragment refocus**: `components.v1.html` with `sl.focus()` — iframe loads too late, key repeat already lost
  2. **Persistent MutationObserver**: injected into parent document, tracked `arrowHeld` flag, refocused slider on DOM mutations — slider DOM destruction during fragment rerun kills key repeat before observer can act
  3. **Input→Change bridge**: dispatched `change` event on every `input` event to force Streamlit to commit each keystep — Streamlit's React wrapper doesn't respond to synthetic change events during held key
- **Root cause**: Streamlit 1.54 slider commits on release only. Fragment rerun destroys/recreates slider DOM, killing browser key repeat. No client-side workaround found.
- **All JS removed** — slider works with single arrow key presses only
- **Possible future fix**: custom Streamlit component with built-in keyboard handling

### Files Modified (Session 4)
- `app/main.py` — added `st.rerun()` after detection/segmentation, removed all held-key JS attempts
