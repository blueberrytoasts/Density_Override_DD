# Fast Slice Viewer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make slice-by-slice navigation fast enough for visual QA (~50-100ms per slice change instead of ~300-500ms)

**Architecture:** Replace matplotlib rendering with direct numpy→PIL conversion for the fast path. Upgrade Streamlit to support `@st.fragment` so only the viewer re-renders on slice change (not the entire page). Add keyboard arrow navigation. Overlay rendering stays as-is but only triggers on-demand.

**Tech Stack:** Streamlit 1.37+, Pillow (PIL), numpy, existing matplotlib for overlay path

---

### Task 1: Upgrade Streamlit

**Files:**
- Modify: `requirements.txt:2`

**Step 1: Update version**

In `requirements.txt`, change:
```
streamlit==1.29.0
```
to:
```
streamlit>=1.37.0
```

`@st.fragment` was introduced in 1.33.0 but 1.37+ has stability fixes.

**Step 2: Install and verify**

Run: `pip install -r requirements.txt --upgrade`

Then verify: `python -c "import streamlit; print(streamlit.__version__)"`

Expected: Version 1.37.0 or higher.

**Step 3: Quick smoke test**

Run: `cd app && streamlit run main.py`

Verify the app loads without errors. Check that the sidebar and slice slider still work.

**Step 4: Commit**

```bash
git add requirements.txt
git commit -m "Upgrade Streamlit for @st.fragment support"
```

---

### Task 2: Create fast slice renderer

**Files:**
- Modify: `app/visualization.py`

**Step 1: Add fast_render_slice function to visualization.py**

Add this function after the existing imports (after line 6):

```python
from PIL import Image

def fast_render_slice(ct_slice, window_center=50, window_width=400):
    """
    Render a CT slice to PNG bytes using numpy windowing + PIL.
    Much faster than matplotlib (~10ms vs ~300ms).

    Args:
        ct_slice: 2D numpy array of HU values
        window_center: center of display window (HU)
        window_width: width of display window (HU)

    Returns:
        bytes: PNG image data ready for st.image()
    """
    # Apply windowing (convert HU to 0-255 grayscale)
    vmin = window_center - window_width / 2
    vmax = window_center + window_width / 2
    windowed = np.clip((ct_slice - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)

    # Convert to PIL Image and encode as PNG
    img = Image.fromarray(windowed, mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf.getvalue()
```

**Step 2: Verify it works standalone**

Run a quick test in Python:
```bash
python -c "
import numpy as np
from visualization import fast_render_slice
test = np.random.uniform(-150, 250, (512, 512)).astype(np.float32)
result = fast_render_slice(test)
print(f'Output size: {len(result)} bytes, type: {type(result)}')
"
```

Expected: Output like `Output size: ~50000 bytes, type: <class 'bytes'>`

**Step 3: Commit**

```bash
git add app/visualization.py
git commit -m "Add fast numpy-to-PNG slice renderer"
```

---

### Task 3: Add fast view toggle and integrate fast renderer

**Files:**
- Modify: `app/main.py:34` (add import)
- Modify: `app/main.py:~1082-1159` (viewer rendering section)

**Step 1: Add import**

At `main.py:34`, add `fast_render_slice` to the visualization import:

```python
from visualization import (create_overlay_image, create_histogram, fig_to_base64,
                            fig_to_png_bytes, fast_render_slice)
```

**Step 2: Add view mode toggle**

Before the display visualization section (around line 1082), add a toggle:

```python
# View mode toggle
view_mode = st.radio(
    "View Mode",
    ["Fast (CT Only)", "Overlays"],
    horizontal=True,
    key="view_mode",
    help="Fast mode for quick scrubbing. Overlays mode shows contours."
)
```

**Step 3: Add fast rendering path**

Replace the rendering section (lines ~1082-1159). The logic should be:

- If `view_mode == "Fast (CT Only)"`: use `fast_render_slice()` → `st.image()`
- If `view_mode == "Overlays"`: use existing `create_overlay_image()` → `st.pyplot(fig)` path (unchanged)
- If no masks exist (simple preview): also use `fast_render_slice()` → `st.image()`

For the fast path:
```python
if view_mode == "Fast (CT Only)" or not st.session_state.masks:
    png_bytes = fast_render_slice(ct_slice, window_center=50, window_width=400)
    st.image(png_bytes, caption=f"Slice {current_slice + 1}", use_container_width=True)
else:
    # existing overlay path (unchanged)
    ...
```

**Step 4: Verify both modes work**

Run the app, load a patient, and:
1. Toggle "Fast (CT Only)" — slices should render noticeably faster
2. Toggle "Overlays" — masks should display as before
3. Run detection, then toggle back and forth

**Step 5: Commit**

```bash
git add app/main.py
git commit -m "Add fast CT-only view mode for quick slice scrubbing"
```

---

### Task 4: Wrap viewer in @st.fragment

**Files:**
- Modify: `app/main.py`

**Step 1: Identify fragment boundary**

The slice viewer section (navigation controls + image display) needs to be extracted into a function decorated with `@st.fragment`. This includes:
- Slice slider and navigation buttons (~695-765)
- Slice info display (~770-771)
- The rendering section (both fast and overlay paths)

The analysis buttons (detect metal, segment artifacts) should stay OUTSIDE the fragment since they modify global state.

**Step 2: Create the fragment function**

Wrap the viewer portion in a fragment:

```python
@st.fragment
def slice_viewer():
    # Slice navigation controls (slider, buttons)
    # ... existing navigation code ...

    # Get current slice data
    ct_slice = st.session_state.ct_volume[current_slice]

    # View mode toggle
    # ... toggle code ...

    # Rendering (fast or overlay)
    # ... rendering code ...
```

Call it where the navigation code currently lives:

```python
slice_viewer()
```

**Important notes:**
- `@st.fragment` means only this function re-runs when its widgets change
- The rest of the page (sidebar, analysis buttons) does NOT re-run
- Session state is shared, so the fragment can read masks and detection results
- Widgets inside the fragment need unique keys (they already have them)

**Step 3: Test fragment isolation**

1. Change slice with slider — should be fast, sidebar should NOT flicker
2. Click "Detect Metal" — should still work (outside fragment, triggers full rerun)
3. Toggle view mode inside fragment — should only re-render the viewer

**Step 4: Commit**

```bash
git add app/main.py
git commit -m "Isolate slice viewer with @st.fragment for faster reruns"
```

---

### Task 5: Add keyboard arrow navigation

**Files:**
- Modify: `app/main.py` (inside the fragment function)

**Step 1: Add keyboard listener**

Streamlit doesn't have native keyboard support, but we can inject a small JavaScript snippet. Add this inside the fragment, after the navigation buttons:

```python
# Keyboard navigation (arrow keys)
st.components.v1.html("""
<script>
document.addEventListener('keydown', function(e) {
    if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
        // Navigate to previous slice
        const slider = window.parent.document.querySelector('[data-testid="stSlider"] input');
        if (slider) {
            slider.value = Math.max(0, parseInt(slider.value) - 1);
            slider.dispatchEvent(new Event('input', { bubbles: true }));
            slider.dispatchEvent(new Event('change', { bubbles: true }));
        }
    } else if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
        // Navigate to next slice
        const slider = window.parent.document.querySelector('[data-testid="stSlider"] input');
        if (slider) {
            slider.value = parseInt(slider.value) + 1;
            slider.dispatchEvent(new Event('input', { bubbles: true }));
            slider.dispatchEvent(new Event('change', { bubbles: true }));
        }
    }
});
</script>
""", height=0)
```

**Note:** This approach manipulates the slider DOM element directly. It may need adjustment based on the Streamlit version's DOM structure. If it doesn't work reliably, we can fall back to using `streamlit-js-eval` or a custom component.

**Step 2: Test keyboard navigation**

1. Click somewhere on the page (to ensure focus)
2. Press Up/Down arrow keys — slice should change
3. Press Left/Right arrow keys — slice should also change
4. Verify it doesn't interfere with other inputs (text fields, etc.)

**Step 3: Commit**

```bash
git add app/main.py
git commit -m "Add keyboard arrow navigation for slice scrubbing"
```

---

### Task 6: Clean up and final testing

**Files:**
- Modify: `app/main.py` (remove old simple preview path if redundant)

**Step 1: Remove redundant code**

The old simple preview path (lines ~1143-1159) that uses matplotlib just for `ax.imshow(ct_slice, cmap='gray')` is now handled by `fast_render_slice`. Remove it if it's no longer reachable.

**Step 2: Full integration test**

Test the complete workflow:
1. Load patient → slices render in fast mode
2. Scrub slider → slices change quickly
3. Arrow keys → slices step smoothly
4. Run metal detection → masks appear in session state
5. Toggle to "Overlays" → full matplotlib overlay renders
6. Toggle back to "Fast" → instant CT-only view
7. Export overlay PNG → still works
8. Export DICOM/NIFTI → still works (unaffected)

**Step 3: Commit**

```bash
git add app/main.py
git commit -m "Clean up redundant preview path, finalize fast viewer"
```
