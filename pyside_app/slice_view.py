"""Interactive 2D slice viewer (axial) built on QGraphicsView.

Interactions (the things Streamlit could not do well):
    - Mouse wheel:            scrub through slices
    - Ctrl + mouse wheel:     zoom in/out at the cursor
    - Left-drag:              pan
    - Right-drag:             adjust window/level (x = level, y = width)
    - Up/Down or Left/Right:  step one slice
    - "F" key:                fit image to view

Rendering goes through ``hu_to_qimage`` so each interaction only re-wraps one
slice's NumPy buffer -- no full-app rerun, no server round-trip.
"""
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QPainter
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsPixmapItem

from pyside_app.hu_render import (
    hu_to_qimage, hu_to_qimage_with_overlay, auto_window_level,
)


class SliceView(QGraphicsView):
    """Displays one axial slice of a 3D HU volume with scrub + window/level."""

    # idx, total
    slice_changed = Signal(int, int)
    # center, width
    window_changed = Signal(float, float)
    # HU value under the cursor (int), or None when outside the image
    cursor_hu = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        self._volume: np.ndarray | None = None
        self._mask: np.ndarray | None = None  # 3D overlay mask, same shape as volume
        self._index = 0
        self._center = 40.0
        self._width = 400.0

        # Window/level drag state (right button).
        self._wl_dragging = False
        self._last_pos = None

        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)  # left-drag pans
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(Qt.GlobalColor.black)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

    # ---- public API -----------------------------------------------------
    def set_volume(self, volume: np.ndarray) -> None:
        """Load a new volume (z, y, x), reset to the middle slice + auto W/L."""
        self._volume = volume
        self._mask = None  # a new volume invalidates any previous overlay
        self._index = volume.shape[0] // 2
        self._center, self._width = auto_window_level(volume)
        self._render(fit=True)
        self.slice_changed.emit(self._index, volume.shape[0])
        self.window_changed.emit(self._center, self._width)

    def set_mask(self, mask: np.ndarray | None) -> None:
        """Set (or clear, with None) the 3D overlay mask and redraw.

        ``mask`` must share the volume's (z, y, x) shape. Passing None removes
        the overlay and returns to plain grayscale.
        """
        if mask is not None and self._volume is not None:
            if mask.shape != self._volume.shape:
                raise ValueError(
                    f"mask shape {mask.shape} != volume shape {self._volume.shape}"
                )
        self._mask = mask
        self._render()

    def set_slice(self, index: int) -> None:
        if self._volume is None:
            return
        index = int(np.clip(index, 0, self._volume.shape[0] - 1))
        if index != self._index:
            self._index = index
            self._render()
            self.slice_changed.emit(self._index, self._volume.shape[0])

    def set_window(self, center: float, width: float) -> None:
        self._center = float(center)
        self._width = max(float(width), 1.0)
        self._render()
        self.window_changed.emit(self._center, self._width)

    def hu_at(self, scene_x: float, scene_y: float):
        """Return the HU value at scene coordinates, or None if out of bounds."""
        if self._volume is None:
            return None
        x, y = int(scene_x), int(scene_y)
        _, h, w = self._volume.shape
        if 0 <= x < w and 0 <= y < h:
            return int(self._volume[self._index, y, x])
        return None

    # ---- rendering ------------------------------------------------------
    def _render(self, fit: bool = False) -> None:
        if self._volume is None:
            return
        slice_hu = self._volume[self._index]
        if self._mask is None:
            image = hu_to_qimage(slice_hu, self._center, self._width)
        else:
            image = hu_to_qimage_with_overlay(
                slice_hu, self._center, self._width, self._mask[self._index]
            )
        self._pixmap_item.setPixmap(QPixmap.fromImage(image))
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        if fit:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    # ---- events ---------------------------------------------------------
    def wheelEvent(self, event):
        if self._volume is None:
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.25 if event.angleDelta().y() > 0 else 1 / 1.25
            self.scale(factor, factor)
        else:
            step = 1 if event.angleDelta().y() > 0 else -1
            self.set_slice(self._index + step)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self._wl_dragging = True
            self._last_pos = event.position()
            self.setCursor(Qt.CursorShape.SizeAllCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._wl_dragging and self._last_pos is not None:
            delta = event.position() - self._last_pos
            self._last_pos = event.position()
            # Horizontal drag shifts level, vertical drag changes width.
            self.set_window(self._center + delta.x() * 2.0,
                            self._width + delta.y() * 4.0)
            event.accept()
            return
        # Report the HU value under the cursor for the status bar.
        scene_pos = self.mapToScene(event.position().toPoint())
        self.cursor_hu.emit(self.hu_at(scene_pos.x(), scene_pos.y()))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton and self._wl_dragging:
            self._wl_dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Up, Qt.Key.Key_Right):
            self.set_slice(self._index + 1)
        elif key in (Qt.Key.Key_Down, Qt.Key.Key_Left):
            self.set_slice(self._index - 1)
        elif key == Qt.Key.Key_F:
            self._render(fit=True)
        else:
            super().keyPressEvent(event)
