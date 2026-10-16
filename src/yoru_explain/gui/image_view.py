# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""A frame with clickable detection boxes, and the map display modes."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from yoru_explain.gui.qtutil import bgr_to_qimage

VIEW_MODES = ("Original", "Heatmap", "Overlay", "Side by side")


class FrameView(QtWidgets.QGraphicsView):
    """Shows one image in frame pixels; boxes are drawn over it.

    ``clicked`` reports the frame pixel under the mouse.  In "Side by side"
    mode the image is three frames wide and clicks are mapped back to the
    first panel.
    """

    clicked = QtCore.pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(QtGui.QColor(40, 40, 40))
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self._pixmap = self.scene().addPixmap(QtGui.QPixmap())
        self._items: List[QtWidgets.QGraphicsItem] = []
        self._frame_width = 0
        self._panels = 1
        self._fitted = False

    def set_image(self, image: Optional[np.ndarray], *, frame_width: Optional[int] = None,
                  panels: int = 1) -> None:
        if image is None:
            self._pixmap.setPixmap(QtGui.QPixmap())
            return
        self._pixmap.setPixmap(QtGui.QPixmap.fromImage(bgr_to_qimage(image)))
        self._frame_width = frame_width or image.shape[1]
        self._panels = panels
        self.scene().setSceneRect(QtCore.QRectF(0, 0, image.shape[1], image.shape[0]))
        if not self._fitted:
            self.fit()

    def fit(self) -> None:
        if not self._pixmap.pixmap().isNull():
            self.fitInView(self._pixmap, QtCore.Qt.KeepAspectRatio)
            self._fitted = True

    def set_boxes(self, boxes: Sequence, *, selected: Optional[int] = None,
                  labels: Sequence[str] = (), dashed: Sequence = ()) -> None:
        for it in self._items:
            self.scene().removeItem(it)
        self._items = []
        for p in range(self._panels):
            dx = p * self._frame_width
            for i, box in enumerate(boxes):
                sel = i == selected
                pen = QtGui.QPen(QtGui.QColor(255, 60, 60) if sel else QtGui.QColor(255, 220, 0))
                pen.setWidthF(3.0 if sel else 1.6)
                pen.setCosmetic(True)
                x1, y1, x2, y2 = box
                r = self.scene().addRect(QtCore.QRectF(x1 + dx, y1, x2 - x1, y2 - y1), pen)
                self._items.append(r)
                if p == 0 and i < len(labels) and labels[i]:
                    t = self.scene().addSimpleText(labels[i])
                    t.setBrush(QtGui.QBrush(pen.color()))
                    t.setPos(x1 + dx, max(0.0, y1 - 16))
                    t.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
                    self._items.append(t)
            for box in dashed:
                pen = QtGui.QPen(QtGui.QColor(120, 200, 255))
                pen.setStyle(QtCore.Qt.DashLine)
                pen.setCosmetic(True)
                x1, y1, x2, y2 = box
                self._items.append(self.scene().addRect(QtCore.QRectF(x1 + dx, y1, x2 - x1, y2 - y1), pen))

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event):
        self.fit()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self._frame_width:
            p = self.mapToScene(event.pos())
            x = p.x() % self._frame_width if self._panels > 1 else p.x()
            self.clicked.emit(float(x), float(p.y()))
        super().mousePressEvent(event)


def compose(frame: np.ndarray, values: Optional[np.ndarray], kind: str, mode: str,
            alpha: float = 0.5):
    """The image to show for *mode*; ``(image, panels)``."""
    from yoru_explain.visualization.overlay import heatmap, overlay

    if values is None or mode == "Original":
        return frame, 1
    heat, valid, _ = heatmap(values, kind)
    if mode == "Heatmap":
        return heat, 1
    over = overlay(frame, heat, valid, alpha)
    if mode == "Overlay":
        return over, 1
    return np.concatenate([frame, heat, over], axis=1), 3


def pick_box(boxes: Sequence, x: float, y: float, margin: float = 6.0) -> Optional[int]:
    """The smallest box containing ``(x, y)``, else the nearest within *margin*."""
    inside = [(abs((b[2] - b[0]) * (b[3] - b[1])), i) for i, b in enumerate(boxes)
              if b[0] <= x <= b[2] and b[1] <= y <= b[3]]
    if inside:
        return min(inside)[1]
    best = None
    for i, b in enumerate(boxes):
        dx = max(b[0] - x, 0, x - b[2])
        dy = max(b[1] - y, 0, y - b[3])
        d = (dx * dx + dy * dy) ** 0.5
        if d <= margin and (best is None or d < best[0]):
            best = (d, i)
    return best[1] if best else None
