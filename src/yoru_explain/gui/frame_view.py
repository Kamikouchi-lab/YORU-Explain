# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""A frame with clickable detection boxes, and the map display modes."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

from yoru_explain.gui import theme as th
from yoru_explain.gui.widgets import texture_data

VIEW_MODES = ("Original", "Heatmap", "Overlay", "Side by side")

#: Largest texture uploaded.  Larger images are shown downscaled; their
#: coordinates stay frame pixels, so boxes, clicks and readouts are exact.
MAX_TEXTURE = (2048, 1536)


class FrameView:
    """One image in frame pixels on a plot: wheel zooms, drag pans, double-click fits.

    The plot's axes are frame pixels (y downwards), so boxes are drawn and
    clicks read in the detector's own coordinates.  *on_click* gets the frame
    pixel under the mouse.  In "Side by side" mode the image is three frames
    wide and clicks are mapped back to the first panel.
    """

    def __init__(self, tag: str, themes: th.Themes,
                 on_click: Optional[Callable[[float, float], None]] = None):
        self.tag = tag
        self.themes = themes
        self.on_click = on_click
        self.x_axis = f"{tag}_x"
        self.y_axis = f"{tag}_y"
        self._registry = None
        self._texture = None
        self._texture_size: Optional[Tuple[int, int]] = None
        self._series = None
        self._items: List[int] = []
        #: ``(width, height)`` of the image shown, in frame pixels; None when empty.
        self.image_size: Optional[Tuple[int, int]] = None
        self.frame_width = 0
        self.panels = 1

    def build(self, texture_registry, width: int = -1, height: int = -1) -> None:
        self._registry = texture_registry
        with dpg.plot(tag=self.tag, width=width, height=height, equal_aspects=True, no_menus=True):
            dpg.add_plot_axis(dpg.mvXAxis, tag=self.x_axis)
            dpg.add_plot_axis(dpg.mvYAxis, tag=self.y_axis, invert=True)
        with dpg.item_handler_registry() as handlers:
            dpg.add_item_clicked_handler(button=dpg.mvMouseButton_Left,
                                         callback=lambda: self._clicked())
        dpg.bind_item_handler_registry(self.tag, handlers)

    def resize(self, width: int, height: int) -> None:
        dpg.configure_item(self.tag, width=max(64, int(width)), height=max(64, int(height)))

    # -- contents ------------------------------------------------------------

    def set_image(self, image: Optional[np.ndarray], *, frame_width: Optional[int] = None,
                  panels: int = 1) -> None:
        if image is None:
            if self._series is not None:
                dpg.configure_item(self._series, show=False)
            self.image_size = None
            self.set_boxes(())
            return
        h, w = image.shape[:2]
        scale = min(1.0, MAX_TEXTURE[0] / w, MAX_TEXTURE[1] / h)
        tw, tht = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        shown = image if scale == 1.0 else cv2.resize(image, (tw, tht), interpolation=cv2.INTER_AREA)
        if self._texture_size != (tw, tht):
            if self._series is not None:
                dpg.delete_item(self._series)
            if self._texture is not None:
                dpg.delete_item(self._texture)
            self._texture = dpg.add_dynamic_texture(tw, tht, texture_data(shown), parent=self._registry)
            first = dpg.get_item_children(self.y_axis, 1) or []
            kwargs = {"before": first[0]} if first else {}
            # ImPlot puts the texture's first row at bounds_max.y, which the
            # inverted (downward) y axis draws at the bottom: flip the UVs.
            self._series = dpg.add_image_series(self._texture, [0, 0], [w, h], uv_min=[0, 1],
                                                uv_max=[1, 0], parent=self.y_axis, **kwargs)
            self._texture_size = (tw, tht)
        else:
            dpg.set_value(self._texture, texture_data(shown))
            dpg.configure_item(self._series, bounds_max=[w, h], show=True)
        refit = self.image_size != (w, h)
        self.image_size = (w, h)
        self.frame_width = int(frame_width or w)
        self.panels = int(panels)
        if refit:
            self.fit()

    def fit(self) -> None:
        if self.image_size is not None:
            dpg.fit_axis_data(self.x_axis)
            dpg.fit_axis_data(self.y_axis)

    def set_boxes(self, boxes: Sequence, *, selected: Optional[int] = None,
                  labels: Sequence[str] = (), dashed: Sequence = ()) -> None:
        for it in self._items:
            if dpg.does_item_exist(it):
                dpg.delete_item(it)
        self._items = []
        if self.image_size is None:
            return
        for p in range(self.panels):
            dx = p * self.frame_width
            for box in dashed:
                self._rect(box, dx, self.themes.box_imported)
            for i, box in enumerate(boxes):
                if i != selected:
                    self._rect(box, dx, self.themes.box)
            if selected is not None and 0 <= selected < len(boxes):
                self._rect(boxes[selected], dx, self.themes.box_selected)
            if p:
                continue
            for i, box in enumerate(boxes):
                if i < len(labels) and labels[i]:
                    color = th.BOX_SELECTED if i == selected else th.BOX
                    self._items.append(dpg.add_plot_annotation(
                        label=labels[i], default_value=(float(box[0]), float(box[1])),
                        offset=(1, -2), color=(*color, 210), clamped=True, parent=self.tag))

    def _rect(self, box, dx: float, theme) -> None:
        x1, y1, x2, y2 = (float(v) for v in box)
        s = dpg.add_line_series([x1 + dx, x2 + dx, x2 + dx, x1 + dx, x1 + dx],
                                [y1, y1, y2, y2, y1], parent=self.y_axis)
        dpg.bind_item_theme(s, theme)
        self._items.append(s)

    @property
    def box_count(self) -> int:
        return sum(1 for it in self._items
                   if dpg.does_item_exist(it) and "LineSeries" in dpg.get_item_type(it))

    # -- the mouse -------------------------------------------------------------

    def mouse_frame_pos(self) -> Optional[Tuple[float, float]]:
        """The frame pixel under the mouse (first panel), or None off the image."""
        if self.image_size is None or not dpg.is_item_hovered(self.tag):
            return None
        x, y = dpg.get_plot_mouse_pos()
        w, h = self.image_size
        if not (0 <= x < w and 0 <= y < h):
            return None
        return (x % self.frame_width if self.panels > 1 else x), y

    def _clicked(self) -> None:
        pos = self.mouse_frame_pos()
        if pos is not None and self.on_click is not None:
            self.on_click(*pos)


def compose(frame: np.ndarray, values: Optional[np.ndarray], kind: str, mode: str,
            alpha: float = 0.5):
    """The image to show for *mode*; ``(image, panels, vrange)``."""
    from yoru_explain.visualization.overlay import heatmap, overlay

    if values is None or mode == "Original":
        return frame, 1, None
    heat, valid, vrange = heatmap(values, kind)
    if mode == "Heatmap":
        return heat, 1, vrange
    over = overlay(frame, heat, valid, alpha)
    if mode == "Overlay":
        return over, 1, vrange
    return np.concatenate([frame, heat, over], axis=1), 3, vrange


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


def map_value(values: Optional[np.ndarray], x: float, y: float) -> Optional[float]:
    """The map's value at frame pixel ``(x, y)``; None outside or where undefined."""
    if values is None:
        return None
    xi, yi = int(np.floor(x)), int(np.floor(y))
    if not (0 <= yi < values.shape[0] and 0 <= xi < values.shape[1]):
        return None
    v = float(values[yi, xi])
    return v if np.isfinite(v) else None
