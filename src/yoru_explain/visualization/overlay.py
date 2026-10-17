# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Pictures of maps: heatmap, overlay, and the preview PNG of a result.

These are for looking at.  The numbers live in ``maps/*.npz`` and the CSV
tables; a preview never rescales or smooths what it shows beyond the colour
scale printed under it.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from yoru_explain.visualization.colormaps import (
    DIVERGING,
    SEQUENTIAL,
    colorize,
    positive_range,
    symmetric_range,
)

#: How each kind of map is coloured: (lut, range function, label).
MAP_STYLES = {
    "delta": (DIVERGING, symmetric_range, "score change when occluded (s_ref - s)"),
    "cam": (SEQUENTIAL, positive_range, "Grad-CAM (ReLU), model units"),
}

_TEXT = (235, 235, 235)
_BG = (32, 32, 32)


def _ascii(text: str) -> str:
    """cv2.putText draws ASCII only."""
    return "".join(c if 32 <= ord(c) < 127 else "?" for c in str(text))


def heatmap(values: np.ndarray, kind: str, vrange: Optional[Tuple[float, float]] = None):
    """``(bgr, valid, (vmin, vmax))`` for a map of *kind* (``delta`` / ``cam``)."""
    lut, rng, _ = MAP_STYLES[kind]
    vmin, vmax = vrange if vrange is not None else rng(values)
    bgr, valid = colorize(values, vmin, vmax, lut)
    bgr[~valid] = (96, 96, 96)
    return bgr, valid, (vmin, vmax)


def overlay(frame: np.ndarray, heat: np.ndarray, valid: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """*heat* over *frame* where the map is defined; the frame elsewhere."""
    base = frame if frame.ndim == 3 else np.repeat(frame[..., None], 3, axis=2)
    out = base.astype(np.float32).copy()
    m = valid[..., None]
    out = np.where(m, (1 - alpha) * out + alpha * heat.astype(np.float32), out)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def draw_box(image: np.ndarray, box: Sequence[float], color=(0, 255, 255), label: str = "",
             thickness: int = 2, obb: Optional[Sequence[float]] = None) -> np.ndarray:
    """*box* on *image* -- or, for a rotated-box model, its rotated box *obb*."""
    import cv2

    x1, y1, x2, y2 = (int(round(v)) for v in box)
    if obb is not None:
        from yoru_explain.core.geometry import obb_corners

        pts = np.round(obb_corners(obb)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [pts], True, color, thickness, cv2.LINE_AA)
    else:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    if label:
        text = _ascii(label)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y = max(th + 2, y1 - 3)
        cv2.rectangle(image, (x1, y - th - 2), (x1 + tw + 2, y + base - 1), color, -1)
        cv2.putText(image, text, (x1 + 1, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1,
                    cv2.LINE_AA)
    return image


def colorbar(width: int, kind: str, vrange: Tuple[float, float], height: int = 34) -> np.ndarray:
    import cv2

    lut, _, label = MAP_STYLES[kind]
    bar = np.full((height, width, 3), _BG, np.uint8)
    x0, x1 = 60, max(61, width - 60)
    idx = np.clip(np.linspace(0, 255, x1 - x0).round().astype(int), 0, 255)
    bar[4:16, x0:x1] = lut[idx][None, :, :]
    vmin, vmax = vrange
    cv2.putText(bar, f"{vmin:.3g}", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _TEXT, 1, cv2.LINE_AA)
    cv2.putText(bar, f"{vmax:.3g}", (x1 + 4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _TEXT, 1, cv2.LINE_AA)
    cv2.putText(bar, _ascii(label), (x0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _TEXT, 1, cv2.LINE_AA)
    return bar


def text_block(width: int, lines: List[str]) -> np.ndarray:
    import cv2

    h = 8 + 16 * len(lines)
    block = np.full((h, width, 3), _BG, np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(block, _ascii(line)[:200], (6, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    _TEXT, 1, cv2.LINE_AA)
    return block


def triptych(frame: np.ndarray, values: np.ndarray, kind: str, box: Sequence[float],
             label: str = "", lines: Sequence[str] = (), roi: Optional[Sequence[int]] = None,
             alpha: float = 0.5, obb: Optional[Sequence[float]] = None) -> np.ndarray:
    """Original | heatmap | overlay, with the colour scale and *lines* of text."""
    import cv2

    heat, valid, vrange = heatmap(values, kind)
    left = draw_box(frame.copy(), box, label=label, obb=obb)
    middle = draw_box(heat.copy(), box, obb=obb)
    right = draw_box(overlay(frame, heat, valid, alpha), box, obb=obb)
    if roi is not None:
        for img in (left, middle, right):
            cv2.rectangle(img, (int(roi[0]), int(roi[1])), (int(roi[2]) - 1, int(roi[3]) - 1),
                          (200, 200, 200), 1, cv2.LINE_4)
    row = np.concatenate([left, middle, right], axis=1)
    parts = [row, colorbar(row.shape[1], kind, vrange)]
    if lines:
        parts.append(text_block(row.shape[1], list(lines)))
    return np.concatenate(parts, axis=0)
