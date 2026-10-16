"""Frames for the synthetic detector, with answers derived by hand.

The toy detector (``yoru_explain.adapters.toy``) puts a candidate on every
16-pixel cell of the letterboxed input; candidate (gy, gx) sees the 32x32
input window ``[16*gx - 8, 16*gx + 24) x [16*gy - 8, 16*gy + 24)``.  A frame
whose sides are multiples of 32 and whose long side is the input size is not
resized or padded, so input and frame pixels coincide.
"""

from __future__ import annotations

import numpy as np

STRIDE = 16
WINDOW = 32
RED = (0, 0, 255)
GREEN = (0, 255, 0)


def blank(h: int = 192, w: int = 320) -> np.ndarray:
    return np.zeros((h, w, 3), np.uint8)


def window(gy: int, gx: int, pad_top: int = 0, pad_left: int = 0):
    """Frame ``(x0, y0, x1, y1)`` of candidate (gy, gx)'s window."""
    x0 = STRIDE * gx - (WINDOW - STRIDE) // 2 - pad_left
    y0 = STRIDE * gy - (WINDOW - STRIDE) // 2 - pad_top
    return (x0, y0, x0 + WINDOW, y0 + WINDOW)


def paint(frame: np.ndarray, box, bgr) -> np.ndarray:
    x0, y0, x1, y1 = (int(v) for v in box)
    frame[max(0, y0):y1, max(0, x0):x1] = bgr
    return frame


def overlaps(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def target_on(adapter, frame, box, **kw):
    """The target made from the detection whose box is *box*."""
    from yoru_explain import api

    dets = adapter.detect(frame)
    for i, d in enumerate(dets):
        if tuple(round(v, 3) for v in d.box) == tuple(float(v) for v in box):
            return api.target_from_detection(d, i, frame, **kw)
    raise AssertionError(f"no detection at {box}: {dets}")
