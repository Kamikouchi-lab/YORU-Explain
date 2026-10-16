"""Pictures: NaN stays visibly undefined, scales are symmetric for deltas."""

from __future__ import annotations

import numpy as np

from yoru_explain.visualization.colormaps import DIVERGING, colorize, symmetric_range
from yoru_explain.visualization.overlay import heatmap, overlay, triptych


def test_symmetric_scale_and_white_zero():
    m = np.array([[-0.2, 0.0, 0.5]])
    assert symmetric_range(m) == (-0.5, 0.5)
    bgr, valid = colorize(m, -0.5, 0.5, DIVERGING)
    assert valid.all()
    assert (bgr[0, 1] > 200).all()  # 0 is (near) white


def test_nan_is_not_painted_as_zero():
    m = np.full((4, 4), np.nan, np.float32)
    m[1, 1] = 0.0
    heat, valid, _ = heatmap(m, "delta")
    assert valid.sum() == 1
    frame = np.full((4, 4, 3), 10, np.uint8)
    out = overlay(frame, heat, valid)
    assert (out[0, 0] == 10).all()           # undefined pixels keep the frame
    assert not (out[1, 1] == 10).all()


def test_triptych_shape():
    frame = np.zeros((48, 64, 3), np.uint8)
    m = np.zeros((48, 64), np.float32)
    m[10:20, 10:20] = 0.3
    img = triptych(frame, m, "delta", (10, 10, 20, 20), label="solo 0.9", lines=["a", "b"])
    assert img.shape[1] == 3 * 64 and img.shape[0] > 48
