# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Colour tables for maps, as 256x3 BGR uint8 arrays (OpenCV order).

Signed maps (occlusion deltas) use a diverging table centred on 0 with a
symmetric range, so equal colours mean equal magnitudes on both sides and
white means "no change".  Non-negative maps (Grad-CAM) use viridis, which is
perceptually ordered and readable with colour-vision deficiencies.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

# Anchor colours in RGB: matplotlib's coolwarm and viridis at a few stops.
_COOLWARM = ((0.0, (59, 76, 192)), (0.25, (141, 176, 254)), (0.5, (221, 221, 221)),
             (0.75, (244, 154, 123)), (1.0, (180, 4, 38)))
_VIRIDIS = ((0.0, (68, 1, 84)), (0.25, (59, 82, 139)), (0.5, (33, 145, 140)),
            (0.75, (94, 201, 98)), (1.0, (253, 231, 37)))


def _lut(anchors: Sequence[Tuple[float, Tuple[int, int, int]]]) -> np.ndarray:
    xs = np.array([a for a, _ in anchors])
    rgb = np.array([c for _, c in anchors], dtype=np.float64)
    t = np.linspace(0.0, 1.0, 256)
    out = np.stack([np.interp(t, xs, rgb[:, i]) for i in range(3)], axis=1)
    return np.rint(out[:, ::-1]).astype(np.uint8)  # RGB -> BGR


DIVERGING = _lut(_COOLWARM)
SEQUENTIAL = _lut(_VIRIDIS)


def colorize(values: np.ndarray, vmin: float, vmax: float, lut: np.ndarray):
    """``(bgr image, valid mask)`` for *values*; NaN pixels are invalid."""
    v = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(v)
    span = (vmax - vmin) if vmax > vmin else 1.0
    idx = np.zeros(v.shape, dtype=np.int64)
    idx[valid] = np.clip(np.rint((v[valid] - vmin) / span * 255), 0, 255).astype(np.int64)
    bgr = lut[idx]
    bgr[~valid] = 0
    return bgr, valid


def symmetric_range(values: np.ndarray) -> Tuple[float, float]:
    v = np.asarray(values, dtype=np.float64)
    m = float(np.nanmax(np.abs(v))) if np.isfinite(v).any() else 0.0
    m = m if m > 0 else 1.0
    return -m, m


def positive_range(values: np.ndarray) -> Tuple[float, float]:
    v = np.asarray(values, dtype=np.float64)
    m = float(np.nanmax(v)) if np.isfinite(v).any() else 0.0
    return 0.0, (m if m > 0 else 1.0)
