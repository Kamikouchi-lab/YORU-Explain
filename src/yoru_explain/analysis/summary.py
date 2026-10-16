# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Per-event numbers that describe one result.

These are descriptive values of one detection's analysis -- where the
largest effect was, how much of the map lies inside the box.  Nothing here
compares events or tests anything: frames of one video are not independent
samples, and a population analysis needs videos, animals and days as its
units (planned for v0.2).
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from yoru_explain.core.types import ScoreStatus


def _box_mask(shape, box) -> np.ndarray:
    h, w = shape
    x1, y1, x2, y2 = box
    mask = np.zeros((h, w), bool)
    xa, ya = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
    xb, yb = min(w, int(math.ceil(x2))), min(h, int(math.ceil(y2)))
    if xb > xa and yb > ya:
        mask[ya:yb, xa:xb] = True
    return mask


def _nanmean(values: np.ndarray) -> float:
    v = values[np.isfinite(values)]
    return float(v.mean()) if v.size else float("nan")


def occlusion_summary(result) -> Dict[str, Any]:
    counts = result.status_counts()
    counted = [p for p in result.patches if p.counted]
    out: Dict[str, Any] = {
        "n_perturbations": len(result.patches),
        "n_counted": len(counted),
        "n_ok": counts.get(ScoreStatus.OK.value, 0),
        "n_matched": counts.get(ScoreStatus.MATCHED.value, 0),
        "n_ambiguous": counts.get(ScoreStatus.AMBIGUOUS.value, 0),
        "n_class_changed": counts.get(ScoreStatus.CLASS_CHANGED.value, 0),
        "n_unmatched": counts.get(ScoreStatus.UNMATCHED.value, 0),
        "n_disappeared": counts.get(ScoreStatus.DISAPPEARED.value, 0),
        "n_non_finite": counts.get(ScoreStatus.NON_FINITE.value, 0),
        "n_error": counts.get(ScoreStatus.ERROR.value, 0),
        # Fixed-candidate: scored, but NMS would no longer report the candidate.
        "n_candidate_dropped": sum(1 for p in result.patches
                                   if p.score.status == ScoreStatus.OK.value
                                   and p.score.candidate_kept is False),
    }
    if counted:
        best = max(counted, key=lambda p: p.delta)
        out.update(
            delta_score=best.delta,
            perturbed_score=best.used_score,
            max_delta_patch_id=best.patch.patch_id,
            delta_min=min(p.delta for p in counted),
            delta_mean=float(np.mean([p.delta for p in counted])),
        )
    if result.delta_map is not None:
        mask = _box_mask(result.delta_map.shape, result.target.box)
        out["in_box_mean_delta"] = _nanmean(result.delta_map[mask])
        out["out_box_mean_delta"] = _nanmean(result.delta_map[~mask])
    noop = result.controls.get("noop")
    if noop is not None:
        out["noop_delta"] = noop.delta
    full = result.controls.get("full_region")
    if full is not None:
        out["full_region_score"] = full.used_score
        out["full_region_status"] = full.score.status
    return out


def gradcam_summary(result) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "gradcam_layer": result.layer,
        "gradcam_stride": result.layer_stride,
        "gradient_support": result.gradient_support,
        "gradient_score": result.gradient_score,
    }
    if result.cam is not None:
        cam = result.cam.astype(np.float64)
        total = float(cam.sum())
        mask = _box_mask(cam.shape, result.target.box)
        out["cam_max"] = float(cam.max())
        out["cam_in_box_fraction"] = float(cam[mask].sum() / total) if total > 0 else float("nan")
    return out
