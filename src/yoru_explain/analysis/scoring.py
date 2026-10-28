# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""The target's score on a batch of inputs, in either identity mode.

Fixed-candidate mode reads the target's own pre-NMS candidate -- the same
anchor slot -- on every input.  The value is always defined, and two
diagnostics say what the detector would have made of it: whether NMS still
keeps the candidate (``candidate_kept``) and how far its box moved
(``iou`` with the original box).

Re-detection mode runs NMS and matches its output to the original detection
(:mod:`yoru_explain.analysis.matching`).  The score is the matched
detection's; there is none when the match fails.

The highest score of the class anywhere in the frame is never used: with two
animals of one class in view it would follow whichever is stronger.
"""

from __future__ import annotations

import math
from typing import List, Optional

from yoru_explain.analysis.matching import match_redetection
from yoru_explain.core.errors import ErrorCode
from yoru_explain.core.geometry import overlap
from yoru_explain.core.types import (
    Detection,
    ExplanationTarget,
    ScoreDefinition,
    ScoreResult,
    ScoreStatus,
)


def _finite(value: Optional[float]) -> bool:
    return value is not None and math.isfinite(value)


def score_batch(adapter, raw, prepared, target: ExplanationTarget, score: ScoreDefinition,
                detections: Optional[List[List[Detection]]] = None) -> List[ScoreResult]:
    """One :class:`ScoreResult` per image of *prepared*."""
    n = prepared.batch_size
    caps = adapter.capabilities
    if score.score_type not in caps.score_types:
        return [ScoreResult(ScoreStatus.ERROR.value, error_code=ErrorCode.MODEL_UNSUPPORTED.value,
                            message=f"{adapter.name} cannot read score {score.score_type!r}")] * n
    if score.identity_mode not in caps.identity_modes:
        return [ScoreResult(ScoreStatus.ERROR.value, error_code=ErrorCode.MODEL_UNSUPPORTED.value,
                            message=f"{adapter.name} cannot follow a target in "
                                    f"{score.identity_mode} mode")] * n
    if detections is None:
        detections = adapter.postprocess(raw, prepared)
    if score.identity_mode == "fixed_candidate":
        return _fixed_candidate(adapter, raw, prepared, target, score, detections)
    return _redetect(adapter, raw, prepared, target, score, detections)


def _kept(detections, cand, target, multi_label: bool) -> bool:
    """Whether NMS still reports the candidate (under the target's class, for multi-label models)."""
    return any(d.candidate is not None and d.candidate.index == cand.index
               and (not multi_label or d.class_id == target.target_class)
               for d in detections)


def _fixed_candidate(adapter, raw, prepared, target, score, detections) -> List[ScoreResult]:
    n = prepared.batch_size
    cand = target.candidate
    if cand is None:
        return [ScoreResult(
            ScoreStatus.ERROR.value, error_code=ErrorCode.SCORE_UNAVAILABLE.value,
            message="the target has no pre-NMS candidate; use re-detection mode")] * n
    readouts = adapter.candidate_readout(raw, prepared, cand.index, target.target_class,
                                         score.score_type)
    multi = adapter.capabilities.multi_label
    out = []
    for b in range(n):
        r = readouts[b]
        kept = _kept(detections[b], cand, target, multi)
        box_iou = overlap(r.box, target.box, r.obb, target.obb) if r.box is not None else None
        if not _finite(r.value):
            out.append(ScoreResult(
                ScoreStatus.NON_FINITE.value, value=None, error_code=ErrorCode.NON_FINITE.value,
                message=f"candidate {cand.index} scored {r.value!r}", iou=box_iou,
                candidate_index=cand.index, candidate_box=r.box, candidate_obb=r.obb,
                candidate_kept=kept, candidate_class=r.argmax_class))
            continue
        out.append(ScoreResult(
            ScoreStatus.OK.value, value=float(r.value), iou=box_iou,
            candidate_index=cand.index, candidate_box=r.box, candidate_obb=r.obb,
            candidate_kept=kept, candidate_class=r.argmax_class))
    return out


def _redetect(adapter, raw, prepared, target, score, detections) -> List[ScoreResult]:
    out = []
    reported = adapter.capabilities.reports(score.score_type)
    for b in range(prepared.batch_size):
        m = match_redetection(target.box, target.target_class, detections[b],
                              score.iou_threshold, target.obb)
        if m.status not in (ScoreStatus.MATCHED.value, ScoreStatus.AMBIGUOUS.value):
            out.append(ScoreResult(m.status, value=None, iou=m.iou, matched=m.detection,
                                   n_overlapping=m.n_overlapping))
            continue
        d = m.detection
        if reported:
            # The matched detection's own confidence: its class is the target class.
            value = d.score
        elif d.candidate is None:
            out.append(ScoreResult(
                ScoreStatus.ERROR.value, error_code=ErrorCode.SCORE_UNAVAILABLE.value,
                message=f"{score.score_type} needs the matched detection's candidate",
                iou=m.iou, matched=d, n_overlapping=m.n_overlapping))
            continue
        else:
            value = adapter.candidate_readout(raw, prepared, d.candidate.index,
                                              target.target_class, score.score_type)[b].value
        if not _finite(value):
            out.append(ScoreResult(
                ScoreStatus.NON_FINITE.value, error_code=ErrorCode.NON_FINITE.value,
                message=f"matched detection scored {value!r}", iou=m.iou, matched=d,
                candidate_index=d.candidate.index if d.candidate else None,
                n_overlapping=m.n_overlapping))
            continue
        ambiguous = m.status == ScoreStatus.AMBIGUOUS.value
        out.append(ScoreResult(
            m.status, value=None if ambiguous else float(value), iou=m.iou, matched=d,
            candidate_index=d.candidate.index if d.candidate else None,
            n_overlapping=m.n_overlapping,
            ambiguous_value=float(value) if ambiguous else None))
    return out
