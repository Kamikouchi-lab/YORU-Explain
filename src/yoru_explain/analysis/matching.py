# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Finding the original detection again in a re-run detector's output.

Re-detection mode runs the whole detector, NMS included, on a perturbed
frame.  The output is a new list of boxes, and which of them -- if any -- is
"the same detection" has to be decided.  The decision is made only from the
original box and class, and every way it can go is a named status:

``matched``
    exactly one detection of the target class overlaps the original box with
    IoU >= threshold.  Its confidence is the target's score.
``ambiguous``
    two or more such detections.  The best-overlapping one is reported, but
    it is not counted as a score by default (which one is "the" target is a
    guess).
``class_changed``
    no detection of the target class overlaps enough, but one of another
    class does: the box is still there, labelled differently.
``unmatched``
    nothing overlaps the original box enough, but something overlaps it a
    little (the box moved, shrank, or merged with a neighbour).
``disappeared``
    nothing overlaps the original box at all.

Only ``matched`` gives a score.  The others leave it unknown: a vanished
detection's score is *not* 0 -- NMS may have suppressed it under a
neighbour, or it may sit just below the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from yoru_explain.core.geometry import iou
from yoru_explain.core.types import Detection, ScoreStatus


@dataclass(frozen=True)
class MatchResult:
    status: str
    #: The matched detection; for ``ambiguous`` the best of them; for
    #: ``class_changed`` / ``unmatched`` the best-overlapping other one.
    detection: Optional[Detection]
    iou: Optional[float]
    #: Same-class detections with IoU >= threshold.
    n_overlapping: int
    #: Best IoU of any detection with the original box (0 when none).
    best_any_iou: float


def match_redetection(original_box: Sequence[float], target_class: int,
                      detections: Sequence[Detection], iou_threshold: float) -> MatchResult:
    """Decide which of *detections* is the original detection, if any."""
    scored = [(iou(original_box, d.box), d) for d in detections]
    best_any = max((v for v, _ in scored), default=0.0)

    same = [(v, d) for v, d in scored if d.class_id == target_class and v >= iou_threshold]
    if same:
        # Highest IoU first; equal IoU: higher confidence.
        v, d = max(same, key=lambda vd: (vd[0], vd[1].score))
        status = ScoreStatus.MATCHED if len(same) == 1 else ScoreStatus.AMBIGUOUS
        return MatchResult(status.value, d, v, len(same), best_any)

    others = [(v, d) for v, d in scored if v >= iou_threshold]
    if others:
        v, d = max(others, key=lambda vd: (vd[0], vd[1].score))
        return MatchResult(ScoreStatus.CLASS_CHANGED.value, d, v, 0, best_any)

    overlapping = [(v, d) for v, d in scored if v > 0.0]
    if overlapping:
        v, d = max(overlapping, key=lambda vd: (vd[0], vd[1].score))
        return MatchResult(ScoreStatus.UNMATCHED.value, d, v, 0, best_any)

    return MatchResult(ScoreStatus.DISAPPEARED.value, None, None, 0, best_any)
