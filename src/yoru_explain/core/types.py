# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""The plain data the whole application passes around.

Every type here is a frozen dataclass with ``to_dict`` / ``from_dict``, so
that what a run used can be written to JSON/CSV and read back unchanged.
None of them imports torch or Qt.

Coordinates
    All boxes and maps are in :data:`COORDINATE_SYSTEM`: pixels of the frame
    the detector was given -- after any flip YORU applied -- with the origin
    at the top-left corner, x to the right and y down.  A box is
    ``(x1, y1, x2, y2)`` exactly as YORU's ``DetectorBase.detect`` reports it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

#: Name of the one coordinate system boxes and maps are stored in.
COORDINATE_SYSTEM = "frame_px_xyxy"
COORDINATE_SYSTEM_DESCRIPTION = (
    "pixels of the frame given to the detector (after any v_flip/h_flip), "
    "origin top-left, x right, y down; boxes are (x1, y1, x2, y2) as YORU reports them"
)

#: The scores an adapter may be asked to explain, and what each one is.
#: ``obj_x_cls`` is what YOLOv5 reports as a detection's confidence.
SCORE_TYPES: Dict[str, str] = {
    "obj_x_cls": "sigmoid(objectness) x sigmoid(class logit) -- the confidence YOLOv5 reports",
    "cls_prob": "sigmoid(class logit) -- class probability given an object",
    "obj_prob": "sigmoid(objectness) -- object probability, class-agnostic",
    "cls_logit": "class logit, before the sigmoid",
    "obj_logit": "objectness logit, before the sigmoid",
}

#: How the explained detection is followed from one input to the next.
IDENTITY_MODES: Dict[str, str] = {
    "fixed_candidate": (
        "the same pre-NMS candidate (anchor slot) is read on every input; its box may move"
    ),
    "redetect": (
        "the detector is run end to end and its output is matched to the original "
        "detection by class and IoU; no score when no unique match"
    ),
}

Box = Tuple[float, float, float, float]


def finite_or_none(value) -> Optional[float]:
    """``float(value)``, or ``None`` for missing / NaN / infinite values."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _box(values: Sequence[float]) -> Box:
    x1, y1, x2, y2 = (float(v) for v in values)
    return (x1, y1, x2, y2)


@dataclass(frozen=True)
class CandidateRef:
    """One pre-NMS candidate of a detector: what fixed-candidate mode follows.

    *index* is the row in the adapter's raw candidate tensor; *locator* says
    where that row sits in the network (for YOLOv5: level, stride, anchor,
    grid_y, grid_x) and is kept for reports only.
    """

    index: int
    locator: Tuple[Tuple[str, int], ...] = ()

    @property
    def locator_dict(self) -> Dict[str, int]:
        return dict(self.locator)

    def describe(self) -> str:
        loc = ", ".join(f"{k}={v}" for k, v in self.locator)
        return f"#{self.index}" + (f" ({loc})" if loc else "")

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "locator": dict(self.locator)}

    @classmethod
    def from_dict(cls, d: Optional[Mapping[str, Any]]) -> Optional["CandidateRef"]:
        if not d:
            return None
        loc = d.get("locator") or {}
        return cls(int(d["index"]), tuple((str(k), int(v)) for k, v in loc.items()))


@dataclass(frozen=True)
class Detection:
    """One detection as the detector reported it, in frame pixels."""

    box: Box
    score: float
    class_id: int
    class_name: str
    candidate: Optional[CandidateRef] = None

    @property
    def x1(self) -> float:
        return self.box[0]

    @property
    def y1(self) -> float:
        return self.box[1]

    @property
    def x2(self) -> float:
        return self.box[2]

    @property
    def y2(self) -> float:
        return self.box[3]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "box": list(self.box),
            "score": self.score,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "candidate": self.candidate.to_dict() if self.candidate else None,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Detection":
        return cls(
            box=_box(d["box"]),
            score=float(d["score"]),
            class_id=int(d["class_id"]),
            class_name=str(d["class_name"]),
            candidate=CandidateRef.from_dict(d.get("candidate")),
        )


@dataclass(frozen=True)
class FrameRef:
    """Which frame of which input a target lives on.

    *frame_index* counts frames from 0 in decoding order -- the ``frame``
    column of YORU's video analysis.  *frame_sha1* is a hash of the decoded
    (and flipped) pixels, so a later run can tell whether it decoded the same
    picture.  *access* says how the frame was read: ``sequential`` (decoded
    in order from the start: exact), ``seek_verified`` (a seek checked against
    the presentation times of an in-order pass), ``seek`` (positioned by the
    container index: usually right, not guaranteed), ``image``, or
    ``unread`` (imported from a detections file; no hash yet).
    """

    source_id: str
    path: str
    kind: str
    frame_index: int
    width: int
    height: int
    timestamp_ms: Optional[float] = None
    v_flip: bool = False
    h_flip: bool = False
    frame_sha1: Optional[str] = None
    access: str = "sequential"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "path": self.path,
            "kind": self.kind,
            "frame_index": self.frame_index,
            "width": self.width,
            "height": self.height,
            "timestamp_ms": finite_or_none(self.timestamp_ms),
            "v_flip": self.v_flip,
            "h_flip": self.h_flip,
            "frame_sha1": self.frame_sha1,
            "access": self.access,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FrameRef":
        return cls(
            source_id=str(d["source_id"]),
            path=str(d["path"]),
            kind=str(d["kind"]),
            frame_index=int(d["frame_index"]),
            width=int(d["width"]),
            height=int(d["height"]),
            timestamp_ms=finite_or_none(d.get("timestamp_ms")),
            v_flip=bool(d.get("v_flip", False)),
            h_flip=bool(d.get("h_flip", False)),
            frame_sha1=d.get("frame_sha1") or None,
            access=str(d.get("access", "sequential")),
        )


@dataclass(frozen=True)
class ScoreDefinition:
    """What number is being explained, and how the target is followed."""

    score_type: str = "obj_x_cls"
    identity_mode: str = "fixed_candidate"
    #: Re-detection: minimum IoU with the original box for a match.
    iou_threshold: float = 0.5

    def __post_init__(self):
        if self.score_type not in SCORE_TYPES:
            raise ValueError(f"unknown score_type {self.score_type!r}; one of {sorted(SCORE_TYPES)}")
        if self.identity_mode not in IDENTITY_MODES:
            raise ValueError(
                f"unknown identity_mode {self.identity_mode!r}; one of {sorted(IDENTITY_MODES)}")
        if not 0.0 < float(self.iou_threshold) <= 1.0:
            raise ValueError(f"iou_threshold must be in (0, 1], got {self.iou_threshold}")

    @property
    def is_probability(self) -> bool:
        return not self.score_type.endswith("_logit")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score_type": self.score_type,
            "identity_mode": self.identity_mode,
            "iou_threshold": float(self.iou_threshold),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ScoreDefinition":
        unknown = set(d) - {"score_type", "identity_mode", "iou_threshold"}
        if unknown:
            raise ValueError(f"unknown score keys: {sorted(unknown)}")
        return cls(**dict(d))


@dataclass(frozen=True)
class ExplanationTarget:
    """One detection chosen for explanation: the thing every score refers to.

    *detection* is the detection as the adapter reproduced it on *frame*
    (with its pre-NMS candidate when the adapter can recover it) and
    *detection_index* its position in that frame's detection list.
    *target_class* is the class whose score is explained -- the detected
    class unless asked otherwise.  *transform* is how the adapter maps the
    frame to the model input and back (for YOLOv5 the letterbox:
    ``LetterboxTransform.to_dict()``), filled in when the target is
    reproduced on its frame.
    """

    event_id: str
    frame: FrameRef
    detection: Detection
    detection_index: int
    target_class: int
    model_id: str = ""
    track_id: Optional[int] = None
    origin: str = "adapter"
    coordinate_system: str = COORDINATE_SYSTEM
    transform: Optional[Mapping[str, Any]] = field(default=None, hash=False)

    @property
    def box(self) -> Box:
        return self.detection.box

    @property
    def candidate(self) -> Optional[CandidateRef]:
        return self.detection.candidate

    def with_detection(self, detection: Detection, detection_index: int) -> "ExplanationTarget":
        return replace(self, detection=detection, detection_index=detection_index)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "frame": self.frame.to_dict(),
            "detection": self.detection.to_dict(),
            "detection_index": self.detection_index,
            "target_class": self.target_class,
            "model_id": self.model_id,
            "track_id": self.track_id,
            "origin": self.origin,
            "coordinate_system": self.coordinate_system,
            "transform": dict(self.transform) if self.transform else None,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ExplanationTarget":
        return cls(
            event_id=str(d["event_id"]),
            frame=FrameRef.from_dict(d["frame"]),
            detection=Detection.from_dict(d["detection"]),
            detection_index=int(d["detection_index"]),
            target_class=int(d["target_class"]),
            model_id=str(d.get("model_id", "")),
            track_id=None if d.get("track_id") in (None, "") else int(d["track_id"]),
            origin=str(d.get("origin", "adapter")),
            coordinate_system=str(d.get("coordinate_system", COORDINATE_SYSTEM)),
            transform=dict(d["transform"]) if d.get("transform") else None,
        )


class ScoreStatus(str, Enum):
    """What happened when a target was scored on one (perturbed) input."""

    #: Fixed-candidate: the candidate's score was read.
    OK = "ok"
    #: Re-detection: exactly one same-class detection overlaps the target.
    MATCHED = "matched"
    #: Re-detection: two or more same-class detections overlap the target.
    AMBIGUOUS = "ambiguous"
    #: Re-detection: only detections of another class overlap the target.
    CLASS_CHANGED = "class_changed"
    #: Re-detection: something overlaps the target, but below the IoU threshold.
    UNMATCHED = "unmatched"
    #: Re-detection: nothing overlaps the target box at all.
    DISAPPEARED = "disappeared"
    #: The score came out NaN or infinite.
    NON_FINITE = "non_finite"
    #: Scoring failed; see the error code.
    ERROR = "error"


#: Statuses whose value is the target's score and enters delta maps.
SCORED_STATUSES = frozenset({ScoreStatus.OK.value, ScoreStatus.MATCHED.value})


@dataclass(frozen=True)
class ScoreResult:
    """The target's score on one input, or why there is none.

    *value* is ``None`` whenever the status is not a scored one: a target
    that was lost has an unknown score, not a score of 0.
    """

    status: str
    value: Optional[float] = None
    error_code: Optional[str] = None
    message: str = ""
    #: Re-detection: IoU of the matched detection with the original box.
    #: Fixed-candidate: IoU of the candidate's own (moved) box with the original.
    iou: Optional[float] = None
    #: Re-detection: the matched detection.
    matched: Optional[Detection] = None
    #: The candidate the value was read from.
    candidate_index: Optional[int] = None
    #: Fixed-candidate: the candidate's box on this input, frame pixels.
    candidate_box: Optional[Box] = None
    #: Fixed-candidate: whether NMS still keeps the candidate on this input.
    candidate_kept: Optional[bool] = None
    #: Fixed-candidate: the class the candidate would be labelled with now.
    candidate_class: Optional[int] = None
    #: Re-detection: number of same-class detections with IoU >= threshold.
    n_overlapping: int = 0
    #: Re-detection, ``ambiguous`` only: the score of the best-overlapping of
    #: the competing detections.  Not the target's score (*value* stays
    #: ``None``); occlusion uses it only when asked to (``include_ambiguous``).
    ambiguous_value: Optional[float] = None

    @property
    def scored(self) -> bool:
        return self.status in SCORED_STATUSES and self.value is not None


@dataclass(frozen=True)
class ModelRecord:
    """A model file a project knows, identified by its hash."""

    model_id: str
    path: str
    sha256: str
    adapter: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "path": self.path,
            "sha256": self.sha256,
            "adapter": self.adapter,
            "options": dict(self.options),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ModelRecord":
        return cls(str(d["model_id"]), str(d["path"]), str(d["sha256"]),
                   str(d["adapter"]), dict(d.get("options") or {}))
