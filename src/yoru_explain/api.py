# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""The Python API: everything the GUI does, without the GUI.

::

    from yoru_explain import api

    adapter = api.load_adapter("yoru-yolov5", "best.pt", device="auto")
    project = api.Project.open_or_create("my_project")
    source = project.add_source("video.mp4")
    model = project.add_model("best.pt", "yoru-yolov5")
    event = api.add_event(project, adapter, model.model_id, source["source_id"],
                          frame_index=120, detection_index=0)
    project.save()
    outcome = api.run_analysis(project, adapter, model.model_id, [event],
                               api.AnalysisRequest(method="occlusion"))

Settings are dataclasses (:class:`OcclusionConfig`, :class:`GradCamConfig`,
:class:`AnalysisRequest`) with ``to_dict`` / ``from_dict``, so they can be
kept in JSON.  There is no command line.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from yoru_explain.core.errors import ErrorCode, ExplainError
from yoru_explain.core.pipeline import (
    AnalysisRequest,
    Progress,
    RunOutcome,
    analyze_frame,
    describe_score,
    reproduce_target,
    run_analysis,
)
from yoru_explain.core.registry import create_adapter, list_adapters
from yoru_explain.core.types import (
    Detection,
    ExplanationTarget,
    FrameRef,
    ScoreDefinition,
)
from yoru_explain.io.project import Project
from yoru_explain.io.runs import RunReader
from yoru_explain.methods.baselines import BaselineSpec
from yoru_explain.methods.gradcam import GradCamConfig
from yoru_explain.methods.occlusion import OcclusionConfig

__all__ = [
    "AnalysisRequest", "BaselineSpec", "Detection", "ExplanationTarget", "FrameRef",
    "GradCamConfig", "OcclusionConfig", "Progress", "Project", "RunOutcome", "RunReader",
    "ScoreDefinition", "add_event", "add_events_from_detections_csv", "analyze_frame",
    "describe_score", "explain",
    "list_adapters", "load_adapter", "reproduce_target", "run_analysis", "target_from_detection",
]


def load_adapter(name: str = "yoru-yolov5", weights=None, device: str = "auto", **options):
    """A loaded adapter (see :mod:`yoru_explain.core.registry` for the names)."""
    adapter = create_adapter(name)
    adapter.load_model(weights, device=device, **options)
    return adapter


def target_from_detection(detection: Detection, detection_index: int, image: np.ndarray, *,
                          source_id: str = "image", path: str = "", kind: str = "image",
                          frame_index: int = 0, timestamp_ms: Optional[float] = None,
                          v_flip: bool = False, h_flip: bool = False, event_id: str = "",
                          model_id: str = "", target_class: Optional[int] = None,
                          access: str = "image", frame_sha1: Optional[str] = None,
                          origin: str = "adapter") -> ExplanationTarget:
    from yoru_explain.core.hashing import frame_sha1 as _sha1

    h, w = image.shape[:2]
    frame = FrameRef(source_id, path, kind, int(frame_index), w, h, timestamp_ms, v_flip, h_flip,
                     frame_sha1 or _sha1(image), access)
    return ExplanationTarget(event_id, frame, detection, int(detection_index),
                             detection.class_id if target_class is None else int(target_class),
                             model_id=model_id, origin=origin)


def add_event(project: Project, adapter, model_id: str, source_id: str, frame_index: int,
              detection_index: int, *, note: str = "", target_class: Optional[int] = None) -> str:
    """Detect on a frame (decoded in order) and add one detection as an event."""
    s = project.source(source_id)
    with project.open_source(source_id) as reader:
        frame = reader.read(frame_index, exact=True)
    dets = adapter.detect(frame.image)
    if not 0 <= detection_index < len(dets):
        raise ExplainError(ErrorCode.MISSING_DATA,
                           f"frame {frame_index} has {len(dets)} detection(s); no index {detection_index}")
    target = target_from_detection(
        dets[detection_index], detection_index, frame.image, source_id=source_id, path=s["path"],
        kind=s["kind"], frame_index=frame_index, timestamp_ms=frame.timestamp_ms,
        v_flip=s["v_flip"], h_flip=s["h_flip"], model_id=model_id, target_class=target_class,
        access=frame.access, frame_sha1=frame.sha1)
    return project.add_event(target, note)


def add_events_from_detections_csv(project: Project, source_id: str, csv_path, *, model_id: str = "",
                                   class_names=None, frames=None, limit: Optional[int] = None,
                                   note: str = "") -> list:
    """Add the detections of a YORU (or YORU-Tracker) detections CSV as events.

    Such events carry neither a pre-NMS candidate nor a frame hash
    (``frame_access="unread"``): a run decodes the frame, runs the detector and
    must find each box again (same class, IoU >= 0.9), or reports
    ``target_not_reproduced``.  *class_names* keeps only those classes,
    *frames* only those frame indices, *limit* at most that many events.
    Returns the new event ids (the project is not saved).
    """
    from yoru_explain.io.detections_csv import load_detections_csv

    s = project.source(source_id)
    _, by_frame = load_detections_csv(csv_path)
    keep_classes = set(class_names) if class_names else None
    keep_frames = set(int(f) for f in frames) if frames is not None else None
    added = []
    for frame_index in sorted(by_frame):
        if keep_frames is not None and frame_index not in keep_frames:
            continue
        for imported in by_frame[frame_index]:
            d = imported.detection
            if keep_classes is not None and d.class_name not in keep_classes:
                continue
            if limit is not None and len(added) >= limit:
                return added
            frame = FrameRef(source_id, s["path"], s["kind"], frame_index, int(s["width"]),
                             int(s["height"]), None, bool(s["v_flip"]), bool(s["h_flip"]), None, "unread")
            target = ExplanationTarget("", frame, d, -1, d.class_id, model_id=model_id,
                                       track_id=imported.track_id, origin="csv_import")
            added.append(project.add_event(target, note))
    return added


def explain(adapter, image: np.ndarray, target: ExplanationTarget,
            request: Optional[AnalysisRequest] = None, *, reproduce: bool = True,
            progress=None, cancel=None):
    """Run one method on one frame and return the result object (no files).

    With *reproduce* the target is first found again on *image*
    (:func:`reproduce_target`).
    """
    request = request or AnalysisRequest()
    if reproduce:
        target, _ = reproduce_target(adapter, image, target)
    return analyze_frame(adapter, image, target, request, progress=progress, cancel=cancel)
