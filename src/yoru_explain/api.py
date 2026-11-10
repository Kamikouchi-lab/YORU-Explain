# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""The Python API: everything the GUI does, without the GUI.

::

    from yoru_explain import api

    adapter = api.load_model("best.pt")       # any YORU model; or api.load_adapter("yoru-yolov5", ...)
    project = api.Project.open_or_create("my_project")
    source = project.add_source("video.mp4")
    model = project.add_model("best.pt", adapter.name)
    event = api.add_event(project, adapter, model.model_id, source["source_id"],
                          frame_index=120, detection_index=0)
    project.save()
    outcome = api.run_analysis(project, adapter, model.model_id, [event],
                               api.AnalysisRequest(method="occlusion"))
    # Both methods, as the GUI's Both methods buttons: Grad-CAM, then occlusion.
    outcomes = api.run_analyses(project, adapter, model.model_id, [event],
                                [api.AnalysisRequest(method=m) for m in api.BOTH_METHODS])

    # For your own analysis: tables, maps on the box grid, frame crops, a gallery.
    project.set_unit(source["source_id"], "fly-01")     # what counts as one independent sample
    export = api.export_run(project, outcome.run_dir)   # -> my_project/exports/<run_id>/
    # Maps turned to orientations you measured (event_id, cx, cy, angle_deg, length[, directed]).
    aligned = api.align_maps(export.export_dir, "my_orientations.csv", out_dir="aligned")

Settings are dataclasses (:class:`OcclusionConfig`, :class:`GradCamConfig`,
:class:`AnalysisRequest`, :class:`ExportOptions`) with ``to_dict`` /
``from_dict``, so they can be kept in JSON.  There is no command line.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from yoru_explain.core.errors import ErrorCode, ExplainError
from yoru_explain.core.pipeline import (
    BOTH_METHODS,
    AnalysisRequest,
    Progress,
    RunOutcome,
    analyze_frame,
    describe_score,
    reproduce_target,
    run_analyses,
    run_analysis,
)
from yoru_explain.analysis.normalize import AlignedMaps, Orientation, align_maps, read_orientations
from yoru_explain.core.export import ExportOptions, ExportOutcome, export_run
from yoru_explain.core.registry import create_adapter, list_adapters, resolve_model
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
from yoru_explain.visualization.gallery import render_gallery

__all__ = [
    "BOTH_METHODS", "AlignedMaps", "AnalysisRequest", "BaselineSpec", "Detection", "ExplanationTarget",
    "ExportOptions", "ExportOutcome", "FrameRef", "GradCamConfig", "OcclusionConfig", "Orientation",
    "Progress", "Project", "RunOutcome", "RunReader",
    "ScoreDefinition", "add_event", "add_events_from_detections_csv", "align_maps", "analyze_frame",
    "describe_score", "explain", "export_run",
    "list_adapters", "load_adapter", "load_model", "min_gap_frames", "read_orientations",
    "render_gallery", "reproduce_target", "resolve_model",
    "run_analyses", "run_analysis", "target_from_detection",
]


def load_adapter(name: str = "yoru-yolov5", weights=None, device: str = "auto", **options):
    """A loaded adapter (see :mod:`yoru_explain.core.registry` for the names)."""
    adapter = create_adapter(name)
    adapter.load_model(weights, device=device, **options)
    return adapter


def load_model(weights, backend: str = "auto", device: str = "auto", *, black_box: bool = False,
               **options):
    """The adapter for a model file of any YORU backend, loaded.

    ``backend`` is a YORU backend name (``auto``, ``yolov5``, ``ultralytics``,
    ``rtdetr``, ``torchvision``, ``onnx``); ``auto`` picks what YORU's
    ``get_detector("auto", ...)`` would.  ``black_box=True`` explains YORU's
    ``detect()`` only (re-detection; no Grad-CAM).  *options* go to the
    adapter (``conf_thresh``, ``iou_thresh``; ``rgb_input`` for YOLOv5).
    """
    name, _backend, extra = resolve_model(weights, backend, black_box=black_box)
    return load_adapter(name, weights, device=device, **extra, **options)


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
                                   per_class: Optional[dict] = None, order: str = "first",
                                   seed: Optional[int] = None, note: str = "",
                                   min_gap_s: Optional[float] = None) -> list:
    """Add the detections of a YORU (or YORU-Tracker) detections CSV as events.

    Such events carry neither a pre-NMS candidate nor a frame hash
    (``frame_access="unread"``): a run decodes the frame, runs the detector and
    must find each box again (same class, IoU >= 0.9), or reports
    ``target_not_reproduced``.  *class_names* keeps only those classes,
    *frames* only those frame indices, *limit* at most that many events --
    or *per_class* ``{class_name: n}``, at most *n* events of each class.

    ``order="first"`` takes them from the first frame on.  ``order="random"``
    draws them at random with *seed* (the same seed, file and project give
    the same events), *n* boxes of a class from *n* different frames where
    the video has them; a box that is an event of this source already (same
    frame and class, IoU >= 0.9) is not drawn again, so a second draw adds
    other boxes.  See :func:`yoru_explain.io.detections_csv.choose_detections`.

    *min_gap_s* keeps the events of one class at least that many seconds
    apart (converted with the video's frame rate), counting the events this
    source has already: neighbouring frames are nearly the same picture and
    add no independent information.  Fewer events than asked come back when
    the video has too few that far apart.
    Returns the new event ids, in frame order (the project is not saved).
    """
    from yoru_explain.io.detections_csv import choose_detections, load_detections_csv

    s = project.source(source_id)
    _, by_frame = load_detections_csv(csv_path)
    skip = _is_event_already(project, source_id) if order == "random" else None
    gap, taken = 0, None
    if min_gap_s:
        gap = min_gap_frames(s, float(min_gap_s))
        taken = {}
        for e in project.events:
            if e["source_id"] == source_id:
                taken.setdefault(e["class_name"], []).append(int(float(e["frame_index"])))
    chosen = choose_detections(by_frame, class_names=class_names or None, frames=frames,
                               limit=limit, per_class=per_class, order=order, seed=seed, skip=skip,
                               min_gap_frames=gap, taken=taken)
    added = []
    for imported in chosen:
        d = imported.detection
        frame = FrameRef(source_id, s["path"], s["kind"], imported.frame_index, int(s["width"]),
                         int(s["height"]), None, bool(s["v_flip"]), bool(s["h_flip"]), None, "unread")
        target = ExplanationTarget("", frame, d, -1, d.class_id, model_id=model_id,
                                   track_id=imported.track_id, origin="csv_import")
        added.append(project.add_event(target, note))
    return added


def min_gap_frames(source: dict, seconds: float) -> int:
    """*seconds* as a number of frames of *source* (rounded up); its frame rate must be known."""
    if seconds < 0:
        raise ExplainError(ErrorCode.INVALID_CONFIG, "the minimum gap is negative")
    fps = float(source.get("fps") or 0.0)
    if source.get("kind") != "video":
        return 0
    if not fps > 0:
        raise ExplainError(ErrorCode.INVALID_CONFIG,
                           f"{source.get('source_id')}: the video does not state its frame rate, so a "
                           f"gap in seconds cannot be turned into frames")
    # Rounded up, but not past a whole number by a rounding residue (0.2 s x 25 fps is 5 frames).
    return int(math.ceil(seconds * fps - 1e-6))


def _is_event_already(project: Project, source_id: str):
    """``ImportedDetection -> bool``: the box is an event of *source_id* already."""
    from yoru_explain.core.geometry import overlap

    existing: dict = {}
    for e in project.events:
        if e["source_id"] == source_id:
            t = project.target(e["event_id"])
            existing.setdefault((t.frame.frame_index, t.detection.class_id), []).append(t.detection)

    def skip(imported) -> bool:
        d = imported.detection
        return any(overlap(d.box, x.box, d.obb, x.obb) >= 0.9
                   for x in existing.get((imported.frame_index, d.class_id), ()))

    return skip


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
