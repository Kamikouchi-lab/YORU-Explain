# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Running an analysis over a project's events and writing the run folder.

For every event, in order of source and frame:

1. the frame is decoded **in order** (exact frame numbering) and its pixel
   hash compared with the one the event was made from (``frame_mismatch``);
2. on the first frame, the adapter is compared with YORU's own detector
   (``identity_check``); a failed check stops the run, since its results
   would not be about YORU's detections;
3. the detector runs on the frame and the stored detection must be found
   again -- same candidate and class, or same class and IoU >= 0.9
   (``target_not_reproduced``);
4. the method runs; the result, its maps and its preview are written at
   once, so an interrupted run keeps what it finished.

Errors become result rows with a status and an error code, and the run
continues with the next event.  The GUI and the Python API both come here;
nothing in the GUI computes an explanation.
"""

from __future__ import annotations

import time
import traceback
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from yoru_explain.analysis.summary import gradcam_summary, occlusion_summary
from yoru_explain.core.environment import hardware, software_versions
from yoru_explain.core.errors import ErrorCode, ExplainError, error_code_of
from yoru_explain.core.geometry import iou
from yoru_explain.core.hashing import json_sha256
from yoru_explain.core.logs import get_logger, run_log
from yoru_explain.core.types import (
    COORDINATE_SYSTEM,
    COORDINATE_SYSTEM_DESCRIPTION,
    IDENTITY_MODES,
    SCORE_TYPES,
    ExplanationTarget,
)
from yoru_explain.io import runs as runio
from yoru_explain.io import schema
from yoru_explain.io.project import Project, now_iso
from yoru_explain.methods.gradcam import GradCamConfig, run_gradcam
from yoru_explain.methods.occlusion import OcclusionConfig, run_occlusion
from yoru_explain.visualization.overlay import triptych

log = get_logger("pipeline")

METHODS = ("occlusion", "gradcam")
#: IoU an imported (or re-anchored) detection needs to count as found again.
REPRODUCE_IOU = 0.9

BASELINE_NOTE = (
    "Occlusion replaces image regions with a baseline the model never saw in training; "
    "the measured effect depends on that choice (see config.method_config.baseline and "
    "perturbations.csv pixel_change). Results describe the model's score, not the animal's behaviour."
)


@dataclass(frozen=True)
class AnalysisRequest:
    method: str = "occlusion"
    occlusion: OcclusionConfig = field(default_factory=OcclusionConfig)
    gradcam: GradCamConfig = field(default_factory=GradCamConfig)
    #: Compare the adapter with YORU's own detector on the first frame.
    verify_identity: bool = True
    #: Reuse a result of an earlier run with the same cache key.
    use_cache: bool = True
    write_previews: bool = True

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")

    def method_config(self) -> Dict[str, Any]:
        return self.occlusion.to_dict() if self.method == "occlusion" else self.gradcam.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "method_config": self.method_config(),
            "verify_identity": self.verify_identity,
            "use_cache": self.use_cache,
            "write_previews": self.write_previews,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AnalysisRequest":
        unknown = set(d) - {"method", "method_config", "verify_identity", "use_cache", "write_previews"}
        if unknown:
            raise ValueError(f"unknown request keys: {sorted(unknown)}")
        method = d.get("method", "occlusion")
        mc = d.get("method_config") or {}
        kw = {}
        if method == "occlusion" and mc:
            kw["occlusion"] = OcclusionConfig.from_dict(mc)
        if method == "gradcam" and mc:
            kw["gradcam"] = GradCamConfig.from_dict(mc)
        return cls(method=method, verify_identity=bool(d.get("verify_identity", True)),
                   use_cache=bool(d.get("use_cache", True)),
                   write_previews=bool(d.get("write_previews", True)), **kw)

    @property
    def score_type(self) -> str:
        return self.occlusion.score.score_type if self.method == "occlusion" else self.gradcam.score_type

    @property
    def identity_mode(self) -> str:
        return self.occlusion.score.identity_mode if self.method == "occlusion" else "fixed_candidate"


@dataclass(frozen=True)
class Progress:
    event_index: int
    n_events: int
    inner_done: int = 0
    inner_total: int = 0
    message: str = ""

    @property
    def fraction(self) -> float:
        if not self.n_events:
            return 1.0
        inner = self.inner_done / self.inner_total if self.inner_total else 0.0
        return min(1.0, (self.event_index + inner) / self.n_events)


@dataclass
class RunOutcome:
    run_id: str
    run_dir: Path
    status: str
    counts: Dict[str, int]
    identity_check: Optional[Dict[str, Any]]
    results: List[Dict[str, Any]] = field(default_factory=list)


ProgressFn = Callable[[Progress], None]
CancelFn = Callable[[], bool]


def reproduce_target(adapter, image: np.ndarray, target: ExplanationTarget, *,
                     same_model: bool = True) -> Tuple[ExplanationTarget, Dict[str, Any]]:
    """*target* as the adapter detects it on *image* now.

    The stored detection is found again (a) by its candidate -- same index,
    same class **and** IoU >= :data:`REPRODUCE_IOU` with the stored box, since
    a candidate index means nothing on another model or input size -- or (b)
    by class and IoU >= :data:`REPRODUCE_IOU` alone.  (a) is skipped when the
    event was made with a different model (*same_model* false).  The returned
    target carries the detection as found -- box, score and candidate of this
    inference, and the adapter's frame-to-input transform -- and *info* says
    how it was found: ``same_candidate``, ``iou`` (no stored candidate),
    ``iou_candidate_changed`` or ``iou_other_model``.
    """
    prepared = adapter.preprocess([image])
    detections = adapter.postprocess(adapter.predict_raw(prepared), prepared)[0]
    target = replace(target, transform=prepared.transform.to_dict())
    stored = target.detection
    info: Dict[str, Any] = {"n_detections": len(detections), "stored_score": stored.score}
    if stored.candidate is not None and same_model:
        for i, d in enumerate(detections):
            if d.candidate is not None and d.candidate.index == stored.candidate.index \
                    and d.class_id == stored.class_id:
                v = iou(d.box, stored.box)
                if v >= REPRODUCE_IOU:
                    info.update(how="same_candidate", iou=v)
                    return target.with_detection(d, i), info
                log.warning("%s: candidate %s is now at IoU %.3f from the stored box; not used",
                            target.event_id, stored.candidate.index, v)
    same = [(iou(d.box, stored.box), i, d) for i, d in enumerate(detections)
            if d.class_id == stored.class_id]
    if same:
        best_iou, i, d = max(same, key=lambda t: (t[0], t[2].score))
        if best_iou >= REPRODUCE_IOU:
            if stored.candidate is None:
                how = "iou"
            else:
                how = "iou_candidate_changed" if same_model else "iou_other_model"
            info.update(how=how, iou=best_iou)
            if how != "iou":
                log.warning("%s: re-anchored by IoU %.3f (%s): candidate %s -> %s", target.event_id,
                            best_iou, how, stored.candidate.index,
                            d.candidate.index if d.candidate else None)
            return target.with_detection(d, i), info
        info["best_iou"] = best_iou
    raise ExplainError(
        ErrorCode.TARGET_NOT_REPRODUCED,
        f"{target.event_id}: no {stored.class_name} detection on frame {target.frame.frame_index} "
        f"matches the stored box (best IoU {info.get('best_iou', 0.0):.3f}); different model, "
        f"thresholds, device, flips or frame?", info)


def describe_score(request: AnalysisRequest, target: ExplanationTarget, class_names: Mapping[int, str]) -> str:
    """One line saying exactly which number is explained."""
    cls = class_names.get(target.target_class, str(target.target_class))
    what = SCORE_TYPES[request.score_type].split(" -- ")[0].replace("class logit", f"'{cls}' logit")
    if request.identity_mode == "fixed_candidate":
        cand = target.candidate.describe() if target.candidate else "(none)"
        return f"{what} of fixed pre-NMS candidate {cand}"
    thr = request.occlusion.score.iou_threshold
    return f"{what} of the post-NMS detection matched to the original (class '{cls}', IoU >= {thr:g})"


def cache_key(adapter, frame_sha1: str, target: ExplanationTarget, request: AnalysisRequest) -> str:
    info = adapter.model_info()
    settings = {k: info.get(k) for k in ("conf_thresh", "nms_conf", "iou_thresh", "channel_order",
                                          "input_size", "device", "gain", "precision")}
    return json_sha256({
        "schema_version": schema.SCHEMA_VERSION,
        "adapter": adapter.name,
        "adapter_version": adapter.version,
        "model_sha256": info.get("sha256") or info.get("weights"),
        "settings": settings,
        "frame_sha1": frame_sha1,
        "target": {
            "candidate": target.candidate.index if target.candidate else None,
            "class_id": target.detection.class_id,
            "target_class": target.target_class,
            "box": [round(v, 3) for v in target.box],
        },
        "method": request.method_config(),
    })


class ResultCache:
    """Finished results of a project's earlier runs, by cache key."""

    def __init__(self, project: Project):
        self._hits: Dict[str, Tuple[Path, Dict[str, str]]] = {}
        for run_dir in project.run_dirs():
            path = run_dir / "results.csv"
            if not path.is_file():
                continue
            for row in schema.read_rows(path):
                if row.get("status") == "ok" and row.get("cache_key") and not row.get("cached_from"):
                    self._hits.setdefault(row["cache_key"], (run_dir, row))

    def lookup(self, key: str) -> Optional[Tuple[Path, Dict[str, str]]]:
        return self._hits.get(key)


def _preview_lines(request, target, result, score_text, method_label) -> List[str]:
    if request.method == "occlusion":
        c = request.occlusion
        return [
            f"{target.event_id}  frame {target.frame.frame_index}  {target.detection.class_name}  "
            f"score {result.reference_score:.4f} (reference {result.reference_score_batch:.4f})",
            f"explained: {score_text}",
            f"occlusion {c.patch_size[0]}x{c.patch_size[1]} stride {c.stride[0]}x{c.stride[1]}, "
            f"baseline {c.baseline.describe()}, region {c.region}, batch {c.batch_size}",
            method_label,
        ]
    return [
        f"{target.event_id}  frame {target.frame.frame_index}  {target.detection.class_name}  "
        f"score {result.reference_score:.4f}",
        f"explained: {score_text}",
        f"Grad-CAM at {result.layer} (stride {result.layer_stride}), gradient at "
        f"{result.gradient_support} cell(s), precision {request.gradcam.precision}",
        method_label,
    ]


def analyze_frame(adapter, image: np.ndarray, target: ExplanationTarget, request: AnalysisRequest, *,
                  progress: Optional[Callable[[int, int], None]] = None,
                  cancel: Optional[CancelFn] = None):
    """Run the requested method on one reproduced target; no files."""
    if request.method == "occlusion":
        return run_occlusion(adapter, image, target, request.occlusion, progress=progress, cancel=cancel)
    return run_gradcam(adapter, image, target, request.gradcam)


def _failure_row(target: ExplanationTarget, source, request, code, message) -> Dict[str, Any]:
    row = runio.target_columns(target, source)
    row.update(method=request.method,
               method_kind="perturbation_effect" if request.method == "occlusion" else "attribution",
               score_type=request.score_type, identity_mode=request.identity_mode,
               baseline=request.occlusion.baseline.describe() if request.method == "occlusion" else None,
               status="failed" if code != ErrorCode.CANCELLED.value else "cancelled",
               error_code=code, message=message)
    return row


def run_analysis(project: Project, adapter, model_id: str, event_ids: Sequence[str],
                 request: AnalysisRequest, *, progress: Optional[ProgressFn] = None,
                 cancel: Optional[CancelFn] = None) -> RunOutcome:
    """Analyse *event_ids* of *project* with the loaded *adapter*; returns how it went."""
    model = project.model(model_id)
    info = adapter.model_info()
    if info.get("sha256") and model.sha256 not in (info.get("sha256"), "synthetic"):
        raise ExplainError(ErrorCode.INVALID_CONFIG,
                           f"the loaded model ({info['sha256'][:12]}) is not project model {model_id} "
                           f"({model.sha256[:12]})")
    run_id = runio.new_run_id(request.method)
    writer = runio.RunWriter(project.runs_dir, run_id)
    t_start = time.perf_counter()
    event_ids = list(event_ids)
    created = now_iso()
    config = {
        "schema_version": schema.SCHEMA_VERSION,
        "kind": schema.CONFIG_KIND,
        "run_id": run_id,
        "created": created,
        "request": request.to_dict(),
        "method": request.method,
        "method_config": request.method_config(),
        "score_types": SCORE_TYPES,
        "identity_modes": IDENTITY_MODES,
        "model_id": model_id,
        "model_options": dict(model.options),
        "adapter": adapter.name,
        "adapter_version": adapter.version,
        "events": event_ids,
        "frame_access": "sequential (decoded in order; frame hash checked against the event)",
        "seed": request.occlusion.seed if request.method == "occlusion" else None,
    }
    writer.write_config(config)
    manifest: Dict[str, Any] = {
        "schema_version": schema.SCHEMA_VERSION,
        "kind": schema.MANIFEST_KIND,
        "run_id": run_id,
        "created": created,
        "status": "running",
        "method": request.method,
        "method_kind": "perturbation_effect" if request.method == "occlusion" else "attribution",
        "project": {"name": project.name, "root": str(project.root)},
        "software": software_versions(),
        "hardware": hardware(),
        "adapter": {"name": adapter.name, "version": adapter.version,
                    "capabilities": adapter.capabilities.to_dict()},
        "model": {"model_id": model_id, **adapter.model_info()},
        "model_record_sha256": model.sha256,
        "coordinate_system": COORDINATE_SYSTEM,
        "coordinate_system_description": COORDINATE_SYSTEM_DESCRIPTION,
        "baseline_note": BASELINE_NOTE if request.method == "occlusion" else None,
        "interpretation": "Perturbation effects and attributions describe one model's score; "
                          "they are not biological interpretations. Frames are not independent samples.",
    }
    writer.write_manifest(manifest)

    counts: Counter = Counter()
    error_codes: Counter = Counter()
    identity: Optional[Dict[str, Any]] = None
    outcome_rows: List[Dict[str, Any]] = []
    inputs: Dict[str, Dict[str, Any]] = {}
    status = "completed"
    with run_log(writer.log_path) as rlog:
        rlog.info("run %s: %s of %d event(s) with %s %s on %s", run_id, request.method, len(event_ids),
                  adapter.name, Path(str(info.get("weights", ""))).name, info.get("device"))
        targets: List[Tuple[str, Optional[ExplanationTarget], Optional[str]]] = []
        for eid in event_ids:
            try:
                targets.append((eid, project.target(eid), None))
            except ExplainError as e:
                targets.append((eid, None, str(e)))
        order = sorted(range(len(targets)), key=lambda i: (
            targets[i][1].frame.source_id if targets[i][1] else "",
            targets[i][1].frame.frame_index if targets[i][1] else -1))
        cache = ResultCache(project) if request.use_cache else None
        readers: Dict[str, Any] = {}
        abort: Optional[Tuple[str, str]] = None
        try:
            for n, i in enumerate(order):
                eid, target, problem = targets[i]
                if target is None:
                    row = {"event_id": eid, "method": request.method, "status": "failed",
                           "error_code": ErrorCode.MISSING_DATA.value, "message": problem}
                    writer.append_result(row)
                    outcome_rows.append(row)
                    counts["failed"] += 1
                    error_codes[ErrorCode.MISSING_DATA.value] += 1
                    continue
                source = project.source(target.frame.source_id)
                if abort is None and cancel is not None and cancel():
                    abort = (ErrorCode.CANCELLED.value, "cancelled before this event")
                    status = "cancelled"
                if abort is not None:
                    row = _failure_row(target, source, request, *abort)
                    writer.append_result(row)
                    outcome_rows.append(row)
                    counts["cancelled" if abort[0] == ErrorCode.CANCELLED.value else "failed"] += 1
                    error_codes[abort[0]] += 1
                    continue
                if progress:
                    progress(Progress(n, len(order), 0, 0, f"{eid}: reading frame {target.frame.frame_index}"))
                identity_holder = {"value": identity}

                def inner(done, total, n=n, eid=eid):
                    if progress:
                        progress(Progress(n, len(order), done, total, f"{eid}: {done}/{total}"))

                try:
                    row = _analyze_event(project, adapter, writer, request, target, source, readers,
                                         cache, inputs, rlog, inner, cancel, identity_holder, model_id)
                    identity = identity_holder["value"]
                except _IdentityFailure as e:
                    identity = e.result
                    abort = (ErrorCode.IDENTITY_CHECK_FAILED.value, str(e))
                    status = "failed"
                    row = _failure_row(target, source, request, *abort)
                    writer.append_result(row)
                    rlog.error("identity check against YORU failed: %s", e.result)
                except Exception as e:
                    # Not expected (each step reports its own failures); still a
                    # row for this event, and the next event runs.
                    rlog.error("%s: unexpected error:\n%s", eid, traceback.format_exc())
                    row = _failure_row(target, source, request, error_code_of(e).value,
                                       f"{type(e).__name__}: {e}")
                    try:
                        writer.append_result(row)
                    except Exception:
                        rlog.error("%s: the failure row could not be written either", eid)
                outcome_rows.append(row)
                st = row.get("status")
                counts[st] += 1
                if row.get("cached_from"):
                    counts["cached"] += 1
                if row.get("error_code"):
                    error_codes[row["error_code"]] += 1
                if st == "cancelled":
                    status = "cancelled"
                    abort = (ErrorCode.CANCELLED.value, "cancelled")
        except BaseException:
            # Whatever escaped (KeyboardInterrupt, a writer that cannot write at
            # all), the manifest must not claim the run completed.
            status = "failed"
            raise
        finally:
            for r in readers.values():
                r.close()
            writer.ensure_tables()
            manifest.update(
                status=status,
                finished=now_iso(),
                elapsed_s=time.perf_counter() - t_start,
                identity_check=identity,
                inputs=list(inputs.values()),
                events={"requested": len(event_ids), **dict(counts)},
                error_codes=dict(error_codes),
                files={
                    "config.json": "the request: method, parameters, score definition, seed",
                    "results.csv": "one row per event (schema_version first)",
                    "perturbations.csv": "one row per perturbed input (occlusion)",
                    "maps/*.npz": "maps and per-patch arrays with metadata (meta_json)",
                    "previews/*.png": "original | heatmap | overlay, for viewing only",
                    "logs/run.log": "this run's log",
                },
            )
            writer.write_manifest(manifest)
            rlog.info("run %s %s: %s", run_id, status, dict(counts))
    return RunOutcome(run_id, writer.dir, status, dict(counts), identity, outcome_rows)


class _IdentityFailure(Exception):
    def __init__(self, result: Dict[str, Any]):
        super().__init__(f"adapter differs from YORU's detector: {result}")
        self.result = result


def _analyze_event(project, adapter, writer, request, target, source, readers, cache, inputs, rlog,
                   inner_progress, cancel, identity_holder, model_id) -> Dict[str, Any]:
    t0 = time.perf_counter()
    sid = source["source_id"]
    try:
        if sid not in readers:
            readers[sid] = project.open_source(sid)
            inputs[sid] = {k: source.get(k) for k in (
                "source_id", "path", "kind", "size", "quick_id", "quick_id_method", "width", "height",
                "fps", "frame_count", "fourcc", "v_flip", "h_flip", "seek_check")}
            inputs[sid]["verify"] = project.verify_source(sid)
        frame = readers[sid].read(target.frame.frame_index, exact=True)
        if target.frame.frame_sha1 and frame.sha1 != target.frame.frame_sha1:
            raise ExplainError(
                ErrorCode.FRAME_MISMATCH,
                f"frame {target.frame.frame_index} decoded in order is not the picture the event was "
                f"made from (event access: {target.frame.access}); re-create the event with exact frame access")
        if request.verify_identity and identity_holder["value"] is None:
            check = adapter.identity_check(frame.image)
            check["frame"] = {"source_id": sid, "frame_index": frame.frame_index, "sha1": frame.sha1}
            identity_holder["value"] = check
            rlog.info("identity check: %s", check)
            if check.get("status") == "failed":
                raise _IdentityFailure(check)
        stored_score = target.detection.score
        # A candidate index is only meaningful for the model the event was made with.
        same_model = not target.model_id or target.model_id == model_id
        target, rep = reproduce_target(adapter, frame.image, target, same_model=same_model)
        target = replace(target, frame=replace(target.frame, frame_sha1=frame.sha1,
                                               timestamp_ms=frame.timestamp_ms, access=frame.access))
        key = cache_key(adapter, frame.sha1, target, request)
        base = runio.target_columns(target, source)
        base.update(stored_score=stored_score, reproduction=rep.get("how"),
                    reproduction_iou=rep.get("iou"), cache_key=key)
        if cache is not None:
            hit = cache.lookup(key)
            if hit is not None:
                return _copy_cached(writer, hit, target, base, rlog)
        score_text = describe_score(request, target, adapter.class_names)
        result = analyze_frame(adapter, frame.image, target, request, progress=inner_progress,
                               cancel=cancel)
    except _IdentityFailure:
        raise
    except Exception as e:
        code = error_code_of(e)
        if not isinstance(e, ExplainError):
            rlog.error("%s failed:\n%s", target.event_id, traceback.format_exc())
        else:
            rlog.warning("%s: %s", target.event_id, e)
        row = _failure_row(target, source, request, code.value, str(e))
        row["elapsed_s"] = time.perf_counter() - t0
        writer.append_result(row)
        return row

    written: List[str] = []
    try:
        row = dict(base)
        row.update(
            method=request.method,
            method_kind="perturbation_effect" if request.method == "occlusion" else "attribution",
            score_type=request.score_type,
            identity_mode=request.identity_mode,
            baseline=request.occlusion.baseline.describe() if request.method == "occlusion" else None,
            original_score=result.reference_score,
            status=result.status,
            error_code=result.error_code,
            message=result.message or score_text,
            elapsed_s=result.elapsed_s,
        )
        method_label = ("Perturbation effect on the model's score -- not a biological interpretation"
                        if request.method == "occlusion" else
                        "Attribution of the model's score -- not a biological interpretation")
        extra_meta = {"run_id": writer.run_id, "score_description": score_text, "cache_key": key}
        perturbations = None
        if request.method == "occlusion":
            row.update(reference_score=result.reference_score_batch,
                       matching_status=result.reference_status)
            if result.ok:
                row.update(occlusion_summary(result))
                arrays, meta = runio.occlusion_npz(result, extra_meta)
                perturbations = runio.perturbation_rows(target.event_id, result)
                row["map_file"] = writer.save_map(target.event_id, "occlusion", arrays, meta)
                written.append(row["map_file"])
                if request.write_previews:
                    img = triptych(frame.image, result.delta_map, "delta", target.box,
                                   label=f"{target.detection.class_name} {result.reference_score:.3f}",
                                   lines=_preview_lines(request, target, result, score_text, method_label),
                                   roi=result.roi)
                    row["preview_file"] = writer.save_preview(target.event_id, "occlusion", img)
                    written.append(row["preview_file"])
        else:
            row.update(reference_score=result.reference_score, matching_status="ok" if result.ok else None)
            row.update(gradcam_summary(result))
            if result.ok:
                arrays, meta = runio.gradcam_npz(result, target.transform, extra_meta)
                row["map_file"] = writer.save_map(target.event_id, "gradcam", arrays, meta)
                written.append(row["map_file"])
                if request.write_previews:
                    img = triptych(frame.image, result.cam, "cam", target.box,
                                   label=f"{target.detection.class_name} {result.reference_score:.3f}",
                                   lines=_preview_lines(request, target, result, score_text, method_label))
                    row["preview_file"] = writer.save_preview(target.event_id, "gradcam", img)
                    written.append(row["preview_file"])
        # The tables last, the event's own row at the very end: results.csv is
        # the record of what finished.
        if perturbations:
            writer.append_perturbations(perturbations)
        writer.append_result(row)
    except Exception as e:
        # Writing failed (disk full, path too long, ...): no half result is
        # left behind as if it were one, and the run goes on.
        for rel in written:
            try:
                (writer.dir / rel).unlink()
            except OSError:
                pass
        rlog.error("%s: writing the results failed:\n%s", target.event_id, traceback.format_exc())
        row = _failure_row(target, source, request, ErrorCode.OUTPUT_FAILED.value,
                           f"writing the results failed: {type(e).__name__}: {e}")
        row.update({k: base.get(k) for k in ("stored_score", "reproduction", "reproduction_iou", "cache_key")})
        row["elapsed_s"] = time.perf_counter() - t0
        writer.append_result(row)
        return row
    rlog.info("%s: %s %s in %.2fs", target.event_id, request.method, result.status, result.elapsed_s)
    return row


def _copy_cached(writer, hit, target: ExplanationTarget, base: Dict[str, Any], rlog) -> Dict[str, Any]:
    run_dir, old = hit
    reader = runio.RunReader(run_dir)
    row = {k: v for k, v in old.items() if k in schema.RESULT_COLUMNS}
    row.update(base)
    row["cached_from"] = reader.run_id
    method = old.get("method", "")
    if old.get("map_file"):
        data = reader.load_map(old["map_file"])
        meta = data.pop("meta")
        meta.update(cached_from=reader.run_id, run_id=writer.run_id, target=target.to_dict())
        row["map_file"] = writer.save_map(target.event_id, method, data, meta)
    if old.get("preview_file"):
        src = run_dir / old["preview_file"]
        if src.is_file():
            rel = f"{runio.PREVIEWS_DIR}/{runio.map_name(target.event_id, method, 'png')}"
            (writer.dir / rel).write_bytes(src.read_bytes())
            row["preview_file"] = rel
    rows = []
    for p in reader.perturbations(old["event_id"]):
        p = {k: v for k, v in p.items() if k in schema.PERTURBATION_COLUMNS}
        p["event_id"] = target.event_id
        rows.append(p)
    if rows:
        writer.append_perturbations(rows)
    writer.append_result(row)
    rlog.info("%s: reused from run %s (cache key %s)", target.event_id, reader.run_id, base["cache_key"][:12])
    return schema.typed_result_row(row)
