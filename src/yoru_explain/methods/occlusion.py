# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Occlusion sensitivity: how the target's score changes when a region is hidden.

For the target's score ``s`` (see :class:`~yoru_explain.core.types.ScoreDefinition`)
on frame ``x`` and a rectangle ``R`` replaced by a baseline,

    delta(R) = s_ref - s(x with R replaced)

where ``s_ref`` is ``s(x)`` scored **at the run's batch size**: a detector's
scores depend slightly on how many images share a forward pass (measured;
see docs/INITIAL_AUDIT.md), so the reference is taken the same way the
perturbed frames are.  At batch size 1 -- the default -- ``s_ref`` is the
confidence YORU itself reports for the detection.

* ``delta > 0``: hiding R lowers the score -- R supports it.  ``delta < 0``:
  hiding R raises it.  Signs are kept; nothing is clipped at 0.
* Rectangles tile a region of interest with a fixed patch size and stride
  (frame pixels).  The last row/column of patches is aligned to the region's
  far edge, so every pixel of the region is covered.
* The map at pixel p is the mean delta of the counted patches covering p:
  ``M(p) = mean{delta(R) : p in R, R counted}``.  A patch is *counted* when
  the target was scored on it (``ok`` in fixed-candidate mode, ``matched``
  in re-detection mode; ``ambiguous`` only when asked).  Pixels with no
  counted patch are NaN -- never 0.
* Two control evaluations run with every target: the unchanged frame inside
  the batch pipeline (``noop``: its delta must be 0, and is reported), and
  the whole region replaced by the baseline (``full_region``: how the score
  responds to the baseline itself).

What this measures is the effect of an input edit on one number of one
model.  It is not an attribution in the gradient sense, and not a
biological finding.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from yoru_explain.core.errors import ErrorCode, ExplainError, error_code_of
from yoru_explain.core.geometry import clip_box, scaled_box
from yoru_explain.core.logs import get_logger
from yoru_explain.core.types import (
    ExplanationTarget,
    ScoreDefinition,
    ScoreResult,
    ScoreStatus,
)
from yoru_explain.methods.baselines import Baseline, BaselineSpec

log = get_logger("occlusion")

#: Version of this method's definition; results record it.
OCCLUSION_VERSION = "0.1.0"
METHOD_KIND = "perturbation_effect"
REGIONS = ("context", "box", "full")


def _pair(v) -> Tuple[int, int]:
    if isinstance(v, (int, float)):
        return (int(v), int(v))
    a, b = v
    return (int(a), int(b))


@dataclass(frozen=True)
class OcclusionConfig:
    """Everything that defines an occlusion run (all of it goes into ``config.json``)."""

    #: ``(height, width)`` of a patch, frame pixels.
    patch_size: Tuple[int, int] = (32, 32)
    #: ``(y, x)`` step between patches, frame pixels.
    stride: Tuple[int, int] = (16, 16)
    baseline: BaselineSpec = field(default_factory=BaselineSpec)
    #: ``context``: the target box scaled by *context_scale*; ``box``: the box;
    #: ``full``: the whole frame.
    region: str = "context"
    context_scale: float = 3.0
    score: ScoreDefinition = field(default_factory=ScoreDefinition)
    #: Images per forward pass.  1 reproduces YORU's per-frame inference.
    batch_size: int = 1
    seed: int = 0
    #: Re-detection: let ``ambiguous`` matches into the map.
    include_ambiguous: bool = False
    method_version: str = OCCLUSION_VERSION

    def __post_init__(self):
        object.__setattr__(self, "patch_size", _pair(self.patch_size))
        object.__setattr__(self, "stride", _pair(self.stride))
        if min(self.patch_size) < 1 or min(self.stride) < 1:
            raise ValueError("patch size and stride must be >= 1 pixel")
        if self.region not in REGIONS:
            raise ValueError(f"region must be one of {REGIONS}, got {self.region!r}")
        if not float(self.context_scale) >= 1.0:
            raise ValueError("context_scale must be >= 1")
        if int(self.batch_size) < 1:
            raise ValueError("batch_size must be >= 1")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": "occlusion",
            "method_kind": METHOD_KIND,
            "method_version": self.method_version,
            "patch_size": list(self.patch_size),
            "stride": list(self.stride),
            "baseline": self.baseline.to_dict(),
            "region": self.region,
            "context_scale": float(self.context_scale),
            "score": self.score.to_dict(),
            "batch_size": int(self.batch_size),
            "seed": int(self.seed),
            "include_ambiguous": bool(self.include_ambiguous),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "OcclusionConfig":
        d = dict(d)
        d.pop("method", None)
        d.pop("method_kind", None)
        known = {"method_version", "patch_size", "stride", "baseline", "region", "context_scale",
                 "score", "batch_size", "seed", "include_ambiguous"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown occlusion keys: {sorted(unknown)}")
        if "baseline" in d:
            d["baseline"] = BaselineSpec.from_dict(d["baseline"])
        if "score" in d:
            d["score"] = ScoreDefinition.from_dict(d["score"])
        return cls(**d)


@dataclass(frozen=True)
class Patch:
    patch_id: int
    #: ``"patch"``, or a control: ``"noop"`` / ``"full_region"``.
    kind: str
    #: ``(x0, y0, x1, y1)``, integer frame pixels, half-open.
    box: Tuple[int, int, int, int]


@dataclass(frozen=True)
class PatchResult:
    patch: Patch
    score: ScoreResult
    #: Mean |perturbed - original| over the region, 0..255.
    pixel_change: float
    #: ``s_ref - used_score``; ``None`` when the patch is not counted.
    delta: Optional[float]
    #: Whether the patch entered the map.
    counted: bool
    #: The score the delta was taken from: the target's score, or with
    #: ``include_ambiguous`` an ambiguous match's; ``None`` when not counted.
    used_score: Optional[float] = None


@dataclass
class OcclusionResult:
    target: ExplanationTarget
    config: OcclusionConfig
    status: str = "ok"
    error_code: Optional[str] = None
    message: str = ""
    #: How the target was scored on the unchanged frame (``ok`` / ``matched``,
    #: or what went wrong).
    reference_status: Optional[str] = None
    #: The target's score on the unchanged frame at batch size 1.
    reference_score: Optional[float] = None
    #: The same at the run's batch size: the deltas' reference.
    reference_score_batch: Optional[float] = None
    roi: Optional[Tuple[int, int, int, int]] = None
    patches: List[PatchResult] = field(default_factory=list)
    controls: Dict[str, PatchResult] = field(default_factory=dict)
    delta_map: Optional[np.ndarray] = None
    coverage: Optional[np.ndarray] = None
    coverage_all: Optional[np.ndarray] = None
    elapsed_s: float = 0.0
    n_forward: int = 0

    @property
    def method(self) -> str:
        return "occlusion"

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def status_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for p in self.patches:
            counts[p.score.status] = counts.get(p.score.status, 0) + 1
        return counts

    @property
    def lost_fraction(self) -> Optional[np.ndarray]:
        """Per pixel: the share of covering patches that were not counted."""
        if self.coverage is None or self.coverage_all is None:
            return None
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = 1.0 - self.coverage.astype(np.float64) / self.coverage_all.astype(np.float64)
        frac[self.coverage_all == 0] = np.nan
        return frac.astype(np.float32)


def region_of_interest(target_box, region: str, context_scale: float,
                       width: int, height: int) -> Tuple[int, int, int, int]:
    """The integer ``(x0, y0, x1, y1)`` the patches tile."""
    if region == "full":
        return (0, 0, int(width), int(height))
    box = target_box if region == "box" else scaled_box(target_box, context_scale, width, height)
    x0, y0, x1, y1 = clip_box(box, width, height)
    x0, y0 = int(math.floor(x0)), int(math.floor(y0))
    x1, y1 = int(math.ceil(x1)), int(math.ceil(y1))
    if x1 <= x0 or y1 <= y0:
        raise ExplainError(ErrorCode.TARGET_NOT_REPRODUCED, f"empty region of interest {box}")
    return (x0, y0, x1, y1)


def _starts(lo: int, hi: int, size: int, step: int) -> List[int]:
    """Patch starts along one axis: lo, lo+step, ..., the last ending at hi."""
    if hi - lo <= size:
        return [lo]
    starts = list(range(lo, hi - size + 1, step))
    if starts[-1] + size < hi:
        starts.append(hi - size)
    return starts


def make_patches(roi: Tuple[int, int, int, int], patch_size: Tuple[int, int],
                 stride: Tuple[int, int], width: int, height: int) -> List[Patch]:
    """Patches tiling *roi*, row by row; each clipped to the frame."""
    x0, y0, x1, y1 = roi
    ph, pw = patch_size
    sy, sx = stride
    patches = []
    for ys in _starts(y0, y1, ph, sy):
        for xs in _starts(x0, x1, pw, sx):
            box = (xs, ys, min(xs + pw, width), min(ys + ph, height))
            patches.append(Patch(len(patches), "patch", box))
    return patches


def _render(frame: np.ndarray, baseline: Baseline, patch: Patch) -> Tuple[np.ndarray, float]:
    out = frame.copy()
    if patch.kind == "noop":
        return out, 0.0
    x0, y0, x1, y1 = patch.box
    baseline.fill(out, y0, y1, x0, x1, patch.patch_id)
    region_before = frame[y0:y1, x0:x1].astype(np.int16)
    change = float(np.abs(out[y0:y1, x0:x1].astype(np.int16) - region_before).mean()) \
        if region_before.size else 0.0
    return out, change


def _target_score(adapter, images: Sequence[np.ndarray], target: ExplanationTarget,
                  score: ScoreDefinition) -> List[ScoreResult]:
    prepared = adapter.preprocess(list(images))
    raw = adapter.predict_raw(prepared, logits=adapter.needs_logits(score.score_type))
    return adapter.score_target(raw, prepared, target, score)


def reference_scores(adapter, frame: np.ndarray, target: ExplanationTarget, score: ScoreDefinition,
                     batch_size: int) -> Tuple[ScoreResult, ScoreResult]:
    """The target's score on the unchanged frame, at batch 1 and at *batch_size*."""
    one = _target_score(adapter, [frame], target, score)[0]
    if batch_size == 1:
        return one, one
    return one, _target_score(adapter, [frame] * batch_size, target, score)[0]


ProgressFn = Callable[[int, int], None]
CancelFn = Callable[[], bool]


def run_occlusion(adapter, frame: np.ndarray, target: ExplanationTarget, config: OcclusionConfig, *,
                  progress: Optional[ProgressFn] = None,
                  cancel: Optional[CancelFn] = None) -> OcclusionResult:
    """Occlusion sensitivity of *target* on *frame*.

    *target* must already be reproduced on *frame* (see
    :func:`yoru_explain.core.pipeline.reproduce_target`).  Failures come back
    as a result with ``status="failed"`` and an error code, never as scores.
    """
    t0 = time.perf_counter()
    result = OcclusionResult(target=target, config=config)
    h, w = frame.shape[:2]
    if (w, h) != (target.frame.width, target.frame.height):
        result.status, result.error_code = "failed", ErrorCode.FRAME_MISMATCH.value
        result.message = f"frame is {w}x{h}, target was made on {target.frame.width}x{target.frame.height}"
        return result
    if config.score.identity_mode not in adapter.capabilities.identity_modes:
        result.status, result.error_code = "failed", ErrorCode.MODEL_UNSUPPORTED.value
        result.message = (f"{adapter.name} cannot follow a target in {config.score.identity_mode} mode; "
                          f"it offers {list(adapter.capabilities.identity_modes)}")
        return result
    if config.score.identity_mode == "fixed_candidate" and target.candidate is None:
        result.status, result.error_code = "failed", ErrorCode.SCORE_UNAVAILABLE.value
        result.message = "fixed-candidate mode needs the target's pre-NMS candidate"
        return result
    if config.score.score_type not in adapter.capabilities.score_types:
        result.status, result.error_code = "failed", ErrorCode.MODEL_UNSUPPORTED.value
        result.message = f"{adapter.name} cannot read {config.score.score_type}"
        return result

    try:
        adapter.assert_inference_mode()
        ref1, ref_b = reference_scores(adapter, frame, target, config.score, config.batch_size)
        result.n_forward += 1 + (1 if config.batch_size > 1 else 0)
        result.reference_status = ref1.status if not ref1.scored else ref_b.status
        if not ref1.scored or not ref_b.scored:
            bad = ref1 if not ref1.scored else ref_b
            result.status = "failed"
            result.error_code = bad.error_code or ErrorCode.TARGET_NOT_REPRODUCED.value
            result.message = f"the target is not scored on the unchanged frame ({bad.status}) {bad.message}"
            return result
        result.reference_score = ref1.value
        result.reference_score_batch = ref_b.value
        s_ref = ref_b.value

        roi = region_of_interest(target.box, config.region, config.context_scale, w, h)
        result.roi = roi
        patches = make_patches(roi, config.patch_size, config.stride, w, h)
        jobs = [Patch(-1, "noop", (0, 0, 0, 0)),
                Patch(len(patches), "full_region", roi)] + patches
        baseline = Baseline(config.baseline, frame, config.seed)
        bs = int(config.batch_size)
        done = 0
        total = len(jobs)
        if progress:
            progress(0, total)
        records: List[PatchResult] = []
        for start in range(0, total, bs):
            if cancel is not None and cancel():
                result.status, result.error_code = "cancelled", ErrorCode.CANCELLED.value
                result.message = f"cancelled after {done} of {total} evaluations"
                return result
            batch = jobs[start:start + bs]
            images, changes = [], []
            for job in batch:
                img, change = _render(frame, baseline, job)
                images.append(img)
                changes.append(change)
            # Every forward pass has the same batch size as the reference.
            pad = bs - len(batch)
            scores = _target_score(adapter, images + [frame] * pad, target, config.score)[:len(batch)]
            result.n_forward += 1
            for job, change, sc in zip(batch, changes, scores):
                if sc.scored:
                    used = sc.value
                elif (config.include_ambiguous and sc.status == ScoreStatus.AMBIGUOUS.value
                      and sc.ambiguous_value is not None):
                    used = sc.ambiguous_value
                else:
                    used = None
                counted = used is not None
                delta = (s_ref - used) if counted else None
                records.append(PatchResult(job, sc, change, delta, counted, used))
            done += len(batch)
            if progress:
                progress(done, total)
    except Exception as e:
        code = error_code_of(e)
        log.error("occlusion of %s failed: %s", target.event_id, e, exc_info=e)
        result.status, result.error_code = "failed", code.value
        result.message = f"{type(e).__name__}: {e}"
        if code == ErrorCode.CUDA_OOM:
            result.message += " -- try a smaller batch size"
        return result

    result.controls = {r.patch.kind: r for r in records if r.patch.kind != "patch"}
    result.patches = [r for r in records if r.patch.kind == "patch"]
    _aggregate(result, h, w)
    noop = result.controls.get("noop")
    if noop is not None and noop.delta not in (None, 0.0):
        log.warning("%s: unchanged frame scored %r against reference %r (noop delta %.3g)",
                    target.event_id, noop.score.value, s_ref, noop.delta)
    result.elapsed_s = time.perf_counter() - t0
    return result


def _aggregate(result: OcclusionResult, h: int, w: int) -> None:
    total = np.zeros((h, w), np.float64)
    count = np.zeros((h, w), np.uint32)
    count_all = np.zeros((h, w), np.uint32)
    for p in result.patches:
        x0, y0, x1, y1 = p.patch.box
        count_all[y0:y1, x0:x1] += 1
        if p.counted:
            total[y0:y1, x0:x1] += p.delta
            count[y0:y1, x0:x1] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        m = total / count
    m[count == 0] = np.nan
    result.delta_map = m.astype(np.float32)
    result.coverage = count.astype(np.uint16) if count.max(initial=0) < 65535 else count
    result.coverage_all = count_all.astype(np.uint16) if count_all.max(initial=0) < 65535 else count_all


def with_score(config: OcclusionConfig, **changes) -> OcclusionConfig:
    """*config* with its score definition changed."""
    return replace(config, score=replace(config.score, **changes))
