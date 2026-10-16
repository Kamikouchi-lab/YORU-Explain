# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Grad-CAM for one pre-NMS candidate, offered only where it was verified.

For the target candidate's score ``s`` and a feature map ``A`` (channels
``c``, cells ``(i, j)``) of a chosen layer:

    alpha_c = mean_ij ds/dA_c(i, j)
    CAM(i, j) = sum_c alpha_c * A_c(i, j)        (signed: ``cam_feature``)
    Grad-CAM  = max(CAM, 0)                      (``cam``, mapped to frame pixels)

(Selvaraju et al., 2017.)  The gradient is of the **fixed candidate** the
target was reproduced as -- never of "the best box of the class" -- and the
run refuses to produce a map unless every check passes:

1. the adapter's gradient probe passed when the model was loaded;
2. the target has a pre-NMS candidate, reproduced on this frame;
3. the score recomputed in the gradient pass equals the inference score
   (within ``score_tolerance``; exact under the default precision);
4. the gradient is finite and not identically zero;
5. the map is finite.

The model stays in eval mode; the graph starts at a copy of the input, the
weights do not require gradients and are never updated.

Reading the map: it is a channel-weighted activation map.  With YOLOv5's 1x1
head, the gradient at the Detect input is non-zero only at the candidate's
own cell (``gradient_support`` reports the count), so the map shows where
the features that cell relies on are active anywhere in the frame -- other
animals included.  It is an attribution of the model's score, not evidence
of what the animal is doing.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

from yoru_explain.core.errors import ErrorCode, ExplainError, error_code_of
from yoru_explain.core.logs import get_logger
from yoru_explain.core.types import SCORE_TYPES, ExplanationTarget

log = get_logger("gradcam")

GRADCAM_VERSION = "0.1.0"
METHOD_KIND = "attribution"
PRECISIONS = ("inference", "fp32")


@dataclass(frozen=True)
class GradCamConfig:
    #: Layer name from the adapter's ``gradient_layers``; ``auto`` = the
    #: adapter's default for the target (YOLOv5: the Detect input at the
    #: candidate's level).
    layer: str = "auto"
    score_type: str = "obj_x_cls"
    #: ``inference``: the same autocast as inference (the explained score);
    #: ``fp32``: autocast off.
    precision: str = "inference"
    #: Largest |gradient-pass score - inference score| accepted.
    score_tolerance: float = 1e-4
    method_version: str = GRADCAM_VERSION

    def __post_init__(self):
        if self.score_type not in SCORE_TYPES:
            raise ValueError(f"unknown score_type {self.score_type!r}")
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        if not float(self.score_tolerance) >= 0:
            raise ValueError("score_tolerance must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": "gradcam",
            "method_kind": METHOD_KIND,
            "method_version": self.method_version,
            "layer": self.layer,
            "score_type": self.score_type,
            "identity_mode": "fixed_candidate",
            "precision": self.precision,
            "score_tolerance": float(self.score_tolerance),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GradCamConfig":
        d = dict(d)
        for k in ("method", "method_kind", "identity_mode"):
            d.pop(k, None)
        unknown = set(d) - {"layer", "score_type", "precision", "score_tolerance", "method_version"}
        if unknown:
            raise ValueError(f"unknown gradcam keys: {sorted(unknown)}")
        return cls(**d)


@dataclass
class GradCamResult:
    target: ExplanationTarget
    config: GradCamConfig
    status: str = "ok"
    error_code: Optional[str] = None
    message: str = ""
    layer: Optional[str] = None
    layer_stride: Optional[int] = None
    #: The target's score in the inference pass, and in the gradient pass.
    reference_score: Optional[float] = None
    gradient_score: Optional[float] = None
    #: Signed CAM on the feature grid, before the ReLU.
    cam_feature: Optional[np.ndarray] = None
    #: max(CAM, 0) on frame pixels.
    cam: Optional[np.ndarray] = None
    #: Feature cells with a non-zero gradient.
    gradient_support: Optional[int] = None
    elapsed_s: float = 0.0

    @property
    def method(self) -> str:
        return "gradcam"

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _fail(result: GradCamResult, code: ErrorCode, message: str) -> GradCamResult:
    result.status, result.error_code, result.message = "failed", code.value, message
    return result


def run_gradcam(adapter, frame: np.ndarray, target: ExplanationTarget,
                config: GradCamConfig) -> GradCamResult:
    import torch

    t0 = time.perf_counter()
    result = GradCamResult(target=target, config=config)
    caps = adapter.capabilities
    if not caps.gradients:
        return _fail(result, ErrorCode.GRADIENTS_UNAVAILABLE,
                     f"{adapter.name}: Grad-CAM is not available ({caps.gradients_reason})")
    cand = target.candidate
    if cand is None:
        return _fail(result, ErrorCode.SCORE_UNAVAILABLE,
                     "Grad-CAM needs the target's pre-NMS candidate")
    try:
        layer = adapter.default_gradient_layer(target) if config.layer == "auto" else config.layer
        layers = {li.name: li for li in adapter.gradient_layers()}
        if layer not in layers:
            return _fail(result, ErrorCode.INVALID_CONFIG,
                         f"layer {layer!r} is not one of {adapter.name}'s gradient layers")
        result.layer, result.layer_stride = layer, layers[layer].stride
        adapter.assert_inference_mode()

        prepared = adapter.preprocess([frame])
        logits = adapter.needs_logits(config.score_type)
        raw = adapter.predict_raw(prepared, logits=logits)
        ref = adapter.candidate_readout(raw, prepared, cand.index, target.target_class,
                                        config.score_type)[0].value
        result.reference_score = ref
        graw = adapter.forward_with_gradients(prepared, [layer], precision=config.precision)
        s = adapter.candidate_score_tensor(graw, 0, cand.index, target.target_class, config.score_type)
        result.gradient_score = float(s.detach())
        if ref is None or not math.isfinite(ref) or not math.isfinite(result.gradient_score):
            return _fail(result, ErrorCode.NON_FINITE, f"score not finite ({ref}, {result.gradient_score})")
        diff = abs(result.gradient_score - ref)
        if diff > config.score_tolerance:
            return _fail(result, ErrorCode.SCORE_MISMATCH,
                         f"gradient-pass score {result.gradient_score!r} differs from the inference "
                         f"score {ref!r} by {diff:.3g} (> {config.score_tolerance:g})")
        act = graw.activations[layer]
        grad = torch.autograd.grad(s, act)[0]
        if not bool(torch.isfinite(grad).all()):
            return _fail(result, ErrorCode.NON_FINITE, "non-finite gradient")
        support = int((grad[0] != 0).any(dim=0).sum())
        result.gradient_support = support
        if support == 0:
            return _fail(result, ErrorCode.GRADIENTS_UNAVAILABLE,
                         f"the score has no gradient at {layer}")
        a = act[0].detach().float()
        g = grad[0].detach().float()
        alpha = g.mean(dim=(1, 2))
        cam = (alpha[:, None, None] * a).sum(0).cpu().numpy().astype(np.float32)
        if not np.isfinite(cam).all():
            return _fail(result, ErrorCode.NON_FINITE, "non-finite CAM")
        result.cam_feature = cam
        stride = layers[layer].stride
        result.cam = prepared.transform.feature_to_frame(np.maximum(cam, 0.0), stride)
        if result.cam.shape != frame.shape[:2]:
            return _fail(result, ErrorCode.INTERNAL,
                         f"map {result.cam.shape} does not match the frame {frame.shape[:2]}")
    except ExplainError as e:
        return _fail(result, e.code, e.message)
    except Exception as e:
        log.error("Grad-CAM of %s failed: %s", target.event_id, e, exc_info=e)
        return _fail(result, error_code_of(e), f"{type(e).__name__}: {e}")
    result.elapsed_s = time.perf_counter() - t0
    return result
