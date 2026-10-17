# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""The contract between the explanation methods and a detector.

The methods (occlusion, Grad-CAM) never touch a model directly.  They prepare
inputs, run the model and read scores only through a :class:`ModelAdapter`,
which is what keeps "what is being explained" defined in one place per model
family:

``preprocess(images)``
    Frames -> model input, with the :class:`LetterboxTransform` that maps
    back.  All images of one call must have the same shape (perturbations of
    one frame do).
``predict_raw(prepared)``
    One forward pass, without gradients, exactly as the detector's own
    inference runs it (same precision, same autocast).
``postprocess(raw, prepared)``
    The detections the detector would report for each image, each carrying
    its pre-NMS :class:`CandidateRef` when the adapter can recover it.
``candidate_readout(...)``
    One candidate's score, box and current class on every image: what
    fixed-candidate mode reads.
``score_target(...)``
    The target's :class:`ScoreResult` on every image, in fixed-candidate or
    re-detection mode (shared implementation in ``analysis.scoring``).
``forward_with_gradients(...)`` / ``candidate_score_tensor(...)``
    The gradient path, for adapters whose capabilities say gradients were
    verified.

What a "candidate" is depends on the family and is stated in the
capabilities (``candidate_kind``): an anchor slot (YOLOv5), an anchor point
(YOLOv8/YOLO11), a default box (SSD) or an output row (ONNX) -- fixed
positions, the same on every input of one size, so fixed-candidate mode can
follow them -- or a decoder query (RT-DETR) or a region proposal (Faster
R-CNN), which are chosen anew for every input and so mean something on one
input only: those adapters offer re-detection, and Grad-CAM of the
candidate on the one frame.

An adapter never puts its model in training mode and never changes weights.
"""

from __future__ import annotations

import contextlib
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from yoru_explain.core.errors import ErrorCode, ExplainError
from yoru_explain.core.geometry import LetterboxTransform
from yoru_explain.core.types import (
    IDENTITY_MODES,
    SCORE_TYPES,
    Box,
    Detection,
    ExplanationTarget,
    Obb,
    ScoreDefinition,
    ScoreResult,
)


@dataclass(frozen=True)
class LayerInfo:
    """A feature map the gradient path can capture."""

    name: str
    index: int
    type: str
    #: Input pixels per feature cell; cell ``(i, j)`` is centred on input pixel
    #: ``((j + .5) * stride, (i + .5) * stride)``.  The grid covers the input
    #: (it tiles it exactly, except SSD's, which overhangs it).
    stride: int
    channels: int

    def describe(self) -> str:
        return f"{self.name} ({self.type}, stride {self.stride}, {self.channels} ch)"


@dataclass(frozen=True)
class AdapterCapabilities:
    """What an adapter can do -- measured where it can be, not assumed."""

    fixed_candidate: bool
    redetection: bool
    score_types: Tuple[str, ...]
    #: True only when a gradient probe on the loaded model passed.
    gradients: bool = False
    #: What the probe found, or why gradients are not offered.
    gradients_reason: str = "not probed"
    gradient_layers: Tuple[str, ...] = ()
    #: Repeated batch-1 forward passes are bit-identical.
    deterministic: Optional[bool] = None
    #: The same image scores identically at batch 1 and batch 2.
    batch_invariant: Optional[bool] = None
    batch_max_abs_diff: Optional[float] = None
    notes: Tuple[str, ...] = ()
    #: The model family, in words ("YOLOv5 (YORU's bundled code)", "RT-DETR (ultralytics)", ...).
    family: str = ""
    #: What one candidate is: "anchor slot", "anchor point", "default box",
    #: "output row", "query", "proposal" -- or "" when there are none (black box).
    candidate_kind: str = "anchor slot"
    #: The family's own name of the score that *is* the reported confidence
    #: (``obj_x_cls`` for YOLOv5, ``cls_prob`` for YOLOv8, ``cls_softmax`` for
    #: torchvision); ``confidence`` means the same number.
    reported_score: str = "confidence"
    #: One candidate can be reported under several classes at once (SSD,
    #: Faster R-CNN), so "NMS still keeps it" also needs the class.
    multi_label: bool = False
    #: ``(score_type, definition for this family)`` pairs; the general
    #: definition (``core.types.SCORE_TYPES``) for those not listed.
    score_descriptions: Tuple[Tuple[str, str], ...] = ()
    #: Things the person loading the model must know (the GUI shows them):
    #: e.g. a checkpoint the installed package runs differently from how it was built.
    warnings: Tuple[str, ...] = ()

    @property
    def identity_modes(self) -> Tuple[str, ...]:
        """The identity modes this adapter can score in, in ``IDENTITY_MODES`` order."""
        ok = {"fixed_candidate": self.fixed_candidate, "redetect": self.redetection}
        return tuple(m for m in IDENTITY_MODES if ok.get(m))

    def describe(self, score_type: str) -> str:
        """What *score_type* is for this model family."""
        own = dict(self.score_descriptions)
        return own.get(score_type) or SCORE_TYPES.get(score_type, score_type)

    def reports(self, score_type: str) -> bool:
        """Whether *score_type* is the confidence the detector itself reports."""
        return score_type in ("confidence", self.reported_score)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "candidate_kind": self.candidate_kind,
            "fixed_candidate": self.fixed_candidate,
            "redetection": self.redetection,
            "identity_modes": list(self.identity_modes),
            "score_types": list(self.score_types),
            "reported_score": self.reported_score,
            "score_definitions": {k: self.describe(k) for k in self.score_types},
            "multi_label": self.multi_label,
            "gradients": self.gradients,
            "gradients_reason": self.gradients_reason,
            "gradient_layers": list(self.gradient_layers),
            "deterministic": self.deterministic,
            "batch_invariant": self.batch_invariant,
            "batch_max_abs_diff": self.batch_max_abs_diff,
            "notes": list(self.notes),
            "warnings": list(self.warnings),
        }


@dataclass
class PreparedInput:
    """A batch of model inputs and the transform shared by all of them.

    *extra* holds what else the adapter's forward pass needs (torchvision:
    the resized image sizes).
    """

    tensor: Any
    transform: LetterboxTransform
    frame_shape: Tuple[int, ...]
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return int(self.tensor.shape[0])


@dataclass
class RawOutput:
    """A forward pass's candidates, before NMS.

    *pred* is the adapter's candidate table.  For the YOLOv5 layout it is
    ``(B, N, 5 + nc)``: ``cx, cy, w, h`` in input pixels, then probabilities
    (objectness, classes); other families document theirs.  *logits* holds
    the pre-sigmoid / pre-softmax values when captured, *activations* the
    feature maps of a gradient pass, *extra* whatever else the adapter's
    postprocessing needs (proposals, anchors, the end-to-end head's output).
    """

    pred: Any
    logits: Any = None
    activations: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateReadout:
    """One candidate on one image: its score, where its box is, its class."""

    value: Optional[float]
    box: Optional[Box]
    argmax_class: Optional[int]
    #: The rotated box, for models that predict one.
    obb: Optional[Obb] = None


class ModelAdapter(ABC):
    """Base class of every model adapter."""

    #: Registry name, e.g. ``"yoru-yolov5"``.
    name: ClassVar[str] = ""
    #: Version of the adapter's own code (results record it).
    version: ClassVar[str] = "0"
    description: ClassVar[str] = ""

    def __init__(self):
        # The model is shared by the GUI's worker thread and anything else;
        # one forward pass at a time.
        self._lock = threading.RLock()

    # -- loading ---------------------------------------------------------

    @abstractmethod
    def load_model(self, weights, device: str = "auto", **options) -> None:
        """Load *weights* onto *device* (``auto``, ``cuda``, ``cpu``, ...)."""

    @property
    @abstractmethod
    def class_names(self) -> Dict[int, str]: ...

    @property
    @abstractmethod
    def capabilities(self) -> AdapterCapabilities: ...

    @abstractmethod
    def model_info(self) -> Dict[str, Any]:
        """Everything a manifest needs to identify the model and its settings."""

    # -- inference -------------------------------------------------------

    @abstractmethod
    def preprocess(self, images: Sequence[np.ndarray]) -> PreparedInput: ...

    @abstractmethod
    def predict_raw(self, prepared: PreparedInput, *, logits: bool = False) -> RawOutput: ...

    @abstractmethod
    def postprocess(self, raw: RawOutput, prepared: PreparedInput) -> List[List[Detection]]: ...

    @abstractmethod
    def candidate_readout(self, raw: RawOutput, prepared: PreparedInput, candidate_index: int,
                          target_class: int, score_type: str) -> List[CandidateReadout]: ...

    def needs_logits(self, score_type: str) -> bool:
        return score_type.endswith("_logit")

    def describe_score_type(self, score_type: str) -> str:
        """What *score_type* means for this adapter's model family."""
        caps = self.capabilities
        return caps.describe(score_type) if caps is not None else SCORE_TYPES.get(score_type, score_type)

    def detect(self, image: np.ndarray) -> List[Detection]:
        """The detections the detector reports for one frame."""
        prepared = self.preprocess([image])
        raw = self.predict_raw(prepared)
        return self.postprocess(raw, prepared)[0]

    def score_target(self, raw: RawOutput, prepared: PreparedInput, target: ExplanationTarget,
                     score: ScoreDefinition,
                     detections: Optional[List[List[Detection]]] = None) -> List[ScoreResult]:
        """The target's score on every image of the batch, or why there is none."""
        from yoru_explain.analysis.scoring import score_batch

        return score_batch(self, raw, prepared, target, score, detections)

    def identity_check(self, image: np.ndarray) -> Dict[str, Any]:
        """Compare this adapter with the detector it stands in for, on *image*."""
        return {"status": "not_applicable",
                "reason": f"{self.name} does not stand in for a YORU detector"}

    # -- gradients -------------------------------------------------------

    def gradient_layers(self) -> List[LayerInfo]:
        return []

    def default_gradient_layer(self, target: ExplanationTarget) -> str:
        raise ExplainError(ErrorCode.GRADIENTS_UNAVAILABLE,
                           f"{self.name} offers no gradient path")

    def forward_with_gradients(self, prepared: PreparedInput, layers: Sequence[str], *,
                               precision: str = "inference") -> RawOutput:
        raise ExplainError(ErrorCode.GRADIENTS_UNAVAILABLE,
                           f"{self.name} offers no gradient path")

    def candidate_score_tensor(self, raw: RawOutput, batch_index: int, candidate_index: int,
                               target_class: int, score_type: str):
        raise ExplainError(ErrorCode.GRADIENTS_UNAVAILABLE,
                           f"{self.name} offers no gradient path")

    def assert_inference_mode(self) -> None:
        """Raise if any module of the model is in training mode."""
        model = getattr(self, "torch_module", None)
        if model is None:
            return
        training = [n for n, m in model.named_modules() if m.training]
        if training:
            raise ExplainError(ErrorCode.INTERNAL,
                               f"model modules in training mode: {training[:5]}")


@contextlib.contextmanager
def keep_cuda_visible():
    """Let ultralytics pick a device without hiding the GPU from the rest of the process.

    ultralytics' ``select_device`` writes ``CUDA_VISIBLE_DEVICES`` -- ``""``
    for the CPU -- which, before CUDA is initialised, hides the GPU from
    everything loaded later in the process.  CUDA is initialised first (the
    variable then no longer changes what this process sees) and the variable
    is put back afterwards.
    """
    import os

    import torch

    torch.cuda.is_available()
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved


# -- comparing an adapter with YORU's own detector -------------------------------

#: Largest difference accepted between an adapter and YORU's detector -- the
#: tolerance YORU's own v1-equivalence test uses.  ``exact`` reports whether
#: the two were equal bit for bit.
IDENTITY_TOLERANCE = 1e-4


def yoru_detection_row(d: Mapping[str, Any], rotated: bool) -> List[float]:
    """One of YORU's ``detect()`` dicts as numbers to compare."""
    row = [d["x1"], d["y1"], d["x2"], d["y2"], d["conf"], d["class_id"]]
    if rotated:
        row += [d.get("cx"), d.get("cy"), d.get("w"), d.get("h"), d.get("angle")]
    return row


def detection_row(d: Detection, rotated: bool) -> List[float]:
    row = [*d.box, d.score, d.class_id]
    if rotated:
        row += list(d.obb) if d.obb is not None else [None] * 5
    return row


def compare_detections(reference: str, ref_rows: Sequence[Sequence[Any]],
                       mine_rows: Sequence[Sequence[Any]]) -> Dict[str, Any]:
    """``identity_check``'s verdict on two lists of detection rows, in order."""
    result: Dict[str, Any] = {"reference": reference, "n_reference": len(ref_rows),
                              "n_adapter": len(mine_rows)}
    if len(ref_rows) != len(mine_rows):
        result.update(status="failed", reason="different numbers of detections",
                      max_abs_diff=None, exact=False)
        return result
    diffs = []
    for r, m in zip(ref_rows, mine_rows):
        if len(r) != len(m) or any((a is None) != (b is None) for a, b in zip(r, m)):
            result.update(status="failed", reason="different fields (rotated vs upright boxes?)",
                          max_abs_diff=None, exact=False)
            return result
        diffs += [abs(float(a) - float(b)) for a, b in zip(r, m) if a is not None]
    worst = max(diffs, default=0.0)
    result.update(status="passed" if worst <= IDENTITY_TOLERANCE else "failed", max_abs_diff=worst,
                  exact=[list(r) for r in ref_rows] == [list(m) for m in mine_rows])
    if result["status"] == "failed":
        result["reason"] = f"values differ by up to {worst:.3g}"
    return result
