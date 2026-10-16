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

An adapter never puts its model in training mode and never changes weights.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np

from yoru_explain.core.errors import ErrorCode, ExplainError
from yoru_explain.core.geometry import LetterboxTransform
from yoru_explain.core.types import Box, Detection, ExplanationTarget, ScoreDefinition, ScoreResult


@dataclass(frozen=True)
class LayerInfo:
    """A feature map the gradient path can capture."""

    name: str
    index: int
    type: str
    #: Input pixels per feature cell (the map tiles the model input exactly).
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fixed_candidate": self.fixed_candidate,
            "redetection": self.redetection,
            "score_types": list(self.score_types),
            "gradients": self.gradients,
            "gradients_reason": self.gradients_reason,
            "gradient_layers": list(self.gradient_layers),
            "deterministic": self.deterministic,
            "batch_invariant": self.batch_invariant,
            "batch_max_abs_diff": self.batch_max_abs_diff,
            "notes": list(self.notes),
        }


@dataclass
class PreparedInput:
    """A batch of model inputs and the transform shared by all of them."""

    tensor: Any
    transform: LetterboxTransform
    frame_shape: Tuple[int, ...]

    @property
    def batch_size(self) -> int:
        return int(self.tensor.shape[0])


@dataclass
class RawOutput:
    """A forward pass's candidates, before NMS.

    *pred* is ``(B, N, 5 + nc)``: ``cx, cy, w, h`` in input pixels, then
    probabilities (objectness, classes).  *logits* has the same layout with
    the pre-sigmoid values, when captured.  *activations* holds feature maps
    of a gradient pass.
    """

    pred: Any
    logits: Any = None
    activations: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateReadout:
    """One candidate on one image: its score, where its box is, its class."""

    value: Optional[float]
    box: Optional[Box]
    argmax_class: Optional[int]


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
