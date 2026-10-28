# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Any YORU detector as a black box: ``get_detector(...).detect()`` and nothing else.

The adapter holds the very detector YORU builds (the documented
``yoru.libs.plugins.get_detector``) and runs its ``detect()`` on every input,
so its detections are YORU's by construction -- for every backend YORU has,
including ones added after this adapter was written, and for model files the
model-specific adapters do not open (exported ultralytics formats, say).

What a black box cannot give: candidates (there is no pre-NMS table), class
logits, feature maps, gradients.  So it offers re-detection only -- the
target is the unique same-class detection overlapping the original box --
with the reported confidence as the score, and no Grad-CAM.  Each input is
detected on its own, as YORU detects frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from yoru_explain.adapters import _yoru
from yoru_explain.adapters.base import (
    AdapterCapabilities,
    ModelAdapter,
    PreparedInput,
    RawOutput,
    keep_cuda_visible,
)
from yoru_explain.core.errors import ErrorCode, ExplainError
from yoru_explain.core.geometry import LetterboxTransform
from yoru_explain.core.hashing import sha256_file
from yoru_explain.core.logs import get_logger
from yoru_explain.core.types import Detection

log = get_logger("blackbox")


def detection_from_yoru(d: Dict[str, Any]) -> Detection:
    """One of YORU's ``detect()`` dicts as a :class:`Detection` (rotated box included)."""
    obb = None
    if d.get("angle") is not None:
        obb = (float(d["cx"]), float(d["cy"]), float(d["w"]), float(d["h"]), float(d["angle"]))
    return Detection((float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"])), float(d["conf"]),
                     int(d["class_id"]), str(d["class_name"]), None, obb)


def identity_transform(frame_shape) -> LetterboxTransform:
    """The frame is the input: nothing to map."""
    h, w = int(frame_shape[0]), int(frame_shape[1])
    return LetterboxTransform((h, w), (h, w), 1.0, (h, w), 0, 0, 0, 0, kind="none",
                              box_scale=(1.0, 1.0), box_offset=(0.0, 0.0), clip_boxes=False, fill=None)


class YoruDetectorAdapter(ModelAdapter):
    name = "yoru-detector"
    version = "0.2.0"
    description = ("any YORU detector as a black box (get_detector(...).detect()): re-detection with the "
                   "reported confidence; no candidates, no Grad-CAM")

    def __init__(self):
        super().__init__()
        self._detector = None
        self._caps: Optional[AdapterCapabilities] = None

    def load_model(self, weights, device: str = "auto", *, backend: str = "auto",
                   conf_thresh: Optional[float] = None, iou_thresh: Optional[float] = None,
                   probe: bool = True, **detector_options) -> None:
        core = _yoru.core()
        self._core = core
        path = Path(weights)
        if not path.exists():
            raise ExplainError(ErrorCode.MISSING_DATA, f"model file not found: {path}")
        self._weights = str(path)
        self._sha256 = sha256_file(path) if path.is_file() else None
        self._device_pref = str(device or "auto")
        self._conf = core.default_conf_thresh if conf_thresh is None else float(conf_thresh)
        self._iou = core.default_iou_thresh if iou_thresh is None else float(iou_thresh)
        self._backend_requested = str(backend or "auto")
        self._backend = _yoru.auto_backend(path) if self._backend_requested == "auto" else self._backend_requested
        self._options = dict(detector_options)
        try:
            with keep_cuda_visible():
                self._detector = core.get_detector(self._backend, self._weights, conf_thresh=self._conf,
                                                   iou_thresh=self._iou, device=self._device_pref,
                                                   **self._options)
        except Exception as e:
            raise ExplainError(ErrorCode.MODEL_LOAD_FAILED,
                               f"YORU could not load {path.name} as {self._backend!r}: "
                               f"{type(e).__name__}: {e}") from e
        self._names = {int(k): str(v) for k, v in dict(self._detector.names).items()}
        log.info("loaded %s (%s) as YORU's %r detector (%s), black box", path.name,
                 (self._sha256 or "")[:12], self._backend, type(self._detector).__name__)
        self._caps = self._probe() if probe else self._make_caps(None)

    @property
    def weights(self) -> str:
        return self._weights

    @property
    def class_names(self) -> Dict[int, str]:
        return dict(self._names)

    @property
    def capabilities(self) -> AdapterCapabilities:
        return self._caps

    def model_info(self) -> Dict[str, Any]:
        return {
            "adapter": self.name,
            "adapter_version": self.version,
            "black_box": True,
            "backend": self._backend,
            "backend_requested": self._backend_requested,
            "detector": type(self._detector).__name__,
            "detector_options": dict(self._options),
            "weights": self._weights,
            "sha256": self._sha256,
            "class_names": {str(k): v for k, v in self._names.items()},
            "conf_thresh": self._conf,
            "iou_thresh": self._iou,
            "device": self._device_pref,
            "device_preference": self._device_pref,
            "yoru_version": self._core.yoru_version,
        }

    # -- inference -----------------------------------------------------------

    def preprocess(self, images: Sequence[np.ndarray]) -> PreparedInput:
        if not images:
            raise ExplainError(ErrorCode.INTERNAL, "no images to preprocess")
        shape = images[0].shape
        if any(im.shape != shape for im in images):
            raise ExplainError(ErrorCode.INTERNAL, "images of one batch must share a shape")
        return PreparedInput(np.stack(images), identity_transform(shape[:2]), tuple(shape))

    def predict_raw(self, prepared: PreparedInput, *, logits: bool = False) -> RawOutput:
        with self._lock, keep_cuda_visible():
            dets = [[detection_from_yoru(d) for d in self._detector.detect(im)] for im in prepared.tensor]
        return RawOutput(pred=dets)

    def postprocess(self, raw: RawOutput, prepared: PreparedInput) -> List[List[Detection]]:
        return [list(d) for d in raw.pred]

    def candidate_readout(self, raw, prepared, candidate_index, target_class, score_type):
        raise ExplainError(ErrorCode.SCORE_UNAVAILABLE,
                           "a black-box detector has no pre-NMS candidates; use re-detection mode")

    def identity_check(self, image: np.ndarray) -> Dict[str, Any]:
        return {"status": "not_applicable",
                "reason": "the black-box adapter runs YORU's own detector (get_detector(...).detect)"}

    # -- probe -----------------------------------------------------------------

    def _make_caps(self, deterministic: Optional[bool]) -> AdapterCapabilities:
        return AdapterCapabilities(
            fixed_candidate=False,
            redetection=True,
            score_types=("confidence",),
            gradients=False,
            gradients_reason="black box: YORU's detect() gives boxes and confidences only",
            deterministic=deterministic,
            batch_invariant=True,
            batch_max_abs_diff=0.0,
            notes=("each input is detected on its own, as YORU detects frames",),
            family=f"YORU {self._backend} detector (black box)",
            candidate_kind="",
            reported_score="confidence",
            score_descriptions=(("confidence", f"the reported confidence -- what YORU's {self._backend} "
                                               "detector reports for the detection"),),
        )

    def _probe(self) -> AdapterCapabilities:
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
        p = self.preprocess([frame])
        a = self.predict_raw(p).pred[0]
        b = self.predict_raw(p).pred[0]
        return self._make_caps(a == b)
