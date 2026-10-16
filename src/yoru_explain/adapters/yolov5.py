# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""YOLOv5 checkpoints, run exactly as YORU runs them.

The model is the object YORU's own YOLOv5 plugin builds
(``load_yolov5_model``: ``AutoShape(DetectMultiBackend(weights, fuse=True))``),
and every step around it reproduces ``YOLOv5Detector.detect``:

* frames go in BGR, as in YORU v1, unless YORU's RGB switch says otherwise
  (``YORU_YOLOV5_RGB`` / ``rgb_input``), decided by YORU's own function;
* AutoShape's preprocessing, step by step, with the bundled ``letterbox``
  and ``make_divisible``: 640 px long side, stride multiple, grey padding,
  ``/255`` in the parameter dtype;
* the forward pass under ``torch.inference_mode`` and
  ``torch.amp.autocast("cuda")``, the context YORU's AutoShape uses;
* the bundled ``non_max_suppression`` at ``min(conf, 0.25)`` / IoU 0.45 /
  1000 boxes, a higher confidence threshold applied after it, and
  ``scale_boxes`` back to the frame.

On top of that the adapter recovers which pre-NMS candidate each detection
came from: NMS keeps candidate rows unchanged, so a detection's box equals
``xywh2xyxy`` of exactly one candidate bit for bit (checked, and anything
else is reported, not guessed).  That candidate is what fixed-candidate mode
follows and what Grad-CAM differentiates.

``identity_check`` compares all of this with YORU's documented
``get_detector(...).detect()``; runs record the result.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from yoru_explain.adapters import _yolo_common as yc
from yoru_explain.adapters import _yoru
from yoru_explain.adapters.base import (
    AdapterCapabilities,
    LayerInfo,
    ModelAdapter,
    PreparedInput,
    RawOutput,
)
from yoru_explain.core.errors import ErrorCode, ExplainError
from yoru_explain.core.geometry import LetterboxTransform
from yoru_explain.core.hashing import sha256_file
from yoru_explain.core.logs import get_logger
from yoru_explain.core.types import SCORE_TYPES, CandidateRef, Detection

log = get_logger("yolov5")

#: AutoShape's default long side; YORU never changes it.
DEFAULT_SIZE = 640
#: Rows of the (detections x candidates) comparison done at once.
_MATCH_CHUNK = 64


def _autocast():
    """The autocast context YORU's AutoShape wraps inference in.

    ``torch.amp.autocast("cuda", enabled=True)``.  Without CUDA it would be
    disabled anyway (with a warning per call), so it is disabled up front.
    """
    return torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available())


@contextlib.contextmanager
def _normal_grids(head):
    """Give the Detect head grid tensors autograd may save.

    The head caches its anchor grids from the first forward pass, which ran
    under inference mode; inference tensors cannot be saved for backward.
    Clones made here are ordinary tensors with the same values.  The cached
    ones are put back afterwards.
    """
    saved = (head.grid, head.anchor_grid)
    head.grid = [g.clone() for g in head.grid]
    head.anchor_grid = [g.clone() for g in head.anchor_grid]
    try:
        yield
    finally:
        head.grid, head.anchor_grid = saved


class YoruYolov5Adapter(ModelAdapter):
    name = "yoru-yolov5"
    version = "0.1.0"
    description = "YOLOv5 detection checkpoints via YORU's bundled YOLOv5 code"

    def __init__(self):
        super().__init__()
        self._model = None
        self._caps: Optional[AdapterCapabilities] = None
        self._layer_info: List[LayerInfo] = []

    # -- loading ---------------------------------------------------------

    def load_model(self, weights, device: str = "auto", *, conf_thresh: Optional[float] = None,
                   iou_thresh: Optional[float] = None, rgb_input=None, size: int = DEFAULT_SIZE,
                   probe: bool = True) -> None:
        y = _yoru.load()
        self._y = y
        path = Path(weights)
        if not path.is_file():
            raise ExplainError(ErrorCode.MISSING_DATA, f"model file not found: {path}")
        self._weights = str(path)
        self._sha256 = sha256_file(path)
        self._device_pref = str(device or "auto")
        self._device = y.torch_device(self._device_pref)
        self._conf_thresh = y.default_conf_thresh if conf_thresh is None else float(conf_thresh)
        self._iou = y.default_iou_thresh if iou_thresh is None else float(iou_thresh)
        # As YOLOv5Detector.load: NMS never runs above v1's 0.25.
        self._nms_conf = min(self._conf_thresh, y.v1_nms_conf)
        self._rgb_setting = rgb_input
        self._rgb = bool(y.rgb_input_requested(rgb_input))
        self._size = int(size)
        try:
            model = y.load_yolov5_model(self._weights, self._device)
        except ExplainError:
            raise
        except ValueError as e:
            raise ExplainError(ErrorCode.MODEL_UNSUPPORTED, str(e)) from e
        except Exception as e:
            raise ExplainError(ErrorCode.MODEL_LOAD_FAILED,
                               f"{path.name} could not be loaded: {type(e).__name__}: {e}") from e
        dmb = model.model
        if not getattr(dmb, "pt", False):
            raise ExplainError(ErrorCode.MODEL_UNSUPPORTED,
                               f"{path.name} is not a PyTorch checkpoint; only .pt models are explained")
        head = dmb.model.model[-1]
        if type(head).__name__ != "Detect":
            raise ExplainError(ErrorCode.MODEL_UNSUPPORTED,
                               f"{path.name}: last layer is {type(head).__name__}, not a YOLOv5 Detect head")
        self._model = model
        self._dmb = dmb
        self._layers = dmb.model.model
        self._head = head
        self._stride = int(model.stride)
        self._param = next(dmb.parameters())
        names = model.names
        if isinstance(names, (list, tuple)):
            names = dict(enumerate(names))
        self._names = {int(k): str(v) for k, v in names.items()}
        log.info("loaded %s (%s) on %s; %s frames; NMS conf %.3g, IoU %.3g, conf threshold %.3g",
                 path.name, self._sha256[:12], self._device, "RGB" if self._rgb else "BGR",
                 self._nms_conf, self._iou, self._conf_thresh)
        self._caps = self._probe() if probe else self._unprobed_caps()

    @property
    def torch_module(self):
        return self._dmb

    @property
    def weights(self) -> str:
        return self._weights

    @property
    def rgb(self) -> bool:
        return self._rgb

    # -- facts -----------------------------------------------------------

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
            "weights": self._weights,
            "sha256": self._sha256,
            "class_names": {str(k): v for k, v in self._names.items()},
            "nc": int(self._head.nc),
            "strides": [int(s) for s in self._head.stride.tolist()],
            "anchors_per_level": int(self._head.na),
            "detect_inputs": [int(f) for f in self._head.f],
            "input_size": self._size,
            "stride": self._stride,
            "conf_thresh": self._conf_thresh,
            "nms_conf": self._nms_conf,
            "iou_thresh": self._iou,
            "max_det": int(self._model.max_det),
            "channel_order": "RGB" if self._rgb else "BGR (YORU v1 default)",
            "rgb_input_setting": self._rgb_setting,
            "device": str(self._device),
            "device_preference": self._device_pref,
            "autocast": "torch.amp.autocast('cuda') around forward and NMS, as YORU's AutoShape",
            "precision": "fp16 autocast" if self._device.type == "cuda" else "fp32",
            "fused": True,
            "yoru_version": self._y.yoru_version,
            "bundled_yolov5": self._y.yolov5_dir,
        }

    # -- inference -------------------------------------------------------

    def preprocess(self, images: Sequence[np.ndarray]) -> PreparedInput:
        """AutoShape's preprocessing, for frames of one shape."""
        import cv2

        if not images:
            raise ExplainError(ErrorCode.INTERNAL, "no images to preprocess")
        shape = images[0].shape
        if any(im.shape != shape for im in images):
            raise ExplainError(ErrorCode.INTERNAL, "images of one batch must share a shape")
        if shape[0] < 5:
            # AutoShape would read such an array as CHW and transpose it.
            raise ExplainError(ErrorCode.MODEL_UNSUPPORTED, f"frame too small: {shape}")
        ims = []
        for im in images:
            if self._rgb and im.ndim == 3 and im.shape[2] >= 3:
                im = im[..., 2::-1]  # BGR(A) -> RGB, as YOLOv5Detector.detect
            im = im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            ims.append(im if im.data.contiguous else np.ascontiguousarray(im))
        s = ims[0].shape[:2]
        g = self._size / max(s)
        shape1 = [int(v * g) for v in s]
        shape1 = [self._y.make_divisible(x, self._stride) for x in np.array([shape1]).max(0)]
        x = [self._y.letterbox(im, shape1, auto=False)[0] for im in ims]
        x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))
        tensor = torch.from_numpy(x).to(self._param.device).type_as(self._param) / 255
        return PreparedInput(tensor, LetterboxTransform.yolov5(s, shape1), tuple(shape))

    def _logit_hooks(self, store: list):
        def hook(_m, _i, out):
            store.append(out)
        return [m.register_forward_hook(hook) for m in self._head.m]

    def _gather_logits(self, outs: list) -> torch.Tensor:
        """The Detect convs' outputs in the candidate order of ``pred``."""
        na, no = int(self._head.na), int(self._head.no)
        parts = []
        for o in outs:
            bs, _, ny, nx = o.shape
            parts.append(o.view(bs, na, no, ny, nx).permute(0, 1, 3, 4, 2).reshape(bs, na * ny * nx, no))
        return torch.cat(parts, 1)

    def predict_raw(self, prepared: PreparedInput, *, logits: bool = False) -> RawOutput:
        outs: list = []
        with self._lock:
            hooks = self._logit_hooks(outs) if logits else []
            try:
                with torch.inference_mode(), _autocast():
                    pred = self._dmb(prepared.tensor)
            finally:
                for h in hooks:
                    h.remove()
        return RawOutput(pred=pred, logits=self._gather_logits(outs) if logits else None)

    def postprocess(self, raw: RawOutput, prepared: PreparedInput) -> List[List[Detection]]:
        y = self._y
        m = self._model
        shape1 = list(prepared.transform.input_shape)
        shape0 = prepared.frame_shape[:2]
        results = []
        with self._lock, torch.inference_mode(), _autocast():
            for b in range(raw.pred.shape[0]):
                # One image per call, as YORU always calls it: YOLOv5's NMS has
                # a time limit per call (0.5 s + 0.05 s per image) after which
                # it returns the remaining images of a batch empty.
                o = y.non_max_suppression(raw.pred[b:b + 1], self._nms_conf, self._iou, m.classes,
                                          m.agnostic, m.multi_label, max_det=m.max_det)[0]
                index = self._recover_candidates(raw.pred[b], o)
                boxes = o[:, :4].clone()
                y.scale_boxes(shape1, boxes, shape0)
                rows = torch.cat((boxes, o[:, 4:6]), 1).cpu().tolist()
                dets = []
                for (x1, y1, x2, y2, conf, cls), k in zip(rows, index):
                    if conf < self._conf_thresh:
                        continue  # YOLOv5Detector.detect: a higher threshold after NMS
                    cid = int(cls)
                    cand = CandidateRef(k, yc.locate_candidate(k, self._levels(prepared))) \
                        if k is not None else None
                    dets.append(Detection((x1, y1, x2, y2), conf, cid,
                                          self._names.get(cid, str(cid)), cand))
                results.append(dets)
        return results

    def _recover_candidates(self, pred_b: torch.Tensor, out: torch.Tensor) -> List[Optional[int]]:
        """The candidate row each NMS output row is, by exact box equality."""
        if not len(out):
            return []
        xyxy = self._y.xywh2xyxy(pred_b[:, :4])
        found: List[Optional[int]] = []
        for start in range(0, len(out), _MATCH_CHUNK):
            rows = out[start:start + _MATCH_CHUNK]
            eq = (xyxy[None, :, :] == rows[:, None, :4]).all(-1)
            for r in range(len(rows)):
                ks = eq[r].nonzero().flatten().tolist()
                if len(ks) > 1:
                    # Same box twice: keep the one whose confidence and class
                    # NMS would have produced.
                    conf = (pred_b[ks, 5:] * pred_b[ks, 4:5])
                    best, cls = conf.max(1)
                    ks = [k for k, v, c in zip(ks, best.tolist(), cls.tolist())
                          if v == float(rows[r, 4]) and c == int(rows[r, 5])]
                if len(ks) == 1:
                    found.append(int(ks[0]))
                else:
                    log.warning("detection %s maps to %d candidates; left without one",
                                rows[r, :6].tolist(), len(ks))
                    found.append(None)
        return found

    def _levels(self, prepared: PreparedInput):
        h1, w1 = prepared.transform.input_shape
        return [(int(s), int(self._head.na), h1 // int(s), w1 // int(s))
                for s in self._head.stride.tolist()]

    def candidate_readout(self, raw, prepared, candidate_index, target_class, score_type):
        return yc.readout(raw, prepared.transform, candidate_index, target_class, score_type)

    # -- the check against YORU itself -----------------------------------

    def identity_check(self, image: np.ndarray) -> Dict[str, Any]:
        """Detections here vs. ``get_detector('yolov5', ...).detect(image)``.

        YORU's detector is loaded with the same file, thresholds, device and
        channel order.  Passed means the same number of detections and every
        value equal within 1e-4 (the tolerance YORU's own v1-equivalence test
        uses); ``exact`` says whether they are equal bit for bit.
        """
        y = self._y
        reference = y.get_detector("yolov5", self._weights, conf_thresh=self._conf_thresh,
                                   iou_thresh=self._iou, device=self._device_pref, rgb_input=self._rgb)
        try:
            ref = [[d["x1"], d["y1"], d["x2"], d["y2"], d["conf"], d["class_id"]]
                   for d in reference.detect(image)]
        finally:
            del reference
        mine = [[*d.box, d.score, d.class_id] for d in self.detect(image)]
        result: Dict[str, Any] = {
            "reference": "yoru.libs.plugins.get_detector('yolov5', ...).detect",
            "n_reference": len(ref),
            "n_adapter": len(mine),
        }
        if len(ref) != len(mine):
            result.update(status="failed", reason="different numbers of detections",
                          max_abs_diff=None, exact=False)
            return result
        diffs = [abs(a - b) for r, m_ in zip(ref, mine) for a, b in zip(r, m_)]
        worst = max(diffs, default=0.0)
        result.update(status="passed" if worst <= 1e-4 else "failed", max_abs_diff=worst,
                      exact=ref == mine)
        return result

    # -- gradients -------------------------------------------------------

    def gradient_layers(self) -> List[LayerInfo]:
        return list(self._layer_info)

    def default_gradient_layer(self, target) -> str:
        """The Detect input at the target candidate's own level.

        That is the last feature map before the prediction -- Grad-CAM's
        usual choice.  The head is a 1x1 convolution, so the target's score
        has a gradient at exactly one cell of it (its own); the map then shows
        where the channels that cell weighs most are active elsewhere.
        """
        cand = target.candidate
        if cand is None:
            raise ExplainError(ErrorCode.SCORE_UNAVAILABLE, "the target has no pre-NMS candidate")
        level = cand.locator_dict.get("level", 0)
        return f"model.{int(self._head.f[level])}"

    def _modules_by_name(self):
        return {f"model.{i}": m for i, m in enumerate(self._layers)}

    def forward_with_gradients(self, prepared: PreparedInput, layers: Sequence[str], *,
                               precision: str = "inference") -> RawOutput:
        """A forward pass that keeps the graph from the input to *layers* and the scores.

        ``precision="inference"`` runs under the same autocast as
        ``predict_raw`` (fp16 on CUDA), so the score is the one being
        explained; ``"fp32"`` turns autocast off.  The weights do not require
        gradients and are not touched: the graph starts at a copy of the input.
        """
        if precision not in ("inference", "fp32"):
            raise ExplainError(ErrorCode.INVALID_CONFIG, f"unknown precision {precision!r}")
        modules = self._modules_by_name()
        unknown = [n for n in layers if n not in modules]
        if unknown:
            raise ExplainError(ErrorCode.INVALID_CONFIG, f"unknown layers {unknown}")
        acts: Dict[str, torch.Tensor] = {}
        outs: list = []
        with self._lock:
            hooks = [modules[n].register_forward_hook(lambda m, i, o, n=n: acts.__setitem__(n, o))
                     for n in layers]
            hooks += self._logit_hooks(outs)
            try:
                with torch.inference_mode(False), torch.enable_grad(), _normal_grids(self._head):
                    x = prepared.tensor.detach().clone().requires_grad_(True)
                    ctx = _autocast() if precision == "inference" else \
                        torch.amp.autocast(device_type="cuda", enabled=False)
                    with ctx:
                        pred = self._dmb(x)
            finally:
                for h in hooks:
                    h.remove()
        return RawOutput(pred=pred, logits=self._gather_logits(outs), activations=acts)

    def candidate_score_tensor(self, raw, batch_index, candidate_index, target_class, score_type):
        col = yc.score_column(raw.pred, raw.logits, candidate_index, target_class, score_type)
        return col[batch_index]

    # -- probe -----------------------------------------------------------

    def _unprobed_caps(self) -> AdapterCapabilities:
        return AdapterCapabilities(True, True, tuple(SCORE_TYPES), False, "not probed")

    def _probe(self) -> AdapterCapabilities:
        """Measure determinism, batch dependence and the gradient path on a fixed input."""
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
        p1 = self.preprocess([frame])

        shapes: Dict[str, tuple] = {}
        modules = self._modules_by_name()
        with self._lock:
            hooks = [m.register_forward_hook(
                lambda mod, i, o, n=n: shapes.__setitem__(n, tuple(o.shape) if torch.is_tensor(o) else None))
                for n, m in modules.items()]
            try:
                a = self.predict_raw(p1).pred
            finally:
                for h in hooks:
                    h.remove()
        b = self.predict_raw(p1).pred
        deterministic = bool(torch.equal(a, b))
        c = self.predict_raw(self.preprocess([frame, frame])).pred
        batch_diff = float((c[0, :, 4:] - a[0, :, 4:]).abs().max())

        h1, w1 = p1.transform.input_shape
        info = []
        for i, (n, m) in enumerate(modules.items()):
            s = shapes.get(n)
            if not s or len(s) != 4 or s[2] == 0 or h1 % s[2] or w1 % s[3] or h1 // s[2] != w1 // s[3]:
                continue
            info.append(LayerInfo(n, i, str(getattr(m, "type", type(m).__name__)).split(".")[-1],
                                  h1 // s[2], int(s[1])))
        self._layer_info = info

        notes = []
        if batch_diff != 0.0:
            notes.append(f"scores depend on the batch size (up to {batch_diff:.2g} between batch 1 and 2); "
                         "occlusion takes its reference at the run's batch size")
        grad_ok, reason = self._probe_gradients(p1, a)
        return AdapterCapabilities(
            fixed_candidate=True,
            redetection=True,
            score_types=tuple(SCORE_TYPES),
            gradients=grad_ok,
            gradients_reason=reason,
            gradient_layers=tuple(li.name for li in info),
            deterministic=deterministic,
            batch_invariant=batch_diff == 0.0,
            batch_max_abs_diff=batch_diff,
            notes=tuple(notes),
        )

    def _probe_gradients(self, prepared: PreparedInput, reference_pred) -> tuple:
        try:
            k = int(reference_pred[0, :, 4].argmax())
            level = yc.locate_candidate(k, self._levels(prepared))[0][1]
            layer = f"model.{int(self._head.f[level])}"
            raw = self.forward_with_gradients(prepared, [layer])
            s = self.candidate_score_tensor(raw, 0, k, 0, "obj_x_cls")
            ref = float(reference_pred[0, k, 5] * reference_pred[0, k, 4])
            grad = torch.autograd.grad(s, raw.activations[layer])[0]
        except Exception as e:  # reported, not raised: occlusion still works
            log.warning("gradient probe failed: %s: %s", type(e).__name__, e)
            return False, f"gradient probe failed: {type(e).__name__}: {e}"
        if not bool(torch.isfinite(grad).all()):
            return False, "gradient probe: non-finite gradient"
        if not bool((grad != 0).any()):
            return False, "gradient probe: zero gradient"
        diff = abs(float(s) - ref)
        if diff > 1e-4:
            return False, f"gradient probe: score differs from inference by {diff:.3g}"
        return True, (f"gradient probe passed on {layer} (candidate {k}); "
                      f"gradient-pass score - inference score = {diff:.3g}")
