# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""ultralytics checkpoints -- YOLOv8, YOLO11, YOLOv5u, YOLOv10, RT-DETR -- run as YORU runs them.

YORU's ultralytics plugin (``yoru/libs/plugins/ultralytics_detector.py``)
loads ``ultralytics.YOLO(path)`` (``RTDETR`` for RT-DETR) and calls it on
each frame with YORU's thresholds: ``model(frame, conf=..., iou=...,
device=..., verbose=False)``.  The first call builds ultralytics' predictor,
and every step of a detection is the predictor's:

* ``preprocess``: ultralytics' ``LetterBox`` at the model's own ``imgsz``
  (kept from its training arguments: 640 unless it was trained otherwise;
  1024 for the OBB models) -- the minimum rectangle (``rect``) for YOLO
  models, a stretch to the square for RT-DETR; BGR to RGB; ``/255`` in fp32
  (``half`` is off);
* the fused model's forward pass under ``torch.inference_mode``, no autocast;
* YOLO: ultralytics' ``non_max_suppression`` at YORU's conf / IoU, best class
  only, at most ``max_det`` (300) boxes, probabilistic rotated IoU for OBB
  models, a confidence cut only for YOLOv10's end-to-end head; RT-DETR: no
  NMS -- the queries whose best class score exceeds conf, sorted by score;
* ``scale_boxes`` back to the frame (OBB: ``regularize_rboxes``, then the
  upright box around the rotated one, as ``Results.obb.xyxy`` computes it).

The adapter builds the same model the same way (one predictor call on a
blank frame) and then runs those steps itself, calling ultralytics' own
functions rather than copies, so that it can keep what the predictor throws
away: which candidate each detection came from (NMS's ``return_idxs``), the
class logits, feature maps and gradients.  ``identity_check`` compares the
result with YORU's documented ``get_detector(...).detect()``.

Candidates.  For YOLO models an *anchor point* of the anchor-free head
(level and grid cell): fixed for a given input size, so fixed-candidate mode
follows it; for YOLOv10, the anchor point of its one-to-one branch.  The
score is ``sigmoid(class logit)`` -- there is no objectness.  RT-DETR's
candidates are its decoder *queries*, chosen anew for every input from the
encoder's best tokens: a query index means nothing on another input, so
RT-DETR is followed by re-detection only (Grad-CAM of a query on its own
frame is well defined).
"""

from __future__ import annotations

import contextlib
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from yoru_explain.adapters import _yolo_common as yc
from yoru_explain.adapters import _yoru
from yoru_explain.adapters.base import (
    AdapterCapabilities,
    CandidateReadout,
    LayerInfo,
    ModelAdapter,
    PreparedInput,
    RawOutput,
    compare_detections,
    detection_row,
    keep_cuda_visible,
    yoru_detection_row,
)
from yoru_explain.core.errors import ErrorCode, ExplainError
from yoru_explain.core.geometry import LetterboxTransform
from yoru_explain.core.hashing import sha256_file
from yoru_explain.core.logs import get_logger
from yoru_explain.core.types import CandidateRef, Detection

log = get_logger("ultralytics")

#: The tasks whose models YORU detects with (their boxes; masks / keypoints unused).
TASKS = ("detect", "segment", "pose", "obb")
#: Heads whose class logits are the outputs of their ``cv3`` convolutions
#: (YOLO-World / YOLOE heads score classes against text embeddings instead).
_CV3_HEADS = ("Detect", "Segment", "Pose", "OBB", "v10Detect")
#: Rows of the (detections x candidates) comparison done at once.
_MATCH_CHUNK = 64

_ul_cache = None


def _ul():
    """``(ultralytics, ops, non_max_suppression)``, imported once."""
    global _ul_cache
    if _ul_cache is None:
        try:
            import ultralytics
            from ultralytics.utils import ops

            try:
                from ultralytics.utils.nms import non_max_suppression
            except ImportError:      # older 8.3 releases kept it in ops
                non_max_suppression = ops.non_max_suppression
        except ImportError as e:
            raise ExplainError(ErrorCode.MODEL_UNSUPPORTED,
                               f"ultralytics could not be imported ({e}); YORU installs it") from e
        _ul_cache = (ultralytics, ops, non_max_suppression)
    return _ul_cache


@contextlib.contextmanager
def _graph_safe(module, names: Sequence[str]):
    """Swap tensors a head cached under inference mode for clones autograd may save.

    The head builds its anchors (and strides / masks) on the first forward
    pass at a shape, which ran under inference mode; inference tensors cannot
    be saved for backward.  The clones have the same values.  Everything is
    put back afterwards (the shape too, in case the pass rebuilt them).
    """
    saved = {n: getattr(module, n, None) for n in names}
    for n, v in saved.items():
        if torch.is_tensor(v):
            setattr(module, n, v.clone())
    try:
        yield
    finally:
        for n, v in saved.items():
            setattr(module, n, v)


@contextlib.contextmanager
def _capture_detect_output(head, store: Dict[str, list]):
    """Keep the boxes-and-scores tensor the head computes before adding masks / keypoints / angles.

    Segment, Pose and OBB heads concatenate their extra outputs to it; a score
    read from that tensor needs no backward pass through the extras (Pose's
    keypoint decoding works in place, which autograd refuses).  The values are
    the ones the head reports.
    """
    inference = head._inference

    def wrapped(x):
        y = inference(x)
        store.setdefault("detect", []).append(y)
        return y

    head._inference = wrapped
    try:
        yield
    finally:
        del head._inference


class _UltralyticsAdapter(ModelAdapter):
    """What YOLO and RT-DETR checkpoints share: loading, preprocessing, the identity check."""

    version = "0.2.0"
    #: The ultralytics class YORU loads the checkpoint with.
    _model_class = "YOLO"
    #: The YORU backend this adapter stands in for.
    yoru_backend = "ultralytics"
    #: The head's cached tensors (see :func:`_graph_safe`).
    _cache_names: Tuple[str, ...] = ()

    def __init__(self):
        super().__init__()
        self._model = None
        self._caps: Optional[AdapterCapabilities] = None
        self._layer_info: List[LayerInfo] = []

    # -- loading -------------------------------------------------------------

    def load_model(self, weights, device: str = "auto", *, conf_thresh: Optional[float] = None,
                   iou_thresh: Optional[float] = None, probe: bool = True) -> None:
        core = _yoru.core()
        ultralytics, ops, nms = _ul()
        self._core, self._ops, self._nms = core, ops, nms
        self._nms_has_idxs = "return_idxs" in inspect.signature(nms).parameters
        path = Path(weights)
        if not path.is_file():
            raise ExplainError(ErrorCode.MISSING_DATA, f"model file not found: {path}")
        if path.suffix.lower() != ".pt":
            raise ExplainError(
                ErrorCode.MODEL_UNSUPPORTED,
                f"{path.name}: {self.name} explains PyTorch checkpoints (.pt); an exported model "
                "runs in the black-box adapter (yoru-detector), or use the .pt it was exported from")
        self._weights = str(path)
        self._sha256 = sha256_file(path)
        self._device_pref = str(device or "auto")
        # The device string YORU's plugin hands ultralytics on every call.
        self._device_arg = core.resolve_device(self._device_pref)
        self._conf = core.default_conf_thresh if conf_thresh is None else float(conf_thresh)
        self._iou = core.default_iou_thresh if iou_thresh is None else float(iou_thresh)
        try:
            with keep_cuda_visible():
                model = getattr(ultralytics, self._model_class)(self._weights)
                if model.task not in TASKS:
                    raise ExplainError(ErrorCode.MODEL_UNSUPPORTED,
                                       f"{path.name} is a {model.task} model; YORU detects with "
                                       f"{', '.join(TASKS)} models")
                # The predictor, built exactly as YORU's first detect() builds it.
                model(np.zeros((64, 64, 3), np.uint8), conf=self._conf, iou=self._iou,
                      device=self._device_arg, verbose=False)
        except ExplainError:
            raise
        except Exception as e:
            raise ExplainError(ErrorCode.MODEL_LOAD_FAILED,
                               f"{path.name} could not be loaded: {type(e).__name__}: {e}") from e
        p = model.predictor
        backend = p.model
        net = getattr(backend, "model", None)
        if not getattr(backend, "pt", False) or not isinstance(net, torch.nn.Module):
            raise ExplainError(ErrorCode.MODEL_UNSUPPORTED,
                               f"{path.name} did not load as a PyTorch model")
        self._model, self._predictor, self._backend, self._net = model, p, backend, net
        self._layers = net.model
        self._head = self._layers[-1]
        self._task = str(model.task)
        names = model.names
        if isinstance(names, (list, tuple)):
            names = dict(enumerate(names))
        self._names = {int(k): str(v) for k, v in names.items()}
        self._imgsz = [int(v) for v in p.imgsz]
        self._stride = int(max(np.atleast_1d(np.asarray(backend.stride, dtype=float))))
        self._check_head(path.name)
        log.info("loaded %s (%s) with ultralytics.%s on %s; %s, imgsz %s, conf %.3g, IoU %.3g",
                 path.name, self._sha256[:12], self._model_class, p.device, type(self._head).__name__,
                 self._imgsz, self._conf, self._iou)
        self._caps = self._probe() if probe else self._unprobed_caps()

    #: What loading found that the person must know (capabilities.warnings).
    _load_notes: Tuple[str, ...] = ()

    def _check_head(self, filename: str) -> None:
        raise NotImplementedError

    @property
    def torch_module(self):
        return self._net

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
        import ultralytics

        a = self._predictor.args
        return {
            "adapter": self.name,
            "adapter_version": self.version,
            "backend": self.yoru_backend,
            "weights": self._weights,
            "sha256": self._sha256,
            "class_names": {str(k): v for k, v in self._names.items()},
            "nc": len(self._names),
            "task": self._task,
            "head": type(self._head).__name__,
            "detect_inputs": [int(f) for f in getattr(self._head, "f", [])],
            "imgsz": list(self._imgsz),
            "input_size": max(self._imgsz),
            "rect": bool(a.rect),
            "stride": self._stride,
            "conf_thresh": self._conf,
            "nms_conf": float(a.conf),
            "iou_thresh": self._iou,
            "max_det": int(a.max_det),
            "agnostic_nms": bool(a.agnostic_nms),
            "channel_order": "RGB (ultralytics converts OpenCV's BGR frames)",
            "device": str(self._predictor.device),
            "device_preference": self._device_pref,
            "precision": "fp16" if self._backend.fp16 else "fp32",
            "fused": True,
            "ultralytics_version": str(ultralytics.__version__),
            "yoru_version": self._core.yoru_version,
            **self._extra_info(),
        }

    def _extra_info(self) -> Dict[str, Any]:
        return {}

    # -- inference -----------------------------------------------------------

    def _transform(self, frame_shape) -> LetterboxTransform:
        raise NotImplementedError

    def preprocess(self, images: Sequence[np.ndarray]) -> PreparedInput:
        """ultralytics' own ``predictor.preprocess`` (letterbox, RGB, /255), for frames of one shape."""
        if not images:
            raise ExplainError(ErrorCode.INTERNAL, "no images to preprocess")
        shape = images[0].shape
        if any(im.shape != shape for im in images):
            raise ExplainError(ErrorCode.INTERNAL, "images of one batch must share a shape")
        if len(shape) != 3 or shape[2] != 3:
            raise ExplainError(ErrorCode.MODEL_UNSUPPORTED, f"expected BGR frames, got shape {shape}")
        with self._lock:
            x = self._predictor.preprocess([np.ascontiguousarray(im) for im in images])
        transform = self._transform(shape[:2])
        if tuple(x.shape[2:]) != tuple(transform.input_shape):
            raise ExplainError(ErrorCode.PREPROCESS_MISMATCH,
                               f"ultralytics made a {tuple(x.shape[2:])} input; the transform expects "
                               f"{transform.input_shape}")
        return PreparedInput(x, transform, tuple(shape))

    def _forward(self, x):
        a = self._predictor.args
        return self._backend(x, augment=a.augment, visualize=False, embed=a.embed)

    def _logit_modules(self) -> List[torch.nn.Module]:
        return []

    def _hooks(self, store: Dict[str, list], logits: bool) -> list:
        hooks = []
        if logits:
            store["logits"] = []
            hooks += [m.register_forward_hook(lambda _m, _i, o: store["logits"].append(o))
                      for m in self._logit_modules()]
        return hooks

    def _raw(self, out, store: Dict[str, list], activations=None) -> RawOutput:
        raise NotImplementedError

    def predict_raw(self, prepared: PreparedInput, *, logits: bool = False) -> RawOutput:
        store: Dict[str, list] = {}
        with self._lock:
            hooks = self._hooks(store, logits)
            try:
                with torch.inference_mode():
                    out = self._forward(prepared.tensor)
            finally:
                for h in hooks:
                    h.remove()
            with torch.inference_mode():
                return self._raw(out, store)

    def needs_logits(self, score_type: str) -> bool:
        return score_type == "cls_logit"

    # -- the check against YORU itself -----------------------------------------

    def identity_check(self, image: np.ndarray) -> Dict[str, Any]:
        """Detections here vs. ``get_detector('<backend>', ...).detect(image)``.

        YORU's detector is loaded with the same file, thresholds and device.
        Passed: the same number of detections, every value within 1e-4;
        ``exact``: equal bit for bit.  Rotated boxes are compared too.
        """
        rotated = self._task == "obb"
        with keep_cuda_visible():
            reference = self._core.get_detector(self.yoru_backend, self._weights,
                                                conf_thresh=self._conf, iou_thresh=self._iou,
                                                device=self._device_pref)
            try:
                ref = [yoru_detection_row(d, rotated) for d in reference.detect(image)]
            finally:
                del reference
        mine = [detection_row(d, rotated) for d in self.detect(image)]
        return compare_detections(f"yoru.libs.plugins.get_detector('{self.yoru_backend}', ...).detect",
                                  ref, mine)

    # -- gradients -----------------------------------------------------------------

    def gradient_layers(self) -> List[LayerInfo]:
        return list(self._layer_info)

    def _modules_by_name(self):
        modules = {f"model.{i}": m for i, m in enumerate(self._layers)}
        modules.update(self._branch_modules())
        return modules

    def _branch_modules(self) -> Dict[str, torch.nn.Module]:
        """Modules inside the head that may serve as Grad-CAM layers."""
        return {}

    def _gradient_layer_ok(self, name: str) -> bool:
        return True

    def forward_with_gradients(self, prepared: PreparedInput, layers: Sequence[str], *,
                               precision: str = "inference") -> RawOutput:
        """A forward pass that keeps the graph from the input to *layers* and the scores.

        ultralytics infers in fp32 without autocast, so ``inference`` and
        ``fp32`` are the same pass.  The weights are not touched: the graph
        starts at a copy of the input.
        """
        if precision not in ("inference", "fp32"):
            raise ExplainError(ErrorCode.INVALID_CONFIG, f"unknown precision {precision!r}")
        modules = self._modules_by_name()
        unknown = [n for n in layers if n not in modules]
        if unknown:
            raise ExplainError(ErrorCode.INVALID_CONFIG, f"unknown layers {unknown}")
        acts: Dict[str, torch.Tensor] = {}
        store: Dict[str, list] = {}
        with self._lock:
            leaves = self._leaf_layers()
            hooks = [modules[n].register_forward_hook(lambda m, i, o, n=n: acts.__setitem__(n, o))
                     for n in layers if n not in leaves]
            hooks += self._hooks(store, True)
            hooks += self._leaf_hooks(acts, layers)
            try:
                with torch.inference_mode(False), torch.enable_grad(), \
                        _graph_safe(self._head, self._cache_names), self._grad_capture(store):
                    x = prepared.tensor.detach().clone().requires_grad_(True)
                    out = self._forward(x)
                    raw = self._raw(out, store, acts)
            finally:
                for h in hooks:
                    h.remove()
        return raw

    def _grad_capture(self, store: Dict[str, list]):
        return contextlib.nullcontext()

    def _leaf_layers(self) -> Dict[str, Any]:
        """Layers whose activation is a leaf copy made where the score's branch reads it."""
        return {}

    def _leaf_hooks(self, acts: Dict[str, torch.Tensor], layers: Sequence[str]) -> list:
        return []

    def candidate_score_tensor(self, raw, batch_index, candidate_index, target_class, score_type):
        return self._score_column(raw, int(candidate_index), int(target_class), score_type)[batch_index]

    def _score_column(self, raw: RawOutput, k: int, c: int, score_type: str) -> torch.Tensor:
        if score_type in ("confidence", "cls_prob"):
            return raw.pred[:, k, 4 + c]
        if score_type == "cls_logit":
            if raw.logits is None:
                raise ExplainError(ErrorCode.SCORE_UNAVAILABLE,
                                   "cls_logit needs the head's logits, which were not captured")
            return raw.logits[:, k, c]
        raise ExplainError(ErrorCode.MODEL_UNSUPPORTED, f"{self.name} cannot read {score_type!r}")

    def _check_candidate(self, raw: RawOutput, k: int, c: int) -> None:
        if not 0 <= k < raw.pred.shape[1]:
            raise ExplainError(ErrorCode.SCORE_UNAVAILABLE,
                               f"candidate {k} out of range (model has {raw.pred.shape[1]})")
        if not 0 <= c < len(self._names):
            raise ExplainError(ErrorCode.SCORE_UNAVAILABLE,
                               f"class {c} out of range ({len(self._names)} classes)")

    # -- probe ---------------------------------------------------------------------

    def _family(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _score_types(self, logits_ok: bool) -> Tuple[str, ...]:
        return ("confidence", "cls_prob") + (("cls_logit",) if logits_ok else ())

    def _unprobed_caps(self) -> AdapterCapabilities:
        return AdapterCapabilities(self._fixed_candidate(), True, self._score_types(False), False,
                                   "not probed", warnings=self._load_notes, **self._family())

    def _fixed_candidate(self) -> bool:
        return True

    def _batch_diff(self, one: torch.Tensor, two: torch.Tensor) -> float:
        """Largest class-score difference of image 0 between batch 1 and batch 2."""
        nc = len(self._names)
        return float((two[0, :, 4:4 + nc] - one[0, :, 4:4 + nc]).abs().max())

    def _probe_target(self, raw: RawOutput) -> Tuple[int, int]:
        """The candidate the gradient probe differentiates: the best-scoring one."""
        nc = len(self._names)
        best, cls = raw.pred[0, :, 4:4 + nc].max(1)
        k = int(best.argmax())
        return k, int(cls[k])

    def _probe(self) -> AdapterCapabilities:
        """Measure determinism, batch dependence, the logits and the gradient path."""
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
        p1 = self.preprocess([frame])
        shapes: Dict[str, tuple] = {}
        modules = self._modules_by_name()
        logits_wanted = bool(self._logit_modules())
        with self._lock:
            hooks = [m.register_forward_hook(
                lambda mod, i, o, n=n: shapes.__setitem__(n, tuple(o.shape) if torch.is_tensor(o) else None))
                for n, m in modules.items()]
            try:
                a = self.predict_raw(p1, logits=logits_wanted)
            finally:
                for h in hooks:
                    h.remove()
        b = self.predict_raw(p1)
        deterministic = bool(torch.equal(a.pred, b.pred))
        c = self.predict_raw(self.preprocess([frame, frame])).pred
        batch_diff = self._batch_diff(a.pred, c)

        notes = []
        logits_ok = False
        if logits_wanted:
            nc = len(self._names)
            logits_ok = a.logits is not None and a.logits.shape[:2] == a.pred.shape[:2] and \
                bool(torch.equal(a.logits.sigmoid(), a.pred[..., 4:4 + nc]))
            if not logits_ok:
                notes.append("the class logits read from the head do not reproduce its scores; "
                             "cls_logit is not offered")
        h1, w1 = p1.transform.input_shape
        info = []
        for i, (n, m) in enumerate(modules.items()):
            s = shapes.get(n)
            if not s or len(s) != 4 or s[2] == 0 or h1 % s[2] or w1 % s[3] or h1 // s[2] != w1 // s[3]:
                continue
            if self._gradient_layer_ok(n):
                info.append(LayerInfo(n, i, type(m).__name__, h1 // s[2], int(s[1])))
        self._layer_info = info
        if batch_diff != 0.0:
            notes.append(f"scores depend on the batch size (up to {batch_diff:.2g} between batch 1 and 2); "
                         "occlusion takes its reference at the run's batch size")
        grad_ok, reason = self._probe_gradients(p1, a)
        return AdapterCapabilities(
            fixed_candidate=self._fixed_candidate(),
            redetection=True,
            score_types=self._score_types(logits_ok),
            gradients=grad_ok,
            gradients_reason=reason,
            gradient_layers=tuple(li.name for li in info),
            deterministic=deterministic,
            batch_invariant=batch_diff == 0.0,
            batch_max_abs_diff=batch_diff,
            notes=tuple(notes),
            warnings=self._load_notes,
            **self._family(),
        )

    def _probe_gradients(self, prepared: PreparedInput, reference: RawOutput) -> Tuple[bool, str]:
        try:
            k, c = self._probe_target(reference)
            cand = CandidateRef(k, self._locate(k, prepared, reference, 0))
            layer = self._default_layer_of(cand)
            raw = self.forward_with_gradients(prepared, [layer])
            s = self.candidate_score_tensor(raw, 0, k, c, "confidence")
            ref = float(self._score_column(reference, k, c, "confidence")[0])
            grad = torch.autograd.grad(s, raw.activations[layer])[0]
        except Exception as e:  # reported, not raised: occlusion still works
            log.warning("gradient probe failed: %s: %s", type(e).__name__, e)
            return False, f"gradient probe failed: {type(e).__name__}: {e}"
        if not bool(torch.isfinite(grad).all()):
            return False, "gradient probe: non-finite gradient"
        if not bool((grad != 0).any()):
            return False, "gradient probe: zero gradient"
        diff = abs(float(s.detach()) - ref)
        if diff > 1e-4:
            return False, f"gradient probe: score differs from inference by {diff:.3g}"
        return True, (f"gradient probe passed on {layer} (candidate {k}); "
                      f"gradient-pass score - inference score = {diff:.3g}")

    def default_gradient_layer(self, target) -> str:
        cand = target.candidate
        if cand is None:
            raise ExplainError(ErrorCode.SCORE_UNAVAILABLE, "the target has no candidate")
        return self._default_layer_of(cand)

    def _default_layer_of(self, cand: CandidateRef) -> str:
        """The head's input at the candidate's level -- the last feature map before its prediction."""
        level = int(cand.locator_dict.get("level", len(self._head.f) - 1))
        return f"model.{int(self._head.f[level])}"

    def _locate(self, k: int, prepared: PreparedInput, raw: RawOutput, b: int):
        raise NotImplementedError


class YoruUltralyticsAdapter(_UltralyticsAdapter):
    """YOLOv8 / YOLO11 / YOLOv5u / YOLOv9 / YOLOv10 (detect, segment, pose, OBB) checkpoints."""

    name = "yoru-ultralytics"
    description = ("YOLOv8 / YOLO11 / YOLOv5u / YOLOv10 checkpoints (detect, OBB, segment, pose) via "
                   "ultralytics, as YORU runs them")
    _model_class = "YOLO"
    yoru_backend = "ultralytics"
    _cache_names = ("anchors", "strides", "shape")

    def _check_head(self, filename: str) -> None:
        from ultralytics.nn.modules.head import Detect

        head = self._head
        if not isinstance(head, Detect):
            raise ExplainError(
                ErrorCode.MODEL_UNSUPPORTED,
                f"{filename}: last layer is {type(head).__name__}, not a YOLO detection head"
                + (" -- an RT-DETR checkpoint: choose the rtdetr model type"
                   if type(head).__name__ == "RTDETRDecoder" else ""))
        self._kind = type(head).__name__
        self._end2end = bool(getattr(head, "end2end", False))
        self._obb = self._task == "obb"
        self._nc = int(head.nc)
        if self._nc != len(self._names):
            raise ExplainError(ErrorCode.MODEL_UNSUPPORTED,
                               f"{filename}: the head has {self._nc} classes, the names {len(self._names)}")
        # End-to-end heads decode xyxy boxes; the others xywh.
        self._xyxy = self._end2end or bool(getattr(head, "xyxy", False))
        yaml = getattr(self._net, "yaml", None) or {}
        if yaml.get("end2end") and not self._end2end:
            import ultralytics

            note = (f"{filename} was built for an end-to-end head (its model YAML says end2end, as "
                    f"YOLO26's does), but ultralytics {ultralytics.__version__} runs it with the "
                    "one-to-many head and NMS; YORU's detections come out that way too, and may be "
                    "wrong -- a newer ultralytics may be needed")
            log.warning(note)
            self._load_notes = self._load_notes + (note,)

    def _extra_info(self) -> Dict[str, Any]:
        return {"end2end": self._end2end, "strides": [int(s) for s in self._head.stride.tolist()],
                "box_layout": "xyxy" if self._xyxy else ("xywhr" if self._obb else "xywh"),
                "letterbox": "ultralytics LetterBox(imgsz, auto=rect, stride) + scale_boxes"}

    def _family(self) -> Dict[str, Any]:
        task = {"detect": "", "segment": ", segmentation model: boxes only", "pose": ", pose model: boxes only",
                "obb": ", rotated boxes"}[self._task]
        e2e = ", end-to-end (one-to-one) head" if self._end2end else ""
        return {
            "family": f"ultralytics YOLO ({self._kind} head{task}{e2e})",
            "candidate_kind": "anchor point",
            "reported_score": "cls_prob",
            # The end-to-end head ranks (anchor point, class) pairs: one
            # anchor point can be reported under two classes.
            "multi_label": self._end2end,
            "score_descriptions": (
                ("confidence", "sigmoid(class logit) -- the confidence YOLOv8/YOLO11 report, as YORU "
                               "reports it (the same number as cls_prob); there is no objectness"),
                ("cls_prob", "sigmoid(class logit) -- the confidence this head reports"),
                ("cls_logit", "class logit, before the sigmoid"),
            ),
        }

    def _transform(self, frame_shape) -> LetterboxTransform:
        a = self._predictor.args
        return LetterboxTransform.ultralytics(frame_shape, self._imgsz, auto=bool(a.rect and self._backend.pt),
                                              stride=self._stride)

    def _logit_modules(self):
        if self._kind not in _CV3_HEADS:
            return []
        return list(self._head.one2one_cv3 if self._end2end else self._head.cv3)

    def _grad_capture(self, store):
        # The end-to-end head's dense scores are decoded by _raw itself.
        return contextlib.nullcontext() if self._end2end else _capture_detect_output(self._head, store)

    def candidate_score_tensor(self, raw, batch_index, candidate_index, target_class, score_type):
        detect = raw.extra.get("detect")
        if detect is not None and score_type in ("confidence", "cls_prob"):
            k, c = int(candidate_index), int(target_class)
            return detect[batch_index, 4 + c, k]
        return super().candidate_score_tensor(raw, batch_index, candidate_index, target_class, score_type)

    # YOLOv10's one-to-one branch -- the one that reports -- reads the head's
    # inputs *detached*, with frozen weights, so nothing in it carries a graph
    # back to the backbone.  For the gradient pass each input is handed to
    # the branch as a leaf copy (same values) that requires a gradient: the
    # score's gradient at the head input, through the branch that made it.
    # Earlier layers have no gradient path to the score and are not offered.

    def _leaf_layers(self) -> Dict[str, int]:
        if not self._end2end:
            return {}
        return {f"model.{int(f)}": lvl for lvl, f in enumerate(self._head.f)}

    def _leaf_hooks(self, acts, layers):
        if not self._end2end:
            return []
        wanted = {lvl: name for name, lvl in self._leaf_layers().items() if name in layers}

        def make(lvl):
            def pre(_module, args):
                leaf = args[0].detach().requires_grad_(True)
                if lvl in wanted:
                    acts[wanted[lvl]] = leaf
                return (leaf, *args[1:])
            return pre

        return [self._head.one2one_cv3[lvl].register_forward_pre_hook(make(lvl))
                for lvl in range(int(self._head.nl))]

    def _branch_modules(self):
        if not self._end2end:
            return {}
        h = len(self._layers) - 1
        return {f"model.{h}.one2one_cv3.{lvl}.{j}": self._head.one2one_cv3[lvl][j]
                for lvl in range(int(self._head.nl)) for j in (0, 1)}

    def _gradient_layer_ok(self, name: str) -> bool:
        if not self._end2end:
            return True
        return name in self._leaf_layers() or ".one2one_cv3." in name

    def _raw(self, out, store, activations=None) -> RawOutput:
        y = out[0] if isinstance(out, (list, tuple)) else out
        extra: Dict[str, Any] = {}
        if self._end2end:
            branches = out[1] if isinstance(out, (list, tuple)) and len(out) > 1 else None
            if not isinstance(branches, dict) or "one2one" not in branches:
                raise ExplainError(ErrorCode.INTERNAL, "the end-to-end head did not return its branches")
            # The one-to-one branch's dense candidates, decoded as the head decodes them.
            dense = self._head._inference(branches["one2one"])
            extra["final"] = y
        else:
            dense = y
        extra["dense"] = dense
        if store.get("detect"):
            extra["detect"] = store["detect"][-1]
        logits = None
        if store.get("logits"):
            b = dense.shape[0]
            logits = torch.cat([o.reshape(b, self._nc, -1) for o in store["logits"]], 2).transpose(1, 2)
        return RawOutput(pred=dense.transpose(1, 2), logits=logits, activations=activations or {},
                         extra=extra)

    def _nms_args(self):
        a = self._predictor.args
        return (a.conf, a.iou, a.classes, a.agnostic_nms), dict(
            max_det=a.max_det, nc=0 if a.task == "detect" else len(self._backend.names),
            end2end=getattr(self._backend, "end2end", False), rotated=a.task == "obb")

    def postprocess(self, raw: RawOutput, prepared: PreparedInput) -> List[List[Detection]]:
        args, kw = self._nms_args()
        shape1 = tuple(prepared.tensor.shape[2:])
        out = []
        with self._lock, torch.inference_mode():
            for b in range(raw.pred.shape[0]):
                # One image per call, as YORU calls it (the NMS has a time limit per call).
                if self._end2end:
                    rows = self._nms(raw.extra["final"][b:b + 1], *args, **kw)[0]
                    index = self._match_rows(raw.pred[b], rows)
                elif self._nms_has_idxs:
                    # NMS rewrites its input's boxes in place: give it a copy.
                    kept, idxs = self._nms(raw.extra["dense"][b:b + 1].clone(), *args, **kw,
                                           return_idxs=True)
                    rows = kept[0]
                    index = [int(k) for k in idxs[0].reshape(-1).tolist()]
                else:
                    rows = self._nms(raw.extra["dense"][b:b + 1].clone(), *args, **kw)[0]
                    index = self._match_rows(raw.pred[b], rows)
                out.append(self._detections(rows, index, shape1, prepared, raw, b))
        return out

    def _match_rows(self, pred_b: torch.Tensor, rows: torch.Tensor) -> List[Optional[int]]:
        """The candidate each output row is, by exact box (and score) equality."""
        if not len(rows):
            return []
        boxes = pred_b[:, :4]
        if not (self._xyxy or self._obb):
            boxes = self._ops.xywh2xyxy(boxes)
        found: List[Optional[int]] = []
        for start in range(0, len(rows), _MATCH_CHUNK):
            chunk = rows[start:start + _MATCH_CHUNK]
            eq = (boxes[None, :, :] == chunk[:, None, :4]).all(-1)
            for r in range(len(chunk)):
                ks = eq[r].nonzero().flatten()
                if len(ks) > 1:
                    c = int(chunk[r, 5])
                    ks = ks[pred_b[ks, 4 + c] == chunk[r, 4]]
                if len(ks) == 1:
                    found.append(int(ks[0]))
                else:
                    log.warning("detection %s maps to %d candidates; left without one",
                                chunk[r, :6].tolist(), len(ks))
                    found.append(None)
        return found

    def _frame_boxes(self, rows4: torch.Tensor, angles: Optional[torch.Tensor], shape1, shape0):
        """``(xyxy, xywhr or None)`` in frame pixels, as ultralytics' Results compute them."""
        ops = self._ops
        if angles is not None:
            rb = ops.regularize_rboxes(torch.cat([rows4, angles], dim=-1))
            rb[:, :4] = ops.scale_boxes(shape1, rb[:, :4], shape0, xywh=True)
            corners = ops.xywhr2xyxyxyxy(rb)
            x, y = corners[..., 0], corners[..., 1]
            return torch.stack([x.amin(1), y.amin(1), x.amax(1), y.amax(1)], -1), rb
        return ops.scale_boxes(shape1, rows4.clone(), shape0), None

    def _detections(self, rows, index, shape1, prepared, raw, b) -> List[Detection]:
        if not len(rows):
            return []
        shape0 = prepared.frame_shape
        xyxy, rb = self._frame_boxes(rows[:, :4], rows[:, -1:] if self._obb else None, shape1, shape0)
        xyxy, conf, cls = xyxy.cpu().tolist(), rows[:, 4].cpu().tolist(), rows[:, 5].cpu().tolist()
        obbs = rb.cpu().tolist() if rb is not None else None
        dets = []
        for i, k in enumerate(index):
            cid = int(cls[i])
            cand = CandidateRef(k, self._locate(k, prepared, raw, b)) if k is not None else None
            dets.append(Detection(tuple(xyxy[i]), conf[i], cid, self._names.get(cid, str(cid)), cand,
                                  tuple(obbs[i]) if obbs is not None else None))
        return dets

    def _locate(self, k: int, prepared: PreparedInput, raw: RawOutput, b: int):
        h1, w1 = prepared.tensor.shape[2:]
        levels = [(int(s), 1, int(h1) // int(s), int(w1) // int(s)) for s in self._head.stride.tolist()]
        return tuple(p for p in yc.locate_candidate(k, levels) if p[0] != "anchor")

    def candidate_readout(self, raw, prepared, candidate_index, target_class, score_type):
        k, c = int(candidate_index), int(target_class)
        self._check_candidate(raw, k, c)
        shape1, shape0 = tuple(prepared.tensor.shape[2:]), prepared.frame_shape
        with torch.inference_mode():
            values = self._score_column(raw, k, c, score_type).detach().float().cpu().numpy()
            row = raw.pred[:, k, :].detach()
            classes = row[:, 4:4 + self._nc].argmax(1).cpu().numpy()
            boxes4 = row[:, :4] if (self._xyxy or self._obb) else self._ops.xywh2xyxy(row[:, :4])
            xyxy, rb = self._frame_boxes(boxes4, row[:, -1:] if self._obb else None, shape1, shape0)
            xyxy = xyxy.cpu().tolist()
            obbs = rb.cpu().tolist() if rb is not None else None
        return [CandidateReadout(float(values[i]), tuple(xyxy[i]), int(classes[i]),
                                 tuple(obbs[i]) if obbs is not None else None)
                for i in range(len(values))]


class YoruRtdetrAdapter(_UltralyticsAdapter):
    """RT-DETR checkpoints (``ultralytics.RTDETR``)."""

    name = "yoru-rtdetr"
    description = "RT-DETR checkpoints via ultralytics, as YORU runs them (re-detection; Grad-CAM per query)"
    _model_class = "RTDETR"
    yoru_backend = "rtdetr"
    _cache_names = ("anchors", "valid_mask", "shapes")

    def _check_head(self, filename: str) -> None:
        from ultralytics.nn.modules.head import RTDETRDecoder

        head = self._head
        if not isinstance(head, RTDETRDecoder):
            raise ExplainError(ErrorCode.MODEL_UNSUPPORTED,
                               f"{filename}: last layer is {type(head).__name__}, not RT-DETR's decoder -- "
                               "choose the ultralytics model type")
        self._nc = int(head.nc)
        self._nq = int(head.num_queries)

    def _extra_info(self) -> Dict[str, Any]:
        return {"num_queries": self._nq, "decoder_inputs": [int(f) for f in self._head.f],
                "letterbox": "ultralytics LetterBox(imgsz, auto=False, scale_fill=True): stretched; "
                             "normalised boxes times the frame size"}

    def _fixed_candidate(self) -> bool:
        # A query index is chosen anew for every input (top-k encoder tokens).
        return False

    def _family(self) -> Dict[str, Any]:
        return {
            "family": "RT-DETR (ultralytics)",
            "candidate_kind": "query",
            "reported_score": "cls_prob",
            "score_descriptions": (
                ("confidence", "sigmoid(class logit) -- the confidence RT-DETR reports for a decoder "
                               "query, as YORU reports it (the same number as cls_prob)"),
                ("cls_prob", "sigmoid(class logit) -- the reported confidence"),
                ("cls_logit", "class logit, before the sigmoid"),
            ),
        }

    def _transform(self, frame_shape) -> LetterboxTransform:
        return LetterboxTransform.ultralytics(frame_shape, self._imgsz, auto=False, scale_fill=True)

    def _logit_modules(self):
        head = self._head
        return [head.dec_score_head[int(head.decoder.eval_idx)]]

    def _hooks(self, store, logits):
        hooks = super()._hooks(store, logits)
        head = self._head
        store["enc_scores"], store["level_sizes"] = [], []
        hooks.append(head.enc_score_head.register_forward_hook(
            lambda _m, _i, o: store["enc_scores"].append(o)))
        hooks.append(head.register_forward_pre_hook(
            lambda _m, args: store["level_sizes"].append([tuple(f.shape[2:]) for f in args[0]])))
        return hooks

    def _raw(self, out, store, activations=None) -> RawOutput:
        y = out[0] if isinstance(out, (list, tuple)) else out
        extra: Dict[str, Any] = {}
        if store.get("enc_scores") and store.get("level_sizes"):
            # The encoder tokens the queries were selected from, as the decoder selects them.
            scores = store["enc_scores"][-1]
            extra["tokens"] = torch.topk(scores.max(-1).values, self._nq, dim=1).indices.detach().cpu()
            extra["level_sizes"] = store["level_sizes"][-1]
        logits = store["logits"][-1] if store.get("logits") else None
        return RawOutput(pred=y, logits=logits, activations=activations or {}, extra=extra)

    def postprocess(self, raw: RawOutput, prepared: PreparedInput) -> List[List[Detection]]:
        """RTDETRPredictor.postprocess, keeping the query of each detection."""
        ops = self._ops
        a = self._predictor.args
        y = raw.pred
        nd = y.shape[-1]
        oh, ow = prepared.frame_shape[:2]
        out = []
        with self._lock, torch.inference_mode():
            bboxes, scores = y.split((4, nd - 4), dim=-1)
            for b in range(y.shape[0]):
                bbox = ops.xywh2xyxy(bboxes[b])
                max_score, cls = scores[b].max(-1, keepdim=True)
                idx = max_score.squeeze(-1) > a.conf
                if a.classes is not None:
                    idx = (cls == torch.tensor(a.classes, device=cls.device)).any(1) & idx
                pred = torch.cat([bbox, max_score, cls], dim=-1)[idx]
                queries = torch.arange(len(idx), device=idx.device)[idx]
                order = pred[:, 4].argsort(descending=True)
                pred, queries = pred[order][: a.max_det], queries[order][: a.max_det]
                pred[..., [0, 2]] *= ow
                pred[..., [1, 3]] *= oh
                rows = pred.cpu().tolist()
                dets = []
                for (x1, y1, x2, y2, conf, c), q in zip(rows, queries.cpu().tolist()):
                    cid = int(c)
                    dets.append(Detection((x1, y1, x2, y2), conf, cid, self._names.get(cid, str(cid)),
                                          CandidateRef(int(q), self._locate(int(q), prepared, raw, b))))
                out.append(dets)
        return out

    def _locate(self, q: int, prepared: PreparedInput, raw: RawOutput, b: int):
        loc = [("query", int(q))]
        tokens, sizes = raw.extra.get("tokens"), raw.extra.get("level_sizes")
        if tokens is not None and sizes:
            t = int(tokens[b, q])
            offset = 0
            for level, (h, w) in enumerate(sizes):
                if t < offset + h * w:
                    gy, gx = divmod(t - offset, w)
                    loc += [("token", t), ("level", level), ("grid_y", int(gy)), ("grid_x", int(gx))]
                    break
                offset += h * w
        return tuple(loc)

    def candidate_readout(self, raw, prepared, candidate_index, target_class, score_type):
        q, c = int(candidate_index), int(target_class)
        self._check_candidate(raw, q, c)
        oh, ow = prepared.frame_shape[:2]
        with torch.inference_mode():
            values = self._score_column(raw, q, c, score_type).detach().float().cpu().numpy()
            row = raw.pred[:, q, :].detach()
            classes = row[:, 4:].argmax(1).cpu().numpy()
            box = self._ops.xywh2xyxy(row[:, :4])
            box[..., [0, 2]] *= ow
            box[..., [1, 3]] *= oh
            box = box.cpu().tolist()
        return [CandidateReadout(float(values[i]), tuple(box[i]), int(classes[i])) for i in range(len(values))]

    def _probe_target(self, raw: RawOutput) -> Tuple[int, int]:
        best, cls = raw.pred[0, :, 4:].max(1)
        q = int(best.argmax())
        return q, int(cls[q])

    def _batch_diff(self, one: torch.Tensor, two: torch.Tensor) -> float:
        # Queries are re-selected per image, so compare the scores as a set.
        a = one[0, :, 4:].max(1).values.sort().values
        b = two[0, :, 4:].max(1).values.sort().values
        return float((a - b).abs().max())
