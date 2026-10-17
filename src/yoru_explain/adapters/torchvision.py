# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""torchvision Faster R-CNN / Mask R-CNN / SSD300 checkpoints, run as YORU runs them.

YORU's torchvision plugin (``yoru/libs/plugins/torchvision_detector.py``)
reads a checkpoint written by its ``train_torchvision.py`` -- ``model_type``,
``num_classes``, ``names``, ``model_state_dict`` -- rebuilds the network
(``fasterrcnn_resnet50_fpn`` / ``maskrcnn_resnet50_fpn`` / ``ssd300_vgg16``
with ``weights=None`` and its own predictor), loads the weights and, per
frame, runs ``model([to_tensor(frame[..., ::-1])])`` under ``no_grad``, then
keeps the boxes scoring **above** its confidence threshold.  Its IoU threshold
is not used: NMS runs inside torchvision at the model's own setting (Faster
/ Mask R-CNN 0.5, SSD 0.45).  Class ids are the labels minus one.

The adapter rebuilds the same network the same way and runs the model's own
pieces in the model's own order -- ``transform``, ``backbone``, ``rpn``,
``roi_heads`` / ``head`` + ``anchor_generator``, ``postprocess_detections``,
``transform.postprocess`` -- so that it can keep the per-candidate scores,
logits, feature maps and gradients.  Each detection's candidate is found by
exact equality of its box and score with the candidate table.

Candidates.  SSD: a *default box* (feature level, grid cell, aspect), fixed
for its 300 x 300 input, so fixed-candidate mode follows it.  Faster / Mask
R-CNN: a region *proposal* of this input's RPN -- proposals are chosen anew
for every input, so these models are followed by re-detection only; Grad-CAM
of a proposal on its own frame is well defined.  Both are multi-label: one
candidate can be reported under several classes.

The score YORU reports is the softmax over the classes and the background
(``cls_softmax``); ``cls_logit`` is the class's logit before it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

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
    yoru_detection_row,
)
from yoru_explain.core.errors import ErrorCode, ExplainError
from yoru_explain.core.geometry import LetterboxTransform
from yoru_explain.core.hashing import sha256_file
from yoru_explain.core.logs import get_logger
from yoru_explain.core.types import CandidateRef, Detection

log = get_logger("torchvision")

MODEL_TYPES = ("fasterrcnn", "maskrcnn", "ssd")
_TWO_STAGE = ("fasterrcnn", "maskrcnn")
#: Rows of the (detections x candidates) comparison done at once.
_MATCH_CHUNK = 32


def build_model(model_type: str, num_classes: int):
    """The network YORU's torchvision plugin builds before loading a checkpoint.

    The same calls as ``TorchvisionDetector._build_model``; the identity
    check confirms on every run that the result detects what YORU detects.
    Building fetches torchvision's ImageNet backbone weights (the checkpoint
    replaces them) because ``weights_backbone`` keeps its default -- which
    also decides the network's frozen batch norms, so it is kept as YORU has it.
    """
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn,
        maskrcnn_resnet50_fpn,
        ssd300_vgg16,
    )
    from torchvision.models.detection._utils import retrieve_out_channels
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    from torchvision.models.detection.ssd import SSDHead

    n = int(num_classes) + 1  # + background
    if model_type == "fasterrcnn":
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n)
    elif model_type == "maskrcnn":
        model = maskrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, n)
    elif model_type == "ssd":
        model = ssd300_vgg16(weights=None)
        in_channels = retrieve_out_channels(model.backbone, (300, 300))
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head = SSDHead(in_channels, num_anchors, n)
    else:
        raise ExplainError(ErrorCode.MODEL_UNSUPPORTED,
                           f"unknown torchvision model_type {model_type!r}; one of {MODEL_TYPES}")
    return model


class YoruTorchvisionAdapter(ModelAdapter):
    name = "yoru-torchvision"
    version = "0.2.0"
    description = "torchvision Faster R-CNN / Mask R-CNN / SSD300 checkpoints written by YORU, as YORU runs them"
    yoru_backend = "torchvision"

    def __init__(self):
        super().__init__()
        self._model = None
        self._caps: Optional[AdapterCapabilities] = None
        self._layer_info: List[LayerInfo] = []

    # -- loading -------------------------------------------------------------

    def load_model(self, weights, device: str = "auto", *, conf_thresh: Optional[float] = None,
                   iou_thresh: Optional[float] = None, probe: bool = True) -> None:
        core = _yoru.core()
        self._core = core
        path = Path(weights)
        if not path.is_file():
            raise ExplainError(ErrorCode.MISSING_DATA, f"model file not found: {path}")
        self._weights = str(path)
        self._sha256 = sha256_file(path)
        self._device_pref = str(device or "auto")
        self._device = core.torch_device(self._device_pref)
        self._conf = core.default_conf_thresh if conf_thresh is None else float(conf_thresh)
        # Recorded as given; YORU's torchvision plugin does not use it.
        self._iou = core.default_iou_thresh if iou_thresh is None else float(iou_thresh)
        try:
            checkpoint = torch.load(self._weights, map_location="cpu", weights_only=False)
            model_type = str(checkpoint["model_type"])
            num_classes = int(checkpoint["num_classes"])
            names = checkpoint["names"]
        except Exception as e:
            raise ExplainError(ErrorCode.MODEL_UNSUPPORTED,
                               f"{path.name} is not a checkpoint written by YORU's train_torchvision.py "
                               f"(model_type, num_classes, names, model_state_dict): "
                               f"{type(e).__name__}: {e}") from e
        try:
            model = build_model(model_type, num_classes)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self._device)
            model.eval()
        except ExplainError:
            raise
        except Exception as e:
            raise ExplainError(ErrorCode.MODEL_LOAD_FAILED,
                               f"{path.name} could not be loaded: {type(e).__name__}: {e}") from e
        del checkpoint
        self._model = model
        self._model_type = model_type
        self._two_stage = model_type in _TWO_STAGE
        self._num_classes = num_classes
        self._names = {int(k): str(v) for k, v in dict(names).items()}
        log.info("loaded %s (%s): %s with %d classes on %s; conf %.3g (strictly above)",
                 path.name, self._sha256[:12], model_type, num_classes, self._device, self._conf)
        self._caps = self._probe() if probe else self._unprobed_caps()

    @property
    def torch_module(self):
        return self._model

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
        import torchvision

        t = self._model.transform
        info: Dict[str, Any] = {
            "adapter": self.name,
            "adapter_version": self.version,
            "backend": self.yoru_backend,
            "weights": self._weights,
            "sha256": self._sha256,
            "model_type": self._model_type,
            "class_names": {str(k): v for k, v in self._names.items()},
            "nc": self._num_classes,
            "conf_thresh": self._conf,
            "conf_rule": "score > conf_thresh (strictly), after torchvision's own NMS",
            "iou_thresh": self._iou,
            "iou_thresh_used": False,
            "transform": {"min_size": list(t.min_size), "max_size": int(t.max_size),
                          "fixed_size": list(t.fixed_size) if t.fixed_size else None,
                          "size_divisible": int(t.size_divisible),
                          "image_mean": [float(v) for v in t.image_mean],
                          "image_std": [float(v) for v in t.image_std]},
            "channel_order": "RGB (YORU converts OpenCV's BGR frames), /255",
            "device": str(self._device),
            "device_preference": self._device_pref,
            "precision": "fp32",
            "torchvision_version": str(torchvision.__version__),
            "yoru_version": self._core.yoru_version,
        }
        if self._two_stage:
            rh, rpn = self._model.roi_heads, self._model.rpn
            info.update(score_thresh=float(rh.score_thresh), nms_conf=float(rh.score_thresh),
                        nms_iou=float(rh.nms_thresh), max_det=int(rh.detections_per_img),
                        rpn_post_nms_top_n=int(rpn._post_nms_top_n["testing"]),
                        rpn_nms_thresh=float(rpn.nms_thresh))
        else:
            m = self._model
            info.update(score_thresh=float(m.score_thresh), nms_conf=float(m.score_thresh),
                        nms_iou=float(m.nms_thresh), max_det=int(m.detections_per_img),
                        topk_candidates=int(m.topk_candidates))
        return info

    # -- inference -----------------------------------------------------------

    def preprocess(self, images: Sequence[np.ndarray]) -> PreparedInput:
        """YORU's ``to_tensor(frame[..., ::-1])``, then the model's own ``transform``."""
        from torchvision.transforms.functional import to_tensor

        if not images:
            raise ExplainError(ErrorCode.INTERNAL, "no images to preprocess")
        shape = images[0].shape
        if any(im.shape != shape for im in images):
            raise ExplainError(ErrorCode.INTERNAL, "images of one batch must share a shape")
        if len(shape) != 3 or shape[2] != 3:
            raise ExplainError(ErrorCode.MODEL_UNSUPPORTED, f"expected BGR frames, got shape {shape}")
        with self._lock, torch.inference_mode():
            tensors = [to_tensor(im[:, :, ::-1].copy()).to(self._device) for im in images]
            image_list, _ = self._model.transform(tensors)
        sizes = [tuple(int(v) for v in s) for s in image_list.image_sizes]
        transform = LetterboxTransform.torchvision(shape[:2], sizes[0], tuple(image_list.tensors.shape[2:]))
        return PreparedInput(image_list.tensors, transform, tuple(shape), extra={"image_sizes": sizes})

    def _image_list(self, tensor, prepared: PreparedInput):
        from torchvision.models.detection.image_list import ImageList

        return ImageList(tensor, list(prepared.extra["image_sizes"]))

    def _feature_hooks(self, acts: Dict[str, torch.Tensor], layers: Sequence[str]) -> list:
        """Forward hooks storing the requested feature maps (see :meth:`gradient_layers`)."""
        modules = self._feature_modules()
        hooks = []
        for name in layers:
            module_name, key = self._split_layer(name)
            module = modules.get(module_name)
            if module is None:
                raise ExplainError(ErrorCode.INVALID_CONFIG, f"unknown layer {name!r}")

            def hook(_m, _i, out, name=name, key=key):
                acts[name] = out[key] if key is not None else out

            hooks.append(module.register_forward_hook(hook))
        return hooks

    def _feature_modules(self) -> Dict[str, torch.nn.Module]:
        b = self._model.backbone
        if self._two_stage:
            mods = {f"backbone.body.{n}": getattr(b.body, n) for n in ("layer1", "layer2", "layer3", "layer4")}
            mods["backbone.fpn"] = b.fpn
            return mods
        return {"backbone": b}

    @staticmethod
    def _split_layer(name: str) -> Tuple[str, Optional[str]]:
        """``backbone.fpn.2`` -> (``backbone.fpn``, ``"2"``); ``backbone.3`` -> (``backbone``, ``"3"``)."""
        if name.startswith("backbone.fpn."):
            return "backbone.fpn", name.split(".", 2)[2]
        if name.startswith("backbone.body."):
            return name, None
        if name.startswith("backbone."):
            return "backbone", name.split(".", 1)[1]
        return name, None

    def _run(self, tensor, prepared: PreparedInput) -> Dict[str, Any]:
        """The model's own pieces, in its own order, up to the per-candidate outputs."""
        import torch.nn.functional as F
        from torchvision.ops import boxes as box_ops

        m = self._model
        images = self._image_list(tensor, prepared)
        out: Dict[str, Any] = {"images": images}
        features = m.backbone(images.tensors)
        if self._two_stage:
            proposals, _ = m.rpn(images, features)
            rh = m.roi_heads
            box_features = rh.box_roi_pool(features, proposals, images.image_sizes)
            box_features = rh.box_head(box_features)
            class_logits, box_regression = rh.box_predictor(box_features)
            # As postprocess_detections computes them: decode, softmax, split, clip.
            pred_boxes = rh.box_coder.decode(box_regression, proposals)
            pred_scores = F.softmax(class_logits, -1)
            counts = [len(p) for p in proposals]
            out.update(proposals=proposals, class_logits=class_logits, box_regression=box_regression,
                       counts=counts, logits_all=class_logits,
                       boxes=[box_ops.clip_boxes_to_image(b, s) for b, s in
                              zip(pred_boxes.split(counts, 0), images.image_sizes)],
                       scores=list(pred_scores.split(counts, 0)),
                       logits=list(class_logits.split(counts, 0)))
        else:
            if isinstance(features, torch.Tensor):
                from collections import OrderedDict

                features = OrderedDict([("0", features)])
            feats = list(features.values())
            head_outputs = m.head(feats)
            anchors = m.anchor_generator(images, feats)
            pred_scores = F.softmax(head_outputs["cls_logits"], dim=-1)
            boxes = []
            for reg, anc, size in zip(head_outputs["bbox_regression"], anchors, images.image_sizes):
                boxes.append(box_ops.clip_boxes_to_image(m.box_coder.decode_single(reg, anc), size))
            out.update(head_outputs=head_outputs, anchors=anchors, boxes=boxes,
                       scores=list(pred_scores), logits=list(head_outputs["cls_logits"]),
                       logits_all=head_outputs["cls_logits"],
                       level_sizes=[tuple(f.shape[-2:]) for f in feats])
        return out

    def predict_raw(self, prepared: PreparedInput, *, logits: bool = False) -> RawOutput:
        with self._lock, torch.inference_mode():
            parts = self._run(prepared.tensor, prepared)
        # pred: per image, the (candidates, classes + 1) softmax scores.
        return RawOutput(pred=parts["scores"], logits=parts["logits"], extra=parts)

    def postprocess(self, raw: RawOutput, prepared: PreparedInput) -> List[List[Detection]]:
        m = self._model
        x = raw.extra
        images = x["images"]
        frame_sizes = [tuple(prepared.frame_shape[:2])] * len(images.image_sizes)
        with self._lock, torch.inference_mode():
            if self._two_stage:
                boxes, scores, labels = m.roi_heads.postprocess_detections(
                    x["class_logits"], x["box_regression"], x["proposals"], images.image_sizes)
                dets = [{"boxes": b, "scores": s, "labels": lb} for b, s, lb in zip(boxes, scores, labels)]
            else:
                dets = m.postprocess_detections(x["head_outputs"], x["anchors"], images.image_sizes)
            boxes_in = [d["boxes"] for d in dets]     # input pixels, before the resize back
            dets = m.transform.postprocess(dets, images.image_sizes, frame_sizes)
            out = []
            for b, d in enumerate(dets):
                index = self._match(x["boxes"][b], raw.pred[b], boxes_in[b], d["scores"], d["labels"])
                fb, fs, fl = d["boxes"].cpu(), d["scores"].cpu(), d["labels"].cpu()
                keep = fs > self._conf              # YORU: scores > conf_thresh
                rows = []
                for i in torch.nonzero(keep).flatten().tolist():
                    cid = int(fl[i]) - 1             # torchvision labels are 1-based
                    k = index[i]
                    cand = CandidateRef(k, self._locate(k, raw, b)) if k is not None else None
                    rows.append(Detection((float(fb[i, 0]), float(fb[i, 1]), float(fb[i, 2]),
                                           float(fb[i, 3])), float(fs[i]), cid,
                                          self._names.get(cid, str(cid)), cand))
                out.append(rows)
        return out

    def _match(self, cand_boxes, cand_scores, boxes, scores, labels) -> List[Optional[int]]:
        """The candidate of each output row: its box and score, bit for bit, at its label."""
        found: List[Optional[int]] = []
        for start in range(0, len(boxes), _MATCH_CHUNK):
            bs, ss, ls = boxes[start:start + _MATCH_CHUNK], scores[start:start + _MATCH_CHUNK], \
                labels[start:start + _MATCH_CHUNK]
            for b, s, lb in zip(bs, ss, ls):
                cb = cand_boxes[:, int(lb)] if cand_boxes.dim() == 3 else cand_boxes
                hit = (cb == b).all(-1) & (cand_scores[:, int(lb)] == s)
                ks = hit.nonzero().flatten().tolist()
                if len(ks) == 1:
                    found.append(int(ks[0]))
                else:
                    log.warning("detection %s maps to %d candidates; left without one",
                                b.tolist(), len(ks))
                    found.append(None)
        return found

    def _locate(self, k: int, raw: RawOutput, b: int):
        x = raw.extra
        if self._two_stage:
            proposal = x["proposals"][b][k:k + 1]
            level = int(self._model.roi_heads.box_roi_pool.map_levels([proposal])[0])
            return (("proposal", int(k)), ("fpn_level", level))
        steps = self._model.anchor_generator.steps
        per_loc = self._model.anchor_generator.num_anchors_per_location()
        offset = 0
        for level, ((h, w), a) in enumerate(zip(x["level_sizes"], per_loc)):
            n = h * w * a
            if k < offset + n:
                cell, box = divmod(k - offset, a)       # y, x, then the aspect
                gy, gx = divmod(cell, w)
                return (("level", level), ("step", int(steps[level])), ("grid_y", int(gy)),
                        ("grid_x", int(gx)), ("box", int(box)))
            offset += n
        return ()

    def candidate_readout(self, raw, prepared, candidate_index, target_class, score_type):
        k, c = int(candidate_index), int(target_class)
        label = c + 1
        if not 0 <= c < self._num_classes:
            raise ExplainError(ErrorCode.SCORE_UNAVAILABLE,
                               f"class {c} out of range ({self._num_classes} classes)")
        out = []
        with torch.inference_mode():
            for b in range(len(raw.pred)):
                scores = raw.pred[b]
                if not 0 <= k < scores.shape[0]:
                    raise ExplainError(ErrorCode.SCORE_UNAVAILABLE,
                                       f"candidate {k} out of range ({scores.shape[0]} on image {b})")
                value = float(self._value(raw, b, k, label, score_type))
                box = raw.extra["boxes"][b][k]
                box = box[label] if box.dim() == 2 else box
                fb = prepared.transform.boxes_to_frame(box.detach().float().cpu().numpy()[None])[0]
                cls = int(scores[k, 1:].argmax()) if scores.shape[1] > 1 else None
                out.append(CandidateReadout(value, tuple(float(v) for v in fb), cls))
        return out

    def _value(self, raw: RawOutput, b: int, k: int, label: int, score_type: str):
        if score_type in ("confidence", "cls_softmax"):
            return raw.pred[b][k, label]
        if score_type == "cls_logit":
            return raw.logits[b][k, label]
        raise ExplainError(ErrorCode.MODEL_UNSUPPORTED, f"{self.name} cannot read {score_type!r}")

    # -- the check against YORU itself -------------------------------------------

    def identity_check(self, image: np.ndarray) -> Dict[str, Any]:
        """Detections here vs. ``get_detector('torchvision', ...).detect(image)``."""
        reference = self._core.get_detector("torchvision", self._weights, conf_thresh=self._conf,
                                            iou_thresh=self._iou, device=self._device_pref)
        try:
            ref = [yoru_detection_row(d, False) for d in reference.detect(image)]
        finally:
            del reference
        mine = [detection_row(d, False) for d in self.detect(image)]
        return compare_detections("yoru.libs.plugins.get_detector('torchvision', ...).detect", ref, mine)

    # -- gradients -------------------------------------------------------------

    def gradient_layers(self) -> List[LayerInfo]:
        return list(self._layer_info)

    def default_gradient_layer(self, target) -> str:
        """The feature map the candidate's score is computed from.

        Faster / Mask R-CNN: the FPN level its proposal is pooled from (the
        ``LevelMapper`` of the box head's ``MultiScaleRoIAlign``); SSD: the
        backbone level of its default box.
        """
        cand = target.candidate
        if cand is None:
            raise ExplainError(ErrorCode.SCORE_UNAVAILABLE, "the target has no candidate")
        loc = cand.locator_dict
        if self._two_stage:
            names = self._model.roi_heads.box_roi_pool.featmap_names
            return f"backbone.fpn.{names[int(loc.get('fpn_level', 0))]}"
        return f"backbone.{int(loc.get('level', 0))}"

    def forward_with_gradients(self, prepared: PreparedInput, layers: Sequence[str], *,
                               precision: str = "inference") -> RawOutput:
        """The model's pieces with the graph kept from a copy of its input (fp32, as inference)."""
        if precision not in ("inference", "fp32"):
            raise ExplainError(ErrorCode.INVALID_CONFIG, f"unknown precision {precision!r}")
        known = {li.name for li in self._layer_info} or None
        unknown = [n for n in layers if known is not None and n not in known]
        if unknown:
            raise ExplainError(ErrorCode.INVALID_CONFIG, f"unknown layers {unknown}")
        acts: Dict[str, torch.Tensor] = {}
        with self._lock:
            hooks = self._feature_hooks(acts, layers)
            try:
                with torch.inference_mode(False), torch.enable_grad():
                    x = prepared.tensor.detach().clone().requires_grad_(True)
                    parts = self._run(x, prepared)
            finally:
                for h in hooks:
                    h.remove()
        return RawOutput(pred=parts["scores"], logits=parts["logits"], activations=acts, extra=parts)

    def candidate_score_tensor(self, raw, batch_index, candidate_index, target_class, score_type):
        return self._value(raw, int(batch_index), int(candidate_index), int(target_class) + 1, score_type)

    # -- probe -----------------------------------------------------------------

    def _family(self) -> Dict[str, Any]:
        label = {"fasterrcnn": "Faster R-CNN", "maskrcnn": "Mask R-CNN (boxes)", "ssd": "SSD300"}
        return {
            "family": f"torchvision {label[self._model_type]}",
            "candidate_kind": "proposal" if self._two_stage else "default box",
            "reported_score": "cls_softmax",
            "multi_label": True,
            "score_descriptions": (
                ("confidence", "softmax probability over the classes and the background -- the "
                               "confidence torchvision reports, as YORU reports it (= cls_softmax)"),
                ("cls_softmax", "softmax probability over the classes and the background"),
                ("cls_logit", "the class's logit, before the softmax (shift-invariant: only differences "
                              "between classes matter to the softmax)"),
            ),
        }

    def _unprobed_caps(self) -> AdapterCapabilities:
        return AdapterCapabilities(not self._two_stage, True, ("confidence", "cls_softmax", "cls_logit"),
                                   False, "not probed", **self._family())

    def _probe(self) -> AdapterCapabilities:
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
        p1 = self.preprocess([frame])
        a = self.predict_raw(p1)
        b = self.predict_raw(p1)
        deterministic = all(torch.equal(x, y) for x, y in zip(a.pred, b.pred)) and \
            all(torch.equal(x, y) for x, y in zip(a.extra["boxes"], b.extra["boxes"]))
        two = self.predict_raw(self.preprocess([frame, frame]))
        batch_diff = self._batch_diff(a, two)
        info = self._probe_layers(p1)
        self._layer_info = info
        notes = []
        if self._two_stage:
            notes.append("region proposals are chosen anew for every input: re-detection only; "
                         "Grad-CAM differentiates the target's proposal on its own frame")
        if batch_diff is None or batch_diff != 0.0:
            notes.append("scores depend on the batch size" + (
                f" (up to {batch_diff:.2g} between batch 1 and 2)" if batch_diff is not None else
                " (batch 1 and 2 kept different proposals)") +
                "; occlusion takes its reference at the run's batch size")
        grad_ok, reason = self._probe_gradients(p1, a)
        return AdapterCapabilities(
            fixed_candidate=not self._two_stage,
            redetection=True,
            score_types=("confidence", "cls_softmax", "cls_logit"),
            gradients=grad_ok,
            gradients_reason=reason,
            gradient_layers=tuple(li.name for li in info),
            deterministic=deterministic,
            batch_invariant=batch_diff == 0.0,
            batch_max_abs_diff=batch_diff,
            notes=tuple(notes),
            **self._family(),
        )

    def _batch_diff(self, one: RawOutput, two: RawOutput) -> Optional[float]:
        a, b = one.pred[0], two.pred[0]
        if a.shape != b.shape:
            return None
        if self._two_stage and not torch.equal(one.extra["proposals"][0], two.extra["proposals"][0]):
            # Different proposals: compare the best foreground scores as a set.
            fa = a[:, 1:].max(1).values.sort().values
            fb = b[:, 1:].max(1).values.sort().values
            return float((fa - fb).abs().max())
        return float((a - b).abs().max())

    def _probe_layers(self, prepared: PreparedInput) -> List[LayerInfo]:
        h1, w1 = prepared.transform.input_shape
        acts: Dict[str, torch.Tensor] = {}
        if self._two_stage:
            names = [f"backbone.body.layer{i}" for i in (1, 2, 3, 4)]
            names += [f"backbone.fpn.{k}" for k in self._model.roi_heads.box_roi_pool.featmap_names]
        else:
            names = [f"backbone.{i}" for i in range(len(self._model.anchor_generator.steps))]
        with self._lock:
            hooks = self._feature_hooks(acts, names)
            try:
                with torch.inference_mode():
                    self._run(prepared.tensor, prepared)
            finally:
                for h in hooks:
                    h.remove()
        info = []
        steps = list(self._model.anchor_generator.steps) if not self._two_stage else None
        for i, n in enumerate(names):
            t = acts.get(n)
            if t is None or t.dim() != 4:
                continue
            gh, gw = int(t.shape[2]), int(t.shape[3])
            if steps is not None:
                stride = int(steps[i])
            elif h1 % gh == 0 and w1 % gw == 0 and h1 // gh == w1 // gw:
                stride = h1 // gh
            else:
                continue
            if gh * stride < h1 or gw * stride < w1:
                continue
            info.append(LayerInfo(n, i, type(t).__name__ if False else ("FPN" if "fpn" in n else "feature"),
                                  stride, int(t.shape[1])))
        return info

    def _probe_gradients(self, prepared: PreparedInput, reference: RawOutput) -> Tuple[bool, str]:
        try:
            scores = reference.pred[0]
            if scores.shape[0] == 0:
                return False, "gradient probe: the model produced no candidates on the probe frame"
            fg = scores[:, 1:]
            flat = int(fg.reshape(-1).argmax())
            k, c = divmod(flat, fg.shape[1])
            cand = CandidateRef(k, self._locate(k, reference, 0))
            target = type("T", (), {"candidate": cand})()
            layer = self.default_gradient_layer(target)
            raw = self.forward_with_gradients(prepared, [layer])
            s = self.candidate_score_tensor(raw, 0, k, c, "confidence")
            ref = float(self._value(reference, 0, k, c + 1, "confidence"))
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
