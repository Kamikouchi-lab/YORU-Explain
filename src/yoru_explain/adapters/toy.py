# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""A synthetic detector whose answers are known in advance, for validation.

It finds pure red and pure green squares on a dark background.  Every
candidate sits on a 16-pixel grid of the (letterboxed) input and looks at a
fixed 32x32 window centred on its cell:

* ``pool``: the window's mean of each channel (``AvgPool2d(32, 16, 8)``);
* ``mix``: per cell, red evidence ``e_r = R - G - B`` and green evidence
  ``e_g = G - R - B`` (a fixed 1x1 convolution);
* head: ``obj = g*(max(e_r, e_g) - 0.75)``, ``cls_red = g*(R - G)``,
  ``cls_green = -cls_red`` (``g`` = gain, 20), box = the 32x32 window.

So a candidate's score depends on the pixels of its window and on nothing
else, exactly: occluding anything outside the window cannot change it, a
black patch over part of the square must lower it, a green patch must turn
it green.  The tests use this to check the occlusion code -- the patch grid,
the deltas, the map, the statuses -- against answers derived by hand, and the
Grad-CAM code against a known feature map.  It is a test instrument, not a
model of anything.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch import nn

from yoru_explain.adapters import _yolo_common as yc
from yoru_explain.adapters.base import (
    AdapterCapabilities,
    LayerInfo,
    ModelAdapter,
    PreparedInput,
    RawOutput,
)
from yoru_explain.core.errors import ErrorCode, ExplainError
from yoru_explain.core.geometry import LetterboxTransform, autoshape_input_shape, letterbox_image
from yoru_explain.core.types import CandidateRef, Detection

STRIDE = 16
WINDOW = 32
GAIN = 20.0
OBJ_LEVEL = 0.75


class _Head(nn.Module):
    """Evidence -> YOLO-layout logits and probabilities (no parameters)."""

    def __init__(self, gain: float):
        super().__init__()
        self.gain = gain

    def forward(self, e):
        b, _, h, w = e.shape
        e_r, e_g = e[:, 0], e[:, 1]
        obj = self.gain * (torch.maximum(e_r, e_g) - OBJ_LEVEL)
        red = self.gain * (e_r - e_g) / 2
        logits = torch.stack((obj, red, -red), dim=1)            # (B, 3, h, w)
        logits = logits.permute(0, 2, 3, 1).reshape(b, h * w, 3)
        gy, gx = torch.meshgrid(torch.arange(h, device=e.device, dtype=e.dtype),
                                torch.arange(w, device=e.device, dtype=e.dtype), indexing="ij")
        box = torch.stack(((gx + 0.5) * STRIDE, (gy + 0.5) * STRIDE,
                           torch.full_like(gx, WINDOW), torch.full_like(gy, WINDOW)), dim=-1)
        box = box.reshape(1, h * w, 4).expand(b, -1, -1)
        pred = torch.cat((box, logits.sigmoid()), dim=2)
        return pred, torch.cat((box, logits), dim=2)


class _ToyNet(nn.Module):
    def __init__(self, gain: float = GAIN):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=WINDOW, stride=STRIDE, padding=(WINDOW - STRIDE) // 2,
                                 count_include_pad=True)
        self.mix = nn.Conv2d(3, 2, kernel_size=1, bias=False)
        # Input channels are B, G, R: frames arrive in OpenCV's order.
        weight = torch.tensor([[-1.0, -1.0, 1.0],    # e_r = R - G - B
                               [-1.0, 1.0, -1.0]])   # e_g = G - R - B
        with torch.no_grad():
            self.mix.weight.copy_(weight.view(2, 3, 1, 1))
        for p in self.parameters():
            p.requires_grad_(False)
        self.head = _Head(gain)

    def forward(self, x):
        return self.head(self.mix(self.pool(x)))


class ToyGridAdapter(ModelAdapter):
    name = "toy-grid"
    version = "0.1.0"
    description = "Synthetic red/green square detector with known receptive fields (tests)"
    CLASS_NAMES = {0: "red", 1: "green"}

    def __init__(self):
        super().__init__()
        self._net = None
        self._caps = None

    @property
    def torch_module(self):
        return self._net

    def load_model(self, weights=None, device: str = "cpu", *, size=None, conf_thresh: float = 0.25,
                   iou_thresh: float = 0.45, gain: float = GAIN) -> None:
        if weights not in (None, "", "synthetic"):
            raise ExplainError(ErrorCode.MODEL_UNSUPPORTED, "the synthetic detector takes no weights")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._net = _ToyNet(gain).to(self._device).eval()
        self._size = size
        self._conf = float(conf_thresh)
        self._iou = float(iou_thresh)
        self._gain = float(gain)
        self._caps = self._probe()

    # -- facts -----------------------------------------------------------

    @property
    def class_names(self) -> Dict[int, str]:
        return dict(self.CLASS_NAMES)

    @property
    def capabilities(self) -> AdapterCapabilities:
        return self._caps

    def model_info(self) -> Dict[str, Any]:
        return {
            "adapter": self.name,
            "adapter_version": self.version,
            "weights": "synthetic",
            "sha256": None,
            "class_names": {str(k): v for k, v in self.CLASS_NAMES.items()},
            "device": str(self._device),
            "input_size": self._size,
            "conf_thresh": self._conf,
            "iou_thresh": self._iou,
            "gain": self._gain,
            "stride": STRIDE,
            "window": WINDOW,
            "channel_order": "BGR",
            "synthetic": True,
        }

    # -- inference -------------------------------------------------------

    def preprocess(self, images: Sequence[np.ndarray]) -> PreparedInput:
        if not images:
            raise ExplainError(ErrorCode.INTERNAL, "no images to preprocess")
        shape = images[0].shape
        if any(im.shape != shape for im in images):
            raise ExplainError(ErrorCode.INTERNAL, "images of one batch must share a shape")
        if len(shape) != 3 or shape[2] != 3:
            raise ExplainError(ErrorCode.MODEL_UNSUPPORTED, "the synthetic detector takes BGR images")
        size = self._size or max(shape[:2])
        transform = LetterboxTransform.yolov5(shape[:2], autoshape_input_shape(shape[:2], size, 32))
        batch = np.stack([letterbox_image(im, transform) for im in images])
        x = torch.from_numpy(np.ascontiguousarray(batch.transpose(0, 3, 1, 2))).to(self._device)
        return PreparedInput(x.float() / 255, transform, tuple(shape))

    def predict_raw(self, prepared: PreparedInput, *, logits: bool = False) -> RawOutput:
        with self._lock, torch.inference_mode():
            pred, lg = self._net(prepared.tensor)
        return RawOutput(pred=pred, logits=lg)

    def postprocess(self, raw: RawOutput, prepared: PreparedInput) -> List[List[Detection]]:
        rows = yc.nms_with_indices(raw.pred, self._conf, self._iou)
        out = []
        for r in rows:
            boxes = prepared.transform.boxes_to_frame(r[:, :4]) if len(r) else np.zeros((0, 4))
            dets = []
            for i in range(len(r)):
                k = int(r[i, 6])
                cid = int(r[i, 5])
                dets.append(Detection(tuple(float(v) for v in boxes[i]), float(np.float32(r[i, 4])),
                                      cid, self.CLASS_NAMES[cid],
                                      CandidateRef(k, self._locate(k, prepared))))
            out.append(dets)
        return out

    def candidate_readout(self, raw, prepared, candidate_index, target_class, score_type):
        return yc.readout(raw, prepared.transform, candidate_index, target_class, score_type)

    def _levels(self, prepared: PreparedInput):
        h1, w1 = prepared.transform.input_shape
        return [(STRIDE, 1, h1 // STRIDE, w1 // STRIDE)]

    def _locate(self, k: int, prepared: PreparedInput):
        return yc.locate_candidate(k, self._levels(prepared))

    # -- gradients -------------------------------------------------------

    def gradient_layers(self) -> List[LayerInfo]:
        return [LayerInfo("pool", 0, "AvgPool2d", STRIDE, 3), LayerInfo("mix", 1, "Conv2d", STRIDE, 2)]

    def default_gradient_layer(self, target) -> str:
        return "pool"

    def forward_with_gradients(self, prepared: PreparedInput, layers, *, precision: str = "inference"):
        modules = dict(self._net.named_children())
        unknown = [n for n in layers if n not in modules]
        if unknown:
            raise ExplainError(ErrorCode.INVALID_CONFIG, f"unknown layers {unknown}")
        acts: Dict[str, torch.Tensor] = {}
        hooks = [modules[n].register_forward_hook(lambda m, i, o, n=n: acts.__setitem__(n, o))
                 for n in layers]
        try:
            with self._lock, torch.enable_grad():
                x = prepared.tensor.detach().clone().requires_grad_(True)
                pred, lg = self._net(x)
        finally:
            for h in hooks:
                h.remove()
        return RawOutput(pred=pred, logits=lg, activations=acts)

    def candidate_score_tensor(self, raw, batch_index, candidate_index, target_class, score_type):
        col = yc.score_column(raw.pred, raw.logits, candidate_index, target_class, score_type)
        return col[batch_index]

    # -- probe -----------------------------------------------------------

    def _probe(self) -> AdapterCapabilities:
        frame = np.zeros((96, 128, 3), np.uint8)
        frame[24:56, 40:72, 2] = 255
        p1 = self.preprocess([frame])
        a = self.predict_raw(p1).pred
        b = self.predict_raw(p1).pred
        p2 = self.preprocess([frame, frame])
        c = self.predict_raw(p2).pred
        diff = float((c[0] - a[0]).abs().max())
        g = self.forward_with_gradients(p1, ["pool"])
        k = int(g.pred[0, :, 4].argmax())
        s = self.candidate_score_tensor(g, 0, k, 0, "obj_x_cls")
        grad = torch.autograd.grad(s, g.activations["pool"])[0]
        ok = bool(torch.isfinite(grad).all() and (grad != 0).any())
        return AdapterCapabilities(
            fixed_candidate=True,
            redetection=True,
            score_types=yc.YOLOV5_SCORE_TYPES,
            gradients=ok,
            gradients_reason=("gradient probe passed" if ok else "gradient probe failed"),
            gradient_layers=("pool", "mix"),
            deterministic=bool(torch.equal(a, b)),
            batch_invariant=diff == 0.0,
            batch_max_abs_diff=diff,
            notes=("synthetic detector for validation; its scores mean nothing biological",),
            family="synthetic (YOLOv5 layout)",
            candidate_kind="grid cell",
            reported_score="obj_x_cls",
            score_descriptions=yc.YOLOV5_SCORES,
        )
