# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Reading YOLO-layout candidate tensors: ``(B, N, 5 + nc)``.

Columns ``cx, cy, w, h`` (input pixels), then ``sigmoid(objectness)`` and
``sigmoid(class)`` for each class -- YOLOv5's ``Detect`` output.  The
``obj_x_cls`` score is computed as ``pred[..., 5 + c] * pred[..., 4]`` in the
tensor's own dtype, the very product YOLOv5's NMS takes for a detection's
confidence, so a candidate's score equals its reported confidence bit for bit.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from yoru_explain.adapters.base import CandidateReadout, RawOutput
from yoru_explain.core.errors import ErrorCode, ExplainError
from yoru_explain.core.geometry import LetterboxTransform

#: The scores of the YOLOv5 layout, the reported confidence first.
YOLOV5_SCORE_TYPES = ("confidence", "obj_x_cls", "cls_prob", "obj_prob", "cls_logit", "obj_logit")
YOLOV5_SCORES = (
    ("confidence", "sigmoid(objectness) x sigmoid(class logit) -- YOLOv5's confidence as YORU "
                   "reports it (the same number as obj_x_cls)"),
    ("obj_x_cls", "sigmoid(objectness) x sigmoid(class logit) -- the confidence YOLOv5 reports"),
    ("cls_prob", "sigmoid(class logit) -- class probability given an object"),
    ("obj_prob", "sigmoid(objectness) -- object probability, class-agnostic"),
    ("cls_logit", "class logit, before the sigmoid"),
    ("obj_logit", "objectness logit, before the sigmoid"),
)


def xywh_to_xyxy(xywh: torch.Tensor) -> torch.Tensor:
    """The same arithmetic as YOLOv5's ``xywh2xyxy``."""
    y = xywh.clone()
    y[..., 0] = xywh[..., 0] - xywh[..., 2] / 2
    y[..., 1] = xywh[..., 1] - xywh[..., 3] / 2
    y[..., 2] = xywh[..., 0] + xywh[..., 2] / 2
    y[..., 3] = xywh[..., 1] + xywh[..., 3] / 2
    return y


def score_column(pred: torch.Tensor, logits: Optional[torch.Tensor], k: int, c: int,
                 score_type: str) -> torch.Tensor:
    """Candidate *k*'s score of class *c* on every image: shape ``(B,)``."""
    if score_type in ("obj_x_cls", "confidence"):
        return pred[:, k, 5 + c] * pred[:, k, 4]
    if score_type == "cls_prob":
        return pred[:, k, 5 + c]
    if score_type == "obj_prob":
        return pred[:, k, 4]
    if logits is None:
        raise ExplainError(ErrorCode.SCORE_UNAVAILABLE,
                           f"{score_type} needs the head's logits, which were not captured")
    if score_type == "cls_logit":
        return logits[:, k, 5 + c]
    if score_type == "obj_logit":
        return logits[:, k, 4]
    raise ExplainError(ErrorCode.MODEL_UNSUPPORTED, f"unknown score type {score_type!r}")


def readout(raw: RawOutput, transform: LetterboxTransform, k: int, c: int,
            score_type: str) -> List[CandidateReadout]:
    pred = raw.pred
    if not 0 <= k < pred.shape[1]:
        raise ExplainError(ErrorCode.SCORE_UNAVAILABLE,
                           f"candidate {k} out of range (model has {pred.shape[1]})")
    nc = pred.shape[2] - 5
    if not 0 <= c < nc:
        raise ExplainError(ErrorCode.SCORE_UNAVAILABLE, f"class {c} out of range ({nc} classes)")
    values = score_column(pred, raw.logits, k, c, score_type).detach().float().cpu().numpy()
    boxes_in = xywh_to_xyxy(pred[:, k, :4]).detach().float().cpu().numpy()
    boxes = transform.boxes_to_frame(boxes_in)
    # The class NMS would label the candidate with: argmax of obj * cls.
    conf = (pred[:, k, 5:] * pred[:, k, 4:5]).detach()
    classes = conf.argmax(dim=1).cpu().numpy()
    return [CandidateReadout(float(values[b]), tuple(float(v) for v in boxes[b]), int(classes[b]))
            for b in range(pred.shape[0])]


def locate_candidate(k: int, levels) -> tuple:
    """``(("level", i), ("stride", s), ("anchor", a), ("grid_y", y), ("grid_x", x))``.

    *levels* lists ``(stride, n_anchors, ny, nx)`` per output level, in the
    order the candidates are concatenated (level, then anchor, y, x).
    """
    offset = 0
    for i, (stride, na, ny, nx) in enumerate(levels):
        n = na * ny * nx
        if k < offset + n:
            a, rem = divmod(k - offset, ny * nx)
            gy, gx = divmod(rem, nx)
            return (("level", i), ("stride", int(stride)), ("anchor", a),
                    ("grid_y", gy), ("grid_x", gx))
        offset += n
    raise ExplainError(ErrorCode.INTERNAL, f"candidate {k} beyond the {offset} the levels hold")


def nms_with_indices(pred: torch.Tensor, conf_thres: float, iou_thres: float,
                     max_det: int = 1000) -> List[np.ndarray]:
    """YOLOv5's NMS rule (best class only, class-offset boxes), keeping row indices.

    Used by the synthetic adapter.  The YOLOv5 adapter runs the bundled
    ``non_max_suppression`` itself and recovers indices by exact matching.
    Returns, per image, an ``(n, 7)`` array ``x1, y1, x2, y2, conf, cls, k``
    with boxes in input pixels.
    """
    import torchvision

    out = []
    max_wh = 7680
    for x in pred:
        idx = torch.arange(x.shape[0], device=x.device)
        keep = x[:, 4] > conf_thres
        x, idx = x[keep].clone(), idx[keep]
        if not x.shape[0]:
            out.append(np.zeros((0, 7)))
            continue
        x[:, 5:] *= x[:, 4:5]
        box = xywh_to_xyxy(x[:, :4])
        conf, j = x[:, 5:].max(1, keepdim=True)
        sel = conf.view(-1) > conf_thres
        det = torch.cat((box, conf, j.float()), 1)[sel]
        idx = idx[sel]
        if not det.shape[0]:
            out.append(np.zeros((0, 7)))
            continue
        order = det[:, 4].argsort(descending=True)
        det, idx = det[order], idx[order]
        kept = torchvision.ops.nms(det[:, :4] + det[:, 5:6] * max_wh, det[:, 4], iou_thres)[:max_det]
        rows = torch.cat((det[kept], idx[kept, None].to(det.dtype)), 1)
        out.append(rows.detach().double().cpu().numpy())
    return out
