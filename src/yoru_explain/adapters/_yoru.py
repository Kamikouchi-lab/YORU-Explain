# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""The one module that imports YORU's YOLOv5 internals.

YORU's documented external API (``docs/external_api.md`` in YORU) stops at
``DetectorBase.detect()``: boxes after NMS.  Explaining a detection needs the
model underneath -- its pre-NMS candidates, head logits, feature maps and
gradients -- so the YOLOv5 adapter also uses the names below, which YORU does
**not** promise to keep:

========================================================  ==============================
``yoru.libs.plugins.yolov5_detector.load_yolov5_model``  the AutoShape model YORU builds
``yoru.libs.plugins.yolov5_detector.rgb_input_requested``  YORU's BGR/RGB rule
``yoru.libs.plugins.yolov5_detector.V1_NMS_CONF``        the confidence NMS runs at
``yoru.libs.yolov5_support.yolov5_importable``           importing the bundled code
``yoru.libs.device.torch_device``                        YORU's device choice
========================================================  ==============================

and, through ``yolov5_importable``, the bundled YOLOv5's ``letterbox``,
``make_divisible``, ``non_max_suppression``, ``scale_boxes`` and
``xywh2xyxy``.  Keeping them here, behind one function, means a YORU change
breaks one import with a clear message instead of an analysis.  Because they
can change, every run also checks the adapter against YORU's documented
``get_detector(...).detect()`` (see ``YoruYolov5Adapter.identity_check``).
``tests/unit/test_yoru_boundary.py`` pins this list; ``docs/YORU_PR_PROPOSAL.md``
proposes documenting it in YORU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from yoru_explain.core.errors import ErrorCode, ExplainError


@dataclass(frozen=True)
class YoruYolov5:
    """YORU's YOLOv5 entry points and the bundled YOLOv5 functions."""

    yoru_version: str
    load_yolov5_model: Callable[..., Any]
    rgb_input_requested: Callable[..., bool]
    v1_nms_conf: float
    torch_device: Callable[..., Any]
    get_detector: Callable[..., Any]
    default_conf_thresh: float
    default_iou_thresh: float
    yolov5_dir: str
    letterbox: Callable[..., Any]
    make_divisible: Callable[..., Any]
    non_max_suppression: Callable[..., Any]
    scale_boxes: Callable[..., Any]
    xywh2xyxy: Callable[..., Any]


_cached = None


def load() -> YoruYolov5:
    """Import everything once; raise :class:`ExplainError` if YORU lacks a piece."""
    global _cached
    if _cached is not None:
        return _cached
    try:
        import yoru
        from yoru.libs.device import torch_device
        from yoru.libs.plugins import DEFAULT_CONF_THRESH, DEFAULT_IOU_THRESH, get_detector
        from yoru.libs.plugins.yolov5_detector import (
            V1_NMS_CONF,
            load_yolov5_model,
            rgb_input_requested,
        )
        from yoru.libs.yolov5_support import YOLOV5_DIR, yolov5_importable

        with yolov5_importable():
            from utils.augmentations import letterbox
            from utils.general import make_divisible, non_max_suppression, scale_boxes, xywh2xyxy
    except ImportError as e:
        raise ExplainError(
            ErrorCode.MODEL_UNSUPPORTED,
            f"YORU's YOLOv5 code could not be imported ({e}). The YOLOv5 adapter needs "
            "YORU >= 2.0.0b3 with yoru/libs/yolov5; see docs/INITIAL_AUDIT.md.") from e
    _cached = YoruYolov5(
        yoru_version=str(yoru.__version__),
        load_yolov5_model=load_yolov5_model,
        rgb_input_requested=rgb_input_requested,
        v1_nms_conf=float(V1_NMS_CONF),
        torch_device=torch_device,
        get_detector=get_detector,
        default_conf_thresh=float(DEFAULT_CONF_THRESH),
        default_iou_thresh=float(DEFAULT_IOU_THRESH),
        yolov5_dir=str(YOLOV5_DIR),
        letterbox=letterbox,
        make_divisible=make_divisible,
        non_max_suppression=non_max_suppression,
        scale_boxes=scale_boxes,
        xywh2xyxy=xywh2xyxy,
    )
    return _cached
