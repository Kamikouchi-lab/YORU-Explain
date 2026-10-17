# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""The one module that imports YORU's internals.

YORU's documented external API (``docs/external_api.md`` in YORU) stops at
``DetectorBase.detect()``: boxes after NMS.  Explaining a detection needs the
model underneath -- its pre-NMS candidates, head logits, feature maps and
gradients -- so the adapters also use the names below, which YORU does
**not** promise to keep:

========================================================  ==============================
``yoru.libs.plugins.yolov5_detector.load_yolov5_model``  the AutoShape model YORU builds
``yoru.libs.plugins.yolov5_detector.rgb_input_requested``  YORU's BGR/RGB rule
``yoru.libs.plugins.yolov5_detector.V1_NMS_CONF``        the confidence NMS runs at
``yoru.libs.yolov5_support.yolov5_importable``           importing the bundled code
``yoru.libs.device.torch_device``                        YORU's device choice (torch backends)
``yoru.libs.device.resolve_device``                      the device string YORU hands ultralytics
``yoru.libs.plugins._auto_detect_backend``               which backend YORU picks for a file
========================================================  ==============================

and, through ``yolov5_importable``, the bundled YOLOv5's ``letterbox``,
``make_divisible``, ``non_max_suppression``, ``scale_boxes`` and
``xywh2xyxy``.  Keeping them here, behind two functions, means a YORU change
breaks one import with a clear message instead of an analysis.  Because they
can change, every run also checks the adapter against YORU's documented
``get_detector(...).detect()`` (see each adapter's ``identity_check``).
``tests/unit/test_yoru_boundary.py`` pins this list.  Kamikouchi-lab/YORU#43
proposes documenting the YOLOv5 ones in YORU (``docs/YORU_PR_PROPOSAL.md``).

The other model families need no YORU internals: YORU runs them through
``ultralytics``, ``torchvision`` and ``onnxruntime`` with settings that its
plugins state, and the adapters call those packages themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List

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


@dataclass(frozen=True)
class YoruCore:
    """YORU's detector registry and device rule, which every adapter needs."""

    yoru_version: str
    get_detector: Callable[..., Any]
    list_detector_backends: Callable[[], List[str]]
    default_conf_thresh: float
    default_iou_thresh: float
    #: ``preference -> torch.device``, as YORU's torch backends resolve it.
    torch_device: Callable[..., Any]
    #: ``preference -> str``, the device YORU passes to ultralytics.
    resolve_device: Callable[..., str]


_core = None


def core() -> YoruCore:
    """YORU's registry and device rule; raise :class:`ExplainError` if YORU lacks them."""
    global _core
    if _core is not None:
        return _core
    try:
        import yoru
        from yoru.libs.device import resolve_device, torch_device
        from yoru.libs.plugins import (
            DEFAULT_CONF_THRESH,
            DEFAULT_IOU_THRESH,
            get_detector,
            list_detector_backends,
        )
    except ImportError as e:
        raise ExplainError(ErrorCode.MODEL_UNSUPPORTED,
                           f"YORU's detector plugins could not be imported ({e}); YORU-Explain needs "
                           "YORU >= 2.0.0b3") from e
    _core = YoruCore(str(yoru.__version__), get_detector, list_detector_backends,
                     float(DEFAULT_CONF_THRESH), float(DEFAULT_IOU_THRESH), torch_device,
                     resolve_device)
    return _core


def auto_backend(model_path) -> str:
    """The backend YORU's ``get_detector("auto", model_path)`` would pick.

    YORU decides from the file -- a YOLOv5 checkpoint by its pickle, whatever
    it is called; ``.onnx``; torchvision checkpoints by their ``model_type``;
    RT-DETR, YOLOv8 and YOLO11 by their names; ultralytics otherwise -- and
    the explanation must use the same model family YORU would.
    """
    try:
        from yoru.libs.plugins import _auto_detect_backend
    except ImportError as e:
        raise ExplainError(ErrorCode.MODEL_UNSUPPORTED,
                           f"YORU's backend detection could not be imported ({e}); choose the model "
                           "type explicitly") from e
    return str(_auto_detect_backend(str(model_path)))
