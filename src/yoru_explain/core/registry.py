# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Adapter names, where their code lives, and which one stands in for each YORU backend.

Adapters are imported only when created, so listing them does not load
torch or YORU.

Every detection backend YORU has (``yoru.libs.plugins.list_detector_backends``)
has a model-specific adapter that reproduces YORU's inference and exposes
the model underneath; a backend without one -- added to YORU later, say --
still runs through the black-box adapter, which is YORU's own ``detect()``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from yoru_explain.core.errors import ErrorCode, ExplainError


@dataclass(frozen=True)
class AdapterEntry:
    name: str
    module: str
    cls: str
    label: str
    #: Shown in the GUI so nobody mistakes the test model for a real one.
    synthetic: bool = False
    #: The YORU backend this adapter stands in for (None: any, or none).
    backend: Optional[str] = None


_ADAPTERS: Dict[str, AdapterEntry] = {
    "yoru-yolov5": AdapterEntry(
        "yoru-yolov5", "yoru_explain.adapters.yolov5", "YoruYolov5Adapter",
        "YOLOv5 (YORU's bundled YOLOv5 code; YORU v1 models)", backend="yolov5"),
    "yoru-ultralytics": AdapterEntry(
        "yoru-ultralytics", "yoru_explain.adapters.ultralytics", "YoruUltralyticsAdapter",
        "YOLOv8 / YOLO11 / YOLOv5u / YOLOv10 (ultralytics; detect, OBB, segment, pose)",
        backend="ultralytics"),
    "yoru-rtdetr": AdapterEntry(
        "yoru-rtdetr", "yoru_explain.adapters.ultralytics", "YoruRtdetrAdapter",
        "RT-DETR (ultralytics)", backend="rtdetr"),
    "yoru-torchvision": AdapterEntry(
        "yoru-torchvision", "yoru_explain.adapters.torchvision", "YoruTorchvisionAdapter",
        "Faster R-CNN / Mask R-CNN / SSD300 (torchvision checkpoints written by YORU)",
        backend="torchvision"),
    "yoru-onnx": AdapterEntry(
        "yoru-onnx", "yoru_explain.adapters.onnx", "YoruOnnxAdapter",
        "ONNX exports of YOLOv5 / YOLOv8 / YOLO11 (ONNX Runtime)", backend="onnx"),
    "yoru-detector": AdapterEntry(
        "yoru-detector", "yoru_explain.adapters.blackbox", "YoruDetectorAdapter",
        "Any YORU detector as a black box (its detect() only: re-detection, no Grad-CAM)"),
    "toy-grid": AdapterEntry(
        "toy-grid", "yoru_explain.adapters.toy", "ToyGridAdapter",
        "Synthetic test detector (colour squares; no weights)", synthetic=True),
}

#: The model-specific adapter for each YORU backend.
BACKEND_ADAPTERS: Dict[str, str] = {e.backend: e.name for e in _ADAPTERS.values() if e.backend}
#: The adapter for everything else, and for any model on request.
BLACK_BOX = "yoru-detector"
#: The other names YORU's get_detector accepts for its backends.
BACKEND_ALIASES: Dict[str, str] = {
    "yolov8": "ultralytics", "yolo11": "ultralytics",
    "fasterrcnn": "torchvision", "maskrcnn": "torchvision", "ssd": "torchvision",
}

#: Models that are planned but not explained, and why.
UNSUPPORTED: Dict[str, str] = {
    "yoake": "YOAKE: temporal, track-aligned behaviour model; adapter planned for v0.4",
}


def list_adapters() -> List[AdapterEntry]:
    return list(_ADAPTERS.values())


def adapter_entry(name: str) -> AdapterEntry:
    if name not in _ADAPTERS:
        why = UNSUPPORTED.get(name)
        raise ExplainError(
            ErrorCode.MODEL_UNSUPPORTED,
            f"no adapter {name!r}" + (f" ({why})" if why else "")
            + f"; available: {sorted(_ADAPTERS)}")
    return _ADAPTERS[name]


def create_adapter(name: str):
    """A new, unloaded adapter instance."""
    entry = adapter_entry(name)
    module = importlib.import_module(entry.module)
    return getattr(module, entry.cls)()


def adapter_for_backend(backend: str) -> str:
    """The adapter that stands in for YORU backend *backend*; the black box when none does."""
    backend = BACKEND_ALIASES.get(backend, backend)
    return BACKEND_ADAPTERS.get(backend, BLACK_BOX)


def resolve_model(path, backend: str = "auto", *, black_box: bool = False) -> Tuple[str, str, Dict[str, Any]]:
    """``(adapter, YORU backend, adapter options)`` for a model file, as YORU would load it.

    ``backend="auto"`` asks YORU which backend it would pick for the file
    (``get_detector("auto", ...)``'s choice), so the explanation is of the
    model family YORU runs.  *black_box* -- or a backend without a
    model-specific adapter -- gives the black-box adapter, told the backend.
    """
    backend = str(backend or "auto")
    if backend == "auto":
        from yoru_explain.adapters import _yoru

        backend = _yoru.auto_backend(path)
    backend = BACKEND_ALIASES.get(backend, backend)
    adapter = BLACK_BOX if black_box else adapter_for_backend(backend)
    return adapter, backend, ({"backend": backend} if adapter == BLACK_BOX else {})
