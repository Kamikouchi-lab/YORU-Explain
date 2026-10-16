# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Adapter names and where their code lives.

Adapters are imported only when created, so listing them does not load
torch or YORU.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Dict, List

from yoru_explain.core.errors import ErrorCode, ExplainError


@dataclass(frozen=True)
class AdapterEntry:
    name: str
    module: str
    cls: str
    label: str
    #: Shown in the GUI so nobody mistakes the test model for a real one.
    synthetic: bool = False


_ADAPTERS: Dict[str, AdapterEntry] = {
    "yoru-yolov5": AdapterEntry(
        "yoru-yolov5", "yoru_explain.adapters.yolov5", "YoruYolov5Adapter",
        "YOLOv5 (YORU's bundled YOLOv5 code)"),
    "toy-grid": AdapterEntry(
        "toy-grid", "yoru_explain.adapters.toy", "ToyGridAdapter",
        "Synthetic test detector (colour squares; no weights)", synthetic=True),
}

#: Adapters that are planned but not implemented, and why.
UNSUPPORTED: Dict[str, str] = {
    "ultralytics": "YOLOv8/YOLO11/YOLO26/RT-DETR: NMS runs inside ultralytics; adapter planned",
    "torchvision": "Faster R-CNN / SSD: two-stage, re-detection only; adapter planned",
    "onnx": "ONNX Runtime: no gradients; black-box re-detection planned",
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
