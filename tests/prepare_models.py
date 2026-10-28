"""Download or build the models the integration tests use, into ``.cache/models/`` (git-ignored).

::

    uv run --with "onnx>=1.16" python -m tests.prepare_models

* ultralytics weights from ultralytics' release assets (YOLO11 / YOLOv8 /
  YOLOv5u / YOLOv10, the YOLO11 OBB, segmentation and pose models, RT-DETR);
* torchvision's COCO Faster R-CNN, Mask R-CNN and SSD300, saved in the layout
  YORU's ``train_torchvision.py`` writes (``model_type``, ``num_classes``,
  ``names``, ``model_state_dict``), so that YORU's torchvision plugin loads
  them as it loads a model trained in YORU (it also fetches the ImageNet
  backbone weights torchvision builds those models with);
* ONNX exports of ``yolov5s.pt`` (YOLOv5 layout, class names in the metadata)
  and ``yolo11n.pt`` (YOLOv8 layout, no metadata, so YORU's row-count rule
  decides the layout) -- the step that needs the ``onnx`` package, hence
  ``--with``.  It is skipped when ``onnx`` is missing.

None of these is a YORU model or a model of animals: they are fixtures that
let the tests compare each adapter with YORU's own detector.  The YOLOv5 and
fly models of v0.1 are described in docs/INITIAL_AUDIT.md section 5.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / ".cache" / "models"
ULTRALYTICS_ASSETS = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"
ULTRALYTICS = ("yolo11n.pt", "yolov8n.pt", "yolov5nu.pt", "yolov10n.pt", "yolo11n-obb.pt",
               "yolo11n-seg.pt", "yolo11n-pose.pt", "rtdetr-l.pt")
YOLOV5_ASSETS = "https://github.com/ultralytics/yolov5/releases/download/v7.0/"


def _download(url: str, path: Path) -> None:
    if path.is_file():
        return
    print(f"downloading {url}")
    tmp = path.with_name(path.name + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(path)


def ultralytics_weights() -> None:
    for name in ULTRALYTICS:
        _download(ULTRALYTICS_ASSETS + name, MODELS / name)
    _download(YOLOV5_ASSETS + "yolov5s.pt", MODELS / "yolov5s.pt")


def torchvision_checkpoints() -> None:
    import torch
    from torchvision.models import VGG16_Weights
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_Weights,
        MaskRCNN_ResNet50_FPN_Weights,
        SSD300_VGG16_Weights,
        fasterrcnn_resnet50_fpn,
        maskrcnn_resnet50_fpn,
        ssd300_vgg16,
    )

    builds = {
        "fasterrcnn": (fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights.COCO_V1),
        "maskrcnn": (maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights.COCO_V1),
        "ssd": (ssd300_vgg16, SSD300_VGG16_Weights.COCO_V1),
    }
    for model_type, (build, weights) in builds.items():
        path = MODELS / f"coco_{model_type}.pt"
        if path.is_file():
            continue
        print(f"building {path.name}")
        model = build(weights=weights)
        categories = list(weights.meta["categories"])     # [0] is the background
        names = {i: str(c) for i, c in enumerate(categories[1:])}
        torch.save({"epoch": 0, "model_state_dict": model.state_dict(), "num_classes": len(names),
                    "names": names, "model_type": model_type}, path)
    # What YORU's plugin builds SSD300 with before loading the checkpoint.
    VGG16_Weights.IMAGENET1K_FEATURES.get_state_dict(progress=True)


def onnx_exports() -> None:
    try:
        import onnx
    except ImportError:
        print("onnx is not installed: ONNX exports skipped (run with --with onnx)")
        return
    import torch

    v5 = MODELS / "yolov5s.onnx"
    if not v5.is_file():
        from yoru_explain.adapters import _yoru

        y = _yoru.load()
        model = y.load_yolov5_model(str(MODELS / "yolov5s.pt"), torch.device("cpu")).model.model
        head = model.model[-1]
        head.inplace, head.export, head.dynamic = False, True, False
        names = {int(k): str(v) for k, v in dict(enumerate(model.names) if isinstance(model.names, list)
                                                  else model.names).items()}
        _export(model.eval(), v5, names)
    v8 = MODELS / "yolo11n.onnx"
    if not v8.is_file():
        from ultralytics import YOLO

        model = YOLO(str(MODELS / "yolo11n.pt")).model.fuse().eval()
        head = model.model[-1]
        head.export, head.format = True, "onnx"
        _export(model, v8, None)
    print("ONNX:", onnx.__version__)


def _export(model, path: Path, names) -> None:
    import onnx
    import torch

    print(f"exporting {path.name}")
    with torch.no_grad():
        torch.onnx.export(model, torch.zeros(1, 3, 640, 640), str(path), opset_version=17,
                          input_names=["images"], output_names=["output0"], dynamo=False)
    if names is not None:
        proto = onnx.load(str(path))
        onnx.helper.set_model_props(proto, {"names": str(names), "stride": "32"})
        onnx.save(proto, str(path))


def main(torchvision: bool = True, onnx: bool = True) -> None:
    """Everything, or -- ``torchvision=False``, as CI does -- without the ~1 GB of torchvision models."""
    MODELS.mkdir(parents=True, exist_ok=True)
    ultralytics_weights()
    if torchvision:
        torchvision_checkpoints()
    if onnx:
        onnx_exports()
    print("models in", MODELS)


if __name__ == "__main__":
    main()
