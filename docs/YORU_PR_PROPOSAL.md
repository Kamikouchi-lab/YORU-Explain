# Proposal for YORU: document the YOLOv5 entry points analysis tools need

**Status: proposal only. Nothing in YORU has been changed.** This is the text
of a possible separate PR to YORU, for the maintainers to accept, change or
reject.

## Why

YORU's external API (`docs/external_api.md`) ends at `DetectorBase.detect()`:
boxes after NMS. An explanation tool needs what is underneath — the model
YORU builds, its pre-NMS candidates and gradients — and so YORU-Explain also
imports five names YORU does not promise to keep:

| Name | Used for |
|---|---|
| `yoru.libs.plugins.yolov5_detector.load_yolov5_model(model_path, device)` | the very `AutoShape(DetectMultiBackend(...))` YORU runs |
| `yoru.libs.plugins.yolov5_detector.rgb_input_requested(value=None)` | YORU's BGR/RGB rule (`YORU_YOLOV5_RGB`) |
| `yoru.libs.plugins.yolov5_detector.V1_NMS_CONF` | the confidence NMS runs at (0.25) |
| `yoru.libs.yolov5_support.yolov5_importable()` (and `YOLOV5_DIR`) | importing the bundled `letterbox`, `non_max_suppression`, `scale_boxes`, ... |
| `yoru.libs.device.torch_device(preference)` | the device YORU would pick |

Today they are confined to one module (`src/yoru_explain/adapters/_yoru.py`),
pinned by a test there, and every run compares the result with `detect()`.
That catches a change, but only after it has happened. Documenting them in
YORU would turn a silent break into a failing YORU test, as for YORU-Tracker.

No behaviour changes; documentation and one test file.

## Proposed change 1 — `docs/external_api.md`

Add a section after "One detection":

```markdown
## The YOLOv5 model underneath — `yoru.libs.plugins.yolov5_detector`, `yoru.libs.yolov5_support`

For tools that analyse a YOLOv5 model rather than only its boxes (YORU-Explain).
They give the same model and settings `get_detector("yolov5", ...)` uses.

| Name | What it is |
|------|------------|
| `load_yolov5_model(model_path, device)` | The bundled YOLOv5's `AutoShape(DetectMultiBackend(weights, fuse=True))` on `device` (a `torch.device`), as the YOLOv5 plugin loads it. Refuses non-YOLOv5 and non-detection checkpoints. |
| `rgb_input_requested(value=None)` | Whether frames go in RGB (`value`, else `$YORU_YOLOV5_RGB`); BGR otherwise, as in YORU v1. |
| `V1_NMS_CONF` | The confidence YOLOv5's NMS runs at; a higher threshold is applied after NMS. |
| `yolov5_support.yolov5_importable()` | Context manager that makes the bundled `models` / `utils` packages importable (as `torch.hub.load` did for v1). |
| `yolov5_support.YOLOV5_DIR` | Where the bundled code lives. |
| `device.torch_device(preference="auto")` | The `torch.device` YORU resolves `preference` to (`$YORU_DEVICE`, CUDA, MPS, CPU). |

The bundled YOLOv5's own functions (`utils.augmentations.letterbox`,
`utils.general.non_max_suppression`, ...) belong to the copy of
ultralytics/yolov5 that YORU v1.1.2 shipped; they are not YORU API, and YORU
keeps the copy as it is.
```

## Proposed change 2 — `tests/test_public_api.py`

```diff
 PUBLIC_NAMES = {
     "yoru.libs.plugins": (
         "get_detector", "list_detector_backends",
         "DEFAULT_CONF_THRESH", "DEFAULT_IOU_THRESH",
     ),
+    "yoru.libs.plugins.yolov5_detector": (
+        "load_yolov5_model", "rgb_input_requested", "V1_NMS_CONF",
+    ),
+    "yoru.libs.yolov5_support": ("yolov5_importable", "YOLOV5_DIR"),
+    "yoru.libs.device": ("torch_device",),
     "yoru.libs.detector_base": (
```

```python
def test_load_yolov5_model_signature():
    pytest.importorskip("torch")
    from yoru.libs.plugins.yolov5_detector import V1_NMS_CONF, load_yolov5_model

    assert _params(load_yolov5_model)[:2] == ["model_path", "device"]
    assert V1_NMS_CONF == 0.25
```

(The `yolov5_detector` module imports torch, so its line in `PUBLIC_NAMES`
needs the same `importorskip` guard the other torch-dependent tests use.)

## Not part of the PR — two observations from YORU-Explain's validation

1. **Windows, PyQt5 before torch.** With PyQt5-Qt5 5.15.2 (YORU's pin),
   `from PyQt5 import QtWidgets` loads `PyQt5/Qt5/bin/MSVCP140.dll`; a torch
   imported afterwards in the same process fails with WinError 1114 in
   `c10.dll` (torch 2.11.0+cu128). YORU-Explain imports torch first. If any
   YORU process imports PyQt5 (labelImg) before torch, it would hit the same
   error.
2. **`test_data/test_video_fly_copulation.mp4` states 30.0032 fps; its frames
   are 33.333 ms apart.** OpenCV's `CAP_PROP_POS_FRAMES` seek therefore lands
   one frame early from about frame 6000 on (seek 6301 → frame 6300). YORU's
   video analysis reads in order and is not affected; anything that seeks by
   frame number is.
