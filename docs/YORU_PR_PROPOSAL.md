# YORU pull request: document the YOLOv5 entry points analysis tools need

**Status: opened 2026-10-17 as
[Kamikouchi-lab/YORU#43](https://github.com/Kamikouchi-lab/YORU/pull/43)**
(branch `document-yolov5-entry-points`, base `yoru-tracker-lite`), awaiting
review. It changes YORU's documentation and tests only. Until it is merged,
the names below remain undocumented in YORU and YORU-Explain keeps treating
them as internal (`src/yoru_explain/adapters/_yoru.py`,
`tests/unit/test_yoru_boundary.py`).

## Why

YORU's external API (`docs/external_api.md`) ends at `DetectorBase.detect()`:
boxes after NMS. An explanation tool needs what is underneath: the model
YORU builds, its pre-NMS candidates and its gradients. So YORU-Explain also
imports these names, which YORU did not promise to keep:

| Name | Used for |
|---|---|
| `yoru.libs.plugins.yolov5_detector.load_yolov5_model(model_path, device)` | the very `AutoShape(DetectMultiBackend(...))` YORU runs |
| `yoru.libs.plugins.yolov5_detector.rgb_input_requested(value=None)` | YORU's BGR/RGB rule (`YORU_YOLOV5_RGB`) |
| `yoru.libs.plugins.yolov5_detector.V1_NMS_CONF` | the confidence NMS runs at (0.25) |
| `yoru.libs.yolov5_support.yolov5_importable()` (and `YOLOV5_DIR`) | importing the bundled `letterbox`, `non_max_suppression`, `scale_boxes`, ... |
| `yoru.libs.device.torch_device(preference)` | the device YORU would pick |

Today they are confined to one module, pinned by a test there, and every
run compares the adapter with `detect()`. That catches a change, but only
after it has happened. Once YORU documents them, renaming one fails a YORU
test, as already happens for YORU-Tracker's names.

## What the pull request changes

* `docs/external_api.md`: a new section, *The YOLOv5 model underneath*,
  listing the six names above. YORU-Explain appears beside YORU-Tracker as a
  sister application, and the dependency rule covers it ("YORU never imports
  `yoru_tracker` or `yoru_explain`").
* `tests/test_public_api.py`: the names go into `PUBLIC_NAMES`.
  `test_yolov5_entry_point_signatures` pins the signatures and
  `V1_NMS_CONF == 0.25`. The dependency-rule test,
  `test_yoru_never_imports_a_sister_application`, checks both packages.
* `CLAUDE.md`: the same facts for future work on YORU.

These modules import without torch (checked), so the new `PUBLIC_NAMES`
entries need no `importorskip` guard. An earlier draft of this proposal said
they did; that was wrong.

Before opening the pull request, YORU's full suite was run on the branch:
678 passed, 4 skipped (Windows, uv environment).

## Since then (not in the pull request)

The other backends (2026-10-17) use two more YORU names, also confined to
`adapters/_yoru.py`: `yoru.libs.device.resolve_device` (the device string the
ultralytics plugin passes) and `yoru.libs.plugins._auto_detect_backend` (the
backend `get_detector("auto", ...)` picks). Documenting them — the second as
a public `detector_backend_of(path)`, say — would be a natural follow-up
pull request; nothing has been proposed yet.

## After it is merged

Move the five names from `INTERNAL` to `DOCUMENTED` in
`tests/unit/test_yoru_boundary.py`, and drop the "not promised" wording in
`adapters/_yoru.py`. The bundled YOLOv5 functions (`letterbox`, ...) stay
internal: they belong to the vendored ultralytics/yolov5 copy, not to YORU.

## Observations, sent with the pull request as notes (nothing changed for them)

1. **Windows, PyQt5 before torch.** With PyQt5-Qt5 5.15.2 (YORU's pin),
   `from PyQt5 import QtWidgets` loads `PyQt5/Qt5/bin/MSVCP140.dll`. A torch
   imported afterwards in the same process then fails with WinError 1114 in
   `c10.dll` (torch 2.11.0+cu128). YORU-Explain imports torch first. A YORU
   process that imports PyQt5 (labelImg) before torch would hit the same
   error.
2. **`test_data/test_video_fly_copulation.mp4` states 30.0032 fps, but its
   frames are 33.333 ms apart.** OpenCV's `CAP_PROP_POS_FRAMES` seek
   therefore lands one frame early from about frame 6000 on (seeking 6301
   decodes frame 6300). YORU's video analysis reads frames in order and is
   not affected; anything that seeks by frame number is.
