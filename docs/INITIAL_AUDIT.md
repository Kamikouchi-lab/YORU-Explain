# Initial audit (phase 0)

Date: 2026-10-16. Everything below was read from the code on disk or measured
on this machine; nothing is inferred from names alone. Where a fact could not
be checked, it says so.

## 1. What was inspected

| Repository | Where | State |
|---|---|---|
| YORU | `../YORU` | branch `yoru-tracker-lite` @ `664fa5c`, clean, version `2.0.0b3` (`yoru/__init__.py`) |
| YORU-Tracker | `../YORU-Tracker` | branch `crowded-arena-tracking` @ `7f14298`, version `0.1.0`; one uncommitted blank-line change in its `pyproject.toml` (not ours, left alone) |
| YOAKE | `~/YOAKE` — **outside the workspace** | read-only overview only (section 7) |
| YORU-Explain | `../YORU-Explain` | empty directory; this repository |

YORU and YORU-Tracker were not modified. Nothing was written inside either
checkout (the test-model dataset was copied out first, because YOLOv5 writes
a `labels.cache` beside the labels it reads).

## 2. How YORU runs a YOLOv5 model (verified)

| Step | Code | Facts |
|---|---|---|
| Backend choice | `yoru/libs/plugins/__init__.py` `get_detector`, `_auto_detect_backend`, `_sniff_checkpoint` | A YOLOv5 checkpoint is recognised from its pickle (`models.yolo` / `models.common` markers), whatever the file is called. |
| Load | `yoru/libs/plugins/yolov5_detector.py` `load_yolov5_model(path, device)` | `AutoShape(DetectMultiBackend(weights, device=device, fuse=True)).to(device)`, built under `yoru.libs.yolov5_support.yolov5_importable()`. Classification/segmentation checkpoints refused. |
| Device | `yoru/libs/device.py` `torch_device` | `auto` → `$YORU_DEVICE`, else CUDA, else MPS, else CPU. |
| Channel order | `rgb_input_requested` | **BGR frames by default** (as YORU v1); RGB only with `YORU_YOLOV5_RGB=1` or `rgb_input=True`: `image[..., 2::-1]`. |
| Thresholds | `YOLOv5Detector.load` | `AutoShape.conf = min(conf_thresh, 0.25)`, `iou = 0.45`; a threshold above 0.25 is applied **after** NMS (`conf < thresh` dropped). `max_det = 1000`, `agnostic=False`, `multi_label=False`, `classes=None`. |
| Preprocess | bundled `models/common.py` `AutoShape.forward` | `im[..., :3]` (grey → `cv2.COLOR_GRAY2BGR`); `g = 640 / max(h, w)`; `shape1 = make_divisible(int(h·g)), make_divisible(int(w·g))` (stride 32); `letterbox(im, shape1, auto=False)` (`cv2.INTER_LINEAR`, pad value 114, pads `round(d/2 ∓ 0.1)`); BCHW; `/255` as the parameter dtype (fp32). |
| Forward | same | `torch.amp.autocast(device_type="cuda", enabled=True)` around the model **and NMS** (a YORU v1 change kept in the copy); `AutoShape.forward` itself is `@smart_inference_mode` (= `torch.inference_mode`). |
| Raw output | bundled `models/yolo.py` `Detect.forward` (`export=True`, `inplace=False` set by AutoShape) | one tensor `(B, N, 5+nc)`: `[cx, cy, w, h, σ(obj), σ(cls_0..nc-1)]` in letterbox pixels; `N = Σ_levels na·ny·nx`, flattened level → anchor → y → x. |
| NMS | bundled `utils/general.py` `non_max_suppression` | `obj > conf`; `conf = obj·cls` (`x[:, 5:] *= x[:, 4:5]`); best class only; `conf > conf`; sort descending; class-offset boxes (`+ cls·7680`); `torchvision.ops.nms`; first 1000. |
| Back to the frame | `scale_boxes(shape1, boxes, shape0)` | `gain = min(h1/h0, w1/w0)`, `pad = ((w1 − w0·gain)/2, (h1 − h0·gain)/2)` **as floats** (letterbox itself pads whole pixels, so the two can differ by ≤ 0.5 letterbox px); clipped to the frame. |
| Output | `YOLOv5Detector.detect` | `list[dict]`: `x1 y1 x2 y2 conf class_id class_name`, Python floats from the fp32 tensor, in NMS order (descending confidence). |

Reported confidence is therefore **σ(objectness) × σ(class logit)** of one
pre-NMS candidate, as YOLOv5 defines it.

### Measured on this machine (`yolov5s.pt`, `zidane.jpg` and the test video)

* Re-implementing AutoShape's preprocessing and calling
  `AutoShape.model(x)` + the bundled NMS + `scale_boxes` reproduces
  `YOLOv5Detector.detect` **exactly** (max |diff| = 0.0) on CUDA and on the CPU.
* Every post-NMS detection maps back to **exactly one** pre-NMS candidate by
  bitwise equality of its `xywh2xyxy` box; recomputing `pred[k,5+c]·pred[k,4]`
  for that candidate gives the reported confidence bit for bit. A
  fixed-candidate ("候補固定") mode is therefore possible for YOLOv5.
* On CUDA the objectness/class values are fp16-quantised (autocast) inside an
  fp32 tensor, so ties are common — the reason YORU keeps NMS at 0.25.
* A repeated batch-1 forward pass is bit-identical (CUDA and CPU).
* **Scores depend on the batch size**: the same image scored in a batch of 4
  differs from batch 1 by up to 2.4e-3 (CUDA autocast; 8.4e-4 in fp32),
  while identical images within one batch agree exactly. Occlusion deltas
  must therefore be taken against a reference scored at the same batch size.
* Gradients: the model's parameters have `requires_grad=False` (EMA weights),
  so gradients are obtained w.r.t. activations with `requires_grad` set on the
  input. The first attempt failed — the Detect head caches its anchor grids
  from the first (inference-mode) call, and inference tensors cannot be saved
  for backward. Cloning the cached grids (outside inference mode) for the
  gradient pass fixes it; YOLOv5's in-place SiLU (`inplace=True` on 57
  modules) did not break autograd. The gradient-pass score equals the
  reference score exactly under autocast.
* Gradient support of one candidate (level 2 / stride 32) at the Detect
  inputs: layer 23 (its own level) — exactly **1** non-zero cell (the head
  is a 1×1 conv); layer 20 — 49 cells; layer 17 — 361 cells; layer 9 (SPPF) —
  49 cells.
* Speed (RTX 5060 Ti / CPU, 640×384 input): 4.5 ms per image at batch 1,
  1.0 ms at batch 8 on CUDA; 23.5 ms / 16.6 ms on the CPU.

## 3. Detection files YORU and YORU-Tracker write

| File | Writer | Columns | Notes |
|---|---|---|---|
| `<video>.csv` | `yoru/libs/analysis.py` (video analysis) | `frame, x1, y1, x2, y2, x_center, y_center, w, h, angle, confidence, class, class_name[, tracking_id]` | `frame` counts frames read sequentially with `cv2.VideoCapture.read()`, from 0. Rows below the GUI threshold are dropped. **`v_flip`/`h_flip` are applied before detection and not recorded in the file**, so the importer must be told. |
| `*_detect.csv` | realtime | `DETECTION_COLUMNS` = `x1 y1 x2 y2 confidence class class_name total_time cx cy w h angle` | Rows repeat until the next detector result; frame numbers need the `*_log.csv`. Not imported by v0.1. |
| `*_detections.csv` | YORU-Tracker | `frame_id` + `DETECTION_COLUMNS` | Readable later (v0.4). |

None of these carries the model hash, thresholds, channel order or flips.
YORU-Explain therefore re-runs the detector on the frame and matches the
imported box to its own detection before analysing it; a box it cannot
reproduce is reported (`target_not_reproduced`), never analysed.

## 4. Models v0.1 can explain

| Model | v0.1 | Why |
|---|---|---|
| YOLOv5 detection checkpoints loaded by YORU's bundled YOLOv5 (every YORU v1 model, upstream `yolov5*.pt`) | **supported**: occlusion in fixed-candidate and re-detection modes; Grad-CAM after a gradient probe passes | Sections 2 and the measurements above. |
| YOLOv5 segmentation / classification | no | YORU refuses them. |
| ultralytics YOLOv8/11/26, RT-DETR (`ultralytics_detector.py`) | no (planned adapter) | NMS happens inside `ultralytics.YOLO.__call__`; candidates would need the ultralytics `nn` model. Different head (anchor-free, no objectness). |
| `yolov5*u` | no | ultralytics' network; never routed to the bundled code (YORU rule). |
| torchvision Faster R-CNN / SSD | no | two-stage; candidate identity is a proposal. Re-detection mode only, later. |
| ONNX (`onnx_detector.py`) | no | no gradients; black-box re-detection possible later. |
| OBB models | no | YOLOv5 has no rotated head; matching would need polygon IoU. |
| YOAKE | no (v0.4) | Section 7. |

## 5. Dependencies and environment

* Python 3.10.22 (YORU and YORU-Tracker require `>=3.10,<3.11`).
* `uv` 0.12.23. `uv sync` here resolves YORU editable from `../YORU`, and the
  same builds YORU and YORU-Tracker use: torch 2.11.0+cu128,
  torchvision 0.26.0+cu128, opencv-python 4.10.0, numpy 1.26.4, PyQt5
  5.15.11 / PyQt5-Qt5 5.15.2. GPU: NVIDIA GeForce RTX 5060 Ti (CUDA 12.8).
* `torch` and `torchvision` are direct dependencies here, so the cu128 index
  is declared in this `pyproject.toml` as in YORU's.
* No YOLOv5 checkpoint existed on this machine (searched the home directory;
  the `E:` drive named in an earlier session is not mounted). For tests:
  * `yolov5s.pt` / `yolov5n.pt` downloaded from the ultralytics/yolov5 v7.0
    release into `.cache/models/` (git-ignored);
  * `.cache/models/fly_copulation_yolov5s.pt`: trained here by YORU's own
    `yoru/libs/train_yolov5.py` (40 epochs from `yolov5s.pt`, classes
    `copulation`, `solo`) on YORU's `test_data/labeled_frames`, every 10th
    frame held out. Held-out mAP50 0.982 — but those frames come from the
    same video, so this number is not a generalisation estimate. It is a
    **test fixture**, not the YORU paper's model.

## 6. GUI framework

> **Revised 2026-10-17:** at the user's request the GUI was rebuilt with
> DearPyGui, like YORU and YORU-Tracker, on YORU's documented GUI primitives
> (`GuiSession`, `apply_default_theme`). The PyQt5 assessment below is kept
> as it was made.

PyQt5, as the plan proposes. Checked:

* YORU's own windows and YORU-Tracker use DearPyGui; PyQt5 is already a YORU
  dependency on Windows (labelImg), pinned to 5.15.11.
* PyQt5 is GPL v3, compatible with an AGPL-3.0 application.
* Qt's `offscreen` platform works here, so GUI smoke tests run headless.
* This application builds its own windows; YORU's DearPyGui primitives
  (`GuiSession`, `apply_default_theme`) are not used. Logs still go to YORU's
  log directory (`get_log_dir()`), as `yoru_explain.log`.
* opencv-python on Linux ships Qt plugins that clash with PyQt5 — not an
  issue on Windows (checked), noted for later platforms.

## 7. YOAKE (outside the workspace)

`~/YOAKE` (git, `73aecd5`), package `htrtdetr` 0.1.0, MIT licence: a
"Hierarchical Temporal RT-DETR" — ResNet+FPN backbone, RT-DETR-OBB detector,
OBB-aware tracking, per-track ring buffer, a temporal module over 3/8/16
track-aligned frames and a behaviour head. Behaviour labels depend on a
track's history, so single-frame occlusion is not defined for them without
deciding what happens to the history. No trained weights in `~/YOAKE/weights`.
It is listed as an **unsupported adapter** for v0.4; its inference API was not
studied beyond the file list, and nothing here depends on it.

## 8. The boundary with YORU

YORU's `docs/external_api.md` lists the supported names. The YOLOv5 adapter
needs more than `DetectorBase.detect()` returns — pre-NMS candidates, head
logits, feature maps and gradients — so it also uses YORU names that are
**not** in that list. They are confined to `src/yoru_explain/adapters/_yoru.py`
and pinned by `tests/unit/test_yoru_boundary.py`:

| Name | Documented? | Used for |
|---|---|---|
| `yoru.__version__` | yes | provenance |
| `yoru.libs.plugins.get_detector`, `DEFAULT_CONF_THRESH`, `DEFAULT_IOU_THRESH` | yes | identity check against YORU's own detector; defaults |
| `yoru.libs.user_paths.get_log_dir` | yes | log location |
| `yoru.libs.plugins.yolov5_detector.load_yolov5_model`, `rgb_input_requested`, `V1_NMS_CONF` | **no** | the very model object YORU builds, its channel-order rule and its NMS threshold |
| `yoru.libs.yolov5_support.yolov5_importable` | **no** | importing the bundled `letterbox`, `make_divisible`, `non_max_suppression`, `scale_boxes`, `xywh2xyxy` |
| `yoru.libs.device.torch_device` | **no** | the same device YORU would pick |

Because these can change without notice, every run checks the adapter
against `get_detector(...).detect()` on its first frame and records the
result in `manifest.json`. A proposal to document them in YORU is in
`docs/YORU_PR_PROPOSAL.md`. It was opened as Kamikouchi-lab/YORU#43 on
2026-10-17; it changes YORU's docs and tests only.

YORU never imports `yoru_explain` (checked by the same test), and YORU's GUI
gets no branches for it.

## 9. Open questions (未確定仕様)

1. Which model and video should the first real demo use? The plan names a
   wing-extension event; no such model is on this machine. The fixture model
   detects `copulation` / `solo`.
2. Are the YORU paper models available, and were they run with BGR (v1
   default) or RGB? Identity with YORU is verified here only for `yolov5s.pt`
   and the fixture model.
3. ~~Default occlusion baseline~~ — decided 2026-10-17: hot pink (#FF69B4,
   BGR 180,105,255). Original note: black is out of distribution for a
   bright-arena video; the 114-grey letterbox value and a blur are offered.
   Which to make the lab default is a scientific choice.
4. Whether `ambiguous` re-detection matches (two same-class boxes over the
   target) should count in maps; v0.1 excludes them and counts them.
5. ~~Should the YORU names in section 8 become documented external API?~~ —
   proposed to YORU on 2026-10-17 as Kamikouchi-lab/YORU#43 (awaiting review).
6. YOAKE's inference contract (single frame vs. track history) before a v0.4
   adapter.

## 10. Licences

* YORU, its bundled YOLOv5 copy and YORU-Tracker: AGPL-3.0(-or-later).
  YORU-Explain links them in-process, so it is AGPL-3.0-or-later too
  (`LICENSE` copied from YORU-Tracker).
* PyQt5 GPL v3, PyQt5-Qt5 LGPL v3 — compatible.
* Downloaded `yolov5*.pt` weights are ultralytics' (AGPL-3.0); they are not
  committed. YOAKE is MIT; nothing of it is used.
* Video data and trained weights may be confidential lab data: the run
  manifest records hashes and paths, never copies of videos or weights.
