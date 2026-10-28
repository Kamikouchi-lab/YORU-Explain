# YORU's detection backends, one by one

Date: 2026-10-17. Every YORU detection backend
(`yoru.libs.plugins.list_detector_backends()`: `yolov5`, `ultralytics`,
`rtdetr`, `torchvision`, `onnx`) has an adapter here that runs the model the
way YORU runs it and exposes what is underneath. A backend without one — one
YORU adds later, say — runs through the black-box adapter. Everything below
was read from the code on disk (YORU 2.0.0b3 @ `34c2e4f`, ultralytics 8.3.222,
torchvision 0.26.0, onnxruntime 1.23.2) or measured on this machine; the
YOLOv5 backend is described in [INITIAL_AUDIT.md](INITIAL_AUDIT.md) §2.

| YORU backend | Models | Adapter | Candidate | Identity modes | Score YORU reports | Grad-CAM |
|---|---|---|---|---|---|---|
| `yolov5` | YOLOv5 checkpoints (every YORU v1 model) | `yoru-yolov5` | anchor slot (level, anchor, cell) | fixed-candidate, re-detection | `obj_x_cls` = σ(obj)·σ(cls) | yes |
| `ultralytics` | YOLOv8, YOLO11, YOLOv5u, YOLOv9, YOLOv10; detect, **OBB**, segment, pose | `yoru-ultralytics` | anchor point (level, cell) | fixed-candidate, re-detection | `cls_prob` = σ(cls) | yes |
| `rtdetr` | RT-DETR | `yoru-rtdetr` | decoder query | re-detection | `cls_prob` = σ(cls) | yes (the query on its frame) |
| `torchvision` | Faster R-CNN, Mask R-CNN (boxes) | `yoru-torchvision` | region proposal | re-detection | `cls_softmax` | yes (the proposal on its frame) |
| `torchvision` | SSD300 | `yoru-torchvision` | default box (level, cell, aspect) | fixed-candidate, re-detection | `cls_softmax` | yes |
| `onnx` | ONNX exports of YOLOv5 / YOLOv8 / YOLO11 | `yoru-onnx` | output row | fixed-candidate, re-detection | `obj_x_cls` or `cls_prob` | no (ONNX Runtime has no gradients) |
| any | anything `get_detector` loads | `yoru-detector` (black box) | none | re-detection | `confidence` | no |

`confidence` is offered by every adapter and is always the number YORU
reports; the family's own name for it is listed per adapter
(`capabilities.reported_score`). Choosing the model type `auto` asks YORU
which backend `get_detector("auto", ...)` would pick for the file
(`adapters/_yoru.auto_backend`), so the explanation is of the model family
YORU runs.

## What every adapter guarantees

* **The detections are YORU's.** `identity_check` loads YORU's own detector
  (`get_detector(backend, path, conf, iou, device=...)`) and compares every
  box, confidence and class — and, for OBB models, the rotated box. A run
  checks this on its first frame and stops if it fails. Measured: equal bit
  for bit for every model listed in §7, on CPU and CUDA.
* **A candidate's readout is its detection.** Reading a detection's
  candidate on the unchanged frame gives its reported confidence bit for bit
  (tested for every detection on the test images).
* **The gradient pass is the inference pass.** Grad-CAM is offered only when
  a probe on the loaded model passed, and a map is made only when the
  gradient-pass score equals the inference score (measured difference 0 for
  every family). Weights are never changed, modules never put in training
  mode; the graph starts at a copy of the input.
* **Fixed-candidate mode only where a candidate is a fixed slot.** RT-DETR's
  queries and Faster R-CNN's proposals are chosen anew for every input, so
  "the same candidate" on a perturbed frame is not defined; those adapters
  refuse fixed-candidate mode (before a run starts) and offer re-detection.

## 1. ultralytics: YOLOv8, YOLO11, YOLOv5u, YOLOv9, YOLOv10 (`yoru-ultralytics`)

**How YORU runs it** (`yoru/libs/plugins/ultralytics_detector.py`):
`ultralytics.YOLO(path)`, then per frame `model(frame, conf=..., iou=...,
device=resolve_device(...), verbose=False)`; boxes from `result.boxes`, or —
for OBB models — `result.obb` (`xyxy` = the upright box around the rotated
one, plus `cx, cy, w, h, angle`).

**ultralytics' predictor**, built by the first call and reused:

| Step | Facts |
|---|---|
| Input size | `imgsz` from the checkpoint's training arguments (640; the OBB models were trained at 1024) |
| Letterbox | `LetterBox(imgsz, auto=rect, stride=32)`, `rect=True` for predict: resize to `round(w0·r) × round(h0·r)`, pad to the next multiple of 32 (`(d mod 32)/2`, `round(·∓0.1)`), grey 114 |
| Colour | BGR → RGB, `/255`, **fp32** (`half=False`); no autocast |
| Forward | the fused model (`AutoBackend(fuse=True)`) under `torch.inference_mode` |
| Raw output | `(B, 4 + nc [+ extra], N)`: `cx, cy, w, h` (input px), `σ(class)`, then mask coefficients (segment), keypoints (pose) or the angle (OBB); `N = Σ_levels h·w`, level → y → x |
| NMS | `non_max_suppression(conf, iou, classes=None, agnostic=False, max_det=300, nc=0 for detect else nc, rotated=obb)`: best class only, class-offset boxes, `torchvision.ops.nms` (OBB: `batch_probiou` fast-NMS) |
| Back to the frame | `scale_boxes` with a **rounded** pad (`round((w1 − w0·g)/2 − 0.1)`), clipped; OBB: `regularize_rboxes`, `scale_boxes(xywh=True)`, upright box from the corners |
| YOLOv10 | an end-to-end head: the one-to-one branch's top-300 (anchor, class) pairs, NMS only cuts at conf |

The adapter calls the predictor's own `preprocess`, the fused model,
ultralytics' `non_max_suppression` with `return_idxs=True` (the candidate
index of every kept box; for YOLOv10 and older ultralytics, exact box
equality) and `scale_boxes`, one image per NMS call as YORU calls it.

**Candidate**: an anchor point — level, grid cell — of the anchor-free head:
fixed for a given input size, so fixed-candidate mode follows it. YOLOv10:
the anchor point of its one-to-one branch; that head can report one anchor
point under two classes, so "NMS keeps it" also needs the class
(`multi_label`).

**Scores**: `confidence` = `cls_prob` = σ(class logit) (there is no
objectness); `cls_logit` from forward hooks on the head's class convolutions
(`cv3`, YOLOv10: `one2one_cv3`), offered only when σ(logit) reproduces the
head's scores bit for bit (it does for every model tested).

**Grad-CAM**: default layer = the head's input at the candidate's level
(`model.<head.f[level]>`), as for YOLOv5; any layer whose map tiles the
input can be chosen. Unlike YOLOv5's 1×1 head, YOLOv8's head has two 3×3
blocks, so the gradient at its input covers 5×5 cells (measured 25), and for
OBB models fewer at the image edge. Two families needed care:

* **Pose**: the score is read from the head's boxes-and-scores tensor before
  the keypoints are appended (the keypoint decoding works in place, which
  autograd refuses); the values are those reported.
* **YOLOv10**: the one-to-one branch reads the head's inputs *detached*, and
  the weights are frozen, so nothing in it carries a graph back to the
  backbone. For the gradient pass each input is handed to that branch as a
  leaf copy (same values): the gradient of the score at the head input,
  through the branch that produced it. Earlier layers have no gradient path
  to the score and are not offered.

**Not explained here**: exported ultralytics formats (TorchScript, TensorRT,
OpenVINO, ...) — they run in the black-box adapter; classification models
(YORU detects nothing with them). YOLO-World / YOLOE heads load and are
compared with YORU like the others, but their class logits are not offered
(they score against text embeddings); no such model was tested.

**YOLO26 with ultralytics 8.3.222** (found while testing): `yolo26n.pt`
declares an end-to-end head in its model YAML (`end2end: True`,
`reg_max: 1`), but this ultralytics ignores the flag and runs the one-to-many
head with NMS. YORU's own detector then reports one "refrigerator" (0.597) on
ultralytics' `bus.jpg`, where YOLO11n reports a bus and four people. The
adapter reproduces YORU exactly, so it explains that output too — and warns
(`capabilities.warnings`, a popup in the GUI) that a newer ultralytics may be
needed. Nothing was changed in YORU.

## 2. RT-DETR (`yoru-rtdetr`)

**How YORU runs it**: `ultralytics.RTDETR(path)` — chosen when the file name
contains `rtdetr` — called like the YOLO models.

| Step | Facts |
|---|---|
| Letterbox | `LetterBox(imgsz, auto=False, scale_fill=True)`: the frame is **stretched** to 640 × 640, no padding |
| Forward | fp32, inference mode; output `(B, 300, 4 + nc)`: normalised `cx, cy, w, h` and σ(class) of the decoder's 300 queries |
| Postprocess | no NMS: per query the best class; queries above conf, sorted by score (`argsort`), at most 300; boxes `xywh2xyxy` × (frame width, height), not clipped |

The adapter repeats `RTDETRPredictor.postprocess` with the query index kept.

**Candidate**: a decoder query. The 300 queries are initialised from the
encoder tokens with the best class scores, so which query a given object
ends up in changes with the input: a query index on a perturbed frame is not
"the same candidate". RT-DETR is therefore followed by **re-detection only**.
On its own frame a query is well defined, and its locator records the encoder
token it was selected from (recomputed from a hook on `enc_score_head`, as
the decoder selects it) and that token's level and cell.

**Grad-CAM**: default layer = the decoder input at the query's token level;
the score's gradient spreads through the deformable attention (measured: 342
cells at stride 32 for the top detection on `bus.jpg`). The decoder's anchor
and mask caches, built under inference mode, are swapped for clones during
the gradient pass.

**Batch dependence**: on CUDA, RT-DETR's scores move with the batch size
(1.6e-3 between batch 1 and 2, §7), and because the queries are re-selected
from the encoder's scores, a batch of *different* images can reorder or swap
low-confidence detections. The occlusion reference is taken at the run's
batch size, as always; use batch size 1 for exact work.

## 3. torchvision: Faster R-CNN, Mask R-CNN, SSD300 (`yoru-torchvision`)

**How YORU runs it** (`yoru/libs/plugins/torchvision_detector.py`): reads a
checkpoint written by YORU's `train_torchvision.py` (`model_type`,
`num_classes`, `names`, `model_state_dict`), rebuilds
`fasterrcnn_resnet50_fpn` / `maskrcnn_resnet50_fpn` / `ssd300_vgg16` with
`weights=None` and its own predictor, loads the weights, and per frame runs
`model([to_tensor(frame[..., ::-1])])` under `no_grad`; keeps scores
**strictly above** conf; class id = label − 1. **YORU's IoU threshold is not
used** — NMS runs inside torchvision at the model's own setting (Faster /
Mask R-CNN 0.5, SSD 0.45); the adapter records it as unused.

| Step | Faster / Mask R-CNN | SSD300 |
|---|---|---|
| Transform | normalise (ImageNet mean/std), bilinear resize by `min(800/short, 1333/long)`, pad bottom/right to a multiple of 32 | normalise, resize to exactly 300 × 300 |
| Candidates | RPN proposals (1000 after NMS), then per class: softmax, decode, clip, `> 0.05`, NMS 0.5, top 100 | 8732 default boxes; per class softmax `> 0.01`, top 400, NMS 0.45, top 200 |
| Back to the frame | `resize_boxes`: each axis × original / resized | same |

The adapter rebuilds the network with the same calls (which fetch
torchvision's ImageNet backbone weights, as YORU's do; they also decide the
frozen batch norms, so they are kept), and runs the model's own pieces in
its own order — `transform`, `backbone`, `rpn`, `roi_heads` pieces /
`head` + `anchor_generator`, `postprocess_detections`,
`transform.postprocess` — keeping the per-candidate softmax, logits and
boxes. Each detection's candidate is found by exact equality of its box and
score at its label. Mask R-CNN's mask branch is not run: YORU uses its boxes
only, and they do not depend on it.

**Candidates**: Faster / Mask R-CNN — a proposal of this input's RPN, chosen
anew for every input: **re-detection only**; Grad-CAM of the proposal on its
own frame. SSD — a default box (level, cell, aspect), fixed for its 300 × 300
input: fixed-candidate mode works. Both are multi-label (one candidate can be
reported under several classes), so "kept" checks the class too.

**Scores**: `confidence` = `cls_softmax` = softmax over the classes and the
background, at the class; `cls_logit` the class's logit (only differences
between classes matter to a softmax).

**Grad-CAM**: Faster / Mask R-CNN — default the FPN level the proposal is
pooled from (the box head's `LevelMapper`), also `backbone.body.layer1..4`
and `backbone.fpn.0..3` (`pool` is left out: it is used by the RPN only, and
its cells are not centred where the grid convention puts them). SSD — the
backbone level of the default box; SSD's grids overhang the 300 px input
(38 cells of 8 px), which `feature_to_frame` handles (cells centred at
`(j + ½)·step`, as the default boxes are).

## 4. ONNX (`yoru-onnx`)

**How YORU runs it** (`yoru/libs/plugins/onnx_detector.py`): ONNX Runtime with
every available provider; per frame: YORU's own letterbox to the model's
fixed input size (YOLOv5's integer pads), BGR → RGB, `/255` float32, one
`session.run`; rows `cx, cy, w, h, [obj,] classes` (a YOLOv8 export
transposed); objectness present when the metadata's class names say
`nc + 5` columns — or, without names, when there are more than 1.5 × the
8/16/32-grid cells of rows; best class; `> conf`; boxes `(c − pad)/ratio`
with the pad of the **rounded** resized size, clipped; `cv2.dnn.NMSBoxes`
with every class shifted into its own band.

These steps are private methods in YORU, so the adapter repeats them in the
same float32 arithmetic, keeping each detection's output row; the identity
check confirms the result on every run.

**Candidate**: an output row of a fixed-size model — an anchor slot (YOLOv5
export) or anchor point (YOLOv8 export), whose grid position is recorded
when the rows are the standard grid. Fixed-candidate mode works.

**Scores**: `confidence`; YOLOv5 exports also `obj_x_cls`, `cls_prob`,
`obj_prob` — all as exported, probabilities only (no logits in the output).
**No gradients**: occlusion only. One image per `session.run`, as YORU.

## 5. Black box (`yoru-detector`)

`get_detector(backend, path, conf, iou, device, ...)` itself: its `detect()`
on every input. Works for every backend YORU has or will have, and for files
the specific adapters do not open. No candidates, logits or gradients:
re-detection with the reported confidence, no Grad-CAM. Choose it in the
GUI with "black box", or `api.load_model(path, black_box=True)`.

## 6. Rotated boxes (OBB models)

* `Detection.obb` = YORU's `(cx, cy, w, h, angle)` (radians, from +x towards
  +y — `yoru.libs.obb`, ultralytics' `xywhr`); `box` stays the upright box
  around it, as YORU reports `x1..y2`.
* Overlap between two rotated boxes is the exact IoU of the two rectangles as
  polygons (re-detection matching, finding a target again, Explore). YOLO's
  OBB NMS uses a probabilistic IoU instead; the matching here does not need
  to agree with it.
* The occlusion region (`box` / `context`) is the upright box (scaled); the
  in-box summaries (`in_box_mean_delta`, `cam_in_box_fraction`) use the
  rotated box.
* Files: `events.csv` / `results.csv` carry `obb_cx … obb_angle`,
  `perturbations.csv` `matched_cx … matched_angle`, the NPZ `target_obb`
  (schema 0.2.0). YORU's detections CSVs bring their rotated boxes along
  (angle ≠ 0).
* The GUI draws rotated boxes as polygons and picks a box by the point being
  inside the rotated rectangle.

## 7. Measured on this machine

Test models (`tests/prepare_models.py`; fixtures, not models of animals):
ultralytics' `yolo11n`, `yolov8n`, `yolov5nu`, `yolov10n`, `yolo11n-obb`,
`yolo11n-seg`, `yolo11n-pose`, `rtdetr-l`; torchvision's COCO Faster R-CNN,
Mask R-CNN and SSD300 saved in YORU's checkpoint layout; ONNX exports of
`yolov5s.pt` (with class names) and `yolo11n.pt` (without, so YORU's
row-count rule decides).

| Check | Result |
|---|---|
| Identity with YORU's detector, `bus.jpg`, `zidane.jpg`, frame 150 of YORU's test video | **bit for bit** for all 13 models, CPU and CUDA (ONNX: CPU, the only provider here) |
| Every detection's candidate recovered, its readout = its confidence | bit for bit, all models |
| Gradient-pass score − inference score | 0 for every gradient-capable model |
| Grad-CAM leaves the model alone | weights (state-dict SHA-256), eval mode, no parameter gradients, detections unchanged |
| Occlusion controls | `noop` Δ = 0 and reference = YORU's confidence in every offered mode; unsupported modes refused |
| Batch of 3 different images | each keeps its own detections |
| A project run with an OBB model | identity exact; rotated-box columns filled; re-detection matches rotated boxes |

Per model, on ultralytics' `bus.jpg` (1080 × 810), RTX 5060 Ti / CPU
(the probe at load time measures the batch column the same way):

| Model | Input (h × w) | Batch 1 vs 2: largest score change, CPU / CUDA | Forward per image, CUDA batch 1 / 8 (CPU batch 1) | Grad-CAM default layer (stride): cells with gradient |
|---|---|---|---|---|
| `yolov5s.pt` (v0.1) | 640 × 480 | 2e-6 / 2.9e-3 (fp16 autocast) | 4.5 / 1.2 ms (23 ms) | `model.23` (32): 1 |
| `yolo11n.pt` | 640 × 480 | 1.7e-8 / 2.8e-5 | 4.3 / 1.1 ms (20 ms) | `model.22` (32): 25 |
| `yolov8n.pt` | 640 × 480 | 4.5e-13 / 1.8e-5 | 3.4 / 1.0 ms (17 ms) | `model.21` (32): 25 |
| `yolov5nu.pt` | 640 × 480 | 9.1e-13 / 5.8e-5 | 3.8 / 1.0 ms (17 ms) | `model.23` (32): 25 |
| `yolov10n.pt` | 640 × 480 | 3e-9 / 4.4e-6 | 5.0 / 1.1 ms (18 ms) | `model.22` (32): 25 (through the one-to-one branch) |
| `yolo11n-obb.pt` | 1024 × 768 | 0 / 7.1e-5 | 5.3 / 3.0 ms (34 ms) | `model.16` (8): 15 |
| `yolo11n-seg.pt` | 640 × 480 | 1.2e-7 / 1.9e-4 | 5.1 / 1.5 ms (24 ms) | `model.22` (32): 25 |
| `yolo11n-pose.pt` | 640 × 480 | 2.1e-8 / 1.7e-5 | 5.6 / 1.2 ms (20 ms) | `model.22` (32): 25 |
| `rtdetr-l.pt` | 640 × 640 | 1.3e-6 / 1.6e-3 | 17 / 13 ms (206 ms) | `model.27` (32): 342 (CPU 400) |
| COCO Faster R-CNN | 1088 × 800 | 1.2e-7 / 3.6e-6 | 40 / 41 ms (604 ms) | `backbone.fpn.2` (16): 280 |
| COCO Mask R-CNN (boxes) | 1088 × 800 | 3.5e-9 / 2.7e-6 | 40 / 40 ms (618 ms) | `backbone.fpn.2` (16): 252 |
| COCO SSD300 | 300 × 300 | 8.9e-7 / 6.0e-5 | 8.6 / 5.9 ms (85 ms) | `backbone.2` (32): 9 |
| `yolov5s.onnx` | 640 × 640 | 0 (one image per run) | CPU 22 ms | — |
| `yolo11n.onnx` | 640 × 640 | 0 (one image per run) | CPU 16 ms | — |

The gradient-pass score equalled the inference score exactly for every row
with a Grad-CAM layer. Batch differences are of the class scores of image 0
(RT-DETR: of its queries' best scores, as a set — the queries are
re-selected per image). The OBB model's top detection sits on the stride-8
level, hence its layer. Rows of Faster R-CNN do not speed up with the batch
because its RoI stage is per image.

## 8. Not verified

* Models trained in YORU with these backends (the fixtures are COCO
  models); every run re-checks identity on its own first frame.
* YOLOv9, YOLOv3u, YOLOv6, YOLO12, YOLO-World / YOLOE, ultralytics' exported
  formats in the black box, ONNX Runtime's CUDA provider, an ONNX export of
  a torchvision model.
* YOLO26 run end to end (needs a newer ultralytics than YORU installs here).
