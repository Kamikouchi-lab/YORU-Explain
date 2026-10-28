# Validation — what the numbers mean and what has been checked

Status at v0.1.0 (2026-10-16), extended 2026-10-17 to every YORU detection
backend ([BACKENDS.md](BACKENDS.md) holds the per-family details). Every
claim below is backed by a test in `tests/` or by a measurement whose script
and output are summarised here. Things not checked are listed as such in
section 5.

## 1. Definitions

### 1.1 The explained score

The default, `confidence`, is in every model family the number YORU reports
for the detection, computed for its pre-NMS candidate `k` and class `c` the
way the family computes it — so on the unchanged frame it equals YORU's
reported confidence bit for bit:

| Family | `confidence` | Its parts |
|---|---|---|
| YOLOv5 (`Detect`: logits `t_obj(k)`, `t_c(k)` per anchor slot) | `σ(t_obj) · σ(t_c)` = `obj_x_cls`, computed as `pred[k,5+c] · pred[k,4]`, the very product YOLOv5's NMS takes | `cls_prob` `σ(t_c)`, `obj_prob` `σ(t_obj)`, `cls_logit` `t_c`, `obj_logit` `t_obj` (forward hooks on the head's convolutions; fp16 on CUDA) |
| YOLOv8 / YOLO11 / YOLOv5u / YOLOv10 (anchor-free, no objectness) | `σ(t_c)` = `cls_prob` | `cls_logit` (hooks on the class convolutions) |
| RT-DETR (a decoder query) | `σ(t_c)` = `cls_prob` | `cls_logit` (hook on the last decoder layer's score head) |
| Faster / Mask R-CNN (a proposal), SSD (a default box) | `softmax(z)[c + 1]` over the classes and the background = `cls_softmax` | `cls_logit` `z[c + 1]` |
| ONNX export | as exported: `obj · cls` (YOLOv5 layout, = `obj_x_cls`) or `cls` (YOLOv8 layout, = `cls_prob`) | YOLOv5 layout: `cls_prob`, `obj_prob`; no logits |
| black box | the reported conf of the matched detection | — |

Each run's `config.json` writes the definitions of the scores its adapter
offers, and `results.csv` says which was used. Runs of v0.1 used `obj_x_cls`
for the same number YOLOv5's `confidence` is now.

### 1.2 Following the detection (identity modes)

* **fixed-candidate**: the target is the candidate `k` its detection came
  from (YOLOv5: by exact equality of the NMS output row with
  `xywh2xyxy(pred[k])`; ultralytics: NMS's own kept indices; SSD, Faster
  R-CNN, ONNX: exact equality of box and score — unique on every frame
  tested). On a perturbed frame the value is the same formula at the same
  `k`. Occlusion does not change the frame size, so the input transform and
  the grid — hence `k`'s meaning — are unchanged. Offered only where a
  candidate is a fixed slot (anchor slot, anchor point, default box, output
  row): RT-DETR's queries and Faster / Mask R-CNN's proposals are chosen anew
  for every input, and those adapters refuse this mode. Recorded diagnostics:
  `candidate_kept` (does NMS still keep `k` — under the target class, for
  multi-label models?), `iou` of `k`'s moved box with the original,
  `candidate_class`.
* **re-detection**: the detector runs end to end (NMS, the conf threshold) and
  the target is the unique detection of class `c` with IoU ≥ τ (default 0.5)
  with the original box — for rotated-box models the exact IoU of the two
  rotated rectangles. Otherwise the state is `ambiguous`, `class_changed`,
  `unmatched` or `disappeared` and **there is no score** (a vanished detection
  may have been suppressed under a neighbour, or sit just below the threshold;
  its score is unknown, not 0).

### 1.3 Occlusion sensitivity (perturbation effect)

For a rectangle `R` replaced by a baseline `b`:

    Δ(R) = s_ref − s(x with R := b)

* `s_ref` is `s(x)` evaluated **at the run's batch size**. At batch size 1
  (default) it equals the confidence YORU reports.
* Rectangles of `patch_size` tile the region of interest with `stride`; the
  last row/column is aligned to the region's edge so every pixel is covered.
* Map: `M(p) = mean{ Δ(R) : p ∈ R, R counted }`, NaN where no counted patch
  covers `p`. Counted = scored (`ok`, `matched`; `ambiguous` only on request).
* Signs are kept: `Δ > 0` — hiding R lowers the score; `Δ < 0` — raises it.
* Controls with every target: `noop` (the unchanged frame through the same
  batch pipeline; must give Δ = 0) and `full_region` (the whole region replaced).
* Baselines: `constant` (**default: hot pink #FF69B4**, BGR `(180, 105, 255)`, the
  lab's choice of 2026-10-17; black, YOLOv5 grey 114, any colour), `image_mean`,
  `patch_mean`, `blur` (σ), `noise` (seeded per patch). Each patch's real
  pixel change is recorded; a patch that changes no pixel cannot have an effect.

### 1.4 Grad-CAM (attribution)

For the fixed candidate's score `s` and a layer's feature map `A` (channels
`c`, cells `i, j`):

    α_c = mean_ij ∂s/∂A_c(i,j),   CAM = Σ_c α_c A_c,   Grad-CAM = max(CAM, 0)

`cam_feature` keeps the signed CAM; `cam` is the ReLU, upsampled bilinearly
(half-pixel centres: cell `(i, j)` ↦ input pixel `((j+½)s, (i+½)s)`), cropped
by the input padding and resized to the frame. Default layer — the feature
map the candidate's score is computed from: YOLOv5 the `Detect` input at the
candidate's own level (`model.17/20/23` for v5s-sized models); YOLOv8 /
YOLO11 / YOLOv10 the head's input at its level (YOLOv10: the gradient
through the one-to-one branch that produced the score); RT-DETR the decoder
input at the level of the query's encoder token; Faster / Mask R-CNN the FPN
level its proposal is pooled from; SSD the backbone level of its default
box. On one frame a query or a proposal is a well-defined candidate, so
Grad-CAM is offered for RT-DETR and Faster R-CNN although fixed-candidate
occlusion is not. ONNX models and the black box have no gradients.

A map is produced only if: the load-time gradient probe passed; the target
has its candidate; the gradient-pass score equals the inference score (≤
`score_tolerance`, default 1e-4; measured difference 0); the gradient is
finite and non-zero; the map is finite. Otherwise the result is `failed` with
a code and no map.

## 2. Constraints of these measures

* **One model, one number.** Both methods describe the score of one candidate
  of one model. They are not evidence about the animal.
* **Baseline dependence.** Occlusion edits the image into something the model
  never saw (black over a bright arena is extreme; the default hot pink is
  strongly red, and a fly's eyes are red, so a pink patch may add eye-like
  evidence as well as hide what it covers). The effect depends on the
  baseline, the patch size and the stride; compare settings before drawing
  conclusions (v0.2 adds systematic comparisons).
* **Fixed vs re-detection answer different questions.** Fixed-candidate asks
  "how does this anchor's score change"; re-detection asks "does the detector
  still report this box, and with what confidence". Measured on the fly
  events below: wherever NMS dropped the fixed candidate, re-detection matched
  a *different* candidate, and per-patch deltas of the two modes differed by
  up to 0.20.
* **Grad-CAM at YOLOv5's Detect input is single-cell.** The head is a 1×1
  convolution, so `∂s/∂A` is non-zero at exactly one cell (measured:
  `gradient_support = 1`; YOLOv8-style heads have two 3×3 blocks: 25 cells;
  RT-DETR and Faster R-CNN spread it over hundreds, [BACKENDS.md](BACKENDS.md) §7). The map shows where the channels *that* cell
  relies on are active anywhere — on another fly of the same class as much as
  on the target (measured: CAM mass split ≈ 50 / 50 between two solo flies).
  Earlier layers (`model.17`, `model.9`) spread the gradient (49–361 cells).
* **Batch size.** Scores depend on the batch size (cuDNN/MKL-DNN kernels):
  up to 2.4e-3 between batch 1 and 4 on CUDA autocast, 2e-6 on the CPU. The
  reference is taken at the run's batch size, so the noop delta stays exactly
  0, but per-patch deltas at batch 8 differ from batch 1 by up to 2.7e-3 on
  the fly model: effects smaller than that are not resolved at batch > 1.
  Batch 1 is exact.
* **Device.** CUDA (fp16 autocast, as YORU) and CPU (fp32) give slightly
  different scores (e.g. 0.857316 vs 0.857245). Runs record the device; the
  cache key includes it.
* **Letterbox conventions.** YOLOv5 pads by whole pixels but maps boxes back
  with fractional pads (`scale_boxes`); the two differ by ≤ ½ letterbox pixel.
  Boxes follow `scale_boxes` (as YORU reports them); maps follow the real pads.

## 3. What was verified

`uv run pytest` — 330 tests (one more opt-in), all passing on 2026-10-17
(Windows 11, Python 3.10.22, torch 2.11.0+cu128, ultralytics 8.3.222,
torchvision 0.26.0, onnxruntime 1.23.2, RTX 5060 Ti; about 2 min with every
model fixture present, 25 s without them).

| Area | How | Result |
|---|---|---|
| Inference identical to YORU | `tests/integration`: adapter vs `get_detector('yolov5', …).detect` on `zidane.jpg` (`yolov5s.pt`) and on frames 0/150/4350/6450/8700 of YORU's test video (fly fixture model), on CPU and CUDA; also with RGB input and with conf 0.85 (applied after NMS) | equal **bit for bit** in every case |
| Candidate = detection | readout of each detection's candidate vs its reported score; `σ(logits)` vs `pred` | bit-exact |
| Letterbox / boxes / maps | `tests/unit/test_geometry.py`: 8 frame shapes vs the bundled `letterbox`, `make_divisible`, `scale_boxes` | input images byte-identical; boxes within 1e-3 px; maps and feature cells land where they belong (incl. padded and non-integer-ratio cases) |
| Occlusion, known answer | `tests/unit/test_occlusion_synthetic.py` with the synthetic detector (each candidate reads exactly a 32×32 window) | patches outside the window: Δ **exactly 0** (pixels changed); over the square: Δ > 0; green inside the window: Δ < 0 and kept negative in the map; noop Δ = 0; a patch changing no pixel has Δ = 0; uncovered pixels NaN |
| Following the target | `tests/unit/test_target_identity.py`, two same-class squares | occluding A lowers A's score by up to 0.49 while "best box of the class" would report no change; fixed candidate reports NMS dropping it; re-detection gives `matched`, `disappeared`, `class_changed`, `unmatched`, `ambiguous` (τ = 0.3) as constructed; lost patches give no score and NaN map pixels |
| Grad-CAM, known answer | `tests/unit/test_gradcam_synthetic.py` | CAM half-max region = the red square in frame pixels, through letterbox padding; signed CAM negative on green; refuses without probe / candidate / with a mismatched score / non-finite gradient / unknown layer |
| Grad-CAM, real model | fly model, frame 150, layers auto/17/9 | gradient-pass score = inference score exactly; support 1 / >1 cells; weights unchanged (state-dict SHA-256), all modules in eval mode, no parameter gradients, detections unchanged afterwards |
| Reproducibility | save → `AnalysisRequest.from_dict(config.json)` → rerun | results, perturbations and NPZ arrays identical (CPU synthetic; CUDA fly model; and across processes: the example script reproduced the earlier run's numbers to the last digit) |
| Failure handling | changed frame hash, box not on the frame, missing video, cancellation, identity failure | `frame_mismatch`, `target_not_reproduced`, `missing_data`, `cancelled` rows; the run continues with the next event; no map, no number |
| Frames | `tests/unit/test_frames.py` | in-order numbering from 0; seeks verified by presentation time equal in-order frames; flips; non-ASCII paths |
| CPU only | CUDA reported missing | whole core runs on CPU; YOLOv5 autocast disabled |
| GUI | `tests/gui`, the DearPyGui window built in process (no viewport) | model loads on a worker; click selects; analysis runs while the loop keeps ticking (no tick > 0.5 s); cancel; views and colour scale; settings export/import; a failing callback is reported, not fatal; popups open one at a time; quitting during a task asks and waits; only the latest frame request is decoded. With `YORU_EXPLAIN_GUI_WINDOW=1`: `python -m yoru_explain` opens a real window and saves a screenshot |
| Boundary | `tests/unit/test_yoru_boundary.py` | YORU internals only in `adapters/_yoru.py`; YORU never imports yoru_explain; no argparse; the core imports without torch or a GUI toolkit; DearPyGui/tkinter only in `gui/`, no Qt |
| Every YORU backend | `tests/integration/test_yoru_backends.py`: 13 models of the ultralytics (YOLO11, YOLOv8, YOLOv5u, YOLOv10, YOLO11 OBB / seg / pose), RT-DETR, torchvision (Faster R-CNN, Mask R-CNN, SSD300) and ONNX (YOLOv5 and YOLO11 exports) backends, on `bus.jpg`, `zidane.jpg` and frame 150 of YORU's test video, CPU and CUDA | detections equal YORU's `get_detector(...).detect` **bit for bit**, rotated boxes included; every detection's candidate recovered and its readout = its confidence bit for bit; gradient-pass score = inference score; weights, eval mode and inference untouched by Grad-CAM; noop Δ = 0 and reference = YORU's confidence in every offered mode, the others refused; each image of a batch keeps its own detections; a project run with an OBB model writes its rotated boxes and is identity-checked exactly. Details: [BACKENDS.md](BACKENDS.md) |
| Which backend | `tests/unit/test_backends_registry.py` | `auto` resolves to the backend YORU's `get_detector("auto")` would pick (YOLOv5 by the pickle, whatever the name); every YORU backend has its own adapter; a request an adapter cannot honour is refused before a run folder is written |
| Transforms of each family | `tests/unit/test_family_transforms.py`: 7 frame shapes | input images byte-identical to ultralytics' `LetterBox` (rect, square, 1024) and YORU's ONNX letterbox; boxes as ultralytics' `scale_boxes`, RT-DETR's normalised boxes and torchvision's `resize_boxes`; SSD's overhanging grids map cells to their centres |
| Rotated boxes | `tests/unit/test_obb.py` | corners = `yoru.libs.obb` and ultralytics' `xywhr`; polygon IoU exact on known cases and within 0.01 of a raster count; re-detection matches rotated rods that upright boxes would confuse; events, tables and YORU CSVs keep them |

### Measured on YORU's test video (fly fixture model, CUDA, 32×32 patches, stride 16, context ×3, black)

| Event | Frame | Class / score | Patches | Largest Δ | Mean Δ in box / outside | Fixed: dropped by NMS | Re-detection: matched from another candidate | max \|Δ_fixed − Δ_redetect\| | Grad-CAM mass in box |
|---|---|---|---|---|---|---|---|---|---|
| ev-0001 | 150 | solo 0.9113 | 500 | 0.146 | 0.022 / 0.0006 | 11 | 11 | 0.113 | 0.50 |
| ev-0002 | 150 | solo 0.8841 | 375 | 0.275 | 0.033 / 0.0023 | 16 | 16 | 0.197 | 0.50 |
| ev-0003 | 6450 | copulation 0.8959 | 528 | 0.523 | 0.029 / 0.0019 | 2 | 3 | 0.074 | 0.999 |

Time per event: 2.7–3.7 s at batch 1, 1.4–1.9 s at batch 8 (CUDA).

The same events with the default **hot pink** baseline instead of black
(2026-10-17, otherwise identical settings):

| Event | Black: largest Δ / mean Δ in box | Hot pink: largest Δ / mean Δ in box |
|---|---|---|
| solo, frame 150 | 0.146 / 0.022 | 0.132 / 0.013 |
| copulation, frame 6450 | 0.523 / 0.029 | **0.055** / 0.0066 |

Hiding the copulating pair with pink patches lowers the score about ten
times less than hiding it with black, though the largest effects sit in the
same place; replacing the whole region kills the detection with either
colour (score < 1e-5). Effects of this size depend on the baseline, so a
figure should name it and, for a claim, show more than one.
These describe the fixture model's score; they are not findings about flies.
The fixture model was trained here on 900 frames of the same video it is
tested on.

### Independent review

A separate review of the code (2026-10-17) found seven defects; all are fixed
and each has a regression test:

| Defect | Fix |
|---|---|
| an error while writing an event's files escaped the run and left a manifest saying `completed` with missing rows | files first, tables last; a write error removes the event's files and becomes an `output_failed` row; any other unexpected error becomes an `internal_error` row and the run continues; an escaping interrupt leaves `status: failed` |
| an `ambiguous` re-detection wrote the competing detection's score as `perturbed_score` | the competing score is kept apart (`ambiguous_value`) and used only with `include_ambiguous` |
| a stored candidate index was accepted without checking the box, so an event from another model could be re-anchored to another animal | same candidate also needs IoU ≥ 0.9; events of another model are found by IoU only; Explore refuses detections of a model that is no longer loaded |
| the bundled NMS's per-call time limit could empty later images of a batch | NMS per image |
| a failed time-index pass raised inside the Explore view | reported as `failed` |
| re-reading a frame decoded the video again from the start | the last exactly read frame is reused |
| quitting during a task could leave a thread running at exit | cancel, then wait for the task (up to 60 s, window responsive) |

### Found while validating

* **Seeking is not exact on YORU's test video.** The file states 30.0032 fps
  while its frames are 33.333 ms apart; OpenCV's seek to frame 6301 decodes
  frame 6300, and reports position 6302. Analysis runs therefore decode in
  order and compare each frame's hash with the event's; the GUI seeks only
  through a presentation-time index (one pass, 0.8 s for 9002 frames) and
  verifies where each seek landed.
* **Windows: torch must be imported before PyQt5.** PyQt5-Qt5 5.15.2 loads
  its own, older `MSVCP140.dll`; a torch imported afterwards fails with
  WinError 1114 in `c10.dll`. Any process that imports both (YORU included)
  has the same constraint. YORU-Explain's GUI no longer uses PyQt5; DearPyGui
  and tkinter imported before torch were checked not to break it.

## 4. Reading results responsibly

* Keep **attribution** (Grad-CAM), **perturbation effect** (occlusion) and
  **biological interpretation** apart. The application never produces the
  third; a claim about body parts needs independently measured features
  (v0.3 region masks) and controls.
* Do not choose settings until a map "looks right". A map that does not
  highlight wings or legs is a measurement.
* Frames of a video are not independent samples. No statistical test is run;
  a group analysis (v0.2) will use videos, animals and days as units.
* Report the baseline, patch size, stride, identity mode, score type, batch
  size and device with any figure — they are all in `config.json`.

## 5. Not verified

* The YORU paper's models and other YORU v1 models: identity with YORU is
  shown for `yolov5s.pt` and the fixture model only; every run re-checks it
  on its own first frame.
* Models trained in YORU with its other backends: the fixtures are COCO
  models of ultralytics and torchvision and ONNX exports made here.
* Linux, macOS, Apple MPS; GPUs other than the RTX 5060 Ti.
* Variable-frame-rate and very long / high-resolution videos (seek
  verification is implemented; only the test video and synthetic clips were
  measured).
* Grad-CAM sanity checks beyond the above (weight randomisation, comparison
  with occlusion across many events) — v0.2.
* The effect of each baseline on real data beyond the single runs above.
* YOLOv9, YOLOv3u, YOLOv6, YOLO12, YOLO-World / YOLOE; ultralytics' exported
  formats (they run in the black box); ONNX Runtime's CUDA provider; YOLO26
  run end to end (ultralytics 8.3.222 runs it with NMS — BACKENDS.md §1).
* YOAKE.
