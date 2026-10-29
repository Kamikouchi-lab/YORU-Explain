# YORU-Explain

Perturbation and attribution analysis of [YORU](https://github.com/Kamikouchi-lab/YORU)'s
behaviour detections: *which parts of the image does the detector's score for
this detection depend on, and by how much?*

YORU-Explain is a sister application of YORU, like
[YORU-Tracker](https://github.com/Kamikouchi-lab/YORU-Tracker): its own window, its
own command, YORU unchanged. It loads a model of any of YORU's detection
backends — YOLOv5 (YORU v1 models), YOLOv8 / YOLO11 / YOLOv5u / YOLOv10
(detect, rotated-box OBB, segmentation, pose), RT-DETR, Faster R-CNN /
Mask R-CNN / SSD, ONNX exports — the way YORU runs it, checks that it detects
exactly what YORU detects, lets you pick one detection on a video frame, and
measures

* **Occlusion sensitivity** — the change in that detection's score when a
  region of the frame is replaced by a baseline (a *perturbation effect*);
* **Grad-CAM** — a gradient-weighted feature map of that score (an
  *attribution*), offered only for models whose gradient path was verified.

Both are about **one model's score**. Neither is evidence of what the animal
does or which body part matters biologically; a map that does not highlight
the wing is a measurement, not a failure. See [docs/VALIDATION.md](docs/VALIDATION.md).

| | YORU | YORU-Tracker | YORU-Explain |
|---|---|---|---|
| Purpose | detection, closed loop | identity, trajectories | why a detection scores as it does |
| Launch | `python -m yoru` | `python -m yoru_tracker` | `python -m yoru_explain` |
| GUI | DearPyGui | DearPyGui | DearPyGui |

## What is measured, exactly

The score is, by default, the confidence YORU reports for the detection
(`confidence`), computed from the **pre-NMS candidate** the detection came
from. What that number and that candidate are depends on the model family:

| YORU backend | Models | Candidate | `confidence` is | Modes | Grad-CAM |
|---|---|---|---|---|---|
| `yolov5` | YOLOv5 (YORU v1) | anchor slot | σ(objectness) × σ(class logit) | fixed, re-detection | yes |
| `ultralytics` | YOLOv8, YOLO11, YOLOv5u, YOLOv10; OBB, seg, pose | anchor point | σ(class logit) | fixed, re-detection | yes |
| `rtdetr` | RT-DETR | decoder query | σ(class logit) | re-detection | yes |
| `torchvision` | Faster / Mask R-CNN | region proposal | softmax over classes + background | re-detection | yes |
| `torchvision` | SSD300 | default box | softmax over classes + background | fixed, re-detection | yes |
| `onnx` | ONNX exports of YOLOv5 / YOLOv8 / YOLO11 | output row | as exported | fixed, re-detection | no |
| any | anything YORU loads, as a black box | — | YORU's reported conf | re-detection | no |

YORU-Explain recovers each detection's candidate (bit-exact) and follows it:

* **fixed-candidate** (default where offered): the same candidate is read on every
  perturbed frame. Its score is always defined; the run also records whether
  NMS would still keep it and how far its box moved. RT-DETR's queries and
  Faster R-CNN's proposals are chosen anew for every input, so those models
  are followed by re-detection only.
* **re-detection**: the detector runs end to end, NMS included, and its
  output is matched to the original box (same class, IoU ≥ 0.5). A failed match
  is a named state — `disappeared`, `unmatched`, `class_changed`,
  `ambiguous` — with **no score**, never a score of 0.

The best box of the class anywhere in the frame is never used: with two flies
of one class it would follow whichever is stronger.

Occlusion: `delta(R) = s_ref − s(frame with R replaced)`, signs kept; the map is
the mean delta of the patches covering each pixel, NaN where none was scored.
`s_ref` is taken at the run's batch size, because the detector's scores depend
on it (measured). The default baseline is a constant hot pink (#FF69B4); black,
grey, the frame or patch mean, a blur and seeded noise are the alternatives.
Other scores, where the family has them: `obj_x_cls`, `cls_prob`, `obj_prob`,
`cls_softmax`, `cls_logit`, `obj_logit`. Definitions and their limits:
[docs/VALIDATION.md](docs/VALIDATION.md); each backend, step by step:
[docs/BACKENDS.md](docs/BACKENDS.md). Rotated boxes (OBB models) are kept as
YORU reports them and matched by their exact polygon IoU.

## Install

YORU-Explain needs a YORU checkout next to it, as YORU-Tracker does:

```text
YORU-dev/
├── YORU/            # https://github.com/Kamikouchi-lab/YORU (2.0.0 Beta 4 or later)
└── YORU-Explain/    # this repository
```

```bash
cd YORU-Explain
uv sync                 # .venv with YORU installed editable from ../YORU (CUDA 12.8 torch on Windows/Linux)
uv run python -m yoru_explain
```

Without the sibling checkout, install YORU first and then this package
(`pip install "yoru @ git+https://github.com/Kamikouchi-lab/YORU"` followed by
`pip install .`); the built package only requires `yoru>=2.0.0b4,<3`, never a
local path.

Verified on Windows 11, Python 3.10.22, torch 2.11.0+cu128 (RTX 5060 Ti and
CPU), DearPyGui 1.11.1. Other platforms are untested.

## Use

**GUI** (`uv run python -m yoru_explain`):

1. *Project* — create a project folder, add videos/images (with YORU's
   `v_flip`/`h_flip` if they were used), add a model (type `auto` picks the
   backend YORU would; *black box* explains YORU's `detect()` only), *Load
   selected model*.
   Optionally attach a detections CSV — YORU's video-analysis `<video>.csv`, or
   YORU-Tracker's `*_tracks.csv` / `*_detections.csv` (a track's events keep
   its `track_id`; set the flips its `*_tracks.json` records): its boxes are
   shown in light blue, and *Events from CSV...* turns them into events in
   bulk (each is found again with the loaded model before it is analysed).
   YORU's real-time `*_detect.csv` is not read directly; track it with
   YORU-Tracker (with its `*_log.csv` beside it) and attach the tracks.
2. *Explore* — move to a frame (slider, `<` `>`, or the arrow keys); the
   loaded model's detections are drawn on the frame, whose axes are frame
   pixels (wheel zooms, drag pans, double-click fits). Click a box, *Add
   selected as event*. Ctrl/Shift-click selects several events; double-click
   one to go to its frame. Results of an event are shown as Original / Heatmap
   / Overlay / Side by side, with the colour scale and the map's value under
   the mouse.
3. *Analyze* — choose occlusion or Grad-CAM and the score, run the selected or
   all events. Progress, cancel and the log are on the right; nothing blocks
   the window.
4. *Statistics* — per-event values of each run (descriptive only).
5. *Settings* — device, logs, settings files, versions, capabilities.

**Python** (the GUI calls the same functions):

```python
from yoru_explain import api

adapter = api.load_model("best.pt", device="auto")      # any YORU model; backend as YORU picks it
project = api.Project.open_or_create("my_project")
src = project.add_source("video.mp4")
mdl = project.add_model("best.pt", adapter.name)
ev = api.add_event(project, adapter, mdl.model_id, src["source_id"], frame_index=6450, detection_index=0)
project.save()
outcome = api.run_analysis(project, adapter, mdl.model_id, [ev], api.AnalysisRequest("occlusion"))
print(outcome.status, outcome.identity_check)
```

Worked scripts: [examples/synthetic_occlusion.py](examples/synthetic_occlusion.py)
(no weights needed) and [examples/yolov5_video_event.py](examples/yolov5_video_event.py).
Settings are dataclasses with `to_dict`/`from_dict` (JSON); there is no command line.

## Output

```text
my_project/
├── project.json, events.csv
└── runs/<run_id>/
    ├── config.json        the request (method, parameters, score definition, seed)
    ├── manifest.json      provenance: versions, GPU, model sha256, input ids, identity check, counts
    ├── results.csv        one row per event
    ├── perturbations.csv  one row per occluded patch
    ├── maps/*.npz         maps and per-patch arrays + metadata
    ├── previews/*.png     original | heatmap | overlay (for viewing)
    └── logs/run.log
```

Every table carries `schema_version`; columns are in [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md).
Videos and weights are referenced by path and hash, never copied.

## Status

* Every YORU detection backend is supported ([docs/BACKENDS.md](docs/BACKENDS.md)):
  YOLOv5 (YORU v1 models), ultralytics (YOLOv8 / YOLO11 / YOLOv5u / YOLOv10;
  detect, OBB, segment, pose), RT-DETR, torchvision (Faster R-CNN, Mask R-CNN,
  SSD300), ONNX; any other backend YORU gains runs as a black box.
* Identity with YORU's own detector is verified bit-for-bit, on CUDA and CPU,
  on `yolov5s.pt`, a fly test model trained here with YORU's training script,
  and test models of every other family (COCO models from ultralytics and
  torchvision, ONNX exports). It is **not** verified on the YORU paper's models
  or on models trained in YORU with the other backends (not available here).
  Every run repeats the check on its first frame.
* 330 tests (synthetic known-answer tests, real-model integration for every
  backend, the GUI in process), plus a real-window test with
  `YORU_EXPLAIN_GUI_WINDOW=1`; see [docs/VALIDATION.md](docs/VALIDATION.md) for
  what they establish. The integration tests' models are fetched or built by
  `uv run --with onnx python -m tests.prepare_models` (into the git-ignored
  `.cache/models/`); without them those tests are skipped.
* Next: [docs/ROADMAP.md](docs/ROADMAP.md).

Logs: `~/.yoru/logs/yoru_explain.log`, beside YORU's (`YORU_HOME` moves both).

## Licence

AGPL-3.0-or-later (it runs YORU's AGPL code, including the bundled YOLOv5, in
process, and ultralytics (AGPL-3.0), torchvision (BSD-3-Clause) and ONNX Runtime
(MIT), with which YORU runs its other backends). DearPyGui is MIT.
