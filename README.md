# YORU-Explain

Perturbation and attribution analysis of [YORU](https://github.com/Kamikouchi-lab/YORU)'s
behaviour detections: *which parts of the image does the detector's score for
this detection depend on, and by how much?*

YORU-Explain is a sister application of YORU, like
[YORU-Tracker](https://github.com/HMYamano/YORU-Tracker): its own window, its
own command, YORU unchanged. It loads a YOLOv5 model through YORU's bundled
YOLOv5 code, checks that it detects exactly what YORU detects, lets you pick
one detection on a video frame, and measures

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

For a YOLOv5 detection the score is, by default, YOLOv5's own confidence:
`sigmoid(objectness) × sigmoid(class logit)` of the **pre-NMS candidate** (anchor
slot) the detection came from. YORU-Explain recovers that candidate for every
detection (bit-exact) and follows it:

* **fixed-candidate** (default): the same candidate is read on every
  perturbed frame. Its score is always defined; the run also records whether
  NMS would still keep it and how far its box moved.
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
grey, the frame or patch mean, a blur and seeded noise are the alternatives. Other scores: `cls_prob`, `obj_prob`, `cls_logit`,
`obj_logit`. Definitions and their limits: [docs/VALIDATION.md](docs/VALIDATION.md).

## Install

YORU-Explain needs a YORU checkout next to it, as YORU-Tracker does:

```text
YORU-dev/
├── YORU/            # https://github.com/Kamikouchi-lab/YORU (2.0.0b3 or later, with yoru/libs/yolov5)
└── YORU-Explain/    # this repository
```

```bash
cd YORU-Explain
uv sync                 # .venv with YORU installed editable from ../YORU (CUDA 12.8 torch on Windows/Linux)
uv run python -m yoru_explain
```

Without the sibling checkout, install YORU first and then this package
(`pip install "yoru @ git+https://github.com/Kamikouchi-lab/YORU"` followed by
`pip install .`); the built package only requires `yoru>=2.0.0b3,<3`, never a
local path.

Verified on Windows 11, Python 3.10.22, torch 2.11.0+cu128 (RTX 5060 Ti and
CPU), DearPyGui 1.11.1. Other platforms are untested.

## Use

**GUI** (`uv run python -m yoru_explain`):

1. *Project* — create a project folder, add videos/images (with YORU's
   `v_flip`/`h_flip` if they were used), add a model, *Load selected model*.
   Optionally attach a YORU detections CSV: its boxes are shown in light blue, and
   *Events from CSV...* turns them into events in bulk (each is found again
   with the loaded model before it is analysed).
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

adapter = api.load_adapter("yoru-yolov5", "best.pt", device="auto")
project = api.Project.open_or_create("my_project")
src = project.add_source("video.mp4")
mdl = project.add_model("best.pt", "yoru-yolov5")
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

## Status (v0.1)

* YOLOv5 detection checkpoints (YORU v1 models included) are supported; other
  YORU backends are not yet ([docs/INITIAL_AUDIT.md](docs/INITIAL_AUDIT.md)).
* Identity with YORU's own detector is verified bit-for-bit on `yolov5s.pt`
  and on a fly test model trained here with YORU's training script, on CUDA
  and CPU. It is **not** verified on the YORU paper's models (not available
  here). Every run repeats the check on its first frame.
* 152 tests (synthetic known-answer tests, real-model integration, the GUI
  in process), plus a real-window test with `YORU_EXPLAIN_GUI_WINDOW=1`; see
  [docs/VALIDATION.md](docs/VALIDATION.md) for what they establish.
* Next: [docs/ROADMAP.md](docs/ROADMAP.md).

Logs: `~/.yoru/logs/yoru_explain.log`, beside YORU's (`YORU_HOME` moves both).

## Licence

AGPL-3.0-or-later (it runs YORU's AGPL code, including the bundled YOLOv5, in
process). DearPyGui is MIT.
