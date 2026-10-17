# YORU-Explain — Claude Code project guide

YORU-Explain explains YORU's behaviour detections: occlusion sensitivity
(perturbation effect) and verified Grad-CAM (attribution) for one chosen
detection, for every YORU detection backend (YOLOv5, ultralytics incl. OBB /
seg / pose / YOLOv10, RT-DETR, torchvision, ONNX; others as a black box;
`docs/BACKENDS.md`). Sister application of YORU (`../YORU`) and YORU-Tracker
(`../YORU-Tracker`). Windows, Python 3.10, DearPyGui (as YORU and
YORU-Tracker). Start with `README.md`;
the definitions are in `docs/VALIDATION.md`, the files in `docs/DATA_SCHEMA.md`.

## Logs — check these first when something fails

`~/.yoru/logs/yoru_explain.log` (beside YORU's `yoru.log`; moved by
`YORU_HOME`), and each run's `runs/<run_id>/logs/run.log`, with tracebacks.

## The rules that make the numbers trustworthy

- Explain **the** detection: its pre-NMS candidate (fixed-candidate) or a
  unique class+IoU match (re-detection). Never the best box of the class.
- A target that is lost, ambiguous or failed has **no score** — empty cell /
  NaN with a status and error code. Never 0.
- Occlusion deltas are taken against a reference scored at the run's batch
  size (scores depend on batch size). Keep the `noop` control.
- Grad-CAM only when the adapter's gradient probe passed and the gradient-pass
  score equals the inference score; otherwise fail with a code, no map.
- Never put a model in training mode, never change weights; gradients are
  taken w.r.t. activations from a copy of the input.
- Every adapter must stay identical to its YORU backend's `detect()`
  (`tests/integration` checks each bit for bit, CPU and CUDA). Change
  preprocessing, NMS or thresholds only together with those tests. Call the
  family's own functions (ultralytics' NMS / `scale_boxes`, torchvision's
  pieces) rather than copies; only YORU's private ONNX steps are repeated.
- Fixed-candidate mode only where a candidate is a fixed slot: RT-DETR queries
  and Faster R-CNN proposals are re-detection only (`check_request` refuses).
- Frames are numbered as YORU's analysis numbers them (decoding order from 0);
  runs read in order and compare frame hashes; seeks are verified by
  presentation time.
- Do not tune settings until a map looks plausible; measurements are reported
  as they come out. Attribution, perturbation effect and biological
  interpretation stay separate in UI and docs.

## The boundary with YORU

- YORU changes only through pull requests to Kamikouchi-lab/YORU that the
  user asked for (#43: documenting the YOLOv5 entry points, see
  `docs/YORU_PR_PROPOSAL.md`). Never edit ../YORU on its own.
- YORU names outside YORU's `docs/external_api.md` are used only in
  `src/yoru_explain/adapters/_yoru.py` (YOLOv5 entry points, `torch_device`,
  `resolve_device`, `_auto_detect_backend`); `tests/unit/test_yoru_boundary.py`
  pins the list. YORU never imports `yoru_explain`.
- Found 2026-10-17, not ours to fix: with ultralytics 8.3.222, YORU runs
  YOLO26 checkpoints with NMS instead of their end-to-end head (wrong boxes);
  the ultralytics adapter warns (`capabilities.warnings`).
- YORU-Explain does not import `yoru_tracker`.

## Code layout and conventions

- Core (`core/`, `analysis/`, `io/`, `visualization/`, `methods/occlusion.py`)
  imports no torch/GUI toolkit/YORU at module level and runs headless.
- The GUI only displays and dispatches to `core/pipeline.py` / `api.py`;
  heavy work runs on `gui/worker.py` threads, which never touch DearPyGui:
  the render loop (`gui/main_window.run`, manual callback management) polls
  them each frame and delivers results through `gui/state.Signal`.
- The window shell and base theme are YORU's documented GUI primitives
  (`GuiSession`, `apply_default_theme`); the screens are our own. No PyQt5.
- DearPyGui's image plot has a downward y axis, so `FrameView` flips the
  texture's UVs; boxes and clicks are in frame pixels.
- No `argparse` anywhere. Settings are dataclasses with strict
  `to_dict`/`from_dict` (unknown keys are errors).
- Every written table/JSON/NPZ carries `schema_version`; change
  `io/schema.py` and `docs/DATA_SCHEMA.md` together.

## Environment and tests

`uv sync` creates `.venv` with YORU editable from `../YORU` (torch cu128).
Run with `uv run ...` or `.venv/Scripts/python.exe`; bare `python` on this
machine is a broken Windows Store stub. `uv run pytest` runs everything
(unit, integration, the GUI in process without a viewport; ~2 min with all
model fixtures, ~25 s without); `YORU_EXPLAIN_GUI_WINDOW=1` adds a
real-window test. Integration tests need `.cache/models/yolov5s.pt` and
`.cache/models/fly_copulation_yolov5s.pt` (or `YORU_EXPLAIN_YOLOV5_WEIGHTS` /
`YORU_EXPLAIN_FLY_WEIGHTS`), the other backends' fixtures
(`uv run --with onnx python -m tests.prepare_models` downloads / builds them)
and YORU's test video; they skip without them. `.cache/` is git-ignored: the fly model is
a test fixture trained with YORU's `train_yolov5.py` (docs/INITIAL_AUDIT.md §5).
