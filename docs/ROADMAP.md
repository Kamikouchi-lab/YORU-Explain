# Roadmap

v0.1 (this release): YOLOv5 adapter identical to YORU, fixed-candidate and
re-detection scoring, occlusion with controls, verified Grad-CAM, PyQt5 GUI,
reproducible run folders, 144 tests. What follows is concrete work, in order.

## v0.2 — validity of the explanations

1. **Deletion / insertion curves** per event: remove (or add back) patches in
   order of their occlusion Δ and record the score after each step;
   `curves.csv` (event, step, fraction, score, status) and the areas under
   them. Compare with random orderings (seeded, ≥ 20) as the null.
2. **Randomised controls**: score change for random patches of the same
   total area as the top-k patches; background-only occlusion; the same
   analysis on the frame with the target region inpainted.
3. **Sensitivity of the settings**: one command (Python API + GUI preset)
   that runs a grid of baselines × patch sizes × strides on the same events
   and reports rank correlation (Spearman) and IoU of top-k regions between
   maps; flag events whose top region moves with the setting.
4. **Grad-CAM sanity checks** (Adebayo et al.): cascading weight
   randomisation of a *copy* of the model (never the loaded one), map
   similarity vs. depth; Grad-CAM vs occlusion agreement per event.
5. **Group analysis with hierarchy**: events carry video, animal and day;
   summaries are computed per animal/video first and only then across them
   (mixed models or cluster bootstrap, units chosen explicitly). Frames are
   never treated as independent replicates; the GUI states the unit used.
6. **Failure statistics**: per run and per class, the share of `disappeared`,
   `class_changed`, `ambiguous`, `candidate_dropped` patches; plots of
   uncertainty (noop residual, batch noise floor from batch-1 vs batch-B reruns).
7. Resume of an interrupted run in place (the cache already skips finished
   events; add `resume_from=<run_id>`).

## v0.3 — regions with names

1. Import masks (PNG label images, polygons from JSON/CSV) per frame or per
   track; a mask editor in Explore (polygon tool) with undo.
2. Region occlusion: replace a whole named region (wing, head, thorax, legs,
   background) with each baseline; area-normalised effect, matched-area random
   control, effect per pixel.
3. Optional external segmentation (e.g. SAM, already used by YORU's labelling)
   behind an adapter, with the masks stored and hashed like any input.
4. Association with independently measured morphology/kinematics (only when
   such features are supplied); reported as association, not cause.

## v0.4 — YORU-Tracker and YOAKE

1. Read YORU-Tracker's track CSV and detections CSV (`frame_id` +
   `DETECTION_COLUMNS`); attach `track_id` to events; select "all frames of
   track 7 between t0 and t1" as one event set.
2. Per-track summaries over time (Δ map statistics per frame), with frames
   still not treated as independent.
3. YOAKE adapter: study `htrtdetr`'s inference API first (single frame vs
   track history; behaviour head inputs: actor OBB crop, context box,
   kinematics). Define what occlusion means for a temporal model
   (occlude in one frame vs across the history window), then temporal
   occlusion / attribution for genuinely temporal inputs only.
4. Further adapters: ultralytics (YOLOv8/11/26, RT-DETR; candidate = query /
   anchor-free cell, no objectness), torchvision (proposal identity),
   ONNX (black-box re-detection only).
5. RISE and other methods, with agreement between methods reported.

## Housekeeping

* CI on Windows like YORU-Tracker's (YORU checked out beside, `uv sync
  --locked`, pytest; download `yolov5s.pt` for the integration tests).
* If YORU documents the YOLOv5 entry points (docs/YORU_PR_PROPOSAL.md),
  move `adapters/_yoru.py` to the documented names and drop the internal list.
