# Data schema — version 0.3.0

Every file YORU-Explain writes carries `schema_version` (`0.3.0`): JSON files as
a key, CSV tables as their **first column on every row**, NPZ archives in their
`meta_json`. A reader should refuse a different major version. Column order is
fixed by `src/yoru_explain/io/schema.py`.

**0.3.0** (2026-11-09) adds patches sized relative to the target
(`patch_fraction`, `stride_fraction` in the occlusion config;
`patch_height_px, patch_width_px, stride_y_px, stride_x_px, target_size_px` in
`results.csv`; `patch_size_px`, `stride_px` in the occlusion NPZ), the
experimental unit of a source (`unit_id` in `project.json` and in the run
manifest's `inputs`), and export folders (`exports/<run_id>/`, below). A 0.1.0
or 0.2.0 project or run reads unchanged: the new columns are just empty, and an
export of an older run takes the pixels used from its configuration.

**0.2.0** (2026-10-17) adds rotated boxes — `obb_cx, obb_cy, obb_w, obb_h,
obb_angle` in `events.csv` and `results.csv`, `matched_cx … matched_angle` in
`perturbations.csv`, `target_obb` in the NPZ archives — the models of every
YORU backend, the score types `confidence` and `cls_softmax`, and per-adapter
score definitions in `config.json`. A 0.1.0 project or run reads unchanged:
the new columns are just empty, and saving a 0.1.0 project writes them.

Conventions in all files:

* **Coordinates** — `frame_px_xyxy`: pixels of the frame the detector was given
  (after any `v_flip` / `h_flip`), origin top-left, x right, y down. Boxes are
  `(x1, y1, x2, y2)` exactly as YORU reports them — for a rotated-box (OBB)
  model the upright box around the rotated one. Rotated boxes are YORU's
  `(cx, cy, w, h, angle)`, `angle` in radians from +x towards +y
  (`yoru.libs.obb`); empty for upright-box models. Patches are integer
  half-open rectangles `[x0, x1) × [y0, y1)`. Maps are `H × W` arrays in the
  same pixels.
* **Missing numbers** are empty CSV cells / JSON `null` / NaN in NPZ — never 0.
  NaN and infinity are never written as text.
* **Floats** in CSV are written with `repr` (round-trip exact).
* **Booleans** in CSV are `true` / `false`.
* Text is UTF-8.

## project.json

| Key | Meaning |
|---|---|
| `schema_version`, `kind` (`yoru-explain-project`), `name`, `created` | |
| `sources[]` | `source_id` (`src-001`…), `path` (absolute), `kind` (`video`/`image`), `v_flip`, `h_flip`, `size` (bytes), `quick_id` + `quick_id_method` (`sha256(size, first 8 MiB, last 8 MiB)`), `width`, `height`, `fps` (as the container states it), `frame_count` (as stated), `fourcc`, `seek_check` (`null` or `{status, samples, mismatched}`), `detections_files[]` (`{path, layout, frames, detections}`), `added`, `unit_id` (optional: the experimental unit — animal, pair, cage — the source's events belong to; several sources may share one; absent: the source is its own unit) |
| `models[]` | `model_id` (`mdl-001`…), `path`, `sha256` (whole file; `synthetic` for the test detector), `adapter` (`yoru-yolov5`, `yoru-ultralytics`, `yoru-rtdetr`, `yoru-torchvision`, `yoru-onnx`, `yoru-detector` (black box), `toy-grid`), `options` (`conf_thresh`, `iou_thresh`; `rgb_input` for YOLOv5; `backend` for the black box) |
| `settings` | reserved |
| `removed` | optional: `{sources[], models[]}`, the sources and models taken out of the project, as they were, each with `removed` (the time). Their ids are never given again, so an event or a run that names one still means that file. Absent until something is removed; readers that do not know it can ignore it (no version change). |

Removing a source also removes the events picked on it from `events.csv`;
removing a model keeps its events (a run with another model finds each one
again by class and IoU, without its candidate index). New ids are one above
the highest in `sources` / `models`, `removed` and `events.csv`.

## events.csv

One row per detection chosen for explanation.

| Column | Meaning |
|---|---|
| `schema_version`, `event_id` (`ev-0001`…) | |
| `source_id`, `frame_index` | frame counted from 0 in decoding order (YORU's `frame`) |
| `timestamp_ms` | presentation time the decoder gave |
| `frame_sha1` | SHA-1 of the decoded, flipped pixels (shape and dtype included); empty for `unread` events |
| `frame_access` | how the frame was read: `sequential`, `seek_verified` (seek checked against a presentation-time index), `seek` (unverified), `image`, `unread` (imported from a detections CSV without decoding; no hash — the run hashes the frame it decodes) |
| `v_flip`, `h_flip`, `frame_width`, `frame_height` | |
| `model_id`, `detection_index` | model, and position in that frame's detection list |
| `class_id`, `class_name`, `x1`, `y1`, `x2`, `y2`, `score` | the detection as detected |
| `obb_cx`, `obb_cy`, `obb_w`, `obb_h`, `obb_angle` | its rotated box (OBB models); empty otherwise |
| `candidate_index`, `candidate_locator` | the candidate (row of the adapter's candidate table), and JSON of where it sits: YOLOv5 `{level, stride, anchor, grid_y, grid_x}`; YOLOv8/YOLO11 `{level, stride, grid_y, grid_x}`; SSD `{level, step, grid_y, grid_x, box}`; RT-DETR `{query, token, level, grid_y, grid_x}`; Faster R-CNN `{proposal, fpn_level}`; empty for an imported detection or a black-box model |
| `target_class` | class whose score is explained (the detected class) |
| `track_id` | optional (YORU's `tracking_id` from an imported CSV) |
| `origin` | `adapter` or `csv_import` |
| `note`, `created` | |

## runs/&lt;run_id&gt;/config.json — what was asked

`run_id` is `YYYYMMDD-HHMMSS_<method>_<6 hex>`.

| Key | Meaning |
|---|---|
| `schema_version`, `kind` (`yoru-explain-run-config`), `run_id`, `created` | |
| `request` | the `AnalysisRequest`: `method`, `method_config`, `verify_identity`, `use_cache`, `write_previews`. `AnalysisRequest.from_dict(config["request"])` re-creates it. |
| `method_config` | occlusion: `method`, `method_kind` (`perturbation_effect`), `method_version`, `patch_size [h, w]`, `stride [y, x]`, and — only when used — `patch_fraction` (square patches of `round(fraction × target size)` pixels per event, the target size being `sqrt(w × h)` of its box, or of its rotated box for OBB models; `patch_size` and `stride` are then not used) and `stride_fraction` (`null`: half the patch), `baseline {kind, value [B,G,R], sigma}` (default `constant` `[180, 105, 255]`, hot pink #FF69B4; older runs keep the colour they recorded), `region` (`context`/`box`/`full`), `context_scale`, `score {score_type, identity_mode, iou_threshold}`, `batch_size`, `seed`, `include_ambiguous`. Grad-CAM: `method`, `method_kind` (`attribution`), `method_version`, `layer`, `score_type`, `identity_mode` (always `fixed_candidate`), `precision`, `score_tolerance`. |
| `score_types`, `identity_modes` | the definitions, in words, of the score types the run's adapter offers — as that model family defines them (`confidence` is YOLOv5's `obj_x_cls`, YOLOv8's sigmoid, torchvision's softmax, ...) — and of the identity modes |
| `model_id`, `model_options`, `adapter`, `adapter_version`, `events`, `frame_access`, `seed` | |

## runs/&lt;run_id&gt;/manifest.json — what it ran with, and how it ended

| Key | Meaning |
|---|---|
| `schema_version`, `kind` (`yoru-explain-run-manifest`), `run_id`, `created`, `finished`, `elapsed_s` | |
| `status` | `running` (while it runs — or the process was killed), `completed` (every event has a row; see `events` for how many failed), `cancelled`, `failed` (identity check failed, or the run itself was interrupted by an error) |
| `method`, `method_kind` | |
| `project` | `{name, root}` |
| `software` | versions of yoru_explain, python, numpy, opencv, torch, torchvision, yoru, dearpygui (pyqt5 in runs made with the PyQt5 GUI), and the git commits of the YORU and YORU-Explain checkouts |
| `hardware` | platform, CUDA/cuDNN versions, `cudnn_benchmark`, `cudnn_deterministic`, GPU names |
| `adapter` | `{name, version, capabilities}`; capabilities hold the `family`, the `candidate_kind`, the `identity_modes` and `score_types` it offers with their `score_definitions`, `reported_score` (the family's name of the reported confidence), `multi_label`, the measured `deterministic`, `batch_invariant`, `batch_max_abs_diff`, the gradient probe's verdict, `notes` and `warnings` |
| `model` | `model_id` + the adapter's `model_info()`: weights path, `sha256`, the YORU `backend`, class names, input size, thresholds (`conf_thresh`, `nms_conf`, `iou_thresh`, `max_det`), channel order, device, precision, and the family's own settings (YOLOv5: strides, bundled YOLOv5 path; ultralytics: task, head, `imgsz`, `end2end`; torchvision: `model_type`, transform, NMS settings, `iou_thresh_used: false`; ONNX: layout, providers, class-name source; black box: `black_box: true`, detector class), package versions, YORU version |
| `identity_check` | adapter vs `yoru.libs.plugins.get_detector(<backend>, ...).detect` on the first frame (rotated boxes included): `status` (`passed`/`failed`/`not_applicable` — the black box *is* YORU's detector), `n_reference`, `n_adapter`, `max_abs_diff`, `exact`, `frame {source_id, frame_index, sha1}` |
| `inputs[]` | the sources used, as in project.json (with their `unit_id` at the time), plus `verify {status: ok/changed/missing}` |
| `events` | `requested`, and counts per result status (`ok`, `failed`, `cancelled`), `cached` |
| `error_codes` | count per error code |
| `coordinate_system`, `coordinate_system_description`, `baseline_note`, `interpretation`, `files` | |

## runs/&lt;run_id&gt;/results.csv — one row per event

Identification:
`schema_version, run_id, event_id, video_id (= source_id), source_path, frame_index, timestamp_ms, frame_sha1, detection_id (= detection_index), candidate_index, track_id, behavior_class (= class_name), class_id, target_class, x1, y1, x2, y2, obb_cx, obb_cy, obb_w, obb_h, obb_angle`

Reproduction of the stored detection on this run's inference:
`stored_score` (score stored with the event), `reproduction` — how the stored detection was found again on this run's inference: `same_candidate` (same candidate, class, and IoU ≥ 0.9), `iou` (no stored candidate; class and IoU ≥ 0.9), `iou_candidate_changed` (class and IoU ≥ 0.9, from another candidate), `iou_other_model` (the event was made with another model, so its candidate index was not used) — and `reproduction_iou`.

What was measured:

| Column | Occlusion | Grad-CAM |
|---|---|---|
| `method`, `method_kind`, `score_type`, `identity_mode`, `baseline` | `occlusion`, `perturbation_effect`, …, baseline description | `gradcam`, `attribution`, …, `fixed_candidate`, empty |
| `patch_height_px`, `patch_width_px`, `stride_y_px`, `stride_x_px`, `target_size_px` | the patches used for this event, frame pixels (the configured ones, or those a `patch_fraction` gave), and `sqrt(w × h)` of the target (its rotated box for OBB models) | empty |
| `original_score` | target's score on the unchanged frame at batch 1 (= YORU's confidence for `confidence`) | score in the inference pass |
| `reference_score` | the same at the run's batch size: the deltas' reference | same as original |
| `perturbed_score`, `delta_score`, `max_delta_patch_id` | score and delta of the patch with the largest delta | empty |
| `delta_min`, `delta_mean` | over counted patches | empty |
| `in_box_mean_delta`, `out_box_mean_delta` | mean of the map over defined pixels inside / outside the target box | empty |
| `noop_delta` | delta of the unchanged frame run through the batch pipeline (expected exactly 0) | empty |
| `full_region_score`, `full_region_status` | the whole region replaced by the baseline | empty |
| `n_perturbations`, `n_counted` | patches, patches in the map | empty |
| `n_ok`, `n_matched`, `n_ambiguous`, `n_class_changed`, `n_unmatched`, `n_disappeared`, `n_non_finite`, `n_error` | patch counts per status | empty |
| `n_candidate_dropped` | fixed-candidate: patches on which NMS would no longer keep the candidate (multi-label models: under the target class) | empty |
| `matching_status` | how the target scored on the unchanged frame (`ok` / `matched`, or the failure) | `ok` |
| `gradcam_layer`, `gradcam_stride`, `gradient_support`, `gradient_score`, `cam_max`, `cam_in_box_fraction` | empty | layer, its stride, cells with non-zero gradient, score in the gradient pass, CAM maximum, CAM mass inside the box / total |

Outcome: `status` (`ok`, `failed`, `cancelled`), `error_code`, `message` (the score definition in words, or the error), `map_file`, `preview_file` (relative to the run folder), `cache_key`, `cached_from` (run the result was copied from), `elapsed_s`.

## runs/&lt;run_id&gt;/perturbations.csv — one row per evaluated input (occlusion)

| Column | Meaning |
|---|---|
| `schema_version`, `run_id`, `event_id`, `patch_id` | patches are numbered row by row from 0; the controls have `-1` (`noop`) and `n` (`full_region`) |
| `kind` | `patch`, `noop`, `full_region` |
| `x0`, `y0`, `x1`, `y1` | the replaced rectangle (`noop`: 0,0,0,0) |
| `pixel_change` | mean absolute change of the replaced pixels, 0–255 |
| `status` | see below |
| `perturbed_score` | the score the delta was taken from — the target's score (`ok`, `matched`), or with `include_ambiguous` the best competing match's; empty for every row that is not counted (an `ambiguous` row's competing score is not the target's) |
| `delta_score` | `reference_score − perturbed_score` for counted rows; empty otherwise |
| `counted` | whether the row entered the map |
| `iou` | re-detection: IoU of the matched detection with the original box; fixed-candidate: IoU of the candidate's (moved) box with the original |
| `candidate_index`, `candidate_kept`, `candidate_class` | candidate the score was read from; fixed-candidate: whether NMS keeps it, and its current argmax class |
| `matched_x1 … matched_y2`, `matched_cx … matched_angle`, `matched_score`, `matched_class` | re-detection: the matched (or best-overlapping) detection, with its rotated box for OBB models |
| `n_overlapping` | re-detection: same-class detections with IoU ≥ threshold |
| `error_code` | |

### Statuses

| Status | Mode | Scored? | Meaning |
|---|---|---|---|
| `ok` | fixed-candidate | yes | the candidate's score was read |
| `matched` | re-detection | yes | exactly one same-class detection with IoU ≥ threshold |
| `ambiguous` | re-detection | reported, counted only with `include_ambiguous` | two or more such detections |
| `class_changed` | re-detection | no | only other-class detections overlap enough |
| `unmatched` | re-detection | no | something overlaps the box, below the threshold |
| `disappeared` | re-detection | no | nothing overlaps the box |
| `non_finite` | both | no | NaN / inf |
| `error` | both | no | see `error_code` |

### Error codes

`model_unsupported`, `model_load_failed`, `gradients_unavailable`,
`score_unavailable`, `score_mismatch`, `target_not_reproduced`,
`frame_mismatch`, `preprocess_mismatch`, `identity_check_failed`,
`non_finite`, `cuda_oom`, `missing_data`, `invalid_config`, `output_failed`
(computed, but the files could not be written — none are left behind),
`cancelled`, `internal_error` (`src/yoru_explain/core/errors.py`).

**results.csv is the record of what finished.** An event's files and its
perturbations.csv rows are written before its results.csv row; rows of an
event whose result is not `ok` are not to be used.

## runs/&lt;run_id&gt;/maps/&lt;event_id&gt;__occlusion.npz

`numpy.load(path, allow_pickle=False)`:

| Array | dtype, shape | Meaning |
|---|---|---|
| `delta_map` | float32 `H×W` | mean delta of the counted patches covering each pixel; NaN where none |
| `coverage` | uint16 `H×W` | counted patches covering the pixel |
| `coverage_all` | uint16 `H×W` | all patches covering the pixel |
| `lost_fraction` | float32 `H×W` | `1 − coverage / coverage_all`; NaN outside the region |
| `patch_id`, `patch_xyxy` | int32 `P`, `P×4` | the patches |
| `patch_score`, `patch_delta` | float64 `P` | the score used for the delta, and the delta; NaN when not counted |
| `patch_counted` | bool `P` | |
| `patch_status` | int8 `P` | code from `meta.status_codes` (`ok` 0, `matched` 1, `ambiguous` 2, `class_changed` 3, `unmatched` 4, `disappeared` 5, `non_finite` 6, `error` 7) |
| `patch_pixel_change` | float32 `P` | |
| `target_box` | float64 `4` | |
| `target_obb` | float64 `5` | the rotated box; NaN for upright-box models |
| `roi` | int32 `4` | the region the patches tile |
| `frame_shape` | int32 `2` | `H, W` |
| `patch_size_px`, `stride_px` | int32 `2` | `[h, w]` and `[y, x]` of the patches used (0.3.0) |
| `meta_json` | 0-d str | JSON: `schema_version`, `method`, `method_kind`, `map`, `map_definition`, coordinate system, `status_codes`, `reference_score`, `reference_score_batch`, `target_size_px`, `config`, `target`, `run_id`, `score_description`, `cache_key` |

## runs/&lt;run_id&gt;/maps/&lt;event_id&gt;__gradcam.npz

| Array | Meaning |
|---|---|
| `cam` | float32 `H×W`: `max(Σ_c α_c A_c, 0)` mapped to frame pixels |
| `cam_feature` | float32 `h×w`: the signed `Σ_c α_c A_c` on the layer's grid, before ReLU |
| `target_box`, `target_obb`, `frame_shape` | |
| `meta_json` | as above, plus `layer`, `layer_stride`, `letterbox` (the full frame ↔ input transform: `kind` — `yolov5_letterbox`, `ultralytics_letterbox`, `ultralytics_stretch`, `torchvision_resize`, `yoru_onnx_letterbox` — shapes, ratio, pads, the box mapping `frame = (input − box_offset) / box_scale` with `clip_boxes`; `box_gain`, `box_pad` as v0.1 named them), `gradient_score`, `gradient_support` |

## Previews and logs

`previews/<event_id>__<method>.png`: original | heatmap | overlay with the
colour scale and the run's settings printed under it. For viewing only; the
numbers are in the NPZ and CSV files. `logs/run.log`: everything logged under
`yoru_explain` during the run, tracebacks included.

## exports/&lt;run_id&gt;/ — a run for the experimenter's own analysis

Written by `api.export_run` (GUI: *4 Results → Export for analysis*) from a
run folder and the project; nothing is recomputed. The folder is built beside
its place (`<run_id>.partial`) and swapped in when complete, so an
interrupted or cancelled export leaves the previous one. Everything in it
describes the model's score; class names are the model's predictions; frames
are not independent samples — compare experimental units (`unit_id`).

**Two grids** appear below. The *box grid* of an event is `n × n` cells around
its target's upright box: cell `(r, c)` holds the map's value at the frame
pixel containing `(cx + u[c]·w, cy + v[r]·h)` (box centre `cx, cy`, width `w`,
height `h`), with `u = v = −E + (k + ½)·2E/n` (`E` the extent; the box edges are
at ±0.5). The *aligned grid* (`api.align_maps`) is turned to an orientation the
experimenter gives: cell `(r, c)` samples `centre + L·(a[r]·f + b[c]·s)` with
`f = (cos θ, sin θ)` the forward direction, `s = (−sin θ, cos θ)` the animal's
right as seen from above, `L` the given length, `a` decreasing down the rows
(row 0 is the most forward) and `b` increasing along the columns. Both sample
the nearest pixel (no interpolation); a cell off the map or on an undefined
pixel is NaN. YORU-Explain never estimates which way an animal faces.

| File | Contents |
|---|---|
| `export.json` | `schema_version`, `kind` (`yoru-explain-export`), `created`, `software`, `run_id`, `run_dir`, `project`, `method`, `method_kind`, `method_config`, `map` (`delta` / `cam`), `map_definition`, `options` (`grid`, `extent`, `crops`, `gallery`, `gallery_scale`), `grid {cells, extent, definition}`, `units {source_id: {unit_id, assigned}}`, `counts`, `files`, `interpretation` |
| `README.txt` | how to read the folder, written with it |
| `events.csv` | one row per event of the run, in its order (columns below) |
| `patches.csv` | occlusion: one row per evaluated patch and control of the finished events — `event_id, unit_id, class_name, patch_id, kind, x0, y0, x1, y1` (frame pixels, half-open), `u0, v0, u1, v1` (box units: `u = (x − cx) / w`, `v = (y − cy) / h`; empty for `noop`), `status, counted, perturbed_score, delta_score, candidate_kept, pixel_change` as in perturbations.csv |
| `summary.csv` | `level` (`unit`: per unit and predicted class, over its finished events; `class`: per class over the **unit means**, each unit weighing the same), `class_name, unit_id, n_units, n_events, n_ok, n_failed`, and `<metric>_mean`, `<metric>_median` for `reference_score, delta_score, in_box_mean_delta, out_box_mean_delta, in_box_positive_share, cam_in_box_fraction`. Descriptive: no interval, no test |
| `box_maps.npz` | `event_id, unit_id, class_name` (str `N`), `map` (float32 `N×n×n`, the box grid), `u`, `v` (float64 `n`, the cell centres), `image_rgb` (uint8 `N×n×n×3`, the frame on the same grid) and `image_valid` (bool `N×n×n`) when crops were exported, `meta_json` (`map`, `map_definition`, `grid`, `extent`, `grid_definition`) |
| `mean_maps.npz` | `class_names` (str `C`), `mean_by_class` (float32 `C×n×n`: mean of the unit means), `n_units_by_class` (int32 `C×n×n`: units defining each cell), `unit_class`, `unit_id` (str `K`), `mean_by_unit` (`K×n×n`), `n_events_by_unit` (`K×n×n`), `u`, `v`, `meta_json` |
| `maps/<event_id>.npz` | `map` (float32: the run's map cropped to the analysed region — the patches' region for occlusion, the box grid's for Grad-CAM; frame pixel `(x, y)` is `map[y − offset[1], x − offset[0]]`), `coverage` (occlusion), `offset` (int32 `[x0, y0]`), `frame_shape`, `box`, `obb` (NaN for upright boxes), `meta_json` |
| `crops/<event_id>.png` | the unchanged frame over the same region (decoded in order and checked against the frame hash the run recorded) |
| `gallery/` | `gallery.csv` (`event_id, unit_id, class_name, frame_index, status, file, scale_group, vmin, vmax`), `<class>/<unit>__<event>__f<frame>.png` (frame \| map \| overlay, with the colour scale), `sheets/<class>__pNN.png` and `sheets/<class>__<unit>__pNN.png` (contact sheets; events without a map are grey tiles with their status). One colour scale per predicted class (`class`, default), per run (`run`) or per event (`event`); `api.render_gallery(export_dir, scale=...)` draws it again with another |

`events.csv` columns: `schema_version, run_id, event_id, unit_id`,
`unit_assigned` (`false`: the video is its own unit), `source_id, source_path,
fps, frame_index, timestamp_ms, frame_sha1, track_id, class_id, class_name`
(predicted), `target_class, x1 … y2, obb_cx … obb_angle, target_size_px,
method, method_kind, score_type, identity_mode, baseline, patch_height_px,
patch_width_px, stride_y_px, stride_x_px`, the run's values
`reference_score, delta_score` (largest patch delta), `delta_min, delta_mean,
in_box_mean_delta, out_box_mean_delta`, `in_box_positive_share` (of the
positive part of the map over its defined pixels, the share inside the box —
by pixels, not an additive attribution), `cam_max, cam_in_box_fraction,
n_perturbations, n_counted, n_disappeared, n_class_changed,
n_candidate_dropped, noop_delta`, `status, error_code, message`,
`stack_index` (row in `box_maps.npz`; empty without a map), `map_file`,
`crop_x0, crop_y0, crop_x1, crop_y1` (the region of `maps/` and `crops/`),
`crop_file`, `crop_status` (`ok`, `frame_unavailable`, `frame_mismatch`),
`run_map_file` (the NPZ in the run folder).

### Inputs the experimenter writes

*Units* (`Project.assign_units_from_csv`, GUI *1 Project → Units from table*):
a CSV with the columns `video` (a source id, the video's full path, or its
file name when only one source has it) and `unit_id`; other columns are
ignored and reported.

*Orientations* (`api.align_maps(export_dir, "orientations.csv")`): one row per
event, `event_id, cx, cy, angle_deg, length` and optionally `directed`
(`true`/`false`, default `true`). `cx, cy` and `length` in frame pixels;
`angle_deg` the forward direction in degrees from +x towards +y (0 = right, 90
= down on screen). `directed=false` (an axis with no known head end) averages
both directions. An empty or non-numeric cell leaves that event out with the
status `invalid_orientation`; an event listed twice, or not in the export, is
an error. `"obb"` instead of a file uses the model's rotated boxes (longer
side, undirected).

`align_maps(..., out_dir=folder)` writes `aligned.npz` (`event_id, unit_id,
class_name, map` (`N×n×n`), `forward`, `right` (the cell coordinates `a`, `b`),
`image_rgb`, `image_valid`, `meta_json`), `aligned_means.npz` (as
`mean_maps.npz`) and `aligned.csv` (`event_id, unit_id, class_name, status`
(`ok`, `no_orientation`, `invalid_orientation`, `not_analysed`),
`stack_index, cx, cy, angle_deg, length, directed`).
