# Data schema — version 0.1.0

Every file YORU-Explain writes carries `schema_version` (`0.1.0`): JSON files as
a key, CSV tables as their **first column on every row**, NPZ archives in their
`meta_json`. A reader should refuse a different major version. Column order is
fixed by `src/yoru_explain/io/schema.py`.

Conventions in all files:

* **Coordinates** — `frame_px_xyxy`: pixels of the frame the detector was given
  (after any `v_flip` / `h_flip`), origin top-left, x right, y down. Boxes are
  `(x1, y1, x2, y2)` exactly as YORU reports them. Patches are integer
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
| `sources[]` | `source_id` (`src-001`…), `path` (absolute), `kind` (`video`/`image`), `v_flip`, `h_flip`, `size` (bytes), `quick_id` + `quick_id_method` (`sha256(size, first 8 MiB, last 8 MiB)`), `width`, `height`, `fps` (as the container states it), `frame_count` (as stated), `fourcc`, `seek_check` (`null` or `{status, samples, mismatched}`), `detections_files[]` (`{path, layout, frames, detections}`), `added` |
| `models[]` | `model_id` (`mdl-001`…), `path`, `sha256` (whole file; `synthetic` for the test detector), `adapter` (`yoru-yolov5`, `toy-grid`), `options` (`conf_thresh`, `iou_thresh`, `rgb_input`) |
| `settings` | reserved |

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
| `candidate_index`, `candidate_locator` | pre-NMS candidate row, and JSON `{level, stride, anchor, grid_y, grid_x}`; empty for an imported detection |
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
| `method_config` | occlusion: `method`, `method_kind` (`perturbation_effect`), `method_version`, `patch_size [h, w]`, `stride [y, x]`, `baseline {kind, value [B,G,R], sigma}` (default `constant` `[180, 105, 255]`, hot pink #FF69B4; older runs keep the colour they recorded), `region` (`context`/`box`/`full`), `context_scale`, `score {score_type, identity_mode, iou_threshold}`, `batch_size`, `seed`, `include_ambiguous`. Grad-CAM: `method`, `method_kind` (`attribution`), `method_version`, `layer`, `score_type`, `identity_mode` (always `fixed_candidate`), `precision`, `score_tolerance`. |
| `score_types`, `identity_modes` | the definitions of every score type and identity mode, in words |
| `model_id`, `model_options`, `adapter`, `adapter_version`, `events`, `frame_access`, `seed` | |

## runs/&lt;run_id&gt;/manifest.json — what it ran with, and how it ended

| Key | Meaning |
|---|---|
| `schema_version`, `kind` (`yoru-explain-run-manifest`), `run_id`, `created`, `finished`, `elapsed_s` | |
| `status` | `running` (while it runs — or the process was killed), `completed` (every event has a row; see `events` for how many failed), `cancelled`, `failed` (identity check failed, or the run itself was interrupted by an error) |
| `method`, `method_kind` | |
| `project` | `{name, root}` |
| `software` | versions of yoru_explain, python, numpy, opencv, torch, torchvision, yoru, pyqt5, and the git commits of the YORU and YORU-Explain checkouts |
| `hardware` | platform, CUDA/cuDNN versions, `cudnn_benchmark`, `cudnn_deterministic`, GPU names |
| `adapter` | `{name, version, capabilities}`; capabilities hold the measured `deterministic`, `batch_invariant`, `batch_max_abs_diff`, and the gradient probe's verdict |
| `model` | `model_id` + the adapter's `model_info()`: weights path, `sha256`, class names, strides, thresholds (`conf_thresh`, `nms_conf`, `iou_thresh`, `max_det`), channel order, device, precision, YORU version, bundled YOLOv5 path |
| `identity_check` | adapter vs `yoru.libs.plugins.get_detector(...).detect` on the first frame: `status` (`passed`/`failed`/`not_applicable`), `n_reference`, `n_adapter`, `max_abs_diff`, `exact`, `frame {source_id, frame_index, sha1}` |
| `inputs[]` | the sources used, as in project.json, plus `verify {status: ok/changed/missing}` |
| `events` | `requested`, and counts per result status (`ok`, `failed`, `cancelled`), `cached` |
| `error_codes` | count per error code |
| `coordinate_system`, `coordinate_system_description`, `baseline_note`, `interpretation`, `files` | |

## runs/&lt;run_id&gt;/results.csv — one row per event

Identification:
`schema_version, run_id, event_id, video_id (= source_id), source_path, frame_index, timestamp_ms, frame_sha1, detection_id (= detection_index), candidate_index, track_id, behavior_class (= class_name), class_id, target_class, x1, y1, x2, y2`

Reproduction of the stored detection on this run's inference:
`stored_score` (score stored with the event), `reproduction` — how the stored detection was found again on this run's inference: `same_candidate` (same candidate, class, and IoU ≥ 0.9), `iou` (no stored candidate; class and IoU ≥ 0.9), `iou_candidate_changed` (class and IoU ≥ 0.9, from another candidate), `iou_other_model` (the event was made with another model, so its candidate index was not used) — and `reproduction_iou`.

What was measured:

| Column | Occlusion | Grad-CAM |
|---|---|---|
| `method`, `method_kind`, `score_type`, `identity_mode`, `baseline` | `occlusion`, `perturbation_effect`, …, baseline description | `gradcam`, `attribution`, …, `fixed_candidate`, empty |
| `original_score` | target's score on the unchanged frame at batch 1 (= YORU's confidence for `obj_x_cls`) | score in the inference pass |
| `reference_score` | the same at the run's batch size: the deltas' reference | same as original |
| `perturbed_score`, `delta_score`, `max_delta_patch_id` | score and delta of the patch with the largest delta | empty |
| `delta_min`, `delta_mean` | over counted patches | empty |
| `in_box_mean_delta`, `out_box_mean_delta` | mean of the map over defined pixels inside / outside the target box | empty |
| `noop_delta` | delta of the unchanged frame run through the batch pipeline (expected exactly 0) | empty |
| `full_region_score`, `full_region_status` | the whole region replaced by the baseline | empty |
| `n_perturbations`, `n_counted` | patches, patches in the map | empty |
| `n_ok`, `n_matched`, `n_ambiguous`, `n_class_changed`, `n_unmatched`, `n_disappeared`, `n_non_finite`, `n_error` | patch counts per status | empty |
| `n_candidate_dropped` | fixed-candidate: patches on which NMS would no longer keep the candidate | empty |
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
| `matched_x1 … matched_y2`, `matched_score`, `matched_class` | re-detection: the matched (or best-overlapping) detection |
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
| `roi` | int32 `4` | the region the patches tile |
| `frame_shape` | int32 `2` | `H, W` |
| `meta_json` | 0-d str | JSON: `schema_version`, `method`, `method_kind`, `map`, `map_definition`, coordinate system, `status_codes`, `reference_score`, `reference_score_batch`, `config`, `target`, `run_id`, `score_description`, `cache_key` |

## runs/&lt;run_id&gt;/maps/&lt;event_id&gt;__gradcam.npz

| Array | Meaning |
|---|---|
| `cam` | float32 `H×W`: `max(Σ_c α_c A_c, 0)` mapped to frame pixels |
| `cam_feature` | float32 `h×w`: the signed `Σ_c α_c A_c` on the layer's grid, before ReLU |
| `target_box`, `frame_shape` | |
| `meta_json` | as above, plus `layer`, `layer_stride`, `letterbox` (the full transform: shapes, ratio, pads, `box_gain`, `box_pad`), `gradient_score`, `gradient_support` |

## Previews and logs

`previews/<event_id>__<method>.png`: original | heatmap | overlay with the
colour scale and the run's settings printed under it. For viewing only; the
numbers are in the NPZ and CSV files. `logs/run.log`: everything logged under
`yoru_explain` during the run, tracebacks included.
