# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Column order of every table this application writes.

The layout of each file is described in ``docs/DATA_SCHEMA.md``.  Every CSV
row starts with ``schema_version``, so a table copied away from its run
still says how to read it; JSON files and NPZ archives carry it too.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

SCHEMA_VERSION = "0.1.0"

PROJECT_KIND = "yoru-explain-project"
MANIFEST_KIND = "yoru-explain-run-manifest"
CONFIG_KIND = "yoru-explain-run-config"

EVENT_COLUMNS = (
    "schema_version", "event_id", "source_id", "frame_index", "timestamp_ms", "frame_sha1",
    "frame_access", "v_flip", "h_flip", "frame_width", "frame_height", "model_id",
    "detection_index", "class_id", "class_name", "x1", "y1", "x2", "y2", "score",
    "candidate_index", "candidate_locator", "target_class", "track_id", "origin", "note", "created",
)

RESULT_COLUMNS = (
    "schema_version", "run_id", "event_id", "video_id", "source_path", "frame_index",
    "timestamp_ms", "frame_sha1", "detection_id", "candidate_index", "track_id",
    "behavior_class", "class_id", "target_class", "x1", "y1", "x2", "y2",
    "stored_score", "reproduction", "reproduction_iou",
    "method", "method_kind", "score_type", "identity_mode", "baseline",
    "original_score", "reference_score", "perturbed_score", "delta_score", "max_delta_patch_id",
    "delta_min", "delta_mean", "in_box_mean_delta", "out_box_mean_delta",
    "noop_delta", "full_region_score", "full_region_status",
    "n_perturbations", "n_counted", "n_ok", "n_matched", "n_ambiguous", "n_class_changed",
    "n_unmatched", "n_disappeared", "n_non_finite", "n_error", "n_candidate_dropped",
    "matching_status", "gradcam_layer", "gradcam_stride", "gradient_support", "gradient_score",
    "cam_max", "cam_in_box_fraction",
    "status", "error_code", "message", "map_file", "preview_file", "cache_key", "cached_from",
    "elapsed_s",
)

PERTURBATION_COLUMNS = (
    "schema_version", "run_id", "event_id", "patch_id", "kind", "x0", "y0", "x1", "y1",
    "pixel_change", "status", "perturbed_score", "delta_score", "counted", "iou",
    "candidate_index", "candidate_kept", "candidate_class",
    "matched_x1", "matched_y1", "matched_x2", "matched_y2", "matched_score", "matched_class",
    "n_overlapping", "error_code",
)


#: results.csv columns holding integers / floats; the rest are text.
RESULT_INT_COLUMNS = frozenset({
    "frame_index", "detection_id", "candidate_index", "track_id", "class_id", "target_class",
    "max_delta_patch_id", "n_perturbations", "n_counted", "n_ok", "n_matched", "n_ambiguous",
    "n_class_changed", "n_unmatched", "n_disappeared", "n_non_finite", "n_error",
    "n_candidate_dropped", "gradcam_stride", "gradient_support",
})
RESULT_FLOAT_COLUMNS = frozenset({
    "timestamp_ms", "x1", "y1", "x2", "y2", "stored_score", "reproduction_iou", "original_score",
    "reference_score", "perturbed_score", "delta_score", "delta_min", "delta_mean",
    "in_box_mean_delta", "out_box_mean_delta", "noop_delta", "full_region_score", "gradient_score",
    "cam_max", "cam_in_box_fraction", "elapsed_s",
})


def typed_result_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """A results.csv row with numbers as numbers and empty cells as ``None``."""
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, str):
            if v.strip() == "":
                out[k] = None
            elif k in RESULT_INT_COLUMNS:
                out[k] = int(float(v))
            elif k in RESULT_FLOAT_COLUMNS:
                out[k] = float(v)
            else:
                out[k] = v
        else:
            out[k] = v
    return out


def cell(value: Any) -> str:
    """One CSV cell: empty for missing / non-finite numbers, never ``nan``."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return repr(value)
    return str(value)


def write_rows(path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]], *,
               append: bool = False) -> int:
    """Write *rows* (dicts) in *columns* order; unknown keys are an error."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not (append and path.is_file() and path.stat().st_size > 0)
    n = 0
    with open(path, "a" if append else "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(columns)
        for row in rows:
            extra = set(row) - set(columns)
            if extra:
                raise ValueError(f"{path.name}: unknown columns {sorted(extra)}")
            out = dict(row)
            out.setdefault("schema_version", SCHEMA_VERSION)
            writer.writerow([cell(out.get(c)) for c in columns])
            n += 1
    return n


def read_rows(path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def as_float(text) -> float:
    """A CSV cell as a float; NaN for an empty cell."""
    if text is None or str(text).strip() == "":
        return float("nan")
    return float(text)


def as_int(text):
    if text is None or str(text).strip() == "":
        return None
    return int(float(text))


def as_bool(text) -> bool:
    return str(text).strip().lower() in ("1", "true", "yes")
