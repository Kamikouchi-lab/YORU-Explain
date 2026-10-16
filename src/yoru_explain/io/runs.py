# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""One analysis run on disk::

    runs/<run_id>/
    ├── config.json         what was asked: method, parameters, score definition, seed
    ├── manifest.json       what it ran with and what came out (provenance, checks, counts)
    ├── results.csv         one row per event
    ├── perturbations.csv   one row per perturbed input (occlusion)
    ├── maps/<event>__<method>.npz     the maps and per-patch arrays, with metadata
    ├── previews/<event>__<method>.png original | heatmap | overlay
    └── logs/run.log

Rows are appended as each event finishes, so a run that is cancelled or
crashes leaves valid tables of what was done; ``manifest.json`` says how it
ended.  Every file carries ``schema_version`` (docs/DATA_SCHEMA.md).
"""

from __future__ import annotations

import json
import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

from yoru_explain.core.types import COORDINATE_SYSTEM, COORDINATE_SYSTEM_DESCRIPTION
from yoru_explain.io import schema
from yoru_explain.io.frames import imwrite

MAPS_DIR = "maps"
PREVIEWS_DIR = "previews"


def new_run_id(method: str) -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}_{method}_{secrets.token_hex(3)}"


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def _clean(value):
    """NaN/inf -> None, recursively, so the JSON is valid."""
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def write_json(path, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_clean(dict(data)), indent=2, ensure_ascii=False, default=_json_default),
                   encoding="utf-8")
    os.replace(tmp, path)


def read_json(path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def map_name(event_id: str, method: str, ext: str) -> str:
    return f"{event_id}__{method}.{ext}"


class RunWriter:
    def __init__(self, runs_dir, run_id: str):
        self.run_id = run_id
        self.dir = Path(runs_dir) / run_id
        if self.dir.exists():
            raise FileExistsError(self.dir)
        (self.dir / MAPS_DIR).mkdir(parents=True)
        (self.dir / PREVIEWS_DIR).mkdir()
        (self.dir / "logs").mkdir()

    @property
    def log_path(self) -> Path:
        return self.dir / "logs" / "run.log"

    def write_config(self, config: Mapping[str, Any]) -> None:
        write_json(self.dir / "config.json", config)

    def write_manifest(self, manifest: Mapping[str, Any]) -> None:
        write_json(self.dir / "manifest.json", manifest)

    def append_result(self, row: Mapping[str, Any]) -> None:
        row = dict(row)
        row["run_id"] = self.run_id
        schema.write_rows(self.dir / "results.csv", schema.RESULT_COLUMNS, [row], append=True)

    def append_perturbations(self, rows: Iterable[Mapping[str, Any]]) -> int:
        rows = [dict(r, run_id=self.run_id) for r in rows]
        return schema.write_rows(self.dir / "perturbations.csv", schema.PERTURBATION_COLUMNS, rows,
                                 append=True)

    def ensure_tables(self) -> None:
        """Write the headers of tables that got no rows, so readers find them."""
        for name, cols in (("results.csv", schema.RESULT_COLUMNS),
                           ("perturbations.csv", schema.PERTURBATION_COLUMNS)):
            if not (self.dir / name).is_file():
                schema.write_rows(self.dir / name, cols, [])

    def save_map(self, event_id: str, method: str, arrays: Mapping[str, np.ndarray],
                 meta: Mapping[str, Any]) -> str:
        rel = f"{MAPS_DIR}/{map_name(event_id, method, 'npz')}"
        payload = {k: np.asarray(v) for k, v in arrays.items()}
        payload["meta_json"] = np.array(json.dumps(_clean(dict(meta)), default=_json_default))
        with open(self.dir / rel, "wb") as f:
            np.savez_compressed(f, **payload)
        return rel

    def save_preview(self, event_id: str, method: str, image: np.ndarray) -> str:
        rel = f"{PREVIEWS_DIR}/{map_name(event_id, method, 'png')}"
        imwrite(self.dir / rel, image)
        return rel


class RunReader:
    """A finished (or interrupted) run, read back."""

    def __init__(self, run_dir):
        self.dir = Path(run_dir)
        self.config = read_json(self.dir / "config.json")
        self.manifest = read_json(self.dir / "manifest.json")
        self.run_id = self.manifest.get("run_id", self.dir.name)
        self._results: Optional[List[Dict[str, str]]] = None

    @property
    def results(self) -> List[Dict[str, str]]:
        """results.csv as written (text cells); read once per reader."""
        if self._results is None:
            path = self.dir / "results.csv"
            self._results = schema.read_rows(path) if path.is_file() else []
        return list(self._results)

    def typed_results(self) -> List[Dict[str, Any]]:
        """results.csv with numbers as numbers and empty cells as ``None``."""
        return [schema.typed_result_row(r) for r in self.results]

    def perturbations(self, event_id: Optional[str] = None) -> List[Dict[str, str]]:
        path = self.dir / "perturbations.csv"
        rows = schema.read_rows(path) if path.is_file() else []
        return [r for r in rows if event_id is None or r["event_id"] == event_id]

    def result(self, event_id: str) -> Optional[Dict[str, str]]:
        for r in self.results:
            if r["event_id"] == event_id:
                return r
        return None

    def load_map(self, rel_path: str) -> Dict[str, Any]:
        """The arrays of ``maps/*.npz`` plus ``meta`` (its JSON metadata)."""
        with np.load(self.dir / rel_path, allow_pickle=False) as data:
            out: Dict[str, Any] = {k: data[k] for k in data.files if k != "meta_json"}
            out["meta"] = json.loads(str(data["meta_json"]))
        return out


# -- what goes into the files -------------------------------------------------


def target_columns(target, source: Mapping[str, Any]) -> Dict[str, Any]:
    d = target.detection
    return {
        "event_id": target.event_id,
        "video_id": source.get("source_id"),
        "source_path": source.get("path"),
        "frame_index": target.frame.frame_index,
        "timestamp_ms": target.frame.timestamp_ms,
        "frame_sha1": target.frame.frame_sha1,
        "detection_id": target.detection_index,
        "candidate_index": d.candidate.index if d.candidate else None,
        "track_id": target.track_id,
        "behavior_class": d.class_name,
        "class_id": d.class_id,
        "target_class": target.target_class,
        "x1": d.box[0], "y1": d.box[1], "x2": d.box[2], "y2": d.box[3],
    }


def perturbation_rows(event_id: str, result) -> List[Dict[str, Any]]:
    rows = []
    records = list(result.controls.values()) + list(result.patches)
    for r in records:
        s = r.score
        m = s.matched
        x0, y0, x1, y1 = r.patch.box
        rows.append({
            "event_id": event_id,
            "patch_id": r.patch.patch_id,
            "kind": r.patch.kind,
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "pixel_change": r.pixel_change,
            "status": s.status,
            # The number the delta was taken from; empty when not counted.
            "perturbed_score": r.used_score,
            "delta_score": r.delta,
            "counted": r.counted,
            "iou": s.iou,
            "candidate_index": s.candidate_index,
            "candidate_kept": s.candidate_kept,
            "candidate_class": s.candidate_class,
            "matched_x1": m.box[0] if m else None,
            "matched_y1": m.box[1] if m else None,
            "matched_x2": m.box[2] if m else None,
            "matched_y2": m.box[3] if m else None,
            "matched_score": m.score if m else None,
            "matched_class": m.class_id if m else None,
            "n_overlapping": s.n_overlapping,
            "error_code": s.error_code,
        })
    return rows


_STATUS_CODES = {name: i for i, name in enumerate(
    ("ok", "matched", "ambiguous", "class_changed", "unmatched", "disappeared", "non_finite", "error"))}


def occlusion_npz(result, extra_meta: Mapping[str, Any]):
    """Arrays and metadata of an occlusion result's NPZ."""
    patches = result.patches
    arrays = {
        "delta_map": result.delta_map,
        "coverage": result.coverage,
        "coverage_all": result.coverage_all,
        "lost_fraction": result.lost_fraction,
        "patch_id": np.array([p.patch.patch_id for p in patches], np.int32),
        "patch_xyxy": np.array([p.patch.box for p in patches], np.int32).reshape(-1, 4),
        "patch_score": np.array([np.nan if p.used_score is None else p.used_score for p in patches],
                                np.float64),
        "patch_delta": np.array([np.nan if p.delta is None else p.delta for p in patches], np.float64),
        "patch_counted": np.array([p.counted for p in patches], bool),
        "patch_status": np.array([_STATUS_CODES.get(p.score.status, -1) for p in patches], np.int8),
        "patch_pixel_change": np.array([p.pixel_change for p in patches], np.float32),
        "target_box": np.array(result.target.box, np.float64),
        "roi": np.array(result.roi, np.int32),
        "frame_shape": np.array(result.delta_map.shape, np.int32),
    }
    meta = {
        "schema_version": schema.SCHEMA_VERSION,
        "method": "occlusion",
        "method_kind": "perturbation_effect",
        "map": "delta_map",
        "map_definition": "mean over counted patches covering the pixel of (reference_score - perturbed_score); "
                          "NaN where no counted patch covers it",
        "coordinate_system": COORDINATE_SYSTEM,
        "coordinate_system_description": COORDINATE_SYSTEM_DESCRIPTION,
        "status_codes": _STATUS_CODES,
        "reference_score": result.reference_score,
        "reference_score_batch": result.reference_score_batch,
        "config": result.config.to_dict(),
        "target": result.target.to_dict(),
        **dict(extra_meta),
    }
    return arrays, meta


def gradcam_npz(result, transform: Optional[Mapping[str, Any]], extra_meta: Mapping[str, Any]):
    arrays = {
        "cam": result.cam,
        "cam_feature": result.cam_feature,
        "target_box": np.array(result.target.box, np.float64),
        "frame_shape": np.array(result.cam.shape, np.int32),
    }
    meta = {
        "schema_version": schema.SCHEMA_VERSION,
        "method": "gradcam",
        "method_kind": "attribution",
        "map": "cam",
        "map_definition": "max(sum_c alpha_c A_c, 0) on the layer's grid (cam_feature is the signed sum), "
                          "alpha_c = mean gradient of the score over the grid; mapped to frame pixels by "
                          "bilinear upsampling and undoing the letterbox",
        "coordinate_system": COORDINATE_SYSTEM,
        "coordinate_system_description": COORDINATE_SYSTEM_DESCRIPTION,
        "layer": result.layer,
        "layer_stride": result.layer_stride,
        "letterbox": transform,
        "reference_score": result.reference_score,
        "gradient_score": result.gradient_score,
        "gradient_support": result.gradient_support,
        "config": result.config.to_dict(),
        "target": result.target.to_dict(),
        **dict(extra_meta),
    }
    return arrays, meta
