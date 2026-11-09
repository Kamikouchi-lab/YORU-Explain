# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""A run's results as one folder of plain tables and arrays, for the experimenter's own analysis.

::

    <project>/exports/<run_id>/
    ├── export.json      what was exported from what: run, options, grid, units, counts, files
    ├── README.txt       how to read the files (written with them)
    ├── events.csv       one row per event of the run, failures included (status, error_code)
    ├── patches.csv      one row per occluded patch: frame pixels and box units (occlusion)
    ├── summary.csv      per unit and class, then per class over units: descriptive only
    ├── box_maps.npz     every finished event's map on the box grid (+ its picture), N x n x n
    ├── mean_maps.npz    unit means, and per class the mean of unit means, with counts per cell
    ├── maps/<event>.npz the map cropped to the analysed region, with its frame offset
    ├── crops/<event>.png the unchanged frame cropped to the same region
    └── gallery/         every event drawn on a shared colour scale, and contact sheets

Nothing here estimates which way an animal faces or names a body part: the
grids are the model's own box (``box_maps``), and :func:`yoru_explain.
analysis.normalize.align_maps` turns the maps to orientations the
experimenter supplies.  Statistics across animals are left to the
experimenter; ``unit_id`` (:meth:`Project.unit_of`) says which events belong
together.  Every value describes the model's score, not the animal.

The folder is built beside its final place and swapped in when complete, so
an interrupted export leaves the previous one (or none), never half of one.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from yoru_explain.analysis.normalize import box_points, box_region, grid_centres, sample, unit_class_means
from yoru_explain.core.errors import ErrorCode, ExplainError
from yoru_explain.core.geometry import region_mask
from yoru_explain.core.logs import get_logger
from yoru_explain.core.types import COORDINATE_SYSTEM
from yoru_explain.io import schema
from yoru_explain.io.frames import imwrite
from yoru_explain.io.project import Project, now_iso
from yoru_explain.io.runs import RunReader, write_json

log = get_logger("export")

EXPORTS_DIR = "exports"
#: How events are coloured in the gallery: one scale for the run, per class, or per event.
GALLERY_SCALES = ("run", "class", "event")

INTERPRETATION = ("Values describe one model's detection score on edited images (occlusion) or its "
                  "gradients (Grad-CAM); they are not findings about the animal. Class names are the "
                  "model's predictions, not ground truth. Frames are not independent samples: compare "
                  "experimental units (unit_id), not events.")

ProgressFn = Callable[[float, str], None]
CancelFn = Callable[[], bool]


@dataclass(frozen=True)
class ExportOptions:
    """How a run is exported (kept in ``export.json``)."""

    #: Cells per side of the box grid.
    grid: int = 64
    #: Half-width of the box grid in box sizes (the box's edges are at 0.5).
    #: ``None``: the analysed region -- half the context scale for occlusion
    #: over the box's context, 0.5 for the box itself, 1.5 otherwise.
    extent: Optional[float] = None
    #: Decode the frames again to write ``crops/`` and the grid pictures.
    crops: bool = True
    #: Draw ``gallery/`` (:func:`yoru_explain.visualization.gallery.render_gallery`).
    gallery: bool = True
    #: The gallery's colour scale: ``run``, ``class`` or ``event``.
    gallery_scale: str = "class"

    def __post_init__(self):
        if not 4 <= int(self.grid) <= 512:
            raise ValueError("grid must be 4..512 cells")
        object.__setattr__(self, "grid", int(self.grid))
        if self.extent is not None:
            object.__setattr__(self, "extent", float(self.extent))
            if not 0.0 < self.extent <= 20.0:
                raise ValueError("extent must be in (0, 20]")
        if self.gallery_scale not in GALLERY_SCALES:
            raise ValueError(f"gallery_scale must be one of {GALLERY_SCALES}")

    def to_dict(self) -> Dict[str, Any]:
        return {"grid": self.grid, "extent": self.extent, "crops": bool(self.crops),
                "gallery": bool(self.gallery), "gallery_scale": self.gallery_scale}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ExportOptions":
        unknown = set(d) - {"grid", "extent", "crops", "gallery", "gallery_scale"}
        if unknown:
            raise ValueError(f"unknown export keys: {sorted(unknown)}")
        return cls(**dict(d))


@dataclass
class ExportOutcome:
    export_dir: Path
    counts: Dict[str, int]
    warnings: List[str] = field(default_factory=list)


def default_extent(method: str, method_config: Mapping[str, Any]) -> float:
    """The box grid's half-width that covers what a run analysed."""
    if method == "occlusion":
        region = method_config.get("region", "context")
        if region == "context":
            return float(method_config.get("context_scale", 3.0)) / 2.0
        if region == "box":
            return 0.5
    return 1.5


def _f(value) -> float:
    return float(value) if value is not None else float("nan")


def _box(row: Mapping[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    b = tuple(_f(row.get(k)) for k in ("x1", "y1", "x2", "y2"))
    return b if all(math.isfinite(v) for v in b) and b[2] > b[0] and b[3] > b[1] else None


def _obb(row: Mapping[str, Any]):
    o = tuple(_f(row.get(k)) for k in ("obb_cx", "obb_cy", "obb_w", "obb_h", "obb_angle"))
    return o if all(math.isfinite(v) for v in o) else None


def _stats(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    v = [x for x in values if x is not None and math.isfinite(x)]
    if not v:
        return None, None
    return float(np.mean(v)), float(np.median(v))


def summarise(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """summary.csv: per (class, unit) the events' values, then per class the units' means.

    ``mean`` / ``median`` at the unit level are over the unit's finished
    events; at the class level over the unit means (each unit weighs the
    same).  No interval and no test: those need the experimenter's design.
    """
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for e in events:
        groups.setdefault((str(e.get("class_name") or ""), str(e.get("unit_id") or "")), []).append(e)
    rows: List[Dict[str, Any]] = []
    by_class: Dict[str, List[Dict[str, Any]]] = {}
    for (cls, unit) in sorted(groups):
        evs = groups[(cls, unit)]
        ok = [e for e in evs if e.get("status") == "ok"]
        row = {"level": "unit", "class_name": cls, "unit_id": unit, "n_units": 1,
               "n_events": len(evs), "n_ok": len(ok), "n_failed": len(evs) - len(ok)}
        for m in schema.EXPORT_SUMMARY_METRICS:
            row[f"{m}_mean"], row[f"{m}_median"] = _stats([e.get(m) for e in ok])
        rows.append(row)
        by_class.setdefault(cls, []).append(row)
    for cls in sorted(by_class):
        units = by_class[cls]
        row = {"level": "class", "class_name": cls, "unit_id": "",
               "n_units": sum(1 for u in units if u["n_ok"]),
               "n_events": sum(u["n_events"] for u in units), "n_ok": sum(u["n_ok"] for u in units),
               "n_failed": sum(u["n_failed"] for u in units)}
        for m in schema.EXPORT_SUMMARY_METRICS:
            row[f"{m}_mean"], row[f"{m}_median"] = _stats([u[f"{m}_mean"] for u in units])
        rows.append(row)
    return rows


class _Frames:
    """The run's frames, decoded in order and checked against the hash the run recorded."""

    def __init__(self, project: Project):
        self.project = project
        self._readers: Dict[str, Any] = {}
        self._broken: Dict[str, str] = {}

    def read(self, source_id: str, frame_index: int, sha1: str) -> Tuple[Optional[np.ndarray], str]:
        if source_id in self._broken:
            return None, self._broken[source_id]
        try:
            if source_id not in self._readers:
                self._readers[source_id] = self.project.open_source(source_id)
            frame = self._readers[source_id].read(int(frame_index), exact=True)
        except Exception as e:   # a missing or unreadable video: the export goes on without its pictures
            if source_id not in self._readers:
                self._broken[source_id] = "frame_unavailable"
            log.warning("export: %s frame %s not read: %s", source_id, frame_index, e)
            return None, "frame_unavailable"
        if sha1 and frame.sha1 != sha1:
            return None, "frame_mismatch"
        return frame.image, "ok"

    def close(self) -> None:
        for r in self._readers.values():
            r.close()
        self._readers.clear()


def _event_row(r: Mapping[str, Any], project: Project, fps_of: Dict[str, Any],
               method_config: Mapping[str, Any]) -> Dict[str, Any]:
    sid = r.get("video_id") or ""
    unit, assigned = project.unit_of(sid) if sid else ("", False)
    row = {k: r.get(k) for k in schema.EXPORT_EVENT_COLUMNS if k in r}
    row.update(
        event_id=r.get("event_id"), unit_id=unit, unit_assigned=assigned, source_id=sid,
        source_path=r.get("source_path"), fps=fps_of.get(sid), class_name=r.get("behavior_class"),
        run_map_file=r.get("map_file"), map_file=None,
    )
    row.pop("schema_version", None)
    row.pop("run_id", None)
    if (r.get("method") == "occlusion" and r.get("status") == "ok" and row.get("patch_height_px") is None
            and method_config.get("patch_fraction") is None and method_config.get("patch_size")):
        # A run written before the pixels used were recorded: they were the configured ones.
        (ph, pw), (sy, sx) = method_config["patch_size"], method_config["stride"]
        row.update(patch_height_px=ph, patch_width_px=pw, stride_y_px=sy, stride_x_px=sx)
    return row


def _replace_folder(stage: Path, final: Path) -> None:
    old = final.with_name(final.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if final.exists():
        os.replace(final, old)
    os.replace(stage, final)
    if old.exists():
        shutil.rmtree(old, ignore_errors=True)


def _readme(info: Mapping[str, Any]) -> str:
    g = info["grid"]
    return f"""YORU-Explain export of run {info['run_id']} ({info['method']}, map: {info['map']})
Created {info['created']}. Schema {info['schema_version']}. See docs/DATA_SCHEMA.md of YORU-Explain.

{INTERPRETATION}

events.csv   one row per event (failed ones too: status, error_code). unit_id is the
             experimental unit of the event's video (unit_assigned=false: the video itself).
             map_file / crop_file are relative to this folder; stack_index is the row in box_maps.npz.
patches.csv  one row per occluded patch: x0..y1 in frame pixels (half-open), u0..v1 in box
             units (u = (x - box centre x) / box width, v likewise with y and height).
summary.csv  descriptive values per unit and class, then per class over the unit means.
box_maps.npz map[k] = the map of event event_id[k] on a {g['cells']}x{g['cells']} grid around its box:
             cell (r, c) is the frame pixel at x = cx + u[c] * w, y = cy + v[r] * h (nearest pixel);
             the box edges are at u, v = +-0.5. NaN = not measured. image_rgb = the frame there.
mean_maps.npz  mean_by_unit (per unit and class) and mean_by_class (mean of the unit means),
             with n_events_by_unit / n_units_by_class per cell.
maps/<event>.npz  map (and coverage) cropped to the analysed region; frame pixel (x, y) is
             map[y - offset[1], x - offset[0]]. crops/<event>.png is the frame over the same region.
gallery/     each event drawn on a shared colour scale, and contact sheets per class (and unit).

To align the maps to orientations you measured yourself (one row per event: event_id, cx, cy,
angle_deg, length[, directed]): yoru_explain.api.align_maps(<this folder>, "orientations.csv").
"""


def export_run(project: Project, run_dir, out_dir=None, options: Optional[ExportOptions] = None, *,
               progress: Optional[ProgressFn] = None, cancel: Optional[CancelFn] = None) -> ExportOutcome:
    """Export the run in *run_dir* of *project* to *out_dir* (default ``<project>/exports/<run_id>``).

    An earlier export in *out_dir* is replaced when the new one is complete;
    a folder there that is not an export is refused.  Cancelling (*cancel*
    returns True) leaves the earlier export as it was and raises
    ``cancelled``.  *progress* gets ``(fraction, message)``.
    """
    options = options or ExportOptions()
    reader = RunReader(run_dir)
    method = str(reader.config.get("method") or reader.manifest.get("method") or "")
    if method not in ("occlusion", "gradcam"):
        raise ExplainError(ErrorCode.INVALID_CONFIG, f"{reader.run_id}: unknown method {method!r}")
    mc = reader.config.get("method_config") or {}
    map_key, kind = ("delta_map", "delta") if method == "occlusion" else ("cam", "cam")
    extent = options.extent if options.extent is not None else default_extent(method, mc)
    n = options.grid
    final = Path(out_dir) if out_dir is not None else project.root / EXPORTS_DIR / reader.run_id
    if final.exists() and any(final.iterdir()) and not (final / "export.json").is_file():
        raise ExplainError(ErrorCode.INVALID_CONFIG,
                           f"{final} holds other files and is not an export; choose another folder")
    stage = final.with_name(final.name + ".partial")
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "maps").mkdir(parents=True)
    if options.crops:
        (stage / "crops").mkdir()

    results = reader.typed_results()
    fps_of = {s["source_id"]: s.get("fps") for s in project.sources + project.removed_entries("sources")}
    perturbations = reader.perturbations_by_event() if method == "occlusion" else {}
    frames = _Frames(project) if options.crops else None
    order = sorted(range(len(results)), key=lambda i: (str(results[i].get("video_id") or ""),
                                                       results[i].get("frame_index") or -1))
    events: List[Optional[Dict[str, Any]]] = [None] * len(results)
    grids: Dict[int, Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]] = {}
    patch_rows: Dict[int, List[Dict[str, Any]]] = {}
    warnings: List[str] = []
    map_definition = None
    try:
        for done, i in enumerate(order):
            if cancel is not None and cancel():
                raise ExplainError(ErrorCode.CANCELLED, "export cancelled")
            r = results[i]
            ev = _event_row(r, project, fps_of, mc)
            events[i] = ev
            if progress:
                progress(done / max(1, len(order)), f"{r.get('event_id')}: exporting")
            box = _box(r)
            if r.get("status") != "ok" or not r.get("map_file") or box is None:
                continue
            data = reader.load_map(r["map_file"])
            m = data[map_key].astype(np.float32)
            map_definition = map_definition or data["meta"].get("map_definition")
            h, w = m.shape
            obb = _obb(r)
            if method == "occlusion" and "roi" in data:
                x0, y0, x1, y1 = (int(v) for v in data["roi"])
            else:
                x0, y0, x1, y1 = box_region(box, extent, w, h)
            eid = str(r["event_id"])
            crop = {"map": m[y0:y1, x0:x1], "offset": np.array([x0, y0], np.int32),
                    "frame_shape": np.array([h, w], np.int32), "box": np.array(box, np.float64),
                    "obb": np.array(obb if obb else [np.nan] * 5, np.float64)}
            if "coverage" in data:
                crop["coverage"] = data["coverage"][y0:y1, x0:x1]
            crop["meta_json"] = np.array(json.dumps({
                "schema_version": schema.SCHEMA_VERSION, "event_id": eid, "run_id": reader.run_id,
                "map": kind, "map_definition": data["meta"].get("map_definition"),
                "coordinate_system": f"{COORDINATE_SYSTEM} cropped: frame pixel (x, y) is "
                                     f"map[y - offset[1], x - offset[0]]"}))
            rel = f"maps/{eid}.npz"
            with open(stage / rel, "wb") as fh:
                np.savez_compressed(fh, **crop)
            ev.update(map_file=rel, crop_x0=x0, crop_y0=y0, crop_x1=x1, crop_y1=y1)

            xs, ys = box_points(box, n, extent)
            grid_map = sample(m, xs, ys)[0]
            if kind == "delta":
                defined = np.isfinite(m)
                pos = np.where(defined, np.clip(m, 0.0, None), 0.0)
                total = float(pos.sum())
                inside = region_mask(m.shape, box, obb)
                ev["in_box_positive_share"] = float(pos[inside].sum() / total) if total > 0 else None
            image, valid = None, None
            if frames is not None:
                picture, status = frames.read(ev["source_id"], int(r["frame_index"]), r.get("frame_sha1") or "")
                ev["crop_status"] = status
                if picture is not None:
                    imwrite(stage / f"crops/{eid}.png", picture[y0:y1, x0:x1])
                    ev["crop_file"] = f"crops/{eid}.png"
                    image, valid = sample(picture[:, :, ::-1], xs, ys)
                else:
                    warnings.append(f"{eid}: frame not exported ({status})")
            grids[i] = (grid_map, image, valid)
            if method == "occlusion":
                patch_rows[i] = _patch_rows(perturbations.get(eid, []), ev, box)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        if frames is not None:
            frames.close()

    try:
        _write(stage, reader, project, options, method, kind, extent, n, results, events, grids,
               patch_rows, map_definition, mc)
        if options.gallery:
            from yoru_explain.visualization.gallery import render_gallery

            if progress:
                progress(0.99, "drawing the gallery")
            render_gallery(stage, scale=options.gallery_scale, cancel=cancel)
        _replace_folder(stage, final)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    counts = {"events": len(results), "ok": sum(1 for e in events if e and e.get("map_file")),
              "failed": sum(1 for r in results if r.get("status") != "ok"),
              "crops": sum(1 for e in events if e and e.get("crop_file"))}
    if progress:
        progress(1.0, f"exported {counts['ok']} of {counts['events']} event(s) to {final}")
    log.info("export of %s to %s: %s", reader.run_id, final, counts)
    return ExportOutcome(final, counts, warnings)


def _patch_rows(rows: List[Mapping[str, str]], ev: Mapping[str, Any], box) -> List[Dict[str, Any]]:
    x1, y1, x2, y2 = box
    cx, cy, w, h = (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1
    out = []
    for p in rows:
        row = {k: p.get(k) for k in ("patch_id", "kind", "x0", "y0", "x1", "y1", "status", "counted",
                                     "perturbed_score", "delta_score", "candidate_kept", "pixel_change")}
        row.update(event_id=ev["event_id"], unit_id=ev["unit_id"], class_name=ev["class_name"])
        if p.get("kind") != "noop":
            px0, py0, px1, py1 = (float(p[k]) for k in ("x0", "y0", "x1", "y1"))
            row.update(u0=(px0 - cx) / w, v0=(py0 - cy) / h, u1=(px1 - cx) / w, v1=(py1 - cy) / h)
        out.append(row)
    return out


def _write(stage: Path, reader: RunReader, project: Project, options: ExportOptions, method: str,
           kind: str, extent: float, n: int, results, events, grids, patch_rows, map_definition,
           method_config) -> None:
    order = [i for i in range(len(results)) if i in grids]
    for k, i in enumerate(order):
        events[i]["stack_index"] = k
    rows = [dict(e, run_id=reader.run_id) for e in events]
    schema.write_rows(stage / "events.csv", schema.EXPORT_EVENT_COLUMNS, rows)
    if method == "occlusion":
        schema.write_rows(stage / "patches.csv", schema.EXPORT_PATCH_COLUMNS,
                          [p for i in order for p in patch_rows.get(i, [])])
    schema.write_rows(stage / "summary.csv", schema.EXPORT_SUMMARY_COLUMNS, summarise(rows))

    g = grid_centres(n, extent)
    stack = np.stack([grids[i][0] for i in order]) if order else np.zeros((0, n, n), np.float32)
    meta = {"schema_version": schema.SCHEMA_VERSION, "run_id": reader.run_id, "map": kind,
            "map_definition": map_definition, "grid": n, "extent": extent,
            "grid_definition": "cell (r, c) samples the frame pixel containing (cx + u[c] * w, cy + v[r] * h) "
                               "of the target's box (centre cx, cy; width w, height h); box edges at "
                               "u, v = +-0.5; NaN where the map is not defined or off the frame"}
    arrays = {"event_id": np.array([str(results[i]["event_id"]) for i in order], dtype=str),
              "unit_id": np.array([events[i]["unit_id"] for i in order], dtype=str),
              "class_name": np.array([events[i]["class_name"] or "" for i in order], dtype=str),
              "map": stack, "u": g, "v": g, "meta_json": np.array(json.dumps(meta))}
    if options.crops:
        arrays["image_rgb"] = np.stack([
            grids[i][1] if grids[i][1] is not None else np.zeros((n, n, 3), np.uint8) for i in order]) \
            if order else np.zeros((0, n, n, 3), np.uint8)
        arrays["image_valid"] = np.stack([
            grids[i][2] if grids[i][2] is not None else np.zeros((n, n), bool) for i in order]) \
            if order else np.zeros((0, n, n), bool)
    np.savez_compressed(stage / "box_maps.npz", **arrays)
    means = unit_class_means(stack, list(arrays["class_name"]), list(arrays["unit_id"])).arrays()
    means.update(u=g, v=g, meta_json=np.array(json.dumps(meta)))
    np.savez_compressed(stage / "mean_maps.npz", **means)

    units = {}
    for e in events:
        if e and e.get("source_id"):
            units[e["source_id"]] = {"unit_id": e["unit_id"], "assigned": bool(e["unit_assigned"])}
    from yoru_explain import __version__

    info = {
        "schema_version": schema.SCHEMA_VERSION,
        "kind": schema.EXPORT_KIND,
        "created": now_iso(),
        "software": {"yoru_explain": __version__},
        "run_id": reader.run_id,
        "run_dir": str(Path(reader.dir).resolve()),
        "project": {"name": project.name, "root": str(project.root)},
        "method": method,
        "method_kind": reader.manifest.get("method_kind"),
        "method_config": method_config,
        "map": kind,
        "map_definition": map_definition,
        "options": options.to_dict(),
        "grid": {"cells": n, "extent": extent,
                 "definition": meta["grid_definition"]},
        "units": units,
        "counts": {"events": len(results), "exported_maps": len(order),
                   "failed": sum(1 for r in results if r.get("status") != "ok"),
                   "crops": sum(1 for e in events if e and e.get("crop_file"))},
        "files": {
            "events.csv": "one row per event of the run (schema_version first)",
            "patches.csv": "one row per occluded patch (occlusion runs)",
            "summary.csv": "per unit and class, then per class over unit means; descriptive",
            "box_maps.npz": "event maps on the box grid (+ pictures), N x cells x cells",
            "mean_maps.npz": "unit means and per-class means of unit means, with counts per cell",
            "maps/*.npz": "each event's map cropped to the analysed region, with its offset",
            "crops/*.png": "the unchanged frame over the same region",
            "gallery/": "pictures on a shared colour scale and contact sheets",
        },
        "interpretation": INTERPRETATION,
    }
    write_json(stage / "export.json", info)
    (stage / "README.txt").write_text(_readme(info), encoding="utf-8")
