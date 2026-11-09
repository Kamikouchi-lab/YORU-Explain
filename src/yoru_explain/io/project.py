# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""A project folder: inputs, models, chosen events, and analysis runs.

::

    project_root/
    ├── project.json      sources, models, settings (schema_version)
    ├── events.csv        one row per detection chosen for explanation
    └── runs/<run_id>/    one folder per analysis run (see io.runs)

Videos and weights are referenced by path and identified by hash; they are
never copied into the project.  A source or model taken out of the project
is kept under ``removed``, so its id is never given to another file: events
and runs that name it keep meaning what they meant.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from yoru_explain.core.errors import ErrorCode, ExplainError
from yoru_explain.core.hashing import QUICK_ID_METHOD, quick_file_id, sha256_file
from yoru_explain.core.types import (
    CandidateRef,
    Detection,
    ExplanationTarget,
    FrameRef,
    ModelRecord,
    finite_or_none,
)
from yoru_explain.io import schema
from yoru_explain.io.frames import kind_of, open_source

PROJECT_FILE = "project.json"
EVENTS_FILE = "events.csv"
RUNS_DIR = "runs"
REMOVED_KEY = "removed"

#: Characters Windows does not allow in a folder name, and the names it reserves.
_BAD_NAME_CHARS = '<>:"/\\|?*'
_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                   *(f"LPT{i}" for i in range(1, 10))}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _next_id(prefix: str, width: int, used: Iterable[str]) -> str:
    """``prefix`` + one more than the highest number among *used* (never a gap left by a removal)."""
    numbers = [int(u[len(prefix):]) for u in used
               if u and u.startswith(prefix) and u[len(prefix):].isdigit()]
    return f"{prefix}{max(numbers, default=0) + 1:0{width}d}"


def new_project_folder(name: str, location) -> Path:
    """The folder ``location/name`` a new project would be created in.

    Raises :class:`ExplainError` (``INVALID_CONFIG``) with a sentence a user
    can act on when the name is not a valid folder name, the location does
    not exist, or the folder is already a project or holds other files.
    """
    name = str(name or "").strip()
    if not name:
        raise ExplainError(ErrorCode.INVALID_CONFIG, "Enter a project name.")
    bad = sorted({c for c in name if c in _BAD_NAME_CHARS or ord(c) < 32})
    if bad:
        raise ExplainError(ErrorCode.INVALID_CONFIG,
                           f"A project name cannot contain {' '.join(bad)}.")
    if name.endswith(".") or name in (".", "..") or name.split(".")[0].upper() in _RESERVED_NAMES:
        raise ExplainError(ErrorCode.INVALID_CONFIG, f"{name!r} cannot be a folder name.")
    if not str(location or "").strip():
        raise ExplainError(ErrorCode.INVALID_CONFIG, "Choose the folder to create the project in.")
    parent = Path(str(location).strip())
    if not parent.is_dir():
        raise ExplainError(ErrorCode.INVALID_CONFIG, f"The location {parent} does not exist.")
    folder = parent / name
    if (folder / PROJECT_FILE).exists():
        raise ExplainError(ErrorCode.INVALID_CONFIG,
                           f"{folder} is already a project; open it with Load project.")
    if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
        raise ExplainError(ErrorCode.INVALID_CONFIG,
                           f"{folder} already exists and is not empty; choose another name.")
    return folder


def target_to_event_row(target: ExplanationTarget, note: str = "", created: str = "") -> Dict[str, Any]:
    f, d = target.frame, target.detection
    return {
        "event_id": target.event_id,
        "source_id": f.source_id,
        "frame_index": f.frame_index,
        "timestamp_ms": finite_or_none(f.timestamp_ms),
        "frame_sha1": f.frame_sha1,
        "frame_access": f.access,
        "v_flip": f.v_flip,
        "h_flip": f.h_flip,
        "frame_width": f.width,
        "frame_height": f.height,
        "model_id": target.model_id,
        "detection_index": target.detection_index,
        "class_id": d.class_id,
        "class_name": d.class_name,
        "x1": d.box[0], "y1": d.box[1], "x2": d.box[2], "y2": d.box[3],
        **schema.obb_columns(d.obb),
        "score": d.score,
        "candidate_index": d.candidate.index if d.candidate else None,
        "candidate_locator": json.dumps(d.candidate.locator_dict) if d.candidate else None,
        "target_class": target.target_class,
        "track_id": target.track_id,
        "origin": target.origin,
        "note": note,
        "created": created or now_iso(),
    }


class Project:
    """An open project folder."""

    def __init__(self, root: Path, data: Dict[str, Any], events: List[Dict[str, str]]):
        self.root = Path(root)
        self.data = data
        self._events = events

    # -- opening ---------------------------------------------------------

    @classmethod
    def create(cls, root, name: Optional[str] = None) -> "Project":
        root = Path(root)
        if (root / PROJECT_FILE).exists():
            raise ExplainError(ErrorCode.INVALID_CONFIG, f"{root} already holds a project")
        root.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": schema.SCHEMA_VERSION,
            "kind": schema.PROJECT_KIND,
            "name": name or root.name,
            "created": now_iso(),
            "sources": [],
            "models": [],
            "settings": {},
        }
        project = cls(root, data, [])
        project.save()
        return project

    @classmethod
    def open(cls, root) -> "Project":
        root = Path(root)
        path = root / PROJECT_FILE
        if not path.is_file():
            raise ExplainError(ErrorCode.MISSING_DATA, f"no {PROJECT_FILE} in {root}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("kind") != schema.PROJECT_KIND:
            raise ExplainError(ErrorCode.INVALID_CONFIG, f"{path} is not a YORU-Explain project")
        major = str(data.get("schema_version", "0")).split(".")[0]
        if major != schema.SCHEMA_VERSION.split(".")[0]:
            raise ExplainError(ErrorCode.INVALID_CONFIG,
                               f"{path}: schema {data.get('schema_version')} is not readable by "
                               f"schema {schema.SCHEMA_VERSION}")
        events_path = root / EVENTS_FILE
        events = schema.read_rows(events_path) if events_path.is_file() else []
        return cls(root, data, events)

    @classmethod
    def open_or_create(cls, root, name: Optional[str] = None) -> "Project":
        root = Path(root)
        return cls.open(root) if (root / PROJECT_FILE).is_file() else cls.create(root, name)

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(self.root / PROJECT_FILE,
                           json.dumps(self.data, indent=2, ensure_ascii=False))
        tmp = self.root / (EVENTS_FILE + ".tmp")
        schema.write_rows(tmp, schema.EVENT_COLUMNS, self._events)
        os.replace(tmp, self.root / EVENTS_FILE)

    # -- paths -----------------------------------------------------------

    @property
    def name(self) -> str:
        return str(self.data.get("name", self.root.name))

    @property
    def runs_dir(self) -> Path:
        return self.root / RUNS_DIR

    @property
    def settings(self) -> Dict[str, Any]:
        return self.data.setdefault("settings", {})

    # -- sources ---------------------------------------------------------

    @property
    def sources(self) -> List[Dict[str, Any]]:
        return self.data.setdefault("sources", [])

    def source(self, source_id: str) -> Dict[str, Any]:
        for s in self.sources:
            if s["source_id"] == source_id:
                return s
        raise ExplainError(ErrorCode.MISSING_DATA, f"no source {source_id!r} in the project")

    def add_source(self, path, *, v_flip: bool = False, h_flip: bool = False,
                   check_seeking: bool = False) -> Dict[str, Any]:
        """Register a video or image; the same file with the same flips is added once."""
        path = Path(path).resolve()
        kind = kind_of(path)
        for s in self.sources:
            if Path(s["path"]) == path and s["v_flip"] == v_flip and s["h_flip"] == h_flip:
                return s
        with open_source(path, v_flip=v_flip, h_flip=h_flip) as reader:
            info = reader.info
            seek = reader.check_seeking() if (check_seeking and kind == "video") else None
        used = ([s["source_id"] for s in self.sources]
                + [s["source_id"] for s in self.removed_entries("sources")]
                + [e.get("source_id", "") for e in self._events])
        entry = {
            "source_id": _next_id("src-", 3, used),
            "path": str(path),
            "kind": kind,
            "v_flip": bool(v_flip),
            "h_flip": bool(h_flip),
            "size": path.stat().st_size,
            "quick_id": quick_file_id(path),
            "quick_id_method": QUICK_ID_METHOD,
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "frame_count": info.frame_count,
            "fourcc": info.fourcc,
            "seek_check": seek,
            "detections_files": [],
            "added": now_iso(),
        }
        self.sources.append(entry)
        return entry

    def verify_source(self, source_id: str) -> Dict[str, Any]:
        """Whether the file is still there and still the same file."""
        s = self.source(source_id)
        p = Path(s["path"])
        if not p.is_file():
            return {"status": "missing", "path": str(p)}
        same = quick_file_id(p) == s["quick_id"]
        return {"status": "ok" if same else "changed", "path": str(p)}

    def source_file_status(self, source_id: str) -> str:
        """A quick look, cheap enough for every redraw: ``ok``, ``missing`` or ``changed``.

        Only the existence and the size are compared; :meth:`verify_source`
        (which every run records) also reads the file's ends.
        """
        s = self.source(source_id)
        p = Path(s["path"])
        if not p.is_file():
            return "missing"
        return "ok" if s.get("size") in (None, p.stat().st_size) else "changed"

    def open_source(self, source_id: str):
        s = self.source(source_id)
        return open_source(s["path"], v_flip=s["v_flip"], h_flip=s["h_flip"])

    def events_of_sources(self, source_ids: Iterable[str]) -> List[str]:
        """Ids of the events picked on these sources."""
        ids = set(source_ids)
        return [e["event_id"] for e in self._events if e.get("source_id") in ids]

    def remove_sources(self, source_ids: Iterable[str]) -> List[str]:
        """Take sources out of the project, with the events picked on them.

        The files are not touched, and runs keep their results.  The entries
        move to ``removed`` (stamped with the time), so their ids are never
        given again.  Returns the ids of the events removed with them.
        """
        drop = set(source_ids)
        for sid in drop:
            self.source(sid)   # an unknown id is an error, not a no-op
        events = self.events_of_sources(drop)
        self.remove_events(events)
        stamp = now_iso()
        gone = [dict(s, removed=stamp) for s in self.sources if s["source_id"] in drop]
        self.data["sources"] = [s for s in self.sources if s["source_id"] not in drop]
        self._removed("sources").extend(gone)
        return events

    # -- detections CSVs -------------------------------------------------

    def attach_detections(self, source_id: str, csv_path) -> Dict[str, Any]:
        """Attach a YORU / YORU-Tracker detections CSV to a source (once per file).

        Its boxes are shown on the frames; turning them into events is
        :func:`yoru_explain.api.add_events_from_detections_csv`.
        """
        from yoru_explain.io.detections_csv import load_detections_csv

        s = self.source(source_id)
        path = Path(csv_path).resolve()
        layout, frames = load_detections_csv(path)
        entry = {"path": str(path), "layout": layout, "frames": len(frames),
                 "detections": sum(len(v) for v in frames.values())}
        files = [e for e in s.get("detections_files", []) if Path(e["path"]) != path]
        s["detections_files"] = files + [entry]
        return entry

    def detach_detections(self, source_id: str, csv_path=None) -> int:
        """Detach one detections CSV (or, with no *csv_path*, all) from a source."""
        s = self.source(source_id)
        files = s.get("detections_files", [])
        keep = [] if csv_path is None else [e for e in files if Path(e["path"]) != Path(csv_path).resolve()]
        s["detections_files"] = keep
        return len(files) - len(keep)

    # -- removed entries -------------------------------------------------

    def removed_entries(self, kind: str) -> List[Dict[str, Any]]:
        """Sources or models (*kind*) taken out of the project, oldest first."""
        return list(self.data.get(REMOVED_KEY, {}).get(kind, []))

    def _removed(self, kind: str) -> List[Dict[str, Any]]:
        return self.data.setdefault(REMOVED_KEY, {}).setdefault(kind, [])

    # -- models ----------------------------------------------------------

    @property
    def models(self) -> List[ModelRecord]:
        return [ModelRecord.from_dict(m) for m in self.data.setdefault("models", [])]

    def model(self, model_id: str) -> ModelRecord:
        for m in self.models:
            if m.model_id == model_id:
                return m
        raise ExplainError(ErrorCode.MISSING_DATA, f"no model {model_id!r} in the project")

    def add_model(self, path, adapter: str, options: Optional[Dict[str, Any]] = None) -> ModelRecord:
        options = dict(options or {})
        if path in (None, "", "synthetic"):
            sha = "synthetic"
            path_text = "synthetic"
        else:
            p = Path(path).resolve()
            if not p.is_file():
                raise ExplainError(ErrorCode.MISSING_DATA, f"model not found: {p}")
            sha = sha256_file(p)
            path_text = str(p)
        for m in self.models:
            if m.sha256 == sha and m.adapter == adapter and dict(m.options) == options:
                return m
        # An event remembers the model it was picked with, and its candidate
        # index is only used with that model: an id is never given twice.
        used = ([m.model_id for m in self.models]
                + [m.get("model_id", "") for m in self.removed_entries("models")]
                + [e.get("model_id", "") for e in self._events])
        record = ModelRecord(_next_id("mdl-", 3, used), path_text, sha, adapter, options)
        self.data.setdefault("models", []).append(record.to_dict())
        return record

    def remove_models(self, model_ids: Iterable[str]) -> int:
        """Take models out of the project; the files are not touched.

        Events picked with a removed model stay: a run with another model
        finds each of them again by class and IoU (its candidate index is not
        used).  The entries move to ``removed``, so their ids are never given
        again.  Returns how many were removed.
        """
        drop = set(model_ids)
        for mid in drop:
            self.model(mid)
        stamp = now_iso()
        models = self.data.setdefault("models", [])
        gone = [dict(m, removed=stamp) for m in models if m["model_id"] in drop]
        self.data["models"] = [m for m in models if m["model_id"] not in drop]
        self._removed("models").extend(gone)
        return len(gone)

    # -- events ----------------------------------------------------------

    @property
    def events(self) -> List[Dict[str, str]]:
        return list(self._events)

    def event(self, event_id: str) -> Dict[str, str]:
        for e in self._events:
            if e["event_id"] == event_id:
                return e
        raise ExplainError(ErrorCode.MISSING_DATA, f"no event {event_id!r} in the project")

    def next_event_id(self) -> str:
        used = {e["event_id"] for e in self._events}
        n = len(self._events) + 1
        while f"ev-{n:04d}" in used:
            n += 1
        return f"ev-{n:04d}"

    def add_event(self, target: ExplanationTarget, note: str = "") -> str:
        if not target.event_id:
            from dataclasses import replace

            target = replace(target, event_id=self.next_event_id())
        if any(e["event_id"] == target.event_id for e in self._events):
            raise ExplainError(ErrorCode.INVALID_CONFIG, f"event {target.event_id} exists already")
        row = {k: schema.cell(v) for k, v in target_to_event_row(target, note).items()}
        row["schema_version"] = schema.SCHEMA_VERSION
        self._events.append(row)
        return target.event_id

    def remove_events(self, event_ids: Iterable[str]) -> int:
        drop = set(event_ids)
        before = len(self._events)
        self._events = [e for e in self._events if e["event_id"] not in drop]
        return before - len(self._events)

    def target(self, event_id: str) -> ExplanationTarget:
        """The :class:`ExplanationTarget` an events.csv row stands for."""
        e = self.event(event_id)
        s = self.source(e["source_id"])
        cand = None
        if str(e.get("candidate_index", "")).strip():
            loc = json.loads(e["candidate_locator"]) if e.get("candidate_locator") else {}
            cand = CandidateRef(int(float(e["candidate_index"])),
                                tuple((str(k), int(v)) for k, v in loc.items()))
        det = Detection((float(e["x1"]), float(e["y1"]), float(e["x2"]), float(e["y2"])),
                        float(e["score"]), int(float(e["class_id"])), e["class_name"], cand,
                        schema.obb_from_row(e))
        frame = FrameRef(
            source_id=s["source_id"], path=s["path"], kind=s["kind"],
            frame_index=int(float(e["frame_index"])),
            width=int(float(e["frame_width"])), height=int(float(e["frame_height"])),
            timestamp_ms=finite_or_none(schema.as_float(e.get("timestamp_ms"))),
            v_flip=schema.as_bool(e.get("v_flip")), h_flip=schema.as_bool(e.get("h_flip")),
            frame_sha1=e.get("frame_sha1") or None, access=e.get("frame_access") or "sequential",
        )
        return ExplanationTarget(
            event_id=e["event_id"], frame=frame, detection=det,
            detection_index=schema.as_int(e.get("detection_index")) if e.get("detection_index") else -1,
            target_class=int(float(e.get("target_class") or e["class_id"])),
            model_id=e.get("model_id", ""),
            track_id=schema.as_int(e.get("track_id")),
            origin=e.get("origin") or "adapter",
        )

    # -- runs ------------------------------------------------------------

    def run_dirs(self) -> List[Path]:
        if not self.runs_dir.is_dir():
            return []
        return sorted(p for p in self.runs_dir.iterdir() if (p / "manifest.json").is_file())
