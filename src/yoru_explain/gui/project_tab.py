# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Project tab: the project folder, its videos/images, its models."""

from __future__ import annotations

from pathlib import Path

import dearpygui.dearpygui as dpg

from yoru_explain.gui import widgets
from yoru_explain.gui.state import project_name
from yoru_explain.gui.widgets import SelectTable, short, tooltip

CHANNELS = ("YORU setting (YORU_YOLOV5_RGB)", "BGR (YORU v1)", "RGB")

#: YORU's detector backends (the names of yoru.libs.plugins.get_detector), as offered here.
BACKENDS = {
    "auto": "auto (the backend YORU picks for the file)",
    "yolov5": "yolov5 - YOLOv5 (YORU v1 models)",
    "ultralytics": "ultralytics - YOLOv8 / YOLO11 / YOLOv5u / YOLOv10 (+ OBB, seg, pose)",
    "rtdetr": "rtdetr - RT-DETR",
    "torchvision": "torchvision - Faster R-CNN / Mask R-CNN / SSD",
    "onnx": "onnx - ONNX exports (YOLOv5 / YOLOv8 / YOLO11)",
}


class ProjectTab:
    def __init__(self, app):
        self.app = app
        self.state = app.state
        self.popups = app.popups
        self.sources = SelectTable("proj_sources", ["id", "file", "kind", "size", "fps", "frames",
                                                    "flips", "seek check", "detections CSV"])
        self.models = SelectTable("proj_models", ["id", "file", "sha256", "adapter", "options"])
        st = self.state
        st.projectChanged.connect(self.refresh)
        st.adapterChanged.connect(self.refresh_model_status)
        st.tasks.busyChanged.connect(self._busy)
        st.tasks.finished.connect(self._task_done)
        st.tasks.failed.connect(self._task_failed)

    # -- building --------------------------------------------------------------

    def build(self) -> None:
        themes = self.app.themes
        with dpg.group(horizontal=True):
            t = dpg.add_text("Project:")
            dpg.bind_item_theme(t, themes.header)
            dpg.add_input_text(tag="proj_name", default_value=project_name(None), readonly=True,
                               width=-380)
            dpg.add_button(label="New project...", callback=lambda: self.new_project())
            dpg.add_button(label="Open project...", callback=lambda: self.open_project())
            dpg.add_button(label="Save", tag="proj_save", width=60, callback=lambda: self.save())
        dpg.add_spacer(height=4)

        widgets.heading("Videos and images", themes)
        self.sources.build()
        with dpg.group(horizontal=True):
            dpg.add_button(label="Add video / image...", callback=lambda: self.add_source())
            for tag, label in (("proj_vflip", "v_flip"), ("proj_hflip", "h_flip")):
                c = dpg.add_checkbox(label=label, tag=tag)
                tooltip(c, "Flip frames as YORU did before detecting. YORU's CSV files do not "
                           "record this.")
            b = dpg.add_button(label="Check seeking", tag="proj_seek",
                               callback=lambda: self.check_seeking())
            tooltip(b, "Compare seeked frames with frames decoded in order on 5 samples")
            dpg.add_button(label="Attach YORU detections CSV...", callback=lambda: self.attach_csv())
            b = dpg.add_button(label="Events from CSV...", callback=lambda: self.events_from_csv())
            tooltip(b, "Add the attached CSV's detections as events (filtered by class). A run "
                       "finds each box again on its frame or reports it.")
            dpg.add_button(label="Verify files", callback=lambda: self.verify())
        dpg.add_spacer(height=8)

        widgets.heading("Models", themes)
        self.models.build()
        with dpg.group(horizontal=True):
            dpg.add_text("type")
            c = dpg.add_combo(list(BACKENDS.values()), tag="proj_backend", default_value=BACKENDS["auto"],
                              width=440)
            tooltip(c, "The YORU backend that runs the model. auto: the one YORU's get_detector('auto', ...) "
                       "picks for the file (a YOLOv5 checkpoint is recognised by its contents, whatever "
                       "it is called; RT-DETR, YOLOv8 and YOLO11 by their names).")
            c = dpg.add_checkbox(label="black box", tag="proj_blackbox")
            tooltip(c, "Explain only what YORU's detect() returns: works for any model YORU loads; "
                       "re-detection with the reported confidence, no candidates, no Grad-CAM.")
            dpg.add_button(label="Add model...", callback=lambda: self.add_model())
            b = dpg.add_button(label="Add synthetic test model", callback=lambda: self.add_toy())
            tooltip(b, "A detector of red/green squares with known answers, for validating the "
                       "pipeline. Not a model of animals.")
        with dpg.group(horizontal=True):
            dpg.add_text("conf")
            c = dpg.add_input_float(tag="proj_conf", default_value=0.25, min_value=0.01,
                                    max_value=0.99, min_clamped=True, max_clamped=True,
                                    step=0.05, format="%.2f", width=110)
            tooltip(c, "Confidence threshold, as YORU applies it (YOLOv5: above 0.25 after NMS; "
                       "torchvision: scores strictly above it).")
            dpg.add_text("IoU")
            dpg.add_input_float(tag="proj_iou", default_value=0.45, min_value=0.05, max_value=0.95,
                                min_clamped=True, max_clamped=True, step=0.05, format="%.2f",
                                width=110)
            dpg.add_text("YOLOv5 channels")
            c = dpg.add_combo(list(CHANNELS), tag="proj_channel", default_value=CHANNELS[0], width=260)
            tooltip(c, "YOLOv5 only: YORU's BGR/RGB rule. The other backends convert to RGB themselves.")
        with dpg.group(horizontal=True):
            b = dpg.add_button(label="Load selected model", tag="proj_load",
                               callback=lambda: self.load_model())
            dpg.bind_item_theme(b, themes.primary_button)
            dpg.add_text("No model loaded.", tag="proj_model_status", wrap=900)
        self.refresh()

    def layout(self, width: int, height: int) -> None:
        table_h = max(90, (height - 262) // 2)
        dpg.configure_item(self.sources.tag, height=table_h)
        dpg.configure_item(self.models.tag, height=table_h)
        dpg.configure_item("proj_model_status", wrap=max(200, width - 220))

    # -- project ---------------------------------------------------------------

    def new_project(self):
        from yoru_explain.io.project import Project

        d = widgets.ask_directory("New project folder (empty or new)")
        if not d:
            return
        try:
            self.state.set_project(Project.create(d))
        except Exception as e:
            self.popups.error("New project", e)

    def open_project(self):
        from yoru_explain.io.project import Project

        d = widgets.ask_directory("Open project folder")
        if not d:
            return
        try:
            self.state.set_project(Project.open(d))
        except Exception as e:
            self.popups.error("Open project", e)

    def save(self):
        try:
            self.state.save_project()
        except Exception as e:
            self.popups.error("Save project", e)

    def _need_project(self):
        if self.state.project is None:
            self.popups.message("YORU-Explain", "Create or open a project first.")
            return None
        return self.state.project

    # -- sources ---------------------------------------------------------------

    def add_source(self, files=None):
        p = self._need_project()
        if p is None:
            return
        if files is None:
            files = widgets.ask_open_files("Add videos or images", widgets.VIDEO_TYPES)
        try:
            for f in files:
                p.add_source(f, v_flip=bool(dpg.get_value("proj_vflip")),
                             h_flip=bool(dpg.get_value("proj_hflip")))
            p.save()
        except Exception as e:
            self.popups.error("Add source", e)
        self.state.projectChanged.emit()   # Explore lists the new source too

    def check_seeking(self):
        p = self._need_project()
        sel = self.sources.selected
        if p is None:
            return
        if not sel:
            self.popups.message("YORU-Explain", "Select a video first.")
            return

        def work(source_id):
            with p.open_source(source_id) as r:
                return source_id, r.check_seeking()

        if not self.state.tasks.submit("check_seeking", work, sel[0]):
            self.popups.message("YORU-Explain", "Another task is running.")

    def attach_csv(self, path=None):
        p = self._need_project()
        sel = self.sources.selected
        if p is None:
            return
        if not sel:
            self.popups.message("YORU-Explain", "Select a video first.")
            return
        f = path or widgets.ask_open_file("YORU detections CSV", widgets.CSV_TYPES)
        if not f:
            return
        try:
            from yoru_explain.io.detections_csv import load_detections_csv

            layout, frames = load_detections_csv(f)
            s = p.source(sel[0])
            entry = {"path": str(Path(f).resolve()), "layout": layout,
                     "frames": len(frames), "detections": sum(len(v) for v in frames.values())}
            s.setdefault("detections_files", []).append(entry)
            p.save()
        except Exception as e:
            self.popups.error("Attach detections", e)
        self.state.projectChanged.emit()   # Explore draws the CSV's boxes too

    def events_from_csv(self):
        p = self._need_project()
        sel = self.sources.selected
        if p is None:
            return
        if not sel:
            self.popups.message("YORU-Explain", "Select a video first.")
            return
        s = p.source(sel[0])
        if not s.get("detections_files"):
            self.popups.message("YORU-Explain", "Attach a detections CSV first.")
            return
        self.popups.form("Events from CSV",
                         [("classes", "Classes to keep (comma-separated; empty = all):", ""),
                          ("limit", "At most this many events:", 100)],
                         lambda v: self.add_csv_events(s["source_id"], v["classes"], v["limit"]),
                         ok="Add events")

    def add_csv_events(self, source_id: str, classes: str, limit: int):
        from yoru_explain import api

        p = self.state.project
        s = p.source(source_id)
        limit = max(1, int(limit))
        names = [c.strip() for c in str(classes).split(",") if c.strip()] or None
        try:
            added = []
            for entry in s.get("detections_files", []):
                added += api.add_events_from_detections_csv(
                    p, source_id, entry["path"], model_id=self.state.model_id or "",
                    class_names=names, limit=limit - len(added))
                if len(added) >= limit:
                    break
            p.save()
        except Exception as e:
            self.popups.error("Events from CSV", e)
            return
        self.state.eventsChanged.emit()
        self.popups.message("Events from CSV", f"{len(added)} event(s) added.")

    def verify(self):
        p = self._need_project()
        if p is None:
            return
        from yoru_explain.core.hashing import sha256_file

        lines = []
        for s in p.sources:
            lines.append(f"{s['source_id']}: {p.verify_source(s['source_id'])['status']}  {s['path']}")
        for m in p.models:
            if m.sha256 == "synthetic":
                lines.append(f"{m.model_id}: synthetic")
                continue
            ok = Path(m.path).is_file() and sha256_file(m.path) == m.sha256
            lines.append(f"{m.model_id}: {'ok' if ok else 'missing or changed'}  {m.path}")
        self.popups.message("Files", "\n".join(lines) or "Nothing to verify.")

    # -- models ------------------------------------------------------------------

    def backend(self) -> str:
        """The backend chosen in the type combo (a YORU backend name, or ``auto``)."""
        label = dpg.get_value("proj_backend")
        return next((k for k, v in BACKENDS.items() if v == label), "auto")

    def _options(self, backend: str = "yolov5"):
        options = {"conf_thresh": round(float(dpg.get_value("proj_conf")), 4),
                   "iou_thresh": round(float(dpg.get_value("proj_iou")), 4)}
        if backend == "yolov5":
            options["rgb_input"] = {0: None, 1: False, 2: True}[CHANNELS.index(dpg.get_value("proj_channel"))]
        return options

    def add_model(self, path=None, backend=None, black_box=None):
        """Register a model file with the adapter for its YORU backend."""
        from yoru_explain.core.registry import resolve_model

        p = self._need_project()
        if p is None:
            return
        f = path or widgets.ask_open_file("Model file", widgets.MODEL_TYPES)
        if not f:
            return
        backend = backend or self.backend()
        black_box = bool(dpg.get_value("proj_blackbox")) if black_box is None else bool(black_box)
        try:
            adapter, resolved, extra = resolve_model(f, backend, black_box=black_box)
            p.add_model(f, adapter, {**self._options(resolved), **extra})
            p.save()
        except Exception as e:
            self.popups.error("Add model", e)
        self.refresh()

    def add_toy(self):
        p = self._need_project()
        if p is None:
            return
        p.add_model(None, "toy-grid", {})
        p.save()
        self.refresh()

    def load_model(self):
        p = self._need_project()
        if p is None:
            return
        sel = self.models.selected
        if not sel:
            if len(p.models) == 1:
                sel = [p.models[0].model_id]
            else:
                self.popups.message("YORU-Explain", "Select a model first.")
                return
        record = p.model(sel[0])
        device = self.state.device

        def work():
            from yoru_explain import api

            weights = None if record.sha256 == "synthetic" else record.path
            dev = device if record.adapter != "toy-grid" else ("cpu" if device == "auto" else device)
            return record.model_id, api.load_adapter(record.adapter, weights, device=dev,
                                                     **dict(record.options))

        if not self.state.tasks.submit("load_model", work):
            self.popups.message("YORU-Explain", "Another task is running.")
            return
        dpg.set_value("proj_model_status", f"Loading {record.model_id}...")

    def refresh_model_status(self):
        a = self.state.adapter
        if a is None:
            dpg.set_value("proj_model_status", "No model loaded.")
            return
        info = a.model_info()
        caps = a.capabilities
        grad = "Grad-CAM available" if caps.gradients else f"Grad-CAM unavailable ({caps.gradients_reason})"
        family = getattr(caps, "family", "") or a.name
        modes = ", ".join(getattr(caps, "identity_modes", ())) or "-"
        kind = getattr(caps, "candidate_kind", "")
        dpg.set_value("proj_model_status",
                      f"{self.state.model_id} loaded: {family} ({a.name}) on {info.get('device')}, "
                      f"{info.get('channel_order', '')}; classes {info.get('class_names')}; "
                      f"candidates: {kind or 'none (black box)'}; identity modes: {modes}; {grad}; "
                      f"batch-invariant: {caps.batch_invariant} (max diff {caps.batch_max_abs_diff})")

    # -- refresh -------------------------------------------------------------------

    def refresh(self):
        p = self.state.project
        dpg.set_value("proj_name", project_name(p))
        if p is None:
            self.sources.set_rows([])
            self.models.set_rows([])
            return
        rows, ids = [], []
        for s in p.sources:
            seek = s.get("seek_check") or {}
            csvs = ", ".join(Path(c["path"]).name for c in s.get("detections_files", []))
            rows.append([s["source_id"], short(s["path"]), s["kind"], f"{s['width']}x{s['height']}",
                         f"{s['fps']:.3f}" if s.get("fps") else "", s.get("frame_count"),
                         ("v" if s["v_flip"] else "") + ("h" if s["h_flip"] else "") or "-",
                         seek.get("status", ""), csvs])
            ids.append(s["source_id"])
        self.sources.set_rows(rows, ids)
        mrows, mids = [], []
        for m in p.models:
            mrows.append([m.model_id, short(m.path), m.sha256[:12], m.adapter, dict(m.options)])
            mids.append(m.model_id)
        self.models.set_rows(mrows, mids)

    def _busy(self, busy):
        for tag in ("proj_load", "proj_seek"):
            dpg.configure_item(tag, enabled=not busy)

    def _task_done(self, name, result):
        if name == "load_model":
            model_id, adapter = result
            self.state.set_adapter(adapter, model_id)
            warnings = list(getattr(adapter.capabilities, "warnings", ()) or ())
            if warnings:
                self.popups.message("Model loaded, with a warning", "\n\n".join(warnings),
                                    kind="warning")
        elif name == "check_seeking" and self.state.project is not None:
            source_id, check = result
            self.state.project.source(source_id)["seek_check"] = check
            self.state.project.save()
            self.refresh()
            if check.get("status") == "failed":
                self.popups.message(
                    "Seeking", f"Seeking is not exact for this video (frames {check['mismatched']}). "
                    "YORU-Explain reads analysed frames in order and verifies seeks by presentation "
                    "time, so results are not affected; plain seeking would show wrong frames.",
                    kind="warning")

    def _task_failed(self, name, text, exc):
        if name in ("load_model", "check_seeking"):
            if name == "load_model":
                dpg.set_value("proj_model_status", "Loading failed.")
            self.popups.error(name.replace("_", " ").capitalize(), exc)
