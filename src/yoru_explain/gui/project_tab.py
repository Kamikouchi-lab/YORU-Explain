# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Project tab (step 1): the project folder, its videos/images, its models.

Everything is saved as it changes.  New projects, models and their options
are asked for in dialogs, so the tab itself only holds what the project
contains and the few actions on it: add, remove, load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import dearpygui.dearpygui as dpg

from yoru_explain.gui import theme as th
from yoru_explain.gui import widgets
from yoru_explain.gui.state import project_name
from yoru_explain.gui.widgets import SelectTable, muted, short, tooltip

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
DIALOG_W = 600


def model_options(backend: str, conf: float = 0.25, iou: float = 0.45,
                  channel: str = CHANNELS[0]) -> Dict[str, object]:
    """The adapter options of a model record: YORU's thresholds (and YOLOv5's channel rule)."""
    options = {"conf_thresh": round(float(conf), 4), "iou_thresh": round(float(iou), 4)}
    if backend == "yolov5":
        options["rgb_input"] = {0: None, 1: False, 2: True}[CHANNELS.index(channel)]
    return options


def _default_location() -> str:
    docs = Path.home() / "Documents"
    return str(docs if docs.is_dir() else Path.home())


class ProjectTab:
    def __init__(self, app):
        self.app = app
        self.state = app.state
        self.popups = app.popups
        self.sources = SelectTable("proj_sources", ["id", "file", "status", "kind", "size", "fps",
                                                    "frames", "flips", "seek check", "YORU CSV"],
                                   multi=True, on_activate=lambda key: self._show_source(key))
        self.models = SelectTable("proj_models", ["id", "file", "status", "adapter", "options",
                                                  "sha256"],
                                  on_activate=lambda key: self.load_model(key))
        #: The open "New project" / "Add model" dialog's widgets (tests drive them).
        self.project_form: Dict[str, object] = {}
        self.model_form: Dict[str, object] = {}
        self._last_location: Optional[str] = None
        st = self.state
        st.projectChanged.connect(self.refresh)
        st.projectChanged.connect(self.refresh_model_status)
        st.modelsChanged.connect(self.refresh)
        st.modelsChanged.connect(self.refresh_model_status)
        st.adapterChanged.connect(self.refresh_model_status)
        st.adapterChanged.connect(self.refresh)
        st.tasks.busyChanged.connect(self._busy)
        st.tasks.finished.connect(self._task_done)
        st.tasks.failed.connect(self._task_failed)

    # -- building --------------------------------------------------------------

    def build(self) -> None:
        themes = self.app.themes
        with dpg.group(horizontal=True):
            t = dpg.add_text("Project")
            dpg.bind_item_theme(t, themes.header)
            dpg.add_input_text(tag="proj_name", default_value=project_name(None), readonly=True,
                               width=-290)
            b = dpg.add_button(label="New project...", tag="proj_new", callback=lambda: self.new_project())
            tooltip(b, "Make a new project folder: give it a name and choose where it goes.")
            b = dpg.add_button(label="Load project...", tag="proj_open",
                               callback=lambda: self.load_project())
            tooltip(b, "Open an existing project: choose the folder that holds its project.json.")
        muted("Changes are saved as you make them.", themes)
        dpg.add_spacer(height=6)

        widgets.heading("Videos and images", themes)
        muted("The videos (or images) whose detections you want to explain. If YORU flipped them, "
              "tick the same flips before adding. Double-click one to open it in 2 Pick events.",
              themes, tag="proj_src_help")
        self.sources.build()
        with dpg.group(horizontal=True):
            b = dpg.add_button(label="Add videos / images...", tag="proj_src_add",
                               callback=lambda: self.add_source())
            dpg.bind_item_theme(b, themes.primary_button)
            b = dpg.add_button(label="Remove", tag="proj_src_remove",
                               callback=lambda: self.remove_sources())
            tooltip(b, "Take the selected videos/images (Ctrl/Shift-click for several) out of the "
                       "project, with the events picked on them. The files are not deleted; run "
                       "results stay on disk.")
            dpg.add_spacer(width=18)
            muted("flips YORU used:", themes)
            for tag, label in (("proj_vflip", "v_flip"), ("proj_hflip", "h_flip")):
                c = dpg.add_checkbox(label=label, tag=tag)
                tooltip(c, "Applied to the files you add next: flip their frames as YORU did "
                           "before detecting. YORU's CSV files do not record this.")
        dpg.add_spacer(height=8)

        widgets.heading("Model", themes)
        muted("The model YORU detected with. The loaded model draws the boxes in 2 Pick events "
              "and is the one analysed in 3 Analyze. Double-click a model to load it.",
              themes, tag="proj_mdl_help")
        self.models.build()
        with dpg.group(horizontal=True):
            b = dpg.add_button(label="Add model...", tag="proj_mdl_add", callback=lambda: self.add_model())
            tooltip(b, "Choose the weights file (.pt / .pth / .onnx), then its type and YORU's "
                       "thresholds.")
            b = dpg.add_button(label="Remove", tag="proj_mdl_remove",
                               callback=lambda: self.remove_models())
            tooltip(b, "Take the selected model out of the project. Events picked with it stay; "
                       "the file is not deleted.")
            b = dpg.add_button(label="Load selected model", tag="proj_load",
                               callback=lambda: self.load_model())
            dpg.bind_item_theme(b, themes.primary_button)
        dpg.add_text("No model loaded.", tag="proj_model_status", wrap=900)
        self.refresh()
        self.refresh_model_status()

    def layout(self, width: int, height: int) -> None:
        table_h = max(90, (height - 330) // 2)
        dpg.configure_item(self.sources.tag, height=table_h)
        dpg.configure_item(self.models.tag, height=table_h)
        for tag in ("proj_model_status", "proj_src_help", "proj_mdl_help"):
            dpg.configure_item(tag, wrap=max(200, width - 20))

    # -- project ---------------------------------------------------------------

    def _location(self) -> str:
        if self._last_location:
            return self._last_location
        if self.state.project is not None:
            return str(Path(self.state.project.root).parent)
        return _default_location()

    def new_project(self) -> int:
        """Ask for the new project's name and location (a dialog)."""
        form: Dict[str, object] = {}
        self.project_form = form
        location = self._location()
        themes = self.app.themes

        def build(tag):
            form["tag"] = tag
            dpg.add_text("Project name")
            form["name"] = dpg.add_input_text(hint="e.g. courtship_2026", width=-1,
                                              callback=lambda: self._project_form_changed())
            dpg.add_spacer(height=4)
            dpg.add_text("Create it in")
            with dpg.group(horizontal=True):
                form["location"] = dpg.add_input_text(default_value=location, width=-110,
                                                      callback=lambda: self._project_form_changed())
                dpg.add_button(label="Browse...", width=100, callback=lambda: self._browse_location())
            dpg.add_spacer(height=4)
            form["preview"] = dpg.add_text("", wrap=DIALOG_W - 30)
            muted("A folder with the project's name is made there. It holds project.json, events.csv "
                  "and the runs; videos and models stay where they are.", themes, wrap=DIALOG_W - 30)
            dpg.add_spacer(height=6)
            with dpg.group(horizontal=True):
                b = dpg.add_button(label="Create", width=100, callback=lambda: self._accept_project_form())
                dpg.bind_item_theme(b, themes.primary_button)
                dpg.add_button(label="Cancel", width=100, callback=lambda: self.popups.close(tag))
            self._project_form_changed()

        return self.popups.dialog("New project", build, width=DIALOG_W)

    def _project_form_values(self):
        form, g = self.project_form, dpg.get_value
        return str(g(form["name"]) or "").strip(), str(g(form["location"]) or "").strip()

    def _project_form_changed(self) -> None:
        from yoru_explain.io.project import new_project_folder

        form, themes = self.project_form, self.app.themes
        if not form or not dpg.does_item_exist(form.get("preview", 0)):
            return
        name, location = self._project_form_values()
        if not name:
            text, theme = "Enter a name for the project.", themes.muted
        else:
            try:
                text, theme = f"Creates {new_project_folder(name, location)}", themes.ok
            except Exception as e:
                text, theme = str(getattr(e, "message", "") or e), themes.warn
        dpg.set_value(form["preview"], text)
        dpg.bind_item_theme(form["preview"], theme)

    def _browse_location(self) -> None:
        form = self.project_form
        d = widgets.ask_directory("Where to create the project", current=dpg.get_value(form["location"]))
        if d:
            dpg.set_value(form["location"], d)
            self._project_form_changed()

    def _accept_project_form(self) -> None:
        name, location = self._project_form_values()
        try:
            self.create_project(name, location)
        except Exception as e:
            # Stay open with the reason; the user corrects the name or the place.
            dpg.set_value(self.project_form["preview"], str(getattr(e, "message", "") or e))
            dpg.bind_item_theme(self.project_form["preview"], self.app.themes.warn)
            return
        self.popups.close(self.project_form["tag"])

    def create_project(self, name: str, location: str):
        """Create ``location/name`` as a new project and open it."""
        from yoru_explain.io.project import Project, new_project_folder

        folder = new_project_folder(name, location)
        project = Project.create(folder, name=name.strip())
        self._last_location = str(Path(location))
        self.state.set_project(project)
        return project

    def load_project(self, path: Optional[str] = None) -> None:
        from yoru_explain.io.project import PROJECT_FILE, Project

        d = path or widgets.ask_directory("Load project: choose the project folder (it holds project.json)",
                                          current=self._location())
        if not d:
            return
        if not (Path(d) / PROJECT_FILE).is_file():
            self.popups.message("Load project", f"{d} is not a YORU-Explain project: it has no "
                                f"{PROJECT_FILE}. Choose the project's own folder, or make a new "
                                "project with New project.", kind="warning")
            return
        try:
            self.state.set_project(Project.open(d))
            self._last_location = str(Path(d).parent)
        except Exception as e:
            self.popups.error("Load project", e)

    def save(self):
        try:
            self.state.save_project()
        except Exception as e:
            self.popups.error("Save project", e)

    def _need_project(self):
        if self.state.project is None:
            self.popups.message("YORU-Explain", "Create a new project or load one first "
                                                "(New project... / Load project...).")
            return None
        return self.state.project

    def _idle(self) -> bool:
        if self.state.tasks.busy:
            self.popups.message("YORU-Explain", f"Wait until the running task "
                                                f"({self.state.tasks.current}) has finished.")
            return False
        return True

    # -- sources ---------------------------------------------------------------

    def add_source(self, files=None):
        p = self._need_project()
        if p is None:
            return
        if files is None:
            files = widgets.ask_open_files("Add videos or images", widgets.VIDEO_TYPES)
        added = []
        try:
            for f in files:
                added.append(p.add_source(f, v_flip=bool(dpg.get_value("proj_vflip")),
                                          h_flip=bool(dpg.get_value("proj_hflip")))["source_id"])
            p.save()
        except Exception as e:
            self.popups.error("Add videos / images", e)
        self.state.projectChanged.emit()   # Pick events lists the new source too
        if added:
            self.sources.select(added)

    def remove_sources(self, source_ids=None, *, confirm: bool = True) -> None:
        p = self._need_project()
        if p is None:
            return
        ids = list(source_ids or self.sources.selected)
        if not ids:
            self.popups.message("Remove", "Select the videos/images to remove first "
                                          "(Ctrl/Shift-click for several).")
            return
        if not self._idle():
            return
        events = p.events_of_sources(ids)

        def remove():
            p.remove_sources(ids)
            p.save()
            self.state.projectChanged.emit()
            self.state.eventsChanged.emit()

        if not confirm:
            remove()
            return
        names = ", ".join(Path(p.source(i)["path"]).name for i in ids)
        text = f"Remove {names} from the project?"
        if events:
            text += f"\n\nThe {len(events)} event(s) picked on {'it' if len(ids) == 1 else 'them'} " \
                    "are removed too."
        text += "\n\nThe files themselves are not deleted, and run results stay on disk."
        self.popups.confirm("Remove videos / images", text, remove, yes="Remove", no="Cancel")

    def _show_source(self, source_id) -> None:
        self.app.explore_tab.show_source(source_id)
        self.app.show_tab("explore")

    def check_seeking(self):
        """Compare seeked frames with frames decoded in order (5 samples) for the selected video."""
        p = self._need_project()
        sel = self.sources.selected
        if p is None:
            return
        if not sel:
            self.popups.message("Check seeking", "Select a video in 1 Project first.")
            return

        def work(source_id):
            with p.open_source(source_id) as r:
                return source_id, r.check_seeking()

        if not self.state.tasks.submit("check_seeking", work, sel[0]):
            self.popups.message("YORU-Explain", "Another task is running.")

    # -- models ------------------------------------------------------------------

    def add_model(self, path: Optional[str] = None) -> Optional[int]:
        """Choose a weights file, then its type and thresholds (a dialog)."""
        p = self._need_project()
        if p is None:
            return None
        f = path or widgets.ask_open_file("Add model: the weights YORU detected with",
                                          widgets.MODEL_TYPES)
        if not f:
            return None
        form: Dict[str, object] = {"path": str(f)}
        self.model_form = form
        themes = self.app.themes

        def build(tag):
            form["tag"] = tag
            dpg.add_text(Path(f).name)
            dpg.bind_item_theme(dpg.last_item(), themes.header)
            muted(str(Path(f).parent), themes, wrap=DIALOG_W - 30)
            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                dpg.add_text("type    ")
                form["backend"] = dpg.add_combo(list(BACKENDS.values()), default_value=BACKENDS["auto"],
                                                width=-1)
            tooltip(form["backend"], "The YORU backend that runs the model. auto: the one YORU's "
                    "get_detector('auto', ...) picks for the file (a YOLOv5 checkpoint is recognised "
                    "by its contents, whatever it is called; RT-DETR, YOLOv8 and YOLO11 by their "
                    "names).")
            form["black_box"] = dpg.add_checkbox(label="black box: explain only what YORU's "
                                                       "detect() returns")
            tooltip(form["black_box"], "Works for any model YORU loads: re-detection with the "
                    "reported confidence, no candidates, no Grad-CAM.")
            with dpg.group(horizontal=True):
                dpg.add_text("conf    ")
                form["conf"] = dpg.add_input_float(default_value=0.25, min_value=0.01, max_value=0.99,
                                                   min_clamped=True, max_clamped=True, step=0.05,
                                                   format="%.2f", width=120)
                tooltip(form["conf"], "Confidence threshold, as YORU applies it (YOLOv5: above 0.25 "
                        "after NMS; torchvision: scores strictly above it).")
                dpg.add_text("IoU")
                form["iou"] = dpg.add_input_float(default_value=0.45, min_value=0.05, max_value=0.95,
                                                  min_clamped=True, max_clamped=True, step=0.05,
                                                  format="%.2f", width=120)
                tooltip(form["iou"], "NMS IoU threshold, as YORU applies it.")
            with dpg.group(horizontal=True):
                dpg.add_text("channels")
                form["channel"] = dpg.add_combo(list(CHANNELS), default_value=CHANNELS[0], width=-1)
                tooltip(form["channel"], "YOLOv5 only: YORU's BGR/RGB rule. The other backends "
                        "convert to RGB themselves.")
            muted("Use the thresholds YORU ran the model with; the defaults are YORU's.", themes,
                  wrap=DIALOG_W - 30)
            form["load"] = dpg.add_checkbox(label="load it now", default_value=True)
            dpg.add_spacer(height=6)
            with dpg.group(horizontal=True):
                b = dpg.add_button(label="Add", width=100, callback=lambda: self._accept_model_form())
                dpg.bind_item_theme(b, themes.primary_button)
                dpg.add_button(label="Cancel", width=100, callback=lambda: self.popups.close(tag))

        return self.popups.dialog("Add model", build, width=DIALOG_W, text=str(f))

    def _accept_model_form(self) -> None:
        form, g = self.model_form, dpg.get_value
        label = g(form["backend"])
        backend = next((k for k, v in BACKENDS.items() if v == label), "auto")
        self.popups.close(form["tag"])
        self.register_model(form["path"], backend, black_box=bool(g(form["black_box"])),
                            conf=g(form["conf"]), iou=g(form["iou"]), channel=g(form["channel"]),
                            load=bool(g(form["load"])))

    def register_model(self, path, backend: str = "auto", *, black_box: bool = False,
                       conf: float = 0.25, iou: float = 0.45, channel: str = CHANNELS[0],
                       load: bool = False):
        """Add a model file with the adapter for its YORU backend; *load* it too."""
        from yoru_explain.core.registry import resolve_model

        p = self._need_project()
        if p is None:
            return None
        record = None
        try:
            adapter, resolved, extra = resolve_model(path, backend or "auto", black_box=black_box)
            record = p.add_model(path, adapter, {**model_options(resolved, conf, iou, channel), **extra})
            p.save()
        except Exception as e:
            self.popups.error("Add model", e)
        self.state.modelsChanged.emit()
        if record is not None:
            self.models.select([record.model_id])
            if load:
                self.load_model(record.model_id)
        return record

    def add_toy(self):
        """The synthetic test detector (red/green squares with known answers)."""
        p = self._need_project()
        if p is None:
            return
        record = p.add_model(None, "toy-grid", {})
        p.save()
        self.state.modelsChanged.emit()
        self.models.select([record.model_id])

    def remove_models(self, model_ids=None, *, confirm: bool = True) -> None:
        p = self._need_project()
        if p is None:
            return
        ids = list(model_ids or self.models.selected)
        if not ids:
            self.popups.message("Remove", "Select the model to remove first.")
            return
        if not self._idle():
            return

        def remove():
            p.remove_models(ids)
            p.save()
            if self.state.model_id in ids:
                self.state.set_adapter(None, None)
            self.state.modelsChanged.emit()

        if not confirm:
            remove()
            return
        names = ", ".join(f"{i} ({Path(p.model(i).path).name})" for i in ids)
        loaded = " It is the loaded model and is unloaded." if self.state.model_id in ids else ""
        self.popups.confirm("Remove model", f"Remove {names} from the project?{loaded}\n\nEvents "
                            "picked with it stay (another model finds each of them again by class "
                            "and IoU). The file is not deleted, and run results stay on disk.",
                            remove, yes="Remove", no="Cancel")

    def load_model(self, model_id: Optional[str] = None):
        p = self._need_project()
        if p is None:
            return
        if model_id is None:
            sel = self.models.selected
            if sel:
                model_id = sel[0]
            elif len(p.models) == 1:
                model_id = p.models[0].model_id
            else:
                self.popups.message("YORU-Explain", "Select a model first." if p.models else
                                    "Add a model first (Add model...).")
                return
        record = p.model(model_id)
        if record.sha256 != "synthetic" and not Path(record.path).is_file():
            self.popups.message("Load model", f"The model file is missing: {record.path}", kind="warning")
            return
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
        dpg.set_value("proj_model_status", f"Loading {record.model_id} ({Path(record.path).name})...")
        dpg.bind_item_theme("proj_model_status", self.app.themes.muted)

    def refresh_model_status(self):
        a = self.state.adapter
        themes = self.app.themes
        if a is None:
            p = self.state.project
            text = ("No model loaded. Select a model and press Load selected model (or double-click it)."
                    if p is not None and p.models else "No model loaded.")
            dpg.set_value("proj_model_status", text)
            dpg.bind_item_theme("proj_model_status", themes.muted)
            return
        info = a.model_info()
        caps = a.capabilities
        grad = "Grad-CAM available" if caps.gradients else f"Grad-CAM unavailable ({caps.gradients_reason})"
        family = getattr(caps, "family", "") or a.name
        modes = ", ".join(getattr(caps, "identity_modes", ())) or "-"
        kind = getattr(caps, "candidate_kind", "")
        dpg.set_value("proj_model_status",
                      f"Loaded {self.state.model_id}: {family} ({a.name}) on {info.get('device')}, "
                      f"{info.get('channel_order', '')}; classes {info.get('class_names')}; "
                      f"candidates: {kind or 'none (black box)'}; identity modes: {modes}; {grad}; "
                      f"batch-invariant: {caps.batch_invariant} (max diff {caps.batch_max_abs_diff})")
        dpg.bind_item_theme("proj_model_status", themes.ok)

    # -- refresh -------------------------------------------------------------------

    def refresh(self):
        p = self.state.project
        dpg.set_value("proj_name", project_name(p))
        if p is None:
            self.sources.set_rows([])
            self.models.set_rows([])
            return
        rows, ids, colors = [], [], []
        for s in p.sources:
            seek = s.get("seek_check") or {}
            csvs = ", ".join(Path(c["path"]).name for c in s.get("detections_files", []))
            status = p.source_file_status(s["source_id"])
            rows.append([s["source_id"], short(s["path"]), status, s["kind"],
                         f"{s['width']}x{s['height']}", f"{s['fps']:.3f}" if s.get("fps") else "",
                         s.get("frame_count"),
                         ("v" if s["v_flip"] else "") + ("h" if s["h_flip"] else "") or "-",
                         seek.get("status", ""), csvs])
            ids.append(s["source_id"])
            colors.append(None if status == "ok" else th.TEXT_WARN)
        self.sources.set_rows(rows, ids, colors)
        mrows, mids, mcolors = [], [], []
        for m in p.models:
            missing = m.sha256 != "synthetic" and not Path(m.path).is_file()
            loaded = m.model_id == self.state.model_id and self.state.adapter is not None
            status = "loaded" if loaded else ("missing" if missing else "")
            mrows.append([m.model_id, short(m.path), status, m.adapter, dict(m.options), m.sha256[:12]])
            mids.append(m.model_id)
            mcolors.append(th.TEXT_OK if loaded else (th.TEXT_WARN if missing else None))
        self.models.set_rows(mrows, mids, mcolors)

    def _busy(self, busy):
        for tag in ("proj_load", "proj_src_remove", "proj_mdl_remove"):
            dpg.configure_item(tag, enabled=not busy)

    def _task_done(self, name, result):
        if name == "load_model":
            model_id, adapter = result
            p = self.state.project
            try:
                record = p.model(model_id) if p is not None else None
            except Exception:
                record = None
            if p is not None and record is None:
                return   # the project changed while the model loaded
            self.state.set_adapter(adapter, model_id, record)
            warnings = list(getattr(adapter.capabilities, "warnings", ()) or ())
            sha = adapter.model_info().get("sha256")
            if record is not None and sha and record.sha256 not in (sha, "synthetic"):
                warnings.append(f"{Path(record.path).name} has changed since it was added to the "
                                f"project (sha256 {sha[:12]}, recorded {record.sha256[:12]}), so "
                                "analyses refuse it. If the new file is intended, remove the model "
                                "and add the file again.")
            if warnings:
                self.popups.message("Model loaded, with a warning", "\n\n".join(warnings),
                                    kind="warning")
        elif name == "check_seeking" and self.state.project is not None:
            source_id, check = result
            try:
                self.state.project.source(source_id)["seek_check"] = check
            except Exception:
                return   # removed meanwhile
            self.state.project.save()
            self.refresh()
            if check.get("status") == "failed":
                self.popups.message(
                    "Seeking", f"Seeking is not exact for this video (frames {check['mismatched']}). "
                    "YORU-Explain reads analysed frames in order and verifies seeks by presentation "
                    "time, so results are not affected; plain seeking would show wrong frames.",
                    kind="warning")
            else:
                self.popups.message("Seeking", f"Seek check: {check.get('status')}.")

    def _task_failed(self, name, text, exc):
        if name in ("load_model", "check_seeking"):
            if name == "load_model":
                dpg.set_value("proj_model_status", "Loading failed.")
                dpg.bind_item_theme("proj_model_status", self.app.themes.error)
            self.popups.error(name.replace("_", " ").capitalize(), exc)
