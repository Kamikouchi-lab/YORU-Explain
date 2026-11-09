# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Pick events tab (step 2): browse frames, pick detections, look at their results.

Frames are decoded exactly (in order, or by a seek verified against
presentation times) on a worker thread, and the loaded model runs on each
shown frame, so the boxes are the detector's own.  Clicking a box selects
it; "Add selected as event" stores it -- with its pre-NMS candidate and the
frame's hash -- for analysis.  A YORU detections CSV gives events in bulk
(Import from YORU CSV...): from the first frame on or at random, so many in
all or so many of each class; its boxes are drawn in light blue.
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dearpygui.dearpygui as dpg

from yoru_explain.gui import widgets
from yoru_explain.gui.frame_view import VIEW_MODES, FrameView, compose, map_value, pick_box
from yoru_explain.gui.widgets import SelectTable, fmt, muted, tooltip
from yoru_explain.gui.worker import FrameRequest

#: What each detections-CSV layout is, in words.
CSV_LAYOUTS = {
    "yoru-analysis": "YORU video-analysis CSV",
    "yoru-tracker": "YORU-Tracker detections CSV",
    "yoru-tracker-tracks": "YORU-Tracker tracks CSV",
}
CSV_DIALOG_W = 620
ALL_FRAMES = "every frame"
#: The import dialog's "Pick" and "How many" choices.
FIRST = "from the first frame on"
RANDOM = "at random"
IN_TOTAL = "in total"
PER_CLASS = "per class"
PER_CLASS_DEFAULT = 10

logger = logging.getLogger("yoru_explain.gui")

DISCLAIMER = ("Perturbation effects and attributions describe the model's score for this detection. "
              "They are not a biological interpretation.")
#: Wait this long after the last frame change before decoding (slider drags).
DEBOUNCE_S = 0.12
NAV_W = 0.58          # share of the width for the frame
SCALE_W = 74


def _stops(lut, n: int = 17):
    """A BGR colour table as RGBA stops for a DearPyGui colormap."""
    idx = [round(i * (len(lut) - 1) / (n - 1)) for i in range(n)]
    return [[int(lut[i][2]), int(lut[i][1]), int(lut[i][0]), 255] for i in idx]


class ExploreTab:
    def __init__(self, app):
        self.app = app
        self.state = app.state
        self.popups = app.popups
        self.frame = None            # LoadedFrame on display
        self.detections: List = []
        self._det_adapter = None     # the adapter that produced self.detections
        self.selected: Optional[int] = None
        self._indexed: Dict[str, dict] = {}
        self._imported: Dict[str, dict] = {}
        self._pending_select = None  # (event target) to select once its frame arrives
        self._result = None          # (reader, row, map dict)
        self._source_ids: List[str] = []
        self._source_labels: List[str] = []
        self._result_data: List[Tuple[str, str]] = []
        self._result_labels: List[str] = []
        self._max_index = 0
        self._request_at: Optional[float] = None
        self._shown: Tuple[Optional[object], str] = (None, "delta")   # (map values, kind)
        self._info_lines: List[Tuple[str, Optional[int]]] = []
        self._wrap = 520
        self._project_ref = None     # the project whose sources _indexed describes
        #: The open "Import from YORU CSV" dialog's widgets (tests drive them).
        self.csv_form: Dict[str, object] = {}
        self.view = FrameView("exp_view", app.themes, on_click=self._clicked)
        self.det_table = SelectTable("exp_dets", ["#", "class", "score", "box (x1, y1, x2, y2)",
                                                  "candidate"],
                                     height=150, on_select=lambda keys: self._det_row_selected(keys))
        self.events = SelectTable("exp_events", ["event", "source", "frame", "class", "score",
                                                 "candidate", "origin", "note"],
                                  multi=True, height=200, sortable=True,
                                  on_select=lambda keys: self._events_clicked(keys),
                                  on_activate=lambda key: self.goto_selected_event())
        st = self.state
        st.projectChanged.connect(self.refresh_sources)
        st.eventsChanged.connect(self.refresh_events)
        st.runsChanged.connect(self._events_selected)
        st.adapterChanged.connect(self._request_frame)
        st.frames.frameReady.connect(self._frame_ready)
        st.frames.indexReady.connect(self._index_ready)
        st.showResult.connect(self.show_result)

    # -- building ----------------------------------------------------------------

    def build(self, texture_registry) -> None:
        from yoru_explain.visualization.colormaps import DIVERGING, SEQUENTIAL

        themes = self.app.themes
        with dpg.colormap_registry():
            self._cmaps = {"delta": dpg.add_colormap(_stops(DIVERGING), False),
                           "cam": dpg.add_colormap(_stops(SEQUENTIAL), False)}
        with dpg.group(horizontal=True):
            with dpg.child_window(tag="exp_left", border=False, no_scrollbar=True):
                with dpg.group(horizontal=True):
                    dpg.add_text("Source")
                    dpg.add_combo([], tag="exp_source", width=260,
                                  callback=lambda: self._source_changed())
                    dpg.add_text("Frame")
                    dpg.add_button(label="<", width=26, callback=lambda: self.step(-1))
                    dpg.add_input_int(tag="exp_index", width=110, step=0, min_value=0,
                                      callback=lambda s, v: self.set_index(v))
                    dpg.add_button(label=">", width=26, callback=lambda: self.step(1))
                    dpg.add_slider_int(tag="exp_slider", width=-1, min_value=0, max_value=0,
                                       callback=lambda s, v: self.set_index(v))
                with dpg.group(horizontal=True):
                    self.view.build(texture_registry)
                    with dpg.group(tag="exp_scale_group", show=False):
                        dpg.add_colormap_scale(tag="exp_scale", colormap=self._cmaps["delta"],
                                               width=SCALE_W, height=300)
                with dpg.group(horizontal=True):
                    dpg.add_text("View")
                    dpg.add_combo(list(VIEW_MODES), tag="exp_mode", default_value=VIEW_MODES[2],
                                  width=130, callback=lambda: self.redraw())
                    dpg.add_text("overlay alpha")
                    dpg.add_slider_float(tag="exp_alpha", default_value=0.5, min_value=0.1,
                                         max_value=0.9, width=110, format="%.2f",
                                         callback=lambda: self.redraw())
                    b = dpg.add_button(label="Fit", width=40, callback=lambda: self.view.fit())
                    tooltip(b, "Show the whole frame (or double-click it). Wheel zooms, drag pans, "
                               "right-drag zooms into a box.")
                    muted("", themes, tag="exp_hover")
                dpg.add_text("", tag="exp_frame_info")
                with dpg.group(horizontal=True, tag="exp_csv_row", show=False):
                    muted("", themes, tag="exp_csv_info")
                    b = dpg.add_button(label="Detach", tag="exp_csv_detach", small=True,
                                       callback=lambda: self.detach_csv())
                    tooltip(b, "Stop drawing this CSV's boxes. Events already imported from it stay.")
            with dpg.child_window(tag="exp_right", border=False):
                t = dpg.add_text("Detections on this frame")
                dpg.bind_item_theme(t, themes.header)
                muted("Click a box on the frame (or a row here), then add it as an event.", themes,
                      tag="exp_dets_help")
                self.det_table.build()
                with dpg.group(horizontal=True):
                    b = dpg.add_button(label="Add selected as event", tag="exp_add",
                                       callback=lambda: self.add_events(False))
                    dpg.bind_item_theme(b, themes.primary_button)
                    dpg.add_button(label="Add all on this frame", callback=lambda: self.add_events(True))
                    b = dpg.add_button(label="Import from YORU CSV...", tag="exp_import",
                                       callback=lambda: self.import_csv())
                    tooltip(b, "Make events in bulk from the detections YORU saved for this video "
                               "(its video-analysis <video>.csv, or YORU-Tracker's tracks / "
                               "detections CSV). You choose the classes, the frames, and how "
                               "many: from the first frame on or at random, in total or per "
                               "class.")
                dpg.add_input_text(tag="exp_note", hint="note for new events (optional)", width=-1)
                dpg.add_spacer(height=4)
                t = dpg.add_text("Events", tag="exp_events_title")
                dpg.bind_item_theme(t, themes.header)
                muted("Select the events to analyse in 3 Analyze: click one to see its frame, "
                      "Ctrl/Shift-click to select several.", themes, tag="exp_events_help")
                self.events.build()
                b = dpg.add_button(label="Remove selected events", callback=lambda: self.remove_events())
                tooltip(b, "Results of earlier runs stay on disk.")
                dpg.add_spacer(height=4)
                t = dpg.add_text("Results of the selected event")
                dpg.bind_item_theme(t, themes.header)
                dpg.add_combo([], tag="exp_results", width=-1, callback=lambda: self._result_chosen())
                with dpg.child_window(tag="exp_info", height=-1, border=True):
                    pass
        self.refresh_sources()
        self._show_info()

    def layout(self, width: int, height: int) -> None:
        left = int(width * NAV_W)
        right = max(300, width - left - 16)
        dpg.configure_item("exp_left", width=left, height=height)
        dpg.configure_item("exp_right", width=right, height=height)
        plot_h = max(160, height - (124 if dpg.is_item_shown("exp_csv_row") else 96))
        scale = SCALE_W + 8 if dpg.is_item_shown("exp_scale_group") else 0
        self.view.resize(left - scale - 8, plot_h)
        dpg.configure_item("exp_scale", height=plot_h)
        rest = max(300, height - 320)
        self.det_table.height = max(90, int(rest * 0.32))
        dpg.configure_item(self.det_table.tag, height=self.det_table.height)
        dpg.configure_item(self.events.tag, height=max(110, int(rest * 0.42)))
        for tag in ("exp_events_help", "exp_dets_help"):
            dpg.configure_item(tag, wrap=right - 10)
        self._wrap = right - 30
        self._render_info()
        self._layout_size = (width, height)

    # -- sources and frames ------------------------------------------------------

    def _current_source(self):
        p = self.state.project
        label = dpg.get_value("exp_source")
        if p is None or label not in self._source_labels:
            return None
        try:
            return p.source(self._source_ids[self._source_labels.index(label)])
        except Exception:
            return None

    def _select_source(self, source_id: str) -> bool:
        if source_id not in self._source_ids:
            return False
        dpg.set_value("exp_source", self._source_labels[self._source_ids.index(source_id)])
        return True

    @property
    def index(self) -> int:
        return int(dpg.get_value("exp_index"))

    def refresh_sources(self):
        previous = self._current_source()
        p = self.state.project
        self._source_ids, self._source_labels = [], []
        if p is not None:
            for s in p.sources:
                self._source_ids.append(s["source_id"])
                self._source_labels.append(f"{s['source_id']}  {Path(s['path']).name}")
        dpg.configure_item("exp_source", items=self._source_labels)
        keep = previous is not None and previous["source_id"] in self._source_ids
        if keep:
            self._select_source(previous["source_id"])
        else:
            dpg.set_value("exp_source", self._source_labels[0] if self._source_labels else "")
        if p is not self._project_ref:
            # Another project: the frame service forgot its readers and their time indexes.
            self._project_ref = p
            self._indexed.clear()
        else:
            # The same project changed (a source added or removed, a CSV attached):
            # the time indexes of the sources still there stay valid.
            self._indexed = {k: v for k, v in self._indexed.items() if k in self._source_ids}
        self._imported.clear()
        self._source_changed(keep_index=keep)

    def show_source(self, source_id: str) -> None:
        """Show *source_id* (from its first frame unless it is already shown)."""
        s = self._current_source()
        if s is not None and s["source_id"] == source_id:
            return
        if self._select_source(source_id):
            self._source_changed()

    def _source_changed(self, keep_index: bool = False):
        s = self._current_source()
        n = max(0, int(s.get("frame_count") or 1) - 1) if s else 0
        summary = self._indexed.get(s["source_id"], {}) if s else {}
        if summary.get("status") == "ok":
            n = int(summary["frames"]) - 1   # the decoded frame count, not the container's estimate
        self._max_index = n
        dpg.configure_item("exp_slider", max_value=n)
        dpg.configure_item("exp_index", max_value=n)
        index = min(self.index, n) if keep_index else 0
        dpg.set_value("exp_index", index)
        dpg.set_value("exp_slider", index)
        self.frame = None
        self.detections = []
        self.selected = None
        self.view.set_image(None)
        self._fill_detections()
        if s is None:
            dpg.set_value("exp_frame_info", "No video or image: add one in 1 Project."
                          if self.state.project is not None else
                          "Create or load a project in 1 Project.")
            self._show_csv_row(None)
            return
        if s["kind"] == "video" and s["source_id"] not in self._indexed:
            self._indexed[s["source_id"]] = {"status": "building"}
            self.state.frames.build_index(self._req(s, 0))
        self._load_imported(s)
        self._show_csv_row(s)
        self._request_frame()

    def _show_csv_row(self, s) -> None:
        files = (s or {}).get("detections_files") or []
        if files:
            n = sum(int(e.get("detections") or 0) for e in files)
            names = ", ".join(Path(e["path"]).name for e in files)
            dpg.set_value("exp_csv_info", f"Light-blue boxes: {n} detections from {names}")
        if bool(files) != dpg.is_item_shown("exp_csv_row"):
            dpg.configure_item("exp_csv_row", show=bool(files))
            size = getattr(self, "_layout_size", None)
            if size is not None:
                self.layout(*size)

    def _load_imported(self, s):
        from yoru_explain.io.detections_csv import load_detections_csv

        sid = s["source_id"]
        if sid in self._imported:
            return
        frames: dict = {}
        for entry in s.get("detections_files", []):
            try:
                _, f = load_detections_csv(entry["path"])
                for k, v in f.items():
                    frames.setdefault(k, []).extend(v)
            except Exception as e:
                self.popups.error("Detections CSV", e)
        self._imported[sid] = frames

    def _req(self, s, index):
        return FrameRequest(s["source_id"], s["path"], s["v_flip"], s["h_flip"], int(index),
                            detect=self.state.adapter is not None)

    def set_index(self, value) -> None:
        """Show frame *value* (clamped); decoded once the value stops changing."""
        value = max(0, min(int(value), self._max_index))
        dpg.set_value("exp_index", value)
        dpg.set_value("exp_slider", value)
        self._request_at = time.monotonic() + DEBOUNCE_S

    def step(self, delta: int) -> None:
        self.set_index(self.index + delta)

    def on_key(self, key) -> None:
        if any(dpg.is_item_active(t) for t in ("exp_note", "exp_index")):
            return
        if key == dpg.mvKey_Left:
            self.step(-1)
        elif key == dpg.mvKey_Right:
            self.step(1)

    def _request_frame(self):
        self._request_at = None
        s = self._current_source()
        if s is None:
            return
        dpg.set_value("exp_frame_info", f"loading frame {self.index}...")
        self.state.frames.request(self._req(s, self.index))

    def _index_ready(self, source_id, summary):
        self._indexed[source_id] = summary
        s = self._current_source()
        if s is not None and s["source_id"] == source_id and summary.get("status") == "ok":
            # The decoded frame count, not the container's estimate.
            self._max_index = int(summary["frames"]) - 1
            dpg.configure_item("exp_slider", max_value=self._max_index)
            dpg.configure_item("exp_index", max_value=self._max_index)
        self._update_frame_info()

    def _frame_ready(self, result):
        s = self._current_source()
        req = result.request
        if s is None or req.source_id != s["source_id"] or req.index != self.index:
            return  # an older request
        if result.error:
            dpg.set_value("exp_frame_info", f"frame {req.index}: {result.exception}")
            return
        self.frame = result.frame
        self.detections = list(result.detections)
        self._det_adapter = result.adapter
        self.selected = None
        if self._pending_select is not None:
            self.selected = self._match_pending()
            self._pending_select = None
        self._fill_detections()
        self._update_frame_info()
        self.redraw()

    def _update_frame_info(self):
        f = self.frame
        s = self._current_source()
        if f is None or s is None:
            return
        idx = self._indexed.get(s["source_id"], {})
        ts = f"{f.timestamp_ms / 1000:.3f} s" if f.timestamp_ms is not None else "-"
        model = self.state.model_id or "no model loaded"
        dpg.set_value("exp_frame_info", f"frame {f.frame_index}  t={ts}  access={f.access}  "
                                        f"time index: {idx.get('status', '-')}  model: {model}")

    def tick(self, now: float) -> None:
        if self._request_at is not None and time.monotonic() >= self._request_at:
            self._request_frame()
        pos = self.view.mouse_frame_pos()
        if pos is None:
            text = ""
        else:
            values, kind = self._shown
            text = f"x {pos[0]:.1f}  y {pos[1]:.1f}"
            if values is not None:
                v = map_value(values, *pos)
                name = "delta" if kind == "delta" else "CAM"
                text += f"   {name} {fmt(v, 4) if v is not None else '(not scored)'}"
        if dpg.get_value("exp_hover") != text:
            dpg.set_value("exp_hover", text)

    # -- detections -----------------------------------------------------------------

    def _fill_detections(self):
        rows = []
        for i, d in enumerate(self.detections):
            box = ", ".join(f"{v:.1f}" for v in d.box)
            if d.obb is not None:
                box += f"  (rotated {math.degrees(d.obb[4]):.1f} deg)"
            rows.append([i, d.class_name, f"{d.score:.4f}", box,
                         d.candidate.describe() if d.candidate else "-"])
        self.det_table.set_rows(rows, list(range(len(rows))))
        self.det_table.select([self.selected] if self.selected is not None else [])

    def _clicked(self, x, y):
        i = pick_box([d.box for d in self.detections], x, y, obbs=[d.obb for d in self.detections])
        if i is not None:
            self.selected = i
            self.det_table.select([i])
            self.redraw()

    def _det_row_selected(self, keys):
        if keys:
            self.selected = keys[0]
            self.redraw()

    def add_events(self, all_on_frame: bool):
        from yoru_explain import api

        p, a, f, s = self.state.project, self.state.adapter, self.frame, self._current_source()
        if p is None or s is None:
            self.popups.message("YORU-Explain", "Add a video or image in 1 Project first.")
            return
        if a is None:
            self.popups.message("YORU-Explain", "Load a model in 1 Project first: its detections are "
                                                "the boxes you pick here.")
            return
        if f is None:
            self.popups.message("YORU-Explain", "Wait until the frame is shown.")
            return
        if self._det_adapter is not a:
            # Detections of a model that is no longer the loaded one: refresh first.
            self._request_frame()
            self.popups.message("YORU-Explain", "The model changed; the detections are being "
                                                "refreshed. Select the box again.")
            return
        idx = list(range(len(self.detections))) if all_on_frame else (
            [self.selected] if self.selected is not None else [])
        if not idx:
            self.popups.message("YORU-Explain", "No detection on this frame." if all_on_frame else
                                "Click a box on the frame (or a row in the list) first.")
            return
        added = []
        try:
            for i in idx:
                t = api.target_from_detection(
                    self.detections[i], i, f.image, source_id=s["source_id"], path=s["path"],
                    kind=s["kind"], frame_index=f.frame_index, timestamp_ms=f.timestamp_ms,
                    v_flip=s["v_flip"], h_flip=s["h_flip"], model_id=self.state.model_id or "",
                    access=f.access, frame_sha1=f.sha1)
                added.append(p.add_event(t, dpg.get_value("exp_note").strip()))
            p.save()
        except Exception as e:
            self.popups.error("Add event", e)
        self.state.eventsChanged.emit()
        if added:
            # New events join the selection, so 3 Analyze runs them next.
            self.events.select(self.events.selected + added)
            self._events_selected()

    # -- a YORU detections CSV ----------------------------------------------------------

    def _suggested_csv(self, s) -> Optional[str]:
        """The CSV to start the file dialog from: one attached, or YORU's ``<video>.csv``."""
        files = s.get("detections_files") or []
        if files:
            return files[-1]["path"]
        beside = Path(s["path"]).with_suffix(".csv")
        return str(beside) if beside.is_file() else s["path"]

    def import_csv(self, path: Optional[str] = None) -> Optional[int]:
        """Choose a YORU detections CSV for the shown video, then what to import (a dialog)."""
        from yoru_explain.io.detections_csv import load_detections_csv

        p, s = self.state.project, self._current_source()
        if p is None or s is None:
            self.popups.message("Import from YORU CSV", "Add a video in 1 Project and show it here first.")
            return None
        f = path or widgets.ask_open_file(f"YORU detections CSV for {Path(s['path']).name}",
                                          widgets.CSV_TYPES, current=self._suggested_csv(s))
        if not f:
            return None
        try:
            layout, by_frame = load_detections_csv(f)
        except Exception as e:
            self.popups.error("Import from YORU CSV", e)
            return None
        counts = Counter(x.detection.class_name for v in by_frame.values() for x in v)
        on_frames: Dict[str, set] = {}
        for k, v in by_frame.items():
            for x in v:
                on_frames.setdefault(x.detection.class_name, set()).add(k)
        total = sum(counts.values())
        frame_index = self.frame.frame_index if self.frame is not None else None
        on_frame = len(by_frame.get(frame_index, [])) if frame_index is not None else 0
        form: Dict[str, object] = {"path": str(f), "source_id": s["source_id"], "classes": {},
                                   "per_class": {}, "frame_index": frame_index}
        self.csv_form = form
        themes = self.app.themes
        wrap = CSV_DIALOG_W - 30

        def build(tag):
            form["tag"] = tag
            dpg.add_text(Path(f).name)
            dpg.bind_item_theme(dpg.last_item(), themes.header)
            muted(f"{CSV_LAYOUTS.get(layout, layout)}: {total} detections on {len(by_frame)} frames. "
                  f"Video: {s['source_id']} {Path(s['path']).name}", themes, wrap=wrap)
            dpg.add_spacer(height=4)
            dpg.add_text("Classes to import")
            for name, n in sorted(counts.items()):
                nf = len(on_frames[name])
                with dpg.group(horizontal=True):
                    # Shown with "per class": how many events of this class.
                    form["per_class"][name] = dpg.add_input_int(
                        default_value=min(PER_CLASS_DEFAULT, nf), min_value=0, min_clamped=True,
                        step=0, width=70, show=False)
                    form["classes"][name] = dpg.add_checkbox(
                        label=f"{name}  ({n} boxes on {nf} frames)", default_value=True)
            if not counts:
                muted("The file holds no detections.", themes)
            with dpg.group(horizontal=True):
                dpg.add_text("Frames  ")
                items = [ALL_FRAMES]
                if frame_index is not None:
                    items.append(f"only the frame shown now ({frame_index}: {on_frame} boxes)")
                form["frames"] = dpg.add_radio_button(items, default_value=ALL_FRAMES, horizontal=True)
            with dpg.group(horizontal=True):
                dpg.add_text("Pick    ")
                form["order"] = dpg.add_radio_button([FIRST, RANDOM], default_value=FIRST,
                                                     horizontal=True,
                                                     callback=lambda: self._csv_form_changed())
            with dpg.group(horizontal=True):
                dpg.add_text("How many")
                form["number"] = dpg.add_radio_button([IN_TOTAL, PER_CLASS], default_value=IN_TOTAL,
                                                      horizontal=True,
                                                      callback=lambda: self._csv_form_changed())
            with dpg.group(horizontal=True) as row:
                form["limit_row"] = row
                dpg.add_text("At most")
                form["limit"] = dpg.add_input_int(default_value=min(100, max(1, total)), min_value=1,
                                                  min_clamped=True, step=0, width=120)
                dpg.add_text("events")
            form["per_class_help"] = muted("Set the number of events beside each class (0 or "
                                           "unticked: none).", themes, wrap=wrap, show=False)
            with dpg.group(horizontal=True):
                dpg.add_text("Min. gap")
                form["gap"] = dpg.add_input_float(default_value=0.0, min_value=0.0, min_clamped=True,
                                                  step=0, width=90, format="%.2f")
                tooltip(form["gap"], "Events of one class at least this many seconds apart, counting "
                                     "the events this video has already. 0: no gap.")
                dpg.add_text("s between events of a class (0: none)")
            muted("Neighbouring frames are nearly the same picture; a gap keeps a burst of them from "
                  "passing for many samples.", themes, wrap=wrap)
            with dpg.group(show=False) as group:
                form["random_rows"] = group
                with dpg.group(horizontal=True):
                    dpg.add_text("Seed")
                    form["seed"] = dpg.add_input_int(default_value=random.randrange(1, 1_000_000),
                                                     step=0, width=120)
                    muted("the same seed draws the same boxes", themes)
                muted("At random: n boxes of a class come from n different frames as long as the "
                      "chosen frames hold that many. Boxes that are events already are not drawn "
                      "again, so drawing again adds other boxes.", themes, wrap=wrap)
            dpg.add_spacer(height=4)
            muted("YORU's CSV does not record the model or its settings, so each imported box is "
                  "only a pointer: before it is analysed, the loaded model is run on its frame and "
                  "must find the same box again (same class, IoU >= 0.9), or the event fails with "
                  "target_not_reproduced. The CSV's boxes are drawn in light blue on the frames.",
                  themes, wrap=wrap)
            dpg.add_spacer(height=6)
            with dpg.group(horizontal=True):
                b = dpg.add_button(label="Import events", callback=lambda: self._accept_csv_form(True))
                dpg.bind_item_theme(b, themes.primary_button)
                b = dpg.add_button(label="Only show the boxes",
                                   callback=lambda: self._accept_csv_form(False))
                tooltip(b, "Draw the CSV's boxes on the frames without making events.")
                dpg.add_button(label="Cancel", width=90, callback=lambda: self.popups.close(tag))

        return self.popups.dialog("Import from YORU CSV", build, width=CSV_DIALOG_W, text=str(f))

    def _csv_form_changed(self) -> None:
        """Show the inputs the chosen "Pick" and "How many" use."""
        form, g = self.csv_form, dpg.get_value
        per_class = g(form["number"]) == PER_CLASS
        for item in form["per_class"].values():
            dpg.configure_item(item, show=per_class)
        dpg.configure_item(form["per_class_help"], show=per_class)
        dpg.configure_item(form["limit_row"], show=not per_class)
        dpg.configure_item(form["random_rows"], show=g(form["order"]) == RANDOM)

    def _accept_csv_form(self, make_events: bool) -> None:
        form, g = self.csv_form, dpg.get_value
        classes = [name for name, item in form["classes"].items() if g(item)]
        only_frame = g(form["frames"]) != ALL_FRAMES and form["frame_index"] is not None
        at_random = g(form["order"]) == RANDOM
        per_class = None
        if g(form["number"]) == PER_CLASS:
            per_class = {name: max(0, int(g(form["per_class"][name]))) for name in classes}
        limit = max(1, int(g(form["limit"]))) if per_class is None else None
        gap = round(float(g(form["gap"])), 4)     # the input is a 32-bit float: 0.2 -> 0.2000000029
        self.popups.close(form["tag"])
        self.import_csv_events(form["source_id"], form["path"], make_events=make_events,
                               class_names=classes, frames=[form["frame_index"]] if only_frame else None,
                               limit=limit, per_class=per_class,
                               order="random" if at_random else "first",
                               seed=int(g(form["seed"])) if at_random else None,
                               min_gap_s=gap if gap > 0 else None)

    def import_csv_events(self, source_id: str, csv_path: str, *, make_events: bool = True,
                          class_names=None, frames=None, limit: Optional[int] = None,
                          per_class: Optional[Dict[str, int]] = None, order: str = "first",
                          seed: Optional[int] = None, min_gap_s: Optional[float] = None) -> List[str]:
        """Attach the CSV to the source (its boxes are drawn) and add its detections as events.

        *per_class* ``{class_name: n}`` asks for *n* events of each class (of
        the ticked *class_names*) instead of *limit* in all; ``order="random"``
        draws them at random with *seed*; *min_gap_s* keeps the events of a
        class that many seconds apart (see
        :func:`yoru_explain.api.add_events_from_detections_csv`).
        """
        from yoru_explain import api

        p = self.state.project
        added: List[str] = []
        try:
            p.attach_detections(source_id, csv_path)
            if make_events and class_names is not None and not class_names:
                self.popups.message("Import from YORU CSV", "No class was ticked; no event was added.")
                make_events = False
            if make_events and per_class is not None:
                per_class = {k: int(n) for k, n in per_class.items()
                             if int(n) > 0 and (class_names is None or k in class_names)}
                if not per_class:
                    self.popups.message("Import from YORU CSV", "Every ticked class is set to 0 "
                                                                "events; no event was added.")
                    make_events = False
            if make_events:
                added = api.add_events_from_detections_csv(
                    p, source_id, csv_path, model_id=self.state.model_id or "",
                    class_names=(class_names or None) if per_class is None else None,
                    frames=frames, limit=limit, per_class=per_class, order=order, seed=seed,
                    note=dpg.get_value("exp_note").strip(), min_gap_s=min_gap_s)
                if order == "random":
                    logger.info("drew %d event(s) at random from %s for %s: seed %s, %s, frames %s%s",
                                len(added), Path(csv_path).name, source_id, seed,
                                f"per class {per_class}" if per_class is not None else
                                f"at most {limit} of {class_names or 'every class'}",
                                frames if frames is not None else "all",
                                f", at least {min_gap_s:g} s apart" if min_gap_s else "")
            p.save()
        except Exception as e:
            self.popups.error("Import from YORU CSV", e)
        self.state.projectChanged.emit()   # the CSV's boxes, and the Project table's CSV column
        if make_events:
            self.state.eventsChanged.emit()
            if added:
                # Selected, so 3 Analyze runs them next (the list's title counts them).
                self.events.select(added)
                self._events_selected()
                self._report_short(added, per_class, order, min_gap_s)
            else:
                self.popups.message("Import from YORU CSV", "No detection of the ticked classes "
                                    "(on the chosen frames) is in the file" +
                                    (" that is not an event already" if order == "random" else "") +
                                    "; no event was added.")
        return added

    def _report_short(self, added: List[str], per_class: Optional[Dict[str, int]],
                      order: str, min_gap_s: Optional[float] = None) -> None:
        """Say which classes got fewer events than asked for."""
        if not per_class:
            return
        got = Counter(self.state.project.event(e)["class_name"] for e in added)
        short = [f"{name}: {got.get(name, 0)} of {n}" for name, n in sorted(per_class.items())
                 if got.get(name, 0) < n]
        if short:
            more = " that are not events already" if order == "random" else ""
            if min_gap_s:
                more += f" and at least {min_gap_s:g} s from the others"
            self.popups.message("Import from YORU CSV",
                                f"Added {len(added)} events. Fewer than asked for ({'; '.join(short)}): "
                                f"the chosen frames hold no more boxes of those classes{more}.")

    def detach_csv(self) -> None:
        p, s = self.state.project, self._current_source()
        if p is None or s is None:
            return
        try:
            p.detach_detections(s["source_id"])
            p.save()
        except Exception as e:
            self.popups.error("Detach CSV", e)
        self.state.projectChanged.emit()

    # -- events -----------------------------------------------------------------------

    def refresh_events(self):
        p = self.state.project
        rows, ids = [], []
        if p is not None:
            for e in p.events:
                cand = e.get("candidate_index", "")
                rows.append([e["event_id"], e["source_id"], e["frame_index"], e["class_name"],
                             fmt(e["score"]), cand, e.get("origin", ""), e.get("note", "")])
                ids.append(e["event_id"])
        self.events.set_rows(rows, ids)
        self._events_selected()

    def remove_events(self):
        p = self.state.project
        ids = self.events.selected
        if p is None or not ids:
            return

        def remove():
            p.remove_events(ids)
            p.save()
            self.state.eventsChanged.emit()

        self.popups.confirm("Remove events", f"Remove {len(ids)} event(s)? Existing run results "
                                             "stay on disk.", remove, yes="Remove")

    def _events_clicked(self, keys):
        self._events_selected()
        if len(keys) == 1:
            self.goto_selected_event()   # one event selected: show it on its frame

    def _events_selected(self):
        ids = self.events.selected
        self.state.select_events(ids)
        n = self.events.row_count
        dpg.set_value("exp_events_title", f"Events ({n}, {len(ids)} selected)" if n else "Events")
        self._result_data, self._result_labels = [], []
        if len(ids) == 1:
            for reader, row in self.state.results_for(ids[0]):
                label = f"{reader.run_id}  {row.get('method')}  {row.get('status')}"
                if row.get("error_code"):
                    label += f" ({row['error_code']})"
                while label in self._result_labels:   # combo items are matched by text
                    label += " "
                self._result_labels.append(label)
                self._result_data.append((str(reader.dir), row.get("event_id")))
        dpg.configure_item("exp_results", items=self._result_labels)
        dpg.set_value("exp_results", self._result_labels[0] if self._result_labels else "")
        self._result_chosen()

    def goto_selected_event(self):
        ids = self.events.selected
        p = self.state.project
        if p is None or len(ids) != 1:
            return
        t = p.target(ids[0])
        s = self._current_source()
        if s is None or s["source_id"] != t.frame.source_id:
            if self._select_source(t.frame.source_id):
                self._source_changed()
        self._pending_select = t
        self.set_index(t.frame.frame_index)

    def _match_pending(self):
        from yoru_explain.core.geometry import overlap

        t = self._pending_select
        if t is None or self.frame is None or t.frame.frame_index != self.frame.frame_index:
            return None
        best = None
        for i, d in enumerate(self.detections):
            if d.class_id != t.detection.class_id:
                continue
            v = overlap(d.box, t.box, d.obb, t.obb)
            if best is None or v > best[0]:
                best = (v, i)
        return best[1] if best and best[0] >= 0.5 else None

    # -- results ------------------------------------------------------------------------

    def show_result(self, run_dir: str, event_id: str):
        self.events.select([event_id])
        self._events_selected()
        self.goto_selected_event()
        key = (run_dir, event_id)
        if key in self._result_data:
            dpg.set_value("exp_results", self._result_labels[self._result_data.index(key)])
            self._result_chosen()

    def _result_chosen(self):
        from yoru_explain.io.runs import RunReader

        label = dpg.get_value("exp_results")
        self._result = None
        if label in self._result_labels:
            run_dir, event_id = self._result_data[self._result_labels.index(label)]
            try:
                reader = RunReader(run_dir)
                row = reader.result(event_id)
                maps = reader.load_map(row["map_file"]) if row and row.get("map_file") else None
                self._result = (reader, row, maps)
            except Exception as e:
                self.popups.error("Result", e)
        self._show_info()
        self.redraw()

    def _show_info(self):
        themes = self.app.themes
        if self._result is None:
            self._info_lines = [("No result selected.", themes.muted), (DISCLAIMER, themes.muted)]
            self._render_info()
            return
        reader, row, maps = self._result
        kind = {"perturbation_effect": "Perturbation effect (occlusion)",
                "attribution": "Attribution (Grad-CAM)"}.get(row.get("method_kind"), row.get("method"))
        status = row.get("status")
        status_theme = themes.ok if status == "ok" else (
            themes.warn if status in ("cancelled", "skipped") else themes.error)
        lines = [(f"{kind} - run {reader.run_id}", themes.header),
                 (f"status: {status} {row.get('error_code') or ''}", status_theme),
                 (f"explained: {row.get('message')}", None),
                 (f"score type: {row.get('score_type')}, identity: {row.get('identity_mode')}", None),
                 (f"original score: {fmt(row.get('original_score'), 6)}; reference at batch size: "
                  f"{fmt(row.get('reference_score'), 6)}", None)]
        if row.get("method") == "occlusion":
            lines += [(f"baseline: {row.get('baseline')}", None),
                      (f"largest drop: {fmt(row.get('delta_score'))} (patch "
                       f"{row.get('max_delta_patch_id')}, score {fmt(row.get('perturbed_score'))}); "
                       f"smallest delta {fmt(row.get('delta_min'))}", None),
                      (f"mean delta in box {fmt(row.get('in_box_mean_delta'))}, outside "
                       f"{fmt(row.get('out_box_mean_delta'))}", None),
                      (f"patches {row.get('n_perturbations')}: counted {row.get('n_counted')}, "
                       f"disappeared {row.get('n_disappeared')}, class changed "
                       f"{row.get('n_class_changed')}, unmatched {row.get('n_unmatched')}, ambiguous "
                       f"{row.get('n_ambiguous')}, candidate dropped by NMS "
                       f"{row.get('n_candidate_dropped')}", None),
                      (f"controls: unchanged-frame delta {fmt(row.get('noop_delta'))}; whole region "
                       f"replaced: {row.get('full_region_status')} {fmt(row.get('full_region_score'))}",
                       None)]
        else:
            lines += [(f"layer {row.get('gradcam_layer')} (stride {row.get('gradcam_stride')}), "
                       f"gradient at {row.get('gradient_support')} cell(s)", None),
                      (f"CAM max {fmt(row.get('cam_max'))}, share inside box "
                       f"{fmt(row.get('cam_in_box_fraction'))}", None)]
        lines.append((DISCLAIMER, themes.muted))
        self._info_lines = lines
        self._render_info()

    def _render_info(self):
        if not dpg.does_item_exist("exp_info"):
            return
        dpg.delete_item("exp_info", children_only=True)
        for text, theme in self._info_lines:
            item = dpg.add_text(text, parent="exp_info", wrap=max(200, self._wrap))
            if theme is not None:
                dpg.bind_item_theme(item, theme)

    def info_text(self) -> str:
        return "\n".join(text for text, _ in self._info_lines)

    # -- drawing --------------------------------------------------------------------------

    def redraw(self):
        f = self.frame
        if f is None:
            return
        values, kind = None, "delta"
        mode = dpg.get_value("exp_mode")
        if self._result is not None and self._result[2] is not None:
            _, row, maps = self._result
            event_frame = int(float(row.get("frame_index") or -1))
            same = event_frame == f.frame_index and str(row.get("frame_sha1", "")) in ("", f.sha1)
            if same:
                if "delta_map" in maps:
                    values, kind = maps["delta_map"], "delta"
                elif "cam" in maps:
                    values, kind = maps["cam"], "cam"
        image, panels, vrange = compose(f.image, values, kind, mode, float(dpg.get_value("exp_alpha")))
        self._shown = (values, kind)
        self.view.set_image(image, frame_width=f.image.shape[1], panels=panels)
        self._show_scale(kind, vrange)
        boxes = [d.box for d in self.detections]
        labels = [f"{i}: {d.class_name} {d.score:.2f}" for i, d in enumerate(self.detections)]
        imported = []
        s = self._current_source()
        if s is not None:
            imported = [x.detection for x in self._imported.get(s["source_id"], {}).get(f.frame_index, [])]
        self.view.set_boxes(boxes, selected=self.selected, labels=labels,
                            dashed=[d.box for d in imported], obbs=[d.obb for d in self.detections],
                            dashed_obbs=[d.obb for d in imported])

    def _show_scale(self, kind: str, vrange) -> None:
        visible = vrange is not None
        was = dpg.is_item_shown("exp_scale_group")
        if visible:
            dpg.configure_item("exp_scale", colormap=self._cmaps[kind],
                               min_scale=float(vrange[0]), max_scale=float(vrange[1]))
        if visible != was:
            dpg.configure_item("exp_scale_group", show=visible)
            size = getattr(self, "_layout_size", None)
            if size is not None:
                self.layout(*size)
