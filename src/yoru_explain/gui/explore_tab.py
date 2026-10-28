# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Explore tab: browse frames, pick a detection, look at its results.

Frames are decoded exactly (in order, or by a seek verified against
presentation times) on a worker thread, and the loaded model runs on each
shown frame, so the boxes are the detector's own.  Clicking a box selects
it; "Add as event" stores it -- with its pre-NMS candidate and the frame's
hash -- for analysis.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dearpygui.dearpygui as dpg

from yoru_explain.gui.frame_view import VIEW_MODES, FrameView, compose, map_value, pick_box
from yoru_explain.gui.widgets import SelectTable, fmt, muted, tooltip
from yoru_explain.gui.worker import FrameRequest

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
        self.view = FrameView("exp_view", app.themes, on_click=self._clicked)
        self.det_table = SelectTable("exp_dets", ["#", "class", "score", "box (x1, y1, x2, y2)",
                                                  "candidate"],
                                     height=150, on_select=lambda keys: self._det_row_selected(keys))
        self.events = SelectTable("exp_events", ["event", "source", "frame", "class", "score",
                                                 "candidate", "origin", "note"],
                                  multi=True, height=200, sortable=True,
                                  on_select=lambda keys: self._events_selected(),
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
            with dpg.child_window(tag="exp_right", border=False):
                t = dpg.add_text("Detections on this frame")
                dpg.bind_item_theme(t, themes.header)
                muted("click a box or a row", themes)
                self.det_table.build()
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Add selected as event", callback=lambda: self.add_events(False))
                    dpg.add_button(label="Add all on frame", callback=lambda: self.add_events(True))
                    b = dpg.add_button(label="Add CSV boxes on frame",
                                       callback=lambda: self.add_csv_events())
                    tooltip(b, "The light-blue boxes of an attached YORU CSV, as events; a run "
                               "finds each again with the loaded model (same class, IoU >= 0.9)")
                dpg.add_input_text(tag="exp_note", hint="note for new events (optional)", width=-1)
                dpg.add_spacer(height=4)
                t = dpg.add_text("Events")
                dpg.bind_item_theme(t, themes.header)
                muted("the selection is what Analyze runs on; Ctrl/Shift-click for several, "
                      "double-click to go to the frame", themes, tag="exp_events_help")
                self.events.build()
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Go to frame", callback=lambda: self.goto_selected_event())
                    dpg.add_button(label="Remove selected events", callback=lambda: self.remove_events())
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
        plot_h = max(160, height - 96)
        scale = SCALE_W + 8 if dpg.is_item_shown("exp_scale_group") else 0
        self.view.resize(left - scale - 8, plot_h)
        dpg.configure_item("exp_scale", height=plot_h)
        rest = max(300, height - 300)
        self.det_table.height = max(90, int(rest * 0.32))
        dpg.configure_item(self.det_table.tag, height=self.det_table.height)
        dpg.configure_item(self.events.tag, height=max(110, int(rest * 0.42)))
        dpg.configure_item("exp_events_help", wrap=right - 10)
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
        self._indexed.clear()
        self._imported.clear()
        self._source_changed(keep_index=keep)

    def _source_changed(self, keep_index: bool = False):
        s = self._current_source()
        n = max(0, int(s.get("frame_count") or 1) - 1) if s else 0
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
            dpg.set_value("exp_frame_info", "No source." if self.state.project is not None
                          else "Open a project in the Project tab.")
            return
        if s["kind"] == "video" and s["source_id"] not in self._indexed:
            self._indexed[s["source_id"]] = {"status": "building"}
            self.state.frames.build_index(self._req(s, 0))
        self._load_imported(s)
        self._request_frame()

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
            rows.append([i, d.class_name, f"{d.score:.4f}",
                         ", ".join(f"{v:.1f}" for v in d.box),
                         d.candidate.describe() if d.candidate else "-"])
        self.det_table.set_rows(rows, list(range(len(rows))))
        self.det_table.select([self.selected] if self.selected is not None else [])

    def _clicked(self, x, y):
        i = pick_box([d.box for d in self.detections], x, y)
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
        if p is None or a is None or f is None or s is None:
            self.popups.message("YORU-Explain", "Open a project, load a model and show a frame first.")
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
            self.popups.message("YORU-Explain", "Select a detection first.")
            return
        try:
            for i in idx:
                t = api.target_from_detection(
                    self.detections[i], i, f.image, source_id=s["source_id"], path=s["path"],
                    kind=s["kind"], frame_index=f.frame_index, timestamp_ms=f.timestamp_ms,
                    v_flip=s["v_flip"], h_flip=s["h_flip"], model_id=self.state.model_id or "",
                    access=f.access, frame_sha1=f.sha1)
                p.add_event(t, dpg.get_value("exp_note").strip())
            p.save()
        except Exception as e:
            self.popups.error("Add event", e)
        self.state.eventsChanged.emit()

    def add_csv_events(self):
        from yoru_explain import api

        p, f, s = self.state.project, self.frame, self._current_source()
        imported = self._imported.get(s["source_id"], {}).get(f.frame_index, []) if (f and s) else []
        if p is None or not imported:
            self.popups.message("YORU-Explain", "No CSV boxes on this frame.")
            return
        try:
            for x in imported:
                t = api.target_from_detection(
                    x.detection, -1, f.image, source_id=s["source_id"], path=s["path"], kind=s["kind"],
                    frame_index=f.frame_index, timestamp_ms=f.timestamp_ms, v_flip=s["v_flip"],
                    h_flip=s["h_flip"], model_id=self.state.model_id or "", access=f.access,
                    frame_sha1=f.sha1, origin="csv_import")
                p.add_event(t, dpg.get_value("exp_note").strip())
            p.save()
        except Exception as e:
            self.popups.error("Add CSV events", e)
        self.state.eventsChanged.emit()

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

    def _events_selected(self):
        ids = self.events.selected
        self.state.selected_events = ids
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
        from yoru_explain.core.geometry import iou

        t = self._pending_select
        if t is None or self.frame is None or t.frame.frame_index != self.frame.frame_index:
            return None
        best = None
        for i, d in enumerate(self.detections):
            if d.class_id != t.detection.class_id:
                continue
            v = iou(d.box, t.box)
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
            imported = [x.detection.box for x in self._imported.get(s["source_id"], {}).get(f.frame_index, [])]
        self.view.set_boxes(boxes, selected=self.selected, labels=labels, dashed=imported)

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
