# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Statistics tab (v0.1): what each run did, event by event.

Counts and per-event values only.  No test is run and no population summary
is drawn: frames of one video are not independent samples, and a group
analysis needs videos, animals and days as its units (planned for v0.2).
"""

from __future__ import annotations

from collections import Counter
from typing import List

import dearpygui.dearpygui as dpg

from yoru_explain.gui import widgets
from yoru_explain.gui.widgets import SelectTable, fmt, muted

NOTE = ("Descriptive, per event. Frames from one video are not independent samples; no statistical "
        "test is computed here and none should be read into these numbers. Values describe the "
        "model's score, not the animal.")


class StatisticsTab:
    def __init__(self, app):
        self.app = app
        self.state = app.state
        self._run_dirs: List[str] = []
        self._run_labels: List[str] = []
        self.table = SelectTable("stat_table", ["event", "frame", "class", "status", "error",
                                                "score type", "identity", "original", "largest drop",
                                                "smallest", "in box", "outside", "counted",
                                                "disappeared", "class changed", "candidate dropped",
                                                "noop", "CAM in box"],
                                 height=300, sortable=True,
                                 on_activate=lambda key: self.state.showResult.emit(*key))
        self.state.runsChanged.connect(self.refresh)
        self.state.projectChanged.connect(self.refresh)

    def build(self) -> None:
        themes = self.app.themes
        with dpg.group(horizontal=True):
            dpg.add_text("Run")
            dpg.add_combo([], tag="stat_run", width=-330, callback=lambda: self.show_run())
            dpg.add_button(label="Refresh", callback=lambda: self.state.runs_changed())
            dpg.add_button(label="Open run folder", callback=lambda: self.open_folder())
        dpg.add_text("", tag="stat_summary", wrap=1000)
        muted(NOTE, themes, tag="stat_note", wrap=1000)
        dpg.add_separator()
        self.table.build()
        with dpg.group(horizontal=True):
            dpg.add_button(label="View in Explore", callback=lambda: self._open_selected())
            muted("or double-click a row; click a header to sort", themes)
        self.refresh()

    def layout(self, width: int, height: int) -> None:
        for tag in ("stat_summary", "stat_note"):
            dpg.configure_item(tag, wrap=max(300, width - 20))
        dpg.configure_item(self.table.tag, height=max(120, height - 190))

    @property
    def run_count(self) -> int:
        return len(self._run_dirs)

    def refresh(self):
        label = dpg.get_value("stat_run")
        current = self._run_dirs[self._run_labels.index(label)] if label in self._run_labels else None
        self._run_dirs, self._run_labels = [], []
        for reader in self.state.run_readers():
            m = reader.manifest
            self._run_dirs.append(str(reader.dir))
            self._run_labels.append(f"{reader.run_id}  {m.get('method')}  {m.get('status')}  "
                                    f"{(m.get('events') or {}).get('requested', '?')} event(s)")
        dpg.configure_item("stat_run", items=self._run_labels)
        if current in self._run_dirs:
            dpg.set_value("stat_run", self._run_labels[self._run_dirs.index(current)])
        else:
            dpg.set_value("stat_run", self._run_labels[0] if self._run_labels else "")
        self.show_run()

    def _reader(self):
        label = dpg.get_value("stat_run")
        if label not in self._run_labels:
            return None
        d = self._run_dirs[self._run_labels.index(label)]
        for r in self.state.run_readers():
            if str(r.dir) == d:
                return r
        return None

    def show_run(self):
        reader = self._reader()
        if reader is None:
            dpg.set_value("stat_summary", "No runs yet.")
            self.table.set_rows([])
            return
        m = reader.manifest
        rows = reader.results
        by_status = Counter(r.get("status") for r in rows)
        by_error = Counter(r.get("error_code") for r in rows if r.get("error_code"))
        ident = m.get("identity_check") or {}
        model = m.get("model") or {}
        cfg = reader.config.get("method_config", {})
        dpg.set_value(
            "stat_summary",
            f"{reader.run_id} - {m.get('method')} ({m.get('method_kind')}), {m.get('status')}, "
            f"created {m.get('created')}\n"
            f"model {model.get('model_id')} {model.get('weights')} sha256 {str(model.get('sha256'))[:12]} "
            f"on {model.get('device')}; identity check vs YORU: {ident.get('status', '-')}"
            f"{' (exact)' if ident.get('exact') else ''}\n"
            f"events by status: {dict(by_status)}; errors: {dict(by_error) or 'none'}\n"
            f"settings: {cfg}")
        out, data = [], []
        for r in rows:
            out.append([r.get("event_id"), r.get("frame_index"), r.get("behavior_class"), r.get("status"),
                        r.get("error_code"), r.get("score_type"), r.get("identity_mode"),
                        fmt(r.get("original_score")), fmt(r.get("delta_score")), fmt(r.get("delta_min")),
                        fmt(r.get("in_box_mean_delta")), fmt(r.get("out_box_mean_delta")),
                        r.get("n_counted"), r.get("n_disappeared"), r.get("n_class_changed"),
                        r.get("n_candidate_dropped"), fmt(r.get("noop_delta")),
                        fmt(r.get("cam_in_box_fraction"))])
            data.append((str(reader.dir), r.get("event_id")))
        self.table.set_rows(out, data)

    def open_folder(self):
        reader = self._reader()
        if reader is not None:
            try:
                widgets.open_folder(reader.dir)
            except Exception as e:
                self.app.popups.error("Open run folder", e)

    def _open_selected(self):
        sel = self.table.selected
        if sel:
            self.state.showResult.emit(*sel[0])
