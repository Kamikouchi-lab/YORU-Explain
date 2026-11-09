# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Results tab (step 4; "statistics", v0.1): what each run did, event by event.

Counts and per-event values only.  No test is run and no population summary
is drawn: frames of one video are not independent samples, and a group
analysis needs videos, animals and days as its units.  *Export for analysis*
writes the run as plain tables, arrays, frame crops and a gallery on a shared
colour scale (``core.export.export_run``, on the task thread), for the
experimenter's own analysis by experimental unit.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Optional

import dearpygui.dearpygui as dpg

from yoru_explain.gui import widgets
from yoru_explain.gui.widgets import SelectTable, fmt, muted, tooltip

NOTE = ("Descriptive, per event. Frames from one video are not independent samples; no statistical "
        "test is computed here and none should be read into these numbers. Values describe the "
        "model's score, not the animal.")
EXPORT_TIP = ("Write this run to <project>/exports/<run id>/: events.csv (one row per event, with its "
              "experimental unit), patches.csv, a summary per unit and class, every map on a grid "
              "around its box (box_maps.npz), map and frame crops, and a gallery of every event on a "
              "shared colour scale with contact sheets. Nothing is recomputed; an earlier export of the "
              "run is replaced.")
SCALE_TIP = ("Colour scale of the gallery: one per predicted class (class), one for the whole run (run), "
             "or each event its own (event, as the previews).")


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
        tasks = self.state.tasks
        tasks.busyChanged.connect(self._busy)
        tasks.progress.connect(self._progress)
        tasks.finished.connect(self._done)
        tasks.failed.connect(self._failed)
        #: The export folder of the shown run, when it has one.
        self.export_dir: Optional[Path] = None
        self.last_export = None

    def build(self) -> None:
        themes = self.app.themes
        with dpg.group(horizontal=True):
            dpg.add_text("Run")
            dpg.add_combo([], tag="stat_run", width=-170, callback=lambda: self.show_run())
            dpg.add_button(label="Open run folder", callback=lambda: self.open_folder())
        dpg.add_text("", tag="stat_summary", wrap=1000)
        muted(NOTE, themes, tag="stat_note", wrap=1000)
        with dpg.group(horizontal=True):
            b = dpg.add_button(label="Export for analysis", tag="stat_export", callback=lambda: self.export())
            dpg.bind_item_theme(b, themes.primary_button)
            tooltip(b, EXPORT_TIP)
            dpg.add_text("gallery colours")
            c = dpg.add_combo(["class", "run", "event"], tag="stat_scale", default_value="class", width=90)
            tooltip(c, SCALE_TIP)
            c = dpg.add_checkbox(label="frame crops", tag="stat_crops", default_value=True)
            tooltip(c, "Decode the frames again to save the unchanged picture under each map. Untick "
                       "when the videos are not at hand; the numbers are exported either way.")
            dpg.add_button(label="Cancel", tag="stat_cancel", enabled=False,
                           callback=lambda: self.state.tasks.cancel())
            dpg.add_button(label="Open export folder", tag="stat_open_export", enabled=False,
                           callback=lambda: self.open_export())
        dpg.add_progress_bar(tag="stat_progress", default_value=0.0, width=-1, overlay="", show=False)
        dpg.add_text("", tag="stat_export_status", wrap=1000)
        dpg.add_separator()
        self.table.build()
        with dpg.group(horizontal=True):
            dpg.add_button(label="Show on the frame", callback=lambda: self._open_selected())
            muted("or double-click a row: its map is shown in 2 Pick events. Click a header to sort.",
                  themes)
        self.refresh()

    def layout(self, width: int, height: int) -> None:
        for tag in ("stat_summary", "stat_note", "stat_export_status"):
            dpg.configure_item(tag, wrap=max(300, width - 20))
        dpg.configure_item(self.table.tag, height=max(120, height - 260))

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
        self._show_export(reader)
        if reader is None:
            dpg.set_value("stat_summary", "No runs yet: run an analysis in 3 Analyze.")
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

    # -- export ------------------------------------------------------------------------

    def _export_folder(self, reader) -> Optional[Path]:
        p = self.state.project
        if p is None or reader is None:
            return None
        from yoru_explain.core.export import EXPORTS_DIR

        folder = p.root / EXPORTS_DIR / reader.run_id
        return folder if (folder / "export.json").is_file() else None

    def _show_export(self, reader) -> None:
        if self.state.tasks.current == "export":
            return
        self.export_dir = self._export_folder(reader)
        dpg.configure_item("stat_open_export", enabled=self.export_dir is not None)
        self._busy(self.state.tasks.busy)
        if reader is None:
            dpg.set_value("stat_export_status", "")
        elif self.export_dir is not None:
            dpg.set_value("stat_export_status", f"Exported to {self.export_dir} (exporting again replaces it).")
        else:
            dpg.set_value("stat_export_status", "Not exported yet: Export for analysis writes tables, maps on "
                                                "a grid around each box, frame crops and a gallery.")

    def export(self):
        """Export the shown run (``core.export.export_run``) on the task thread."""
        from yoru_explain.core.export import ExportOptions, export_run

        reader, p = self._reader(), self.state.project
        if reader is None or p is None:
            self.app.popups.message("Export", "Choose a run first.")
            return
        tasks = self.state.tasks
        options = ExportOptions(gallery_scale=dpg.get_value("stat_scale"),
                                crops=bool(dpg.get_value("stat_crops")))

        def work():
            return export_run(p, reader.dir, options=options,
                              progress=lambda f, m: tasks.report(f, m), cancel=tasks.cancelled)

        if not tasks.submit("export", work):
            self.app.popups.message("Export", "Another task is running.")
            return
        dpg.configure_item("stat_progress", show=True)
        dpg.set_value("stat_progress", 0.0)
        dpg.set_value("stat_export_status", f"Exporting {reader.run_id}...")
        self._busy(True)

    def _busy(self, busy):
        if not dpg.does_item_exist("stat_export"):
            return
        mine = busy and self.state.tasks.current == "export"
        dpg.configure_item("stat_export", enabled=not busy and self._reader() is not None)
        dpg.configure_item("stat_cancel", enabled=mine)

    def _progress(self, fraction, message, detail=None):
        if self.state.tasks.current != "export":
            return
        dpg.set_value("stat_progress", float(fraction))
        dpg.configure_item("stat_progress", overlay=f"{fraction * 100:.0f}%")
        dpg.set_value("stat_export_status", message)

    def _done(self, name, outcome):
        if name != "export":
            return
        self.last_export = outcome
        self.export_dir = outcome.export_dir
        dpg.configure_item("stat_progress", show=False)
        c = outcome.counts
        text = (f"Exported {c['ok']} of {c['events']} event(s) to {outcome.export_dir}"
                f" ({c['failed']} failed, listed in events.csv with their error codes).")
        if outcome.warnings:
            text += f"\n{len(outcome.warnings)} frame(s) not exported: {outcome.warnings[0]}" + \
                    (" ..." if len(outcome.warnings) > 1 else "")
        dpg.set_value("stat_export_status", text)
        dpg.configure_item("stat_open_export", enabled=True)

    def _failed(self, name, text, exc):
        if name != "export":
            return
        dpg.configure_item("stat_progress", show=False)
        self._show_export(self._reader())
        cancelled = getattr(getattr(exc, "code", None), "value", "") == "cancelled"
        dpg.set_value("stat_export_status", "Export cancelled; an earlier export is unchanged." if cancelled
                      else f"Export failed: {exc}")
        if not cancelled:
            self.app.popups.error("Export", exc)

    def open_export(self):
        if self.export_dir is not None:
            try:
                widgets.open_folder(self.export_dir)
            except Exception as e:
                self.app.popups.error("Open export folder", e)

    def _open_selected(self):
        sel = self.table.selected
        if sel:
            self.state.showResult.emit(*sel[0])
