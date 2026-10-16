# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Statistics tab (v0.1): what each run did, event by event.

Counts and per-event values only.  No test is run and no population summary
is drawn: frames of one video are not independent samples, and a group
analysis needs videos, animals and days as its units (planned for v0.2).
"""

from __future__ import annotations

from collections import Counter

from PyQt5 import QtCore, QtGui, QtWidgets

from yoru_explain.gui.qtutil import fill_table, fmt, table
from yoru_explain.gui.state import AppState

NOTE = ("<small><b>Descriptive, per event.</b> Frames from one video are not independent samples; "
        "no statistical test is computed here and none should be read into these numbers. "
        "Values describe the model's score, not the animal.</small>")


class StatisticsTab(QtWidgets.QWidget):
    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        layout = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        self.run = QtWidgets.QComboBox()
        b_refresh = QtWidgets.QPushButton("Refresh")
        b_open = QtWidgets.QPushButton("Open run folder")
        row.addWidget(QtWidgets.QLabel("Run"))
        row.addWidget(self.run, 1)
        row.addWidget(b_refresh)
        row.addWidget(b_open)
        layout.addLayout(row)
        self.summary = QtWidgets.QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.summary)
        note = QtWidgets.QLabel(NOTE)
        note.setWordWrap(True)
        layout.addWidget(note)
        self.table = table(["event", "frame", "class", "status", "error", "score type", "identity",
                            "original", "largest drop", "smallest", "in box", "outside", "counted",
                            "disappeared", "class changed", "candidate dropped", "noop", "CAM in box"])
        self.table.setSortingEnabled(True)
        layout.addWidget(self.table, 1)
        self.run.currentIndexChanged.connect(self.show_run)
        b_refresh.clicked.connect(self.state.runs_changed)
        b_open.clicked.connect(self.open_folder)
        self.table.itemDoubleClicked.connect(self._open_result)
        state.runsChanged.connect(self.refresh)
        state.projectChanged.connect(self.refresh)
        self.refresh()

    def refresh(self):
        self.run.blockSignals(True)
        current = self.run.currentData()
        self.run.clear()
        for reader in self.state.run_readers():
            m = reader.manifest
            self.run.addItem(f"{reader.run_id}  {m.get('method')}  {m.get('status')}  "
                             f"{(m.get('events') or {}).get('requested', '?')} event(s)", str(reader.dir))
        i = self.run.findData(current)
        self.run.setCurrentIndex(i if i >= 0 else 0)
        self.run.blockSignals(False)
        self.show_run()

    def _reader(self):
        d = self.run.currentData()
        for r in self.state.run_readers():
            if str(r.dir) == d:
                return r
        return None

    def show_run(self):
        reader = self._reader()
        if reader is None:
            self.summary.setText("No runs yet.")
            fill_table(self.table, [])
            return
        m = reader.manifest
        rows = reader.results
        by_status = Counter(r.get("status") for r in rows)
        by_error = Counter(r.get("error_code") for r in rows if r.get("error_code"))
        ident = m.get("identity_check") or {}
        model = m.get("model") or {}
        cfg = reader.config.get("method_config", {})
        self.summary.setText(
            f"<b>{reader.run_id}</b> — {m.get('method')} ({m.get('method_kind')}), {m.get('status')}, "
            f"created {m.get('created')}<br>"
            f"model {model.get('model_id')} {model.get('weights')} sha256 {str(model.get('sha256'))[:12]} "
            f"on {model.get('device')}; identity check vs YORU: {ident.get('status', '-')}"
            f"{' (exact)' if ident.get('exact') else ''}<br>"
            f"events by status: {dict(by_status)}; errors: {dict(by_error) or 'none'}<br>"
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
        fill_table(self.table, out, data=data)

    def open_folder(self):
        reader = self._reader()
        if reader is not None:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(reader.dir)))

    def _open_result(self, item):
        data = self.table.item(item.row(), 0).data(QtCore.Qt.UserRole)
        if data:
            self.state.showResult.emit(*data)
