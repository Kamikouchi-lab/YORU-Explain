# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Settings tab: device, logs, settings files, versions, adapters."""

from __future__ import annotations

import json

from PyQt5 import QtCore, QtGui, QtWidgets

from yoru_explain.gui.qtutil import error_box
from yoru_explain.gui.state import AppState


class SettingsTab(QtWidgets.QWidget):
    def __init__(self, state: AppState, analyze_tab, parent=None):
        super().__init__(parent)
        self.state = state
        self.analyze = analyze_tab
        layout = QtWidgets.QVBoxLayout(self)

        dev = QtWidgets.QGroupBox("Compute device (used when a model is loaded)")
        dl = QtWidgets.QHBoxLayout(dev)
        self.device = QtWidgets.QComboBox()
        self.device.addItems(["auto", "cuda", "cpu"])
        self.device.setToolTip("auto follows YORU: $YORU_DEVICE, else CUDA, else CPU. Scores differ "
                               "slightly between devices (fp16 autocast on CUDA), so runs record it.")
        self.device.currentTextChanged.connect(self._device)
        dl.addWidget(self.device)
        dl.addStretch(1)
        layout.addWidget(dev)

        logs = QtWidgets.QGroupBox("Logs")
        ll = QtWidgets.QHBoxLayout(logs)
        from yoru_explain.core.logs import log_file

        self.log_path = QtWidgets.QLineEdit(str(log_file()))
        self.log_path.setReadOnly(True)
        b_logs = QtWidgets.QPushButton("Open log folder")
        b_logs.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(str(log_file().parent))))
        ll.addWidget(self.log_path, 1)
        ll.addWidget(b_logs)
        layout.addWidget(logs)

        cfg = QtWidgets.QGroupBox("Analysis settings file (JSON)")
        cl = QtWidgets.QHBoxLayout(cfg)
        b_export = QtWidgets.QPushButton("Export current settings...")
        b_import = QtWidgets.QPushButton("Import settings...")
        b_export.clicked.connect(self.export_settings)
        b_import.clicked.connect(self.import_settings)
        cl.addWidget(b_export)
        cl.addWidget(b_import)
        cl.addStretch(1)
        layout.addWidget(cfg)

        info = QtWidgets.QGroupBox("Versions, adapters and the loaded model")
        il = QtWidgets.QVBoxLayout(info)
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        il.addWidget(self.text)
        b_refresh = QtWidgets.QPushButton("Refresh")
        b_refresh.clicked.connect(self.refresh)
        il.addWidget(b_refresh)
        layout.addWidget(info, 1)
        state.adapterChanged.connect(self.refresh)
        self.refresh()

    def _device(self, text):
        self.state.device = text

    def export_settings(self):
        f, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export analysis settings", "analysis.json",
                                                     "JSON (*.json)")
        if not f:
            return
        try:
            with open(f, "w", encoding="utf-8") as fh:
                json.dump(self.analyze.request().to_dict(), fh, indent=2)
        except Exception as e:
            error_box(self, "Export settings", e)

    def import_settings(self):
        from yoru_explain.core.pipeline import AnalysisRequest

        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import analysis settings", "", "JSON (*.json)")
        if not f:
            return
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            # A run's config.json holds the request under "request".
            req = AnalysisRequest.from_dict(data.get("request", data))
            self.state.requestImported.emit(req)
        except Exception as e:
            error_box(self, "Import settings", e)

    def refresh(self):
        from yoru_explain.core.environment import hardware, software_versions
        from yoru_explain.core.registry import UNSUPPORTED, list_adapters
        from yoru_explain.io.schema import SCHEMA_VERSION

        lines = [f"schema_version: {SCHEMA_VERSION}", "", "software:"]
        lines += [f"  {k}: {v}" for k, v in software_versions().items()]
        lines += ["", "hardware:"] + [f"  {k}: {v}" for k, v in hardware().items()]
        lines += ["", "adapters:"]
        for e in list_adapters():
            lines.append(f"  {e.name}: {e.label}{'  [synthetic]' if e.synthetic else ''}")
        lines += ["", "not supported yet:"] + [f"  {k}: {v}" for k, v in UNSUPPORTED.items()]
        a = self.state.adapter
        if a is not None:
            lines += ["", f"loaded model ({self.state.model_id}):",
                      json.dumps(a.model_info(), indent=2, default=str),
                      "capabilities:", json.dumps(a.capabilities.to_dict(), indent=2)]
        self.text.setPlainText("\n".join(lines))
