# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Settings tab: device, logs, settings files, versions, adapters."""

from __future__ import annotations

import json

import dearpygui.dearpygui as dpg

from yoru_explain.gui import widgets
from yoru_explain.gui.widgets import tooltip


class SettingsTab:
    def __init__(self, app, analyze_tab):
        self.app = app
        self.state = app.state
        self.popups = app.popups
        self.analyze = analyze_tab
        self.state.adapterChanged.connect(self.refresh)

    def build(self) -> None:
        from yoru_explain.core.logs import log_file

        themes = self.app.themes
        widgets.heading("Compute device (used when a model is loaded)", themes)
        c = dpg.add_combo(["auto", "cuda", "cpu"], tag="set_device", default_value=self.state.device,
                          width=160, callback=lambda s, v: self._device(v))
        tooltip(c, "auto follows YORU: $YORU_DEVICE, else CUDA, else CPU. Scores differ slightly "
                   "between devices (fp16 autocast on CUDA), so runs record it.")
        dpg.add_spacer(height=6)

        widgets.heading("Logs", themes)
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="set_log_path", default_value=str(log_file()), readonly=True,
                               width=-180)
            dpg.add_button(label="Open log folder", callback=lambda: self._open_logs())
        dpg.add_spacer(height=6)

        widgets.heading("Analysis settings file (JSON)", themes)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Export current settings...", callback=lambda: self.export_settings())
            dpg.add_button(label="Import settings...", callback=lambda: self.import_settings())
        dpg.add_spacer(height=6)

        widgets.heading("Versions, adapters and the loaded model", themes)
        dpg.add_input_text(tag="set_info", multiline=True, readonly=True, width=-1, height=300)
        dpg.add_button(label="Refresh", callback=lambda: self.refresh())
        self.refresh()

    def layout(self, width: int, height: int) -> None:
        dpg.configure_item("set_info", height=max(150, height - 300))

    def _device(self, text):
        self.state.device = text

    def _open_logs(self):
        from yoru_explain.core.logs import log_file

        try:
            widgets.open_folder(log_file().parent)
        except Exception as e:
            self.popups.error("Open log folder", e)

    def export_settings(self, path=None):
        f = path or widgets.ask_save_file("Export analysis settings", widgets.JSON_TYPES, ".json",
                                          initial_file="analysis.json")
        if not f:
            return
        try:
            with open(f, "w", encoding="utf-8") as fh:
                json.dump(self.analyze.request().to_dict(), fh, indent=2)
        except Exception as e:
            self.popups.error("Export settings", e)

    def import_settings(self, path=None):
        from yoru_explain.core.pipeline import AnalysisRequest

        f = path or widgets.ask_open_file("Import analysis settings", widgets.JSON_TYPES)
        if not f:
            return
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            # A run's config.json holds the request under "request".
            req = AnalysisRequest.from_dict(data.get("request", data))
            self.state.requestImported.emit(req)
        except Exception as e:
            self.popups.error("Import settings", e)

    def info_text(self) -> str:
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
        return "\n".join(lines)

    def refresh(self):
        dpg.set_value("set_info", self.info_text())
