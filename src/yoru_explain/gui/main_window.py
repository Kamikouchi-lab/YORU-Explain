# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""The main window: five tabs over one shared state."""

from __future__ import annotations

from PyQt5 import QtWidgets

from yoru_explain import __version__
from yoru_explain.gui.analyze_tab import AnalyzeTab
from yoru_explain.gui.explore_tab import ExploreTab
from yoru_explain.gui.project_tab import ProjectTab
from yoru_explain.gui.qtutil import error_box
from yoru_explain.gui.settings_tab import SettingsTab
from yoru_explain.gui.state import AppState, project_name
from yoru_explain.gui.statistics_tab import StatisticsTab


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, state: AppState = None):
        super().__init__()
        self.state = state or AppState(self)
        self.setWindowTitle(f"YORU-Explain {__version__}")
        self.resize(1500, 920)
        self.tabs = QtWidgets.QTabWidget()
        self.project_tab = ProjectTab(self.state)
        self.explore_tab = ExploreTab(self.state)
        self.analyze_tab = AnalyzeTab(self.state)
        self.statistics_tab = StatisticsTab(self.state)
        self.settings_tab = SettingsTab(self.state, self.analyze_tab)
        for name, tab in (("Project", self.project_tab), ("Explore", self.explore_tab),
                          ("Analyze", self.analyze_tab), ("Statistics", self.statistics_tab),
                          ("Settings", self.settings_tab)):
            self.tabs.addTab(tab, name)
        self.setCentralWidget(self.tabs)
        self.status_label = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self.status_label, 1)
        self.state.projectChanged.connect(self._status)
        self.state.adapterChanged.connect(self._status)
        self.state.tasks.busyChanged.connect(self._status)
        self.state.showResult.connect(lambda *_: self.tabs.setCurrentWidget(self.explore_tab))
        self.state.tasks.failed.connect(self._unhandled)
        self._status()

    def _status(self, *_):
        a = self.state.adapter
        model = (f"{self.state.model_id} ({a.name} on {a.model_info().get('device')})"
                 if a is not None else "no model")
        busy = f"  |  running: {self.state.tasks.current}" if self.state.tasks.busy else ""
        self.status_label.setText(f"Project: {project_name(self.state.project)}  |  Model: {model}{busy}")

    def _unhandled(self, name, text, exc):
        # Tabs report their own tasks; anything else still reaches the user.
        if name not in ("analysis", "load_model", "check_seeking"):
            error_box(self, name, exc)

    def closeEvent(self, event):
        if self.state.tasks.busy:
            if QtWidgets.QMessageBox.question(
                    self, "YORU-Explain", "A task is running. Cancel it and quit?") != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
            # Cancellation is checked between batches and events; let the task
            # get there, so its run folder is closed with a manifest.
            self.state.tasks.cancel()
            self.statusBar().showMessage("Stopping the running task...")
            if not self.state.tasks.wait_idle(60.0):
                from yoru_explain.core.logs import get_logger

                get_logger("gui").warning("quit with task %r still running", self.state.tasks.current)
        try:
            self.state.save_project()
        except Exception:
            pass
        self.analyze_tab.detach_log()
        self.state.shutdown()
        super().closeEvent(event)
