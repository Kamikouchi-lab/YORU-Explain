# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""What the tabs share: the open project, the loaded model, the workers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from PyQt5 import QtCore

from yoru_explain.gui.worker import FrameService, TaskRunner


class AppState(QtCore.QObject):
    projectChanged = QtCore.pyqtSignal()
    adapterChanged = QtCore.pyqtSignal()
    eventsChanged = QtCore.pyqtSignal()
    runsChanged = QtCore.pyqtSignal()
    #: (run directory, event id): show this result in Explore.
    showResult = QtCore.pyqtSignal(str, str)
    #: An AnalysisRequest to put into the Analyze tab (settings import).
    requestImported = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.adapter = None
        self.model_id: Optional[str] = None
        self.device = "auto"
        self.selected_events: List[str] = []
        self.tasks = TaskRunner(self)
        self.frames = FrameService(self)
        self._runs: Dict[str, object] = {}

    # -- project ---------------------------------------------------------

    def set_project(self, project) -> None:
        self.project = project
        self.selected_events = []
        self._runs.clear()
        self.frames.reset()
        self.projectChanged.emit()
        self.eventsChanged.emit()
        self.runsChanged.emit()

    def save_project(self) -> None:
        if self.project is not None:
            self.project.save()

    # -- model -----------------------------------------------------------

    def set_adapter(self, adapter, model_id: Optional[str]) -> None:
        self.adapter = adapter
        self.model_id = model_id
        self.frames.adapter = adapter
        self.adapterChanged.emit()

    # -- runs ------------------------------------------------------------

    def run_readers(self):
        """Every run of the project, newest first (cached until runs change)."""
        from yoru_explain.io.runs import RunReader

        if self.project is None:
            return []
        out = []
        for d in reversed(self.project.run_dirs()):
            key = str(d)
            if key not in self._runs:
                try:
                    self._runs[key] = RunReader(d)
                except Exception:
                    continue
            out.append(self._runs[key])
        return out

    def runs_changed(self) -> None:
        self._runs.clear()
        self.runsChanged.emit()

    def results_for(self, event_id: str):
        """``[(reader, row)]`` of every result for *event_id*."""
        found = []
        for reader in self.run_readers():
            for row in reader.results:
                if row.get("event_id") == event_id:
                    found.append((reader, row))
        return found

    def shutdown(self) -> None:
        self.tasks.shutdown()
        self.frames.shutdown()


def short(path, n: int = 48) -> str:
    text = str(path)
    return text if len(text) <= n else "..." + text[-(n - 3):]


def project_name(project) -> str:
    return f"{project.name} ({Path(project.root)})" if project is not None else "(no project)"
