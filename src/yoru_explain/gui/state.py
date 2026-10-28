# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""What the tabs share: the open project, the loaded model, the workers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional


class Signal:
    """A list of callbacks, called in order by :meth:`emit` on the render thread.

    A callback that raises does not stop the others; the exception goes to
    *on_error* ``(context, exc)``, or to the log.
    """

    def __init__(self, on_error: Optional[Callable[[str, BaseException], None]] = None):
        self._slots: List[Callable] = []
        self._on_error = on_error

    def connect(self, fn: Callable) -> None:
        self._slots.append(fn)

    def disconnect(self, fn: Callable) -> None:
        self._slots = [s for s in self._slots if s is not fn]

    def emit(self, *args) -> None:
        for fn in list(self._slots):
            try:
                fn(*args)
            except Exception as exc:
                name = getattr(fn, "__qualname__", repr(fn))
                if self._on_error is not None:
                    self._on_error(f"Error in {name}", exc)
                else:
                    from yoru_explain.core.logs import log_exception

                    log_exception(f"Error in {name}", exc)


class AppState:
    def __init__(self):
        #: Set by the window: ``(context, exc)`` -> log and show.  Callback
        #: errors go there instead of breaking the render loop.
        self.report_error: Callable[[str, BaseException], None] = self._log_error
        err = lambda context, exc: self.report_error(context, exc)  # noqa: E731
        self.projectChanged = Signal(err)
        self.adapterChanged = Signal(err)
        self.eventsChanged = Signal(err)
        self.runsChanged = Signal(err)
        #: ``(run directory, event id)``: show this result in Explore.
        self.showResult = Signal(err)
        #: An AnalysisRequest to put into the Analyze tab (settings import).
        self.requestImported = Signal(err)
        self.project = None
        self.adapter = None
        self.model_id: Optional[str] = None
        self.device = "auto"
        self.selected_events: List[str] = []
        from yoru_explain.gui.worker import FrameService, TaskRunner

        self.tasks = TaskRunner(err)
        self.frames = FrameService(err)
        self._runs: Dict[str, object] = {}

    @staticmethod
    def _log_error(context: str, exc: BaseException) -> None:
        from yoru_explain.core.logs import log_exception

        log_exception(context, exc)

    def poll(self) -> None:
        """Deliver what the workers finished; once per frame."""
        self.tasks.poll()
        self.frames.poll()

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


def project_name(project) -> str:
    return f"{project.name} ({Path(project.root)})" if project is not None else "(no project)"
