# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Work off the GUI thread.

:class:`TaskRunner` runs one long task at a time -- loading a model, an
analysis run -- on its own thread, with progress and cancellation.
:class:`FrameService` decodes frames and runs the detector for the Explore
view on another thread; when requests pile up (dragging the slider) only the
latest is served.  Results come back as Qt signals, so widgets are only ever
touched on the GUI thread.
"""

from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from PyQt5 import QtCore


class _Worker(QtCore.QObject):
    done = QtCore.pyqtSignal(str, object)
    error = QtCore.pyqtSignal(str, str, object)

    @QtCore.pyqtSlot(str, object, object, object)
    def run(self, name, fn, args, kwargs):
        try:
            result = fn(*args, **kwargs)
        except BaseException as e:  # reported to the GUI, never swallowed
            self.error.emit(name, traceback.format_exc(), e)
            return
        self.done.emit(name, result)


class TaskRunner(QtCore.QObject):
    """One background task at a time."""

    progress = QtCore.pyqtSignal(float, str)
    finished = QtCore.pyqtSignal(str, object)
    failed = QtCore.pyqtSignal(str, str, object)
    busyChanged = QtCore.pyqtSignal(bool)
    _start = QtCore.pyqtSignal(str, object, object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = QtCore.QThread()
        self._thread.setObjectName("yoru-explain-task")
        self._worker = _Worker()
        self._worker.moveToThread(self._thread)
        self._start.connect(self._worker.run)
        self._worker.done.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._thread.start()
        self._cancel = threading.Event()
        self._busy = False
        self.current: Optional[str] = None

    @property
    def busy(self) -> bool:
        return self._busy

    def submit(self, name: str, fn: Callable, *args, **kwargs) -> bool:
        if self._busy:
            return False
        self._busy = True
        self.current = name
        self._cancel.clear()
        self.busyChanged.emit(True)
        self._start.emit(name, fn, args, kwargs)
        return True

    def cancel(self) -> None:
        self._cancel.set()

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def report(self, fraction: float, message: str = "") -> None:
        """Called from the task's thread; delivered on the GUI thread."""
        self.progress.emit(float(fraction), message)

    def _on_done(self, name, result):
        self._busy = False
        self.current = None
        self.busyChanged.emit(False)
        self.finished.emit(name, result)

    def _on_error(self, name, text, exc):
        self._busy = False
        self.current = None
        self.busyChanged.emit(False)
        self.failed.emit(name, text, exc)

    def wait_idle(self, timeout_s: float) -> bool:
        """Keep the GUI responsive until the running task returns; False on timeout."""
        import time

        from PyQt5 import QtWidgets

        end = time.monotonic() + timeout_s
        while self._busy and time.monotonic() < end:
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)
            time.sleep(0.01)
        return not self._busy

    def shutdown(self, wait_ms: int = 5000) -> None:
        self._cancel.set()
        self._thread.quit()
        self._thread.wait(wait_ms)


@dataclass
class FrameRequest:
    source_id: str
    path: str
    v_flip: bool
    h_flip: bool
    index: int
    detect: bool = True


@dataclass
class FrameResult:
    request: FrameRequest
    frame: Any = None              # io.frames.LoadedFrame
    detections: List[Any] = field(default_factory=list)
    #: The adapter that produced *detections* (the model may change meanwhile).
    adapter: Any = None
    error: Optional[str] = None
    exception: Optional[BaseException] = None


class _FrameWorker(QtCore.QObject):
    ready = QtCore.pyqtSignal(object)
    indexed = QtCore.pyqtSignal(str, object)

    def __init__(self, service: "FrameService"):
        super().__init__()
        self._service = service
        self._readers: Dict[str, Any] = {}

    def _reader(self, req: FrameRequest):
        from yoru_explain.io.frames import open_source

        key = f"{req.source_id}|{req.path}|{req.v_flip}|{req.h_flip}"
        if key not in self._readers:
            self._readers[key] = open_source(req.path, v_flip=req.v_flip, h_flip=req.h_flip)
        return self._readers[key]

    @QtCore.pyqtSlot()
    def drain(self):
        while True:
            req = self._service._take()
            if req is None:
                return
            if isinstance(req, _IndexRequest):
                try:
                    reader = self._reader(req.frame)
                    summary = reader.build_index(cancel=self._service.closing)                         if hasattr(reader, "build_index") else {"status": "not_applicable"}
                except BaseException as e:
                    summary = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
                self.indexed.emit(req.frame.source_id, summary)
                continue
            out = FrameResult(req)
            try:
                reader = self._reader(req)
                out.frame = reader.read(req.index, exact=True)
                adapter = self._service.adapter
                if req.detect and adapter is not None:
                    out.detections = adapter.detect(out.frame.image)
                    out.adapter = adapter
            except BaseException as e:
                out.error = traceback.format_exc()
                out.exception = e
            self.ready.emit(out)

    @QtCore.pyqtSlot()
    def close_all(self):
        for r in self._readers.values():
            try:
                r.close()
            except Exception:
                pass
        self._readers.clear()


@dataclass
class _IndexRequest:
    frame: FrameRequest


class FrameService(QtCore.QObject):
    """Frames (and their detections) for the Explore view; latest request wins."""

    frameReady = QtCore.pyqtSignal(object)
    indexReady = QtCore.pyqtSignal(str, object)
    _wake = QtCore.pyqtSignal()
    _close = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.adapter = None
        self._lock = threading.Lock()
        self._closing = threading.Event()
        self._pending: Optional[FrameRequest] = None
        self._index_queue: List[_IndexRequest] = []
        self._thread = QtCore.QThread()
        self._thread.setObjectName("yoru-explain-frames")
        self._worker = _FrameWorker(self)
        self._worker.moveToThread(self._thread)
        self._wake.connect(self._worker.drain)
        self._close.connect(self._worker.close_all)
        self._worker.ready.connect(self.frameReady)
        self._worker.indexed.connect(self.indexReady)
        self._thread.start()

    def request(self, req: FrameRequest) -> None:
        with self._lock:
            self._pending = req
        self._wake.emit()

    def build_index(self, req: FrameRequest) -> None:
        with self._lock:
            self._index_queue.append(_IndexRequest(req))
        self._wake.emit()

    def _take(self):
        with self._lock:
            if self._index_queue:
                return self._index_queue.pop(0)
            req, self._pending = self._pending, None
            return req

    def closing(self) -> bool:
        return self._closing.is_set()

    def reset(self) -> None:
        """Forget open readers (the project changed)."""
        self._close.emit()

    def shutdown(self, wait_ms: int = 5000) -> None:
        self._closing.set()   # stops an index pass at its next check
        self._close.emit()
        self._thread.quit()
        self._thread.wait(wait_ms)
