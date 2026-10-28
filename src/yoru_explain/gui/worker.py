# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Work off the render thread.

:class:`TaskRunner` runs one long task at a time -- loading a model, an
analysis run, a seek check -- on its own thread, with progress and
cancellation.  :class:`FrameService` decodes frames and runs the detector for
the Explore view on another thread; when requests pile up (dragging the
slider) only the latest is served.

Neither ever calls into the GUI.  Their threads leave results behind, and
the render loop calls :meth:`poll` once per frame, which delivers them
through :class:`~yoru_explain.gui.state.Signal` objects on the render thread
-- so widgets are only ever touched there.
"""

from __future__ import annotations

import collections
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from yoru_explain.gui.state import Signal


class TaskRunner:
    """One background task at a time."""

    def __init__(self, on_slot_error=None):
        #: ``(fraction, message)``, the latest the task reported.
        self.progress = Signal(on_slot_error)
        #: ``(name, result)``
        self.finished = Signal(on_slot_error)
        #: ``(name, traceback text, exception)``
        self.failed = Signal(on_slot_error)
        #: ``(busy)``
        self.busyChanged = Signal(on_slot_error)
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._outcome = None
        self._progress = None
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
        self._outcome = None
        self._progress = None

        def work():
            try:
                result = fn(*args, **kwargs)
                outcome = ("done", result, None, None)
            except BaseException as e:  # reported to the GUI, never swallowed
                outcome = ("error", None, traceback.format_exc(), e)
            with self._lock:
                self._outcome = outcome

        self._thread = threading.Thread(target=work, name=f"yoru-explain-{name}", daemon=True)
        self._thread.start()
        self.busyChanged.emit(True)
        return True

    def cancel(self) -> None:
        self._cancel.set()

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def report(self, fraction: float, message: str = "") -> None:
        """Called from the task's thread; delivered by the next :meth:`poll`."""
        with self._lock:
            self._progress = (float(fraction), message)

    def poll(self) -> None:
        """Deliver progress and the outcome; on the render thread, once per frame."""
        with self._lock:
            progress, self._progress = self._progress, None
            outcome = self._outcome
            if outcome is not None:
                self._outcome = None
        if progress is not None and self._busy:
            self.progress.emit(*progress)
        if outcome is None:
            return
        name = self.current
        self._busy = False
        self.current = None
        self._thread = None
        self.busyChanged.emit(False)
        kind, result, text, exc = outcome
        if kind == "done":
            self.finished.emit(name, result)
        else:
            self.failed.emit(name, text, exc)

    def wait_idle(self, timeout_s: float) -> bool:
        """Wait for the running task to return (and deliver it); False on timeout."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout_s)
            if thread.is_alive():
                return False
        self.poll()
        return not self._busy

    def shutdown(self, timeout_s: float = 5.0) -> None:
        self._cancel.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout_s)


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


@dataclass
class _IndexRequest:
    frame: FrameRequest


class FrameService:
    """Frames (and their detections) for the Explore view; latest request wins."""

    def __init__(self, on_slot_error=None):
        #: ``(FrameResult)``
        self.frameReady = Signal(on_slot_error)
        #: ``(source_id, index summary)``
        self.indexReady = Signal(on_slot_error)
        self.adapter = None
        self._cond = threading.Condition()
        self._closing = threading.Event()
        self._reset = False
        self._pending: Optional[FrameRequest] = None
        self._index_queue: List[_IndexRequest] = []
        self._ready = collections.deque()
        self._readers: Dict[str, Any] = {}
        self._thread = threading.Thread(target=self._loop, name="yoru-explain-frames", daemon=True)
        self._thread.start()

    # -- the render thread's side -------------------------------------------

    def request(self, req: FrameRequest) -> None:
        with self._cond:
            self._pending = req
            self._cond.notify()

    def build_index(self, req: FrameRequest) -> None:
        with self._cond:
            self._index_queue.append(_IndexRequest(req))
            self._cond.notify()

    def reset(self) -> None:
        """Forget open readers (the project changed) before the next request."""
        with self._cond:
            self._reset = True
            self._cond.notify()

    def poll(self) -> None:
        while self._ready:
            kind, *payload = self._ready.popleft()
            if kind == "frame":
                self.frameReady.emit(*payload)
            else:
                self.indexReady.emit(*payload)

    def closing(self) -> bool:
        return self._closing.is_set()

    def wait_ready(self, timeout_s: float) -> bool:
        """Wait until something is ready to poll (tests); False on timeout."""
        end = time.monotonic() + timeout_s
        while not self._ready and time.monotonic() < end:
            time.sleep(0.005)
        return bool(self._ready)

    def shutdown(self, timeout_s: float = 5.0) -> None:
        self._closing.set()   # stops an index pass at its next check
        with self._cond:
            self._cond.notify()
        self._thread.join(timeout_s)

    # -- the worker thread --------------------------------------------------

    def _reader(self, req: FrameRequest):
        from yoru_explain.io.frames import open_source

        key = f"{req.source_id}|{req.path}|{req.v_flip}|{req.h_flip}"
        if key not in self._readers:
            self._readers[key] = open_source(req.path, v_flip=req.v_flip, h_flip=req.h_flip)
        return self._readers[key]

    def _close_readers(self) -> None:
        for r in self._readers.values():
            try:
                r.close()
            except Exception:
                pass
        self._readers.clear()

    def _take(self):
        with self._cond:
            while not (self._closing.is_set() or self._reset or self._index_queue
                       or self._pending is not None):
                self._cond.wait()
            if self._closing.is_set():
                return "close", None
            if self._reset:
                self._reset = False
                return "reset", None
            if self._index_queue:
                return "index", self._index_queue.pop(0)
            req, self._pending = self._pending, None
            return "frame", req

    def _loop(self) -> None:
        try:
            while True:
                kind, req = self._take()
                if kind == "close":
                    return
                if kind == "reset":
                    self._close_readers()
                elif kind == "index":
                    self._ready.append(("index", req.frame.source_id, self._index(req)))
                else:
                    self._ready.append(("frame", self._frame(req)))
        finally:
            self._close_readers()

    def _index(self, req: _IndexRequest) -> dict:
        try:
            reader = self._reader(req.frame)
            if not hasattr(reader, "build_index"):
                return {"status": "not_applicable"}
            return reader.build_index(cancel=self.closing)
        except BaseException as e:
            return {"status": "failed", "error": f"{type(e).__name__}: {e}"}

    def _frame(self, req: FrameRequest) -> FrameResult:
        out = FrameResult(req)
        try:
            reader = self._reader(req)
            out.frame = reader.read(req.index, exact=True)
            adapter = self.adapter
            if req.detect and adapter is not None:
                out.detections = adapter.detect(out.frame.image)
                out.adapter = adapter
        except BaseException as e:
            out.error = traceback.format_exc()
            out.exception = e
        return out
