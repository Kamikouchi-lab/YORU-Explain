# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Small DearPyGui pieces the tabs share: tables, popups, file dialogs, the log.

File dialogs are tkinter's, as in YORU's own windows and YORU-Tracker; they
block the render loop while open, like a native dialog.  Message boxes,
questions and forms are DearPyGui modal windows and never block: the answer
arrives as a callback.
"""

from __future__ import annotations

import collections
import logging
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

from yoru_explain.gui import theme as th

VIDEO_TYPES = (("Videos and images", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v *.mpg *.mpeg "
                                     "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
               ("All files", "*.*"))
MODEL_TYPES = (("Models (PyTorch, ONNX)", "*.pt *.pth *.onnx"), ("All files", "*.*"))
CSV_TYPES = (("CSV", "*.csv"), ("All files", "*.*"))
JSON_TYPES = (("JSON", "*.json"), ("All files", "*.*"))


def fmt(value, digits: int = 4) -> str:
    """A number for a table cell; empty for missing values and NaN, never 0."""
    if value is None or value == "":
        return ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v != v:  # NaN
        return ""
    return f"{v:.{digits}g}"


def short(path, n: int = 48) -> str:
    text = str(path)
    return text if len(text) <= n else "..." + text[-(n - 3):]


def texture_data(image_bgr: np.ndarray) -> np.ndarray:
    """A BGR (or grey) uint8 image as the flat float32 RGBA a dynamic texture takes."""
    a = np.ascontiguousarray(image_bgr)
    code = cv2.COLOR_GRAY2RGBA if a.ndim == 2 else cv2.COLOR_BGR2RGBA
    return (cv2.cvtColor(a, code).astype(np.float32) * (1.0 / 255.0)).ravel()


def tooltip(item, text: str) -> None:
    with dpg.tooltip(item):
        dpg.add_text(text, wrap=420)


def heading(text: str, themes: th.Themes) -> int:
    item = dpg.add_text(text)
    dpg.bind_item_theme(item, themes.header)
    dpg.add_separator()
    return item


def muted(text: str, themes: th.Themes, **kwargs) -> int:
    item = dpg.add_text(text, **kwargs)
    dpg.bind_item_theme(item, themes.muted)
    return item


def open_folder(path) -> None:
    folder = str(path)
    if sys.platform == "win32":
        os.startfile(folder)  # noqa: S606 - a folder the user asked to open
    else:
        import subprocess

        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", folder])


# -- file dialogs (tkinter) ---------------------------------------------------

def _tk_root():
    import tkinter

    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    return root


def _initial_dir(current: Optional[str]) -> Optional[str]:
    if not current:
        return None
    p = Path(current)
    if p.is_dir():
        return str(p)
    return str(p.parent) if p.parent.exists() else None


def ask_open_file(title: str, filetypes=(("All files", "*.*"),), current: Optional[str] = None) -> str:
    from tkinter import filedialog

    root = _tk_root()
    try:
        return filedialog.askopenfilename(title=title, filetypes=list(filetypes),
                                          initialdir=_initial_dir(current)) or ""
    finally:
        root.destroy()


def ask_open_files(title: str, filetypes=(("All files", "*.*"),),
                   current: Optional[str] = None) -> List[str]:
    from tkinter import filedialog

    root = _tk_root()
    try:
        return list(filedialog.askopenfilenames(title=title, filetypes=list(filetypes),
                                                initialdir=_initial_dir(current)) or ())
    finally:
        root.destroy()


def ask_directory(title: str, current: Optional[str] = None) -> str:
    from tkinter import filedialog

    root = _tk_root()
    try:
        return filedialog.askdirectory(title=title, initialdir=_initial_dir(current)) or ""
    finally:
        root.destroy()


def ask_save_file(title: str, filetypes, default_ext: str, initial_file: str = "",
                  current: Optional[str] = None) -> str:
    from tkinter import filedialog

    root = _tk_root()
    try:
        return filedialog.asksaveasfilename(title=title, filetypes=list(filetypes),
                                            defaultextension=default_ext, initialfile=initial_file,
                                            initialdir=_initial_dir(current)) or ""
    finally:
        root.destroy()


# -- popups -------------------------------------------------------------------

def _centre(width: int, height: int) -> List[int]:
    try:
        vw, vh = dpg.get_viewport_client_width(), dpg.get_viewport_client_height()
    except Exception:  # no viewport (tests)
        vw, vh = 1400, 900
    return [max(0, (vw - width) // 2), max(0, (vh - height) // 3)]


class Popups:
    """Modal message boxes, questions, forms and error reports, one at a time.

    Each call returns the popup's tag.  ``shown`` keeps ``(kind, title, text)``
    of every popup asked for, for tests and the log.

    DearPyGui 1.11 closes a modal at once when it is opened while another
    modal is open, or in the frame one was closed (measured), so popups are
    queued: the next opens on a :meth:`tick` once no popup is showing and
    the last one closed has been off screen for a frame.  A closed popup is
    hidden at once and deleted a few frames later.
    """

    WIDTH = 560

    def __init__(self, themes: th.Themes):
        self.themes = themes
        self.shown: List[Tuple[str, str, str]] = []
        self._ticks = 0
        self._queue: List[Tuple[int, Callable[[int], None]]] = []
        self._open: List[int] = []
        self._closed: List[Tuple[int, int]] = []

    # -- the queue -------------------------------------------------------------

    def _request(self, kind: str, title: str, text: str, build: Callable[[int], None]) -> int:
        self.shown.append((kind, title, text))
        tag = dpg.generate_uuid()
        self._queue.append((tag, build))
        self._pump()
        return tag

    def _pump(self) -> None:
        self._open = [t for t in self._open if dpg.does_item_exist(t)
                      and all(t != c for _, c in self._closed)]
        if not self._queue or self._open:
            return
        if any(self._ticks - when < 2 for when, _ in self._closed):
            return
        tag, build = self._queue.pop(0)
        build(tag)
        dpg.bind_item_theme(tag, self.themes.window)
        self._open.append(tag)

    def tick(self) -> None:
        """Once per frame: delete closed popups, open the next one."""
        self._ticks += 1
        keep = []
        for when, tag in self._closed:
            if self._ticks - when >= 3:
                if dpg.does_item_exist(tag):
                    dpg.delete_item(tag)
            else:
                keep.append((when, tag))
        self._closed = keep
        self._pump()

    @property
    def showing(self) -> List[int]:
        """Popups on screen now (the queue's head)."""
        self._open = [t for t in self._open if dpg.does_item_exist(t)
                      and all(t != c for _, c in self._closed)]
        return list(self._open)

    @property
    def waiting(self) -> int:
        return len(self._queue)

    def close(self, tag) -> None:
        self._queue = [(t, b) for t, b in self._queue if t != tag]
        if dpg.does_item_exist(tag) and all(t != tag for _, t in self._closed):
            dpg.configure_item(tag, show=False)
            self._closed.append((self._ticks, tag))

    def _window(self, tag: int, title: str, width: int = WIDTH):
        return dpg.window(label=title, modal=True, tag=tag, width=width, autosize=True,
                          no_collapse=True, pos=_centre(width, 240),
                          on_close=lambda s: self.close(s))

    # -- the kinds ---------------------------------------------------------------

    def message(self, title: str, text: str, *, kind: str = "info") -> int:
        def build(tag):
            with self._window(tag, title):
                t = dpg.add_text(text, wrap=self.WIDTH - 30)
                if kind == "warning":
                    dpg.bind_item_theme(t, self.themes.warn)
                dpg.add_spacer(height=6)
                dpg.add_button(label="OK", width=90, callback=lambda: self.close(tag))

        return self._request(kind, title, text, build)

    def confirm(self, title: str, text: str, on_yes: Callable[[], None], *,
                yes: str = "Yes", no: str = "No") -> int:
        def build(tag):
            def answer(ok):
                self.close(tag)
                if ok:
                    on_yes()

            with self._window(tag, title):
                dpg.add_text(text, wrap=self.WIDTH - 30)
                dpg.add_spacer(height=6)
                with dpg.group(horizontal=True):
                    b = dpg.add_button(label=yes, callback=lambda: answer(True))
                    dpg.bind_item_theme(b, self.themes.primary_button)
                    dpg.add_button(label=no, callback=lambda: answer(False))

        return self._request("question", title, text, build)

    def form(self, title: str, fields: Sequence[Tuple[str, str, object]],
             on_ok: Callable[[Dict[str, object]], None], *, ok: str = "OK") -> int:
        """*fields*: ``(key, label, default)``; ints get an int input, the rest text."""
        def build(tag):
            inputs = {}

            def accept():
                values = {k: dpg.get_value(i) for k, i in inputs.items()}
                self.close(tag)
                on_ok(values)

            with self._window(tag, title):
                for key, label, default in fields:
                    dpg.add_text(label)
                    if isinstance(default, int) and not isinstance(default, bool):
                        inputs[key] = dpg.add_input_int(default_value=default, width=160, step=0)
                    else:
                        inputs[key] = dpg.add_input_text(default_value=str(default), width=-1)
                dpg.add_spacer(height=6)
                with dpg.group(horizontal=True):
                    b = dpg.add_button(label=ok, callback=lambda: accept())
                    dpg.bind_item_theme(b, self.themes.primary_button)
                    dpg.add_button(label="Cancel", width=90, callback=lambda: self.close(tag))

        return self._request("form", title, ", ".join(label for _, label, _ in fields), build)

    def error(self, title: str, exc_or_text) -> int:
        """Show an error with its traceback (folded), and log it."""
        from yoru_explain.core.logs import log_exception, log_file

        if isinstance(exc_or_text, BaseException):
            log_exception(title, exc_or_text)
            text = f"{type(exc_or_text).__name__}: {exc_or_text}"
            detail = "".join(traceback.format_exception(type(exc_or_text), exc_or_text,
                                                        exc_or_text.__traceback__))
        else:
            detail = str(exc_or_text)
            lines = detail.strip().splitlines()
            text = lines[-1] if lines else title
        width = 680

        def build(tag):
            with self._window(tag, title, width):
                t = dpg.add_text(text, wrap=width - 30)
                dpg.bind_item_theme(t, self.themes.error)
                with dpg.collapsing_header(label="Details"):
                    dpg.add_input_text(default_value=detail, multiline=True, readonly=True,
                                       width=-1, height=220)
                muted(f"Logged to {log_file()}", self.themes, wrap=width - 30)
                dpg.add_spacer(height=6)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Close", width=90, callback=lambda: self.close(tag))
                    dpg.add_button(label="Copy details",
                                   callback=lambda: dpg.set_clipboard_text(detail))

        return self._request("error", title, text, build)


# -- tables ---------------------------------------------------------------------

def _sort_key(text: str):
    """Numbers by value, then text; empty cells last."""
    if text == "":
        return (2, 0.0, "")
    try:
        return (0, float(text), "")
    except ValueError:
        return (1, 0.0, text.lower())


class SelectTable:
    """A table whose rows are selected by clicking: Ctrl/Shift extend with *multi*.

    Each row carries a *key* (any value); ``selected`` lists the keys of the
    selected rows in display order.  *on_select* is called after a click
    changes the selection, *on_activate* with the key of a double-clicked row.
    """

    def __init__(self, tag: str, headers: Sequence[str], *, multi: bool = False, height: int = 200,
                 sortable: bool = False, on_select: Optional[Callable[[list], None]] = None,
                 on_activate: Optional[Callable[[object], None]] = None):
        self.tag = tag
        self.headers = list(headers)
        self.multi = multi
        self.height = height
        self.sortable = sortable
        self.on_select = on_select
        self.on_activate = on_activate
        self._rows: List[Tuple[List[str], object, Optional[tuple]]] = []
        self._order: List[int] = []
        self._selected: List[object] = []
        self._anchor = None
        self._selectables: List[int] = []
        self._sort: Optional[Tuple[int, bool]] = None
        self._handlers = None

    def build(self) -> int:
        with dpg.item_handler_registry() as self._handlers:
            dpg.add_item_double_clicked_handler(callback=self._double_clicked)
        # Not resizable: fixed-fit columns then follow their contents whenever
        # the rows change, instead of keeping the header's width.
        with dpg.table(tag=self.tag, header_row=True, resizable=False, row_background=True,
                       borders_innerV=True, borders_outerV=True, borders_outerH=True,
                       scrollY=True, scrollX=True, freeze_rows=1, height=self.height,
                       policy=dpg.mvTable_SizingFixedFit, sortable=self.sortable,
                       sort_tristate=self.sortable,
                       callback=(lambda s, a: self._sorted(a)) if self.sortable else None):
            for h in self.headers:
                dpg.add_table_column(label=h)
        return self.tag

    # -- contents ----------------------------------------------------------

    def set_rows(self, rows: Sequence[Sequence[object]], keys: Optional[Sequence[object]] = None,
                 colors: Optional[Sequence[Optional[tuple]]] = None) -> None:
        """Replace every row; the selection keeps the keys that are still there."""
        keys = list(range(len(rows))) if keys is None else list(keys)
        self._rows = [(["" if v is None else str(v) for v in row], keys[i],
                       colors[i] if colors is not None else None) for i, row in enumerate(rows)]
        present = set(map(_hashable, keys))
        self._selected = [k for k in self._selected if _hashable(k) in present]
        self._apply_order()
        self._render()

    @property
    def row_count(self) -> int:
        return len(self._rows)

    def cell(self, i: int, j: int) -> str:
        """Text of column *j* of the *i*-th displayed row."""
        return self._rows[self._order[i]][0][j]

    def key_at(self, i: int):
        return self._rows[self._order[i]][1]

    @property
    def keys(self) -> list:
        return [self._rows[r][1] for r in self._order]

    # -- selection -----------------------------------------------------------

    @property
    def selected(self) -> list:
        chosen = set(map(_hashable, self._selected))
        return [k for k in self.keys if _hashable(k) in chosen]

    def select(self, keys, *, notify: bool = False) -> None:
        self._selected = list(keys)
        self._anchor = self._selected[-1] if self._selected else None
        self._sync()
        if notify and self.on_select is not None:
            self.on_select(self.selected)

    def select_row(self, i: int, *, notify: bool = False) -> None:
        self.select([self.key_at(i)], notify=notify)

    def click(self, i: int, *, ctrl: bool = False, shift: bool = False) -> None:
        """What a click on displayed row *i* does (the callback and tests use it)."""
        key = self.key_at(i)
        if self.multi and ctrl:
            if _hashable(key) in set(map(_hashable, self._selected)):
                self._selected = [k for k in self._selected if _hashable(k) != _hashable(key)]
            else:
                self._selected.append(key)
            self._anchor = key
        elif self.multi and shift and self._anchor is not None and self._anchor in self.keys:
            a = self.keys.index(self._anchor)
            lo, hi = sorted((a, i))
            self._selected = self.keys[lo:hi + 1]
        else:
            self._selected = [key]
            self._anchor = key
        self._sync()
        if self.on_select is not None:
            self.on_select(self.selected)

    def _clicked(self, user_data) -> None:
        ctrl = dpg.is_key_down(dpg.mvKey_Control)
        shift = dpg.is_key_down(dpg.mvKey_Shift)
        self.click(int(user_data), ctrl=ctrl, shift=shift)

    def _double_clicked(self, sender, app_data) -> None:
        item = app_data[1] if isinstance(app_data, (list, tuple)) and len(app_data) > 1 else app_data
        try:
            i = int(dpg.get_item_user_data(item))
        except Exception:
            return
        if self.on_activate is not None and 0 <= i < len(self._order):
            self.on_activate(self.key_at(i))

    def _sync(self) -> None:
        chosen = set(map(_hashable, self._selected))
        for i, item in enumerate(self._selectables):
            if dpg.does_item_exist(item):
                dpg.set_value(item, _hashable(self.key_at(i)) in chosen)

    # -- drawing -------------------------------------------------------------

    def _apply_order(self) -> None:
        order = list(range(len(self._rows)))
        if self._sort is not None:
            col, descending = self._sort
            order.sort(key=lambda r: _sort_key(self._rows[r][0][col]), reverse=descending)
        self._order = order

    def _sorted(self, specs) -> None:
        if not specs:
            self._sort = None
        else:
            column, direction = specs[0]
            cols = list(dpg.get_item_children(self.tag, 0) or [])
            self._sort = (cols.index(column), direction < 0) if column in cols else None
        self._apply_order()
        self._render()

    def _render(self) -> None:
        for row in dpg.get_item_children(self.tag, 1) or []:
            dpg.delete_item(row)
        self._selectables = []
        chosen = set(map(_hashable, self._selected))
        for d, r in enumerate(self._order):
            cells, key, color = self._rows[r]
            with dpg.table_row(parent=self.tag):
                sel = dpg.add_selectable(label=cells[0] if cells else "", span_columns=True,
                                         default_value=_hashable(key) in chosen, user_data=d,
                                         callback=lambda s, a, u: self._clicked(u))
                dpg.bind_item_handler_registry(sel, self._handlers)
                self._selectables.append(sel)
                for text in cells[1:]:
                    if color is not None:
                        dpg.add_text(text, color=color)
                    else:
                        dpg.add_text(text)


def _hashable(key):
    return tuple(key) if isinstance(key, list) else key


# -- the log --------------------------------------------------------------------

class LogBuffer(logging.Handler):
    """Collects log records from any thread; the GUI drains them each frame."""

    def __init__(self, level=logging.INFO, maxlen: int = 5000):
        super().__init__(level)
        self._lines = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s",
                                            "%H:%M:%S"))

    def emit(self, record):
        try:
            text = self.format(record)
            with self._lock:
                self._lines.append((text, record.levelno))
        except Exception:
            pass

    def drain(self) -> List[Tuple[str, int]]:
        with self._lock:
            out = list(self._lines)
            self._lines.clear()
        return out


class LogPanel:
    """The last *max_lines* log lines, warnings and errors in colour."""

    def __init__(self, tag: str, themes: th.Themes, max_lines: int = 2000):
        self.tag = tag
        self.themes = themes
        self.max_lines = max_lines
        self.lines: collections.deque = collections.deque(maxlen=max_lines)
        self._items: collections.deque = collections.deque()

    def build(self, height: int = 200) -> int:
        with dpg.child_window(tag=self.tag, height=height, border=True, horizontal_scrollbar=True):
            pass
        return self.tag

    def append(self, text: str, level: int) -> None:
        self.lines.append(text)
        follow = True
        try:
            follow = dpg.get_y_scroll(self.tag) >= dpg.get_y_scroll_max(self.tag) - 4
        except Exception:
            pass
        item = dpg.add_text(text, parent=self.tag)
        if level >= logging.ERROR:
            dpg.bind_item_theme(item, self.themes.error)
        elif level >= logging.WARNING:
            dpg.bind_item_theme(item, self.themes.warn)
        self._items.append(item)
        while len(self._items) > self.max_lines:
            dpg.delete_item(self._items.popleft())
        if follow:
            dpg.set_y_scroll(self.tag, -1.0)  # -1: the bottom, once laid out

    def text(self) -> str:
        return "\n".join(self.lines)
