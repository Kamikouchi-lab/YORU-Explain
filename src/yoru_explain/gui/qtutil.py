# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Small Qt helpers: images, log forwarding, message boxes."""

from __future__ import annotations

import logging
import traceback

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


def bgr_to_qimage(image: np.ndarray) -> QtGui.QImage:
    """A QImage that owns a copy of *image* (BGR or grey uint8)."""
    a = np.ascontiguousarray(image)
    h, w = a.shape[:2]
    if a.ndim == 2:
        q = QtGui.QImage(a.data, w, h, w, QtGui.QImage.Format_Grayscale8)
    else:
        rgb = np.ascontiguousarray(a[..., 2::-1])
        q = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    return q.copy()


class _Emitter(QtCore.QObject):
    record = QtCore.pyqtSignal(str, int)


class QtLogHandler(logging.Handler):
    """Forwards log records to the GUI thread as ``(text, level)``."""

    def __init__(self, level=logging.INFO):
        super().__init__(level)
        self.emitter = _Emitter()
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s",
                                            "%H:%M:%S"))

    def emit(self, record):
        try:
            self.emitter.record.emit(self.format(record), record.levelno)
        except Exception:
            pass


def error_box(parent, title: str, exc_or_text) -> None:
    """Show an error, and log it with its traceback."""
    from yoru_explain.core.logs import log_exception

    if isinstance(exc_or_text, BaseException):
        log_exception(title, exc_or_text)
        text = f"{type(exc_or_text).__name__}: {exc_or_text}"
        detail = "".join(traceback.format_exception(type(exc_or_text), exc_or_text,
                                                    exc_or_text.__traceback__))
    else:
        detail = str(exc_or_text)
        lines = detail.strip().splitlines()
        text = lines[-1] if lines else title
    box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Critical, title, text,
                                QtWidgets.QMessageBox.Ok, parent)
    box.setDetailedText(detail)
    box.open()


def form_row(layout: QtWidgets.QFormLayout, label: str, widget, tip: str = ""):
    if tip:
        widget.setToolTip(tip)
    layout.addRow(label, widget)
    return widget


def table(headers, *, stretch_last=True, select_rows=True) -> QtWidgets.QTableWidget:
    t = QtWidgets.QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    if select_rows:
        t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    t.horizontalHeader().setStretchLastSection(stretch_last)
    t.setAlternatingRowColors(True)
    return t


def fill_table(t: QtWidgets.QTableWidget, rows, *, data=None) -> None:
    """Replace *t*'s rows; ``data[i]`` is stored on column 0 under UserRole."""
    t.setSortingEnabled(False)
    t.setRowCount(len(rows))
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            item = QtWidgets.QTableWidgetItem("" if value is None else str(value))
            if j == 0 and data is not None:
                item.setData(QtCore.Qt.UserRole, data[i])
            t.setItem(i, j, item)
    t.resizeColumnsToContents()


def selected_data(t: QtWidgets.QTableWidget):
    rows = sorted({i.row() for i in t.selectedIndexes()})
    out = []
    for r in rows:
        item = t.item(r, 0)
        if item is not None:
            out.append(item.data(QtCore.Qt.UserRole))
    return out


def fmt(value, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v != v:  # NaN
        return ""
    return f"{v:.{digits}g}"
