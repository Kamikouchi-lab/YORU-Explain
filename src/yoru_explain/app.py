# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Starting the GUI: ``python -m yoru_explain`` (or ``yoru-explain``).

There is no command line to parse; :func:`main` takes its options as
function arguments, for scripts and tests.
"""

from __future__ import annotations

import sys
from typing import Optional


def main(project_dir: Optional[str] = None, *, exit_after_ms: Optional[int] = None) -> int:
    """Open the window; with *project_dir*, open (or create) that project first.

    *exit_after_ms* closes the window after that long (smoke tests).
    """
    import yoru_explain.gui  # noqa: F401  imports torch before PyQt5 (see gui/__init__.py)
    from PyQt5 import QtCore, QtWidgets

    from yoru_explain.core.logs import log_exception, setup_logging

    log = setup_logging()
    app = QtWidgets.QApplication.instance()
    if app is None:
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("YORU-Explain")
    app.setStyle("Fusion")

    def excepthook(exc_type, exc, tb):
        log_exception("unhandled exception", exc)
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = excepthook

    from yoru_explain.gui.main_window import MainWindow

    window = MainWindow()
    if project_dir:
        from yoru_explain.io.project import Project

        window.state.set_project(Project.open_or_create(project_dir))
    window.show()
    log.info("YORU-Explain started")
    if exit_after_ms is not None:
        QtCore.QTimer.singleShot(int(exit_after_ms), window.close)
    return app.exec_()
