# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Starting the GUI: ``python -m yoru_explain`` (or ``yoru-explain``).

There is no command line to parse; :func:`main` takes its options as
function arguments, for scripts and tests.
"""

from __future__ import annotations

import sys
from typing import Optional


def main(project_dir: Optional[str] = None, *, exit_after_ms: Optional[int] = None,
         screenshot: Optional[str] = None, tab: Optional[str] = None) -> int:
    """Open the window; with *project_dir*, open (or create) that project first.

    *tab* shows that tab first (``"project"``, ``"explore"``, ``"analyze"``,
    ``"statistics"``, ``"settings"``).  *exit_after_ms* closes the window after
    that long and *screenshot* saves it to that PNG just before (smoke tests).
    """
    from yoru_explain.core.logs import log_exception, setup_logging

    log = setup_logging()

    def excepthook(exc_type, exc, tb):
        log_exception("unhandled exception", exc)
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = excepthook

    from yoru_explain.gui.main_window import run

    log.info("YORU-Explain started")
    return run(project_dir, exit_after_ms=exit_after_ms, screenshot=screenshot, tab=tab)
