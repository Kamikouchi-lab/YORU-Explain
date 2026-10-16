# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""The YORU-Explain log file, next to YORU's own, and per-run logs.

``~/.yoru/logs/yoru_explain.log`` (the directory moves with ``YORU_HOME``,
as YORU's ``yoru.log`` and YORU-Tracker's ``yoru_tracker.log`` do), rotated
at 5 MB with three backups.  Every error the GUI shows and every failed
analysis row is logged there with its traceback.  A run additionally writes
``runs/<run_id>/logs/run.log`` (see :func:`run_log`).
"""

from __future__ import annotations

import contextlib
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FILENAME = "yoru_explain.log"
LOGGER_NAME = "yoru_explain"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def log_dir() -> Path:
    """YORU's log directory, or ``$YORU_HOME/logs`` (``~/.yoru/logs``) without YORU."""
    try:
        from yoru.libs.user_paths import get_log_dir

        return Path(get_log_dir())
    except Exception:
        home = Path(os.environ.get("YORU_HOME") or Path.home() / ".yoru")
        path = home / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path


def log_file() -> Path:
    return log_dir() / LOG_FILENAME


def get_logger(name: str = "") -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Attach the rotating file handler once; safe to call repeatedly."""
    global _configured
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    if _configured:
        return logger
    _configured = True
    try:
        handler = RotatingFileHandler(log_file(), maxBytes=5 * 1024 * 1024,
                                      backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
    except Exception:
        # No log file is no reason not to run.
        pass
    return logger


def log_exception(context: str, exc: BaseException) -> None:
    try:
        setup_logging().error("%s: %s", context, exc, exc_info=exc)
    except Exception:
        pass


@contextlib.contextmanager
def run_log(path):
    """Copy everything logged under ``yoru_explain`` into *path* for the block."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger(LOGGER_NAME)
    previous = logger.level
    if logger.level == logging.NOTSET or logger.level > logging.INFO:
        logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        yield logger
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(previous)
