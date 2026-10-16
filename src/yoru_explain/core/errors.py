# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Error codes every result row can carry, and the exception that carries one.

A failure is always reported with one of these codes -- in ``results.csv``,
in ``perturbations.csv`` and in the run log -- and never turned into a score.
In particular a target that cannot be scored is *not* given 0: the value
stays empty and the code says why.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Optional


class ErrorCode(str, Enum):
    #: The model, or the requested method/score, is not supported by the adapter.
    MODEL_UNSUPPORTED = "model_unsupported"
    MODEL_LOAD_FAILED = "model_load_failed"
    #: Gradients could not be obtained (or were not verified) for this model.
    GRADIENTS_UNAVAILABLE = "gradients_unavailable"
    #: The requested score cannot be read for this target (e.g. no candidate).
    SCORE_UNAVAILABLE = "score_unavailable"
    #: A score recomputed on the same input differs from the reference.
    SCORE_MISMATCH = "score_mismatch"
    #: The stored target was not found again on its frame.
    TARGET_NOT_REPRODUCED = "target_not_reproduced"
    #: The decoded frame is not the frame the event was made from.
    FRAME_MISMATCH = "frame_mismatch"
    #: The adapter's preprocessing does not reproduce YORU's inference.
    PREPROCESS_MISMATCH = "preprocess_mismatch"
    #: The adapter's detections differ from YORU's own detector.
    IDENTITY_CHECK_FAILED = "identity_check_failed"
    #: NaN or infinity where a number was expected.
    NON_FINITE = "non_finite"
    CUDA_OOM = "cuda_oom"
    #: Something the project needs (video, model, event field) is missing.
    MISSING_DATA = "missing_data"
    INVALID_CONFIG = "invalid_config"
    #: The result was computed but could not be written (disk full, path too long).
    OUTPUT_FAILED = "output_failed"
    CANCELLED = "cancelled"
    INTERNAL = "internal_error"


class ExplainError(Exception):
    """An error with an :class:`ErrorCode`, raised by the core and the adapters."""

    def __init__(self, code: ErrorCode, message: str,
                 details: Optional[Mapping[str, Any]] = None):
        super().__init__(message)
        self.code = ErrorCode(code)
        self.message = message
        self.details = dict(details or {})

    def __str__(self) -> str:
        return f"[{self.code.value}] {self.message}"


def error_code_of(exc: BaseException) -> ErrorCode:
    """The code an arbitrary exception is reported under."""
    if isinstance(exc, ExplainError):
        return exc.code
    name = type(exc).__name__
    if name == "OutOfMemoryError" or "out of memory" in str(exc).lower():
        return ErrorCode.CUDA_OOM
    return ErrorCode.INTERNAL
