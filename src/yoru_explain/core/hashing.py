# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Identities of the files and frames a result was computed from.

Models are hashed in full (SHA-256): a model is tens of megabytes, and
"the same weights" must mean the same bytes.  Videos can be many gigabytes,
so they get a *quick id* -- SHA-256 over the size, the first and the last
8 MiB -- recorded together with the method, and the frames actually used are
hashed one by one (:func:`frame_sha1`), which is what really ties a result to
its pixels.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

_CHUNK = 1 << 20
_QUICK_SPAN = 8 << 20

QUICK_ID_METHOD = "sha256(size, first 8 MiB, last 8 MiB)"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def quick_file_id(path) -> str:
    """A fast identity for a large file; see :data:`QUICK_ID_METHOD`."""
    size = os.path.getsize(path)
    h = hashlib.sha256()
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(_QUICK_SPAN))
        if size > 2 * _QUICK_SPAN:
            f.seek(size - _QUICK_SPAN)
            h.update(f.read(_QUICK_SPAN))
        elif size > _QUICK_SPAN:
            h.update(f.read())
    return "q1:" + h.hexdigest()


def frame_sha1(image: np.ndarray) -> str:
    """SHA-1 of a decoded frame's pixels, shape and dtype included."""
    a = np.ascontiguousarray(image)
    h = hashlib.sha1()
    h.update(f"{a.shape}|{a.dtype.str}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def json_sha256(value: Any) -> str:
    """SHA-256 of a JSON-serialisable value in canonical form (sorted keys)."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_identity(path) -> dict:
    """Path, size and quick id of a file, for manifests."""
    p = Path(path)
    return {"path": str(p), "size": p.stat().st_size, "quick_id": quick_file_id(p),
            "quick_id_method": QUICK_ID_METHOD}
