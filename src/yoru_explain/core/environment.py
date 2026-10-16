# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""What software and hardware a run was computed with, for its manifest."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Dict, Optional


def _version(module_name: str, attr: str = "__version__") -> Optional[str]:
    try:
        module = __import__(module_name, fromlist=["_"])
    except Exception:
        return None
    return str(getattr(module, attr, None) or "unknown")


def _dist_version(distribution: str) -> Optional[str]:
    """An installed distribution's version, without importing it (a GUI toolkit)."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def git_commit(path) -> Optional[str]:
    """The commit a git checkout containing *path* is at, read from ``.git``.

    Read from the files rather than by running git, so it works without git
    installed.  ``None`` when *path* is not in a checkout.
    """
    p = Path(path).resolve()
    for root in (p, *p.parents):
        git = root / ".git"
        if git.is_file():  # worktree: "gitdir: <path>"
            text = git.read_text(encoding="utf-8").strip()
            if text.startswith("gitdir:"):
                git = (root / text.split(":", 1)[1].strip()).resolve()
        if not git.is_dir():
            continue
        try:
            head = (git / "HEAD").read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not head.startswith("ref:"):
            return head
        ref = head.split(":", 1)[1].strip()
        ref_file = git / ref
        if ref_file.is_file():
            return ref_file.read_text(encoding="utf-8").strip()
        packed = git / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.endswith(" " + ref):
                    return line.split(" ", 1)[0]
        common = git / "commondir"
        if common.is_file():
            base = (git / common.read_text(encoding="utf-8").strip()).resolve()
            ref_file = base / ref
            if ref_file.is_file():
                return ref_file.read_text(encoding="utf-8").strip()
        return None
    return None


def software_versions() -> Dict[str, Optional[str]]:
    from yoru_explain import __version__

    versions = {
        "yoru_explain": __version__,
        "python": sys.version.split()[0],
        "numpy": _version("numpy"),
        "opencv": _version("cv2"),
        "torch": _version("torch"),
        "torchvision": _version("torchvision"),
        "yoru": _version("yoru"),
        "dearpygui": _dist_version("dearpygui"),
    }
    try:
        import yoru
        versions["yoru_commit"] = git_commit(Path(yoru.__file__).parent)
    except Exception:
        versions["yoru_commit"] = None
    try:
        import yoru_explain
        versions["yoru_explain_commit"] = git_commit(Path(yoru_explain.__file__).parent)
    except Exception:
        versions["yoru_explain_commit"] = None
    return versions


def hardware() -> Dict[str, object]:
    info: Dict[str, object] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        info["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        info["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        if torch.cuda.is_available():
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as e:  # torch is optional for the core
        info["torch_error"] = f"{type(e).__name__}: {e}"
    return info
