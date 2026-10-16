from __future__ import annotations

import os
from pathlib import Path

import pytest

# Logs of test runs (ours and YORU's) go to a throwaway YORU_HOME, not the
# user's ~/.yoru/logs.
import tempfile  # noqa: E402

os.environ["YORU_HOME"] = tempfile.mkdtemp(prefix="yoru_explain_tests_")

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / ".cache" / "models"


def _first_file(*candidates):
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    return None


def yoru_root():
    try:
        import yoru
    except Exception:
        return None
    return Path(yoru.__file__).resolve().parents[1]


def yolov5s_weights():
    """$YORU_EXPLAIN_YOLOV5_WEIGHTS, else .cache/models/yolov5s.pt."""
    return _first_file(os.environ.get("YORU_EXPLAIN_YOLOV5_WEIGHTS"), MODELS / "yolov5s.pt")


def fly_weights():
    """$YORU_EXPLAIN_FLY_WEIGHTS, else the test-fixture fly model (docs/INITIAL_AUDIT.md)."""
    return _first_file(os.environ.get("YORU_EXPLAIN_FLY_WEIGHTS"),
                       MODELS / "fly_copulation_yolov5s.pt")


def fly_video():
    root = yoru_root()
    return _first_file(os.environ.get("YORU_EXPLAIN_TEST_VIDEO"),
                       root / "test_data" / "test_video_fly_copulation.mp4" if root else None)


@pytest.fixture
def toy():
    from yoru_explain.adapters.toy import ToyGridAdapter

    adapter = ToyGridAdapter()
    adapter.load_model(None, device="cpu")
    return adapter
