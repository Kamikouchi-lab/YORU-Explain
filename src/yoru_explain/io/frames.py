# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Frames of videos and images, numbered the way YORU numbers them.

YORU's video analysis reads a video with ``cv2.VideoCapture.read()`` from the
first frame and counts from 0; its ``frame`` column is that count.  Reading
in order (``exact=True``) reproduces the count by construction.  Seeking
(``CAP_PROP_POS_FRAMES``) is fast and right for most constant-frame-rate
files, but the container's index can disagree with the decoding order
(variable frame rate, edit lists, broken timestamps), so:

* every frame handed out says how it was reached (``access``), with its
  presentation time and a hash of its pixels;
* :meth:`VideoFrameReader.check_seeking` compares seeks with in-order reads
  on sample frames, and analysis runs read frames in order and compare their
  hash with the one the event was created from (``frame_mismatch``).

Flips are applied after decoding, as YORU applies ``v_flip`` / ``h_flip``
before detection.  Paths may contain non-ASCII characters (OpenCV's
``imread``/``imwrite`` cannot open those on Windows).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from yoru_explain.core.errors import ErrorCode, ExplainError
from yoru_explain.core.hashing import frame_sha1

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".mpg", ".mpeg")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

#: A frame at most this far ahead is reached by reading on, not by seeking.
READ_AHEAD = 64


def kind_of(path) -> str:
    ext = Path(path).suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    raise ExplainError(ErrorCode.MISSING_DATA, f"{Path(path).name}: not a video or image file")


def flip(frame: np.ndarray, v_flip: bool, h_flip: bool) -> np.ndarray:
    import cv2

    if v_flip and h_flip:
        return cv2.flip(frame, -1)
    if v_flip:
        return cv2.flip(frame, 0)
    if h_flip:
        return cv2.flip(frame, 1)
    return frame


def imread(path) -> np.ndarray:
    """``cv2.imread`` that also opens non-ASCII paths."""
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ExplainError(ErrorCode.MISSING_DATA, f"could not read image {path}")
    return image


def imwrite(path, image: np.ndarray) -> None:
    """``cv2.imwrite`` that also writes non-ASCII paths."""
    import cv2

    path = Path(path)
    ok, buf = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ExplainError(ErrorCode.INTERNAL, f"could not encode {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))


def _opencv_path(path: str) -> str:
    """A path ``cv2.VideoCapture`` can open.

    On Windows OpenCV's video backends do not open paths with non-ASCII
    characters; the 8.3 short name of such a file is pure ASCII.
    """
    if path.isascii():
        return path
    import sys

    if sys.platform != "win32":
        return path
    import ctypes
    from ctypes import wintypes

    get_short = ctypes.windll.kernel32.GetShortPathNameW
    get_short.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    get_short.restype = wintypes.DWORD
    buf = ctypes.create_unicode_buffer(32768)
    n = get_short(str(Path(path).resolve()), buf, len(buf))
    return buf.value if 0 < n < len(buf) and buf.value.isascii() else path


@dataclass(frozen=True)
class LoadedFrame:
    image: np.ndarray
    frame_index: int
    timestamp_ms: Optional[float]
    access: str
    sha1: str


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    fourcc: str

    def to_dict(self) -> Dict[str, object]:
        return {"path": self.path, "fps": self.fps, "frame_count": self.frame_count,
                "width": self.width, "height": self.height, "fourcc": self.fourcc}


class VideoFrameReader:
    """Frames of one video by index; see the module docstring for *exact*."""

    def __init__(self, path, *, v_flip: bool = False, h_flip: bool = False):
        import cv2

        self.path = str(path)
        if not Path(self.path).is_file():
            raise ExplainError(ErrorCode.MISSING_DATA, f"video not found: {self.path}")
        self.v_flip, self.h_flip = bool(v_flip), bool(h_flip)
        self._cap = None
        self._pts: Optional[List[float]] = None
        self.decoded_frame_count: Optional[int] = None
        #: The last frame read exactly; reading it again needs no decoding.
        self._last: Optional[LoadedFrame] = None
        self._open()
        cap = self._cap
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        fourcc_text = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00")
        self.info = VideoInfo(
            path=self.path,
            fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            fourcc=fourcc_text,
        )

    def _open(self) -> None:
        import cv2

        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(_opencv_path(self.path))
        if not self._cap.isOpened():
            raise ExplainError(ErrorCode.MISSING_DATA, f"could not open video {self.path}")
        self._next = 0

    def _position_ms(self) -> Optional[float]:
        import cv2

        ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        return float(ms) if ms is not None and math.isfinite(ms) else None

    def _retrieve(self, index: int, access: str) -> LoadedFrame:
        ok, frame = self._cap.retrieve()
        if not ok or frame is None:
            raise ExplainError(ErrorCode.MISSING_DATA, f"{Path(self.path).name}: frame {index} could not be decoded")
        image = flip(frame, self.v_flip, self.h_flip)
        loaded = LoadedFrame(image, index, self._position_ms(), access, frame_sha1(image))
        if access in ("sequential", "seek_verified"):
            self._last = loaded
        return loaded

    def read(self, index: int, *, exact: bool = True) -> LoadedFrame:
        """Frame *index* (0-based).

        ``exact=True`` decodes in order from the start (re-opening the file
        when *index* is behind the current position) -- always the frame YORU
        numbers *index*.  ``exact=False`` seeks when the frame is far ahead
        or behind.
        """
        import cv2

        index = int(index)
        if index < 0:
            raise ExplainError(ErrorCode.MISSING_DATA, f"frame {index} is outside the video")
        # No upper check against CAP_PROP_FRAME_COUNT: containers often state
        # it wrongly.  Running out of frames is reported by grab().
        last = self._last
        if exact and last is not None and last.frame_index == index:
            # Several events on one frame: no decoding again (from the start).
            return LoadedFrame(last.image.copy(), index, last.timestamp_ms, last.access, last.sha1)
        ahead = index - self._next
        if exact and self._pts is not None and not 0 <= ahead <= READ_AHEAD:
            frame = self._read_verified(index)
            if frame is not None:
                return frame
            self._open()  # could not verify: read in order instead
            ahead = index
        if exact or 0 <= ahead <= READ_AHEAD:
            if ahead < 0:
                self._open()
            while self._next < index:
                if not self._cap.grab():
                    raise ExplainError(ErrorCode.MISSING_DATA,
                                       f"{Path(self.path).name} ends before frame {index}")
                self._next += 1
            if not self._cap.grab():
                raise ExplainError(ErrorCode.MISSING_DATA,
                                   f"{Path(self.path).name} ends before frame {index}")
            self._next = index + 1
            return self._retrieve(index, "sequential")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        if not self._cap.grab():
            raise ExplainError(ErrorCode.MISSING_DATA, f"could not seek to frame {index}")
        self._next = index + 1
        return self._retrieve(index, "seek")

    # -- presentation-time index -----------------------------------------

    @property
    def indexed(self) -> bool:
        return self._pts is not None

    def build_index(self, progress=None, cancel=None) -> Dict[str, object]:
        """Record every frame's presentation time in one in-order pass.

        With the index, :meth:`read` reaches a far frame by seeking and then
        *verifies* where the seek landed from the presentation time -- the
        container's own frame position cannot be trusted (measured: on the
        YORU test video a seek to 6301 decodes frame 6300 while OpenCV reports
        position 6302).  Returns a summary; the index is unusable when times
        are missing or not strictly increasing.
        """
        import cv2

        cap = cv2.VideoCapture(_opencv_path(self.path))
        times: List[float] = []
        try:
            while cap.grab():
                ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                times.append(float(ms) if ms is not None else float("nan"))
                if progress and len(times) % 500 == 0:
                    progress(len(times), self.info.frame_count)
                if cancel is not None and len(times) % 500 == 0 and cancel():
                    return {"status": "cancelled", "frames": len(times)}
        finally:
            cap.release()
        steps = [b - a for a, b in zip(times, times[1:])]
        if not times or not all(math.isfinite(t) for t in times) or any(s <= 0 for s in steps):
            self._pts = None
            return {"status": "unusable", "frames": len(times),
                    "reason": "presentation times missing or not strictly increasing"}
        self._pts = times
        self.decoded_frame_count = len(times)
        return {"status": "ok", "frames": len(times),
                "stated_frame_count": self.info.frame_count, "stated_fps": self.info.fps,
                "fps_from_times": 1000.0 / (sorted(steps)[len(steps) // 2]) if steps else None}

    def _frame_of_time(self, ms: Optional[float]) -> Optional[int]:
        import bisect

        times = self._pts
        if ms is None or not math.isfinite(ms) or not times:
            return None
        i = bisect.bisect_left(times, ms)
        best = min((j for j in (i - 1, i) if 0 <= j < len(times)), key=lambda j: abs(times[j] - ms))
        gaps = [times[k] - times[k - 1] for k in (best, best + 1) if 0 < k < len(times)]
        tolerance = min(gaps) / 2 if gaps else 0.5
        return best if abs(times[best] - ms) < tolerance else None

    def _read_verified(self, index: int) -> Optional[LoadedFrame]:
        """Seek near *index*, find out from the time where the seek landed, read on."""
        import cv2

        aim = index
        for _ in range(4):
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, aim)
            if not self._cap.grab():
                return None
            at = self._frame_of_time(self._position_ms())
            if at is None:
                return None
            if at > index:  # landed past it: aim further back
                aim = max(0, aim - 2 * (at - index) - 1)
                continue
            self._next = at + 1
            while self._next <= index:
                if not self._cap.grab():
                    return None
                self._next += 1
            if self._frame_of_time(self._position_ms()) != index:
                return None
            return self._retrieve(index, "seek_verified")
        return None

    def check_seeking(self, samples: Sequence[int] = ()) -> Dict[str, object]:
        """Compare seeked frames with in-order frames at *samples* (default: 5 spread out)."""
        n = self.info.frame_count
        if not samples:
            samples = sorted({int(n * f) for f in (0.1, 0.3, 0.5, 0.7, 0.9)} - {0}) if n > 10 else []
        exact = VideoFrameReader(self.path, v_flip=self.v_flip, h_flip=self.h_flip)
        mismatched: List[int] = []
        try:
            for i in samples:
                self._open()  # seek from a fresh capture, as a user's jump would
                seeked = self.read(i, exact=False)
                if exact.read(i, exact=True).sha1 != seeked.sha1:
                    mismatched.append(i)
        finally:
            exact.close()
            self._open()
        return {"status": "passed" if not mismatched else "failed",
                "samples": list(samples), "mismatched": mismatched}

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ImageReader:
    """A single image file, as a one-frame source."""

    def __init__(self, path, *, v_flip: bool = False, h_flip: bool = False):
        self.path = str(path)
        if not Path(self.path).is_file():
            raise ExplainError(ErrorCode.MISSING_DATA, f"image not found: {self.path}")
        self.v_flip, self.h_flip = bool(v_flip), bool(h_flip)
        image = imread(self.path)
        self._image = flip(image, self.v_flip, self.h_flip)
        h, w = self._image.shape[:2]
        self.info = VideoInfo(self.path, 0.0, 1, w, h, "image")

    def read(self, index: int = 0, *, exact: bool = True) -> LoadedFrame:
        if int(index) != 0:
            raise ExplainError(ErrorCode.MISSING_DATA, "an image has only frame 0")
        return LoadedFrame(self._image.copy(), 0, None, "image", frame_sha1(self._image))

    def check_seeking(self, samples: Sequence[int] = ()) -> Dict[str, object]:
        return {"status": "not_applicable"}

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def open_source(path, *, v_flip: bool = False, h_flip: bool = False):
    """A :class:`VideoFrameReader` or :class:`ImageReader` for *path*."""
    if kind_of(path) == "video":
        return VideoFrameReader(path, v_flip=v_flip, h_flip=h_flip)
    return ImageReader(path, v_flip=v_flip, h_flip=h_flip)
