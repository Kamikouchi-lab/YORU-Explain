"""Frame numbering, seeking, hashing, flips and non-ASCII paths."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from yoru_explain.core.errors import ExplainError
from yoru_explain.io.frames import (
    ImageReader,
    VideoFrameReader,
    imread,
    imwrite,
    kind_of,
    open_source,
)


def _write_video(path, n=150, size=(96, 64)):
    w, h = size
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (w, h))
    assert out.isOpened()
    for i in range(n):
        f = np.zeros((h, w, 3), np.uint8)
        f[:, : w // 2] = (i * 7) % 256        # frame number, readable after JPEG
        f[: h // 2, w // 2:] = (i * 13) % 256
        f[h - 8:, :8] = 255                   # an asymmetric mark, to see flips
        out.write(f)
    out.release()
    return path


@pytest.fixture
def video(tmp_path):
    return _write_video(tmp_path / "clip.avi")


def test_in_order_reading_numbers_frames_from_zero(video):
    with VideoFrameReader(video) as r:
        assert r.info.frame_count == 150 and (r.info.width, r.info.height) == (96, 64)
        f0, f5 = r.read(0), r.read(5)
        assert f0.frame_index == 0 and f0.access == "sequential"
        assert abs(int(f5.image[10, 10, 0]) - 35) <= 3
        # Going back re-reads from the start: still exact.
        assert r.read(0).sha1 == f0.sha1


def test_seeks_are_verified_against_presentation_times(video):
    with VideoFrameReader(video) as seq, VideoFrameReader(video) as idx:
        summary = idx.build_index()
        assert summary["status"] == "ok" and summary["frames"] == 150
        for i in (140, 3, 100, 77, 149):
            a = idx.read(i, exact=True)
            assert a.access in ("seek_verified", "sequential")
            assert a.sha1 == seq.read(i, exact=True).sha1
        assert idx.check_seeking()["status"] in ("passed", "failed")


def test_unverified_seeks_say_so(video):
    with VideoFrameReader(video) as r:
        assert r.read(120, exact=False).access == "seek"


def test_reading_past_the_end_is_an_error(video):
    with VideoFrameReader(video) as r:
        with pytest.raises(ExplainError):
            r.read(10_000)


def test_flips_are_applied_after_decoding(video):
    with VideoFrameReader(video) as plain, VideoFrameReader(video, v_flip=True, h_flip=True) as both:
        a, b = plain.read(3).image, both.read(3).image
        np.testing.assert_array_equal(b, a[::-1, ::-1])


def test_non_ascii_paths(tmp_path):
    d = tmp_path / "解析データ"
    img = np.zeros((20, 30, 3), np.uint8)
    img[5:10, 5:10] = (10, 200, 30)
    imwrite(d / "フレーム.png", img)
    np.testing.assert_array_equal(imread(d / "フレーム.png"), img)
    with ImageReader(d / "フレーム.png") as r:
        f = r.read(0)
        assert f.access == "image" and f.frame_index == 0
    _write_video(d / "動画.avi", n=10)
    with VideoFrameReader(d / "動画.avi") as r:
        assert r.read(9).frame_index == 9


def test_kinds():
    assert kind_of("a.MP4") == "video" and kind_of("b.png") == "image"
    with pytest.raises(ExplainError):
        kind_of("c.txt")
    with pytest.raises(ExplainError):
        open_source("missing.mp4")


def test_reading_the_same_frame_again_does_not_decode_again(video):
    with VideoFrameReader(video) as r:
        a = r.read(120)
        opened = []
        r._open = lambda: opened.append(1)
        b = r.read(120)
        assert b.sha1 == a.sha1 and b.access == a.access and not opened
        b.image[...] = 0                         # a copy: the kept frame is not changed
        assert r.read(120).sha1 == a.sha1
    with VideoFrameReader(video) as r:
        r.read(120)
        assert r.read(120, exact=False).access in ("seek", "sequential")  # never the kept exact frame
