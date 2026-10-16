"""Coordinates: the letterbox both ways, against the bundled YOLOv5 code."""

from __future__ import annotations

import numpy as np
import pytest

from yoru_explain.core.geometry import (
    LetterboxTransform,
    autoshape_input_shape,
    clip_box,
    iou,
    letterbox_image,
    scaled_box,
)

SHAPES = [(720, 1280), (480, 640), (200, 320), (333, 517), (1080, 1920), (640, 480), (37, 1000),
          (601, 601)]


def _bundled():
    try:
        from yoru_explain.adapters import _yoru

        return _yoru.load()
    except Exception as e:
        pytest.skip(f"YORU's bundled YOLOv5 is not importable: {e}")


def test_iou_and_boxes():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0
    assert iou((0, 0, 32, 32), (16, 0, 48, 32)) == pytest.approx(1 / 3)
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0
    assert clip_box((-5, -5, 50, 50), 40, 30) == (0, 0, 40, 30)
    assert scaled_box((10, 10, 20, 30), 2.0, 100, 100) == (5, 0, 25, 40)


@pytest.mark.parametrize("shape", SHAPES)
def test_input_shape_and_letterbox_match_yolov5(shape):
    y = _bundled()
    h0, w0 = shape
    g = 640 / max(shape)
    expected = [y.make_divisible(x, 32) for x in np.array([[int(v * g) for v in shape]]).max(0)]
    assert list(autoshape_input_shape(shape)) == expected

    rng = np.random.default_rng(1)
    im = rng.integers(0, 256, size=(h0, w0, 3), dtype=np.uint8)
    theirs, ratio, (dw, dh) = y.letterbox(im, expected, auto=False)
    t = LetterboxTransform.yolov5(shape, expected)
    mine = letterbox_image(im, t)
    assert mine.shape == theirs.shape
    assert np.array_equal(mine, theirs)
    assert t.ratio == pytest.approx(ratio[0])
    # The pads letterbox really added: where the original pixels start.
    assert t.pad_left == int(round(dw - 0.1)) and t.pad_top == int(round(dh - 0.1))


@pytest.mark.parametrize("shape", SHAPES)
def test_boxes_go_back_as_scale_boxes_maps_them(shape):
    import torch

    y = _bundled()
    t = LetterboxTransform.yolov5(shape, autoshape_input_shape(shape))
    rng = np.random.default_rng(2)
    h1, w1 = t.input_shape
    xy = rng.uniform(0, [w1, h1], size=(50, 2))
    wh = rng.uniform(1, 200, size=(50, 2))
    boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1).astype(np.float32)
    theirs = torch.from_numpy(boxes.copy())
    y.scale_boxes(list(t.input_shape), theirs, shape)
    np.testing.assert_allclose(t.boxes_to_frame(boxes), theirs.numpy(), atol=1e-3)


@pytest.mark.parametrize("shape", [(200, 320), (480, 640), (333, 517)])
def test_a_map_on_input_pixels_comes_back_where_it_was(shape):
    """letterbox an image, map the result back: the content lines up again."""
    h0, w0 = shape
    t = LetterboxTransform.yolov5(shape, autoshape_input_shape(shape))
    frame = np.zeros((h0, w0), np.float32)
    frame[h0 // 3:h0 // 3 + 24, w0 // 4:w0 // 4 + 24] = 1.0
    forward = letterbox_image(np.repeat((frame * 255).astype(np.uint8)[..., None], 3, 2), t)[..., 0]
    back = t.map_to_frame(forward.astype(np.float32) / 255)
    assert back.shape == (h0, w0)
    inside = back[h0 // 3 + 4:h0 // 3 + 20, w0 // 4 + 4:w0 // 4 + 20]
    assert inside.min() > 0.9
    far = back[h0 // 3 + 40:, :]
    assert far.max() < 0.5  # 114-grey padding never comes back into the frame


@pytest.mark.parametrize("shape,size,pads", [
    ((200, 320), 320, (12, 12, 0)),   # padded, not resized
    ((200, 320), 640, (8, 8, 0)),     # resized x2, then padded
    ((333, 517), 640, (2, 2, 0)),     # resized by a non-integer ratio (1.238)
])
def test_a_feature_cell_lands_on_its_centre(shape, size, pads):
    t = LetterboxTransform.yolov5(shape, autoshape_input_shape(shape, size))
    assert (t.pad_top, t.pad_bottom, t.pad_left) == pads
    stride = 16
    f = np.zeros((t.input_shape[0] // stride, t.input_shape[1] // stride), np.float32)
    f[5, 7] = 1.0
    m = t.feature_to_frame(f, stride)
    y, x = np.unravel_index(np.argmax(m), m.shape)
    cx, cy = t.points_to_frame(np.array([[(7 + 0.5) * stride, (5 + 0.5) * stride]]))[0]
    tol = 1.0 / t.ratio + 0.5
    assert abs(x + 0.5 - cx) <= tol and abs(y + 0.5 - cy) <= tol


def test_points_round_trip():
    t = LetterboxTransform.yolov5((333, 517), autoshape_input_shape((333, 517)))
    p = np.array([[0.0, 0.0], [517.0, 333.0], [100.5, 200.25]])
    np.testing.assert_allclose(t.points_to_frame(t.points_to_input(p)), p, atol=1e-9)


def test_transform_dict_round_trip():
    t = LetterboxTransform.yolov5((333, 517), autoshape_input_shape((333, 517)))
    assert LetterboxTransform.from_dict(t.to_dict()) == t
