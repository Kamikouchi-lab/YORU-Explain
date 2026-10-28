"""Each family's frame <-> input transform, against that family's own code (no weights needed)."""

from __future__ import annotations

import numpy as np
import pytest

from yoru_explain.core.geometry import LetterboxTransform, letterbox_image

SHAPES = [(1080, 810), (720, 1280), (480, 640), (333, 517), (37, 1000), (601, 601), (200, 320)]


def _image(shape, seed=0):
    return np.random.default_rng(seed).integers(0, 256, size=(*shape, 3), dtype=np.uint8)


def _boxes(t, n=40, seed=1):
    rng = np.random.default_rng(seed)
    h1, w1 = t.input_shape
    xy = rng.uniform(0, [w1, h1], size=(n, 2))
    wh = rng.uniform(1, 200, size=(n, 2))
    return np.concatenate([xy - wh / 2, xy + wh / 2], 1).astype(np.float32)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("imgsz,auto", [((640, 640), True), ((640, 640), False), ((1024, 1024), True)])
def test_ultralytics_letterbox_and_scale_boxes(shape, imgsz, auto):
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils import ops

    im = _image(shape)
    theirs = LetterBox(imgsz, auto=auto, stride=32)(image=im)
    t = LetterboxTransform.ultralytics(shape, imgsz, auto=auto, stride=32)
    assert theirs.shape[:2] == t.input_shape
    assert np.array_equal(letterbox_image(im, t), theirs)
    boxes = _boxes(t)
    scaled = ops.scale_boxes(t.input_shape, torch.from_numpy(boxes.copy()), shape)
    np.testing.assert_allclose(t.boxes_to_frame(boxes), scaled.numpy(), atol=1e-3)


@pytest.mark.parametrize("shape", SHAPES)
def test_rtdetr_stretches_and_scales_normalised_boxes(shape):
    pytest.importorskip("ultralytics")
    from ultralytics.data.augment import LetterBox

    im = _image(shape)
    theirs = LetterBox((640, 640), auto=False, scale_fill=True)(image=im)
    t = LetterboxTransform.ultralytics(shape, (640, 640), auto=False, scale_fill=True)
    assert theirs.shape[:2] == t.input_shape == (640, 640)
    norm = np.array([[0.1, 0.2, 0.5, 0.9], [0.0, 0.0, 1.0, 1.0]])
    h0, w0 = shape
    expected = norm * [w0, h0, w0, h0]                 # RTDETRPredictor.postprocess
    np.testing.assert_allclose(t.boxes_to_frame(norm * 640), expected, atol=1e-6)


@pytest.mark.parametrize("shape", SHAPES)
def test_torchvision_resize_and_resize_boxes(shape):
    torch = pytest.importorskip("torch")
    from torchvision.models.detection.transform import GeneralizedRCNNTransform, resize_boxes

    tr = GeneralizedRCNNTransform(800, 1333, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    il, _ = tr([torch.rand(3, *shape)])
    t = LetterboxTransform.torchvision(shape, il.image_sizes[0], il.tensors.shape[2:])
    assert t.input_shape == tuple(il.tensors.shape[2:]) and t.input_shape[0] % 32 == 0
    boxes = _boxes(t)
    theirs = resize_boxes(torch.from_numpy(boxes.copy()), il.image_sizes[0], shape)
    np.testing.assert_allclose(t.boxes_to_frame(boxes), theirs.numpy(), rtol=1e-6, atol=1e-3)


def test_ssd_is_stretched_to_300():
    torch = pytest.importorskip("torch")
    from torchvision.models.detection.transform import GeneralizedRCNNTransform

    tr = GeneralizedRCNNTransform(300, 300, [0.48, 0.46, 0.41], [1 / 255] * 3, size_divisible=1,
                                  fixed_size=(300, 300))
    il, _ = tr([torch.rand(3, 720, 1280)])
    t = LetterboxTransform.torchvision((720, 1280), il.image_sizes[0], il.tensors.shape[2:])
    assert t.input_shape == t.resized_shape == (300, 300)
    assert t.box_scale == pytest.approx((300 / 1280, 300 / 720))


@pytest.mark.parametrize("shape", SHAPES)
def test_yoru_onnx_letterbox(shape):
    pytest.importorskip("onnxruntime")
    from yoru.libs.plugins.onnx_detector import ONNXDetector

    im = _image(shape)
    theirs, ratio, (pad_w, pad_h) = ONNXDetector._letterbox(im, (640, 640))
    t = LetterboxTransform.yoru_onnx(shape, (640, 640))
    assert np.array_equal(letterbox_image(im, t), theirs)
    assert t.box_scale == (ratio, ratio) and t.box_offset == (pad_w, pad_h)


def test_a_grid_overhanging_the_input_puts_cells_at_their_centres():
    """SSD300: 38 cells of 8 px over a 300 px input (default boxes centred at (j + .5) * 8)."""
    t = LetterboxTransform.torchvision((300, 300), (300, 300), (300, 300))
    f = np.zeros((38, 38), np.float32)
    f[10, 20] = 1.0
    m = t.feature_to_frame(f, 8)
    y, x = np.unravel_index(np.argmax(m), m.shape)
    assert abs(x + 0.5 - 164) <= 1 and abs(y + 0.5 - 84) <= 1
    with pytest.raises(ValueError):
        t.feature_to_frame(np.zeros((30, 30), np.float32), 8)   # does not cover the input


def test_every_kind_round_trips_and_v01_dicts_still_read():
    kinds = [LetterboxTransform.yolov5((333, 517), (416, 640)),
             LetterboxTransform.ultralytics((1080, 810), (640, 640), auto=True),
             LetterboxTransform.ultralytics((1080, 810), (640, 640), auto=False, scale_fill=True),
             LetterboxTransform.torchvision((720, 1280), (750, 1333), (768, 1344)),
             LetterboxTransform.yoru_onnx((720, 1280), (640, 640))]
    for t in kinds:
        assert LetterboxTransform.from_dict(t.to_dict()) == t
    v01 = {k: v for k, v in kinds[0].to_dict().items() if k not in ("kind", "box_scale", "box_offset",
                                                                      "clip_boxes")}
    assert LetterboxTransform.from_dict(v01) == kinds[0]
