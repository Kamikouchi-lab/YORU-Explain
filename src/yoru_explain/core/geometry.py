# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Boxes, IoU, and the letterbox between a frame and the model input.

:class:`LetterboxTransform` records how YOLOv5's ``AutoShape`` letterboxes a
frame (``utils.augmentations.letterbox`` with ``auto=False``) and maps both
ways.  Two conventions coexist in YOLOv5 and both are kept, on purpose:

* the image is resized to ``round(w0*r) x round(h0*r)`` and padded by whole
  pixels, ``round(d/2 - 0.1)`` before and ``round(d/2 + 0.1)`` after;
* boxes are mapped back by ``scale_boxes``, which uses the *fractional* pad
  ``(w1 - w0*gain) / 2``.

So a box goes back with :meth:`boxes_to_frame` (what YORU reports), while a
map -- Grad-CAM, anything defined on input pixels -- goes back with
:meth:`map_to_frame`, which undoes the real resize and padding.  The two can
differ by up to half a letterbox pixel; ``tests/unit/test_geometry.py`` pins
both against the bundled YOLOv5 code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

Box = Tuple[float, float, float, float]

#: YOLOv5's letterbox fill value (grey).
LETTERBOX_FILL = 114


def box_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two ``xyxy`` boxes; 0 when either is empty."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def clip_box(box: Sequence[float], width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    return (min(max(x1, 0.0), width), min(max(y1, 0.0), height),
            min(max(x2, 0.0), width), min(max(y2, 0.0), height))


def scaled_box(box: Sequence[float], scale: float, width: int, height: int) -> Box:
    """*box* scaled by *scale* about its centre, clipped to the frame."""
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    hw, hh = (box[2] - box[0]) * scale / 2.0, (box[3] - box[1]) * scale / 2.0
    return clip_box((cx - hw, cy - hh, cx + hw, cy + hh), width, height)


def make_divisible(x: float, divisor: int) -> int:
    """The smallest multiple of *divisor* that is >= *x* (YOLOv5's rule)."""
    return int(math.ceil(x / divisor) * divisor)


def autoshape_input_shape(frame_shape: Sequence[int], size: int = 640,
                          stride: int = 32) -> Tuple[int, int]:
    """The ``(h, w)`` YOLOv5's AutoShape feeds the network for one frame.

    ``g = size / max(h0, w0)``, each side ``int(side * g)`` rounded up to a
    multiple of the stride.
    """
    h0, w0 = int(frame_shape[0]), int(frame_shape[1])
    g = size / max(h0, w0)
    return (make_divisible(int(h0 * g), stride), make_divisible(int(w0 * g), stride))


@dataclass(frozen=True)
class LetterboxTransform:
    """How one frame was letterboxed into the model input, and back."""

    #: ``(h0, w0)`` of the frame.
    frame_shape: Tuple[int, int]
    #: ``(h1, w1)`` of the model input.
    input_shape: Tuple[int, int]
    #: Resize ratio ``r = min(h1/h0, w1/w0)``.
    ratio: float
    #: ``(h, w)`` the frame was resized to: ``round(h0*r), round(w0*r)``.
    resized_shape: Tuple[int, int]
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int

    @classmethod
    def yolov5(cls, frame_shape: Sequence[int], input_shape: Sequence[int]) -> "LetterboxTransform":
        """The transform ``letterbox(im, input_shape, auto=False)`` performs."""
        h0, w0 = int(frame_shape[0]), int(frame_shape[1])
        h1, w1 = int(input_shape[0]), int(input_shape[1])
        r = min(h1 / h0, w1 / w0)
        new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
        dw, dh = (w1 - new_w) / 2, (h1 - new_h) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        return cls((h0, w0), (h1, w1), r, (new_h, new_w), top, bottom, left, right)

    @property
    def box_gain(self) -> float:
        """``scale_boxes``' gain: ``min(h1/h0, w1/w0)``."""
        return min(self.input_shape[0] / self.frame_shape[0],
                   self.input_shape[1] / self.frame_shape[1])

    @property
    def box_pad(self) -> Tuple[float, float]:
        """``scale_boxes``' fractional ``(pad_x, pad_y)``."""
        g = self.box_gain
        return ((self.input_shape[1] - self.frame_shape[1] * g) / 2,
                (self.input_shape[0] - self.frame_shape[0] * g) / 2)

    def boxes_to_frame(self, boxes: np.ndarray) -> np.ndarray:
        """Input-pixel ``xyxy`` boxes as YOLOv5's ``scale_boxes`` maps them, clipped."""
        b = np.array(boxes, dtype=np.float64, copy=True).reshape(-1, 4)
        px, py = self.box_pad
        g = self.box_gain
        b[:, [0, 2]] -= px
        b[:, [1, 3]] -= py
        b /= g
        h0, w0 = self.frame_shape
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, w0)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, h0)
        return b

    def points_to_input(self, xy: np.ndarray) -> np.ndarray:
        """Frame points (continuous pixel coordinates) on the resized, padded input."""
        p = np.array(xy, dtype=np.float64, copy=True).reshape(-1, 2)
        sx = self.resized_shape[1] / self.frame_shape[1]
        sy = self.resized_shape[0] / self.frame_shape[0]
        p[:, 0] = p[:, 0] * sx + self.pad_left
        p[:, 1] = p[:, 1] * sy + self.pad_top
        return p

    def points_to_frame(self, xy: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`points_to_input`."""
        p = np.array(xy, dtype=np.float64, copy=True).reshape(-1, 2)
        sx = self.resized_shape[1] / self.frame_shape[1]
        sy = self.resized_shape[0] / self.frame_shape[0]
        p[:, 0] = (p[:, 0] - self.pad_left) / sx
        p[:, 1] = (p[:, 1] - self.pad_top) / sy
        return p

    def map_to_frame(self, input_map: np.ndarray) -> np.ndarray:
        """A map on input pixels ``(h1, w1)`` as a map on frame pixels ``(h0, w0)``.

        The padding is cut off and the resize undone (bilinear, half-pixel
        centres -- the same sampling grid the forward resize used).
        """
        import cv2

        m = np.asarray(input_map, dtype=np.float32)
        if m.shape[:2] != tuple(self.input_shape):
            raise ValueError(f"map shape {m.shape[:2]} is not the input shape {self.input_shape}")
        rh, rw = self.resized_shape
        crop = m[self.pad_top:self.pad_top + rh, self.pad_left:self.pad_left + rw]
        h0, w0 = self.frame_shape
        return cv2.resize(crop, (w0, h0), interpolation=cv2.INTER_LINEAR)

    def feature_to_frame(self, feature_map: np.ndarray, stride: int) -> np.ndarray:
        """A map on a stride-*stride* feature grid as a map on frame pixels.

        Cell ``(i, j)`` is centred on input pixel ``((j + 0.5) * stride,
        (i + 0.5) * stride)`` -- YOLOv5's grid convention -- which is what a
        bilinear resize with half-pixel centres assumes.
        """
        import cv2

        f = np.asarray(feature_map, dtype=np.float32)
        h1, w1 = self.input_shape
        if f.shape[0] * stride != h1 or f.shape[1] * stride != w1:
            raise ValueError(
                f"feature map {f.shape[:2]} at stride {stride} does not tile the input {self.input_shape}")
        up = cv2.resize(f, (w1, h1), interpolation=cv2.INTER_LINEAR)
        return self.map_to_frame(up)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "yolov5_letterbox",
            "frame_shape": list(self.frame_shape),
            "input_shape": list(self.input_shape),
            "ratio": self.ratio,
            "resized_shape": list(self.resized_shape),
            "pad_top": self.pad_top,
            "pad_bottom": self.pad_bottom,
            "pad_left": self.pad_left,
            "pad_right": self.pad_right,
            "box_gain": self.box_gain,
            "box_pad": list(self.box_pad),
            "fill": LETTERBOX_FILL,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LetterboxTransform":
        return cls(
            tuple(int(v) for v in d["frame_shape"]),
            tuple(int(v) for v in d["input_shape"]),
            float(d["ratio"]),
            tuple(int(v) for v in d["resized_shape"]),
            int(d["pad_top"]), int(d["pad_bottom"]), int(d["pad_left"]), int(d["pad_right"]),
        )


def letterbox_image(image: np.ndarray, transform: LetterboxTransform) -> np.ndarray:
    """Apply *transform* to *image* the way YOLOv5's ``letterbox`` does.

    Used by the synthetic test adapter; the YOLOv5 adapter calls the bundled
    function itself.
    """
    import cv2

    h0, w0 = transform.frame_shape
    if image.shape[:2] != (h0, w0):
        raise ValueError(f"image {image.shape[:2]} is not the transform's frame {transform.frame_shape}")
    rh, rw = transform.resized_shape
    im = image
    if (w0, h0) != (rw, rh):
        im = cv2.resize(im, (rw, rh), interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(im, transform.pad_top, transform.pad_bottom,
                              transform.pad_left, transform.pad_right,
                              cv2.BORDER_CONSTANT, value=(LETTERBOX_FILL,) * 3)
