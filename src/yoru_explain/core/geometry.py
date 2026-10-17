# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Boxes, rotated boxes, IoU, and the transform between a frame and the model input.

:class:`LetterboxTransform` records how a detector turns a frame into its
input -- resize, then pad -- and maps both ways.  Each model family does
this its own way, and each has a constructor that reproduces it:

``yolov5``
    YOLOv5's ``AutoShape`` (``utils.augmentations.letterbox`` with
    ``auto=False``): the image is resized to ``round(w0*r) x round(h0*r)`` and
    padded by whole pixels, ``round(d/2 - 0.1)`` before and ``round(d/2 +
    0.1)`` after, while boxes are mapped back by ``scale_boxes`` with the
    *fractional* pad ``(w1 - w0*gain) / 2``.
``ultralytics``
    ultralytics' ``LetterBox`` (YOLOv8/YOLO11/...: the minimum rectangle,
    ``auto=True``; RT-DETR: stretched to a square, ``scale_fill=True``), with
    its own ``scale_boxes``, whose pad is *rounded*.
``torchvision``
    ``GeneralizedRCNNTransform``: a bilinear resize by one factor (Faster /
    Mask R-CNN) or to a fixed size (SSD), padded at the bottom and right;
    boxes go back by the per-axis size ratios (``resize_boxes``).
``yoru_onnx``
    YORU's ONNX plugin: the same letterbox as YOLOv5, but boxes go back with
    the pad of the *rounded* resized size.

So a box goes back with :meth:`boxes_to_frame` (what YORU reports), while a
map -- Grad-CAM, anything defined on input pixels -- goes back with
:meth:`map_to_frame`, which undoes the real resize and padding.  The two can
differ by up to half an input pixel; ``tests/unit/test_geometry.py`` pins
both against each family's own code.

Rotated boxes (OBB) are ``(cx, cy, w, h, angle)`` in frame pixels, ``angle``
in radians from the image's +x axis towards +y -- YORU's convention
(``yoru.libs.obb``), which is ultralytics' ``xywhr``.  Their overlap is the
exact IoU of the two rectangles as polygons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

Box = Tuple[float, float, float, float]
Obb = Tuple[float, float, float, float, float]

#: YOLOv5's (and ultralytics') letterbox fill value (grey).
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


# -- rotated boxes ---------------------------------------------------------------


def obb_corners(obb: Sequence[float]) -> np.ndarray:
    """The four corners of ``(cx, cy, w, h, angle)`` as a ``(4, 2)`` array.

    In YORU's canonical order -- ``(-w/2,-h/2), (+w/2,-h/2), (+w/2,+h/2),
    (-w/2,+h/2)`` rotated by ``angle`` -- the order of
    ``yoru.libs.obb.obb_corners`` (pinned by a test).
    """
    cx, cy, w, h, a = (float(v) for v in obb)
    hw, hh = w / 2.0, h / 2.0
    c, s = math.cos(a), math.sin(a)
    local = ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
    return np.array([(cx + lx * c - ly * s, cy + lx * s + ly * c) for lx, ly in local], np.float64)


def obb_aabb(obb: Sequence[float]) -> Box:
    """The upright box enclosing a rotated one."""
    p = obb_corners(obb)
    return (float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max()))


def point_in_obb(x: float, y: float, obb: Sequence[float]) -> bool:
    cx, cy, w, h, a = (float(v) for v in obb)
    c, s = math.cos(a), math.sin(a)
    dx, dy = x - cx, y - cy
    return abs(dx * c + dy * s) <= w / 2.0 and abs(-dx * s + dy * c) <= h / 2.0


def polygon_area(points: np.ndarray) -> float:
    """Signed area (shoelace); positive for counter-clockwise points in x-right, y-up terms."""
    p = np.asarray(points, np.float64)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


def _clip_convex(subject: List[Tuple[float, float]],
                 clip: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Sutherland-Hodgman: *subject* clipped by the convex, positively oriented *clip*."""
    out = subject
    for i in range(len(clip)):
        if not out:
            break
        (ax, ay), (bx, by) = clip[i], clip[(i + 1) % len(clip)]

        def side(p):
            return (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax)

        def cut(p, q):
            sp, sq = side(p), side(q)
            t = sp / (sp - sq)
            return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))

        src, out = out, []
        for j in range(len(src)):
            p, q = src[j], src[(j + 1) % len(src)]
            p_in, q_in = side(p) >= 0, side(q) >= 0
            if p_in:
                out.append(p)
                if not q_in:
                    out.append(cut(p, q))
            elif q_in:
                out.append(cut(p, q))
    return out


def _positive(points: np.ndarray) -> List[Tuple[float, float]]:
    pts = [(float(x), float(y)) for x, y in points]
    return pts if polygon_area(points) >= 0 else pts[::-1]


def obb_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Exact IoU of two rotated rectangles, as polygons; 0 when either is empty."""
    area_a = abs(float(a[2]) * float(a[3]))
    area_b = abs(float(b[2]) * float(b[3]))
    if area_a <= 0 or area_b <= 0:
        return 0.0
    pa, pb = obb_corners(a), obb_corners(b)
    # Disjoint enclosing boxes: no clipping needed.
    if (pa[:, 0].max() <= pb[:, 0].min() or pb[:, 0].max() <= pa[:, 0].min()
            or pa[:, 1].max() <= pb[:, 1].min() or pb[:, 1].max() <= pa[:, 1].min()):
        return 0.0
    inter = _clip_convex(_positive(pa), _positive(pb))
    inter_area = abs(polygon_area(np.array(inter))) if len(inter) >= 3 else 0.0
    union = area_a + area_b - inter_area
    return float(min(1.0, max(0.0, inter_area / union))) if union > 0 else 0.0


def overlap(box_a: Sequence[float], box_b: Sequence[float],
            obb_a: Optional[Sequence[float]] = None, obb_b: Optional[Sequence[float]] = None) -> float:
    """IoU of two detections: of their rotated boxes when both have one, else of the upright boxes."""
    if obb_a is not None and obb_b is not None:
        return obb_iou(obb_a, obb_b)
    return iou(box_a, box_b)


def region_mask(shape: Sequence[int], box: Sequence[float],
                obb: Optional[Sequence[float]] = None) -> np.ndarray:
    """Pixels inside *box* (or inside *obb* when given), by their centres."""
    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros((h, w), bool)
    if obb is None:
        x1, y1, x2, y2 = box
        xa, ya = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
        xb, yb = min(w, int(math.ceil(x2))), min(h, int(math.ceil(y2)))
        if xb > xa and yb > ya:
            mask[ya:yb, xa:xb] = True
        return mask
    x1, y1, x2, y2 = obb_aabb(obb)
    xa, ya = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
    xb, yb = min(w, int(math.ceil(x2)) + 1), min(h, int(math.ceil(y2)) + 1)
    if xb <= xa or yb <= ya:
        return mask
    cx, cy, bw, bh, a = (float(v) for v in obb)
    yy, xx = np.mgrid[ya:yb, xa:xb]
    dx, dy = xx + 0.5 - cx, yy + 0.5 - cy
    c, s = math.cos(a), math.sin(a)
    inside = (np.abs(dx * c + dy * s) <= bw / 2.0) & (np.abs(-dx * s + dy * c) <= bh / 2.0)
    mask[ya:yb, xa:xb] = inside
    return mask


# -- frame <-> model input ---------------------------------------------------------


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


def _upsample_grid(f: np.ndarray, stride: float, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear upsampling of a grid whose cell ``(i, j)`` is centred on ``((j+.5)s, (i+.5)s)``.

    For grids that do not tile the output exactly (SSD's 38 x 8 px over a
    300 px input); edges are clamped, as ``cv2.resize`` clamps them.
    """
    def axis(n_out, n_in):
        u = (np.arange(n_out, dtype=np.float64) + 0.5) / stride - 0.5
        u = np.clip(u, 0.0, n_in - 1)
        i0 = np.floor(u).astype(int)
        i1 = np.minimum(i0 + 1, n_in - 1)
        return i0, i1, (u - i0)

    y0, y1, wy = axis(out_h, f.shape[0])
    x0, x1, wx = axis(out_w, f.shape[1])
    top = f[y0][:, x0] * (1 - wx) + f[y0][:, x1] * wx
    bottom = f[y1][:, x0] * (1 - wx) + f[y1][:, x1] * wx
    return (top * (1 - wy)[:, None] + bottom * wy[:, None]).astype(np.float32)


@dataclass(frozen=True)
class LetterboxTransform:
    """How one frame was resized and padded into the model input, and back."""

    #: ``(h0, w0)`` of the frame.
    frame_shape: Tuple[int, int]
    #: ``(h1, w1)`` of the model input (padding included).
    input_shape: Tuple[int, int]
    #: The nominal resize ratio (YOLO: ``r = min(h1/h0, w1/w0)``; torchvision: the scale factor).
    ratio: float
    #: ``(h, w)`` the frame was resized to before padding.
    resized_shape: Tuple[int, int]
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int
    #: Which detector's convention: ``yolov5_letterbox``, ``ultralytics_letterbox``,
    #: ``ultralytics_stretch``, ``torchvision_resize``, ``yoru_onnx_letterbox``.
    kind: str = "yolov5_letterbox"
    #: Boxes go back as ``frame = (input - box_offset) / box_scale``, per axis ``(x, y)``.
    #: Left empty, the YOLOv5 ``scale_boxes`` rule fills them in.
    box_scale: Optional[Tuple[float, float]] = None
    box_offset: Optional[Tuple[float, float]] = None
    #: Whether boxes are clipped to the frame on the way back.
    clip_boxes: bool = True
    #: What the padding holds (a pixel value, a description, or None).
    fill: Any = field(default=LETTERBOX_FILL, compare=False)

    def __post_init__(self):
        if self.box_scale is None or self.box_offset is None:
            h0, w0 = self.frame_shape
            h1, w1 = self.input_shape
            g = min(h1 / h0, w1 / w0)
            object.__setattr__(self, "box_scale", (g, g))
            object.__setattr__(self, "box_offset", ((w1 - w0 * g) / 2, (h1 - h0 * g) / 2))
        object.__setattr__(self, "box_scale", tuple(float(v) for v in self.box_scale))
        object.__setattr__(self, "box_offset", tuple(float(v) for v in self.box_offset))

    # -- the families ----------------------------------------------------------

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

    @classmethod
    def ultralytics(cls, frame_shape: Sequence[int], imgsz: Sequence[int], *, auto: bool,
                    stride: int = 32, scale_fill: bool = False) -> "LetterboxTransform":
        """ultralytics' ``LetterBox(imgsz, auto, scale_fill, stride=...)`` and its ``scale_boxes``.

        ``scale_fill`` (RT-DETR) stretches the frame to *imgsz*; RT-DETR's
        boxes come out normalised and go back by the frame size, which is
        the same as ``frame = input / (w1/w0, h1/h0)``, unclipped.
        """
        h0, w0 = int(frame_shape[0]), int(frame_shape[1])
        nh, nw = (int(imgsz), int(imgsz)) if isinstance(imgsz, (int, float)) else \
            (int(imgsz[0]), int(imgsz[1]))
        if scale_fill:
            return cls((h0, w0), (nh, nw), min(nh / h0, nw / w0), (nh, nw), 0, 0, 0, 0,
                       kind="ultralytics_stretch", box_scale=(nw / w0, nh / h0), box_offset=(0.0, 0.0),
                       clip_boxes=False, fill=None)
        r = min(nh / h0, nw / w0)
        new_w, new_h = round(w0 * r), round(h0 * r)
        dw, dh = nw - new_w, nh - new_h
        if auto:
            dw, dh = dw % stride, dh % stride
        dw, dh = dw / 2, dh / 2
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        h1, w1 = new_h + top + bottom, new_w + left + right
        gain = min(h1 / h0, w1 / w0)
        pad = (round((w1 - w0 * gain) / 2 - 0.1), round((h1 - h0 * gain) / 2 - 0.1))
        return cls((h0, w0), (h1, w1), r, (new_h, new_w), top, bottom, left, right,
                   kind="ultralytics_letterbox", box_scale=(gain, gain), box_offset=pad)

    @classmethod
    def torchvision(cls, frame_shape: Sequence[int], resized_shape: Sequence[int],
                    input_shape: Sequence[int]) -> "LetterboxTransform":
        """``GeneralizedRCNNTransform``: resized to *resized_shape*, padded bottom/right to *input_shape*.

        Boxes go back by ``resize_boxes``: each axis times original / resized.
        The padding is 0 after normalisation, i.e. the normalisation mean.
        """
        h0, w0 = int(frame_shape[0]), int(frame_shape[1])
        rh, rw = int(resized_shape[0]), int(resized_shape[1])
        h1, w1 = int(input_shape[0]), int(input_shape[1])
        return cls((h0, w0), (h1, w1), rh / h0, (rh, rw), 0, h1 - rh, 0, w1 - rw,
                   kind="torchvision_resize", box_scale=(rw / w0, rh / h0), box_offset=(0.0, 0.0),
                   clip_boxes=False, fill="0 after normalisation (the normalisation mean)")

    @classmethod
    def yoru_onnx(cls, frame_shape: Sequence[int], input_shape: Sequence[int]) -> "LetterboxTransform":
        """YORU's ONNX plugin: YOLOv5's letterbox; boxes back by ``(coord - pad) / ratio``.

        ``pad = (w1 - round(w0 * ratio)) / 2`` -- from the rounded size, unlike YOLOv5.
        """
        t = cls.yolov5(frame_shape, input_shape)
        h1, w1 = t.input_shape
        new_h, new_w = t.resized_shape
        return cls(t.frame_shape, t.input_shape, t.ratio, t.resized_shape, t.pad_top, t.pad_bottom,
                   t.pad_left, t.pad_right, kind="yoru_onnx_letterbox", box_scale=(t.ratio, t.ratio),
                   box_offset=((w1 - new_w) / 2, (h1 - new_h) / 2))

    # -- mapping -----------------------------------------------------------------

    @property
    def box_gain(self) -> float:
        """The x scale of the box mapping (YOLOv5's ``scale_boxes`` gain)."""
        return self.box_scale[0]

    @property
    def box_pad(self) -> Tuple[float, float]:
        """The ``(x, y)`` offset of the box mapping (YOLOv5's fractional pad)."""
        return self.box_offset

    def boxes_to_frame(self, boxes: np.ndarray) -> np.ndarray:
        """Input-pixel ``xyxy`` boxes as the detector maps them back to the frame."""
        b = np.array(boxes, dtype=np.float64, copy=True).reshape(-1, 4)
        (sx, sy), (ox, oy) = self.box_scale, self.box_offset
        b[:, [0, 2]] = (b[:, [0, 2]] - ox) / sx
        b[:, [1, 3]] = (b[:, [1, 3]] - oy) / sy
        if self.clip_boxes:
            h0, w0 = self.frame_shape
            b[:, [0, 2]] = b[:, [0, 2]].clip(0, w0)
            b[:, [1, 3]] = b[:, [1, 3]].clip(0, h0)
        return b

    def obbs_to_frame(self, obbs: np.ndarray) -> np.ndarray:
        """Input-pixel ``(cx, cy, w, h, angle)`` in frame pixels (isotropic transforms)."""
        o = np.array(obbs, dtype=np.float64, copy=True).reshape(-1, 5)
        (sx, sy), (ox, oy) = self.box_scale, self.box_offset
        o[:, 0] = (o[:, 0] - ox) / sx
        o[:, 1] = (o[:, 1] - oy) / sy
        o[:, 2] /= sx
        o[:, 3] /= sy
        return o

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

    def feature_to_frame(self, feature_map: np.ndarray, stride: float) -> np.ndarray:
        """A map on a stride-*stride* feature grid as a map on frame pixels.

        Cell ``(i, j)`` is centred on input pixel ``((j + 0.5) * stride,
        (i + 0.5) * stride)`` -- the grid convention of YOLO and of SSD's
        default boxes -- which is what a bilinear resize with half-pixel
        centres assumes.  A grid may overhang the input (SSD: 38 cells of
        8 px over 300 px); it must cover it.
        """
        import cv2

        f = np.asarray(feature_map, dtype=np.float32)
        h1, w1 = self.input_shape
        if f.shape[0] * stride == h1 and f.shape[1] * stride == w1:
            up = cv2.resize(f, (w1, h1), interpolation=cv2.INTER_LINEAR)
        elif f.shape[0] * stride >= h1 and f.shape[1] * stride >= w1:
            up = _upsample_grid(f, float(stride), h1, w1)
        else:
            raise ValueError(
                f"feature map {f.shape[:2]} at stride {stride} does not cover the input {self.input_shape}")
        return self.map_to_frame(up)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "frame_shape": list(self.frame_shape),
            "input_shape": list(self.input_shape),
            "ratio": self.ratio,
            "resized_shape": list(self.resized_shape),
            "pad_top": self.pad_top,
            "pad_bottom": self.pad_bottom,
            "pad_left": self.pad_left,
            "pad_right": self.pad_right,
            "box_scale": list(self.box_scale),
            "box_offset": list(self.box_offset),
            "clip_boxes": self.clip_boxes,
            # The YOLOv5 names of the same two numbers, as v0.1 wrote them.
            "box_gain": self.box_gain,
            "box_pad": list(self.box_pad),
            "fill": self.fill,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LetterboxTransform":
        scale = d.get("box_scale")
        offset = d.get("box_offset")
        return cls(
            tuple(int(v) for v in d["frame_shape"]),
            tuple(int(v) for v in d["input_shape"]),
            float(d["ratio"]),
            tuple(int(v) for v in d["resized_shape"]),
            int(d["pad_top"]), int(d["pad_bottom"]), int(d["pad_left"]), int(d["pad_right"]),
            kind=str(d.get("kind") or "yolov5_letterbox"),
            box_scale=tuple(float(v) for v in scale) if scale is not None else None,
            box_offset=tuple(float(v) for v in offset) if offset is not None else None,
            clip_boxes=bool(d.get("clip_boxes", True)),
            fill=d.get("fill", LETTERBOX_FILL),
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
