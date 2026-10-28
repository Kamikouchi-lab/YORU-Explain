# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""ONNX exports of YOLOv5 / YOLOv8 / YOLO11, run as YORU's ONNX plugin runs them.

YORU's ``yoru/libs/plugins/onnx_detector.py`` opens the file with ONNX
Runtime (every available provider), and per frame:

* letterboxes it to the model's fixed input size (``cv2.resize`` to
  ``round(w0*r) x round(h0*r)``, grey 114 padding), converts BGR to RGB,
  divides by 255 in float32;
* reads the single output as rows ``cx, cy, w, h, [objectness,] classes``
  (transposing a YOLOv8 export), deciding whether there is an objectness
  column from the class names in the metadata -- ``nc + 5`` columns -- or,
  without names, from the row count (YOLOv5 exports three rows per cell);
* scores each row by ``objectness x class`` (YOLOv5) or ``class`` (YOLOv8),
  best class only, keeps rows **above** the confidence threshold, maps boxes
  back as ``(coord - pad) / ratio`` and clips them, and runs
  ``cv2.dnn.NMSBoxes`` with every class shifted into its own band.

The adapter does the same, step for step, in the same float32 arithmetic --
YORU keeps these steps as private methods, so they are repeated here, and
``identity_check`` compares the result with YORU's documented
``get_detector("onnx", ...).detect()`` on every run.  What the adapter adds
is the row each detection came from: an *output row* of a fixed-size model
is a fixed anchor slot / anchor point, so fixed-candidate mode follows it.

ONNX Runtime computes no gradients: occlusion only.  The output holds
probabilities, not logits, so no logit scores either.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from yoru_explain.adapters import _yoru
from yoru_explain.adapters.base import (
    AdapterCapabilities,
    CandidateReadout,
    ModelAdapter,
    PreparedInput,
    RawOutput,
    compare_detections,
    detection_row,
    yoru_detection_row,
)
from yoru_explain.core.errors import ErrorCode, ExplainError
from yoru_explain.core.geometry import LetterboxTransform, letterbox_image
from yoru_explain.core.hashing import sha256_file
from yoru_explain.core.logs import get_logger
from yoru_explain.core.types import CandidateRef, Detection

log = get_logger("onnx")

#: The strides YORU's row-count rule assumes.
_STRIDES = (8, 16, 32)


def _boxes_to_frame(rows: np.ndarray, pad_w: float, pad_h: float, ratio: float, orig_w: int,
                    orig_h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """YORU's arithmetic, in the rows' float32: ``(c -+ size/2 - pad) / ratio``, clipped."""
    cx, cy, bw, bh = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    x1 = (cx - bw / 2 - pad_w) / ratio
    y1 = (cy - bh / 2 - pad_h) / ratio
    x2 = (cx + bw / 2 - pad_w) / ratio
    y2 = (cy + bh / 2 - pad_h) / ratio
    return (np.clip(x1, 0, orig_w), np.clip(y1, 0, orig_h), np.clip(x2, 0, orig_w),
            np.clip(y2, 0, orig_h))


class YoruOnnxAdapter(ModelAdapter):
    name = "yoru-onnx"
    version = "0.2.0"
    description = "ONNX exports of YOLOv5 / YOLOv8 / YOLO11 via ONNX Runtime, as YORU runs them (occlusion only)"
    yoru_backend = "onnx"

    def __init__(self):
        super().__init__()
        self._session = None
        self._caps: Optional[AdapterCapabilities] = None

    # -- loading -------------------------------------------------------------

    def load_model(self, weights, device: str = "auto", *, conf_thresh: Optional[float] = None,
                   iou_thresh: Optional[float] = None, names: Optional[Dict[int, str]] = None,
                   probe: bool = True) -> None:
        core = _yoru.core()
        self._core = core
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ExplainError(ErrorCode.MODEL_UNSUPPORTED,
                               f"onnxruntime could not be imported ({e}); YORU installs it") from e
        path = Path(weights)
        if not path.is_file():
            raise ExplainError(ErrorCode.MISSING_DATA, f"model file not found: {path}")
        self._weights = str(path)
        self._sha256 = sha256_file(path)
        self._device_pref = str(device or "auto")
        self._conf = core.default_conf_thresh if conf_thresh is None else float(conf_thresh)
        self._iou = core.default_iou_thresh if iou_thresh is None else float(iou_thresh)
        try:
            self._providers = list(ort.get_available_providers())
            self._session = ort.InferenceSession(self._weights, providers=self._providers)
        except Exception as e:
            raise ExplainError(ErrorCode.MODEL_LOAD_FAILED,
                               f"{path.name} could not be opened by ONNX Runtime: {type(e).__name__}: {e}") from e
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        try:
            self._input_hw = (int(inp.shape[2]), int(inp.shape[3]))
        except (TypeError, ValueError, IndexError) as e:
            raise ExplainError(ErrorCode.MODEL_UNSUPPORTED,
                               f"{path.name}: input shape {inp.shape} is not fixed; YORU's ONNX plugin "
                               "needs a fixed height and width") from e
        # The class names as YORU finds them: the option, else the metadata.
        self._names_option = dict(names) if names else None
        found: Dict[int, str] = dict(names) if names else {}
        self._names_source = "option" if names else "none"
        if not found:
            meta = self._session.get_modelmeta().custom_metadata_map
            if "names" in meta:
                try:
                    found = ast.literal_eval(meta["names"])
                    self._names_source = "metadata"
                except Exception:
                    found = {}
        self._yoru_names = found          # what YORU's plugin holds (may be empty)
        # Run once to learn the output layout (rows, columns) and decide the
        # objectness column the way YORU decides it.
        h, w = self._input_hw
        rows = self._run_blob(np.zeros((1, 3, h, w), np.float32))
        self._n_rows, self._n_cols = rows.shape
        self._objectness = self._has_objectness(self._n_rows, self._n_cols)
        self._nc = self._n_cols - (5 if self._objectness else 4)
        if self._nc < 1:
            raise ExplainError(ErrorCode.MODEL_UNSUPPORTED,
                               f"{path.name}: output rows of {self._n_cols} columns hold no classes")
        self._names = {int(k): str(v) for k, v in found.items()} if found else \
            {i: str(i) for i in range(self._nc)}
        self._levels = self._grid_levels()
        log.info("opened %s (%s) with %s; input %s, %d rows x %d columns (%s); conf %.3g, IoU %.3g",
                 path.name, self._sha256[:12], self._providers, self._input_hw, self._n_rows,
                 self._n_cols, self.layout, self._conf, self._iou)
        self._caps = self._probe() if probe else self._unprobed_caps()

    def _has_objectness(self, n_rows: int, num_cols: int) -> bool:
        """YORU's rule (``ONNXDetector._has_objectness``)."""
        if self._yoru_names:
            nc = len(self._yoru_names)
            if num_cols == nc + 5:
                return True
            if num_cols == nc + 4:
                return False
            log.warning("ONNX output has %d columns, matching neither %d (YOLOv8/11) nor %d (YOLOv5) "
                        "for %d classes; YORU assumes the YOLOv8/11 layout", num_cols, nc + 4, nc + 5, nc)
            return False
        h, w = self._input_hw
        cells = sum((h // s) * (w // s) for s in _STRIDES)
        return n_rows > cells * 1.5

    def _grid_levels(self):
        """``(stride, anchors, ny, nx)`` per level when the rows are the standard grid, else None."""
        h, w = self._input_hw
        na = 3 if self._objectness else 1
        levels = [(s, na, h // s, w // s) for s in _STRIDES]
        return levels if sum(a * y * x for _, a, y, x in levels) == self._n_rows else None

    @property
    def layout(self) -> str:
        return "YOLOv5 (objectness column)" if self._objectness else "YOLOv8/YOLO11 (no objectness)"

    @property
    def weights(self) -> str:
        return self._weights

    @property
    def class_names(self) -> Dict[int, str]:
        return dict(self._names)

    @property
    def capabilities(self) -> AdapterCapabilities:
        return self._caps

    def model_info(self) -> Dict[str, Any]:
        import onnxruntime

        return {
            "adapter": self.name,
            "adapter_version": self.version,
            "backend": self.yoru_backend,
            "weights": self._weights,
            "sha256": self._sha256,
            "class_names": {str(k): v for k, v in self._names.items()},
            "class_names_source": self._names_source,
            "nc": self._nc,
            "layout": self.layout,
            "rows": self._n_rows,
            "columns": self._n_cols,
            "input_name": self._input_name,
            "input_size": max(self._input_hw),
            "imgsz": list(self._input_hw),
            "conf_thresh": self._conf,
            "conf_rule": "score > conf_thresh (strictly), before NMS",
            "nms_conf": self._conf,
            "iou_thresh": self._iou,
            "nms": "cv2.dnn.NMSBoxes, each class offset into its own band (YORU's)",
            "channel_order": "RGB (YORU converts OpenCV's BGR frames), /255",
            "letterbox": "YORU's ONNX letterbox (114 padding); boxes back by (coord - pad) / ratio",
            "providers": list(self._providers),
            "device": "onnxruntime: " + ", ".join(self._providers),
            "device_preference": self._device_pref,
            "precision": "as exported",
            "onnxruntime_version": str(onnxruntime.__version__),
            "yoru_version": self._core.yoru_version,
        }

    # -- inference -----------------------------------------------------------

    def preprocess(self, images: Sequence[np.ndarray]) -> PreparedInput:
        import cv2

        if not images:
            raise ExplainError(ErrorCode.INTERNAL, "no images to preprocess")
        shape = images[0].shape
        if any(im.shape != shape for im in images):
            raise ExplainError(ErrorCode.INTERNAL, "images of one batch must share a shape")
        if len(shape) != 3 or shape[2] != 3:
            raise ExplainError(ErrorCode.MODEL_UNSUPPORTED, f"expected BGR frames, got shape {shape}")
        transform = LetterboxTransform.yoru_onnx(shape[:2], self._input_hw)
        blobs = []
        for im in images:
            img = cv2.cvtColor(letterbox_image(im, transform), cv2.COLOR_BGR2RGB)
            blobs.append(np.ascontiguousarray((img.astype(np.float32) / 255.0).transpose(2, 0, 1)))
        return PreparedInput(np.stack(blobs), transform, tuple(shape))

    def _run_blob(self, blob: np.ndarray) -> np.ndarray:
        """One ``session.run``; the output as ``(rows, columns)``, as YORU reads it."""
        pred = self._session.run(None, {self._input_name: blob})[0]
        pred = np.squeeze(pred, axis=0)
        if pred.shape[0] < pred.shape[1]:   # YOLOv8 exports: (features, boxes)
            pred = pred.T
        return pred

    def predict_raw(self, prepared: PreparedInput, *, logits: bool = False) -> RawOutput:
        # One image per run, as YORU runs it (exports have a batch of 1).
        with self._lock:
            rows = [self._run_blob(prepared.tensor[b:b + 1]) for b in range(prepared.batch_size)]
        return RawOutput(pred=rows)

    def _scores(self, rows: np.ndarray) -> np.ndarray:
        return rows[:, 5:] * rows[:, 4:5] if self._objectness else rows[:, 4:]

    def postprocess(self, raw: RawOutput, prepared: PreparedInput) -> List[List[Detection]]:
        return [self._detections(rows, prepared) for rows in raw.pred]

    def _detections(self, pred: np.ndarray, prepared: PreparedInput) -> List[Detection]:
        """``ONNXDetector._postprocess``, keeping the row of each detection."""
        import cv2

        orig_h, orig_w = prepared.frame_shape[:2]
        t = prepared.transform
        ratio = t.box_scale[0]
        pad_w, pad_h = t.box_offset
        n_rows, num_cols = pred.shape
        if n_rows == 0 or num_cols < 5:
            return []
        cls_scores = self._scores(pred)
        if cls_scores.shape[1] == 0:
            return []
        class_ids = cls_scores.argmax(axis=1)
        confidences = cls_scores[np.arange(len(cls_scores)), class_ids]
        mask = confidences > self._conf
        if not mask.any():
            return []
        rows = np.nonzero(mask)[0]
        pred = pred[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]
        x1, y1, x2, y2 = _boxes_to_frame(pred, pad_w, pad_h, ratio, orig_w, orig_h)
        offset = class_ids.astype(np.float32) * (max(orig_w, orig_h) + 1.0)
        boxes_nms = np.stack([x1 + offset, y1 + offset, x2 - x1, y2 - y1], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(boxes_nms, confidences.tolist(), self._conf, self._iou)
        if len(indices) == 0:
            return []
        out = []
        for i in np.array(indices).flatten():
            cid = int(class_ids[i])
            k = int(rows[i])
            out.append(Detection((float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
                                 float(confidences[i]), cid, self._yoru_names.get(cid, str(cid))
                                 if self._yoru_names else str(cid),
                                 CandidateRef(k, self._locate(k))))
        return out

    def _locate(self, k: int):
        if self._levels is None:
            return ()
        from yoru_explain.adapters._yolo_common import locate_candidate

        loc = locate_candidate(k, self._levels)
        return loc if self._objectness else tuple(p for p in loc if p[0] != "anchor")

    def candidate_readout(self, raw, prepared, candidate_index, target_class, score_type):
        k, c = int(candidate_index), int(target_class)
        if not 0 <= k < self._n_rows:
            raise ExplainError(ErrorCode.SCORE_UNAVAILABLE, f"row {k} out of range ({self._n_rows})")
        if not 0 <= c < self._nc:
            raise ExplainError(ErrorCode.SCORE_UNAVAILABLE, f"class {c} out of range ({self._nc} classes)")
        t = prepared.transform
        h0, w0 = prepared.frame_shape[:2]
        out = []
        for rows in raw.pred:
            row = rows[k:k + 1]                # a one-row array: the same float32 arithmetic
            value = float(self._value(row, c, score_type)[0])
            x1, y1, x2, y2 = _boxes_to_frame(row, *t.box_offset, t.box_scale[0], w0, h0)
            cls = int(self._scores(row).argmax(axis=1)[0])
            out.append(CandidateReadout(value, (float(x1[0]), float(y1[0]), float(x2[0]), float(y2[0])), cls))
        return out

    def _value(self, rows: np.ndarray, c: int, score_type: str) -> np.ndarray:
        if score_type == "confidence":
            return self._scores(rows)[:, c]
        if self._objectness:
            if score_type == "obj_x_cls":
                return self._scores(rows)[:, c]
            if score_type == "cls_prob":
                return rows[:, 5 + c]
            if score_type == "obj_prob":
                return rows[:, 4]
        elif score_type == "cls_prob":
            return rows[:, 4 + c]
        raise ExplainError(ErrorCode.MODEL_UNSUPPORTED, f"{self.name} ({self.layout}) cannot read {score_type!r}")

    # -- the check against YORU itself -------------------------------------------

    def identity_check(self, image: np.ndarray) -> Dict[str, Any]:
        """Detections here vs. ``get_detector('onnx', ...).detect(image)``."""
        kwargs = {"names": self._names_option} if self._names_option else {}
        reference = self._core.get_detector("onnx", self._weights, conf_thresh=self._conf,
                                            iou_thresh=self._iou, **kwargs)
        try:
            ref = [yoru_detection_row(d, False) for d in reference.detect(image)]
        finally:
            del reference
        mine = [detection_row(d, False) for d in self.detect(image)]
        return compare_detections("yoru.libs.plugins.get_detector('onnx', ...).detect", ref, mine)

    # -- probe -----------------------------------------------------------------

    def _score_types(self) -> Tuple[str, ...]:
        if self._objectness:
            return ("confidence", "obj_x_cls", "cls_prob", "obj_prob")
        return ("confidence", "cls_prob")

    def _family(self) -> Dict[str, Any]:
        if self._objectness:
            desc = (("confidence", "objectness x class probability -- YOLOv5's confidence as YORU's "
                                   "ONNX plugin computes it (= obj_x_cls)"),
                    ("obj_x_cls", "objectness x class probability, as exported"),
                    ("cls_prob", "class probability given an object, as exported"),
                    ("obj_prob", "objectness, as exported"))
        else:
            desc = (("confidence", "class probability -- the YOLOv8/YOLO11 confidence as YORU's "
                                   "ONNX plugin reads it (= cls_prob)"),
                    ("cls_prob", "class probability, as exported"))
        return {"family": f"ONNX export, {self.layout}",
                "candidate_kind": "output row",
                "reported_score": "obj_x_cls" if self._objectness else "cls_prob",
                "score_descriptions": desc}

    def _unprobed_caps(self) -> AdapterCapabilities:
        return AdapterCapabilities(True, True, self._score_types(), False,
                                   "ONNX Runtime computes no gradients", **self._family())

    def _probe(self) -> AdapterCapabilities:
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
        p = self.preprocess([frame])
        a = self.predict_raw(p).pred[0]
        b = self.predict_raw(p).pred[0]
        return AdapterCapabilities(
            fixed_candidate=True,
            redetection=True,
            score_types=self._score_types(),
            gradients=False,
            gradients_reason="ONNX Runtime computes no gradients: occlusion only",
            deterministic=bool(np.array_equal(a, b)),
            # One image per run, as YORU: a batch is a loop, so nothing depends on its size.
            batch_invariant=True,
            batch_max_abs_diff=0.0,
            notes=("one image per session.run, as YORU's ONNX plugin runs it",)
                  + (() if self._levels else ("the output rows are not the standard 8/16/32 grid: "
                                              "candidates carry no grid position",)),
            **self._family(),
        )
