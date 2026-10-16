# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""What an occluded region is replaced with.

Every choice here makes an image the model never saw in training, and the
choice changes the measured effect: black over a bright arena is a strong,
out-of-distribution edit; a blur keeps the colours and removes detail; the
region's own mean removes texture only.  So the baseline is part of the
result -- it is written into every run's ``config.json``, its description
into ``manifest.json``, and each patch's actual pixel change into
``perturbations.csv`` (a patch whose pixels did not change cannot have an
effect, whatever its delta says).

The default is a constant hot pink (CSS ``hotpink``, #FF69B4), the lab's
choice.  Like any constant colour it is out of distribution; it is also
strongly red, and a fly's eyes are red, so compare it with another baseline
before reading much into a small effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import numpy as np

#: The default baseline colour, CSS ``hotpink`` (#FF69B4), in OpenCV's BGR order.
HOT_PINK: Tuple[int, int, int] = (180, 105, 255)

#: Names printed for well-known constant colours (BGR).
NAMED_COLOURS: Dict[Tuple[int, int, int], str] = {
    HOT_PINK: "hot pink #FF69B4",
    (0, 0, 0): "black",
    (114, 114, 114): "YOLOv5 letterbox grey",
}

BASELINE_KINDS: Dict[str, str] = {
    "constant": "every occluded pixel set to one BGR colour (value); the default (180,105,255) is "
                "hot pink #FF69B4, (0,0,0) black, (114,114,114) YOLOv5's letterbox grey",
    "image_mean": "the frame's mean colour",
    "patch_mean": "the occluded region's own mean colour: removes texture, keeps colour",
    "blur": "the same region of a Gaussian-blurred frame (sigma): removes detail, keeps colour",
    "noise": "uniform random colours, seeded per patch",
}


@dataclass(frozen=True)
class BaselineSpec:
    kind: str = "constant"
    #: ``constant``: BGR colour; hot pink unless set.
    value: Tuple[int, int, int] = HOT_PINK
    #: ``blur``: Gaussian sigma in pixels.
    sigma: float = 8.0

    def __post_init__(self):
        if self.kind not in BASELINE_KINDS:
            raise ValueError(f"unknown baseline {self.kind!r}; one of {sorted(BASELINE_KINDS)}")
        if len(self.value) != 3 or any(not 0 <= int(v) <= 255 for v in self.value):
            raise ValueError(f"baseline value must be three numbers in 0..255, got {self.value}")
        if self.kind == "blur" and not float(self.sigma) > 0:
            raise ValueError("blur sigma must be > 0")

    def describe(self) -> str:
        if self.kind == "constant":
            value = tuple(int(v) for v in self.value)
            name = NAMED_COLOURS.get(value)
            return f"constant BGR {value}" + (f" ({name})" if name else "")
        if self.kind == "blur":
            return f"Gaussian blur, sigma {float(self.sigma):g} px"
        return self.kind.replace("_", " ")

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "value": [int(v) for v in self.value], "sigma": float(self.sigma)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "BaselineSpec":
        unknown = set(d) - {"kind", "value", "sigma"}
        if unknown:
            raise ValueError(f"unknown baseline keys: {sorted(unknown)}")
        # Every config written records its value, so an older run's colour
        # (black, before hot pink became the default) is read back as it was.
        return cls(kind=str(d.get("kind", "constant")),
                   value=tuple(int(v) for v in d.get("value", HOT_PINK)),
                   sigma=float(d.get("sigma", 8.0)))


class Baseline:
    """A :class:`BaselineSpec` prepared for one frame."""

    def __init__(self, spec: BaselineSpec, frame: np.ndarray, seed: int = 0):
        self.spec = spec
        self.frame = frame
        self.seed = int(seed)
        self._full = None
        if spec.kind == "constant":
            self._full = np.empty_like(frame)
            self._full[...] = np.array(spec.value, dtype=frame.dtype)[: frame.shape[2]] \
                if frame.ndim == 3 else int(spec.value[0])
        elif spec.kind == "image_mean":
            mean = frame.reshape(-1, frame.shape[2]).mean(0) if frame.ndim == 3 else frame.mean()
            self._full = np.empty_like(frame)
            self._full[...] = np.rint(mean).astype(frame.dtype)
        elif spec.kind == "blur":
            import cv2

            self._full = cv2.GaussianBlur(frame, (0, 0), sigmaX=float(spec.sigma),
                                          borderType=cv2.BORDER_REFLECT)

    def fill(self, out: np.ndarray, y0: int, y1: int, x0: int, x1: int, patch_id: int) -> None:
        """Replace ``out[y0:y1, x0:x1]`` by the baseline (in place)."""
        if self._full is not None:
            out[y0:y1, x0:x1] = self._full[y0:y1, x0:x1]
        elif self.spec.kind == "patch_mean":
            region = self.frame[y0:y1, x0:x1]
            mean = region.reshape(-1, region.shape[-1]).mean(0) if region.ndim == 3 else region.mean()
            out[y0:y1, x0:x1] = np.rint(mean).astype(out.dtype)
        elif self.spec.kind == "noise":
            # Seeded by (seed, patch id): a patch gets the same noise whatever
            # batch it lands in and however the run is split.
            rng = np.random.default_rng((self.seed, int(patch_id)))
            out[y0:y1, x0:x1] = rng.integers(0, 256, size=out[y0:y1, x0:x1].shape, dtype=np.uint8)
        else:  # pragma: no cover - guarded by BaselineSpec
            raise ValueError(self.spec.kind)
