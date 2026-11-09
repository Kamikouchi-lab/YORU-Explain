# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Every event of an export drawn the same way, on a colour scale they share.

::

    gallery/
    ├── gallery.csv                       one row per event: its picture (or why there is none), its scale
    ├── <class>/<unit>__<event>__f<frame>.png   frame | map | overlay | map on its own scale
    └── sheets/<class>__p01.png, <class>__<unit>__p01.png   contact sheets, failures included

A run preview scales each map to its own largest value, so two events'
colours cannot be compared.  Here one scale is used for the whole run
(``run``), for each predicted class (``class``, the default), or -- as in
the previews -- for each event (``event``); a number fixes it (``+-v`` for
occlusion deltas, ``0..v`` for Grad-CAM).  A group's scale is its largest
value, so no colour saturates; events with small effects then look faint,
which is what they are -- each event's picture adds the map on its own scale
as a fourth panel, for its pattern.  Overlays are as opaque as the value is
large, so the frame shows through where the map is near 0.  The scale is
printed on every picture and sheet.  Everything is drawn from the export
folder (its map crops and frame crops); nothing is recomputed.  For looking
at only: the numbers are in the export's tables and arrays.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from yoru_explain.core.errors import ErrorCode, ExplainError
from yoru_explain.io import schema
from yoru_explain.io.frames import imread, imwrite
from yoru_explain.visualization.overlay import colorbar, draw_box, heatmap, text_block

SCALES = ("run", "class", "event")
_GREY = (64, 64, 64)
_BG = (32, 32, 32)
_TEXT = (235, 235, 235)
NOTE = "Change of the model's score when a region is replaced -- not a biological interpretation"
CAM_NOTE = "Grad-CAM attribution of the model's score -- not a biological interpretation"


@dataclass
class GalleryOutcome:
    gallery_dir: Path
    pictures: int
    sheets: int
    scales: Dict[str, Tuple[float, float]] = field(default_factory=dict)


def safe_name(text: str, limit: int = 60) -> str:
    """*text* as part of a file name on any system."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(text)).strip(" .")
    return (s or "_")[:limit]


def _range(kind: str, values: Sequence[np.ndarray]) -> Tuple[float, float]:
    finite = [v[np.isfinite(v)] for v in values]
    finite = [v for v in finite if v.size]
    if not finite:
        return (-1.0, 1.0) if kind == "delta" else (0.0, 1.0)
    allv = np.concatenate(finite)
    if kind == "delta":
        m = float(np.max(np.abs(allv)))
        m = m if m > 0 else 1.0
        return -m, m
    m = float(np.max(allv))
    return 0.0, (m if m > 0 else 1.0)


def _upscale(image: np.ndarray, k: int) -> np.ndarray:
    return image if k == 1 else np.repeat(np.repeat(image, k, axis=0), k, axis=1)


def _wrap(width: int, lines: Sequence[str]) -> List[str]:
    """*lines* broken at spaces so that each fits *width* pixels of :func:`text_block`'s font."""
    import cv2

    def fits(text: str) -> bool:
        return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0] <= width - 12

    out: List[str] = []
    for line in lines:
        current = ""
        for word in str(line).split(" "):
            trial = f"{current} {word}" if current else word
            if current and not fits(trial):
                out.append(current)
                current = word
            else:
                current = trial
        out.append(current)
    return out


def weighted_overlay(base: np.ndarray, heat: np.ndarray, values: np.ndarray,
                     vrange: Tuple[float, float], alpha_max: float = 0.75) -> np.ndarray:
    """*heat* over *base*, each pixel as opaque as its value is large on the scale.

    Where the map is about 0 (or undefined) the frame shows through
    unchanged, so the picture under a small effect stays visible.  For
    looking at only.
    """
    v = np.asarray(values, np.float64)
    span = max(abs(float(vrange[0])), abs(float(vrange[1]))) or 1.0
    a = np.zeros(v.shape, np.float64)
    finite = np.isfinite(v)
    a[finite] = np.clip(np.abs(v[finite]) / span, 0.0, 1.0) * alpha_max
    a = a[..., None]
    out = base.astype(np.float64) * (1.0 - a) + heat.astype(np.float64) * a
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _event_picture(crop: Optional[np.ndarray], m: np.ndarray, kind: str, vrange, box, offset, obb,
                   label: str, lines: List[str], own_scale: bool) -> np.ndarray:
    """crop | map | overlay (| map on its own scale) over the analysed region.

    Enlarged (nearest) when the region is small.
    """
    h, w = m.shape
    k = max(1, int(math.ceil(160 / max(1, h))))
    heat, valid, _ = heatmap(m, kind, vrange)
    base = crop if crop is not None else np.full((h, w, 3), _GREY, np.uint8)
    over = weighted_overlay(base, heat, m, vrange)
    images = [base, heat, over]
    if own_scale:
        own, _, own_range = heatmap(m, kind)
        images.append(own)
        lines = list(lines) + [f"right: this event on its own scale {own_range[0]:.3g} .. {own_range[1]:.3g} "
                               f"(its pattern; not comparable with other events)"]
    x0, y0 = offset
    b = [(box[0] - x0) * k, (box[1] - y0) * k, (box[2] - x0) * k, (box[3] - y0) * k]
    o = None if obb is None else [(obb[0] - x0) * k, (obb[1] - y0) * k, obb[2] * k, obb[3] * k, obb[4]]
    panels = [draw_box(_upscale(p, k).copy(), b, label=label if i == 0 else "", obb=o, thickness=1)
              for i, p in enumerate(images)]
    if crop is None:
        import cv2

        cv2.putText(panels[0], "frame not exported", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _TEXT, 1,
                    cv2.LINE_AA)
    row = np.concatenate(panels, axis=1)
    return np.concatenate([row, colorbar(row.shape[1], kind, vrange),
                           text_block(row.shape[1], _wrap(row.shape[1], lines))], axis=0)


def _tile(image: Optional[np.ndarray], size: int, caption: Sequence[str], status: str) -> np.ndarray:
    import cv2

    tile = np.full((size, size, 3), _BG, np.uint8)
    if image is not None:
        h, w = image.shape[:2]
        s = size / max(h, w)
        nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
        small = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST)
        ty, tx = (size - nh) // 2, (size - nw) // 2
        tile[ty:ty + nh, tx:tx + nw] = small
    else:
        tile[:] = _GREY
        cv2.putText(tile, status[:22], (6, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 180, 255), 1,
                    cv2.LINE_AA)
    strip = text_block(size, list(caption))
    return np.concatenate([tile, strip], axis=0)


def _sheet(tiles: List[np.ndarray], columns: int, title: str, kind: str,
           vrange: Optional[Tuple[float, float]]) -> np.ndarray:
    th, tw = tiles[0].shape[:2]
    columns = max(1, min(int(columns), len(tiles)))
    rows = int(math.ceil(len(tiles) / columns))
    gap = 4
    width = columns * tw + (columns + 1) * gap
    grid = np.full((rows * th + (rows + 1) * gap, width, 3), _BG, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, columns)
        y, x = gap + r * (th + gap), gap + c * (tw + gap)
        grid[y:y + th, x:x + tw] = t
    parts = [text_block(width, _wrap(width, [title])), grid]
    if vrange is not None:
        parts.append(colorbar(width, kind, vrange))
    else:
        parts.append(text_block(width, _wrap(width, [
            "each tile on its own colour scale (largest value of that event)"])))
    parts.append(text_block(width, _wrap(width, [NOTE if kind == "delta" else CAM_NOTE])))
    return np.concatenate(parts, axis=0)


def render_gallery(export_dir, out_dir=None, *, scale: Union[str, float] = "class", columns: int = 8,
                   rows: int = 6, tile_px: int = 180, by_unit: bool = True,
                   progress: Optional[Callable[[float, str], None]] = None,
                   cancel: Optional[Callable[[], bool]] = None) -> GalleryOutcome:
    """Draw every event of the export folder *export_dir* into *out_dir* (default ``gallery/``).

    *scale*: ``run``, ``class``, ``event`` or a number (see the module
    docstring).  Contact sheets hold *columns* x *rows* tiles each, per
    class and -- with *by_unit* -- per unit and class, sorted by unit and
    frame; events without a map appear as grey tiles with their status, so
    what is missing stays visible.  An earlier gallery in *out_dir* is
    replaced.
    """
    export_dir = Path(export_dir)
    info = json.loads((export_dir / "export.json").read_text(encoding="utf-8"))
    kind = info.get("map", "delta")
    events = schema.read_rows(export_dir / "events.csv")
    out = Path(out_dir) if out_dir is not None else export_dir / "gallery"
    if out.exists():
        if any(out.iterdir()) and not (out / "gallery.csv").is_file():
            raise ValueError(f"{out} holds other files and is not a gallery; choose another folder")
        shutil.rmtree(out)
    (out / "sheets").mkdir(parents=True)
    if isinstance(scale, str) and scale not in SCALES:
        try:
            scale = float(scale)
        except ValueError:
            raise ValueError(f"scale must be one of {SCALES} or a number, not {scale!r}") from None
    if not isinstance(scale, str) and not float(scale) > 0:
        raise ValueError("a fixed scale must be > 0")

    maps: Dict[str, Dict[str, Any]] = {}
    for e in events:
        if e.get("map_file"):
            with np.load(export_dir / e["map_file"], allow_pickle=False) as d:
                maps[e["event_id"]] = {"map": d["map"], "offset": tuple(int(v) for v in d["offset"]),
                                       "box": tuple(float(v) for v in d["box"]),
                                       "obb": None if not np.isfinite(d["obb"]).all()
                                       else tuple(float(v) for v in d["obb"])}

    def group(e) -> str:
        return {"run": "run", "class": f"class:{e['class_name']}", "event": f"event:{e['event_id']}"}.get(
            scale, "fixed") if isinstance(scale, str) else "fixed"

    scales: Dict[str, Tuple[float, float]] = {}
    if isinstance(scale, str):
        members: Dict[str, List[np.ndarray]] = {}
        for e in events:
            if e["event_id"] in maps:
                members.setdefault(group(e), []).append(maps[e["event_id"]]["map"])
        scales = {g: _range(kind, v) for g, v in members.items()}
    else:
        v = float(scale)
        scales = {"fixed": (-v, v) if kind == "delta" else (0.0, v)}

    order = sorted(events, key=lambda e: (e.get("class_name") or "", e.get("unit_id") or "",
                                          int(float(e.get("frame_index") or -1)), e["event_id"]))
    index_rows, tiles_by_class, tiles_by_unit = [], {}, {}
    pictures = 0
    for n, e in enumerate(order):
        if cancel is not None and cancel():
            raise ExplainError(ErrorCode.CANCELLED, "gallery cancelled")
        if progress:
            progress(n / max(1, len(order)), f"{e['event_id']}: drawing")
        eid, cls, unit = e["event_id"], e.get("class_name") or "", e.get("unit_id") or ""
        frame = e.get("frame_index") or ""
        g = group(e)
        row = {"event_id": eid, "unit_id": unit, "class_name": cls, "frame_index": frame,
               "status": e.get("status") or "", "scale_group": g}
        image = None
        if eid in maps:
            mp = maps[eid]
            vr = scales[g]
            crop = imread(export_dir / e["crop_file"]) if e.get("crop_file") else None
            if crop is not None and crop.shape[:2] != mp["map"].shape:
                crop = None   # not the region of the map: draw the map alone
            ref = schema.as_float(e.get("reference_score"))
            lines = [f"{eid}  unit {unit}  frame {frame}  predicted class {cls}  score {ref:.4f}",
                     f"{e.get('method')} {e.get('score_type')} {e.get('identity_mode')}  patch "
                     f"{e.get('patch_height_px') or '-'} px  baseline {e.get('baseline') or '-'}",
                     f"colour scale {vr[0]:.3g} .. {vr[1]:.3g} ({g})",
                     NOTE if kind == "delta" else CAM_NOTE]
            picture = _event_picture(crop, mp["map"], kind, vr, mp["box"], mp["offset"], mp["obb"],
                                     f"{cls} {ref:.3f}", lines, own_scale=not g.startswith("event:"))
            rel = f"{safe_name(cls)}/{safe_name(unit)}__{safe_name(eid)}__f{frame}.png"
            imwrite(out / rel, picture)
            pictures += 1
            row.update(file=rel, vmin=vr[0], vmax=vr[1])
            heat, valid, _ = heatmap(mp["map"], kind, vr)
            base = crop if crop is not None else np.full(heat.shape, _GREY, np.uint8)
            x0, y0 = mp["offset"]
            b = mp["box"]
            image = draw_box(weighted_overlay(base, heat, mp["map"], vr),
                             [b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0],
                             thickness=1, obb=None if mp["obb"] is None else
                             [mp["obb"][0] - x0, mp["obb"][1] - y0, *mp["obb"][2:]])
        index_rows.append(row)
        tile = _tile(image, tile_px, [f"{eid} f{frame}", f"unit {unit}"],
                     e.get("error_code") or e.get("status") or "no map")
        tiles_by_class.setdefault(cls, []).append(tile)
        if by_unit:
            tiles_by_unit.setdefault((cls, unit), []).append(tile)
    schema.write_rows(out / "gallery.csv", schema.GALLERY_COLUMNS, index_rows)

    sheets = 0
    per_page = max(1, int(columns) * int(rows))

    def write_sheets(name: str, title: str, tiles: List[np.ndarray], vr) -> int:
        pages = int(math.ceil(len(tiles) / per_page))
        for p in range(pages):
            chunk = tiles[p * per_page:(p + 1) * per_page]
            img = _sheet(chunk, int(columns), f"{title}  page {p + 1}/{pages}", kind, vr)
            imwrite(out / "sheets" / f"{name}__p{p + 1:02d}.png", img)
        return pages

    for cls, tiles in tiles_by_class.items():
        vr = scales.get("run") if scale == "run" else scales.get(f"class:{cls}") if scale == "class" \
            else scales.get("fixed")
        sheets += write_sheets(safe_name(cls), f"predicted class {cls}: {len(tiles)} event(s)", tiles, vr)
    for (cls, unit), tiles in tiles_by_unit.items():
        vr = scales.get("run") if scale == "run" else scales.get(f"class:{cls}") if scale == "class" \
            else scales.get("fixed")
        sheets += write_sheets(f"{safe_name(cls)}__{safe_name(unit)}",
                               f"predicted class {cls}, unit {unit}: {len(tiles)} event(s)", tiles, vr)
    if progress:
        progress(1.0, f"drew {pictures} picture(s) and {sheets} sheet(s)")
    return GalleryOutcome(out, pictures, sheets, scales)
