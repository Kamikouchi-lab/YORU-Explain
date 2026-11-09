# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Maps on grids that do not depend on where, or how large, the animal was.

A map (or picture) in frame pixels is sampled on an ``n x n`` grid whose
cell centres are ``-E + (k + 0.5) * 2E / n`` (``k = 0 .. n-1``) along both
axes, ``E`` being the grid's *extent*:

``box`` grid -- :func:`box_points`
    Centred on the target's box and scaled by its width and height: cell
    ``(row r, column c)`` looks at frame point ``x = cx + u_c * w``,
    ``y = cy + v_r * h``.  The box's edges are at ``u, v = +-0.5``; nothing
    about the animal's orientation is assumed (up is the frame's up).
``aligned`` grid -- :func:`aligned_points`
    Centred on a point, turned and scaled by a direction and a length that
    **the experimenter** gives (from their own tracking, pose estimation or
    judgement; YORU-Explain does not estimate orientation).  The given
    direction points up: row 0 is the most forward, and columns run to the
    animal's right as seen from above with its head up.  ``a`` (forward)
    and ``b`` (right) are in units of the length.  With ``directed=False``
    (an axis without a known head end) an event's grid is the mean of the
    two opposite directions.

Sampling is nearest-pixel: a cell takes the value of the pixel its centre
falls in (pixel ``(i, j)`` covers ``[j, j+1) x [i, i+1)`` in frame
coordinates), so no value is made up by interpolation; a cell whose pixel
is undefined (NaN) or off the map is NaN.  Means over events keep, per
cell, the number of events -- or units -- that defined it, and the mean
over a class is a mean of **unit** means (frames are not independent
samples).  Missing values are never 0.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from yoru_explain.core.errors import ErrorCode, ExplainError
from yoru_explain.io import schema


def grid_centres(n: int, extent: float) -> np.ndarray:
    """The ``n`` cell centres spanning ``[-extent, extent]``, exactly antisymmetric.

    ``g[n-1-k] == -g[k]`` holds bit for bit, so a grid turned by 180 degrees
    samples exactly the mirrored points.
    """
    if int(n) < 1 or not float(extent) > 0:
        raise ValueError("a grid needs n >= 1 cells and an extent > 0")
    step = 2.0 * float(extent) / int(n)
    g = -float(extent) + (np.arange(int(n), dtype=np.float64) + 0.5) * step
    return (g - g[::-1]) / 2.0


def box_points(box: Sequence[float], n: int, extent: float) -> Tuple[np.ndarray, np.ndarray]:
    """Frame ``(x, y)`` of every cell of the box grid: two ``n x n`` arrays."""
    x1, y1, x2, y2 = (float(v) for v in box)
    g = grid_centres(n, extent)
    cx, cy, w, h = (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1
    xs = np.broadcast_to(cx + g[None, :] * w, (len(g), len(g)))
    ys = np.broadcast_to(cy + g[:, None] * h, (len(g), len(g)))
    return xs, ys


def box_region(box: Sequence[float], extent: float, width: int, height: int) -> Tuple[int, int, int, int]:
    """Integer frame ``(x0, y0, x1, y1)`` the box grid of *extent* covers, clipped to the frame."""
    x1, y1, x2, y2 = (float(v) for v in box)
    cx, cy, w, h = (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1
    x0 = max(0, int(math.floor(cx - extent * w)))
    y0 = max(0, int(math.floor(cy - extent * h)))
    xa = min(int(width), int(math.ceil(cx + extent * w)))
    ya = min(int(height), int(math.ceil(cy + extent * h)))
    return x0, y0, max(x0, xa), max(y0, ya)


def _unit(v: float) -> float:
    """cos / sin with the rounding residue of multiples of 90 degrees removed."""
    if abs(v) < 1e-12:
        return 0.0
    if abs(abs(v) - 1.0) < 1e-12:
        return math.copysign(1.0, v)
    return v


def aligned_points(cx: float, cy: float, angle_rad: float, length: float, n: int,
                   extent: float, *, reverse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Frame ``(x, y)`` of every cell of an aligned grid (see the module docstring).

    *angle_rad* is the forward direction, from the frame's +x axis towards
    +y (clockwise on screen, as YORU's rotated boxes), *length* the unit of
    the grid in pixels.  *reverse* turns the grid by exactly 180 degrees
    (the other end of an axis forward).
    """
    g = grid_centres(n, extent)
    a = -g[:, None]          # forward: row 0 is the most forward
    b = g[None, :]           # to the animal's right with its head up
    hx, hy = _unit(math.cos(angle_rad)), _unit(math.sin(angle_rad))
    if reverse:
        hx, hy = -hx, -hy
    rx, ry = -hy, hx         # the forward direction turned a quarter clockwise on screen
    xs = cx + length * (a * hx + b * rx)
    ys = cy + length * (a * hy + b * ry)
    return xs, ys


def sample(values: np.ndarray, xs: np.ndarray, ys: np.ndarray,
           offset: Sequence[int] = (0, 0)) -> Tuple[np.ndarray, np.ndarray]:
    """``(samples, inside)``: *values* at frame points ``(xs, ys)``, nearest pixel.

    *values* is a map or a picture whose pixel ``(0, 0)`` is frame pixel
    *offset* ``(x0, y0)`` (a crop).  Points off it are NaN for float maps
    and 0 for pictures; *inside* says which were on it.
    """
    v = np.asarray(values)
    ix = np.floor(np.asarray(xs, np.float64)).astype(np.int64) - int(offset[0])
    iy = np.floor(np.asarray(ys, np.float64)).astype(np.int64) - int(offset[1])
    h, w = v.shape[:2]
    inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    shape = ix.shape + v.shape[2:]
    if np.issubdtype(v.dtype, np.floating):
        out = np.full(shape, np.nan, np.float32)
    else:
        out = np.zeros(shape, v.dtype)
    out[inside] = v[iy[inside], ix[inside]]
    return out, inside


# -- means over events, then over units ---------------------------------------------


def nanmean_stack(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(mean, count)`` over axis 0 of the finite values; NaN where none."""
    s = np.asarray(stack, np.float64)
    finite = np.isfinite(s)
    count = finite.sum(axis=0)
    total = np.where(finite, s, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
    mean[count == 0] = np.nan
    return mean.astype(np.float32), count.astype(np.int32)


@dataclass
class MeanMaps:
    """Means of a stack of grids: per unit and class, then per class over units.

    ``mean_by_class[i]`` is the mean over units of the unit means of class
    ``class_names[i]`` (each unit weighs the same however many events it
    has); ``n_units_by_class[i]`` counts, per cell, the units that defined
    it.  ``mean_by_unit[k]`` is the mean of the events of unit
    ``unit_keys[k] = (class_name, unit_id)``, ``n_events_by_unit[k]`` the
    events that defined each cell.  Descriptive: no interval or test.
    """

    class_names: List[str]
    mean_by_class: np.ndarray
    n_units_by_class: np.ndarray
    unit_keys: List[Tuple[str, str]]
    mean_by_unit: np.ndarray
    n_events_by_unit: np.ndarray

    def arrays(self) -> Dict[str, np.ndarray]:
        return {
            "class_names": np.array(self.class_names, dtype=str),
            "mean_by_class": self.mean_by_class,
            "n_units_by_class": self.n_units_by_class,
            "unit_class": np.array([k[0] for k in self.unit_keys], dtype=str),
            "unit_id": np.array([k[1] for k in self.unit_keys], dtype=str),
            "mean_by_unit": self.mean_by_unit,
            "n_events_by_unit": self.n_events_by_unit,
        }


def unit_class_means(stack: np.ndarray, class_names: Sequence[str],
                     unit_ids: Sequence[str]) -> MeanMaps:
    """:class:`MeanMaps` of *stack* (``N x ...``), events labelled by class and unit."""
    stack = np.asarray(stack)
    cell = stack.shape[1:]
    groups: Dict[Tuple[str, str], List[int]] = {}
    for i, key in enumerate(zip(class_names, unit_ids)):
        groups.setdefault((str(key[0]), str(key[1])), []).append(i)
    keys = sorted(groups)
    by_unit, n_events = [], []
    for k in keys:
        m, c = nanmean_stack(stack[groups[k]])
        by_unit.append(m)
        n_events.append(c)
    classes = sorted({k[0] for k in keys})
    by_class, n_units = [], []
    for name in classes:
        rows = [i for i, k in enumerate(keys) if k[0] == name]
        m, c = nanmean_stack(np.stack([by_unit[i] for i in rows]))
        by_class.append(m)
        n_units.append(c)
    empty = np.zeros((0,) + cell, np.float32)
    return MeanMaps(classes, np.stack(by_class) if by_class else empty,
                    np.stack(n_units) if n_units else empty.astype(np.int32), keys,
                    np.stack(by_unit) if by_unit else empty,
                    np.stack(n_events) if n_events else empty.astype(np.int32))


# -- the experimenter's orientations ------------------------------------------------

ORIENTATION_COLUMNS = ("event_id", "cx", "cy", "angle_deg", "length")


@dataclass(frozen=True)
class Orientation:
    """Where an event's animal is and which way it points, as the experimenter says.

    ``cx, cy``: the centre, frame pixels.  ``angle_deg``: the forward
    direction in degrees from the frame's +x axis towards +y (clockwise on
    screen; 0 = right, 90 = down).  ``length``: the grid's unit in pixels
    (a body length, say).  ``directed``: whether the head end is known;
    ``False`` averages both ends.
    """

    event_id: str
    cx: float
    cy: float
    angle_deg: float
    length: float
    directed: bool = True

    def valid(self) -> bool:
        return all(math.isfinite(v) for v in (self.cx, self.cy, self.angle_deg, self.length)) \
            and self.length > 0


def _bool_cell(text: str, where: str) -> bool:
    t = str(text).strip().lower()
    if t in ("", "1", "true", "yes", "y"):
        return True
    if t in ("0", "false", "no", "n"):
        return False
    raise ExplainError(ErrorCode.INVALID_CONFIG, f"{where}: directed must be true or false, not {text!r}")


def read_orientations(path) -> List[Orientation]:
    """An orientation table: ``event_id, cx, cy, angle_deg, length[, directed]``.

    Other columns are ignored (the experimenter's own).  An empty or
    non-numeric cell makes that event's orientation unusable (it is then
    left out of the aligned maps with the status ``invalid_orientation``),
    not an error; an event listed twice is an error.
    """
    path = Path(path)
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = [h.strip() for h in (reader.fieldnames or [])]
        missing = [c for c in ORIENTATION_COLUMNS if c not in header]
        if missing:
            raise ExplainError(ErrorCode.INVALID_CONFIG,
                               f"{path.name}: missing column(s) {', '.join(missing)} "
                               f"(needs {', '.join(ORIENTATION_COLUMNS)}[, directed])")
        out: List[Orientation] = []
        seen = set()
        for n, raw in enumerate(reader, start=2):
            row = {str(k).strip(): (v or "").strip() for k, v in raw.items() if k is not None}
            eid = row["event_id"]
            if not eid:
                continue
            if eid in seen:
                raise ExplainError(ErrorCode.INVALID_CONFIG, f"{path.name}: event {eid} is listed twice")
            seen.add(eid)

            def num(key):
                try:
                    return float(row[key]) if row[key] else float("nan")
                except ValueError:
                    return float("nan")

            out.append(Orientation(eid, num("cx"), num("cy"), num("angle_deg"), num("length"),
                                   _bool_cell(row.get("directed", ""), f"{path.name} line {n}")))
    return out


def obb_orientations(events: Iterable[Mapping[str, Any]]) -> List[Orientation]:
    """Orientations from the model's own rotated boxes (OBB models only).

    The axis is the box's longer side and the length that side; the box
    does not say which end is the head, so they are undirected.  Events
    without a rotated box are left out.
    """
    out = []
    for e in events:
        try:
            cx, cy, w, h, ang = (float(e[k]) for k in ("obb_cx", "obb_cy", "obb_w", "obb_h", "obb_angle"))
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (cx, cy, w, h, ang)):
            continue
        if h > w:
            w, h, ang = h, w, ang + math.pi / 2
        out.append(Orientation(str(e["event_id"]), cx, cy, math.degrees(ang), w, directed=False))
    return out


# -- aligned maps from an export folder ---------------------------------------------


@dataclass
class AlignedMaps:
    """Maps (and pictures) of an export, turned to the experimenter's orientations.

    ``maps[k]`` / ``images[k]`` belong to ``event_ids[k]``; ``rows`` has one
    entry per event of the export with its status (``ok``,
    ``no_orientation``, ``invalid_orientation``, ``not_analysed``).
    ``images`` are RGB, 0 where the picture was not exported
    (``image_valid`` False).
    """

    grid: int
    extent: float
    event_ids: List[str]
    unit_ids: List[str]
    class_names: List[str]
    maps: np.ndarray
    images: Optional[np.ndarray]
    image_valid: Optional[np.ndarray]
    rows: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def mean_by_class(self) -> MeanMaps:
        """Unit means, then their mean per class (see :class:`MeanMaps`)."""
        return unit_class_means(self.maps, self.class_names, self.unit_ids)

    def save(self, folder) -> Path:
        """``aligned.npz`` (+ ``aligned_means.npz``) and ``aligned.csv`` in *folder*."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        g = grid_centres(self.grid, self.extent)
        meta = dict(self.meta, schema_version=schema.SCHEMA_VERSION, grid=self.grid, extent=self.extent)
        arrays = {
            "event_id": np.array(self.event_ids, dtype=str),
            "unit_id": np.array(self.unit_ids, dtype=str),
            "class_name": np.array(self.class_names, dtype=str),
            "map": self.maps,
            "forward": -g,
            "right": g,
            "meta_json": np.array(json.dumps(meta)),
        }
        if self.images is not None:
            arrays["image_rgb"] = self.images
            arrays["image_valid"] = self.image_valid
        np.savez_compressed(folder / "aligned.npz", **arrays)
        means = self.mean_by_class().arrays()
        means["meta_json"] = np.array(json.dumps(meta))
        np.savez_compressed(folder / "aligned_means.npz", **means)
        schema.write_rows(folder / "aligned.csv", schema.ALIGNED_COLUMNS, self.rows)
        return folder


def _orientations(source, events: Sequence[Mapping[str, Any]]) -> Dict[str, Orientation]:
    if isinstance(source, str) and source == "obb":
        items = obb_orientations(events)
    elif isinstance(source, (str, Path)):
        items = read_orientations(source)
    else:
        items = [o if isinstance(o, Orientation) else Orientation(
            str(o["event_id"]), float(o["cx"]), float(o["cy"]), float(o["angle_deg"]),
            float(o["length"]), bool(o.get("directed", True))) for o in source]
    out: Dict[str, Orientation] = {}
    for o in items:
        if o.event_id in out:
            raise ExplainError(ErrorCode.INVALID_CONFIG, f"event {o.event_id} has two orientations")
        out[o.event_id] = o
    return out


def align_maps(export_dir, orientations: Union[str, Path, Iterable[Any]], *, grid: int = 64,
               extent: float = 1.0, images: bool = True,
               out_dir: Optional[Union[str, Path]] = None) -> AlignedMaps:
    """The maps of an export folder on grids aligned to the experimenter's orientations.

    *orientations* is an orientation CSV (:func:`read_orientations`), a list
    of :class:`Orientation` (or dicts with its fields), or ``"obb"`` for the
    model's own rotated boxes (undirected).  Maps come from the export's
    ``maps/`` crops and pictures from its ``crops/``, so the run and the
    videos are not needed.  With *out_dir* the result is also saved there
    (:meth:`AlignedMaps.save`).
    """
    from yoru_explain.io.frames import imread

    export_dir = Path(export_dir)
    info = json.loads((export_dir / "export.json").read_text(encoding="utf-8"))
    events = schema.read_rows(export_dir / "events.csv")
    wanted = _orientations(orientations, events)
    known = {e["event_id"] for e in events}
    unknown = sorted(set(wanted) - known)
    if unknown:
        raise ExplainError(ErrorCode.INVALID_CONFIG,
                           f"orientations name events not in this export: {', '.join(unknown[:8])}"
                           + (" ..." if len(unknown) > 8 else ""))
    ids, units, classes, maps, pics, valid, rows = [], [], [], [], [], [], []
    for e in events:
        eid = e["event_id"]
        o = wanted.get(eid)
        row = {"event_id": eid, "unit_id": e["unit_id"], "class_name": e["class_name"]}
        if o is not None:
            row.update(cx=o.cx, cy=o.cy, angle_deg=o.angle_deg, length=o.length, directed=o.directed)
        if not e.get("map_file"):
            row["status"] = "not_analysed"
        elif o is None:
            row["status"] = "no_orientation"
        elif not o.valid():
            row["status"] = "invalid_orientation"
        else:
            with np.load(export_dir / e["map_file"], allow_pickle=False) as data:
                m, offset = data["map"], data["offset"]
            pic = None
            if images and e.get("crop_file"):
                rgb = imread(export_dir / e["crop_file"])[:, :, ::-1]
                pic = (rgb, (int(e["crop_x0"]), int(e["crop_y0"])))
            angle = math.radians(o.angle_deg)
            m_samples, p_samples, p_valid = [], [], []
            for reverse in ((False,) if o.directed else (False, True)):
                xs, ys = aligned_points(o.cx, o.cy, angle, o.length, grid, extent, reverse=reverse)
                m_samples.append(sample(m, xs, ys, offset)[0])
                if pic is not None:
                    p, inside = sample(pic[0], xs, ys, pic[1])
                    p_samples.append(p)
                    p_valid.append(inside)
            mean, _ = nanmean_stack(np.stack(m_samples))
            row.update(status="ok", stack_index=len(ids))
            ids.append(eid)
            units.append(e["unit_id"])
            classes.append(e["class_name"])
            maps.append(mean)
            if images:
                if p_samples:
                    # Undirected: the picture of the first direction (an average of two
                    # pictures would show neither); the map is the mean of both.
                    pics.append(p_samples[0])
                    valid.append(p_valid[0])
                else:
                    pics.append(np.zeros((grid, grid, 3), np.uint8))
                    valid.append(np.zeros((grid, grid), bool))
        rows.append(row)
    empty = np.zeros((0, grid, grid), np.float32)
    result = AlignedMaps(
        grid=int(grid), extent=float(extent), event_ids=ids, unit_ids=units, class_names=classes,
        maps=np.stack(maps) if maps else empty,
        images=(np.stack(pics) if pics else np.zeros((0, grid, grid, 3), np.uint8)) if images else None,
        image_valid=(np.stack(valid) if valid else np.zeros((0, grid, grid), bool)) if images else None,
        rows=rows,
        meta={
            "kind": "yoru-explain-aligned-maps",
            "export_run_id": info.get("run_id"),
            "map": info.get("map"),
            "grid_definition": "cell (r, c) samples the frame at centre + length * (a_r * forward + "
                               "b_c * right); a_r = forward[r] (row 0 most forward), b_c = right[c]; "
                               "forward = (cos angle, sin angle), right = (-sin angle, cos angle) "
                               "in frame pixels (y down); nearest pixel; undirected events: mean of "
                               "both directions",
            "orientations": "experimenter" if not (isinstance(orientations, str) and orientations == "obb")
                            else "model rotated boxes (undirected)",
            "interpretation": info.get("interpretation"),
        },
    )
    if out_dir is not None:
        result.save(out_dir)
    return result
