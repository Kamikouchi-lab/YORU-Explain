# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Detections saved by YORU (and YORU-Tracker), read as candidate events.

Three layouts are recognised by their header:

``yoru-analysis``
    YORU's video-analysis table: ``frame, x1, y1, x2, y2, x_center,
    y_center, w, h, angle, confidence, class, class_name[, tracking_id]``.
``yoru-tracker``
    YORU-Tracker's saved detections (``*_detections.csv``): ``frame_id``
    followed by YORU's ``DETECTION_COLUMNS``.
``yoru-tracker-tracks``
    YORU-Tracker's tracks (``*_tracks.csv``): YORU's ``DETECTION_COLUMNS``
    followed by ``frame_id, track_id, track_state, predicted, ...``.  A row
    the tracker did not predict is the detection its track matched, so it is
    read with its ``track_id``; a predicted row is the tracker's estimate of a
    lost track, not a detection, and is skipped.

YORU's real-time ``*_detect.csv`` is refused: its rows do not say which
video frame they belong to (the ``*_log.csv`` beside it does).  YORU-Tracker
reads the pair; its ``*_tracks.csv`` can be imported here.

These files do not record the model, thresholds, channel order or flips
they were made with, so an imported detection is only a pointer: before it
is analysed, the detector is run on the frame and the imported box must be
found again (same class, IoU >= 0.9); otherwise the event fails with
``target_not_reproduced``.
"""

from __future__ import annotations

import bisect
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from yoru_explain.core.errors import ErrorCode, ExplainError
from yoru_explain.core.types import Detection

_ANALYSIS_PREFIX = ("frame", "x1", "y1", "x2", "y2", "x_center", "y_center")
_TRACKER_PREFIX = ("frame_id", "x1", "y1", "x2", "y2", "confidence", "class", "class_name")
#: The start of YORU's ``DETECTION_COLUMNS``: the whole header of a real-time
#: ``*_detect.csv``, and the first columns of a YORU-Tracker tracks CSV.
_DETECTION_PREFIX = ("x1", "y1", "x2", "y2", "confidence", "class", "class_name", "total_time")
_TRACKS_COLUMNS = ("frame_id", "track_id", "predicted")


@dataclass(frozen=True)
class ImportedDetection:
    frame_index: int
    detection: Detection
    track_id: Optional[int] = None


#: Where each layout keeps the oriented box (YORU's ``obb_of``: the upright
#: box with angle 0 for models without rotation).
_OBB_COLUMNS = {
    "yoru-analysis": ("x_center", "y_center", "w", "h", "angle"),
    "yoru-tracker": ("cx", "cy", "w", "h", "angle"),
    "yoru-tracker-tracks": ("cx", "cy", "w", "h", "angle"),
}
#: Where each layout keeps the ID of the track a detection belongs to.
_TRACK_ID_COLUMN = {
    "yoru-analysis": "tracking_id",
    "yoru-tracker-tracks": "track_id",
}


def _rotated_box(line, col, layout):
    """The row's rotated box, or None when it is upright (angle 0) or absent."""
    names = _OBB_COLUMNS[layout]
    if not all(n in col for n in names):
        return None
    cells = [line[col[n]].strip() if col[n] < len(line) else "" for n in names]
    if any(c == "" for c in cells):
        return None
    obb = tuple(float(c) for c in cells)
    return obb if obb[4] != 0.0 else None


def load_detections_csv(path) -> Tuple[str, Dict[int, List[ImportedDetection]]]:
    """``(layout, {frame_index: [detections]})``."""
    path = Path(path)
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = tuple(h.strip() for h in next(reader, ()))
        if header[:len(_ANALYSIS_PREFIX)] == _ANALYSIS_PREFIX:
            layout, frame_col, conf_col = "yoru-analysis", "frame", "confidence"
        elif header[:len(_TRACKER_PREFIX)] == _TRACKER_PREFIX:
            layout, frame_col, conf_col = "yoru-tracker", "frame_id", "confidence"
        elif (header[:len(_DETECTION_PREFIX)] == _DETECTION_PREFIX
              and set(_TRACKS_COLUMNS) <= set(header)):
            layout, frame_col, conf_col = "yoru-tracker-tracks", "frame_id", "confidence"
        elif header[:len(_DETECTION_PREFIX)] == _DETECTION_PREFIX:
            raise ExplainError(
                ErrorCode.MISSING_DATA,
                f"{path.name}: a YORU real-time detections file. Its rows do not say which "
                f"video frame they belong to (the *_log.csv beside it does), so it is not "
                f"read here; track the pair with YORU-Tracker and import its *_tracks.csv")
        else:
            raise ExplainError(
                ErrorCode.MISSING_DATA,
                f"{path.name}: not a YORU detections table (header starts {', '.join(header[:4])})")
        col = {name: i for i, name in enumerate(header)}
        track_col = _TRACK_ID_COLUMN.get(layout)
        frames: Dict[int, List[ImportedDetection]] = {}
        for line in reader:
            if not line or not "".join(line).strip():
                continue
            if layout == "yoru-tracker" and not line[col["x1"]].strip():
                continue  # a frame on which nothing was detected
            if layout == "yoru-tracker-tracks" and line[col["predicted"]].strip() not in ("", "0"):
                continue  # a lost track's estimated position: there is no detection
            frame = int(float(line[col[frame_col]]))
            det = Detection(
                box=(float(line[col["x1"]]), float(line[col["y1"]]),
                     float(line[col["x2"]]), float(line[col["y2"]])),
                score=float(line[col[conf_col]]),
                class_id=int(float(line[col["class"]])),
                class_name=line[col["class_name"]],
                obb=_rotated_box(line, col, layout),
            )
            track = None
            if track_col in col and line[col[track_col]].strip():
                track = int(float(line[col[track_col]]))
            frames.setdefault(frame, []).append(ImportedDetection(frame, det, track))
    return layout, frames


#: How :func:`choose_detections` goes through a file.
ORDERS = ("first", "random")


class _Spacing:
    """Keeps chosen detections of one class at least *gap* frames apart.

    *taken* holds frames already used per class name (events picked
    earlier); they count as chosen.  With a gap of 0 everything is admitted.
    """

    def __init__(self, gap: int, taken: Optional[Mapping[str, Iterable[int]]] = None):
        self.gap = int(gap)
        self._frames: Dict[str, List[int]] = {
            str(k): sorted(int(f) for f in v) for k, v in (taken or {}).items()}

    def admit(self, frame: int, class_name: str) -> bool:
        """Whether a box of *class_name* on *frame* may be chosen; if so it is recorded."""
        if self.gap > 0:
            frames = self._frames.setdefault(class_name, [])
            i = bisect.bisect_left(frames, frame)
            if (i < len(frames) and frames[i] - frame < self.gap) or \
                    (i > 0 and frame - frames[i - 1] < self.gap):
                return False
            frames.insert(i, frame)
        return True


def _spread(items: list, n: int, rng: random.Random, spacing: Optional[_Spacing] = None) -> list:
    """*n* of *items* (``(frame, position, detection)``) at random, spread over frames.

    The frames are shuffled and one box is taken from each in turn (a box
    at random on a frame with several), so *n* boxes come from *n*
    different frames whenever the items cover that many; only then does a
    frame give a second box.  *spacing* passes over boxes too close in time
    to one already chosen of the same class.
    """
    by_frame: Dict[int, list] = {}
    for item in items:
        by_frame.setdefault(item[0], []).append(item)
    order = sorted(by_frame)
    rng.shuffle(order)
    for f in order:
        rng.shuffle(by_frame[f])
    out: list = []
    depth = 0
    while len(out) < n:
        layer = [by_frame[f][depth] for f in order if depth < len(by_frame[f])]
        if not layer:
            break
        for item in layer:
            if len(out) >= n:
                break
            if spacing is None or spacing.admit(item[0], item[2].detection.class_name):
                out.append(item)
        depth += 1
    return out


def choose_detections(by_frame: Dict[int, List[ImportedDetection]], *, class_names=None,
                      frames=None, limit: Optional[int] = None,
                      per_class: Optional[Dict[str, int]] = None, order: str = "first",
                      seed: Optional[int] = None, skip=None, min_gap_frames: int = 0,
                      taken: Optional[Mapping[str, Iterable[int]]] = None) -> List[ImportedDetection]:
    """The detections of a :func:`load_detections_csv` table to make events of.

    *class_names* keeps only those classes, *frames* only those frame
    indices, *skip* (``ImportedDetection -> bool``) leaves out the ones it
    returns True for.  How many: *limit* at most that many in all, or
    *per_class* ``{class_name: n}`` at most *n* of each of those classes
    (instead of *class_names* and *limit*).

    ``order="first"`` takes them from the first frame on, in the file's
    order.  ``order="random"`` draws them with ``random.Random(seed)``,
    spread over frames: *n* boxes of a class come from *n* different frames
    whenever the chosen frames hold that many (see :func:`_spread`).  The
    same file, choices and seed give the same detections.

    *min_gap_frames* keeps the chosen boxes of one class at least that many
    frames apart -- also from *taken* (``{class_name: frames}``, events
    picked before) -- so that a burst of near-identical neighbouring frames
    does not pass for many samples.  Fewer than asked come back when the
    video has too few that far apart.

    Returned in frame order (and the file's order on a frame).
    """
    if order not in ORDERS:
        raise ExplainError(ErrorCode.INVALID_CONFIG,
                           f"order must be one of {', '.join(ORDERS)}, not {order!r}")
    if int(min_gap_frames) < 0:
        raise ExplainError(ErrorCode.INVALID_CONFIG, "min_gap_frames is negative")
    spacing = _Spacing(min_gap_frames, taken) if int(min_gap_frames) > 0 else None
    if per_class is not None:
        if class_names is not None or limit is not None:
            raise ExplainError(ErrorCode.INVALID_CONFIG,
                               "give per_class, or class_names and limit, not both")
        if any(int(n) < 0 for n in per_class.values()):
            raise ExplainError(ErrorCode.INVALID_CONFIG, "a per_class count is negative")
        per_class = {str(k): int(n) for k, n in per_class.items() if int(n) > 0}
        keep_classes = set(per_class)
    else:
        keep_classes = set(class_names) if class_names is not None else None
    if limit is not None and int(limit) < 0:
        raise ExplainError(ErrorCode.INVALID_CONFIG, "limit is negative")
    keep_frames = set(int(f) for f in frames) if frames is not None else None
    items = []   # (frame, position on the frame, detection), in the file's order
    for frame_index in sorted(by_frame):
        if keep_frames is not None and frame_index not in keep_frames:
            continue
        for pos, imported in enumerate(by_frame[frame_index]):
            if keep_classes is not None and imported.detection.class_name not in keep_classes:
                continue
            if skip is not None and skip(imported):
                continue
            items.append((frame_index, pos, imported))

    def admit(x) -> bool:
        return spacing is None or spacing.admit(x[0], x[2].detection.class_name)

    if order == "random":
        rng = random.Random(seed)
        if per_class is not None:
            chosen = []
            for name in sorted(per_class):
                of_class = [x for x in items if x[2].detection.class_name == name]
                chosen += _spread(of_class, per_class[name], rng, spacing)
        else:
            chosen = _spread(items, len(items) if limit is None else int(limit), rng, spacing)
        chosen.sort(key=lambda x: (x[0], x[1]))
    elif per_class is not None:
        counts: Dict[str, int] = {}
        chosen = []
        for x in items:
            name = x[2].detection.class_name
            if counts.get(name, 0) < per_class[name] and admit(x):
                counts[name] = counts.get(name, 0) + 1
                chosen.append(x)
    else:
        n = len(items) if limit is None else int(limit)
        chosen = []
        for x in items:
            if len(chosen) >= n:
                break
            if admit(x):
                chosen.append(x)
    return [x[2] for x in chosen]
