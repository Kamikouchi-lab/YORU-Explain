"""Which detections of a YORU CSV become events: first ones, or at random; in all or per class."""

from __future__ import annotations

from collections import Counter

import pytest

from yoru_explain.core.errors import ErrorCode, ExplainError
from yoru_explain.core.types import Detection
from yoru_explain.io.detections_csv import ImportedDetection, choose_detections


def _table():
    """100 frames: "walk" on every frame (two boxes on even frames), "sing" on every 10th."""
    by_frame = {}
    for f in range(100):
        dets = [Detection((0, 0, 10, 10), 0.9, 0, "walk")]
        if f % 2 == 0:
            dets.append(Detection((20, 0, 30, 10), 0.8, 0, "walk"))
        if f % 10 == 0:
            dets.append(Detection((40, 0, 50, 10), 0.7, 1, "sing"))
        by_frame[f] = [ImportedDetection(f, d) for d in dets]
    return by_frame


def _keys(chosen):
    return [(x.frame_index, x.detection.box) for x in chosen]


def test_first_takes_them_in_frame_order():
    chosen = choose_detections(_table(), class_names=["walk"], limit=5)
    assert _keys(chosen) == [(0, (0, 0, 10, 10)), (0, (20, 0, 30, 10)), (1, (0, 0, 10, 10)),
                             (2, (0, 0, 10, 10)), (2, (20, 0, 30, 10))]


def test_first_per_class_takes_the_first_of_each_class():
    chosen = choose_detections(_table(), per_class={"walk": 2, "sing": 3})
    assert [(x.frame_index, x.detection.class_name) for x in chosen] == [
        (0, "walk"), (0, "walk"), (0, "sing"), (10, "sing"), (20, "sing")]


def test_random_is_reproducible_with_its_seed_and_in_frame_order():
    a = choose_detections(_table(), limit=20, order="random", seed=7)
    b = choose_detections(_table(), limit=20, order="random", seed=7)
    c = choose_detections(_table(), limit=20, order="random", seed=8)
    assert _keys(a) == _keys(b) and _keys(a) != _keys(c)
    assert len(a) == 20
    assert [x.frame_index for x in a] == sorted(x.frame_index for x in a)


def test_random_spreads_over_frames_before_taking_a_second_box():
    chosen = choose_detections(_table(), class_names=["walk"], limit=100, order="random", seed=1)
    assert len({x.frame_index for x in chosen}) == 100          # one box on each of 100 frames
    more = choose_detections(_table(), class_names=["walk"], limit=120, order="random", seed=1)
    assert len(more) == 120 and len({x.frame_index for x in more}) == 100
    everything = choose_detections(_table(), class_names=["walk"], limit=1000, order="random", seed=1)
    assert len(everything) == 150                               # no more than there are


def test_random_per_class_gives_each_class_its_number():
    chosen = choose_detections(_table(), per_class={"walk": 12, "sing": 4}, order="random", seed=3)
    assert Counter(x.detection.class_name for x in chosen) == {"walk": 12, "sing": 4}
    walk = [x.frame_index for x in chosen if x.detection.class_name == "walk"]
    assert len(set(walk)) == 12                                 # on 12 different frames
    sing = choose_detections(_table(), per_class={"sing": 50}, order="random", seed=3)
    assert len(sing) == 10                                      # only 10 frames have it
    assert choose_detections(_table(), per_class={"walk": 0, "sing": 2}, order="random",
                             seed=3)[0].detection.class_name == "sing"


def test_random_keeps_to_the_chosen_frames_and_leaves_out_skipped_boxes():
    chosen = choose_detections(_table(), limit=3, frames=[4], order="random", seed=0)
    assert {x.frame_index for x in chosen} == {4} and len(chosen) == 2
    skip = lambda x: x.frame_index < 90                         # noqa: E731
    chosen = choose_detections(_table(), per_class={"walk": 50}, order="random", seed=0, skip=skip)
    assert all(x.frame_index >= 90 for x in chosen) and len(chosen) == 15


def test_contradicting_choices_are_refused():
    for kwargs in ({"per_class": {"walk": 1}, "limit": 3},
                   {"per_class": {"walk": 1}, "class_names": ["walk"]},
                   {"per_class": {"walk": -1}},
                   {"order": "shuffled"}):
        with pytest.raises(ExplainError) as info:
            choose_detections(_table(), **kwargs)
        assert info.value.code == ErrorCode.INVALID_CONFIG
