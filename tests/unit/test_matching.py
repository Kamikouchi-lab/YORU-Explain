"""Re-detection: every way of finding (or not finding) the original detection."""

from __future__ import annotations

from yoru_explain.analysis.matching import match_redetection
from yoru_explain.core.types import Detection

ORIG = (100.0, 100.0, 132.0, 132.0)


def d(box, cls=0, score=0.9):
    return Detection(tuple(float(v) for v in box), score, cls, f"c{cls}")


def test_one_overlapping_detection_of_the_class_is_a_match():
    m = match_redetection(ORIG, 0, [d((101, 100, 133, 132), score=0.7), d((300, 300, 332, 332))], 0.5)
    assert m.status == "matched" and m.detection.score == 0.7 and m.n_overlapping == 1


def test_two_overlapping_detections_of_the_class_are_ambiguous():
    m = match_redetection(ORIG, 0, [d((100, 100, 132, 132), score=0.5),
                                    d((104, 100, 136, 132), score=0.95)], 0.5)
    assert m.status == "ambiguous" and m.n_overlapping == 2
    assert m.detection.box == (100, 100, 132, 132)  # best IoU, not best score


def test_only_another_class_overlapping_is_a_class_change():
    m = match_redetection(ORIG, 0, [d(ORIG, cls=1)], 0.5)
    assert m.status == "class_changed" and m.detection.class_id == 1


def test_a_weak_overlap_is_unmatched():
    m = match_redetection(ORIG, 0, [d((116, 100, 148, 132))], 0.5)  # IoU 1/3
    assert m.status == "unmatched" and 0 < m.iou < 0.5


def test_nothing_overlapping_has_disappeared():
    assert match_redetection(ORIG, 0, [], 0.5).status == "disappeared"
    m = match_redetection(ORIG, 0, [d((0, 0, 10, 10))], 0.5)
    assert m.status == "disappeared" and m.detection is None and m.iou is None


def test_the_threshold_decides_between_matched_and_ambiguous():
    dets = [d(ORIG), d((116, 100, 148, 132))]
    assert match_redetection(ORIG, 0, dets, 0.5).status == "matched"
    assert match_redetection(ORIG, 0, dets, 0.3).status == "ambiguous"
