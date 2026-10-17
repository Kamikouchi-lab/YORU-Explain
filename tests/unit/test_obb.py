"""Rotated boxes (OBB models): YORU's convention, the exact IoU, matching, the files."""

from __future__ import annotations

import math

import numpy as np
import pytest

from yoru_explain.analysis.matching import match_redetection
from yoru_explain.core.geometry import (
    obb_aabb,
    obb_corners,
    obb_iou,
    overlap,
    point_in_obb,
    region_mask,
)
from yoru_explain.core.types import Detection, ScoreStatus

OBBS = [(10, 20, 30, 8, 0.0), (100.5, 50.25, 40, 12, 0.7), (5, 5, 3, 9, -1.2),
        (0, 0, 10, 4, math.pi / 2), (64, 64, 50, 7, 1.4)]


@pytest.mark.parametrize("obb", OBBS)
def test_corners_are_yorus(obb):
    from yoru.libs.obb import obb_corners as yoru_corners   # documented, pure Python

    np.testing.assert_allclose(obb_corners(obb), np.array(yoru_corners(obb)), atol=1e-12)


@pytest.mark.parametrize("obb", OBBS)
def test_ultralytics_xywhr_is_the_same_rectangle(obb):
    torch = pytest.importorskip("torch")
    ops = pytest.importorskip("ultralytics.utils.ops")

    theirs = ops.xywhr2xyxyxyxy(torch.tensor([obb], dtype=torch.float64))[0].numpy()
    mine = obb_corners(obb)
    key = lambda p: (round(p[0], 6), round(p[1], 6))  # noqa: E731
    assert sorted(map(tuple, theirs), key=key) == pytest.approx(sorted(map(tuple, mine), key=key))
    assert obb_aabb(obb) == pytest.approx((theirs[:, 0].min(), theirs[:, 1].min(),
                                           theirs[:, 0].max(), theirs[:, 1].max()))


def test_obb_iou_known_values():
    a = (0, 0, 10, 10, 0.0)
    assert obb_iou(a, a) == pytest.approx(1.0)
    assert obb_iou(a, (20, 0, 10, 10, 0.0)) == 0.0
    assert obb_iou(a, (5, 0, 10, 10, 0.0)) == pytest.approx(1 / 3)           # as upright boxes
    # A square and the same square turned by 45 degrees: a regular octagon.
    assert obb_iou(a, (0, 0, 10, 10, math.pi / 4)) == pytest.approx(1 / math.sqrt(2), rel=1e-9)
    b = (3, 4, 20, 6, 0.3)
    assert obb_iou(b, (3, 4, 20, 6, 0.3 + math.pi)) == pytest.approx(1.0)     # angle mod pi
    assert obb_iou(b, (3, 4, 6, 20, 0.3 + math.pi / 2)) == pytest.approx(1.0)  # w, h swapped
    assert obb_iou(b, (3, 4, 0, 6, 0.3)) == 0.0                              # empty box


def test_obb_iou_agrees_with_a_raster_count():
    rng = np.random.default_rng(0)
    scale = 10.0                                   # 0.1 px raster
    for _ in range(15):
        a = (rng.uniform(40, 60), rng.uniform(40, 60), rng.uniform(5, 40), rng.uniform(5, 40),
             rng.uniform(-math.pi, math.pi))
        b = (a[0] + rng.uniform(-12, 12), a[1] + rng.uniform(-12, 12), rng.uniform(5, 40),
             rng.uniform(5, 40), rng.uniform(-math.pi, math.pi))
        up = lambda o: (o[0] * scale, o[1] * scale, o[2] * scale, o[3] * scale, o[4])  # noqa: E731
        ma = region_mask((1000, 1000), None, up(a))
        mb = region_mask((1000, 1000), None, up(b))
        est = (ma & mb).sum() / max(1, (ma | mb).sum())
        assert obb_iou(a, b) == pytest.approx(est, abs=0.01)
        assert obb_iou(a, b) == pytest.approx(obb_iou(b, a), abs=1e-12)


def _rod(angle):
    """A thin 60 x 6 rod centred at (50, 50)."""
    return (50.0, 50.0, 60.0, 6.0, angle)


def test_crossing_rods_overlap_as_boxes_but_not_as_rods():
    a, b = _rod(math.pi / 4), _rod(-math.pi / 4)
    box_a, box_b = obb_aabb(a), obb_aabb(b)
    assert overlap(box_a, box_b) > 0.9                     # upright boxes: nearly the same
    assert overlap(box_a, box_b, a, b) < 0.1               # rotated boxes: a small cross
    assert overlap(box_a, box_b, a, None) == overlap(box_a, box_b)   # both needed


def test_re_detection_matches_rotated_boxes():
    a, b = _rod(math.pi / 4), _rod(-math.pi / 4)
    other = Detection(obb_aabb(b), 0.9, 0, "rod", obb=b)
    same = Detection(obb_aabb(a), 0.8, 0, "rod", obb=a)
    assert match_redetection(obb_aabb(a), 0, [other], 0.5, a).status == ScoreStatus.UNMATCHED.value
    assert match_redetection(obb_aabb(a), 0, [other], 0.5).status == ScoreStatus.MATCHED.value
    m = match_redetection(obb_aabb(a), 0, [other, same], 0.5, a)
    assert m.status == ScoreStatus.MATCHED.value and m.detection is same


def test_the_target_region_follows_the_rotated_box():
    obb = (40.0, 30.0, 50.0, 10.0, 0.6)
    mask = region_mask((64, 96), None, obb)
    for y in range(64):
        for x in range(96):
            assert mask[y, x] == point_in_obb(x + 0.5, y + 0.5, obb)
    upright = region_mask((64, 96), obb_aabb(obb))
    assert mask.sum() < upright.sum() and not (mask & ~upright).any()


def test_rotated_boxes_survive_events_and_tables(tmp_path):
    from yoru_explain import api
    from yoru_explain.io import schema
    from yoru_explain.io.project import Project
    from yoru_explain.io.runs import target_columns

    image = np.zeros((96, 128, 3), np.uint8)
    path = tmp_path / "frame.png"
    import cv2

    cv2.imwrite(str(path), image)
    project = Project.create(tmp_path / "proj")
    s = project.add_source(path)
    obb = (60.0, 40.0, 50.0, 12.0, 0.4)
    det = Detection(obb_aabb(obb), 0.75, 1, "rod", obb=obb)
    t = api.target_from_detection(det, 0, image, source_id=s["source_id"], path=s["path"])
    eid = project.add_event(t)
    project.save()
    back = Project.open(project.root).target(eid)
    assert back.obb == pytest.approx(obb) and back.box == pytest.approx(det.box)
    row = target_columns(back, s)
    assert (row["obb_cx"], row["obb_angle"]) == (60.0, 0.4)
    assert set(schema.OBB_COLUMNS) <= set(schema.EVENT_COLUMNS) and set(schema.OBB_COLUMNS) <= set(
        schema.RESULT_COLUMNS)
    # A v0.1 row has no such columns: no rotated box.
    assert schema.obb_from_row({"x1": "1"}) is None
    assert schema.obb_from_row(schema.obb_columns(None)) is None


def test_yoru_csvs_bring_their_rotated_boxes(tmp_path):
    from yoru_explain.io.detections_csv import load_detections_csv

    path = tmp_path / "video.csv"
    path.write_text(
        "frame,x1,y1,x2,y2,x_center,y_center,w,h,angle,confidence,class,class_name\n"
        "3,10,20,50,40,30,30,40,20,0.0,0.9,0,fly\n"
        "3,5,5,60,60,32.5,32.5,70,12,0.785,0.8,1,wing\n", encoding="utf-8")
    _, frames = load_detections_csv(path)
    upright, rotated = (x.detection for x in frames[3])
    assert upright.obb is None
    assert rotated.obb == (32.5, 32.5, 70.0, 12.0, 0.785)
