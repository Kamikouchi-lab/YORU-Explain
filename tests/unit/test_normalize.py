"""Box and aligned grids, nearest-pixel sampling, and means that weigh units, by hand."""

from __future__ import annotations

import math

import numpy as np
import pytest

from yoru_explain.analysis.normalize import (
    Orientation,
    aligned_points,
    box_points,
    box_region,
    grid_centres,
    nanmean_stack,
    obb_orientations,
    read_orientations,
    sample,
    unit_class_means,
)
from yoru_explain.core.errors import ExplainError


def test_grid_centres_are_cell_centres_and_exactly_antisymmetric():
    np.testing.assert_allclose(grid_centres(4, 0.5), [-0.375, -0.125, 0.125, 0.375])
    for n, e in ((7, 1.0), (12, 1.0), (64, 1.5), (33, 0.7)):
        g = grid_centres(n, e)
        assert (g[::-1] == -g).all()
        assert g[0] == pytest.approx(-e + e / n)


def test_box_points_scale_by_the_box_width_and_height():
    xs, ys = box_points((10, 20, 50, 100), 4, 0.5)
    np.testing.assert_allclose(xs[0], [15, 25, 35, 45])          # cx 30, w 40
    np.testing.assert_allclose(ys[:, 0], [30, 50, 70, 90])       # cy 60, h 80
    assert box_region((10, 20, 50, 100), 1.5, 64, 200) == (0, 0, 64, 180)


def test_sampling_takes_the_pixel_the_point_falls_in_and_nan_off_the_map():
    m = np.arange(12, dtype=np.float32).reshape(3, 4)
    m[1, 2] = np.nan
    xs = np.array([[0.0, 0.999, 3.5, 4.0, 2.2]])
    ys = np.array([[0.0, 2.999, 0.5, 0.0, 1.0]])
    out, inside = sample(m, xs, ys)
    assert out[0, 0] == 0 and out[0, 1] == 8 and out[0, 2] == 3
    assert np.isnan(out[0, 3]) and not inside[0, 3]                 # x = 4 is off a 4-wide map
    assert np.isnan(out[0, 4]) and inside[0, 4]                     # on the map, but not measured
    # A crop: its pixel (0, 0) is frame pixel (10, 20).
    out, _ = sample(m, np.array([10.5, 13.5]), np.array([20.5, 22.5]), offset=(10, 20))
    assert out.tolist() == [0.0, 11.0]
    pic = np.full((3, 4, 3), 7, np.uint8)
    out, inside = sample(pic, np.array([[1.0, -1.0]]), np.array([[1.0, 1.0]]))
    assert out.dtype == np.uint8 and out[0, 0].tolist() == [7, 7, 7] and out[0, 1].tolist() == [0, 0, 0]


def test_the_forward_direction_points_up_and_the_right_side_is_on_the_right():
    n = 5
    # Facing right on screen (0 deg): up in the grid is +x in the frame, right is +y (down).
    xs, ys = aligned_points(100.0, 50.0, 0.0, 10.0, n, 1.0)
    assert xs[0, 2] > 100 and ys[0, 2] == pytest.approx(50)        # top centre: ahead of the animal
    assert ys[2, 4] > 50 and xs[2, 4] == pytest.approx(100)        # right centre: on its right
    # Facing up on screen (-90 deg): the grid is the frame's own orientation.
    xs, ys = aligned_points(100.0, 50.0, -math.pi / 2, 10.0, n, 1.0)
    g = grid_centres(n, 1.0) * 10
    np.testing.assert_array_equal(xs, np.broadcast_to(100 + g[None, :], (n, n)))
    np.testing.assert_array_equal(ys, np.broadcast_to(50 + g[:, None], (n, n)))
    # reverse turns it by exactly 180 degrees: cell (r, c) looks where cell (n-1-r, n-1-c) did.
    xs1, ys1 = aligned_points(37.3, 21.9, 0.6, 13.0, 8, 1.2)
    xs2, ys2 = aligned_points(37.3, 21.9, 0.6, 13.0, 8, 1.2, reverse=True)
    np.testing.assert_array_equal(xs2, xs1[::-1, ::-1])
    np.testing.assert_array_equal(ys2, ys1[::-1, ::-1])


@pytest.mark.parametrize("angle", [0.0, 90.0, 180.0, -37.0, 123.0])
def test_a_spot_at_the_head_lands_at_the_top_whatever_the_heading(angle):
    """A map that is high only near the head, drawn for any heading, aligns to the same place."""
    h, w = 200, 200
    cx, cy, length = 100.0, 100.0, 40.0
    t = math.radians(angle)
    hx, hy = cx + 0.4 * length * math.cos(t), cy + 0.4 * length * math.sin(t)
    yy, xx = np.mgrid[0:h, 0:w]
    m = np.where((xx + 0.5 - hx) ** 2 + (yy + 0.5 - hy) ** 2 < 36, 1.0, 0.0).astype(np.float32)
    xs, ys = aligned_points(cx, cy, t, length, 21, 1.0)
    out, _ = sample(m, xs, ys)
    rows, cols = np.mgrid[0:21, 0:21]
    r, c = (rows * out).sum() / out.sum(), (cols * out).sum() / out.sum()
    # 0.4 lengths ahead is row (1 - 0.4) / (2 / 21) - 0.5 = 5.8; the centre column is 10.
    assert r == pytest.approx(5.8, abs=0.6) and c == pytest.approx(10, abs=0.6)
    assert out[15:, :].max() == 0                               # nothing behind


def test_means_weigh_each_unit_once_whatever_its_number_of_events():
    stack = np.array([[[1.0]], [[1.0]], [[1.0]], [[4.0]], [[np.nan]]], np.float32)
    units = ["a", "a", "a", "b", "b"]
    mm = unit_class_means(stack, ["x"] * 5, units)
    assert mm.class_names == ["x"] and mm.unit_keys == [("x", "a"), ("x", "b")]
    assert mm.mean_by_unit[:, 0, 0].tolist() == [1.0, 4.0]
    assert mm.n_events_by_unit[:, 0, 0].tolist() == [3, 1]     # the NaN event did not count
    assert mm.mean_by_class[0, 0, 0] == pytest.approx(2.5)     # not (1+1+1+4)/4
    assert mm.n_units_by_class[0, 0, 0] == 2
    mean, count = nanmean_stack(np.full((3, 2), np.nan))
    assert np.isnan(mean).all() and (count == 0).all()          # undefined stays undefined, never 0


def _csv(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_an_orientation_table_is_read_strictly_but_keeps_your_columns(tmp_path):
    rows = read_orientations(_csv(tmp_path / "o.csv",
                                  "event_id,cx,cy,angle_deg,length,directed,my_note\n"
                                  "ev-0001,10,20,90,30,,walking\n"
                                  "ev-0002,10,20,,30,false,lost\n"))
    assert rows[0] == Orientation("ev-0001", 10.0, 20.0, 90.0, 30.0, True) and rows[0].valid()
    assert not rows[1].directed and not rows[1].valid()
    with pytest.raises(ExplainError):
        read_orientations(_csv(tmp_path / "a.csv", "event_id,cx,cy,length\nev-0001,1,2,3\n"))
    with pytest.raises(ExplainError):
        read_orientations(_csv(tmp_path / "b.csv", "event_id,cx,cy,angle_deg,length,directed\n"
                                                   "ev-0001,1,2,3,4,maybe\n"))


def test_model_rotated_boxes_give_undirected_axes_along_the_longer_side():
    rows = [{"event_id": "e1", "obb_cx": "5", "obb_cy": "6", "obb_w": "10", "obb_h": "40", "obb_angle": "0"},
            {"event_id": "e2", "obb_cx": "", "obb_cy": "", "obb_w": "", "obb_h": "", "obb_angle": ""}]
    (o,) = obb_orientations(rows)
    assert o.event_id == "e1" and o.length == 40 and o.angle_deg == pytest.approx(90) and not o.directed
