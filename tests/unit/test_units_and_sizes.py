"""Patches sized to the animal, experimental units of videos, and events spaced in time."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from tests.unit.test_export import _index_of, _video
from yoru_explain import api
from yoru_explain.core.errors import ExplainError
from yoru_explain.core.types import Detection
from yoru_explain.io import schema
from yoru_explain.io.detections_csv import ImportedDetection, choose_detections
from yoru_explain.io.runs import RunReader
from yoru_explain.methods.occlusion import make_patches, region_of_interest

# -- patches relative to the target ------------------------------------------------------


def test_a_fraction_of_the_target_size_gives_whole_pixels():
    c = api.OcclusionConfig(patch_fraction=0.25)
    assert c.patch_geometry((0, 0, 100, 64)) == ((20, 20), (10, 10), 80.0)     # sqrt(100 * 64) = 80
    c = api.OcclusionConfig(patch_fraction=0.2, stride_fraction=0.05)
    # A rotated box: its own w x h, not the upright box around it.
    assert c.patch_geometry((0, 0, 200, 200), (50, 50, 120, 30, 0.3)) == ((12, 12), (3, 3), 60.0)
    fixed = api.OcclusionConfig(patch_size=(24, 16), stride=(12, 8))
    assert fixed.patch_geometry((0, 0, 100, 64))[:2] == ((24, 16), (12, 8))


def test_fractions_are_kept_in_the_config_and_pixel_configs_are_unchanged():
    c = api.OcclusionConfig(patch_fraction=0.25)
    d = c.to_dict()
    assert d["patch_fraction"] == 0.25 and d["stride_fraction"] is None
    assert api.OcclusionConfig.from_dict(d) == c
    # A pixel configuration writes the dict it always did: earlier cache keys still match.
    assert "patch_fraction" not in api.OcclusionConfig().to_dict()
    with pytest.raises(ValueError):
        api.OcclusionConfig(stride_fraction=0.1)
    with pytest.raises(ValueError):
        api.OcclusionConfig(patch_fraction=0.0)


def test_twice_the_magnification_tiles_the_box_alike():
    c = api.OcclusionConfig(patch_fraction=0.25)

    def in_box_units(box):
        (ps, st, _) = c.patch_geometry(box)
        roi = region_of_interest(box, "context", 3.0, 1000, 1000)
        cx, cy, w = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0]
        return [((p.box[0] - cx) / w, (p.box[1] - cy) / w, (p.box[2] - cx) / w, (p.box[3] - cy) / w)
                for p in make_patches(roi, ps, st, 1000, 1000)]

    small, large = in_box_units((100, 100, 180, 180)), in_box_units((200, 200, 360, 360))
    assert len(small) == len(large) == 23 * 23
    np.testing.assert_allclose(small, large)


@pytest.fixture
def one_video(tmp_path, toy):
    project = api.Project.create(tmp_path / "proj", "sizes")
    sid = project.add_source(_video(tmp_path / "clip.avi"))["source_id"]
    model = project.add_model(None, "toy-grid")
    ev = api.add_event(project, toy, model.model_id, sid, 3, _index_of(toy, project, sid, 3, 0))
    project.save()
    return project, toy, model, sid, ev


def test_a_run_records_the_pixels_a_fraction_gave(one_video):
    project, toy, model, sid, ev = one_video
    req = api.AnalysisRequest("occlusion", occlusion=api.OcclusionConfig(patch_fraction=0.5), use_cache=False)
    out = api.run_analysis(project, toy, model.model_id, [ev], req)
    (row,) = RunReader(out.run_dir).results
    assert row["status"] == "ok"
    assert (row["patch_height_px"], row["patch_width_px"], row["stride_y_px"], row["stride_x_px"]) == \
        ("16", "16", "8", "8")                               # the toy box is 32 x 32
    assert float(row["target_size_px"]) == 32.0
    box = tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2"))
    roi = region_of_interest(box, "context", 3.0, 320, 192)
    assert int(row["n_perturbations"]) == len(make_patches(roi, (16, 16), (8, 8), 320, 192))
    data = RunReader(out.run_dir).load_map(row["map_file"])
    assert data["patch_size_px"].tolist() == [16, 16] and data["stride_px"].tolist() == [8, 8]
    assert data["meta"]["config"]["patch_fraction"] == 0.5


def test_a_target_too_small_for_the_fraction_fails_with_a_code(one_video):
    project, toy, model, sid, ev = one_video
    req = api.AnalysisRequest("occlusion", occlusion=api.OcclusionConfig(patch_fraction=0.03), use_cache=False)
    out = api.run_analysis(project, toy, model.model_id, [ev], req)
    (row,) = RunReader(out.run_dir).results
    assert row["status"] == "failed" and row["error_code"] == "invalid_config"
    assert row["delta_score"] == "" and row["map_file"] == ""


# -- experimental units -------------------------------------------------------------------


def test_a_source_is_its_own_unit_until_one_is_set_and_the_unit_is_kept(one_video, tmp_path):
    project, toy, model, sid, ev = one_video
    assert project.unit_of(sid) == (sid, False)
    project.set_unit(sid, "  fly-07 ")
    assert project.unit_of(sid) == ("fly-07", True)
    project.save()
    again = api.Project.open(project.root)
    assert again.unit_of(sid) == ("fly-07", True)
    with pytest.raises(ExplainError):
        again.set_unit(sid, "a\tb")
    # A run records it with its inputs; a removed source keeps it for the runs that used it.
    req = api.AnalysisRequest("occlusion", occlusion=api.OcclusionConfig(patch_size=32, stride=32),
                              use_cache=False)
    out = api.run_analysis(again, toy, model.model_id, [ev], req)
    assert RunReader(out.run_dir).manifest["inputs"][0]["unit_id"] == "fly-07"
    again.remove_sources([sid])
    assert again.unit_of(sid) == ("fly-07", True)
    p2 = api.Project.open(project.root)
    p2.set_unit(sid, None)
    assert p2.unit_of(sid) == (sid, False)


def test_units_are_read_from_a_table_by_id_path_or_file_name(tmp_path):
    project = api.Project.create(tmp_path / "proj", "units")
    (tmp_path / "day1").mkdir()
    (tmp_path / "day2").mkdir()
    a = project.add_source(_video(tmp_path / "day1" / "arena.avi"))["source_id"]
    b = project.add_source(_video(tmp_path / "day2" / "arena.avi"))["source_id"]
    c = project.add_source(_video(tmp_path / "solo.avi"))["source_id"]
    table = tmp_path / "units.csv"
    with open(table, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "unit_id", "condition"])
        w.writerow([a, "pair-1", "ctrl"])
        w.writerow([str(tmp_path / "day2" / "arena.avi"), "pair-2", "test"])
        w.writerow(["solo.avi", "fly-9", "ctrl"])
        w.writerow(["arena.avi", "pair-x", ""])      # two sources have this name: not guessed
        w.writerow(["missing.avi", "fly-0", ""])
    report = project.assign_units_from_csv(table)
    assert report["assigned"] == {a: "pair-1", b: "pair-2", c: "fly-9"}
    assert report["ambiguous"] == ["arena.avi"] and report["unmatched"] == ["missing.avi"]
    assert report["ignored_columns"] == ["condition"]
    assert [project.unit_of(s)[0] for s in (a, b, c)] == ["pair-1", "pair-2", "fly-9"]
    bad = tmp_path / "bad.csv"
    bad.write_text("file,animal\nx,y\n", encoding="utf-8")
    with pytest.raises(ExplainError):
        project.assign_units_from_csv(bad)


# -- events spaced in time ------------------------------------------------------------------


def _every_frame(n=100, classes=("a",)):
    return {f: [ImportedDetection(f, Detection((0, 0, 10, 10), 0.9, i, name))
                for i, name in enumerate(classes)] for f in range(n)}


def test_chosen_boxes_of_a_class_keep_the_gap():
    by_frame = _every_frame(100, ("a", "b"))
    first = choose_detections(by_frame, per_class={"a": 50}, min_gap_frames=10)
    assert [d.frame_index for d in first] == list(range(0, 100, 10))
    taken = choose_detections(by_frame, per_class={"a": 50}, min_gap_frames=10, taken={"a": [5]})
    assert [d.frame_index for d in taken] == list(range(15, 100, 10))
    for seed in range(5):
        drawn = choose_detections(by_frame, per_class={"a": 50, "b": 3}, order="random", seed=seed,
                                  min_gap_frames=10)
        frames_a = sorted(d.frame_index for d in drawn if d.detection.class_name == "a")
        assert all(y - x >= 10 for x, y in zip(frames_a, frames_a[1:])) and len(frames_a) <= 10
        # The gap is per class: b's boxes may share frames with a's.
        assert sum(1 for d in drawn if d.detection.class_name == "b") == 3
    # Without a gap nothing changes.
    assert len(choose_detections(by_frame, per_class={"a": 50}, order="random", seed=1)) == 50


def test_a_gap_in_seconds_uses_the_frame_rate():
    assert api.min_gap_frames({"kind": "video", "fps": 25.0}, 1.0) == 25
    assert api.min_gap_frames({"kind": "video", "fps": 25.0}, 0.5) == 13      # rounded up
    assert api.min_gap_frames({"kind": "image", "fps": 0.0}, 3.0) == 0
    with pytest.raises(ExplainError):
        api.min_gap_frames({"kind": "video", "fps": 0.0, "source_id": "src-001"}, 1.0)


def test_events_imported_with_a_gap_are_that_far_apart_also_from_earlier_ones(one_video, tmp_path):
    project, toy, model, sid, ev = one_video
    path = tmp_path / "clip.csv"   # YORU's video-analysis layout: the red square on every frame
    with project.open_source(sid) as r, open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "x1", "y1", "x2", "y2", "x_center", "y_center", "w", "h", "angle",
                    "confidence", "class", "class_name"])
        for i in range(40):
            d = next(d for d in toy.detect(r.read(i).image) if d.class_id == 0)
            x1, y1, x2, y2 = d.box
            w.writerow([i, x1, y1, x2, y2, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, 0,
                        round(d.score, 4), d.class_id, d.class_name])
    # ev (frame 3, red) exists already; 0.4 s at 25 fps is 10 frames.
    added = api.add_events_from_detections_csv(project, sid, path, per_class={"red": 40}, min_gap_s=0.4)
    frames = sorted(int(project.event(e)["frame_index"]) for e in added)
    assert frames == [13, 23, 33]
    all_red = sorted(int(e["frame_index"]) for e in project.events if e["class_name"] == "red")
    assert all(y - x >= 10 for x, y in zip(all_red, all_red[1:]))
    rows = schema.read_rows(path)
    assert len(rows) == 40
