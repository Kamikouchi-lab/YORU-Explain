"""Export folders: the run's numbers, unchanged, in forms an experimenter can analyse further."""

from __future__ import annotations

import csv
import json
import math

import cv2
import numpy as np
import pytest

from tests import synthetic as syn
from yoru_explain import api
from yoru_explain.analysis.normalize import box_points, grid_centres, sample
from yoru_explain.core.errors import ExplainError
from yoru_explain.io import schema
from yoru_explain.io.runs import RunReader

N_FRAMES = 40
REQ = api.AnalysisRequest("occlusion", occlusion=api.OcclusionConfig(patch_size=16, stride=8),
                          use_cache=False)


def _video(path, shift=0):
    h, w = 192, 320
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (w, h))
    for i in range(N_FRAMES):
        f = syn.blank(h, w)
        syn.paint(f, syn.window(5, 5 + shift + i // 10), syn.RED)
        syn.paint(f, syn.window(8, 14), syn.GREEN)
        out.write(f)
    out.release()
    return path


def _index_of(adapter, project, source_id, frame, cls):
    with project.open_source(source_id) as r:
        dets = adapter.detect(r.read(frame).image)
    return next(i for i, d in enumerate(dets) if d.class_id == cls)


@pytest.fixture
def study(tmp_path, toy):
    """Two videos -- two animals -- with three and two events, and one event that fails."""
    project = api.Project.create(tmp_path / "proj", "study")
    a = project.add_source(_video(tmp_path / "fly_a.avi"))["source_id"]
    b = project.add_source(_video(tmp_path / "fly_b.avi", shift=2))["source_id"]
    model = project.add_model(None, "toy-grid")
    events = []
    for sid, frame, cls in ((a, 3, 0), (a, 25, 1), (a, 31, 0), (b, 5, 0), (b, 22, 1)):
        events.append(api.add_event(project, toy, model.model_id, sid, frame,
                                    _index_of(toy, project, sid, frame, cls)))
    project.set_unit(a, "fly-A")
    project.set_unit(b, "fly-B")
    project.save()
    # One event whose box is not on its frame: the run reports it, the export keeps the row.
    rows = schema.read_rows(project.root / "events.csv")
    rows[2].update(x1="0", y1="0", x2="20", y2="20", candidate_index="", candidate_locator="")
    schema.write_rows(project.root / "events.csv", schema.EVENT_COLUMNS, rows)
    project = api.Project.open(project.root)
    out = api.run_analysis(project, toy, model.model_id, events, REQ)
    return project, toy, model, events, out


def _export(project, out, **kw):
    return api.export_run(project, out.run_dir, options=api.ExportOptions(**kw))


def test_the_export_holds_every_event_with_its_unit_and_status(study):
    project, toy, model, events, out = study
    ex = _export(project, out)
    d = ex.export_dir
    assert d == project.root / "exports" / out.run_id
    for name in ("export.json", "README.txt", "events.csv", "patches.csv", "summary.csv", "box_maps.npz",
                 "mean_maps.npz", "gallery/gallery.csv"):
        assert (d / name).is_file(), name
    info = json.loads((d / "export.json").read_text(encoding="utf-8"))
    assert info["kind"] == schema.EXPORT_KIND and info["run_id"] == out.run_id
    assert info["units"] == {"src-001": {"unit_id": "fly-A", "assigned": True},
                             "src-002": {"unit_id": "fly-B", "assigned": True}}
    rows = schema.read_rows(d / "events.csv")
    assert [r["event_id"] for r in rows] == [r["event_id"] for r in RunReader(out.run_dir).results]
    assert all(r["schema_version"] == schema.SCHEMA_VERSION for r in rows)
    by_id = {r["event_id"]: r for r in rows}
    failed = by_id[events[2]]
    assert failed["status"] == "failed" and failed["error_code"] == "target_not_reproduced"
    assert failed["map_file"] == "" and failed["stack_index"] == "" and failed["unit_id"] == "fly-A"
    ok = [r for r in rows if r["status"] == "ok"]
    assert len(ok) == 4 and ex.counts == {"events": 5, "ok": 4, "failed": 1, "crops": 4}
    assert sorted(int(r["stack_index"]) for r in ok) == [0, 1, 2, 3]
    assert {r["unit_id"] for r in ok} == {"fly-A", "fly-B"}
    assert all(r["patch_height_px"] == "16" and r["stride_x_px"] == "8" for r in ok)
    assert all(0.0 <= float(r["in_box_positive_share"]) <= 1.0 for r in ok)


def test_map_crops_are_the_run_maps_and_frame_crops_the_frames(study):
    project, toy, model, events, out = study
    d = _export(project, out).export_dir
    run = RunReader(out.run_dir)
    for r in schema.read_rows(d / "events.csv"):
        if r["status"] != "ok":
            continue
        full = run.load_map(r["run_map_file"])
        with np.load(d / r["map_file"], allow_pickle=False) as crop:
            x0, y0 = (int(v) for v in crop["offset"])
            h, w = crop["map"].shape
            np.testing.assert_array_equal(crop["map"], full["delta_map"][y0:y0 + h, x0:x0 + w])
            np.testing.assert_array_equal(crop["coverage"], full["coverage"][y0:y0 + h, x0:x0 + w])
            assert (x0, y0, x0 + w, y0 + h) == tuple(int(v) for v in full["roi"])
        # Outside the crop the run's map has no value: nothing is lost.
        outside = np.ones(full["delta_map"].shape, bool)
        outside[y0:y0 + h, x0:x0 + w] = False
        assert not np.isfinite(full["delta_map"][outside]).any()
        with project.open_source(r["source_id"]) as reader:
            frame = reader.read(int(r["frame_index"]))
        picture = cv2.imread(str(d / r["crop_file"]))
        np.testing.assert_array_equal(picture, frame.image[y0:y0 + h, x0:x0 + w])
        assert r["crop_status"] == "ok"


def test_box_maps_sample_the_frame_maps_at_the_grid(study):
    project, toy, model, events, out = study
    d = _export(project, out, grid=16).export_dir
    run = RunReader(out.run_dir)
    rows = {r["event_id"]: r for r in schema.read_rows(d / "events.csv")}
    with np.load(d / "box_maps.npz", allow_pickle=False) as z:
        assert z["map"].shape == (4, 16, 16) and z["image_rgb"].shape == (4, 16, 16, 3)
        extent = json.loads(str(z["meta_json"]))["extent"]
        assert extent == 1.5                     # context x3: the grid covers what was analysed
        np.testing.assert_allclose(z["u"], grid_centres(16, 1.5))
        for k, eid in enumerate(z["event_id"]):
            r = rows[str(eid)]
            assert int(r["stack_index"]) == k and z["unit_id"][k] == r["unit_id"]
            m = run.load_map(r["run_map_file"])["delta_map"]
            box = tuple(float(r[c]) for c in ("x1", "y1", "x2", "y2"))
            xs, ys = box_points(box, 16, extent)
            np.testing.assert_array_equal(z["map"][k], sample(m, xs, ys)[0])
            # Cell (r, c) is the pixel holding (cx + u[c] w, cy + v[r] h).
            cx, w = (box[0] + box[2]) / 2, box[2] - box[0]
            cy, h = (box[1] + box[3]) / 2, box[3] - box[1]
            assert z["map"][k][3, 12] == m[int(math.floor(cy + z["v"][3] * h)),
                                            int(math.floor(cx + z["u"][12] * w))] or \
                np.isnan(z["map"][k][3, 12])


def test_patches_carry_frame_and_box_coordinates(study):
    project, toy, model, events, out = study
    d = _export(project, out).export_dir
    rows = {r["event_id"]: r for r in schema.read_rows(d / "events.csv")}
    patches = schema.read_rows(d / "patches.csv")
    run = RunReader(out.run_dir)
    assert {p["event_id"] for p in patches} == {e for e, r in rows.items() if r["status"] == "ok"}
    for eid in {p["event_id"] for p in patches}:
        mine = [p for p in patches if p["event_id"] == eid]
        assert len(mine) == len(run.perturbations(eid))
        r = rows[eid]
        x1, y1, x2, y2 = (float(r[c]) for c in ("x1", "y1", "x2", "y2"))
        for p in mine:
            assert p["unit_id"] == r["unit_id"] and p["class_name"] == r["class_name"]
            if p["kind"] == "noop":
                assert p["u0"] == ""
                continue
            assert float(p["u0"]) == pytest.approx((float(p["x0"]) - (x1 + x2) / 2) / (x2 - x1))
            assert float(p["v1"]) == pytest.approx((float(p["y1"]) - (y1 + y2) / 2) / (y2 - y1))


def test_class_means_weigh_units_not_events(study):
    project, toy, model, events, out = study
    d = _export(project, out, grid=8).export_dir
    with np.load(d / "box_maps.npz", allow_pickle=False) as z:
        stack, units, classes = z["map"], list(z["unit_id"]), list(z["class_name"])
    with np.load(d / "mean_maps.npz", allow_pickle=False) as mm:
        names = list(mm["class_names"])
        for ci, cls in enumerate(names):
            unit_means = []
            for unit in sorted(set(u for u, c in zip(units, classes) if c == cls)):
                rows = [k for k, (u, c) in enumerate(zip(units, classes)) if u == unit and c == cls]
                unit_means.append(np.nanmean(stack[rows], axis=0))
            expected = np.nanmean(np.stack(unit_means), axis=0)
            np.testing.assert_allclose(mm["mean_by_class"][ci], expected, rtol=1e-6)
            assert mm["n_units_by_class"][ci].max() == len(unit_means)
    summary = schema.read_rows(d / "summary.csv")
    red = [s for s in summary if s["class_name"] == "red"]
    by_unit = {s["unit_id"]: s for s in red if s["level"] == "unit"}
    cls_row = next(s for s in red if s["level"] == "class")
    # fly-A has one finished red event (one failed), fly-B one.
    assert by_unit["fly-A"]["n_events"] == "2" and by_unit["fly-A"]["n_failed"] == "1"
    assert cls_row["n_units"] == "2" and cls_row["n_failed"] == "1"
    unit_values = [float(by_unit[u]["in_box_mean_delta_mean"]) for u in ("fly-A", "fly-B")]
    assert float(cls_row["in_box_mean_delta_mean"]) == pytest.approx(np.mean(unit_values))


def test_a_source_without_a_unit_is_its_own_unit(study):
    project, toy, model, events, out = study
    project.set_unit("src-002", None)
    d = _export(project, out, gallery=False).export_dir
    rows = [r for r in schema.read_rows(d / "events.csv") if r["source_id"] == "src-002"]
    assert {(r["unit_id"], r["unit_assigned"]) for r in rows} == {("src-002", "false")}


def test_a_missing_video_leaves_the_numbers_and_says_why_there_is_no_picture(study, tmp_path):
    project, toy, model, events, out = study
    (tmp_path / "fly_b.avi").unlink()
    ex = _export(project, out)
    rows = [r for r in schema.read_rows(ex.export_dir / "events.csv") if r["status"] == "ok"]
    b = [r for r in rows if r["source_id"] == "src-002"]
    assert b and all(r["crop_status"] == "frame_unavailable" and r["crop_file"] == "" for r in b)
    assert all(r["map_file"] for r in rows)
    assert ex.counts["crops"] == len(rows) - len(b)


def test_without_crops_no_video_is_read(study, tmp_path):
    project, toy, model, events, out = study
    for name in ("fly_a.avi", "fly_b.avi"):
        (tmp_path / name).unlink()
    d = _export(project, out, crops=False).export_dir
    assert not (d / "crops").exists()
    with np.load(d / "box_maps.npz", allow_pickle=False) as z:
        assert "image_rgb" not in z.files and z["map"].shape[0] == 4


def test_a_new_export_replaces_the_old_and_a_cancelled_one_changes_nothing(study):
    project, toy, model, events, out = study
    first = _export(project, out, grid=8).export_dir
    marker = first / "box_maps.npz"
    before = marker.read_bytes()
    with pytest.raises(ExplainError) as e:
        api.export_run(project, out.run_dir, options=api.ExportOptions(grid=16), cancel=lambda: True)
    assert e.value.code.value == "cancelled"
    assert marker.read_bytes() == before
    assert not first.with_name(first.name + ".partial").exists()
    _export(project, out, grid=16)
    with np.load(marker, allow_pickle=False) as z:
        assert z["map"].shape[1:] == (16, 16)


def test_a_folder_that_is_not_an_export_is_not_overwritten(study, tmp_path):
    project, toy, model, events, out = study
    target = tmp_path / "mine"
    target.mkdir()
    (target / "notes.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ExplainError):
        api.export_run(project, out.run_dir, target)
    assert (target / "notes.txt").read_text(encoding="utf-8") == "keep"


def test_a_gradcam_run_exports_its_cam_on_the_same_grids(study):
    project, toy, model, events, out = study
    cam = api.run_analysis(project, toy, model.model_id, events, api.AnalysisRequest("gradcam", use_cache=False))
    ex = api.export_run(project, cam.run_dir, options=api.ExportOptions(grid=16))
    d = ex.export_dir
    info = json.loads((d / "export.json").read_text(encoding="utf-8"))
    assert info["map"] == "cam" and info["grid"]["extent"] == 1.5
    assert not (d / "patches.csv").exists()
    run = RunReader(cam.run_dir)
    rows = {r["event_id"]: r for r in schema.read_rows(d / "events.csv")}
    with np.load(d / "box_maps.npz", allow_pickle=False) as z:
        for k, eid in enumerate(z["event_id"]):
            r = rows[str(eid)]
            box = tuple(float(r[c]) for c in ("x1", "y1", "x2", "y2"))
            full = run.load_map(r["run_map_file"])["cam"]
            np.testing.assert_array_equal(z["map"][k], sample(full, *box_points(box, 16, 1.5))[0])
            assert r["cam_in_box_fraction"] != "" and r["in_box_positive_share"] == ""
    summary = [s for s in schema.read_rows(d / "summary.csv") if s["level"] == "class"]
    assert all(s["cam_in_box_fraction_mean"] for s in summary)
    drawn = [r for r in schema.read_rows(d / "gallery" / "gallery.csv") if r["file"]]
    assert drawn and all(float(r["vmin"]) == 0.0 for r in drawn)      # a CAM is not signed


def test_export_options_are_strict():
    o = api.ExportOptions(grid=32, extent=1.0, gallery_scale="run")
    assert api.ExportOptions.from_dict(o.to_dict()) == o
    with pytest.raises(ValueError):
        api.ExportOptions.from_dict({"grid": 32, "colour": "red"})
    with pytest.raises(ValueError):
        api.ExportOptions(gallery_scale="pretty")


# -- the gallery ------------------------------------------------------------------------


def test_the_gallery_draws_every_event_on_a_shared_scale(study):
    project, toy, model, events, out = study
    d = _export(project, out).export_dir
    rows = schema.read_rows(d / "gallery" / "gallery.csv")
    assert len(rows) == 5
    drawn = [r for r in rows if r["file"]]
    assert len(drawn) == 4 and all((d / "gallery" / r["file"]).is_file() for r in drawn)
    failed = next(r for r in rows if r["event_id"] == events[2])
    assert failed["status"] == "failed" and failed["file"] == ""
    # One scale per predicted class, and it is the largest |delta| of that class.
    for cls in {r["class_name"] for r in drawn}:
        ranges = {(r["vmin"], r["vmax"]) for r in drawn if r["class_name"] == cls}
        assert len(ranges) == 1
        vmax = float(next(iter(ranges))[1])
        evs = [e for e in schema.read_rows(d / "events.csv") if e["class_name"] == cls and e["map_file"]]
        largest = max(np.nanmax(np.abs(np.load(d / e["map_file"])["map"])) for e in evs)
        assert vmax == pytest.approx(float(largest))
    sheets = sorted(p.name for p in (d / "gallery" / "sheets").iterdir())
    assert "red__p01.png" in sheets and "red__fly-A__p01.png" in sheets and "green__fly-B__p01.png" in sheets


def test_the_gallery_scale_can_be_the_run_each_event_or_a_number(study):
    project, toy, model, events, out = study
    d = _export(project, out, gallery=False).export_dir
    api.render_gallery(d, scale="run")
    rows = [r for r in schema.read_rows(d / "gallery" / "gallery.csv") if r["file"]]
    assert len({(r["vmin"], r["vmax"]) for r in rows}) == 1
    api.render_gallery(d, scale="event")
    rows = [r for r in schema.read_rows(d / "gallery" / "gallery.csv") if r["file"]]
    assert {r["scale_group"] for r in rows} == {f"event:{r['event_id']}" for r in rows}
    api.render_gallery(d, scale=0.25)
    rows = [r for r in schema.read_rows(d / "gallery" / "gallery.csv") if r["file"]]
    assert {(float(r["vmin"]), float(r["vmax"])) for r in rows} == {(-0.25, 0.25)}
    with pytest.raises(ValueError):
        api.render_gallery(d, scale="pretty")


# -- maps aligned to the experimenter's orientations ---------------------------------------


def _orientation_csv(path, rows, header=("event_id", "cx", "cy", "angle_deg", "length", "directed")):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def test_pointing_up_with_the_box_size_gives_the_box_grid(study, tmp_path):
    """Forward = up (-90 deg) and length = the (square) box side: the aligned grid is the box grid."""
    project, toy, model, events, out = study
    d = _export(project, out, grid=12, extent=1.0, gallery=False).export_dir
    rows = [r for r in schema.read_rows(d / "events.csv") if r["status"] == "ok"]
    table = []
    for r in rows:
        x1, y1, x2, y2 = (float(r[c]) for c in ("x1", "y1", "x2", "y2"))
        assert x2 - x1 == y2 - y1
        table.append((r["event_id"], (x1 + x2) / 2, (y1 + y2) / 2, -90, x2 - x1, "true"))
    aligned = api.align_maps(d, _orientation_csv(tmp_path / "o.csv", table), grid=12, extent=1.0,
                             out_dir=tmp_path / "aligned")
    with np.load(d / "box_maps.npz", allow_pickle=False) as z:
        for k, eid in enumerate(aligned.event_ids):
            j = list(z["event_id"]).index(eid)
            np.testing.assert_array_equal(aligned.maps[k], z["map"][j])
            np.testing.assert_array_equal(aligned.images[k], z["image_rgb"][j])
    # Pointing right instead turns the grid a quarter: the box grid rotated anticlockwise.
    turned = api.align_maps(d, [api.Orientation(t[0], t[1], t[2], 0.0, t[4]) for t in table],
                            grid=12, extent=1.0, images=False)
    with np.load(d / "box_maps.npz", allow_pickle=False) as z:
        for k, eid in enumerate(turned.event_ids):
            j = list(z["event_id"]).index(eid)
            np.testing.assert_array_equal(turned.maps[k], np.rot90(z["map"][j]))
    saved = schema.read_rows(tmp_path / "aligned" / "aligned.csv")
    assert {r["status"] for r in saved} == {"ok", "not_analysed"}
    with np.load(tmp_path / "aligned" / "aligned_means.npz", allow_pickle=False) as mm:
        assert set(mm["class_names"]) == {"red", "green"}


def test_an_undirected_axis_averages_both_ends(study):
    project, toy, model, events, out = study
    d = _export(project, out, grid=10, extent=1.0, gallery=False).export_dir
    rows = [r for r in schema.read_rows(d / "events.csv") if r["status"] == "ok"]
    o = [api.Orientation(r["event_id"], (float(r["x1"]) + float(r["x2"])) / 2,
                         (float(r["y1"]) + float(r["y2"])) / 2, 37.0, 30.0, directed=False) for r in rows]
    aligned = api.align_maps(d, o, grid=10, extent=1.0, images=False)
    for m in aligned.maps:
        np.testing.assert_allclose(m, np.rot90(m, 2), equal_nan=True)


def test_orientations_left_out_or_unusable_are_reported_not_guessed(study, tmp_path):
    project, toy, model, events, out = study
    d = _export(project, out, gallery=False).export_dir
    ok = [r["event_id"] for r in schema.read_rows(d / "events.csv") if r["status"] == "ok"]
    path = _orientation_csv(tmp_path / "o.csv", [(ok[0], 100, 80, 45, 30, ""), (ok[1], 100, 80, "", 30, "")],
                            header=("event_id", "cx", "cy", "angle_deg", "length", "my_note"))
    aligned = api.align_maps(d, path, images=False)
    status = {r["event_id"]: r["status"] for r in aligned.rows}
    assert status[ok[0]] == "ok" and status[ok[1]] == "invalid_orientation"
    assert all(status[e] == "no_orientation" for e in ok[2:])
    assert aligned.event_ids == [ok[0]]
    with pytest.raises(ExplainError):
        api.align_maps(d, [api.Orientation("ev-9999", 1, 1, 0, 10)])
    with pytest.raises(ExplainError):
        api.align_maps(d, _orientation_csv(tmp_path / "bad.csv", [(ok[0], 1, 1, 0)],
                                           header=("event_id", "cx", "cy", "angle_deg")))
    with pytest.raises(ExplainError):
        api.align_maps(d, _orientation_csv(tmp_path / "twice.csv",
                                           [(ok[0], 1, 1, 0, 9, ""), (ok[0], 2, 2, 0, 9, "")]))
