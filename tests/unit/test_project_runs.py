"""Projects and run folders: written, read back, analysed again -- identically."""

from __future__ import annotations

import csv
import json

import cv2
import numpy as np
import pytest

from tests import synthetic as syn
from yoru_explain import api
from yoru_explain.io import schema
from yoru_explain.io.detections_csv import load_detections_csv
from yoru_explain.io.runs import RunReader

N_FRAMES = 40


def _video(path):
    h, w = 192, 320
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (w, h))
    for i in range(N_FRAMES):
        f = syn.blank(h, w)
        syn.paint(f, syn.window(5, 5 + i // 10), syn.RED)
        syn.paint(f, syn.window(8, 14), syn.GREEN)
        out.write(f)
    out.release()
    return path


def _index_of(adapter, project, source_id, frame, cls):
    with project.open_source(source_id) as r:
        dets = adapter.detect(r.read(frame).image)
    return next(i for i, d in enumerate(dets) if d.class_id == cls)


@pytest.fixture
def setup(tmp_path, toy):
    video = _video(tmp_path / "clip.avi")
    project = api.Project.create(tmp_path / "proj", "test")
    source = project.add_source(video)
    model = project.add_model(None, "toy-grid")
    sid = source["source_id"]
    e1 = api.add_event(project, toy, model.model_id, sid, 3, _index_of(toy, project, sid, 3, 0))
    e2 = api.add_event(project, toy, model.model_id, sid, 25, _index_of(toy, project, sid, 25, 1))
    project.save()
    return project, toy, model, [e1, e2], video


REQ = api.AnalysisRequest("occlusion", occlusion=api.OcclusionConfig(patch_size=16, stride=16),
                          use_cache=False)


def test_events_survive_saving(setup):
    project, toy, model, events, _ = setup
    again = api.Project.open(project.root)
    assert [e["event_id"] for e in again.events] == events
    t = again.target(events[0])
    assert t.frame.frame_index == 3 and t.frame.access == "sequential"
    assert t.candidate is not None and t.detection.class_name == "red"
    assert t.to_dict() == project.target(events[0]).to_dict()
    rows = schema.read_rows(project.root / "events.csv")
    assert all(r["schema_version"] == schema.SCHEMA_VERSION for r in rows)


def test_a_run_writes_consistent_files(setup):
    project, toy, model, events, _ = setup
    out = api.run_analysis(project, toy, model.model_id, events, REQ)
    assert out.status == "completed" and out.counts == {"ok": 2}
    run = RunReader(out.run_dir)
    for name in ("config.json", "manifest.json", "results.csv", "perturbations.csv", "logs/run.log"):
        assert (out.run_dir / name).is_file(), name
    assert run.manifest["schema_version"] == schema.SCHEMA_VERSION
    assert run.manifest["status"] == "completed"
    assert run.manifest["events"] == {"requested": 2, "ok": 2}
    assert run.manifest["identity_check"]["status"] == "not_applicable"
    assert run.config["request"]["method_config"]["patch_size"] == [16, 16]
    assert "run " in (out.run_dir / "logs" / "run.log").read_text(encoding="utf-8")

    results = run.results
    assert [r["event_id"] for r in results] == events
    for r in results:
        assert r["schema_version"] == schema.SCHEMA_VERSION and r["status"] == "ok"
        assert r["method_kind"] == "perturbation_effect"
        data = run.load_map(r["map_file"])
        assert data["delta_map"].shape == (192, 320)
        assert data["meta"]["target"]["event_id"] == r["event_id"]
        # The target carries the frame -> input transform it was scored through.
        assert data["meta"]["target"]["transform"]["input_shape"] == [192, 320]
        assert data["meta"]["coordinate_system"] == "frame_px_xyxy"
        assert (out.run_dir / r["preview_file"]).is_file()
        rows = run.perturbations(r["event_id"])
        patches = [p for p in rows if p["kind"] == "patch"]
        assert len(patches) == int(r["n_perturbations"]) == len(data["patch_delta"])
        assert {p["kind"] for p in rows} == {"patch", "noop", "full_region"}
        noop = next(p for p in rows if p["kind"] == "noop")
        assert float(noop["delta_score"]) == 0.0
        counted = [float(p["delta_score"]) for p in patches if p["counted"] == "true"]
        assert max(counted) == float(r["delta_score"])
        np.testing.assert_array_equal(
            np.array([schema.as_float(p["delta_score"]) for p in patches]), data["patch_delta"])


def test_rerunning_the_saved_config_reproduces_the_numbers(setup):
    project, toy, model, events, _ = setup
    first = api.run_analysis(project, toy, model.model_id, events, REQ)
    saved = RunReader(first.run_dir)
    request = api.AnalysisRequest.from_dict(saved.config["request"])
    assert request == REQ
    second = api.run_analysis(project, toy, model.model_id, saved.config["events"], request)
    again = RunReader(second.run_dir)
    skip = {"run_id", "elapsed_s", "map_file", "preview_file"}
    for a, b in zip(saved.results, again.results):
        assert {k: v for k, v in a.items() if k not in skip} == {k: v for k, v in b.items() if k not in skip}
        ma, mb = saved.load_map(a["map_file"]), again.load_map(b["map_file"])
        for k in ("delta_map", "coverage", "patch_delta", "patch_score"):
            np.testing.assert_array_equal(ma[k], mb[k])
    strip = lambda rows: [{k: v for k, v in r.items() if k != "run_id"} for r in rows]
    assert strip(saved.perturbations()) == strip(again.perturbations())


def test_a_cached_result_is_reused_and_says_so(setup):
    project, toy, model, events, _ = setup
    first = api.run_analysis(project, toy, model.model_id, events, REQ)
    cached = api.run_analysis(project, toy, model.model_id, events,
                              api.AnalysisRequest(REQ.method, occlusion=REQ.occlusion, use_cache=True))
    assert cached.counts == {"ok": 2, "cached": 2}
    a, b = RunReader(first.run_dir), RunReader(cached.run_dir)
    typed = {r["event_id"]: r for r in a.typed_results()}
    for r in cached.results:        # numbers, as for computed rows -- not CSV text
        assert isinstance(r["delta_score"], float) and isinstance(r["n_perturbations"], int)
        assert r["delta_score"] == typed[r["event_id"]]["delta_score"]
    for ra, rb in zip(a.results, b.results):
        assert rb["cached_from"] == a.run_id and rb["delta_score"] == ra["delta_score"]
        np.testing.assert_array_equal(a.load_map(ra["map_file"])["delta_map"],
                                      b.load_map(rb["map_file"])["delta_map"])
    # A different setting is not a cache hit.
    other = api.run_analysis(project, toy, model.model_id, events, api.AnalysisRequest(
        "occlusion", occlusion=api.OcclusionConfig(patch_size=16, stride=8)))
    assert "cached" not in other.counts


def _edit_event(project, event_id, **changes):
    rows = schema.read_rows(project.root / "events.csv")
    for r in rows:
        if r["event_id"] == event_id:
            r.update({k: str(v) for k, v in changes.items()})
    schema.write_rows(project.root / "events.csv", schema.EVENT_COLUMNS, rows)
    return api.Project.open(project.root)


def test_a_changed_frame_is_a_frame_mismatch(setup):
    project, toy, model, events, _ = setup
    project = _edit_event(project, events[0], frame_sha1="0" * 40)
    out = api.run_analysis(project, toy, model.model_id, events, REQ)
    rows = {r["event_id"]: r for r in RunReader(out.run_dir).results}
    assert rows[events[0]]["status"] == "failed"
    assert rows[events[0]]["error_code"] == "frame_mismatch"
    assert rows[events[1]]["status"] == "ok"      # one failure does not stop the run
    assert out.counts == {"failed": 1, "ok": 1}


def test_a_detection_that_is_not_there_is_not_analysed(setup):
    project, toy, model, events, _ = setup
    project = _edit_event(project, events[0], x1=0, y1=0, x2=20, y2=20, candidate_index="",
                          candidate_locator="")
    out = api.run_analysis(project, toy, model.model_id, events[:1], REQ)
    (row,) = RunReader(out.run_dir).results
    assert row["error_code"] == "target_not_reproduced" and not row["map_file"]
    assert row["delta_score"] == ""          # no number at all, not 0


def test_a_missing_video_is_reported(setup):
    project, toy, model, events, video = setup
    video.unlink()
    out = api.run_analysis(project, toy, model.model_id, events, REQ)
    assert {r["error_code"] for r in RunReader(out.run_dir).results} == {"missing_data"}


def test_cancelling_marks_the_rest_cancelled(setup):
    project, toy, model, events, _ = setup
    out = api.run_analysis(project, toy, model.model_id, events, REQ, cancel=lambda: True)
    assert out.status == "cancelled"
    run = RunReader(out.run_dir)
    assert run.manifest["status"] == "cancelled"
    assert {r["status"] for r in run.results} == {"cancelled"}


def test_progress_is_reported(setup):
    project, toy, model, events, _ = setup
    seen = []
    api.run_analysis(project, toy, model.model_id, events, REQ, progress=seen.append)
    assert seen and seen[-1].n_events == 2
    assert all(0.0 <= p.fraction <= 1.0 for p in seen)


def test_yoru_analysis_csv_import(setup, tmp_path):
    project, toy, model, events, video = setup
    with project.open_source("src-001") as r:
        frame = r.read(12)
    dets = toy.detect(frame.image)
    path = tmp_path / "clip.csv"  # YORU's video-analysis layout
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "x1", "y1", "x2", "y2", "x_center", "y_center", "w", "h", "angle",
                    "confidence", "class", "class_name"])
        for d in dets:
            x1, y1, x2, y2 = d.box
            w.writerow([12, x1 + 0.5, y1, x2, y2 - 0.5, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, 0,
                        round(d.score, 4), d.class_id, d.class_name])
    layout, frames = load_detections_csv(path)
    assert layout == "yoru-analysis" and list(frames) == [12]
    imported = frames[12][0].detection
    assert imported.candidate is None
    target = api.target_from_detection(imported, -1, frame.image, source_id="src-001",
                                       path=str(video), kind="video", frame_index=12,
                                       model_id=model.model_id, origin="csv_import",
                                       access=frame.access, frame_sha1=frame.sha1)
    eid = project.add_event(target)
    project.save()
    out = api.run_analysis(project, toy, model.model_id, [eid], REQ)
    (row,) = RunReader(out.run_dir).results
    assert row["status"] == "ok" and row["reproduction"] == "iou"
    assert float(row["reproduction_iou"]) > 0.9 and row["candidate_index"]


def test_json_files_hold_no_nan(setup):
    project, toy, model, events, _ = setup
    out = api.run_analysis(project, toy, model.model_id, events, REQ)
    for name in ("config.json", "manifest.json"):
        text = (out.run_dir / name).read_text(encoding="utf-8")
        json.loads(text)
        assert "NaN" not in text and "Infinity" not in text


def test_events_from_a_yoru_csv_in_bulk(setup, tmp_path):
    project, toy, model, events, video = setup
    path = tmp_path / "bulk.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "x1", "y1", "x2", "y2", "x_center", "y_center", "w", "h", "angle",
                    "confidence", "class", "class_name"])
        with project.open_source("src-001") as r:
            for frame in (2, 15, 33):
                for d in toy.detect(r.read(frame).image):
                    x1, y1, x2, y2 = d.box
                    w.writerow([frame, x1, y1, x2, y2, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, 0,
                                d.score, d.class_id, d.class_name])
        w.writerow([20, 0, 0, 30, 30, 15, 15, 30, 30, 0, 0.9, 0, "red"])   # not on that frame
    added = api.add_events_from_detections_csv(project, "src-001", path, model_id=model.model_id,
                                               class_names=["red"])
    assert len(added) == 4
    project.save()
    t = project.target(added[0])
    assert t.origin == "csv_import" and t.candidate is None
    assert t.frame.access == "unread" and t.frame.frame_sha1 is None
    out = api.run_analysis(project, toy, model.model_id, added, REQ)
    rows = {r["event_id"]: r for r in RunReader(out.run_dir).results}
    by_frame = {project.target(e).frame.frame_index: rows[e] for e in added}
    assert {f: r["status"] for f, r in by_frame.items()} == {2: "ok", 15: "ok", 20: "failed", 33: "ok"}
    assert by_frame[20]["error_code"] == "target_not_reproduced"   # a box that is not there
    good = [by_frame[f] for f in (2, 15, 33)]
    assert all(r["reproduction"] == "iou" and r["frame_sha1"] for r in good)  # hashed when decoded
    assert len(api.add_events_from_detections_csv(project, "src-001", path, limit=2)) == 2


def test_a_failed_identity_check_stops_the_run(setup, monkeypatch):
    project, toy, model, events, _ = setup
    monkeypatch.setattr(toy, "identity_check", lambda image: {
        "status": "failed", "reason": "simulated", "exact": False})
    out = api.run_analysis(project, toy, model.model_id, events, REQ)
    assert out.status == "failed"
    run = RunReader(out.run_dir)
    assert run.manifest["status"] == "failed"
    assert run.manifest["identity_check"]["reason"] == "simulated"
    assert {r["error_code"] for r in run.results} == {"identity_check_failed"}
    assert not any(r["map_file"] for r in run.results)


def test_the_loaded_model_must_be_the_projects_model(setup, monkeypatch):
    from yoru_explain.core.errors import ExplainError

    project, toy, model, events, _ = setup
    other = project.add_model(__file__, "yoru-yolov5")       # any file: a different sha256
    info = toy.model_info()
    monkeypatch.setattr(toy, "model_info", lambda: dict(info, sha256="0" * 64))
    runs_before = list(project.runs_dir.glob("*")) if project.runs_dir.exists() else []
    with pytest.raises(ExplainError, match="not project model"):
        api.run_analysis(project, toy, other.model_id, events, REQ)
    runs_after = list(project.runs_dir.glob("*")) if project.runs_dir.exists() else []
    assert runs_after == runs_before                       # no half-made run folder


def test_a_failure_to_write_results_is_a_row_and_the_run_goes_on(setup, monkeypatch):
    from yoru_explain.io import runs as runio

    project, toy, model, events, _ = setup
    calls = []
    original = runio.RunWriter.save_preview

    def flaky(self, event_id, method, image):
        calls.append(event_id)
        if len(calls) == 1:
            raise OSError("disk full (simulated)")
        return original(self, event_id, method, image)

    monkeypatch.setattr(runio.RunWriter, "save_preview", flaky)
    out = api.run_analysis(project, toy, model.model_id, events, REQ)
    assert out.status == "completed" and out.counts == {"failed": 1, "ok": 1}
    run = RunReader(out.run_dir)
    rows = {r["event_id"]: r for r in run.results}
    bad, good = rows[calls[0]], rows[calls[1]]
    assert bad["error_code"] == "output_failed" and "disk full" in bad["message"]
    assert not bad["map_file"] and not bad["delta_score"]
    assert not (out.run_dir / "maps" / f"{calls[0]}__occlusion.npz").exists()
    assert not run.perturbations(calls[0])            # nothing half-written for it
    assert good["status"] == "ok" and run.perturbations(calls[1])
    assert run.manifest["error_codes"] == {"output_failed": 1}


def test_an_unexpected_error_in_one_event_does_not_end_the_run(setup, monkeypatch):
    from yoru_explain.core import pipeline

    project, toy, model, events, _ = setup
    original = pipeline._analyze_event

    def boom(*args, **kwargs):
        if args[4].event_id == events[0]:
            raise RuntimeError("a defect")
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_analyze_event", boom)
    out = api.run_analysis(project, toy, model.model_id, events, REQ)
    rows = {r["event_id"]: r for r in RunReader(out.run_dir).results}
    assert rows[events[0]]["error_code"] == "internal_error"
    assert rows[events[1]]["status"] == "ok"
    assert len(out.results) == 2


def test_an_escaping_interrupt_leaves_a_failed_manifest(setup, monkeypatch):
    from yoru_explain.core import pipeline

    project, toy, model, events, _ = setup

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline, "_analyze_event", interrupt)
    with pytest.raises(KeyboardInterrupt):
        api.run_analysis(project, toy, model.model_id, events, REQ)
    (run_dir,) = project.run_dirs()
    assert RunReader(run_dir).manifest["status"] == "failed"
