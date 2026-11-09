"""The GUI's way to the experimenter's own analysis: units, patch sizes, spacing, export.

In process, without a viewport, with the synthetic detector (as test_gui_smoke).
"""

from __future__ import annotations

import csv
import time

import cv2
import dearpygui.dearpygui as dpg
import pytest

from tests import synthetic as syn

pytestmark = pytest.mark.gui


@pytest.fixture
def app():
    from yoru_explain.gui.main_window import ExplainApp

    dpg.create_context()
    application = ExplainApp()
    application.build()
    yield application
    application.shutdown()
    dpg.destroy_context()


def tick_until(app, condition, timeout=60.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.tick()
        if condition():
            return True
        time.sleep(0.005)
    return False


def errors(app):
    return [p for p in app.popups.shown if p[0] == "error"]


def _video(path, n=12):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (320, 192))
    for _ in range(n):
        f = syn.blank()
        syn.paint(f, syn.window(5, 5), syn.RED)
        syn.paint(f, syn.window(8, 14), syn.GREEN)
        out.write(f)
    out.release()
    return path


@pytest.fixture
def project(tmp_path):
    from yoru_explain.io.project import Project

    p = Project.create(tmp_path / "proj")
    p.add_source(_video(tmp_path / "fly_a.avi"))
    p.add_source(_video(tmp_path / "fly_b.avi"))
    p.add_model(None, "toy-grid")
    p.save()
    return Project.open(p.root)


def test_units_are_set_for_selected_videos_or_from_a_table(app, project, tmp_path):
    st = app.state
    st.set_project(project)
    pt = app.project_tab
    assert pt.sources.cell(0, 10) == "(src-001)"            # its own unit until one is set
    pt.set_unit(["src-001"], unit="fly-A")
    assert pt.sources.cell(0, 10) == "fly-A" and st.project.unit_of("src-001") == ("fly-A", True)
    pt.sources.select(["src-002"])
    pt.set_unit()                                           # asks for the name in a form
    assert app.popups.showing and app.popups.shown[-1][0] == "form"
    table = tmp_path / "units.csv"
    with open(table, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows([["video", "unit_id", "sex"], ["fly_b.avi", "fly-B", "f"], ["other.avi", "x", "m"]])
    app.popups.close(app.popups.showing[0])
    pt.units_from_csv(str(table))
    assert st.project.unit_of("src-002") == ("fly-B", True)
    assert "other.avi" in app.popups.shown[-1][2] and "sex" in app.popups.shown[-1][2]
    from yoru_explain.io.project import Project

    assert Project.open(st.project.root).unit_of("src-002") == ("fly-B", True)   # saved
    assert not errors(app)


def test_patches_can_be_sized_relative_to_the_target(app):
    from yoru_explain.gui.analyze_tab import PATCH_FRACTION, PATCH_PX

    an = app.analyze_tab
    assert an.request().occlusion.patch_fraction is None and dpg.is_item_shown("ana_px_rows")
    dpg.set_value("ana_patch_unit", PATCH_FRACTION)
    an._show_rows()
    assert dpg.is_item_shown("ana_frac_rows") and not dpg.is_item_shown("ana_px_rows")
    dpg.set_value("ana_patch_frac", 0.3)
    dpg.set_value("ana_stride_frac", 0.1)
    req = an.request()
    assert req.occlusion.patch_fraction == 0.3 and req.occlusion.stride_fraction == 0.1
    dpg.set_value("ana_patch_unit", PATCH_PX)
    an._show_rows()
    an.set_request(req)                                    # e.g. a settings file read back
    assert dpg.get_value("ana_patch_unit") == PATCH_FRACTION and dpg.is_item_shown("ana_frac_rows")
    assert an.request().occlusion == req.occlusion


def test_a_run_is_exported_from_the_results_tab(app, project):
    from yoru_explain import api

    st = app.state
    adapter = api.load_adapter("toy-grid", device="cpu")
    st.set_project(project)
    st.set_adapter(adapter, project.models[0].model_id)
    project.set_unit("src-001", "fly-A")
    events = []
    for sid, frame in (("src-001", 2), ("src-002", 5)):
        with project.open_source(sid) as r:
            dets = adapter.detect(r.read(frame).image)
        events.append(api.add_event(project, adapter, project.models[0].model_id, sid, frame,
                                    next(i for i, d in enumerate(dets) if d.class_name == "red")))
    project.save()
    out = api.run_analysis(project, adapter, project.models[0].model_id, events, api.AnalysisRequest(
        "occlusion", occlusion=api.OcclusionConfig(patch_fraction=0.5), use_cache=False))
    sts = app.statistics_tab
    st.runs_changed()
    sts.refresh()
    assert "Not exported yet" in dpg.get_value("stat_export_status")
    assert not dpg.get_item_configuration("stat_open_export")["enabled"]
    dpg.set_value("stat_scale", "run")
    sts.export()
    assert st.tasks.busy and st.tasks.current == "export"
    assert dpg.get_item_configuration("stat_cancel")["enabled"]
    assert tick_until(app, lambda: not st.tasks.busy)
    assert sts.last_export is not None, dpg.get_value("stat_export_status")
    d = sts.last_export.export_dir
    assert d == project.root / "exports" / out.run_id
    assert (d / "events.csv").is_file() and (d / "gallery" / "gallery.csv").is_file()
    assert "Exported 2 of 2" in dpg.get_value("stat_export_status")
    assert dpg.get_item_configuration("stat_open_export")["enabled"]
    import json

    assert json.loads((d / "export.json").read_text(encoding="utf-8"))["options"]["gallery_scale"] == "run"
    # Shown again later: the tab knows the run has an export.
    sts.refresh()
    assert "Exported to" in dpg.get_value("stat_export_status")
    assert not errors(app), errors(app)


def test_a_cancelled_export_says_so_and_is_not_an_error(app, project, monkeypatch):
    import yoru_explain.core.export as export_module
    from yoru_explain import api
    from yoru_explain.core.errors import ErrorCode, ExplainError

    st = app.state
    adapter = api.load_adapter("toy-grid", device="cpu")
    st.set_project(project)
    ev = api.add_event(project, adapter, project.models[0].model_id, "src-001", 1, 0)
    project.save()
    api.run_analysis(project, adapter, project.models[0].model_id, [ev],
                     api.AnalysisRequest("occlusion", use_cache=False))
    seen = {}

    def cancelled(project, run_dir, options=None, *, progress=None, cancel=None):
        seen["cancel"] = cancel
        raise ExplainError(ErrorCode.CANCELLED, "export cancelled")

    monkeypatch.setattr(export_module, "export_run", cancelled)
    sts = app.statistics_tab
    st.runs_changed()
    sts.refresh()
    sts.export()
    assert tick_until(app, lambda: not st.tasks.busy)
    assert seen["cancel"] == st.tasks.cancelled             # the Cancel button reaches the export
    assert sts.last_export is None
    assert "cancelled" in dpg.get_value("stat_export_status")
    assert not errors(app)


def test_imported_events_keep_the_gap_asked_for(app, project, tmp_path):
    from yoru_explain import api

    adapter = api.load_adapter("toy-grid", device="cpu")
    path = tmp_path / "fly_a.csv"
    with open(path, "w", newline="", encoding="utf-8") as f, project.open_source("src-001") as r:
        w = csv.writer(f)
        w.writerow(["frame", "x1", "y1", "x2", "y2", "x_center", "y_center", "w", "h", "angle",
                    "confidence", "class", "class_name"])
        for i in range(12):
            for d in adapter.detect(r.read(i).image):
                x1, y1, x2, y2 = d.box
                w.writerow([i, x1, y1, x2, y2, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, 0,
                            d.score, d.class_id, d.class_name])
    st = app.state
    st.set_project(project)
    st.set_adapter(adapter, project.models[0].model_id)
    ex = app.explore_tab
    ex.set_index(0)
    assert tick_until(app, lambda: ex.frame is not None)
    ex.import_csv(str(path))
    form = ex.csv_form
    dpg.set_value(form["classes"]["green"], False)
    dpg.set_value(form["limit"], 12)
    dpg.set_value(form["gap"], 0.2)                    # 5 frames at 25 fps
    ex._accept_csv_form(True)
    frames = sorted(int(e["frame_index"]) for e in st.project.events)
    assert frames == [0, 5, 10]
    assert not errors(app)
