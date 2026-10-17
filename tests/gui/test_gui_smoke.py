"""The GUI end to end: project -> model -> pick a box -> analyse -> view.

The in-process tests build the real window in a DearPyGui context without a
viewport, with the synthetic detector, so they need neither a display nor
model weights.  The render loop is replaced by calls to ``app.tick()``; the
workflow test checks that a tick never waits on the analysis running on the
worker thread.  The native-window test opens the real window for a moment;
it is opt-in (``YORU_EXPLAIN_GUI_WINDOW=1``) because it needs a desktop.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
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


def tick_until(app, condition, timeout=30.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.tick()
        if condition():
            return True
        time.sleep(0.005)
    return False


def errors(app):
    return [p for p in app.popups.shown if p[0] == "error"]


def _video(path, n=30):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (320, 192))
    for i in range(n):
        f = syn.blank()
        syn.paint(f, syn.window(5, 5), syn.RED)
        syn.paint(f, syn.window(8, 14), syn.GREEN)
        out.write(f)
    out.release()
    return path


def test_the_whole_gui_workflow(app, tmp_path):
    from yoru_explain.io.project import Project

    video = _video(tmp_path / "clip.avi")
    project = Project.create(tmp_path / "proj")
    project.add_source(video)
    model = project.add_model(None, "toy-grid")
    project.save()

    st = app.state
    st.set_project(Project.open(project.root))
    assert app.project_tab.sources.row_count == 1 and app.project_tab.models.row_count == 1
    app.tick()                                  # the status line follows once per frame
    assert str(project.root) in app.status_text

    # Load the model on the task thread.
    app.project_tab.models.select_row(0)
    app.project_tab.load_model()
    assert tick_until(app, lambda: st.adapter is not None), "model did not load"
    assert st.model_id == model.model_id
    assert app.analyze_tab.gradcam_offered          # the synthetic model's probe passed
    assert model.model_id in app.status_text

    # Explore: a frame arrives with its detections; click the red box.
    ex = app.explore_tab
    ex.set_index(4)
    assert tick_until(app, lambda: ex.frame is not None and ex.frame.frame_index == 4 and ex.detections)
    assert ex.frame.access in ("sequential", "seek_verified")
    assert ex.view.image_size == (320, 192)
    assert ex.view.box_count == len(ex.detections)
    red = next(i for i, d in enumerate(ex.detections) if d.class_name == "red")
    x1, y1, x2, y2 = ex.detections[red].box
    ex._clicked((x1 + x2) / 2, (y1 + y2) / 2)
    assert ex.selected == red and ex.det_table.selected == [red]
    ex.add_events(all_on_frame=False)
    assert len(st.project.events) == 1
    ex.events.click(0)
    assert st.selected_events == ["ev-0001"]

    # Analyze, and time every tick meanwhile.
    an = app.analyze_tab
    assert an.request().occlusion.baseline.value == (180, 105, 255)   # hot pink by default
    assert an.colour_rgb() == (255, 105, 180)                          # the swatch shows it
    dpg.set_value("ana_patch", 8)
    dpg.set_value("ana_stride", 4)
    dpg.set_value("ana_region", "full")
    an.run(selected_only=True)
    assert st.tasks.busy
    assert dpg.get_item_configuration("ana_cancel")["enabled"]
    assert not dpg.get_item_configuration("ana_run_sel")["enabled"]
    ticks = []
    end = time.monotonic() + 120
    while st.tasks.busy and time.monotonic() < end:
        t0 = time.perf_counter()
        app.tick()
        ticks.append(time.perf_counter() - t0)
        time.sleep(0.005)
    assert not st.tasks.busy
    assert an.last_run is not None and an.last_run.status == "completed", dpg.get_value("ana_status")
    assert an.results.row_count == 1 and an.results.cell(0, 2) == "ok"
    assert len(ticks) > 5           # the loop kept running
    assert max(ticks) < 0.5         # and never waited on the analysis
    assert "ev-0001" in an.log.text()

    # View the result: heatmap, overlay, side by side, with the colour scale.
    run_dir = str(an.last_run.run_dir)
    st.showResult.emit(run_dir, "ev-0001")
    assert app.current_tab() == "explore"
    assert tick_until(app, lambda: ex._result is not None and ex.frame is not None
                      and ex.frame.frame_index == 4)
    for mode, panels in (("Heatmap", 1), ("Overlay", 1), ("Side by side", 3)):
        dpg.set_value("exp_mode", mode)
        ex.redraw()
        assert ex.view.panels == panels
        assert dpg.is_item_shown("exp_scale_group")
        assert ex._shown[0] is not None
    assert ex.view.image_size == (3 * 320, 192)
    dpg.set_value("exp_mode", "Original")
    ex.redraw()
    assert not dpg.is_item_shown("exp_scale_group")
    assert "Perturbation effect" in ex.info_text()
    assert "not a biological interpretation" in ex.info_text()

    # Grad-CAM through the same buttons.
    an.set_method("gradcam")
    assert dpg.is_item_shown("ana_cam") and not dpg.is_item_shown("ana_occ")
    an.run(selected_only=True)
    assert tick_until(app, lambda: not st.tasks.busy, timeout=60)
    assert an.last_run.status == "completed" and an.results.cell(0, 2) == "ok"

    # Statistics lists both runs; Settings knows the model.
    app.statistics_tab.refresh()
    assert app.statistics_tab.run_count == 2
    assert app.statistics_tab.table.row_count == 1
    app.settings_tab.refresh()
    assert "capabilities" in dpg.get_value("set_info")
    assert not errors(app), errors(app)


def test_cancel_from_the_gui(app, tmp_path):
    from yoru_explain import api
    from yoru_explain.io.project import Project

    video = _video(tmp_path / "clip.avi", n=10)
    project = Project.create(tmp_path / "proj")
    sid = project.add_source(video)["source_id"]
    mid = project.add_model(None, "toy-grid").model_id
    adapter = api.load_adapter("toy-grid", device="cpu")
    for frame in range(5):
        api.add_event(project, adapter, mid, sid, frame, 0)
    project.save()

    st = app.state
    st.set_project(Project.open(project.root))
    st.set_adapter(adapter, mid)
    an = app.analyze_tab
    dpg.set_value("ana_patch", 4)
    dpg.set_value("ana_stride", 2)
    dpg.set_value("ana_region", "full")
    dpg.set_value("ana_cache", False)
    an.run(selected_only=False)
    an._cancel()
    assert tick_until(app, lambda: not st.tasks.busy, timeout=120)
    assert an.last_run.status == "cancelled"
    assert any(r.get("status") == "cancelled" for r in an.last_run.results)


def test_settings_survive_export_and_import(app, tmp_path):
    an = app.analyze_tab
    dpg.set_value("ana_patch", 24)
    dpg.set_value("ana_baseline", "patch_mean")
    dpg.set_value("ana_value", "0,0,0")
    dpg.set_value("ana_identity", "redetect")
    path = tmp_path / "analysis.json"
    app.settings_tab.export_settings(str(path))
    exported = an.request().to_dict()
    assert json.loads(path.read_text(encoding="utf-8")) == json.loads(json.dumps(exported))

    dpg.set_value("ana_patch", 32)
    dpg.set_value("ana_baseline", "constant")
    dpg.set_value("ana_value", "180,105,255")
    app.settings_tab.import_settings(str(path))
    assert an.request().to_dict() == exported
    assert an.colour_rgb() == (0, 0, 0)
    assert not errors(app)


def test_picking_a_colour_sets_a_constant_baseline(app):
    an = app.analyze_tab
    dpg.set_value("ana_baseline", "blur")
    dpg.set_value("ana_colour", (10, 20, 30, 255))
    an._colour_picked()
    req = an.request()
    assert req.occlusion.baseline.kind == "constant"
    assert req.occlusion.baseline.value == (30, 20, 10)      # B, G, R


def test_without_a_gradient_probe_gradcam_is_not_offered(app):
    class NoGradients:
        name = "fake"

        class capabilities:
            gradients = False
            gradients_reason = "no gradient path"
            score_types = ("obj_x_cls",)
            batch_invariant = True
            batch_max_abs_diff = 0.0

            @staticmethod
            def to_dict():
                return {"gradients": False}

        @staticmethod
        def gradient_layers():
            return []

        @staticmethod
        def model_info():
            return {"device": "cpu"}

    app.analyze_tab.set_method("gradcam")      # not offered without a model
    assert app.analyze_tab.method() == "occlusion"
    app.state.set_adapter(NoGradients(), "m1")
    assert not app.analyze_tab.gradcam_offered
    assert "no gradient path" in dpg.get_value("ana_cam_reason")
    assert dpg.get_item_configuration("ana_score")["items"] == ["obj_x_cls"]
    assert not errors(app), errors(app)


def test_event_selection_with_ctrl_and_shift(app):
    table = app.explore_tab.events
    table.set_rows([[f"ev-{i}"] + [""] * 7 for i in range(5)], [f"ev-{i}" for i in range(5)])
    table.click(1)
    table.click(3, shift=True)
    assert table.selected == ["ev-1", "ev-2", "ev-3"]
    table.click(2, ctrl=True)
    assert table.selected == ["ev-1", "ev-3"]
    assert app.state.selected_events == ["ev-1", "ev-3"]
    columns = dpg.get_item_children(table.tag, 0)
    table._sorted([[columns[0], -1]])                    # a click on the header, descending
    assert table.keys == [f"ev-{i}" for i in (4, 3, 2, 1, 0)]
    assert table.selected == ["ev-3", "ev-1"]            # display order; the selection is kept
    table._sorted(None)
    table.set_rows([["ev-3"] + [""] * 7], ["ev-3"])     # rows that remain stay selected
    assert table.selected == ["ev-3"]


def test_a_failing_callback_is_reported_not_fatal(app):
    from yoru_explain.gui.main_window import run_callbacks

    calls = []
    run_callbacks(app, [(lambda: 1 / 0, None, None, None),
                        (lambda s, a: calls.append((s, a)), "item", 3, None)])
    assert calls == [("item", 3)]                           # the next callback still ran
    assert [e[2] for e in errors(app)] == ["ZeroDivisionError: division by zero"]

    def bad_slot():
        raise RuntimeError("slot failed")

    app.state.eventsChanged.connect(bad_slot)
    app.state.eventsChanged.emit()                          # the tabs' slots still run
    assert errors(app)[-1][2] == "RuntimeError: slot failed"


def test_popups_open_one_at_a_time(app):
    # DearPyGui closes a modal opened over another, or in the frame one closed.
    popups = app.popups
    first = popups.message("A", "first")
    second = popups.message("B", "asked for while the first is showing")
    assert popups.showing == [first] and popups.waiting == 1
    popups.close(first)
    app.tick()                          # the closed one is off screen for a frame first
    assert popups.showing == [] and popups.waiting == 1
    for _ in range(3):
        app.tick()
    assert popups.showing == [second] and popups.waiting == 0
    assert not dpg.does_item_exist(first)

    answered = []
    question = popups.confirm("Q", "really?", lambda: answered.append(popups.message("C", "done")))
    popups.close(second)
    for _ in range(3):
        app.tick()
    assert popups.showing == [question]


def test_closing_while_a_task_runs_asks_and_waits(app):
    gate = threading.Event()
    seen = []

    def task():
        while not app.state.tasks.cancelled() and not gate.is_set():
            time.sleep(0.01)
        seen.append(app.state.tasks.cancelled())

    assert app.state.tasks.submit("long", task)
    app.request_close()
    assert app.popups.shown[-1][0] == "question" and not app.closed
    app._cancel_and_quit()                     # "Cancel and quit"
    assert tick_until(app, lambda: app.closed, timeout=10)
    assert seen == [True]


def test_a_failed_time_index_is_reported_not_raised(tmp_path):
    from yoru_explain.gui.worker import FrameRequest, FrameService

    fs = FrameService()
    got, frames = [], []
    fs.indexReady.connect(lambda sid, summary: got.append((sid, summary)))
    fs.frameReady.connect(frames.append)
    fs.build_index(FrameRequest("src-x", str(tmp_path / "gone.mp4"), False, False, 0))
    assert fs.wait_ready(20)
    fs.poll()
    assert got[0][0] == "src-x" and got[0][1]["status"] == "failed" and not frames
    fs.shutdown()


def test_only_the_latest_frame_request_is_served(tmp_path):
    from yoru_explain.gui.worker import FrameRequest, FrameService

    video = str(_video(tmp_path / "clip.avi", n=20))
    fs = FrameService()
    frames = []
    fs.frameReady.connect(frames.append)
    gate = threading.Event()
    original = fs._frame
    fs._frame = lambda req: (gate.wait(5), original(req))[1]
    fs.request(FrameRequest("s", video, False, False, 1, detect=False))   # taken at once
    time.sleep(0.1)
    for i in (2, 3, 4):
        fs.request(FrameRequest("s", video, False, False, i, detect=False))
    gate.set()
    end = time.monotonic() + 20
    while len(frames) < 2 and time.monotonic() < end:
        fs.poll()
        time.sleep(0.01)
    time.sleep(0.2)
    fs.poll()
    assert [f.request.index for f in frames] == [1, 4]
    assert frames[1].frame.frame_index == 4 and frames[1].error is None
    fs.shutdown()


def test_models_are_added_with_the_adapter_for_their_backend(app, tmp_path):
    import zipfile

    from yoru_explain.gui.project_tab import BACKENDS
    from yoru_explain.io.project import Project

    app.state.set_project(Project.create(tmp_path / "proj"))
    files = {}
    for name in ("flies_rtdetr.pt", "flies.onnx", "best.pt", "other.onnx"):
        files[name] = tmp_path / name
        files[name].write_bytes(name.encode())          # unreadable: YORU decides by the name
    v1 = tmp_path / "renamed_v1_model.pt"               # a YOLOv5 checkpoint, by its pickle
    with zipfile.ZipFile(v1, "w") as zf:
        zf.writestr("archive/data.pkl", b"\x80\x02cmodels.yolo\nDetectionModel\n")
    tab = app.project_tab
    for f in (files["flies_rtdetr.pt"], files["flies.onnx"], v1):
        tab.add_model(str(f))                           # type: auto
    dpg.set_value("proj_backend", BACKENDS["ultralytics"])
    tab.add_model(str(files["best.pt"]))
    dpg.set_value("proj_blackbox", True)
    dpg.set_value("proj_backend", BACKENDS["auto"])
    tab.add_model(str(files["other.onnx"]))
    models = app.state.project.models
    assert [m.adapter for m in models] == ["yoru-rtdetr", "yoru-onnx", "yoru-yolov5", "yoru-ultralytics",
                                           "yoru-detector"]
    assert "rgb_input" in models[2].options and "rgb_input" not in models[0].options
    assert models[4].options["backend"] == "onnx"
    assert tab.models.row_count == 5 and not errors(app)


class _RedetectOnly:
    """What an RT-DETR / Faster R-CNN adapter reports: no fixed candidates."""

    name = "fake-rtdetr"

    class capabilities:
        gradients = True
        gradients_reason = "probe passed"
        score_types = ("confidence", "cls_prob")
        identity_modes = ("redetect",)
        family = "RT-DETR (test)"
        candidate_kind = "query"
        batch_invariant = True
        batch_max_abs_diff = 0.0

        @staticmethod
        def to_dict():
            return {"identity_modes": ["redetect"]}

    @staticmethod
    def gradient_layers():
        return []

    @staticmethod
    def describe_score_type(score_type):
        return {"confidence": "sigmoid of the query's class logit"}.get(score_type, score_type)

    @staticmethod
    def model_info():
        return {"device": "cpu"}


def test_a_redetection_only_model_offers_only_redetection(app):
    an = app.analyze_tab
    dpg.set_value("ana_identity", "fixed_candidate")
    app.state.set_adapter(_RedetectOnly(), "m1")
    assert dpg.get_item_configuration("ana_identity")["items"] == ["redetect"]
    assert an.request().occlusion.score.identity_mode == "redetect"
    assert dpg.get_item_configuration("ana_score")["items"] == ["confidence", "cls_prob"]
    dpg.set_value("ana_score", "confidence")
    an._score_help()
    assert "sigmoid of the query's class logit" in dpg.get_value("ana_score_help")
    an.set_method("gradcam")                     # Grad-CAM: the query on its own frame
    assert an.request().method == "gradcam" and not errors(app)
    assert "candidates: query" in dpg.get_value("proj_model_status")


def test_a_model_loaded_with_a_warning_says_so(app):
    adapter = _RedetectOnly()
    adapter.capabilities = type("Caps", (_RedetectOnly.capabilities,),
                                {"warnings": ("built for an end-to-end head, run with NMS",)})
    app.project_tab._task_done("load_model", ("m1", adapter))
    shown = [p for p in app.popups.shown if p[0] == "warning"]
    assert shown and "end-to-end head" in shown[-1][2]


def test_rotated_boxes_are_drawn_and_picked(app):
    import math

    from yoru_explain.core.geometry import obb_aabb
    from yoru_explain.gui.frame_view import pick_box

    rod_a = (100.0, 100.0, 160.0, 10.0, math.pi / 4)
    rod_b = (100.0, 100.0, 160.0, 10.0, -math.pi / 4)
    boxes = [obb_aabb(rod_a), obb_aabb(rod_b)]
    # On rod a, away from where the rods cross: inside both upright boxes, inside rod a only.
    x, y = 100 + 50 * math.cos(math.pi / 4), 100 + 50 * math.sin(math.pi / 4)
    assert pick_box(boxes, x, y, obbs=[rod_a, rod_b]) == 0
    x, y = 100 + 50 * math.cos(-math.pi / 4), 100 + 50 * math.sin(-math.pi / 4)
    assert pick_box(boxes, x, y, obbs=[rod_a, rod_b]) == 1
    assert pick_box(boxes, 45, 100, obbs=[rod_a, rod_b], margin=6) is None   # an empty corner
    assert pick_box(boxes, 45, 100, margin=6) == 0                            # upright: inside both
    view = app.explore_tab.view
    view.set_image(np.zeros((200, 200, 3), np.uint8))
    view.set_boxes(boxes, obbs=[rod_a, rod_b], selected=0)
    assert view.box_count == 2


@pytest.mark.slow
def test_the_gui_workflow_with_a_rotated_box_model(app, tmp_path):
    """A real ultralytics OBB model through the window: auto type, rotated boxes, analysis."""
    from tests.conftest import MODELS
    from yoru_explain.core.geometry import point_in_obb
    from yoru_explain.io.project import Project

    weights = MODELS / "yolo11n-obb.pt"
    ul = pytest.importorskip("ultralytics")
    if not weights.is_file():
        pytest.skip("yolo11n-obb.pt not in .cache/models (run tests/prepare_models.py)")
    image = tmp_path / "bus.png"
    cv2.imwrite(str(image), cv2.imread(os.path.join(os.path.dirname(ul.__file__), "assets", "bus.jpg")))
    st = app.state
    st.set_project(Project.create(tmp_path / "proj"))
    app.project_tab.add_source([str(image)])
    app.project_tab.add_model(str(weights))                       # type: auto
    assert st.project.models[0].adapter == "yoru-ultralytics"
    app.project_tab.models.select_row(0)
    app.project_tab.load_model()
    assert tick_until(app, lambda: st.adapter is not None, timeout=120), errors(app)
    assert "rotated boxes" in dpg.get_value("proj_model_status")

    ex = app.explore_tab
    ex.set_index(0)
    assert tick_until(app, lambda: ex.frame is not None and ex.detections, timeout=60)
    d = ex.detections[0]
    assert d.obb is not None and ex.view.box_count == len(ex.detections)
    cx, cy = d.obb[0], d.obb[1]
    assert point_in_obb(cx, cy, d.obb)
    ex._clicked(cx, cy)
    assert ex.selected == 0
    ex.add_events(all_on_frame=False)
    assert st.project.target("ev-0001").obb == pytest.approx(d.obb)

    an = app.analyze_tab
    assert dpg.get_item_configuration("ana_identity")["items"] == ["fixed_candidate", "redetect"]
    dpg.set_value("ana_identity", "redetect")
    dpg.set_value("ana_patch", 128)
    dpg.set_value("ana_stride", 128)
    dpg.set_value("ana_region", "box")
    ex.events.click(0)
    an.run(selected_only=True)
    assert tick_until(app, lambda: an.last_run is not None, timeout=300), errors(app)
    run = an.last_run
    assert run.status == "completed" and run.counts == {"ok": 1}, run.results
    assert run.identity_check["exact"]
    assert run.results[0]["obb_angle"] is not None and not errors(app)


@pytest.mark.skipif(os.environ.get("YORU_EXPLAIN_GUI_WINDOW") != "1",
                    reason="set YORU_EXPLAIN_GUI_WINDOW=1 to open a real window")
def test_main_opens_and_closes(tmp_path):
    shot = tmp_path / "window.png"
    code = ("import sys\nfrom yoru_explain.app import main\n"
            f"sys.exit(main({str(tmp_path / 'proj')!r}, exit_after_ms=2500, "
            f"screenshot={str(shot)!r}, tab='explore'))\n")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "proj" / "project.json").is_file()
    assert shot.is_file() and shot.stat().st_size > 0
