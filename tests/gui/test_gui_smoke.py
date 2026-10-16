"""The GUI end to end, offscreen: project -> model -> pick a box -> analyse -> view.

Runs headless (``QT_QPA_PLATFORM=offscreen``, set in conftest) with the
synthetic detector, so it needs neither a display nor model weights.  It also
checks that the window keeps processing events while an analysis runs on the
worker thread.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from tests import synthetic as syn

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp():
    from PyQt5 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication(["test"])


def wait_until(app, condition, timeout=30.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


def _video(path, n=30):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (320, 192))
    for i in range(n):
        f = syn.blank()
        syn.paint(f, syn.window(5, 5), syn.RED)
        syn.paint(f, syn.window(8, 14), syn.GREEN)
        out.write(f)
    out.release()
    return path


def test_the_whole_gui_workflow(qapp, tmp_path):
    from PyQt5 import QtCore

    from yoru_explain.gui.main_window import MainWindow
    from yoru_explain.io.project import Project

    video = _video(tmp_path / "clip.avi")
    project = Project.create(tmp_path / "proj")
    project.add_source(video)
    model = project.add_model(None, "toy-grid")
    project.save()

    w = MainWindow()
    w.show()
    st = w.state
    st.set_project(Project.open(project.root))
    assert w.project_tab.sources.rowCount() == 1 and w.project_tab.models.rowCount() == 1

    # Load the model on the task thread.
    w.project_tab.models.selectRow(0)
    w.project_tab.load_model()
    assert wait_until(qapp, lambda: st.adapter is not None), "model did not load"
    assert st.model_id == model.model_id
    assert w.analyze_tab.m_cam.isEnabled()          # the synthetic model's probe passed

    # Explore: a frame arrives with its detections; click the red box.
    ex = w.explore_tab
    ex.index.setValue(4)
    assert wait_until(qapp, lambda: ex.frame is not None and ex.frame.frame_index == 4 and ex.detections)
    assert ex.frame.access in ("sequential", "seek_verified")
    red = next(i for i, d in enumerate(ex.detections) if d.class_name == "red")
    x1, y1, x2, y2 = ex.detections[red].box
    ex._clicked((x1 + x2) / 2, (y1 + y2) / 2)
    assert ex.selected == red
    ex.add_events(all_on_frame=False)
    assert len(st.project.events) == 1
    ex.events.selectRow(0)
    assert st.selected_events == ["ev-0001"]

    # Analyze, and keep an eye on the event loop meanwhile.
    an = w.analyze_tab
    assert an.request().occlusion.baseline.value == (180, 105, 255)   # hot pink by default
    assert "rgb(255, 105, 180)" in an.colour.styleSheet()                # the swatch shows it
    an.patch.setValue(8)
    an.stride.setValue(4)
    an.region.setCurrentText("full")
    ticks = []
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start(10)
    an.run(selected_only=True)
    assert st.tasks.busy and an.b_cancel.isEnabled() and not an.b_sel.isEnabled()
    assert wait_until(qapp, lambda: not st.tasks.busy, timeout=120)
    timer.stop()
    assert an._last_run is not None and an._last_run.status == "completed", an.status.text()
    assert an.results.rowCount() == 1 and an.results.item(0, 2).text() == "ok"
    assert len(ticks) > 5          # the GUI thread kept running
    gaps = np.diff(ticks) if len(ticks) > 1 else np.array([0.0])
    assert gaps.max() < 1.0        # never blocked for long

    # View the result: heatmap, overlay, side by side.
    run_dir = str(an._last_run.run_dir)
    st.showResult.emit(run_dir, "ev-0001")
    assert w.tabs.currentWidget() is ex
    assert wait_until(qapp, lambda: ex._result is not None and ex.frame is not None)
    for mode, panels in (("Heatmap", 1), ("Overlay", 1), ("Side by side", 3)):
        ex.mode.setCurrentText(mode)
        ex.redraw()
        assert ex.view._panels == panels
    assert "Perturbation effect" in ex.info.toPlainText()
    assert "not a biological interpretation" in ex.info.toPlainText()

    # Grad-CAM through the same buttons.
    an.m_cam.setChecked(True)
    an.run(selected_only=True)
    assert wait_until(qapp, lambda: not st.tasks.busy, timeout=60)
    assert an._last_run.status == "completed" and an.results.item(0, 2).text() == "ok"

    # Statistics lists both runs; Settings knows the model.
    w.statistics_tab.refresh()
    assert w.statistics_tab.run.count() == 2
    assert w.statistics_tab.table.rowCount() == 1
    w.settings_tab.refresh()
    assert "capabilities" in w.settings_tab.text.toPlainText()
    w.close()


def test_cancel_from_the_gui(qapp, tmp_path):
    from yoru_explain import api
    from yoru_explain.gui.main_window import MainWindow
    from yoru_explain.io.project import Project

    video = _video(tmp_path / "clip.avi", n=10)
    project = Project.create(tmp_path / "proj")
    sid = project.add_source(video)["source_id"]
    mid = project.add_model(None, "toy-grid").model_id
    adapter = api.load_adapter("toy-grid", device="cpu")
    for frame in range(5):
        api.add_event(project, adapter, mid, sid, frame, 0)
    project.save()

    w = MainWindow()
    st = w.state
    st.set_project(Project.open(project.root))
    st.set_adapter(adapter, mid)
    an = w.analyze_tab
    an.patch.setValue(4)
    an.stride.setValue(2)
    an.region.setCurrentText("full")
    an.cache.setChecked(False)
    an.run(selected_only=False)
    an._cancel()
    assert wait_until(qapp, lambda: not st.tasks.busy, timeout=120)
    assert an._last_run.status == "cancelled"
    assert any(r.get("status") == "cancelled" for r in an._last_run.results)
    w.close()


def test_main_opens_and_closes(qapp, tmp_path):
    from yoru_explain.app import main

    assert main(str(tmp_path / "proj"), exit_after_ms=300) == 0
    assert (tmp_path / "proj" / "project.json").is_file()


def test_a_failed_time_index_is_reported_not_raised(qapp, tmp_path):
    from yoru_explain.gui.worker import FrameRequest, FrameService

    fs = FrameService()
    got, frames = [], []
    fs.indexReady.connect(lambda sid, summary: got.append((sid, summary)))
    fs.frameReady.connect(frames.append)
    fs.build_index(FrameRequest("src-x", str(tmp_path / "gone.mp4"), False, False, 0))
    assert wait_until(qapp, lambda: got, timeout=20)
    assert got[0][0] == "src-x" and got[0][1]["status"] == "failed" and not frames
    fs.shutdown()
