"""Which adapter explains which YORU model, and what each adapter refuses (no weights needed)."""

from __future__ import annotations

import dataclasses
import zipfile

import numpy as np
import pytest

from tests import synthetic as syn
from yoru_explain.adapters.base import AdapterCapabilities
from yoru_explain.core.errors import ErrorCode, ExplainError
from yoru_explain.core.registry import (
    BACKEND_ADAPTERS,
    BLACK_BOX,
    adapter_for_backend,
    create_adapter,
    list_adapters,
    resolve_model,
)


def test_every_yoru_backend_has_its_own_adapter():
    from yoru_explain.adapters import _yoru

    backends = [b for b in _yoru.core().list_detector_backends() if b != "auto"]
    assert set(backends) <= set(BACKEND_ADAPTERS), backends
    for b in backends:
        assert adapter_for_backend(b) != BLACK_BOX
    assert adapter_for_backend("yolov8") == "yoru-ultralytics"      # YORU's aliases
    assert adapter_for_backend("ssd") == "yoru-torchvision"
    assert adapter_for_backend("a-backend-added-later") == BLACK_BOX


def test_every_adapter_is_importable():
    for entry in list_adapters():
        adapter = create_adapter(entry.name)
        assert adapter.name == entry.name


def _yolov5_like_checkpoint(path):
    """A file YORU recognises as a YOLOv5 checkpoint by its pickle, whatever its name."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", b"\x80\x02cmodels.yolo\nDetectionModel\n")
    return path


@pytest.mark.parametrize("name,adapter,backend", [
    ("model.onnx", "yoru-onnx", "onnx"),
    ("my_rtdetr.pt", "yoru-rtdetr", "rtdetr"),
    ("fasterrcnn_best.pt", "yoru-torchvision", "torchvision"),
    ("ssd_best.pt", "yoru-torchvision", "torchvision"),
    ("yolo11_flies.pt", "yoru-ultralytics", "ultralytics"),
    ("best.pt", "yoru-ultralytics", "ultralytics"),
])
def test_auto_follows_yorus_choice(tmp_path, name, adapter, backend):
    path = tmp_path / name
    path.write_bytes(b"not a checkpoint")      # unreadable: YORU decides by the name
    assert resolve_model(path) == (adapter, backend, {})
    assert resolve_model(path, black_box=True) == (BLACK_BOX, backend, {"backend": backend})


def test_a_yolov5_checkpoint_is_recognised_by_its_contents(tmp_path):
    path = _yolov5_like_checkpoint(tmp_path / "yolov8_named_by_mistake.pt")
    assert resolve_model(path) == ("yoru-yolov5", "yolov5", {})
    assert resolve_model(path, "ultralytics") == ("yoru-ultralytics", "ultralytics", {})


def test_capabilities_say_which_modes_and_scores_there_are():
    caps = AdapterCapabilities(False, True, ("confidence", "cls_prob"), reported_score="cls_prob",
                               score_descriptions=(("cls_prob", "sigmoid of the query's logit"),))
    assert caps.identity_modes == ("redetect",)
    assert caps.reports("confidence") and caps.reports("cls_prob") and not caps.reports("cls_logit")
    assert caps.describe("cls_prob") == "sigmoid of the query's logit"
    assert caps.describe("cls_logit").startswith("class logit")      # the general definition
    d = caps.to_dict()
    assert d["identity_modes"] == ["redetect"] and set(d["score_definitions"]) == {"confidence", "cls_prob"}


class _NoFixedCandidate:
    """The synthetic detector, told it cannot follow a fixed candidate (as RT-DETR / Faster R-CNN)."""

    def __init__(self, toy):
        self._toy = toy
        self.capabilities = dataclasses.replace(toy.capabilities, fixed_candidate=False,
                                                candidate_kind="query", family="test")

    def __getattr__(self, name):
        return getattr(self._toy, name)


def test_a_request_the_adapter_cannot_honour_is_refused_before_a_run_folder(tmp_path, toy):
    import cv2

    from yoru_explain import api

    frame = syn.blank()
    syn.paint(frame, syn.window(5, 5), syn.RED)
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), frame)
    project = api.Project.create(tmp_path / "proj")
    sid = project.add_source(path)["source_id"]
    mid = project.add_model(None, "toy-grid").model_id
    ev = api.add_event(project, toy, mid, sid, 0, 0)
    adapter = _NoFixedCandidate(toy)
    with pytest.raises(ExplainError) as e:
        api.run_analysis(project, adapter, mid, [ev], api.AnalysisRequest("occlusion"))
    assert e.value.code == ErrorCode.MODEL_UNSUPPORTED and "redetect" in str(e.value)
    with pytest.raises(ExplainError) as e:
        api.run_analysis(project, toy, mid, [ev], api.AnalysisRequest(
            "occlusion", occlusion=api.OcclusionConfig(score=api.ScoreDefinition("cls_softmax"))))
    assert e.value.code == ErrorCode.MODEL_UNSUPPORTED
    assert project.run_dirs() == []
    # Re-detection is what such an adapter offers, and it runs.
    req = api.AnalysisRequest("occlusion", occlusion=api.OcclusionConfig(
        patch_size=32, stride=32, region="box", score=api.ScoreDefinition(identity_mode="redetect")))
    out = api.run_analysis(project, adapter, mid, [ev], req)
    assert out.status == "completed" and out.counts == {"ok": 1}


def test_occlusion_itself_refuses_an_unsupported_mode(toy):
    from yoru_explain import api

    frame = syn.blank()
    syn.paint(frame, syn.window(5, 5), syn.RED)
    target = syn.target_on(toy, frame, (72.0, 72.0, 104.0, 104.0))
    r = api.explain(_NoFixedCandidate(toy), frame, target, api.AnalysisRequest("occlusion"))
    assert r.status == "failed" and r.error_code == ErrorCode.MODEL_UNSUPPORTED.value


def test_confidence_is_the_reported_score(toy):
    frame = syn.blank()
    syn.paint(frame, syn.window(4, 9), syn.RED)
    p = toy.preprocess([frame])
    raw = toy.predict_raw(p)
    (d,) = toy.postprocess(raw, p)[0]
    for st in ("confidence", "obj_x_cls"):
        assert toy.candidate_readout(raw, p, d.candidate.index, d.class_id, st)[0].value == d.score
    assert toy.capabilities.reports("obj_x_cls")


def test_kept_needs_the_class_for_multi_label_models():
    from yoru_explain.analysis.scoring import _kept
    from yoru_explain.core.types import CandidateRef, Detection

    cand = CandidateRef(7)
    target = type("T", (), {"target_class": 1})()
    other_class = [Detection((0, 0, 1, 1), 0.9, 0, "a", CandidateRef(7))]
    assert _kept(other_class, cand, target, multi_label=False)      # YOLO: one label per candidate
    assert not _kept(other_class, cand, target, multi_label=True)   # SSD: that class is gone


def test_the_score_is_described_as_the_family_defines_it(toy):
    from yoru_explain import api
    from yoru_explain.core.pipeline import describe_score

    frame = syn.blank()
    syn.paint(frame, syn.window(5, 5), syn.RED)
    target = syn.target_on(toy, frame, (72.0, 72.0, 104.0, 104.0))
    text = describe_score(api.AnalysisRequest(), target, toy.class_names, toy)
    assert text.startswith("sigmoid(objectness) x sigmoid('red' logit)") and "grid cell" in text


def test_black_box_detections_carry_rotated_boxes():
    from yoru_explain.adapters.blackbox import detection_from_yoru, identity_transform

    upright = detection_from_yoru({"x1": 1, "y1": 2, "x2": 3, "y2": 4, "conf": 0.5, "class_id": 0,
                                   "class_name": "a"})
    rotated = detection_from_yoru({"x1": 1, "y1": 2, "x2": 3, "y2": 4, "conf": 0.5, "class_id": 0,
                                   "class_name": "a", "cx": 2, "cy": 3, "w": 2, "h": 1, "angle": 0.3})
    assert upright.obb is None and rotated.obb == (2.0, 3.0, 2.0, 1.0, 0.3)
    t = identity_transform((48, 64))
    np.testing.assert_array_equal(t.boxes_to_frame(np.array([[1.5, 2, 70, 3]])), [[1.5, 2, 70, 3]])
