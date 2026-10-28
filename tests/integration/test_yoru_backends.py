"""Every YORU backend against YORU itself, on real models.

The models are fixtures in ``.cache/models`` (``tests/prepare_models.py``
fetches or builds them); a test whose model is missing is skipped, not
failed.  For each model and device:

* the adapter's detections equal ``yoru.libs.plugins.get_detector(...).detect``
  bit for bit, rotated boxes included;
* every detection has its candidate, and reading that candidate gives the
  reported confidence bit for bit;
* the gradient pass reproduces the score, and leaves weights, modes and
  inference as they were;
* occlusion's controls hold in every identity mode the adapter offers, and
  the modes it does not offer are refused.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest
import torch

from tests.conftest import MODELS, fly_video

pytestmark = pytest.mark.slow

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
BOTH = ("fixed_candidate", "redetect")
REDETECT = ("redetect",)

#: file, adapter, identity modes, gradients
CASES = [
    ("yolo11n.pt", "yoru-ultralytics", BOTH, True),
    ("yolov8n.pt", "yoru-ultralytics", BOTH, True),
    ("yolov5nu.pt", "yoru-ultralytics", BOTH, True),
    ("yolov10n.pt", "yoru-ultralytics", BOTH, True),
    ("yolo11n-obb.pt", "yoru-ultralytics", BOTH, True),
    ("yolo11n-seg.pt", "yoru-ultralytics", BOTH, True),
    ("yolo11n-pose.pt", "yoru-ultralytics", BOTH, True),
    ("rtdetr-l.pt", "yoru-rtdetr", REDETECT, True),
    ("coco_fasterrcnn.pt", "yoru-torchvision", REDETECT, True),
    ("coco_maskrcnn.pt", "yoru-torchvision", REDETECT, True),
    ("coco_ssd.pt", "yoru-torchvision", BOTH, True),
    ("yolov5s.onnx", "yoru-onnx", BOTH, False),
    ("yolo11n.onnx", "yoru-onnx", BOTH, False),
]
IDS = [c[0] for c in CASES]


def _model(name):
    path = MODELS / name
    if not path.is_file():
        pytest.skip(f"{name} not in .cache/models (run tests/prepare_models.py)")
    return path


@pytest.fixture(scope="module")
def images():
    import cv2

    ul = pytest.importorskip("ultralytics")
    assets = os.path.join(os.path.dirname(ul.__file__), "assets")
    out = {name: cv2.imread(os.path.join(assets, f"{name}.jpg")) for name in ("bus", "zidane")}
    video = fly_video()
    if video is not None:
        from yoru_explain.io.frames import VideoFrameReader

        with VideoFrameReader(video) as r:
            out["fly150"] = r.read(150).image
    return out


_loaded = {}


def _adapter(name, device):
    """One loaded adapter per (model, device) for the whole module."""
    key = (name, device)
    if key not in _loaded:
        from yoru_explain import api

        _loaded[key] = api.load_model(_model(name), device=device)
    return _loaded[key]


def _state_hash(module):
    h = hashlib.sha256()
    for k, v in module.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _best(adapter, images):
    """``(image, detections, index)`` of the highest-scoring detection on any test image."""
    best = None
    for im in images.values():
        dets = adapter.detect(im)
        for i, d in enumerate(dets):
            if best is None or d.score > best[1][best[2]].score:
                best = (im, dets, i)
    if best is None:
        pytest.skip("the model detects nothing on the test images")
    return best


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,adapter_name,modes,gradients", CASES, ids=IDS)
def test_detections_are_yorus_exactly(name, adapter_name, modes, gradients, device, images):
    a = _adapter(name, device)
    assert a.name == adapter_name                      # api.load_model chose as YORU would
    n = 0
    for key, im in images.items():
        check = a.identity_check(im)
        assert check["status"] == "passed" and check["exact"], (key, check)
        n += check["n_adapter"]
    assert n > 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,adapter_name,modes,gradients", CASES, ids=IDS)
def test_every_detection_has_its_candidate_and_its_score(name, adapter_name, modes, gradients, device,
                                                         images):
    a = _adapter(name, device)
    multi = a.capabilities.multi_label
    for im in images.values():
        p = a.preprocess([im])
        raw = a.predict_raw(p)
        for d in a.postprocess(raw, p)[0]:
            assert d.candidate is not None, d
            r = a.candidate_readout(raw, p, d.candidate.index, d.class_id, "confidence")[0]
            assert r.value == d.score                  # bit for bit
            np.testing.assert_allclose(r.box, d.box, atol=1e-3)
            if not multi:
                assert r.argmax_class == d.class_id
            if d.obb is not None:
                np.testing.assert_allclose(r.obb, d.obb, atol=1e-4)


@pytest.mark.parametrize("name,adapter_name,modes,gradients", CASES, ids=IDS)
def test_capabilities_are_measured(name, adapter_name, modes, gradients, images):
    a = _adapter(name, DEVICES[-1])
    caps = a.capabilities
    assert caps.identity_modes == modes
    assert caps.gradients is gradients, caps.gradients_reason
    assert caps.deterministic is True
    assert caps.family and caps.candidate_kind
    assert "confidence" in caps.score_types and caps.reports(caps.reported_score)
    if gradients:
        assert caps.gradient_layers
    if name.endswith("-obb.pt"):
        assert all(d.obb is not None for im in images.values() for d in a.detect(im))


@pytest.mark.parametrize("name,adapter_name,modes,gradients", [c for c in CASES if c[3]],
                         ids=[c[0] for c in CASES if c[3]])
def test_gradcam_reproduces_the_score_and_leaves_the_model_alone(name, adapter_name, modes, gradients,
                                                                 images):
    from yoru_explain import api

    a = _adapter(name, DEVICES[-1])
    im, dets, i = _best(a, images)
    target = api.target_from_detection(dets[i], i, im)
    before = _state_hash(a.torch_module)
    r = api.explain(a, im, target, api.AnalysisRequest("gradcam"))
    assert r.ok, r.message
    assert r.gradient_score == r.reference_score == dets[i].score
    assert r.gradient_support and r.gradient_support > 0
    assert r.cam.shape == im.shape[:2] and np.isfinite(r.cam).all()
    assert _state_hash(a.torch_module) == before
    assert not any(m.training for m in a.torch_module.modules())
    assert all(p.grad is None for p in a.torch_module.parameters())
    assert a.detect(im) == dets                         # inference unchanged afterwards


@pytest.mark.parametrize("name,adapter_name,modes,gradients", CASES, ids=IDS)
def test_occlusion_controls_in_every_offered_mode(name, adapter_name, modes, gradients, images):
    from yoru_explain import api

    a = _adapter(name, DEVICES[-1])
    im, dets, i = _best(a, images)
    target = api.target_from_detection(dets[i], i, im)
    for mode in BOTH:
        cfg = api.OcclusionConfig(patch_size=96, stride=96, region="box",
                                  score=api.ScoreDefinition(identity_mode=mode))
        r = api.explain(a, im, target, api.AnalysisRequest("occlusion", occlusion=cfg))
        if mode not in modes:
            assert r.status == "failed" and r.error_code == "model_unsupported"
            continue
        assert r.ok, r.message
        assert r.reference_score == dets[i].score       # batch 1: YORU's confidence
        assert r.controls["noop"].delta == 0.0
        assert r.patches and np.isfinite([p.delta for p in r.patches if p.counted]).all()


def test_black_box_is_yorus_own_detector(images):
    from yoru_explain import api

    path = _model("yolo11n.pt")
    a = api.load_model(path, device=DEVICES[-1], black_box=True)
    assert a.name == "yoru-detector" and a.model_info()["backend"] == "ultralytics"
    assert a.identity_check(images["bus"])["status"] == "not_applicable"
    white = _adapter("yolo11n.pt", DEVICES[-1])
    for im in images.values():
        mine = [(d.box, d.score, d.class_id) for d in a.detect(im)]
        assert mine == [(d.box, d.score, d.class_id) for d in white.detect(im)]
    im, dets, i = _best(a, images)
    target = api.target_from_detection(dets[i], i, im)
    cfg = api.OcclusionConfig(patch_size=96, stride=96, region="box",
                              score=api.ScoreDefinition(identity_mode="redetect"))
    r = api.explain(a, im, target, api.AnalysisRequest("occlusion", occlusion=cfg))
    assert r.ok and r.controls["noop"].delta == 0.0 and r.reference_score == dets[i].score
    r = api.explain(a, im, target, api.AnalysisRequest("occlusion"))
    assert r.status == "failed" and r.error_code == "model_unsupported"


@pytest.mark.parametrize("name", ["yolo11n.pt", "coco_ssd.pt", "rtdetr-l.pt"])
def test_every_image_of_a_batch_gets_its_own_postprocessing(name, images):
    """Each image of a batch keeps its own detections (scores move with the batch size, measured)."""
    from yoru_explain.core.geometry import iou

    a = _adapter(name, DEVICES[-1])
    frames = [images["bus"], images["bus"][:, ::-1].copy(), images["bus"][::-1].copy()]
    p = a.preprocess(frames)
    batch = a.postprocess(a.predict_raw(p), p)
    checked = 0
    for frame, dets in zip(frames, batch):
        for s in (s for s in a.detect(frame) if s.score >= 0.5):
            same = [d for d in dets if d.class_id == s.class_id and iou(d.box, s.box) >= 0.9]
            assert len(same) == 1 and abs(same[0].score - s.score) < 0.05, (s, same)
            checked += 1
    assert checked > 0


def test_a_project_run_with_rotated_boxes(tmp_path, images):
    import cv2

    from yoru_explain import api
    from yoru_explain.io.runs import RunReader

    a = _adapter("yolo11n-obb.pt", DEVICES[-1])
    path = tmp_path / "bus.png"
    cv2.imwrite(str(path), images["bus"])
    project = api.Project.create(tmp_path / "proj")
    sid = project.add_source(path)["source_id"]
    mid = project.add_model(_model("yolo11n-obb.pt"), a.name).model_id
    ev = api.add_event(project, a, mid, sid, 0, 0)
    project.save()
    assert project.target(ev).obb is not None
    req = api.AnalysisRequest("occlusion", occlusion=api.OcclusionConfig(
        patch_size=64, stride=64, region="context", score=api.ScoreDefinition(identity_mode="redetect")),
        use_cache=False)
    out = api.run_analysis(project, a, mid, [ev], req)
    assert out.status == "completed" and out.counts == {"ok": 1}
    assert out.identity_check["status"] == "passed" and out.identity_check["exact"]
    run = RunReader(out.run_dir)
    (row,) = run.typed_results()
    assert row["obb_angle"] is not None and row["obb_w"] > 0
    assert run.manifest["adapter"]["capabilities"]["family"].startswith("ultralytics YOLO (OBB")
    assert run.config["score_types"]["confidence"].startswith("sigmoid(class logit)")
    rows = run.perturbations(ev)
    assert any(r["matched_angle"] for r in rows if r["status"] == "matched")


def test_an_end_to_end_checkpoint_run_with_nms_is_flagged():
    """YOLO26-style checkpoints need a newer ultralytics; YORU runs them with NMS regardless."""
    from yoru_explain import api

    path = MODELS / "yolo26n.pt"
    if not path.is_file():
        pytest.skip("yolo26n.pt not in .cache/models")
    a = api.load_model(path, device="cpu", probe=False)
    if a.model_info()["end2end"]:
        pytest.skip("this ultralytics runs YOLO26 end to end")
    assert any("end-to-end head" in w for w in a.capabilities.warnings)
    assert not _adapter("yolo11n.pt", DEVICES[-1]).capabilities.warnings
