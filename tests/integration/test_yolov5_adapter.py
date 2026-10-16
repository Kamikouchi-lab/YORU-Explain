"""The YOLOv5 adapter against YORU itself, on real models.

Needs YORU's bundled YOLOv5 and weights: ``yolov5s.pt`` (upstream COCO model)
and the fly fixture model, from ``.cache/models`` or the environment
variables in ``tests/conftest.py``.  Skipped, not failed, without them.

What "the same as YORU" means here: for each frame, the adapter's detections
equal ``yoru.libs.plugins.get_detector(...).detect(frame)`` bit for bit, on
every device present.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from tests.conftest import fly_video, fly_weights, yolov5s_weights

pytestmark = pytest.mark.slow

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
FLY_FRAMES = (0, 150, 4350, 6450, 8700)


def _need(path, what):
    if path is None:
        pytest.skip(f"{what} not available (see tests/conftest.py)")
    return path


def _adapter(weights, device, **kw):
    from yoru_explain.adapters.yolov5 import YoruYolov5Adapter

    a = YoruYolov5Adapter()
    a.load_model(weights, device=device, **kw)
    return a


@pytest.fixture(scope="module")
def zidane():
    import cv2

    from yoru_explain.adapters import _yoru

    return cv2.imread(_yoru.load().yolov5_dir + "/data/images/zidane.jpg")


@pytest.fixture(scope="module")
def fly_frames():
    from yoru_explain.io.frames import VideoFrameReader

    path = _need(fly_video(), "YORU's test video")
    with VideoFrameReader(path) as r:
        return {i: r.read(i).image for i in FLY_FRAMES}


@pytest.fixture(scope="module", params=DEVICES)
def fly(request):
    return _adapter(_need(fly_weights(), "the fly fixture model"), request.param)


@pytest.fixture(scope="module", params=DEVICES)
def coco(request):
    return _adapter(_need(yolov5s_weights(), "yolov5s.pt"), request.param)


def test_coco_detections_are_yorus_exactly(coco, zidane):
    check = coco.identity_check(zidane)
    assert check["status"] == "passed" and check["exact"], check
    assert check["n_adapter"] >= 2


def test_fly_detections_are_yorus_exactly(fly, fly_frames):
    for i, frame in fly_frames.items():
        check = fly.identity_check(frame)
        assert check["status"] == "passed" and check["exact"], (i, check)


def test_every_detection_has_its_candidate_and_its_score(fly, fly_frames):
    for frame in fly_frames.values():
        p = fly.preprocess([frame])
        raw = fly.predict_raw(p, logits=True)
        dets = fly.postprocess(raw, p)[0]
        assert dets
        for d in dets:
            assert d.candidate is not None
            r = fly.candidate_readout(raw, p, d.candidate.index, d.class_id, "obj_x_cls")[0]
            assert r.value == d.score                     # bit for bit
            assert r.argmax_class == d.class_id
            np.testing.assert_allclose(r.box, d.box, atol=1e-3)
            k = d.candidate.index
            # The head's logits, through the head's own (fp16 on CUDA) sigmoid.
            assert torch.equal(raw.logits[0, k, 4:].sigmoid().float(), raw.pred[0, k, 4:].float())


def test_a_higher_threshold_is_applied_after_nms_as_yoru_does(fly_frames):
    w = _need(fly_weights(), "the fly fixture model")
    strict = _adapter(w, "cpu", conf_thresh=0.85, probe=False)
    for frame in fly_frames.values():
        check = strict.identity_check(frame)
        assert check["status"] == "passed" and check["exact"], check
    assert strict.model_info()["nms_conf"] == 0.25


def test_rgb_mode_is_yorus_rgb_mode(zidane):
    w = _need(yolov5s_weights(), "yolov5s.pt")
    bgr = _adapter(w, "cpu", rgb_input=False, probe=False)
    rgb = _adapter(w, "cpu", rgb_input=True, probe=False)
    assert rgb.identity_check(zidane)["exact"]
    assert [d.score for d in rgb.detect(zidane)] != [d.score for d in bgr.detect(zidane)]
    assert rgb.model_info()["channel_order"] == "RGB"


def test_capabilities_are_measured(fly):
    caps = fly.capabilities
    assert caps.deterministic is True
    assert caps.gradients, caps.gradients_reason
    assert caps.batch_max_abs_diff is not None
    assert "model.17" in caps.gradient_layers and "model.23" in caps.gradient_layers


def _state_hash(module):
    h = hashlib.sha256()
    for k, v in module.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def test_gradcam_on_a_real_model(fly, fly_frames):
    from yoru_explain import api

    frame = fly_frames[150]
    dets = fly.detect(frame)
    target = api.target_from_detection(dets[0], 0, frame)
    before = _state_hash(fly.torch_module)
    for layer in ("auto", "model.17", "model.9"):
        r = api.explain(fly, frame, target, api.AnalysisRequest(
            "gradcam", gradcam=api.GradCamConfig(layer=layer)))
        assert r.ok, r.message
        assert r.gradient_score == r.reference_score == dets[0].score
        assert r.cam.shape == frame.shape[:2] and np.isfinite(r.cam).all()
        if layer == "auto":
            assert r.gradient_support == 1     # the 1x1 head: one cell
        else:
            assert r.gradient_support > 1
    assert _state_hash(fly.torch_module) == before
    assert not any(m.training for m in fly.torch_module.modules())
    assert all(p.grad is None for p in fly.torch_module.parameters())
    assert fly.detect(frame) == dets          # inference unchanged afterwards


@pytest.mark.parametrize("batch_size", [1, 4])
def test_occlusion_controls_on_a_real_model(fly, fly_frames, batch_size):
    from yoru_explain import api

    frame = fly_frames[150]
    target = api.target_from_detection(fly.detect(frame)[0], 0, frame)
    cfg = api.OcclusionConfig(patch_size=32, stride=32, region="box", batch_size=batch_size)
    a = api.explain(fly, frame, target, api.AnalysisRequest("occlusion", occlusion=cfg))
    b = api.explain(fly, frame, target, api.AnalysisRequest("occlusion", occlusion=cfg))
    assert a.ok and b.ok
    assert a.controls["noop"].delta == 0.0
    assert a.reference_score == target.detection.score
    assert [p.delta for p in a.patches] == [p.delta for p in b.patches]
    assert all(p.score.status == "ok" for p in a.patches)
    np.testing.assert_array_equal(a.delta_map, b.delta_map)


def test_project_pipeline_on_the_test_video(tmp_path):
    from yoru_explain import api
    from yoru_explain.io.runs import RunReader

    weights = _need(fly_weights(), "the fly fixture model")
    video = _need(fly_video(), "YORU's test video")
    device = DEVICES[-1]
    adapter = _adapter(weights, device)
    project = api.Project.create(tmp_path / "proj")
    sid = project.add_source(video)["source_id"]
    mid = project.add_model(weights, "yoru-yolov5").model_id
    events = [api.add_event(project, adapter, mid, sid, 150, 1),
              api.add_event(project, adapter, mid, sid, 6450, 0)]
    project.save()
    req = api.AnalysisRequest("occlusion", occlusion=api.OcclusionConfig(
        patch_size=32, stride=16, region="box"), use_cache=False)
    first = api.run_analysis(project, adapter, mid, events, req)
    assert first.status == "completed" and first.counts == {"ok": 2}
    assert first.identity_check["status"] == "passed" and first.identity_check["exact"]
    run = RunReader(first.run_dir)
    assert run.manifest["model"]["sha256"] == project.model(mid).sha256
    assert run.manifest["inputs"][0]["verify"]["status"] == "ok"
    second = api.run_analysis(project, adapter, mid, events,
                              api.AnalysisRequest.from_dict(run.config["request"]))
    again = RunReader(second.run_dir)
    for a, b in zip(run.results, again.results):
        assert a["delta_score"] == b["delta_score"] and a["original_score"] == b["original_score"]
        np.testing.assert_array_equal(run.load_map(a["map_file"])["delta_map"],
                                      again.load_map(b["map_file"])["delta_map"])


def test_every_image_of_a_batch_gets_its_own_nms(fly, fly_frames):
    frames = [fly_frames[i] for i in (0, 150, 6450)]
    p = fly.preprocess(frames)
    batch = fly.postprocess(fly.predict_raw(p), p)
    for frame, dets in zip(frames, batch):
        single = fly.detect(frame)
        assert len(dets) == len(single) > 0
        for d, s in zip(dets, single):
            # Scores move slightly with the batch size (measured); boxes stay.
            assert d.class_id == s.class_id and abs(d.score - s.score) < 1e-2
            np.testing.assert_allclose(d.box, s.box, atol=2.0)
