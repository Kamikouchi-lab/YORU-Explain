"""Without a GPU everything still runs (on the CPU), and nothing asks for CUDA."""

from __future__ import annotations

import numpy as np
import torch

from tests import synthetic as syn
from yoru_explain import api


def test_the_whole_core_runs_with_cuda_reported_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    adapter = api.load_adapter("toy-grid", device="auto")
    assert adapter.model_info()["device"] == "cpu"
    frame = syn.blank()
    syn.paint(frame, syn.window(5, 5), syn.RED)
    target = syn.target_on(adapter, frame, syn.window(5, 5))
    occ = api.explain(adapter, frame, target, api.AnalysisRequest(
        "occlusion", occlusion=api.OcclusionConfig(patch_size=16, stride=16)))
    cam = api.explain(adapter, frame, target, api.AnalysisRequest("gradcam"))
    assert occ.ok and cam.ok
    assert np.isfinite(cam.cam).all()


def test_the_yolov5_autocast_is_disabled_without_cuda(monkeypatch):
    from yoru_explain.adapters import yolov5

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with yolov5._autocast():
        assert not torch.is_autocast_enabled("cuda")
