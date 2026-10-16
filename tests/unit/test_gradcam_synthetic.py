"""Grad-CAM on the synthetic detector: gradients flow, the map lands in place,
and every check that should refuse a map does."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from tests import synthetic as syn
from yoru_explain.methods.gradcam import GradCamConfig, run_gradcam


def _frame_and_target(toy, h=200, w=320):
    frame = syn.blank(h, w)
    pad = (224 - h) // 2 if h == 200 else 0
    box = syn.window(5, 5, pad_top=pad)
    syn.paint(frame, box, syn.RED)
    syn.paint(frame, syn.window(5, 13, pad_top=pad), syn.GREEN)
    return frame, syn.target_on(toy, frame, tuple(float(v) for v in box)), box


@pytest.mark.parametrize("layer", ["pool", "mix", "auto"])
def test_gradcam_lands_on_the_red_square_in_frame_pixels(toy, layer):
    frame, target, box = _frame_and_target(toy)  # letterboxed: 12 px of padding on top
    r = run_gradcam(toy, frame, target, GradCamConfig(layer=layer))
    assert r.ok, r.message
    assert r.gradient_score == r.reference_score == target.detection.score
    assert r.gradient_support == 1           # one candidate, one cell
    assert r.cam.shape == frame.shape[:2] and np.isfinite(r.cam).all()
    assert np.isfinite(r.cam_feature).all()
    ys, xs = np.nonzero(r.cam > 0.5 * r.cam.max())
    assert box[0] <= xs.min() and xs.max() < box[2]
    assert box[1] <= ys.min() and ys.max() < box[3]
    yy, xx = np.indices(r.cam.shape)
    cy, cx = (yy * r.cam).sum() / r.cam.sum(), (xx * r.cam).sum() / r.cam.sum()
    assert abs(cx - (box[0] + box[2]) / 2) < 4 and abs(cy - (box[1] + box[3]) / 2) < 4


def test_gradcam_changes_no_weights_and_no_mode(toy):
    frame, target, _ = _frame_and_target(toy)
    before = {k: v.clone() for k, v in toy.torch_module.state_dict().items()}
    dets_before = toy.detect(frame)
    assert run_gradcam(toy, frame, target, GradCamConfig()).ok
    after = toy.torch_module.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert not any(m.training for m in toy.torch_module.modules())
    assert toy.detect(frame) == dets_before
    assert all(p.grad is None for p in toy.torch_module.parameters())


def test_without_a_verified_gradient_path_there_is_no_map(toy):
    frame, target, _ = _frame_and_target(toy)
    toy._caps = replace(toy.capabilities, gradients=False, gradients_reason="probe failed")
    r = run_gradcam(toy, frame, target, GradCamConfig())
    assert r.status == "failed" and r.error_code == "gradients_unavailable" and r.cam is None


def test_without_a_candidate_there_is_no_map(toy):
    frame, target, _ = _frame_and_target(toy)
    target = target.with_detection(replace(target.detection, candidate=None), 0)
    r = run_gradcam(toy, frame, target, GradCamConfig())
    assert r.error_code == "score_unavailable" and r.cam is None


def test_an_unknown_layer_is_refused(toy):
    frame, target, _ = _frame_and_target(toy)
    r = run_gradcam(toy, frame, target, GradCamConfig(layer="head.conv9"))
    assert r.error_code == "invalid_config" and r.cam is None


def test_a_score_that_differs_from_inference_is_refused(toy, monkeypatch):
    frame, target, _ = _frame_and_target(toy)
    original = toy.candidate_score_tensor
    monkeypatch.setattr(toy, "candidate_score_tensor", lambda *a: original(*a) * 0.9)
    r = run_gradcam(toy, frame, target, GradCamConfig())
    assert r.error_code == "score_mismatch" and r.cam is None


def test_a_non_finite_gradient_is_refused(toy, monkeypatch):
    frame, target, _ = _frame_and_target(toy)
    original = toy.forward_with_gradients

    def poisoned(prepared, layers, precision="inference"):
        raw = original(prepared, layers, precision=precision)
        raw.activations = {k: v * float("nan") for k, v in raw.activations.items()}
        return raw

    monkeypatch.setattr(toy, "forward_with_gradients", poisoned)
    r = run_gradcam(toy, frame, target, GradCamConfig())
    assert r.status == "failed" and r.cam is None


def test_the_signed_cam_keeps_negative_evidence(toy):
    frame, target, _ = _frame_and_target(toy)
    r = run_gradcam(toy, frame, target, GradCamConfig(layer="pool"))
    # Green lowers the red candidate's score: negative before the ReLU.
    assert r.cam_feature.min() < 0 < r.cam_feature.max()
    assert r.cam.min() >= 0
