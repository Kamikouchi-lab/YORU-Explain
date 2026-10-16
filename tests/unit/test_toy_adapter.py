"""The synthetic detector itself: boxes, candidates, scores as designed."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests import synthetic as syn


def test_squares_are_detected_where_they_are(toy):
    frame = syn.blank()
    syn.paint(frame, syn.window(5, 5), syn.RED)
    syn.paint(frame, syn.window(5, 13), syn.GREEN)
    dets = toy.detect(frame)
    assert [(d.class_name, d.box) for d in dets] == [
        ("red", (72.0, 72.0, 104.0, 104.0)), ("green", (200.0, 72.0, 232.0, 104.0))]
    loc = dets[0].candidate.locator_dict
    assert (loc["grid_y"], loc["grid_x"], loc["stride"]) == (5, 5, 16)
    # sigmoid(20 * 0.25) * sigmoid(20 * 1)
    assert dets[0].score == pytest.approx(1 / (1 + np.exp(-5)) / (1 + np.exp(-20)), rel=1e-6)


def test_a_padded_frame_maps_boxes_back_to_frame_pixels(toy):
    frame = syn.blank(200, 320)  # letterboxed to 224: 12 px on top
    syn.paint(frame, syn.window(5, 5, pad_top=12), syn.RED)
    (d,) = toy.detect(frame)
    assert d.box == (72.0, 60.0, 104.0, 92.0)


def test_the_candidate_readout_is_the_reported_score(toy):
    frame = syn.blank()
    syn.paint(frame, syn.window(4, 9), syn.RED)
    p = toy.preprocess([frame])
    raw = toy.predict_raw(p)
    (d,) = toy.postprocess(raw, p)[0]
    r = toy.candidate_readout(raw, p, d.candidate.index, d.class_id, "obj_x_cls")[0]
    assert r.value == d.score and r.box == d.box and r.argmax_class == 0
    lg = raw.logits[0, d.candidate.index]
    assert torch.equal(lg[4:].sigmoid(), raw.pred[0, d.candidate.index, 4:])
    obj = toy.candidate_readout(raw, p, d.candidate.index, 0, "obj_logit")[0].value
    assert obj == pytest.approx(20 * (1.0 - 0.75))


def test_probe_reports_measured_capabilities(toy):
    caps = toy.capabilities
    assert caps.deterministic and caps.batch_invariant and caps.gradients
    assert caps.fixed_candidate and caps.redetection
    assert toy.model_info()["synthetic"] is True


def test_no_weights_are_accepted(toy):
    from yoru_explain.adapters.toy import ToyGridAdapter
    from yoru_explain.core.errors import ExplainError

    with pytest.raises(ExplainError):
        ToyGridAdapter().load_model("best.pt")
