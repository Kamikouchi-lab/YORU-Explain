"""Data types and settings survive JSON, and nothing unknown slips through."""

from __future__ import annotations

import json

import pytest

from yoru_explain.core.pipeline import AnalysisRequest
from yoru_explain.core.types import (
    CandidateRef,
    Detection,
    ExplanationTarget,
    FrameRef,
    ScoreDefinition,
    ScoreResult,
    ScoreStatus,
)
from yoru_explain.methods.baselines import BaselineSpec
from yoru_explain.methods.gradcam import GradCamConfig
from yoru_explain.methods.occlusion import OcclusionConfig


def _target():
    cand = CandidateRef(14774, (("level", 2), ("stride", 32), ("anchor", 1), ("grid_y", 6), ("grid_x", 14)))
    det = Detection((1.5, 2.25, 30.0, 40.125), 0.857316255569458, 0, "person", cand)
    frame = FrameRef("src-001", "C:/v.mp4", "video", 120, 1280, 720, 4000.0, True, False, "ab" * 20,
                     "seek_verified")
    from yoru_explain.core.geometry import LetterboxTransform

    transform = LetterboxTransform.yolov5((720, 1280), (384, 640)).to_dict()
    return ExplanationTarget("ev-0001", frame, det, 0, 0, model_id="mdl-001", track_id=3,
                             transform=transform)


def test_target_round_trips_through_json():
    t = _target()
    back = ExplanationTarget.from_dict(json.loads(json.dumps(t.to_dict())))
    assert back == t
    assert back.candidate.locator_dict["grid_x"] == 14
    assert back.transform["pad_top"] == 12 and back.transform["input_shape"] == [384, 640]


def test_configs_round_trip():
    occ = OcclusionConfig(patch_size=(24, 40), stride=8, baseline=BaselineSpec("blur", sigma=3.0),
                          region="full", score=ScoreDefinition("cls_logit", "redetect", 0.3),
                          batch_size=4, seed=7, include_ambiguous=True)
    assert OcclusionConfig.from_dict(json.loads(json.dumps(occ.to_dict()))) == occ
    assert occ.stride == (8, 8)
    g = GradCamConfig(layer="model.17", score_type="obj_prob", precision="fp32")
    assert GradCamConfig.from_dict(g.to_dict()) == g
    for req in (AnalysisRequest("occlusion", occlusion=occ, use_cache=False),
                AnalysisRequest("gradcam", gradcam=g, verify_identity=False)):
        assert AnalysisRequest.from_dict(json.loads(json.dumps(req.to_dict()))) == req


@pytest.mark.parametrize("bad", [
    lambda: ScoreDefinition("max_score"),
    lambda: ScoreDefinition(identity_mode="best_box"),
    lambda: ScoreDefinition(iou_threshold=0.0),
    lambda: OcclusionConfig(patch_size=0),
    lambda: OcclusionConfig(region="everything"),
    lambda: OcclusionConfig(context_scale=0.5),
    lambda: OcclusionConfig(batch_size=0),
    lambda: BaselineSpec("paint"),
    lambda: BaselineSpec("constant", value=(0, 0, 300)),
    lambda: GradCamConfig(precision="fp8"),
    lambda: AnalysisRequest("lime"),
])
def test_invalid_settings_are_refused(bad):
    with pytest.raises(ValueError):
        bad()


def test_unknown_keys_are_refused_not_ignored():
    with pytest.raises(ValueError, match="unknown"):
        OcclusionConfig.from_dict({"patch": 16})
    with pytest.raises(ValueError, match="unknown"):
        GradCamConfig.from_dict({"layers": "x"})
    with pytest.raises(ValueError, match="unknown"):
        ScoreDefinition.from_dict({"score": "obj_x_cls"})
    with pytest.raises(ValueError, match="unknown"):
        AnalysisRequest.from_dict({"method": "occlusion", "speed": 2})


def test_a_lost_target_has_no_score():
    lost = ScoreResult(ScoreStatus.DISAPPEARED.value)
    assert lost.value is None and not lost.scored
    ambiguous = ScoreResult(ScoreStatus.AMBIGUOUS.value, value=0.4)
    assert not ambiguous.scored  # reported, but not a score of the target
    assert ScoreResult(ScoreStatus.MATCHED.value, value=0.4).scored
