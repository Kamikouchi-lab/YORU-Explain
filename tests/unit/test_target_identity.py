"""Following *the* detection, not the class's best box; every lost-target state.

Two animals of one class are the normal case in YORU's videos.  These tests
pin down that a perturbation of animal A is measured on A's own score, and
that each way of losing A in re-detection mode is a named state with no score.
"""

from __future__ import annotations

import numpy as np

from tests import synthetic as syn
from yoru_explain.core.types import ScoreDefinition
from yoru_explain.methods.occlusion import OcclusionConfig, run_occlusion

A = syn.window(5, 5)          # (72, 72, 104, 104)
B = syn.window(5, 13)         # (200, 72, 232, 104), same class


def _two_reds():
    frame = syn.blank()
    syn.paint(frame, A, syn.RED)
    syn.paint(frame, B, syn.RED)
    return frame


def _scores(toy, target, images, mode, iou_threshold=0.5):
    p = toy.preprocess(images)
    raw = toy.predict_raw(p)
    return toy.score_target(raw, p, target, ScoreDefinition(identity_mode=mode,
                                                            iou_threshold=iou_threshold))


def test_occluding_a_lowers_as_score_even_though_b_still_scores_high(toy):
    frame = _two_reds()
    target = syn.target_on(toy, frame, A)
    r = run_occlusion(toy, frame, target, OcclusionConfig(patch_size=16, stride=16, region="full"))
    assert r.ok
    over_a = [p for p in r.patches if syn.overlaps(p.patch.box, A)]
    assert all(p.delta > 0 for p in over_a)
    # A patch inside A hides a quarter of its window: sigmoid(5) -> sigmoid(0).
    assert max(p.delta for p in over_a) > 0.4
    # "Best red box anywhere" -- the comparison that must not be made -- would
    # see B's unchanged score and report (almost) no effect for the same patches.
    for p in over_a:
        img = frame.copy()
        x0, y0, x1, y1 = p.patch.box
        img[y0:y1, x0:x1] = 0
        best_red = max(d.score for d in toy.detect(img) if d.class_id == 0)
        assert r.reference_score - best_red < 1e-6
    # And B's patches do nothing to A.
    over_b = [p for p in r.patches if syn.overlaps(p.patch.box, B)]
    assert all(p.delta == 0.0 for p in over_b)


def test_fixed_candidate_follows_the_slot_and_says_when_nms_drops_it(toy):
    frame = _two_reds()
    target = syn.target_on(toy, frame, A)
    gone = frame.copy()
    syn.paint(gone, A, (0, 0, 0))
    (here, lost) = _scores(toy, target, [frame, gone], "fixed_candidate")
    assert here.status == "ok" and here.candidate_kept and here.value == target.detection.score
    assert lost.status == "ok" and lost.value < 1e-6 and lost.candidate_kept is False
    assert lost.candidate_index == target.candidate.index


def test_redetection_states(toy):
    frame = _two_reds()
    target = syn.target_on(toy, frame, A)
    covered = frame.copy()
    syn.paint(covered, A, (0, 0, 0))                    # A gone, B left
    green = frame.copy()
    syn.paint(green, A, syn.GREEN)                       # A relabelled
    shifted = frame.copy()
    syn.paint(shifted, A, (0, 0, 0))
    syn.paint(shifted, syn.window(5, 6), syn.RED)        # A one cell to the right: IoU 1/3
    doubled = frame.copy()
    syn.paint(doubled, (72, 72, 120, 104), syn.RED)      # A widened: two cells detect it
    states = _scores(toy, target, [frame, covered, green, shifted, doubled], "redetect")
    assert [s.status for s in states] == ["matched", "disappeared", "class_changed", "unmatched",
                                          "matched"]
    assert states[0].value == target.detection.score
    for s in states[1:4]:
        assert s.value is None and not s.scored   # unknown, not zero
    # With a looser IoU threshold the widened case has two candidates for "A".
    loose = _scores(toy, target, [doubled], "redetect", iou_threshold=0.3)[0]
    assert loose.status == "ambiguous" and loose.n_overlapping == 2 and not loose.scored


def test_occlusion_maps_count_only_scored_patches(toy):
    frame = _two_reds()
    target = syn.target_on(toy, frame, A)
    r = run_occlusion(toy, frame, target, OcclusionConfig(
        patch_size=32, stride=8, region="context", score=ScoreDefinition(identity_mode="redetect")))
    statuses = {p.score.status for p in r.patches}
    assert "disappeared" in statuses and "matched" in statuses
    lost = [p for p in r.patches if not p.counted]
    assert all(p.delta is None for p in lost)
    # Pixels covered only by lost patches are NaN, never 0.
    only_lost = (r.coverage == 0) & (r.coverage_all > 0)
    assert np.isnan(r.delta_map[only_lost]).all()


def test_ambiguous_patches_have_no_score_unless_asked(toy):
    from yoru_explain.io.runs import perturbation_rows
    from yoru_explain.methods.baselines import BaselineSpec

    frame = syn.blank()
    syn.paint(frame, A, syn.RED)
    target = syn.target_on(toy, frame, A)
    # Red patches beside A make a neighbouring cell detect red too.
    base = dict(patch_size=16, stride=8, region="context",
                baseline=BaselineSpec("constant", value=(0, 0, 255)),
                score=ScoreDefinition(identity_mode="redetect", iou_threshold=0.3))
    r = run_occlusion(toy, frame, target, OcclusionConfig(**base))
    amb = [p for p in r.patches if p.score.status == "ambiguous"]
    assert amb
    for p in amb:
        assert p.score.value is None and p.score.ambiguous_value is not None
        assert not p.counted and p.used_score is None and p.delta is None
    rows = {row["patch_id"]: row for row in perturbation_rows("ev", r)}
    assert all(rows[p.patch.patch_id]["perturbed_score"] is None for p in amb)

    asked = run_occlusion(toy, frame, target, OcclusionConfig(include_ambiguous=True, **base))
    amb2 = [p for p in asked.patches if p.score.status == "ambiguous"]
    assert amb2 and all(p.counted and p.used_score == p.score.ambiguous_value for p in amb2)
    assert all(p.delta == asked.reference_score_batch - p.used_score for p in amb2)


def test_a_candidate_index_alone_does_not_reproduce_a_target(toy):
    from dataclasses import replace

    from yoru_explain import api
    from yoru_explain.core.pipeline import reproduce_target

    frame = _two_reds()
    dets = toy.detect(frame)
    a = next(d for d in dets if d.box == A)
    b = next(d for d in dets if d.box == B)
    # A's box carrying B's candidate index (as from another model or input size).
    forged = api.target_from_detection(replace(a, candidate=b.candidate), 0, frame)
    found, info = reproduce_target(toy, frame, forged)
    assert found.box == A and info["how"] == "iou_candidate_changed"
    assert found.candidate.index == a.candidate.index
    # Made with another model: the candidate is not consulted at all.
    found, info = reproduce_target(toy, frame, api.target_from_detection(a, 0, frame), same_model=False)
    assert found.box == A and info["how"] == "iou_other_model"
