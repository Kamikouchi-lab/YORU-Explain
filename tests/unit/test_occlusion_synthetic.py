"""Occlusion against answers known in advance.

The synthetic detector's candidate reads exactly its 32x32 window, so:
hiding anything outside the window must leave the score *bit for bit*
unchanged (delta exactly 0), hiding part of the red square must lower it,
and hiding green inside the window must raise it (a negative delta).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests import synthetic as syn
from yoru_explain.core.types import ScoreDefinition
from yoru_explain.methods.baselines import BaselineSpec
from yoru_explain.methods.occlusion import (
    OcclusionConfig,
    make_patches,
    region_of_interest,
    run_occlusion,
)


def _noisy_frame(seed=0):
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 60, size=(192, 320, 3), dtype=np.uint8)  # dark: nothing detected
    syn.paint(frame, syn.window(5, 5), syn.RED)
    return frame


def _run(toy, frame, box=(72.0, 72.0, 104.0, 104.0), **cfg):
    target = syn.target_on(toy, frame, box)
    return run_occlusion(toy, frame, target, OcclusionConfig(**cfg))


def test_only_patches_over_the_candidates_window_change_its_score(toy):
    frame = _noisy_frame()
    r = _run(toy, frame, patch_size=16, stride=8, region="full")
    assert r.ok, r.message
    win = syn.window(5, 5)
    inside = [p for p in r.patches if syn.overlaps(p.patch.box, win)]
    outside = [p for p in r.patches if not syn.overlaps(p.patch.box, win)]
    assert inside and outside
    # Outside the window: pixels changed, the score did not -- exactly.
    assert all(p.pixel_change > 0 for p in outside)
    assert all(p.delta == 0.0 for p in outside)
    # Over the red square: the score drops.
    assert all(p.delta > 0 for p in inside)
    # The map: positive only where an overlapping patch was, 0 elsewhere.
    m = r.delta_map
    assert np.isfinite(m).all()
    y, x = np.unravel_index(np.nanargmax(m), m.shape)
    assert win[0] <= x < win[2] and win[1] <= y < win[3]
    assert m[:40, :].max() == 0.0 and m[:, 150:].max() == 0.0


def test_the_controls_noop_and_full_region(toy):
    r = _run(toy, _noisy_frame(), patch_size=16, stride=16, region="context")
    assert r.controls["noop"].delta == 0.0
    assert r.controls["noop"].pixel_change == 0.0
    full = r.controls["full_region"]
    assert full.score.status == "ok" and full.score.value < 1e-3  # the square is gone
    assert r.reference_score == r.reference_score_batch == r.target.detection.score


def test_hiding_suppressing_evidence_gives_a_negative_delta(toy):
    frame = syn.blank()
    x0, y0, x1, y1 = syn.window(5, 5)
    syn.paint(frame, (x0, y0, x1, y1), syn.RED)
    syn.paint(frame, (x1 - 4, y0, x1, y1), syn.GREEN)  # a green stripe inside the window
    r = _run(toy, frame, patch_size=4, stride=4, region="box")
    assert r.ok, r.message
    green = [p for p in r.patches if p.patch.box[0] >= x1 - 4]
    red = [p for p in r.patches if p.patch.box[2] <= x1 - 4]
    assert green and red
    assert all(p.delta < 0 for p in green)       # hiding green raises the red score
    assert all(p.delta > 0 for p in red)
    assert np.nanmin(r.delta_map) < 0            # kept negative in the map, not clipped


def test_a_patch_that_changes_no_pixel_has_no_effect(toy):
    frame = syn.blank()
    syn.paint(frame, syn.window(5, 5), syn.RED)
    r = _run(toy, frame, patch_size=16, stride=16, region="full")
    flat = [p for p in r.patches if p.pixel_change == 0.0]
    assert flat and all(p.delta == 0.0 for p in flat)


@pytest.mark.parametrize("batch_size", [1, 3, 8])
def test_batch_size_does_not_change_deltas_here(toy, batch_size):
    """The synthetic model is batch-invariant, so every batch size must agree."""
    frame = _noisy_frame(1)
    ref = _run(toy, frame, patch_size=16, stride=8, region="context")
    r = _run(toy, frame, patch_size=16, stride=8, region="context", batch_size=batch_size)
    assert r.controls["noop"].delta == 0.0
    assert [p.delta for p in r.patches] == [p.delta for p in ref.patches]
    np.testing.assert_array_equal(r.delta_map, ref.delta_map)


def test_rerunning_gives_identical_results(toy):
    frame = _noisy_frame(2)
    a = _run(toy, frame, patch_size=16, stride=8, baseline=BaselineSpec("noise"), seed=5)
    b = _run(toy, frame, patch_size=16, stride=8, baseline=BaselineSpec("noise"), seed=5)
    c = _run(toy, frame, patch_size=16, stride=8, baseline=BaselineSpec("noise"), seed=6)
    np.testing.assert_array_equal(a.delta_map, b.delta_map)
    assert [p.delta for p in a.patches] == [p.delta for p in b.patches]
    assert [p.pixel_change for p in a.patches] != [p.pixel_change for p in c.patches]


@pytest.mark.parametrize("kind", ["constant", "image_mean", "patch_mean", "blur", "noise"])
def test_every_baseline_runs_and_is_recorded(toy, kind):
    r = _run(toy, _noisy_frame(), patch_size=16, stride=16, baseline=BaselineSpec(kind))
    assert r.ok
    assert r.config.to_dict()["baseline"]["kind"] == kind


def test_redetection_reports_a_lost_target_as_lost_not_zero(toy):
    frame = syn.blank()
    syn.paint(frame, syn.window(5, 5), syn.RED)
    r = _run(toy, frame, patch_size=32, stride=8, region="context",
             score=ScoreDefinition(identity_mode="redetect"))
    assert r.ok
    covering = [p for p in r.patches if p.patch.box == syn.window(5, 5)]
    assert covering and covering[0].score.status == "disappeared"
    assert covering[0].score.value is None and covering[0].delta is None and not covering[0].counted
    assert r.controls["full_region"].score.status == "disappeared"
    lost = r.lost_fraction
    assert np.nanmax(lost) > 0


def test_patches_tile_the_region_to_its_edges():
    roi = (10, 20, 75, 61)
    patches = make_patches(roi, (16, 16), (8, 8), 100, 80)
    cover = np.zeros((80, 100), int)
    for p in patches:
        x0, y0, x1, y1 = p.box
        assert 0 <= x0 < x1 <= 100 and 0 <= y0 < y1 <= 80
        cover[y0:y1, x0:x1] += 1
    assert (cover[20:61, 10:75] > 0).all()
    assert max(p.box[2] for p in patches) == 75 and max(p.box[3] for p in patches) == 61
    # A region smaller than a patch gets one patch.
    assert len(make_patches((5, 5, 9, 9), (16, 16), (8, 8), 100, 80)) == 1


def test_regions_of_interest():
    assert region_of_interest((10, 10, 20, 20), "full", 3, 100, 80) == (0, 0, 100, 80)
    assert region_of_interest((10.2, 10.7, 20.1, 20.0), "box", 3, 100, 80) == (10, 10, 21, 20)
    assert region_of_interest((0, 0, 20, 20), "context", 3, 100, 80) == (0, 0, 40, 40)


def test_a_frame_of_another_size_is_refused(toy):
    frame = _noisy_frame()
    target = syn.target_on(toy, frame, (72.0, 72.0, 104.0, 104.0))
    r = run_occlusion(toy, frame[:160], target, OcclusionConfig())
    assert r.status == "failed" and r.error_code == "frame_mismatch"


def test_fixed_candidate_mode_needs_a_candidate(toy):
    from dataclasses import replace

    frame = _noisy_frame()
    target = syn.target_on(toy, frame, (72.0, 72.0, 104.0, 104.0))
    target = target.with_detection(replace(target.detection, candidate=None), 0)
    r = run_occlusion(toy, frame, target, OcclusionConfig())
    assert r.status == "failed" and r.error_code == "score_unavailable"


def test_cancelling_stops_without_a_map(toy):
    frame = _noisy_frame()
    target = syn.target_on(toy, frame, (72.0, 72.0, 104.0, 104.0))
    calls = []
    r = run_occlusion(toy, frame, target, OcclusionConfig(patch_size=8, stride=4),
                      cancel=lambda: len(calls) > 2, progress=lambda d, t: calls.append(d))
    assert r.status == "cancelled" and r.error_code == "cancelled" and r.delta_map is None
