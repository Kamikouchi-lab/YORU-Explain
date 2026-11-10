"""The time left of a running task, from the fractions it reports (a fake clock)."""

from __future__ import annotations

import pytest

from yoru_explain.core.eta import TimeLeft, format_duration


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_nothing_to_estimate_from_before_progress():
    clock = Clock()
    eta = TimeLeft(clock=clock)
    assert eta.remaining() is None
    clock.t += 5
    eta.update(0.0)                       # "reading frame": no progress yet
    assert eta.remaining() is None


def test_a_steady_task_is_estimated_exactly():
    clock = Clock()
    eta = TimeLeft(clock=clock)
    for step in range(1, 5):              # 10 % every 2 s
        clock.t += 2.0
        eta.update(0.1 * step)
    assert eta.remaining() == pytest.approx(12.0)          # 60 % left at 5 %/s
    assert eta.elapsed() == pytest.approx(8.0)


def test_set_up_before_the_first_progress_does_not_slow_the_estimate():
    clock = Clock()
    eta = TimeLeft(clock=clock)
    clock.t += 30.0                       # identity check, decoding the first frame ...
    eta.update(0.1)                       # ... then 10 % per second
    for _ in range(3):
        clock.t += 1.0
        eta.update(eta.fraction + 0.1)
    assert eta.remaining() == pytest.approx(6.0)           # not 60 % at (40 % / 33 s)


def test_the_estimate_counts_down_between_reports():
    clock = Clock()
    eta = TimeLeft(clock=clock)
    eta.update(0.0)
    for f in (0.25, 0.5):
        clock.t += 10.0
        eta.update(f)
    assert eta.remaining() == pytest.approx(20.0)
    clock.t += 5.0
    assert eta.remaining() == pytest.approx(15.0)


def test_a_stalled_task_is_not_said_to_be_nearly_done():
    clock = Clock()
    eta = TimeLeft(clock=clock)
    eta.update(0.0)
    for f in (0.25, 0.5):
        clock.t += 10.0
        eta.update(f)
    clock.t += 30.0                       # 20 s were estimated; nothing for 30 s
    left = eta.remaining()
    assert left == pytest.approx(0.5 * 40.0 / 0.25)        # 50 % left at the pace up to now
    clock.t += 10.0
    assert eta.remaining() > left                         # and growing while nothing comes


def test_a_short_task_before_min_span_is_not_estimated_and_restart_forgets():
    clock = Clock()
    eta = TimeLeft(clock=clock, min_span_s=0.5)
    clock.t += 0.1
    eta.update(0.5)
    assert eta.remaining() is None
    clock.t += 1.0
    eta.update(0.75)
    assert eta.remaining() is not None
    eta.update(1.0)
    assert eta.remaining() == 0.0
    eta.restart()
    assert eta.remaining() is None and eta.fraction == 0.0 and eta.elapsed() == 0.0


@pytest.mark.parametrize("seconds, text", [
    (0, "0 s"), (3.4, "3 s"), (9.4, "9 s"), (12, "10 s"), (44, "45 s"), (57.6, "1 min 00 s"),
    (125, "2 min 00 s"), (126, "2 min 10 s"), (3594, "59 min 50 s"), (3600, "1 h 00 min"),
    (3 * 3600 + 5 * 60 + 20, "3 h 05 min"), (-4, "0 s"),
])
def test_durations_are_rounded_as_estimates(seconds, text):
    assert format_duration(seconds) == text
