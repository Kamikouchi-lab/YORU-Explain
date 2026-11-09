# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""How long a running task still needs, from the progress it reports.

:class:`TimeLeft` is given the task's progress fraction (0..1) each time the
task reports it and estimates the time left at the pace measured so far:

* The pace is measured from the first report with progress (``fraction >
  0``), so set-up work before it -- opening the video, the identity check,
  decoding the first frame -- does not slow the estimate down for the whole
  task.  Until that span is :attr:`min_span_s` long, the pace since the start
  is used instead.
* Between reports the estimate counts down.  If it runs out before the task
  ends, the task is slower than estimated, and the time left is derived again
  from the pace measured up to now (it grows while nothing is reported).

It is an estimate: events differ in size, cached results take no time, and
frames far apart take longer to decode.
"""

from __future__ import annotations

import time
from typing import Callable, Optional, Tuple


class TimeLeft:
    """The time left of one task (restart it for the next)."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic, min_span_s: float = 0.5):
        self._clock = clock
        #: Shortest span a pace is measured over.
        self.min_span_s = float(min_span_s)
        self.restart()

    def restart(self, now: Optional[float] = None) -> None:
        """Start timing a new task (at *now*)."""
        self._t0 = self._now(now)
        self._fraction = 0.0
        self._anchor: Optional[Tuple[float, float]] = None
        self._end: Optional[float] = None

    def _now(self, now: Optional[float]) -> float:
        return self._clock() if now is None else float(now)

    @property
    def fraction(self) -> float:
        """The latest fraction reported."""
        return self._fraction

    def elapsed(self, now: Optional[float] = None) -> float:
        return max(0.0, self._now(now) - self._t0)

    def update(self, fraction: float, now: Optional[float] = None) -> None:
        """The task reports *fraction* done (at *now*)."""
        now = self._now(now)
        self._fraction = f = min(1.0, max(0.0, float(fraction)))
        if f <= 0.0:
            return
        if self._anchor is None:
            self._anchor = (now, f)
        rate = self._rate(now)
        if rate is not None:
            self._end = now + (1.0 - f) / rate

    def _rate(self, now: float) -> Optional[float]:
        """Fraction per second: since the first progress, else since the start."""
        f = self._fraction
        if self._anchor is not None:
            ta, fa = self._anchor
            if f > fa and now - ta >= self.min_span_s:
                return (f - fa) / (now - ta)
        span = now - self._t0
        if f > 0.0 and span >= self.min_span_s:
            return f / span
        return None

    def remaining(self, now: Optional[float] = None) -> Optional[float]:
        """Seconds left, or ``None`` while there is nothing to estimate from."""
        if self._end is None:
            return None
        if self._fraction >= 1.0:
            return 0.0
        now = self._now(now)
        left = self._end - now
        if left > 0.0:
            return left
        # Slower than estimated: what is left at the pace measured up to now.
        rate = self._rate(now)
        return (1.0 - self._fraction) / rate if rate else 0.0


def format_duration(seconds: float) -> str:
    """``7 s``, ``45 s``, ``2 min 10 s``, ``1 h 05 min``: rounded as an estimate deserves."""
    s = max(0.0, float(seconds))
    if s < 9.5:
        return f"{int(round(s))} s"
    if s < 57.5:
        return f"{int(5 * round(s / 5.0))} s"
    if s < 3595.0:
        m, sec = divmod(int(10 * round(s / 10.0)), 60)
        return f"{m} min {sec:02d} s"
    h, m = divmod(int(round(s / 60.0)), 60)
    return f"{h} h {m:02d} min"
