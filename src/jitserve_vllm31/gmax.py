# SPDX-License-Identifier: Apache-2.0
# Adapted for this vLLM V1 plugin from the JITServe artifact. See NOTICE.
"""GMAX cutoff adaptation, request-level goodput, and frame bookkeeping."""
from __future__ import annotations

import heapq
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Optional

DEFAULT_SCHEDULE_INTERVAL = 20
DEFAULT_INTERVAL_TIME = 0.02


class CutoffTuner:
    """Online adaptation of the GMAX cutoff p (paper Sec 4.2).

    The paper only says p is adapted online "by exploring different thresholds
    and converging to those that maximize end-to-end goodput". We realize this
    as epsilon-greedy local search over a fixed grid. Each observed goodput
    rate updates the EMA for the cutoff that earned it. Most decisions use
    the best cutoff so far, and the rest explore a neighbouring value.
    """
    GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)

    def __init__(self, initial: float, adaptive: bool = True,
                 epsilon: float = 0.1, alpha: float = 0.2,
                 seed: int = 0) -> None:
        self.adaptive = adaptive
        self.grid = sorted(set(self.GRID) | {initial})
        self.value: dict[float, Optional[float]] = {p: None for p in self.grid}
        self.current = initial
        self.epsilon = epsilon
        self.alpha = alpha
        self._rng = random.Random(seed)

    def observe(self, cutoff: float, goodput_rate: float) -> None:
        if self.adaptive:
            v = self.value[cutoff]
            self.value[cutoff] = goodput_rate if v is None else (
                (1 - self.alpha) * v + self.alpha * goodput_rate)

    def choose(self, *, warmup: bool = False) -> float:
        if not self.adaptive:
            return self.current
        tried = [p for p in self.grid if self.value[p] is not None]
        if not tried:
            return self.current
        best = max(tried, key=lambda p: self.value[p] or 0.0)
        i = self.grid.index(best)
        neighbours = self.grid[max(0, i - 1):i] + self.grid[i + 1:i + 2]
        if warmup and self.epsilon > 0:
            if self.value[self.current] is None:
                return self.current  # Wait for this trial's delayed outcomes.
            for candidate in neighbours:
                if self.value[candidate] is None:
                    self.current = candidate
                    return candidate
        if self._rng.random() < self.epsilon:
            self.current = self._rng.choice(neighbours) if neighbours else best
        else:
            self.current = best
        return self.current

    def update(self, goodput_rate: float) -> float:
        self.observe(self.current, goodput_rate)
        return self.choose()


@dataclass(slots=True)
class _ArrivalWindow:
    started_at: float
    cutoff: float
    arrivals: int = 0
    pending: int = 0
    on_time: int = 0
    contended: bool = False


@dataclass(frozen=True, slots=True)
class RequestGoodputResult:
    started_at: float
    window_s: float
    cutoff: float
    arrivals: int
    on_time: int
    qpm: float
    contended: bool
    trained: bool


class RequestGoodputTracker:
    """Credit terminal outcomes to the cutoff in effect on scheduler admission.

    The first arrival opens a fixed-length, half-open window. A sealed window
    reports once its requests have finished or expired, even when a younger
    window finishes first. Newly observed rewards affect the next window's
    cutoff choice. An open window always uses the same cutoff.
    """

    def __init__(self, tuner: CutoffTuner, window_s: float = 30.0) -> None:
        if not math.isfinite(window_s) or window_s <= 0:
            raise ValueError("window_s must be positive and finite")
        self.tuner = tuner
        self.window_s = window_s
        self._active: _ArrivalWindow | None = None
        self._pending: dict[str, tuple[float, _ArrivalWindow]] = {}
        self._deadlines: list[tuple[float, str]] = []
        self._ready: list[_ArrivalWindow] = []
        self._choice_due = False

    def arrive(self, request_id: str, deadline_at: float, now: float) -> None:
        """Enroll one eligible arrival after the caller has advanced to now."""
        if request_id in self._pending:
            return
        if not math.isfinite(deadline_at):
            raise ValueError("deadline_at must be finite")
        if self._active is None:
            if self._choice_due:
                self.tuner.choose(warmup=True)
                self._choice_due = False
            self._active = _ArrivalWindow(now, self.tuner.current)
        cohort = self._active
        cohort.arrivals += 1
        if deadline_at >= now:
            cohort.pending += 1
            self._pending[request_id] = (deadline_at, cohort)
            heapq.heappush(self._deadlines, (deadline_at, request_id))

    def finish(self, request_id: str, successful: bool, now: float) -> None:
        entry = self._pending.pop(request_id, None)
        if entry is None:
            return
        deadline_at, cohort = entry
        if successful and now <= deadline_at:
            cohort.on_time += 1
        cohort.pending -= 1
        if cohort.pending == 0 and cohort is not self._active:
            self._ready.append(cohort)

    def mark_contention(self) -> None:
        """Called only when GMAX makes a choice among competing requests."""
        if self._active is not None:
            self._active.contended = True

    def advance(self, now: float) -> list[RequestGoodputResult]:
        cohort = self._active
        if cohort is not None and now >= cohort.started_at + self.window_s:
            self._active = None
            if cohort.pending == 0:
                self._ready.append(cohort)

        # Keep the deadline instant eligible for an on-time terminal callback.
        while self._deadlines and self._deadlines[0][0] < now:
            deadline, request_id = heapq.heappop(self._deadlines)
            entry = self._pending.get(request_id)
            if entry is not None and entry[0] == deadline:
                self.finish(request_id, False, now)
        # Lazy deletion makes finish O(1). Compact only after enough finishes
        # to amortize the rebuild, so long deadlines cannot retain old IDs.
        if len(self._deadlines) > 2 * len(self._pending):
            self._deadlines = [(deadline, rid)
                               for rid, (deadline, _) in self._pending.items()]
            heapq.heapify(self._deadlines)

        results = []
        for cohort in self._ready:
            qpm = cohort.on_time * 60.0 / self.window_s
            trained = self.tuner.adaptive and cohort.contended
            if trained:
                self.tuner.observe(cohort.cutoff, qpm)
                self._choice_due = True
            results.append(RequestGoodputResult(
                started_at=cohort.started_at, window_s=self.window_s,
                cutoff=cohort.cutoff, arrivals=cohort.arrivals,
                on_time=cohort.on_time, qpm=qpm, contended=cohort.contended,
                trained=trained,
            ))
        self._ready.clear()
        return results


class FrameTracker:
    """V0 ``BasePolicy`` frame bookkeeping, ported for V1.

    One frame = ``schedule_interval`` scheduler invocations. At each boundary
    the wall-clock interval between boundaries is measured (deque-smoothed)
    and supplied to the priority computation as ``interval_time`` /
    ``frame_time``.
    """
    def __init__(self, schedule_interval: int = DEFAULT_SCHEDULE_INTERVAL,
                 window_size: int = 10) -> None:
        self.schedule_interval = schedule_interval
        self.num_schedule_count = 0
        self.last_schedule_time: Optional[float] = None
        self.intervals: Deque[float] = deque(maxlen=window_size)
        self.interval_time: Optional[float] = None

    def update_schedule_count(self) -> bool:
        """Return True exactly on frame boundaries."""
        self.num_schedule_count += 1
        if self.num_schedule_count % self.schedule_interval == 0:
            self.update_interval_time(time.time())
            return True
        return False

    def update_interval_time(self, cur_time: float) -> None:
        if self.last_schedule_time is None:
            self.last_schedule_time = cur_time
            self.intervals.append(DEFAULT_INTERVAL_TIME)
        else:
            new_interval = (cur_time - self.last_schedule_time) / (
                self.schedule_interval)
            self.intervals.append(new_interval)
            self.last_schedule_time = cur_time
        if self.intervals:
            self.interval_time = sum(self.intervals) / len(self.intervals)
        else:
            self.interval_time = DEFAULT_INTERVAL_TIME

    def _get_interval_time(self) -> float:
        return self.interval_time if self.interval_time else DEFAULT_INTERVAL_TIME

    def frame_time(self) -> float:
        """Wall-clock length of one scheduling frame."""
        return self.schedule_interval * self._get_interval_time()


class FrameGoodputAccumulator:
    """V0 ``_accumulate_goodput`` / ``_end_frame``, ported for V1.

    On every scheduler invocation the goodput earned by running requests (as
    reported by the V1 stats layer) is added to the frame bucket. On frame
    boundaries the bucket is drained into ``CutoffTuner.update``.
    """
    def __init__(self, cutoff_tuner: CutoffTuner) -> None:
        self.cutoff_tuner = cutoff_tuner
        self._frame_goodput = 0.0
        self._frame_start = time.time()

    def accumulate(self, running_goodput: float) -> None:
        if running_goodput > 0:
            self._frame_goodput += running_goodput

    def end_frame(self) -> float:
        now = time.time()
        elapsed = now - self._frame_start
        rate = self._frame_goodput / elapsed if elapsed > 0 else 0.0
        new_cutoff = self.cutoff_tuner.update(rate)
        self._frame_goodput = 0.0
        self._frame_start = now
        return new_cutoff


def select_gmax_window(
    tagged: list[Any],
    density: dict[str, float],
    window_size: int,
    cutoff: float,
) -> list[Any]:
    """GMAX (paper Algorithm 1) window selection, ported from V0
    ``_recompute_bins_sliding_window``.

    Keep requests whose priority (goodput-density, higher is better) is at
    least ``cutoff * priority(r_B)`` where ``r_B`` is the B-th ranked request;
    sort survivors by input length and pick the contiguous window of size
    ``window_size`` (B) with the largest total priority.
    """
    if not tagged or window_size <= 0:
        return list(tagged)
    ordered = sorted(tagged, key=lambda r: -density.get(str(r.request_id), 0.0))
    batch_size = min(window_size, len(ordered))
    if len(ordered) <= batch_size:
        return list(ordered)
    threshold = cutoff * density.get(str(ordered[batch_size - 1].request_id), 0.0)
    candidates = [
        r for r in ordered
        if density.get(str(r.request_id), 0.0) >= threshold
    ]
    candidates.sort(key=lambda r: int(getattr(r, "num_prompt_tokens", 0)))
    if len(candidates) <= batch_size:
        return candidates

    cur_sum = sum(
        density.get(str(candidates[i].request_id), 0) for i in range(batch_size)
    )

    best_start = 0
    best_sum = cur_sum
    for i in range(1, len(candidates) - batch_size + 1):
        cur_sum -= density.get(str(candidates[i - 1].request_id), 0)
        cur_sum += density.get(str(candidates[i + batch_size - 1].request_id), 0)
        if cur_sum > best_sum:
            best_sum = cur_sum
            best_start = i
    return candidates[best_start:best_start + batch_size]


def prioritize_window(waiters, priorities: dict[str, int], size: int, cutoff: float) -> None:
    """Keep the selected window ahead of all other tagged waiting requests."""
    if size <= 0 or len(waiters) <= size:
        return
    densities = {r.request_id: -priorities[r.request_id] for r in waiters}
    selected = {r.request_id for r in select_gmax_window(waiters, densities, size, cutoff)}
    # Scores can exceed any fixed offset. Preserve ordering within each group.
    offset = max(densities.values()) - min(densities.values()) + 1
    for r in waiters:
        if r.request_id not in selected:
            priorities[r.request_id] += offset
