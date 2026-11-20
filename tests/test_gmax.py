# SPDX-License-Identifier: Apache-2.0
"""GMAX window behaviors (offline, no vLLM import)."""
import pytest
from jitserve_vllm31.gmax import CutoffTuner, FrameTracker, FrameGoodputAccumulator, select_gmax_window


class Req:
    def __init__(self, rid, prompt=100, density=1.0):
        self.request_id = rid
        self.num_prompt_tokens = prompt
        self._density = density

    def __repr__(self):
        return f"Req({self.request_id},p={self.num_prompt_tokens})"


def _densities(*reqs):
    return {r.request_id: r._density for r in reqs}


def test_cutoff_tuner_ema_and_epsilon_greedy():
    tuner = CutoffTuner(0.7, epsilon=0.0, alpha=0.2)
    # Drive 0.7 with high goodput, 0.8 with low: best stays 0.7.
    for rate in (10.0,) * 10:
        assert tuner.current == 0.7
        tuner.update(rate)
    assert tuner.value[0.7] == pytest.approx(10.0, rel=1e-6)
    for rate in (1.0,) * 10:
        tuner.update(rate)
    assert tuner.current == 0.7  # epsilon=0 → greedy on best


def test_frame_tracker_boundary_every_interval():
    tracker = FrameTracker(schedule_interval=5)
    boundaries = [tracker.update_schedule_count() for _ in range(12)]
    assert boundaries[:5] == [False] * 4 + [True]
    assert sum(boundaries) == 2  # boundaries at counts 5 and 10
    assert tracker.interval_time is not None


def test_frame_goodput_accumulator_drains_into_tuner():
    tuner = CutoffTuner(0.7, epsilon=0.0, alpha=0.5)
    acc = FrameGoodputAccumulator(tuner)
    acc.accumulate(3.0)
    acc.accumulate(2.0)
    new_p = acc.end_frame()
    assert new_p in CutoffTuner.GRID or new_p == 0.7
    assert tuner.value[0.7] > 0
    assert acc._frame_goodput == 0.0


def test_window_keeps_high_density_above_cutoff():
    lo = Req("lo", prompt=50, density=0.1)
    mid = Req("mid", prompt=100, density=1.0)
    high = Req("high", prompt=20, density=5.0)
    window = select_gmax_window([lo, mid, high], _densities(lo, mid, high), 2, 0.8)
    assert "lo" not in [r.request_id for r in window]
    assert sorted(r.request_id for r in window) == ["high", "mid"]


def test_window_picks_best_input_length_contiguous_slice():
    candidates = [
        Req("a", prompt=100, density=4.0),
        Req("b", prompt=50, density=3.0),
        Req("c", prompt=10, density=2.0),
        Req("d", prompt=200, density=3.5),
    ]
    window = select_gmax_window(candidates, _densities(*candidates), 3, 1.0)
    assert len(window) == 3
    ids = [r.request_id for r in window]
    # Sorted by prompt len: c(10), b(50), a(100), d(200). Best contiguous window
    # of size 3 by total density: b+a+d = 3+4+3.5 = 10.5 > c+b+a = 9.
    assert ids == ["b", "a", "d"]
    assert [r.num_prompt_tokens for r in window] == [50, 100, 200]


def test_window_small_pool_returns_all():
    r1 = Req("one", prompt=10, density=1.0)
    r2 = Req("two", prompt=20, density=2.0)
    window = select_gmax_window([r1, r2], _densities(r1, r2), 10, 0.9)
    assert sorted(r.request_id for r in window) == ["one", "two"]


def test_window_empty_or_zero_size():
    assert select_gmax_window([], {}, 5, 0.8) == []
    r = Req("x", prompt=5, density=1.0)
    assert select_gmax_window([r], _densities(r), 0, 0.8) == [r]


def test_window_keeps_first_of_tied_best_slices():
    reqs = [Req(str(i), prompt=i, density=density)
            for i, density in enumerate([1, 4, 3, 4, 1])]
    window = select_gmax_window(reqs, _densities(*reqs), 2, 0.0)
    assert window == reqs[1:3]
