# SPDX-License-Identifier: Apache-2.0
import pytest

from jitserve_vllm31.gmax import CutoffTuner, RequestGoodputTracker


def test_terminal_outcomes_and_expiry_count_complete_requests_once():
    tuner = CutoffTuner(0.7, epsilon=0.0)
    tracker = RequestGoodputTracker(tuner)
    for rid in ("complete", "cancelled", "late", "never-finished"):
        tracker.arrive(rid, deadline_at=40.0, now=0.0)
    tracker.arrive("already-late", deadline_at=-1.0, now=0.0)
    tracker.mark_contention()
    tracker.finish("cancelled", False, now=5.0)
    tracker.finish("cancelled", True, now=6.0)
    tracker.finish("untracked", True, now=10.0)
    assert tracker.advance(30.0) == []
    assert tracker.advance(40.0) == []
    tracker.finish("complete", True, now=40.0)
    tracker.finish("complete", True, now=40.0)
    tracker.finish("late", True, now=41.0)
    # The fourth pending arrival times out without a terminal callback. The
    # delayed poll has no bearing on the 30 seconds of arrival exposure.
    result, = tracker.advance(90.0)
    assert (result.started_at, result.window_s) == (0.0, 30.0)
    assert (result.arrivals, result.on_time, result.qpm) == (5, 1, 2.0)
    assert result.cutoff == 0.7 and result.contended and result.trained
    assert tuner.value[0.7] == 2.0
    tracker.finish("never-finished", True, now=91.0)
    assert tracker.advance(150.0) == []


def test_late_feedback_credits_arrival_cutoff_and_can_close_out_of_order():
    tuner = CutoffTuner(0.7, epsilon=1.0, alpha=0.5)
    tracker = RequestGoodputTracker(tuner, window_s=10.0)
    tracker.arrive("slow", deadline_at=100.0, now=0.0)
    tracker.mark_contention()
    assert tracker.advance(10.0) == []
    tracker.arrive("fast", deadline_at=12.0, now=10.0)
    tracker.mark_contention()
    tracker.finish("fast", True, now=11.0)
    second, = tracker.advance(20.0)
    assert (second.started_at, second.cutoff, second.qpm) == (10.0, 0.7, 6.0)
    assert tuner.current == 0.7

    tracker.arrive("exploration", deadline_at=30.0, now=20.0)
    explored = tuner.current
    assert explored in (0.6, 0.8)
    tracker.mark_contention()
    tracker.finish("slow", False, now=21.0)
    first, = tracker.advance(21.0)
    assert (first.started_at, first.cutoff, first.qpm) == (0.0, 0.7, 0.0)
    assert first.trained and tuner.value[0.7] == 3.0
    assert tuner.value[explored] is None
    assert tuner.current == explored
    tracker.arrive("same-window", deadline_at=30.0, now=21.0)
    tracker.finish("same-window", True, now=23.0)
    tracker.finish("exploration", True, now=25.0)
    assert tracker.advance(29.9) == []
    third, = tracker.advance(30.0)
    assert (third.cutoff, third.on_time, third.qpm) == (explored, 2, 12.0)
    assert tuner.value[explored] == 12.0


@pytest.mark.parametrize("adaptive, contended", [(True, False), (False, True)])
def test_uncontested_and_fixed_cutoff_windows_report_without_training(adaptive, contended):
    tuner = CutoffTuner(0.7, adaptive=adaptive, epsilon=1.0)
    tracker = RequestGoodputTracker(tuner)
    tracker.mark_contention()
    assert tracker.advance(1_000.0) == []
    tracker.arrive("after-idle", deadline_at=1_001.0, now=1_000.0)
    if contended:
        tracker.mark_contention()
    tracker.finish("after-idle", True, now=1_001.0)
    result, = tracker.advance(20_000.0)
    assert result.started_at == 1_000.0 and result.qpm == 2.0
    assert result.contended == contended and not result.trained
    assert tracker.advance(30_000.0) == []
    tracker.arrive("later", deadline_at=30_010.0, now=30_000.0)
    assert tuner.current == 0.7 and all(v is None for v in tuner.value.values())


def test_reused_id_survives_previous_deadline_and_heap_releases_early_finishes():
    tracker = RequestGoodputTracker(CutoffTuner(0.7), window_s=10.0)
    tracker.arrive("reused", deadline_at=4.0, now=0.0)
    tracker.arrive("other", deadline_at=1_000_000.0, now=0.0)
    tracker.finish("reused", True, now=1.0)
    assert tracker.advance(1.0) == []
    tracker.arrive("reused", deadline_at=8.0, now=1.0)
    assert tracker.advance(5.0) == []
    tracker.finish("reused", True, now=7.0)
    tracker.finish("other", True, now=7.0)
    result, = tracker.advance(10.0)
    assert (result.arrivals, result.on_time) == (3, 3)
    assert not tracker._deadlines and not tracker._pending


def test_warmup_probes_neighbours_then_keeps_higher_measured_goodput():
    tuner = CutoffTuner(0.95, seed=0)
    tracker = RequestGoodputTracker(tuner, window_s=10)
    tested = []
    for i, success in enumerate([True, False, False, True]):
        t = 10.0 * i
        tracker.advance(t)
        tracker.arrive(str(i), t + 8, t)
        tested.append(tuner.current)
        tracker.mark_contention()
        tracker.finish(str(i), success, t + 1)
    assert tested == [0.95, 0.9, 1.0, 0.95]
