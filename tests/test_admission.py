# SPDX-License-Identifier: Apache-2.0
"""Admission control selection tests (no vLLM import required)."""

from jitserve_vllm31.admission import select_admission_drops

WAITING = "WAITING"
RUNNING = "RUNNING"


class Req:
    def __init__(self, rid, status, computed=0, arrival=None):
        self.request_id = rid
        self.status = status
        self.num_computed_tokens = computed
        self.arrival_time = arrival


def test_no_drops_under_limit():
    now = 100.0
    reqs = [Req("a", WAITING, 0, 99.0), Req("b", WAITING, 0, 95.0)]
    assert select_admission_drops(reqs, now, 30.0, WAITING) == []


def test_drops_only_expired_waiting_unscheduled():
    now = 100.0
    reqs = [
        Req("old", WAITING, 0, 60.0),       # waited 40s > 30 -> drop
        Req("fresh", WAITING, 0, 99.0),     # waited 1s -> keep
        Req("running", RUNNING, 0, 60.0),   # running estado -> keep
        Req("preempted", WAITING, 120, 60.0),  # has computed tokens -> keep
    ]
    assert select_admission_drops(reqs, now, 30.0, WAITING) == ["old"]


def test_limit_zero_disables():
    now = 100.0
    reqs = [Req("old", WAITING, 0, 0.0)]
    assert select_admission_drops(reqs, now, 0.0, WAITING) == []


def test_multiple_drops_sorted_by_arrival():
    now = 100.0
    reqs = [Req("b", WAITING, 0, 50.0), Req("a", WAITING, 0, 60.0)]
    drops = select_admission_drops(reqs, now, 30.0, WAITING)
    assert set(drops) == {"a", "b"}


def test_ever_scheduled_guards_preempted_waiter():
    now = 100.0
    reqs = [Req("old", WAITING, 0, 60.0)]  # waited 40s, computed reset by recompute preemption
    assert select_admission_drops(reqs, now, 30.0, WAITING, ever_scheduled={"old"}) == []
    assert select_admission_drops(reqs, now, 30.0, WAITING, ever_scheduled=set()) == ["old"]
