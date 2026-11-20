# SPDX-License-Identifier: Apache-2.0
"""Dynamic priority queue behavior (no vLLM import required)."""

from jitserve_vllm31.request_queue import DynamicPriorityRequestQueue


class Req:
    def __init__(self, rid, arrival, priority=0):
        self.request_id = rid
        self.arrival_time = arrival
        self.priority = priority


def test_initial_order_respects_priority_and_arrival():
    q = DynamicPriorityRequestQueue()
    q.add_request(Req("a", 1.0, 3))
    q.add_request(Req("b", 2.0, 1))
    q.add_request(Req("c", 0.5, 1))
    assert [r.request_id for r in q] == ["c", "b", "a"]
    assert q.pop_request().request_id == "c"
    assert len(q) == 2


def test_refresh_reorders_heap():
    q = DynamicPriorityRequestQueue()
    q.add_request(Req("wide", 0.0, 2))
    q.add_request(Req("tight", 0.0, 2))
    q.refresh({"wide": 5, "tight": -9})
    assert q.pop_request().request_id == "tight"
    assert q.pop_request().request_id == "wide"


def test_remove_isolates_target():
    q = DynamicPriorityRequestQueue()
    q.add_request(Req("a", 0.0))
    q.add_request(Req("b", 1.0))
    q.remove_request(q.peek_request())
    assert [r.request_id for r in q] == ["b"]
    q.remove_requests([Req("b", 1.0)])
    assert not q
