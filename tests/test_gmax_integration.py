# SPDX-License-Identifier: Apache-2.0
"""Production GMAX priorities determine the next requests dequeued by vLLM."""
from types import SimpleNamespace

from jitserve_vllm31.gmax import prioritize_window
from jitserve_vllm31.request_queue import DynamicPriorityRequestQueue


def test_cutoff_changes_next_batch_with_tagged_contention():
    # c bridges two high-density prompts by length. With the lower cutoff
    # it enters the candidate set and splits a/b into different windows.
    requests = [SimpleNamespace(request_id=key, num_prompt_tokens=length,
                                arrival_time=i, priority=0)
                for i, (key, length) in enumerate([('a', 10), ('b', 100), ('c', 20)])]
    queue = DynamicPriorityRequestQueue()
    scores = {'a': -10_000_000_000, 'b': -9_000_000_000, 'c': -8_500_000_000}
    for cutoff, expected in [(0.90, ['a', 'c']), (0.95, ['a', 'b'])]:
        for request in requests:
            queue.add_request(request)
        priorities = scores.copy()
        prioritize_window(requests, priorities, 2, cutoff)
        queue.refresh(priorities)
        assert [queue.pop_request().request_id for _ in range(2)] == expected
        queue.clear()


def test_uncontested_pool_preserves_exact_priorities():
    scores = {'a': -40, 'b': -37}
    requests = [SimpleNamespace(request_id=key) for key in scores]
    prioritize_window(requests, scores, 2, 0.90)
    assert scores == {'a': -40, 'b': -37}
