# SPDX-License-Identifier: Apache-2.0
"""Heap-backed dynamic-priority queue compatible with vLLM's RequestQueue API."""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Iterator, Mapping
from typing import Any


class _Entry:
    __slots__ = ("priority", "arrival_time", "request_id", "sequence", "request")

    def __init__(self, request: Any, priority: int, sequence: int) -> None:
        self.priority = priority
        self.arrival_time = float(request.arrival_time)
        self.request_id = str(request.request_id)
        self.sequence = sequence
        self.request = request

    def __lt__(self, other: _Entry) -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        if self.request_id != other.request_id:
            return self.request_id < other.request_id
        return self.sequence < other.sequence


class DynamicPriorityRequestQueue:
    """A min-heap ordered by score, arrival time, request ID, then insertion."""

    def __init__(self) -> None:
        self._heap: list[_Entry] = []
        self._next_sequence = 0

    def add_request(self, request: Any) -> None:
        self._next_sequence += 1
        entry = _Entry(request, int(getattr(request, "priority", 0)), self._next_sequence)
        heapq.heappush(self._heap, entry)

    def pop_request(self) -> Any:
        if not self._heap:
            raise IndexError("pop from empty priority queue")
        return heapq.heappop(self._heap).request

    def peek_request(self) -> Any:
        if not self._heap:
            raise IndexError("peek from an empty priority queue")
        return self._heap[0].request

    def prepend_request(self, request: Any) -> None:
        self.add_request(request)

    def prepend_requests(self, requests: Iterable[Any]) -> None:
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Any) -> None:
        for index, entry in enumerate(self._heap):
            if entry.request is request:
                del self._heap[index]
                heapq.heapify(self._heap)
                return
        raise ValueError("request is not in the priority queue")

    def remove_requests(self, requests: Iterable[Any]) -> None:
        request_ids = {str(request.request_id) for request in requests}
        if not request_ids:
            return
        self._heap = [entry for entry in self._heap if entry.request_id not in request_ids]
        heapq.heapify(self._heap)

    def refresh(self, priorities: Mapping[str, int]) -> None:
        for entry in self._heap:
            entry.priority = priorities.get(
                entry.request_id, int(getattr(entry.request, "priority", 0))
            )
        heapq.heapify(self._heap)

    def clear(self) -> None:
        self._heap.clear()

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __len__(self) -> int:
        return len(self._heap)

    def __iter__(self) -> Iterator[Any]:
        pending = list(self._heap)
        while pending:
            yield heapq.heappop(pending).request
