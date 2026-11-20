# SPDX-License-Identifier: Apache-2.0
# Adapted for this vLLM V1 plugin from the JITServe artifact. See NOTICE.
"""Pure admission-control selection (no vLLM imports, testable offline).

Mirrors JITServe ``slo_scheduler.py:199`` ``_admission_control``: requests
waiting longer than ``limit`` seconds without ever being scheduled are
selected for drop. ``num_computed_tokens == 0`` combined with absence from
``ever_scheduled`` is the "never scheduled" proxy (preempted requests are
left alone).
"""

from collections.abc import Iterable
from typing import Any


def select_admission_drops(
    requests: Iterable[Any],
    now: float,
    limit: float,
    waiting_status: Any,
    ever_scheduled: "set[str] | None" = None,
) -> list[str]:
    """Return request_ids that exceeded their admission deadline.

    Args:
        requests: candidate requests (any object with ``status``,
            ``num_computed_tokens``, ``arrival_time`` and ``request_id``).
        now: current wall-clock ``time.time()``.
        limit: max waiting seconds; ``<= 0`` disables admission control.
        waiting_status: the scheduler's WAITING status sentinel.
    """
    if limit <= 0:
        return []
    dropped: list[str] = []
    for request in requests:
        if getattr(request, "status", None) != waiting_status:
            continue
        if getattr(request, "num_computed_tokens", 0) != 0:
            continue
        if ever_scheduled is not None and request.request_id in ever_scheduled:
            continue
        arrival = getattr(request, "arrival_time", None)
        if arrival is None:
            continue
        if now - float(arrival) > limit:
            dropped.append(request.request_id)
    return dropped
