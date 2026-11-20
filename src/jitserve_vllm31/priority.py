# SPDX-License-Identifier: Apache-2.0
# Adapted for this vLLM V1 plugin from the JITServe artifact. See NOTICE.
"""Pure-Python port of JITServe's SLO goodput-density ranking."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from math import isfinite
from typing import Any

from .metadata import SLO_EXTRA_ARGS_KEY

_PRIORITY_SCALE = 1_000_000


class OutputLengthHistory:
    """Recent normally completed throughput lengths, in decode tokens."""

    def __init__(self, capacity: int = 128) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._lengths: deque[int] = deque(maxlen=capacity)

    def observe(self, num_tokens: int) -> None:
        if isinstance(num_tokens, int) and not isinstance(num_tokens, bool) and num_tokens > 0:
            self._lengths.append(num_tokens)

    def estimate(self, max_tokens: int | None) -> int | None:
        if not self._lengths:
            return None
        lengths = sorted(self._lengths)
        # Nearest-rank 95th percentile, rounded up to an observed length.
        estimate = lengths[(95 * len(lengths) - 1) // 100]
        if max_tokens is not None and max_tokens > 0:
            return min(estimate, max_tokens)
        return estimate


def request_deadline_at(request: Any, metadata: Mapping[str, Any]) -> float:
    """Return the absolute wall-clock deadline, including frontend time if available."""
    try:
        if "deadline_at" in metadata:
            deadline = metadata["deadline_at"]
        else:
            arrival = getattr(request, "arrival_time", None)
            duration = metadata.get("deadline_s")
            if (type(arrival) not in (int, float) or arrival < 0
                    or type(duration) not in (int, float) or duration <= 0):
                raise ValueError("invalid arrival_time or deadline_s")
            deadline = arrival + duration
        if type(deadline) not in (int, float) or not isfinite(deadline) or deadline <= 0:
            raise ValueError("deadline_at must be a positive finite timestamp")
        return float(deadline)
    except OverflowError as exc:
        raise ValueError("deadline exceeds wall-clock range") from exc


def _metadata(request: Any) -> Mapping[str, Any] | None:
    sampling_params = getattr(request, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None)
    if not isinstance(extra_args, Mapping):
        return None
    metadata = extra_args.get(SLO_EXTRA_ARGS_KEY)
    if not isinstance(metadata, Mapping) or metadata.get("version") != 1:
        return None
    return metadata


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0 or not isfinite(denominator):
        return 1e-6
    return max(1e-6, min(1.0, numerator / denominator))


def _lengths(request: Any, metadata: Mapping[str, Any]) -> tuple[int, int]:
    prompt_len = max(0, int(getattr(request, "num_prompt_tokens", 0)))
    predicted = max(1, int(metadata.get("output_len", 1)))
    max_tokens = getattr(request, "max_tokens", None)
    if isinstance(max_tokens, int) and max_tokens > 0:
        predicted = min(predicted, max_tokens)
    output_ids = getattr(request, "output_token_ids", ())
    remaining = max(predicted - len(output_ids), 1)
    return prompt_len, remaining


def _starvation_bonus(
    request: Any,
    now: float,
    last_scheduled_at: Mapping[str, float],
    starvation_delta: float,
    frame_time_s: float,
) -> float:
    if starvation_delta <= 0:
        return 0.0
    request_id = str(request.request_id)
    anchor = last_scheduled_at.get(request_id, float(request.arrival_time))
    return starvation_delta * max(0.0, now - anchor) / max(frame_time_s, 1e-6)


def _reward(
    request: Any,
    metadata: Mapping[str, Any],
    now: float,
    unit_time_s: float,
    penalty_factor: int,
    remaining_time_s: float | None = None,
) -> tuple[float, float]:
    prompt_len, remaining_len = _lengths(request, metadata)
    predicted_len = max(remaining_len, 1)
    remaining_time = (
        max(unit_time_s * remaining_len, 1e-6)
        if remaining_time_s is None
        else max(remaining_time_s, 1e-6)
    )
    if metadata["request_type"] == "throughput":
        budget = request_deadline_at(request, metadata) - now
        feasibility = max(0.0, min(1.0, budget / remaining_time))
        return feasibility**penalty_factor, remaining_time

    serve_time = max(0.0, now - float(request.arrival_time))
    deadline = float(metadata["deadline_s"])
    request_type = metadata["request_type"]

    if request_type == "latency":
        # Once a decode token exists, TTFT reward has already been realized.
        if len(getattr(request, "output_token_ids", ())) > 0:
            decode_ratio = _ratio(deadline, serve_time + remaining_time)
            reward = predicted_len * 8 * decode_ratio**penalty_factor
        else:
            ttft_slo = float(metadata["ttft_slo_s"])
            prefill_ratio = _ratio(ttft_slo, serve_time + unit_time_s)
            decode_ratio = _ratio(deadline, serve_time + remaining_time)
            reward = prompt_len * prefill_ratio**penalty_factor
            reward += predicted_len * 8 * decode_ratio**penalty_factor
    else:
        finish_ratio = _ratio(deadline, serve_time + remaining_time)
        reward = (prompt_len + predicted_len * 8) * finish_ratio**penalty_factor

    return reward, remaining_time


def _to_priority(density: float) -> int:
    if not isfinite(density) or density <= 0:
        return 0
    return -int(round(density * _PRIORITY_SCALE))


def compute_request_priorities(
    requests: Iterable[Any],
    *,
    now: float,
    unit_time_s: float,
    frame_time_s: float,
    penalty_factor: int,
    starvation_delta: float = 0.0,
    last_scheduled_at: Mapping[str, float] | None = None,
) -> dict[str, int]:
    """Return lower-is-earlier integer priorities for active V1 requests.

    Requests without JITServe metadata retain their native vLLM priority. For
    collective requests, rewards are summed and divided by the sum of peer
    remaining service times, matching the original collection-density rule.
    """
    active = list(requests)
    last_scheduled_at = last_scheduled_at or {}
    unit_time_s = max(float(unit_time_s), 1e-6)
    scores = {str(request.request_id): int(getattr(request, "priority", 0)) for request in active}
    metadata_by_id: dict[str, Mapping[str, Any]] = {}
    collective_groups: dict[tuple[type, int | str], list[Any]] = {}

    for request in active:
        request_id = str(request.request_id)
        metadata = _metadata(request)
        if metadata is None:
            continue
        request_type = metadata.get("request_type")
        if request_type not in {"latency", "throughput", "collective"}:
            continue
        try:
            if request_type == "collective":
                collection_id = metadata.get("collection_id")
                if isinstance(collection_id, bool) or not isinstance(collection_id, (int, str)):
                    continue
                collective_groups.setdefault((type(collection_id), collection_id), []).append(request)
            else:
                reward, remaining_time = _reward(
                    request, metadata, now, unit_time_s, penalty_factor
                )
                if request_type == "latency":
                    reward += _starvation_bonus(
                        request, now, last_scheduled_at, starvation_delta, frame_time_s
                    )
                scores[request_id] = _to_priority(reward / remaining_time)
            metadata_by_id[request_id] = metadata
        except (KeyError, TypeError, ValueError, OverflowError):
            continue

    for peers in collective_groups.values():
        peer_data: list[tuple[Any, Mapping[str, Any], int, int]] = []
        group_remaining_time = 0.0
        for request in peers:
            request_id = str(request.request_id)
            metadata = metadata_by_id.get(request_id)
            if metadata is None:
                continue
            prompt_len, remaining_len = _lengths(request, metadata)
            group_remaining_time += unit_time_s * remaining_len
            peer_data.append((request, metadata, prompt_len, remaining_len))
        if not peer_data:
            continue

        group_remaining_time = max(group_remaining_time, 1e-6)
        total_reward = 0.0
        for request, metadata, prompt_len, remaining_len in peer_data:
            reward, _ = _reward(
                request,
                metadata,
                now,
                unit_time_s,
                penalty_factor,
                remaining_time_s=group_remaining_time,
            )
            total_reward += reward
            total_reward += _starvation_bonus(
                request, now, last_scheduled_at, starvation_delta, frame_time_s
            )
        priority = _to_priority(total_reward / group_remaining_time)
        for request, _, _, _ in peer_data:
            scores[str(request.request_id)] = priority

    return scores


def request_metadata(request: Any) -> Mapping[str, Any] | None:
    """Public accessor for the JITServe metadata attached to a request."""
    return _metadata(request)
