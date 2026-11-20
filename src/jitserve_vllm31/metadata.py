# SPDX-License-Identifier: Apache-2.0
"""Validation and normalization for JITServe request metadata."""

from __future__ import annotations

from math import isfinite
from typing import Any

SLO_EXTRA_ARGS_KEY = "jitserve_v1"
REQUEST_TYPES = {0: "latency", 1: "throughput", 2: "collective"}


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _request_type(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError("request_type must be latency, throughput, or collective")
    if isinstance(value, int):
        value = REQUEST_TYPES.get(value)
    elif isinstance(value, str):
        lowered = value.strip().lower()
        if lowered.isdigit():
            value = REQUEST_TYPES.get(int(lowered))
        else:
            value = lowered
    if value not in {"latency", "throughput", "collective"}:
        raise ValueError("request_type must be 0/latency, 1/throughput, or 2/collective")
    return value


def normalize_request_metadata(
    request_info: dict[str, Any],
    slo_constraint: tuple[float, float, float] | list[float] | None = None,
    *,
    timeout_s: float | None = None,
    received_at: float | None = None,
) -> dict[str, Any]:
    """Normalize a nearline timeout or a legacy SLO tuple into engine metadata."""
    if not isinstance(request_info, dict):
        raise ValueError("request_info must be an object")
    if timeout_s is not None:
        if slo_constraint is not None or request_info.get("deadline") is not None:
            raise ValueError("timeout_s cannot be combined with slo_constraint or request_info.deadline")
        request_type = _request_type(request_info.get("request_type", "throughput"))
        if request_type != "throughput":
            raise ValueError("timeout_s requires a throughput request")
        # Only the end-to-end deadline participates in throughput scoring.
        ttft_slo = tbt_slo = ttlt_slo = _finite_positive(timeout_s, "timeout_s")
    else:
        if not isinstance(slo_constraint, (tuple, list)) or len(slo_constraint) != 3:
            raise ValueError("provide timeout_s or slo_constraint containing (ttft, tbt, ttlt)")
        ttft_slo, tbt_slo, ttlt_slo = (
            _finite_positive(value, f"slo_constraint[{index}]")
            for index, value in enumerate(slo_constraint)
        )
        request_type = _request_type(request_info.get("request_type"))
    output_len = _positive_int(request_info.get("output_len"), "output_len")
    stages = _positive_int(request_info.get("num_stages", 1), "num_stages")

    collection_id = request_info.get("collection_id")
    if request_type == "collective":
        if isinstance(collection_id, bool) or not isinstance(collection_id, (int, str)):
            raise ValueError("collective requests require an integer or string collection_id")
    elif collection_id is not None and (
        isinstance(collection_id, bool) or not isinstance(collection_id, (int, str))
    ):
        raise ValueError("collection_id must be an integer, string, or null")

    deadline_value = request_info.get("deadline")
    if deadline_value is None:
        deadline = ttlt_slo * stages if request_type == "collective" else ttlt_slo
    else:
        deadline = _finite_positive(deadline_value, "deadline")

    metadata = {
        "version": 1,
        "request_type": request_type,
        "collection_id": collection_id,
        "num_stages": stages,
        "deadline_s": deadline,
        "ttft_slo_s": ttft_slo,
        "tbt_slo_s": tbt_slo,
        "ttlt_slo_s": ttlt_slo,
        "output_len": output_len,
    }
    if request_type == "throughput" and received_at is not None:
        received = _finite_positive(received_at, "received_at")
        metadata["deadline_at"] = _finite_positive(received + deadline, "deadline_at")
    return metadata
