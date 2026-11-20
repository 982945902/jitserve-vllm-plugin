# SPDX-License-Identifier: Apache-2.0
"""Metadata normalization tests (no vLLM import required)."""

import pytest

from jitserve_vllm31.metadata import normalize_request_metadata


def test_throughput_defaults():
    m = normalize_request_metadata(
        {"request_type": 1, "output_len": 2560}, (10.0, 0.05, 60.0)
    )
    assert m["request_type"] == "throughput"
    assert m["deadline_s"] == 60.0
    assert m["ttlt_slo_s"] == 60.0
    assert m["num_stages"] == 1
    assert m["version"] == 1


def test_latency_alias_and_string_type():
    m = normalize_request_metadata(
        {"request_type": "latency", "output_len": 400}, (5.0, 0.05, 20.0)
    )
    assert m["request_type"] == "latency"
    assert m["deadline_s"] == 20.0


def test_collective_expands_deadline_and_requires_collection_id():
    m = normalize_request_metadata(
        {"request_type": 2, "collection_id": 7, "output_len": 800, "num_stages": 3},
        (1.0, 0.08, 8.0),
    )
    assert m["deadline_s"] == 24.0
    assert m["request_type"] == "collective"
    with pytest.raises(ValueError):
        normalize_request_metadata({"request_type": 2, "output_len": 800}, (1.0, 0.08, 8.0))


def test_bad_slo_constraint_length():
    with pytest.raises(ValueError):
        normalize_request_metadata({"request_type": 1, "output_len": 100}, (10.0, 0.05))


def test_bad_output_len_and_request_type():
    with pytest.raises(ValueError):
        normalize_request_metadata({"request_type": 1, "output_len": 0}, (10.0, 0.05, 60.0))
    with pytest.raises(ValueError):
        normalize_request_metadata({"request_type": 9, "output_len": 100}, (10.0, 0.05, 60.0))


def test_timeout_defaults_to_throughput_and_ignores_client_deadline_at():
    meta = normalize_request_metadata({"output_len": 8, "deadline_at": 9999},
                                      timeout_s=30, received_at=100)
    assert meta["request_type"] == "throughput"
    assert meta["deadline_s"] == 30
    assert meta["deadline_at"] == 130


@pytest.mark.parametrize("invalid", [0, -5, True, float("nan"), float("inf")])
def test_invalid_timeout_or_receipt(invalid):
    with pytest.raises(ValueError):
        normalize_request_metadata({"output_len": 8}, timeout_s=invalid)
    with pytest.raises(ValueError):
        normalize_request_metadata({"output_len": 8}, timeout_s=30, received_at=invalid)


def test_timeout_and_legacy_deadline_definitions():
    info = {"output_len": 8, "request_type": 1, "deadline": 5}
    legacy = normalize_request_metadata(info, [1, 0.1, 10], received_at=100)
    assert legacy["deadline_at"] == 105
    with pytest.raises(ValueError):
        normalize_request_metadata(info, timeout_s=5)
    del info["deadline"]
    with pytest.raises(ValueError):
        normalize_request_metadata(info, [1, 0.1, 5], timeout_s=5)
    info["request_type"] = "latency"
    with pytest.raises(ValueError):
        normalize_request_metadata(info, timeout_s=5)
    assert "deadline_at" not in normalize_request_metadata(info, [1, 0.1, 5], received_at=100)
