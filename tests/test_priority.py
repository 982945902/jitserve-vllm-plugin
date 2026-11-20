# SPDX-License-Identifier: Apache-2.0
"""Priority density invariants (no vLLM import required)."""

import pytest

from jitserve_vllm31.metadata import normalize_request_metadata
from jitserve_vllm31.priority import OutputLengthHistory, compute_request_priorities, request_deadline_at

NOW = 100.0


class _SP:
    def __init__(self, meta):
        self.extra_args = {"jitserve_v1": meta}


class Req:
    def __init__(self, rid, arrival, rtype, out_len, slo=(10, 0.05, 60), cid=None):
        self.request_id = rid
        self.num_prompt_tokens = 2500
        self.arrival_time = arrival
        self.max_tokens = None
        self.output_token_ids = []
        self.priority = 0
        self.sampling_params = _SP(
            normalize_request_metadata(
                {"request_type": rtype, "output_len": out_len, "collection_id": cid}, slo
            )
        )


def _p(reqs, **kw):
    return compute_request_priorities(
        reqs,
        now=NOW,
        unit_time_s=0.02,
        frame_time_s=1.0,
        penalty_factor=100,
        **kw,
    )


def test_tighter_deadline_outranks_loose():
    p = _p(
        [
            Req("tight", NOW - 1.0, "throughput", 400, (10, 0.05, 30)),
            Req("loose", NOW - 1.0, "throughput", 2560, (10, 0.05, 120)),
        ]
    )
    assert p["tight"] < p["loose"]


def test_shorter_job_has_higher_density():
    p = _p(
        [
            Req("small", NOW - 1.0, "throughput", 100),
            Req("big", NOW - 1.0, "throughput", 2560),
        ]
    )
    assert p["small"] < p["big"]


def test_missing_metadata_falls_back_to_native_priority():
    req = Req("r", NOW - 1.0, "throughput", 2560)
    req.sampling_params = object()
    req.priority = 37
    assert _p([req])["r"] == 37


def test_collective_peers_share_priority():
    p = _p(
        [
            Req("ca", NOW - 1.0, "collective", 1000, cid=5),
            Req("cb", NOW - 1.0, "collective", 500, cid=5),
        ]
    )
    assert p["ca"] == p["cb"]


def test_latency_decode_reward_is_finite():
    lat = Req("lat", NOW - 1.0, "latency", 400, (1.0, 0.05, 20.0))
    lat.output_token_ids = [1, 2, 3]
    assert isinstance(_p([lat])["lat"], int)


def test_deadline_near_zero_clamps_without_crash():
    req = Req("expiring", NOW - 19.99, "throughput", 2560, (10, 0.05, 20.0))
    assert _p([req])["expiring"] >= 0


def test_throughput_payoff_is_one_completed_request():
    short = Req("short", NOW - 10, "throughput", 50)
    long_prompt = Req("long_prompt", NOW - 1, "throughput", 50)
    long_prompt.num_prompt_tokens = 90_000
    decoding = Req("decoding", NOW - 1, "throughput", 200)
    decoding.output_token_ids = [0] * 150
    long = Req("long", NOW - 1, "throughput", 100)
    scores = _p([short, long_prompt, decoding, long], starvation_delta=1)
    assert scores == {"short": -1_000_000, "long_prompt": -1_000_000,
                      "decoding": -1_000_000, "long": -500_000}


def test_throughput_discounts_by_actual_remaining_deadline():
    req = Req("r", NOW, "throughput", 100)
    meta = req.sampling_params.extra_args["jitserve_v1"]
    assert request_deadline_at(req, meta) == NOW + 60
    meta["deadline_at"] = NOW + 1
    scores = compute_request_priorities([req], now=NOW, unit_time_s=0.02,
                                        frame_time_s=1, penalty_factor=2)
    # One second left for two seconds of work. (1/2)^2 Q / 2 s.
    assert scores["r"] == -125_000
    meta["deadline_at"] = NOW
    assert _p([req])["r"] == 0


@pytest.mark.parametrize("invalid", [None, True, 0, -1, "100", float("inf"), float("nan"), 10**400])
def test_invalid_absolute_deadline_does_not_restart_budget(invalid):
    req = Req("bad", NOW, "throughput", 10)
    meta = req.sampling_params.extra_args["jitserve_v1"]
    meta["deadline_at"] = invalid
    with pytest.raises(ValueError):
        request_deadline_at(req, meta)
    req.priority = 12
    assert _p([req])["bad"] == 12


def test_legacy_deadline_rejects_invalid_duration_or_arrival():
    req = Req("old", NOW, "throughput", 10)
    for duration in [0, -1, None, True, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            request_deadline_at(req, {"deadline_s": duration})
    req.arrival_time = float("nan")
    with pytest.raises(ValueError):
        request_deadline_at(req, {"deadline_s": 10})


def test_original_collective_and_latency_token_rewards():
    a = Req("a", NOW, "collective", 50, cid=1)
    b = Req("b", NOW, "collective", 100, cid=1)
    lat = Req("lat", NOW, "latency", 100)
    peers = -round((2 * 2500 + 150 * 8) / 3 * 1_000_000)
    assert _p([a, b, lat]) == {"a": peers, "b": peers, "lat": -1_650_000_000}


def test_output_length_history_uses_recent_positive_lengths_and_nearest_rank():
    history = OutputLengthHistory(capacity=20)
    history.observe(0)
    assert history.estimate(max_tokens=32) is None
    for length in range(1, 22):
        history.observe(length)
    # Recent lengths are 2..21, and the nineteenth observation is 20.
    assert history.estimate(None) == 20
    assert history.estimate(12) == 12
