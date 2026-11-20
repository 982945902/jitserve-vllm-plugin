# SPDX-License-Identifier: Apache-2.0
"""Phase 3: SLO-ordered preemption victim selection (offline, no vLLM).

V1's PRIORITY preemption branch takes its victim from ``running[-1]``.
The mixin sorts ``running`` by density (lower priority int = higher
scheduling rank, i.e. more earnable goodput per second, first), so the
request with the *lowest* density lands last and becomes the victim.

Goodput-density semantics (ported from V0 ``SLOPolicy``): a request whose
deadline has already slipped earns ~0 goodput, so its density collapses to
0 and it is deprioritized (soft graceful degradation), not rescued.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jitserve_vllm31.metadata import normalize_request_metadata  # noqa: E402
from jitserve_vllm31.priority import compute_request_priorities  # noqa: E402


class _SP:
    def __init__(self, meta):
        self.extra_args = {"jitserve_v1": meta}


class _Req:
    """Minimal stand-in for a running-mode V1 request."""

    def __init__(self, rid, ttlt, arrival=0.0, out_len=100, decoded=20):
        self.request_id = rid
        self.arrival_time = arrival
        self.num_prompt_tokens = 500
        self.max_tokens = None
        self.output_token_ids = list(range(decoded))
        self.priority = 0
        meta = normalize_request_metadata(
            {"request_type": 1, "output_len": out_len}, (2.0, 0.05, ttlt)
        )
        self.sampling_params = _SP(meta)


def _pri(running, now):
    return compute_request_priorities(
        running, now=now, unit_time_s=0.02, frame_time_s=1.0, penalty_factor=100
    )


def test_shorter_viable_job_outranks_longer_victim_last():
    """Both requests have full deadline headroom; the shorter remaining
    job has higher density and sorts first, leaving the longer one as
    ``running[-1]`` — the V1 preemption victim slot."""
    tight = _Req("tight", ttlt=30.0, out_len=200)   # remain 3.6s of work
    loose = _Req("loose", ttlt=600.0, out_len=400)  # remain 7.6s of work
    running = [loose, tight]  # deliberately wrong order
    priorities = _pri(running, now=10.0)
    assert priorities["tight"] < priorities["loose"] < 0
    running.sort(key=lambda r: priorities[r.request_id])
    assert [r.request_id for r in running] == ["tight", "loose"]
    assert running[-1].request_id == "loose"  # victim = lowest density


def test_doomed_request_density_collapses_to_victim_slot():
    """A request whose deadline already slipped earns ~0 goodput: its
    density collapses, priority falls back to 0, and it sorts LAST —
        soft graceful degradation, matching V0 ``-0.0`` underflow behavior.
    """
    healthy = _Req("healthy", ttlt=600.0, out_len=400)
    doomed = _Req("doomed", ttlt=5.0, out_len=200)  # deadline passed at now
    running = [doomed, healthy]
    priorities = _pri(running, now=10.0)
    assert priorities["doomed"] == 0  # density <= 0 -> native fallback
    assert priorities["healthy"] < 0
    running.sort(key=lambda r: priorities[r.request_id])
    assert running[-1].request_id == "doomed"


def test_no_metadata_request_keeps_native_zero_and_bubbles_last():
    """A request without plugin metadata falls back to priority 0; with
    other negative densities present it bubbles to the victim slot."""

    class _BareSP:
        pass

    urgent = _Req("urgent", ttlt=1.0, out_len=50)
    bare = _Req("bare", ttlt=999.0)
    bare.sampling_params = _BareSP()  # no jitserve_v1 key
    running = [bare, urgent]
    priorities = _pri(running, now=0.1)  # before urgent's deadline
    assert priorities["bare"] == 0
    assert priorities["urgent"] < priorities["bare"]
    running.sort(key=lambda r: priorities[r.request_id])
    assert [r.request_id for r in running] == ["urgent", "bare"]
    assert running[-1].request_id == "bare"


def test_equal_density_preserves_relative_order():
    """Stable-sort contract: equal priorities preserve insertion order."""
    a = _Req("a", ttlt=10.0, arrival=0.0, out_len=100)
    b = _Req("b", ttlt=10.0, arrival=0.0, out_len=100)
    running = [a, b]
    priorities = _pri(running, now=1.0)
    assert priorities["a"] == priorities["b"]
    running.sort(key=lambda r: priorities[r.request_id])
    assert [r.request_id for r in running] == ["a", "b"]
