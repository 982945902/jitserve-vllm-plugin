# SPDX-License-Identifier: Apache-2.0
"""Offline tests for the first-token watchdog drop-frame path."""

import asyncio
import importlib.util
from pathlib import Path
import sys
import types

import pytest

_PKG_SRC = Path(__file__).resolve().parents[1] / "src" / "jitserve_vllm31"


class _HTTPException(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code


def _load_module(monkeypatch):
    fake = types.ModuleType("fastapi")
    fake.FastAPI = fake.Request = object
    fake.HTTPException = _HTTPException
    fake_responses = types.ModuleType("fastapi.responses")
    fake_responses.JSONResponse = fake_responses.StreamingResponse = lambda value: value
    fake_pydantic = types.ModuleType("pydantic")
    fake_pydantic.BaseModel = object
    fake_pydantic.Field = lambda *a, **k: None
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.SamplingParams = types.SimpleNamespace
    monkeypatch.setitem(sys.modules, "fastapi", fake)
    monkeypatch.setitem(sys.modules, "fastapi.responses", fake_responses)
    monkeypatch.setitem(sys.modules, "pydantic", fake_pydantic)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.syspath_prepend(str(_PKG_SRC.parent))
    spec = importlib.util.spec_from_file_location("jitserve_vllm31.endpoint", _PKG_SRC / "endpoint.py")
    endpoint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(endpoint)
    return endpoint


def test_drop_frame_shape(monkeypatch):
    frame = _load_module(monkeypatch)._drop_frame(1.0)
    assert frame["text"] == [""]
    assert frame["ttft"] is None and frame["tbt"] is None
    assert frame["output_tokens"] == 0 and frame["delta_tokens"] == 0
    assert frame["finish_reason"] == "abort"
    assert frame["dropped"] is True
    assert frame["e2e"] >= 0


def test_watchdog_default_and_env(monkeypatch):
    mod = _load_module(monkeypatch)
    monkeypatch.delenv("JITSERVE_FIRST_TOKEN_WATCHDOG_S", raising=False)
    assert mod._first_token_watchdog_s(2.0) == 10.0
    assert mod._first_token_watchdog_s(20.0) == 40.0
    monkeypatch.setenv("JITSERVE_FIRST_TOKEN_WATCHDOG_S", "0.5")
    assert mod._first_token_watchdog_s(2.0) == 0.5
    monkeypatch.setenv("JITSERVE_FIRST_TOKEN_WATCHDOG_S", "0")
    assert mod._first_token_watchdog_s(2.0) is None


def _submit(endpoint, payload):
    routes = {}
    submitted = {}

    async def generate(prompt, params, request_id, priority):
        submitted.update(params.extra_args)
        completion = types.SimpleNamespace(text="done", token_ids=[1], finish_reason="stop")
        yield types.SimpleNamespace(outputs=[completion], finished=True)

    app = types.SimpleNamespace(post=lambda path: lambda handler: routes.setdefault(path, handler),
                                add_middleware=lambda cls: None, state=types.SimpleNamespace())
    engine = types.SimpleNamespace(check_admission=lambda **kwargs: None, generate=generate)
    plugin = endpoint.JitServeEndpointPlugin()
    asyncio.run(plugin.init_state(engine, app.state, None))
    plugin.attach_router(app)
    body = dict(prompt="hello", sampling_params={}, timeout_s=None, slo_constraint=None,
                request_info=None, output_len=None, request_id="r", priority=0, stream=False)
    body.update(payload)
    info = body["request_info"]
    if info is not None:
        body["request_info"] = types.SimpleNamespace(model_dump=lambda **kwargs: info)
    response = asyncio.run(routes["/plugins/jitserve/generate"](
        types.SimpleNamespace(**body), types.SimpleNamespace(app=app, scope={})
    ))
    assert response["finish_reason"] == "stop"
    return submitted


def test_nearline_route_deadline_precedes_qrf_and_cannot_be_overridden(monkeypatch):
    endpoint = _load_module(monkeypatch)
    wall = [100.0]
    monkeypatch.setattr(endpoint, "time", types.SimpleNamespace(time=lambda: wall[0], monotonic=lambda: wall[0]))
    monkeypatch.setattr(endpoint, "_QRF_ENABLED", True)

    def predict(prompts):
        assert prompts == ["hello"]
        wall[0] += 15
        return 80

    monkeypatch.setattr(endpoint.qrf, "predict_output_len", predict)
    observed = []
    real_watchdog = endpoint._first_token_watchdog_s
    monkeypatch.setenv("JITSERVE_FIRST_TOKEN_WATCHDOG_S", "1")

    def capture_watchdog(*args):
        observed.append(real_watchdog(*args))
        return observed[-1]

    monkeypatch.setattr(endpoint, "_first_token_watchdog_s", capture_watchdog)
    args = {"max_tokens": 64, "extra_args": {"jitserve_v1": {"deadline_at": 999}, "other": 1}}
    result = _submit(endpoint, {"timeout_s": 60, "sampling_params": args})
    meta = result["jitserve_v1"]
    assert result["other"] == 1 and meta["deadline_at"] == 160
    assert observed == [45]
    assert meta["output_len"] == 64 and "learn_output_len" not in meta


def test_endpoint_predictor_sources_and_multi_output(monkeypatch):
    endpoint = _load_module(monkeypatch)
    monkeypatch.setattr(endpoint, "_QRF_ENABLED", True)
    calls = []
    monkeypatch.setattr(endpoint.qrf, "predict_output_len", lambda prompts: calls.append(prompts))
    nearline = {"timeout_s": 30, "output_len": 42}
    meta = _submit(endpoint, nearline)["jitserve_v1"]
    assert calls == [] and meta["output_len"] == 42 and "learn_output_len" not in meta
    del nearline["output_len"]
    meta = _submit(endpoint, nearline)["jitserve_v1"]
    assert calls == [["hello"]] and meta["learn_output_len"] is True
    assert meta["output_len"] == 1024 and "multi_output" not in meta
    legacy = {"slo_constraint": [1, 0.01, 30], "sampling_params": {"n": 2},
              "request_info": {"request_type": 1, "deadline": 20}}
    meta = _submit(endpoint, legacy)["jitserve_v1"]
    assert meta["deadline_s"] == 20 and meta["multi_output"] is True
    assert "learn_output_len" not in meta


@pytest.mark.parametrize("overrides", [
    {"sampling_params": {"n": 2}}, {"sampling_params": {"n": True}},
    {"slo_constraint": [1, 0.1, 60]}, {"request_info": {"deadline": 20}},
    {"request_info": {"request_type": "collective"}},
    {"request_info": {"output_len": 32}, "output_len": 64},
])
def test_nearline_route_rejects_ambiguous_payloads(monkeypatch, overrides):
    endpoint = _load_module(monkeypatch)
    with pytest.raises(_HTTPException) as exc:
        _submit(endpoint, {"timeout_s": 30, **overrides})
    assert exc.value.status_code == 422
