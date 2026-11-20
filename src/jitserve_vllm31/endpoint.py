# SPDX-License-Identifier: Apache-2.0
"""Opt-in HTTP route carrying JITServe SLO metadata into vLLM V1."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from vllm import SamplingParams

from .delivery import DeliveryMiddleware, delivery_sender
from .metadata import SLO_EXTRA_ARGS_KEY, normalize_request_metadata
from . import qrf

_QRF_ENABLED = qrf.configure_from_env()
_FALLBACK_OUTPUT_LEN = 1024
_STATE_ATTRIBUTE = "jitserve_vllm31_engine_client"
_FIRST_TOKEN_WATCHDOG_MULT = 2.0
_FIRST_TOKEN_WATCHDOG_MIN_S = 10.0


def _first_token_watchdog_s(ttft_slo: float, remaining_timeout_s: float | None = None) -> float | None:
    """Seconds to wait for the first stream frame before declaring a drop.

    Scheduler-side admission removes the request, but V1's batch-queue
    output path may never deliver a terminal frame for it, leaving the
    frontend generator pending forever. The watchdog converts that hang
    into an explicit drop frame. ``<=0`` disables it (original behavior).
    """
    raw = os.environ.get("JITSERVE_FIRST_TOKEN_WATCHDOG_S")
    if raw is not None:
        try:
            value = float(raw)
        except ValueError:
            return None
        if value <= 0:
            return None
        return max(value, remaining_timeout_s or 0.0)
    if remaining_timeout_s is not None:
        return max(0.0, remaining_timeout_s)
    return max(_FIRST_TOKEN_WATCHDOG_MULT * ttft_slo, _FIRST_TOKEN_WATCHDOG_MIN_S)


class _FirstTokenWatchdog(Exception):
    """First stream frame did not arrive within the watchdog window."""


class _RequestInfo(BaseModel):
    request_type: int | str = "throughput"
    output_len: int | None = Field(default=None, gt=0)
    collection_id: int | str | None = None
    deadline: float | None = None
    num_stages: int = Field(default=1, ge=1)


class _GenerateRequest(BaseModel):
    prompt: str
    request_info: _RequestInfo | None = None
    slo_constraint: tuple[float, float, float] | None = None
    timeout_s: float | None = Field(default=None, gt=0, allow_inf_nan=False, strict=True)
    output_len: int | None = Field(default=None, gt=0, strict=True)
    sampling_params: dict[str, Any]
    request_id: str | None = None
    priority: int = 0
    stream: bool = True


def _drop_frame(started_at: float) -> dict[str, Any]:
    return {
        "text": [""],
        "ttft": None,
        "tbt": None,
        "output_tokens": 0,
        "delta_tokens": 0,
        "finish_reason": "abort",
        "dropped": True,
        "e2e": time.monotonic() - started_at,
    }


class JitServeEndpointPlugin:
    """Expose `/plugins/jitserve/generate` for SLO-aware V1 scheduling."""

    name = "jitserve_api"
    required_tasks: tuple[Literal["generate"], ...] = ("generate",)

    def attach_router(self, app: FastAPI) -> None:
        app.add_middleware(DeliveryMiddleware)
        @app.post("/plugins/jitserve/generate")
        async def generate(body: _GenerateRequest, raw_request: Request):
            received_at = raw_request.scope.get("jitserve.received_at", time.time())
            started_at = time.monotonic()
            engine_client = getattr(raw_request.app.state, _STATE_ATTRIBUTE, None)
            if engine_client is None:
                raise HTTPException(status_code=503, detail="vLLM engine is unavailable")

            try:
                info = body.request_info.model_dump(exclude_unset=True) if body.request_info else {}
                if body.output_len is not None:
                    if info.get("output_len") is not None:
                        raise ValueError("specify output_len at only one level")
                    info["output_len"] = body.output_len
                needs_prediction = info.get("output_len") is None
                if needs_prediction:
                    info["output_len"] = _FALLBACK_OUTPUT_LEN
                request_metadata = normalize_request_metadata(
                    info, body.slo_constraint, timeout_s=body.timeout_s, received_at=received_at
                )
                n = body.sampling_params.get("n", 1)
                if body.timeout_s is not None and (type(n) is not int or n != 1):
                    raise ValueError("timeout_s requires sampling_params.n = 1")
                if request_metadata["request_type"] == "throughput" and n != 1:
                    request_metadata["multi_output"] = True
                sender = raw_request.app.state.jitserve_delivery_sender
                if sender is not None and request_metadata["request_type"] == "throughput" and n == 1:
                    ticket = uuid.uuid4()
                    request_metadata["delivery_ticket"] = ticket.hex
                    raw_request.scope["jitserve.delivery"] = (sender, ticket.bytes)
                if needs_prediction:
                    predicted = (
                        await asyncio.to_thread(qrf.predict_output_len, [body.prompt])
                        if _QRF_ENABLED else None
                    )
                    if predicted is not None and predicted > 0:
                        limit = body.sampling_params.get("max_tokens")
                        if isinstance(limit, int) and limit > 0:
                            predicted = min(predicted, limit)
                        request_metadata["output_len"] = predicted
                    elif request_metadata["request_type"] == "throughput" and n == 1:
                        request_metadata["learn_output_len"] = True
                sampling_args = dict(body.sampling_params)
                extra_args = sampling_args.pop("extra_args", {}) or {}
                if not isinstance(extra_args, dict):
                    raise ValueError("sampling_params.extra_args must be an object")
                extra_args = dict(extra_args)
                extra_args[SLO_EXTRA_ARGS_KEY] = request_metadata
                sampling_args["extra_args"] = extra_args
                sampling_params = SamplingParams(**sampling_args)
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

            request_id = body.request_id or uuid.uuid4().hex
            engine_client.check_admission(n=1, request_id=request_id)
            watchdog = _first_token_watchdog_s(
                request_metadata["ttft_slo_s"],
                request_metadata["deadline_at"] - time.time() if body.timeout_s is not None else None,
            )

            async def _emit(output: Any) -> dict[str, Any] | None:
                if not output.outputs:
                    return None
                nonlocal first_token_at, previous_chunk_at, previous_token_count
                completion = output.outputs[0]
                now = time.monotonic()
                token_count = len(completion.token_ids)
                delta_tokens = max(0, token_count - previous_token_count)
                if first_token_at is None and token_count:
                    first_token_at = now
                ttft = (
                    first_token_at - started_at
                    if first_token_at is not None else None
                )
                tbt = (
                    (now - previous_chunk_at) / delta_tokens
                    if previous_chunk_at is not None and delta_tokens
                    else None
                )
                if delta_tokens:
                    previous_chunk_at = now
                previous_token_count = token_count
                if (output.finished and "jitserve.delivery" in raw_request.scope
                        and completion.finish_reason == "stop" and completion.text):
                    raw_request.scope["jitserve.completed_tokens"] = token_count
                return {
                    "text": [completion.text],
                    "ttft": ttft,
                    "tbt": tbt,
                    "output_tokens": token_count,
                    "delta_tokens": delta_tokens,
                    "finish_reason": completion.finish_reason,
                    "dropped": (
                        completion.finish_reason == "abort"
                        and token_count == 0
                    ),
                    "e2e": now - started_at if output.finished else None,
                }

            async def output_events():
                generator = engine_client.generate(
                    body.prompt,
                    sampling_params,
                    request_id,
                    priority=body.priority,
                )
                try:
                    try:
                        first = (
                            await asyncio.wait_for(
                                generator.__anext__(), timeout=watchdog
                            )
                            if watchdog is not None
                            else await generator.__anext__()
                        )
                    except (asyncio.TimeoutError, StopAsyncIteration):
                        # Either V1 delivered no terminal frame for a
                        # scheduler-side admission drop (generator ends
                        # with zero frames), or the first frame never
                        # arrived within the watchdog (generator hangs).
                        # The scheduler-side drop is logged on the engine
                        # core; surface it to the client explicitly.
                        await engine_client.abort(request_id)
                        yield _drop_frame(started_at)
                        return
                    frame = await _emit(first)
                    if frame is not None:
                        yield frame
                    async for output in generator:
                        frame = await _emit(output)
                        if frame is not None:
                            yield frame
                except asyncio.CancelledError:
                    await engine_client.abort(request_id)
                    raise
                finally:
                    await generator.aclose()

            first_token_at: float | None = None
            previous_chunk_at: float | None = None
            previous_token_count = 0

            if body.stream:
                async def encoded_events():
                    async for event in output_events():
                        yield (json.dumps(event, separators=(",", ":")) + "\0").encode()

                return StreamingResponse(
                    encoded_events(), media_type="application/json"
                )

            final_event: dict[str, Any] | None = None
            async for event in output_events():
                final_event = event
            if final_event is None:
                raise HTTPException(status_code=502, detail="vLLM returned no output")
            return JSONResponse(final_event)

    async def init_state(self, engine_client, state, args) -> None:
        setattr(state, _STATE_ATTRIBUTE, engine_client)
        state.jitserve_delivery_sender = delivery_sender(getattr(engine_client, "vllm_config", None))
