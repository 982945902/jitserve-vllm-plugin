# SPDX-License-Identifier: Apache-2.0
# Adapted for this vLLM V1 plugin from the JITServe artifact. See NOTICE.
"""vLLM 0.31 scheduler adapters for JITServe SLO priorities."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import math
import os
import time
from typing import Any

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus
from vllm.logger import init_logger as _init_logger

_logger = _init_logger(__name__)
import logging as _logging
_logging.getLogger("jitserve_vllm31").parent = _logging.getLogger("vllm")

from .admission import select_admission_drops
from .delivery import DeliveryReceiver
from .gmax import CutoffTuner, RequestGoodputTracker, prioritize_window
from .priority import (
    OutputLengthHistory, compute_request_priorities, request_deadline_at, request_metadata,
)
from .request_queue import DynamicPriorityRequestQueue

_SCHEDULE_INTERVAL = 50
_DEFAULT_UNIT_TIME_S = 0.02


def _check_vllm_version() -> None:
    try:
        installed = version("vllm")
    except PackageNotFoundError as exc:
        raise RuntimeError("JITServe vLLM plugin requires vLLM 0.31.x") from exc
    if not installed.startswith("0.31."):
        raise RuntimeError(
            f"JITServe vLLM plugin supports vLLM 0.31.x; found {installed}"
        )


def _read_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _read_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    try:
        value = default if raw is None else float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return value


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class _JitServeSchedulerMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _check_vllm_version()
        vllm_config = kwargs.get("vllm_config", args[0] if args else None)
        if vllm_config is None:
            raise TypeError("vllm_config is required")
        policy = vllm_config.scheduler_config.policy
        if policy != "priority":
            raise ValueError(
                "JitServeScheduler requires --scheduling-policy priority"
            )

        super().__init__(*args, **kwargs)
        self.waiting = DynamicPriorityRequestQueue()
        self.kv_holding_waiting = DynamicPriorityRequestQueue()

        self._jitserve_penalty_factor = _read_int_env(
            "JITSERVE_PENALTY_FACTOR", 100, 1, 1000
        )
        self._jitserve_starvation_delta = _read_float_env(
            "JITSERVE_STARVATION_DELTA", 0.0
        )
        self._jitserve_max_waiting_s = _read_float_env(
            "JITSERVE_MAX_WAITING_S", 30.0
        )
        self._jitserve_preemption_order = os.environ.get(
            "JITSERVE_PREEMPTION_ORDER", "slo"
        ).lower()
        if self._jitserve_preemption_order not in ("slo", "none"):
            raise ValueError(
                "JITSERVE_PREEMPTION_ORDER must be 'slo' or 'none'"
            )

        # Phase 4 GMAX (paper Sec 4.2): online cutoff adaptation + window cap.
        self._jitserve_gmax_enabled = _read_bool_env("JITSERVE_GMAX", True)
        self._jitserve_gmax_window = _read_int_env(
            "JITSERVE_GMAX_WINDOW", 0, 0, 512
        )
        initial_cutoff = _read_float_env("JITSERVE_GMAX_CUTOFF", 0.95)
        adaptive = _read_bool_env("JITSERVE_GMAX_ADAPTIVE", True)
        self._jitserve_cutoff_tuner = CutoffTuner(
            initial_cutoff, adaptive=adaptive
        )
        feedback_window = _read_float_env("JITSERVE_GMAX_FEEDBACK_WINDOW_S", 30.0)
        self._jitserve_feedback = RequestGoodputTracker(self._jitserve_cutoff_tuner, feedback_window)
        self._jitserve_receipts = DeliveryReceiver(vllm_config) if self._jitserve_gmax_enabled else None
        self._jitserve_lengths = OutputLengthHistory()
        trace_path = os.environ.get("JITSERVE_TRACE_FILE", "")
        self._jitserve_trace_file = trace_path if trace_path else None

        self._jitserve_unit_time_s = _DEFAULT_UNIT_TIME_S
        self._jitserve_schedule_count = 0
        self._jitserve_last_step: float | None = None
        self._jitserve_busy_s = 0.0
        self._jitserve_busy_steps = 0
        self._jitserve_last_scheduled_at: dict[str, float] = {}

    def _update_unit_time(self, now_monotonic: float) -> bool:
        # A completed idle period has no running requests. Including the gap
        # would make the next burst look as if every token took seconds.
        if self.running and self._jitserve_last_step is not None:
            self._jitserve_busy_s += now_monotonic - self._jitserve_last_step
            self._jitserve_busy_steps += 1
        self._jitserve_last_step = now_monotonic
        self._jitserve_schedule_count += 1
        if self._jitserve_schedule_count % _SCHEDULE_INTERVAL != 0:
            return False
        if self._jitserve_busy_steps:
            sample = max(self._jitserve_busy_s / self._jitserve_busy_steps, 1e-6)
            self._jitserve_unit_time_s = 0.8 * self._jitserve_unit_time_s + 0.2 * sample
        self._jitserve_busy_s = 0.0
        self._jitserve_busy_steps = 0
        return True

    def _advance_feedback(self, now: float) -> None:
        if self._jitserve_receipts is not None:
            self._jitserve_receipts.drain(self._jitserve_feedback, self._jitserve_lengths)
        for result in self._jitserve_feedback.advance(now):
            line = (
                f"GMAX cohort cutoff={result.cutoff:.4f} arrivals={result.arrivals} "
                f"on_time={result.on_time} qpm={result.qpm:.4f} trained={int(result.trained)}"
            )
            _logger.info("%s", line)
            if self._jitserve_trace_file:
                try:
                    with open(self._jitserve_trace_file, "a") as trace:
                        trace.write(line + "\n")
                except OSError as exc:
                    _logger.warning("Disabling JITServe trace file: %s", exc)
                    self._jitserve_trace_file = None

    def add_request(self, request: Request) -> None:
        new_request = request.request_id not in self.requests
        super().add_request(request)
        if not new_request or request.resumable:
            return
        metadata = request_metadata(request)
        if not metadata or metadata.get("request_type") != "throughput" or metadata.get("multi_output"):
            return
        if metadata.get("learn_output_len") and isinstance(metadata, dict):
            estimate = self._jitserve_lengths.estimate(request.max_tokens)
            if estimate is not None:
                metadata["output_len"] = estimate
        if self._jitserve_gmax_enabled:
            try:
                deadline = request_deadline_at(request, metadata)
            except (ValueError, TypeError, KeyError, OverflowError):
                return
            now = time.time()
            self._advance_feedback(now)
            key = metadata.get("delivery_ticket", request.request_id)
            if isinstance(key, str):
                self._jitserve_feedback.arrive(key, deadline, now)

    def _free_request(self, request: Request, delay_free_blocks: bool = False):
        # Both update_from_output and finish_requests use this hook. KV block
        # release can be deferred, so settling on disappearance is too late.
        now = time.time()
        result = super()._free_request(request, delay_free_blocks=delay_free_blocks)
        metadata = request_metadata(request)
        if metadata and metadata.get("request_type") == "throughput" and not (
            metadata.get("multi_output") or request.resumable
        ):
            complete = request.status == RequestStatus.FINISHED_STOPPED and request.num_output_tokens > 0
            if complete and "delivery_ticket" not in metadata:
                self._jitserve_lengths.observe(request.num_output_tokens)
            if self._jitserve_gmax_enabled:
                # The HTTP frontend can stop on a string or lose the client
                # after generation. Its acknowledgment is authoritative.
                if "delivery_ticket" not in metadata:
                    self._jitserve_feedback.finish(request.request_id, complete, now)
                self._advance_feedback(now)
        return result

    def shutdown(self) -> None:
        try:
            if self._jitserve_receipts is not None:
                self._advance_feedback(time.time())
                self._jitserve_receipts.close()
        finally:
            super().shutdown()

    def _apply_gmax_window(self, priorities: dict[str, int]) -> None:
        free_slots = self.max_num_active_reqs - len(self.running) - self.num_waiting_for_streaming_input
        window_size = min(free_slots, self.max_num_scheduled_tokens)
        if self._jitserve_gmax_window:
            window_size = min(window_size, self._jitserve_gmax_window)
        if window_size <= 0 or len(self.waiting) + len(self.kv_holding_waiting) <= window_size:
            return
        waiters = [
            r for r in self.requests.values()
            if r.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED) and request_metadata(r)
        ]
        if len(waiters) <= window_size:
            return
        prioritize_window(waiters, priorities, window_size, self._jitserve_cutoff_tuner.current)
        if window_size > 1 and any(request_metadata(r).get("request_type") == "throughput" for r in waiters):
            self._jitserve_feedback.mark_contention()

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        now_monotonic = time.monotonic()
        now_wall = time.time()
        frame_boundary = self._update_unit_time(now_monotonic)
        if frame_boundary and self._jitserve_gmax_enabled:
            self._advance_feedback(now_wall)

        priorities = compute_request_priorities(
            self.requests.values(),
            now=now_wall,
            unit_time_s=self._jitserve_unit_time_s,
            frame_time_s=_SCHEDULE_INTERVAL * self._jitserve_unit_time_s,
            penalty_factor=self._jitserve_penalty_factor,
            starvation_delta=self._jitserve_starvation_delta,
            last_scheduled_at=self._jitserve_last_scheduled_at,
        )

        if self._jitserve_gmax_enabled:
            self._apply_gmax_window(priorities)
        self.waiting.refresh(priorities)
        self.kv_holding_waiting.refresh(priorities)

        # Phase 3: SLO-ordered preemption victim selection. V1's PRIORITY
        # branch takes the preemption victim from ``self.running[-1]``;
        # sorting running by density leaves the lowest-SLO request last.
        # Recompute-only (V1 has no SWAP); the V0 cost-estimate
        # (``_preemption_worth_it``) is intentionally not ported.
        if self._jitserve_preemption_order == "slo":
            self.running.sort(
                key=lambda r: priorities.get(str(r.request_id), 0)
            )

        dropped_ids = select_admission_drops(
            self.requests.values(),
            now=now_wall,
            limit=self._jitserve_max_waiting_s,
            waiting_status=RequestStatus.WAITING,
            ever_scheduled=set(self._jitserve_last_scheduled_at),
        )
        if _logger.isEnabledFor(20):  # INFO
            n_wait = sum(
                1
                for r in self.requests.values()
                if r.status == RequestStatus.WAITING
            )
            n_run = len(self.running)
            _logger.info(
                "JITServe scan: requests=%d waiting=%d running=%d "
                "drop_limit=%s gmax_p=%.3f",
                len(self.requests),
                n_wait,
                n_run,
                self._jitserve_max_waiting_s,
                self._jitserve_cutoff_tuner.current,
            )
        if dropped_ids:
            _logger.warning(
                "JITServe admission: dropping %d requests (%s)",
                len(dropped_ids),
                ",".join(sorted(dropped_ids)),
            )
            self.finish_requests(dropped_ids, RequestStatus.FINISHED_ABORTED)

        output = super().schedule(throttle_prefills=throttle_prefills)
        preempted = getattr(output, "preempted_req_ids", None) or ()
        if preempted:
            _logger.warning(
                "JITServe preemption: recompute victims=%s",
                sorted(preempted),
            )

        for request_id in output.num_scheduled_tokens:
            self._jitserve_last_scheduled_at[str(request_id)] = now_wall
        self._jitserve_last_scheduled_at = {
            request_id: ts
            for request_id, ts in self._jitserve_last_scheduled_at.items()
            if request_id in self.requests
        }
        return output


class JitServeScheduler(_JitServeSchedulerMixin, AsyncScheduler):
    """Async vLLM scheduler with dynamic JITServe density ranking."""


class JitServeSyncScheduler(_JitServeSchedulerMixin, Scheduler):
    """Synchronous vLLM scheduler with dynamic JITServe density ranking."""
