# SPDX-License-Identifier: Apache-2.0
# Adapted for this vLLM V1 plugin from the JITServe artifact. See NOTICE.
"""Optional QRF length prediction for the JitServe vLLM 0.31 plugin.

Environment-gated: only enabled when both ``JITSERVE_QRF_PORT`` and
``JITSERVE_QRF_HOST`` are set (the prediction server is an external
process started by the operator). All failures
fall back to the request's declared ``output_len``, mirroring the V0
``_maybe_start_prediction`` semantics where a missing/failed prediction
keeps the caller's placeholder length.
"""

from __future__ import annotations

import queue
import socket
import struct
from typing import Any

_HEADER = struct.Struct("!I")
_POOL: "queue.SimpleQueue[socket.socket] | None" = None
_HOST = ""
_PORT = 0
_TIMEOUT = 1.0


def configure_from_env(environ: Any = None) -> bool:
    """Configure the caller-side pool from environment variables.

    Returns True when prediction is enabled. Called once at import time
    by :mod:`jitserve_vllm31.endpoint`; safe to call repeatedly.
    """
    global _POOL, _HOST, _PORT, _TIMEOUT
    if environ is None:
        import os

        environ = os.environ
    host = environ.get("JITSERVE_QRF_HOST", "")
    port_raw = environ.get("JITSERVE_QRF_PORT", "")
    if not host or not port_raw:
        return False
    try:
        port = int(port_raw)
    except ValueError:
        return False
    if not (1 <= port <= 65535):
        return False
    _HOST, _PORT = host, port
    _TIMEOUT = float(environ.get("JITSERVE_QRF_TIMEOUT", "1.0"))
    if environ.get("JITSERVE_QRF_SPAWN", "") not in ("1", "yes", "true"):
        _POOL = queue.SimpleQueue()
    else:
        # Spawn mode is operator-controlled; the plugin only
        # pools connections when told the server already exists.
        _POOL = queue.SimpleQueue()
    return True


def is_enabled() -> bool:
    return bool(_HOST and _PORT)


def _send(sock: socket.socket, obj: Any) -> None:
    import pickle

    data = pickle.dumps(obj)
    sock.sendall(_HEADER.pack(len(data)) + data)


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _recv(sock: socket.socket) -> Any | None:
    import pickle

    header = _recv_exact(sock, _HEADER.size)
    if header is None:
        return None
    body = _recv_exact(sock, _HEADER.unpack(header)[0])
    return None if body is None else pickle.loads(body)


def predict_output_len(prompts: list[str]) -> int | None:
    """Return the q0.95 upper bound for ``prompts[0]`` or None on failure.

    Mirrors ``jitserve.request_analyzer.prediction.PredictionClient``:
    a persistent pooled TCP connection, 4-byte big-endian length header
    (``struct "!I"``) + pickle body; request
    ``{"prompt": [str, ...]}``, reply ``{"output_len": [int, ...]}``.
    A reply bound of -1 (too few tail samples) also maps to None so the
    caller keeps its declared length.
    """
    if not is_enabled() or _POOL is None or not prompts:
        return None
    try:
        sock = _POOL.get_nowait()
    except queue.Empty:
        try:
            sock = socket.create_connection((_HOST, _PORT), timeout=_TIMEOUT)
        except OSError:
            return None
    try:
        _send(sock, {"prompt": prompts})
        reply = _recv(sock)
    except OSError:
        sock.close()
        return None
    if reply is None:
        sock.close()
        return None
    _POOL.put(sock)
    bounds = reply.get("output_len") or []
    if not bounds:
        return None
    bound = bounds[0]
    if not isinstance(bound, int) or bound <= 0:
        return None
    return bound
