# SPDX-License-Identifier: Apache-2.0
"""Local frontend-to-EngineCore acknowledgments for request-level goodput."""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import struct
import tempfile
import time

import zmq

_log = logging.getLogger(__name__)
# Private ticket, ASGI delivery timestamp, output tokens in a complete answer (zero on failure).
_RECEIPT = struct.Struct("!16sdI")


def _directory(config) -> Path | None:
    if config is None or config.parallel_config.data_parallel_size != 1:
        return None
    digest = hashlib.sha256(config.instance_id.encode()).hexdigest()[:20]
    return Path(tempfile.gettempdir()) / f"jitserve-{os.getuid()}-{digest}"


class DeliveryReceiver:
    def __init__(self, config):
        self.directory = _directory(config)
        self.socket = None
        if self.directory is None:
            return
        self.directory.mkdir(mode=0o700)
        try:
            self.socket = zmq.Context.instance().socket(zmq.PULL)
            self.socket.bind(f"ipc://{self.directory}/receipts")
        except Exception:
            self.close()
            raise

    def drain(self, feedback, length_history):
        if self.socket is None or not self.socket.poll(0, zmq.POLLIN):
            return
        while True:
            try:
                message = self.socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            if len(message) == _RECEIPT.size:
                ticket, when, tokens = _RECEIPT.unpack(message)
                if tokens:
                    length_history.observe(tokens)
                feedback.finish(ticket.hex(), tokens > 0, when)

    def close(self):
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        if self.directory is not None:
            (self.directory / 'receipts').unlink(missing_ok=True)
            self.directory.rmdir()
            self.directory = None


def delivery_sender(config):
    directory = _directory(config)
    if directory is None or not (directory / 'receipts').is_socket():
        return None
    sock = zmq.Context.instance().socket(zmq.PUSH)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"ipc://{directory}/receipts")
    _log.info('JITServe HTTP delivery feedback connected to %s', directory)
    return sock


class DeliveryMiddleware:
    """Report only after the final ASGI body write has succeeded."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['path'] != '/plugins/jitserve/generate':
            return await self.app(scope, receive, send)
        scope['jitserve.received_at'] = time.time()

        async def send_body(message):
            await send(message)
            if message['type'] == 'http.response.body' and not message.get('more_body'):
                receipt = scope.pop('jitserve.delivery', None)
                if receipt is not None:
                    sock, ticket = receipt
                    tokens = scope.get('jitserve.completed_tokens', 0)
                    try:
                        sock.send(_RECEIPT.pack(ticket, time.time(), tokens), zmq.NOBLOCK)
                    except zmq.ZMQError as exc:
                        _log.warning('JITServe delivery receipt failed: %s', exc)

        await self.app(scope, receive, send_body)
