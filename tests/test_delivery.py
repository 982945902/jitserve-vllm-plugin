# SPDX-License-Identifier: Apache-2.0
"""Exercise the ASGI write boundary and the real cross-process receipt protocol."""
import asyncio
import time
import uuid
from types import SimpleNamespace

import pytest
import zmq

from jitserve_vllm31.delivery import DeliveryMiddleware, DeliveryReceiver, delivery_sender
from jitserve_vllm31.gmax import CutoffTuner, RequestGoodputTracker
from jitserve_vllm31.priority import OutputLengthHistory


def test_final_body_controls_qpm_and_learned_length():
    config = SimpleNamespace(instance_id=uuid.uuid4().hex,
                             parallel_config=SimpleNamespace(data_parallel_size=1))
    receiver = DeliveryReceiver(config)
    sender = delivery_sender(config)
    history = OutputLengthHistory()
    tracker = RequestGoodputTracker(CutoffTuner(0.95), window_s=5)
    start = time.time()

    async def transfer(tokens, disconnect):
        ticket = uuid.uuid4()
        tracker.arrive(ticket.hex, start + 5, start)
        scope = {'type': 'http', 'path': '/plugins/jitserve/generate',
                 'jitserve.delivery': (sender, ticket.bytes), 'jitserve.completed_tokens': tokens}

        async def app(scope, receive, send):
            await send({'type': 'http.response.body', 'body': b'first', 'more_body': True})
            assert not receiver.socket.poll(0, zmq.POLLIN)
            await send({'type': 'http.response.body', 'body': b'', 'more_body': False})

        async def write_body(message):
            if disconnect and not message['more_body']:
                raise ConnectionResetError

        await DeliveryMiddleware(app)(scope, None, write_body)

    try:
        for tokens in (7, 0):
            asyncio.run(transfer(tokens, False))
            assert receiver.socket.poll(1000, zmq.POLLIN)
            receiver.drain(tracker, history)
        with pytest.raises(ConnectionResetError):
            asyncio.run(transfer(123, True))
        receiver.drain(tracker, history)
        result, = tracker.advance(start + 6)
        assert (result.arrivals, result.on_time, result.qpm) == (3, 1, 12)
        assert history.estimate(1024) == 7
    finally:
        sender.close(linger=0)
        receiver.close()
