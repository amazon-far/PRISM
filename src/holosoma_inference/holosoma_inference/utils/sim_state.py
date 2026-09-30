"""ZMQ subscriber for MuJoCo simulator state."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger


class SimStateSub:
    """Subscribe to simulator root/object state for split sim rollout observations."""

    def __init__(self, port: int = 5557) -> None:
        self.port = int(port)
        self.context: Any | None = None
        self.socket: Any | None = None
        self._zmq: Any | None = None
        self.last_state: dict | None = None

    def start(self) -> None:
        import zmq

        self._zmq = zmq
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://localhost:{self.port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.RCVTIMEO, 10)
        logger.info("Sim state subscriber started, connecting to port {}", self.port)

    def _drain_messages(self) -> None:
        zmq = self._zmq
        if self.socket is None or zmq is None:
            return
        while True:
            try:
                self.last_state = json.loads(self.socket.recv_string(zmq.NOBLOCK))
            except zmq.Again:
                break

    def get_state(self) -> dict | None:
        self._drain_messages()
        return self.last_state

    def close(self) -> None:
        socket = self.socket
        context = self.context
        self.socket = None
        self.context = None
        self._zmq = None
        if socket is not None:
            socket.close(0)
        if context is not None:
            context.term()
