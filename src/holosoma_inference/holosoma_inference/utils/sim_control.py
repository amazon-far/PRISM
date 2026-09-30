"""ZMQ helpers for split sim2sim simulator control requests."""

from __future__ import annotations

import json
import time
from typing import Any

from loguru import logger


class SimControlPush:
    """Send split-sim control and lowcmd payloads."""

    def __init__(self, port: int = 5559) -> None:
        self.port = int(port)
        self.context: Any | None = None
        self.socket: Any | None = None
        self._zmq: Any | None = None
        self.enabled = False

    def start(self) -> None:
        try:
            import zmq

            self._zmq = zmq
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUSH)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.connect(f"tcp://localhost:{self.port}")
            self.enabled = True
            logger.info("Sim control publisher connected to port {}", self.port)
        except Exception as exc:
            logger.error("Failed to start sim control publisher: {}", exc)
            self._zmq = None
            self.enabled = False

    def publish(self, payload: dict) -> None:
        zmq = self._zmq
        if not self.enabled or self.socket is None or zmq is None:
            return
        try:
            self.socket.send_string(json.dumps(payload), zmq.NOBLOCK)
        except zmq.Again:
            pass
        except Exception as exc:
            logger.warning("Sim control publish failed: {}", exc)

    def request_reset(self, reason: str) -> None:
        payload = json.dumps({"action": "reset", "reason": str(reason)})
        zmq = self._zmq
        if not self.enabled or self.socket is None or zmq is None:
            return
        for _ in range(20):
            try:
                self.socket.send_string(payload, zmq.NOBLOCK)
                return
            except zmq.Again:
                time.sleep(0.01)
            except Exception as exc:
                logger.warning("Sim control reset publish failed: {}", exc)
                return
        logger.warning("Sim control reset publish dropped after retries")

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
        self.enabled = False
