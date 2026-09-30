"""Receive IR stereo pairs over the network and present them as a camera.

`RemoteStereoCamerasWrapper` has the same interface as `RealSenseCamerasWrapper`
(`cameras`, `num_cameras`, `get_frames()`), so `image_server.py` can run a
learned stereo depth predictor on a host that does not have the camera plugged
in. The camera host runs `scripts/stereo_relay_pub.py`.

Wire format (one ZMQ part per frame, so subscribers can use ZMQ_CONFLATE):

    header  <4sQdIIII   magic b"HSS1", seq, send time, H, W, n_cam, has_depth
    calib   50 x f32    intrinsics (2,3,3) then extrinsics (2,4,4)
    left    H*W  u8     left IR
    right   H*W  u8     right IR
    depth   H*W  u16    optional hardware depth in mm (only if has_depth)

The relay carries the raw pair because a learned stereo model needs it; the
hardware depth is optional and off by default since it doubles the bandwidth
and the point of this path is not to use it.
"""

from __future__ import annotations

import os
import struct
import threading
import time
from dataclasses import dataclass

import numpy as np
import zmq

HEADER_FMT = "<4sQdIIII"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
MAGIC = b"HSS1"
CALIB_FLOATS = 18 + 32
CALIB_BYTES = CALIB_FLOATS * 4


@dataclass(frozen=True)
class RemoteStereoCameraConfig:
    """Configuration for one relayed stereo camera."""

    connect: str = os.environ.get("HOLOSOMA_REMOTE_STEREO_CONNECT", "tcp://127.0.0.1:5602")
    """ZMQ address of stereo_relay_pub.py on the camera host (env HOLOSOMA_REMOTE_STEREO_CONNECT)."""

    img_shape: tuple[int, int] = (480, 848)
    """Expected (height, width) per eye; frames of another size are rejected."""

    stale_after_s: float = 1.0
    """Warn once when no frame has arrived for this long."""

    recv_timeout_s: float = 45.0
    """How long get_frames() may block waiting for the first frame. Long enough
    to survive one relay-side camera pipeline restart (3 x 5 s misses, stop,
    reopen), which is how the D435i recovers from a 'started but no frames' stall."""


@dataclass(frozen=True)
class RemoteStereoCamerasConfig:
    terms: dict[str, RemoteStereoCameraConfig] | None = None

    def __post_init__(self):
        if self.terms is None:
            object.__setattr__(self, "terms", {"d435i_depth": RemoteStereoCameraConfig()})


class RemoteStereoCamera:
    """One relayed stereo camera; a background thread keeps the newest frame."""

    def __init__(self, config: RemoteStereoCameraConfig):
        self.config = config
        self.calibration: dict[str, np.ndarray] | None = None
        # Attributes the image server's deployment audit reads off every camera.
        self.depth_scale = None
        self.last_capture_metadata: dict = {}

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        # Always consume the newest frame; a queued stale one is worse than a drop.
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.setsockopt(zmq.RCVHWM, 1)
        self._sock.connect(config.connect)

        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._latest_at = 0.0
        self._seq = -1
        self._dropped = 0
        self._recv_count = 0
        self._stats_at = time.monotonic()
        self._stats_dropped = 0
        self._warned_stale = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._recv_loop, name="remote-stereo-recv", daemon=True)
        self._thread.start()
        print(f"[RemoteStereo] subscribed to {config.connect}, expecting {config.img_shape[1]}x{config.img_shape[0]}")

    def _recv_loop(self) -> None:
        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        h_exp, w_exp = self.config.img_shape
        while not self._stop.is_set():
            if not dict(poller.poll(timeout=200)):
                if self._latest is not None and time.monotonic() - self._latest_at > self.config.stale_after_s:
                    if not self._warned_stale:
                        print("[RemoteStereo] WARNING: no frames arriving; is stereo_relay_pub.py running on the camera host?")
                        self._warned_stale = True
                continue
            msg = self._sock.recv(copy=True)
            if len(msg) < HEADER_SIZE + CALIB_BYTES:
                continue
            magic, seq, t_send, h, w, n_cam, has_depth = struct.unpack(HEADER_FMT, msg[:HEADER_SIZE])
            if magic != MAGIC or (h, w) != (h_exp, w_exp) or n_cam != 1 or has_depth not in (0, 1):
                continue
            if len(msg) != HEADER_SIZE + CALIB_BYTES + h * w * (2 + 2 * has_depth):
                continue
            off = HEADER_SIZE
            calib = np.frombuffer(msg[off : off + CALIB_BYTES], dtype=np.float32)
            off += CALIB_BYTES
            n = h * w
            left = np.frombuffer(msg[off : off + n], dtype=np.uint8).reshape(h, w)
            off += n
            right = np.frombuffer(msg[off : off + n], dtype=np.uint8).reshape(h, w)
            off += n
            depth = None
            if has_depth:
                depth = np.frombuffer(msg[off : off + 2 * n], dtype=np.uint16).reshape(h, w).astype(np.float32) * 1e-3

            if self.calibration is None:
                self.calibration = {
                    "intrinsics": calib[:18].reshape(2, 3, 3).copy(),
                    "extrinsics": calib[18:].reshape(2, 4, 4).copy(),
                }
                k = self.calibration["intrinsics"][0]
                print(f"[RemoteStereo] calibration received: fx={k[0,0]:.2f} baseline={abs(self.calibration['extrinsics'][1][0,3]):.6f} m")

            if self._seq >= 0 and seq > self._seq + 1:
                self._dropped += seq - self._seq - 1
            self._seq = seq
            self._recv_count += 1

            # Periodic receiver health line: makes receive-side starvation (the
            # image server not draining frames because something else holds the
            # GIL) visible without needing the deployment audit.
            now_m = time.monotonic()
            if now_m - self._stats_at >= 5.0:
                dt = now_m - self._stats_at
                print(
                    f"[RemoteStereo] recv {self._recv_count / dt:.1f} Hz  seq={seq}  "
                    f"gaps(dropped frames) +{self._dropped - self._stats_dropped}  total {self._dropped}"
                )
                self._stats_at = now_m
                self._recv_count = 0
                self._stats_dropped = self._dropped

            with self._lock:
                self._latest = {"left": left, "right": right, "depth": depth, "t_send": t_send, "seq": seq}
                self._latest_at = time.monotonic()
            self._warned_stale = False

    def capture(self) -> dict:
        """Newest frame as `RealSenseCamera.capture()` would return it.

        Returns ``{"depth": (H, W) float32 metres or zeros, "rgb": (H, 2W, 3) uint8
        side-by-side IR, "total_latency_ms": float | None}``.
        """
        deadline = time.monotonic() + self.config.recv_timeout_s
        while True:
            with self._lock:
                frame = self._latest
            if frame is not None:
                break
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"no stereo frames received from {self.config.connect} within "
                    f"{self.config.recv_timeout_s:.0f}s. Start stereo_relay_pub.py on the camera host first."
                )
            time.sleep(0.005)

        left, right = frame["left"], frame["right"]
        self.last_capture_metadata = {
            "relay_seq": int(frame["seq"]),
            "relay_send_time": float(frame["t_send"]),
            "received_at_monotonic": self._latest_at,
            "dropped_total": int(self._dropped),
        }
        # Match the RealSense IR-stereo path: grey replicated to 3 channels, side by side.
        rgb = np.concatenate([np.repeat(left[:, :, None], 3, axis=2), np.repeat(right[:, :, None], 3, axis=2)], axis=1)
        depth = frame["depth"]
        if depth is None:
            depth = np.zeros(left.shape, dtype=np.float32)
        # Clocks on the two hosts are not synchronised; report None rather than a
        # number that is mostly clock offset.
        return {"depth": depth, "rgb": rgb, "total_latency_ms": None}

    def release(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._sock.close(linger=0)


class RemoteStereoCamerasWrapper:
    """Same interface as RealSenseCamerasWrapper / ZedCamerasWrapper."""

    def __init__(self, config: RemoteStereoCamerasConfig):
        self.cameras = {name: RemoteStereoCamera(cfg) for name, cfg in config.terms.items()}
        self.num_cameras = len(self.cameras)

    def get_frames(self) -> dict:
        depth_data: dict[str, np.ndarray] = {}
        rgb_data: dict[str, np.ndarray] = {}
        calibration_data: dict[str, dict[str, np.ndarray]] = {}
        for name, camera in self.cameras.items():
            frame = camera.capture()
            depth_data[name] = frame["depth"]
            rgb_data[name] = frame["rgb"]
            if camera.calibration is None:
                raise RuntimeError(f"[RemoteStereo] frames arrived for '{name}' but calibration is missing")
            calibration_data[name] = camera.calibration
        return {"depth": depth_data, "rgb": rgb_data, "calibration": calibration_data}
