"""Receive relayed depth frames and republish them into local shared memory.

Runs on the machine doing the policy compute (e.g. a laptop) when the depth
camera is attached to a different host. Subscribes to `depth_relay_pub.py` and
writes each frame into a local `depth_img_shm` segment with exactly the layout
`run_policy.py` expects, so the policy runs unmodified.

Start this BEFORE run_policy.py: the policy attaches to an existing segment and
raises if it is missing.

Usage (on the compute host):

    python scripts/depth_relay_sub.py --connect tcp://<camera-host>:5601
"""

from __future__ import annotations

import argparse
import signal
import struct
import sys
import time
from multiprocessing import resource_tracker, shared_memory

import numpy as np
import zmq

HEADER_FMT = "<4sQdIII"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
MAGIC = b"HSD1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--connect", required=True, help="ZMQ address of depth_relay_pub.py, e.g. tcp://10.0.0.5:5601")
    parser.add_argument("--shm-name", default="depth_img_shm", help="shared memory segment to write")
    parser.add_argument("--recv-timeout", type=float, default=5.0, help="seconds before warning about no frames")
    parser.add_argument("--stats-interval", type=float, default=5.0, help="seconds between stats lines (0 disables)")
    return parser.parse_args()


def open_shm(name: str, nbytes: int) -> tuple[shared_memory.SharedMemory, bool]:
    """Create the segment, or attach if something already made it."""
    try:
        shm = shared_memory.SharedMemory(create=True, size=nbytes, name=name)
        return shm, True
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=name)
        if shm.size < nbytes:
            shm.close()
            raise RuntimeError(
                f"existing shared memory '{name}' is {shm.size} bytes, need {nbytes}. "
                "Stop whatever created it (image_server.py?) and retry."
            ) from None
        # Someone else owns it; don't unlink it on our way out.
        resource_tracker.unregister(shm._name, "shared_memory")
        return shm, False


def main() -> int:
    args = parse_args()

    def _stop_on_sigterm(_signum, _frame):
        # Without this, a SIGTERM (which is how supervisors and `kill` stop us)
        # skips the cleanup below and leaves the segment behind, so the next run
        # attaches to a stale buffer instead of creating a fresh one.
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _stop_on_sigterm)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    # Always hand the policy the newest frame available, never a queued stale one.
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.RCVHWM, 1)
    sock.connect(args.connect)
    print(f"[depth-relay-sub] subscribed to {args.connect}")

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    shm: shared_memory.SharedMemory | None = None
    owns_shm = False
    frame: np.ndarray | None = None
    shape: tuple[int, int, int, int] | None = None

    received = 0
    dropped = 0
    last_seq: int | None = None
    last_stats = time.monotonic()
    last_frame_at = time.monotonic()
    warned_stalled = False

    try:
        while True:
            events = dict(poller.poll(timeout=500))
            now = time.monotonic()

            if sock not in events:
                if now - last_frame_at > args.recv_timeout and not warned_stalled:
                    print(
                        f"[depth-relay-sub] WARNING: no frames for {now - last_frame_at:.1f}s. "
                        "Is depth_relay_pub.py running on the camera host, and the port reachable?"
                    )
                    warned_stalled = True
                continue

            msg = sock.recv(copy=True)
            last_frame_at = now
            warned_stalled = False

            if len(msg) < HEADER_SIZE:
                print(f"[depth-relay-sub] short message ({len(msg)} bytes), ignoring")
                continue
            magic, seq, t_send, n_cam, height, width = struct.unpack(HEADER_FMT, msg[:HEADER_SIZE])
            if magic != MAGIC:
                print(f"[depth-relay-sub] bad magic {magic!r}, ignoring")
                continue

            expected = (n_cam, 1, height, width)
            payload = msg[HEADER_SIZE:]
            nbytes = int(np.prod(expected)) * np.dtype(np.float32).itemsize
            if len(payload) != nbytes:
                print(f"[depth-relay-sub] payload {len(payload)} bytes, expected {nbytes}; ignoring")
                continue

            if shm is None:
                shape = expected
                shm, owns_shm = open_shm(args.shm_name, nbytes)
                frame = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)
                print(
                    f"[depth-relay-sub] shared memory '{args.shm_name}' ready: shape={shape} "
                    f"({'created' if owns_shm else 'attached to existing'})"
                )
                print("[depth-relay-sub] you can start run_policy.py now")
            elif expected != shape:
                print(f"[depth-relay-sub] shape changed {shape} -> {expected}; ignoring frame")
                continue

            frame[:] = np.frombuffer(payload, dtype=np.float32).reshape(shape)
            received += 1

            if last_seq is not None and seq > last_seq + 1:
                # CONFLATE drops frames by design when we fall behind; count them
                # so a genuinely bad link is visible rather than silent.
                dropped += seq - last_seq - 1
            last_seq = seq

            if args.stats_interval > 0 and now - last_stats >= args.stats_interval:
                dt = now - last_stats
                finite = frame[np.isfinite(frame)]
                rng = f"{finite.min():.3f}..{finite.max():.3f}" if finite.size else "empty"
                # Clocks on the two hosts are not synchronised, so treat this as a
                # drift indicator, not an absolute latency measurement.
                skew_ms = (time.time() - t_send) * 1000.0
                print(
                    f"[depth-relay-sub] seq={seq} rate={received / dt:.1f} Hz dropped={dropped} "
                    f"range={rng} send->recv={skew_ms:+.1f}ms (unsynced clocks)"
                )
                received = 0
                dropped = 0
                last_stats = now
    except KeyboardInterrupt:
        print("\n[depth-relay-sub] stopping")
    finally:
        sock.close(linger=0)
        ctx.term()
        if shm is not None:
            shm.close()
            if owns_shm:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
