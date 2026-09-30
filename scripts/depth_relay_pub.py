"""Publish the image server's depth shared memory over the network.

Runs on the machine the depth camera is physically attached to (e.g. the robot's
onboard computer), alongside `image_server.py`. Reads the `depth_img_shm` segment
that the image server writes and republishes each frame over ZMQ PUB so that
`run_policy.py` can consume it from another host via `depth_relay_sub.py`.

Frames are sent as a single ZMQ part (fixed-size header + raw float32 payload) so
that subscribers can use ZMQ_CONFLATE and always read the newest frame.

Usage (on the camera host, after image_server.py is already running):

    python scripts/depth_relay_pub.py --bind tcp://*:5601
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from multiprocessing import resource_tracker, shared_memory

import numpy as np
import zmq

# magic, seq, send time (epoch s), n_cam, height, width
HEADER_FMT = "<4sQdIII"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
MAGIC = b"HSD1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bind", default="tcp://*:5601", help="ZMQ PUB bind address")
    parser.add_argument("--shm-name", default="depth_img_shm", help="shared memory segment to read")
    parser.add_argument("--num-cameras", type=int, default=1)
    parser.add_argument("--height", type=int, default=58, help="resized_height used by the image server")
    parser.add_argument("--width", type=int, default=87, help="resized_width used by the image server")
    parser.add_argument("--rate", type=float, default=30.0, help="publish rate in Hz")
    parser.add_argument("--wait-timeout", type=float, default=30.0, help="seconds to wait for the shm to appear")
    parser.add_argument("--stats-interval", type=float, default=5.0, help="seconds between stats lines (0 disables)")
    return parser.parse_args()


def attach_shm(name: str, timeout: float) -> shared_memory.SharedMemory:
    """Attach to the image server's segment, waiting for it to show up."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            shm = shared_memory.SharedMemory(name=name)
            # The image server owns this segment; don't let our resource tracker
            # unlink it when we exit.
            resource_tracker.unregister(shm._name, "shared_memory")
            return shm
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"shared memory '{name}' not found after {timeout:.0f}s. "
                    "Start image_server.py (e.g. via real_depth.sh) on this host first."
                ) from None
            time.sleep(0.2)


def main() -> int:
    args = parse_args()
    shape = (args.num_cameras, 1, args.height, args.width)
    nbytes = int(np.prod(shape)) * np.dtype(np.float32).itemsize

    shm = attach_shm(args.shm_name, args.wait_timeout)
    if shm.size < nbytes:
        raise RuntimeError(f"shared memory '{args.shm_name}' is {shm.size} bytes, expected at least {nbytes}")
    frame = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)
    print(f"[depth-relay-pub] attached to '{args.shm_name}': shape={shape}, {nbytes} bytes/frame")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    # Never queue stale depth: a late frame is worse than a dropped one.
    sock.setsockopt(zmq.SNDHWM, 1)
    sock.bind(args.bind)
    print(f"[depth-relay-pub] publishing on {args.bind} at {args.rate:.0f} Hz")

    period = 1.0 / args.rate if args.rate > 0 else 0.0
    seq = 0
    sent = 0
    nonfinite_frames = 0
    next_tick = time.monotonic()
    last_stats = time.monotonic()

    try:
        while True:
            payload = frame.tobytes()
            header = struct.pack(HEADER_FMT, MAGIC, seq, time.time(), args.num_cameras, args.height, args.width)
            sock.send(header + payload, copy=False)
            seq += 1
            sent += 1

            # Cheap sanity check; a constant or all-NaN buffer means the camera
            # side is not actually producing depth.
            if seq % 30 == 0 and not np.isfinite(frame).all():
                nonfinite_frames += 1

            now = time.monotonic()
            if args.stats_interval > 0 and now - last_stats >= args.stats_interval:
                dt = now - last_stats
                finite = frame[np.isfinite(frame)]
                rng = f"{finite.min():.3f}..{finite.max():.3f}" if finite.size else "empty"
                print(
                    f"[depth-relay-pub] seq={seq} rate={sent / dt:.1f} Hz range={rng} "
                    f"nonfinite_checks={nonfinite_frames}"
                )
                sent = 0
                last_stats = now

            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # Fell behind; resync rather than accumulate drift.
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        print("\n[depth-relay-pub] stopping")
    finally:
        sock.close(linger=0)
        ctx.term()
        shm.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
