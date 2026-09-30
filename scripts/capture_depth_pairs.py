"""Record D435i hardware depth alongside the raw IR stereo pair.

Runs on the machine the camera is attached to. Each frame captures what the two
depth sources need in order to be compared later: the RealSense's own stereo
depth, and the left/right infrared images that Fast-FoundationStereo consumes,
plus the stereo calibration needed to turn its disparity into metres.

Pair this with `compare_depth_sources.py`, which runs FFS over the recording and
reports how far the two depth sources diverge in the 58x87 space the policy
actually sees.

Usage (on the camera host):

    python scripts/capture_depth_pairs.py --frames 200 --out depth_pairs.npz
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=200, help="number of frames to record")
    parser.add_argument("--out", default="depth_pairs.npz", help="output .npz path")
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30, help="frames to discard while auto-exposure settles")
    parser.add_argument(
        "--emitter",
        choices=["on", "off"],
        default="on",
        help="IR projector. 'on' matches how the D435i produces its own depth; "
        "'off' removes the speckle pattern, which a stereo network may prefer.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("pyrealsense2 not installed in this interpreter", file=sys.stderr)
        return 1

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.infrared, 1, args.width, args.height, rs.format.y8, args.fps)
    config.enable_stream(rs.stream.infrared, 2, args.width, args.height, rs.format.y8, args.fps)

    profile = pipeline.start(config)

    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if args.emitter == "on" else 0.0)
        print(f"[capture] IR emitter: {args.emitter}")
    depth_scale = depth_sensor.get_depth_scale()
    print(f"[capture] depth scale: {depth_scale} m/unit")

    # Same calibration convention as holosoma.sensors.realsense: left eye is the
    # reference frame and the baseline lives in the right eye's translation.
    left_p = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    right_p = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()

    def build_k(intr):
        return np.array(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    li, ri = left_p.get_intrinsics(), right_p.get_intrinsics()
    intrinsics = np.stack([build_k(li), build_k(ri)], axis=0)

    rs_extr = right_p.get_extrinsics_to(left_p)
    t = np.array(rs_extr.translation, dtype=np.float32)
    t[0] *= -1.0
    left_ext = np.eye(4, dtype=np.float32)
    right_ext = np.eye(4, dtype=np.float32)
    right_ext[:3, :3] = np.array(rs_extr.rotation, dtype=np.float32).reshape(3, 3)
    right_ext[:3, 3] = t
    extrinsics = np.stack([left_ext, right_ext], axis=0)
    print(f"[capture] fx={li.fx:.2f}  baseline={abs(t[0]):.6f} m")

    depths, lefts, rights = [], [], []
    try:
        # The first frames after start can take a while with three streams up;
        # give them longer, but never block indefinitely - a hung wait leaves the
        # process holding the camera so nothing else can open it.
        try:
            for i in range(args.warmup):
                pipeline.wait_for_frames(10000 if i == 0 else 5000)
        except RuntimeError as exc:
            print(f"[capture] no frames during warmup: {exc}", file=sys.stderr)
            print(
                "[capture] usually the camera is stalled from a previous session; "
                "unplug/replug it, or try --fps 15 or a lower resolution.",
                file=sys.stderr,
            )
            return 1
        print(f"[capture] warmup done, recording {args.frames} frames...")

        t0 = time.time()
        for i in range(args.frames):
            frames = pipeline.wait_for_frames(5000)
            d = frames.get_depth_frame()
            l = frames.get_infrared_frame(1)
            r = frames.get_infrared_frame(2)
            if not d or not l or not r:
                continue
            # Store metric depth so the comparison needs no scale bookkeeping.
            depths.append(np.asanyarray(d.get_data()).astype(np.float32) * depth_scale)
            lefts.append(np.asanyarray(l.get_data()).copy())
            rights.append(np.asanyarray(r.get_data()).copy())
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{args.frames}  ({(i + 1) / (time.time() - t0):.1f} Hz)")
    except KeyboardInterrupt:
        print("\n[capture] interrupted, saving what we have")
    finally:
        pipeline.stop()

    if not depths:
        print("[capture] no frames captured", file=sys.stderr)
        return 1

    np.savez_compressed(
        args.out,
        depth=np.stack(depths),
        ir_left=np.stack(lefts),
        ir_right=np.stack(rights),
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        emitter=args.emitter,
    )
    print(f"[capture] wrote {args.out}: {len(depths)} frames, {args.width}x{args.height}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
