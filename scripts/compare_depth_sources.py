"""Quantify how far Fast-FoundationStereo depth diverges from the D435i's own.

Takes a recording from `capture_depth_pairs.py`, runs FFS over the stored IR
stereo pairs, and pushes both depth sources through the exact post-processing
`image_server.py` applies, so the comparison happens in the 58x87 normalised
space the policy consumes rather than in raw metres.

The point is to decide whether swapping the depth source needs the policy
retrained. A policy trained on D435i depth sees whatever this script reports as
a distribution shift.

Usage (on a CUDA host):

    HOLOSOMA_FFS_REPO=... HOLOSOMA_FFS_MODEL=... \
    python scripts/compare_depth_sources.py depth_pairs.npz
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", help=".npz from capture_depth_pairs.py")
    parser.add_argument("--limit", type=int, default=0, help="only process the first N frames (0 = all)")
    parser.add_argument("--ffs-height", type=int, default=288, help="FFS inference height")
    parser.add_argument("--ffs-width", type=int, default=512, help="FFS inference width (per eye)")
    parser.add_argument("--valid-iters", type=int, default=8)
    # Defaults mirror config_values.image_server.real_d435i.
    parser.add_argument("--near-clip", type=float, default=0.3)
    parser.add_argument("--far-clip", type=float, default=3.0)
    parser.add_argument("--min-valid-depth", type=float, default=None)
    parser.add_argument("--crop-y-start", type=int, default=16)
    parser.add_argument("--crop-x-start", type=int, default=32)
    parser.add_argument("--crop-x-end", type=int, default=-32)
    parser.add_argument("--resized-height", type=int, default=58)
    parser.add_argument("--resized-width", type=int, default=87)
    parser.add_argument("--save-npz", default="", help="optional path to dump the processed pairs")
    return parser.parse_args()


def policy_space(depth_m: np.ndarray, a: argparse.Namespace) -> np.ndarray:
    """Replicate image_server._resize_clip_expand_transpose exactly."""
    f = depth_m[a.crop_y_start :, a.crop_x_start : a.crop_x_end]
    f = cv2.resize(f, (a.resized_width, a.resized_height), cv2.INTER_CUBIC)
    f = np.clip(f, a.near_clip, a.far_clip)
    min_valid = a.near_clip if a.min_valid_depth is None else a.min_valid_depth
    f = f.copy()
    f[f < min_valid] = a.far_clip
    return (f - a.near_clip) / (a.far_clip - a.near_clip) - 0.5


def pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q))


def main() -> int:
    a = parse_args()
    data = np.load(a.recording, allow_pickle=True)
    depth = data["depth"]
    ir_l, ir_r = data["ir_left"], data["ir_right"]
    intr, extr = data["intrinsics"], data["extrinsics"]
    n = len(depth) if a.limit <= 0 else min(a.limit, len(depth))
    H, W = depth.shape[1:3]
    print(f"recording: {len(depth)} frames at {W}x{H}, emitter={data['emitter']}, processing {n}")

    sys.path.insert(0, "src/holosoma")
    from holosoma.models.ffs.infer import FFSConfig, FastFoundationStereo

    ffs = FastFoundationStereo(
        cfg=dataclasses.replace(FFSConfig(), device="cuda", valid_iters=a.valid_iters, max_disp=192)
    )

    # FFS runs at reduced resolution for speed; its output is resampled back to
    # the native frame before cropping, because the crop constants are expressed
    # in native pixels and would otherwise change the field of view.
    scale_x = a.ffs_width / W
    hw_all, ffs_all = [], []
    for i in range(n):
        left = cv2.resize(ir_l[i], (a.ffs_width, a.ffs_height), interpolation=cv2.INTER_AREA)
        right = cv2.resize(ir_r[i], (a.ffs_width, a.ffs_height), interpolation=cv2.INTER_AREA)
        sbs = np.concatenate([left, right], axis=1)
        sbs = np.repeat(sbs[:, :, None], 3, axis=2)  # FFS expects 3 channels

        k = intr.copy().astype(np.float64)
        k[:, 0, :] *= scale_x
        k[:, 1, :] *= a.ffs_height / H

        d = ffs.predict(sbs, k, extr)
        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)

        hw_all.append(policy_space(depth[i], a))
        ffs_all.append(policy_space(d, a))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n}")

    hw = np.stack(hw_all)
    fs = np.stack(ffs_all)
    far_norm = 0.5  # what far_clip maps to after normalisation

    diff = fs - hw
    absdiff = np.abs(diff)
    span = a.far_clip - a.near_clip

    print("\n" + "=" * 68)
    print("POLICY-SPACE COMPARISON  (58x87, normalised to [-0.5, +0.5])")
    print("=" * 68)
    print(f"{'':22}{'D435i hardware':>20}{'Fast-FoundationStereo':>24}")
    for label, fn in (
        ("mean", np.mean),
        ("std", np.std),
        ("p01", lambda x: pct(x, 1)),
        ("p50", lambda x: pct(x, 50)),
        ("p99", lambda x: pct(x, 99)),
    ):
        print(f"  {label:20}{fn(hw):>20.4f}{fn(fs):>24.4f}")

    hw_far = float(np.mean(np.isclose(hw, far_norm)))
    fs_far = float(np.mean(np.isclose(fs, far_norm)))
    print(f"  {'at far plane':20}{hw_far * 100:>19.2f}%{fs_far * 100:>23.2f}%")
    print("     (D435i holes land here; a stereo network fills them instead)")

    print("\n  disagreement between the two")
    print(f"    MAE  {absdiff.mean():.4f} normalised  =  {absdiff.mean() * span:.4f} m")
    print(f"    RMSE {np.sqrt((diff ** 2).mean()):.4f} normalised  =  {np.sqrt((diff ** 2).mean()) * span:.4f} m")
    print(f"    bias {diff.mean():+.4f} normalised  =  {diff.mean() * span:+.4f} m  (FFS minus D435i)")
    print(f"    p50  {pct(absdiff, 50):.4f}    p95 {pct(absdiff, 95):.4f}    p99 {pct(absdiff, 99):.4f}")

    both_near = ~np.isclose(hw, far_norm) & ~np.isclose(fs, far_norm)
    if both_near.any():
        d2 = np.abs(diff[both_near])
        print(f"\n  excluding far-plane pixels ({both_near.mean() * 100:.1f}% of pixels)")
        print(f"    MAE  {d2.mean():.4f} normalised  =  {d2.mean() * span:.4f} m")

    corr = float(np.corrcoef(hw.ravel(), fs.ravel())[0, 1])
    print(f"\n  pearson correlation: {corr:.4f}")

    tv_hw = float(np.mean(np.abs(np.diff(hw, axis=-1))))
    tv_fs = float(np.mean(np.abs(np.diff(fs, axis=-1))))
    print(f"  horizontal roughness: D435i {tv_hw:.4f}   FFS {tv_fs:.4f}   (lower = smoother)")

    tj_hw = float(np.mean(np.abs(np.diff(hw, axis=0))))
    tj_fs = float(np.mean(np.abs(np.diff(fs, axis=0))))
    print(f"  frame-to-frame jitter: D435i {tj_hw:.4f}   FFS {tj_fs:.4f}")

    print("\n" + "=" * 68)
    print("A policy trained on D435i depth sees the MAE above as distribution")
    print("shift at every timestep. Compare it against the observation noise the")
    print("policy was trained to tolerate before deciding whether to retrain.")
    print("=" * 68)

    if a.save_npz:
        np.savez_compressed(a.save_npz, hardware=hw, ffs=fs)
        print(f"\nwrote {a.save_npz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
