#!/usr/bin/env python3
"""Validate interval-only contact data against immutable motion truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REGIONS = ("left_wrist", "right_wrist")
ROLLOUT_REQUIRED_ARRAYS = (
    "valid_steps",
    "body_pos_local",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "object_pos_local",
    "object_quat_w",
    "object_lin_vel_w",
    "object_ang_vel_w",
)


def _clip_id(directory_name: str, motion_ids: set[str]) -> str:
    if directory_name in motion_ids:
        return directory_name
    prefix, separator, suffix = directory_name.partition("_")
    if separator and prefix.isdecimal() and suffix in motion_ids:
        return suffix
    return directory_name


def _load_expected_intervals(motion_path: Path) -> tuple[int, dict[str, list[int]]]:
    with np.load(motion_path, allow_pickle=False) as payload:
        if "hand_contact_valid" not in payload.files:
            raise ValueError(f"motion omits hand_contact_valid: {motion_path}")
        valid = np.asarray(payload["hand_contact_valid"], dtype=np.bool_)
    if valid.ndim != 2 or valid.shape[1] != 2 or valid.shape[0] <= 0:
        raise ValueError(f"invalid hand_contact_valid shape {valid.shape}: {motion_path}")
    intervals: dict[str, list[int]] = {}
    for index, region in enumerate(REGIONS):
        frames = np.flatnonzero(valid[:, index])
        if frames.size == 0:
            raise ValueError(f"{region} has no valid contact frames: {motion_path}")
        start = int(frames[0])
        end = int(frames[-1]) + 1
        if int(frames.size) != end - start:
            raise ValueError(f"{region} contact truth is non-contiguous: {motion_path}")
        intervals[region] = [start, end]
    return int(valid.shape[0]), intervals


def _validate_rollout_reference(path: Path, clip_id: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing teacher rollout reference: {path}")
    with np.load(path, allow_pickle=False) as payload:
        missing = sorted({"clip_id", "trajectory_length", *ROLLOUT_REQUIRED_ARRAYS}.difference(payload.files))
        if missing:
            raise ValueError(f"teacher rollout reference omits fields {missing}: {path}")
        stored_clip_id = str(np.asarray(payload["clip_id"]).item())
        if stored_clip_id != clip_id:
            raise ValueError(
                f"teacher rollout clip identity mismatch: expected={clip_id} actual={stored_clip_id}"
            )
        trajectory_length = int(np.asarray(payload["trajectory_length"]).item())
        if trajectory_length <= 0:
            raise ValueError(f"invalid teacher rollout trajectory_length={trajectory_length}: {path}")
        for name in ROLLOUT_REQUIRED_ARRAYS:
            value = np.asarray(payload[name])
            if value.ndim < 1 or value.shape[0] != trajectory_length:
                raise ValueError(
                    f"teacher rollout field {name} has shape {value.shape}, "
                    f"expected leading dimension {trajectory_length}: {path}"
                )
            if name == "valid_steps":
                if value.dtype != np.bool_ or not bool(value.any()):
                    raise ValueError(f"teacher rollout valid_steps is invalid: {path}")
            elif not bool(np.isfinite(value).all()):
                raise ValueError(f"teacher rollout field {name} contains non-finite values: {path}")


def validate(motion_dir: Path, contact_root: Path, expected_total: int | None) -> Path:
    motion_dir = motion_dir.expanduser().resolve()
    contact_root = contact_root.expanduser().resolve()
    clips_root = contact_root / "clips" if (contact_root / "clips").is_dir() else contact_root
    motion_paths = {path.stem: path for path in sorted(motion_dir.glob("*.npz"))}
    motion_ids = set(motion_paths)
    if not motion_ids:
        raise ValueError(f"no motion NPZ files under {motion_dir}")
    if expected_total is not None and len(motion_ids) != expected_total:
        raise ValueError(f"expected {expected_total} motions, found {len(motion_ids)}")
    if not clips_root.is_dir():
        raise ValueError(f"contact root does not exist: {clips_root}")

    directories: dict[str, Path] = {}
    for directory in sorted(path for path in clips_root.iterdir() if path.is_dir()):
        clip_id = _clip_id(directory.name, motion_ids)
        if clip_id not in motion_ids:
            raise ValueError(f"contact directory does not resolve to an active motion: {directory}")
        if clip_id in directories:
            raise ValueError(f"duplicate contact directory for {clip_id}")
        directories[clip_id] = directory
    if set(directories) != motion_ids:
        missing = sorted(motion_ids.difference(directories))
        raise ValueError(f"contact intervals missing {len(missing)} motions: {missing[:20]}")

    for clip_id in sorted(motion_ids):
        frame_count, expected = _load_expected_intervals(motion_paths[clip_id])
        interval_path = directories[clip_id] / "contact_intervals.json"
        raw = json.loads(interval_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"contact_intervals.json must contain an object: {interval_path}")
        for region in REGIONS:
            value = raw.get(region)
            if (
                not isinstance(value, list)
                or len(value) != 2
                or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
            ):
                raise ValueError(f"invalid {region} interval in {interval_path}: {value!r}")
            if value != expected[region]:
                raise ValueError(
                    f"{region} interval disagrees with hand_contact_valid in {clip_id}: "
                    f"actual={value} expected={expected[region]}"
                )
            if not 0 <= value[0] < value[1] <= frame_count:
                raise ValueError(f"out-of-range {region} interval in {interval_path}: {value}")
        _validate_rollout_reference(
            directories[clip_id] / "teacher_rollout_reference.npz",
            clip_id,
        )
    print(
        f"[INFO] runtime_contact_intervals_verified clips={len(motion_ids)} "
        "regions=left_wrist,right_wrist semantics=motion_frame_half_open_v1 "
        "teacher_rollout_references=required",
        file=sys.stderr,
    )
    return clips_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-dir", required=True, type=Path)
    parser.add_argument("--contact-root", required=True, type=Path)
    parser.add_argument("--expected-total", type=int)
    args = parser.parse_args()
    try:
        clips_root = validate(args.motion_dir, args.contact_root, args.expected_total)
    except Exception as exc:
        print(f"[ERROR] Runtime contact interval validation failed: {exc}", file=sys.stderr)
        return 2
    print(clips_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
