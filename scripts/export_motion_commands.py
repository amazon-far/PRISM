#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np



RUNTIME_PICKUP_LIFT_HEIGHT_THRESHOLD = 0.10
CLIP_PICKUP_LIFT_RATIO_THRESHOLD = 0.35
RUNTIME_PICKUP_CONSECUTIVE_STEPS = 5
ADAPTIVE_SAMPLING_CONTACT_STAGE_RELEASE_LEAD_STEPS = 30


def wrap_to_pi(value: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(value), np.cos(value))


def yaw_from_quat(quat: np.ndarray, *, order: str) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    if order == "wxyz":
        qw, qx, qy, qz = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    elif order == "xyzw":
        qx, qy, qz, qw = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    else:
        raise ValueError(f"Unsupported quaternion order: {order}")
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def rotate_world_xy_to_heading_frame(delta_xy_w: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    delta_xy_w = np.asarray(delta_xy_w, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    out = np.empty_like(delta_xy_w, dtype=np.float64)
    out[..., 0] = cy * delta_xy_w[..., 0] + sy * delta_xy_w[..., 1]
    out[..., 1] = -sy * delta_xy_w[..., 0] + cy * delta_xy_w[..., 1]
    return out


def first_sustained_true_index(mask: np.ndarray, consecutive_steps: int) -> int | None:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.size == 0:
        return None
    if consecutive_steps <= 1:
        idx = np.flatnonzero(mask)
        return None if idx.size == 0 else int(idx[0])

    run_length = 0
    for idx, flag in enumerate(mask.tolist()):
        run_length = run_length + 1 if flag else 0
        if run_length >= consecutive_steps:
            return idx - consecutive_steps + 1
    return None


def first_sustained_true_index_from(mask: np.ndarray, consecutive_steps: int, start_idx: int) -> int | None:
    if start_idx <= 0:
        return first_sustained_true_index(mask, consecutive_steps)
    if start_idx >= int(np.asarray(mask).size):
        return None
    rel = first_sustained_true_index(np.asarray(mask)[start_idx:], consecutive_steps)
    return None if rel is None else int(start_idx + rel)


def pickup_threshold_from_rel_z(rel_z: np.ndarray) -> float:
    rel_z = np.asarray(rel_z, dtype=np.float64).reshape(-1)
    if rel_z.size == 0:
        return float(RUNTIME_PICKUP_LIFT_HEIGHT_THRESHOLD)
    z_min = float(np.min(rel_z))
    z_range = max(float(np.max(rel_z) - z_min), 0.0)
    return z_min + max(float(RUNTIME_PICKUP_LIFT_HEIGHT_THRESHOLD), z_range * float(CLIP_PICKUP_LIFT_RATIO_THRESHOLD))


def pickup_step_and_threshold_from_rel_z(rel_z: np.ndarray) -> tuple[int, float]:
    rel_z = np.asarray(rel_z, dtype=np.float64).reshape(-1)
    threshold = pickup_threshold_from_rel_z(rel_z)
    if rel_z.size == 0:
        return 0, threshold
    lifted = rel_z >= threshold
    pickup_step = first_sustained_true_index(lifted, RUNTIME_PICKUP_CONSECUTIVE_STEPS)
    if pickup_step is None:
        lifted_idx = np.flatnonzero(lifted)
        pickup_step = int(lifted_idx[0]) if lifted_idx.size else int(np.argmax(rel_z))
    return int(pickup_step), float(threshold)


def contact_aware_carry_window_from_rel_z(
    rel_z: np.ndarray,
    *,
    contact_interval: tuple[int, int] | None = None,
) -> tuple[int, int, int, float]:
    rel_z = np.asarray(rel_z, dtype=np.float64).reshape(-1)
    if rel_z.size == 0:
        return 0, 0, 0, float(RUNTIME_PICKUP_LIFT_HEIGHT_THRESHOLD)

    pickup_step, pickup_threshold = pickup_step_and_threshold_from_rel_z(rel_z)
    total_steps = int(rel_z.shape[0])
    carry_start = max(0, min(int(pickup_step), total_steps))
    carry_end = total_steps

    lowered = rel_z < pickup_threshold
    lowering_step = first_sustained_true_index_from(
        lowered,
        RUNTIME_PICKUP_CONSECUTIVE_STEPS,
        min(carry_start + 1, total_steps),
    )
    if lowering_step is not None:
        carry_end = min(carry_end, int(lowering_step))

    if contact_interval is not None:
        t1, t2 = sorted((int(contact_interval[0]), int(contact_interval[1])))
        if t2 > t1:
            release_start = max(
                0,
                min(t2 - max(int(ADAPTIVE_SAMPLING_CONTACT_STAGE_RELEASE_LEAD_STEPS), 0), total_steps),
            )
            carry_end = min(carry_end, release_start)

    carry_end = max(carry_start, min(carry_end, total_steps))
    return int(carry_start), int(carry_end), int(pickup_step), float(pickup_threshold)


def scalar_string(value: Any) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0].item())
    return str(value)


def load_motion_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key].copy() for key in data.keys()}


def load_teacher_reference_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key].copy() for key in data.keys()}


def command_from_motion_delta(data: dict[str, Any], *, lookahead_steps: int) -> dict[str, Any]:
    root_pos = np.asarray(data["body_pos_w"], dtype=np.float64)[:, 0, :]
    root_quat = np.asarray(data["body_quat_w"], dtype=np.float64)[:, 0, :]
    object_pos = np.asarray(data.get("object_pos_w", np.zeros_like(root_pos)), dtype=np.float64)
    fps = float(np.asarray(data.get("fps", [30.0])).reshape(-1)[0])
    n = int(root_pos.shape[0])
    step = max(1, int(lookahead_steps))
    target_idx = np.minimum(np.arange(n) + step, n - 1)

    root_yaw = yaw_from_quat(root_quat, order="wxyz")
    target_yaw = root_yaw[target_idx]
    delta_xy_w = root_pos[target_idx, :2] - root_pos[:, :2]
    rel_xy = rotate_world_xy_to_heading_frame(delta_xy_w, root_yaw)
    rel_yaw = wrap_to_pi(target_yaw - root_yaw)
    command = np.concatenate([rel_xy, np.asarray(rel_yaw).reshape(-1, 1)], axis=1)

    dt = np.maximum((target_idx - np.arange(n)).astype(np.float64), 1.0) / max(fps, 1.0e-6)
    velocity_command = command.copy()
    velocity_command[:, 0:2] /= dt[:, None]
    velocity_command[:, 2] /= dt

    rel_z = object_pos[:, 2] - root_pos[:, 2]
    carry_start, carry_end, pickup_step, pickup_threshold = contact_aware_carry_window_from_rel_z(rel_z)
    active = np.zeros((n,), dtype=bool)
    active[carry_start:carry_end] = True
    return {
        "mode": "motion_delta",
        "quat_order": "wxyz",
        "root_pos": root_pos,
        "target_root_pos": root_pos[target_idx],
        "root_yaw": root_yaw,
        "target_yaw": target_yaw,
        "object_pos": object_pos,
        "command": command,
        "velocity_command": velocity_command,
        "rel_z": rel_z,
        "active": active,
        "carry_start": carry_start,
        "carry_end": carry_end,
        "pickup_step": pickup_step,
        "pickup_threshold": pickup_threshold,
        "fps": fps,
    }


def command_from_teacher_reference(data: dict[str, Any]) -> dict[str, Any]:
    root_pos = np.asarray(data["root_pos_local"], dtype=np.float64)
    target_root_pos = np.asarray(data["target_root_pos_local"], dtype=np.float64)
    root_quat = np.asarray(data["root_quat_w"], dtype=np.float64)
    target_root_quat = np.asarray(data["target_root_quat_w"], dtype=np.float64)
    object_pos = np.asarray(data.get("target_object_pos_local", data.get("object_pos_local")), dtype=np.float64)
    valid = np.asarray(data.get("valid_steps", np.ones((root_pos.shape[0],), dtype=bool)), dtype=bool).reshape(-1)
    n = int(root_pos.shape[0])
    fps = float(np.asarray(data.get("fps", [30.0])).reshape(-1)[0]) if "fps" in data else 30.0

    root_yaw = yaw_from_quat(root_quat, order="xyzw")
    target_yaw = yaw_from_quat(target_root_quat, order="xyzw")
    delta_xy_w = target_root_pos[:, :2] - root_pos[:, :2]
    rel_xy = rotate_world_xy_to_heading_frame(delta_xy_w, root_yaw)
    rel_yaw = wrap_to_pi(target_yaw - root_yaw)
    command = np.concatenate([rel_xy, np.asarray(rel_yaw).reshape(-1, 1)], axis=1)
    velocity_command = np.full_like(command, np.nan)

    rel_z = object_pos[:, 2] - target_root_pos[:, 2]
    carry_start, carry_end, pickup_step, pickup_threshold = contact_aware_carry_window_from_rel_z(rel_z)
    active = np.zeros((n,), dtype=bool)
    active[carry_start:carry_end] = True
    active &= valid
    return {
        "mode": "teacher_reference",
        "quat_order": "xyzw",
        "root_pos": root_pos,
        "target_root_pos": target_root_pos,
        "root_yaw": root_yaw,
        "target_yaw": target_yaw,
        "object_pos": object_pos,
        "command": command,
        "velocity_command": velocity_command,
        "rel_z": rel_z,
        "active": active,
        "valid": valid,
        "carry_start": carry_start,
        "carry_end": carry_end,
        "pickup_step": pickup_step,
        "pickup_threshold": pickup_threshold,
        "fps": fps,
    }


def finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def stats(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": finite_or_none(np.mean(values)),
        "std": finite_or_none(np.std(values)),
        "min": finite_or_none(np.min(values)),
        "max": finite_or_none(np.max(values)),
    }


def summarize_clip(clip_id: str, source_path: Path, result: dict[str, Any]) -> dict[str, Any]:
    command = np.asarray(result["command"], dtype=np.float64)
    root_pos = np.asarray(result["root_pos"], dtype=np.float64)
    object_pos = np.asarray(result["object_pos"], dtype=np.float64)
    root_yaw = np.asarray(result["root_yaw"], dtype=np.float64)
    active = np.asarray(result["active"], dtype=bool)
    active_idx = np.flatnonzero(active)
    active_cmd = command[active] if active_idx.size else command[:0]
    active_root = root_pos[active] if active_idx.size else root_pos[:0]
    active_object = object_pos[active] if active_idx.size else object_pos[:0]

    cmd_x = stats(active_cmd[:, 0] if active_cmd.size else np.array([], dtype=np.float64))
    cmd_y = stats(active_cmd[:, 1] if active_cmd.size else np.array([], dtype=np.float64))
    cmd_yaw_deg = stats(np.rad2deg(active_cmd[:, 2]) if active_cmd.size else np.array([], dtype=np.float64))
    cmd_norm = stats(np.linalg.norm(active_cmd[:, :2], axis=1) if active_cmd.size else np.array([], dtype=np.float64))

    start_idx = int(active_idx[0]) if active_idx.size else 0
    end_idx = int(active_idx[-1]) if active_idx.size else max(int(root_pos.shape[0]) - 1, 0)
    root_delta = root_pos[end_idx, :2] - root_pos[start_idx, :2]
    object_delta = object_pos[end_idx, :2] - object_pos[start_idx, :2]
    root_motion_dir = math.atan2(float(root_delta[1]), float(root_delta[0])) if np.linalg.norm(root_delta) > 1e-9 else 0.0
    object_motion_dir = (
        math.atan2(float(object_delta[1]), float(object_delta[0])) if np.linalg.norm(object_delta) > 1e-9 else 0.0
    )

    first_cmd = active_cmd[0] if active_cmd.size else np.full((3,), np.nan)
    last_cmd = active_cmd[-1] if active_cmd.size else np.full((3,), np.nan)
    return {
        "clip_id": clip_id,
        "source_path": str(source_path),
        "mode": result["mode"],
        "quat_order": result["quat_order"],
        "num_frames": int(command.shape[0]),
        "fps": finite_or_none(result["fps"]),
        "carry_start": int(result["carry_start"]),
        "carry_end": int(result["carry_end"]),
        "active_frames": int(active_idx.size),
        "pickup_step": int(result["pickup_step"]),
        "pickup_threshold": finite_or_none(result["pickup_threshold"]),
        "root_yaw_start_deg": finite_or_none(np.rad2deg(root_yaw[0])),
        "root_yaw_active_start_deg": finite_or_none(np.rad2deg(root_yaw[start_idx])),
        "root_yaw_active_end_deg": finite_or_none(np.rad2deg(root_yaw[end_idx])),
        "root_yaw_active_delta_deg": finite_or_none(np.rad2deg(wrap_to_pi(root_yaw[end_idx] - root_yaw[start_idx]))),
        "root_active_delta_x_w": finite_or_none(root_delta[0]),
        "root_active_delta_y_w": finite_or_none(root_delta[1]),
        "root_active_distance_w": finite_or_none(np.linalg.norm(root_delta)),
        "root_active_motion_dir_deg": finite_or_none(np.rad2deg(root_motion_dir)),
        "object_active_delta_x_w": finite_or_none(object_delta[0]),
        "object_active_delta_y_w": finite_or_none(object_delta[1]),
        "object_active_distance_w": finite_or_none(np.linalg.norm(object_delta)),
        "object_active_motion_dir_deg": finite_or_none(np.rad2deg(object_motion_dir)),
        "first_active_cmd_x": finite_or_none(first_cmd[0]),
        "first_active_cmd_y": finite_or_none(first_cmd[1]),
        "first_active_cmd_yaw_deg": finite_or_none(np.rad2deg(first_cmd[2])),
        "last_active_cmd_x": finite_or_none(last_cmd[0]),
        "last_active_cmd_y": finite_or_none(last_cmd[1]),
        "last_active_cmd_yaw_deg": finite_or_none(np.rad2deg(last_cmd[2])),
        "active_cmd_x_mean": cmd_x["mean"],
        "active_cmd_x_std": cmd_x["std"],
        "active_cmd_x_min": cmd_x["min"],
        "active_cmd_x_max": cmd_x["max"],
        "active_cmd_y_mean": cmd_y["mean"],
        "active_cmd_y_std": cmd_y["std"],
        "active_cmd_y_min": cmd_y["min"],
        "active_cmd_y_max": cmd_y["max"],
        "active_cmd_yaw_deg_mean": cmd_yaw_deg["mean"],
        "active_cmd_yaw_deg_std": cmd_yaw_deg["std"],
        "active_cmd_yaw_deg_min": cmd_yaw_deg["min"],
        "active_cmd_yaw_deg_max": cmd_yaw_deg["max"],
        "active_cmd_xy_norm_mean": cmd_norm["mean"],
        "active_cmd_xy_norm_max": cmd_norm["max"],
    }


def clip_id_from_motion_data(path: Path, data: dict[str, Any]) -> str:
    if "clip_id" in data:
        value = scalar_string(data["clip_id"]).strip()
        if value:
            return value
    return path.stem


def iter_motion_sources(motion_dir: Path) -> list[Path]:
    return sorted(path for path in motion_dir.glob("*.npz") if path.is_file())


def iter_teacher_reference_sources(root: Path) -> list[Path]:
    if root.name == "teacher_rollout_reference.npz" and root.is_file():
        return [root]
    direct = sorted(root.glob("*/teacher_rollout_reference.npz"))
    if direct:
        return direct
    return sorted(root.rglob("teacher_rollout_reference.npz"))


def write_frame_jsonl(path: Path, rows: list[tuple[str, Path, dict[str, Any]]], *, stride: int) -> int:
    count = 0
    stride = max(1, int(stride))
    with path.open("w", encoding="utf-8") as f:
        for clip_id, source_path, result in rows:
            command = np.asarray(result["command"], dtype=np.float64)
            velocity_command = np.asarray(result["velocity_command"], dtype=np.float64)
            root_pos = np.asarray(result["root_pos"], dtype=np.float64)
            target_root_pos = np.asarray(result["target_root_pos"], dtype=np.float64)
            object_pos = np.asarray(result["object_pos"], dtype=np.float64)
            root_yaw = np.asarray(result["root_yaw"], dtype=np.float64)
            target_yaw = np.asarray(result["target_yaw"], dtype=np.float64)
            active = np.asarray(result["active"], dtype=bool)
            rel_z = np.asarray(result["rel_z"], dtype=np.float64)
            n = int(command.shape[0])
            for idx in range(0, n, stride):
                record = {
                    "clip_id": clip_id,
                    "source_path": str(source_path),
                    "mode": result["mode"],
                    "frame": idx,
                    "active": bool(active[idx]),
                    "root_pos": root_pos[idx].astype(float).tolist(),
                    "target_root_pos": target_root_pos[idx].astype(float).tolist(),
                    "object_pos": object_pos[idx].astype(float).tolist(),
                    "root_yaw": finite_or_none(root_yaw[idx]),
                    "root_yaw_deg": finite_or_none(np.rad2deg(root_yaw[idx])),
                    "target_yaw": finite_or_none(target_yaw[idx]),
                    "target_yaw_deg": finite_or_none(np.rad2deg(target_yaw[idx])),
                    "command": command[idx].astype(float).tolist(),
                    "command_yaw_deg": finite_or_none(np.rad2deg(command[idx, 2])),
                    "velocity_command": [
                        finite_or_none(value) if math.isfinite(float(value)) else None
                        for value in velocity_command[idx].astype(float).tolist()
                    ],
                    "rel_z": finite_or_none(rel_z[idx]),
                }
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
                count += 1
    return count


def write_npz(path: Path, rows: list[tuple[str, Path, dict[str, Any]]]) -> None:
    payload: dict[str, Any] = {}
    clip_ids: list[str] = []
    source_paths: list[str] = []
    modes: list[str] = []
    for idx, (clip_id, source_path, result) in enumerate(rows):
        prefix = f"clip_{idx:04d}"
        clip_ids.append(clip_id)
        source_paths.append(str(source_path))
        modes.append(str(result["mode"]))
        for key in (
            "command",
            "velocity_command",
            "active",
            "root_pos",
            "target_root_pos",
            "object_pos",
            "root_yaw",
            "target_yaw",
            "rel_z",
        ):
            payload[f"{prefix}_{key}"] = np.asarray(result[key])
        payload[f"{prefix}_carry_window"] = np.asarray([result["carry_start"], result["carry_end"]], dtype=np.int32)
        payload[f"{prefix}_pickup"] = np.asarray([result["pickup_step"], result["pickup_threshold"]], dtype=np.float64)
    payload["clip_ids"] = np.asarray(clip_ids)
    payload["source_paths"] = np.asarray(source_paths)
    payload["modes"] = np.asarray(modes)
    np.savez_compressed(path, **payload)


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(summaries[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def export_motion_commands(args: argparse.Namespace) -> tuple[Path, Path, Path, int]:
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, Path, dict[str, Any]]] = []
    summaries: list[dict[str, Any]] = []

    if args.teacher_reference_root is not None:
        mode = "teacher_reference"
        sources = iter_teacher_reference_sources(Path(args.teacher_reference_root).expanduser())
    else:
        mode = "motion_delta"
        sources = iter_motion_sources(Path(args.motion_dir).expanduser())

    if args.limit is not None:
        sources = sources[: max(0, int(args.limit))]
    if not sources:
        raise FileNotFoundError("No NPZ sources found for command export.")

    for source in sources:
        if mode == "teacher_reference":
            data = load_teacher_reference_npz(source)
            result = command_from_teacher_reference(data)
            clip_id = clip_id_from_motion_data(source.parent, data)
        else:
            data = load_motion_npz(source)
            result = command_from_motion_delta(data, lookahead_steps=args.lookahead_steps)
            clip_id = clip_id_from_motion_data(source, data)
        rows.append((clip_id, source, result))
        summaries.append(summarize_clip(clip_id, source, result))

    summary_path = out_dir / "motion_command_summary.csv"
    frames_path = out_dir / "motion_command_frames.jsonl"
    npz_path = out_dir / "motion_command_frames.npz"
    write_summary_csv(summary_path, summaries)
    frame_count = write_frame_jsonl(frames_path, rows, stride=args.frame_stride)
    write_npz(npz_path, rows)

    metadata = {
        "created_unix_time": time.time(),
        "mode": mode,
        "motion_dir": None if args.motion_dir is None else str(Path(args.motion_dir).expanduser()),
        "teacher_reference_root": None
        if args.teacher_reference_root is None
        else str(Path(args.teacher_reference_root).expanduser()),
        "num_clips": len(rows),
        "frame_jsonl_rows": frame_count,
        "lookahead_steps": int(args.lookahead_steps),
        "frame_stride": int(args.frame_stride),
        "outputs": {
            "summary_csv": str(summary_path),
            "frames_jsonl": str(frames_path),
            "frames_npz": str(npz_path),
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_path, frames_path, npz_path, len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-motion root-frame commands. For motion banks this exports next-frame pelvis/root "
            "delta commands in the current pelvis heading frame. For teacher rollout references it exports "
            "the exact sparse command target_root - rollout_root in the rollout root heading frame."
        )
    )
    parser.add_argument("--motion-dir", type=Path, required=True)
    parser.add_argument(
        "--teacher-reference-root",
        type=Path,
        default=None,
        help="Optional contact-export clips root containing */teacher_rollout_reference.npz.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("logs/runtime/motion_commands"))
    parser.add_argument("--lookahead-steps", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path, frames_path, npz_path, clip_count = export_motion_commands(args)
    print(f"exported_clips={clip_count}")
    print(f"summary_csv={summary_path}")
    print(f"frames_jsonl={frames_path}")
    print(f"frames_npz={npz_path}")


if __name__ == "__main__":
    main()
