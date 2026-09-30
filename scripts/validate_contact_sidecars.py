#!/usr/bin/env python3
"""Validate AS contact/teacher-rollout sidecars before simulator startup."""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path

import numpy as np

from holosoma.utils.contact_intervals import (
    CONTACT_INTERVAL_FALLBACK_FILES,
    convert_contact_interval_timebase,
    infer_contact_export_clip_id,
    resolve_contact_export_clip_id,
    select_primary_contact_interval,
)


DEFAULT_TRACKED_BODIES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]
DEFAULT_OFFLINE_WRIST_REGION_NAMES = ["left_wrist", "right_wrist"]
DEFAULT_OFFLINE_CONTACT_REGION_NAMES = [
    "left_wrist",
    "right_wrist",
    "left_elbow",
    "right_elbow",
    "left_wrist_roll",
    "right_wrist_roll",
    "left_wrist_pitch",
    "right_wrist_pitch",
    "torso",
]
SUPPORTED_OFFLINE_CONTACT_REGION_NAMES = frozenset(DEFAULT_OFFLINE_CONTACT_REGION_NAMES)


def _parse_string_list(raw_value: str, field_name: str) -> list[str]:
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        try:
            value = ast.literal_eval(raw_value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"{field_name} must be a JSON/Python list of strings") from exc
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a JSON/Python list of strings")
    return list(value)


def infer_clip_id(directory_name: str) -> str:
    return infer_contact_export_clip_id(directory_name)


def _scalar_text(data: np.lib.npyio.NpzFile, key: str) -> str:
    if key not in data.files:
        raise ValueError(f"missing {key}")
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{key} must be scalar, got shape {value.shape}")
    return str(value.item()).strip()


def _require_shape(data: np.lib.npyio.NpzFile, key: str, shape: tuple[int, ...]) -> np.ndarray:
    if key not in data.files:
        raise ValueError(f"missing {key}")
    value = np.asarray(data[key])
    if value.shape != shape:
        raise ValueError(f"{key} shape {value.shape} != {shape}")
    if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
        raise ValueError(f"{key} contains non-finite values")
    return value


def _validate_contact_arrays(clip_dir: Path, side: str, trajectory_length: int) -> tuple[int, int] | None:
    """Validate raw rollout-step contact arrays without applying runtime timebase conversion."""

    points = np.load(clip_dir / f"{side}_contact_points.npy", allow_pickle=False)
    counts = np.load(clip_dir / f"{side}_contact_point_counts.npy", allow_pickle=False)
    interval = np.load(clip_dir / f"{side}_contact_interval_steps.npy", allow_pickle=False)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError(f"{side}_contact_points shape {points.shape} must be (N, 3)")
    if counts.ndim != 1 or counts.shape[0] != points.shape[0]:
        raise ValueError(
            f"{side}_contact_point_counts shape {counts.shape} must match points count {points.shape[0]}"
        )
    if interval.shape != (2,):
        raise ValueError(f"{side}_contact_interval_steps shape {interval.shape} must be (2,)")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{side}_contact_points contains non-finite values")
    if not np.issubdtype(counts.dtype, np.integer) or np.any(counts <= 0):
        raise ValueError(f"{side}_contact_point_counts must contain positive integers")
    if not np.issubdtype(interval.dtype, np.integer):
        raise ValueError(f"{side}_contact_interval_steps must contain integer rollout-step indices")
    start, end = (int(interval[0]), int(interval[1]))
    if (start, end) == (-1, -1):
        return None
    if start < 0 or end < start:
        raise ValueError(
            f"{side}_contact_interval_steps [{start}, {end}] is neither an ordered interval nor the [-1, -1] sentinel"
        )
    # Do not compare raw interval indices with trajectory_length here. Contact
    # intervals use physical rollout-step time; runtime subtracts the explicit
    # default-pose prepend offset before checking against the motion clip.
    del trajectory_length
    return start, end


def _load_runtime_contact_window(
    clip_dir: Path,
    *,
    metadata: dict[str, object],
    motion_fps: float,
    motion_length: int,
    runtime_prepend_compensation: bool,
    runtime_prepend_duration_s: float,
) -> tuple[int, int] | None:
    """Load and validate the same primary/secondary interval used by training."""

    intervals_by_region: dict[str, object] = {}
    contact_intervals_path = clip_dir / "contact_intervals.json"
    if contact_intervals_path.is_file():
        try:
            payload = json.loads(contact_intervals_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"invalid contact_intervals.json: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("contact_intervals.json must be a JSON object")
        intervals_by_region.update(payload)

    if not intervals_by_region:
        for region_name, file_name in CONTACT_INTERVAL_FALLBACK_FILES.items():
            interval_path = clip_dir / file_name
            if interval_path.is_file():
                intervals_by_region[region_name] = np.load(interval_path, allow_pickle=False)

    interval = select_primary_contact_interval(intervals_by_region)
    if interval is None:
        return None
    interval = convert_contact_interval_timebase(
        interval,
        metadata=metadata,
        motion_fps=motion_fps,
    )

    runtime_prepend_offset = (
        round(float(runtime_prepend_duration_s) * motion_fps)
        if runtime_prepend_compensation
        else 0
    )
    start_step = max(0, int(interval[0]) - runtime_prepend_offset)
    end_step = int(interval[1]) - runtime_prepend_offset
    if end_step <= start_step:
        return None
    if start_step >= motion_length or end_step > motion_length:
        raise ValueError(
            "runtime contact interval is outside the active motion-time range: "
            f"raw={interval} prepend_steps={runtime_prepend_offset} "
            f"runtime={[start_step, end_step]} motion_length={motion_length}"
        )
    return start_step, end_step


def _validate_offline_contact_target_regions(
    clip_dir: Path,
    expected_clip_id: str,
    region_names: list[str],
    *,
    require_stable_contact: bool,
) -> set[str]:
    """Return configured regions with finite, non-empty point targets for one clip."""

    metadata_path = clip_dir / "metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"invalid contact metadata: {exc}") from exc
        if not isinstance(metadata, dict):
            raise ValueError("contact metadata must be a JSON object")
        embedded_clip_id = str(metadata.get("clip_id", "")).strip()
        if embedded_clip_id and embedded_clip_id != expected_clip_id:
            raise ValueError(
                f"contact metadata clip_id {embedded_clip_id!r} != active clip {expected_clip_id!r}"
            )
        # Match OfflineContactPointGuidance exactly: missing metadata (or an
        # empty object) remains legacy-compatible, while any non-empty metadata
        # must explicitly mark the rollout stable when the runtime requires it.
        if require_stable_contact and metadata and not bool(metadata.get("stable_contact_success", False)):
            return set()

    active_regions: set[str] = set()
    for region_name in region_names:
        points_path = clip_dir / f"{region_name}_contact_points.npy"
        counts_path = clip_dir / f"{region_name}_contact_point_counts.npy"
        if not points_path.is_file():
            if counts_path.is_file():
                raise ValueError(f"{region_name}_contact_point_counts exists without contact_points")
            continue
        points = np.load(points_path, allow_pickle=False)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError(f"{region_name}_contact_points shape {points.shape} must be (N, 3)")
        if not np.all(np.isfinite(points)):
            raise ValueError(f"{region_name}_contact_points contains non-finite values")
        if not counts_path.is_file():
            if points.shape[0] > 0:
                raise ValueError(f"{region_name}_contact_points has targets but contact_point_counts is missing")
            continue
        counts = np.load(counts_path, allow_pickle=False)
        if counts.ndim != 1 or counts.shape[0] != points.shape[0]:
            raise ValueError(
                f"{region_name}_contact_point_counts shape {counts.shape} must match points count {points.shape[0]}"
            )
        if not np.issubdtype(counts.dtype, np.integer) or np.any(counts <= 0):
            raise ValueError(f"{region_name}_contact_point_counts must contain positive integers")
        if points.shape[0] > 0:
            active_regions.add(region_name)
    return active_regions


def _required_rollout_reference_steps(motion_length: int, motion_end_mode: str) -> int:
    """Return the reference prefix that can actually be sampled by training.

    ``MotionCommand.motion_end_mask()`` terminates episodic WBT rollouts when
    ``time_steps >= motion_length - 2``.  BaseTask evaluates termination and
    reward before advancing the command clock, so the last reward-bearing
    index is ``motion_length - 2`` and the required prefix has ``L - 1``
    frames.  Continuing mode rolls through index ``L - 1`` and therefore
    requires the complete motion-length prefix.
    """

    if motion_end_mode == "episodic":
        return max(1, motion_length - 1)
    if motion_end_mode == "continuing":
        return motion_length
    raise ValueError(f"motion_end_mode must be episodic or continuing, got {motion_end_mode!r}")


def validate_clip(
    clip_dir: Path,
    expected_clip_id: str,
    tracked_body_names: list[str],
    ref_body_name: str,
    *,
    motion_path: Path,
    motion_end_mode: str,
    runtime_prepend_compensation: bool,
    runtime_prepend_duration_s: float,
) -> bool:
    required_files = [
        "teacher_rollout_reference.npz",
        "left_wrist_contact_points.npy",
        "left_wrist_contact_point_counts.npy",
        "left_wrist_contact_interval_steps.npy",
        "right_wrist_contact_points.npy",
        "right_wrist_contact_point_counts.npy",
        "right_wrist_contact_interval_steps.npy",
    ]
    missing = [name for name in required_files if not (clip_dir / name).is_file()]
    if missing:
        raise ValueError(f"missing required files: {missing}")

    with np.load(motion_path, allow_pickle=False) as motion:
        motion_length = None
        for key in ("body_pos_w", "joint_pos", "object_pos_w"):
            if key in motion.files:
                value = np.asarray(motion[key])
                if value.ndim >= 1:
                    motion_length = int(value.shape[0])
                    break
        if motion_length is None or motion_length < 1:
            raise ValueError(f"cannot determine motion length from {motion_path}")
        required_reference_steps = _required_rollout_reference_steps(motion_length, motion_end_mode)
        motion_fps = None
        if "fps" in motion.files:
            fps_value = np.asarray(motion["fps"])
            if fps_value.size == 1 and np.isfinite(float(fps_value.item())) and float(fps_value.item()) > 0.0:
                motion_fps = float(fps_value.item())

    rollout_path = clip_dir / "teacher_rollout_reference.npz"
    with np.load(rollout_path, allow_pickle=False) as data:
        embedded_clip_id = _scalar_text(data, "clip_id")
        if embedded_clip_id != expected_clip_id:
            raise ValueError(f"embedded clip_id {embedded_clip_id!r} != active clip {expected_clip_id!r}")

        if "tracked_body_names" not in data.files:
            raise ValueError("missing tracked_body_names")
        loaded_body_names = [str(value) for value in np.asarray(data["tracked_body_names"]).reshape(-1).tolist()]
        if loaded_body_names != tracked_body_names:
            raise ValueError(
                f"tracked_body_names {loaded_body_names!r} != motion config {tracked_body_names!r}"
            )
        loaded_ref_body = _scalar_text(data, "ref_body_name")
        if loaded_ref_body != ref_body_name:
            raise ValueError(f"ref_body_name {loaded_ref_body!r} != motion config {ref_body_name!r}")

        if "valid_steps" not in data.files:
            raise ValueError("missing valid_steps")
        valid_steps = np.asarray(data["valid_steps"], dtype=np.bool_).reshape(-1)
        trajectory_length = int(valid_steps.size)
        if trajectory_length == 0:
            raise ValueError("valid_steps is empty")
        if not np.any(valid_steps):
            raise ValueError("valid_steps contains no valid rollout step")
        if trajectory_length < required_reference_steps:
            raise ValueError(
                f"rollout reference has {trajectory_length} frames but {motion_end_mode} motion execution "
                f"requires {required_reference_steps} reward-bearing frames (motion_length={motion_length})"
            )
        if not np.all(valid_steps[:required_reference_steps]):
            first_invalid = int(np.flatnonzero(~valid_steps[:required_reference_steps])[0])
            raise ValueError(
                "valid_steps has a gap inside the reward-bearing motion range: "
                f"first_invalid_step={first_invalid}, required_reference_steps={required_reference_steps}, "
                f"motion_length={motion_length}, motion_end_mode={motion_end_mode}"
            )
        if "trajectory_length" in data.files and int(np.asarray(data["trajectory_length"]).item()) != trajectory_length:
            raise ValueError("trajectory_length metadata does not match valid_steps")

        num_bodies = len(tracked_body_names)
        required_arrays = {
            "body_pos_local": (trajectory_length, num_bodies, 3),
            "body_quat_w": (trajectory_length, num_bodies, 4),
            "body_lin_vel_w": (trajectory_length, num_bodies, 3),
            "body_ang_vel_w": (trajectory_length, num_bodies, 3),
            "ref_pos_local": (trajectory_length, 3),
            "ref_quat_w": (trajectory_length, 4),
            "ref_lin_vel_w": (trajectory_length, 3),
            "ref_ang_vel_w": (trajectory_length, 3),
            "root_pos_local": (trajectory_length, 3),
            "root_quat_w": (trajectory_length, 4),
            "root_lin_vel_w": (trajectory_length, 3),
            "root_ang_vel_w": (trajectory_length, 3),
            "object_pos_local": (trajectory_length, 3),
            "object_quat_w": (trajectory_length, 4),
            "object_lin_vel_w": (trajectory_length, 3),
            "object_ang_vel_w": (trajectory_length, 3),
        }
        loaded_arrays = {
            key: _require_shape(data, key, shape)
            for key, shape in required_arrays.items()
        }
        for quat_key in ("body_quat_w", "ref_quat_w", "root_quat_w", "object_quat_w"):
            valid_quat = loaded_arrays[quat_key][valid_steps]
            quat_norm = np.linalg.norm(valid_quat, axis=-1)
            if quat_norm.size == 0 or not np.all(np.abs(quat_norm - 1.0) <= 1.0e-3):
                min_norm = float(quat_norm.min()) if quat_norm.size else float("nan")
                max_norm = float(quat_norm.max()) if quat_norm.size else float("nan")
                raise ValueError(
                    f"{quat_key} must contain unit quaternions on valid steps; "
                    f"norm range=[{min_norm}, {max_norm}]"
                )

    _validate_contact_arrays(clip_dir, "left_wrist", trajectory_length)
    _validate_contact_arrays(clip_dir, "right_wrist", trajectory_length)
    if motion_fps is None:
        raise ValueError(f"motion clip has no valid fps metadata required for contact windows: {motion_path}")

    metadata: dict[str, object] = {}
    metadata_path = clip_dir / "metadata.json"
    if metadata_path.is_file():
        try:
            raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"invalid contact metadata: {exc}") from exc
        if not isinstance(raw_metadata, dict):
            raise ValueError("contact metadata must be a JSON object")
        metadata = raw_metadata

    runtime_window = _load_runtime_contact_window(
        clip_dir,
        metadata=metadata,
        motion_fps=motion_fps,
        motion_length=motion_length,
        runtime_prepend_compensation=runtime_prepend_compensation,
        runtime_prepend_duration_s=runtime_prepend_duration_s,
    )
    return runtime_window is not None


def validate_contact_root(
    motion_dir: Path,
    contact_root: Path,
    *,
    expected_total: int | None,
    tracked_body_names: list[str],
    ref_body_name: str,
    motion_end_mode: str = "continuing",
    runtime_prepend_compensation: bool = False,
    runtime_prepend_duration_s: float = 0.2,
    offline_contact_region_names: list[str] | None = None,
    offline_wrist_region_names: list[str] | None = None,
    require_stable_contact: bool = True,
    require_offline_contact_targets: bool = True,
    expected_valid_runtime_windows: int | None = None,
) -> Path:
    motion_dir = motion_dir.expanduser().resolve()
    contact_root = contact_root.expanduser().resolve()
    if not math.isfinite(runtime_prepend_duration_s) or runtime_prepend_duration_s < 0.0:
        raise ValueError(
            f"runtime prepend duration must be finite and non-negative, got {runtime_prepend_duration_s}"
        )
    if motion_end_mode not in {"episodic", "continuing"}:
        raise ValueError(f"motion_end_mode must be episodic or continuing, got {motion_end_mode!r}")
    if expected_valid_runtime_windows is not None and expected_valid_runtime_windows < 0:
        raise ValueError(
            "expected_valid_runtime_windows must be non-negative, got "
            f"{expected_valid_runtime_windows}"
        )
    contact_region_names = list(
        DEFAULT_OFFLINE_CONTACT_REGION_NAMES
        if offline_contact_region_names is None
        else offline_contact_region_names
    )
    wrist_region_names = list(
        DEFAULT_OFFLINE_WRIST_REGION_NAMES
        if offline_wrist_region_names is None
        else offline_wrist_region_names
    )
    for field_name, region_names in (
        ("offline_contact_region_names", contact_region_names),
        ("offline_wrist_region_names", wrist_region_names),
    ):
        if not region_names or len(region_names) != len(set(region_names)):
            raise ValueError(f"{field_name} must be a non-empty list without duplicates: {region_names!r}")
        unsupported = sorted(set(region_names) - SUPPORTED_OFFLINE_CONTACT_REGION_NAMES)
        if unsupported:
            raise ValueError(f"{field_name} contains unsupported regions: {unsupported}")
    clips_root = contact_root / "clips" if (contact_root / "clips").is_dir() else contact_root
    if not clips_root.is_dir():
        raise ValueError(f"Contact export root does not exist: {contact_root}")

    motion_paths = {path.stem: path for path in sorted(motion_dir.glob("*.npz"))}
    motion_ids = sorted(motion_paths)
    if expected_total is not None and len(motion_ids) != expected_total:
        raise ValueError(f"Expected {expected_total} active clips under {motion_dir}, found {len(motion_ids)}")
    if not motion_ids:
        raise ValueError(f"No .npz clips found under active motion dir: {motion_dir}")

    dirs_by_id: dict[str, list[Path]] = {}
    for clip_dir in sorted(path for path in clips_root.iterdir() if path.is_dir()):
        resolved_clip_id = resolve_contact_export_clip_id(clip_dir.name, motion_paths)
        dirs_by_id.setdefault(resolved_clip_id, []).append(clip_dir)

    errors: list[str] = []
    runtime_window_count = 0
    contact_region_clip_counts = {region_name: 0 for region_name in contact_region_names}
    contact_target_clip_count = 0
    wrist_target_clip_count = 0
    for clip_id in motion_ids:
        candidates = dirs_by_id.get(clip_id, [])
        if len(candidates) != 1:
            errors.append(f"{clip_id}: expected one contact directory, found {len(candidates)}")
            continue
        try:
            runtime_window_count += int(
                validate_clip(
                    candidates[0],
                    clip_id,
                    tracked_body_names,
                    ref_body_name,
                    motion_path=motion_paths[clip_id],
                    motion_end_mode=motion_end_mode,
                    runtime_prepend_compensation=runtime_prepend_compensation,
                    runtime_prepend_duration_s=runtime_prepend_duration_s,
                )
            )
            active_contact_regions = _validate_offline_contact_target_regions(
                candidates[0],
                clip_id,
                contact_region_names,
                require_stable_contact=require_stable_contact,
            )
            if require_offline_contact_targets and not active_contact_regions:
                raise ValueError(
                    "no non-empty offline contact target among configured regions "
                    f"{contact_region_names!r}"
                )
            for region_name in active_contact_regions:
                contact_region_clip_counts[region_name] += 1
            contact_target_clip_count += int(bool(active_contact_regions))
            active_wrist_regions = _validate_offline_contact_target_regions(
                candidates[0],
                clip_id,
                wrist_region_names,
                require_stable_contact=require_stable_contact,
            )
            wrist_target_clip_count += int(bool(active_wrist_regions))
        except Exception as exc:
            errors.append(f"{clip_id}: {exc}")

    if errors:
        preview = "\n  - ".join(errors[:20])
        suffix = "" if len(errors) <= 20 else f"\n  ... and {len(errors) - 20} more error(s)"
        raise ValueError("Contact sidecar contract validation failed:\n  - " + preview + suffix)
    if (
        expected_valid_runtime_windows is not None
        and runtime_window_count != expected_valid_runtime_windows
    ):
        raise ValueError(
            "Runtime contact-window coverage mismatch: "
            f"expected {expected_valid_runtime_windows}/{len(motion_ids)}, "
            f"found {runtime_window_count}/{len(motion_ids)}"
        )
    if runtime_prepend_compensation:
        print(
            "[INFO] runtime_contact_window_preflight "
            f"compensation=True prepend_duration_s={runtime_prepend_duration_s} "
            f"valid_windows={runtime_window_count}/{len(motion_ids)}",
            file=sys.stderr,
        )
    coverage = ",".join(
        f"{region_name}:{contact_region_clip_counts[region_name]}"
        for region_name in contact_region_names
    )
    print(
        "[INFO] offline_contact_target_preflight "
        f"required={require_offline_contact_targets} "
        f"contact_clip_coverage={contact_target_clip_count}/{len(motion_ids)} "
        f"wrist_clip_coverage={wrist_target_clip_count}/{len(motion_ids)} "
        f"per_region={coverage}",
        file=sys.stderr,
    )
    return clips_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-dir", required=True, type=Path)
    parser.add_argument("--contact-root", required=True, type=Path)
    parser.add_argument("--expected-total", type=int)
    parser.add_argument("--tracked-body-names", default=json.dumps(DEFAULT_TRACKED_BODIES))
    parser.add_argument("--ref-body-name", default="torso_link")
    parser.add_argument(
        "--motion-end-mode",
        choices=("episodic", "continuing"),
        default="continuing",
        help="Whether motion_ends terminates before the final clip frame or clips roll over continuously.",
    )
    parser.add_argument("--runtime-prepend-compensation", action="store_true")
    parser.add_argument("--runtime-prepend-duration-s", type=float, default=0.2)
    parser.add_argument(
        "--offline-contact-region-names",
        default=json.dumps(DEFAULT_OFFLINE_CONTACT_REGION_NAMES),
    )
    parser.add_argument(
        "--offline-wrist-region-names",
        default=json.dumps(DEFAULT_OFFLINE_WRIST_REGION_NAMES),
    )
    parser.add_argument(
        "--allow-missing-offline-contact-targets",
        action="store_true",
        help=(
            "Validate all present contact arrays but allow clips with no non-empty contact target. "
            "This is only appropriate when positive offline/contact-guidance rewards are disabled."
        ),
    )
    parser.add_argument(
        "--expected-valid-runtime-windows",
        type=int,
        help="Require this exact number of clips to have a valid runtime contact window.",
    )
    args = parser.parse_args()
    tracked_body_names = json.loads(args.tracked_body_names)
    if not isinstance(tracked_body_names, list) or not all(isinstance(value, str) for value in tracked_body_names):
        raise SystemExit("[ERROR] --tracked-body-names must be a JSON list of strings")
    try:
        offline_contact_region_names = _parse_string_list(
            args.offline_contact_region_names,
            "--offline-contact-region-names",
        )
        offline_wrist_region_names = _parse_string_list(
            args.offline_wrist_region_names,
            "--offline-wrist-region-names",
        )
        clips_root = validate_contact_root(
            args.motion_dir,
            args.contact_root,
            expected_total=args.expected_total,
            tracked_body_names=tracked_body_names,
            ref_body_name=args.ref_body_name,
            motion_end_mode=args.motion_end_mode,
            runtime_prepend_compensation=args.runtime_prepend_compensation,
            runtime_prepend_duration_s=args.runtime_prepend_duration_s,
            offline_contact_region_names=offline_contact_region_names,
            offline_wrist_region_names=offline_wrist_region_names,
            require_offline_contact_targets=not args.allow_missing_offline_contact_targets,
            expected_valid_runtime_windows=args.expected_valid_runtime_windows,
        )
    except Exception as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
    print(clips_root)


if __name__ == "__main__":
    main()
