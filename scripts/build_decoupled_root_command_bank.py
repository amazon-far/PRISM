#!/usr/bin/env python3
"""Build an immutable motion view with precomputed turn-then-forward commands."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
import xml.etree.ElementTree as ET
import zipfile

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
COMMAND_MODULE_PATH = SCRIPT_DIR / "export_heading_path_commands.py"
COMMAND_KEY = "policy_command_xy_yaw"
PHASE_KEY = "policy_command_phase"
FILE_MODE = 0o444
DIRECTORY_MODE = 0o555


def _load_command_module():
    spec = importlib.util.spec_from_file_location("heading_path_commands", COMMAND_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import command generator: {COMMAND_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_stable(path: Path) -> bytes:
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    if not stat.S_ISREG(before.st_mode):
        raise FileNotFoundError(f"Expected a regular file: {path}")
    payload = resolved.read_bytes()
    after = resolved.stat()
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after) or len(payload) != before.st_size:
        raise RuntimeError(f"Source changed while being read: {path}")
    return payload


def _write_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _deterministic_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for key in sorted(arrays):
            if not key or "/" in key or "\\" in key:
                raise ValueError(f"Invalid NPZ array key: {key!r}")
            array_buffer = io.BytesIO()
            np.lib.format.write_array(array_buffer, np.asarray(arrays[key]), allow_pickle=False)
            member = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o600 << 16
            archive.writestr(member, array_buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
    return buffer.getvalue()


def _arrays_exact(left: np.ndarray, right: np.ndarray) -> bool:
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    if left.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _canonical_records_digest(records: list[dict[str, object]]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return _sha256_bytes(payload)


def _object_category(clip_id: str) -> str:
    lowered = clip_id.lower()
    for category in ("barrel", "bin", "ball", "box"):
        if category in lowered:
            return category
    raise ValueError(f"Cannot infer an allowed object category from clip ID: {clip_id}")


def _mesh_targets(urdf_path: Path) -> list[Path]:
    try:
        root = ET.fromstring(_read_stable(urdf_path))
    except ET.ParseError as exc:
        raise ValueError(f"Invalid URDF XML: {urdf_path}: {exc}") from exc
    targets: list[Path] = []
    for mesh in root.findall(".//mesh"):
        raw = str(mesh.attrib.get("filename", "")).strip()
        if not raw:
            raise ValueError(f"URDF mesh element is missing filename: {urdf_path}")
        if "://" in raw or raw.startswith("package:"):
            raise ValueError(f"Remote/package mesh references are forbidden: {urdf_path}: {raw}")
        target = Path(raw).expanduser()
        if not target.is_absolute():
            target = urdf_path.parent / target
        target = target.resolve(strict=True)
        if not target.is_file():
            raise FileNotFoundError(f"URDF mesh target is not a regular file: {target}")
        targets.append(target)
    if not targets:
        raise ValueError(f"URDF contains no visual/collision mesh: {urdf_path}")
    return targets


def _copy_and_validate_object_closure(
    source: Path,
    temporary: Path,
    clip_ids: list[str],
) -> tuple[bytes, list[dict[str, object]]]:
    map_source = source / "_clip_object_urdf_map.json"
    map_bytes = _read_stable(map_source)
    try:
        object_map = json.loads(map_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid object map: {map_source}: {exc}") from exc
    clips = object_map.get("clips") if isinstance(object_map, dict) else None
    if not isinstance(clips, dict) or sorted(clips) != clip_ids:
        raise ValueError("Object map clip IDs do not exactly match the source NPZ bank.")
    _write_file(temporary / "_clip_object_urdf_map.json", map_bytes)

    records: list[dict[str, object]] = []
    for clip_id in clip_ids:
        entry = clips[clip_id]
        if not isinstance(entry, dict):
            raise ValueError(f"Object-map entry is not an object: {clip_id}")
        raw_urdf = str(entry.get("object_urdf_path", "")).strip()
        if not raw_urdf:
            raise ValueError(f"Object-map entry has no object_urdf_path: {clip_id}")
        if Path(raw_urdf).is_absolute() or ".." in Path(raw_urdf).parts:
            raise ValueError(f"Object-map URDF path must be local and relative: {clip_id}: {raw_urdf}")
        source_urdf = source / raw_urdf
        target_urdf = temporary / raw_urdf
        urdf_bytes = _read_stable(source_urdf)
        _write_file(target_urdf, urdf_bytes)

        source_meshes = _mesh_targets(source_urdf)
        target_meshes = _mesh_targets(target_urdf)
        if source_meshes != target_meshes:
            raise ValueError(
                "Copied URDF no longer resolves to the exact source real meshes; "
                f"output directory depth is incompatible for clip {clip_id}."
            )
        records.append(
            {
                "path": target_urdf.relative_to(temporary).as_posix(),
                "sha256": _sha256_bytes(urdf_bytes),
                "size": len(urdf_bytes),
                "mesh_targets": [os.path.relpath(path, source) for path in source_meshes],
            }
        )
    return map_bytes, records


def _copy_portable_source_tree(
    source: Path,
    temporary: Path,
    clip_ids: list[str],
) -> tuple[bytes, list[dict[str, object]], list[dict[str, object]]]:
    """Copy a self-contained bank while reserving top-level motions for rewriting."""

    map_source = source / "_clip_object_urdf_map.json"
    map_bytes = _read_stable(map_source)
    try:
        object_map = json.loads(map_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid object map: {map_source}: {exc}") from exc
    clips = object_map.get("clips") if isinstance(object_map, dict) else None
    if not isinstance(clips, dict) or sorted(clips) != clip_ids:
        raise ValueError("Object map clip IDs do not exactly match the source NPZ bank.")

    source_records: list[dict[str, object]] = []
    generated_records: list[dict[str, object]] = []
    for source_path in sorted(source.rglob("*"), key=lambda path: path.relative_to(source).as_posix()):
        if source_path.is_symlink():
            raise ValueError(f"Portable source banks must not contain symlinks: {source_path}")
        if source_path.is_dir():
            continue
        if not source_path.is_file():
            raise ValueError(f"Portable source banks may contain only regular files: {source_path}")
        relative = source_path.relative_to(source)
        payload = _read_stable(source_path)
        record = {
            "path": relative.as_posix(),
            "sha256": _sha256_bytes(payload),
            "size": len(payload),
        }
        source_records.append(record)
        if relative.parent == Path() and (relative.suffix == ".npz" or relative.name == "manifest.json"):
            continue
        _write_file(temporary / relative, payload)
        generated_records.append(dict(record))

    output_map = temporary / "_clip_object_urdf_map.json"
    if _read_stable(output_map) != map_bytes:
        raise AssertionError("Portable object map changed while being copied.")
    for clip_id in clip_ids:
        entry = clips[clip_id]
        if not isinstance(entry, dict):
            raise ValueError(f"Object-map entry is not an object: {clip_id}")
        raw_urdf = str(entry.get("object_urdf_path", "")).strip()
        if not raw_urdf:
            raise ValueError(f"Object-map entry has no object_urdf_path: {clip_id}")
        relative_urdf = Path(raw_urdf)
        if relative_urdf.is_absolute() or ".." in relative_urdf.parts:
            raise ValueError(f"Object-map URDF path must be local and relative: {clip_id}: {raw_urdf}")
        output_urdf = (temporary / relative_urdf).resolve(strict=True)
        try:
            output_urdf.relative_to(temporary.resolve(strict=True))
        except ValueError as exc:
            raise ValueError(f"Object URDF escapes the portable bank: {clip_id}: {raw_urdf}") from exc
        for mesh_target in _mesh_targets(output_urdf):
            try:
                mesh_target.relative_to(temporary.resolve(strict=True))
            except ValueError as exc:
                raise ValueError(
                    f"Object mesh escapes the portable bank: {clip_id}: {mesh_target}"
                ) from exc

    return map_bytes, source_records, generated_records


def _seal_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        os.chmod(path, DIRECTORY_MODE if path.is_dir() else FILE_MODE)
    os.chmod(root, DIRECTORY_MODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoothing-steps", type=int, default=5)
    parser.add_argument("--rdp-epsilon-m", type=float, default=0.06)
    parser.add_argument("--minimum-leg-m", type=float, default=0.10)
    parser.add_argument("--minimum-turn-deg", type=float, default=25.0)
    parser.add_argument("--forward-command-m", type=float, default=0.15)
    parser.add_argument("--minimum-turn-steps", type=int, default=8)
    parser.add_argument("--maximum-turn-steps", type=int, default=30)
    parser.add_argument("--minimum-forward-steps-between-turns", type=int, default=5)
    parser.add_argument("--expected-clip-count", type=int, default=79)
    parser.add_argument(
        "--expected-category-counts-json",
        default='{"box":25,"ball":4,"barrel":35,"bin":15}',
    )
    parser.add_argument(
        "--copy-portable-source-tree",
        action="store_true",
        help="Copy all non-motion payload, including contact labels and in-bank real meshes.",
    )
    parser.add_argument("--expected-source-payload-digest", default="")
    parser.add_argument("--expected-source-manifest-sha256", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to replace an existing output: {output}")

    source_npz = sorted(source.glob("*.npz"), key=lambda path: path.name)
    if args.expected_clip_count <= 0:
        raise ValueError("expected_clip_count must be positive.")
    if len(source_npz) != args.expected_clip_count:
        raise ValueError(
            f"Expected {args.expected_clip_count} top-level NPZ files, found {len(source_npz)}."
        )
    clip_ids = [path.stem for path in source_npz]
    if len(set(clip_ids)) != len(clip_ids):
        raise ValueError("Duplicate source clip IDs are forbidden.")
    try:
        expected_category_counts = json.loads(args.expected_category_counts_json)
    except json.JSONDecodeError as exc:
        raise ValueError("expected_category_counts_json must be valid JSON.") from exc
    if not isinstance(expected_category_counts, dict) or set(expected_category_counts) != {
        "box",
        "ball",
        "barrel",
        "bin",
    }:
        raise ValueError("expected_category_counts_json must define box, ball, barrel, and bin.")
    expected_category_counts = {
        key: int(value) for key, value in expected_category_counts.items()
    }
    if any(value < 0 for value in expected_category_counts.values()) or sum(
        expected_category_counts.values()
    ) != args.expected_clip_count:
        raise ValueError("Expected category counts must be non-negative and sum to expected_clip_count.")

    source_manifest_sha = ""
    source_payload_digest = ""
    if (
        args.copy_portable_source_tree
        or args.expected_source_manifest_sha256
        or args.expected_source_payload_digest
    ):
        source_manifest_path = source / "manifest.json"
        source_manifest_bytes = _read_stable(source_manifest_path)
        source_manifest_sha = _sha256_bytes(source_manifest_bytes)
        if (
            args.expected_source_manifest_sha256
            and source_manifest_sha != args.expected_source_manifest_sha256
        ):
            raise ValueError(
                "Source manifest SHA256 mismatch: "
                f"expected={args.expected_source_manifest_sha256} actual={source_manifest_sha}"
            )
        try:
            source_manifest = json.loads(source_manifest_bytes)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid source manifest: {source_manifest_path}") from exc
        # Merged training banks call this immutable identity
        # ``payload_digest``.  Audited policy-rollout banks expose the same
        # contract as ``source_digest``.  Accept either spelling, while the
        # caller-provided expected digest is still checked exactly below.
        source_payload_digest = str(
            source_manifest.get(
                "payload_digest", source_manifest.get("source_digest", "")
            )
        )
        if (
            args.expected_source_payload_digest
            and source_payload_digest != args.expected_source_payload_digest
        ):
            raise ValueError(
                "Source payload digest mismatch: "
                f"expected={args.expected_source_payload_digest} actual={source_payload_digest}"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=output.parent))
    command_module = _load_command_module()
    parameters = {
        "smoothing_steps": args.smoothing_steps,
        "rdp_epsilon_m": args.rdp_epsilon_m,
        "minimum_leg_m": args.minimum_leg_m,
        "minimum_turn_deg": args.minimum_turn_deg,
        "forward_command_m": args.forward_command_m,
        "minimum_turn_steps": args.minimum_turn_steps,
        "maximum_turn_steps": args.maximum_turn_steps,
        "minimum_forward_steps_between_turns": args.minimum_forward_steps_between_turns,
    }
    source_records: list[dict[str, object]] = []
    generated_records: list[dict[str, object]] = []
    clip_records: list[dict[str, object]] = []
    category_counts: dict[str, int] = {key: 0 for key in ("box", "ball", "barrel", "bin")}
    total_phase_counts = np.zeros((3,), dtype=np.int64)

    try:
        if args.copy_portable_source_tree:
            map_bytes, portable_source_records, portable_generated_records = (
                _copy_portable_source_tree(source, temporary, clip_ids)
            )
            source_records.extend(portable_source_records)
            generated_records.extend(portable_generated_records)
            urdf_records: list[dict[str, object]] = []
        else:
            map_bytes, urdf_records = _copy_and_validate_object_closure(source, temporary, clip_ids)
            source_records.append(
                {
                    "path": "_clip_object_urdf_map.json",
                    "sha256": _sha256_bytes(map_bytes),
                    "size": len(map_bytes),
                }
            )
            generated_records.append(dict(source_records[-1]))
            generated_records.extend(urdf_records)

        for source_path in source_npz:
            source_bytes = _read_stable(source_path)
            source_sha = _sha256_bytes(source_bytes)
            with np.load(io.BytesIO(source_bytes), allow_pickle=False) as data:
                if COMMAND_KEY in data.files or PHASE_KEY in data.files:
                    raise ValueError(f"Source clip already contains derived command fields: {source_path}")
                arrays = {key: np.asarray(data[key]).copy() for key in data.files}
            required = {"body_pos_w", "body_quat_w", "object_pos_w", "fps"}
            missing = sorted(required.difference(arrays))
            if missing:
                raise KeyError(f"Source clip {source_path} is missing required arrays: {missing}")
            root_pos = np.asarray(arrays["body_pos_w"], dtype=np.float64)[:, 0]
            root_quat = np.asarray(arrays["body_quat_w"], dtype=np.float64)[:, 0]
            object_pos = np.asarray(arrays["object_pos_w"], dtype=np.float64)
            carry_start, carry_end, pickup_threshold = command_module.xm0_post_pickup_window_from_rel_z(
                root_pos,
                object_pos,
            )
            result = command_module.compute_turn_then_forward_commands(
                root_pos,
                root_quat,
                carry_start=carry_start,
                carry_end=carry_end,
                **parameters,
            )
            command = np.asarray(result["command"], dtype=np.float32)
            phase = np.asarray(result["phase"], dtype=np.uint8)
            arrays[COMMAND_KEY] = command
            arrays[PHASE_KEY] = phase
            target_bytes = _deterministic_npz_bytes(arrays)
            target_path = temporary / source_path.name
            _write_file(target_path, target_bytes)

            with np.load(target_path, allow_pickle=False) as rewritten:
                if set(rewritten.files) != set(arrays):
                    raise AssertionError(f"Rewritten NPZ key mismatch: {source_path.name}")
                for key, expected in arrays.items():
                    actual = np.asarray(rewritten[key])
                    if not _arrays_exact(actual, expected):
                        raise AssertionError(f"Rewritten array changed: {source_path.name}:{key}")

            category = _object_category(source_path.stem)
            category_counts[category] += 1
            phase_counts = np.asarray(result["phase_counts"], dtype=np.int64)
            total_phase_counts += phase_counts
            if not args.copy_portable_source_tree:
                source_records.append(
                    {"path": source_path.name, "sha256": source_sha, "size": len(source_bytes)}
                )
            generated_records.append(
                {
                    "path": source_path.name,
                    "sha256": _sha256_bytes(target_bytes),
                    "size": len(target_bytes),
                }
            )
            clip_records.append(
                {
                    "clip_id": source_path.stem,
                    "category": category,
                    "frame_count": int(command.shape[0]),
                    "fps": float(np.asarray(arrays["fps"]).reshape(-1)[0]),
                    "carry_window": [carry_start, carry_end],
                    "pickup_threshold_rel_z_m": pickup_threshold,
                    "phase_counts": phase_counts.tolist(),
                    "raw_rdp_vertex_frames": result["raw_rdp_vertex_frames"].astype(int).tolist(),
                    "vertex_frames": result["vertex_frames"].astype(int).tolist(),
                    "legs": result["legs"],
                    "turn_windows": result["turn_windows"],
                    "path_arc_length_m": result["path_arc_length_m"],
                    "path_net_displacement_m": result["path_net_displacement_m"],
                    "source_npz_sha256": source_sha,
                    "derived_npz_sha256": _sha256_bytes(target_bytes),
                }
            )

        for record in urdf_records:
            relative = str(record["path"])
            source_payload = _read_stable(source / relative)
            source_records.append(
                {"path": relative, "sha256": _sha256_bytes(source_payload), "size": len(source_payload)}
            )

        source_records.sort(key=lambda record: str(record["path"]))
        generated_records.sort(key=lambda record: str(record["path"]))
        clip_records.sort(key=lambda record: str(record["clip_id"]))
        if category_counts != expected_category_counts:
            raise ValueError(
                f"Unexpected category counts: actual={category_counts} expected={expected_category_counts}"
            )
        if total_phase_counts[1] <= 0 or total_phase_counts[2] <= 0:
            raise ValueError(f"Derived bank lacks forward or yaw command rows: {total_phase_counts.tolist()}")

        manifest = {
            "schema_version": 1,
            "semantics": "precomputed_open_loop_heading_relative_turn_then_forward_actor_input",
            "training_behavior": {
                "actor_command_only": True,
                "drop_button_and_contact_labels_unchanged": True,
                "reference_tracking_reward_unchanged": True,
                "reference_motion_timeline_unchanged": True,
                "runtime_pickup_latch_required": True,
                "policy_checkpoint_used": False,
            },
            "invariants": {
                "dy_always_zero": True,
                "dx_nonnegative": True,
                "dx_and_dyaw_never_overlap": True,
                "pre_pickup_command_zero": True,
                "fallback_geometry_allowed": False,
                "source_arrays_exactly_preserved": True,
            },
            "command_keys": {"command": COMMAND_KEY, "phase": PHASE_KEY},
            "phase_values": {"0": "zero", "1": "forward_only", "2": "yaw_only"},
            "parameters": parameters,
            "clip_count": len(clip_records),
            "category_counts": category_counts,
            "total_phase_counts": total_phase_counts.tolist(),
            "source_payload_digest": source_payload_digest,
            "source_manifest_sha256": source_manifest_sha,
            "source_view_digest": _canonical_records_digest(source_records),
            "derived_payload_digest": _canonical_records_digest(generated_records),
            "source_records": source_records,
            "generated_records": generated_records,
            "clips": clip_records,
        }
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()
        _write_file(temporary / "manifest.json", manifest_bytes)

        for directory in sorted(
            [path for path in temporary.rglob("*") if path.is_dir()], reverse=True
        ):
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        root_descriptor = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(root_descriptor)
        finally:
            os.close(root_descriptor)
        _seal_tree(temporary)
        os.replace(temporary, output)
        parent_descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        if temporary.exists():
            for path in temporary.rglob("*"):
                try:
                    os.chmod(path, 0o700 if path.is_dir() else 0o600)
                except OSError:
                    pass
            shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "output": str(output),
                "clip_count": len(clip_records),
                "phase_counts": total_phase_counts.tolist(),
                "source_view_digest": manifest["source_view_digest"],
                "derived_payload_digest": manifest["derived_payload_digest"],
                "manifest_sha256": _sha256_bytes(manifest_bytes),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
