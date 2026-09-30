#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import fcntl
import hashlib
import io
import json
import os
import shutil
import stat
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np



MARKER_NAME = ".generated_by_realmesh_rollout"
INTEGRITY_MANIFEST_NAME = "generation_integrity.json"
PUBLICATION_ROOT_METADATA_PATHS = frozenset(
    {
        "realmesh_rollout_manifest.json",
        "merge_manifest.json",
        INTEGRITY_MANIFEST_NAME,
    }
)
DEFAULT_CONTACT_EXPORT_NAME = "contact_export_from_teacher_realmesh_rollout"
DATA_BANK_ROOT = REPO_ROOT / "data" / "ds_as_data"
TEACHER_ROLLOUT_SHARD_ROOT = DATA_BANK_ROOT / "_teacher_rollout_shards"
TEACHER_ROLLOUT_GENERATION_ROOT = DATA_BANK_ROOT / "_teacher_rollout_generations"
TEACHER_ROLLOUT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "teacher_as_contacts"
_OBJECT_ASSET_PATH_KEYS = (
    "object_urdf_path",
    "object_mesh_path",
    "object_visual_mesh_path",
    "object_collision_mesh_path",
)
_OBJECT_ASSET_PATH_LIST_KEYS = (
    "object_mesh_paths",
    "object_visual_mesh_paths",
    "object_collision_mesh_paths",
)


def _parse_clip_map_bytes(raw: bytes, *, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(raw.decode("utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
        return {key: value for key, value in payload.items() if key != "clips"}, payload["clips"]
    if isinstance(payload, dict):
        return {}, payload
    raise ValueError(f"Invalid object map: {path}")


def _stable_regular_file_bytes(path: Path, *, label: str) -> tuple[bytes, str]:
    if not hasattr(os, "O_NOFOLLOW"):
        raise SystemExit(f"[ERROR] O_NOFOLLOW is unavailable while reading {label}: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise SystemExit(f"[ERROR] {label} is not a one-link regular file: {path}")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if before_identity != after_identity or sum(map(len, chunks)) != after.st_size:
            raise SystemExit(f"[ERROR] {label} changed while it was read: {path}")
        return b"".join(chunks), digest.hexdigest()
    finally:
        os.close(descriptor)


def _load_clip_map(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, _digest = _stable_regular_file_bytes(path, label="clip object map")
    return _parse_clip_map_bytes(raw, path=path)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_content_identity(path: Path) -> dict[str, Any]:
    if not hasattr(os, "O_NOFOLLOW"):
        raise SystemExit(f"[ERROR] O_NOFOLLOW is unavailable while hashing: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise SystemExit(f"[ERROR] Hashed input is not a one-link regular file: {path}")
        digest = hashlib.sha256()
        bytes_read = 0
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            bytes_read += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if before_identity != after_identity or bytes_read != after.st_size:
            raise SystemExit(f"[ERROR] Hashed input changed while it was read: {path}")
        return {"sha256": digest.hexdigest(), "size": int(after.st_size)}
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    return str(_file_content_identity(path)["sha256"])


def _teacher_lineage_from_summary(
    summary_path: Path,
    *,
    expected_checkpoint_sha256: str,
    expected_checkpoint_path: Path,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if payload is None:
        raw, _digest = _stable_regular_file_bytes(summary_path, label="teacher shard summary")
        payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"[ERROR] Shard summary must be a JSON object: {summary_path}")
    checkpoint_sha256 = payload.get("source_checkpoint_sha256")
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise SystemExit(
            "[ERROR] Shard used a different teacher checkpoint: "
            f"summary={summary_path} actual={checkpoint_sha256!r} expected={expected_checkpoint_sha256!r}"
        )
    raw_checkpoint_path = payload.get("checkpoint")
    if not isinstance(raw_checkpoint_path, str) or not raw_checkpoint_path:
        raise SystemExit(f"[ERROR] Shard summary has no checkpoint path: {summary_path}")
    if Path(raw_checkpoint_path).expanduser().resolve() != expected_checkpoint_path:
        raise SystemExit(
            "[ERROR] Shard summary checkpoint path differs from the immutable launch file: "
            f"summary={summary_path} actual={raw_checkpoint_path!r} expected={str(expected_checkpoint_path)!r}"
        )
    export_config = payload.get("export_config")
    if not isinstance(export_config, dict):
        raise SystemExit(f"[ERROR] Shard summary has no export_config mapping: {summary_path}")
    semantic_export_config = dict(export_config)
    semantic_export_config.pop("output_dir", None)
    return {
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_training_provenance": payload.get("source_training_provenance"),
        "saved_wandb_path": payload.get("saved_wandb_path"),
        "export_config": semantic_export_config,
        "num_envs": payload.get("num_envs"),
    }


def _lexical_absolute(path: Path) -> Path:
    """Return an absolute normalized path without resolving symlinks."""
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _validate_destructive_path(path: Path, *, allowed_root: Path, label: str) -> Path:
    """Resolve a deletion/copy target and fail closed outside its owned root."""
    raw = str(path).strip()
    if not raw or raw == ".":
        raise SystemExit(f"[ERROR] Refusing empty or current-directory {label}: {path}")

    allowed_lexical = _lexical_absolute(allowed_root)
    allowed_resolved = allowed_lexical.resolve(strict=False)
    if allowed_lexical != allowed_resolved:
        raise SystemExit(
            f"[ERROR] Refusing symlinked/aliased allowed root for {label}: "
            f"{allowed_lexical} -> {allowed_resolved}"
        )
    target_lexical = _lexical_absolute(path)
    try:
        relative = target_lexical.relative_to(allowed_lexical)
    except ValueError as exc:
        raise SystemExit(
            f"[ERROR] Refusing {label} outside allowed root {allowed_resolved}: {target_lexical}"
        ) from exc
    if not relative.parts:
        raise SystemExit(f"[ERROR] Refusing to use allowed root itself as {label}: {allowed_resolved}")

    # Reject every symlink below the trusted root, including a symlink at the
    # final target.  Merely checking Path.resolve() would catch escapes but
    # would still let a generated cleanup unexpectedly erase another in-root
    # directory through a symlink alias.
    cursor = allowed_lexical
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise SystemExit(f"[ERROR] Refusing symlink component in {label}: {cursor}")

    target_resolved = target_lexical.resolve(strict=False)
    try:
        target_resolved.relative_to(allowed_resolved)
    except ValueError as exc:
        raise SystemExit(
            f"[ERROR] Refusing symlink escape for {label}: {target_lexical} -> {target_resolved}"
        ) from exc
    if target_resolved == Path(target_resolved.anchor):
        raise SystemExit(f"[ERROR] Refusing filesystem root as {label}: {target_resolved}")
    if target_resolved in {REPO_ROOT.resolve(), DATA_BANK_ROOT.resolve(strict=False)}:
        raise SystemExit(f"[ERROR] Refusing protected repository/data root as {label}: {target_resolved}")
    return target_resolved


def _validate_alias_path(path: Path, *, allowed_root: Path, label: str) -> Path:
    """Validate an owned publication alias while allowing the final symlink."""

    raw = str(path).strip()
    if not raw or raw == ".":
        raise SystemExit(f"[ERROR] Refusing empty or current-directory {label}: {path}")
    root = _lexical_absolute(allowed_root)
    if root != root.resolve(strict=False):
        raise SystemExit(f"[ERROR] Refusing symlinked allowed root for {label}: {root}")
    alias = _lexical_absolute(path)
    try:
        relative = alias.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"[ERROR] Refusing {label} outside allowed root {root}: {alias}") from exc
    if not relative.parts:
        raise SystemExit(f"[ERROR] Refusing allowed root itself as {label}: {root}")
    cursor = root
    for part in relative.parts[:-1]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise SystemExit(f"[ERROR] Refusing symlinked parent in {label}: {cursor}")
    return alias


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _scoped_no_follow_lock(path: Path, *, allowed_root: Path, label: str):
    """Hold one owned regular-file flock without following a hostile symlink."""

    lock_path = _validate_destructive_path(path, allowed_root=allowed_root, label=label)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        identity = os.fstat(descriptor)
        if not stat.S_ISREG(identity.st_mode) or identity.st_nlink != 1:
            raise SystemExit(f"[ERROR] Refusing non-regular or multiply-linked {label}: {lock_path}")
        timeout_raw = os.environ.get("TEACHER_ROLLOUT_PUBLICATION_LOCK_TIMEOUT_S", "120")
        try:
            timeout_seconds = float(timeout_raw)
        except ValueError as exc:
            raise SystemExit(
                f"[ERROR] TEACHER_ROLLOUT_PUBLICATION_LOCK_TIMEOUT_S must be numeric: {timeout_raw!r}"
            ) from exc
        if not (0.0 <= timeout_seconds <= 3600.0):
            raise SystemExit("[ERROR] Teacher rollout publication lock timeout must be in [0, 3600] seconds.")
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise SystemExit(
                        f"[ERROR] Timed out after {timeout_seconds:g}s waiting for {label}: {lock_path}"
                    )
                time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        yield lock_path
    finally:
        os.close(descriptor)


def _ensure_static_view_alias(
    alias: Path,
    target: Path,
    *,
    allowed_root: Path,
) -> None:
    """Create a compatibility symlink once; never rewrite an existing view."""

    alias = _validate_alias_path(alias, allowed_root=allowed_root, label="teacher rollout static view alias")
    alias.parent.mkdir(parents=True, exist_ok=True)
    expected_target = os.path.relpath(_lexical_absolute(target), alias.parent)
    if alias.is_symlink():
        actual_target = os.readlink(alias)
        if actual_target != expected_target:
            raise SystemExit(
                "[ERROR] Existing teacher rollout view alias has an unexpected target: "
                f"alias={alias} actual={actual_target!r} expected={expected_target!r}"
            )
        return
    if alias.exists():
        raise SystemExit(f"[ERROR] Refusing to replace existing non-symlink rollout view: {alias}")

    temporary_alias = alias.with_name(f".{alias.name}.static-next-{os.getpid()}")
    if temporary_alias.exists() or temporary_alias.is_symlink():
        raise SystemExit(f"[ERROR] Temporary static view alias already exists: {temporary_alias}")
    temporary_alias.symlink_to(expected_target, target_is_directory=target.suffix == "")
    try:
        os.replace(temporary_alias, alias)
        _fsync_directory(alias.parent)
    finally:
        if temporary_alias.is_symlink():
            temporary_alias.unlink()


def _preflight_static_view_alias(alias: Path, target: Path, *, allowed_root: Path) -> None:
    alias = _validate_alias_path(alias, allowed_root=allowed_root, label="teacher rollout static view alias")
    expected_target = os.path.relpath(_lexical_absolute(target), alias.parent)
    if alias.is_symlink() and os.readlink(alias) == expected_target:
        return
    if not alias.exists() and not alias.is_symlink():
        return
    raise SystemExit(
        "[ERROR] OUTPUT_ROOT contains a legacy or foreign compatibility artifact that cannot be "
        "transactionally migrated in place. Use a fresh RUN_ID/OUTPUT_ROOT; --force cannot bypass "
        f"the single-alias publication contract. path={alias} expected_symlink_target={expected_target!r}"
    )


def _publish_directory_noreplace(staging: Path, generation: Path) -> bool:
    """Publish a directory iff its content-addressed destination is absent."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise SystemExit("[ERROR] Atomic generation publication requires renameat2().")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(staging), -100, os.fsencode(generation), 1)
    if result == 0:
        return True
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        return False
    raise OSError(error, os.strerror(error), str(generation))


def _atomic_switch_generated_alias(
    alias: Path,
    generation: Path,
    *,
    allowed_root: Path,
) -> None:
    """Atomically publish a complete generation, including directory migration."""

    alias = _validate_alias_path(alias, allowed_root=allowed_root, label="teacher rollout bank alias")
    generation = _validate_destructive_path(
        generation,
        allowed_root=allowed_root,
        label="teacher rollout bank generation",
    )
    generation_marker = generation / MARKER_NAME
    if not generation.is_dir() or not generation_marker.is_file() or generation_marker.is_symlink():
        raise SystemExit(f"[ERROR] Refusing to publish incomplete generation: {generation}")
    alias.parent.mkdir(parents=True, exist_ok=True)
    if alias.is_symlink() and alias.resolve() == generation:
        return
    if alias.exists() and not alias.is_symlink():
        legacy_marker = alias / MARKER_NAME
        if not alias.is_dir() or not legacy_marker.is_file() or legacy_marker.is_symlink():
            raise SystemExit(f"[ERROR] Refusing to replace non-generated publication alias: {alias}")

    temporary_alias = alias.with_name(f".{alias.name}.next-{os.getpid()}")
    if temporary_alias.exists() or temporary_alias.is_symlink():
        raise SystemExit(f"[ERROR] Temporary publication alias already exists: {temporary_alias}")
    temporary_alias.symlink_to(os.path.relpath(generation, alias.parent), target_is_directory=True)
    try:
        if not alias.exists() and not alias.is_symlink():
            os.replace(temporary_alias, alias)
        elif alias.is_symlink():
            os.replace(temporary_alias, alias)
        else:
            # Exchanging the complete legacy directory with the new symlink is
            # one namespace operation: active readers see either generation,
            # never a missing or partially rebuilt bank.
            retirement_root = _validate_destructive_path(
                allowed_root / "_retired_generated_aliases",
                allowed_root=allowed_root,
                label="retired generated-alias namespace",
            )
            retirement_root.mkdir(parents=True, exist_ok=True)
            alias_identity = hashlib.sha256(str(alias).encode("utf-8")).hexdigest()[:16]
            legacy_inode = int(alias.stat().st_ino)
            retirement_path = _validate_destructive_path(
                retirement_root / f"{alias_identity}-{alias.name}-legacy-{legacy_inode}-{os.getpid()}",
                allowed_root=allowed_root,
                label="retired generated alias",
            )
            if retirement_path.exists() or retirement_path.is_symlink():
                raise SystemExit(f"[ERROR] Retired generated-alias path already exists: {retirement_path}")
            libc = ctypes.CDLL(None, use_errno=True)
            renameat2 = getattr(libc, "renameat2", None)
            if renameat2 is None:
                raise SystemExit("[ERROR] Atomic legacy-directory migration requires renameat2().")
            renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
            renameat2.restype = ctypes.c_int
            result = renameat2(
                -100,
                os.fsencode(temporary_alias),
                -100,
                os.fsencode(alias),
                2,
            )
            if result != 0:
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error), str(alias))
            # The old generated directory now lives at temporary_alias.  Keep
            # it under an owned retirement namespace instead of deleting it:
            # a pre-existing reader may still hold a directory fd and must be
            # able to finish its current epoch against the old complete bank.
            os.replace(temporary_alias, retirement_path)
            _fsync_directory(retirement_root)
        _fsync_directory(alias.parent)
    finally:
        if temporary_alias.is_symlink():
            temporary_alias.unlink()


def _directory_content_manifest(
    path: Path,
    *,
    excluded_relative_paths: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for candidate in sorted(path.rglob("*")):
        relative_path = candidate.relative_to(path).as_posix()
        if relative_path in excluded_relative_paths:
            continue
        if candidate.is_symlink():
            raise SystemExit(f"[ERROR] Refusing symlink in immutable publication input: {candidate}")
        if not candidate.is_file():
            continue
        file_path = candidate
        identity = _file_content_identity(file_path)
        records.append(
            {
                "path": relative_path,
                **identity,
            }
        )
    return records


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pretty_json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _generation_manifest_from_publication_payload(
    publication_id: str,
    publication_payload: dict[str, Any],
) -> dict[str, Any]:
    summary_counts = publication_payload.get("summary_counts")
    all_rows = publication_payload.get("all_summary_rows")
    successful_object_map = publication_payload.get("successful_object_map")
    if (
        not isinstance(summary_counts, dict)
        or not isinstance(all_rows, list)
        or any(not isinstance(row, dict) for row in all_rows)
        or not isinstance(successful_object_map, dict)
    ):
        raise SystemExit("[ERROR] Teacher rollout publication payload cannot derive its generation manifest.")
    success_rows = [
        row
        for row in all_rows
        if str(row.get("success", "")).strip().lower() == "true"
    ]
    failure_rows = [
        row
        for row in all_rows
        if str(row.get("success", "")).strip().lower() != "true"
    ]
    category_counts = Counter(
        _category_for(str(clip_id), entry)
        for clip_id, entry in successful_object_map.items()
    )
    return {
        "publication_id": publication_id,
        "publication_payload": publication_payload,
        "source_bank": publication_payload.get("source_bank"),
        "contact_export_name": publication_payload.get("contact_export_name"),
        "teacher_lineage": publication_payload.get("teacher_lineage"),
        "total_rollout_rows": summary_counts.get("total"),
        "success_count": summary_counts.get("success"),
        "failure_count": summary_counts.get("failure"),
        "category_counts": dict(sorted(category_counts.items())),
        "save_visualization": publication_payload.get("save_visualization"),
        "save_visualization_preview_png": publication_payload.get("save_visualization_preview_png"),
        "save_visualization_face_heatmap_png": publication_payload.get("save_visualization_face_heatmap_png"),
        "success_status_counts": dict(
            sorted(Counter(row.get("status", "") for row in success_rows).items())
        ),
        "failure_status_counts": dict(
            sorted(Counter(row.get("status", "") for row in failure_rows).items())
        ),
    }


def _rebased_object_entry(entry: dict[str, Any], *, nested_motion_bank: bool) -> dict[str, Any]:
    rebased = dict(entry)

    def rebase_one(raw_value: Any, *, key: str) -> str:
        raw = str(raw_value).strip()
        if not raw:
            return raw
        path = Path(raw).expanduser()
        if path.is_absolute() or ".." in path.parts or raw.startswith(("package://", "http://", "https://", "file://")):
            raise SystemExit(f"[ERROR] Cannot rebase unsafe published object asset path {key}={raw!r}.")
        return (Path("..") / path).as_posix() if nested_motion_bank else path.as_posix()

    for key in _OBJECT_ASSET_PATH_KEYS:
        if key in rebased and str(rebased[key]).strip():
            rebased[key] = rebase_one(rebased[key], key=key)
    for key in _OBJECT_ASSET_PATH_LIST_KEYS:
        if key not in rebased:
            continue
        values = rebased[key]
        if not isinstance(values, list):
            raise SystemExit(f"[ERROR] Published object asset list {key} must be a JSON list.")
        rebased[key] = [rebase_one(value, key=key) for value in values]
    return rebased


def _deterministic_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for key in sorted(arrays):
            array_buffer = io.BytesIO()
            np.lib.format.write_array(array_buffer, np.asarray(arrays[key]), allow_pickle=False)
            member = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o600 << 16
            archive.writestr(
                member,
                array_buffer.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            )
    return output.getvalue()


def _normalized_rollout_motion_npz_bytes(
    source_path: Path,
    *,
    published_urdf_path: str,
    expected_source_identity: dict[str, Any],
) -> bytes:
    """Deterministically rewrite only the location-dependent URDF string."""
    raw, _digest = _stable_regular_file_bytes(source_path, label="teacher rollout motion normalization input")
    actual_source_identity = {"sha256": _digest, "size": len(raw)}
    if actual_source_identity != expected_source_identity:
        raise SystemExit(
            f"[ERROR] Teacher rollout motion changed before deterministic normalization: "
            f"path={source_path} actual={actual_source_identity!r} expected={expected_source_identity!r}"
        )
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as source:
            source_arrays = {key: np.asarray(source[key]).copy() for key in source.files}
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as exc:
        raise SystemExit(f"[ERROR] Cannot normalize rollout motion NPZ: {source_path}: {exc}") from exc
    if set(source_arrays) != _ROLLOUT_MOTION_KEYS:
        raise SystemExit(f"[ERROR] Cannot normalize unexpected rollout motion schema: {source_path}")
    normalized_arrays = dict(source_arrays)
    normalized_arrays["object_urdf_path"] = np.asarray(published_urdf_path)
    normalized_bytes = _deterministic_npz_bytes(normalized_arrays)
    try:
        with np.load(io.BytesIO(normalized_bytes), allow_pickle=False) as normalized:
            if set(normalized.files) != set(source_arrays):
                raise SystemExit(f"[ERROR] Normalized rollout motion keys changed: {source_path}")
            for key, source_array in source_arrays.items():
                normalized_array = np.asarray(normalized[key])
                if key == "object_urdf_path":
                    if _npz_scalar_string(normalized_array, field=key, path=source_path) != published_urdf_path:
                        raise SystemExit(f"[ERROR] Normalized rollout URDF metadata is incorrect: {source_path}")
                    continue
                if (
                    normalized_array.dtype != source_array.dtype
                    or normalized_array.shape != source_array.shape
                    or not np.array_equal(normalized_array, source_array)
                ):
                    raise SystemExit(
                        f"[ERROR] Rollout normalization changed non-URDF array {key}: {source_path}"
                    )
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as exc:
        raise SystemExit(f"[ERROR] Normalized rollout motion is unreadable: {source_path}: {exc}") from exc
    return normalized_bytes


def _bytes_identity(raw: bytes) -> dict[str, Any]:
    return {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}


def _validate_published_motion_bank_view(
    generation_root: Path,
    motion_bank_dir: Path,
    *,
    expected_map_payload: dict[str, Any],
    expected_motion_identities: dict[str, dict[str, Any]],
) -> None:
    """Check paths exactly as MotionLoader resolves them from each bank view."""
    map_path = motion_bank_dir / "_clip_object_urdf_map.json"
    map_metadata, clip_map = _load_clip_map(map_path)
    expected_metadata = {key: value for key, value in expected_map_payload.items() if key != "clips"}
    expected_clips = expected_map_payload.get("clips")
    if map_metadata != expected_metadata or clip_map != expected_clips or not isinstance(expected_clips, dict):
        raise SystemExit(f"[ERROR] Published motion-bank map differs from its expected view: {map_path}")
    if set(expected_motion_identities) != set(expected_clips):
        raise SystemExit(f"[ERROR] Published motion identity set differs from its object map: {motion_bank_dir}")

    generation_lexical = _lexical_absolute(generation_root)

    def require_asset(raw: str, *, clip_id: str, key: str) -> Path:
        candidate = _lexical_absolute(motion_bank_dir / raw)
        try:
            relative = candidate.relative_to(generation_lexical)
        except ValueError as exc:
            raise SystemExit(
                f"[ERROR] Published {key} escapes its immutable generation for {clip_id}: {raw}"
            ) from exc
        cursor = generation_lexical
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise SystemExit(f"[ERROR] Published {key} traverses a symlink for {clip_id}: {cursor}")
        if not candidate.is_file() or candidate.is_symlink():
            raise SystemExit(f"[ERROR] Published {key} is missing or symlinked for {clip_id}: {candidate}")
        identity = candidate.stat()
        if not stat.S_ISREG(identity.st_mode) or identity.st_nlink != 1:
            raise SystemExit(f"[ERROR] Published {key} is not a one-link regular file for {clip_id}: {candidate}")
        return candidate

    for clip_id, entry in sorted(expected_clips.items()):
        if not isinstance(entry, dict):
            raise SystemExit(f"[ERROR] Published object-map entry is invalid: {clip_id}")
        for key in _OBJECT_ASSET_PATH_KEYS:
            raw_value = str(entry.get(key, "")).strip()
            if raw_value:
                require_asset(raw_value, clip_id=clip_id, key=key)
        for key in _OBJECT_ASSET_PATH_LIST_KEYS:
            values = entry.get(key, [])
            if not isinstance(values, list):
                raise SystemExit(f"[ERROR] Published object asset list is invalid for {clip_id}: {key}")
            for raw_value in values:
                normalized_value = str(raw_value).strip()
                if normalized_value:
                    require_asset(normalized_value, clip_id=clip_id, key=key)

        motion_path = motion_bank_dir / f"{clip_id}.npz"
        if _file_content_identity(motion_path) != expected_motion_identities[clip_id]:
            raise SystemExit(f"[ERROR] Published rollout motion identity differs for {clip_id}: {motion_path}")
        motion_bytes, _motion_sha256 = _stable_regular_file_bytes(
            motion_path,
            label="published teacher rollout motion",
        )
        try:
            with np.load(io.BytesIO(motion_bytes), allow_pickle=False) as payload:
                embedded_urdf = _npz_scalar_string(
                    np.asarray(payload["object_urdf_path"]),
                    field="object_urdf_path",
                    path=motion_path,
                )
        except (EOFError, KeyError, OSError, ValueError, zipfile.BadZipFile) as exc:
            raise SystemExit(f"[ERROR] Published rollout motion metadata is unreadable: {motion_path}: {exc}") from exc
        map_urdf = str(entry.get("object_urdf_path", "")).strip()
        if embedded_urdf != map_urdf:
            raise SystemExit(
                f"[ERROR] Published NPZ/map URDF paths differ for {clip_id}: "
                f"npz={embedded_urdf!r} map={map_urdf!r}"
            )
        require_asset(embedded_urdf, clip_id=clip_id, key="NPZ object_urdf_path")


def _fsync_tree(path: Path) -> None:
    for file_path in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        with file_path.open("rb") as stream:
            os.fsync(stream.fileno())
    for directory in sorted((candidate for candidate in path.rglob("*") if candidate.is_dir()), reverse=True):
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    root_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(root_fd)
    finally:
        os.close(root_fd)


def _validate_generation_integrity(
    generation: Path,
    *,
    publication_id: str,
    expected_manifest: dict[str, Any] | None = None,
    expected_content_files: list[dict[str, Any]] | None = None,
) -> None:
    if generation.name != publication_id:
        raise SystemExit(
            f"[ERROR] Teacher rollout generation path is not content-addressed by publication_id: {generation}"
        )
    manifest_path = generation / "realmesh_rollout_manifest.json"
    merge_manifest_path = generation / "merge_manifest.json"
    integrity_path = generation / INTEGRITY_MANIFEST_NAME
    marker_path = generation / MARKER_NAME
    if (
        not marker_path.is_file()
        or marker_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.is_symlink()
        or not merge_manifest_path.is_file()
        or merge_manifest_path.is_symlink()
        or not integrity_path.is_file()
        or integrity_path.is_symlink()
    ):
        raise SystemExit(f"[ERROR] Teacher rollout generation is incomplete: {generation}")
    try:
        manifest_raw, _manifest_sha256 = _stable_regular_file_bytes(
            manifest_path,
            label="teacher rollout generation manifest",
        )
        merge_manifest_raw, _merge_manifest_sha256 = _stable_regular_file_bytes(
            merge_manifest_path,
            label="teacher rollout merge manifest",
        )
        integrity_raw, _integrity_sha256 = _stable_regular_file_bytes(
            integrity_path,
            label="teacher rollout generation integrity manifest",
        )
        manifest = json.loads(manifest_raw.decode("utf-8"))
        merge_manifest = json.loads(merge_manifest_raw.decode("utf-8"))
        integrity = json.loads(integrity_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"[ERROR] Teacher rollout generation metadata is invalid: {generation}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("publication_id") != publication_id:
        raise SystemExit(f"[ERROR] Teacher rollout generation manifest identity mismatch: {generation}")
    publication_payload = manifest.get("publication_payload")
    if not isinstance(publication_payload, dict) or _canonical_json_sha256(publication_payload) != publication_id:
        raise SystemExit(f"[ERROR] Teacher rollout generation payload digest mismatch: {generation}")
    derived_manifest = _generation_manifest_from_publication_payload(publication_id, publication_payload)
    if manifest != derived_manifest or manifest_raw != _pretty_json_bytes(derived_manifest):
        raise SystemExit(
            f"[ERROR] Teacher rollout generation manifest is not the canonical payload-derived document: {generation}"
        )
    if expected_manifest is not None and manifest != expected_manifest:
        raise SystemExit(f"[ERROR] Existing teacher rollout generation does not match this publication: {generation}")

    bound_content_files = publication_payload.get("final_content_manifest")
    bound_content_sha256 = publication_payload.get("final_content_manifest_sha256")
    if (
        not isinstance(bound_content_files, list)
        or not isinstance(bound_content_sha256, str)
        or len(bound_content_sha256) != 64
        or any(character not in "0123456789abcdef" for character in bound_content_sha256)
        or _canonical_json_sha256(bound_content_files) != bound_content_sha256
    ):
        raise SystemExit(f"[ERROR] Teacher rollout publication payload has invalid final-content binding: {generation}")
    content_paths: list[str] = []
    for record in bound_content_files:
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "sha256", "size"}
            or not isinstance(record.get("path"), str)
            or not record["path"]
            or Path(record["path"]).is_absolute()
            or ".." in Path(record["path"]).parts
            or record["path"] in PUBLICATION_ROOT_METADATA_PATHS
            or not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in record["sha256"])
            or not isinstance(record.get("size"), int)
            or isinstance(record.get("size"), bool)
            or record["size"] < 0
        ):
            raise SystemExit(f"[ERROR] Teacher rollout publication payload has invalid content record: {record!r}")
        content_paths.append(record["path"])
    if content_paths != sorted(content_paths) or len(content_paths) != len(set(content_paths)):
        raise SystemExit(f"[ERROR] Teacher rollout publication content records are not unique and sorted: {generation}")

    expected_integrity = {
        "schema": "teacher_realmesh_rollout_generation_integrity_v2",
        "publication_id": publication_id,
        "final_content_manifest_sha256": bound_content_sha256,
        "files": bound_content_files,
    }
    if integrity != expected_integrity:
        raise SystemExit(
            f"[ERROR] Teacher rollout generation integrity metadata differs from its publication payload: {generation}"
        )
    if integrity_raw != _pretty_json_bytes(expected_integrity):
        raise SystemExit(f"[ERROR] Teacher rollout generation integrity metadata is not canonical: {generation}")
    actual_files = _directory_content_manifest(
        generation,
        excluded_relative_paths=PUBLICATION_ROOT_METADATA_PATHS,
    )
    if actual_files != bound_content_files:
        raise SystemExit(f"[ERROR] Teacher rollout generation content changed after publication: {generation}")
    if expected_content_files is not None and actual_files != expected_content_files:
        raise SystemExit(f"[ERROR] Existing teacher rollout generation content differs from this publication: {generation}")

    expected_merge_manifest = {
        **manifest,
        "output_generation": str(generation),
        "target_generation": str(generation),
        "compatibility_view": "OUTPUT_ROOT/current -> TARGET_BANK -> target_generation",
    }
    if merge_manifest != expected_merge_manifest:
        raise SystemExit(f"[ERROR] Teacher rollout merge manifest differs from its committed generation: {generation}")
    if merge_manifest_raw != _pretty_json_bytes(expected_merge_manifest):
        raise SystemExit(f"[ERROR] Teacher rollout merge manifest is not canonical: {generation}")


def _readonly_generation(path: Path) -> None:
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_symlink():
            continue
        os.chmod(child, 0o555 if child.is_dir() else 0o444)
    os.chmod(path, 0o555)


def _require_disjoint_paths(first: Path, second: Path, *, first_label: str, second_label: str) -> None:
    first_resolved = first.resolve(strict=False)
    second_resolved = second.resolve(strict=False)
    if (
        first_resolved == second_resolved
        or first_resolved in second_resolved.parents
        or second_resolved in first_resolved.parents
    ):
        raise SystemExit(
            f"[ERROR] Refusing overlapping {first_label}/{second_label}: "
            f"{first_resolved} <-> {second_resolved}"
        )


def _require_outside_root(path: Path, forbidden_root: Path, *, label: str, forbidden_label: str) -> None:
    path_resolved = path.resolve(strict=False)
    forbidden_resolved = forbidden_root.resolve(strict=False)
    if path_resolved == forbidden_resolved or forbidden_resolved in path_resolved.parents:
        raise SystemExit(
            f"[ERROR] Refusing {label} inside reserved {forbidden_label}: "
            f"{path_resolved} (reserved={forbidden_resolved})"
        )


def _require_lexically_outside_root(path: Path, forbidden_root: Path, *, label: str, forbidden_label: str) -> None:
    path_absolute = _lexical_absolute(path)
    forbidden_absolute = _lexical_absolute(forbidden_root)
    if path_absolute == forbidden_absolute or forbidden_absolute in path_absolute.parents:
        raise SystemExit(
            f"[ERROR] Refusing {label} inside reserved {forbidden_label}: "
            f"{path_absolute} (reserved={forbidden_absolute})"
        )


def _safe_remove_generated(
    path: Path,
    *,
    allowed_root: Path,
    label: str,
    force: bool,
    preserve_names: frozenset[str] = frozenset(),
) -> Path:
    path = _validate_destructive_path(path, allowed_root=allowed_root, label=label)
    if not path.exists():
        return path
    if not path.is_dir():
        raise SystemExit(f"[ERROR] Refusing to replace non-directory {label}: {path}")

    marker = path / MARKER_NAME
    marker_ok = marker.is_file() and not marker.is_symlink()
    unexpected = [child for child in path.iterdir() if child.name not in preserve_names]
    if unexpected and not marker_ok:
        force_note = " (--force cannot bypass the generated marker)" if force else ""
        raise SystemExit(f"[ERROR] Refusing to overwrite non-generated {label}: {path}{force_note}")
    # Immutable prepared snapshots are read-only between preparation and merge.
    # Marker authentication above is required before restoring owner write bits
    # for a same-scope, launch-lock-protected rebuild.
    for descendant in sorted(path.rglob("*"), reverse=True):
        if not descendant.is_symlink():
            os.chmod(descendant, 0o755 if descendant.is_dir() else 0o644)
    os.chmod(path, 0o755)
    for child in path.iterdir():
        if child.name in preserve_names:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    return path


def _copy_or_symlink(src: Path, dst: Path, *, symlink: bool, allowed_root: Path) -> None:
    dst = _validate_destructive_path(dst, allowed_root=allowed_root, label="generated copy target")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if symlink:
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
    elif src.is_dir():
        shutil.copytree(src, dst, symlinks=False)
    else:
        shutil.copy2(src, dst)


def _category_for(clip_id: str, entry: object) -> str:
    parts = [clip_id]
    if isinstance(entry, dict):
        for key in ("object_name", "object_urdf_path", "object_mesh_path", "object_category", "category", "object_type"):
            value = str(entry.get(key, "")).strip()
            if value:
                path = Path(value)
                parts.extend([path.name, path.stem] if key.endswith("_path") else [value])
    raw = " ".join(parts).lower().replace("-", "_")
    if "barrel" in raw:
        return "barrel"
    if "bin" in raw or "trash" in raw or "basket" in raw:
        return "bin"
    if "ball" in raw or "sphere" in raw:
        return "ball"
    if "box" in raw or "cube" in raw or "largebox" in raw:
        return "box"
    return "other"


def _parse_csv_set(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def _relative_top_level_dirs(entries: list[dict[str, Any]]) -> set[str]:
    dirs: set[str] = set()
    for entry in entries:
        for key in ("object_urdf_path", "object_mesh_path"):
            raw = str(entry.get(key, "")).strip()
            if not raw:
                continue
            path = Path(raw).expanduser()
            if path.is_absolute() or raw.startswith(("package://", "http://", "https://", "file://")):
                continue
            parts = path.parts
            if len(parts) > 1:
                dirs.add(parts[0])
    return dirs


def _validated_asset_snapshot_dirs(
    entries: dict[str, Any],
    *,
    source_bank: Path,
) -> set[str]:
    """Validate direct and transitive URDF assets before any shard is written."""

    top_dirs: set[str] = set()

    def resolve_relative(raw: str, *, clip_id: str, label: str, base_dir: Path) -> Path:
        raw = raw.strip()
        if not raw:
            raise SystemExit(f"[ERROR] Missing {label} for selected clip {clip_id}.")
        if raw.startswith(("package://", "http://", "https://", "file://")):
            raise SystemExit(f"[ERROR] External {label} is not allowed for selected clip {clip_id}: {raw}")
        relative = Path(raw).expanduser()
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit(f"[ERROR] {label} must stay inside the source bank for {clip_id}: {raw}")
        lexical = base_dir / relative
        try:
            lexical_relative = lexical.relative_to(source_bank)
        except ValueError as exc:
            raise SystemExit(f"[ERROR] {label} escapes the source bank for {clip_id}: {raw}") from exc
        cursor = source_bank
        for part in lexical_relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise SystemExit(f"[ERROR] Symlinked {label} is not allowed for {clip_id}: {cursor}")
        resolved = lexical.resolve()
        try:
            bank_relative = resolved.relative_to(source_bank)
        except ValueError as exc:
            raise SystemExit(f"[ERROR] {label} escapes the source bank for {clip_id}: {raw}") from exc
        if len(bank_relative.parts) < 2:
            raise SystemExit(f"[ERROR] {label} must be nested below one asset directory for {clip_id}: {raw}")
        if not resolved.is_file():
            raise SystemExit(f"[ERROR] Missing {label} for selected clip {clip_id}: {resolved}")
        top_dirs.add(bank_relative.parts[0])
        return resolved

    for clip_id, raw_entry in sorted(entries.items()):
        if not isinstance(raw_entry, dict):
            raise SystemExit(f"[ERROR] Selected object-map entry is not a mapping: {clip_id}")
        object_name = str(raw_entry.get("object_name", "")).strip()
        try:
            object_size = np.asarray(raw_entry.get("object_size"), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"[ERROR] Invalid object_size for selected clip {clip_id}.") from exc
        if (
            not object_name
            or object_size.shape != (3,)
            or not np.isfinite(object_size).all()
            or np.any(object_size <= 0)
        ):
            raise SystemExit(f"[ERROR] Selected clip has incomplete object_name/object_size metadata: {clip_id}")
        urdf_path = resolve_relative(
            str(raw_entry.get("object_urdf_path", "")),
            clip_id=clip_id,
            label="object_urdf_path",
            base_dir=source_bank,
        )
        for key in _OBJECT_ASSET_PATH_KEYS:
            if key == "object_urdf_path":
                continue
            raw_asset = str(raw_entry.get(key, "")).strip()
            if raw_asset:
                resolve_relative(raw_asset, clip_id=clip_id, label=key, base_dir=source_bank)
        for key in _OBJECT_ASSET_PATH_LIST_KEYS:
            values = raw_entry.get(key, [])
            if not isinstance(values, list):
                raise SystemExit(f"[ERROR] {key} must be a list for selected clip {clip_id}.")
            for raw_asset in values:
                resolve_relative(str(raw_asset), clip_id=clip_id, label=key, base_dir=source_bank)
        try:
            urdf_root = ET.parse(urdf_path).getroot()
        except Exception as exc:
            raise SystemExit(f"[ERROR] Unable to parse object URDF for {clip_id}: {urdf_path}: {exc}") from exc
        mesh_nodes = list(urdf_root.findall(".//mesh")) + list(urdf_root.findall(".//texture"))
        if not urdf_root.findall(".//mesh"):
            raise SystemExit(f"[ERROR] Object URDF has no mesh geometry for {clip_id}: {urdf_path}")
        for node in mesh_nodes:
            filename = str(node.get("filename", "")).strip()
            resolve_relative(
                filename,
                clip_id=clip_id,
                label=f"URDF {node.tag} filename",
                base_dir=urdf_path.parent,
            )
    return top_dirs


def _summary_row_success(row: dict[str, str]) -> bool:
    return str(row.get("success", "")).strip().lower() == "true"


def _infer_clip_id_from_dir_name(dir_name: str) -> str:
    normalized = dir_name.strip()
    prefix, separator, suffix = normalized.partition("_")
    if separator and prefix.isdecimal() and suffix.strip():
        return suffix.strip()
    return normalized


def _read_csv(path: Path) -> list[dict[str, str]]:
    raw, _digest = _stable_regular_file_bytes(path, label="teacher shard summary CSV")
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")))


def _parse_summary_success(row: dict[str, str], *, summary_path: Path) -> bool:
    raw = str(row.get("success", "")).strip().lower()
    if raw not in {"true", "false"}:
        raise SystemExit(f"[ERROR] Shard summary has a non-boolean success value in {summary_path}: {raw!r}")
    return raw == "true"


_ROLLOUT_MOTION_KEYS = frozenset(
    {
        "fps",
        "body_names",
        "joint_names",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "object_name",
        "object_urdf_path",
        "object_size",
        "object_pos_w",
        "object_quat_w",
        "object_lin_vel_w",
        "object_ang_vel_w",
    }
)
_ROLLOUT_QUATERNION_NORM_ATOL = 1.0e-3


def _npz_scalar_string(value: np.ndarray, *, field: str, path: Path) -> str:
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in {"U", "S"}:
        raise SystemExit(f"[ERROR] Rollout motion {field} must be one string scalar: {path}")
    item = array.reshape(-1)[0]
    decoded = item.decode("utf-8") if isinstance(item, bytes) else str(item)
    if not decoded.strip():
        raise SystemExit(f"[ERROR] Rollout motion {field} is empty: {path}")
    return decoded


def _npz_string_list(value: np.ndarray, *, field: str, path: Path) -> list[str]:
    array = np.asarray(value)
    if array.ndim != 1 or array.size == 0 or array.dtype.kind not in {"U", "S"}:
        raise SystemExit(f"[ERROR] Rollout motion {field} must be a nonempty 1-D string array: {path}")
    values = [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in array.tolist()]
    if any(not item.strip() for item in values) or len(set(values)) != len(values):
        raise SystemExit(f"[ERROR] Rollout motion {field} contains empty or duplicate names: {path}")
    return values


def _finite_numeric_array(value: np.ndarray, *, field: str, path: Path) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in {"f", "i", "u"}:
        raise SystemExit(f"[ERROR] Rollout motion {field} must be numeric: {path}")
    if not np.isfinite(array).all():
        raise SystemExit(f"[ERROR] Rollout motion {field} contains NaN or infinity: {path}")
    return array


def _require_unit_quaternion_array(value: np.ndarray, *, field: str, path: Path) -> None:
    array = np.asarray(value)
    if array.ndim < 1 or array.shape[-1] != 4:
        raise SystemExit(f"[ERROR] Rollout motion {field} must end in quaternion dimension 4: {path}")
    norms = np.linalg.norm(array.astype(np.float64, copy=False), axis=-1)
    deviations = np.abs(norms - 1.0)
    if not np.isfinite(norms).all() or np.any(deviations > _ROLLOUT_QUATERNION_NORM_ATOL):
        max_deviation = float(np.max(deviations)) if deviations.size else float("inf")
        raise SystemExit(
            f"[ERROR] Rollout motion {field} must contain unit quaternions within "
            f"atol={_ROLLOUT_QUATERNION_NORM_ATOL:g}: max_norm_deviation={max_deviation:g} path={path}"
        )


def _resolved_rollout_urdf(raw_path: str, *, shard_input_dir: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (shard_input_dir / candidate).resolve()


def _summary_float(row: dict[str, str], field: str, *, summary_path: Path) -> float:
    raw = str(row.get(field, "")).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise SystemExit(f"[ERROR] Shard summary has invalid {field} in {summary_path}: {raw!r}") from exc
    if not np.isfinite(value):
        raise SystemExit(f"[ERROR] Shard summary has non-finite {field} in {summary_path}: {raw!r}")
    return value


def _validate_rollout_motion_npz(
    path: Path,
    *,
    clip_id: str,
    expected_object_entry: dict[str, Any],
    shard_input_dir: Path,
    summary_row: dict[str, str],
    summary_path: Path,
) -> dict[str, Any]:
    """Validate one exported teacher trajectory from one stable no-follow byte snapshot."""
    raw, digest = _stable_regular_file_bytes(path, label="teacher rollout motion output")
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as payload:
            payload_keys = list(payload.files)
            if len(payload_keys) != len(set(payload_keys)) or set(payload_keys) != _ROLLOUT_MOTION_KEYS:
                raise SystemExit(
                    f"[ERROR] Rollout motion schema differs for {clip_id}: "
                    f"actual={sorted(payload_keys)!r} expected={sorted(_ROLLOUT_MOTION_KEYS)!r}"
                )
            arrays = {key: np.asarray(payload[key]) for key in payload_keys}
    except SystemExit:
        raise
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as exc:
        raise SystemExit(f"[ERROR] Rollout motion is not a valid no-pickle NPZ for {clip_id}: {path}: {exc}") from exc

    body_names = _npz_string_list(arrays["body_names"], field="body_names", path=path)
    joint_names = _npz_string_list(arrays["joint_names"], field="joint_names", path=path)
    object_name = _npz_scalar_string(arrays["object_name"], field="object_name", path=path)
    object_urdf_path = _npz_scalar_string(arrays["object_urdf_path"], field="object_urdf_path", path=path)
    fps = _finite_numeric_array(arrays["fps"], field="fps", path=path).reshape(-1)
    if fps.size != 1 or float(fps[0]) <= 0:
        raise SystemExit(f"[ERROR] Rollout motion fps must contain one positive value: {path}")

    numeric = {
        key: _finite_numeric_array(arrays[key], field=key, path=path)
        for key in _ROLLOUT_MOTION_KEYS
        if key
        not in {
            "body_names",
            "joint_names",
            "object_name",
            "object_urdf_path",
            "fps",
        }
    }
    joint_pos = numeric["joint_pos"]
    if joint_pos.ndim != 2 or joint_pos.shape[0] <= 0:
        raise SystemExit(f"[ERROR] Rollout motion joint_pos must have shape [T, D] with T>0: {path}")
    trajectory_length = int(joint_pos.shape[0])
    expected_shapes = {
        "joint_pos": (trajectory_length, len(joint_names) + 7),
        "joint_vel": (trajectory_length, len(joint_names) + 6),
        "body_pos_w": (trajectory_length, len(body_names), 3),
        "body_quat_w": (trajectory_length, len(body_names), 4),
        "body_lin_vel_w": (trajectory_length, len(body_names), 3),
        "body_ang_vel_w": (trajectory_length, len(body_names), 3),
        "object_pos_w": (trajectory_length, 3),
        "object_quat_w": (trajectory_length, 4),
        "object_lin_vel_w": (trajectory_length, 3),
        "object_ang_vel_w": (trajectory_length, 3),
        "object_size": (3,),
    }
    for field, expected_shape in expected_shapes.items():
        if numeric[field].shape != expected_shape:
            raise SystemExit(
                f"[ERROR] Rollout motion {field} shape differs for {clip_id}: "
                f"actual={numeric[field].shape!r} expected={expected_shape!r}"
            )
    _require_unit_quaternion_array(numeric["body_quat_w"], field="body_quat_w", path=path)
    _require_unit_quaternion_array(numeric["object_quat_w"], field="object_quat_w", path=path)

    expected_name = str(expected_object_entry.get("object_name", "")).strip()
    expected_urdf_raw = str(expected_object_entry.get("object_urdf_path", "")).strip()
    try:
        expected_size = np.asarray(expected_object_entry.get("object_size"), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"[ERROR] Prepared object_size is invalid for {clip_id}.") from exc
    if (
        not expected_name
        or not expected_urdf_raw
        or expected_size.shape != (3,)
        or not np.isfinite(expected_size).all()
        or np.any(expected_size <= 0)
    ):
        raise SystemExit(f"[ERROR] Prepared object metadata is incomplete or invalid for {clip_id}.")
    expected_urdf = _resolved_rollout_urdf(expected_urdf_raw, shard_input_dir=shard_input_dir)
    if object_name != expected_name:
        raise SystemExit(
            f"[ERROR] Rollout motion object_name differs for {clip_id}: "
            f"actual={object_name!r} expected={expected_name!r}"
        )
    if _resolved_rollout_urdf(object_urdf_path, shard_input_dir=shard_input_dir) != expected_urdf:
        raise SystemExit(f"[ERROR] Rollout motion object_urdf_path differs for {clip_id}.")
    if not np.allclose(numeric["object_size"], expected_size, rtol=0.0, atol=1e-6):
        raise SystemExit(f"[ERROR] Rollout motion object_size differs for {clip_id}.")

    summary_name = str(summary_row.get("object_name", "")).strip()
    summary_urdf = str(summary_row.get("object_urdf_path", "")).strip()
    summary_size = np.asarray(
        [
            _summary_float(summary_row, "primitive_extent_x", summary_path=summary_path),
            _summary_float(summary_row, "primitive_extent_y", summary_path=summary_path),
            _summary_float(summary_row, "primitive_extent_z", summary_path=summary_path),
        ],
        dtype=np.float64,
    )
    if summary_name != expected_name:
        raise SystemExit(f"[ERROR] Shard summary object_name differs for {clip_id}.")
    if not summary_urdf or _resolved_rollout_urdf(summary_urdf, shard_input_dir=shard_input_dir) != expected_urdf:
        raise SystemExit(f"[ERROR] Shard summary object_urdf_path differs for {clip_id}.")
    if not np.allclose(summary_size, expected_size, rtol=0.0, atol=1e-6):
        raise SystemExit(f"[ERROR] Shard summary object extents differ for {clip_id}.")
    num_steps = _summary_float(summary_row, "num_steps", summary_path=summary_path)
    if not num_steps.is_integer() or num_steps < trajectory_length:
        raise SystemExit(
            f"[ERROR] Shard summary num_steps is shorter than rollout motion for {clip_id}: "
            f"num_steps={num_steps!r} trajectory_length={trajectory_length}"
        )
    return {"sha256": digest, "size": len(raw)}


def _require_recorded_file_identity(path: Path, expected: Any, *, label: str) -> dict[str, Any]:
    if (
        not isinstance(expected, dict)
        or set(expected) != {"sha256", "size"}
        or not isinstance(expected.get("sha256"), str)
        or len(expected["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in expected["sha256"])
        or not isinstance(expected.get("size"), int)
        or expected["size"] < 0
    ):
        raise SystemExit(f"[ERROR] Shard output manifest has an invalid {label} identity.")
    actual = _file_content_identity(path)
    if actual != expected:
        raise SystemExit(
            f"[ERROR] {label} changed after the shard output manifest was committed: "
            f"path={path} actual={actual!r} expected={expected!r}"
        )
    return actual


def _load_prepared_manifest(
    path: Path,
    *,
    expected_sha256: str,
    source_bank: Path,
) -> tuple[Path, dict[str, Any], list[str], dict[str, list[str]]]:
    prepared_path = _validate_destructive_path(
        path,
        allowed_root=TEACHER_ROLLOUT_SHARD_ROOT,
        label="prepared teacher rollout manifest",
    )
    if not prepared_path.is_file() or prepared_path.is_symlink():
        raise SystemExit(f"[ERROR] Prepared shard manifest is missing or symlinked: {prepared_path}")
    if len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256):
        raise SystemExit("[ERROR] --prepared-manifest-sha256 must be 64 lowercase hexadecimal characters.")
    prepared_bytes, actual_sha256 = _stable_regular_file_bytes(
        prepared_path,
        label="prepared shard manifest",
    )
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            "[ERROR] Prepared shard manifest changed after launch preparation: "
            f"actual={actual_sha256} expected={expected_sha256}"
        )
    payload = json.loads(prepared_bytes.decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != "teacher_realmesh_rollout_prepared_shards_v2":
        raise SystemExit(f"[ERROR] Unsupported prepared shard manifest: {prepared_path}")
    if Path(str(payload.get("source_bank", ""))).expanduser().resolve() != source_bank:
        raise SystemExit("[ERROR] Prepared shard manifest source bank differs from --source-bank.")

    selected_ids = payload.get("selected_clip_ids")
    if (
        not isinstance(selected_ids, list)
        or not selected_ids
        or any(not isinstance(clip_id, str) or not clip_id.strip() for clip_id in selected_ids)
        or len(set(selected_ids)) != len(selected_ids)
        or payload.get("selected_clip_count") != len(selected_ids)
    ):
        raise SystemExit("[ERROR] Prepared shard manifest has an invalid selected clip set.")
    selected_object_map = payload.get("selected_object_map")
    if not isinstance(selected_object_map, dict) or set(selected_object_map) != set(selected_ids):
        raise SystemExit("[ERROR] Prepared shard manifest object map does not match its selected clip set.")

    raw_shards = payload.get("shards")
    if not isinstance(raw_shards, list) or payload.get("num_shards") != len(raw_shards) or not raw_shards:
        raise SystemExit("[ERROR] Prepared shard manifest has an invalid shard list.")
    expected_shards: dict[str, list[str]] = {}
    concatenated_ids: list[str] = []
    for expected_index, shard in enumerate(raw_shards):
        if not isinstance(shard, dict) or shard.get("shard_index") != expected_index:
            raise SystemExit("[ERROR] Prepared shard manifest indices are not contiguous and ordered.")
        shard_name = f"shard_{expected_index:02d}"
        raw_ids = shard.get("clip_ids")
        if (
            not isinstance(raw_ids, list)
            or any(not isinstance(clip_id, str) for clip_id in raw_ids)
            or shard.get("count") != len(raw_ids)
        ):
            raise SystemExit(f"[ERROR] Prepared shard manifest has invalid clip IDs for {shard_name}.")
        expected_shards[shard_name] = list(raw_ids)
        concatenated_ids.extend(raw_ids)
    if concatenated_ids != selected_ids:
        raise SystemExit("[ERROR] Prepared shard clip partitions do not exactly reproduce the selected clip order.")
    return prepared_path, payload, list(selected_ids), expected_shards


def _validate_prepared_snapshot(
    prepared_path: Path,
    payload: dict[str, Any],
    expected_shards: dict[str, list[str]],
) -> None:
    snapshot_root = prepared_path.parent
    selected_inputs = payload.get("selected_motion_inputs")
    selected_object_map = payload.get("selected_object_map")
    asset_inputs = payload.get("source_asset_inputs")
    if not isinstance(selected_inputs, dict) or not isinstance(selected_object_map, dict):
        raise SystemExit("[ERROR] Prepared manifest lacks authenticated motion/object inputs.")
    if not isinstance(asset_inputs, dict):
        raise SystemExit("[ERROR] Prepared manifest lacks authenticated asset inputs.")

    expected_top_level = {
        MARKER_NAME,
        "manifest.json",
        "_asset_snapshot",
        *expected_shards.keys(),
    }
    actual_top_level = {path.name for path in snapshot_root.iterdir()}
    if actual_top_level != expected_top_level:
        raise SystemExit(
            "[ERROR] Prepared snapshot has unexpected or missing top-level entries: "
            f"actual={sorted(actual_top_level)!r} expected={sorted(expected_top_level)!r}"
        )

    asset_snapshot = snapshot_root / "_asset_snapshot"
    if {path.name for path in asset_snapshot.iterdir()} != set(asset_inputs):
        raise SystemExit("[ERROR] Prepared asset snapshot directory set differs from its manifest.")
    for directory_name, expected_files in sorted(asset_inputs.items()):
        if Path(directory_name).name != directory_name or directory_name in {".", ".."}:
            raise SystemExit(f"[ERROR] Prepared manifest has an unsafe asset directory: {directory_name!r}")
        asset_directory = asset_snapshot / directory_name
        if _directory_content_manifest(asset_directory) != expected_files:
            raise SystemExit(f"[ERROR] Prepared asset snapshot changed after preparation: {asset_directory}")

    for shard_name, expected_ids in expected_shards.items():
        shard_dir = snapshot_root / shard_name
        if not shard_dir.is_dir() or shard_dir.is_symlink():
            raise SystemExit(f"[ERROR] Prepared shard input is missing or symlinked: {shard_dir}")
        expected_shard_entries = {
            "_clip_object_urdf_map.json",
            "clip_ids.txt",
            *(f"{clip_id}.npz" for clip_id in expected_ids),
            *asset_inputs.keys(),
        }
        if {path.name for path in shard_dir.iterdir()} != expected_shard_entries:
            raise SystemExit(f"[ERROR] Prepared shard has unexpected or missing inputs: {shard_dir}")
        actual_motion_names = {
            path.stem
            for path in shard_dir.glob("*.npz")
            if path.is_file() and not path.is_symlink()
        }
        if actual_motion_names != set(expected_ids):
            raise SystemExit(f"[ERROR] Prepared input motions differ from manifest for {shard_name}.")
        for clip_id in expected_ids:
            motion_path = shard_dir / f"{clip_id}.npz"
            actual_identity = _file_content_identity(motion_path)
            if actual_identity != selected_inputs.get(clip_id):
                raise SystemExit(f"[ERROR] Prepared input motion changed after preparation: {motion_path}")
        expected_clip_list = "\n".join(expected_ids) + ("\n" if expected_ids else "")
        clip_ids_path = shard_dir / "clip_ids.txt"
        if clip_ids_path.is_symlink() or clip_ids_path.read_text(encoding="utf-8") != expected_clip_list:
            raise SystemExit(f"[ERROR] Prepared clip_ids.txt changed after preparation: {clip_ids_path}")
        object_map_path = shard_dir / "_clip_object_urdf_map.json"
        if object_map_path.is_symlink():
            raise SystemExit(f"[ERROR] Prepared object map is symlinked: {object_map_path}")
        shard_metadata, shard_object_map = _load_clip_map(object_map_path)
        expected_object_map = {clip_id: selected_object_map[clip_id] for clip_id in expected_ids}
        if shard_metadata != payload.get("source_map_metadata") or shard_object_map != expected_object_map:
            raise SystemExit(f"[ERROR] Prepared shard object map changed after preparation: {object_map_path}")
        for asset_name in asset_inputs:
            link_path = shard_dir / asset_name
            expected_target = snapshot_root / "_asset_snapshot" / asset_name
            if not link_path.is_symlink() or link_path.resolve() != expected_target:
                raise SystemExit(f"[ERROR] Prepared shard asset view changed after preparation: {link_path}")

    for candidate in [snapshot_root, *(path for path in snapshot_root.rglob("*") if not path.is_symlink())]:
        if candidate.stat().st_mode & 0o222:
            raise SystemExit(f"[ERROR] Prepared shard snapshot is unexpectedly writable: {candidate}")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_entry_path(raw: str, *, base_dir: Path) -> Path:
    path = Path(str(raw).strip()).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _copy_clip_dir_with_bank_metadata(
    src: Path,
    dst: Path,
    *,
    object_entry: dict[str, Any] | None,
    generation_root: Path,
    motion_bank_dir: Path,
    expected_input_manifest: list[dict[str, Any]],
) -> None:
    shutil.copytree(src, dst, symlinks=False)
    copied_input_manifest = _directory_content_manifest(dst)
    if copied_input_manifest != expected_input_manifest:
        raise SystemExit(f"[ERROR] Teacher contact sidecar changed while copying: {src}")
    metadata_path = dst / "metadata.json"
    if not metadata_path.is_file():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    clip_id = str(metadata.get("clip_id") or _infer_clip_id_from_dir_name(dst.name))
    metadata["teacher_rollout_motion_bank_path"] = os.path.relpath(motion_bank_dir / f"{clip_id}.npz", dst)
    if object_entry:
        if object_entry.get("object_name"):
            metadata["object_name"] = str(object_entry["object_name"])
        if object_entry.get("object_size"):
            metadata["primitive_extents_xyz"] = object_entry["object_size"]
        raw_urdf = str(object_entry.get("object_urdf_path", "")).strip()
        if raw_urdf:
            resolved_urdf = _resolve_entry_path(raw_urdf, base_dir=generation_root)
            metadata["object_urdf_path"] = os.path.relpath(resolved_urdf, dst)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_object_frame_visualization(
    source_clip_dir: Path,
    vis_clip_dir: Path,
    *,
    save_preview_png: bool,
    save_face_heatmap_png: bool,
) -> None:
    from holosoma.export_teacher_box_contacts import (  # noqa: PLC0415
        _EXPORT_REGION_LABELS,
        _save_overlay_assets,
    )

    metadata_path = source_clip_dir / "metadata.json"
    if not metadata_path.is_file():
        return
    vis_clip_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    object_urdf_path = str(metadata.get("object_urdf_path", "")).strip()
    if object_urdf_path and not Path(object_urdf_path).is_absolute():
        object_urdf_path = str((source_clip_dir / object_urdf_path).resolve())
    primitive_points = np.load(source_clip_dir / "primitive_contact_points.npy")
    primitive_counts = np.load(source_clip_dir / "primitive_contact_point_counts.npy")
    region_points_by_label: dict[str, np.ndarray] = {}
    for label in _EXPORT_REGION_LABELS:
        points_path = source_clip_dir / f"{label}_contact_points.npy"
        region_points_by_label[label] = (
            np.load(points_path) if points_path.is_file() else np.zeros((0, 3), dtype=np.float32)
        )
    _save_overlay_assets(
        vis_clip_dir,
        clip_id=str(metadata["clip_id"]),
        object_name=str(metadata.get("object_name", "")),
        object_urdf_path=object_urdf_path,
        extents_xyz=np.asarray(metadata["primitive_extents_xyz"], dtype=np.float32),
        retained_points_xyz=primitive_points,
        retained_counts=primitive_counts,
        display_points_xyz=primitive_points,
        display_point_labels=[_EXPORT_REGION_LABELS[0]] * int(primitive_points.shape[0]),
        region_points_by_label=region_points_by_label,
        save_glb=True,
        save_preview_png=save_preview_png,
        save_face_heatmap_png=save_face_heatmap_png,
    )


def prepare_shards(args: argparse.Namespace) -> None:
    dry_run = bool(getattr(args, "dry_run", False))
    try:
        rollout_contract = json.loads(str(getattr(args, "rollout_contract_json", "{}")))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[ERROR] Invalid --rollout-contract-json: {exc}") from exc
    if not isinstance(rollout_contract, dict):
        raise SystemExit("[ERROR] --rollout-contract-json must encode a JSON object.")
    source_bank_input = args.source_bank.expanduser()
    source_map_input = args.source_map.expanduser()
    if source_bank_input.is_symlink() or source_map_input.is_symlink():
        raise SystemExit("[ERROR] Source bank and object map must not be symlink aliases.")
    source_bank = source_bank_input.resolve()
    source_map = source_map_input.resolve()
    shard_root = _validate_destructive_path(
        args.shard_root,
        allowed_root=TEACHER_ROLLOUT_SHARD_ROOT,
        label="teacher rollout shard root",
    )
    _require_disjoint_paths(
        shard_root,
        source_bank,
        first_label="teacher rollout shard root",
        second_label="source bank",
    )
    allowed = _parse_csv_set(args.allowed_categories)
    excluded = _parse_csv_set(args.exclude_clips)

    if not source_bank.is_dir():
        raise SystemExit(f"[ERROR] Missing source bank: {source_bank}")
    if not source_map.is_file():
        raise SystemExit(f"[ERROR] Missing source object map: {source_map}")
    npz_files = sorted(source_bank.glob("*.npz"))
    if args.expected_total and len(npz_files) != args.expected_total:
        raise SystemExit(f"[ERROR] Expected {args.expected_total} source npz files, found {len(npz_files)}")

    source_map_bytes, source_map_sha256 = _stable_regular_file_bytes(source_map, label="source object map")
    metadata, clips = _parse_clip_map_bytes(source_map_bytes, path=source_map)
    if args.expected_total and len(clips) != args.expected_total:
        raise SystemExit(f"[ERROR] Expected {args.expected_total} object-map entries, found {len(clips)}")

    selected_ids: list[str] = []
    skipped_by_category: Counter[str] = Counter()
    for npz_path in npz_files:
        clip_id = npz_path.stem
        entry = clips.get(clip_id)
        if entry is None:
            raise SystemExit(f"[ERROR] Missing object-map entry for {clip_id}")
        category = _category_for(clip_id, entry)
        if allowed and category not in allowed:
            skipped_by_category[category] += 1
            continue
        if clip_id in excluded:
            continue
        selected_ids.append(clip_id)

    if not selected_ids:
        raise SystemExit("[ERROR] No clips selected for realmesh rollout.")
    if args.num_shards <= 0 or args.num_shards > len(selected_ids):
        raise SystemExit(
            f"[ERROR] num_shards must be in [1, {len(selected_ids)}] for the selected rollout clips; "
            f"got {args.num_shards}."
        )

    base, rem = divmod(len(selected_ids), args.num_shards)
    shard_counts = [base + (1 if idx < rem else 0) for idx in range(args.num_shards)]
    per_gpu_envs = int(args.per_gpu_envs)
    if per_gpu_envs <= 0:
        per_gpu_envs = max(shard_counts)
    if max(shard_counts) > per_gpu_envs:
        raise SystemExit(
            f"[ERROR] PER_GPU_ENVS={per_gpu_envs} is below largest shard count {max(shard_counts)}. "
            "Increase PER_GPU_ENVS or NUM_SHARDS."
        )

    selected_object_entries = {clip_id: clips[clip_id] for clip_id in selected_ids}
    selected_asset_dirs = _validated_asset_snapshot_dirs(selected_object_entries, source_bank=source_bank)
    source_asset_inputs: dict[str, list[dict[str, Any]]] = {}
    for directory_name in sorted(selected_asset_dirs):
        source_directory = source_bank / directory_name
        if not source_directory.is_dir() or source_directory.is_symlink():
            raise SystemExit(f"[ERROR] Selected object asset directory is missing or symlinked: {source_directory}")
        source_asset_inputs[directory_name] = _directory_content_manifest(source_directory)

    offset = 0
    manifest: dict[str, Any] = {
        "schema": "teacher_realmesh_rollout_prepared_shards_v2",
        "source_bank": str(source_bank),
        "source_map": str(source_map),
        "source_map_sha256": source_map_sha256,
        "source_map_metadata": metadata,
        "allowed_categories": sorted(allowed),
        "excluded_clips": sorted(excluded),
        "source_npz_count": len(npz_files),
        "selected_clip_count": len(selected_ids),
        "selected_clip_ids": selected_ids,
        "selected_object_map": {clip_id: clips[clip_id] for clip_id in selected_ids},
        "selected_motion_inputs": {
            clip_id: _file_content_identity(source_bank / f"{clip_id}.npz")
            for clip_id in selected_ids
        },
        "source_asset_inputs": source_asset_inputs,
        "rollout_contract": rollout_contract,
        "rollout_code_sha256": {
            "prepare_teacher_as_realmesh_rollout.py": _sha256_file(Path(__file__).resolve()),
            "launch_teacher_as_realmesh_rollout.sh": _sha256_file(REPO_ROOT / "scripts" / "launch_teacher_as_realmesh_rollout.sh"),
            "infer_teacher_as_contacts.sh": _sha256_file(REPO_ROOT / "infer_teacher_as_contacts.sh"),
            "export_teacher_box_contacts.py": _sha256_file(
                REPO_ROOT / "src" / "holosoma" / "holosoma" / "export_teacher_box_contacts.py"
            ),
        },
        "selected_category_counts": dict(sorted(Counter(_category_for(cid, clips[cid]) for cid in selected_ids).items())),
        "skipped_by_category": dict(sorted(skipped_by_category.items())),
        "num_shards": args.num_shards,
        "per_gpu_envs": per_gpu_envs,
        "shards": [],
    }
    if not dry_run:
        shard_root = _safe_remove_generated(
            shard_root,
            allowed_root=TEACHER_ROLLOUT_SHARD_ROOT,
            label="teacher rollout shard root",
            force=False,
        )
        shard_root.mkdir(parents=True, exist_ok=True)
        (shard_root / MARKER_NAME).write_text(
            "generated by prepare_teacher_as_realmesh_rollout.py prepare-shards\n",
            encoding="utf-8",
        )
        asset_snapshot_root = shard_root / "_asset_snapshot"
        asset_snapshot_root.mkdir()
        for directory_name in sorted(selected_asset_dirs):
            source_directory = source_bank / directory_name
            snapshot_directory = asset_snapshot_root / directory_name
            shutil.copytree(source_directory, snapshot_directory, symlinks=False)
            if _directory_content_manifest(snapshot_directory) != source_asset_inputs[directory_name]:
                raise SystemExit(f"[ERROR] Object assets changed while preparing shard snapshot: {source_directory}")
    for shard_idx, count in enumerate(shard_counts):
        shard_ids = selected_ids[offset : offset + count]
        offset += count
        shard_dir = shard_root / f"shard_{shard_idx:02d}"
        if not dry_run:
            shard_dir.mkdir(parents=True)
            for rel_dir in sorted(selected_asset_dirs):
                snapshot_dir = shard_root / "_asset_snapshot" / rel_dir
                if snapshot_dir.is_dir() and not (shard_dir / rel_dir).exists():
                    _copy_or_symlink(snapshot_dir, shard_dir / rel_dir, symlink=True, allowed_root=shard_dir)
            for clip_id in shard_ids:
                copied_motion = shard_dir / f"{clip_id}.npz"
                _copy_or_symlink(
                    source_bank / f"{clip_id}.npz",
                    copied_motion,
                    symlink=False,
                    allowed_root=shard_dir,
                )
                copied_identity = _file_content_identity(copied_motion)
                if copied_identity != manifest["selected_motion_inputs"][clip_id]:
                    raise SystemExit(f"[ERROR] Source motion changed while preparing shard snapshot: {clip_id}")
            shard_payload = dict(metadata)
            shard_payload["clips"] = {clip_id: clips[clip_id] for clip_id in shard_ids}
            _write_json(shard_dir / "_clip_object_urdf_map.json", shard_payload)
            (shard_dir / "clip_ids.txt").write_text(
                "\n".join(shard_ids) + ("\n" if shard_ids else ""),
                encoding="utf-8",
            )
        manifest["shards"].append(
            {
                "shard_index": shard_idx,
                "count": count,
                "motion_dir": str(shard_dir),
                "object_map": str(shard_dir / "_clip_object_urdf_map.json"),
                "clip_ids": shard_ids,
                "first_clip": shard_ids[0] if shard_ids else None,
                "last_clip": shard_ids[-1] if shard_ids else None,
            }
        )

    if not dry_run:
        _write_json(shard_root / "manifest.json", manifest)
        _fsync_tree(shard_root)
        _readonly_generation(shard_root)
        _fsync_tree(shard_root)
        _fsync_directory(shard_root.parent)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _merge_outputs_locked(args: argparse.Namespace) -> None:
    expected_checkpoint_sha256 = str(args.expected_teacher_checkpoint_sha256).strip()
    if len(expected_checkpoint_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_checkpoint_sha256
    ):
        raise SystemExit(
            "[ERROR] --expected-teacher-checkpoint-sha256 must be 64 lowercase hexadecimal characters."
        )
    expected_checkpoint_path = args.teacher_checkpoint_path.expanduser().resolve()
    if not expected_checkpoint_path.is_file():
        raise SystemExit(f"[ERROR] Immutable teacher checkpoint is missing: {expected_checkpoint_path}")
    actual_checkpoint_sha256 = _sha256_file(expected_checkpoint_path)
    if actual_checkpoint_sha256 != expected_checkpoint_sha256:
        raise SystemExit(
            "[ERROR] Immutable teacher checkpoint changed before merge: "
            f"actual={actual_checkpoint_sha256} expected={expected_checkpoint_sha256}"
        )
    output_root = _validate_destructive_path(
        args.output_root,
        allowed_root=TEACHER_ROLLOUT_OUTPUT_ROOT,
        label="teacher rollout output root",
    )
    target_alias = _validate_alias_path(
        args.target_bank,
        allowed_root=DATA_BANK_ROOT,
        label="teacher rollout target bank alias",
    )
    static_views = {
        "current": target_alias,
        "clips": output_root / "current" / "clips",
        "motion_bank": output_root / "current" / "motion_bank",
        "summary_all.csv": output_root / "current" / "summary_all.csv",
        "summary.csv": output_root / "current" / "summary.csv",
        "success_clips.txt": output_root / "current" / "success_clips.txt",
        "failure_clips.txt": output_root / "current" / "failure_clips.txt",
        "merge_manifest.json": output_root / "current" / "merge_manifest.json",
    }
    for view_name, view_target in static_views.items():
        _preflight_static_view_alias(
            output_root / view_name,
            view_target,
            allowed_root=TEACHER_ROLLOUT_OUTPUT_ROOT,
        )
    generation_root = _validate_destructive_path(
        DATA_BANK_ROOT / "_teacher_rollout_generations",
        allowed_root=DATA_BANK_ROOT,
        label="teacher rollout generation namespace",
    )
    _require_lexically_outside_root(
        target_alias,
        TEACHER_ROLLOUT_SHARD_ROOT,
        label="teacher rollout target bank alias",
        forbidden_label="teacher rollout shard namespace",
    )
    _require_lexically_outside_root(
        target_alias,
        generation_root,
        label="teacher rollout target bank alias",
        forbidden_label="teacher rollout generation namespace",
    )
    source_bank = args.source_bank.expanduser().resolve()
    if not source_bank.is_dir():
        raise SystemExit(f"[ERROR] Missing source bank: {source_bank}")
    prepared_manifest_path, prepared_manifest, selected_clip_ids, expected_shards = _load_prepared_manifest(
        args.prepared_manifest,
        expected_sha256=str(args.prepared_manifest_sha256).strip(),
        source_bank=source_bank,
    )
    _validate_prepared_snapshot(prepared_manifest_path, prepared_manifest, expected_shards)
    current_rollout_code_sha256 = {
        "prepare_teacher_as_realmesh_rollout.py": _sha256_file(Path(__file__).resolve()),
        "launch_teacher_as_realmesh_rollout.sh": _sha256_file(REPO_ROOT / "scripts" / "launch_teacher_as_realmesh_rollout.sh"),
        "infer_teacher_as_contacts.sh": _sha256_file(REPO_ROOT / "infer_teacher_as_contacts.sh"),
        "export_teacher_box_contacts.py": _sha256_file(
            REPO_ROOT / "src" / "holosoma" / "holosoma" / "export_teacher_box_contacts.py"
        ),
    }
    if prepared_manifest.get("rollout_code_sha256") != current_rollout_code_sha256:
        raise SystemExit("[ERROR] Teacher rollout/merge code changed after shard preparation.")
    _require_disjoint_paths(
        target_alias,
        source_bank,
        first_label="teacher rollout target bank alias",
        second_label="source bank",
    )
    contact_export_name = args.contact_export_name.strip() or DEFAULT_CONTACT_EXPORT_NAME
    if contact_export_name in {".", ".."} or Path(contact_export_name).name != contact_export_name:
        raise SystemExit(f"[ERROR] Contact export name must be one safe path component: {contact_export_name!r}")
    shards_root = output_root / "shards"
    if not shards_root.is_dir():
        raise SystemExit(f"[ERROR] Missing shard outputs root: {shards_root}")

    source_map_sha256 = str(prepared_manifest.get("source_map_sha256", ""))
    if len(source_map_sha256) != 64:
        raise SystemExit("[ERROR] Prepared shard manifest has no authenticated source object-map digest.")
    source_clip_map = prepared_manifest["selected_object_map"]
    source_map_metadata = prepared_manifest.get("source_map_metadata")
    if not isinstance(source_map_metadata, dict):
        raise SystemExit("[ERROR] Prepared shard manifest has invalid source object-map metadata.")
    current_exporter_identity = _file_content_identity(
        REPO_ROOT / "src" / "holosoma" / "holosoma" / "export_teacher_box_contacts.py"
    )

    all_rows: list[dict[str, str]] = []
    success_rows: list[dict[str, str]] = []
    failure_rows: list[dict[str, str]] = []
    clip_object_map: dict[str, Any] = {}
    success_clip_dirs: dict[str, Path] = {}
    success_motion_sources: dict[str, Path] = {}
    all_rollout_motion_sources: dict[str, Path] = {}
    all_rollout_motion_inputs: dict[str, dict[str, Any]] = {}
    seen_clip_ids: set[str] = set()
    teacher_lineage: dict[str, Any] | None = None
    shard_summary_payloads: dict[str, dict[str, Any]] = {}
    shard_output_manifest_payloads: dict[str, dict[str, Any]] = {}
    shard_entries = sorted(shards_root.iterdir())
    shard_outputs = [path for path in shard_entries if path.is_dir() and not path.is_symlink()]
    actual_shard_names = {path.name for path in shard_outputs}
    if actual_shard_names != set(expected_shards) or len(shard_entries) != len(expected_shards):
        raise SystemExit(
            "[ERROR] Rollout shard outputs do not exactly match the prepared manifest: "
            f"actual={sorted(actual_shard_names)!r} expected={sorted(expected_shards)!r}"
        )

    for shard_output in shard_outputs:
        expected_clip_ids = expected_shards[shard_output.name]
        expected_clip_set = set(expected_clip_ids)
        summary_csv = shard_output / "summary.csv"
        summary_json = shard_output / "summary.json"
        success_clips_path = shard_output / "success_clips.txt"
        failure_clips_path = shard_output / "failure_clips.txt"
        shard_output_manifest_path = shard_output / "shard_output_manifest.json"
        clips_src = shard_output / "clips"
        motion_src = shard_output / "motion_bank"
        if (
            not summary_csv.is_file()
            or summary_csv.is_symlink()
            or not summary_json.is_file()
            or summary_json.is_symlink()
            or not clips_src.is_dir()
            or clips_src.is_symlink()
            or not motion_src.is_dir()
            or motion_src.is_symlink()
            or not success_clips_path.is_file()
            or success_clips_path.is_symlink()
            or not failure_clips_path.is_file()
            or failure_clips_path.is_symlink()
            or not shard_output_manifest_path.is_file()
            or shard_output_manifest_path.is_symlink()
        ):
            raise SystemExit(f"[ERROR] Shard output is incomplete: {shard_output}")
        expected_output_entries = {
            "clips",
            "motion_bank",
            "summary.csv",
            "summary.json",
            "success_clips.txt",
            "failure_clips.txt",
            "shard_output_manifest.json",
        }
        if {path.name for path in shard_output.iterdir()} != expected_output_entries:
            raise SystemExit(f"[ERROR] {shard_output.name} has unexpected or missing top-level outputs.")

        shard_manifest_bytes, _shard_manifest_sha256 = _stable_regular_file_bytes(
            shard_output_manifest_path,
            label="teacher shard output manifest",
        )
        try:
            shard_output_manifest = json.loads(shard_manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SystemExit(f"[ERROR] Invalid shard output manifest: {shard_output_manifest_path}: {exc}") from exc
        expected_object_map_payload = {
            **source_map_metadata,
            "clips": {clip_id: source_clip_map[clip_id] for clip_id in expected_clip_ids},
        }
        if (
            not isinstance(shard_output_manifest, dict)
            or shard_output_manifest.get("schema") != "teacher_realmesh_rollout_shard_output_v1"
            or shard_output_manifest.get("shard_name") != shard_output.name
            or shard_output_manifest.get("prepared_manifest_sha256") != str(args.prepared_manifest_sha256)
            or shard_output_manifest.get("teacher_checkpoint_sha256") != expected_checkpoint_sha256
            or shard_output_manifest.get("expected_clip_ids") != expected_clip_ids
            or shard_output_manifest.get("object_map_payload") != expected_object_map_payload
            or shard_output_manifest.get("exporter_code") != current_exporter_identity
        ):
            raise SystemExit(f"[ERROR] {shard_output.name} output manifest differs from its launch contract.")
        _require_recorded_file_identity(
            summary_csv,
            shard_output_manifest.get("summary_csv"),
            label="teacher shard summary CSV",
        )
        _require_recorded_file_identity(
            summary_json,
            shard_output_manifest.get("summary_json"),
            label="teacher shard summary JSON",
        )
        _require_recorded_file_identity(
            success_clips_path,
            shard_output_manifest.get("success_clips"),
            label="teacher shard success clip list",
        )
        _require_recorded_file_identity(
            failure_clips_path,
            shard_output_manifest.get("failure_clips"),
            label="teacher shard failure clip list",
        )
        shard_output_manifest_payloads[shard_output.name] = shard_output_manifest

        shard_summary_bytes, _shard_summary_sha256 = _stable_regular_file_bytes(
            summary_json,
            label="teacher shard summary",
        )
        shard_summary_payload = json.loads(shard_summary_bytes.decode("utf-8"))
        if not isinstance(shard_summary_payload, dict):
            raise SystemExit(f"[ERROR] Shard summary must be a JSON object: {summary_json}")
        shard_summary_payloads[shard_output.name] = shard_summary_payload

        shard_lineage = _teacher_lineage_from_summary(
            summary_json,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_checkpoint_path=expected_checkpoint_path,
            payload=shard_summary_payload,
        )
        if teacher_lineage is None:
            teacher_lineage = shard_lineage
        elif shard_lineage != teacher_lineage:
            raise SystemExit(
                "[ERROR] Teacher/export lineage differs across rollout shards: "
                f"first={teacher_lineage!r} shard={shard_output.name!r} actual={shard_lineage!r}"
            )

        shard_rows = _read_csv(summary_csv)
        row_clip_ids = [str(row.get("clip_id", "")).strip() for row in shard_rows]
        if (
            len(shard_rows) != len(expected_clip_ids)
            or len(set(row_clip_ids)) != len(row_clip_ids)
            or set(row_clip_ids) != expected_clip_set
        ):
            raise SystemExit(
                f"[ERROR] {shard_output.name} summary rows do not exactly match its prepared clip set."
            )
        row_success = [_parse_summary_success(row, summary_path=summary_csv) for row in shard_rows]
        expected_success_count = sum(row_success)
        if (
            shard_summary_payload.get("num_clips") != len(expected_clip_ids)
            or shard_summary_payload.get("num_success") != expected_success_count
            or shard_summary_payload.get("num_failure") != len(expected_clip_ids) - expected_success_count
            or shard_summary_payload.get("num_envs") != prepared_manifest.get("per_gpu_envs")
        ):
            raise SystemExit(f"[ERROR] {shard_output.name} summary JSON counts disagree with summary.csv.")
        expected_success_text = "".join(
            f"{clip_id}\n" for clip_id, succeeded in zip(row_clip_ids, row_success, strict=True) if succeeded
        )
        expected_failure_text = "".join(
            f"{clip_id}\n" for clip_id, succeeded in zip(row_clip_ids, row_success, strict=True) if not succeeded
        )
        success_list_bytes, _success_list_sha256 = _stable_regular_file_bytes(
            success_clips_path,
            label="teacher shard success clip list",
        )
        failure_list_bytes, _failure_list_sha256 = _stable_regular_file_bytes(
            failure_clips_path,
            label="teacher shard failure clip list",
        )
        if success_list_bytes.decode("utf-8") != expected_success_text:
            raise SystemExit(f"[ERROR] {shard_output.name} success_clips.txt disagrees with summary.csv.")
        if failure_list_bytes.decode("utf-8") != expected_failure_text:
            raise SystemExit(f"[ERROR] {shard_output.name} failure_clips.txt disagrees with summary.csv.")

        source_map = motion_src / "_clip_object_urdf_map.json"
        if not source_map.is_file() or source_map.is_symlink():
            raise SystemExit(f"[ERROR] Shard motion object map is missing or symlinked: {source_map}")
        _require_recorded_file_identity(
            source_map,
            shard_output_manifest.get("object_map_output"),
            label="teacher shard object map",
        )
        shard_map_metadata, shard_map = _load_clip_map(source_map)
        expected_shard_map = {clip_id: source_clip_map[clip_id] for clip_id in expected_clip_ids}
        if shard_map_metadata != source_map_metadata or shard_map != expected_shard_map:
            raise SystemExit(
                f"[ERROR] {shard_output.name} motion object map is not exactly equal to its prepared subset."
            )
        expected_motion_entries = {"_clip_object_urdf_map.json", *(f"{clip_id}.npz" for clip_id in expected_clip_ids)}
        if {path.name for path in motion_src.iterdir()} != expected_motion_entries:
            raise SystemExit(f"[ERROR] {shard_output.name} motion directory has unexpected or missing entries.")
        motion_clip_ids = {path.stem for path in motion_src.glob("*.npz") if path.is_file() and not path.is_symlink()}
        if motion_clip_ids != expected_clip_set:
            raise SystemExit(f"[ERROR] {shard_output.name} motion files differ from its prepared clip set.")
        recorded_motion_outputs = shard_output_manifest.get("motion_outputs")
        if not isinstance(recorded_motion_outputs, dict) or set(recorded_motion_outputs) != expected_clip_set:
            raise SystemExit(f"[ERROR] {shard_output.name} output manifest has an invalid rollout motion set.")
        rows_by_clip_id = {str(row["clip_id"]).strip(): row for row in shard_rows}
        shard_input_dir = prepared_manifest_path.parent / shard_output.name
        for clip_id in expected_clip_ids:
            motion_path = motion_src / f"{clip_id}.npz"
            recorded_identity = _require_recorded_file_identity(
                motion_path,
                recorded_motion_outputs.get(clip_id),
                label=f"teacher rollout motion {clip_id}",
            )
            validated_identity = _validate_rollout_motion_npz(
                motion_path,
                clip_id=clip_id,
                expected_object_entry=expected_shard_map[clip_id],
                shard_input_dir=shard_input_dir,
                summary_row=rows_by_clip_id[clip_id],
                summary_path=summary_csv,
            )
            if validated_identity != recorded_identity:
                raise SystemExit(f"[ERROR] Teacher rollout motion changed during merge validation: {motion_path}")
            all_rollout_motion_sources[clip_id] = motion_path
            all_rollout_motion_inputs[clip_id] = validated_identity
        clip_dir_matches: dict[str, list[Path]] = {clip_id: [] for clip_id in expected_clip_ids}
        unexpected_clip_dirs: list[str] = []
        for path in clips_src.iterdir():
            if not path.is_dir() or path.is_symlink():
                unexpected_clip_dirs.append(path.name)
                continue
            inferred_id = _infer_clip_id_from_dir_name(path.name)
            if path.name in expected_clip_set:
                inferred_id = path.name
            if inferred_id not in clip_dir_matches:
                unexpected_clip_dirs.append(path.name)
                continue
            clip_dir_matches[inferred_id].append(path)
        if unexpected_clip_dirs or any(len(matches) != 1 for matches in clip_dir_matches.values()):
            raise SystemExit(
                f"[ERROR] {shard_output.name} clip directories differ from its prepared clip set: "
                f"unexpected={unexpected_clip_dirs!r}"
            )
        recorded_clip_outputs = shard_output_manifest.get("clip_outputs")
        if not isinstance(recorded_clip_outputs, dict) or set(recorded_clip_outputs) != expected_clip_set:
            raise SystemExit(f"[ERROR] {shard_output.name} output manifest has an invalid contact clip set.")
        for clip_id in expected_clip_ids:
            recorded_clip = recorded_clip_outputs.get(clip_id)
            clip_directory = clip_dir_matches[clip_id][0]
            if (
                not isinstance(recorded_clip, dict)
                or set(recorded_clip) != {"directory_name", "files"}
                or recorded_clip.get("directory_name") != clip_directory.name
                or not isinstance(recorded_clip.get("files"), list)
                or _directory_content_manifest(clip_directory) != recorded_clip["files"]
            ):
                raise SystemExit(
                    f"[ERROR] Teacher rollout contact output changed after its shard manifest was committed: "
                    f"{clip_directory}"
                )
        for row in shard_rows:
            clip_id = str(row.get("clip_id", "")).strip()
            if not clip_id:
                raise SystemExit(f"[ERROR] Shard summary contains an empty clip_id: {summary_csv}")
            if clip_id in seen_clip_ids:
                raise SystemExit(f"[ERROR] Clip appears more than once across rollout summaries: {clip_id}")
            seen_clip_ids.add(clip_id)
            all_rows.append(row)
            if not _parse_summary_success(row, summary_path=summary_csv):
                failure_rows.append(row)
                continue

            if clip_id not in shard_map:
                raise SystemExit(f"[ERROR] Successful clip is missing from shard object map: {clip_id}")
            source_entry = source_clip_map.get(clip_id)
            shard_entry = shard_map[clip_id]
            if source_entry is None or source_entry != shard_entry:
                raise SystemExit(f"[ERROR] Source/shard object metadata differs for successful clip: {clip_id}")
            object_entry = source_entry
            if not isinstance(object_entry, dict):
                raise SystemExit(f"[ERROR] Successful clip has a non-mapping object entry: {clip_id}")

            matches = clip_dir_matches[clip_id]
            if len(matches) != 1:
                raise SystemExit(f"[ERROR] Expected one clip dir for {clip_id}, found {len(matches)} in {clips_src}")
            motion_path = motion_src / f"{clip_id}.npz"
            if not motion_path.is_file() or motion_path.is_symlink():
                raise SystemExit(f"[ERROR] Successful clip motion is missing or symlinked: {motion_path}")
            if matches[0].name in {path.name for path in success_clip_dirs.values()}:
                raise SystemExit(f"[ERROR] Successful clip directories collide by name: {matches[0].name}")

            success_rows.append(row)
            success_clip_dirs[clip_id] = matches[0]
            success_motion_sources[clip_id] = motion_path
            clip_object_map[clip_id] = object_entry

    if teacher_lineage is None:
        raise SystemExit(f"[ERROR] No complete shard outputs found under {shards_root}")
    if seen_clip_ids != set(selected_clip_ids):
        raise SystemExit("[ERROR] Merged shard summaries do not cover the complete prepared clip set.")
    if set(all_rollout_motion_inputs) != set(selected_clip_ids):
        raise SystemExit("[ERROR] Validated rollout motions do not cover the complete prepared clip set.")
    if not success_rows:
        raise SystemExit("[ERROR] Refusing to publish an empty successful teacher rollout bank.")

    # A published generation must be self-contained.  Absolute/external object
    # paths would silently leave the supposedly immutable bank dependent on a
    # mutable location, so only verified in-bank relative assets are accepted.
    asset_top_dirs: set[str] = set()
    prepared_asset_root = _validate_destructive_path(
        prepared_manifest_path.parent / "_asset_snapshot",
        allowed_root=TEACHER_ROLLOUT_SHARD_ROOT,
        label="prepared teacher rollout asset snapshot",
    )
    if not prepared_asset_root.is_dir() or prepared_asset_root.is_symlink():
        raise SystemExit(f"[ERROR] Prepared object asset snapshot is missing or symlinked: {prepared_asset_root}")
    for clip_id, entry in sorted(clip_object_map.items()):
        asset_values: list[tuple[str, str]] = []
        for key in _OBJECT_ASSET_PATH_KEYS:
            asset_values.append((key, str(entry.get(key, "")).strip()))
        for key in _OBJECT_ASSET_PATH_LIST_KEYS:
            values = entry.get(key, [])
            if not isinstance(values, list):
                raise SystemExit(f"[ERROR] {key} must be a list for published clip {clip_id}.")
            asset_values.extend((key, str(value).strip()) for value in values)
        for key, raw_path in asset_values:
            if not raw_path:
                continue
            if raw_path.startswith(("package://", "http://", "https://", "file://")):
                raise SystemExit(f"[ERROR] External {key} is not publishable for {clip_id}: {raw_path}")
            relative_path = Path(raw_path).expanduser()
            if relative_path.is_absolute() or len(relative_path.parts) < 2 or ".." in relative_path.parts:
                raise SystemExit(f"[ERROR] {key} must be a nested source-bank-relative path for {clip_id}: {raw_path}")
            lexical_asset = prepared_asset_root / relative_path
            if lexical_asset.is_symlink():
                raise SystemExit(f"[ERROR] Object asset is symlinked for {clip_id}: {lexical_asset}")
            resolved_asset = lexical_asset.resolve()
            try:
                resolved_asset.relative_to(prepared_asset_root)
            except ValueError as exc:
                raise SystemExit(f"[ERROR] Object asset escapes source bank for {clip_id}: {raw_path}") from exc
            if not resolved_asset.is_file():
                raise SystemExit(f"[ERROR] Object asset is missing for {clip_id}: {resolved_asset}")
            asset_top_dirs.add(relative_path.parts[0])

    asset_sources: dict[str, Path] = {}
    prepared_asset_inputs = prepared_manifest.get("source_asset_inputs")
    if not isinstance(prepared_asset_inputs, dict):
        raise SystemExit("[ERROR] Prepared shard manifest has no authenticated object asset snapshot.")
    if not asset_top_dirs.issubset(set(prepared_asset_inputs)):
        raise SystemExit("[ERROR] Successful object assets are absent from the prepared asset snapshot set.")
    # Preserve the complete authenticated rollout asset snapshot, including
    # objects whose teacher rollout failed and therefore has no published clip.
    asset_top_dirs = set(prepared_asset_inputs)
    for directory_name in sorted(asset_top_dirs):
        source_directory = prepared_asset_root / directory_name
        if not source_directory.is_dir() or source_directory.is_symlink():
            raise SystemExit(f"[ERROR] Object asset directory is missing or symlinked: {source_directory}")
        if _directory_content_manifest(source_directory) != prepared_asset_inputs[directory_name]:
            raise SystemExit(f"[ERROR] Prepared object asset snapshot changed after rollout: {source_directory}")
        asset_sources[directory_name] = source_directory

    for clip_id, source_path in sorted(all_rollout_motion_sources.items()):
        if _file_content_identity(source_path) != all_rollout_motion_inputs[clip_id]:
            raise SystemExit(f"[ERROR] Teacher rollout motion changed after semantic validation: {source_path}")
    motion_inputs = {
        clip_id: all_rollout_motion_inputs[clip_id]
        for clip_id in sorted(success_motion_sources)
    }
    contact_inputs = {
        clip_id: {
            "directory_name": path.name,
            "files": _directory_content_manifest(path),
        }
        for clip_id, path in sorted(success_clip_dirs.items())
    }
    asset_inputs = {
        directory_name: _directory_content_manifest(path)
        for directory_name, path in sorted(asset_sources.items())
    }
    root_object_map_payload = {
        **source_map_metadata,
        "clips": {
            clip_id: _rebased_object_entry(entry, nested_motion_bank=False)
            for clip_id, entry in sorted(clip_object_map.items())
        },
    }
    nested_object_map_payload = {
        **source_map_metadata,
        "clips": {
            clip_id: _rebased_object_entry(entry, nested_motion_bank=True)
            for clip_id, entry in sorted(clip_object_map.items())
        },
    }
    normalized_motion_identities: dict[str, dict[str, dict[str, Any]]] = {
        "root": {},
        "nested": {},
    }
    for clip_id, source_path in sorted(success_motion_sources.items()):
        root_urdf = str(root_object_map_payload["clips"][clip_id]["object_urdf_path"])
        nested_urdf = str(nested_object_map_payload["clips"][clip_id]["object_urdf_path"])
        normalized_motion_identities["root"][clip_id] = _bytes_identity(
            _normalized_rollout_motion_npz_bytes(
                source_path,
                published_urdf_path=root_urdf,
                expected_source_identity=motion_inputs[clip_id],
            )
        )
        normalized_motion_identities["nested"][clip_id] = _bytes_identity(
            _normalized_rollout_motion_npz_bytes(
                source_path,
                published_urdf_path=nested_urdf,
                expected_source_identity=motion_inputs[clip_id],
            )
        )
    full_teacher_lineage = {
        **teacher_lineage,
        "checkpoint_path": str(expected_checkpoint_path),
        "checkpoint_source": str(args.teacher_checkpoint_source),
    }
    publication_input_payload = {
        "schema": "teacher_realmesh_rollout_generation_v1",
        "source_bank": str(source_bank),
        "source_object_map_sha256": source_map_sha256,
        "prepared_manifest_sha256": str(args.prepared_manifest_sha256),
        "prepared_selected_clip_ids": selected_clip_ids,
        "prepared_shards": expected_shards,
        "prepared_rollout_contract": prepared_manifest.get("rollout_contract"),
        "publisher_code_sha256": current_rollout_code_sha256,
        "teacher_lineage": full_teacher_lineage,
        "contact_export_name": contact_export_name,
        "all_summary_rows": all_rows,
        "shard_summary_payloads": shard_summary_payloads,
        "shard_output_manifests": shard_output_manifest_payloads,
        "summary_counts": {
            "total": len(all_rows),
            "success": len(success_rows),
            "failure": len(failure_rows),
        },
        "successful_object_map": dict(sorted(clip_object_map.items())),
        "published_motion_object_maps": {
            "root": root_object_map_payload,
            "nested": nested_object_map_payload,
        },
        "motion_inputs": motion_inputs,
        "all_rollout_motion_inputs": dict(sorted(all_rollout_motion_inputs.items())),
        "normalized_motion_identities": normalized_motion_identities,
        "contact_inputs": contact_inputs,
        "asset_inputs": asset_inputs,
        "save_visualization": bool(args.save_visualization),
        "save_visualization_preview_png": bool(args.save_visualization_preview_png),
        "save_visualization_face_heatmap_png": bool(args.save_visualization_face_heatmap_png),
    }
    generation_root_preexisting = generation_root.exists()
    generation_root.mkdir(parents=True, exist_ok=True)
    if not generation_root_preexisting:
        _fsync_directory(DATA_BANK_ROOT)

    staging = _validate_destructive_path(
        generation_root
        / f".candidate-{_canonical_json_sha256(publication_input_payload)[:16]}.staging-{os.getpid()}",
        allowed_root=DATA_BANK_ROOT,
        label="teacher rollout bank staging generation",
    )
    if staging.exists() or staging.is_symlink():
        raise SystemExit(f"[ERROR] Teacher rollout staging generation already exists: {staging}")
    staging.mkdir(parents=True)
    (staging / MARKER_NAME).write_text(
        "generated by prepare_teacher_as_realmesh_rollout.py merge\n",
        encoding="utf-8",
    )
    expected_content_files: list[dict[str, Any]]
    publication_id: str
    generation: Path
    generation_manifest: dict[str, Any]
    merge_manifest: dict[str, Any]
    try:
        slot_bank = staging / "_single_slot_motion_bank"
        output_motion = staging / "motion_bank"
        output_clips = staging / "clips"
        slot_bank.mkdir()
        output_motion.mkdir()
        output_clips.mkdir()
        (output_motion / MARKER_NAME).write_text("generated success motion merge\n", encoding="utf-8")
        (output_clips / MARKER_NAME).write_text("generated success clip merge\n", encoding="utf-8")
        for clip_id, source_path in sorted(success_motion_sources.items()):
            root_motion_bytes = _normalized_rollout_motion_npz_bytes(
                source_path,
                published_urdf_path=str(root_object_map_payload["clips"][clip_id]["object_urdf_path"]),
                expected_source_identity=motion_inputs[clip_id],
            )
            nested_motion_bytes = _normalized_rollout_motion_npz_bytes(
                source_path,
                published_urdf_path=str(nested_object_map_payload["clips"][clip_id]["object_urdf_path"]),
                expected_source_identity=motion_inputs[clip_id],
            )
            if (
                _bytes_identity(root_motion_bytes) != normalized_motion_identities["root"][clip_id]
                or _bytes_identity(nested_motion_bytes) != normalized_motion_identities["nested"][clip_id]
            ):
                raise SystemExit(f"[ERROR] Deterministic rollout normalization changed for {clip_id}.")
            root_destination = staging / f"{clip_id}.npz"
            slot_destination = slot_bank / f"{clip_id}.npz"
            output_destination = output_motion / f"{clip_id}.npz"
            root_destination.write_bytes(root_motion_bytes)
            slot_destination.write_bytes(nested_motion_bytes)
            shutil.copy2(slot_destination, output_destination)
            if (
                _file_content_identity(root_destination) != normalized_motion_identities["root"][clip_id]
                or _file_content_identity(slot_destination) != normalized_motion_identities["nested"][clip_id]
                or _file_content_identity(output_destination) != normalized_motion_identities["nested"][clip_id]
            ):
                raise SystemExit(f"[ERROR] Normalized rollout motion changed while publishing: {clip_id}")
        _write_json(staging / "_clip_object_urdf_map.json", root_object_map_payload)
        _write_json(slot_bank / "_clip_object_urdf_map.json", nested_object_map_payload)
        _write_json(output_motion / "_clip_object_urdf_map.json", nested_object_map_payload)

        for directory_name, source_directory in sorted(asset_sources.items()):
            _copy_or_symlink(
                source_directory,
                staging / directory_name,
                symlink=False,
                allowed_root=staging,
            )
            if _directory_content_manifest(staging / directory_name) != asset_inputs[directory_name]:
                raise SystemExit(f"[ERROR] Object assets changed while copying: {source_directory}")

        _validate_published_motion_bank_view(
            staging,
            staging,
            expected_map_payload=root_object_map_payload,
            expected_motion_identities=normalized_motion_identities["root"],
        )
        _validate_published_motion_bank_view(
            staging,
            slot_bank,
            expected_map_payload=nested_object_map_payload,
            expected_motion_identities=normalized_motion_identities["nested"],
        )
        _validate_published_motion_bank_view(
            staging,
            output_motion,
            expected_map_payload=nested_object_map_payload,
            expected_motion_identities=normalized_motion_identities["nested"],
        )

        contact_root = staging / contact_export_name / "clips"
        contact_root.mkdir(parents=True)
        copied_contact_dirs: dict[str, Path] = {}
        for clip_id, source_clip_dir in sorted(success_clip_dirs.items()):
            copied_dir = contact_root / source_clip_dir.name
            _copy_clip_dir_with_bank_metadata(
                source_clip_dir,
                copied_dir,
                object_entry=clip_object_map[clip_id],
                generation_root=staging,
                motion_bank_dir=slot_bank,
                expected_input_manifest=contact_inputs[clip_id]["files"],
            )
            copied_contact_dirs[clip_id] = copied_dir

            output_clip_dir = output_clips / source_clip_dir.name
            normalized_contact_manifest = _directory_content_manifest(copied_dir)
            _copy_clip_dir_with_bank_metadata(
                copied_dir,
                output_clip_dir,
                object_entry=clip_object_map[clip_id],
                generation_root=staging,
                motion_bank_dir=output_motion,
                expected_input_manifest=normalized_contact_manifest,
            )

        if args.save_visualization:
            vis_root = staging / "object_frame_contact_vis" / "clips"
            for source_clip_dir in sorted(copied_contact_dirs.values()):
                _save_object_frame_visualization(
                    source_clip_dir,
                    vis_root / source_clip_dir.name,
                    save_preview_png=bool(args.save_visualization_preview_png),
                    save_face_heatmap_png=bool(args.save_visualization_face_heatmap_png),
                )

        _write_csv(staging / "summary_all.csv", all_rows)
        _write_csv(staging / "summary.csv", success_rows)
        _write_json(staging / "all_rollout_motion_inputs.json", dict(sorted(all_rollout_motion_inputs.items())))
        (staging / "success_clips.txt").write_text(
            "\n".join(row["clip_id"] for row in success_rows) + "\n",
            encoding="utf-8",
        )
        (staging / "failure_clips.txt").write_text(
            "\n".join(row["clip_id"] for row in failure_rows) + ("\n" if failure_rows else ""),
            encoding="utf-8",
        )

        # Root publication metadata is excluded from this list to avoid a hash
        # cycle: each excluded document contains publication_id.  Every other
        # final byte is bound into publication_payload, while the three root
        # documents are reconstructed and cross-checked by
        # _validate_generation_integrity().
        expected_content_files = _directory_content_manifest(
            staging,
            excluded_relative_paths=PUBLICATION_ROOT_METADATA_PATHS,
        )
        final_content_manifest_sha256 = _canonical_json_sha256(expected_content_files)
        publication_payload = {
            **publication_input_payload,
            "final_content_manifest_sha256": final_content_manifest_sha256,
            "final_content_manifest": expected_content_files,
        }
        publication_id = _canonical_json_sha256(publication_payload)
        generation = _validate_destructive_path(
            generation_root / publication_id,
            allowed_root=DATA_BANK_ROOT,
            label="teacher rollout bank generation",
        )
        generation_manifest = _generation_manifest_from_publication_payload(
            publication_id,
            publication_payload,
        )
        merge_manifest = {
            **generation_manifest,
            "output_generation": str(generation),
            "target_generation": str(generation),
            "compatibility_view": "OUTPUT_ROOT/current -> TARGET_BANK -> target_generation",
        }
        _write_json(staging / "realmesh_rollout_manifest.json", generation_manifest)
        _write_json(staging / "merge_manifest.json", merge_manifest)
        _write_json(
            staging / INTEGRITY_MANIFEST_NAME,
            {
                "schema": "teacher_realmesh_rollout_generation_integrity_v2",
                "publication_id": publication_id,
                "final_content_manifest_sha256": final_content_manifest_sha256,
                "files": expected_content_files,
            },
        )
        _fsync_tree(staging)

        if generation.exists():
            _validate_generation_integrity(
                generation,
                publication_id=publication_id,
                expected_manifest=generation_manifest,
                expected_content_files=expected_content_files,
            )
        else:
            published = _publish_directory_noreplace(staging, generation)
            if published:
                _fsync_directory(generation_root)
                _readonly_generation(generation)
                _fsync_tree(generation)
                _fsync_directory(generation_root)
            else:
                _validate_generation_integrity(
                    generation,
                    publication_id=publication_id,
                    expected_manifest=generation_manifest,
                    expected_content_files=expected_content_files,
                )
        _validate_generation_integrity(
            generation,
            publication_id=publication_id,
            expected_manifest=generation_manifest,
            expected_content_files=expected_content_files,
        )
        _readonly_generation(generation)
        _fsync_tree(generation)
        _fsync_directory(generation_root)
    finally:
        if staging.exists():
            for child in staging.rglob("*"):
                if not child.is_symlink():
                    os.chmod(child, 0o755 if child.is_dir() else 0o644)
            os.chmod(staging, 0o755)
            shutil.rmtree(staging)

    _validate_published_motion_bank_view(
        generation,
        generation,
        expected_map_payload=root_object_map_payload,
        expected_motion_identities=normalized_motion_identities["root"],
    )
    _validate_published_motion_bank_view(
        generation,
        generation / "_single_slot_motion_bank",
        expected_map_payload=nested_object_map_payload,
        expected_motion_identities=normalized_motion_identities["nested"],
    )
    _validate_published_motion_bank_view(
        generation,
        generation / "motion_bank",
        expected_map_payload=nested_object_map_payload,
        expected_motion_identities=normalized_motion_identities["nested"],
    )

    # Every compatibility path is a one-time static view through TARGET_BANK.
    # Only TARGET_BANK moves per publication, so consumers never observe clips,
    # motions, summaries, and manifests from independently switched generations.
    for view_name, view_target in static_views.items():
        _ensure_static_view_alias(
            output_root / view_name,
            view_target,
            allowed_root=TEACHER_ROLLOUT_OUTPUT_ROOT,
        )
    _fsync_directory(output_root)
    _fsync_directory(output_root.parent)

    _atomic_switch_generated_alias(target_alias, generation, allowed_root=DATA_BANK_ROOT)
    for view_name in static_views:
        resolved_view = (output_root / view_name).resolve()
        expected_view = generation if view_name == "current" else generation / view_name
        if resolved_view != expected_view:
            raise SystemExit(
                f"[ERROR] Published rollout view does not resolve into the committed generation: {view_name}"
            )
    published_manifest = {
        **merge_manifest,
        "published_output_root": str(output_root),
        "published_target_bank": str(target_alias),
    }
    print(json.dumps(published_manifest, indent=2, sort_keys=True))


def merge_outputs(args: argparse.Namespace) -> None:
    publication_lock = DATA_BANK_ROOT / ".teacher_realmesh_publication.lock"
    with _scoped_no_follow_lock(
        publication_lock,
        allowed_root=DATA_BANK_ROOT,
        label="teacher rollout publication lock",
    ):
        _merge_outputs_locked(args)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and merge parallel AS real-mesh teacher rollout exports.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-shards")
    prepare.add_argument("--source-bank", required=True, type=Path)
    prepare.add_argument("--source-map", required=True, type=Path)
    prepare.add_argument("--shard-root", required=True, type=Path)
    prepare.add_argument("--num-shards", required=True, type=int)
    prepare.add_argument("--per-gpu-envs", required=True, type=int)
    prepare.add_argument("--expected-total", type=int, default=0)
    prepare.add_argument("--allowed-categories", default="box,ball,bin,barrel")
    prepare.add_argument("--exclude-clips", default="")
    prepare.add_argument("--rollout-contract-json", default="{}")
    prepare.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the prospective shard manifest without writing or deleting anything.",
    )
    prepare.set_defaults(func=prepare_shards)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--output-root", required=True, type=Path)
    merge.add_argument("--target-bank", required=True, type=Path)
    merge.add_argument("--source-bank", required=True, type=Path)
    merge.add_argument("--prepared-manifest", required=True, type=Path)
    merge.add_argument("--prepared-manifest-sha256", required=True)
    merge.add_argument("--contact-export-name", default=DEFAULT_CONTACT_EXPORT_NAME)
    merge.add_argument("--expected-teacher-checkpoint-sha256", required=True)
    merge.add_argument("--teacher-checkpoint-path", required=True, type=Path)
    merge.add_argument("--teacher-checkpoint-source", required=True)
    merge.add_argument("--save-visualization", action="store_true")
    merge.add_argument("--save-visualization-preview-png", action="store_true")
    merge.add_argument("--save-visualization-face-heatmap-png", action="store_true")
    merge.add_argument("--force", action="store_true")
    merge.set_defaults(func=merge_outputs)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
