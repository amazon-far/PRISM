#!/usr/bin/env python3
"""Filter and atomically publish an immutable, content-addressed AS solid bank.

Motion clips, contact sidecars, and optional metadata are regular read-only
copies in the published generation.  They must not remain symlinks to a source
tree that can change while a training run is consuming the generation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any
import xml.etree.ElementTree as ET

from prepare_as_rank_shards import (
    MOTION_TRANSITION_SOURCE_KEY,
    canonical_motion_transition_source,
    collect_local_mesh_asset_paths,
    collect_local_urdf_asset_paths,
    resolve_motion_transition_source,
)
from motion_generator_teacher import (
    MOTION_GENERATOR_TEACHER_KEY,
    motion_generator_teacher_from_rollout_manifest,
)


_VERSION = 5
_MARKER_NAME = ".generated_by_prepare_immutable_solid_bank"
_MANIFEST_NAME = "manifest.json"
_OBJECT_MAP_NAME = "_clip_object_urdf_map.json"
_ALLOWED_CATEGORIES = {"box", "bin", "barrel", "ball", "anything"}
_OPTIONAL_METADATA = (
    "teacher_export_summary.json",
    "teacher_export_summary.csv",
    "source_teacher_export.txt",
)
_ENTRY_SINGLE_MESH_KEYS = (
    "object_mesh_path",
    "object_visual_mesh_path",
    "object_collision_mesh_path",
)
_ENTRY_MULTI_MESH_KEYS = (
    "object_mesh_paths",
    "object_visual_mesh_paths",
    "object_collision_mesh_paths",
)


@dataclass(frozen=True)
class Selection:
    selected: dict[str, dict[str, Any]]
    category_counts: dict[str, int]
    filtered_payload: dict[str, Any]
    source_identity: dict[str, Any]
    source_digest: str
    contact_root: Path
    metadata_paths: tuple[Path, ...]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"Expected a regular file: {resolved}")
    before = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    after = resolved.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise RuntimeError(f"File changed while hashing: {resolved}")
    record_path = os.path.relpath(path, relative_to) if relative_to is not None else str(resolved)
    return {"path": record_path, "size": int(after.st_size), "sha256": digest.hexdigest()}


def _copy_verified_payload(source: str | Path, destination: str | Path) -> str:
    """Copy a file as read-only and verify its complete byte identity."""

    source_path = Path(source).expanduser().resolve(strict=True)
    destination_path = Path(destination)
    expected = _stable_file_record(source_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, destination_path)
    os.chmod(destination_path, 0o444)
    actual = _stable_file_record(destination_path)
    if (actual["size"], actual["sha256"]) != (expected["size"], expected["sha256"]):
        raise RuntimeError(
            "Copied solid-bank payload does not match its source: "
            f"source={source_path} destination={destination_path}"
        )
    return str(destination_path)


def _tree_records(
    root: Path,
    *,
    reject_symlinks: bool = False,
    require_read_only: bool = False,
) -> list[dict[str, Any]]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Expected a directory: {root}")
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: str(item.relative_to(root))):
        if reject_symlinks and path.is_symlink():
            raise ValueError(f"Published immutable tree contains a symlink: {path}")
        if path.is_symlink() and path.is_dir():
            raise ValueError(f"Directory symlink has no closed traversal semantics: {path}")
        if path.is_file():
            if require_read_only and path.stat().st_mode & 0o222:
                raise ValueError(f"Published immutable tree contains a writable file: {path}")
            records.append(_stable_file_record(path, relative_to=root))
        elif not path.is_dir():
            raise ValueError(f"Unsupported contact-tree entry: {path}")
    if not records:
        raise ValueError(f"Contact export tree contains no files: {root}")
    return records


def _load_clip_map(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    clips = payload.get("clips") if isinstance(payload, dict) else None
    if clips is None and isinstance(payload, dict):
        return {}, payload
    if not isinstance(clips, dict) or not clips:
        raise ValueError(f"Invalid or empty object map: {path}")
    return {key: value for key, value in payload.items() if key != "clips"}, clips


def _category_for(clip_id: str, entry: object) -> str:
    parts = [clip_id]
    if isinstance(entry, dict):
        for key in (
            "object_name",
            "object_urdf_path",
            "object_mesh_path",
            "object_category",
            "category",
            "object_type",
        ):
            value = str(entry.get(key, "")).strip()
            if not value:
                continue
            if key.endswith("_path"):
                path = Path(value)
                parts.extend((path.name, path.stem))
            else:
                parts.append(value)
    else:
        path = Path(str(entry).strip())
        parts.extend((path.name, path.stem))
    raw = " ".join(parts).lower().replace("-", "_")
    if "barrel" in raw:
        return "barrel"
    if "bin" in raw or "trash" in raw or "basket" in raw:
        return "bin"
    if "ball" in raw or "sphere" in raw:
        return "ball"
    if "box" in raw or "cube" in raw or "largebox" in raw:
        return "box"
    if "anything" in raw:
        return "anything"
    return "other"


def _resolve_path(raw: str, *, base_dir: Path, role: str) -> Path:
    path = Path(str(raw).strip()).expanduser()
    if not str(path):
        raise ValueError(f"{role} path is empty")
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{role} does not exist: {path}")
    return path


def _absolutize_and_validate_entry(
    clip_id: str,
    raw_entry: object,
    *,
    source_map: Path,
    mesh_cache: dict[Path, tuple[Path, ...]],
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    if not isinstance(raw_entry, dict):
        raise ValueError(f"Solid object-map entry for {clip_id!r} must be a mapping")
    entry = dict(raw_entry)
    raw_urdf = str(entry.get("object_urdf_path", "")).strip()
    if not raw_urdf:
        raise ValueError(f"Solid clip {clip_id!r} is missing object_urdf_path")
    urdf_path = _resolve_path(raw_urdf, base_dir=source_map.parent, role=f"{clip_id} URDF")
    entry["object_urdf_path"] = str(urdf_path)

    try:
        root = ET.parse(urdf_path).getroot()
    except Exception as exc:
        raise ValueError(f"Invalid object URDF for {clip_id!r}: {urdf_path}: {exc}") from exc
    primitives = [name for name in ("box", "sphere", "cylinder", "capsule") if root.findall(f".//{name}")]
    if primitives:
        raise ValueError(
            f"Solid clip {clip_id!r} uses primitive geometry {primitives} in {urdf_path}; mesh geometry is required"
        )
    if not root.findall(".//mesh"):
        raise ValueError(f"Solid clip {clip_id!r} URDF has no mesh geometry: {urdf_path}")

    assets = list(collect_local_urdf_asset_paths(urdf_path, mesh_cache=mesh_cache))
    for key in _ENTRY_SINGLE_MESH_KEYS:
        raw_mesh = str(entry.get(key, "")).strip()
        if not raw_mesh:
            continue
        mesh_path = _resolve_path(raw_mesh, base_dir=source_map.parent, role=f"{clip_id} {key}")
        entry[key] = str(mesh_path)
        mesh_assets = mesh_cache.get(mesh_path)
        if mesh_assets is None:
            mesh_assets = collect_local_mesh_asset_paths(mesh_path)
            mesh_cache[mesh_path] = mesh_assets
        assets.extend(mesh_assets)
    for key in _ENTRY_MULTI_MESH_KEYS:
        raw_values = entry.get(key)
        if raw_values is None:
            continue
        if isinstance(raw_values, str):
            raw_values = [raw_values]
        if not isinstance(raw_values, (list, tuple)) or not raw_values:
            raise ValueError(f"Solid clip {clip_id!r} has invalid {key}; expected a non-empty path list")
        resolved_values: list[str] = []
        for index, raw_mesh in enumerate(raw_values):
            mesh_path = _resolve_path(
                str(raw_mesh),
                base_dir=source_map.parent,
                role=f"{clip_id} {key}[{index}]",
            )
            resolved_values.append(str(mesh_path))
            mesh_assets = mesh_cache.get(mesh_path)
            if mesh_assets is None:
                mesh_assets = collect_local_mesh_asset_paths(mesh_path)
                mesh_cache[mesh_path] = mesh_assets
            assets.extend(mesh_assets)
        entry[key] = resolved_values
    return entry, tuple(dict.fromkeys(assets))


def _read_allowlist(path: Path | None) -> tuple[set[str] | None, dict[str, Any] | None]:
    if path is None:
        return None, None
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Solid clip allowlist does not exist: {path}")
    clip_ids = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not clip_ids:
        raise ValueError(f"Solid clip allowlist is empty: {path}")
    return clip_ids, _stable_file_record(path)


def _select(
    *,
    source_bank: Path,
    source_map: Path,
    allowed: set[str],
    contact_export_name: str,
    clip_list_path: Path | None,
    contact_root_override: Path | None,
) -> Selection:
    source_bank = source_bank.expanduser().resolve()
    source_map = source_map.expanduser().resolve()
    if not source_bank.is_dir():
        raise FileNotFoundError(f"Solid source bank does not exist: {source_bank}")
    if not source_map.is_file():
        raise FileNotFoundError(f"Solid source object map does not exist: {source_map}")
    if not allowed or not allowed.issubset(_ALLOWED_CATEGORIES):
        raise ValueError(f"Allowed categories must be a non-empty subset of {sorted(_ALLOWED_CATEGORIES)}")
    if not contact_export_name or Path(contact_export_name).name != contact_export_name:
        raise ValueError(f"contact_export_name must be one path component: {contact_export_name!r}")

    source_metadata, clips = _load_clip_map(source_map)
    source_npz_paths = sorted(source_bank.glob("*.npz"), key=lambda path: path.name)
    source_clip_ids = [path.stem for path in source_npz_paths]
    if not source_clip_ids:
        raise FileNotFoundError(f"Solid source bank has no .npz motion clips: {source_bank}")
    source_clip_set = set(source_clip_ids)
    map_clip_set = {str(clip_id) for clip_id in clips}
    if source_clip_set != map_clip_set:
        raise ValueError(
            "Cannot infer or validate transition lineage from a non-closed solid source bank: "
            f"motion_only={sorted(source_clip_set - map_clip_set)[:20]}, "
            f"map_only={sorted(map_clip_set - source_clip_set)[:20]}"
        )
    transition_source = resolve_motion_transition_source(
        source_metadata,
        active_clip_count=len(source_clip_ids),
        infer_if_missing=True,
        role=f"solid source map {source_map} {MOTION_TRANSITION_SOURCE_KEY}",
    )
    assert transition_source is not None
    allowed_clip_ids, clip_list_record = _read_allowlist(clip_list_path)
    missing_allowlist = set(allowed_clip_ids or ())
    selected: dict[str, dict[str, Any]] = {}
    category_counts: Counter[str] = Counter()
    asset_paths: set[Path] = set()
    mesh_cache: dict[Path, tuple[Path, ...]] = {}
    motion_paths: list[Path] = []
    for clip_id_raw, raw_entry in clips.items():
        clip_id = str(clip_id_raw)
        category = _category_for(clip_id, raw_entry)
        category_counts[category] += 1
        if allowed_clip_ids is not None:
            if clip_id not in allowed_clip_ids:
                continue
            missing_allowlist.discard(clip_id)
        if category not in allowed:
            continue
        motion_path = source_bank / f"{clip_id}.npz"
        if not motion_path.is_file():
            raise FileNotFoundError(f"Selected solid clip is missing motion data: {motion_path}")
        entry, entry_assets = _absolutize_and_validate_entry(
            clip_id,
            raw_entry,
            source_map=source_map,
            mesh_cache=mesh_cache,
        )
        selected[clip_id] = entry
        motion_paths.append(motion_path)
        asset_paths.update(entry_assets)
    if missing_allowlist:
        raise ValueError(
            "Solid clip allowlist contains entries missing from the object map: "
            + ", ".join(sorted(missing_allowlist)[:20])
        )
    if not selected:
        raise ValueError(f"No clips matched allowed solid categories {sorted(allowed)} in {source_bank}")

    contact_root = (
        contact_root_override.expanduser().resolve()
        if contact_root_override is not None
        else (source_bank / contact_export_name).resolve()
    )
    contact_records = _tree_records(contact_root)
    metadata_paths = tuple(path for name in _OPTIONAL_METADATA if (path := source_bank / name).is_file())
    rollout_manifest_path = source_bank / "realmesh_rollout_manifest.json"
    source_rollout_manifest: dict[str, Any] | None = None
    motion_generator_teacher: dict[str, Any] | None = None
    if rollout_manifest_path.exists() or rollout_manifest_path.is_symlink():
        if rollout_manifest_path.is_symlink() or not rollout_manifest_path.is_file():
            raise ValueError(
                "Teacher-rollout source manifest must be one regular non-symlink file: "
                f"{rollout_manifest_path}"
            )
        source_rollout_manifest = _stable_file_record(rollout_manifest_path)
        motion_generator_teacher = motion_generator_teacher_from_rollout_manifest(
            rollout_manifest_path
        )
    # Preserve the transition lineage from the unfiltered source even when
    # the active optimization view contains one clip.  Re-inferring after the
    # filter would silently turn a global runtime-hold experiment into a
    # standalone static-splice experiment.
    filtered_payload = {
        MOTION_TRANSITION_SOURCE_KEY: transition_source,
        "clips": {clip_id: selected[clip_id] for clip_id in sorted(selected)},
    }
    source_identity = {
        "version": _VERSION,
        "source_bank": str(source_bank),
        "source_object_map": _stable_file_record(source_map),
        "allowed_categories": sorted(allowed),
        "clip_allowlist": clip_list_record,
        "selected_clip_ids": sorted(selected),
        "source_motion_clip_ids": source_clip_ids,
        "source_rollout_manifest": source_rollout_manifest,
        MOTION_GENERATOR_TEACHER_KEY: motion_generator_teacher,
        MOTION_TRANSITION_SOURCE_KEY: transition_source,
        "motion_files": [_stable_file_record(path) for path in sorted(motion_paths)],
        "object_assets": [_stable_file_record(path) for path in sorted(asset_paths, key=str)],
        "filtered_object_map_sha256": _sha256_json(filtered_payload),
        "contact_export_name": contact_export_name,
        "contact_root": str(contact_root),
        "contact_files": contact_records,
        "metadata_files": [_stable_file_record(path) for path in metadata_paths],
    }
    return Selection(
        selected=selected,
        category_counts=dict(category_counts),
        filtered_payload=filtered_payload,
        source_identity=source_identity,
        source_digest=_sha256_json(source_identity),
        contact_root=contact_root,
        metadata_paths=metadata_paths,
    )


def _safe_target_component(raw_base: str, digest: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_base.strip()).strip("_.") or "solid_bank"
    suffix = f"__src_{digest}"
    # Leave room for the incoming-directory and lock suffixes on filesystems
    # with the common 255-byte NAME_MAX.
    max_component_bytes = 220
    max_prefix = max_component_bytes - len(suffix)
    if len(cleaned.encode("utf-8")) > max_prefix:
        base_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:16]
        cleaned = f"{cleaned[: max_prefix - 17]}_{base_hash}"
    return cleaned + suffix


@contextmanager
def _output_lock(output_root: Path):
    output_root.parent.mkdir(parents=True, exist_ok=True)
    lock_key = hashlib.sha256(output_root.name.encode("utf-8")).hexdigest()
    lock_path = output_root.parent / f".solid-bank-{lock_key}.lock"
    with lock_path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _fsync_tree(root: Path) -> None:
    directories = [root]
    for current, child_dirs, child_files in os.walk(root):
        current_path = Path(current)
        directories.extend(current_path / name for name in child_dirs)
        for name in child_files:
            path = current_path / name
            if path.is_symlink():
                raise ValueError(f"Immutable solid generation contains a symlink: {path}")
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _freeze_regular_files(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Immutable solid generation contains a symlink: {path}")
        if path.is_file():
            os.chmod(path, 0o444)


def _published_is_valid(output_root: Path, *, selection: Selection) -> tuple[bool, dict[str, Any] | None]:
    try:
        marker_path = output_root / _MARKER_NAME
        manifest_path = output_root / _MANIFEST_NAME
        if (
            output_root.is_symlink()
            or not output_root.is_dir()
            or marker_path.is_symlink()
            or manifest_path.is_symlink()
            or not marker_path.is_file()
            or not manifest_path.is_file()
            or marker_path.stat().st_mode & 0o222
            or manifest_path.stat().st_mode & 0o222
        ):
            return False, None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return False, None
        if (
            manifest.get("version") != _VERSION
            or manifest.get("source_digest") != selection.source_digest
            or manifest.get("source_identity") != selection.source_identity
            or manifest.get(MOTION_TRANSITION_SOURCE_KEY)
            != selection.source_identity[MOTION_TRANSITION_SOURCE_KEY]
            or manifest.get("output_root") != str(output_root)
        ):
            return False, manifest
        expected_npz = [f"{clip_id}.npz" for clip_id in sorted(selection.selected)]
        if sorted(path.name for path in output_root.glob("*.npz")) != expected_npz:
            return False, manifest
        for name in expected_npz:
            if (
                (output_root / name).is_symlink()
                or not (output_root / name).is_file()
                or (output_root / name).stat().st_mode & 0o222
            ):
                return False, manifest
        published_motion = manifest.get("published_motion_files")
        actual_motion = [
            _stable_file_record(output_root / name, relative_to=output_root)
            for name in expected_npz
        ]
        if published_motion != actual_motion:
            return False, manifest
        map_record = manifest.get("published_object_map")
        map_path = output_root / _OBJECT_MAP_NAME
        if map_path.is_symlink() or map_path.stat().st_mode & 0o222:
            return False, manifest
        if _stable_file_record(map_path, relative_to=output_root) != map_record:
            return False, manifest
        contact_name = str(selection.source_identity["contact_export_name"])
        contact_snapshot = output_root / contact_name
        if contact_snapshot.is_symlink() or not contact_snapshot.is_dir():
            return False, manifest
        if manifest.get("published_contact_files") != _tree_records(
            contact_snapshot,
            reject_symlinks=True,
            require_read_only=True,
        ):
            return False, manifest
        published_metadata = manifest.get("published_metadata_files")
        actual_metadata = [
            _stable_file_record(output_root / path.name, relative_to=output_root)
            for path in selection.metadata_paths
        ]
        if published_metadata != actual_metadata:
            return False, manifest
        for metadata_path in selection.metadata_paths:
            snapshot = output_root / metadata_path.name
            if snapshot.is_symlink() or not snapshot.is_file() or snapshot.stat().st_mode & 0o222:
                return False, manifest
        return True, manifest
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
        return False, None


def prepare_immutable_solid_bank(
    *,
    source_bank: Path,
    source_map: Path,
    allowed_categories: set[str],
    contact_export_name: str,
    clip_list_path: Path | None,
    target_bank_name: str | None,
    contact_root_override: Path | None = None,
) -> dict[str, Any]:
    source_bank = source_bank.expanduser().resolve()
    selection = _select(
        source_bank=source_bank,
        source_map=source_map,
        allowed=allowed_categories,
        contact_export_name=contact_export_name,
        clip_list_path=clip_list_path,
        contact_root_override=contact_root_override,
    )
    slug = "_".join(name for name in ("box", "bin", "barrel", "ball") if name in allowed_categories)
    raw_base = target_bank_name or f"{source_bank.name}_solid_{slug}"
    output_root = source_bank.parent / _safe_target_component(raw_base, selection.source_digest)

    with _output_lock(output_root):
        current, manifest = _published_is_valid(output_root, selection=selection)
        if current:
            assert manifest is not None
            return manifest
        if os.path.lexists(output_root):
            raise ValueError(
                "Refusing to replace invalid content at immutable solid-bank identity: "
                f"{output_root}"
            )

        temp_root = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.incoming-", dir=output_root.parent))
        try:
            for clip_id in sorted(selection.selected):
                source_npz = (source_bank / f"{clip_id}.npz").resolve()
                _copy_verified_payload(source_npz, temp_root / f"{clip_id}.npz")
            object_map_path = temp_root / _OBJECT_MAP_NAME
            object_map_path.write_text(
                json.dumps(selection.filtered_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            for metadata_path in selection.metadata_paths:
                _copy_verified_payload(metadata_path, temp_root / metadata_path.name)
            contact_name = str(selection.source_identity["contact_export_name"])
            contact_snapshot = temp_root / contact_name
            shutil.copytree(
                selection.contact_root,
                contact_snapshot,
                symlinks=False,
                copy_function=_copy_verified_payload,
            )

            final_selection = _select(
                source_bank=source_bank,
                source_map=source_map,
                allowed=allowed_categories,
                contact_export_name=contact_export_name,
                clip_list_path=clip_list_path,
                contact_root_override=contact_root_override,
            )
            if final_selection.source_digest != selection.source_digest:
                raise RuntimeError(
                    "Solid-bank source changed while its immutable view was being built: "
                    f"before={selection.source_digest}, after={final_selection.source_digest}"
                )
            manifest = {
                "version": _VERSION,
                "source_digest": selection.source_digest,
                "source_identity": selection.source_identity,
                "output_root": str(output_root),
                "selected_clip_count": len(selection.selected),
                MOTION_TRANSITION_SOURCE_KEY: canonical_motion_transition_source(
                    selection.source_identity[MOTION_TRANSITION_SOURCE_KEY],
                    active_clip_count=len(selection.selected),
                    role=f"solid manifest {MOTION_TRANSITION_SOURCE_KEY}",
                ),
                "category_counts": selection.category_counts,
                "published_object_map": _stable_file_record(object_map_path, relative_to=temp_root),
                "published_motion_files": [
                    _stable_file_record(temp_root / f"{clip_id}.npz", relative_to=temp_root)
                    for clip_id in sorted(selection.selected)
                ],
                "published_contact_files": _tree_records(contact_snapshot),
                "published_metadata_files": [
                    _stable_file_record(temp_root / path.name, relative_to=temp_root)
                    for path in selection.metadata_paths
                ],
            }
            (temp_root / _MARKER_NAME).write_text(
                "generated by prepare_immutable_solid_bank.py\n",
                encoding="utf-8",
            )
            (temp_root / _MANIFEST_NAME).write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _freeze_regular_files(temp_root)
            _fsync_tree(temp_root)
            os.replace(temp_root, output_root)
            parent_fd = os.open(output_root.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        finally:
            if temp_root.exists():
                shutil.rmtree(temp_root)

        current, manifest = _published_is_valid(output_root, selection=selection)
        if not current or manifest is None:
            raise RuntimeError(f"Published immutable solid bank failed validation: {output_root}")
        return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bank", required=True, type=Path)
    parser.add_argument("--source-map", required=True, type=Path)
    parser.add_argument("--allowed-categories-json", required=True)
    parser.add_argument("--contact-export-name", required=True)
    parser.add_argument("--clip-list", type=Path)
    parser.add_argument("--target-bank-name")
    parser.add_argument("--contact-root", type=Path)
    args = parser.parse_args()
    try:
        raw_allowed = json.loads(args.allowed_categories_json)
        if not isinstance(raw_allowed, list) or not all(isinstance(item, str) for item in raw_allowed):
            raise ValueError("--allowed-categories-json must be a JSON list of strings")
        manifest = prepare_immutable_solid_bank(
            source_bank=args.source_bank,
            source_map=args.source_map,
            allowed_categories=set(raw_allowed),
            contact_export_name=args.contact_export_name,
            clip_list_path=args.clip_list,
            target_bank_name=args.target_bank_name,
            contact_root_override=args.contact_root,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to prepare immutable solid AS bank: {exc}", file=sys.stderr)
        return 2

    output_root = Path(str(manifest["output_root"]))
    print(f"SOLID_BANK_NAME={output_root.name}")
    print(f"SOLID_BANK_DIR={output_root}")
    print(f"SOLID_OBJECT_MAP={output_root / _OBJECT_MAP_NAME}")
    print(f"SOLID_SELECTED_CLIP_COUNT={manifest['selected_clip_count']}")
    counts = manifest["category_counts"]
    print("SOLID_CATEGORY_COUNTS=" + ",".join(f"{key}:{counts[key]}" for key in sorted(counts)))
    print(f"SOLID_SOURCE_DIGEST={manifest['source_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
