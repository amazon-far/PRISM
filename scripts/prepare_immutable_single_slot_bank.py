#!/usr/bin/env python3
"""Publish a content-addressed single-slot AS motion/object-map snapshot.

The published motion payload is copied into a new generation before an atomic
rename.  A changed motion clip, object map, URDF, mesh, or canonicalization
contract selects a new digest path, so a reader can never observe a launcher
clearing/rebuilding the directory or later source-NPZ mutations through a
symbolic link.  Object mesh dependencies referenced by the generated URDFs are
content-closed again by the training provenance preflight immediately before
simulator startup.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

from prepare_as_rank_shards import (
    MOTION_TRANSITION_SOURCE_KEY,
    canonical_motion_transition_source,
    compute_rank_shard_source_digest,
)
from prepare_single_slot_object_map import prepare_single_slot_map
from motion_generator_teacher import (
    MOTION_GENERATOR_TEACHER_KEY,
    motion_generator_teacher_from_solid_manifest,
)


_MANIFEST_VERSION = 5
# Version 2 also guarantees that launcher-only metadata is kept outside the
# immutable payload.  Bumping the view identity lets deployments containing a
# v1 generation polluted by the historical ``_object_bank_wandb.env`` writer
# publish a clean generation without deleting or mutating the old directory.
_CANONICALIZATION_CONTRACT_VERSION = 2
_MARKER_NAME = ".generated_by_prepare_immutable_single_slot_bank"
_MANIFEST_NAME = "manifest.json"
_OBJECT_MAP_NAME = "_clip_object_urdf_map.json"


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
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    if not resolved.is_file():
        raise FileNotFoundError(f"Expected a regular file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    after = resolved.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise RuntimeError(f"File changed while hashing: {resolved}")
    record_path = (
        os.path.relpath(path, relative_to)
        if relative_to is not None
        else str(resolved)
    )
    return {
        "path": record_path,
        "size": int(after.st_size),
        "sha256": digest.hexdigest(),
    }


def _copy_verified_payload(source: Path, destination: Path) -> None:
    """Copy one regular file and prove that the published bytes match source.

    ``shutil.copy2`` deliberately is not used: inherited writable mode bits
    would make a supposedly immutable generation easy to mutate in place.
    Source identity is checked by ``_stable_file_record`` on both sides of the
    copy, and the full source closure is rehashed once more before publication.
    """

    expected = _stable_file_record(source)
    shutil.copyfile(source, destination)
    os.chmod(destination, 0o444)
    actual = _stable_file_record(destination)
    if (actual["size"], actual["sha256"]) != (expected["size"], expected["sha256"]):
        raise RuntimeError(
            "Copied AS motion payload does not match its source: "
            f"source={source} destination={destination}"
        )


def _view_digest(
    source_digest: str,
    *,
    motion_generator_teacher: dict[str, Any] | None,
    source_lineage_manifest: dict[str, Any] | None,
) -> str:
    return _sha256_json(
        {
            "manifest_version": _MANIFEST_VERSION,
            "canonicalization_contract_version": _CANONICALIZATION_CONTRACT_VERSION,
            "rank_shard_source_digest_ws1": source_digest,
            MOTION_GENERATOR_TEACHER_KEY: motion_generator_teacher,
            "source_lineage_manifest": source_lineage_manifest,
        }
    )


def _source_motion_generator_lineage(
    source_motion_dir: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Read lineage from an immutable solid source, if one is present."""

    manifest_path = source_motion_dir / _MANIFEST_NAME
    if not manifest_path.exists() and not manifest_path.is_symlink():
        return None, None
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(
            f"Source lineage manifest must be one regular non-symlink file: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not parse source lineage manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Source lineage manifest must be a mapping: {manifest_path}")
    source_identity = manifest.get("source_identity")
    if source_identity is None:
        # A generic motion directory may have an unrelated manifest.  It does
        # not claim authenticated teacher-rollout lineage.
        return None, None
    if not isinstance(source_identity, dict):
        raise ValueError(f"Source solid manifest has malformed source_identity: {manifest_path}")
    if manifest.get("version") == _MANIFEST_VERSION \
            and MOTION_GENERATOR_TEACHER_KEY not in source_identity:
        raise ValueError(
            f"Version {_MANIFEST_VERSION} solid manifest omitted {MOTION_GENERATOR_TEACHER_KEY}: "
            f"{manifest_path}"
        )
    output_root = manifest.get("output_root")
    if not isinstance(output_root, str) or Path(output_root).expanduser().resolve() != source_motion_dir:
        raise ValueError(f"Source solid manifest points at a different output root: {manifest_path}")
    identity = motion_generator_teacher_from_solid_manifest(
        manifest,
        role=f"source solid manifest {manifest_path}",
    )
    record = _stable_file_record(manifest_path, relative_to=source_motion_dir)
    return identity, record


@contextmanager
def _output_lock(output_root: Path):
    output_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_root.parent / f".{output_root.name}.lock"
    with lock_path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_tree(root: Path) -> None:
    # Data files first, then directories bottom-up, so the final rename cannot
    # publish a generation whose manifest reached disk without its payload.
    directories = [root]
    for current, child_dirs, child_files in os.walk(root):
        current_path = Path(current)
        directories.extend(current_path / name for name in child_dirs)
        for name in child_files:
            path = current_path / name
            if not path.is_symlink():
                _fsync_file(path)
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _freeze_payload_tree(root: Path) -> None:
    """Make the payload namespace read-only, leaving one derivative namespace.

    Rank shards are published after the single-slot view, so they need a
    dedicated writable parent.  Everything else, including the generation
    root and canonical-URDF directory, is non-writable.  This prevents an
    auxiliary launcher from adding metadata beside immutable NPZ files while
    still allowing separately content-addressed rank shards below
    ``_rank_shards``.
    """

    rank_shards_root = root / "_rank_shards"
    rank_shards_root.mkdir(mode=0o755)
    directories = [root]
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Immutable single-slot generation contains a symlink: {path}")
        if path.is_file():
            os.chmod(path, 0o444)
        elif path.is_dir():
            directories.append(path)
    for directory in directories:
        os.chmod(directory, 0o755 if directory == rank_shards_root else 0o555)


def _remove_unpublished_tree(root: Path) -> None:
    """Best-effort cleanup for a frozen tree that was never atomically published."""

    if not root.exists():
        return
    for current, _child_dirs, _child_files in os.walk(root):
        os.chmod(current, 0o700)
    shutil.rmtree(root)


def _published_manifest_is_valid(
    output_root: Path,
    *,
    source_motion_dir: Path,
    source_object_map: Path,
    source_digest: str,
    view_digest: str,
    motion_generator_teacher: dict[str, Any] | None,
    source_lineage_manifest: dict[str, Any] | None,
) -> tuple[bool, dict[str, Any] | None]:
    try:
        if (
            output_root.is_symlink()
            or not output_root.is_dir()
            or output_root.stat().st_mode & 0o222
        ):
            return False, None
        marker_path = output_root / _MARKER_NAME
        manifest_path = output_root / _MANIFEST_NAME
        if (
            marker_path.is_symlink()
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
        expected_identity = {
            "version": _MANIFEST_VERSION,
            "canonicalization_contract_version": _CANONICALIZATION_CONTRACT_VERSION,
            "source_motion_dir": str(source_motion_dir),
            "source_object_map": str(source_object_map),
            "source_digest": source_digest,
            "view_digest": view_digest,
            "output_root": str(output_root),
            MOTION_GENERATOR_TEACHER_KEY: motion_generator_teacher,
            "source_lineage_manifest": source_lineage_manifest,
        }
        if any(manifest.get(key) != value for key, value in expected_identity.items()):
            return False, manifest

        npz_records = manifest.get("motion_files")
        generated_records = manifest.get("generated_files")
        if not isinstance(npz_records, list) or not npz_records:
            return False, manifest
        if not isinstance(generated_records, list) or not generated_records:
            return False, manifest
        expected_npz_names = sorted(str(record["path"]) for record in npz_records)
        actual_npz_names = sorted(path.name for path in output_root.glob("*.npz"))
        if actual_npz_names != expected_npz_names:
            return False, manifest
        expected_generated_names = sorted(str(record["path"]) for record in generated_records)
        expected_urdf_names = sorted(
            name for name in expected_generated_names if name.startswith("_single_slot_urdfs/")
        )
        actual_urdf_names = sorted(
            os.path.relpath(path, output_root)
            for path in (output_root / "_single_slot_urdfs").glob("*.urdf")
        )
        if actual_urdf_names != expected_urdf_names:
            return False, manifest
        canonical_urdf_root = output_root / "_single_slot_urdfs"
        rank_shards_root = output_root / "_rank_shards"
        rank_shards_mode = rank_shards_root.stat().st_mode
        if (
            canonical_urdf_root.is_symlink()
            or not canonical_urdf_root.is_dir()
            or canonical_urdf_root.stat().st_mode & 0o222
            or rank_shards_root.is_symlink()
            or not rank_shards_root.is_dir()
            or (rank_shards_mode & 0o022) != 0
        ):
            return False, manifest
        for expected in [*npz_records, *generated_records]:
            relative = str(expected["path"])
            candidate = output_root / relative
            if candidate.is_symlink() or candidate.stat().st_mode & 0o222:
                return False, manifest
            if _stable_file_record(candidate, relative_to=output_root) != expected:
                return False, manifest
        published_map = json.loads((output_root / _OBJECT_MAP_NAME).read_text(encoding="utf-8"))
        if not isinstance(published_map, dict):
            return False, manifest
        transition_source = canonical_motion_transition_source(
            published_map.get(MOTION_TRANSITION_SOURCE_KEY),
            active_clip_count=len(npz_records),
            role=f"published single-slot map {MOTION_TRANSITION_SOURCE_KEY}",
        )
        if manifest.get(MOTION_TRANSITION_SOURCE_KEY) != transition_source:
            return False, manifest
        allowed_top_level = {
            _MARKER_NAME,
            _MANIFEST_NAME,
            _OBJECT_MAP_NAME,
            "_single_slot_urdfs",
            # Rank-local shards are a separately content-addressed derivative
            # intentionally published after this payload generation.
            "_rank_shards",
            *expected_npz_names,
        }
        if any(path.name not in allowed_top_level for path in output_root.iterdir()):
            return False, manifest
        return True, manifest
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
        return False, None


def _build_manifest(
    temp_root: Path,
    *,
    output_root: Path,
    source_motion_dir: Path,
    source_object_map: Path,
    source_digest: str,
    view_digest: str,
    clip_count: int,
    urdf_count: int,
    motion_transition_source: dict[str, Any],
    motion_generator_teacher: dict[str, Any] | None,
    source_lineage_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    motion_records = [
        _stable_file_record(path, relative_to=temp_root)
        for path in sorted(temp_root.glob("*.npz"))
    ]
    generated_paths = [temp_root / _OBJECT_MAP_NAME]
    generated_paths.extend(sorted((temp_root / "_single_slot_urdfs").glob("*.urdf")))
    generated_records = [
        _stable_file_record(path, relative_to=temp_root)
        for path in generated_paths
    ]
    return {
        "version": _MANIFEST_VERSION,
        "canonicalization_contract_version": _CANONICALIZATION_CONTRACT_VERSION,
        "source_motion_dir": str(source_motion_dir),
        "source_object_map": str(source_object_map),
        "source_digest": source_digest,
        "view_digest": view_digest,
        "output_root": str(output_root),
        "clip_count": int(clip_count),
        "canonical_urdf_count": int(urdf_count),
        MOTION_GENERATOR_TEACHER_KEY: motion_generator_teacher,
        "source_lineage_manifest": source_lineage_manifest,
        MOTION_TRANSITION_SOURCE_KEY: canonical_motion_transition_source(
            motion_transition_source,
            active_clip_count=clip_count,
            role=f"single-slot manifest {MOTION_TRANSITION_SOURCE_KEY}",
        ),
        "motion_files": motion_records,
        "generated_files": generated_records,
    }


def prepare_immutable_single_slot_bank(
    *,
    source_motion_dir: Path,
    source_object_map: Path,
    output_base: Path,
) -> dict[str, Any]:
    source_motion_dir = source_motion_dir.expanduser().resolve()
    source_object_map = source_object_map.expanduser().resolve()
    output_base = output_base.expanduser().resolve()
    if not source_motion_dir.is_dir():
        raise FileNotFoundError(f"Source motion directory does not exist: {source_motion_dir}")
    if not source_object_map.is_file():
        raise FileNotFoundError(f"Source object map does not exist: {source_object_map}")
    if output_base == source_motion_dir:
        raise ValueError("Output base must not replace the source motion directory")

    source_digest = compute_rank_shard_source_digest(
        motion_dir=source_motion_dir,
        object_map=source_object_map,
        world_size=1,
    )
    motion_generator_teacher, source_lineage_manifest = _source_motion_generator_lineage(
        source_motion_dir
    )
    view_digest = _view_digest(
        source_digest,
        motion_generator_teacher=motion_generator_teacher,
        source_lineage_manifest=source_lineage_manifest,
    )
    output_root = output_base / "by-source" / view_digest

    # Published content-addressed views are intentionally read-only.  Reusing
    # one must therefore be a read-only operation: creating the sibling lock
    # file is needed only when a generation still has to be published.
    current, manifest = _published_manifest_is_valid(
        output_root,
        source_motion_dir=source_motion_dir,
        source_object_map=source_object_map,
        source_digest=source_digest,
        view_digest=view_digest,
        motion_generator_teacher=motion_generator_teacher,
        source_lineage_manifest=source_lineage_manifest,
    )
    if current:
        assert manifest is not None
        return manifest

    with _output_lock(output_root):
        current, manifest = _published_manifest_is_valid(
            output_root,
            source_motion_dir=source_motion_dir,
            source_object_map=source_object_map,
            source_digest=source_digest,
            view_digest=view_digest,
            motion_generator_teacher=motion_generator_teacher,
            source_lineage_manifest=source_lineage_manifest,
        )
        if current:
            assert manifest is not None
            return manifest
        if os.path.lexists(output_root):
            raise ValueError(
                "Refusing to replace an invalid or non-generated immutable single-slot bank: "
                f"{output_root}. Its content-addressed identity must never be mutated."
            )

        output_root.parent.mkdir(parents=True, exist_ok=True)
        temp_root = Path(
            tempfile.mkdtemp(prefix=f".{view_digest}.incoming-", dir=output_root.parent)
        )
        try:
            source_npz_files = sorted(source_motion_dir.glob("*.npz"))
            if not source_npz_files:
                raise FileNotFoundError(f"No .npz clips found under {source_motion_dir}")
            for source_npz in source_npz_files:
                _copy_verified_payload(source_npz.resolve(), temp_root / source_npz.name)

            object_map_path = temp_root / _OBJECT_MAP_NAME
            clip_count, urdf_count = prepare_single_slot_map(
                motion_dir=temp_root,
                object_map=source_object_map,
                output_map=object_map_path,
                active_clip_ids={path.stem for path in source_npz_files},
            )
            if clip_count != len(source_npz_files):
                raise ValueError(
                    "Single-slot object-map coverage differs from the active motion view: "
                    f"map_clips={clip_count}, npz_clips={len(source_npz_files)}"
                )

            generated_map_payload = json.loads(object_map_path.read_text(encoding="utf-8"))
            if not isinstance(generated_map_payload, dict):
                raise ValueError(f"Generated single-slot object map is not a JSON object: {object_map_path}")
            transition_source = canonical_motion_transition_source(
                generated_map_payload.get(MOTION_TRANSITION_SOURCE_KEY),
                active_clip_count=clip_count,
                role=f"generated single-slot map {MOTION_TRANSITION_SOURCE_KEY}",
            )

            final_source_digest = compute_rank_shard_source_digest(
                motion_dir=source_motion_dir,
                object_map=source_object_map,
                world_size=1,
            )
            if final_source_digest != source_digest:
                raise RuntimeError(
                    "Single-slot source changed while the immutable view was being built: "
                    f"before={source_digest}, after={final_source_digest}"
                )
            final_generator_teacher, final_lineage_manifest = _source_motion_generator_lineage(
                source_motion_dir
            )
            if (
                final_generator_teacher != motion_generator_teacher
                or final_lineage_manifest != source_lineage_manifest
            ):
                raise RuntimeError(
                    "Single-slot source teacher lineage changed while the immutable view was built"
                )

            manifest = _build_manifest(
                temp_root,
                output_root=output_root,
                source_motion_dir=source_motion_dir,
                source_object_map=source_object_map,
                source_digest=source_digest,
                view_digest=view_digest,
                clip_count=clip_count,
                urdf_count=urdf_count,
                motion_transition_source=transition_source,
                motion_generator_teacher=motion_generator_teacher,
                source_lineage_manifest=source_lineage_manifest,
            )
            (temp_root / _MARKER_NAME).write_text(
                "generated by prepare_immutable_single_slot_bank.py\n",
                encoding="utf-8",
            )
            (temp_root / _MANIFEST_NAME).write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _freeze_payload_tree(temp_root)
            _fsync_tree(temp_root)
            os.replace(temp_root, output_root)
            parent_fd = os.open(output_root.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        finally:
            if temp_root.exists():
                _remove_unpublished_tree(temp_root)

        current, published_manifest = _published_manifest_is_valid(
            output_root,
            source_motion_dir=source_motion_dir,
            source_object_map=source_object_map,
            source_digest=source_digest,
            view_digest=view_digest,
            motion_generator_teacher=motion_generator_teacher,
            source_lineage_manifest=source_lineage_manifest,
        )
        if not current or published_manifest is None:
            raise RuntimeError(f"Published single-slot bank failed post-publish validation: {output_root}")
        return published_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-motion-dir", required=True, type=Path)
    parser.add_argument("--source-object-map", required=True, type=Path)
    parser.add_argument("--output-base", required=True, type=Path)
    args = parser.parse_args()

    try:
        manifest = prepare_immutable_single_slot_bank(
            source_motion_dir=args.source_motion_dir,
            source_object_map=args.source_object_map,
            output_base=args.output_base,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to prepare immutable AS single-slot bank: {exc}", file=sys.stderr)
        return 2

    output_root = Path(str(manifest["output_root"]))
    print(
        "[INFO] Prepared immutable AS single-slot bank: "
        f"{output_root} clips={manifest['clip_count']} urdfs={manifest['canonical_urdf_count']} "
        f"digest={manifest['view_digest']}",
        file=sys.stderr,
    )
    print(str(output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
