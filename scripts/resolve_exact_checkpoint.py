#!/usr/bin/env python3
"""Resolve one exact local/W&B checkpoint into a validated local cache file."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import os
import re
import stat
import sys
import tempfile
from pathlib import Path

from holosoma.utils.checkpoint_validation import load_verified_torch_checkpoint


def parse_wandb_ref(ref: str) -> tuple[str, str]:
    if ref.startswith("https://wandb.ai/"):
        clean = ref.split("?", 1)[0]
        parts = clean.removeprefix("https://wandb.ai/").split("/")
        if len(parts) < 6 or parts[2] != "runs" or parts[4] != "files":
            raise ValueError("W&B checkpoint URL must include /runs/<id>/files/<model.pt>")
        return f"{parts[0]}/{parts[1]}/{parts[3]}", "/".join(parts[5:])

    if ref.startswith("wandb://"):
        parts = ref.removeprefix("wandb://").split("/")
        if len(parts) >= 5 and parts[2] == "runs":
            entity, project, run_id = parts[0], parts[1], parts[3]
            file_name = "/".join(parts[4:])
        elif len(parts) >= 4:
            entity, project, run_id = parts[:3]
            file_name = "/".join(parts[3:])
        else:
            raise ValueError("W&B checkpoint URI must include an exact model .pt file")
        return f"{entity}/{project}/{run_id}", file_name

    raise ValueError("not a W&B reference")


def validate_checkpoint(path: Path, *, expected_sha256: str | None = None) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint, checkpoint_sha256 = load_verified_torch_checkpoint(
        path,
        expected_sha256=expected_sha256,
        map_location="cpu",
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint payload must be a dictionary: {path}")
    return checkpoint_sha256


def _publish_downloaded_checkpoint(source: Path, *, cache_dir: Path, sha256: str) -> Path:
    """Publish validated download bytes once under their immutable digest."""

    digest_dir = cache_dir / "by-sha256"
    digest_dir.mkdir(parents=True, exist_ok=True)
    target = digest_dir / f"{sha256}.pt"
    lock_path = digest_dir / ".publish.lock"
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        if target.exists():
            validate_checkpoint(target, expected_sha256=sha256)
            return target
        if target.is_symlink():
            raise RuntimeError(f"Refusing symlink checkpoint cache target: {target}")
        os.chmod(source, 0o444)
        os.replace(source, target)
        with target.open("rb") as published_stream:
            os.fsync(published_stream.fileno())
        directory_fd = os.open(digest_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        validate_checkpoint(target, expected_sha256=sha256)
    return target


def _publish_local_checkpoint(source: Path, *, cache_dir: Path) -> Path:
    """Copy a local checkpoint through one stable no-follow descriptor.

    Returning the caller's mutable path would undo the resolver's exact-file
    contract for direct (non-W&B) launches.  Copy the verified bytes into the
    same content-addressed, read-only publication layout used for downloads.
    """

    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="local_", dir=cache_dir) as temp_dir:
        staged = Path(temp_dir) / "checkpoint.pt"
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(source, flags)
        except OSError as exc:
            raise OSError(
                f"Unable to open local checkpoint as a no-follow regular file: {source}: {exc}"
            ) from exc
        with os.fdopen(descriptor, "rb", closefd=True) as source_stream:
            initial_stat = os.fstat(source_stream.fileno())
            if not stat.S_ISREG(initial_stat.st_mode):
                raise ValueError(f"Local checkpoint is not a regular file: {source}")
            initial_identity = (
                initial_stat.st_dev,
                initial_stat.st_ino,
                initial_stat.st_size,
                initial_stat.st_mtime_ns,
                initial_stat.st_ctime_ns,
            )
            with staged.open("xb") as staged_stream:
                for chunk in iter(lambda: source_stream.read(4 * 1024 * 1024), b""):
                    staged_stream.write(chunk)
                staged_stream.flush()
                os.fsync(staged_stream.fileno())
            final_stat = os.fstat(source_stream.fileno())
            final_identity = (
                final_stat.st_dev,
                final_stat.st_ino,
                final_stat.st_size,
                final_stat.st_mtime_ns,
                final_stat.st_ctime_ns,
            )
            if final_identity != initial_identity:
                raise RuntimeError(
                    "Local checkpoint changed while it was copied into the immutable cache: "
                    f"path={source} before={initial_identity} after={final_identity}"
                )

        checkpoint_sha256 = validate_checkpoint(staged)
        return _publish_downloaded_checkpoint(
            staged,
            cache_dir=cache_dir,
            sha256=checkpoint_sha256,
        )


def resolve(ref: str, cache_root: Path) -> Path:
    try:
        run_path, file_name = parse_wandb_ref(ref)
    except ValueError as exc:
        if ref.startswith(("wandb://", "https://wandb.ai/")):
            raise SystemExit(f"[ERROR] Invalid exact checkpoint reference {ref!r}: {exc}") from exc
        local_path = Path(os.path.abspath(Path(ref).expanduser()))
        if local_path.suffix != ".pt":
            raise SystemExit(f"[ERROR] Expected an exact .pt checkpoint, got: {local_path}")
        local_cache_dir = Path(os.path.abspath(cache_root.expanduser())) / "local"
        return _publish_local_checkpoint(local_path, cache_dir=local_cache_dir)

    if not file_name.endswith(".pt"):
        raise SystemExit(f"[ERROR] Expected an exact .pt checkpoint, got: {file_name}")
    file_parts = file_name.split("/")
    if (
        file_name.startswith("/")
        or "\\" in file_name
        or any(part in {"", ".", ".."} for part in file_parts)
    ):
        raise SystemExit(
            f"[ERROR] W&B checkpoint file must be a safe relative path without traversal: {file_name!r}"
        )
    safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_path)
    cache_dir = cache_root.expanduser().resolve() / safe_run
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Keep the reference hash only as a diagnostic download-directory prefix;
    # the published path below is keyed by the actual checkpoint bytes.
    file_key = hashlib.sha256(file_name.encode("utf-8")).hexdigest()[:12]

    import wandb

    run = wandb.Api(timeout=60).run(run_path)
    # W&B run files are resolved afresh into a temporary directory.  A merely
    # loadable cache entry is not enough to prove that every node received the
    # currently pinned run/file bytes; the subsequent SHA256 provenance check
    # compares the staged result across all ranks.
    with tempfile.TemporaryDirectory(prefix=f"download_{file_key}_", dir=cache_dir) as temp_dir:
        temp_root = Path(temp_dir).resolve()
        with contextlib.redirect_stdout(sys.stderr):
            downloaded = run.file(file_name).download(root=temp_dir, replace=True)
        downloaded_path = Path(downloaded.name)
        if not downloaded_path.is_absolute():
            downloaded_path = (Path.cwd() / downloaded_path).resolve()
        else:
            downloaded_path = downloaded_path.resolve()
        if not downloaded_path.is_relative_to(temp_root):
            raise RuntimeError(
                "W&B checkpoint download escaped its private staging directory: "
                f"downloaded={downloaded_path} staging={temp_root}"
            )
        checkpoint_sha256 = validate_checkpoint(downloaded_path)
        target = _publish_downloaded_checkpoint(
            downloaded_path,
            cache_dir=cache_dir,
            sha256=checkpoint_sha256,
        )
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True)
    parser.add_argument("--cache-root", required=True, type=Path)
    args = parser.parse_args()
    print(resolve(args.ref, args.cache_root))


if __name__ == "__main__":
    main()
