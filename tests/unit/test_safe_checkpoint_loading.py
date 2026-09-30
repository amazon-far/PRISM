from __future__ import annotations

import pickle
import re
import stat
import subprocess
from pathlib import Path

import pytest
import torch

from scripts.compute_training_provenance import _teacher_contract
from scripts.resolve_exact_checkpoint import (
    _publish_downloaded_checkpoint,
    resolve,
    validate_checkpoint,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRODUCTION_CHECKPOINT_SURFACES = (
    "scripts/resolve_exact_checkpoint.py",
    "scripts/compute_training_provenance.py",
    "scripts/box23k_policy_init_preflight.py",
    "src/holosoma/holosoma/utils/eval_utils.py",
    "src/holosoma/holosoma/agents/ppo/ppo.py",
)


class _UnsafeCheckpointPayload:
    """Create a marker if a checkpoint reader grants arbitrary pickle access."""

    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __reduce__(self):
        return subprocess.call, (["touch", self.marker],)


@pytest.mark.parametrize("reader", [validate_checkpoint, _teacher_contract])
def test_launcher_checkpoint_readers_never_execute_pickle_globals(tmp_path, reader):
    marker = tmp_path / "pickle_executed"
    checkpoint = tmp_path / "unsafe.pt"
    torch.save(
        {
            "experiment_config": {},
            "actor_model_state_dict": {},
            "unused": _UnsafeCheckpointPayload(str(marker)),
        },
        checkpoint,
    )

    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        reader(checkpoint)
    assert not marker.exists()


@pytest.mark.parametrize("relative_path", _PRODUCTION_CHECKPOINT_SURFACES)
def test_production_checkpoint_surfaces_use_verified_weights_only_loader(relative_path):
    source = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")

    assert re.search(r"torch\s*\.\s*load\s*\(", source) is None
    assert "load_verified_torch_checkpoint" in source


def test_resolved_download_is_published_by_content_digest(tmp_path: Path) -> None:
    downloaded = tmp_path / "downloaded.pt"
    torch.save({"actor_model_state_dict": {"weight": torch.tensor([1.0])}}, downloaded)
    digest = validate_checkpoint(downloaded)

    published = _publish_downloaded_checkpoint(
        downloaded,
        cache_dir=tmp_path / "cache",
        sha256=digest,
    )

    assert published.name == f"{digest}.pt"
    assert published.parent.name == "by-sha256"
    assert not downloaded.exists()
    assert validate_checkpoint(published, expected_sha256=digest) == digest
    assert stat.S_IMODE(published.stat().st_mode) == 0o444


def test_resolved_download_never_replaces_existing_digest_entry(tmp_path: Path) -> None:
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    payload = {"actor_model_state_dict": {"weight": torch.tensor([2.0])}}
    torch.save(payload, first)
    second.write_bytes(first.read_bytes())
    digest = validate_checkpoint(first)
    published = _publish_downloaded_checkpoint(first, cache_dir=tmp_path / "cache", sha256=digest)
    before_identity = published.stat()

    reused = _publish_downloaded_checkpoint(second, cache_dir=tmp_path / "cache", sha256=digest)

    after_identity = reused.stat()
    assert reused == published
    assert (after_identity.st_dev, after_identity.st_ino) == (
        before_identity.st_dev,
        before_identity.st_ino,
    )
    assert validate_checkpoint(reused, expected_sha256=digest) == digest


def test_local_resolution_publishes_immutable_content_addressed_copy(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    torch.save({"actor_model_state_dict": {"weight": torch.tensor([3.0])}}, source)
    source_digest = validate_checkpoint(source)

    published = resolve(str(source), tmp_path / "cache")
    source.write_bytes(b"changed after resolution")

    assert published != source
    assert published.name == f"{source_digest}.pt"
    assert published.parent.name == "by-sha256"
    assert validate_checkpoint(published, expected_sha256=source_digest) == source_digest
    assert stat.S_IMODE(published.stat().st_mode) == 0o444


def test_local_resolution_rejects_symlink_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    alias = tmp_path / "alias.pt"
    torch.save({"actor_model_state_dict": {}}, source)
    alias.symlink_to(source)

    with pytest.raises(OSError, match="no-follow regular file"):
        resolve(str(alias), tmp_path / "cache")


@pytest.mark.parametrize(
    "reference",
    (
        "wandb://entity/project/run/../model.pt",
        "wandb://entity/project/run/subdir//model.pt",
        r"wandb://entity/project/run/subdir\model.pt",
    ),
)
def test_wandb_resolution_rejects_unsafe_file_paths_before_download(
    tmp_path: Path,
    reference: str,
) -> None:
    with pytest.raises(SystemExit, match="safe relative path"):
        resolve(reference, tmp_path / "cache")
