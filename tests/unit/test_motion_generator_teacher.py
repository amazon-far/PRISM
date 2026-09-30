from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from motion_generator_teacher import (  # noqa: E402
    motion_generator_teacher_from_rollout_manifest,
    validate_motion_generator_teacher,
)


def _canonical_sha(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_rollout_manifest(path: Path, *, checkpoint_sha256: str) -> None:
    lineage = {
        "source_checkpoint_sha256": checkpoint_sha256,
        "checkpoint_source": "wandb://entity/project/run/model_05000.pt",
        "saved_wandb_path": "entity/project/run",
    }
    payload = {"teacher_lineage": lineage, "content": "fixture"}
    manifest = {
        "publication_id": _canonical_sha(payload),
        "publication_payload": payload,
        "teacher_lineage": lineage,
    }
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def test_authenticated_rollout_manifest_returns_generator_identity(tmp_path: Path) -> None:
    manifest = tmp_path / "realmesh_rollout_manifest.json"
    digest = "1" * 64
    _write_rollout_manifest(manifest, checkpoint_sha256=digest)

    assert motion_generator_teacher_from_rollout_manifest(manifest) == {
        "version": 1,
        "checkpoint_sha256": digest,
        "checkpoint_source": "wandb://entity/project/run/model_05000.pt",
        "saved_wandb_path": "entity/project/run",
    }


def test_legacy_rollout_manifest_requires_external_exact_binding(tmp_path: Path) -> None:
    manifest = tmp_path / "realmesh_rollout_manifest.json"
    manifest.write_text('{"legacy":true}\n', encoding="utf-8")

    assert motion_generator_teacher_from_rollout_manifest(manifest) is None


def test_rollout_lineage_cannot_self_assert_outside_publication_payload(tmp_path: Path) -> None:
    manifest = tmp_path / "realmesh_rollout_manifest.json"
    _write_rollout_manifest(manifest, checkpoint_sha256="2" * 64)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["teacher_lineage"]["source_checkpoint_sha256"] = "3" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="not authenticated"):
        motion_generator_teacher_from_rollout_manifest(manifest)


def test_generator_identity_rejects_partial_or_malformed_payload() -> None:
    with pytest.raises(ValueError, match="exactly"):
        validate_motion_generator_teacher(
            {"version": 1, "checkpoint_sha256": "4" * 64},
            role="fixture generator",
        )
    with pytest.raises(ValueError, match="lowercase hexadecimal"):
        validate_motion_generator_teacher(
            {
                "version": 1,
                "checkpoint_sha256": "NOT-A-DIGEST",
                "checkpoint_source": "wandb://entity/project/run/model.pt",
                "saved_wandb_path": None,
            },
            role="fixture generator",
        )
