#!/usr/bin/env python3
"""Canonical identity helpers for teachers that generated motion banks.

The policy used to generate a kinematic input motion and the policy queried for
distillation labels are separate roles.  This module makes the former an
authenticated, content-addressed part of every derived AS bank instead of
guessing it from the current training checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any


MOTION_GENERATOR_TEACHER_KEY = "motion_generator_teacher"
MOTION_GENERATOR_TEACHER_VERSION = 1
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_motion_generator_teacher(
    value: object,
    *,
    role: str,
) -> dict[str, Any]:
    """Return one canonical motion-generator identity or fail closed."""

    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a mapping")
    expected_keys = {
        "version",
        "checkpoint_sha256",
        "checkpoint_source",
        "saved_wandb_path",
    }
    if set(value) != expected_keys:
        raise ValueError(
            f"{role} must contain exactly {sorted(expected_keys)}; got {sorted(value)}"
        )
    version = value.get("version")
    digest = value.get("checkpoint_sha256")
    checkpoint_source = value.get("checkpoint_source")
    saved_wandb_path = value.get("saved_wandb_path")
    if version != MOTION_GENERATOR_TEACHER_VERSION:
        raise ValueError(f"{role} has unsupported version {version!r}")
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{role}.checkpoint_sha256 must be 64 lowercase hexadecimal characters")
    for field, field_value in (
        ("checkpoint_source", checkpoint_source),
        ("saved_wandb_path", saved_wandb_path),
    ):
        if field_value is not None and (
            not isinstance(field_value, str) or not field_value.strip()
        ):
            raise ValueError(f"{role}.{field} must be null or a non-empty string")
    if checkpoint_source is None and saved_wandb_path is None:
        raise ValueError(f"{role} must retain at least one checkpoint reference")
    return {
        "version": MOTION_GENERATOR_TEACHER_VERSION,
        "checkpoint_sha256": digest,
        "checkpoint_source": checkpoint_source,
        "saved_wandb_path": saved_wandb_path,
    }


def motion_generator_teacher_from_rollout_manifest(
    manifest_path: Path,
) -> dict[str, Any] | None:
    """Extract authenticated lineage from a teacher-rollout generation.

    Legacy rollout manifests did not record the generator checkpoint.  They
    return ``None`` and must be bound by an explicit exact SHA at launch.  A
    manifest that attempts to provide lineage but lacks the modern
    publication-payload digest is rejected rather than treated as legacy.
    """

    manifest_path = manifest_path.expanduser().resolve(strict=True)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not parse rollout manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Rollout manifest must be a mapping: {manifest_path}")
    lineage = manifest.get("teacher_lineage")
    if lineage is None:
        return None
    if not isinstance(lineage, dict):
        raise ValueError(f"Rollout teacher_lineage must be a mapping: {manifest_path}")

    publication_id = manifest.get("publication_id")
    publication_payload = manifest.get("publication_payload")
    if (
        not isinstance(publication_id, str)
        or _SHA256_PATTERN.fullmatch(publication_id) is None
        or not isinstance(publication_payload, dict)
        or _canonical_json_sha256(publication_payload) != publication_id
        or publication_payload.get("teacher_lineage") != lineage
    ):
        raise ValueError(
            "Rollout manifest teacher lineage is not authenticated by its canonical publication payload: "
            f"{manifest_path}"
        )

    digest = lineage.get("source_checkpoint_sha256")
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(
            f"Rollout teacher_lineage.source_checkpoint_sha256 is malformed: {manifest_path}"
        )
    checkpoint_source = lineage.get("checkpoint_source")
    saved_wandb_path = lineage.get("saved_wandb_path")
    return validate_motion_generator_teacher(
        {
            "version": MOTION_GENERATOR_TEACHER_VERSION,
            "checkpoint_sha256": digest,
            "checkpoint_source": checkpoint_source,
            "saved_wandb_path": saved_wandb_path,
        },
        role=f"rollout manifest {MOTION_GENERATOR_TEACHER_KEY}",
    )


def motion_generator_teacher_from_solid_manifest(
    manifest: object,
    *,
    role: str,
) -> dict[str, Any] | None:
    """Read generator lineage sealed into an immutable solid manifest."""

    if not isinstance(manifest, dict):
        raise ValueError(f"{role} must be a mapping")
    source_identity = manifest.get("source_identity")
    if not isinstance(source_identity, dict):
        raise ValueError(f"{role}.source_identity must be a mapping")
    if MOTION_GENERATOR_TEACHER_KEY not in source_identity:
        return None
    value = source_identity[MOTION_GENERATOR_TEACHER_KEY]
    if value is None:
        return None
    return validate_motion_generator_teacher(
        value,
        role=f"{role}.source_identity.{MOTION_GENERATOR_TEACHER_KEY}",
    )
