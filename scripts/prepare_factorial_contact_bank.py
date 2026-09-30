#!/usr/bin/env python3
"""Bind real-mesh contact sidecars to the exact final40K rollout command bank."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from validate_contact_sidecars import _validate_contact_arrays

TEACHER_SHA = "14e644323b8e6a7b769dbf641d9f625bee895742b7064368b55afd0710a0d665"
REGIONS = ("left_wrist", "right_wrist", "left_elbow", "right_elbow", "left_wrist_roll", "right_wrist_roll", "left_wrist_pitch", "right_wrist_pitch", "torso")
FIELDS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_commands(command, phase):
    if command.ndim != 2 or command.shape[1] != 3 or phase.shape != command.shape[:1]:
        raise ValueError("Precomputed command/phase shape mismatch")
    if not np.issubdtype(phase.dtype, np.integer) or not np.isfinite(command).all() or not np.isin(phase, [0, 1, 2]).all():
        raise ValueError("Invalid command values or phases")
    if np.any(command[:, 1] != 0) or np.any((command[:, 0] != 0) & (command[:, 2] != 0)):
        raise ValueError("Lateral command or simultaneous forward/yaw")
    if np.any(command[phase == 0] != 0) or np.any(command[phase == 1, 2] != 0) or np.any(command[phase == 2, 0] != 0):
        raise ValueError("Command phase semantics mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--raw-bank", type=Path, required=True)
    parser.add_argument("--command-bank", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    RAW, BANK, OUTPUT = args.raw_bank.resolve(), args.command_bank.resolve(), args.output_root.resolve()
    if sha(RAW / "manifest.json") != "418ae32cd8f16ec1d427b05d67f56ea04d93d1acd1f0bd417ebd32f7d248060e":
        raise ValueError("Wrong raw final40K bank")
    if sha(BANK / "manifest.json") != "1047c5c3a2aba297127683054d9d0cea33fda43ea171b964e7ab9cb6c5b150e7":
        raise ValueError("Wrong precomputed final40K bank")
    raw = json.loads((RAW / "manifest.json").read_text())
    commands = json.loads((BANK / "manifest.json").read_text())
    lineage = raw["source_identity"]
    if lineage["checkpoint"]["pt_sha256"] != TEACHER_SHA:
        raise ValueError("Wrong rollout producer")
    source_records = {row["clip_id"]: row for row in lineage["records"]}
    command_records = {row["clip_id"]: row for row in commands["clips"]}
    summaries = sorted(args.source.glob("shard_*/summary.json"))
    if len(summaries) != 8 or any(json.loads(p.read_text())["source_checkpoint_sha256"] != TEACHER_SHA for p in summaries):
        raise ValueError("Missing or mismatched contact producer shard")
    metadata_paths = sorted(args.source.glob("shard_*/clips/*/metadata.json"))
    seen, records, copies, clip_proofs = set(), [], [], []
    phase_counts = np.zeros(3, dtype=np.int64)
    for path in metadata_paths:
        metadata = json.loads(path.read_text())
        clip = metadata["clip_id"]
        if clip in seen or clip not in source_records or clip not in command_records:
            raise ValueError("Duplicate or unknown contact clip: " + clip)
        seen.add(clip)
        if metadata["contact_surface_projection"] != "mesh":
            raise ValueError("Non-mesh contact projection: " + clip)
        if metadata["teacher_rollout_valid_step_count"] != 359:
            raise ValueError("Contact timeline differs: " + clip)
        reference = path.parent / "teacher_rollout_reference.npz"
        if sha(reference) != source_records[clip]["source_rollout_reference"]["sha256"]:
            raise ValueError("Contact rollout does not match published trajectory: " + clip)
        motion = path.parents[2] / "motion_bank" / (clip + ".npz")
        if sha(motion) != source_records[clip]["source_motion"]["sha256"]:
            raise ValueError("Source motion changed: " + clip)
        command_path = BANK / (clip + ".npz")
        if sha(command_path) != command_records[clip]["derived_npz_sha256"]:
            raise ValueError("Command file changed: " + clip)
        with np.load(motion, allow_pickle=False) as original, np.load(RAW / (clip + ".npz"), allow_pickle=False) as published, np.load(command_path, allow_pickle=False) as current:
            for field in FIELDS:
                if not np.array_equal(original[field], published[field]) or not np.array_equal(published[field], current[field]):
                    raise ValueError(f"Trajectory changed: {clip}/{field}")
            validate_commands(current["policy_command_xy_yaw"], current["policy_command_phase"])
            phase_counts += np.bincount(current["policy_command_phase"], minlength=3)
        for region in REGIONS:
            _validate_contact_arrays(path.parent, region, metadata["num_steps"])
        required = [path, path.parent / "contact_intervals.json"]
        required += sorted(path.parent.glob("*.npy"))
        for source in required:
            if not source.is_file() or source.is_symlink():
                raise ValueError(f"Missing regular contact asset: {source}")
            relative = f"clips/{clip}/{source.name}"
            records.append({"path": relative, "size": source.stat().st_size, "sha256": sha(source)})
            copies.append((source, relative))
        clip_proofs.append({"clip_id": clip, "reference_sha256": sha(reference), "metadata_sha256": sha(path),
                            "command_sha256": sha(command_path), "stable_contact_success": metadata["stable_contact_success"]})
    if len(seen) != 137 or seen != set(source_records) or seen != set(command_records):
        raise ValueError("Contact bank is not exact137")
    if phase_counts.tolist() != commands["total_phase_counts"]:
        raise ValueError("Command phase count mismatch")
    payload = {"schema_version": 1, "semantics": "mesh_contact_sidecars_from_teacher_rollouts",
               "producer_checkpoint_sha256": TEACHER_SHA, "clip_count": 137,
               "raw_manifest_sha256": sha(RAW / "manifest.json"), "command_manifest_sha256": sha(BANK / "manifest.json"),
               "source_audit_root": str(args.source.resolve().parent),
               "records": sorted(records, key=lambda x: x["path"]), "clip_proofs": sorted(clip_proofs, key=lambda x: x["clip_id"]),
               "phase_counts_zero_forward_yaw": phase_counts.tolist(), "trajectory_fields_exactly_preserved": list(FIELDS)}
    digest = canonical_sha(payload)
    manifest = {**payload, "payload_digest": digest}
    target = OUTPUT / digest
    if target.exists():
        if json.loads((target / "manifest.json").read_text()) != manifest:
            raise ValueError("Existing contact generation differs")
        for row in records:
            if sha(target / row["path"]) != row["sha256"]:
                raise ValueError("Existing contact payload changed")
    else:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        staging = OUTPUT / ("." + digest + ".staging-" + str(os.getpid()))
        staging.mkdir()
        for source, relative in copies:
            dest = staging / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        for file in staging.rglob("*"):
            if file.is_file():
                with file.open("rb") as stream:
                    os.fsync(stream.fileno())
                file.chmod(0o444)
        for directory in sorted([p for p in staging.rglob("*") if p.is_dir()], key=lambda p: len(p.parts), reverse=True) + [staging]:
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            directory.chmod(0o555)
        staging.rename(target)
        fd = os.open(OUTPUT, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    print(json.dumps({"contact_root": str(target), "manifest_sha256": sha(target / "manifest.json"), "clip_count": len(seen), "phase_counts": phase_counts.tolist()}, sort_keys=True))


if __name__ == "__main__":
    main()
