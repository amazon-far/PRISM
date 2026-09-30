from __future__ import annotations

import copy
import dataclasses
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pytest
import torch

from holosoma.train_agent import (
    _emit_batch_worker_preflight_ready,
    _preflight_checkpoint_lineage_before_sim,
    _preflight_cross_rank_provenance_before_sim,
    _preflight_data_assets_before_sim,
    _validate_hierarchical_small_collectives_launch_contract,
)
from holosoma.utils.atomic_output import emit_atomic_stdout_record
from holosoma.utils.training_provenance import (
    ALLOW_LEGACY_UNVERIFIED_TEACHER_LOAD_ENV,
    ENV_NAME,
    EXECUTION_RUNTIME_KEY,
    MOTION_GENERATOR_TEACHER_SHA256_KEY,
    RUNTIME_ASSET_DIGEST_KEY,
    RUNTIME_ASSET_MANIFEST_KEY,
    RUNTIME_ASSET_PHASE_FINAL,
    RUNTIME_ASSET_PHASE_KEY,
    REQUIRE_MOTION_GENERATOR_TEACHER_MATCH_KEY,
    TEACHER_ENABLED_KEY,
    TRAINING_REGIME_DISTILLATION,
    TRAINING_REGIME_KEY,
    TRAINING_REGIME_PURE_RL,
    _execution_runtime_binding_from_environ,
    checkpoint_lineage_enabled,
    disabled_checkpoint_sha256,
    disabled_teacher_sha256,
    embedded_runtime_asset_manifest_sha256,
    training_provenance_from_env,
    validate_execution_runtime_binding,
    validate_training_provenance,
)
import scripts.compute_training_provenance as provenance_module
from scripts.compute_training_provenance import (
    PYTHON_RUNTIME_MANIFEST_SHA256_ENV,
    PYTHON_RUNTIME_SITEPACKAGES_ENV,
    SEMANTIC_ENVIRONMENT_FIELDS,
    SEMANTIC_ENVIRONMENT_KEY,
    SOURCE_MANIFEST_SHA256_ENV,
    SOURCE_SNAPSHOT_ID_ENV,
    _environment_metadata,
    compute_generalist_provenance,
    compute_provenance,
    revalidate_data_asset_provenance,
    revalidate_data_asset_provenance_cached,
)
from scripts.prepare_as_rank_shards import prepare_rank_shards


@pytest.fixture(autouse=True)
def _clear_source_snapshot_identity(monkeypatch):
    monkeypatch.delenv(SOURCE_SNAPSHOT_ID_ENV, raising=False)
    monkeypatch.delenv(SOURCE_MANIFEST_SHA256_ENV, raising=False)
    monkeypatch.setenv("TORCH_DIST_BACKEND", "gloo")
    for name in (
        ALLOW_LEGACY_UNVERIFIED_TEACHER_LOAD_ENV,
        "NCCL_LIB_SHA256",
        "PYTHONHASHSEED",
        "CUBLAS_WORKSPACE_CONFIG",
        PYTHON_RUNTIME_SITEPACKAGES_ENV,
        PYTHON_RUNTIME_MANIFEST_SHA256_ENV,
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
        "HOLOSOMA_GLOO_BARRIER",
        "HOLOSOMA_GLOO_GRAD_REDUCE",
        "HOLOSOMA_GLOO_SMALL_COLLECTIVES",
        "HOLOSOMA_HIERARCHICAL_GRAD_REDUCE",
        "HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES",
        "HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER",
        "HOLOSOMA_RANK_VISIBLE_DEVICES",
        "HOLOSOMA_CONTIGUOUS_MINIBATCHES",
        "HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY",
        "HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP",
        "HOLOSOMA_DAGGER_SUPERVISED_ONLY",
        "HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH",
        "HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD",
        "HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC",
        "NPROC",
        "NNODES",
        "HOLOSOMA_LAUNCH_TOKEN",
        "HOLOSOMA_LAUNCH_EPOCH",
        "HOLOSOMA_ORIGINAL_LOCAL_RANK",
        "HOLOSOMA_ORIGINAL_LOCAL_WORLD_SIZE",
        "HOLOSOMA_ORIGINAL_CUDA_VISIBLE_DEVICES",
        "HOLOSOMA_RANK_VISIBLE_PHYSICAL_DEVICE",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "NODE_RANK",
        "CUDA_VISIBLE_DEVICES",
        ENV_NAME,
    ):
        monkeypatch.delenv(name, raising=False)
    for name in SEMANTIC_ENVIRONMENT_FIELDS:
        monkeypatch.delenv(name, raising=False)


class _WorkerReadyDist:
    def __init__(self) -> None:
        self.barriers: list[dict] = []

    @staticmethod
    def is_initialized() -> bool:
        return True

    def barrier(self, **kwargs) -> None:
        self.barriers.append(kwargs)


class _DescriptorStdout:
    def __init__(self, descriptor: int) -> None:
        self.descriptor = descriptor
        self.flush_count = 0

    def fileno(self) -> int:
        return self.descriptor

    def flush(self) -> None:
        self.flush_count += 1

    def write(self, _value: str) -> int:
        raise AssertionError("descriptor-backed launch records must use one os.write")


def test_atomic_stdout_record_uses_one_portably_atomic_write(monkeypatch):
    stream = _DescriptorStdout(37)
    writes: list[tuple[int, bytes]] = []

    def fake_write(descriptor: int, payload: bytes) -> int:
        writes.append((descriptor, payload))
        return len(payload)

    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(os, "write", fake_write)

    emit_atomic_stdout_record("[INFO] final_worker_preflight_verified rank=7")

    assert stream.flush_count == 1
    assert writes == [(37, b"\n[INFO] final_worker_preflight_verified rank=7\n")]
    assert len(writes[0][1]) <= 512


@pytest.mark.parametrize(
    "record",
    ["", "two\nlines", "carriage\rreturn", "x" * 511, "é" * 256],
)
def test_atomic_stdout_record_rejects_noncanonical_or_oversized_lines(record):
    with pytest.raises(ValueError):
        emit_atomic_stdout_record(record)


def test_atomic_stdout_record_portable_boundary_includes_both_newlines(monkeypatch):
    stream = _DescriptorStdout(37)
    writes: list[bytes] = []

    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(
        os,
        "write",
        lambda _descriptor, payload: writes.append(payload) or len(payload),
    )

    emit_atomic_stdout_record("x" * 510)

    assert writes == [b"\n" + b"x" * 510 + b"\n"]
    assert len(writes[0]) == 512


def test_atomic_stdout_record_starts_new_line_after_unterminated_pipe_fragment(monkeypatch):
    read_descriptor, write_descriptor = os.pipe()
    try:
        monkeypatch.setattr(sys, "stdout", _DescriptorStdout(write_descriptor))
        os.write(write_descriptor, b"prior-partial")
        emit_atomic_stdout_record("[INFO] marker")
        os.close(write_descriptor)
        write_descriptor = -1

        assert os.read(read_descriptor, 4096) == b"prior-partial\n[INFO] marker\n"
    finally:
        os.close(read_descriptor)
        if write_descriptor >= 0:
            os.close(write_descriptor)


def test_atomic_stdout_record_rejects_short_capture_stream_write(monkeypatch):
    class _ShortCapture:
        def fileno(self) -> int:
            raise OSError("no descriptor")

        def write(self, value: str) -> int:
            return len(value) - 1

        def flush(self) -> None:
            raise AssertionError("a short fallback write must fail before flush")

    monkeypatch.setattr(sys, "stdout", _ShortCapture())

    with pytest.raises(RuntimeError, match="fallback record was only partially written"):
        emit_atomic_stdout_record("[INFO] marker")


def test_cross_rank_provenance_marker_is_atomically_reemitted_by_worker(monkeypatch):
    provenance = {"training_regime": "distillation"}
    marker = (
        "[INFO] cross_rank_training_provenance_verified "
        "world_size=104 training_regime=distillation"
    )
    emitted: list[str] = []
    monkeypatch.setenv("WORLD_SIZE", "104")
    monkeypatch.setenv("MASTER_PORT", "29881")

    with (
        patch(
            "holosoma.train_agent.training_provenance_from_env",
            return_value=provenance,
        ),
        patch(
            "holosoma.train_agent.canonical_training_provenance_json",
            return_value="{}",
        ),
        patch(
            "holosoma.train_agent.subprocess.run",
            return_value=subprocess.CompletedProcess(
                [],
                0,
                f"\n[WARN] child diagnostic\n\n{marker}\n",
                "",
            ),
        ),
        patch(
            "holosoma.train_agent.emit_atomic_stdout_record",
            side_effect=emitted.append,
        ),
    ):
        assert _preflight_cross_rank_provenance_before_sim() == provenance

    assert emitted == [marker]


def test_final_worker_ready_is_launch_bound_rank_unique_and_after_barrier(monkeypatch, capsys):
    token = "a" * 64
    snapshot = f"src-{'b' * 64}"
    monkeypatch.setenv("HOLOSOMA_LAUNCH_TOKEN", token)
    monkeypatch.setenv("HOLOSOMA_LAUNCH_EPOCH", "1712345678")
    monkeypatch.setenv("HOLOSOMA_SOURCE_SNAPSHOT_ID", snapshot)
    monkeypatch.setenv("RANK", "13")
    monkeypatch.setenv("WORLD_SIZE", "16")
    # Rank-visible execution remaps CUDA LOCAL_RANK to zero, but the controller
    # must bind the marker to torchrun's original node-local rank.
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("HOLOSOMA_ORIGINAL_LOCAL_RANK", "5")
    monkeypatch.setenv("HOLOSOMA_ORIGINAL_LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("NPROC", "8")
    monkeypatch.setenv("NNODES", "2")
    monkeypatch.setenv("NODE_RANK", "1")
    dist = _WorkerReadyDist()

    emitted = _emit_batch_worker_preflight_ready(
        dist_module=dist,
        distributed_conf={"global_rank": 13, "local_rank": 0, "world_size": 16},
    )

    assert emitted is True
    assert dist.barriers == [{"device_ids": [0]}]
    assert capsys.readouterr().out.strip() == (
        "[INFO] final_worker_preflight_verified "
        "global_rank=13 local_rank=5 world_size=16 "
        f"source_snapshot={snapshot} launch_token={token} launch_epoch=1712345678"
    )


def test_final_worker_ready_skips_non_batch_and_rejects_partial_identity(monkeypatch):
    dist = _WorkerReadyDist()
    assert _emit_batch_worker_preflight_ready(dist_module=dist, distributed_conf=None) is False
    assert dist.barriers == []

    monkeypatch.setenv("HOLOSOMA_LAUNCH_TOKEN", "a" * 64)
    with pytest.raises(RuntimeError, match="HOLOSOMA_LAUNCH_EPOCH"):
        _emit_batch_worker_preflight_ready(dist_module=dist, distributed_conf=None)
    assert dist.barriers == []


def test_final_worker_ready_rejects_inconsistent_batch_topology(monkeypatch):
    monkeypatch.setenv("HOLOSOMA_LAUNCH_TOKEN", "a" * 64)
    monkeypatch.setenv("HOLOSOMA_LAUNCH_EPOCH", "1712345678")
    monkeypatch.setenv("HOLOSOMA_SOURCE_SNAPSHOT_ID", f"src-{'b' * 64}")
    monkeypatch.setenv("RANK", "13")
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("HOLOSOMA_ORIGINAL_LOCAL_RANK", "5")
    monkeypatch.setenv("HOLOSOMA_ORIGINAL_LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("NPROC", "8")
    monkeypatch.setenv("NNODES", "2")
    monkeypatch.setenv("NODE_RANK", "0")

    with pytest.raises(RuntimeError, match="topology is inconsistent"):
        _emit_batch_worker_preflight_ready(
            dist_module=_WorkerReadyDist(),
            distributed_conf={"global_rank": 13, "local_rank": 0, "world_size": 16},
        )


def _teacher(path, *, motion_ends=True, actions=True):
    terms = {"actions": {"func": "pkg:actions"}} if actions else {"dof_pos": {"func": "pkg:dof_pos"}}
    termination_terms = {
        "timeout": {"func": "pkg:timeout_exceeded", "params": {}, "is_timeout": True}
    }
    if motion_ends:
        termination_terms["motion_ends"] = {"func": "pkg:motion_ends", "params": {}, "is_timeout": False}
    torch.save(
        {
            "experiment_config": {
                "algo": {"config": {"module_dict": {"actor": {"input_dim": ["actor_obs"]}}}},
                "observation": {"groups": {"actor_obs": {"terms": terms}}},
                "termination": {"terms": termination_terms},
            }
        },
        path,
    )


def _fixture(tmp_path):
    teacher = tmp_path / "teacher.pt"
    _teacher(teacher)
    motion = tmp_path / "motion"
    motion.mkdir()
    np.savez(
        motion / "clip_a.npz",
        fps=np.asarray([50.0]),
        body_pos_w=np.zeros((2, 1, 3), dtype=np.float32),
    )
    asset_dir = motion / "assets"
    asset_dir.mkdir()
    mesh = asset_dir / "object.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    urdf = asset_dir / "object.urdf"
    urdf.write_text(
        "<robot name='object'><link name='base'><visual><geometry>"
        "<mesh filename='object.obj'/></geometry></visual><collision><geometry>"
        "<mesh filename='object.obj'/></geometry></collision></link></robot>",
        encoding="utf-8",
    )
    object_map = motion / "_clip_object_urdf_map.json"
    object_map.write_text(
        json.dumps(
            {
                "clips": {
                    "clip_a": {
                        "object_urdf_path": "assets/object.urdf",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    clip_dir = tmp_path / "contact" / "clips" / "0000_clip_a"
    clip_dir.mkdir(parents=True)
    for side in ("left_wrist", "right_wrist"):
        np.save(clip_dir / f"{side}_contact_points.npy", np.zeros((1, 3), dtype=np.float32))
        np.save(clip_dir / f"{side}_contact_point_counts.npy", np.ones((1,), dtype=np.int32))
        np.save(clip_dir / f"{side}_contact_interval_steps.npy", np.asarray([0, 1], dtype=np.int32))
    np.savez(clip_dir / "teacher_rollout_reference.npz", valid_steps=np.asarray([True]))
    source_root = tmp_path / "source"
    package_root = source_root / "src" / "holosoma" / "holosoma"
    package_root.mkdir(parents=True)
    (package_root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    inference_root = source_root / "src" / "holosoma_inference" / "holosoma_inference"
    inference_root.mkdir(parents=True)
    (inference_root / "policy.py").write_text("POLICY_VERSION = 1\n", encoding="utf-8")
    return teacher, motion, object_map, tmp_path / "contact", source_root


def _compute(
    teacher,
    motion,
    object_map,
    contact,
    source_root,
    *,
    policy_init_checkpoint=None,
    training_resume_checkpoint=None,
    motion_shard_manifest=None,
):
    return compute_provenance(
        teacher_checkpoint=teacher,
        motion_dir=motion,
        object_map=object_map,
        contact_root=contact,
        motion_shard_manifest=motion_shard_manifest,
        student_motion_end_mode="episodic",
        contact_interval_runtime_prepend_compensation=True,
        source_root=source_root,
        policy_init_checkpoint=policy_init_checkpoint,
        training_resume_checkpoint=training_resume_checkpoint,
    )


def _finalized(provenance):
    runtime_manifest = {"version": 2, "fixture": "data-asset-revalidation"}
    return {
        **provenance,
        RUNTIME_ASSET_PHASE_KEY: RUNTIME_ASSET_PHASE_FINAL,
        RUNTIME_ASSET_DIGEST_KEY: embedded_runtime_asset_manifest_sha256(
            runtime_manifest
        ),
        RUNTIME_ASSET_MANIFEST_KEY: runtime_manifest,
    }


def test_motion_generator_teacher_binding_is_recorded_without_breaking_legacy_v2(
    tmp_path,
    monkeypatch,
):
    teacher, motion, object_map, contact, source_root = _fixture(tmp_path)
    monkeypatch.delenv("HOLOSOMA_EXTERNAL_AS_MOTION_GENERATOR_TEACHER_SHA256", raising=False)
    monkeypatch.delenv("REQUIRE_MOTION_GENERATOR_TEACHER_MATCH", raising=False)
    legacy = _compute(teacher, motion, object_map, contact, source_root)
    assert MOTION_GENERATOR_TEACHER_SHA256_KEY not in legacy
    assert REQUIRE_MOTION_GENERATOR_TEACHER_MATCH_KEY not in legacy
    validate_training_provenance(legacy)

    monkeypatch.setenv(
        "HOLOSOMA_EXTERNAL_AS_MOTION_GENERATOR_TEACHER_SHA256",
        legacy["teacher_sha256"],
    )
    monkeypatch.setenv("REQUIRE_MOTION_GENERATOR_TEACHER_MATCH", "1")
    bound = _compute(teacher, motion, object_map, contact, source_root)
    assert bound[MOTION_GENERATOR_TEACHER_SHA256_KEY] == legacy["teacher_sha256"]
    assert bound[REQUIRE_MOTION_GENERATOR_TEACHER_MATCH_KEY] is True
    validate_training_provenance(bound)

    monkeypatch.setenv(
        "HOLOSOMA_EXTERNAL_AS_MOTION_GENERATOR_TEACHER_SHA256",
        "0" * 64,
    )
    mismatch = _compute(teacher, motion, object_map, contact, source_root)
    with pytest.raises(ValueError, match="requires.*match.*differ"):
        validate_training_provenance(mismatch)

    mismatch[REQUIRE_MOTION_GENERATOR_TEACHER_MATCH_KEY] = False
    validate_training_provenance(mismatch)


def _minimal_finalized_execution_runtime_provenance():
    runtime_manifest = {"version": 2, "fixture": "execution-runtime-binding"}
    return validate_training_provenance(
        {
            "version": 2,
            TRAINING_REGIME_KEY: TRAINING_REGIME_DISTILLATION,
            TEACHER_ENABLED_KEY: True,
            "teacher_sha256": "a" * 64,
            "policy_init_sha256": disabled_checkpoint_sha256("policy_init"),
            "training_resume_sha256": disabled_checkpoint_sha256("training_resume"),
            "motion_shard_manifest_sha256": "b" * 64,
            "contact_sidecar_manifest_sha256": "c" * 64,
            "source_bundle_sha256": "d" * 64,
            RUNTIME_ASSET_PHASE_KEY: RUNTIME_ASSET_PHASE_FINAL,
            RUNTIME_ASSET_DIGEST_KEY: embedded_runtime_asset_manifest_sha256(
                runtime_manifest
            ),
            RUNTIME_ASSET_MANIFEST_KEY: runtime_manifest,
            "policy_init_enabled": False,
            "training_resume_enabled": False,
            "environment": _environment_metadata(),
        },
        require_finalized=True,
    )


def test_execution_runtime_live_binding_covers_every_generated_cross_rank_field():
    recorded = _environment_metadata()[EXECUTION_RUNTIME_KEY]
    live = _execution_runtime_binding_from_environ(os.environ)
    separately_bound = {
        "PYTHONHASHSEED",
        "CUBLAS_WORKSPACE_CONFIG",
        SEMANTIC_ENVIRONMENT_KEY,
    }

    assert set(live) == set(recorded).difference(separately_bound)
    assert live == {name: recorded[name] for name in live}


def test_training_provenance_from_env_binds_normalized_execution_runtime_without_rank_local_false_positives(
    monkeypatch,
):
    monkeypatch.setenv("TORCH_DIST_BACKEND", " GLOO ")
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")
    monkeypatch.setenv("HOLOSOMA_GLOO_BARRIER", "on")
    monkeypatch.setenv("HOLOSOMA_GLOO_GRAD_REDUCE", "true")
    monkeypatch.setenv("HOLOSOMA_GLOO_SMALL_COLLECTIVES", "1")
    monkeypatch.setenv("HOLOSOMA_RANK_VISIBLE_DEVICES", "true")
    monkeypatch.setenv("HOLOSOMA_DAGGER_SUPERVISED_ONLY", "yes")
    monkeypatch.setenv("HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP", "on")
    monkeypatch.setenv("HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH", "0016")
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC", "00300")
    monkeypatch.setenv("HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD", "true")
    monkeypatch.setenv("NPROC", "08")
    monkeypatch.setenv("NNODES", "02")
    provenance = _minimal_finalized_execution_runtime_provenance()

    # Equivalent spellings normalize to the same execution contract.
    monkeypatch.setenv("TORCH_DIST_BACKEND", "gloo")
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")
    monkeypatch.setenv("HOLOSOMA_GLOO_BARRIER", "YES")
    monkeypatch.setenv("HOLOSOMA_GLOO_GRAD_REDUCE", "on")
    monkeypatch.setenv("HOLOSOMA_GLOO_SMALL_COLLECTIVES", "true")
    monkeypatch.setenv("HOLOSOMA_RANK_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("HOLOSOMA_DAGGER_SUPERVISED_ONLY", "1")
    monkeypatch.setenv("HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP", "YES")
    monkeypatch.setenv("HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH", "16")
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC", "300")
    monkeypatch.setenv("HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD", "1")
    monkeypatch.setenv("NPROC", "8")
    monkeypatch.setenv("NNODES", "2")

    # torchrun and train_agent_rank_visible legitimately specialize these for
    # each worker; none are process-wide provenance fields.
    monkeypatch.setenv("RANK", "13")
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("NODE_RANK", "1")
    monkeypatch.setenv("HOLOSOMA_ORIGINAL_LOCAL_RANK", "5")
    monkeypatch.setenv("HOLOSOMA_ORIGINAL_LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")

    # These two are checked at the pre-interpreter/pre-CUDA boundary by
    # train_agent, rather than duplicated by the ambient live-binding helper.
    monkeypatch.setenv("PYTHONHASHSEED", "0007")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv(ENV_NAME, json.dumps(provenance))

    assert training_provenance_from_env() == provenance


@pytest.mark.parametrize(
    ("name", "before", "after"),
    [
        ("HOLOSOMA_DAGGER_SUPERVISED_ONLY", "0", "1"),
        ("HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP", "0", "1"),
        ("HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH", "16", "8"),
        ("HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD", "1", "0"),
        ("HOLOSOMA_GLOO_BARRIER", "0", "1"),
        ("HOLOSOMA_GLOO_GRAD_REDUCE", "0", "1"),
        ("HOLOSOMA_GLOO_SMALL_COLLECTIVES", "0", "1"),
        ("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE", "0", "1"),
        ("HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES", "0", "1"),
        ("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER", "0", "1"),
        ("HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC", "300", "301"),
        ("HOLOSOMA_RANK_VISIBLE_DEVICES", "0", "1"),
        ("HOLOSOMA_CONTIGUOUS_MINIBATCHES", "0", "1"),
        ("HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY", "0", "1"),
        ("HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC", "0", "1"),
        ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0", "1"),
        ("TORCH_DIST_BACKEND", "gloo", "nccl"),
        ("NCCL_LIB_SHA256", "a" * 64, "b" * 64),
        ("NPROC", "8", "4"),
        ("NNODES", "2", "1"),
    ],
)
def test_training_provenance_from_env_rejects_execution_runtime_drift(
    monkeypatch,
    name,
    before,
    after,
):
    # Keeping a valid digest present makes both Gloo/NCCL and hierarchical
    # transition cases independently reach the binding comparison.
    monkeypatch.setenv("NCCL_LIB_SHA256", "a" * 64)
    if name == "HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES":
        monkeypatch.setenv("HOLOSOMA_GLOO_SMALL_COLLECTIVES", "1")
        monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE", "1")
    monkeypatch.setenv(name, before)
    provenance = _minimal_finalized_execution_runtime_provenance()
    monkeypatch.setenv(name, after)
    monkeypatch.setenv(ENV_NAME, json.dumps(provenance))

    with pytest.raises(
        ValueError,
        match=rf"execution runtime changed.*{name}",
    ):
        training_provenance_from_env()


@pytest.mark.parametrize(
    ("name", "recorded", "expected"),
    [
        ("HOLOSOMA_DAGGER_SUPERVISED_ONLY", "0", "must be a boolean"),
        ("HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP", None, "must be a boolean"),
        ("HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH", "16", "must be an integer"),
        ("HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC", "300", "must be an integer"),
        ("TORCH_DIST_BACKEND", "GLOO", "must be 'nccl' or 'gloo'"),
        ("NCCL_LIB_SHA256", "A" * 64, "lowercase SHA256"),
        ("PYTHONHASHSEED", "0007", "canonical integer string"),
        ("CUBLAS_WORKSPACE_CONFIG", " :4096:8", "must be '<unset>'"),
    ],
)
def test_execution_runtime_binding_rejects_noncanonical_recorded_values(
    monkeypatch,
    name,
    recorded,
    expected,
):
    provenance = _minimal_finalized_execution_runtime_provenance()
    provenance["environment"][EXECUTION_RUNTIME_KEY][name] = recorded

    with pytest.raises(ValueError, match=expected):
        validate_execution_runtime_binding(provenance)


def test_execution_runtime_binding_rejects_invalid_live_values(monkeypatch):
    provenance = _minimal_finalized_execution_runtime_provenance()
    monkeypatch.setenv("HOLOSOMA_DAGGER_SUPERVISED_ONLY", "sometimes")

    with pytest.raises(ValueError, match="HOLOSOMA_DAGGER_SUPERVISED_ONLY must be a boolean"):
        validate_execution_runtime_binding(provenance)


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        (
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
            "true",
            "must be exactly 0 or 1",
        ),
        (
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
            "",
            "must be exactly 0 or 1",
        ),
        (
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
            " 1",
            "must not contain surrounding whitespace",
        ),
        (
            "HOLOSOMA_GLOO_BARRIER",
            " true ",
            "must not contain surrounding whitespace",
        ),
        (
            "HOLOSOMA_CONTIGUOUS_MINIBATCHES",
            "off",
            "runtime consumer interprets that spelling as enabled",
        ),
        (
            "HOLOSOMA_RANK_VISIBLE_DEVICES",
            "",
            "must not be explicitly empty",
        ),
    ],
)
def test_execution_bool_normalization_rejects_values_that_consumers_interpret_differently(
    monkeypatch,
    name,
    value,
    expected,
):
    provenance = _minimal_finalized_execution_runtime_provenance()
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=expected):
        _environment_metadata()
    with pytest.raises(ValueError, match=expected):
        validate_execution_runtime_binding(provenance)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH", "+8"),
        ("HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC", "+300"),
        ("NPROC", " 8 "),
        ("NNODES", "８"),
    ],
)
def test_execution_int_normalization_rejects_non_consumer_equivalent_spellings(
    monkeypatch,
    name,
    value,
):
    provenance = _minimal_finalized_execution_runtime_provenance()
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match="ASCII|base-10"):
        _environment_metadata()
    with pytest.raises(ValueError, match="ASCII|base-10"):
        validate_execution_runtime_binding(provenance)


def test_nccl_digest_rejects_surrounding_whitespace_in_generator_and_live_binding(
    monkeypatch,
):
    provenance = _minimal_finalized_execution_runtime_provenance()
    monkeypatch.setenv("NCCL_LIB_SHA256", f" {'a' * 64} ")

    with pytest.raises(ValueError, match="NCCL_LIB_SHA256 must not contain surrounding whitespace"):
        _environment_metadata()
    with pytest.raises(ValueError, match="NCCL_LIB_SHA256 must not contain surrounding whitespace"):
        validate_execution_runtime_binding(provenance)


@pytest.mark.parametrize(("value", "expected"), [("0", False), ("1", True)])
def test_tf32_override_generator_matches_exact_pytorch_check_env_semantics(
    monkeypatch,
    value,
    expected,
):
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", value)

    execution_runtime = _environment_metadata()[EXECUTION_RUNTIME_KEY]

    assert execution_runtime["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] is expected


def test_execution_runtime_binding_normalizes_legacy_additive_fields():
    provenance = _minimal_finalized_execution_runtime_provenance()
    execution_runtime = provenance["environment"][EXECUTION_RUNTIME_KEY]
    execution_runtime.pop("HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES")
    execution_runtime.pop("HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC")
    execution_runtime.pop("HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY")

    normalized_provenance = validate_training_provenance(provenance)
    normalized_execution = normalized_provenance["environment"][EXECUTION_RUNTIME_KEY]
    assert normalized_execution["HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES"] is False
    assert normalized_execution["HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC"] == 300
    assert normalized_execution["HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY"] is False

    normalized = validate_execution_runtime_binding(provenance)

    assert normalized["HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES"] is False
    assert normalized["HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC"] == 300
    assert normalized["HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY"] is False


def test_execution_runtime_binding_still_rejects_unexpected_schema_keys():
    provenance = _minimal_finalized_execution_runtime_provenance()

    provenance["environment"][EXECUTION_RUNTIME_KEY]["TYPO_GLOO_BARRIER"] = False
    with pytest.raises(ValueError, match="keys must exactly match.*unexpected=.*TYPO_GLOO_BARRIER"):
        validate_execution_runtime_binding(provenance)


def test_execution_runtime_binding_accepts_derived_normal_topology(monkeypatch):
    monkeypatch.setenv("NPROC", "8")
    monkeypatch.setenv("NNODES", "2")
    provenance = _minimal_finalized_execution_runtime_provenance()
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("RANK", "13")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "5")
    monkeypatch.setenv("NODE_RANK", "1")

    assert validate_execution_runtime_binding(provenance)["NPROC"] == 8


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("WORLD_SIZE", "15", "WORLD_SIZE=15.*expected WORLD_SIZE=16"),
        ("RANK", "16", "RANK must be in"),
        ("LOCAL_WORLD_SIZE", "4", "LOCAL_WORLD_SIZE=4.*recorded NPROC=8"),
        ("LOCAL_RANK", "6", "global/local ranks are inconsistent"),
        ("NODE_RANK", "0", "NODE_RANK is inconsistent"),
    ],
)
def test_execution_runtime_binding_rejects_inconsistent_normal_topology(
    monkeypatch,
    name,
    value,
    expected,
):
    monkeypatch.setenv("NPROC", "8")
    monkeypatch.setenv("NNODES", "2")
    provenance = _minimal_finalized_execution_runtime_provenance()
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("RANK", "13")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "5")
    monkeypatch.setenv("NODE_RANK", "1")
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=expected):
        validate_execution_runtime_binding(provenance)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("missing_aliases", "requires HOLOSOMA_ORIGINAL_LOCAL_RANK"),
        ("original_world", "ORIGINAL_LOCAL_WORLD_SIZE=4.*recorded NPROC=8"),
        ("remapped_world", "must expose one remapped local rank"),
        ("original_rank", "global/local ranks are inconsistent"),
    ],
)
def test_execution_runtime_binding_rejects_inconsistent_rank_visible_topology(
    monkeypatch,
    mutation,
    expected,
):
    monkeypatch.setenv("NPROC", "8")
    monkeypatch.setenv("NNODES", "2")
    monkeypatch.setenv("HOLOSOMA_RANK_VISIBLE_DEVICES", "1")
    provenance = _minimal_finalized_execution_runtime_provenance()
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("RANK", "13")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("HOLOSOMA_ORIGINAL_LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("HOLOSOMA_ORIGINAL_LOCAL_RANK", "5")
    if mutation == "missing_aliases":
        monkeypatch.delenv("HOLOSOMA_ORIGINAL_LOCAL_WORLD_SIZE")
        monkeypatch.delenv("HOLOSOMA_ORIGINAL_LOCAL_RANK")
    elif mutation == "original_world":
        monkeypatch.setenv("HOLOSOMA_ORIGINAL_LOCAL_WORLD_SIZE", "4")
    elif mutation == "remapped_world":
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    else:
        monkeypatch.setenv("HOLOSOMA_ORIGINAL_LOCAL_RANK", "6")

    with pytest.raises(ValueError, match=expected):
        validate_execution_runtime_binding(provenance)


def test_environment_metadata_normalizes_numerical_execution_runtime(monkeypatch):
    monkeypatch.setenv("TORCH_DIST_BACKEND", " NcCl ")
    monkeypatch.setenv("NCCL_LIB_SHA256", "a" * 64)
    monkeypatch.setenv("PYTHONHASHSEED", "0007")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv(PYTHON_RUNTIME_SITEPACKAGES_ENV, "/runtime/numpy/site-packages")
    monkeypatch.setenv(PYTHON_RUNTIME_MANIFEST_SHA256_ENV, "b" * 64)
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")
    monkeypatch.setenv("HOLOSOMA_GLOO_BARRIER", "on")
    monkeypatch.setenv("HOLOSOMA_GLOO_GRAD_REDUCE", "off")
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE", "true")
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER", "0")
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC", "0037")
    monkeypatch.setenv("HOLOSOMA_RANK_VISIBLE_DEVICES", "true")
    monkeypatch.setenv("NPROC", "8")
    monkeypatch.setenv("NNODES", "3")
    monkeypatch.setenv("HOLOSOMA_CONTIGUOUS_MINIBATCHES", "on")
    monkeypatch.setenv("HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY", "yes")
    monkeypatch.setenv("HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP", "1")
    monkeypatch.setenv("HOLOSOMA_DAGGER_SUPERVISED_ONLY", "false")
    monkeypatch.setenv("HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH", "0016")
    monkeypatch.setenv("HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD", "yes")
    monkeypatch.setenv("HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC", "no")
    monkeypatch.setenv("HOLOSOMA_DEFM_FORWARD_BATCH_SIZE", " 0 ")
    monkeypatch.setenv("HOLOSOMA_DISABLE_AUTO_RESET", " False ")
    monkeypatch.setenv("HOLOSOMA_PERCEPTION_SENSOR_OFFSET_DELTA", " 0, 0, 0 ")

    environment = _environment_metadata()
    execution = environment[EXECUTION_RUNTIME_KEY]
    expected_semantic_environment = {
        name: None for name in SEMANTIC_ENVIRONMENT_FIELDS
    }
    expected_semantic_environment.update(
        {
            "HOLOSOMA_DEFM_FORWARD_BATCH_SIZE": "0",
            "HOLOSOMA_DISABLE_AUTO_RESET": "False",
            "HOLOSOMA_PERCEPTION_SENSOR_OFFSET_DELTA": "0, 0, 0",
        }
    )
    assert environment["python_runtime_manifest_sha256"] == "b" * 64
    assert set(environment["packages"]) == {
        "torch",
        "isaacsim",
        "isaaclab",
        "numpy",
        "omegaconf",
        "antlr4-python3-runtime",
        "PyYAML",
        "attrs",
    }
    assert execution == {
        "NCCL_LIB_SHA256": "a" * 64,
        "TORCH_DIST_BACKEND": "nccl",
        "PYTHONHASHSEED": "7",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": True,
        "HOLOSOMA_GLOO_BARRIER": True,
        "HOLOSOMA_GLOO_GRAD_REDUCE": False,
        "HOLOSOMA_GLOO_SMALL_COLLECTIVES": False,
        "HOLOSOMA_HIERARCHICAL_GRAD_REDUCE": True,
        "HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES": False,
        "HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER": False,
        "HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC": 37,
        "HOLOSOMA_RANK_VISIBLE_DEVICES": True,
        "NPROC": 8,
        "NNODES": 3,
        "HOLOSOMA_CONTIGUOUS_MINIBATCHES": True,
        "HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY": True,
        "HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP": True,
        "HOLOSOMA_DAGGER_SUPERVISED_ONLY": False,
        "HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH": 16,
        "HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD": True,
        "HOLOSOMA_SKIP_CRITIC_WEIGHT_SYNC": False,
        SEMANTIC_ENVIRONMENT_KEY: expected_semantic_environment,
    }


def test_environment_metadata_semantic_environment_has_fixed_keys_and_preserves_unset(
    monkeypatch,
):
    monkeypatch.setenv("HOLOSOMA_ONLINE_CONTACT_PRIOR", " 0 ")

    semantic_environment = _environment_metadata()[EXECUTION_RUNTIME_KEY][
        SEMANTIC_ENVIRONMENT_KEY
    ]

    assert tuple(semantic_environment) == SEMANTIC_ENVIRONMENT_FIELDS
    assert semantic_environment["HOLOSOMA_ONLINE_CONTACT_PRIOR"] == "0"
    assert semantic_environment["HOLOSOMA_DISABLE_ONLINE_CONTACT_PRIOR"] is None


def test_environment_metadata_uses_digest_sentinels_when_overlays_are_disabled():
    environment = _environment_metadata()
    execution = environment[EXECUTION_RUNTIME_KEY]
    assert re.fullmatch(r"[0-9a-f]{64}", environment["python_runtime_manifest_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", execution["NCCL_LIB_SHA256"])
    assert environment["python_runtime_manifest_sha256"] != execution["NCCL_LIB_SHA256"]


@pytest.mark.parametrize(
    ("sitepackages", "manifest"),
    [("/runtime/site-packages", ""), ("", "a" * 64)],
)
def test_environment_metadata_rejects_partial_python_overlay_identity(
    monkeypatch,
    sitepackages,
    manifest,
):
    monkeypatch.setenv(PYTHON_RUNTIME_SITEPACKAGES_ENV, sitepackages)
    monkeypatch.setenv(PYTHON_RUNTIME_MANIFEST_SHA256_ENV, manifest)
    with pytest.raises(ValueError, match="must be set together or both be disabled"):
        _environment_metadata()


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("TORCH_DIST_BACKEND", "mpi", "must be nccl or gloo"),
        (
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
            "maybe",
            "must be exactly 0 or 1",
        ),
        (
            "HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES",
            "maybe",
            "must be a boolean",
        ),
        (
            "HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC",
            "0",
            "must be a positive integer",
        ),
        ("HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH", "-1", "non-negative integer"),
        ("NPROC", "0", "positive integer"),
        ("NNODES", "0", "positive integer"),
        ("PYTHONHASHSEED", "random", "integer in"),
        ("CUBLAS_WORKSPACE_CONFIG", ":4096:2", "must be :4096:8 or :16:8"),
    ],
)
def test_environment_metadata_rejects_invalid_execution_runtime(
    monkeypatch,
    name,
    value,
    expected,
):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=expected):
        _environment_metadata()


def test_environment_metadata_requires_nccl_digest_for_nccl_backend(monkeypatch):
    monkeypatch.setenv("TORCH_DIST_BACKEND", "nccl")
    monkeypatch.delenv("NCCL_LIB_SHA256", raising=False)
    with pytest.raises(ValueError, match="NCCL_LIB_SHA256 is required"):
        _environment_metadata()


def test_environment_metadata_requires_nccl_digest_for_hierarchical_local_groups(monkeypatch):
    monkeypatch.setenv("TORCH_DIST_BACKEND", "gloo")
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE", "1")
    monkeypatch.delenv("NCCL_LIB_SHA256", raising=False)
    with pytest.raises(ValueError, match="hierarchical local gradient reduction uses NCCL"):
        _environment_metadata()


@pytest.mark.parametrize(
    ("gloo_small_collectives", "hierarchical_grad_reduce"),
    [(False, False), (False, True), (True, False)],
)
def test_hierarchical_small_collectives_requires_both_control_planes(
    monkeypatch,
    gloo_small_collectives,
    hierarchical_grad_reduce,
):
    monkeypatch.setenv("NCCL_LIB_SHA256", "a" * 64)
    monkeypatch.setenv(
        "HOLOSOMA_GLOO_SMALL_COLLECTIVES",
        "1" if gloo_small_collectives else "0",
    )
    monkeypatch.setenv(
        "HOLOSOMA_HIERARCHICAL_GRAD_REDUCE",
        "1" if hierarchical_grad_reduce else "0",
    )
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES", "1")

    with pytest.raises(
        ValueError,
        match=(
            "HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES=1 requires "
            "HOLOSOMA_GLOO_SMALL_COLLECTIVES=1 and "
            "HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=1"
        ),
    ):
        _environment_metadata()
    with pytest.raises(
        ValueError,
        match=(
            "HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES=1 requires "
            "HOLOSOMA_GLOO_SMALL_COLLECTIVES=1 and "
            "HOLOSOMA_HIERARCHICAL_GRAD_REDUCE=1"
        ),
    ):
        _validate_hierarchical_small_collectives_launch_contract()


def test_hierarchical_small_collectives_allows_gpu_gradient_leader(monkeypatch):
    monkeypatch.setenv("NCCL_LIB_SHA256", "a" * 64)
    monkeypatch.setenv("NPROC", "8")
    monkeypatch.setenv("NNODES", "2")
    monkeypatch.setenv("HOLOSOMA_GLOO_SMALL_COLLECTIVES", "1")
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE", "1")
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES", "1")
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER", "0")

    execution_runtime = _environment_metadata()[EXECUTION_RUNTIME_KEY]

    assert execution_runtime["HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES"] is True
    assert execution_runtime["HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER"] is False
    _validate_hierarchical_small_collectives_launch_contract()


def test_hierarchical_small_collectives_changes_execution_identity_fingerprint(monkeypatch):
    monkeypatch.setenv("NCCL_LIB_SHA256", "a" * 64)
    monkeypatch.setenv("NPROC", "8")
    monkeypatch.setenv("NNODES", "2")
    monkeypatch.setenv("HOLOSOMA_GLOO_SMALL_COLLECTIVES", "1")
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE", "1")
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES", "0")
    disabled_identity = _environment_metadata()[EXECUTION_RUNTIME_KEY]

    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES", "1")
    enabled_identity = _environment_metadata()[EXECUTION_RUNTIME_KEY]

    assert [
        name
        for name in disabled_identity
        if disabled_identity[name] != enabled_identity[name]
    ] == ["HOLOSOMA_HIERARCHICAL_SMALL_COLLECTIVES"]
    assert provenance_module.sha256_json(disabled_identity) != provenance_module.sha256_json(
        enabled_identity
    )


def test_fresh_checkpoint_metadata_retains_execution_runtime_provenance():
    from holosoma.agents.base_algo.base_algo import BaseAlgo

    class _Config:
        @staticmethod
        def to_serializable_dict():
            return {"training": {"name": "fresh"}}

    runtime_asset_manifest = {"version": 2, "fixture": "checkpoint-metadata"}
    provenance = {
        "environment": _environment_metadata(),
        "runtime_asset_manifest": runtime_asset_manifest,
    }
    algo = object.__new__(BaseAlgo)
    algo._experiment_config = None
    algo._wandb_run_path = None
    algo._training_provenance = None
    with patch(
        "holosoma.agents.base_algo.base_algo.training_provenance_from_env",
        return_value=provenance,
    ):
        algo.attach_checkpoint_metadata(_Config())

    checkpoint_metadata = algo._checkpoint_metadata(iteration=0)
    assert checkpoint_metadata["training_provenance"]["environment"][EXECUTION_RUNTIME_KEY] == (
        provenance["environment"][EXECUTION_RUNTIME_KEY]
    )
    assert checkpoint_metadata["training_provenance"]["runtime_asset_manifest"] == runtime_asset_manifest


def test_training_provenance_hashes_teacher_motion_and_contact_inputs(tmp_path):
    teacher, motion, object_map, contact, source_root = _fixture(tmp_path)
    first = _compute(teacher, motion, object_map, contact, source_root)
    assert first[TRAINING_REGIME_KEY] == TRAINING_REGIME_DISTILLATION
    assert first[TEACHER_ENABLED_KEY] is True
    assert first["teacher_motion_end_mode"] == "episodic"
    assert first["teacher_uses_action_history"] is True
    assert "source_snapshot_id" not in first
    assert "source_manifest_sha256" not in first
    assert first["policy_init_enabled"] is False
    assert first["policy_init_sha256"] == disabled_checkpoint_sha256("policy_init")
    assert first["training_resume_enabled"] is False
    assert first["training_resume_sha256"] == disabled_checkpoint_sha256("training_resume")
    assert first["environment"][EXECUTION_RUNTIME_KEY]["TORCH_DIST_BACKEND"] == "gloo"
    assert isinstance(
        first["environment"][EXECUTION_RUNTIME_KEY]["HOLOSOMA_CONTIGUOUS_MINIBATCHES"],
        bool,
    )
    assert all(len(first[key]) == 64 for key in (
        "teacher_sha256",
        "policy_init_sha256",
        "training_resume_sha256",
        "motion_shard_manifest_sha256",
        "contact_sidecar_manifest_sha256",
        "source_bundle_sha256",
    ))

    np.savez(
        motion / "clip_a.npz",
        fps=np.asarray([50.0]),
        body_pos_w=np.ones((2, 1, 3), dtype=np.float32),
    )
    motion_changed = _compute(teacher, motion, object_map, contact, source_root)
    assert motion_changed["motion_shard_manifest_sha256"] != first["motion_shard_manifest_sha256"]

    points = contact / "clips" / "0000_clip_a" / "left_wrist_contact_points.npy"
    np.save(points, np.ones((1, 3), dtype=np.float32))
    second = _compute(teacher, motion, object_map, contact, source_root)
    assert second["contact_sidecar_manifest_sha256"] != first["contact_sidecar_manifest_sha256"]


def test_pure_rl_generalist_provenance_is_explicitly_teacher_free_and_content_closed(tmp_path):
    _teacher_checkpoint, motion, object_map, contact, source_root = _fixture(tmp_path)
    provenance = compute_generalist_provenance(
        motion_dir=motion,
        object_map=object_map,
        contact_root=contact,
        motion_shard_manifest=None,
        source_root=source_root,
    )

    assert provenance[TRAINING_REGIME_KEY] == TRAINING_REGIME_PURE_RL
    assert provenance[TEACHER_ENABLED_KEY] is False
    assert provenance["teacher_sha256"] == disabled_teacher_sha256()
    assert "teacher_motion_end_mode" not in provenance
    assert "teacher_uses_action_history" not in provenance
    assert validate_training_provenance(provenance) == provenance

    np.savez(
        motion / "clip_a.npz",
        fps=np.asarray([50.0]),
        body_pos_w=np.ones((2, 1, 3), dtype=np.float32),
    )
    changed = compute_generalist_provenance(
        motion_dir=motion,
        object_map=object_map,
        contact_root=contact,
        motion_shard_manifest=None,
        source_root=source_root,
    )
    assert changed["motion_shard_manifest_sha256"] != provenance["motion_shard_manifest_sha256"]


def test_teacher_mode_validation_normalizes_legacy_v2_and_rejects_false_claims(tmp_path):
    teacher, motion, object_map, contact, source_root = _fixture(tmp_path)
    distillation = _compute(teacher, motion, object_map, contact, source_root)
    legacy = dict(distillation)
    legacy.pop(TRAINING_REGIME_KEY)
    legacy.pop(TEACHER_ENABLED_KEY)

    normalized = validate_training_provenance(legacy)
    assert normalized[TRAINING_REGIME_KEY] == TRAINING_REGIME_DISTILLATION
    assert normalized[TEACHER_ENABLED_KEY] is True

    with pytest.raises(ValueError, match="pure-RL.*disabled teacher_sha256"):
        validate_training_provenance(
            {
                **distillation,
                TRAINING_REGIME_KEY: TRAINING_REGIME_PURE_RL,
                TEACHER_ENABLED_KEY: False,
            }
        )
    with pytest.raises(ValueError, match="must not claim teacher semantics"):
        validate_training_provenance(
            {
                **distillation,
                TRAINING_REGIME_KEY: TRAINING_REGIME_PURE_RL,
                TEACHER_ENABLED_KEY: False,
                "teacher_sha256": disabled_teacher_sha256(),
            }
        )


@pytest.mark.parametrize("mutation", ["motion", "object_map", "mesh", "contact"])
def test_pre_simulator_data_revalidation_rejects_launcher_input_mutation(
    tmp_path,
    mutation,
):
    teacher, motion, object_map, contact, source_root = _fixture(tmp_path)
    provenance = _finalized(_compute(teacher, motion, object_map, contact, source_root))

    revalidate_data_asset_provenance(
        provenance,
        motion_dir=motion,
        object_map=object_map,
        contact_root=contact,
        motion_shard_manifest=None,
    )

    if mutation == "motion":
        np.savez(
            motion / "clip_a.npz",
            fps=np.asarray([50.0]),
            body_pos_w=np.ones((2, 1, 3), dtype=np.float32),
        )
    elif mutation == "object_map":
        payload = json.loads(object_map.read_text(encoding="utf-8"))
        payload["metadata"] = {"mutated": True}
        object_map.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "mesh":
        mesh = motion / "assets" / "object.obj"
        mesh.write_bytes(mesh.read_bytes() + b"\n# mutated after launcher preflight\n")
    else:
        sidecar = contact / "clips" / "0000_clip_a" / "left_wrist_contact_points.npy"
        np.save(sidecar, np.ones((1, 3), dtype=np.float32))

    with pytest.raises(RuntimeError, match="changed after launcher preflight"):
        revalidate_data_asset_provenance(
            provenance,
            motion_dir=motion,
            object_map=object_map,
            contact_root=contact,
            motion_shard_manifest=None,
        )


def test_cached_pre_simulator_revalidation_never_skips_source_bundle(tmp_path):
    teacher, motion, object_map, contact, source_root = _fixture(tmp_path)
    provenance = _finalized(_compute(teacher, motion, object_map, contact, source_root))
    kwargs = {
        "motion_dir": motion,
        "object_map": object_map,
        "contact_root": contact,
        "motion_shard_manifest": None,
        "source_root": source_root,
        "cache_root": tmp_path / "cache",
        "node_id": "node-a",
    }
    revalidate_data_asset_provenance_cached(provenance, **kwargs)

    source_file = source_root / "src" / "holosoma" / "holosoma" / "module.py"
    source_file.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source bundle changed after launcher preflight"):
        revalidate_data_asset_provenance_cached(provenance, **kwargs)


def test_per_node_data_revalidation_cache_reuses_only_unchanged_identity_closure(
    tmp_path,
    monkeypatch,
):
    teacher, motion, object_map, contact, source_root = _fixture(tmp_path)
    provenance = _finalized(_compute(teacher, motion, object_map, contact, source_root))
    cache_root = tmp_path / "cache"
    real_revalidate = provenance_module.revalidate_data_asset_provenance
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_revalidate(*args, **kwargs)

    monkeypatch.setattr(provenance_module, "revalidate_data_asset_provenance", counted)
    kwargs = {
        "motion_dir": motion,
        "object_map": object_map,
        "contact_root": contact,
        "motion_shard_manifest": None,
        "cache_root": cache_root,
        "node_id": "node-a",
    }
    revalidate_data_asset_provenance_cached(provenance, **kwargs)
    revalidate_data_asset_provenance_cached(provenance, **kwargs)
    assert calls == 1

    sidecar = contact / "clips" / "0000_clip_a" / "left_wrist_contact_points.npy"
    np.save(sidecar, np.ones((1, 3), dtype=np.float32))
    with pytest.raises(RuntimeError, match="changed after launcher preflight"):
        revalidate_data_asset_provenance_cached(provenance, **kwargs)
    assert calls == 2


def test_data_revalidation_cache_rejects_retargeted_motion_symlink(tmp_path):
    teacher, motion, object_map, contact, source_root = _fixture(tmp_path)
    motion_link = motion / "clip_a.npz"
    original_target = tmp_path / "clip_a_original.npz"
    replacement_target = tmp_path / "clip_a_replacement.npz"
    motion_link.replace(original_target)
    motion_link.symlink_to(original_target)
    np.savez(
        replacement_target,
        fps=np.asarray([50.0]),
        body_pos_w=np.ones((2, 1, 3), dtype=np.float32),
    )
    provenance = _finalized(_compute(teacher, motion, object_map, contact, source_root))
    kwargs = {
        "motion_dir": motion,
        "object_map": object_map,
        "contact_root": contact,
        "motion_shard_manifest": None,
        "cache_root": tmp_path / "cache",
        "node_id": "node-a",
    }
    revalidate_data_asset_provenance_cached(provenance, **kwargs)

    motion_link.unlink()
    motion_link.symlink_to(replacement_target)
    with pytest.raises(RuntimeError, match="changed after launcher preflight"):
        revalidate_data_asset_provenance_cached(provenance, **kwargs)


def test_data_revalidation_cache_isolated_per_node(tmp_path, monkeypatch):
    teacher, motion, object_map, contact, source_root = _fixture(tmp_path)
    provenance = _finalized(_compute(teacher, motion, object_map, contact, source_root))
    cache_root = tmp_path / "shared-cache"
    real_revalidate = provenance_module.revalidate_data_asset_provenance
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_revalidate(*args, **kwargs)

    monkeypatch.setattr(provenance_module, "revalidate_data_asset_provenance", counted)
    common = {
        "motion_dir": motion,
        "object_map": object_map,
        "contact_root": contact,
        "motion_shard_manifest": None,
        "cache_root": cache_root,
    }
    revalidate_data_asset_provenance_cached(provenance, node_id="node-a", **common)
    revalidate_data_asset_provenance_cached(provenance, node_id="node-b", **common)

    assert calls == 2
    assert len(list(cache_root.glob("*.json"))) == 2


def test_concurrent_local_ranks_share_one_full_data_revalidation(tmp_path, monkeypatch):
    teacher, motion, object_map, contact, source_root = _fixture(tmp_path)
    provenance = _finalized(_compute(teacher, motion, object_map, contact, source_root))
    cache_root = tmp_path / "cache"
    real_revalidate = provenance_module.revalidate_data_asset_provenance
    calls = 0
    calls_lock = threading.Lock()

    def counted(*args, **kwargs):
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        return real_revalidate(*args, **kwargs)

    monkeypatch.setattr(provenance_module, "revalidate_data_asset_provenance", counted)
    kwargs = {
        "motion_dir": motion,
        "object_map": object_map,
        "contact_root": contact,
        "motion_shard_manifest": None,
        "cache_root": cache_root,
        "node_id": "node-a",
    }
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                lambda _index: revalidate_data_asset_provenance_cached(
                    provenance,
                    **kwargs,
                ),
                range(4),
            )
        )

    assert calls == 1
    assert all(result == results[0] for result in results)


def test_training_provenance_requires_content_closed_rank_shard_manifest(tmp_path):
    teacher, motion, object_map, contact, source_root = _fixture(tmp_path)
    shard_root = tmp_path / "rank_shards" / "ws1"
    prepare_rank_shards(
        motion_dir=motion,
        object_map=object_map,
        output_root=shard_root,
        world_size=1,
    )
    manifest_path = shard_root / "manifest.json"

    provenance = _compute(
        teacher,
        motion,
        object_map,
        contact,
        source_root,
        motion_shard_manifest=manifest_path,
    )
    assert len(provenance["motion_shard_manifest_sha256"]) == 64

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shards"][0]["npz_files"][0]["sha256"] = "f" * 64
    # Published rank shards are intentionally sealed read-only.  This test is
    # an explicit attacker/fault injection, so thaw only the temporary target
    # file before mutating it rather than weakening the publisher contract.
    manifest_path.chmod(0o644)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="aggregate digest is inconsistent"):
        _compute(
            teacher,
            motion,
            object_map,
            contact,
            source_root,
            motion_shard_manifest=manifest_path,
        )

    manifest_path.write_text(json.dumps({"world_size": 1, "shards": []}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="predates the content-closed"):
        _compute(
            teacher,
            motion,
            object_map,
            contact,
            source_root,
            motion_shard_manifest=manifest_path,
        )


def test_cached_pre_simulator_validation_rejects_selected_rank_shard_tamper(
    tmp_path,
    monkeypatch,
):
    teacher, motion, object_map, contact, source_root = _fixture(tmp_path)
    shard_root = tmp_path / "rank_shards" / "ws1"
    prepare_rank_shards(
        motion_dir=motion,
        object_map=object_map,
        output_root=shard_root,
        world_size=1,
    )
    manifest_path = shard_root / "manifest.json"
    provenance = _finalized(
        _compute(
            teacher,
            motion,
            object_map,
            contact,
            source_root,
            motion_shard_manifest=manifest_path,
        )
    )
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    kwargs = {
        "motion_dir": motion,
        "object_map": object_map,
        "contact_root": contact,
        "motion_shard_manifest": manifest_path,
        "cache_root": tmp_path / "cache",
        "node_id": "node-a",
    }
    revalidate_data_asset_provenance_cached(provenance, **kwargs)

    selected_npz = shard_root / "rank_0" / "clip_a.npz"
    replacement = tmp_path / "replacement.npz"
    np.savez(
        replacement,
        fps=np.asarray([50.0]),
        body_pos_w=np.ones((2, 1, 3), dtype=np.float32),
    )
    # Unlinking the sealed symlink requires write permission on its temporary
    # fixture directory.  Production shard namespaces remain read-only.
    selected_npz.parent.chmod(0o755)
    selected_npz.unlink()
    selected_npz.symlink_to(replacement)

    with pytest.raises(RuntimeError, match="rank-local NPZ content changed"):
        revalidate_data_asset_provenance_cached(provenance, **kwargs)


def test_source_bundle_digest_covers_holosoma_inference_python_sources(tmp_path):
    teacher, motion, object_map, contact, source_root = _fixture(tmp_path)
    first = _compute(teacher, motion, object_map, contact, source_root)

    inference_source = (
        source_root / "src" / "holosoma_inference" / "holosoma_inference" / "policy.py"
    )
    inference_source.write_text("POLICY_VERSION = 2\n", encoding="utf-8")
    second = _compute(teacher, motion, object_map, contact, source_root)

    assert second["source_bundle_sha256"] != first["source_bundle_sha256"]
    for key in (
        "teacher_sha256",
        "policy_init_sha256",
        "training_resume_sha256",
        "motion_shard_manifest_sha256",
        "contact_sidecar_manifest_sha256",
    ):
        assert second[key] == first[key]


def test_source_bundle_digest_covers_pinned_defm_runtime_sources(tmp_path):
    teacher, motion, object_map, contact, source_root = _fixture(tmp_path)
    defm_source = source_root / "submodules" / "defm" / "defm" / "model_factory.py"
    defm_source.parent.mkdir(parents=True)
    defm_source.write_text("DEFM_VERSION = 1\n", encoding="utf-8")
    first = _compute(teacher, motion, object_map, contact, source_root)

    defm_source.write_text("DEFM_VERSION = 2\n", encoding="utf-8")
    second = _compute(teacher, motion, object_map, contact, source_root)

    assert second["source_bundle_sha256"] != first["source_bundle_sha256"]


def test_training_provenance_hashes_runtime_contact_metadata_and_all_regions(tmp_path):
    teacher, motion, object_map, contact, source_root = _fixture(tmp_path)
    clip_dir = contact / "clips" / "0000_clip_a"
    (clip_dir / "contact_intervals.json").write_text(
        json.dumps({"left_elbow": [2, 5]}), encoding="utf-8"
    )
    (clip_dir / "metadata.json").write_text(json.dumps({"motion_fps": 50.0}), encoding="utf-8")
    np.save(clip_dir / "left_elbow_contact_points.npy", np.zeros((1, 3), dtype=np.float32))
    np.save(clip_dir / "left_elbow_contact_point_counts.npy", np.ones((1,), dtype=np.int32))
    np.save(clip_dir / "left_elbow_contact_interval_steps.npy", np.asarray([2, 5], dtype=np.int32))
    np.save(clip_dir / "left_elbow_contact_active_mask.npy", np.asarray([False, True, True]))
    first = _compute(teacher, motion, object_map, contact, source_root)

    np.save(clip_dir / "arm_contact_points.npy", np.ones((4, 3), dtype=np.float32))
    ignored_legacy_changed = _compute(teacher, motion, object_map, contact, source_root)
    assert ignored_legacy_changed["contact_sidecar_manifest_sha256"] == first["contact_sidecar_manifest_sha256"]

    (clip_dir / "contact_intervals.json").write_text(
        json.dumps({"left_elbow": [3, 6]}), encoding="utf-8"
    )
    interval_changed = _compute(teacher, motion, object_map, contact, source_root)
    assert interval_changed["contact_sidecar_manifest_sha256"] != first["contact_sidecar_manifest_sha256"]

    (clip_dir / "metadata.json").write_text(json.dumps({"motion_fps": 60.0}), encoding="utf-8")
    metadata_changed = _compute(teacher, motion, object_map, contact, source_root)
    assert metadata_changed["contact_sidecar_manifest_sha256"] != interval_changed["contact_sidecar_manifest_sha256"]

    np.save(clip_dir / "left_elbow_contact_active_mask.npy", np.asarray([True, True, True]))
    schedule_changed = _compute(teacher, motion, object_map, contact, source_root)
    assert schedule_changed["contact_sidecar_manifest_sha256"] != metadata_changed["contact_sidecar_manifest_sha256"]

    np.save(clip_dir / "left_elbow_contact_points.npy", np.ones((1, 3), dtype=np.float32))
    region_changed = _compute(teacher, motion, object_map, contact, source_root)
    assert (
        region_changed["contact_sidecar_manifest_sha256"]
        != schedule_changed["contact_sidecar_manifest_sha256"]
    )


def test_training_provenance_hashes_urdf_and_transitive_mesh_assets(tmp_path):
    teacher, motion, object_map, contact, source_root = _fixture(tmp_path)
    first = _compute(teacher, motion, object_map, contact, source_root)

    mesh = motion / "assets" / "object.obj"
    mesh.write_text("v 0 0 0\nv 2 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    mesh_changed = _compute(teacher, motion, object_map, contact, source_root)

    assert mesh_changed["motion_shard_manifest_sha256"] != first["motion_shard_manifest_sha256"]
    for key in (
        "teacher_sha256",
        "policy_init_sha256",
        "training_resume_sha256",
        "contact_sidecar_manifest_sha256",
        "source_bundle_sha256",
    ):
        assert mesh_changed[key] == first[key]


def _replace_object_mesh_reference(motion: Path, filename: str) -> None:
    urdf = motion / "assets" / "object.urdf"
    urdf.write_text(
        "<robot name='object'><link name='base'><visual><geometry>"
        f"<mesh filename='{filename}'/></geometry></visual><collision><geometry>"
        f"<mesh filename='{filename}'/></geometry></collision></link></robot>",
        encoding="utf-8",
    )


@pytest.mark.parametrize("dependency_kind", ["obj_mtl_texture", "gltf_buffer", "gltf_image", "dae", "ply"])
def test_training_provenance_hashes_object_mesh_external_dependencies(tmp_path, dependency_kind):
    teacher, motion, object_map, contact, source_root = _fixture(tmp_path)
    assets = motion / "assets"
    texture = assets / "texture.png"
    texture.write_bytes(b"texture-v1")

    if dependency_kind == "obj_mtl_texture":
        (assets / "object.obj").write_text(
            "mtllib object.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
            encoding="utf-8",
        )
        (assets / "object.mtl").write_text("newmtl material\nmap_Kd texture.png\n", encoding="utf-8")
        dependency = texture
    elif dependency_kind in {"gltf_buffer", "gltf_image"}:
        _replace_object_mesh_reference(motion, "object.gltf")
        (assets / "buffer.bin").write_bytes(b"buffer-v1")
        (assets / "object.gltf").write_text(
            json.dumps(
                {
                    "asset": {"version": "2.0"},
                    "buffers": [{"uri": "buffer.bin", "byteLength": 9}],
                    "images": [{"uri": "texture.png"}],
                }
            ),
            encoding="utf-8",
        )
        dependency = assets / ("buffer.bin" if dependency_kind == "gltf_buffer" else "texture.png")
    elif dependency_kind == "dae":
        _replace_object_mesh_reference(motion, "object.dae")
        (assets / "object.dae").write_text(
            "<COLLADA><library_images><image id='tex'><init_from>texture.png</init_from>"
            "</image></library_images></COLLADA>",
            encoding="utf-8",
        )
        dependency = texture
    else:
        _replace_object_mesh_reference(motion, "object.ply")
        (assets / "object.ply").write_bytes(
            b"ply\nformat ascii 1.0\ncomment TextureFile texture.png\n"
            b"element vertex 0\nend_header\n"
        )
        dependency = texture

    first = _compute(teacher, motion, object_map, contact, source_root)
    dependency.write_bytes(dependency.read_bytes() + b"-changed")
    second = _compute(teacher, motion, object_map, contact, source_root)

    assert second["motion_shard_manifest_sha256"] != first["motion_shard_manifest_sha256"]


def test_training_provenance_rejects_unknown_object_mesh_format(tmp_path):
    teacher, motion, object_map, contact, source_root = _fixture(tmp_path)
    _replace_object_mesh_reference(motion, "object.fbx")
    (motion / "assets" / "object.fbx").write_bytes(b"opaque-format")

    with pytest.raises(ValueError, match="transitive external-asset closure is not implemented"):
        _compute(teacher, motion, object_map, contact, source_root)


def test_training_provenance_rejects_missing_or_empty_object_asset_closure(tmp_path):
    teacher, motion, object_map, contact, source_root = _fixture(tmp_path)
    (motion / "assets" / "object.obj").unlink()
    with pytest.raises(FileNotFoundError, match="clip 'clip_a'.*mesh.*asset does not exist"):
        _compute(teacher, motion, object_map, contact, source_root)

    object_map.write_text(json.dumps({"clips": {"clip_a": {}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="active clip 'clip_a' has no object_urdf_path"):
        _compute(teacher, motion, object_map, contact, source_root)


def test_training_provenance_includes_matching_source_snapshot_identity(tmp_path, monkeypatch):
    teacher, motion, object_map, contact, source_root = _fixture(tmp_path)
    digest = "a" * 64
    monkeypatch.setenv(SOURCE_SNAPSHOT_ID_ENV, f"src-{digest}")
    monkeypatch.setenv(SOURCE_MANIFEST_SHA256_ENV, digest)

    provenance = _compute(teacher, motion, object_map, contact, source_root)

    assert provenance["source_snapshot_id"] == f"src-{digest}"
    assert provenance["source_manifest_sha256"] == digest
    assert validate_training_provenance(provenance) == provenance


@pytest.mark.parametrize(
    ("present_name", "present_value"),
    [
        (SOURCE_SNAPSHOT_ID_ENV, f"src-{'a' * 64}"),
        (SOURCE_MANIFEST_SHA256_ENV, "a" * 64),
    ],
)
def test_training_provenance_rejects_partial_source_snapshot_identity(
    tmp_path,
    monkeypatch,
    present_name,
    present_value,
):
    teacher, motion, object_map, contact, source_root = _fixture(tmp_path)
    monkeypatch.setenv(present_name, present_value)

    with pytest.raises(ValueError, match="must be set together"):
        _compute(teacher, motion, object_map, contact, source_root)


@pytest.mark.parametrize(
    ("snapshot_id", "manifest_sha256", "error"),
    [
        (f"source-{'a' * 64}", "a" * 64, "must have format"),
        (f"src-{'A' * 64}", "a" * 64, "must have format"),
        (f"src-{'a' * 64}", "A" * 64, "must be a 64-character lowercase"),
        (f"src-{'a' * 64}", "b" * 64, "digest does not match"),
    ],
)
def test_training_provenance_rejects_invalid_source_snapshot_identity(
    tmp_path,
    monkeypatch,
    snapshot_id,
    manifest_sha256,
    error,
):
    teacher, motion, object_map, contact, source_root = _fixture(tmp_path)
    monkeypatch.setenv(SOURCE_SNAPSHOT_ID_ENV, snapshot_id)
    monkeypatch.setenv(SOURCE_MANIFEST_SHA256_ENV, manifest_sha256)

    with pytest.raises(ValueError, match=error):
        _compute(teacher, motion, object_map, contact, source_root)


def test_training_provenance_validator_rejects_partial_or_mismatched_snapshot_identity(tmp_path):
    teacher, motion, object_map, contact, source_root = _fixture(tmp_path)
    provenance = _compute(teacher, motion, object_map, contact, source_root)
    digest = "a" * 64

    with pytest.raises(ValueError, match="must be present together"):
        validate_training_provenance({**provenance, "source_snapshot_id": f"src-{digest}"})
    with pytest.raises(ValueError, match="must match"):
        validate_training_provenance(
            {
                **provenance,
                "source_snapshot_id": f"src-{digest}",
                "source_manifest_sha256": "b" * 64,
            }
        )


def test_optional_checkpoint_modes_are_explicit_and_strict_in_v2(tmp_path):
    teacher, motion, object_map, contact, source_root = _fixture(tmp_path)
    provenance = _compute(teacher, motion, object_map, contact, source_root)

    assert checkpoint_lineage_enabled(provenance, "policy_init") is False
    assert checkpoint_lineage_enabled(provenance, "training_resume") is False

    missing_modes = dict(provenance)
    missing_modes.pop("policy_init_enabled")
    missing_modes.pop("training_resume_enabled")
    with pytest.raises(ValueError, match="policy_init_enabled is required"):
        validate_training_provenance(missing_modes)

    with pytest.raises(ValueError, match="policy_init_enabled=False requires"):
        validate_training_provenance({**provenance, "policy_init_sha256": "a" * 64})
    with pytest.raises(ValueError, match="training_resume_enabled=True cannot use"):
        validate_training_provenance({**provenance, "training_resume_enabled": True})
    with pytest.raises(ValueError, match="policy_init_enabled must be a boolean"):
        validate_training_provenance({**provenance, "policy_init_enabled": 0})


def _teacher_consuming_config():
    from holosoma.config_values.experiment import DEFAULTS

    config = copy.deepcopy(DEFAULTS["g1_29dof_wbt_w_object_distill_sparse_root_cmd"])
    distill = dataclasses.replace(
        config.algo.config.distill,
        enabled=True,
        teacher_checkpoint="/tmp/teacher.pt",
    )
    algo_config = dataclasses.replace(config.algo.config, distill=distill)
    return dataclasses.replace(
        config,
        algo=dataclasses.replace(config.algo, config=algo_config),
    )


def test_train_agent_binds_checkpoint_lineage_to_cli_mode(tmp_path):
    teacher, motion, object_map, contact, source_root = _fixture(tmp_path)
    fresh_provenance = _compute(teacher, motion, object_map, contact, source_root)
    config = _teacher_consuming_config()

    with patch(
        "holosoma.train_agent.training_provenance_from_env",
        return_value=fresh_provenance,
    ):
        _preflight_checkpoint_lineage_before_sim(config)

    claimed_policy_init = {
        **fresh_provenance,
        "policy_init_enabled": True,
        "policy_init_sha256": "a" * 64,
    }
    with (
        patch(
            "holosoma.train_agent.training_provenance_from_env",
            return_value=claimed_policy_init,
        ),
        pytest.raises(ValueError, match="policy_init_enabled=True.*presence=False"),
    ):
        _preflight_checkpoint_lineage_before_sim(config)

    policy_init_config = dataclasses.replace(
        config,
        training=dataclasses.replace(config.training, policy_init_checkpoint="/tmp/policy.pt"),
    )
    with (
        patch(
            "holosoma.train_agent.training_provenance_from_env",
            return_value=fresh_provenance,
        ),
        pytest.raises(ValueError, match="policy_init_enabled=False.*presence=True"),
    ):
        _preflight_checkpoint_lineage_before_sim(policy_init_config)

    claimed_resume = dict(fresh_provenance)
    claimed_resume["training_resume_enabled"] = True
    claimed_resume["training_resume_sha256"] = "b" * 64
    with (
        patch(
            "holosoma.train_agent.training_provenance_from_env",
            return_value=claimed_resume,
        ),
        pytest.raises(ValueError, match="training_resume_enabled=True.*presence=False"),
    ):
        _preflight_checkpoint_lineage_before_sim(config)


def test_train_agent_requires_teacher_identity_before_simulator_by_default(
    monkeypatch,
):
    config = _teacher_consuming_config()
    with (
        patch(
            "holosoma.train_agent.training_provenance_from_env",
            return_value=None,
        ),
        pytest.raises(ValueError, match="requires finalized current training provenance"),
    ):
        _preflight_checkpoint_lineage_before_sim(config)


def test_train_agent_teacher_identity_hatch_is_exact_and_explicit(
    monkeypatch,
    capsys,
):
    config = _teacher_consuming_config()
    monkeypatch.setenv(ALLOW_LEGACY_UNVERIFIED_TEACHER_LOAD_ENV, "true")
    with (
        patch(
            "holosoma.train_agent.training_provenance_from_env",
            return_value=None,
        ),
        pytest.raises(ValueError, match="must be exactly 0 or 1"),
    ):
        _preflight_checkpoint_lineage_before_sim(config)

    monkeypatch.setenv(ALLOW_LEGACY_UNVERIFIED_TEACHER_LOAD_ENV, "1")
    with patch(
        "holosoma.train_agent.training_provenance_from_env",
        return_value=None,
    ):
        _preflight_checkpoint_lineage_before_sim(config)

    assert "legacy_unverified_teacher_load_allowed" in capsys.readouterr().out


def test_train_agent_rejects_teacher_disabled_provenance_for_distillation(tmp_path):
    teacher, motion, object_map, contact, source_root = _fixture(tmp_path)
    provenance = _compute(teacher, motion, object_map, contact, source_root)
    provenance[TRAINING_REGIME_KEY] = TRAINING_REGIME_PURE_RL
    provenance[TEACHER_ENABLED_KEY] = False
    provenance["teacher_sha256"] = disabled_teacher_sha256()
    provenance.pop("teacher_motion_end_mode", None)
    provenance.pop("teacher_uses_action_history", None)
    provenance = validate_training_provenance(provenance)
    config = _teacher_consuming_config()

    with (
        patch(
            "holosoma.train_agent.training_provenance_from_env",
            return_value=provenance,
        ),
        pytest.raises(ValueError, match="consumes a teacher.*disables it"),
    ):
        _preflight_checkpoint_lineage_before_sim(config)

def test_train_agent_data_asset_preflight_passes_effective_launcher_paths(
    tmp_path,
    monkeypatch,
):
    teacher, motion, object_map, contact, source_root = _fixture(tmp_path)
    provenance = _finalized(_compute(teacher, motion, object_map, contact, source_root))
    shard_root = tmp_path / "rank-shards"
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("MOTION_DIR", str(motion))
    monkeypatch.setenv("OBJECT_SPEC_PATH", str(object_map))
    monkeypatch.setenv("OBJECT_URDF", str(object_map))
    monkeypatch.setenv("CONTACT_EXPORT_ROOT", str(contact))
    monkeypatch.setenv("HOLOSOMA_SOURCE_ROOT", str(source_root))
    monkeypatch.setenv("HOLOSOMA_RANK_LOCAL_MOTION_ROOT", str(shard_root))
    monkeypatch.setenv("HOLOSOMA_DATA_PROVENANCE_CACHE_ROOT", str(cache_root))

    with (
        patch(
            "holosoma.train_agent.training_provenance_from_env",
            return_value=provenance,
        ),
        patch(
            "holosoma.train_agent.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, "verified\n", ""),
        ) as run,
    ):
        assert _preflight_data_assets_before_sim() == provenance

    command = run.call_args.args[0]
    assert command[1:4] == ["-m", "holosoma.utils.compute_training_provenance", "--revalidate-data-assets"]
    assert command[command.index("--motion-dir") + 1] == str(motion)
    assert command[command.index("--object-map") + 1] == str(object_map)
    assert command[command.index("--source-root") + 1] == str(source_root)
    assert command[command.index("--contact-root") + 1] == str(contact)
    assert command[command.index("--motion-shard-manifest") + 1] == str(
        shard_root / "manifest.json"
    )
    assert command[command.index("--cache-root") + 1] == str(cache_root)
    assert json.loads(run.call_args.kwargs["input"]) == provenance


def test_train_agent_data_asset_preflight_executes_revalidation_helper(
    tmp_path,
    monkeypatch,
    capsys,
):
    teacher, motion, object_map, contact, source_root = _fixture(tmp_path)
    provenance = _finalized(_compute(teacher, motion, object_map, contact, source_root))
    monkeypatch.setenv("MOTION_DIR", str(motion))
    monkeypatch.setenv("OBJECT_SPEC_PATH", str(object_map))
    monkeypatch.delenv("OBJECT_URDF", raising=False)
    monkeypatch.setenv("CONTACT_EXPORT_ROOT", str(contact))
    monkeypatch.setenv("HOLOSOMA_SOURCE_ROOT", str(source_root))
    monkeypatch.setenv(
        "HOLOSOMA_DATA_PROVENANCE_CACHE_ROOT",
        str(tmp_path / "cache"),
    )
    monkeypatch.delenv("HOLOSOMA_RANK_LOCAL_MOTION_ROOT", raising=False)
    monkeypatch.delenv("HOLOSOMA_MOTION_SHARD_MANIFEST", raising=False)

    with patch(
        "holosoma.train_agent.training_provenance_from_env",
        return_value=provenance,
    ):
        assert _preflight_data_assets_before_sim() == provenance

    assert "pre-simulator training data provenance verified" in capsys.readouterr().out


def test_train_agent_data_asset_preflight_fails_closed_without_exported_paths(
    tmp_path,
    monkeypatch,
):
    teacher, motion, object_map, contact, source_root = _fixture(tmp_path)
    provenance = _finalized(_compute(teacher, motion, object_map, contact, source_root))
    for name in (
        "MOTION_DIR",
        "OBJECT_SPEC_PATH",
        "OBJECT_URDF",
        "CONTACT_EXPORT_ROOT",
        "AS_CONTACT_EXPORT_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)

    with (
        patch(
            "holosoma.train_agent.training_provenance_from_env",
            return_value=provenance,
        ),
        pytest.raises(RuntimeError, match="requires exported MOTION_DIR"),
    ):
        _preflight_data_assets_before_sim()


def test_training_provenance_hashes_policy_init_and_resume_checkpoints(tmp_path):
    teacher, motion, object_map, contact, source_root = _fixture(tmp_path)
    policy_init = tmp_path / "student_init.pt"
    training_resume = tmp_path / "resume.pt"
    torch.save({"actor_model_state_dict": {"x": torch.tensor([1.0])}}, policy_init)
    torch.save({"actor_model_state_dict": {"x": torch.tensor([2.0])}}, training_resume)

    first = _compute(
        teacher,
        motion,
        object_map,
        contact,
        source_root,
        policy_init_checkpoint=policy_init,
        training_resume_checkpoint=training_resume,
    )
    torch.save({"actor_model_state_dict": {"x": torch.tensor([3.0])}}, policy_init)
    second = _compute(
        teacher,
        motion,
        object_map,
        contact,
        source_root,
        policy_init_checkpoint=policy_init,
        training_resume_checkpoint=training_resume,
    )

    assert first["policy_init_enabled"] is True
    assert first["training_resume_enabled"] is True
    assert first["policy_init_sha256"] != second["policy_init_sha256"]
    assert first["training_resume_sha256"] == second["training_resume_sha256"]


def test_training_provenance_rejects_teacher_contract_mismatch(tmp_path):
    teacher, motion, object_map, contact, source_root = _fixture(tmp_path)
    _teacher(teacher, motion_ends=False)
    with pytest.raises(ValueError, match="no motion_ends"):
        _compute(teacher, motion, object_map, contact, source_root)
    _teacher(teacher, actions=False)
    with pytest.raises(ValueError, match="no actions term"):
        _compute(teacher, motion, object_map, contact, source_root)
