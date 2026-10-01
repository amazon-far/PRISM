import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts._teacher import CLI, ENVIRONMENT
from scripts._student import CLI as STUDENT_CLI, ENVIRONMENT as STUDENT_ENVIRONMENT
from scripts._training import bind_rank_shards, prepare
from scripts.prepare_as_rank_shards import prepare_rank_shards


def test_teacher_launch_uses_installed_module(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/unrelated/source/tree")
    _, _, _, env, command = prepare("teacher", [], {}, 4, [
        "--motion-bank", "/external/motions", "--entity", "test", "--check",
    ])
    assert command[-2:] == ["--module", "holosoma.train_agent_rank_visible"]
    assert env["PYTHONPATH"] == str(Path(__file__).resolve().parents[2])
    assert not (Path(env["PYTHONPATH"]) / "src").exists()


def test_formal_launch_detects_source_identity_automatically():
    args, _, _, _, _ = prepare("teacher", [], {}, 4, [
        "--motion-bank", "/external/motions", "--entity", "test", "--master-addr", "localhost",
    ])
    assert args.source_commit is None
    assert args.runtime_source is None
    assert args.runtime_commit is None


def test_explicit_runtime_commit_must_be_immutable():
    with pytest.raises(SystemExit):
        prepare("teacher", [], {}, 4, [
            "--motion-bank", "/external/motions", "--entity", "test",
            "--runtime-commit", "main", "--master-addr", "localhost",
        ])


def test_single_node_teacher_binds_batch_topology_and_explicit_shards(tmp_path):
    shards = tmp_path / "eight rank shards"
    _, _, cli, env, command = prepare("teacher", CLI, ENVIRONMENT, 4, [
        "--motion-bank", "/external/motions", "--entity", "test", "--check",
        "--nodes", "1", "--rank-shards", str(shards),
    ])
    assert "--nnodes=1" in command and "--master_addr=127.0.0.1" in command
    assert "--training.num-envs=16384" in cli
    assert env["NNODES"] == "1" and env["HOLOSOMA_EXTERNAL_AS_WORLD_SIZE"] == "8"
    assert env["HOLOSOMA_GLOO_GRAD_REDUCE"] == "1"
    assert env["HOLOSOMA_HIERARCHICAL_GRAD_REDUCE"] == "0"
    assert env["HOLOSOMA_MOTION_SHARD_MANIFEST"] == str(shards / "manifest.json")
    assert "HOLOSOMA_EXTERNAL_AS_RANK_SHARD_SOURCE_DIGEST" not in env
    assert not shards.exists()  # --check never prepares or reads data.


@pytest.mark.parametrize("extra", [[], ["--rank-shards", "/shards", "--node-rank", "1"]])
def test_single_node_teacher_rejects_missing_shards_or_invalid_node_rank(extra):
    with pytest.raises(SystemExit):
        prepare("teacher", CLI, ENVIRONMENT, 4, [
            "--motion-bank", "/external/motions", "--entity", "test", "--check",
            "--nodes", "1", *extra,
        ])


def test_default_teacher_preserves_four_node_recipe():
    _, _, cli, env, command = prepare("teacher", CLI, ENVIRONMENT, 4, [
        "--motion-bank", "/external/motions", "--entity", "test", "--check",
    ])
    assert "--nnodes=4" in command and "--training.num-envs=65536" in cli
    assert env["HOLOSOMA_EXTERNAL_AS_WORLD_SIZE"] == "32"
    assert env["HOLOSOMA_HIERARCHICAL_GRAD_REDUCE"] == "1"
    assert env["HOLOSOMA_MOTION_SHARD_MANIFEST"].endswith("/ws32/manifest.json")


def test_student_accepts_new_dataset_shards_without_historical_identity(tmp_path):
    shards = tmp_path / "student shards"
    _, _, cli, env, command = prepare("distillation", STUDENT_CLI, STUDENT_ENVIRONMENT, 1, [
        "--motion-bank", "/external/motions", "--contact-bank", "/external/contacts",
        "--robot-assets", "/external/robot", "--teacher-checkpoint", "/teacher.pt",
        "--initializer-checkpoint", "/box.pt", "--rank-shards", str(shards),
        "--entity", "test", "--check",
    ])
    assert "--nnodes=1" in command and "--training.num-envs=16384" in cli
    assert "--training.export-onnx=True" in cli
    assert "--algo.config.distill.strict-teacher-load=True" in cli
    assert "--command.setup-terms.motion-command.params.motion-config.contact-aware-button-window-mode=peak_height" in cli
    assert env["HOLOSOMA_MOTION_SHARD_MANIFEST"] == str(shards / "manifest.json")
    assert env["HOLOSOMA_RANK_LOCAL_MOTION_ROOT"] == str(shards)
    for key in ("HOLOSOMA_EXTERNAL_AS_RANK_SHARD_SOURCE_DIGEST",
                "HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_SOURCE_DIGEST",
                "HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_VIEW_DIGEST"):
        assert key not in env
    assert not shards.exists()


def test_shard_binding_validates_source_and_topology(tmp_path):
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "object.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    (bank / "object.urdf").write_text(
        '<robot name="object"><link name="baseLink"><visual><geometry>'
        '<mesh filename="object.obj"/></geometry></visual></link></robot>'
    )
    clips = {}
    for i in range(9):
        (bank / f"clip_{i}.npz").write_bytes(b"motion payload")
        clips[f"clip_{i}"] = {"object_urdf_path": "object.urdf"}
    (bank / "_clip_object_urdf_map.json").write_text(json.dumps({"clips": clips}))
    shards = tmp_path / "shards"
    published = prepare_rank_shards(motion_dir=bank, object_map=bank / "_clip_object_urdf_map.json",
                                    output_root=shards, world_size=8, environments_per_rank=2048)
    env = {}
    args = SimpleNamespace(nodes=1, rank_shards=shards)
    bound = bind_rank_shards(args, {"MOTION_BANK": str(bank)}, env, world_size=8)
    assert bound["clip_count"] == 9
    assert env["HOLOSOMA_EXTERNAL_AS_RANK_SHARD_SOURCE_DIGEST"] == published["source_digest"]
    assert set(bound["clip_cover_counts"].values()) == {1}
    with pytest.raises((ValueError, RuntimeError)):
        bind_rank_shards(args, {"MOTION_BANK": str(bank)}, {}, world_size=32)
    (bank / "clip_0.npz").write_bytes(b"changed motion payload")
    rejected_env = {}
    with pytest.raises((ValueError, RuntimeError)):
        bind_rank_shards(args, {"MOTION_BANK": str(bank)}, rejected_env, world_size=8)
    assert "HOLOSOMA_EXTERNAL_AS_RANK_SHARD_SOURCE_DIGEST" not in rejected_env
