import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts._teacher import CLI, ENVIRONMENT
from scripts._student import CLI as STUDENT_CLI, ENVIRONMENT as STUDENT_ENVIRONMENT
from scripts._training import ROOT, STUDENT_DATA, bind_rank_shards, prepare
from scripts.prepare_as_rank_shards import prepare_rank_shards


@pytest.fixture(autouse=True)
def cpu_launch_planning(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setattr("scripts._training_topology.subprocess.run",
                        lambda *a, **kw: SimpleNamespace(stdout="8"))
    monkeypatch.setattr("scripts._training_topology.fit_environments", lambda bank, world, budget: budget)


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
    assert "--nnodes=1" in command and "--standalone" in command
    assert "--training.num-envs=16384" in cli
    assert env["NNODES"] == "1" and env["HOLOSOMA_EXTERNAL_AS_WORLD_SIZE"] == "8"
    assert env["HOLOSOMA_GLOO_GRAD_REDUCE"] == "1"
    assert env["HOLOSOMA_HIERARCHICAL_GRAD_REDUCE"] == "0"
    assert env["HOLOSOMA_MOTION_SHARD_MANIFEST"] == str(shards / "manifest.json")
    assert "HOLOSOMA_EXTERNAL_AS_RANK_SHARD_SOURCE_DIGEST" not in env
    assert not shards.exists()  # --check never prepares or reads data.


def test_single_machine_teacher_prepares_shards_automatically():
    args, _, cli, env, command = prepare("teacher", CLI, ENVIRONMENT, 1, [
        "--motion-bank", "/external/motions", "--entity", "test", "--check",
    ])
    assert args.prepare_shards
    assert "--nnodes=1" in command and "--training.num-envs=16384" in cli
    assert env["HOLOSOMA_HIERARCHICAL_GRAD_REDUCE"] == "0"
    assert env["HOLOSOMA_MOTION_SHARD_MANIFEST"].endswith("/rank_shards/manifest.json")


def test_invalid_machine_rank():
    with pytest.raises(SystemExit):
        prepare("teacher", CLI, ENVIRONMENT, 1, [
            "--motion-bank", "/external/motions", "--entity", "test", "--check", "--machine-rank", "1",
        ])


def test_explicit_multiple_machines_preserve_batch_and_reduction():
    _, _, cli, env, command = prepare("teacher", CLI, ENVIRONMENT, 1, [
        "--motion-bank", "/external/motions", "--entity", "test", "--check", "--machines", "4",
    ])
    assert "--nnodes=4" in command and "--training.num-envs=65536" in cli
    assert env["HOLOSOMA_EXTERNAL_AS_WORLD_SIZE"] == "32"
    assert env["HOLOSOMA_HIERARCHICAL_GRAD_REDUCE"] == "1"


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


def test_short_student_command_matches_explicit_recipe_outside_repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    short = prepare("distillation", STUDENT_CLI, STUDENT_ENVIRONMENT, 1,
                    ["--entity", "test", "--check"])
    explicit = prepare("distillation", STUDENT_CLI, STUDENT_ENVIRONMENT, 1, [
        "--motion-bank", str(STUDENT_DATA / "motion_bank"),
        "--contact-bank", str(STUDENT_DATA / "contact_sidecars"),
        "--robot-assets", str(STUDENT_DATA / "robot_assets"),
        "--rank-shards", str(ROOT / "data" / "student_shards_ws8"),
        "--teacher-checkpoint", str(ROOT / "_ckpts" / "teacher_40000.pt"),
        "--initializer-checkpoint", str(ROOT / "_ckpts" / "box_23000.pt"),
        "--entity", "test", "--check",
    ])
    # The complete effective CLI, environment and distributed command agree.
    assert short[1:] == explicit[1:]
    assert short[0].rank_shards == ROOT / "data" / "student_shards_ws8"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("extra", [[], ["--contact-bank", "/custom/contacts"],
                                  ["--robot-assets", "/custom/robot"]])
def test_custom_student_bank_cannot_implicitly_mix_with_downloaded_assets(extra):
    with pytest.raises(SystemExit):
        prepare("distillation", STUDENT_CLI, STUDENT_ENVIRONMENT, 1, [
            "--motion-bank", "/custom/motions", "--entity", "test", "--check", *extra,
        ])


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
    args = SimpleNamespace(rank_shards=shards, prepare_shards=False, environments_per_rank=2048)
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
