"""Local GPU selection and complete-bank batching, independent of the host GPUs."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import _student, _teacher, _training_topology as topology
from scripts._training import bind_rank_shards, prepare


@pytest.mark.parametrize("count", range(1, 9))
@pytest.mark.parametrize("role,module", [("teacher", _teacher), ("distillation", _student)])
def test_visible_gpu_count_drives_entire_launch(count, role, module, monkeypatch):
    devices = ",".join(str(i) for i in reversed(range(count)))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", devices)
    args, bindings, cli, env, command = prepare(role, module.CLI, module.ENVIRONMENT, 1, [
        "--entity", "test", "--motion-bank", "/bank", "--check",
        *(["--contact-bank", "/contacts", "--robot-assets", "/robot"] if role == "distillation" else []),
    ])
    assert env["CUDA_VISIBLE_DEVICES"] == devices
    assert env["NPROC"] == env["HOLOSOMA_EXTERNAL_AS_WORLD_SIZE"] == str(count)
    assert f"--training.num-envs={count * 2048}" in cli
    assert f"--training.multigpu={count > 1}" in cli
    assert "--training.export-onnx=True" in cli
    assert f"--nproc_per_node={count}" in command
    assert "--standalone" in command and "--nnodes=1" in command
    assert args.prepare_shards


def test_auto_detection_respects_cuda_device_order(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    def query(command, **kwargs):
        assert kwargs["env"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        return SimpleNamespace(stdout="3\n")
    monkeypatch.setattr(topology.subprocess, "run", query)
    assert topology.visible_devices() == ["0", "1", "2"]
    assert topology.visible_devices(count=2) == ["0", "1"]
    with pytest.raises(ValueError, match="exceeds"):
        topology.visible_devices(count=4)


@pytest.mark.parametrize("value", ["", "-1", "0,0", "0,,1", "bad", "01,1"])
def test_bad_device_selections_fail(value, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    with pytest.raises(ValueError):
        topology.visible_devices(check=True)


def test_cpu_check_never_probes_cuda(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(topology.subprocess, "run", lambda *a, **k: pytest.fail("CUDA queried"))
    assert topology.visible_devices(check=True, count=5) == list("01234")


def test_invalid_physical_device_fails_before_workers(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,999")
    monkeypatch.setattr(topology.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="1\n"))
    with pytest.raises(ValueError, match="unavailable"):
        topology.visible_devices()


@pytest.mark.parametrize("count", range(1, 9))
def test_all_clips_survive_automatic_shards(count, tmp_path):
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "mesh.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    clips = {}
    for i in range(129):
        clip = f"clip_{i:03}"
        (bank / f"{clip}.npz").write_bytes(b"motion")
        (bank / f"{clip}.urdf").write_text(
            '<robot name="object"><link name="object"><visual><geometry>'
            '<mesh filename="mesh.obj"/></geometry></visual></link></robot>'
        )
        clips[clip] = {"object_urdf_path": f"{clip}.urdf"}
    (bank / "_clip_object_urdf_map.json").write_text(json.dumps({"clips": clips}))
    environments = topology.fit_environments(bank, count, 2048)
    assert environments == (1935 if count == 1 else 2048)
    args = SimpleNamespace(rank_shards=tmp_path / "shards", prepare_shards=True,
                           environments_per_rank=environments)
    manifest = bind_rank_shards(args, {"MOTION_BANK": str(bank)}, {}, world_size=count)
    assert set(manifest["clip_cover_counts"]) == set(clips)
    assert set(manifest["clip_cover_counts"].values()) == {1}
    counts = [len(list((args.rank_shards / f"rank_{rank}").glob("*.npz"))) for rank in range(count)]
    assert all(environments % n == 0 for n in counts)
    assert sum(counts) == 129
    # An existing tree cannot be silently rebuilt after its source changes.
    (bank / "clip_000.npz").write_bytes(b"changed")
    with pytest.raises((ValueError, RuntimeError)):
        bind_rank_shards(args, {"MOTION_BANK": str(bank)}, {}, world_size=count)


def test_too_small_environment_budget_does_not_drop_clips(tmp_path):
    for i in range(9):
        (tmp_path / f"{i}.npz").touch()
    with pytest.raises(ValueError, match="Cannot fit all 9 clips"):
        topology.fit_environments(tmp_path, 1, 8)


def test_cache_paths_follow_output_and_isolate_concurrent_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    shared = tmp_path / "shared cache"
    environments = []
    for name in ["run_a", "run_b"]:
        _, _, _, env, _ = prepare("distillation", _student.CLI, _student.ENVIRONMENT, 1, [
            "--entity", "test", "--output", str(tmp_path / name), "--cache-dir", str(shared), "--check",
        ])
        environments.append(env)
        for key in ["TMPDIR", "HOLOSOMA_OBJECT_USD_CACHE_DIR", "HOLOSOMA_ROBOT_USD_CACHE_DIR",
                    "HOLOSOMA_PERCEPTION_MESH_CACHE_DIR"]:
            assert Path(env[key]).is_relative_to(shared)
    assert environments[0]["TMPDIR"] != environments[1]["TMPDIR"]
    assert environments[0]["HOLOSOMA_ROBOT_USD_CACHE_DIR"] != environments[1]["HOLOSOMA_ROBOT_USD_CACHE_DIR"]
    assert environments[0]["HOLOSOMA_PERCEPTION_MESH_CACHE_DIR"] == environments[1]["HOLOSOMA_PERCEPTION_MESH_CACHE_DIR"]
    assert not shared.exists()
