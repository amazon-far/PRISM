"""Rollout must cover the supplied bank without inheriting training state."""

import json
from pathlib import Path

import pytest

from scripts._rollout import prepare
from holosoma.simulator.isaacsim.usd_cache import resolve_robot_usd_conversion_dir


def inputs(tmp_path):
    bank = tmp_path / "motion bank"
    bank.mkdir()
    (bank / "_clip_object_urdf_map.json").write_text(json.dumps({"clips": {"a": {}, "b": {}}}))
    for clip in ("a", "b"):
        (bank / f"{clip}.npz").touch()
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.touch()
    return bank, ["--checkpoint", str(checkpoint), "--motion-bank", str(bank),
                  "--output", str(tmp_path / "output"), "--check"]


def test_full_coverage_and_native_rollout_environment(tmp_path, monkeypatch):
    bank, argv = inputs(tmp_path)
    for key in ("WANDB_RUN_ID", "HOLOSOMA_TRAINING_PROVENANCE", "HOLOSOMA_EVAL_POLICY", "WORLD_SIZE"):
        monkeypatch.setenv(key, "stale")
    _, cli, env, command = prepare(argv)
    assert "--training.num-envs=2" in cli
    assert f"--command.setup-terms.motion-command.params.motion-config.motion-file={bank}" in cli
    assert "--max-rollout-steps" not in cli
    assert env["HOLOSOMA_EVAL_POLICY"] == "checkpoint_actor"
    assert env["HOLOSOMA_DISABLE_AUTO_RESET"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert all(key not in env for key in ("WANDB_RUN_ID", "HOLOSOMA_TRAINING_PROVENANCE", "WORLD_SIZE"))
    assert command[1:3] == ["-m", "holosoma.export_teacher_box_contacts"]
    assert not (tmp_path / "output").exists()


def test_missing_clip_cannot_be_silently_filtered(tmp_path):
    bank, argv = inputs(tmp_path)
    (bank / "b.npz").unlink()
    with pytest.raises(SystemExit):
        prepare(argv)


def test_existing_output_is_preserved(tmp_path):
    _, argv = inputs(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / "keep.txt").write_text("existing output")
    with pytest.raises(SystemExit):
        prepare(argv)
    assert (output / "keep.txt").read_text() == "existing output"


def test_parallel_rollouts_do_not_share_forced_robot_usd_conversion(tmp_path, monkeypatch):
    _, argv = inputs(tmp_path)
    monkeypatch.setenv("HOLOSOMA_ROBOT_USD_CACHE_DIR", str(tmp_path / "stale_shared_cache"))
    _, _, first_env, _ = prepare(argv)
    second_argv = list(argv)
    second_argv[second_argv.index("--output") + 1] = str(tmp_path / "other_output")
    _, _, second_env, _ = prepare(second_argv)
    asset_root = tmp_path / "shared_checkout" / "robots"
    first_cache = resolve_robot_usd_conversion_dir(asset_root, 0, environ=first_env)
    second_cache = resolve_robot_usd_conversion_dir(asset_root, 0, environ=second_env)
    assert first_cache != second_cache
    assert not first_cache.is_relative_to(asset_root)
    assert not second_cache.is_relative_to(asset_root)
    assert first_env["HOLOSOMA_OBJECT_USD_CACHE_DIR"] != second_env["HOLOSOMA_OBJECT_USD_CACHE_DIR"]
    assert not Path(first_env["HOLOSOMA_ROBOT_USD_CACHE_DIR"]).exists()
