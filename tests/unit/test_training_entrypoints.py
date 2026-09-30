"""Exercise the public entry points without a simulator or W&B writes."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import _training
from scripts import _teacher as train_teacher
from scripts import _student as train_distillation


def arguments(role, tmp_path, *, check=False):
    args = ["--motion-bank", str(tmp_path / "motion bank"), "--entity", "test",
            "--output", str(tmp_path / "output"), "--master-addr", "10.0.0.1",
            "--source-commit", "a" * 40]
    if role == "distillation":
        for flag in ("contact-bank", "robot-assets", "teacher-checkpoint", "initializer-checkpoint"):
            args.extend(["--" + flag, str(tmp_path / flag)])
    return args + (["--check"] if check else [])


@pytest.mark.parametrize("role,module,nodes", [("teacher", train_teacher, 4), ("distillation", train_distillation, 1)])
def test_check_has_no_launch_side_effects(role, module, nodes, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", *arguments(role, tmp_path, check=True)])
    calls = []
    monkeypatch.setattr(_training, "run", lambda command, env: calls.append(command))
    monkeypatch.setattr(_training.os, "execve", lambda *args: pytest.fail("training started during --check"))
    _training.launch(role, module.CLI, module.ENVIRONMENT, nodes=nodes)
    assert len(calls) == 1 and Path(calls[0][1]).name == "validate_train_cli.py"
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("role,module,nodes", [("teacher", train_teacher, 4), ("distillation", train_distillation, 1)])
@pytest.mark.parametrize("fail_preflight", [False, True])
def test_launch_runs_preflight_before_torchrun(role, module, nodes, fail_preflight, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", *arguments(role, tmp_path)])
    monkeypatch.setattr(_training.os, "chdir", lambda path: None)
    monkeypatch.setattr(_training, "bind_nccl", lambda env: None)
    monkeypatch.setattr("scripts._runtime_source.resolve_runtime_source", lambda *args, **kwargs: {
        "root": tmp_path / "runtime", "remote": "https://example.org/runtime.git",
        "commit": "c" * 40, "ref": "c" * 40,
    })
    monkeypatch.setattr("holosoma.utils.package_source.verify_runtime_installation", lambda *args: {})
    calls, executions = [], []

    def run(command, env, *, capture=False):
        words = [str(word) for word in command]
        calls.append(words)
        if words[:3] == ["git", "remote", "get-url"]:
            return "https://example.org/repo.git\n"
        if words[0] == "git" and "rev-parse" in words:
            return "b" * 40 + "\n"
        if "compute_training_provenance.py" in words[1]:
            return json.dumps({"teacher_enabled": role == "distillation"})
        if words[1].endswith(("box23k_policy_init_preflight.py", "_training.py")) and fail_preflight:
            raise subprocess.CalledProcessError(1, words)

    monkeypatch.setattr(_training, "run", run)
    monkeypatch.setattr(_training.os, "execve", lambda *args: executions.append(args))
    if fail_preflight:
        with pytest.raises(subprocess.CalledProcessError):
            _training.launch(role, module.CLI, module.ENVIRONMENT, nodes=nodes)
        assert not executions
    else:
        _training.launch(role, module.CLI, module.ENVIRONMENT, nodes=nodes)
        assert len(executions) == 1
        command, env = executions[0][1:]
        assert command[1:3] == ["-m", "torch.distributed.run"]
        assert f"--nnodes={nodes}" in command and "--max_restarts=0" in command
        assert json.loads(env["HOLOSOMA_TRAINING_PROVENANCE"])["teacher_enabled"] == (role == "distillation")
        assert "--training.export-onnx=True" in command


def test_stale_training_identity_and_semantic_overrides_are_cleared(tmp_path, monkeypatch):
    for key in ("WANDB_RUN_ID", "WANDB_DISABLED", "HOLOSOMA_TRAINING_PROVENANCE", "HOLOSOMA_SKIP_INITIAL_CHECKPOINT"):
        monkeypatch.setenv(key, "stale")
    _, _, cli, env, command = _training.prepare("teacher", train_teacher.CLI, train_teacher.ENVIRONMENT, 4,
                                              arguments("teacher", tmp_path, check=True))
    assert "WANDB_RUN_ID" not in env and "WANDB_DISABLED" not in env
    assert "HOLOSOMA_TRAINING_PROVENANCE" not in env
    assert env["HOLOSOMA_SKIP_INITIAL_CHECKPOINT"] == "1"
    # Paths are argv elements; spaces must never be split or shell-evaluated.
    assert f"--command.setup-terms.motion-command.params.motion-config.motion-file={tmp_path / 'motion bank'}" in cli


def test_mutable_source_commit_is_rejected_before_launch(tmp_path):
    args = arguments("teacher", tmp_path)
    args[-1] = "main"
    with pytest.raises(SystemExit):
        _training.prepare("teacher", train_teacher.CLI, train_teacher.ENVIRONMENT, 4, args)
