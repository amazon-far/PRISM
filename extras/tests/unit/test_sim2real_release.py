"""Offline deployment regression checks; never construct a robot interface."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from holosoma.config_values.image_server import real_d435i
from holosoma.sensors.image_server import ImageServer
from holosoma_inference.config.config_values.inference import (
    g1_root_pos_contact_aware_drop_button_actions_no_linvel_h1 as preset,
)
from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy
from holosoma_inference.run_policy import _select_policy_class


ROOT = Path(__file__).resolve().parents[3]


def test_copied_sources_match_the_release_manifest():
    manifest = json.loads((ROOT / "extras/provenance/source_manifest.json").read_text())
    assert manifest["source_commit"] == "87760c7b8676ca7f9acc295de9bb2ab233e91537"
    for record in manifest["files"]:
        actual = hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest()
        assert actual == record["release_sha256"], record["path"]
        if record["unchanged"]:
            assert actual == record["source_sha256"], record["path"]


def test_physical_camera_profile_and_depth_processing():
    assert (real_d435i.frame_rate, real_d435i.latency_frame, real_d435i.buffer_len) == (30, (3, 3), 4)
    assert (real_d435i.near_clip, real_d435i.far_clip) == (0.3, 3.0)
    server = SimpleNamespace(cfg=real_d435i)
    raw = np.random.default_rng(42).uniform(0, 4, (480, 848)).astype(np.float32)
    raw[::7] = 0
    # The source revision's positional OpenCV call uses effective linear interpolation.
    expected = cv2.resize(raw[16:, 32:-32], (87, 58), interpolation=cv2.INTER_LINEAR)
    expected = (np.clip(expected, 0.3, 3.0) - 0.3) / 2.7 - 0.5
    actual = ImageServer._resize_clip_expand_transpose(server, raw)
    np.testing.assert_array_equal(actual, expected[None])
    zeros = ImageServer._resize_clip_expand_transpose(server, np.zeros_like(raw))
    np.testing.assert_array_equal(zeros, np.full((1, 58, 87), -0.5, np.float32))


@pytest.mark.parametrize("axes,keys,expected", [
    ((0, 0.04, 0), 0, [0, 0, 0]),
    ((0, 0.05, 0), 0, [0, 0, 0]),
    ((0, 0.06, 0), 0, [0.07, 0, 0]),
    ((0, 1, 0), 0, [0.07, 0, 0]),
    ((0, -1, 0), 0, [-0.07, 0, 0]),
    ((0.5, 0, -0.5), 0, [0, -0.035, 0.05]),
    ((1, 1, 1), 1, [0, 0, 0]),
])
def test_joystick_retains_source_commands(axes, keys, expected):
    policy = WholeBodyTrackingPolicy.__new__(WholeBodyTrackingPolicy)
    message = SimpleNamespace(lx=axes[0], ly=axes[1], rx=axes[2], keys=keys)
    policy.interface = SimpleNamespace(get_joystick_msg=lambda: message)
    policy._joystick_sparse_root_command_offset = np.ones((1, 3), np.float32)
    policy._update_sparse_root_joystick_command()
    np.testing.assert_allclose(policy._joystick_sparse_root_command_offset, [expected], atol=1e-8)


def test_drop_button_latches_on_key_press():
    policy = WholeBodyTrackingPolicy.__new__(WholeBodyTrackingPolicy)
    policy.obs_dims = {"drop_button": 1}
    policy._drop_button_command = 0.0
    policy._drop_button_key_down = False
    policy.logger = SimpleNamespace(info=lambda *args: None)
    assert policy._handle_drop_button_keyboard_command("g")
    assert policy._drop_button_command == 1.0
    assert policy._handle_drop_button_keyboard_command("g")
    assert policy._drop_button_command == 1.0
    policy._drop_button_key_down = False
    policy._handle_drop_button_keyboard_command("g")
    assert policy._drop_button_command == 0.0


def test_real_preset_keeps_policy_selection_and_explicit_model_input():
    assert _select_policy_class(preset) is WholeBodyTrackingPolicy
    assert not any(preset.task.model_path)
    assert preset.task.rl_rate == 50
    assert preset.task.auto_start_policy is False
    assert preset.task.auto_start_motion_clip is False
    assert preset.observation.obs_dims["drop_button"] == 1


def test_launcher_preserves_model_with_spaces_and_selected_interface(tmp_path):
    launcher = tmp_path / "real_drop.sh"
    shutil.copy2(ROOT / "real_drop.sh", launcher)
    model = tmp_path / "policy with spaces.onnx"
    model.write_bytes(b"launcher argument test only")
    fake_python = tmp_path / "fake_python"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "if sys.argv[1:3] == ['-m', 'holosoma_inference.run_policy']:\n"
        "    Path(os.environ['ARGUMENT_RECORD']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    fake_python.chmod(0o755)
    output = tmp_path / "arguments.json"
    env = {**os.environ, "PRISM_PYTHON": str(fake_python), "ARGUMENT_RECORD": str(output)}
    result = subprocess.run(["bash", str(launcher), "--model", model.name, "--interface", "eth9"],
                            cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    args = json.loads(output.read_text())
    assert args[args.index("--task.model-path") + 1] == str(model)
    assert args[args.index("--task.interface") + 1] == "eth9"
    assert args[args.index("--task.rl-rate") + 1] == "50"
    assert "--task.use-joystick" in args


def test_missing_model_fails_before_starting_any_process(tmp_path):
    launcher = tmp_path / "real_drop.sh"
    shutil.copy2(ROOT / "real_drop.sh", launcher)
    env = {key: value for key, value in os.environ.items() if key != "HOLOSOMA_REAL_MODEL_PATH"}
    env["PRISM_PYTHON"] = "/must/not/be/executed"
    result = subprocess.run(["bash", str(launcher)], cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert "Provide an existing ONNX" in result.stderr
    assert not (tmp_path / "logs").exists()
