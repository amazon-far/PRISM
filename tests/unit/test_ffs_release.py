"""Offline checks for FFS geometry, relays, startup and policy restart state."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zmq
from holosoma.config_values.image_server import real_d435i, remote_ffs_d435i
from holosoma.models.ffs.infer import FFSConfig, FastFoundationStereo
from holosoma.sensors.remote_stereo import RemoteStereoCamera, RemoteStereoCameraConfig, HEADER_FMT, MAGIC
from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('depth_check', ROOT / 'scripts/check_depth_stream.py')
depth_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(depth_check)


def test_ffs_depth_profile_preserves_the_new_source_latency():
    assert remote_ffs_d435i.latency_frame == (1, 1)
    assert remote_ffs_d435i.buffer_len == 2
    assert remote_ffs_d435i.camera_type == 'remote_stereo'
    assert remote_ffs_d435i.depth_source == 'depth_gum'
    assert real_d435i.latency_frame == (3, 3)
    assert (remote_ffs_d435i.near_clip, remote_ffs_d435i.far_clip) == (0.3, 3.0)


@pytest.mark.parametrize('disparity,expected', [(2., 1.), (0., 10.), (-1., 10.), (float('nan'), 10.)])
def test_metric_depth_accounts_for_resizing_and_invalid_disparity(disparity, expected):
    sys.path.insert(0, str(ROOT / 'third_party/Fast-FoundationStereo'))
    from core.utils.utils import InputPadder
    predictor = FastFoundationStereo.__new__(FastFoundationStereo)
    predictor.cfg = FFSConfig(device='cpu', infer_height=32, infer_width=32)
    predictor.torch, predictor.device = torch, torch.device('cpu')
    predictor._InputPadder = InputPadder
    def forward(left, right, **kwargs):
        assert left.max() == 255 and right.max() == 0  # Raw 0–255 input contract.
        assert kwargs['optimize_build_volume'] == 'pytorch1'
        return torch.full((1, 1, *left.shape[-2:]), disparity)
    predictor.model = SimpleNamespace(forward=forward)
    pair = np.concatenate([np.full((64, 64, 3), 255, np.uint8), np.zeros((64, 64, 3), np.uint8)], axis=1)
    intr = np.stack([np.diag([80., 80., 1.])] * 2)
    extr = np.stack([np.eye(4)] * 2)
    extr[1, 0, 3] = .05
    depth = predictor.predict(pair, intr, extr)
    assert depth.shape == (64, 64)
    np.testing.assert_allclose(depth, expected)


def test_zero_baseline_is_rejected():
    with pytest.raises(ValueError, match='Non-positive'):
        FastFoundationStereo._baseline_from_extrinsics(np.stack([np.eye(4)] * 2))


def test_stereo_receiver_survives_bad_packets_and_preserves_calibration():
    context = zmq.Context.instance()
    publisher = context.socket(zmq.PUB)
    port = publisher.bind_to_random_port('tcp://127.0.0.1')
    camera = RemoteStereoCamera(RemoteStereoCameraConfig(connect=f'tcp://127.0.0.1:{port}', img_shape=(4, 6), recv_timeout_s=1))
    intr = np.stack([np.diag([80., 80., 1.])] * 2).astype(np.float32)
    extr = np.stack([np.eye(4)] * 2).astype(np.float32)
    extr[1, 0, 3] = .05
    calibration = np.concatenate([intr.ravel(), extr.ravel()]).tobytes()
    left = np.arange(24, dtype=np.uint8).reshape(4, 6)
    right = left + 1
    def packet(seq, payload):
        return struct.pack(HEADER_FMT, MAGIC, seq, 0., 4, 6, 1, 0) + calibration + payload
    try:
        time.sleep(.15)
        publisher.send(packet(0, b'broken'))
        time.sleep(.05)
        for seq in range(1, 30):
            publisher.send(packet(seq, left.tobytes() + right.tobytes()))
            time.sleep(.01)
            if camera.calibration is not None:
                break
        frame = camera.capture()
        np.testing.assert_array_equal(camera.calibration['intrinsics'], intr)
        np.testing.assert_array_equal(camera.calibration['extrinsics'], extr)
        np.testing.assert_array_equal(frame['rgb'][:, :6, 0], left)
        np.testing.assert_array_equal(frame['rgb'][:, 6:, 2], right)
        assert camera._thread.is_alive()
    finally:
        camera.release()
        publisher.close(linger=0)


def test_depth_check_rejects_stale_and_nonfinite_frames():
    with pytest.raises(ValueError, match='not updating'):
        depth_check.inspect_frames([np.zeros((2, 2))] * 5)
    with pytest.raises(ValueError, match='Non-finite'):
        depth_check.inspect_frames([np.full((2, 2), np.nan)])
    distinct, _ = depth_check.inspect_frames([np.full((2, 2), i / 10) for i in range(5)])
    assert distinct == 5
    assert depth_check.inspect_frames([np.full((2, 2), .5)] * 5) == (1, 1.)


def test_policy_restart_resets_actions_and_operator_buttons():
    policy = WholeBodyTrackingPolicy.__new__(WholeBodyTrackingPolicy)
    policy.last_policy_action = np.ones((1, 29))
    policy.scaled_policy_action = np.ones((1, 29))
    policy._pickup_button_command, policy._drop_button_command = 0, 1
    policy._pickup_button_startup, policy._drop_button_startup = 1, 0
    policy.logger = SimpleNamespace(info=lambda *a: None)
    policy._reset_policy_io_state('test')
    assert not policy.last_policy_action.any() and not policy.scaled_policy_action.any()
    assert (policy._pickup_button_command, policy._drop_button_command) == (1, 0)


@pytest.mark.parametrize('script', ['real_ffs.sh', 'real_ffs_depth.sh', 'real_ffs_run.sh', 'install.sh'])
def test_launch_help_does_not_require_models_or_start_devices(script):
    env = {**os.environ, 'PRISM_PYTHON': '/not/an/interpreter'}
    result = subprocess.run(['bash', str(ROOT / script), '--help'], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'Usage:' in result.stdout


def test_standalone_depth_launcher_uses_submodule_and_explicit_model(tmp_path):
    (tmp_path / 'scripts').mkdir()
    for name in ['real_ffs_depth.sh', 'scripts/ffs_common.sh']:
        shutil.copy2(ROOT / name, tmp_path / name)
    upstream = tmp_path / 'third_party/Fast-FoundationStereo/core'
    upstream.mkdir(parents=True)
    (upstream / 'foundation_stereo.py').touch()
    model = tmp_path / 'model with spaces.pth'
    model.touch()
    (tmp_path / 'cfg.yaml').touch()
    fake = tmp_path / 'python'
    record = tmp_path / 'arguments.json'
    fake.write_text('#!/usr/bin/env python3\nimport os,sys,json\nfrom pathlib import Path\nPath(os.environ["ARGUMENT_RECORD"]).write_text(json.dumps({"args":sys.argv[1:],"repo":os.environ["HOLOSOMA_FFS_REPO"],"model":os.environ["HOLOSOMA_FFS_MODEL"]}))\n')
    fake.chmod(0o755)
    # Prevent touching any real shared-memory segment during the launcher test.
    pgrep = tmp_path / 'pgrep'
    pgrep.write_text('#!/bin/sh\nexit 0\n')
    pgrep.chmod(0o755)
    env = {k:v for k,v in os.environ.items() if not k.startswith('HOLOSOMA_')}
    env.update(PRISM_PYTHON=str(fake), ARGUMENT_RECORD=str(record), PATH=f'{tmp_path}:{os.environ["PATH"]}')
    result = subprocess.run(['bash', str(tmp_path / 'real_ffs_depth.sh'), '--ffs-model', model.name,
                             '--connect', 'tcp://127.0.0.1:5602'], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr + result.stdout
    data = json.loads(record.read_text())
    assert data['model'] == str(model)
    assert data['repo'] == str(upstream.parent)
    assert data['args'][:3] == ['-m', 'holosoma.sensors.image_server', 'remote_ffs_d435i']
