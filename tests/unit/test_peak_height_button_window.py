from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from holosoma.managers.command.terms.wbt import (
    MotionCommand,
    _contact_aware_carry_window_from_peak_height,
)
from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy
from holosoma_inference.utils.button_window_contract import (
    EMBEDDED_BUTTON_WINDOW_CONTRACT_KEY,
    EMBEDDED_BUTTON_WINDOW_CONTRACT_SHA256_KEY,
    build_source_button_window_contract,
    height_button_window_from_motion_np,
)
from holosoma_inference.utils.contact_sidecar_contract import policy_requires_contact_window


PEAK_CONFIG = {"contact_aware_button_window_mode": "peak_height"}


@pytest.mark.parametrize("smoothing", [1, 5, 11])
@pytest.mark.parametrize("alpha", [0.0, 0.91, 1.0])
def test_peak_button_numpy_matches_training_including_short_and_flat_traces(smoothing, alpha):
    config = dict(PEAK_CONFIG, contact_aware_peak_height_alpha=alpha,
                  contact_aware_peak_height_smoothing_steps=smoothing)
    rng = np.random.default_rng(42)
    traces = [np.zeros(0), np.ones(1), np.full(20, 0.5),
              np.r_[np.zeros(10), np.linspace(0, 1, 15), np.ones(25), np.linspace(1, 0, 20)],
              rng.uniform(0.1, 0.9, 200)]
    for trace in traces:
        z = np.asarray(trace, dtype=np.float32)
        expected = _contact_aware_carry_window_from_peak_height(
            torch.from_numpy(z.copy()), peak_height_alpha=alpha, smoothing_steps=smoothing,
        )
        assert height_button_window_from_motion_np(z, rng.uniform(0, 2, len(z)), config) == expected


def test_training_peak_buttons_ignore_root_motion_and_contact_sidecars():
    z = torch.tensor([0.2] * 10 + [0.8] * 20 + [0.2] * 10)
    command = object.__new__(MotionCommand)
    command.device = "cpu"
    command.motion_cfg = SimpleNamespace(**PEAK_CONFIG, contact_aware_carry_window_mode="rel_z")
    command.motion = SimpleNamespace(
        has_object=True, num_clips=2, clip_offsets=torch.tensor([0, len(z)]),
        clip_lengths=torch.tensor([len(z), len(z)]),
        object_pos_w=torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], -1).repeat(2, 1),
        body_pos_w=torch.rand(2 * len(z), 1, 3),
    )
    command._adaptive_sampling_contact_window_by_clip = torch.tensor([[1, 2], [3, 4]])
    command._adaptive_sampling_contact_window_valid_by_clip = torch.tensor([True, True])
    expected = _contact_aware_carry_window_from_peak_height(z)
    assert command._get_contact_aware_button_window_by_clip().tolist() == [list(expected)] * 2
    command.num_envs = 3
    command.clip_ids = torch.tensor([0, 1, 1])
    command.time_steps = torch.tensor([expected[1] - 1, expected[1], expected[1] + 1])
    assert command.get_contact_aware_drop_button().tolist() == [False, True, True]


def _policy_and_contract(prepend=10):
    z = np.array([0.2] * 10 + [0.8] * 20 + [0.2] * 10, dtype=np.float32)
    window = height_button_window_from_motion_np(z, np.zeros_like(z), PEAK_CONFIG)
    contract, digest = build_source_button_window_contract(
        motion_config=PEAK_CONFIG, clip_id="clip", source_motion_sha256="a" * 64,
        source_motion_size=100, source_frame_count=len(z), motion_fps=50,
        source_window=window, motion_transition_contract_sha256="b" * 64,
        source_semantics="global_multi_clip_runtime", effective_prepend_steps=prepend,
        effective_append_steps=0,
    )
    materialized = np.r_[np.full(prepend, z[0]), z].astype(np.float32)
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._motion_cfg = dict(PEAK_CONFIG)
    policy._motion_transition_prepend_steps = prepend
    policy._effective_motion_transition_settings = {
        "source_semantics": "global_multi_clip_runtime", "contract_sha256": "b" * 64,
    }
    policy._motion_data = SimpleNamespace(
        has_object=True, object_pos_w=np.stack([np.zeros_like(materialized)] * 2 + [materialized], -1),
        root_pos_w=np.zeros((len(materialized), 3), dtype=np.float32),
        source_frame_count=len(z), frame_count=len(materialized), motion_path=Path("clip.npz"),
        source_sha256="a" * 64, source_size=100, fps=50,
    )
    policy._onnx_metadata = {
        EMBEDDED_BUTTON_WINDOW_CONTRACT_KEY: contract,
        EMBEDDED_BUTTON_WINDOW_CONTRACT_SHA256_KEY: digest,
    }
    return policy, window


def test_peak_inference_binds_mode_threshold_and_runtime_prepend():
    policy, window = _policy_and_contract()
    assert policy._load_contact_aware_button_window(Path("unused.onnx")) == tuple(t + 10 for t in window)
    policy._motion_cfg["contact_aware_peak_height_alpha"] = 0.8
    with pytest.raises(RuntimeError, match="peak_height_alpha"):
        policy._load_contact_aware_button_window(Path("unused.onnx"))


def test_peak_inference_rejects_missing_contract_and_wrong_mode():
    policy, _ = _policy_and_contract()
    with pytest.raises(RuntimeError, match="mode"):
        policy._load_kinematic_button_window()
    policy._onnx_metadata = {}
    with pytest.raises(RuntimeError, match="require a digest-bound"):
        policy._load_contact_aware_button_window(Path("unused.onnx"))


def test_peak_precomputed_policy_does_not_require_contact_sidecar():
    cfg = dict(PEAK_CONFIG, contact_aware_sparse_root_command_mode="precomputed_turn_then_forward")
    metadata = {"experiment_config": {
        "algo": {"config": {"module_dict": {"actor": {"input_dim": ["actor"]}}}},
        "observation": {"groups": {"actor": {"terms": {
            "drop_button": {}, "pickup_button": {}, "sparse_target_root_trajectory_command_contact_aware": {},
        }}}},
        "command": {"setup_terms": {"motion_command": {"params": {"motion_config": cfg}}}},
    }}
    assert not policy_requires_contact_window(metadata)
    cfg["use_adaptive_timesteps_sampler"] = True
    assert policy_requires_contact_window(metadata)


@pytest.mark.parametrize("key,value", [
    ("contact_aware_peak_height_alpha", float("nan")),
    ("contact_aware_peak_height_alpha", True),
    ("contact_aware_peak_height_smoothing_steps", 0),
    ("contact_aware_peak_height_smoothing_steps", 4097),
])
def test_peak_parameters_fail_closed(key, value):
    with pytest.raises(ValueError):
        height_button_window_from_motion_np(np.ones(10), np.zeros(10), dict(PEAK_CONFIG, **{key: value}))
