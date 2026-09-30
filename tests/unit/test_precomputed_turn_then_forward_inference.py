from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from holosoma.managers.command.terms.wbt import (
    _pickup_step_and_threshold_from_rel_z,
)
from holosoma_inference.policies.wbt import (
    MotionData,
    WholeBodyTrackingPolicy,
    _apply_transition_segment_np,
    _pickup_step_and_threshold_from_rel_z_np,
)


def test_inference_pickup_threshold_is_bitwise_equal_to_training() -> None:
    rel_z = np.asarray(
        [0.02, 0.03, 0.04, 0.19, 0.20, 0.21, 0.22, 0.23, 0.05],
        dtype=np.float32,
    )

    inference_step, inference_threshold = _pickup_step_and_threshold_from_rel_z_np(
        rel_z
    )
    training_step, training_threshold = _pickup_step_and_threshold_from_rel_z(
        torch.from_numpy(rel_z.copy())
    )

    assert inference_step == training_step
    assert inference_threshold.tobytes() == (
        training_threshold.detach().cpu().numpy().astype(np.float32).tobytes()
    )


def test_inference_motion_parser_preserves_exact_decoupled_arrays() -> None:
    command = np.asarray(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, -0.4]],
        dtype=np.float32,
    )
    phase = np.asarray([0, 1, 2], dtype=np.uint8)

    loaded = MotionData._extract_precomputed_root_command_np(
        {
            "policy_command_xy_yaw": command,
            "policy_command_phase": phase,
        },
        3,
        source=Path("fixture.npz"),
    )

    assert loaded is not None
    np.testing.assert_array_equal(loaded[0], command)
    np.testing.assert_array_equal(loaded[1], phase)


def test_authenticated_transition_padding_never_invents_actor_command() -> None:
    identity = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    zeros3 = np.zeros(3, dtype=np.float32)
    motion = {
        "joint_pos": np.zeros((1, 2), dtype=np.float32),
        "joint_vel": np.zeros((1, 2), dtype=np.float32),
        "root_pos_w": np.zeros((1, 3), dtype=np.float32),
        "root_quat_w": identity.reshape(1, 4),
        "ref_pos_w": np.zeros((1, 3), dtype=np.float32),
        "ref_quat_w": identity.reshape(1, 4),
        "precomputed_root_command": np.asarray(
            [[0.25, 0.0, 0.0]], dtype=np.float32
        ),
        "precomputed_root_command_phase": np.asarray([1], dtype=np.uint8),
    }
    state = {
        "joint_pos": np.zeros(2, dtype=np.float32),
        "joint_vel": np.zeros(2, dtype=np.float32),
        "root_pos": zeros3,
        "root_quat": identity,
        "ref_pos": zeros3,
        "ref_quat": identity,
    }

    _apply_transition_segment_np(
        motion,
        start_state=state,
        target_state=state,
        num_steps=3,
        prepend=True,
        drop_first=False,
        drop_last=True,
    )

    np.testing.assert_array_equal(
        motion["precomputed_root_command"],
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.25, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        motion["precomputed_root_command_phase"],
        np.asarray([0, 0, 0, 1], dtype=np.uint8),
    )


def _runtime_policy_fixture() -> WholeBodyTrackingPolicy:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy.config = SimpleNamespace(task=SimpleNamespace(sim_object_name="object"))
    policy._precomputed_turn_then_forward_enabled = True
    policy._runtime_pickup_threshold_rel_z = np.float32(0.2)
    policy._runtime_reference_pickup_step = 8
    policy._motion_data = SimpleNamespace(
        has_precomputed_root_command=True,
        precomputed_root_command=np.asarray(
            [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, 0.5]],
            dtype=np.float32,
        ),
    )
    policy._motion_index_for_test = 1
    policy._get_motion_index = lambda: policy._motion_index_for_test
    policy._object_z_for_test = np.float32(0.25)
    policy._get_sim_actor_state = lambda _name: np.asarray(
        [[0.0, 0.0, policy._object_z_for_test]], dtype=np.float32
    )
    policy._apply_external_sparse_root_command = lambda command: command
    policy._last_sparse_manual_enabled = True
    policy._control_tick_sim_state_snapshot = SimpleNamespace(
        sim_time_ms=0.0,
        episode_generation=4,
    )
    policy._reset_runtime_pickup_latch()
    return policy


def test_runtime_pickup_latch_matches_training_five_unique_step_gate() -> None:
    policy = _runtime_policy_fixture()
    robot_state = np.zeros((1, 7), dtype=np.float32)

    for tick in range(4):
        policy._control_tick_sim_state_snapshot.sim_time_ms = float(tick)
        np.testing.assert_array_equal(
            policy._get_precomputed_turn_then_forward_command(robot_state),
            np.zeros((1, 3), dtype=np.float32),
        )

    # Re-reading one pinned simulator state must not count as another step.
    policy._get_precomputed_turn_then_forward_command(robot_state)
    assert policy._runtime_pickup_consecutive_counter == 4

    policy._control_tick_sim_state_snapshot.sim_time_ms = 4.0
    np.testing.assert_array_equal(
        policy._get_precomputed_turn_then_forward_command(robot_state),
        np.asarray([[0.3, 0.0, 0.0]], dtype=np.float32),
    )
    assert policy._runtime_pickup_latched is True

    # The latch is sticky within one episode, even if the object later drops.
    policy._object_z_for_test = np.float32(0.0)
    policy._motion_index_for_test = 2
    policy._control_tick_sim_state_snapshot.sim_time_ms = 5.0
    np.testing.assert_array_equal(
        policy._get_precomputed_turn_then_forward_command(robot_state),
        np.asarray([[0.0, 0.0, 0.5]], dtype=np.float32),
    )


def test_runtime_pickup_latch_resets_on_authenticated_episode_change() -> None:
    policy = _runtime_policy_fixture()
    robot_state = np.zeros((1, 7), dtype=np.float32)
    for tick in range(5):
        policy._control_tick_sim_state_snapshot.sim_time_ms = float(tick)
        policy._get_precomputed_turn_then_forward_command(robot_state)
    assert policy._runtime_pickup_latched is True

    policy._control_tick_sim_state_snapshot.episode_generation = 5
    policy._control_tick_sim_state_snapshot.sim_time_ms = 0.0
    policy._object_z_for_test = np.float32(0.0)

    np.testing.assert_array_equal(
        policy._get_precomputed_turn_then_forward_command(robot_state),
        np.zeros((1, 3), dtype=np.float32),
    )
    assert policy._runtime_pickup_latched is False
    assert policy._runtime_pickup_consecutive_counter == 0


def _drop_exclusive_inference_fixture(*, enabled: bool) -> WholeBodyTrackingPolicy:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._motion_cfg = {"zero_root_command_when_drop_active": enabled}
    policy._uses_sparse_root_command_contact_aware = True
    policy._get_sparse_target_root_trajectory_command = lambda _state: np.asarray(
        [[0.15, -0.2, 0.4]],
        dtype=np.float32,
    )
    policy._get_sparse_target_root_trajectory_command_contact_aware = (
        lambda _state, _base: np.asarray([[0.1, 0.0, -0.7]], dtype=np.float32)
    )
    policy._get_drop_button = lambda: np.asarray([[1.0]], dtype=np.float32)
    policy._get_pickup_button = lambda: np.asarray([[0.0]], dtype=np.float32)
    policy._get_base_lin_vel_obs = lambda _state: np.zeros((1, 3), dtype=np.float32)
    policy._get_base_ang_vel_obs = lambda _state: np.zeros((1, 3), dtype=np.float32)
    policy.num_dofs = 2
    policy.default_dof_angles = np.zeros((1, 2), dtype=np.float32)
    policy.last_policy_action = np.zeros((1, 2), dtype=np.float32)
    return policy


def test_inference_observation_buffer_uses_same_effective_drop_for_button_and_command() -> None:
    policy = _drop_exclusive_inference_fixture(enabled=True)
    robot_state = np.zeros((1, 7 + 2 + 6 + 2), dtype=np.float32)

    obs = policy._get_depth_distill_obs_buffer_dict(robot_state)

    np.testing.assert_array_equal(obs["drop_button"], [[1.0]])
    np.testing.assert_array_equal(
        obs["sparse_target_root_trajectory_command"],
        np.zeros((1, 3), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        obs["sparse_target_root_trajectory_command_contact_aware"],
        np.zeros((1, 3), dtype=np.float32),
    )


def test_inference_drop_exclusivity_disabled_preserves_historical_command() -> None:
    policy = _drop_exclusive_inference_fixture(enabled=False)
    command = np.asarray([[0.15, -0.2, 0.4]], dtype=np.float32)

    actual = policy._apply_drop_exclusive_root_command(
        command,
        np.asarray([[1.0]], dtype=np.float32),
    )

    assert actual is command
