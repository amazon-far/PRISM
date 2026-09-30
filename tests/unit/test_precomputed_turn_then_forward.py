from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from holosoma.managers.command.terms.wbt import MotionCommand, MotionLoader
from holosoma.managers.observation.terms.wbt import (
    _evaluation_precomputed_dx_only_command,
    _zero_root_command_during_drop,
    sparse_target_root_trajectory_command_contact_aware,
)


def _valid_arrays() -> tuple[np.ndarray, np.ndarray]:
    command = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.15, 0.0, 0.0],
            [0.0, 0.0, 0.6],
        ],
        dtype=np.float32,
    )
    phase = np.asarray([0, 1, 2], dtype=np.uint8)
    return command, phase


def test_loader_accepts_exact_decoupled_phase_contract() -> None:
    command, phase = _valid_arrays()

    loaded = MotionLoader._extract_precomputed_root_command_np(
        {
            "policy_command_xy_yaw": command,
            "policy_command_phase": phase,
        },
        3,
        source="fixture.npz",
    )

    assert loaded is not None
    np.testing.assert_array_equal(loaded[0], command)
    np.testing.assert_array_equal(loaded[1], phase)


@pytest.mark.parametrize(
    ("command", "phase", "error"),
    [
        (
            np.asarray([[0.1, 0.0, 0.2]], dtype=np.float32),
            np.asarray([1], dtype=np.uint8),
            "couples dx and dyaw",
        ),
        (
            np.asarray([[0.0, 0.1, 0.0]], dtype=np.float32),
            np.asarray([0], dtype=np.uint8),
            "dy exactly zero",
        ),
        (
            np.asarray([[0.1, 0.0, 0.0]], dtype=np.float32),
            np.asarray([2], dtype=np.uint8),
            "Yaw-phase",
        ),
    ],
)
def test_loader_rejects_nonexclusive_command_contract(
    command: np.ndarray,
    phase: np.ndarray,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        MotionLoader._extract_precomputed_root_command_np(
            {
                "policy_command_xy_yaw": command,
                "policy_command_phase": phase,
            },
            1,
            source="fixture.npz",
        )


def test_loader_rejects_partial_command_fields() -> None:
    command, _ = _valid_arrays()
    with pytest.raises(ValueError, match="must contain both"):
        MotionLoader._extract_precomputed_root_command_np(
            {"policy_command_xy_yaw": command},
            3,
            source="fixture.npz",
        )


def test_runtime_gathers_motion_time_and_applies_pickup_gate() -> None:
    command = MotionCommand.__new__(MotionCommand)
    command.motion_cfg = SimpleNamespace(
        contact_aware_sparse_root_command_mode="precomputed_turn_then_forward"
    )
    command.motion = SimpleNamespace(
        precomputed_root_command=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.15, 0.0, 0.0],
                [0.0, 0.0, -0.7],
            ],
            dtype=torch.float32,
        ),
        precomputed_root_command_phase=torch.tensor([0, 1, 2], dtype=torch.uint8),
    )
    command.time_steps = torch.tensor([2, 1], dtype=torch.long)
    command.pickup_anchor_set = torch.tensor([True, False])
    command._get_motion_indices = lambda time_steps: time_steps

    actual = command.get_precomputed_turn_then_forward_command()
    actual_phase = command.get_precomputed_turn_then_forward_phase()

    torch.testing.assert_close(
        actual,
        torch.tensor([[0.0, 0.0, -0.7], [0.0, 0.0, 0.0]], dtype=torch.float32),
    )
    torch.testing.assert_close(actual_phase, torch.tensor([2, 0], dtype=torch.uint8))


def test_evaluation_dx_only_override_removes_every_nonforward_component() -> None:
    motion_command = SimpleNamespace(
        _evaluation_precomputed_dx_only_after_pickup=True,
        precomputed_turn_then_forward_enabled=lambda: True,
    )
    command = torch.tensor(
        [
            [0.15, 0.0, 0.0],
            [0.0, 0.0, -0.7],
            [0.08, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    actual = _evaluation_precomputed_dx_only_command(motion_command, command)

    torch.testing.assert_close(
        actual,
        torch.tensor(
            [
                [0.15, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.08, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )


def test_evaluation_dx_only_override_is_default_off() -> None:
    motion_command = SimpleNamespace(
        precomputed_turn_then_forward_enabled=lambda: True,
    )
    command = torch.tensor([[0.0, 0.0, 0.7]], dtype=torch.float32)

    actual = _evaluation_precomputed_dx_only_command(motion_command, command)

    assert actual is command


def _drop_exclusive_training_fixture(*, enabled: bool) -> tuple[MotionCommand, SimpleNamespace]:
    motion_command = MotionCommand.__new__(MotionCommand)
    motion_command.motion_cfg = SimpleNamespace(
        zero_root_command_when_drop_active=enabled,
        hybrid_stage2_enabled=False,
        hybrid_velocity_enabled=False,
    )
    motion_command.motion = SimpleNamespace(has_object=True)
    motion_command.manual_control_enabled = False
    motion_command.manual_drop_button_override_enabled = False
    motion_command.get_contact_aware_drop_button = lambda: torch.tensor(
        [False, True, True]
    )
    env = SimpleNamespace(num_envs=3, device="cpu")
    return motion_command, env


def test_training_drop_exclusivity_zeros_all_three_command_dimensions_per_env() -> None:
    motion_command, env = _drop_exclusive_training_fixture(enabled=True)
    command = torch.tensor(
        [
            [0.15, 0.0, 0.0],
            [0.20, -0.10, 0.30],
            [0.00, 0.00, -0.70],
        ],
        dtype=torch.float32,
    )

    actual = _zero_root_command_during_drop(motion_command, env, command)

    torch.testing.assert_close(
        actual,
        torch.tensor(
            [
                [0.15, 0.0, 0.0],
                [0.00, 0.0, 0.0],
                [0.00, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )


def test_training_drop_exclusivity_disabled_is_identity_and_does_not_read_drop() -> None:
    motion_command, env = _drop_exclusive_training_fixture(enabled=False)
    motion_command.get_contact_aware_drop_button = lambda: (_ for _ in ()).throw(
        AssertionError("legacy-disabled gate must not compute the drop button")
    )
    command = torch.tensor([[0.1, 0.2, 0.3]] * 3, dtype=torch.float32)

    actual = _zero_root_command_during_drop(motion_command, env, command)

    assert actual is command


def test_contact_aware_actor_command_applies_drop_exclusivity_after_precomputed_lookup() -> None:
    motion_command, env = _drop_exclusive_training_fixture(enabled=True)
    motion_command.motion_cfg.contact_aware_sparse_root_command_mode = (
        "precomputed_turn_then_forward"
    )
    motion_command.pure_rl_policy_command_after_lift_enabled = lambda: False
    motion_command.get_precomputed_turn_then_forward_command = lambda: torch.tensor(
        [
            [0.15, 0.0, 0.0],
            [0.15, 0.0, 0.0],
            [0.0, 0.0, -0.7],
        ],
        dtype=torch.float32,
    )

    env.command_manager = SimpleNamespace(
        get_state=lambda name: motion_command if name == "motion_command" else None
    )
    actual = sparse_target_root_trajectory_command_contact_aware(env)

    torch.testing.assert_close(
        actual,
        torch.tensor(
            [
                [0.15, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )
