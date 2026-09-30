from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from holosoma.managers.command.terms.wbt import MotionCommand


def _bare_motion_command() -> MotionCommand:
    command = MotionCommand.__new__(MotionCommand)
    command.num_envs = 1
    command.device = torch.device("cpu")
    command.motion = SimpleNamespace(has_object=True)
    command.manual_control_enabled = False
    command.manual_xy_rel = torch.full((1, 2), 9.0)
    command.manual_yaw_rel = torch.full((1, 1), 9.0)
    command.manual_drop_button_override_enabled = False
    command.manual_drop_button = torch.full((1, 1), 9.0)
    command._simulator_object_state_snapshot = torch.zeros((1, 13))
    command._simulator_object_state_snapshot[0, 2] = 0.1
    command._simulator_object_state_snapshot_ready = True
    robot_root_states = torch.zeros((1, 13), dtype=torch.float32)
    robot_root_states[0, 6] = 1.0
    command._env = SimpleNamespace(
        episode_length_buf=torch.tensor([20], dtype=torch.long),
        simulator=SimpleNamespace(robot_root_states=robot_root_states),
    )
    return command


def _set_robot_xy_yaw(command: MotionCommand, *, x: float, y: float, yaw: float) -> None:
    root_states = command._env.simulator.robot_root_states
    root_states[0, 0] = x
    root_states[0, 1] = y
    root_states[0, 3:7] = torch.tensor(
        [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)],
        dtype=torch.float32,
    )


def test_manual_forward_after_lift_holds_zero_then_replaces_command() -> None:
    command = _bare_motion_command()
    command.configure_manual_forward_after_lift(
        command_m=0.15,
        rel_z_delta_m=0.3,
        consecutive_steps=3,
    )

    assert command.manual_control_enabled is True
    assert command.manual_drop_button_override_enabled is True
    torch.testing.assert_close(command.manual_xy_rel, torch.zeros((1, 2)))
    torch.testing.assert_close(command.manual_yaw_rel, torch.zeros((1, 1)))
    torch.testing.assert_close(command.manual_drop_button, torch.zeros((1, 1)))

    command._simulator_object_state_snapshot[0, 2] = 0.399
    command._update_manual_forward_after_lift()
    assert command.get_manual_forward_after_lift_status()["consecutive_count"] == 0

    command._simulator_object_state_snapshot[0, 2] = 0.401
    for expected_count in (1, 2):
        command._update_manual_forward_after_lift()
        status = command.get_manual_forward_after_lift_status()
        assert status["phase"] == "pickup_zero"
        assert status["consecutive_count"] == expected_count
        assert status["active_forward_command_m"] == 0.0

    command._env.episode_length_buf[0] = 23
    command._update_manual_forward_after_lift()
    status = command.get_manual_forward_after_lift_status()
    assert status["phase"] == "forward"
    assert status["triggered"] is True
    assert status["trigger_episode_step"] == 23
    assert status["active_forward_command_m"] == pytest.approx(0.15)
    torch.testing.assert_close(command.manual_xy_rel, torch.tensor([[0.15, 0.0]]))
    torch.testing.assert_close(command.manual_yaw_rel, torch.zeros((1, 1)))
    assert status["command_semantics"] == "legacy_constant_robot_heading_frame"
    assert status["heading_lock"] is None


def test_manual_forward_after_lift_zero_steps_triggers_on_first_threshold_crossing() -> None:
    command = _bare_motion_command()
    command.configure_manual_forward_after_lift(
        command_m=0.15,
        rel_z_delta_m=0.3,
        consecutive_steps=0,
    )

    command._simulator_object_state_snapshot[0, 2] = 0.399
    command._update_manual_forward_after_lift()
    status = command.get_manual_forward_after_lift_status()
    assert status["phase"] == "pickup_zero"
    assert status["triggered"] is False
    assert status["consecutive_count"] == 0
    torch.testing.assert_close(command.manual_xy_rel, torch.zeros((1, 2)))

    command._simulator_object_state_snapshot[0, 2] = 0.401
    command._env.episode_length_buf[0] = 21
    command._update_manual_forward_after_lift()
    status = command.get_manual_forward_after_lift_status()
    assert status["phase"] == "forward"
    assert status["triggered"] is True
    assert status["trigger_episode_step"] == 21
    assert status["required_consecutive_steps"] == 0
    assert status["consecutive_count"] == 1
    torch.testing.assert_close(command.manual_xy_rel, torch.tensor([[0.15, 0.0]]))
    torch.testing.assert_close(command.manual_yaw_rel, torch.zeros((1, 1)))


def test_manual_forward_after_lift_can_preserve_native_contact_buttons() -> None:
    command = _bare_motion_command()
    command.manual_pickup_button_override_enabled = True
    command.manual_pickup_button = torch.ones((1, 1))
    command.configure_manual_forward_after_lift(
        command_m=0.15,
        rel_z_delta_m=0.3,
        consecutive_steps=0,
        preserve_native_contact_buttons=True,
    )

    assert command.manual_control_enabled is True
    assert command.manual_pickup_button_override_enabled is False
    assert command.manual_drop_button_override_enabled is False
    assert command.get_manual_forward_after_lift_status()[
        "preserve_native_contact_buttons"
    ] is True
    assert command.get_manual_forward_after_lift_status()[
        "preserve_native_pickup_button"
    ] is True
    assert command.get_manual_forward_after_lift_status()[
        "preserve_native_drop_button"
    ] is True


def test_manual_forward_after_lift_can_preserve_only_native_pickup() -> None:
    command = _bare_motion_command()
    command.manual_pickup_button_override_enabled = True
    command.manual_pickup_button = torch.ones((1, 1))
    command.configure_manual_forward_after_lift(
        command_m=0.15,
        rel_z_delta_m=0.3,
        consecutive_steps=0,
        preserve_native_pickup_button=True,
    )

    status = command.get_manual_forward_after_lift_status()
    assert command.manual_pickup_button_override_enabled is False
    assert command.manual_drop_button_override_enabled is True
    torch.testing.assert_close(command.manual_drop_button, torch.zeros((1, 1)))
    assert status["preserve_native_contact_buttons"] is False
    assert status["preserve_native_pickup_button"] is True
    assert status["preserve_native_drop_button"] is False


def test_manual_forward_after_lift_restores_latched_world_heading() -> None:
    command = _bare_motion_command()
    command.configure_manual_forward_after_lift(
        command_m=0.15,
        rel_z_delta_m=0.3,
        consecutive_steps=1,
        heading_lock=True,
    )
    command._simulator_object_state_snapshot[0, 2] = 0.401
    command._update_manual_forward_after_lift()
    command._update_manual_forward_heading_lock()

    _set_robot_xy_yaw(command, x=0.2, y=0.04, yaw=math.pi / 6.0)
    command._update_manual_forward_heading_lock()

    expected = torch.tensor(
        [[0.15 * math.cos(-math.pi / 6.0), 0.15 * math.sin(-math.pi / 6.0)]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(command.manual_xy_rel, expected)
    torch.testing.assert_close(command.manual_yaw_rel, torch.tensor([[-math.pi / 6.0]]))
    status = command.get_manual_forward_after_lift_status()
    assert status["heading_lock"]["heading_error_rad"] == pytest.approx(-math.pi / 6.0)
    assert status["heading_lock"]["along_track_displacement_m"] == pytest.approx(0.2)
    assert status["heading_lock"]["cross_track_error_m"] == pytest.approx(0.04)


def test_immediate_manual_forward_uses_same_policy_interface_and_heading_lock() -> None:
    command = _bare_motion_command()
    _set_robot_xy_yaw(command, x=1.0, y=-2.0, yaw=-math.pi / 4.0)
    command.configure_manual_heading_locked_forward(command_m=0.1)

    assert command.manual_control_enabled is True
    torch.testing.assert_close(command.manual_xy_rel, torch.tensor([[0.1, 0.0]]))
    torch.testing.assert_close(command.manual_yaw_rel, torch.zeros((1, 1)))

    _set_robot_xy_yaw(command, x=1.0, y=-2.0, yaw=-math.pi / 4.0 + 0.2)
    command._update_manual_forward_heading_lock()
    torch.testing.assert_close(
        command.manual_xy_rel,
        torch.tensor([[0.1 * math.cos(-0.2), 0.1 * math.sin(-0.2)]]),
    )
    torch.testing.assert_close(command.manual_yaw_rel, torch.tensor([[-0.2]]))


def test_manual_forward_after_lift_can_reproduce_legacy_body_frame_command_for_ab() -> None:
    command = _bare_motion_command()
    command.configure_manual_forward_after_lift(
        command_m=0.15,
        rel_z_delta_m=0.3,
        consecutive_steps=1,
        heading_lock=False,
    )
    command._simulator_object_state_snapshot[0, 2] = 0.401
    command._update_manual_forward_after_lift()
    _set_robot_xy_yaw(command, x=0.2, y=0.04, yaw=math.pi / 2.0)
    command._update_manual_forward_heading_lock()

    torch.testing.assert_close(command.manual_xy_rel, torch.tensor([[0.15, 0.0]]))
    torch.testing.assert_close(command.manual_yaw_rel, torch.zeros((1, 1)))
    status = command.get_manual_forward_after_lift_status()
    assert status["command_semantics"] == "legacy_constant_robot_heading_frame"
    assert status["heading_lock"] is None


def test_manual_forward_after_lift_records_explicit_actor_command_semantics() -> None:
    command = _bare_motion_command()
    command.configure_manual_forward_after_lift(
        command_m=0.05,
        rel_z_delta_m=0.3,
        consecutive_steps=0,
        command_semantics="world_velocity_mps",
    )
    command._simulator_object_state_snapshot[0, 2] = 0.401
    command._update_manual_forward_after_lift()

    status = command.get_manual_forward_after_lift_status()
    assert status["command_semantics"] == "world_velocity_mps"
    assert status["active_forward_command_m"] == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"command_m": float("nan"), "rel_z_delta_m": 0.3, "consecutive_steps": 3}, "command_m"),
        ({"command_m": 0.1, "rel_z_delta_m": 0.0, "consecutive_steps": 3}, "rel_z_delta_m"),
        ({"command_m": 0.1, "rel_z_delta_m": 0.3, "consecutive_steps": -1}, "consecutive_steps"),
        ({"command_m": 0.1, "rel_z_delta_m": 0.3, "consecutive_steps": 1.5}, "consecutive_steps"),
        (
            {
                "command_m": 0.1,
                "rel_z_delta_m": 0.3,
                "consecutive_steps": 1,
                "heading_lock": 1,
            },
            "heading_lock",
        ),
        (
            {
                "command_m": 0.1,
                "rel_z_delta_m": 0.3,
                "consecutive_steps": 1,
                "preserve_native_contact_buttons": 1,
            },
            "preserve_native_contact_buttons",
        ),
        (
            {
                "command_m": 0.1,
                "rel_z_delta_m": 0.3,
                "consecutive_steps": 1,
                "preserve_native_pickup_button": 1,
            },
            "preserve_native_pickup_button",
        ),
        (
            {
                "command_m": 0.1,
                "rel_z_delta_m": 0.3,
                "consecutive_steps": 1,
                "preserve_native_drop_button": 1,
            },
            "preserve_native_drop_button",
        ),
        (
            {
                "command_m": 0.1,
                "rel_z_delta_m": 0.3,
                "consecutive_steps": 1,
                "command_semantics": "unknown",
            },
            "command_semantics",
        ),
        (
            {
                "command_m": 0.1,
                "rel_z_delta_m": 0.3,
                "consecutive_steps": 1,
                "heading_lock": True,
                "command_semantics": "world_velocity_mps",
            },
            "heading_lock",
        ),
    ],
)
def test_manual_forward_after_lift_rejects_invalid_contract(kwargs, error: str) -> None:
    command = _bare_motion_command()
    with pytest.raises(ValueError, match=error):
        command.configure_manual_forward_after_lift(**kwargs)
