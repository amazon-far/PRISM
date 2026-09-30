from __future__ import annotations

from types import SimpleNamespace

import pytest
from holosoma.managers.command.terms.wbt import MotionCommand


ROOT_TERM = (
    "holosoma.managers.observation.terms.wbt:"
    "sparse_target_root_trajectory_command_contact_aware"
)
DROP_TERM = "holosoma.managers.observation.terms.wbt:drop_button"


def _command(*, root_mode: str, button_mode: str) -> MotionCommand:
    command = object.__new__(MotionCommand)
    command.motion_cfg = SimpleNamespace(
        contact_aware_sparse_root_command_mode=root_mode,
        contact_aware_button_window_mode=button_mode,
    )
    terms = {
        "root": SimpleNamespace(func=ROOT_TERM),
        "drop": SimpleNamespace(func=DROP_TERM),
    }
    command._env = SimpleNamespace(
        observation_manager=SimpleNamespace(
            cfg=SimpleNamespace(groups={"actor": SimpleNamespace(terms=terms)})
        )
    )
    return command


@pytest.mark.parametrize("button_mode", ["kinematic_lift", "peak_height"])
def test_precomputed_command_and_height_buttons_do_not_consume_contact_windows(button_mode) -> None:
    command = _command(
        root_mode="precomputed_turn_then_forward",
        button_mode=button_mode,
    )
    assert command._has_contact_window_observation_consumer() is False


def test_tracking_command_still_requires_complete_contact_window_coverage() -> None:
    command = _command(root_mode="tracking_error", button_mode="kinematic_lift")
    assert command._has_contact_window_observation_consumer() is True


def test_contact_interval_button_still_requires_complete_contact_window_coverage() -> None:
    command = _command(
        root_mode="precomputed_turn_then_forward",
        button_mode="contact_interval",
    )
    assert command._has_contact_window_observation_consumer() is True
