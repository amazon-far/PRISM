from types import SimpleNamespace

import pytest
import torch

from holosoma.managers.reward.terms.wbt import penalty_action_rate


def _env(actions: torch.Tensor, previous_actions: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        action_manager=SimpleNamespace(
            action=actions,
            prev_action=previous_actions,
        )
    )


def test_action_rate_penalty_is_unchanged_without_optional_cap():
    env = _env(torch.tensor([[3.0, 4.0]]), torch.zeros(1, 2))

    assert penalty_action_rate(env).item() == pytest.approx(25.0)


def test_action_rate_penalty_optional_cap_limits_only_large_values():
    env = _env(
        torch.tensor([[3.0, 4.0], [1.0, 2.0]]),
        torch.zeros(2, 2),
    )

    result = penalty_action_rate(env, max_penalty=10.0)

    assert torch.allclose(result, torch.tensor([10.0, 5.0]))


def test_action_rate_penalty_nonpositive_cap_keeps_legacy_behavior():
    env = _env(torch.ones(1, 2), torch.zeros(1, 2))

    assert penalty_action_rate(env, max_penalty=-1.0).item() == pytest.approx(2.0)
