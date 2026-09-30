from __future__ import annotations

from types import SimpleNamespace

import torch

import holosoma.managers.reward.terms.wbt as reward_wbt


class _ObjectContactSimulator:
    def __init__(self) -> None:
        self.body_names = ["pelvis", "left_hand", "right_hand"]
        self.contact_history = torch.tensor(
            [[[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]],
            dtype=torch.float32,
        )
        self.requested_body_names: list[list[str]] = []

    def get_object_contact_force_history(self, body_names: list[str]) -> torch.Tensor:
        self.requested_body_names.append(body_names)
        indexes = torch.tensor([self.body_names.index(name) for name in body_names], dtype=torch.long)
        return self.contact_history.index_select(2, indexes)


def test_object_contact_helper_caches_host_names_and_device_indexes_without_cpu_sync(
    monkeypatch,
) -> None:
    simulator = _ObjectContactSimulator()
    env = SimpleNamespace(
        device="cpu",
        simulator=simulator,
        command_manager=SimpleNamespace(get_state=lambda _name: None),
    )

    def forbidden_cpu_transfer(_tensor, *args, **kwargs):
        raise AssertionError("object-contact reward hot path must not transfer indexes to the CPU")

    monkeypatch.setattr(torch.Tensor, "cpu", forbidden_cpu_transfer)

    first = reward_wbt._get_object_contact_force_history(
        env,
        body_names=["right_hand", "left_hand"],
    )
    second = reward_wbt._get_object_contact_force_history(
        env,
        body_names=["right_hand", "left_hand"],
    )

    expected = torch.tensor([[[[3.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]], dtype=torch.float32)
    assert torch.equal(first, expected)
    assert torch.equal(second, expected)
    assert simulator.requested_body_names == [
        ["right_hand", "left_hand"],
        ["right_hand", "left_hand"],
    ]
    assert simulator.requested_body_names[0] is not simulator.requested_body_names[1]

    cached_names, cached_indexes = env._wbt_reward_sim_body_subset_cache[
        (("right_hand", "left_hand"), None)
    ]
    assert cached_names == ("right_hand", "left_hand")
    assert cached_indexes.tolist() == [2, 1]
    assert cached_indexes.device == torch.device(env.device)
