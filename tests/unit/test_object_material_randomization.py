from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from holosoma.managers.randomization.terms.locomotion import (
    _couple_friction_material_buckets,
    _isaacsim_randomize_rigid_body_material,
)


def test_coupled_friction_bucket_distribution_contract() -> None:
    generator = torch.Generator(device="cpu").manual_seed(1234)
    draws = torch.rand((8192, 3), generator=generator)
    independent_buckets = torch.stack(
        (
            0.1 + 0.6 * draws[:, 0],
            0.7 + 0.29 * draws[:, 1],
            draws[:, 2],
        ),
        dim=-1,
    )
    buckets = _couple_friction_material_buckets(
        independent_buckets,
        static_friction_range=(0.1, 0.7),
        dynamic_friction_ratio_range=(0.7, 0.99),
        restitution_range=(0.0, 1.0),
    )

    static = buckets[:, 0]
    dynamic = buckets[:, 1]
    ratio = dynamic / static
    restitution = buckets[:, 2]

    assert buckets.shape == (8192, 3)
    assert torch.all((static >= 0.1) & (static <= 0.7))
    assert torch.all((ratio >= 0.7) & (ratio <= 0.99))
    assert torch.all(dynamic < static)
    assert torch.all((restitution >= 0.0) & (restitution <= 1.0))
    assert static.mean().item() == pytest.approx(0.4, abs=0.01)
    assert ratio.mean().item() == pytest.approx(0.845, abs=0.01)


@pytest.mark.parametrize(
    ("static_range", "ratio_range"),
    (
        ((0.7, 0.1), (0.7, 0.99)),
        ((0.1, 0.7), (0.0, 0.99)),
        ((0.1, 0.7), (0.7, 1.01)),
    ),
)
def test_coupled_friction_bucket_rejects_invalid_ranges(
    static_range: tuple[float, float],
    ratio_range: tuple[float, float],
) -> None:
    buckets = torch.tensor([[0.4, 0.8, 0.5]], dtype=torch.float32)
    with pytest.raises(ValueError):
        _couple_friction_material_buckets(
            buckets,
            static_friction_range=static_range,
            dynamic_friction_ratio_range=ratio_range,
            restitution_range=(0.0, 1.0),
        )


def test_isaaclab_assignment_receives_coupled_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakePhysxView:
        def __init__(self):
            self.materials = torch.zeros((3, 1, 3), dtype=torch.float32)

        def get_material_properties(self):
            return self.materials

    physx_view = FakePhysxView()

    class FakeMaterialRandomizer:
        def __init__(self, cfg, env):
            captured["cfg"] = cfg
            captured["constructor_env"] = env
            params = cfg.params
            ranges = torch.tensor(
                [
                    params["static_friction_range"],
                    params["dynamic_friction_range"],
                    params["restitution_range"],
                ],
                dtype=torch.float32,
            )
            draws = torch.rand((params["num_buckets"], 3), generator=torch.Generator().manual_seed(7))
            self.material_buckets = ranges[:, 0] + (ranges[:, 1] - ranges[:, 0]) * draws

        def __call__(self, env, env_ids, **kwargs):
            captured["call_env"] = env
            captured["env_ids"] = env_ids
            captured["kwargs"] = kwargs
            captured["material_buckets"] = self.material_buckets.clone()
            env.scene["object"].root_physx_view.materials[env_ids, 0] = self.material_buckets[
                : len(env_ids)
            ]

    class FakeEventTermCfg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    isaaclab_module = ModuleType("isaaclab")
    envs_module = ModuleType("isaaclab.envs")
    managers_module = ModuleType("isaaclab.managers")
    envs_module.mdp = SimpleNamespace(randomize_rigid_body_material=FakeMaterialRandomizer)
    managers_module.EventTermCfg = FakeEventTermCfg
    isaaclab_module.envs = envs_module
    isaaclab_module.managers = managers_module
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab_module)
    monkeypatch.setitem(sys.modules, "isaaclab.envs", envs_module)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers_module)

    simulator = SimpleNamespace(
        scene={"object": SimpleNamespace(root_physx_view=physx_view)}
    )
    env_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    _isaacsim_randomize_rigid_body_material(
        simulator,
        env_ids,
        SimpleNamespace(name="object"),
        static_friction_range=(0.1, 0.7),
        dynamic_friction_range=(0.07, 0.693),
        restitution_range=(0.0, 1.0),
        num_buckets=64,
        dynamic_friction_ratio_range=(0.7, 0.99),
    )

    buckets = captured["material_buckets"]
    assert isinstance(buckets, torch.Tensor)
    ratio = buckets[:, 1] / buckets[:, 0]
    assert torch.all((buckets[:, 0] >= 0.1) & (buckets[:, 0] <= 0.7))
    assert torch.all((ratio >= 0.7) & (ratio <= 0.99))
    assert torch.all(buckets[:, 1] < buckets[:, 0])
    cfg = captured["cfg"]
    assert cfg.params["dynamic_friction_range"] == (0.7, 0.99)
    assert captured["call_env"] is simulator
    assert torch.equal(captured["env_ids"], env_ids)
