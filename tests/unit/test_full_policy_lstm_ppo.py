from __future__ import annotations

from types import SimpleNamespace

import torch
from torch.distributions import Normal

from holosoma.agents.modules.ppo_modules import PPOActor, PPOCritic
from holosoma.agents.ppo.ppo import PPO
from holosoma.config_types.algo import LayerConfig, ModuleConfig


def _module(output_dim: int) -> ModuleConfig:
    return ModuleConfig(
        type="LSTM",
        input_dim=["obs"],
        output_dim=[output_dim],
        layer_config=LayerConfig(
            hidden_dims=[16, 8],
            lstm_hidden_dim=12,
            lstm_num_layers=1,
        ),
    )


def test_full_policy_lstm_ppo_sequence_loss_backpropagates() -> None:
    torch.manual_seed(11)
    ppo = object.__new__(PPO)
    ppo.actor = PPOActor(
        {"obs": 5},
        _module(3),
        num_actions=3,
        init_noise_std=0.5,
        history_length={"obs": 1},
    )
    ppo.critic = PPOCritic(
        {"obs": 7},
        _module(1),
        history_length={"obs": 1},
    )
    ppo.config = SimpleNamespace(clip_param=0.2, value_loss_coef=1.0)
    ppo.device = "cpu"
    ppo.use_symmetry = False
    ppo.actor_perception_key = ""
    ppo.critic_perception_key = ""
    ppo.distill_enabled = False
    ppo.distill_mode = "mse"
    ppo.dagger_enabled = False
    ppo.ppo_coeff = 1.0
    ppo.is_multi_gpu = False
    ppo._operational_entropy_coefficient = lambda: 0.005

    time_steps, batch_size = 5, 4
    actor_obs = torch.randn(time_steps, batch_size, 5)
    critic_obs = torch.randn(time_steps, batch_size, 7)
    dones = torch.zeros(time_steps, batch_size, 1, dtype=torch.bool)
    dones[1, 0] = True
    dones[3, 2] = True
    actor_hidden = torch.zeros(time_steps, batch_size, 1, 12)
    actor_cell = torch.zeros_like(actor_hidden)
    critic_hidden = torch.zeros_like(actor_hidden)
    critic_cell = torch.zeros_like(actor_hidden)

    with torch.no_grad():
        mean, _, _ = ppo.actor.actor_module.forward_recurrent_sequence(
            actor_obs,
            dones=dones,
            initial_hidden=actor_hidden[0].permute(1, 0, 2),
            initial_cell=actor_cell[0].permute(1, 0, 2),
        )
    sigma = torch.full_like(mean, 0.5)
    actions = mean + 0.05
    old_log_prob = Normal(mean, sigma).log_prob(actions).sum(-1, keepdim=True)
    minibatch = {
        "actor_obs": actor_obs,
        "critic_obs": critic_obs,
        "actions": actions,
        "values": torch.zeros(time_steps, batch_size, 1),
        "advantages": torch.randn(time_steps, batch_size, 1),
        "returns": torch.randn(time_steps, batch_size, 1),
        "actions_log_prob": old_log_prob,
        "action_mean": mean,
        "action_sigma": sigma,
        "dones": dones,
        "actor_lstm_hidden": actor_hidden,
        "actor_lstm_cell": actor_cell,
        "critic_lstm_hidden": critic_hidden,
        "critic_lstm_cell": critic_cell,
    }

    losses = ppo._compute_ppo_loss_lstm(minibatch)
    total_loss = losses["actor_loss"] + losses["critic_loss"]
    assert torch.isfinite(total_loss)
    total_loss.backward()
    assert ppo.actor.actor_module.module.lstm.weight_ih_l0.grad is not None
    assert ppo.critic.critic_module.module.lstm.weight_ih_l0.grad is not None
    assert torch.isfinite(ppo.actor.actor_module.module.lstm.weight_ih_l0.grad).all()
    assert torch.isfinite(ppo.critic.critic_module.module.lstm.weight_ih_l0.grad).all()
