"""Validate the public model artifacts without data, a simulator, or network access."""

import copy
import json
from pathlib import Path

import pytest
import torch

from holosoma.agents.modules.module_utils import setup_ppo_actor_module
from holosoma.agents.ppo.ppo import PPO
from holosoma.utils.checkpoint_validation import load_verified_torch_checkpoint, validate_finite_tree
from holosoma.utils.eval_utils import CheckpointConfig, load_saved_experiment_config
from holosoma.utils.policy_init_preflight import validate_policy_init_payload_identity

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT / "_ckpts/manifest.json").read_text())
ALLOWED_FIELDS = {
    "actor_model_state_dict", "critic_model_state_dict", "actor_obs_normalizer_state",
    "critic_obs_normalizer_state", "experiment_config", "iter", "next_iter", "iteration",
    "motion_transition_contract", "motion_transition_contract_sha256",
    "precomputed_turn_then_forward_deployment_contract",
    "precomputed_turn_then_forward_deployment_contract_sha256",
    "actor_perception_training_geometry_support", "env_state_by_rank", "release_format",
}


def string_values(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from string_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from string_values(item)
    elif isinstance(value, str):
        yield value


@pytest.mark.parametrize("entry", MANIFEST["checkpoints"], ids=lambda entry: entry["role"])
def test_release_checkpoint_privacy_and_loadability(entry):
    torch.set_num_threads(2)
    assert set(entry) == {"file", "role", "checkpoint_step", "sha256", "size_bytes"}
    path = ROOT / "_ckpts" / entry["file"]
    checkpoint, _ = load_verified_torch_checkpoint(path, expected_sha256=entry["sha256"])
    assert path.stat().st_size == entry["size_bytes"]
    assert set(checkpoint) <= ALLOWED_FIELDS
    assert checkpoint["release_format"]["supports_exact_training_resume"] is False
    for value in string_values(checkpoint):
        assert not value.startswith(("/", "~", "wandb://", "http://", "https://"))
    config, origin = load_saved_experiment_config(CheckpointConfig(checkpoint=str(path)))
    assert origin is None
    assert config.logger.type == "disabled"
    assert config.training.project == "prism"
    assert config.training.name == entry["role"]
    assert config.training.checkpoint is None
    assert config.training.policy_init_checkpoint is None
    runtime = copy.deepcopy(checkpoint["experiment_config"])
    validate_policy_init_payload_identity(checkpoint, runtime)
    validate_finite_tree(checkpoint, path="release")

    teacher = entry["role"] == "teacher"
    dims = {"actor_obs": 178} if teacher else {
        "actor_obs_root_contact_aware": 3, "actor_obs_drop_button": 1,
        "actor_obs_proprio_with_actions_no_linvel": 90, "perception_obs": 5046,
    }
    actor = setup_ppo_actor_module(
        dims, config.algo.config.module_dict.actor, 29, 0.01, "cpu", {key: 1 for key in dims}
    ).eval()
    actor.load_state_dict(checkpoint["actor_model_state_dict"], strict=True)
    ppo = object.__new__(PPO)
    ppo.actor = actor
    ppo._validate_checkpoint_actor_std(checkpoint["actor_model_state_dict"], path="actor")
    example = {"actor_obs": torch.zeros(4, 178 if teacher else 94)}
    if not teacher:
        example["perception_obs"] = torch.zeros(4, 5046)
    with torch.inference_mode():
        actions = actor.act_inference(example)
    assert actions.shape == (4, 29) and torch.isfinite(actions).all()

    # The student keeps static geometry evidence for the strict evaluation loader.
    if "actor_perception_training_geometry_support" in checkpoint:
        for env in checkpoint["env_state_by_rank"].values():
            assert set(env) == {"perception_managers"}
            for state in env["perception_managers"]["states"].values():
                assert set(state) == {"version", "semantics"}
                assert set(state["semantics"]) == {
                    "num_envs", "camera_source", "far_tracking_geometry", "far_tracking_topology",
                }
        ppo.actor_perception_key = "perception_obs"
        derived = ppo._aggregate_actor_perception_geometry_support(
            checkpoint["env_state_by_rank"], allow_legacy_missing=False
        )
        assert derived == checkpoint["actor_perception_training_geometry_support"]
