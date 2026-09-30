#!/usr/bin/env python3
"""Validate the released box actor and assets before student initialization."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, choices=(8, 32), default=32)
    parser.add_argument("--allow-distillation", action="store_true")
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cli = args.train_args
    if cli and cli[0] == "--":
        cli = cli[1:]

    import torch
    import tyro
    from holosoma.agents.modules.module_utils import setup_ppo_actor_module
    from holosoma.config_types.algo import LayerConfig, ModuleConfig
    from holosoma.config_values.experiment import AnnotatedExperimentConfig
    from holosoma.observation.config_utils import apply_observation_overrides
    from holosoma.perception.config_utils import apply_perception_overrides
    from holosoma.utils.checkpoint_validation import load_verified_torch_checkpoint
    from holosoma.utils.inference_helpers import export_policy_as_onnx, validate_exported_policy_onnx
    from holosoma.utils.policy_init_preflight import validate_policy_init_checkpoint
    from holosoma.utils.runtime_asset_manifest import finalize_runtime_asset_provenance
    from holosoma.utils.tyro_utils import TYRO_CONIFG

    torch.set_num_threads(2)
    config = tyro.cli(AnnotatedExperimentConfig, args=cli, config=TYRO_CONIFG)
    if config.training.num_envs != args.world_size * 2048:
        raise ValueError(f"Box23K profile requires exactly {args.world_size} ranks x 2048 environments.")
    config = dataclasses.replace(config, training=dataclasses.replace(config.training, num_envs=2048))
    config = apply_perception_overrides(apply_observation_overrides(config))
    if not config.training.export_onnx:
        raise ValueError("Box23K profile requires ONNX export.")
    if config.algo.config.distill.enabled and not args.allow_distillation:
        raise ValueError("Box23K distillation preflight requires explicit --allow-distillation.")
    if config.training.checkpoint is not None or config.training.stage4_init_checkpoint is not None:
        raise ValueError("Only actor initialization is permitted, not resume or actor-critic initialization.")
    # Match train_agent's pre-simulator asset closure, not the launch-time
    # pending sentinel. The parent worker retains its own unmodified environment.
    provenance = finalize_runtime_asset_provenance(config)
    if provenance is None:
        raise ValueError("Box initialization requires authenticated launch provenance.")
    checkpoint_path = Path(config.training.policy_init_checkpoint)
    validate_policy_init_checkpoint(checkpoint_path, config.to_serializable_dict(), current_provenance=provenance)
    release_manifest = json.loads((ROOT / "_ckpts" / "manifest.json").read_text())
    initializers = [entry for entry in release_manifest["checkpoints"]
                    if entry["role"] == "box_actor_initializer"]
    if len(initializers) != 1:
        raise ValueError("Release manifest must identify exactly one box initializer.")
    checkpoint, checkpoint_sha = load_verified_torch_checkpoint(
        checkpoint_path, expected_sha256=initializers[0]["sha256"],
    )

    binding = json.loads((ROOT / "scripts" / "box23k_robot_assets.json").read_text())
    asset_root = Path(config.robot.asset.asset_root).expanduser().resolve()
    manifest_bytes = (asset_root / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != binding["manifest_sha256"]:
        raise ValueError("Robot asset manifest SHA mismatch.")
    asset_records = {record["path"]: record for record in json.loads(manifest_bytes)["files"]}
    for record in asset_records.values():
        path = asset_root / record["path"]
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing regular robot asset: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Robot asset changed: {path}")
    depth_meshes = {}
    for link, name in config.perception.camera_mesh_file_map.items():
        relative = "g1/meshes/" + name
        if relative not in asset_records or not (asset_root / relative).is_file():
            raise ValueError(f"Missing exact camera mesh; remapping is forbidden: {name}")
        depth_meshes[link] = {"file": name, "sha256": asset_records[relative]["sha256"]}

    actor_cfg = config.algo.config.module_dict.actor
    dims = dict(zip(actor_cfg.input_dim, [3, 1, 90], strict=True))
    dims["perception_obs"] = 5046
    history = {key: 1 for key in dims}
    actor = setup_ppo_actor_module(dims, actor_cfg, 29, 0.01, "cpu", history)
    actor.load_state_dict(checkpoint["actor_model_state_dict"], strict=True)
    actor.eval()
    old = dict(checkpoint["experiment_config"]["algo"]["config"]["module_dict"]["actor"])
    old["layer_config"] = LayerConfig(**old["layer_config"])
    original = setup_ppo_actor_module(dims, ModuleConfig(**old), 29, 0.01, "cpu", history)
    original.load_state_dict(checkpoint["actor_model_state_dict"], strict=True)
    original.eval()
    if not all(p.requires_grad for p in actor.actor_module.perception_encoder.parameters()):
        raise ValueError("The native depth CNN must remain trainable.")
    generator = torch.Generator().manual_seed(42)
    example = {
        "actor_obs": torch.randn(14, 94, generator=generator),
        "perception_obs": torch.rand(14, 5046, generator=generator) - 0.5,
    }
    with torch.inference_mode():
        source_output = original.act_inference(example)
        target_output = actor.act_inference(example)
    torch.testing.assert_close(source_output, target_output, rtol=0, atol=0)

    class Wrapper(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.actor = policy

        def forward(self, actor_obs, perception_obs):
            return self.actor.act_inference({"actor_obs": actor_obs, "perception_obs": perception_obs})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx_path = args.output.with_suffix(".onnx")
    wrapper = Wrapper(actor).eval()
    export_policy_as_onnx(wrapper, str(onnx_path), example, perception_input_name="perception_obs")
    parity = validate_exported_policy_onnx(
        wrapper=wrapper, onnx_file_path=str(onnx_path), example_obs_dict=example,
        perception_input_name="perception_obs",
    )
    report = {
        "accepted": True, "scope": "cpu_initializer_and_exact_robot_assets_not_live_simulator_acceptance",
        "checkpoint_sha256": checkpoint_sha, "source_vs_target_actor_max_abs_error": 0.0,
        "depth_encoder_trainable": True, "onnx_validation": parity,
        "robot_asset_binding": binding, "per_rank_config": config.to_serializable_dict(),
        "effective_depth_meshes": depth_meshes,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"[INFO] box23k_initialization_preflight_ok checkpoint={checkpoint_sha} report={args.output}")


if __name__ == "__main__":
    main()
