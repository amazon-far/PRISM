"""Collect checkpoint-native teacher trajectories and mesh contact sidecars."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def prepare(argv=None):
    parser = argparse.ArgumentParser(
        prog="rollout.sh",
        description="Export every clip in a prepared motion bank using the teacher policy. "
                    "Outputs include motion_bank/, clips/ contact sidecars and success/failure summaries.",
    )
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "_ckpts" / "teacher_40000.pt",
                        help="Teacher checkpoint (default: _ckpts/teacher_40000.pt)")
    parser.add_argument("--motion-bank", required=True, type=Path,
                        help="Single-slot motion bank or one prepared rank shard, with its object map")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "rollout",
                        help="New output directory (default: outputs/rollout)")
    parser.add_argument("--gpu", default="0", help="One GPU index or UUID")
    parser.add_argument("--check", action="store_true", help="Validate checkpoint/config on CPU without rollout")
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    bank = args.motion_bank.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")
    if output.exists():
        parser.error(f"Use a new output directory: {output}")
    if output == bank or bank in output.parents or output in bank.parents:
        parser.error("Output and input motion bank must be separate directories")
    if not args.gpu.strip() or "," in args.gpu or args.gpu == "-1":
        parser.error("--gpu must select one GPU")
    object_map = bank / "_clip_object_urdf_map.json"
    if not object_map.is_file():
        parser.error(f"Missing object map: {object_map}")
    clips = json.loads(object_map.read_text())["clips"]
    motion_ids = {path.stem for path in bank.glob("*.npz")}
    if not clips or motion_ids != set(clips):
        parser.error("Motion NPZ files and object-map clips must match exactly")

    # Independent rollout processes all use LOCAL_RANK=0. Give each output its
    # own conversion caches so forced robot USD imports cannot overwrite assets
    # being loaded by another process from the same checkout.
    runtime_cache = output.parent / (output.name + "_runtime")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("HOLOSOMA_", "WANDB_", "TEACHER_ROLLOUT_")) and key not in {
               "WORLD_SIZE", "RANK", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE",
               "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
               "CONTACT_EXPORT_ROOT", "CONTACT_SIDECAR_MODE",
               "PERCEPTION_INJECT_INTO_POLICY_MODULES", "RESET_TO_DEFAULT_POSE",
           }}
    env.update({
        "PYTHONPATH": str(ROOT),
        "PYTHONDONTWRITEBYTECODE": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "" if args.check else args.gpu, "LOCAL_RANK": "0",
        "HOLOSOMA_DEVICE": "cuda:0", "WANDB_MODE": "disabled",
        "HOLOSOMA_EVAL_POLICY": "checkpoint_actor",
        "HOLOSOMA_EVAL_DISABLE_ROLLOUT_REFERENCE_REWARDS": "1",
        "HOLOSOMA_EVAL_ALLOW_MISSING_CAMERA_SENSOR_NOISE_STATE": "1",
        "HOLOSOMA_DISABLE_AUTO_RESET": "1", "HOLOSOMA_DISABLE_CLIP_END_RESET": "1",
        "HOLOSOMA_OBJECT_SPAWN_MODE": "mesh",
        "HOLOSOMA_FORCE_HETEROGENEOUS_OBJECT_SINGLE_SLOT": "1",
        "HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS": "1",
        "HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK": "0",
        "HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE": "mesh",
        "HOLOSOMA_OBJECT_COLLIDER_TYPE": "convex_decomposition",
        "HOLOSOMA_ROBOT_USD_CACHE_DIR": str(runtime_cache / "robot_usd"),
        "HOLOSOMA_OBJECT_USD_CACHE_DIR": str(runtime_cache / "object_usd"),
    })
    motion = "--command.setup-terms.motion-command.params.motion-config."
    # Native tracking generates the distillation reference trajectories. Every
    # clip is retained, including failures; no manual forward command is applied.
    cli = ["--checkpoint", str(checkpoint), "--output-dir", str(output),
           "--min-contact-frames", "10", "--contact-force-threshold", "1.0",
           "--contact-voxel-size", "0.01", "--success-position-threshold", "0.5",
           "--require-final-position-success-for-success",
           "--require-no-middle-foot-object-contact-for-success",
           "--middle-foot-contact-start-frac", "0.20", "--middle-foot-contact-end-frac", "0.80",
           "--foot-object-contact-force-threshold", "1.0",
           "--no-save-glb", "--no-save-preview-png", "--no-save-face-heatmap-png",
           "randomization:disabled", "logger:disabled",
           f"--training.num-envs={len(clips)}", "--training.headless=True", "--training.seed=42",
           "--simulator.config.sim.max-episode-length-s=1000000",
           "--simulator.config.sim.physx.gpu-collision-stack-size=268435456",
           "--robot.object.enabled=True", f"--robot.object.object-urdf-path={object_map}",
           motion + f"motion-file={bank}", motion + "use-adaptive-timesteps-sampler=False",
           motion + "start-at-timestep-zero-prob=1.0", motion + "freeze-at-timestep-zero-prob=0.0",
           motion + "noise-to-initial-pose.overall-noise-scale=0.0", "--perception.object-geometry-mode=mesh",
           f"--logger.base-dir={output.parent / (output.name + '_logs')}"]
    command = [sys.executable, "-m", "holosoma.export_teacher_box_contacts", *cli]
    return args, cli, env, command


def validate_config(cli):
    import tyro
    from holosoma.config_types.experiment import ExperimentConfig
    from holosoma.export_teacher_box_contacts import ExportConfig
    from holosoma.eval_agent import _validate_eval_policy_contract
    from holosoma.observation import apply_observation_overrides
    from holosoma.perception import apply_perception_overrides
    from holosoma.utils.eval_utils import CheckpointConfig, load_saved_experiment_config
    from holosoma.utils.tyro_utils import TYRO_CONIFG

    checkpoint, remaining = tyro.cli(CheckpointConfig, args=cli, return_unknown_args=True, add_help=False)
    _, remaining = tyro.cli(ExportConfig, args=remaining, return_unknown_args=True, add_help=False)
    saved, _ = load_saved_experiment_config(checkpoint)
    config = tyro.cli(ExperimentConfig, default=saved.get_eval_config(), args=remaining, config=TYRO_CONIFG)
    config = apply_perception_overrides(apply_observation_overrides(config))
    _validate_eval_policy_contract(saved, config)
    print("Rollout checkpoint/config check passed; no simulator, GPU rollout or W&B run started.")


def main():
    args, cli, env, command = prepare()
    if args.check:
        subprocess.run([sys.executable, "-m", "scripts._rollout", "--validate-config", *cli],
                       cwd=ROOT, env=env, check=True)
        print(shlex.join(command))
        return
    os.chdir(ROOT)
    os.execve(sys.executable, command, env)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--validate-config":
        validate_config(sys.argv[2:])
    else:
        main()
