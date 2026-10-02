"""Shared launch plumbing for the two public training scripts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
from string import Template
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STUDENT_DATA = ROOT / "data" / "far-prism-data" / "data" / "train-student" / "data"


def bind_nccl(env):
    """Pin the installed collective library before child processes import Torch."""
    package = importlib.metadata.distribution("nvidia-nccl-cu12")
    libraries = [Path(package.locate_file(file)).resolve() for file in package.files or []
                 if str(file).endswith("/libnccl.so.2")]
    if len(libraries) != 1:
        raise ValueError("Install the nvidia-nccl-cu12 library matching PyTorch before training")
    library = libraries[0]
    with library.open("rb") as stream:
        env["NCCL_LIB_SHA256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    env["NCCL_LIB_DIR"] = str(library.parent)
    env["LD_PRELOAD"] = str(library)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [str(library.parent), env.get("LD_LIBRARY_PATH")]))


def run(command, env, *, capture=False):
    return subprocess.run(
        [str(value) for value in command], cwd=ROOT, env=env, check=True,
        text=True, stdout=subprocess.PIPE if capture else None,
    ).stdout


def prepare(role, cli_template, environment, nodes, argv=None):
    parser = argparse.ArgumentParser(
        prog="train_teacher.sh" if role == "teacher" else "train_student.sh",
        description=f"Train {role} on the visible GPUs of this machine."
    )
    parser.add_argument("--motion-bank", type=Path, required=role == "teacher",
                        help="Prepared single-slot motion bank (student default: downloaded Hugging Face bank)")
    parser.add_argument("--rank-shards", type=Path,
                        help="Use existing matching shards (default: prepare automatically)")
    parser.add_argument("--gpus", type=int, help="Use the first N visible GPUs (default: all)")
    parser.add_argument("--envs-per-gpu", type=int, default=2048,
                        help="Environment budget per GPU; rounded down to fit all clips (default: 2048)")
    parser.add_argument("--iterations", type=int, default=40000, help="Training updates (default: 40000)")
    parser.add_argument("--machines", dest="nodes", type=int, default=nodes,
                        help="Number of machines (default: 1)")
    parser.add_argument("--nodes", dest="nodes", type=int, help=argparse.SUPPRESS)
    if role == "distillation":
        parser.add_argument("--contact-bank", type=Path,
                            help="Contact sidecars (default: downloaded dataset; required for a custom bank)")
        parser.add_argument("--robot-assets", type=Path,
                            help="Robot assets (default: downloaded dataset; required for a custom bank)")
        parser.add_argument("--teacher-checkpoint", type=Path, default=ROOT / "_ckpts" / "teacher_40000.pt",
                            help="Distillation teacher (default: _ckpts/teacher_40000.pt)")
        parser.add_argument("--initializer-checkpoint", type=Path, default=ROOT / "_ckpts" / "box_23000.pt",
                            help="Actor initialization checkpoint (default: _ckpts/box_23000.pt)")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"), required=not os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--project", default="carry")
    parser.add_argument("--name", default=role)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / role)
    parser.add_argument("--cache-dir", type=Path,
                        help="Mesh and simulator cache directory (default: OUTPUT/cache)")
    parser.add_argument("--machine-rank", dest="node_rank", type=int, default=0)
    parser.add_argument("--node-rank", dest="node_rank", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--master-addr", help="First machine address (multiple machines only)")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--source-ref", default="main")
    parser.add_argument("--source-commit", help="Override the project commit (default: current HEAD)")
    parser.add_argument("--runtime-source", type=Path,
                        help="Use an existing runtime checkout (default: fetch automatically)")
    parser.add_argument("--runtime-commit", help="Require a specific installed runtime commit")
    parser.add_argument("--runtime-ref", help="Override the runtime ref in requirements.txt")
    parser.add_argument("--check", action="store_true",
                        help="Parse and print the training command on CPU, without starting training")
    args = parser.parse_args(argv)
    downloaded_bank = False
    if role == "distillation":
        default_bank = STUDENT_DATA / "motion_bank"
        if args.motion_bank is None or args.motion_bank.expanduser().resolve() == default_bank:
            args.motion_bank = default_bank
            args.contact_bank = args.contact_bank or STUDENT_DATA / "contact_sidecars"
            args.robot_assets = args.robot_assets or STUDENT_DATA / "robot_assets"
            downloaded_bank = True
        elif args.contact_bank is None or args.robot_assets is None:
            parser.error("A custom --motion-bank requires --contact-bank and --robot-assets")
    nodes = args.nodes
    if nodes < 1 or args.envs_per_gpu < 1 or args.iterations < 1:
        parser.error("--machines, --envs-per-gpu and --iterations must be positive")
    if not 0 <= args.node_rank < nodes:
        parser.error(f"--machine-rank must be in [0, {nodes - 1}]")
    from scripts._training_topology import visible_devices, fit_environments, shard_directory
    try:
        devices = visible_devices(check=args.check, count=args.gpus)
        world_size = nodes * len(devices)
        environments = args.envs_per_gpu if args.check else fit_environments(
            args.motion_bank.expanduser().resolve(), world_size, args.envs_per_gpu,
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.environments_per_rank = environments
    args.prepare_shards = args.rank_shards is None
    if args.rank_shards is None:
        args.rank_shards = (shard_directory(ROOT / "data", world_size, environments)
                            if downloaded_bank else args.output / "rank_shards")
    if not 1024 <= args.master_port < 65535:
        parser.error("--master-port must be between 1024 and 65534")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.name):
        parser.error("--name must be a simple run name without spaces or slashes")
    if not args.check:
        if nodes > 1 and not args.master_addr:
            parser.error("--master-addr must be the first machine address")
        if args.source_commit is not None and not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
            parser.error("--source-commit must be the full SHA pushed to --source-ref")
        if args.runtime_commit is not None and not re.fullmatch(r"[0-9a-f]{40}", args.runtime_commit):
            parser.error("--runtime-commit must be the full SHA used to install both packages")
    bindings = {
        "MOTION_BANK": str(args.motion_bank.expanduser().resolve()),
        "TEACHER_MOTION_BANK": str(args.motion_bank.expanduser().resolve()),
        "WANDB_ENTITY": args.entity, "WANDB_PROJECT": args.project,
        "RUN_NAME": args.name, "LOG_DIR": str(args.output.expanduser().resolve()),
        "NODES": str(nodes), "NPROC": str(len(devices)), "WORLD_SIZE": str(world_size),
        "TOTAL_ENVS": str(world_size * environments), "MULTIGPU": str(world_size > 1),
        "ITERATIONS": str(args.iterations), "LAST_ITERATION": str(args.iterations - 1),
    }
    if role == "distillation":
        for key in ("contact_bank", "robot_assets", "teacher_checkpoint", "initializer_checkpoint"):
            bindings[key.upper()] = str(getattr(args, key).expanduser().resolve())
    cli = [Template(value).substitute(bindings) for value in cli_template]
    # Do not inherit semantic switches or an existing training/logging identity.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("HOLOSOMA_", "WANDB_")) and key not in {
               "CONTACT_EXPORT_ROOT", "CONTACT_SIDECAR_MODE", "FORCE_EIGHT_GPU_CONFIG",
               "PERCEPTION_INJECT_INTO_POLICY_MODULES", "RESET_TO_DEFAULT_POSE",
               "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
           }}
    if os.environ.get("WANDB_API_KEY"):
        env["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    env.update({key: Template(value).substitute(bindings) for key, value in environment.items()})
    if role == "teacher":
        if nodes == 1:
            env.update({"HOLOSOMA_GLOO_GRAD_REDUCE": "1",
                        "HOLOSOMA_HIERARCHICAL_GRAD_REDUCE": "0",
                        "HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER": "0"})
    if args.rank_shards is not None:
        shard_root = args.rank_shards.expanduser().absolute()
        env["HOLOSOMA_RANK_LOCAL_MOTION_ROOT"] = str(shard_root)
        env["HOLOSOMA_MOTION_SHARD_MANIFEST"] = str(shard_root / "manifest.json")
        # Bind the new bank only after validating its complete published tree.
        # Historical view identities do not describe a newly prepared subset.
        for key in ("HOLOSOMA_EXTERNAL_AS_RANK_SHARD_SOURCE_DIGEST",
                    "HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_SOURCE_DIGEST",
                    "HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_VIEW_DIGEST"):
            env.pop(key, None)
    env.update({
        "PYTHONPATH": str(ROOT),
        "CUDA_VISIBLE_DEVICES": ",".join(devices),
        "NODE_RANK": str(args.node_rank), "MASTER_ADDR": args.master_addr or "127.0.0.1",
        "MASTER_PORT": str(args.master_port), "HEADLESS": "1",
        "MOTION_DIR": bindings["MOTION_BANK"],
        "OBJECT_SPEC_PATH": bindings["MOTION_BANK"] + "/_clip_object_urdf_map.json",
        "HOLOSOMA_SOURCE_ROOT": str(ROOT),
    })
    cache = (args.cache_dir or args.output / "cache").expanduser().resolve()
    run_key = hashlib.sha256(bindings["LOG_DIR"].encode()).hexdigest()[:16]
    machine_cache = cache / "runs" / run_key / f"machine_{args.node_rank}"
    env.update({
        "TMPDIR": str(machine_cache / "tmp"),
        "HOLOSOMA_OBJECT_USD_CACHE_DIR": str(machine_cache / "object_usd"),
        "HOLOSOMA_PERCEPTION_MESH_CACHE_DIR": str(cache / "perception_meshes"),
        "HOLOSOMA_ROBOT_USD_CACHE_DIR": str(machine_cache / "robot_usd"),
    })
    command = [sys.executable, "-m", "torch.distributed.run", f"--nnodes={nodes}",
               f"--nproc_per_node={len(devices)}", "--max_restarts=0"]
    if nodes == 1:
        command += ["--standalone"]
    else:
        command += [f"--node_rank={args.node_rank}", f"--master_addr={env['MASTER_ADDR']}",
                    f"--master_port={args.master_port}"]
    command += ["--module", "holosoma.train_agent_rank_visible", *cli]
    return args, bindings, cli, env, command


def bind_rank_shards(args, bindings, env, *, world_size):
    """Validate explicit shards against their source before binding their identity."""
    from scripts.prepare_as_rank_shards import prepare_rank_shards, validate_published_rank_shards

    shard_root = args.rank_shards.expanduser().absolute()
    # Never regenerate an existing tree: another training process may use it.
    validator = (prepare_rank_shards if args.prepare_shards and not os.path.lexists(shard_root)
                 else validate_published_rank_shards)
    manifest = validator(
        motion_dir=Path(bindings["MOTION_BANK"]),
        object_map=Path(bindings["MOTION_BANK"]) / "_clip_object_urdf_map.json",
        output_root=args.rank_shards.expanduser().absolute(),
        world_size=world_size,
        environments_per_rank=args.environments_per_rank,
        **({"replace_existing": False} if validator is prepare_rank_shards else {}),
    )
    env["HOLOSOMA_EXTERNAL_AS_RANK_SHARD_SOURCE_DIGEST"] = manifest["source_digest"]
    return manifest


def launch(role, cli_template, environment, *, nodes):
    args, bindings, cli, env, command = prepare(role, cli_template, environment, nodes)
    run([sys.executable, ROOT / "scripts/validate_train_cli.py",
         "--expected-motion-end-mode", "episodic", "--", *cli], env)
    if args.check:
        print(shlex.join(command))
        print("CLI syntax check passed; no assets, Git remote, GPU or W&B run were accessed. "
              "GPU detection and environment fitting happen at launch; use --gpus to model a GPU count.")
        return

    print(f"Training on {bindings['NPROC']} GPUs per machine: "
          f"{args.environments_per_rank} environments per GPU, {bindings['TOTAL_ENVS']} total.", flush=True)
    if args.environments_per_rank != args.envs_per_gpu:
        print(f"Adjusted environment budget from {args.envs_per_gpu} to {args.environments_per_rank} "
              "per GPU to retain every motion clip.", flush=True)
    output = Path(bindings["LOG_DIR"])
    # Separate preflight files for each machine, including on shared storage.
    work = output / "launch" / f"machine_{args.node_rank}"
    work.mkdir(parents=True, exist_ok=False)
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    remote = run(["git", "remote", "get-url", "origin"], env, capture=True).strip()
    source_commit = args.source_commit or run(["git", "rev-parse", "HEAD"], env, capture=True).strip()
    tree = run(["git", "rev-parse", source_commit + "^{tree}"], env, capture=True).strip()
    verification = work / "git.json"
    run([sys.executable, ROOT / "scripts/verify_formal_git_checkout.py", "--source-root", ROOT,
         "--remote-url", remote, "--remote-ref", args.source_ref,
         "--commit", source_commit, "--tree", tree, "--output", verification], env)
    env["HOLOSOMA_FORMAL_GIT_VERIFICATION_PATH"] = str(verification)
    from scripts._runtime_source import resolve_runtime_source
    runtime = resolve_runtime_source(ROOT / "requirements.txt", checkout=args.runtime_source,
                                     commit=args.runtime_commit, ref=args.runtime_ref)
    runtime_root, runtime_remote = runtime["root"], runtime["remote"]
    runtime_tree = run(["git", "-C", runtime_root, "rev-parse", runtime["commit"] + "^{tree}"], env, capture=True).strip()
    runtime_verification = work / "runtime_git.json"
    run([sys.executable, ROOT / "scripts/verify_formal_git_checkout.py", "--source-root", runtime_root,
         "--remote-url", runtime_remote, "--remote-ref", runtime["ref"],
         "--commit", runtime["commit"], "--tree", runtime_tree, "--output", runtime_verification], env)
    from holosoma.utils.package_source import verify_runtime_installation
    packages = verify_runtime_installation(runtime_root, runtime["commit"], runtime_remote)
    (work / "runtime_packages.json").write_text(json.dumps(packages, indent=2) + "\n")
    env["HOLOSOMA_RUNTIME_GIT_VERIFICATION_PATH"] = str(runtime_verification)
    if args.rank_shards is not None:
        shards = bind_rank_shards(args, bindings, env, world_size=int(bindings["WORLD_SIZE"]))
        (work / "rank_shards.json").write_text(json.dumps(shards, indent=2) + "\n")
    bind_nccl(env)
    provenance_args = [sys.executable, ROOT / "scripts/compute_training_provenance.py",
                       "--training-regime", "pure_rl" if role == "teacher" else "distillation",
                       "--motion-dir", bindings["MOTION_BANK"], "--object-map", env["OBJECT_SPEC_PATH"],
                       "--motion-shard-manifest", env["HOLOSOMA_MOTION_SHARD_MANIFEST"],
                       "--contact-interval-runtime-prepend-compensation", "false" if role == "teacher" else "true",
                       "--source-root", ROOT]
    if role == "distillation":
        provenance_args += ["--contact-root", bindings["CONTACT_BANK"], "--contact-sidecar-mode", "runtime-intervals",
                            "--policy-init-checkpoint", bindings["INITIALIZER_CHECKPOINT"],
                            "--teacher-checkpoint", bindings["TEACHER_CHECKPOINT"], "--student-motion-end-mode", "episodic"]
    provenance = json.loads(run(provenance_args, env, capture=True))
    env["HOLOSOMA_TRAINING_PROVENANCE"] = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
    (work / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (work / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    if role == "distillation":
        run([sys.executable, ROOT / "scripts/box23k_policy_init_preflight.py", "--world-size", bindings["WORLD_SIZE"],
             "--environments-per-rank", str(args.environments_per_rank),
             "--allow-distillation", "--output", work / "initializer.json", "--", *cli], env)
    else:
        run([sys.executable, Path(__file__), "--teacher-onnx", str(work / "teacher.onnx"),
             str(args.environments_per_rank), *cli], env)
    os.chdir(ROOT)
    os.execve(sys.executable, command, env)


def teacher_onnx_preflight(path, cli, environments_per_rank=2048):
    """Export and check the fresh teacher actor before starting the simulator."""
    import dataclasses
    import torch
    import tyro
    from holosoma.agents.modules.module_utils import setup_ppo_actor_module
    from holosoma.config_values.experiment import AnnotatedExperimentConfig
    from holosoma.observation.config_utils import apply_observation_overrides
    from holosoma.perception.config_utils import apply_perception_overrides
    from holosoma.utils.inference_helpers import export_policy_as_onnx, validate_exported_policy_onnx
    from holosoma.utils.runtime_asset_manifest import finalize_runtime_asset_provenance
    from holosoma.utils.tyro_utils import TYRO_CONIFG

    torch.set_num_threads(2)
    config = tyro.cli(AnnotatedExperimentConfig, args=cli, config=TYRO_CONIFG)
    config = dataclasses.replace(config, training=dataclasses.replace(config.training, num_envs=environments_per_rank))
    config = apply_perception_overrides(apply_observation_overrides(config))
    finalize_runtime_asset_provenance(config)
    torch.manual_seed(config.training.seed)
    actor = setup_ppo_actor_module({"actor_obs": 178}, config.algo.config.module_dict.actor,
                                   29, 0.01, "cpu", {"actor_obs": 1})

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = actor

        def forward(self, actor_obs):
            return self.policy.act_inference({"actor_obs": actor_obs})

    wrapper = Wrapper().eval()
    example = {"actor_obs": torch.randn(14, 178)}
    export_policy_as_onnx(wrapper, path, example)
    report = validate_exported_policy_onnx(wrapper=wrapper, onnx_file_path=path, example_obs_dict=example)
    Path(path).with_suffix(".json").write_text(json.dumps({"scope": "fresh_teacher_actor", "parity": report}, indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 4 or sys.argv[1] != "--teacher-onnx":
        raise SystemExit("Use train_teacher.sh or train_student.sh")
    teacher_onnx_preflight(sys.argv[2], sys.argv[4:], int(sys.argv[3]))
