"""Select local CUDA devices and fit complete motion banks to a GPU budget."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys


def visible_devices(*, check=False, count=None):
    if count is not None and count < 1:
        raise ValueError("--gpus must be positive")
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = None
    if raw is not None:
        devices = [value.strip() for value in raw.split(",")]
        if (not devices or len(set(devices)) != len(devices)
                or any(not re.fullmatch(r"(?:0|[1-9][0-9]*|GPU-[a-fA-F0-9-]+)", d) for d in devices)):
            raise ValueError("CUDA_VISIBLE_DEVICES must list distinct GPU indices or GPU UUIDs")
    if not check:
        # Match the worker's device order; querying in a child keeps CUDA out of
        # the launcher. PyTorch respects scheduler/container CUDA visibility.
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID")
        if devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
        result = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
            env=env, check=True, text=True, capture_output=True,
        )
        available = int(result.stdout.strip())
        if available < 1:
            raise ValueError("No CUDA GPU is visible; check your driver and CUDA_VISIBLE_DEVICES")
        if devices is not None and available != len(devices):
            raise ValueError("CUDA_VISIBLE_DEVICES contains an unavailable or duplicate GPU")
        devices = devices or [str(i) for i in range(available)]
    elif devices is None:
        # CPU-only syntax checking never queries CUDA. --gpus models another
        # topology; without it, use a one-GPU command for this check only.
        devices = [str(i) for i in range(count or 1)]
    if count is not None:
        if count > len(devices):
            raise ValueError(f"--gpus={count} exceeds the {len(devices)} visible GPUs")
        devices = devices[:count]
    return devices


def fit_environments(motion_bank: Path, world_size: int, budget: int) -> int:
    """Keep every clip and choose the largest compatible per-GPU env count."""
    from scripts.prepare_as_rank_shards import _compatible_rank_clip_capacities

    if world_size < 1 or budget < 1:
        raise ValueError("GPU and environment counts must be positive")
    clips = len(list(motion_bank.glob("*.npz")))
    if not clips:
        raise ValueError(f"No motion NPZs found in {motion_bank}")
    if clips < world_size:
        return budget  # The shard builder records and weights duplicated coverage.
    minimum = (clips + world_size - 1) // world_size
    for environments in range(budget, minimum - 1, -1):
        try:
            _compatible_rank_clip_capacities(
                clip_count=clips, world_size=world_size, environments_per_rank=environments,
            )
            return environments
        except ValueError:
            continue
    raise ValueError(
        f"Cannot fit all {clips} clips on {world_size} GPUs within --envs-per-gpu={budget}; "
        "increase --envs-per-gpu or use more GPUs"
    )


def shard_directory(parent: Path, world_size: int, environments: int) -> Path:
    suffix = "" if environments == 2048 else f"_e{environments}"
    return parent / f"student_shards_ws{world_size}{suffix}"


def physics_buffers(environments: int) -> dict[str, str]:
    """Size PhysX capacity per GPU; preserve the published 2048-env recipe."""
    # Reference capacities are the release recipe. Floors are Isaac Lab's
    # PhysxCfg defaults; round upwards to whole MiB/count blocks for headroom.
    profiles = {
        "PHYSX_FOUND_LOST": (335544320, 2**21),
        "PHYSX_FOUND_LOST_AGGREGATE": (469762048, 2**25),
        "PHYSX_TOTAL_AGGREGATE": (83886080, 2**21),
        "PHYSX_COLLISION_STACK": (268435456, 2**26),
    }
    if environments < 1:
        raise ValueError("Environment count must be positive")
    quantum = 2**20
    return {key: str(max(floor, ((reference * environments + 2048 * quantum - 1)
                                // (2048 * quantum)) * quantum))
            for key, (reference, floor) in profiles.items()}
