"""Prepare downloaded student data for the locally visible GPUs."""

import argparse
from pathlib import Path

from scripts._training_topology import fit_environments, shard_directory, visible_devices
from scripts.prepare_as_rank_shards import prepare_rank_shards, validate_published_rank_shards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--envs-per-gpu", type=int, default=2048)
    args = parser.parse_args()
    devices = visible_devices(count=args.gpus)
    bank = args.data_dir.resolve() / "far-prism-data/data/train-student/data/motion_bank"
    environments = fit_environments(bank, len(devices), args.envs_per_gpu)
    shards = shard_directory(args.data_dir.resolve(), len(devices), environments)
    prepare = validate_published_rank_shards if shards.exists() or shards.is_symlink() else prepare_rank_shards
    manifest = prepare(motion_dir=bank, object_map=bank / "_clip_object_urdf_map.json",
                       world_size=len(devices), environments_per_rank=environments, output_root=shards,
                       **({"replace_existing": False} if prepare is prepare_rank_shards else {}))
    print(f"Prepared {manifest['clip_count']} clips for {len(devices)} GPUs, "
          f"{environments} environments per GPU: {shards}")


if __name__ == "__main__":
    main()
