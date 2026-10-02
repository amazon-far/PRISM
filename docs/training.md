# Training

Run from the repository root after [installation](../README.md#installation).
Both scripts use all visible GPUs on this machine and prepare matching motion
shards automatically. Each GPU runs one process. Select GPUs with
`CUDA_VISIBLE_DEVICES`; `--gpus N` uses the first N visible GPUs.

```bash
CUDA_VISIBLE_DEVICES=0 bash train_student.sh --entity YOUR_WANDB_ENTITY
CUDA_VISIBLE_DEVICES=0,1 bash train_student.sh --entity YOUR_WANDB_ENTITY
```

Use a fresh `--output` for each run. Replace `YOUR_WANDB_ENTITY` with your W&B
user or team. `--envs-per-gpu` limits environments on each GPU (default: 2,048).
The launcher rounds this down when needed to distribute every motion clip
without truncation. For example, the 129-clip dataset uses 1,935 environments
on one GPU and 2,048 per GPU on eight GPUs. The effective count is printed at
startup and recorded in the training config and shard manifest. PhysX retains
the original fixed buffer capacities at every GPU and environment count.
Full-resolution teacher meshes can exceed 48 GB on one or two GPUs at the
default budget. Changing GPU or environment counts changes the global batch;
convergence is not guaranteed to match the released checkpoints.

Mesh conversion and simulator caches are stored under `OUTPUT/cache/`. Use
`--cache-dir /path/to/cache` to reuse mesh caches on a disk with sufficient space.
The first launch can spend several minutes preparing large object banks.

Training runs for 40K updates and saves PT/ONNX pairs. Set `--iterations` for a
short validation run. The launcher verifies the
clean PRISM revision, installed HoloSoma revision, assets and ONNX contract before
training. Append `--check` for a CPU command-syntax check; it does not read assets
or run CUDA. Use `--gpus N --check` to inspect a specific topology, or `--help`
for options.

## Teacher

Train a privileged motion-tracking policy with PPO from scratch. Supply a
prepared teacher bank; teacher-training data is still under review.

```bash
bash train_teacher.sh --motion-bank /path/to/teacher_bank --entity YOUR_WANDB_ENTITY
```

## Rollout

Collect teacher trajectories and contact sidecars. Every clip is retained,
including failures.

```bash
bash rollout.sh --motion-bank /path/to/teacher_bank
```

Rollouts use the released teacher and write to `outputs/rollout/`.
Pass `--output` for subsequent collections; existing outputs are never overwritten.

## Student

Distill the teacher into a depth policy using the [downloaded data](data.md)
and released checkpoints in [`_ckpts/`](../_ckpts/README.md). The dataset already
contains rollout commands, so no new rollout is needed. Use the checkpoints
from this code repository; older copies in the dataset archive do not satisfy
the launcher's checkpoint contract.

```bash
bash train_student.sh --entity YOUR_WANDB_ENTITY
```

For another dataset, pass `--motion-bank`, `--contact-bank` and `--robot-assets`.
Matching shards are prepared under the run's output directory. Optionally pass
`--rank-shards` to reuse a prepared set; its source, GPU count and environment
count must match. Existing shard trees are verified and never overwritten.

## Multiple machines (optional)

Use the same clean, pushed PRISM commit, pinned HoloSoma installation, complete
data bank and GPU count on each machine. Add `--machines M --machine-rank R
--master-addr ADDRESS`, with `R` from 0 to M−1 and `ADDRESS` the first machine's
reachable address. Run the command on every participating machine. This is
optional; the default requires only one machine.
