# Training

Run these commands from the repository root after [installation](../README.md#installation).
Use `--help` on any launcher to see its options.

## Student

Online distillation with PPO, **1 node × 8 GPUs**. After the
[dataset and shard preparation](data.md), use the released teacher 40K and box 23K actor
initializer from this code repository's [`_ckpts/`](../_ckpts/README.md) directory. These are the
checkpoint files validated by the launcher; do not substitute the older copies
inside the dataset archive. The published motion bank already contains the
rollout commands, so no new teacher rollout or command generation is needed.

```bash
STUDENT_DATA="$PWD/data/far-prism-data/data/train-student/data"
STUDENT_SHARDS="$PWD/data/student_shards_ws8"
wandb login
bash train_student.sh \
  --motion-bank "$STUDENT_DATA/motion_bank" \
  --rank-shards "$STUDENT_SHARDS" \
  --contact-bank "$STUDENT_DATA/contact_sidecars" \
  --robot-assets "$STUDENT_DATA/robot_assets" \
  --teacher-checkpoint _ckpts/teacher_40000.pt \
  --initializer-checkpoint _ckpts/box_23000.pt \
  --entity YOUR_WANDB_ENTITY
```

Replace `YOUR_WANDB_ENTITY` with your W&B user or team. Training uses 2,048
environments per GPU, runs for 40K iterations and saves PT/ONNX pairs. Use the same
clean PRISM revision and installed HoloSoma revision on every node. The launchers
detect these revisions and fetch/verify runtime source automatically. Use a new
`--output` for each run; append `--check` for a CPU CLI check or `--help` for options.
`--check` only validates the command syntax; it does not validate assets or run
the simulator.

## Teacher

PPO from scratch, 4 nodes × 8 GPUs. This requires a separate prepared
teacher bank (not yet included in the Hugging Face release). Run on each node
with `--node-rank` set to 0–3.

```bash
bash train_teacher.sh \
  --motion-bank /path/to/teacher_motion_bank \
  --entity YOUR_WANDB_ENTITY \
  --node-rank 0 --master-addr NODE_0_IP
```

For a single-node teacher run, prepare eight rank shards and add
`--nodes 1 --rank-shards /path/to/teacher_shards_ws8`:

```bash
python scripts/prepare_as_rank_shards.py \
  --motion-dir /path/to/teacher_motion_bank \
  --object-map /path/to/teacher_motion_bank/_clip_object_urdf_map.json \
  --world-size 8 --environments-per-rank 2048 \
  --output-root /path/to/teacher_shards_ws8
```

This keeps 2,048 environments per GPU and all clips, with a smaller global batch
than the four-node configuration. Training still runs for 40K updates.

## Rollout

Collect teacher trajectories and contact sidecars. Every clip is retained, including failures.

```bash
bash rollout.sh \
  --checkpoint _ckpts/teacher_40000.pt \
  --motion-bank /path/to/teacher_motion_bank \
  --output /path/to/new_rollouts --gpu 0
```
