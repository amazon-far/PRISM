<h1 align="center">
  Counterfactual Video Generation Enables Scalable Humanoid Loco-Manipulation
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2609.38172"><img src="https://img.shields.io/badge/arXiv-2609.38172-b31b1b.svg" alt="arXiv"></a>
  <a href="https://prism-real2sim2real.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
  <img src="https://img.shields.io/badge/Dataset-under%20review-orange.svg" alt="Dataset: under review">
</p>

<p align="center">
  <strong>Zihan Wang, Zhen Wu, Pieter Abbeel, Rocky Duan, Jitendra Malik,<br>
  Carmelo Sferrazza, C. Karen Liu, Guanya Shi, Angjoo Kanazawa.</strong>
</p>

<p align="center">Conference on Robot Learning (CoRL), 2026.</p>

<p align="center">
  This repository provides simulation training and real-robot deployment for PRISM, based on <a href="https://github.com/amazon-far/holosoma">HoloSoma</a>.
</p>

<p align="center"><strong>This release is currently agent organized. Stay tuned for the final verified version.</strong></p>

## Code branches

| Branch | Use |
|---|---|
| [`main`](https://github.com/amazon-far/PRISM/tree/main) | Simulation: teacher training, rollout collection and student distillation |
| [`sim2real`](https://github.com/amazon-far/PRISM/tree/sim2real) | Real-robot deployment with FastFoundationStereo |

For real-robot deployment, switch an existing clone to `sim2real` and follow that branch's README:

```bash
git fetch origin
git switch sim2real
git submodule update --init --recursive
```

## Simulation setup

Use **Linux x86_64 (Ubuntu 22.04/24.04), Python 3.11** and an NVIDIA driver
supporting CUDA 12.8 and Isaac Sim 5.1. Install the system packages `git`,
`build-essential`, `python3.11-venv`, `python3.11-dev`, `libgl1`, `libglib2.0-0`,
`libglu1-mesa` and `libxrender1`. Python 3.11 may need to be installed separately
on your distribution; venv, Conda and uv environments are supported.

Clone the simulation branch, then run the commands from the repository root:

```bash
git clone --branch main https://github.com/amazon-far/PRISM.git
cd PRISM
python3.11 -m venv .venv
source .venv/bin/activate
bash install.sh
```

The installer downloads PyTorch 2.7.0 (CUDA 12.8), Isaac Sim 5.1.0, a fixed
Isaac Lab revision (package version 0.47.2), and the HoloSoma revision in
[`requirements.txt`](requirements.txt). It uses the active environment and
does not require an existing HoloSoma checkout, Conda environment name or local CUDA toolkit.
Keep this environment separate from `sim2real`; do not use `--system-site-packages`.

Run `python scripts/check_environment.py` to verify the installation, or add
`--simulator` for ten headless steps on one GPU (no training or W&B run).
See [installation details](docs/installation.md) for the two explicit vendor
dependency overrides and the supported hardware requirements.

**Dataset: under review.** Training requires prepared motion banks, object assets,
contact sidecars and rank shards. The G1 robot model is included in HoloSoma.

## Train

**Teacher:** PPO from scratch, 4 nodes × 8 GPUs. Run on each node with `--node-rank` set to 0–3.

```bash
bash train_teacher.sh \
  --motion-bank /path/to/teacher_motion_bank \
  --entity YOUR_WANDB_ENTITY --name teacher \
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

**Rollout:** collect teacher trajectories and contact sidecars. Every clip is retained, including failures.

```bash
bash rollout.sh \
  --checkpoint _ckpts/teacher_40000.pt \
  --motion-bank /path/to/teacher_motion_bank \
  --output /path/to/new_rollouts --gpu 0
```

**Student:** online distillation with PPO, 1 node × 8 GPUs. Initialize the actor from box 23K. Use the prepared rollout command bank and matching contacts; raw rollout output requires command-bank and shard preparation using `scripts/`.

```bash
bash train_student.sh \
  --motion-bank /path/to/student_motion_bank \
  --contact-bank /path/to/contact_sidecars \
  --robot-assets /path/to/box23k_robot_assets \
  --teacher-checkpoint _ckpts/teacher_40000.pt \
  --initializer-checkpoint _ckpts/box_23000.pt \
  --entity YOUR_WANDB_ENTITY --name student
```

Both stages use 2,048 environments per GPU, run for 40K iterations and save PT/ONNX pairs. Use the same clean PRISM revision and installed HoloSoma revision on every node. The launchers detect these revisions and fetch/verify runtime source automatically. Use a new `--output` for each run; append `--check` for a CPU CLI check or `--help` for options.

## Checkpoints

| Checkpoint | Purpose |
|---|---|
| [`teacher_40000.pt`](_ckpts/teacher_40000.pt) | Privileged teacher · 40K |
| [`student_28000.pt`](_ckpts/student_28000.pt) | Distilled depth policy · 28K |
| [`box_23000.pt`](_ckpts/box_23000.pt) | Box policy warm-start · 23K |

The checkpoints include model weights and runtime configuration, without optimizer or training-session state. Use them for inference or actor initialization. File checksums are in [`_ckpts/manifest.json`](_ckpts/manifest.json).

New training uses `peak_height` button labels; released policies retain their trained command semantics. Supply motion banks, contact sidecars and object assets separately.

See [LICENSE](LICENSE), [NOTICE](NOTICE) and [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES) for attribution and terms.

## Citation

If you use PRISM in your research, please cite:

```bibtex
@article{wang2026counterfactual,
  title={Counterfactual Video Generation Enables Scalable Humanoid Loco-Manipulation},
  author={Wang, Zihan and Wu, Zhen and Abbeel, Pieter and Duan, Rocky and Malik, Jitendra and Sferrazza, Carmelo and Liu, C. Karen and Shi, Guanya and Kanazawa, Angjoo},
  journal={arXiv preprint arXiv:2609.38172},
  year={2026}
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for security reporting.
