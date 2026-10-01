<h1 align="center">
  Counterfactual Video Generation Enables Scalable Humanoid Loco-Manipulation
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2609.38172"><img src="https://img.shields.io/badge/arXiv-2609.38172-b31b1b.svg" alt="arXiv"></a>
  <a href="https://prism-real2sim2real.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
  <a href="https://huggingface.co/datasets/Amazon-FAR/far-prism-data"><img src="https://img.shields.io/badge/Dataset-Hugging%20Face-yellow.svg" alt="Dataset on Hugging Face"></a>
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

## Dataset

Download [Amazon-FAR/far-prism-data](https://huggingface.co/datasets/Amazon-FAR/far-prism-data)
from Hugging Face. The current release contains **129 successful teacher-rollout
trajectories** (35 box, 32 bin, 34 barrel, 28 ball), precomputed policy commands,
matching contact sidecars, G1 assets and reference-replay videos. This is a
filtered subset of the original 137 trajectories; the raw-video, reconstruction
and teacher-training sections are still under review.

From the repository root, download the single **317 MB** archive to avoid
thousands of individual requests. These commands pin the dataset revision and
verify both the archive and its extracted files (requires `curl`):

```bash
mkdir -p data
curl -fL --retry 3 \
  https://huggingface.co/datasets/Amazon-FAR/far-prism-data/resolve/2498b1dbfda63a8000d28aa1b5d504cf9ff6b21b/far-prism-data.tar.gz \
  -o data/far-prism-data.tar.gz && \
  (cd data && \
   echo 'eb1fbcd2ef2714218292f9faf98e6cf57ac7f792da0becf89b3ca89d65850f5d  far-prism-data.tar.gz' | sha256sum --check && \
   tar -xzf far-prism-data.tar.gz && \
   cd far-prism-data && sha256sum --check SHA256SUMS --quiet)
```

The extracted training inputs are:

```text
data/far-prism-data/
  clips.csv                                # Per-clip index
  data/train-student/data/motion_bank/      # Motion NPZs, commands and object map
  data/train-student/data/contact_sidecars/ # Matching contact intervals and points
  data/train-student/data/robot_assets/     # G1 URDF and depth-rendering meshes
  data/train-student/vis/                   # Reference replays
```

**Current asset dependency:** this dataset revision does not include the object
visual/collision meshes referenced by its URDFs under `data/recon-hoi/`.
Those exact assets must be supplied before preparing shards or starting
distillation. The preparation command below checks these dependencies and stops
if any are missing; downloading the archive alone is not yet sufficient to train.

Once those assets are present, prepare eight rank shards locally. Keep the
extracted directory layout intact so relative object paths resolve correctly:

```bash
STUDENT_DATA="$PWD/data/far-prism-data/data/train-student/data"
STUDENT_SHARDS="$PWD/data/student_shards_ws8"
python scripts/prepare_as_rank_shards.py \
  --motion-dir "$STUDENT_DATA/motion_bank" \
  --object-map "$STUDENT_DATA/motion_bank/_clip_object_urdf_map.json" \
  --world-size 8 --environments-per-rank 2048 \
  --output-root "$STUDENT_SHARDS"
```

The shards cover all 129 published clips. Follow the student command below to
use them. The dataset and generated shards remain local under Git-ignored `data/`.

## Train

**Teacher:** PPO from scratch, 4 nodes × 8 GPUs. This requires a separate prepared
teacher bank (not yet included in the Hugging Face release). Run on each node
with `--node-rank` set to 0–3.

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

**Student:** online distillation with PPO, **1 node × 8 GPUs**. After the dataset
and shard preparation above, use the released teacher 40K and box 23K actor
initializer from this code repository's `_ckpts/` directory. These are the
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
  --entity YOUR_WANDB_ENTITY --name student \
  --output outputs/student
```

Replace `YOUR_WANDB_ENTITY` with your W&B user or team. Both stages use 2,048
environments per GPU, run for 40K iterations and save PT/ONNX pairs. Use the same
clean PRISM revision and installed HoloSoma revision on every node. The launchers
detect these revisions and fetch/verify runtime source automatically. Use a new
`--output` for each run; append `--check` for a CPU CLI check or `--help` for options.
`--check` only validates the command syntax; it does not validate assets or run
the simulator.

## Checkpoints

| Checkpoint | Purpose |
|---|---|
| [`teacher_40000.pt`](_ckpts/teacher_40000.pt) | Privileged teacher · 40K |
| [`student_28000.pt`](_ckpts/student_28000.pt) | Distilled depth policy · 28K |
| [`box_23000.pt`](_ckpts/box_23000.pt) | Box policy warm-start · 23K |

The checkpoints include model weights and runtime configuration, without optimizer or training-session state. Use them for inference or actor initialization. File checksums are in [`_ckpts/manifest.json`](_ckpts/manifest.json).

New training uses `peak_height` button labels; released policies retain their
trained command semantics. See [Dataset](#dataset) for the training inputs and
current asset availability.

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
