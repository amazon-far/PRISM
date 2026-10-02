<h1 align="center">
  Counterfactual Video Generation Enables Scalable Humanoid Loco-Manipulation
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2609.38172"><img src="https://img.shields.io/badge/arXiv-2609.38172-b31b1b.svg" alt="arXiv"></a>
  <a href="https://prism-real2sim2real.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
  <a href="https://huggingface.co/datasets/Amazon-FAR/far-prism-data"><img src="https://img.shields.io/badge/Dataset-Hugging%20Face-yellow.svg" alt="Dataset on Hugging Face"></a>
</p>

<p align="center">
  This repository provides simulation training and real-robot deployment for PRISM, based on <a href="https://github.com/amazon-far/holosoma">HoloSoma</a>.
</p>

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

## Installation

Linux, Python 3.11 and a compatible NVIDIA GPU are required.
See [system requirements and environment details](docs/installation.md).

```bash
git clone --branch main https://github.com/amazon-far/PRISM.git
cd PRISM
python3.11 -m venv .venv
source .venv/bin/activate
bash install.sh
wandb login
```

## Data

Download and prepare the [Hugging Face dataset](https://huggingface.co/datasets/Amazon-FAR/far-prism-data) under `data/`:

```bash
bash download_data.sh
```

Object meshes are pending release; setup currently stops at the asset check.
[Dataset status](docs/data.md).

## Teacher Training

Train a privileged motion-tracking policy with PPO. Run on each node with
`NODE_RANK` set to 0–3 and `NODE_0_IP` set to the first node's address.
Teacher-training data is still under review; supply a prepared teacher bank.

```bash
bash train_teacher.sh --motion-bank /path/to/teacher_bank --entity YOUR_WANDB_ENTITY \
  --node-rank NODE_RANK --master-addr NODE_0_IP
```

## Rollout

Collect teacher trajectories and contact sidecars into `outputs/rollout/`.

```bash
bash rollout.sh --motion-bank /path/to/teacher_bank
```

## Student Distillation

Distill the teacher into a depth policy using the prepared data under `data/`.

```bash
bash train_student.sh --entity YOUR_WANDB_ENTITY
```

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

## License and security

See [LICENSE](LICENSE), [NOTICE](NOTICE) and [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for security reporting.
