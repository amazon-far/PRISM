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

## Installation

Linux, Python 3.11 and a compatible NVIDIA GPU are required.
See [system requirements and environment details](docs/installation.md).

```bash
git clone --branch main https://github.com/amazon-far/PRISM.git
cd PRISM
python3.11 -m venv .venv
source .venv/bin/activate
bash install.sh
```

## Data

Download and extract the [Hugging Face dataset](https://huggingface.co/datasets/Amazon-FAR/far-prism-data) into `data/`:

```bash
bash download_data.sh
```

Add the required object meshes and [prepare training shards](docs/data.md) before training.

## Teacher Training

Train a [privileged motion-tracking policy](docs/training.md#teacher) with PPO using [`train_teacher.sh`](train_teacher.sh).

## Rollout

Collect [teacher trajectories and contact sidecars](docs/training.md#rollout) using [`rollout.sh`](rollout.sh).

## Student Distillation

Distill the teacher into a [depth policy](docs/training.md#student) using [`train_student.sh`](train_student.sh).

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
