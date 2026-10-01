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

## Branches

Use [`main`](https://github.com/amazon-far/PRISM/tree/main) for simulation training
and [`sim2real`](https://github.com/amazon-far/PRISM/tree/sim2real) for real-robot
deployment with FastFoundationStereo.

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

The dataset is available on [Hugging Face](https://huggingface.co/datasets/Amazon-FAR/far-prism-data).
See [download, extraction and preparation](docs/data.md) to place it under `data/`.
The documented dataset snapshot still needs the object meshes before training.

## Teacher Training

Train the teacher with [`train_teacher.sh`](train_teacher.sh).
See [teacher training instructions](docs/training.md#teacher).

## Rollout

Collect teacher trajectories with [`rollout.sh`](rollout.sh).
See [rollout instructions](docs/training.md#rollout).

## Student Distillation

Train the student with [`train_student.sh`](train_student.sh).
See [distillation instructions](docs/training.md#student).

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
