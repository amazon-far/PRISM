# Simulation environment

The supported installation uses Linux x86_64, glibc 2.35 or newer, Python 3.11,
and a CUDA 12.8 compatible NVIDIA driver. Follow NVIDIA's
[Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
for GPU support. The training recipes were validated with eight 48 GB GPUs
per node and 2,048 environments per GPU. Memory use also depends on the motion
bank and object meshes. Changing the GPU count changes the global batch and
requires matching motion shards.
The software is not tied to a hostname, username, mount point or network interface.

Create a fresh environment and run `bash install.sh` from the repository.
The installer uses `python3` from that environment; set `PRISM_PYTHON` to use
another interpreter. The CUDA libraries are installed as Python wheels;
a separate CUDA toolkit is not required.
For uv environments, use `uv venv --python 3.11 --seed .venv` so that pip is available.
Isaac Lab is cloned into the environment's `share/prism/IsaacLab` directory.
`PRISM_ISAACLAB_ROOT` can point to another clean checkout of the documented revision.
No local simulator checkout is copied, and the upstream source is not patched.
`requirements/simulation-lock.txt` fixes the resolved dependency versions for
Python 3.11 on Linux x86_64, including the CUDA wheel variants.

The installer follows the [Isaac Lab pip installation workflow](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html).
Its simulator downloads are large; allow at least 30 GB for the environment and caches.
The first simulator launch follows NVIDIA's EULA acceptance flow.

## Dependency compatibility

HoloSoma's configuration parser and visualization dependency require newer
pure-Python libraries than the exact pins in Isaac Sim 5.1's package metadata:

| Package | Isaac Sim pin | Installed compatibility version | Reason |
|---|---|---|---|
| `typing-extensions` | 4.12.2 | 4.15.0 | Tyro requires >=4.13 |
| `websockets` | 12.0 | 16.0 | Viser requires >=13.1 |

These are explicit [uv dependency overrides](https://docs.astral.sh/uv/pip/compile/#overriding-dependency-versions)
in `requirements/simulation-overrides.txt`. `python scripts/check_environment.py`
checks installed dependencies and permits only these exact package/version
exceptions. Any other missing or incompatible dependency fails the check.
A plain `pip check` therefore reports these two known metadata conflicts;
they are not silently discarded. Build isolation also constrains setuptools
below 81 because the upstream `flatdict` build imports `pkg_resources`.

The checker verifies the pinned HoloSoma installation and clean Isaac Lab Git
revision, imports the training and rollout entry points, and reports CUDA availability.
`python scripts/check_environment.py --simulator` additionally loads the
HoloSoma simulator backend and advances an empty headless simulation ten steps.
This smoke check does not establish training convergence or policy success rates.

Motion banks, object assets, contacts, checkpoints, node addresses and output
directories are provided through the launchers' command-line arguments.
An installation check does not need training data or W&B credentials.
