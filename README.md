# PRISM · Sim-to-real

Deploy the student policy with **D435i stereo IR → FastFoundationStereo (FFS) → policy**.
The robot streams stereo images; an external GPU computer runs FFS and the policy.
Training and released checkpoints are on [`main`](https://github.com/amazon-far/PRISM/tree/main).

**Training uses a 37° neck/camera pitch. Match this mounting angle: 47° points
10° further down and changes the depth view seen by the policy.**

![Neck/camera pitch: trained at 37 degrees, compared with 47 degrees](docs/images/neck_pitch_37_vs_47.svg)

## Install

**GPU computer:** Linux x86_64, Python 3.10, and an NVIDIA driver supporting CUDA 12.4.
On Ubuntu, the system packages are `git`, `build-essential`, `python3.10-venv`,
`python3.10-dev`, `libgl1` and `libglib2.0-0`.

```bash
git clone --branch sim2real --recurse-submodules https://github.com/amazon-far/PRISM.git
cd PRISM
python3.10 -m venv .venv
source .venv/bin/activate
bash install.sh
```

The installer initializes the pinned FFS submodule and installs CUDA PyTorch,
FFS dependencies, pyrealsense2, OpenCV, ZeroMQ and the Unitree SDK.

**Robot camera host:** clone the same branch, create and activate a Python environment,
then run `bash install.sh --relay`. This installs NumPy, pyrealsense2 and ZeroMQ;
also install the system packages `usbutils` and `psmisc`. Configure SSH access
from the GPU computer to this host.

## Run

Prepare the student **ONNX** policy and the
[C-Fast-FoundationStereo weights](https://huggingface.co/nvidia/c-fast-foundationstereo/tree/9b446878c81ddb27593036767b29b2859d46103e).
Keep `model_best_bp2_serialize.pth` and `cfg.yaml` in the same directory.
Then run on the GPU computer:

```bash
bash real_ffs.sh \
  --model /path/to/student.onnx \
  --ffs-model /path/to/ffs/model_best_bp2_serialize.pth \
  --relay-host user@camera-host \
  --relay-python /absolute/path/to/camera-env/bin/python \
  --interface eth0
```

Replace the paths, SSH host and robot network interface for your setup.
This command starts the robot's stereo relay, FFS depth and the policy.
**The robot receives hold-position commands as soon as the policy starts.**
Ctrl-C stops the session; logs are saved under `logs/`.

The FFS submodule retains its [upstream license](third_party/Fast-FoundationStereo/LICENSE.txt),
including its non-commercial use restriction; weights have their own terms.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for security reporting.
