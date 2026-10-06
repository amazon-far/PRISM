# PRISM · Sim-to-real

Deploy the student policy with **D435i stereo IR → FastFoundationStereo (FFS) → policy**.
The robot streams stereo images; an Ubuntu GPU laptop runs FFS and the policy.
Training and released checkpoints are on [`main`](https://github.com/amazon-far/PRISM/tree/main).

## Depth

The canonical G1 camera points **47° downward**; training uses **37°**.
**Tilt the neck/head back (up) by 10° before deployment** to match the training
view. Angles are measured below the torso-forward horizontal.

![Local G1 mesh: canonical 47-degree default; tilt the head back and up by 10 degrees to match the trained 37-degree camera pitch](docs/images/neck_pitch_37_vs_47.png)

To verify the alignment, an agent can help build a live comparison of real and
simulated point clouds at a matched robot pose, checking ground-plane tilt and
height in a common metric coordinate frame. **This point-cloud comparison tool
is not included in this repository.**

## Install

**GPU computer:** Linux x86_64, Python 3.10, and an NVIDIA driver supporting CUDA 12.4.
On Ubuntu, the system packages are `git`, `build-essential`, `python3.10-venv`,
`python3.10-dev`, `libgl1`, `libegl1`, `libglib2.0-0` and `libusb-1.0-0`.
Python 3.10 may need to be installed separately on your distribution.

```bash
git clone --branch sim2real --recurse-submodules https://github.com/amazon-far/PRISM.git
cd PRISM
python3.10 -m venv .venv
source .venv/bin/activate
bash install.sh
```

The installer initializes the pinned FFS submodule and installs CUDA PyTorch,
FFS dependencies, pyrealsense2, OpenCV, ZeroMQ and the Unitree SDK.
The default FFS installation uses the resolved versions in `constraints-ffs.txt`.
Use a fresh environment without `--system-site-packages`, separate from the
Python 3.11 simulation environment. No pre-existing HoloSoma environment or
local CUDA toolkit is required. Run `python scripts/check_install.py` and
`python -m pip check` to verify installation without connecting to the robot.

**Robot camera host:** clone the same branch, create and activate a Python 3.10 environment,
then run `bash install.sh --relay`. This installs NumPy, pyrealsense2 and ZeroMQ;
also install the system packages `libusb-1.0-0`, `usbutils` and `psmisc`. Configure SSH access
from the GPU computer to this host.

## Local Ethernet Inference Setup

Connect the **Ubuntu GPU laptop to the G1 by Ethernet (a network cable)**.
Use `ip -br addr` to identify the wired interface and confirm its robot-network
address. Pass it to `--interface`; `eth0` below is an example. The camera host
specified by `--relay-host` must also be reachable over this connection.

**Keep the laptop on its full-rated high-power AC adapter and confirm Performance
mode is enabled** to reduce power-saving downclocking and FFS latency. Our deployment
laptop requires this adapter to enable the mode; a charging indicator alone is
insufficient. Use the supported [Performance power profile](https://teams.pages.gitlab.gnome.org/Websites/help.gnome.org/gnome-help/power-profile.html).

An agent can help confirm the active profile, power limits and GPU clocks under
inference load. Run on the Ubuntu laptop:

```bash
powerprofilesctl list              # Confirm performance is available.
powerprofilesctl set performance
powerprofilesctl get               # Must report performance.
bash scripts/check_gpu_perf.sh     # Inspect clocks/power while FFS is running.
```

## Run

The released **student 28K ONNX** is included at
[`_ckpts/student_28000.onnx`](_ckpts/student_28000.onnx), matching the student
checkpoint on `main` and the **37° camera mounting** shown above.
Its checksum and model interface are recorded in [`_ckpts/manifest.json`](_ckpts/manifest.json).
Download the
[C-Fast-FoundationStereo weights](https://huggingface.co/nvidia/c-fast-foundationstereo/tree/9b446878c81ddb27593036767b29b2859d46103e).
Keep `model_best_bp2_serialize.pth` and `cfg.yaml` in the same directory.
After checking the laptop's power mode and wired connection, run on the GPU laptop:

```bash
bash real_ffs.sh \
  --model _ckpts/student_28000.onnx \
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

## Repository layout

The root keeps the installation files and required FFS/policy launchers. `src/`,
`scripts/`, `_ckpts/` and `third_party/` contain the runtime, checks and model;
`docs/images/` contains the camera-mounting guide used above.
Optional native-depth launchers, offline analysis tools, tests, community documents
and source provenance are collected in [`extras/`](extras/README.md).
`LICENSE`, `NOTICE` and `THIRD_PARTY_LICENSES` remain at the root for distribution.

## Security

See [CONTRIBUTING](extras/community/CONTRIBUTING.md#security-issue-notifications) for security reporting.
