# PRISM · Sim-to-real

Deploy the student policy with **D435i stereo IR → FastFoundationStereo (FFS) → policy**.
The robot streams stereo images; an Ubuntu GPU laptop connected to the G1 by
Ethernet runs FFS and the policy. Keep the laptop connected to its full-rated
high-power AC adapter throughout deployment.
Training and released checkpoints are on [`main`](https://github.com/amazon-far/PRISM/tree/main).

**The canonical G1 mounting has a default downward camera pitch of 47°.
Training uses 37°: tilt the head back / up by 10° from the canonical position
before deployment.** This physical adjustment aligns the camera pitch with the
depth view used during training. Angles are measured below the torso-forward horizontal.

![Local G1 mesh: canonical 47-degree default; tilt the head back and up by 10 degrees to match the trained 37-degree camera pitch](docs/images/neck_pitch_37_vs_47.png)

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

## Laptop, Ethernet and power

1. Connect the **Ubuntu laptop to the G1 with an Ethernet cable**. Use `ip -br addr`
   to identify the wired interface and verify its address matches your robot-network
   configuration. Pass that interface to `--interface`; `eth0` below is an example.
   The camera host in `--relay-host` must be reachable over this wired connection.
2. **Plug in the laptop's full-rated high-power adapter before selecting Performance
   mode.** For this deployment laptop, battery power or a lower-power charger cannot
   enable the required performance mode. A charging indicator alone is insufficient;
   use the adapter specified for the laptop. The required wattage depends on the model.
3. On the **Ubuntu laptop**, have the agent check and enable the supported Performance
   profile before starting FFS:

   ```bash
   powerprofilesctl list
   # Continue only if the performance profile is available.
   powerprofilesctl set performance
   powerprofilesctl get
   powerprofilesctl list
   bash scripts/check_gpu_perf.sh
   ```

   Confirm `get` reports `performance` and inspect `list` for any degraded-performance
   reason. If the command is missing or the profile is unavailable, check Ubuntu's
   **Settings → Power**, AC-adapter detection and the laptop's vendor/firmware settings.
   Do not treat an unavailable profile as successfully enabled.

Here “turbo/overclock mode” means the supported **high-performance power profile**;
these commands do not apply manual CPU/GPU clock or voltage overclocks. Availability
depends on the hardware and power state; see the [GNOME power-profile guide](https://teams.pages.gitlab.gnome.org/Websites/help.gnome.org/gnome-help/power-profile.html)
and [Ubuntu command reference](https://manpages.ubuntu.com/manpages/noble/man1/powerprofilesctl.1.html).
Check GPU utilization, clocks and power with `check_gpu_perf.sh` while FFS is running,
and inspect its latency logs after warm-up. Performance mode is part of this laptop's
low-latency setup, not a latency guarantee; FFS speed also depends on GPU, input size,
iteration count and inference backend ([upstream guidance](https://github.com/NVlabs/Fast-FoundationStereo#weights-and-trade-off)).

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
