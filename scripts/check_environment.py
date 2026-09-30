"""Check an isolated simulation installation; GPU simulation is opt-in."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_COMMIT = "3c6e67bb5c7ada942a6d1884ab69338f57596f77"
# Only these exact vendor metadata conflicts are accepted, never arbitrary pip errors.
VENDOR_OVERRIDES = {
    ("isaacsim-kernel", "5.1.0.0", "typing-extensions==4.12.2"): "4.15.0",
    ("isaacsim-kernel", "5.1.0.0", "websockets==12.0"): "16.0",
}


def check_platform():
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Simulation requires Python 3.11; sim2real uses a separate Python 3.10 environment.")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("The simulation installer supports Linux x86_64.")
    libc, version = platform.libc_ver()
    if libc != "glibc" or tuple(map(int, version.split(".")[:2])) < (2, 35):
        raise RuntimeError("Isaac Sim 5.1 pip wheels require glibc >= 2.35 (Ubuntu 22.04 or newer).")
    prefix = Path(sys.prefix)
    if sys.prefix == sys.base_prefix and not (prefix / "conda-meta").is_dir():
        raise RuntimeError("Activate a fresh virtual environment before installing.")
    cfg = prefix / "pyvenv.cfg"
    if cfg.is_file() and "include-system-site-packages = true" in cfg.read_text().lower():
        raise RuntimeError("Create the virtual environment without --system-site-packages.")


def check_dependencies(distributions=None):
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    distributions = list(metadata.distributions() if distributions is None else distributions)
    installed = {canonicalize_name(d.metadata["Name"]): d for d in distributions}
    errors, overrides = [], []
    for name, distribution in installed.items():
        for value in distribution.requires or []:
            requirement = Requirement(value)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            dependency = installed.get(canonicalize_name(requirement.name))
            actual = dependency.version if dependency else None
            if actual is not None and (requirement.url or requirement.specifier.contains(actual, prereleases=True)):
                continue
            normalized = f"{canonicalize_name(requirement.name)}{requirement.specifier}"
            approved = VENDOR_OVERRIDES.get((name, distribution.version, normalized))
            message = f"{name} {distribution.version}: {requirement}; installed {actual}"
            if approved is not None and actual == approved:
                overrides.append(message)
            else:
                errors.append(message)
    if errors:
        raise RuntimeError("Dependency check failed:\n" + "\n".join(errors))
    return overrides


def check_installation():
    # The helper lives beside this script; it reads the public requirements file.
    from _runtime_source import dependency_source, installed_commit

    expected = {"torch": "2.7.0+cu128", "torchvision": "0.22.0+cu128",
                "torchaudio": "2.7.0+cu128", "isaacsim": "5.1.0.0", "isaaclab": "0.47.2"}
    from packaging.requirements import Requirement
    for line in (ROOT / "requirements/simulation-constraints.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            requirement = Requirement(line)
            if not requirement.specifier.contains(metadata.version(requirement.name)):
                raise RuntimeError(f"Expected {line}; rerun bash install.sh in a fresh environment.")
    for name, version in expected.items():
        if metadata.version(name) != version:
            raise RuntimeError(f"Expected {name}=={version}; rerun bash install.sh.")
    overrides = check_dependencies()
    remote, commit = dependency_source(ROOT / "requirements.txt")
    if installed_commit(remote) != commit:
        raise RuntimeError("Installed HoloSoma does not match requirements.txt.")
    import isaaclab
    lab_root = Path(isaaclab.__file__).resolve().parents[3]
    actual = subprocess.check_output(["git", "-C", str(lab_root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(lab_root), "status", "--porcelain"], text=True).strip()
    if actual != ISAACLAB_COMMIT or dirty:
        raise RuntimeError("Isaac Lab source must match the clean revision installed by install.sh.")
    for module in ["torch", "cv2", "pinocchio", "onnx", "onnxruntime", "warp", "trimesh", "h5py",
                   "holosoma.train_agent", "holosoma.export_teacher_box_contacts"]:
        imported = importlib.import_module(module)
        if not Path(imported.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError(f"{module} was imported from outside the active environment: {imported.__file__}")
    import torch
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "python_prefix": sys.prefix, "packages": expected, "runtime_commit": commit,
              "isaaclab_commit": actual, "documented_vendor_overrides": overrides,
              "cuda_available": torch.cuda.is_available()}
    print(json.dumps(result, indent=2), flush=True)


def check_simulator():
    # SimulationApp.close can terminate Python with exit code zero even after
    # an exception. Require a child-written result as well as a successful exit.
    with tempfile.TemporaryDirectory(prefix="prism-simulator-check-") as directory:
        result = Path(directory) / "result.json"
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--simulator-worker", str(result)],
                       check=True, timeout=180)
        if not result.is_file() or json.loads(result.read_text()).get("steps") != 10:
            raise RuntimeError("Simulator exited without completing ten steps; inspect the errors above.")
    print("Simulator smoke check passed: 10 headless steps; no training or robot connection.", flush=True)


def simulator_worker(result):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Simulator smoke check requires an NVIDIA GPU and a CUDA 12.8 compatible driver.")
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device="cuda:0", multi_gpu=False).app
    try:
        from isaaclab.sim import SimulationCfg, SimulationContext
        import holosoma.simulator.isaacsim.isaacsim  # noqa: F401
        simulation = SimulationContext(SimulationCfg(dt=0.02, device="cuda:0"))
        simulation.reset()
        for _ in range(10):
            simulation.step(render=False)
        result.write_text(json.dumps({"steps": 10, "device": torch.cuda.get_device_name(0)}))
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        # Use the same headless shutdown path as training and rollout.
        from holosoma.utils.sim_utils import close_simulation_app
        from holosoma.utils.simulator_config import SimulatorType, set_simulator_type_enum
        set_simulator_type_enum(SimulatorType.ISAACSIM)
        close_simulation_app(app)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform-only", action="store_true")
    parser.add_argument("--simulator", action="store_true", help="Also run ten headless simulator steps on one GPU")
    parser.add_argument("--simulator-worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.simulator_worker is not None:
        simulator_worker(args.simulator_worker)
        return
    check_platform()
    if not args.platform_only:
        check_installation()
        if args.simulator:
            check_simulator()


if __name__ == "__main__":
    main()
