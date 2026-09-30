from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

import holosoma.train_agent_rank_visible as launcher
from holosoma.train_agent_rank_visible import (
    _apply_rank_local_cpu_affinity,
    _remap_rank_to_single_visible_gpu,
)


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _install_fake_numa_topology(sysfs_root: Path) -> None:
    gpu_nodes = {
        "0000:10:00.0": 0,
        "0000:11:00.0": 0,
        "0000:20:00.0": 1,
        "0000:21:00.0": 1,
    }
    for pci_bus_id, numa_node in gpu_nodes.items():
        _write(sysfs_root / "bus/pci/devices" / pci_bus_id / "numa_node", f"{numa_node}\n")

    _write(sysfs_root / "devices/system/node/node0/cpulist", "0-3,8-11\n")
    _write(sysfs_root / "devices/system/node/node1/cpulist", "4-7,12-15\n")
    _write(sysfs_root / "devices/system/cpu/online", "0-15\n")
    for first, second in zip(range(8), range(8, 16), strict=True):
        siblings = f"{first},{second}\n"
        _write(
            sysfs_root
            / f"devices/system/cpu/cpu{first}/topology/thread_siblings_list",
            siblings,
        )
        _write(
            sysfs_root
            / f"devices/system/cpu/cpu{second}/topology/thread_siblings_list",
            siblings,
        )


def _fake_nvidia_smi(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
    del args, kwargs
    return subprocess.CompletedProcess(
        args=["nvidia-smi"],
        returncode=0,
        stdout=(
            "0, GPU-a, 00000000:10:00.0\n"
            "1, GPU-b, 00000000:11:00.0\n"
            "2, GPU-c, 00000000:20:00.0\n"
            "3, GPU-d, 00000000:21:00.0\n"
        ),
        stderr="",
    )


def test_rank_visible_remap_overwrites_stale_original_topology(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOLOSOMA_RANK_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c,GPU-d")
    monkeypatch.setenv("HOLOSOMA_ORIGINAL_LOCAL_RANK", "7")
    monkeypatch.setenv("HOLOSOMA_ORIGINAL_LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("HOLOSOMA_ORIGINAL_CUDA_VISIBLE_DEVICES", "stale")

    _remap_rank_to_single_visible_gpu()

    assert os.environ["HOLOSOMA_ORIGINAL_LOCAL_RANK"] == "2"
    assert os.environ["HOLOSOMA_ORIGINAL_LOCAL_WORLD_SIZE"] == "4"
    assert os.environ["HOLOSOMA_ORIGINAL_CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-b,GPU-c,GPU-d"
    assert os.environ["HOLOSOMA_RANK_VISIBLE_PHYSICAL_DEVICE"] == "GPU-c"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-c"
    assert os.environ["LOCAL_RANK"] == "0"
    assert os.environ["LOCAL_WORLD_SIZE"] == "1"


def test_rank_visible_remap_rejects_invalid_boolean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOLOSOMA_RANK_VISIBLE_DEVICES", "treu")

    with pytest.raises(SystemExit, match="must be a boolean"):
        _remap_rank_to_single_visible_gpu()


def test_rank_visible_remap_rejects_negative_local_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOLOSOMA_RANK_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("LOCAL_RANK", "-1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    with pytest.raises(SystemExit, match="must be non-negative"):
        _remap_rank_to_single_visible_gpu()


@pytest.mark.parametrize("devices", ["0,0", "0,", ",0", "-1"])
def test_rank_visible_remap_rejects_invalid_gpu_list(
    monkeypatch: pytest.MonkeyPatch,
    devices: str,
) -> None:
    monkeypatch.setenv("HOLOSOMA_RANK_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", devices)

    with pytest.raises(SystemExit, match="CUDA_VISIBLE_DEVICES"):
        _remap_rank_to_single_visible_gpu()


def test_rank_local_cpu_affinity_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HOLOSOMA_RANK_LOCAL_CPU_AFFINITY", raising=False)

    def unexpected_query(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        raise AssertionError("disabled affinity must not query NVIDIA topology")

    monkeypatch.setattr(launcher.subprocess, "run", unexpected_query)

    assert _apply_rank_local_cpu_affinity() is False


def test_rank_local_cpu_affinity_uses_gpu_numa_and_preserves_smt_siblings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sysfs_root = tmp_path / "sys"
    _install_fake_numa_topology(sysfs_root)
    monkeypatch.setenv("HOLOSOMA_RANK_LOCAL_CPU_AFFINITY", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c,GPU-d")
    monkeypatch.setattr(launcher.subprocess, "run", _fake_nvidia_smi)
    monkeypatch.setattr(launcher.os, "sched_getaffinity", lambda pid: set(range(16)))
    applied: list[tuple[int, set[int]]] = []
    monkeypatch.setattr(
        launcher.os,
        "sched_setaffinity",
        lambda pid, cpus: applied.append((pid, set(cpus))),
    )
    original_torchrun_env = {
        name: os.environ[name]
        for name in ("LOCAL_RANK", "LOCAL_WORLD_SIZE", "CUDA_VISIBLE_DEVICES")
    }

    assert _apply_rank_local_cpu_affinity(sysfs_root=sysfs_root) is True

    assert applied == [(0, {2, 3, 10, 11})]
    assert {
        name: os.environ[name]
        for name in ("LOCAL_RANK", "LOCAL_WORLD_SIZE", "CUDA_VISIBLE_DEVICES")
    } == original_torchrun_env
    assert os.environ["HOLOSOMA_RANK_LOCAL_CPU_AFFINITY_APPLIED"] == "1"
    assert os.environ["HOLOSOMA_RANK_LOCAL_CPU_AFFINITY_NUMA_NODE"] == "0"
    assert os.environ["HOLOSOMA_RANK_LOCAL_CPU_AFFINITY_CPUS"] == "2-3,10-11"
    assert os.environ["HOLOSOMA_RANK_LOCAL_CPU_AFFINITY_GPU"] == "GPU-b"
    assert os.environ["HOLOSOMA_RANK_LOCAL_CPU_AFFINITY_PCI_BUS_ID"] == "0000:11:00.0"


def test_rank_local_cpu_affinity_invalid_opt_in_is_fail_open(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("HOLOSOMA_RANK_LOCAL_CPU_AFFINITY", "treu")

    def unexpected_query(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        raise AssertionError("invalid opt-in must fail open before topology discovery")

    monkeypatch.setattr(launcher.subprocess, "run", unexpected_query)

    assert _apply_rank_local_cpu_affinity() is False
    assert "fail-open" in capsys.readouterr().err


def test_rank_local_cpu_affinity_missing_topology_is_fail_open_without_env_remap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("HOLOSOMA_RANK_LOCAL_CPU_AFFINITY", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-unknown")
    monkeypatch.setattr(launcher.subprocess, "run", _fake_nvidia_smi)
    monkeypatch.setattr(
        launcher.os,
        "sched_setaffinity",
        lambda pid, cpus: pytest.fail("failed topology discovery must not change affinity"),
    )
    original_torchrun_env = {
        name: os.environ[name]
        for name in ("LOCAL_RANK", "LOCAL_WORLD_SIZE", "CUDA_VISIBLE_DEVICES")
    }

    assert _apply_rank_local_cpu_affinity(sysfs_root=tmp_path / "missing-sys") is False

    assert {
        name: os.environ[name]
        for name in ("LOCAL_RANK", "LOCAL_WORLD_SIZE", "CUDA_VISIBLE_DEVICES")
    } == original_torchrun_env
    assert "did not resolve uniquely" in capsys.readouterr().err
