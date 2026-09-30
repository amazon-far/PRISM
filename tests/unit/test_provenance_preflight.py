from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

from holosoma.utils.provenance_preflight import scientific_provenance_contract
from holosoma.utils.training_provenance import embedded_runtime_asset_manifest_sha256


_DIGEST_KEYS = (
    "teacher_sha256",
    "policy_init_sha256",
    "training_resume_sha256",
    "motion_shard_manifest_sha256",
    "contact_sidecar_manifest_sha256",
    "source_bundle_sha256",
    "runtime_asset_manifest_sha256",
)


def test_scientific_contract_records_but_does_not_require_kernel_build_identity() -> None:
    first = {
        "environment": {
            "platform": "Linux-6.17.0-1007-aws",
            "python": "3.11.15",
            "torch": "2.7.0+cu128",
            "packages": {"numpy": "1.26.0"},
        }
    }
    second = json.loads(json.dumps(first))
    second["environment"]["platform"] = "Linux-6.17.0-1017-aws"

    assert first != second
    assert scientific_provenance_contract(first) == scientific_provenance_contract(second)


def test_scientific_contract_still_requires_numpy_and_runtime_overlay_identity() -> None:
    first = {
        "environment": {
            "platform": "Linux-a",
            "packages": {"numpy": "1.26.0"},
            "python_runtime_manifest_sha256": "a" * 64,
        }
    }
    second = json.loads(json.dumps(first))
    second["environment"]["platform"] = "Linux-b"
    second["environment"]["packages"]["numpy"] = "1.23.5"
    assert scientific_provenance_contract(first) != scientific_provenance_contract(second)

    second["environment"]["packages"]["numpy"] = "1.26.0"
    second["environment"]["python_runtime_manifest_sha256"] = "b" * 64
    assert scientific_provenance_contract(first) != scientific_provenance_contract(second)


def _unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def test_standalone_preflight_ignores_inherited_torchelastic_agent_store() -> None:
    """The provenance rendezvous must create its own store on its separate port."""

    world_size = 2
    port = _unused_local_port()
    runtime_asset_manifest = {"version": 2, "fixture": "cross-rank-preflight"}
    payload = json.dumps(
        {
            "version": 2,
            **{key: format(index, "x") * 64 for index, key in enumerate(_DIGEST_KEYS, start=1)},
            "policy_init_enabled": True,
            "training_resume_enabled": True,
            "runtime_asset_manifest_phase": "final",
            "runtime_asset_manifest_sha256": embedded_runtime_asset_manifest_sha256(
                runtime_asset_manifest
            ),
            "runtime_asset_manifest": runtime_asset_manifest,
        }
    )
    repo_root = Path(__file__).resolve().parents[2]
    python_path = os.pathsep.join(
        filter(None, (str(repo_root / "src" / "holosoma"), os.environ.get("PYTHONPATH", "")))
    )

    processes: list[subprocess.Popen[str]] = []
    try:
        for rank in range(world_size):
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": "",
                    "MASTER_ADDR": "127.0.0.1",
                    "MASTER_PORT": "unused-torchrun-agent-store-port",
                    "PYTHONPATH": python_path,
                    "RANK": str(rank),
                    # This is what torchrun workers inherit.  Without the
                    # module's explicit override, every provenance rank acts
                    # as a TCPStore client and waits for a server forever.
                    "TORCHELASTIC_USE_AGENT_STORE": "True",
                    "WORLD_SIZE": str(world_size),
                    "HOLOSOMA_PROVENANCE_TIMEOUT_SEC": "5",
                }
            )
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "holosoma.utils.provenance_preflight",
                        "--world-size",
                        str(world_size),
                        "--master-port",
                        str(port),
                    ],
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )

        # Feed every rank before waiting on rank zero; each module reads all of
        # stdin before entering the collective rendezvous.
        for process in processes:
            assert process.stdin is not None
            process.stdin.write(payload)
            process.stdin.close()
            process.stdin = None
        results = [process.communicate(timeout=15) for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()

    for process, (stdout, stderr) in zip(processes, results, strict=True):
        assert process.returncode == 0, stderr
        assert "cross_rank_training_provenance_verified world_size=2" in stdout
