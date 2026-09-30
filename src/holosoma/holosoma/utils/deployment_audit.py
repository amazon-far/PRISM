"""Bounded, asynchronous evidence capture. Never changes a policy's inputs.

Recording errors invalidate the evidence, not the robot control loop. A missing
closed manifest or any dropped records must not be reported as a complete test.
"""

from __future__ import annotations

import atexit
from dataclasses import asdict, is_dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import queue
import socket
import shutil
import subprocess
import sys
import threading
import time

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def depth_digest(array):
    return hashlib.sha256(np.asarray(array, dtype="<f4").tobytes()).hexdigest()


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported audit metadata: {type(value).__name__}")


def _identity():
    root = Path(__file__).resolve().parents[4]

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], timeout=10).decode().strip()

    packages = {}
    for name in ("numpy", "opencv-python", "onnxruntime", "onnxruntime-gpu", "pyrealsense2"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "source_root": str(root), "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"), "remote": git("remote", "get-url", "origin"),
        "tracked_diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
        "source_status": git("status", "--short", "--untracked-files=all", "--", "src", "scripts", "*.sh"),
        "hostname": socket.gethostname(), "pid": os.getpid(), "python": sys.version,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "packages": packages, "argv": sys.argv,
        "command_environment": {key: value for key, value in os.environ.items()
                                if key.startswith(("HOLOSOMA_POLICY_", "HOLOSOMA_FORCE_", "HOLOSOMA_DEPTH_"))},
    }


class DeploymentAudit:
    def __init__(self, directory, metadata, *, every=1, limit=6000, queue_size=16):
        if every < 1 or limit < 1 or queue_size < 1:
            raise ValueError("Audit sampling, limit and queue size must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.every, self.limit = every, limit
        self.seen = self.accepted = self.written = self.dropped = 0
        self.error = None
        self.closed = False
        self._limit_reported = False
        self.metadata = {"schema_version": 1, "sampling_every": every, "record_limit": limit, **metadata}
        (self.directory / "session.json").write_text(json.dumps(self.metadata, default=_json_default, indent=2))
        if metadata.get("role") == "policy" and "model_path" in metadata:
            target = self.directory / "model.onnx"
            shutil.copyfile(metadata["model_path"], target)
            if sha256_file(target) != metadata["model_sha256"]:
                raise ValueError("Checkpoint changed while archiving deployment evidence")
        self.queue = queue.Queue(maxsize=queue_size)
        self._publish_manifest()
        self.worker = threading.Thread(target=self._write_records, name="DeploymentAudit", daemon=True)
        self.worker.start()
        atexit.register(self.close)

    def _publish_manifest(self):
        payload = {"schema_version": 1, "seen": self.seen, "accepted": self.accepted,
                   "written": self.written, "dropped": self.dropped, "error": self.error,
                   "closed": self.closed, "limit_reached": self._limit_reported,
                   "complete_sampled_window": self.closed and not self.error and not self.dropped
                   and self.written == self.accepted}
        temporary = self.directory / "manifest.json.tmp"
        temporary.write_text(json.dumps(payload, default=_json_default, indent=2))
        temporary.replace(self.directory / "manifest.json")

    def record(self, arrays, metadata):
        """Only copies/enqueues on the caller; never waits for disk or queue space."""
        if self.closed:
            return
        sequence = self.seen
        self.seen += 1
        if sequence % self.every:
            return
        if self.accepted >= self.limit:
            if not self._limit_reported:
                print(f"[deployment_audit] capture limit reached: {self.directory}", file=sys.stderr)
                self._limit_reported = True
            return
        try:
            if self.error:
                self.dropped += 1
                return
            values = arrays() if callable(arrays) else arrays
            copied = {name: np.array(value, copy=True) for name, value in values.items()}
            if any(value.dtype.hasobject for value in copied.values()):
                raise ValueError("Object arrays are not permitted in evidence")
            detail = {"sequence": sequence, "recorded_at_monotonic": time.monotonic(),
                      "recorded_at_unix": time.time(), **metadata}
            copied["metadata_json"] = np.asarray(json.dumps(detail, default=_json_default))
            self.queue.put_nowait((sequence, copied))
            self.accepted += 1
        except queue.Full:
            self.dropped += 1
            if self.dropped == 1:
                print(f"[deployment_audit] INCOMPLETE: queue overflow at {self.directory}", file=sys.stderr)
        except Exception as exc:
            self.error = repr(exc)
            self.dropped += 1
            print(f"[deployment_audit] INCOMPLETE: {exc}", file=sys.stderr)

    def _write_records(self):
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                sequence, arrays = item
                if not self.error:
                    target = self.directory / f"{sequence:08d}.npz"
                    temporary = target.with_suffix(".tmp")
                    with temporary.open("wb") as stream:
                        np.savez_compressed(stream, **arrays)
                    temporary.replace(target)
                    self.written += 1
                self._publish_manifest()
            except Exception as exc:
                self.error = repr(exc)
                print(f"[deployment_audit] INCOMPLETE: {exc}", file=sys.stderr)
            finally:
                self.queue.task_done()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.queue.put(None)
        self.worker.join(timeout=10)
        if self.worker.is_alive():
            self.error = "Audit writer did not finish within 10 seconds"
            print(f"[deployment_audit] INCOMPLETE: {self.error}", file=sys.stderr)
            return
        try:
            self._publish_manifest()
        except Exception as exc:
            print(f"[deployment_audit] INCOMPLETE manifest: {exc}", file=sys.stderr)


def create_deployment_audit(role, metadata):
    root = os.environ.get("HOLOSOMA_DEPLOYMENT_AUDIT_DIR", "")
    if not root:
        return None
    every = int(os.environ.get(f"HOLOSOMA_AUDIT_{role.upper()}_EVERY", "6" if role == "depth" else "1"))
    limit = int(os.environ.get(f"HOLOSOMA_AUDIT_{role.upper()}_LIMIT", "1800" if role == "depth" else "6000"))
    directory = Path(root) / f"{role}_{time.time_ns()}_{os.getpid()}"
    recorder = DeploymentAudit(directory, {"role": role, "identity": _identity(), **metadata},
                               every=every, limit=limit)
    print(f"[deployment_audit] {role} evidence: {directory}")
    return recorder
