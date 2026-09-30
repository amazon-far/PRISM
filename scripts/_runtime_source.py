"""Resolve the installed HoloSoma revision and fetch its verification checkout."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


PACKAGES = {"holosoma": "holosoma", "holosoma-inference": "holosoma_inference"}


def dependency_source(requirements: Path) -> tuple[str, str]:
    """Read the shared Git URL/ref from the same requirements users install."""
    sources = {}
    for line in requirements.read_text().splitlines():
        name, separator, requirement = line.strip().partition(" @ git+")
        if not separator or name not in PACKAGES:
            continue
        source, separator, subdirectory = requirement.partition("#subdirectory=")
        url, ref_separator, ref = source.rpartition("@")
        if not separator or not ref_separator or not url or not ref:
            raise ValueError(f"Expected a Git URL and revision for {name} in {requirements}")
        if subdirectory != f"src/{PACKAGES[name]}":
            raise ValueError(f"Unexpected package subdirectory for {name}")
        if name in sources:
            raise ValueError(f"Duplicate dependency: {name}")
        sources[name] = (url, ref)
    if set(sources) != set(PACKAGES) or len(set(sources.values())) != 1:
        raise ValueError("Both HoloSoma packages must use the same Git repository and revision")
    return sources["holosoma"]


def installed_commit(remote: str) -> str:
    """Require a matching, immutable PEP 610 Git identity for both packages."""
    commits = set()
    for distribution_name, package_name in PACKAGES.items():
        distribution = importlib.metadata.distribution(distribution_name)
        direct = json.loads(distribution.read_text("direct_url.json") or "{}")
        vcs = direct.get("vcs_info", {})
        commit = vcs.get("commit_id", "")
        if (direct.get("url", "").removesuffix(".git") != remote.removesuffix(".git")
                or direct.get("subdirectory") != f"src/{package_name}"
                or vcs.get("vcs") != "git" or not re.fullmatch(r"[0-9a-f]{40}", commit)):
            raise ValueError(f"Install {distribution_name} with: pip install -r requirements.txt")
        commits.add(commit)
    if len(commits) != 1:
        raise ValueError("Installed HoloSoma packages use different commits; reinstall requirements.txt")
    return commits.pop()


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *arguments], check=True,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.strip()


def fetch_checkout(remote: str, ref: str, commit: str, cache_root: Path) -> Path:
    """Fetch code directly from Git on this node, without modifying reused trees."""
    repository_key = hashlib.sha256(remote.encode()).hexdigest()[:16]
    parent = cache_root / "prism" / "holosoma" / repository_key
    parent.mkdir(parents=True, exist_ok=True)
    checkout = parent / commit
    with (parent / f"{commit}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not checkout.exists():
            with tempfile.TemporaryDirectory(prefix="fetch-", dir=parent) as temporary:
                staging = Path(temporary)
                _git(staging, "init", "--quiet")
                _git(staging, "remote", "add", "origin", remote)
                _git(staging, "fetch", "--quiet", "--filter=blob:none", "--no-tags", "origin", ref)
                _git(staging, "merge-base", "--is-ancestor", commit, "FETCH_HEAD")
                _git(staging, "checkout", "--quiet", "--detach", commit)
                if (staging / ".gitmodules").is_file():
                    _git(staging, "submodule", "update", "--init", "--recursive")
                staging.rename(checkout)
    # The launcher runs the full clean-checkout and installed-file verification
    # on every use. An existing dirty checkout is rejected, never reset silently.
    return checkout


def resolve_runtime_source(requirements: Path, *, checkout: Path | None = None,
                           commit: str | None = None, ref: str | None = None) -> dict:
    remote, configured_ref = dependency_source(requirements)
    actual_commit = installed_commit(remote)
    if commit is not None and commit != actual_commit:
        raise ValueError("--runtime-commit differs from the installed HoloSoma packages")
    ref = ref or configured_ref
    if re.fullmatch(r"[0-9a-f]{40}", ref) and ref != actual_commit:
        raise ValueError("Installed runtime differs from the revision pinned in requirements.txt")
    if checkout is None:
        cache_root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))).expanduser()
        checkout = fetch_checkout(remote, ref, actual_commit, cache_root)
    return {"root": checkout.expanduser().resolve(), "remote": remote,
            "ref": ref, "commit": actual_commit}
