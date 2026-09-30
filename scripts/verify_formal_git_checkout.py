#!/usr/bin/env python3
"""Fail-closed verification for a formal-training Git checkout."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _run(root: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if check and proc.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed in {root}:\n{proc.stdout}")
    return proc.stdout.strip()


def _gitlinks(root: Path) -> dict[str, str]:
    rows = _run(root, "ls-files", "-s").splitlines()
    result: dict[str, str] = {}
    for row in rows:
        prefix, path = row.split("\t", 1)
        mode, sha, _stage = prefix.split()
        if mode == "160000":
            result[path] = sha
    return result


def _declared_submodules(root: Path) -> dict[str, str]:
    modules_path = root / ".gitmodules"
    if not modules_path.is_file():
        return {}
    raw = subprocess.run(
        ["git", "config", "-f", str(modules_path), "--get-regexp", r"^submodule\..*\.path$"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if raw.returncode not in (0, 1):
        raise RuntimeError(raw.stderr)
    result: dict[str, str] = {}
    for line in raw.stdout.splitlines():
        key, path = line.split(maxsplit=1)
        name = key[len("submodule.") : -len(".path")]
        url = subprocess.check_output(
            ["git", "config", "-f", str(modules_path), "--get", f"submodule.{name}.url"],
            text=True,
        ).strip()
        result[path] = url
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--remote-url", required=True)
    parser.add_argument("--remote-ref", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.source_root.resolve()
    if not (root / ".git").exists():
        raise RuntimeError(f"Formal source is not a Git checkout: {root}")
    actual_remote = _run(root, "remote", "get-url", "origin")
    if actual_remote != args.remote_url:
        raise RuntimeError(f"origin URL mismatch: expected={args.remote_url!r}, actual={actual_remote!r}")

    _run(root, "fetch", "--no-tags", "origin", args.remote_ref)
    actual_commit = _run(root, "rev-parse", "HEAD")
    actual_tree = _run(root, "rev-parse", "HEAD^{tree}")
    if actual_commit != args.commit:
        raise RuntimeError(f"HEAD mismatch: expected={args.commit}, actual={actual_commit}")
    if actual_tree != args.tree:
        raise RuntimeError(f"tree mismatch: expected={args.tree}, actual={actual_tree}")
    fetched_ref = _run(root, "rev-parse", "FETCH_HEAD")
    ancestor = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", args.commit, fetched_ref]
    )
    if ancestor.returncode != 0:
        raise RuntimeError(
            f"Contract commit {args.commit} is not reachable from origin/{args.remote_ref} ({fetched_ref})."
        )

    status = _run(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError(f"Formal source checkout is dirty:\n{status}")

    gitlinks = _gitlinks(root)
    declared = _declared_submodules(root)
    declared_results: dict[str, dict[str, str]] = {}
    for path, url in sorted(declared.items()):
        if path not in gitlinks:
            raise RuntimeError(f"Declared submodule {path!r} is not a gitlink in the superproject.")
        subroot = root / path
        if not (subroot / ".git").exists():
            raise RuntimeError(f"Declared submodule is not checked out: {path}")
        head = _run(subroot, "rev-parse", "HEAD")
        if head != gitlinks[path]:
            raise RuntimeError(
                f"Submodule SHA mismatch for {path}: expected={gitlinks[path]}, actual={head}"
            )
        sub_status = _run(subroot, "status", "--porcelain=v1", "--untracked-files=all")
        if sub_status:
            raise RuntimeError(f"Submodule {path} is dirty:\n{sub_status}")
        actual_url = _run(subroot, "remote", "get-url", "origin")
        if actual_url != url:
            raise RuntimeError(
                f"Submodule URL mismatch for {path}: expected={url!r}, actual={actual_url!r}"
            )
        declared_results[path] = {"sha": head, "remote_url": actual_url, "status": "clean"}

    unmapped = {path: sha for path, sha in gitlinks.items() if path not in declared}
    for path in unmapped:
        subroot = root / path
        if subroot.exists() and any(subroot.iterdir()):
            raise RuntimeError(
                f"Legacy unmapped gitlink {path!r} contains files and could affect execution."
            )

    payload = {
        "accepted": True,
        "source_root": str(root),
        "remote_url": actual_remote,
        "remote_ref": args.remote_ref,
        "fetched_ref_commit": fetched_ref,
        "commit_sha": actual_commit,
        "tree_sha": actual_tree,
        "tracked_diff_clean": True,
        "untracked_clean": True,
        "declared_submodules": declared_results,
        "legacy_unmapped_gitlinks": unmapped,
        "legacy_unmapped_gitlinks_inactive_and_empty": True,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
