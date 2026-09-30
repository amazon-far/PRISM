import json
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

import pytest

from scripts import _runtime_source as runtime


def requirements(path, url, ref="dev-prism-release"):
    path.write_text("".join(f"{distribution} @ git+{url}@{ref}#subdirectory=src/{package}\n"
                            for distribution, package in runtime.PACKAGES.items()))
    return path


def mock_install(monkeypatch, url, commit):
    metadata = {
        name: {"url": url, "subdirectory": f"src/{package}",
               "vcs_info": {"vcs": "git", "commit_id": commit}}
        for name, package in runtime.PACKAGES.items()
    }
    monkeypatch.setattr(runtime.importlib.metadata, "distribution", lambda name: SimpleNamespace(
        read_text=lambda _: json.dumps(metadata[name]),
    ))
    return metadata


def test_dependency_url_and_branch_are_read_from_requirements(tmp_path):
    path = requirements(tmp_path / "requirements.txt", "https://example.org/a/fork.git", "feature")
    assert runtime.dependency_source(path) == ("https://example.org/a/fork.git", "feature")


def test_dependency_revisions_must_match(tmp_path):
    path = requirements(tmp_path / "requirements.txt", "https://example.org/runtime.git")
    path.write_text(path.read_text().replace("@dev-prism-release", "@other", 1))
    with pytest.raises(ValueError, match="same Git"):
        runtime.dependency_source(path)


@pytest.mark.parametrize("mode", ["mixed_commits", "editable", "wrong_remote", "wrong_package"])
def test_invalid_installations_are_rejected(monkeypatch, mode):
    remote = "https://example.org/runtime.git"
    metadata = mock_install(monkeypatch, remote, "a" * 40)
    direct = metadata["holosoma-inference"]
    if mode == "mixed_commits":
        direct["vcs_info"]["commit_id"] = "b" * 40
    elif mode == "editable":
        direct.clear()
    elif mode == "wrong_remote":
        direct["url"] = "https://example.org/different.git"
    else:
        direct["subdirectory"] = "src/other"
    with pytest.raises(ValueError):
        runtime.installed_commit(remote)


def test_explicit_commit_cannot_override_installed_revision(tmp_path, monkeypatch):
    remote = "https://example.org/runtime.git"
    path = requirements(tmp_path / "requirements.txt", remote)
    mock_install(monkeypatch, remote, "a" * 40)
    with pytest.raises(ValueError, match="differs"):
        runtime.resolve_runtime_source(path, commit="b" * 40)


def test_fetches_exact_git_source_and_reuses_without_reset(tmp_path, monkeypatch):
    origin = tmp_path / "origin"
    origin.mkdir()
    runtime._git(origin, "init", "--quiet", "--initial-branch=dev-prism-release")
    (origin / "runtime.py").write_text("VALUE = 1\n")
    runtime._git(origin, "add", "runtime.py")
    runtime._git(origin, "-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "--quiet", "-m", "Initial")
    commit = runtime._git(origin, "rev-parse", "HEAD")
    path = requirements(tmp_path / "requirements.txt", origin.as_uri())
    mock_install(monkeypatch, origin.as_uri(), commit)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    result = runtime.resolve_runtime_source(path)
    checkout = result["root"]
    assert runtime._git(checkout, "rev-parse", "HEAD") == commit
    assert runtime._git(checkout, "remote", "get-url", "origin") == origin.as_uri()
    assert runtime._git(checkout, "status", "--porcelain") == ""
    (checkout / "runtime.py").write_text("VALUE = 2\n")
    again = runtime.resolve_runtime_source(path)
    assert again["root"] == checkout
    assert (checkout / "runtime.py").read_text() == "VALUE = 2\n"
    verifier = Path(__file__).resolve().parents[2] / "scripts/verify_formal_git_checkout.py"
    result = subprocess.run([sys.executable, str(verifier), "--source-root", str(checkout),
                             "--remote-url", origin.as_uri(), "--remote-ref", "dev-prism-release",
                             "--commit", commit, "--tree", runtime._git(checkout, "rev-parse", "HEAD^{tree}")],
                            text=True, capture_output=True)
    assert result.returncode != 0 and "dirty" in result.stderr
