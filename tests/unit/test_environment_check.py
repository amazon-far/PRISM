"""Installation checks must not hide missing dependencies or inherited packages."""
from pathlib import Path
import json
from types import SimpleNamespace

import pytest

from scripts import check_environment as check


def distribution(name, version, requires=()):
    return SimpleNamespace(metadata={"Name": name}, version=version, requires=list(requires))


def test_only_exact_vendor_conflicts_are_allowed():
    kernel = distribution("isaacsim-kernel", "5.1.0.0", ["typing_extensions==4.12.2", "websockets==12.0"])
    packages = [kernel, distribution("typing-extensions", "4.15.0"), distribution("websockets", "16.0")]
    assert len(check.check_dependencies(packages)) == 2
    for bad in [distribution("typing-extensions", "4.16.0"), distribution("typing-extensions", "4.13.0")]:
        with pytest.raises(RuntimeError, match="Dependency check failed"):
            check.check_dependencies([kernel, bad, packages[2]])
    kernel.version = "5.2.0.0"
    with pytest.raises(RuntimeError, match="Dependency check failed"):
        check.check_dependencies(packages)


def test_missing_or_unrelated_incompatible_dependency_is_rejected():
    package = distribution("application", "1.0", ["numpy<2", "missing-package>=1; extra == 'optional'"])
    assert check.check_dependencies([package, distribution("numpy", "1.26.0")]) == []
    for dependencies in [[], [distribution("numpy", "2.2.6")]]:
        with pytest.raises(RuntimeError, match="numpy"):
            check.check_dependencies([package, *dependencies])


def test_system_site_packages_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(check.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(check.sys, "base_prefix", str(tmp_path / "base"))
    monkeypatch.setattr(check.sys, "version_info", (3, 11, 0))
    monkeypatch.setattr(check.platform, "system", lambda: "Linux")
    monkeypatch.setattr(check.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(check.platform, "libc_ver", lambda: ("glibc", "2.35"))
    config = tmp_path / "pyvenv.cfg"
    config.write_text("include-system-site-packages = false\n")
    check.check_platform()
    config.write_text("include-system-site-packages = true\n")
    with pytest.raises(RuntimeError, match="without --system-site-packages"):
        check.check_platform()


def test_python_or_platform_mismatch_is_reported(monkeypatch):
    monkeypatch.setattr(check.sys, "version_info", (3, 10, 0))
    with pytest.raises(RuntimeError, match="Python 3.11"):
        check.check_platform()
    monkeypatch.setattr(check.sys, "version_info", (3, 11, 0))
    monkeypatch.setattr(check.platform, "system", lambda: "Darwin")
    with pytest.raises(RuntimeError, match="Linux x86_64"):
        check.check_platform()


def test_simulator_exit_zero_without_completed_steps_is_rejected(monkeypatch):
    monkeypatch.setattr(check.subprocess, "run", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="without completing ten steps"):
        check.check_simulator()


def test_simulator_requires_child_completion_record(monkeypatch):
    def finish(command, **kwargs):
        Path(command[-1]).write_text(json.dumps({"steps": 10}))
    monkeypatch.setattr(check.subprocess, "run", finish)
    check.check_simulator()
