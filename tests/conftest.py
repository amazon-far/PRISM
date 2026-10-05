"""Configure release-script imports and simulator-compatible test collection."""

from pathlib import Path
import sys

# Preserve root-level script imports when pytest is invoked via its console entry point.
repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Import torch safely before any isaacgym imports during test collection
from holosoma.utils.safe_torch_import import torch  # noqa: F401


def pytest_configure(config):
    """Register custom markers for pytest."""
    config.addinivalue_line(
        "markers", "isaacsim: marks tests as requiring Isaac Sim (deselect with '-m \"not isaacsim\"')"
    )
    config.addinivalue_line(
        "markers", "multi_gpu: marks tests as requiring multiple GPUs (deselect with '-m \"not multi_gpu\"')"
    )
