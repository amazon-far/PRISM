"""Top-level inference configuration types for holosoma_inference."""

from __future__ import annotations

from pydantic.dataclasses import dataclass

from .camera import CameraConfig
from .observation import ObservationConfig
from .robot import RobotConfig
from .task import TaskConfig


@dataclass(frozen=True)
class InferenceConfig:
    """Top-level configuration for policy inference.

    Combines robot, observation, task, and camera configurations
    for running policies on real robots or in simulation.
    """

    robot: RobotConfig
    """Robot hardware and control configuration."""

    observation: ObservationConfig
    """Observation space configuration."""

    task: TaskConfig
    """Task execution configuration."""

    camera: CameraConfig | None = None
    """Camera system configuration (optional)."""
