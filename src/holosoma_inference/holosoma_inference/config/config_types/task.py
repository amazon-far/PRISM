"""Task configuration types for holosoma_inference."""

from __future__ import annotations

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    """Task execution configuration for policy inference."""

    model_path: str | list[str]
    """Path to ONNX model(s). Supports local paths and wandb:// URIs. Required field."""

    rl_rate: float = 50
    """Policy inference rate in Hz."""

    policy_action_scale: float = 0.25
    """Scaling factor applied to policy actions."""

    use_phase: bool = True
    """Whether to use gait phase observations."""

    gait_period: float = 1.0
    """Gait cycle period in seconds."""

    domain_id: int = 0
    """DDS domain ID for communication."""

    interface: str = "lo"
    """Network interface name."""

    use_joystick: bool = False
    """Enable joystick control input."""

    joystick_type: str = "xbox"
    """Joystick type."""

    joystick_device: int = 0
    """Joystick device index."""

    use_sim_time: bool = False
    """Use synchronized simulation time for WBT policies."""

    use_sim_state: bool = False
    """Use MuJoCo simulator root state for sim rollout observations."""

    sim_state_port: int = 5557
    """ZMQ port used to receive MuJoCo simulator state."""

    prefer_sim_ref_from_sim_state: bool = False
    """Use simulator-measured reference body pose when sim state is available."""

    sim_control_port: int = 5559
    """ZMQ port used to send split sim2sim lowcmd/control requests."""

    use_zmq_lowcmd: bool = False
    """Send lowcmd over split sim-control ZMQ instead of Unitree DDS."""

    auto_start_policy: bool = False
    """Start policy actions immediately after initialization."""

    auto_start_motion_clip: bool = False
    """Start WBT motion clip immediately after policy start."""

    motion_file: str = ""
    """Optional motion file used by sparse-root WBT observations."""

    wandb_download_dir: str = "/tmp"
    """Directory for downloading W&B checkpoints."""

    # Deprecation candidates:
    desired_base_height: float = 0.75
    """Target base height in meters."""

    residual_upper_body_action: bool = False
    """Whether to use residual control for upper body."""

    use_ros: bool = False
    """Use ROS2 for rate limiting."""

    print_observations: bool = False
    """Print observation vectors for debugging."""

    policy_type: str = ""
    """Policy type override. When set, forces a specific policy class (e.g. 'blind_fall_recovery')."""
