"""Configuration types for image server pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ImageSaverConfig:
    """Configuration for image saver."""

    image_root_dir: str = "image_server_images"
    """Root directory for saving images."""

    save_queue_maxsize: int = 0
    """Maximum queue size for image saving. 0 means unlimited."""

    save_workers: int = 2
    """Number of worker threads for parallel image saving."""


@dataclass(frozen=True)
class ImageVisualizerConfig:
    """Configuration for image visualization."""

    near_clip: float = 0.1
    """Near clipping plane for depth visualization."""

    far_clip: float = 2.0
    """Far clipping plane for depth visualization."""

    scale: float = 1.0
    """Uniform display scale factor applied to frames (0, 1]."""


@dataclass(frozen=True)
class GUMConfig:
    """Configuration for GUM depth predictor."""

    model_checkpoint: str = ""
    """Path to GUM model checkpoint (.pt file)."""

    root_folder: str = "."
    """Workspace/git root folder used to resolve relative GUM paths."""

    torch_ops_dir: str = ""
    """Directory containing torch operations .so files."""

    device: str = "cuda"
    """Device to run GUM model on ("cuda" or "cpu")."""

    depth_min: float = 0.2
    """Minimum depth value in meters for GUM prediction."""

    depth_max: float = 10.0
    """Maximum depth value in meters for GUM prediction."""

    target_h: int = 480
    """Target input height before GUM inference."""

    target_w: int = 768
    """Target input width (per-eye width) before GUM inference."""


@dataclass(frozen=True)
class ImageServerConfig:
    """Configuration for image server."""

    near_clip: float = 0.1
    far_clip: float = 2.0
    min_valid_depth: float | None = None

    resized_height: int = 27
    resized_width: int = 48

    enable_gum_depth_prediction: bool = False
    """If True, run depth prediction with GUM."""

    enable_camera_depth_prediction: bool = True
    """If True, use the camera's own depth stream (ZED or RealSense)."""

    enable_rgb: bool = True
    """If True, capture/render RGB frames. Disable to save rendering cost when
    RGB is not needed (e.g. when only camera depth is used for the policy).
    Must be True when enable_gum_depth_prediction is True (GUM requires RGB)."""

    depth_source: Literal["depth", "depth_gum"] = "depth"
    """Depth source to send to policy shared memory."""

    gum_config: GUMConfig = field(default_factory=GUMConfig)
    """Configuration for GUM."""

    visualize_images: bool = False
    """Enable image visualization."""

    image_visualizer_config: ImageVisualizerConfig = field(default_factory=ImageVisualizerConfig)
    """Configuration for image visualizer."""

    save_images: bool = True
    """Enable image saving."""

    image_saver_config: ImageSaverConfig = field(default_factory=ImageSaverConfig)
    """Configuration for image saver."""

    frame_rate: int = 10
    """Capture and send frequency in Hz."""

    latency_frame: int | tuple[int, int] = 0
    """Frame latency for depth images. Can be a fixed int or a (min, max) tuple for stochastic delay."""

    buffer_len: int = 1
    """Number of frames kept in the latency ring buffer. Must be > latency_frame
    (or > max of the range when latency_frame is a tuple)."""

    crop_y_start: int | None = None
    """Start row for cropping depth frames before resize. None means no crop."""

    crop_y_end: int | None = None
    """End row for cropping depth frames before resize. None means no crop."""

    crop_x_start: int | None = None
    """Start column for cropping depth frames before resize. None means no crop."""

    crop_x_end: int | None = None
    """End column for cropping depth frames before resize. None means no crop."""

    camera_type: Literal["zed", "realsense", "remote_stereo"] = "zed"
    """Camera backend. "remote_stereo" receives IR stereo pairs over the network
    from stereo_relay_pub.py running on the camera host, for running a learned
    depth predictor on a different machine than the camera."""
    """Camera backend to use when running the standalone image server."""

    def __post_init__(self):
        if self.depth_source == "depth_gum" and not self.enable_gum_depth_prediction:
            raise ValueError(
                "Invalid ImageServerConfig: depth_source='depth_gum' requires enable_gum_depth_prediction=True."
            )
        if self.depth_source == "depth" and not self.enable_camera_depth_prediction:
            raise ValueError(
                "Invalid ImageServerConfig: depth_source='depth' requires enable_camera_depth_prediction=True."
            )
        if self.enable_gum_depth_prediction and not self.enable_rgb and self.camera_type == "zed":
            raise ValueError(
                "Invalid ImageServerConfig: enable_gum_depth_prediction=True requires enable_rgb=True "
                "(GUM needs RGB images)."
            )
