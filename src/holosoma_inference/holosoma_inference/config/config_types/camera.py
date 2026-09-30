"""Camera configuration types for holosoma_inference."""

from __future__ import annotations

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class CameraPose:
    """Configuration for a single camera pose.

    Defines the position and orientation of a camera relative to a robot link.
    """

    parent_link: str
    """Robot link to which the camera is attached."""

    camera_offset: tuple[float, float, float]
    """Camera position offset from the link in [x, y, z] meters."""

    camera_rotation: tuple[float, float, float]
    """Camera orientation as Euler angles [roll, pitch, yaw] in degrees."""


@dataclass(frozen=True)
class CameraProps:
    """Camera properties and settings.

    Defines the technical specifications and capture settings for cameras.
    """

    image_type: str
    """Type of image to capture (e.g., 'depth', 'rgb', 'rgbd')."""

    width: int
    """Original image width in pixels."""

    height: int
    """Original image height in pixels."""

    resized_width: int
    """Target width after resizing (if if_resize is True)."""

    resized_height: int
    """Target height after resizing (if if_resize is True)."""

    horizontal_fov: float
    """Horizontal field of view in degrees."""

    vertical_fov: float
    """Vertical field of view in degrees."""

    near_clip: float
    """Near clipping plane distance in meters."""

    far_clip: float
    """Far clipping plane distance in meters."""

    frame_rate: int
    """Capture frame rate in Hz."""

    image_show: bool = False
    """Whether to display captured images (for debugging)."""

    depth_delay: int = 1
    """Depth sensor delay in frames. Total delay = depth_delay / frame_rate [s]."""

    crop_y_start: int|None = None
    """Start index of the y-axis to crop."""

    crop_x_start: int|None = None
    """Start index of the x-axis to crop."""

    crop_y_end: int|None = None
    """End index of the y-axis to crop."""

    crop_x_end: int|None = None
    """End index of the x-axis to crop."""


@dataclass(frozen=True)
class CameraConfig:
    """Complete camera system configuration.

    Defines all cameras in the system with their poses and shared properties.

    Examples
    --------
    Dual depth camera setup:
    >>> camera_config = CameraConfig(
    ...     poses={
    ...         "cam_front_depth": CameraPose(
    ...             parent_link="robot/torso_link",
    ...             camera_offset=(0.1, 0.0, 0.1),
    ...             camera_rotation=(0.0, 75.0, 0.0)
    ...         ),
    ...         "cam_back_depth": CameraPose(
    ...             parent_link="robot/torso_link",
    ...             camera_offset=(-0.1, 0.0, 0.1),
    ...             camera_rotation=(0.0, 75.0, 180.0)
    ...         )
    ...     },
    ...     props=CameraProps(
    ...         image_type="depth",
    ...         width=240,
    ...         height=135,
    ...         # ... other properties
    ...     )
    ... )
    """

    poses: dict[str, CameraPose]
    """Dictionary mapping camera names to their pose configurations.

    Each key is a camera identifier (e.g., "cam_front_depth", "cam_back_depth"),
    and each value is a CameraPose defining where and how the camera is mounted.
    """

    props: CameraProps
    """Shared camera properties applied to all cameras.

    All cameras in the system use the same technical specifications
    (resolution, FOV, frame rate, etc.) as defined in this CameraProps instance.
    """
