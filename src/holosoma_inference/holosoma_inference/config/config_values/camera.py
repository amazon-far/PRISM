"""Default camera configurations for holosoma_inference.

This module provides pre-configured camera setups for different
robot types and tasks, migrated from the original YAML configurations.
"""

from __future__ import annotations

from holosoma_inference.config.config_types.camera import CameraConfig, CameraPose, CameraProps

# =============================================================================
# Camera Configurations
# =============================================================================

# Default dual depth camera setup (migrated from camera_old.yaml)
dual_depth_cameras = CameraConfig(
    poses={
        "cam_front_depth": CameraPose(
            parent_link="robot/torso_link",
            camera_offset=(0.1, 0.0, 0.1),  # x, y, z [m]
            camera_rotation=(0.0, 75.0, 0.0),  # roll, pitch, yaw [deg]
        ),
        "cam_back_depth": CameraPose(
            parent_link="robot/torso_link",
            camera_offset=(-0.1, 0.0, 0.1),  # x, y, z [m]
            camera_rotation=(0.0, 75.0, 180.0),  # roll, pitch, yaw [deg]
        ),
    },
    props=CameraProps(
        image_type="depth",
        width=240,
        height=135,
        resized_width=48,
        resized_height=27,
        # ZED 2i - HD@720p FOV settings
        horizontal_fov=101.41,
        vertical_fov=69.00,
        # Alternative ZED 2i - HD@1080p FOV settings:
        # horizontal_fov=84.38,
        # vertical_fov=54.03,
        near_clip=0.1,
        far_clip=2.0,
        frame_rate=10,
        image_show=False,
        depth_delay=1,  # total delay = depth_delay / frame_rate [s]
    ),
)


# Single D435i depth camera setup (for far-tracking depth distillation)
single_d435i_depth = CameraConfig(
    poses={
        "cam_d435i_depth": CameraPose(
            parent_link="robot/torso_link",
            camera_offset=(0.01, 0.01, 0.44),  # x, y, z [m]
            # Equivalent to far-tracking mount (1, 27, 1) plus camera_pitch_deg=10.
            camera_rotation=(1.21741572, 36.99830943, 1.11565009),  # roll, pitch, yaw [deg]
        ),
    },
    props=CameraProps(
        image_type="depth",
        width=848,
        height=480,
        resized_width=87,
        resized_height=58,
        # D435i FOV settings
        horizontal_fov=89.5,
        vertical_fov=58.6,
        near_clip=0.3,
        far_clip=3.0,
        frame_rate=30,  # Training far_tracking_warp camera_fps.
        image_show=False,
        depth_delay=0,
        crop_y_start=16,
        crop_x_start=32,
        crop_x_end=-32,
    ),
)


# Single ZED 2i depth camera setup (for far-tracking G1FlatZed2iConfig distillation)
# Native: 1280x720, warp raycast: 240x135, policy resize: (58, 87)
# Training uses "resize": (58, 87) in distillation_env_cfg.py WarpObservationsCfg
single_zed2i_depth = CameraConfig(
    poses={
        "cam_front_depth": CameraPose(
            parent_link="robot/torso_link",
            camera_offset=(0.1, 0.0, 0.1),    # x, y, z [m] relative to torso_link
            camera_rotation=(0.0, 75.0, 0.0),  # warp convention, same as training G1FlatZed2iConfig
        ),
    },
    props=CameraProps(
        image_type="depth",
        width=240,
        height=135,
        resized_width=87,
        resized_height=58,
        horizontal_fov=101.41,
        vertical_fov=69.00,
        near_clip=0.1,
        far_clip=2.0,
        frame_rate=10,
        image_show=False,
        depth_delay=0,
    ),
)


# =============================================================================
# Default Configurations Dictionary
# =============================================================================

DEFAULTS = {
    "dual-depth": dual_depth_cameras,
    "single-d435i-depth": single_d435i_depth,
    "single-zed2i-depth": single_zed2i_depth,
}
"""Dictionary of all available camera configurations.

Keys use hyphen-case naming convention for CLI compatibility.
"""
