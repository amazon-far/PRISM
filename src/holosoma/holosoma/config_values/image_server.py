"""Default image server configurations."""

import dataclasses

from holosoma.config_types.image_server import ImageServerConfig, ImageSaverConfig, ImageVisualizerConfig

# Default for sim2sim MuJoCo image streaming.
base = ImageServerConfig()
base_saver = ImageSaverConfig()
base_visualizer = ImageVisualizerConfig()

mujoco = dataclasses.replace(
    base,
    enable_gum_depth_prediction=False,
    enable_camera_depth_prediction=True,
    depth_source="depth",
    save_images=True,
)

# MuJoCo sim2sim preset for single D435i depth camera (far-tracking distillation).
# Note: near_clip, far_clip, resized_height, resized_width, crop_*, and frame_rate
# are auto-synced from CameraProps by sim_utils._sync_image_server_config().
mujoco_d435i = dataclasses.replace(
    base,
    enable_gum_depth_prediction=False,
    enable_camera_depth_prediction=True,
    enable_rgb=False,
    depth_source="depth",
    min_valid_depth=0.15,
    save_images=False,
    visualize_images=False,
    latency_frame=(3, 4),
    buffer_len=5,
)

# Debug-friendly ZED profile with visualization and both depth sources enabled.
real_verbose = dataclasses.replace(
    base,
    enable_gum_depth_prediction=True,
    enable_camera_depth_prediction=True,
    depth_source="depth",
    visualize_images=True,
    save_images=True,
)

# Default ZED inference profile.
real = dataclasses.replace(
    base,
    enable_gum_depth_prediction=False,
    enable_camera_depth_prediction=True,
    depth_source="depth",
    visualize_images=False,
)

# Add gum depth prediction,
real_enable_gum = dataclasses.replace(real,
    enable_gum_depth_prediction=True,
    depth_source="depth",
    visualize_images=False,
)

# Add gum depth prediction and use ZED depth as the source for policy.
real_depth_gum = dataclasses.replace(real,
    enable_gum_depth_prediction=True,
    depth_source="depth_gum",
    visualize_images=False,
)


# MuJoCo sim2sim D435i with GUM depth prediction as the policy depth source.
mujoco_depth_gum_d435i = dataclasses.replace(
    mujoco_d435i,
    enable_gum_depth_prediction=True,
    enable_rgb=True,
    depth_source="depth_gum",
    visualize_images=True,
    save_images=False,
)

# Real-robot D435i.
real_d435i = dataclasses.replace(
    base,
    enable_gum_depth_prediction=False,
    enable_camera_depth_prediction=True,
    depth_source="depth",
    near_clip=0.3,
    far_clip=3.0,
    resized_height=58,
    resized_width=87,
    frame_rate=30,
    crop_y_start=16,
    crop_x_start=32,
    crop_x_end=-32,
    save_images=True,
    visualize_images=False,
    camera_type="realsense",
    enable_rgb=False,
    latency_frame=(3, 3),  # 80-100ms RealSense latency, training uses (7,8)*20ms
    buffer_len=4,
)

# Debug-friendly D435i profile with visualization and both depth sources enabled.
real_verbose_d435i = dataclasses.replace(
    real_d435i,
    enable_gum_depth_prediction=True,
    visualize_images=True,
    save_images=True,
)

# D435i with GUM depth prediction enabled (policy still uses camera depth).
real_enable_gum_d435i = dataclasses.replace(
    real_d435i,
    enable_gum_depth_prediction=True,
    visualize_images=True,
)

# D435i with GUM depth prediction as the policy depth source.
real_depth_gum_d435i = dataclasses.replace(
    real_d435i,
    enable_gum_depth_prediction=True,
    depth_source="depth_gum",
    visualize_images=False,
)

# Real-robot ZED 2i (single front camera, far-tracking G1FlatZed2iConfig).
# Training: 240x135 inference -> 48x27 policy output, depth range [0.1, 2.0]m.
# No cropping needed: native 1280x720 -> 240x135 -> 48x27 all share 16:9 aspect ratio.
real_zed2i = dataclasses.replace(
    base,
    enable_gum_depth_prediction=False,
    enable_camera_depth_prediction=True,
    depth_source="depth",
    near_clip=0.1,
    far_clip=2.0,
    resized_height=58,
    resized_width=87,
    frame_rate=10,
    save_images=False,
    visualize_images=False,
    camera_type="zed",
    enable_rgb=False,
    latency_frame=0,
    buffer_len=1,
)

# Debug-friendly ZED 2i profile with visualization and saving.
real_verbose_zed2i = dataclasses.replace(
    real_zed2i,
    enable_gum_depth_prediction=True,
    enable_rgb=True,
    visualize_images=True,
    save_images=True,
)

# ZED 2i with GUM depth prediction enabled (policy still uses camera depth).
real_enable_gum_zed2i = dataclasses.replace(
    real_zed2i,
    enable_gum_depth_prediction=True,
    enable_rgb=True,
    visualize_images=False,
)

# ZED 2i with GUM depth prediction as the policy depth source.
real_depth_gum_zed2i = dataclasses.replace(
    real_zed2i,
    enable_gum_depth_prediction=True,
    enable_rgb=True,
    depth_source="depth_gum",
    visualize_images=False,
)

# MuJoCo sim2sim preset for single ZED 2i depth camera (far-tracking distillation).
mujoco_zed2i = dataclasses.replace(
    base,
    enable_gum_depth_prediction=False,
    enable_camera_depth_prediction=True,
    depth_source="depth",
    near_clip=0.1,
    far_clip=2.0,
    resized_height=58,
    resized_width=87,
    enable_rgb=False,
    save_images=False,
    visualize_images=True,
    latency_frame=0,
    buffer_len=1,
)

# Real robot, but the D435i is on another host: its IR stereo pair arrives over the
# network (stereo_relay_pub.py) and Fast-FoundationStereo runs here to produce the
# depth the policy consumes. Everything downstream of depth_gum matches real_d435i.
# Select FFS over GUM with HOLOSOMA_DEPTH_PREDICTOR=ffs.
remote_ffs_d435i = dataclasses.replace(
    real_d435i,
    camera_type="remote_stereo",
    enable_gum_depth_prediction=True,
    enable_rgb=True,
    depth_source="depth_gum",
    save_images=False,
    visualize_images=False,
    # real_d435i adds 3 frames (100 ms) on top of the RealSense's own ~80-100 ms
    # to land near the ~140-160 ms the policy was trained with. This path already
    # carries ~66 ms more on the laptop (frame arrival -> FFS -> shm, measured
    # median) plus network, so the artificial delay is cut to 1 frame (33 ms) to
    # end up in the same place rather than ~100 ms beyond it.
    latency_frame=(1, 1),
    buffer_len=2,
)

DEFAULTS = {
    "remote_ffs_d435i": remote_ffs_d435i,
    "mujoco": mujoco,
    "mujoco_d435i": mujoco_d435i,
    "mujoco_depth_gum_d435i": mujoco_depth_gum_d435i,
    "mujoco_zed2i": mujoco_zed2i,
    "real_verbose": real_verbose,
    "real": real,
    "real_enable_gum": real_enable_gum,
    "real_depth_gum": real_depth_gum,
    "real_d435i": real_d435i,
    "real_verbose_d435i": real_verbose_d435i,
    "real_enable_gum_d435i": real_enable_gum_d435i,
    "real_depth_gum_d435i": real_depth_gum_d435i,
    "real_zed2i": real_zed2i,
    "real_verbose_zed2i": real_verbose_zed2i,
    "real_enable_gum_zed2i": real_enable_gum_zed2i,
    "real_depth_gum_zed2i": real_depth_gum_zed2i,
}
