"""Intel RealSense D435i camera wrapper for real-robot depth capture.

Follows the same interface as ZedCamerasWrapper so it can be used
interchangeably with ImageServer.
"""

import time
import os
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RealSenseCameraConfig:
    """Configuration for a single RealSense camera."""

    serial_number: str = ""
    """Camera serial number. Empty string means use the first available device."""

    img_shape: tuple[int, int] = (480, 848)
    """Image shape as (height, width)."""

    fps: int = 30
    """Capture frame rate for both depth and color streams."""

    enable_rgb: bool = True
    """Enable the color (RGB) stream."""

    align_depth_to_color: bool = True
    """Align depth frames to the color sensor viewport."""

    positive_inf_depth_value: float = 3.0
    """How +inf should be mapped in the depth image, in meters."""

    negative_inf_depth_value: float = 0.1
    """How -inf should be mapped in the depth image, in meters."""

    nan_depth_value: float = 0.0
    """How nan / zero-depth should be mapped in the depth image, in meters."""

    enable_ir_stereo: bool = False
    """Enable left/right infrared stereo streams (for GUM depth prediction).
    When True, capture() returns a side-by-side (H, 2*W, 3) IR image as 'rgb'
    and calibration includes stereo intrinsics (2, 3, 3) and extrinsics (2, 4, 4)."""

    emitter_enabled: bool = False
    """Enable the IR emitter (dot projector). Set to False to turn it off,
    e.g. when using stereo IR images for GUM where the projected pattern
    can interfere with stereo matching."""


class RealSenseCamera:
    """Manages a single Intel RealSense D435i camera via pyrealsense2."""

    def __init__(self, config: RealSenseCameraConfig):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise ImportError(
                "pyrealsense2 is not available. "
                "Install it with: pip install pyrealsense2"
            ) from exc

        self.rs = rs
        self.config = config
        self._audit_capture_timestamps = bool(os.environ.get("HOLOSOMA_DEPLOYMENT_AUDIT_DIR"))
        self._init_camera()

    def _init_camera(self):
        rs = self.rs
        height, width = self.config.img_shape

        self.pipeline = rs.pipeline()
        rs_config = rs.config()

        if self.config.serial_number:
            rs_config.enable_device(self.config.serial_number)

        # Enable depth stream
        rs_config.enable_stream(
            rs.stream.depth, width, height, rs.format.z16, self.config.fps,
        )

        # Enable color stream
        if self.config.enable_rgb:
            rs_config.enable_stream(
                rs.stream.color, width, height, rs.format.bgr8, self.config.fps,
            )

        # Enable infrared stereo streams (left IR1 + right IR2) for GUM
        if self.config.enable_ir_stereo:
            rs_config.enable_stream(
                rs.stream.infrared, 1, width, height, rs.format.y8, self.config.fps,
            )
            rs_config.enable_stream(
                rs.stream.infrared, 2, width, height, rs.format.y8, self.config.fps,
            )

        profile = self.pipeline.start(rs_config)

        # Depth scale: converts raw uint16 depth values to meters
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # IR emitter control
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(
                rs.option.emitter_enabled, 1.0 if self.config.emitter_enabled else 0.0,
            )
            print(f"[RealSense] IR emitter: {'on' if self.config.emitter_enabled else 'off'}")

        # Align object (reusable across frames)
        if self.config.align_depth_to_color and self.config.enable_rgb:
            self.align = rs.align(rs.stream.color)
        else:
            self.align = None

        # Extract intrinsics
        self._compute_intrinsics(profile)

        print(
            f"[RealSense] Initialized: serial={self.config.serial_number or 'auto'}, "
            f"resolution={width}x{height}@{self.config.fps}fps, "
            f"depth_scale={self.depth_scale:.6f}"
        )

    def _compute_intrinsics(self, profile):
        """Extract intrinsics from the depth (or aligned color) stream profile.

        When ``enable_ir_stereo`` is True, builds stereo calibration with shapes
        ``(2, 3, 3)`` for intrinsics and ``(2, 4, 4)`` for extrinsics, matching
        the format that GUM expects (same as ZED stereo).
        """
        rs = self.rs

        if self.config.enable_ir_stereo:
            self._compute_stereo_ir_intrinsics(profile)
            return

        # When aligned to color, use the color stream intrinsics; otherwise depth.
        if self.align is not None:
            stream = rs.stream.color
        else:
            stream = rs.stream.depth

        video_profile = profile.get_stream(stream).as_video_stream_profile()
        intr = video_profile.get_intrinsics()

        intrinsics = np.array([
            [intr.fx, 0.0, intr.ppx],
            [0.0, intr.fy, intr.ppy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        # Single camera: shape (1, 3, 3) to match ZED's (num_cams, 3, 3) layout.
        extrinsics = np.eye(4, dtype=np.float32)

        self.calibration: dict[str, np.ndarray] = {
            "intrinsics": intrinsics[np.newaxis],
            "extrinsics": extrinsics[np.newaxis],
        }

        print(
            f"[RealSense] Intrinsics: fx={intr.fx:.2f}, fy={intr.fy:.2f}, "
            f"cx={intr.ppx:.2f}, cy={intr.ppy:.2f}"
        )

    def _compute_stereo_ir_intrinsics(self, profile):
        """Extract stereo IR calibration (intrinsics + extrinsics) for GUM."""
        rs = self.rs

        left_ir_profile = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        right_ir_profile = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()

        left_intr = left_ir_profile.get_intrinsics()
        right_intr = right_ir_profile.get_intrinsics()

        def _build_K(intr):
            return np.array([
                [intr.fx, 0.0, intr.ppx],
                [0.0, intr.fy, intr.ppy],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)

        # intrinsics shape (2, 3, 3): [left, right]
        intrinsics = np.stack([_build_K(left_intr), _build_K(right_intr)], axis=0)

        # Extrinsics: right-to-left transform from RealSense SDK
        rs_extr = right_ir_profile.get_extrinsics_to(left_ir_profile)
        R = np.array(rs_extr.rotation, dtype=np.float32).reshape(3, 3)
        t = np.array(rs_extr.translation, dtype=np.float32)
        t[0] *= -1.0

        # Left camera is identity (reference frame)
        left_ext = np.eye(4, dtype=np.float32)

        # Right camera extrinsics: right-to-left transform
        right_ext = np.eye(4, dtype=np.float32)
        right_ext[:3, :3] = R
        right_ext[:3, 3] = t

        # extrinsics shape (2, 4, 4): [left, right]
        extrinsics = np.stack([left_ext, right_ext], axis=0)

        self.calibration = {
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
        }

        print(
            f"[RealSense] Stereo IR intrinsics:\n"
            f"  Left  IR1: fx={left_intr.fx:.2f}, fy={left_intr.fy:.2f}, "
            f"cx={left_intr.ppx:.2f}, cy={left_intr.ppy:.2f}\n"
            f"  Right IR2: fx={right_intr.fx:.2f}, fy={right_intr.fy:.2f}, "
            f"cx={right_intr.ppx:.2f}, cy={right_intr.ppy:.2f}\n"
            f"  Baseline (tx): {t[0]:.6f} m"
        )

    def capture(self) -> dict:
        """Capture aligned depth and color frames.

        Returns
        -------
        dict
            ``{"depth": np.ndarray (H, W) float32 in meters,
               "rgb": np.ndarray (H, W, 3) uint8 BGR or None}``
        """
        t_start = time.perf_counter()
        frames = self.pipeline.wait_for_frames()
        t_received = time.time() * 1000  # system time in ms when frame arrived

        # Hardware latency: compare frame's global_time timestamp to system clock
        depth_frame_raw = frames.get_depth_frame()
        frame_ts = depth_frame_raw.get_timestamp() if depth_frame_raw else None
        ts_domain = depth_frame_raw.get_frame_timestamp_domain() if depth_frame_raw else None
        # Observational only: do not enable timestamps or change camera options.
        if self._audit_capture_timestamps:
            self.last_capture_metadata = {
                "frame_number": depth_frame_raw.get_frame_number() if depth_frame_raw else None,
                "sensor_timestamp_ms": frame_ts, "timestamp_domain": str(ts_domain),
                "received_unix_ms": t_received, "received_monotonic": time.monotonic(),
                "sensor_age_at_receive_ms": t_received - frame_ts
                if frame_ts is not None and ts_domain == self.rs.timestamp_domain.global_time else None,
            }

        if self.align is not None:
            frames = self.align.process(frames)

        # Depth
        depth_frame = frames.get_depth_frame()
        if depth_frame:
            depth_data = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
            # Replace zeros (invalid depth) with nan_depth_value
            depth_data[depth_data == 0.0] = self.config.nan_depth_value
            depth_data = np.nan_to_num(
                depth_data,
                nan=self.config.nan_depth_value,
                posinf=self.config.positive_inf_depth_value,
                neginf=self.config.negative_inf_depth_value,
            )
        else:
            depth_data = None

        # Stereo IR: side-by-side (H, 2*W, 3) for GUM
        if self.config.enable_ir_stereo:
            left_ir = frames.get_infrared_frame(1)
            right_ir = frames.get_infrared_frame(2)
            if left_ir and right_ir:
                left_arr = np.asanyarray(left_ir.get_data())   # (H, W) uint8
                right_arr = np.asanyarray(right_ir.get_data())  # (H, W) uint8
                # Convert grayscale to 3-channel and concatenate side-by-side
                left_rgb = np.stack([left_arr] * 3, axis=-1)   # (H, W, 3)
                right_rgb = np.stack([right_arr] * 3, axis=-1)  # (H, W, 3)
                rgb_data = np.concatenate([left_rgb, right_rgb], axis=1)  # (H, 2*W, 3)
            else:
                rgb_data = None
            t_end = time.perf_counter()
            total_latency_ms = self._compute_latency(frame_ts, ts_domain, t_received, t_start, t_end)
            return {"depth": depth_data, "rgb": rgb_data, "total_latency_ms": total_latency_ms}

        # Color
        rgb_data = None
        if self.config.enable_rgb:
            color_frame = frames.get_color_frame()
            if color_frame:
                rgb_data = np.asanyarray(color_frame.get_data())  # BGR uint8

        t_end = time.perf_counter()
        total_latency_ms = self._compute_latency(frame_ts, ts_domain, t_received, t_start, t_end)
        return {"depth": depth_data, "rgb": rgb_data, "total_latency_ms": total_latency_ms}

    def _compute_latency(
        self,
        frame_ts: float | None,
        ts_domain,
        t_received: float,
        t_start: float,
        t_end: float,
    ) -> float | None:
        """Compute capture latency breakdown.

        Returns the total latency in ms (hw + process) when the hardware
        timestamp is available, otherwise None.
        """
        rs = self.rs
        process_ms = (t_end - t_start) * 1000

        if frame_ts is not None and ts_domain == rs.timestamp_domain.global_time:
            hw_latency_ms = t_received - frame_ts
            return hw_latency_ms + process_ms
        return None

    def release(self):
        """Stop the RealSense pipeline."""
        self.pipeline.stop()
        print("[RealSense] Released")


# ───────────────────────────────────────────────────────
# RealSense Cameras Wrapper
# ───────────────────────────────────────────────────────

DEFAULT_REALSENSE_CAMERA_CONFIGS: dict[str, RealSenseCameraConfig] = {
    "d435i_depth": RealSenseCameraConfig(),
}


@dataclass(frozen=True)
class RealSenseCamerasConfig:
    """Configuration for one or more RealSense cameras."""

    terms: dict[str, RealSenseCameraConfig] = field(
        default_factory=lambda: DEFAULT_REALSENSE_CAMERA_CONFIGS.copy(),
    )


class RealSenseCamerasWrapper:
    """Wraps one or more RealSense cameras with the same interface as ZedCamerasWrapper."""

    def __init__(self, config: RealSenseCamerasConfig):
        self.cameras = {
            name: RealSenseCamera(cfg) for name, cfg in config.terms.items()
        }
        self.num_cameras = len(self.cameras)

    def get_frames(self) -> dict:
        """Capture frames from all cameras.

        Returns
        -------
        dict
            ``{"depth": {name: ndarray}, "rgb": {name: ndarray},
               "calibration": {name: {"intrinsics": ..., "extrinsics": ...}}}``
        """
        depth_data: dict[str, np.ndarray] = {}
        rgb_data: dict[str, np.ndarray] = {}
        calibration_data: dict[str, dict[str, np.ndarray]] = {}
        latency_values: list[float] = []
        for name, camera in self.cameras.items():
            frame = camera.capture()
            depth_data[name] = frame["depth"]
            rgb_data[name] = frame["rgb"]
            calibration_data[name] = camera.calibration
            if frame["total_latency_ms"] is not None:
                latency_values.append(frame["total_latency_ms"])
        result = {"depth": depth_data, "rgb": rgb_data, "calibration": calibration_data}
        if latency_values:
            result["total_latency_ms"] = sum(latency_values) / len(latency_values)
        return result
