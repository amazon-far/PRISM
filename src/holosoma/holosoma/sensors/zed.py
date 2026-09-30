import configparser
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class ZEDCameraConfig:
    """Configuration for ZED Camera initialization."""

    img_shape: tuple[int, int] = (720, 1280)
    """Image shape as (height, width)."""

    fps: int = 10
    """Frames per second."""

    serial_number: int = 36920975
    """Camera serial number."""

    resolution: Literal["2K", "FHD", "HD", "VGA"] = "HD"
    """Image resolution. Options: 2K, FHD, HD, VGA."""

    zed_settings_dir: str = "/usr/local/zed/settings"
    """Directory containing ZED camera settings files (`SN<serial>.conf`)."""

    depth_mode: Literal["NONE", "NEURAL", "NEURAL_LIGHT", "PERFORMANCE", "QUALITY", "ULTRA"] = "NEURAL"
    """Depth mode. Options: NONE, NEURAL, NEURAL_LIGHT, PERFORMANCE, QUALITY, ULTRA."""

    confidence_threshold: int = 100
    """Confidence threshold for depth measurement (0-100)."""

    positive_inf_depth_value: float = 2.0
    """How +inf should be mapped to in the depth image, in meters."""

    negative_inf_depth_value: float = 0.1
    """How -inf should be mapped to in the depth image, in meters."""

    nan_depth_value: float = 0.0
    """How nan should be mapped to in the depth image, in meters."""

class ZEDCamera:
    def __init__(self, config: ZEDCameraConfig):
        # Lazy import ZED SDK, in case using simulator-camera
        try:
            import pyzed.sl as sl
        except ImportError as exc:
            raise ImportError(
                "pyzed.sl is not available. Please install the ZED SDK Python API to use ZED cameras."
            ) from exc

        self.sl = sl
        self.config = config
        self._init_zed()

    def _init_zed(self):
        """Initialize ZED camera."""
        sl = self.sl
        self.zed = sl.Camera()

        self.rgb_mat_side_by_side = sl.Mat()
        self.depth_mat = sl.Mat()

        self.runtime_params = sl.RuntimeParameters()
        self.runtime_params.confidence_threshold = self.config.confidence_threshold

        self.init_params = sl.InitParameters()
        self.init_params.camera_fps = 60
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.set_from_serial_number(self.config.serial_number)
        self.init_params.camera_resolution = self._get_zed_resolution_enum(self.config.img_shape)
        self.init_params.sdk_verbose = 1
        self.init_params.depth_mode = self._get_depth_mode_enum()

        print(f"[ZED Camera] Init parameters: {self.init_params.depth_mode}")
        print(f"[ZED Camera] Resolution: {self.init_params.camera_resolution}")

        status = self.zed.open(self.init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {repr(status)}")

        self._compute_intrinsics_and_extrinsics()

        # ZED camera initialized
        print(f"[ZED Camera] Initialized successfully")

    def _compute_intrinsics_and_extrinsics(self):
        """Compute intrinsics and extrinsics from ZED camera."""

        cam_info = self.zed.get_camera_information()
        calib = cam_info.camera_configuration.calibration_parameters
        left_cam = calib.left_cam
        right_cam = calib.right_cam

        # use the cx, cy, fx, fy from the config to build the intrinsics
        intrinsics_left = np.array([
            [left_cam.fx, 0.0, left_cam.cx],
            [0.0, left_cam.fy, left_cam.cy],
            [0.0, 0.0, 1.0],
        ])
        intrinsics_right = np.array([
            [right_cam.fx, 0.0, right_cam.cx],
            [0.0, right_cam.fy, right_cam.cy],
            [0.0, 0.0, 1.0],
        ])
        intrinsics = np.stack([intrinsics_left, intrinsics_right], axis=0).astype(np.float32)

        # left extrinsics is the identity
        extrinsics_left = np.eye(4)
        # right extrinsics is the inverse of the left extrinsics
        extrinsics_right = np.eye(4)
        extrinsics_right[0, 3] = -calib.stereo_transform[0, 3]
        extrinsics = np.stack([extrinsics_left, extrinsics_right], axis=0).astype(np.float32)

        self.calibration: dict[str, np.ndarray] = {"intrinsics": intrinsics, "extrinsics": extrinsics}

        print(f"[ZED Camera] FOV: {left_cam.h_fov:.2f}° x {left_cam.v_fov:.2f}°")
        print("[ZED Camera] Intrinsics:")
        print(
            f"  left_cam : fx={left_cam.fx:.2f}, fy={left_cam.fy:.2f}, "
            f"cx={left_cam.cx:.2f}, cy={left_cam.cy:.2f}"
        )
        print(
            f"  right_cam: fx={right_cam.fx:.2f}, fy={right_cam.fy:.2f}, "
            f"cx={right_cam.cx:.2f}, cy={right_cam.cy:.2f}"
        )


    def _get_depth_mode_enum(self):
        """Convert depth_mode string from config to ZED SDK enum."""
        sl = self.sl
        depth_mode_map = {
            "NONE": sl.DEPTH_MODE.NONE,
            "NEURAL": sl.DEPTH_MODE.NEURAL,
            # "NEURAL_LIGHT": sl.DEPTH_MODE.NEURAL_LIGHT,  # Not available in SDK 4.2
            "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
            "QUALITY": sl.DEPTH_MODE.QUALITY,
            "ULTRA": sl.DEPTH_MODE.ULTRA,
        }
        return depth_mode_map.get(self.config.depth_mode, self.sl.DEPTH_MODE.NEURAL)

    def _get_zed_resolution_enum(self, img_shape):
        """Convert image shape to ZED resolution enum."""
        sl = self.sl
        height, width = img_shape[0], img_shape[1]

        if width == 1280 and height == 720:
            return sl.RESOLUTION.HD720
        if width == 1920 and height == 1080:
            return sl.RESOLUTION.HD1080
        if width == 2208 and height == 1242:
            return sl.RESOLUTION.HD2K
        if width == 672 and height == 376:
            return sl.RESOLUTION.VGA

        print(f"[ZED Camera] Warning: Resolution {width}x{height} not exactly matched, using HD720")
        return sl.RESOLUTION.HD720

    def _get_depth_data(self):
        """Get depth data from ZED camera, in meters."""
        sl = self.sl
        self.zed.retrieve_measure(self.depth_mat, sl.MEASURE.DEPTH)
        depth_data = self.depth_mat.get_data()
        depth_data = np.nan_to_num(
            depth_data,
            nan=self.config.nan_depth_value,
            posinf=self.config.positive_inf_depth_value,
            neginf=self.config.negative_inf_depth_value,
        )
        return depth_data

    def _get_rgb_data(self):
        """Get RGB data from ZED camera."""
        sl = self.sl
        self.zed.retrieve_image(self.rgb_mat_side_by_side, sl.VIEW.SIDE_BY_SIDE)
        image_data = self.rgb_mat_side_by_side.get_data()
        return image_data[:, :, :3]

    def capture(self):
        """Capture rgb and depth data from ZED camera."""
        sl = self.sl
        t_start = time.perf_counter()
        if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            t_received = time.time() * 1000  # system time in ms when frame arrived
            frame_ts = self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
            depth_data = self._get_depth_data() if self.config.depth_mode != "NONE" else None
            rgb_data = self._get_rgb_data()
            t_end = time.perf_counter()
            hw_latency_ms = t_received - frame_ts
            process_ms = (t_end - t_start) * 1000
            total_latency_ms = hw_latency_ms + process_ms
            return {"depth": depth_data, "rgb": rgb_data, "total_latency_ms": total_latency_ms}
        print("[ZED Camera] Grab error: failed to grab frame")
        return {"depth": None, "rgb": None, "total_latency_ms": None}

    def release(self):
        """Release ZED camera resources."""
        if self.zed.is_opened():
            self.zed.close()
            print("[ZED Camera] Released")

#########################################################
# ZED Cameras Wrapper
#########################################################

# Dual ZED setup (front + back) for the original robot
DUAL_ZED_CAMERA_CONFIGS: dict[str, ZEDCameraConfig] = {
    "front": ZEDCameraConfig(serial_number=35996713),
    "back": ZEDCameraConfig(serial_number=33082869),
}

# Single front ZED 2i for depth distillation (G1FlatZed2iConfig)
SINGLE_FRONT_ZED_CAMERA_CONFIGS: dict[str, ZEDCameraConfig] = {
    "front": ZEDCameraConfig(serial_number=35996713),
}

DEFAULT_ZED_CAMERA_CONFIGS: dict[str, ZEDCameraConfig] = SINGLE_FRONT_ZED_CAMERA_CONFIGS


@dataclass(frozen=True)
class ZedCamerasConfig:
    """Configuration for ZED Cameras."""

    terms: dict[str, ZEDCameraConfig] = field(default_factory=lambda: DEFAULT_ZED_CAMERA_CONFIGS.copy())


class ZedCamerasWrapper:
    def __init__(self, config: ZedCamerasConfig):
        self.cameras = {name: ZEDCamera(config.terms[name]) for name in config.terms.keys()}
        self.num_cameras = len(self.cameras)

    def get_frames(self):
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
