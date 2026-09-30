import time
import os
import signal
import threading
import random
from queue import Queue
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TypedDict
import cv2
import numpy as np
from multiprocessing import resource_tracker, shared_memory
import tyro
from typing_extensions import Annotated, NotRequired
import holosoma.config_values.image_server
from holosoma.utils.rate import RateLimiter
from datetime import datetime
from holosoma.config_types.image_server import (
    ImageSaverConfig,
    ImageServerConfig,
    ImageVisualizerConfig,
)
from holosoma.sensors.zed import ZedCamerasConfig, ZedCamerasWrapper
from holosoma.sensors.realsense import RealSenseCameraConfig, RealSenseCamerasConfig, RealSenseCamerasWrapper
from holosoma.sensors.utils import _prepare_depth_for_visualization


class FrameBundle(TypedDict):
    rgb: NotRequired[dict[str, np.ndarray | None]]
    depth: NotRequired[dict[str, np.ndarray | None]]
    depth_gum: NotRequired[dict[str, np.ndarray | None]]
    calibration: NotRequired[dict[str, dict[str, np.ndarray]]]


ImageServerCliConfig = Annotated[
    ImageServerConfig,
    tyro.conf.arg(
        constructor=tyro.extras.subcommand_type_from_defaults(
            holosoma.config_values.image_server.DEFAULTS
        )
    ),
]


class TimeProfiler:
    """Profiler for tracking time statistics."""

    def __init__(self):
        self.times = []
        self.count = 0
        self.total_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0

    def record(self, elapsed_ms: float):
        """Record a timing measurement."""
        self.times.append(elapsed_ms)
        self.count += 1
        self.total_time += elapsed_ms
        self.min_time = min(self.min_time, elapsed_ms)
        self.max_time = max(self.max_time, elapsed_ms)

    @contextmanager
    def measure(self):
        """Context manager for measuring and recording elapsed time."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.record(elapsed_ms)

    def get_stats(self) -> dict[str, float]:
        """Get statistics about recorded timings."""
        if self.count == 0:
            return {}
        avg_time = self.total_time / self.count
        return {
            "count": self.count,
            "avg_ms": avg_time,
            "min_ms": self.min_time,
            "max_ms": self.max_time,
            "total_ms": self.total_time,
        }

    def reset(self):
        """Reset all statistics."""
        self.times.clear()
        self.count = 0
        self.total_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0


class ImageSaver:
    """Handles image saving logic with background worker threads."""

    def __init__(self, config: ImageSaverConfig, near_clip: float, far_clip: float):
        """Initialize ImageSaver.

        Args:
            config: Configuration for image saving
            near_clip: Near clipping plane for depth visualization
            far_clip: Far clipping plane for depth visualization
        """
        self.config = config
        self.image_root_dir = config.image_root_dir
        self.save_queue_maxsize = config.save_queue_maxsize
        self.save_workers = config.save_workers
        self.near_clip = near_clip
        self.far_clip = far_clip

        # Initialize save directory
        self._init_save_images_dir()

        # Initialize profiler
        self.profiler = TimeProfiler()

        # Initialize queue
        self.save_queue = Queue(maxsize=self.save_queue_maxsize if self.save_queue_maxsize > 0 else 0)
        self.save_queue_dropped_count = 0

        # Directory management
        self.created_camera_channel_dirs: set[tuple[str, str]] = set()
        self._dirs_lock = threading.Lock()

        # Start worker threads
        self.save_threads = []
        for i in range(self.save_workers):
            thread = threading.Thread(target=self._save_images_worker, daemon=True, name=f"SaveWorker-{i}")
            thread.start()
            self.save_threads.append(thread)

    def _init_save_images_dir(self):
        """Create a subdirectory for each session with timestamp."""
        session_dir = Path(self.image_root_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_dir.mkdir(exist_ok=True, parents=True)
        self.save_images_dir = session_dir
        print(f"[Image Saver] Initialized save images directory: {self.save_images_dir}")

    def _ensure_camera_dirs(self, names: list[str], channels: list[str]):
        """Pre-create directories for all channels and cameras."""
        with self._dirs_lock:
            for cam_name in names:
                for channel in channels:
                    camera_channel = (cam_name, channel)
                    if camera_channel in self.created_camera_channel_dirs:
                        continue
                    (self.save_images_dir / cam_name / channel).mkdir(exist_ok=True, parents=True)
                    self.created_camera_channel_dirs.add(camera_channel)

    def save_calibration(self, calibration_by_camera: dict[str, dict[str, np.ndarray]]):
        """Save camera intrinsics/extrinsics under each camera calibration directory."""
        if not calibration_by_camera:
            return

        for cam_name, calibration in calibration_by_camera.items():
            calibration_dir = self.save_images_dir / cam_name / "calibration"
            calibration_dir.mkdir(exist_ok=True, parents=True)

            intrinsics = calibration.get("intrinsics")
            extrinsics = calibration.get("extrinsics")

            if intrinsics is not None:
                np.save(calibration_dir / "intrinsics.npy", intrinsics)
            if extrinsics is not None:
                np.save(calibration_dir / "extrinsics.npy", extrinsics)

        print(f"[Image Saver] Saved calibration for cameras: {list(calibration_by_camera.keys())}")

    def _save_images_worker(self):
        """Background thread worker that saves images from the queue."""
        while True:
            try:
                item = self.save_queue.get()
                if item is None:  # Sentinel to stop
                    break

                raw_images, step_count, timestamp = item
                channels = [channel for channel in ("rgb", "depth", "depth_gum") if raw_images.get(channel)]
                # Derive camera names from the first available channel
                names = []
                for ch in channels:
                    names = list(raw_images[ch].keys())
                    if names:
                        break
                self._ensure_camera_dirs(names, channels)

                for cam_name in names:
                    for channel in channels:
                        channel_frames = raw_images.get(channel, {})
                        frame = channel_frames.get(cam_name)
                        if frame is None:
                            continue

                        save_path = self.save_images_dir / cam_name / channel / f"{cam_name}_{step_count}_{timestamp}.png"
                        frame_for_png = (
                            _prepare_depth_for_visualization(
                                frame,
                                near_clip=self.near_clip,
                                far_clip=self.far_clip,
                            )
                            if channel in {"depth", "depth_gum"}
                            else frame
                        )
                        cv2.imwrite(str(save_path), frame_for_png)

                self.save_queue.task_done()
            except Exception as e:
                print(f"[Image Saver] Error in save_images_worker: {e}")

    def save(self, raw_images: FrameBundle, step_count: int):
        """Enqueue images for background saving. Blocks if queue is full to avoid dropping frames.

        Args:
            raw_images: Dictionary with optional image channels
            step_count: Step count for naming files
        """
        with self.profiler.measure():
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            images_copy: FrameBundle = {}
            for channel in ("rgb", "depth", "depth_gum"):
                channel_frames = raw_images.get(channel)
                if not channel_frames:
                    continue
                images_copy[channel] = {
                    name: (frame.copy() if frame is not None else None)
                    for name, frame in channel_frames.items()
                }

            self.save_queue.put((images_copy, step_count, timestamp))

    def get_stats(self) -> dict[str, float]:
        """Get statistics about image saving operations."""
        stats = self.profiler.get_stats()
        stats["queue_size"] = self.save_queue.qsize()
        return stats

    def shutdown(self):
        """Shutdown worker threads gracefully."""
        # Send sentinel to each worker thread
        for _ in self.save_threads:
            self.save_queue.put(None)

        # Wait for threads to finish
        for thread in self.save_threads:
            thread.join()


class ImageVisualizer:
    """Handles image visualization logic."""

    def __init__(self, config: ImageVisualizerConfig):
        """Initialize ImageVisualizer.

        Args:
            config: Visualization window sizing configuration
        """
        self.near_clip = config.near_clip
        self.far_clip = config.far_clip
        self.scale = config.scale

    def _display_frame(self, name: str, frame: np.ndarray) -> bool:
        """Display a frame in a window.

        Args:
            name: Window name
            frame: Image frame to display

        Returns:
            True if 'q' key was pressed, False otherwise
        """
        if self.scale < 1.0:
            h, w = frame.shape[:2]
            frame = cv2.resize(
                frame,
                (max(1, int(w * self.scale)), max(1, int(h * self.scale))),
                interpolation=cv2.INTER_CUBIC,
            )

        # window_name = f"Image Server Stream {name}"
        # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # cv2.imshow(window_name, frame)
        cv2.imshow(f"Image Server Stream {name}", frame)
        return cv2.waitKey(1) & 0xFF == ord("q")

    def _depth_or_placeholder(
        self,
        depth: np.ndarray | None,
        height: int,
        width: int,
        label: str,
    ) -> np.ndarray:
        """Return depth visualization or a labeled placeholder when depth is unavailable."""
        if depth is None:
            vis = np.zeros((height, width), dtype=np.uint8)
            label = f"{label} not predicted"
        else:
            vis = _prepare_depth_for_visualization(
                depth,
                near_clip=self.near_clip,
                far_clip=self.far_clip,
            )

        cv2.putText(
            vis,
            label,
            (10, min(height - 10, 40)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            255,
            2,
            cv2.LINE_AA,
        )
        return vis

    def visualize(self, raw_image: FrameBundle):
        """Visualize depth and RGB images.

        Args:
            raw_image: Dictionary with optional image channels
        """
        rgb_by_camera = raw_image.get("rgb", {})
        names = []
        for channel in ("rgb", "depth", "depth_gum"):
            names.extend(raw_image.get(channel, {}).keys())
        names = list(dict.fromkeys(names))
        if not names:
            return

        depth_channels = [
            channel for channel in ("depth", "depth_gum") if raw_image.get(channel)
        ]
        combined_depth_rows = []
        for cam_name in names:
            reference_frame = rgb_by_camera.get(cam_name)
            if reference_frame is None:
                for channel in depth_channels:
                    reference_frame = raw_image.get(channel, {}).get(cam_name)
                    if reference_frame is not None:
                        break
            if reference_frame is None:
                continue
            h, w = reference_frame.shape[:2]

            depth_tiles = [
                self._depth_or_placeholder(
                    raw_image.get(channel, {}).get(cam_name),
                    h,
                    w,
                    f"{cam_name}: {channel}",
                )
                for channel in depth_channels
            ]
            if depth_tiles:
                combined_depth_rows.append(np.concatenate(depth_tiles, axis=1))

        if combined_depth_rows:
            concatenated_depth = np.concatenate(combined_depth_rows, axis=0)
            self._display_frame("depth", concatenated_depth)

        rgb_frames = [rgb_by_camera[name] for name in names if rgb_by_camera.get(name) is not None]
        if not rgb_frames:
            return

        concatenated_rgb = np.concatenate(rgb_frames, axis=0)
        self._display_frame("rgb", concatenated_rgb)


class ImageServer:
    def __init__(self, camera_wrapper: ZedCamerasWrapper | RealSenseCamerasWrapper, cfg: ImageServerConfig):
        self.cfg: ImageServerConfig = cfg
        self._stop_event = threading.Event()
        self._closed = False

        # Initialize camera wrapper
        self.camera_wrapper: ZedCamerasWrapper | RealSenseCamerasWrapper = camera_wrapper
        self._deployment_audit = None
        if os.environ.get("HOLOSOMA_DEPLOYMENT_AUDIT_DIR") and cfg.camera_type in ("realsense", "remote_stereo"):
            from holosoma.utils.deployment_audit import create_deployment_audit

            self._deployment_audit = create_deployment_audit("depth", {
                "image_server_config": cfg,
                "cameras": {name: {"config": camera.config, "calibration": camera.calibration,
                                    "depth_scale": camera.depth_scale}
                            for name, camera in camera_wrapper.cameras.items()},
                "training_depth_profile": None,
            })

        # Initialize shared memory
        self._init_shared_memory()

        # GUM is optional and is not needed by the RealSense depth server.
        if self.cfg.enable_gum_depth_prediction:
            import torch

            # Fast-FoundationStereo is an opt-in swap for GUM; it produces the same
            # `depth_gum` channel, so nothing downstream of the image server changes.
            if os.environ.get("HOLOSOMA_DEPTH_PREDICTOR", "gum").lower() == "ffs":
                from holosoma.models.ffs.infer import FastFoundationStereo, FFSConfig

                self.gum = FastFoundationStereo(cfg=FFSConfig(), dtype=torch.bfloat16)
            else:
                raise ValueError("GUM is not bundled. Set HOLOSOMA_DEPTH_PREDICTOR=ffs for learned depth.")
        else:
            self.gum = None

        # Latency buffer config (matches old sim2sim ring-buffer approach)
        if isinstance(self.cfg.latency_frame, (tuple, list)) and len(self.cfg.latency_frame) == 2:
            self.latency_frame_range = self.cfg.latency_frame
            self.latency_frame = None
        else:
            self.latency_frame_range = None
            self.latency_frame = self.cfg.latency_frame
        self.buffer_len = self.cfg.buffer_len

        if self.latency_frame_range is not None:
            assert self.latency_frame_range[1] < self.buffer_len, \
                f"Max latency frame ({self.latency_frame_range[1]}) must be less than buffer length ({self.buffer_len})"
        elif self.latency_frame is not None and self.latency_frame > 0:
            assert self.latency_frame < self.buffer_len, \
                f"Latency frame ({self.latency_frame}) must be less than buffer length ({self.buffer_len})"

        # Pre-filled ring buffer: [buffer_len, num_cameras, 1, H, W]
        self._frame_buffer = np.zeros(
            (self.buffer_len, self.camera_wrapper.num_cameras, 1, self.cfg.resized_height, self.cfg.resized_width),
            dtype=np.float32,
        )

        # Initialize profilers for capturing and depth prediction
        self.capture_profiler = TimeProfiler()
        self.hw_latency_profiler = TimeProfiler()
        if self.gum:
            self.gum_profiler = TimeProfiler()

        # Initialize image saver and visualizer, for debugging and visualization
        if self.cfg.save_images:
            self.image_saver = ImageSaver(
                self.cfg.image_saver_config,
                near_clip=self.cfg.near_clip,
                far_clip=self.cfg.far_clip,
            )
            self._save_camera_calibration()
        else:
            self.image_saver = None

        if self.cfg.visualize_images:
            self.image_visualizer = ImageVisualizer(
                config=self.cfg.image_visualizer_config,
            )
        else:
            self.image_visualizer = None

    def _save_camera_calibration(self):
        """Save camera calibration right after ImageSaver initialization."""
        cameras = getattr(self.camera_wrapper, "cameras", None)
        if cameras is None:
            return

        calibration_by_camera: dict[str, dict[str, np.ndarray]] = {}
        for cam_name, camera in cameras.items():
            calibration = getattr(camera, "calibration", None)
            if calibration is None:
                continue
            calibration_by_camera[cam_name] = calibration

        if not calibration_by_camera:
            print("[Image Server] No camera calibration found to save.")
            return

        self.image_saver.save_calibration(calibration_by_camera)

    def _init_shared_memory(self):
        img_shm_name = "depth_img_shm"
        self._owns_image_shm = False

        # Initialize shared memory
        expected_shape = [self.camera_wrapper.num_cameras, 1, self.cfg.resized_height, self.cfg.resized_width]
        dtype = np.float32

        try:
            memory_size = np.prod(expected_shape) * np.dtype(dtype).itemsize
            # Try to create new shared memory, if it exists, connect to existing one
            try:
                self.image_shm = shared_memory.SharedMemory(create=True, size=memory_size, name=img_shm_name)
                self._owns_image_shm = True
                print(f"[Image Server] Created new shared memory: {img_shm_name}")
            except FileExistsError:
                self.image_shm = shared_memory.SharedMemory(name=img_shm_name)
                self._owns_image_shm = True
                print(f"[Image Server] Connected to existing shared memory: {img_shm_name}")

            self.img_array = np.ndarray(expected_shape, dtype=dtype, buffer=self.image_shm.buf)
            print(f"[Image Server] Shared memory: shape={expected_shape}, dtype={dtype}")
        except Exception as e:
            print(f"[Image Server] Failed to initialize shared memory: {e}")
            raise

        print("ImageServer initialized")

    def request_stop(self) -> None:
        self._stop_event.set()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.request_stop()

        if self._deployment_audit is not None:
            self._deployment_audit.close()

        for camera in getattr(self.camera_wrapper, "cameras", {}).values():
            release = getattr(camera, "release", None)
            if release is not None:
                try:
                    release()
                except Exception as exc:
                    print(f"[Image Server] Camera release warning: {exc}")

        self.image_shm.close()
        if self._owns_image_shm:
            try:
                self.image_shm.unlink()
            except FileNotFoundError:
                # A legacy client may already have unlinked this owner segment.
                resource_tracker.unregister(self.image_shm._name, "shared_memory")
        print("[Image Server] Shutdown complete")

    def _resize_clip_expand_transpose(self, frame):
        # crop
        if any(v is not None for v in (self.cfg.crop_y_start, self.cfg.crop_y_end,
                                        self.cfg.crop_x_start, self.cfg.crop_x_end)):
            frame = frame[self.cfg.crop_y_start:self.cfg.crop_y_end,
                          self.cfg.crop_x_start:self.cfg.crop_x_end]

        # resize
        frame = cv2.resize(frame, (self.cfg.resized_width, self.cfg.resized_height), cv2.INTER_CUBIC)

        # Match distillation: clamp first, then apply the optional min-valid-depth mask.
        frame = np.clip(frame, self.cfg.near_clip, self.cfg.far_clip)
        min_valid_depth = self.cfg.near_clip if self.cfg.min_valid_depth is None else self.cfg.min_valid_depth
        frame[frame < min_valid_depth] = self.cfg.far_clip
        # scale to [-0.5, 0.5] range
        frame = (frame - self.cfg.near_clip) / (self.cfg.far_clip - self.cfg.near_clip) - 0.5

        # [H, W] -> [1, H, W]
        frame = np.expand_dims(frame, axis=0)
        return frame

    def _audit_arrays(self, all_frames: FrameBundle, processed: np.ndarray, published: np.ndarray) -> dict[str, np.ndarray]:
        """Arrays for one deployment-audit record.

        raw_depth is the metric depth the policy frame was derived from,
        processed_depth the crop/resize/clip result, published_depth the delayed
        frame actually written to shared memory. When a stereo pair is present
        (learned depth predictors), its grey planes go in too, so the depth can
        be regenerated offline with other settings.
        """
        arrays = {
            "raw_depth": np.stack(list(all_frames[self.cfg.depth_source].values())),
            "processed_depth": processed,
            "published_depth": published,
        }
        rgb = all_frames.get("rgb")
        if rgb:
            # Side-by-side (H, 2W, 3) with identical channels -> keep one plane.
            arrays["stereo_ir"] = np.stack([frame[..., 0] for frame in rgb.values()])
        calibration = all_frames.get("calibration")
        if calibration:
            # Per record rather than in session.json: for a network camera the
            # calibration only arrives with the first frame, after the session
            # metadata has already been written. 50 floats, so it costs nothing.
            arrays["intrinsics"] = np.stack([c["intrinsics"] for c in calibration.values()])
            arrays["extrinsics"] = np.stack([c["extrinsics"] for c in calibration.values()])
        return arrays

    def _predict_gum_depth(self, frames: FrameBundle) -> dict[str, np.ndarray]:
        rgb_by_camera = frames.get("rgb", {})
        calibration_by_camera = frames.get("calibration")

        gum_depth: dict[str, np.ndarray] = {}
        for name, rgb in rgb_by_camera.items():
            calibration = calibration_by_camera.get(name)
            gum_depth[name] = self.gum.predict(
                side_by_side_image=rgb,
                camera_intrinsics=calibration["intrinsics"],
                camera_extrinsics=calibration["extrinsics"],
            )
        return gum_depth

    def _get_policy_depth_frames(self, frames: FrameBundle) -> dict[str, np.ndarray]:
        return frames.get(self.cfg.depth_source, {})


    def send_process(self):
        render_frequency = self.cfg.frame_rate
        rate_limiter = RateLimiter(render_frequency)

        step_count = 0
        while not self._stop_event.is_set():

            # 0. grab frames from cameras
            with self.capture_profiler.measure():
                all_frames: FrameBundle = dict(self.camera_wrapper.get_frames())

            # Record hardware latency if available
            total_latency_ms = all_frames.pop("total_latency_ms", None)
            if total_latency_ms is not None:
                self.hw_latency_profiler.record(total_latency_ms)

            # Strip RGB frames when RGB is disabled to avoid downstream use
            if not self.cfg.enable_rgb:
                all_frames.pop("rgb", None)

            if self.cfg.enable_gum_depth_prediction:
                with self.gum_profiler.measure():
                    all_frames["depth_gum"] = self._predict_gum_depth(all_frames)

            # 2. Process data to be sent to policy;
            depth_for_policy = all_frames[self.cfg.depth_source]
            depth_for_policy = [self._resize_clip_expand_transpose(frame) for frame in depth_for_policy.values()]
             # Concatenate frames before channel dimension (axis=0)
            # [C, H, W] -> [N, C, H, W] N is the number of cameras; front camera is first, back camera is second
            full_depth_for_policy = np.stack(depth_for_policy, axis=0)

            # 2.5. Shift ring buffer and select delayed frame
            self._frame_buffer[:-1] = self._frame_buffer[1:]
            self._frame_buffer[-1] = full_depth_for_policy.astype(np.float32)

            if self.latency_frame_range is not None:
                current_latency = random.randint(self.latency_frame_range[0], self.latency_frame_range[1])
            else:
                current_latency = self.latency_frame
            delayed_image = self._frame_buffer[-1 - current_latency]

            # 3. copy to shared memory for policy
            try:
                # print(f"[Image Server] Current step count: {step_count}, delayed step count: {delayed_step_count}")
                np.copyto(self.img_array, delayed_image)
                if self._deployment_audit is not None:
                    published_at = time.monotonic()
            except Exception as e:
                print(f"[Image Server] Failed to copy to shared memory: {e}")
                continue

            if self._deployment_audit is not None:
                self._deployment_audit.record(lambda: self._audit_arrays(all_frames, full_depth_for_policy, delayed_image), {
                    "capture_step": step_count, "published_at_monotonic": published_at,
                    "sampled_additional_delay_frames": current_latency,
                    "legacy_total_latency_ms": total_latency_ms,
                    "depth_shm_name": "depth_img_shm",
                    "capture": {name: camera.last_capture_metadata
                                for name, camera in self.camera_wrapper.cameras.items()},
                })

            # 4. save and visualize images
            if self.cfg.save_images:
                self.image_saver.save(
                    all_frames,
                    step_count,
                )

            if self.cfg.visualize_images:
                self.image_visualizer.visualize(
                    all_frames,
                )
                # Also show the processed depth that is actually sent to the policy
                # delayed_image: [N, 1, H, W] in [-0.5, 0.5]
                panels = [delayed_image[i, 0] for i in range(delayed_image.shape[0])]
                combined = np.concatenate(panels, axis=1)
                display = ((combined + 0.5) * 255.0).astype(np.uint8)
                cv2.imshow("Processed Depth (policy input)", display)
                cv2.waitKey(1)

            if step_count % 50 == 0:
                print(f"[Image Server] Rate limiter stats: {rate_limiter.get_stats()}")
                rate_limiter.reset()
                print(f"[Image Server] capture stats: {self.capture_profiler.get_stats()}")
                hw_stats = self.hw_latency_profiler.get_stats()
                if hw_stats:
                    print(f"[Image Server] hw total latency stats: {hw_stats}")
                if self.cfg.enable_gum_depth_prediction:
                    print(f"[Image Server] GUM prediction stats: {self.gum_profiler.get_stats()}")

            rate_limiter.sleep()
            step_count += 1

if __name__ == "__main__":
    # Parse command line arguments using subcommand presets from config_values.image_server.
    cfg = tyro.cli(ImageServerCliConfig, default=holosoma.config_values.image_server.real)

    if cfg.camera_type == "remote_stereo":
        # The camera is on another host; stereo_relay_pub.py streams the IR pair
        # here and the learned depth predictor runs locally.
        from holosoma.sensors.remote_stereo import RemoteStereoCamerasConfig, RemoteStereoCamerasWrapper

        camera_wrapper = RemoteStereoCamerasWrapper(RemoteStereoCamerasConfig())
    elif cfg.camera_type == "realsense":
        # Enable IR stereo streams when GUM depth prediction is requested;
        # GUM requires stereo image pairs and stereo calibration.
        if cfg.enable_gum_depth_prediction:
            rs_cam_cfg = RealSenseCameraConfig(enable_ir_stereo=True)
            rs_cfg = RealSenseCamerasConfig(terms={"d435i_depth": rs_cam_cfg})
        else:
            rs_cam_cfg = RealSenseCameraConfig(enable_rgb=cfg.enable_rgb)
            rs_cfg = RealSenseCamerasConfig(terms={"d435i_depth": rs_cam_cfg})
        camera_wrapper = RealSenseCamerasWrapper(rs_cfg)
    else:
        # Default: ZED cameras
        depth_mode = "NEURAL" if cfg.enable_camera_depth_prediction else "NONE"
        zed_cfg = ZedCamerasConfig(
            terms={
                name: replace(camera_cfg, depth_mode=depth_mode)
                for name, camera_cfg in ZedCamerasConfig().terms.items()
            }
        )
        camera_wrapper = ZedCamerasWrapper(zed_cfg)

    # Create image server with parsed config
    image_server = ImageServer(camera_wrapper, cfg)

    def request_stop(_signum, _frame):
        image_server.request_stop()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGHUP, request_stop)

    thread = threading.Thread(target=image_server.send_process)
    thread.start()
    try:
        # Python only runs signal handlers on the main thread, and a join() with
        # no timeout parks the main thread on a lock it never returns from when
        # the signal lands on the worker thread instead. SIGTERM (systemd,
        # `timeout`, a plain `kill`) then never reaches request_stop() and the
        # server keeps serving - and never unlinks depth_img_shm. Polling the
        # join keeps the main thread returning to the interpreter.
        while thread.is_alive():
            thread.join(timeout=0.5)
    except KeyboardInterrupt:
        print("[Image Server] Stop requested")
    finally:
        image_server.request_stop()
        thread.join(timeout=6.0)
        image_server.close()
