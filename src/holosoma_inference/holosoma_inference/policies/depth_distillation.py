"""Depth distillation policy for far-tracking student models.

Loads two ONNX models:
- depth_backbone.onnx: depth image (1, H, W) -> depth latent (1, D)
- student.onnx: obs -> actions

The depth backbone processes depth images from shared memory into a latent
vector that is concatenated with proprioceptive observations before being
fed to the student network.
"""

from __future__ import annotations

import json
import os
from collections import deque
from multiprocessing import resource_tracker, shared_memory
import math

import numpy as np
import onnx
import onnxruntime
from loguru import logger

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.policies.base import BasePolicy
from holosoma_inference.policies.locomotion import LocomotionPolicy
from holosoma_inference.utils.math.quat import quat_from_angle_axis, quat_mul


class DepthDistillationPolicy(LocomotionPolicy):
    """Policy for far-tracking depth distillation student models.

    Uses a two-model ONNX architecture:
    - depth_backbone: CNN that converts depth images to latent vectors
    - student: MLP that takes [proprioceptive_obs, command, depth_latent] -> actions
    """
    # Joystick-to-velocity-command mapping.
    # Command codes per speed mode, matching the C++ MotionTrackingController.
    # speed_mode 0 = low, 1 = high, 2 = madmax
    CMD_CODES = {
        0: {"stand": 0, "forward": 1, "left_45": 2, "left_90": 3, "right_45": 4, "right_90": 5, "back": 11},
        1: {"stand": 0, "forward": 6, "left_45": 7, "left_90": 8, "right_45": 9, "right_90": 10, "back": 11},
        2: {"stand": 0, "forward": 12, "left_45": 13, "left_90": 13, "right_45": 14, "right_90": 14, "back": 11},
    }
    JOYSTICK_DEADZONE = 0.05

    def __init__(self, config: InferenceConfig):
        self.speed_mode_high = False
        self.speed_mode = 0  # 0=low, 1=high, 2=madmax
        self._stiff_hold_active = True
        self._damping_mode_active = False
        self._stiff_blend_count: int = 0
        self._stiff_blend_duration: int = 100
        self._stiff_blend_start_q: np.ndarray | None = None

        # Will be populated in _init_policy_components
        self.depth_backbone_session = None
        self.depth_backbone_input_name = None
        self.depth_backbone_output_name = None
        self.depth_latent_dim = None
        self.depth_image_shape = None  # (H, W) from backbone input

        # Velocity command: one-hot vector selecting the active command class.
        # During training this comes from the "command" obs group (vel_cmd in motion data).
        # During inference the user sets it via keyboard/joystick.
        self.velocity_command_dim = config.observation.obs_dims.get("velocity_command", 0)
        self.velocity_command = np.zeros((1, self.velocity_command_dim), dtype=np.float32)
        self.active_velocity_command_idx =  self.CMD_CODES[0]["stand"]  # default to standing
        self.set_velocity_command(self.active_velocity_command_idx)

        super().__init__(config)

        # Load stiff startup parameters from robot config
        if config.robot.stiff_startup_pos is not None:
            self._stiff_hold_q = np.array(config.robot.stiff_startup_pos, dtype=np.float32).reshape(1, -1)
        else:
            self._stiff_hold_q = np.array(config.robot.default_dof_angles, dtype=np.float32).reshape(1, -1)

        if config.robot.stiff_startup_kp is not None:
            self._stiff_hold_kp = np.array(config.robot.stiff_startup_kp, dtype=np.float32)
        else:
            raise ValueError("Robot config must specify stiff_startup_kp for DepthDistillationPolicy")

        if config.robot.stiff_startup_kd is not None:
            self._stiff_hold_kd = np.array(config.robot.stiff_startup_kd, dtype=np.float32)
        else:
            raise ValueError("Robot config must specify stiff_startup_kd for DepthDistillationPolicy")

        if self._stiff_hold_q.shape[1] != self.num_dofs:
            raise ValueError("Stiff startup pose dimension mismatch with robot DOFs")

        # Initialize depth shared memory client
        self._init_depth_shm()

        # Initialize depth frame buffer
        depth_history_len = config.observation.history_length_dict.get("depth_obs", 3)
        self.depth_frame_buffer = deque(maxlen=depth_history_len)

    def _init_depth_shm(self):
        """Initialize depth image client using shared memory."""
        camera_config = self.config.camera
        if camera_config is None:
            raise ValueError("DepthDistillationPolicy requires a camera configuration.")

        img_shape = [camera_config.props.resized_height, camera_config.props.resized_width]
        channels = 1  # depth is always 1 channel
        num_cameras = len(camera_config.poses)
        expected_shape = [num_cameras, channels, img_shape[0], img_shape[1]]

        self.depth_img_shm = shared_memory.SharedMemory(name="depth_img_shm")
        resource_tracker.unregister(self.depth_img_shm._name, "shared_memory")
        self.depth_img_array = np.ndarray(expected_shape, dtype=np.float32, buffer=self.depth_img_shm.buf)
        logger.info(f"[DepthDistillationPolicy] Depth SHM client initialized: shape={expected_shape}")

    def _init_policy_components(self, model_path, policy_action_scale, rl_rate):
        """Override to load two ONNX models: depth_backbone and student."""
        self.policy_action_scale = policy_action_scale
        self.rl_rate = rl_rate

        # Collect and resolve model paths
        self.model_paths = self._collect_model_paths(model_path)
        if len(self.model_paths) != 2:
            raise ValueError(
                f"DepthDistillationPolicy requires exactly 2 model paths "
                f"[depth_backbone.onnx, student.onnx], got {len(self.model_paths)}"
            )

        resolved_paths = []
        for path in self.model_paths:
            local_path = self._resolve_model_path(str(path))
            resolved_paths.append(local_path)
        self.model_paths = resolved_paths

        backbone_path = self.model_paths[0]
        student_path = self.model_paths[1]

        # Load depth backbone
        self._load_depth_backbone(backbone_path)

        # Load student model
        self._load_student_model(student_path)

        # TODO: Remove once onnx model is trained with holosoma.
        # Setup joint reordering (robot order <-> model's expected order)
        self._setup_joint_reordering()

        # Pre-compute waist joint indices for anchor body FK
        self._init_waist_joint_indices()

        # Initialize action buffers
        self.last_policy_action = np.zeros((1, self.num_dofs))
        self.scaled_policy_action = np.zeros((1, self.num_dofs))

        # Initialize policy states for multi-policy support
        self._policy_states = [self._capture_policy_state()]
        self.active_policy_index = 0
        self.active_model_path = student_path

        # Resolve control gains from ONNX metadata or config
        self._resolve_control_gains()

    def _load_depth_backbone(self, model_path: str):
        """Load the depth backbone ONNX model."""
        self.depth_backbone_session = onnxruntime.InferenceSession(model_path)

        inputs = self.depth_backbone_session.get_inputs()
        outputs = self.depth_backbone_session.get_outputs()

        self.depth_backbone_input_name = inputs[0].name   # "depth_image"
        self.depth_backbone_output_name = outputs[0].name  # "depth_latent"

        # Extract input shape to auto-detect (H, W)
        input_shape = inputs[0].shape  # e.g. [1, 3, 58, 87] or [1, 58, 87]
        if len(input_shape) == 4:
            # (batch, buffer_len, H, W)
            self.depth_image_shape = (input_shape[2], input_shape[3])
            self.depth_buffer_len = input_shape[1]
        elif len(input_shape) == 3:
            # (batch, H, W)
            self.depth_image_shape = (input_shape[1], input_shape[2])
            self.depth_buffer_len = 1
        else:
            raise ValueError(f"Unexpected depth backbone input shape: {input_shape}")

        # Extract output dim
        output_shape = outputs[0].shape  # e.g. [1, 32]
        self.depth_latent_dim = output_shape[-1]

        logger.info(
            f"[DepthDistillationPolicy] Depth backbone loaded: "
            f"input={input_shape}, latent_dim={self.depth_latent_dim}"
        )

    def _setup_joint_reordering(self):
        """Compute joint reordering indices between robot and model joint orders.

        Uses ``joint_names`` from the ONNX model metadata (the training joint order)
        and ``self.robot_config.dof_names`` (the robot's canonical joint order).

        - ``_real2model_index``: indexing real-order data with this yields model order.
          Used on observations (dof_pos, dof_vel) before feeding the ONNX model.
        - ``_model2real_index``: indexing model-order data with this yields real order.
          Used on actions output by the ONNX model for motor commands.
        """
        model_joint_names = self._model_joint_names
        real_joint_names = list(self.robot_config.dof_names)

        if model_joint_names is None or list(model_joint_names) == real_joint_names:
            self._real2model_index = None
            self._model2real_index = None
            return

        from holosoma_inference.utils.math.misc import get_index_of_a_in_b

        # real2model: for each model joint, find its position in real joint order
        self._real2model_index = get_index_of_a_in_b(model_joint_names, real_joint_names)

        # model2real: inverse mapping
        n = len(self._real2model_index)
        self._model2real_index = [0] * n
        for model_pos, real_pos in enumerate(self._real2model_index):
            self._model2real_index[real_pos] = model_pos


        logger.info(
            f"[DepthDistillationPolicy] Joint reordering enabled via ONNX metadata: "
            f"model={model_joint_names}, real={real_joint_names}"
        )

    def _init_waist_joint_indices(self):
        """Pre-compute waist joint indices for anchor-body FK.

        The training observation ``robot_anchor_projected_gravity`` projects
        gravity into the anchor body frame (``torso_link``), which is connected
        to the root (pelvis) through waist_yaw → waist_roll → waist_pitch.
        We store the indices so ``_get_gravity_frame_quat`` can chain the
        rotations at runtime.
        """
        # (joint_name_substring, local rotation axis)
        waist_chain = [
            ("waist_yaw_joint", np.array([0.0, 0.0, 1.0])),
            ("waist_roll_joint", np.array([1.0, 0.0, 0.0])),
            ("waist_pitch_joint", np.array([0.0, 1.0, 0.0])),
        ]

        self._waist_joint_info: list[tuple[int, np.ndarray]] = []
        for name, axis in waist_chain:
            if name in self.dof_names:
                self._waist_joint_info.append((self.dof_names.index(name), axis))

        if self._waist_joint_info:
            names = [self.dof_names[idx] for idx, _ in self._waist_joint_info]
            logger.info(f"[DepthDistillationPolicy] Anchor-body FK through waist joints: {names}")

    def _get_gravity_frame_quat(self, robot_state_data, base_quat):
        """Compute the anchor body (torso) quaternion for projected gravity.

        Matches the training's ``robot_anchor_projected_gravity`` observation
        which uses ``body_quat_w[:, torso_link_index]`` rather than the root
        body quaternion.

        On the real G1 robot the IMU is mounted in the torso, so ``base_quat``
        already equals the torso quaternion and the FK chain is a no-op
        (waist joint angles are near zero relative to the IMU frame).  For
        sim-to-sim (e.g. MuJoCo) the floating-base quaternion is the pelvis,
        so we chain the waist joint rotations to obtain the torso quaternion.
        """
        if not self._waist_joint_info:
            return base_quat

        # Raw (absolute) joint positions — before default-angle subtraction
        raw_dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]

        anchor_quat = base_quat.copy()
        for idx, axis in self._waist_joint_info:
            angle = float(raw_dof_pos[0, idx])
            joint_quat = quat_from_angle_axis(angle, axis)
            anchor_quat = quat_mul(anchor_quat, joint_quat)

        return anchor_quat

    def _load_student_model(self, model_path: str):
        """Load the student ONNX model and extract metadata."""
        self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
        self.onnx_input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        self.onnx_output_names = [out.name for out in self.onnx_policy_session.get_outputs()]

        # Extract metadata
        onnx_model = onnx.load(model_path)
        metadata = {}
        for prop in onnx_model.metadata_props:
            try:
                metadata[prop.key] = json.loads(prop.value)
            except (json.JSONDecodeError, ValueError):
                metadata[prop.key] = prop.value

        # Extract KP/KD from metadata
        self.onnx_kp = np.array(metadata["kp"]) if "kp" in metadata else None
        self.onnx_kd = np.array(metadata["kd"]) if "kd" in metadata else None

        if self.onnx_kp is not None:
            logger.info(f"Loaded KP/KD from ONNX metadata: {os.path.basename(model_path)}")

        # Extract joint names from metadata for joint reordering
        self._model_joint_names = metadata.get("joint_names", None).split(",")

        # Extract action scale from metadata if available
        if "action_scale" in metadata:
            action_scale = metadata["action_scale"]
            if isinstance(action_scale, (int, float)):
                self.policy_action_scale = float(action_scale)
                logger.info(f"Using action_scale from ONNX metadata: {self.policy_action_scale}")

        # Build the policy callable
        def policy_act(obs_dict):
            input_feed = {name: obs_dict[name] for name in self.onnx_input_names}
            outputs = self.onnx_policy_session.run(self.onnx_output_names, input_feed)
            return outputs  # Return all outputs, not just actions

        self.policy = policy_act

        logger.info(
            f"[DepthDistillationPolicy] Student model loaded: "
            f"inputs={self.onnx_input_names}, outputs={self.onnx_output_names}"
        )

    def _capture_policy_state(self) -> dict:
        """Capture the current policy state for later reuse."""
        return {
            "onnx_policy_session": self.onnx_policy_session,
            "onnx_input_names": self.onnx_input_names,
            "onnx_output_names": self.onnx_output_names,
            "policy_callable": self.policy,
            "onnx_kp": self.onnx_kp,
            "onnx_kd": self.onnx_kd,
            "depth_backbone_session": self.depth_backbone_session,
        }

    def _restore_policy_state(self, state: dict):
        """Restore a previously captured policy state."""
        super()._restore_policy_state(state)
        self.depth_backbone_session = state["depth_backbone_session"]

    def _run_depth_backbone(self, depth_image: np.ndarray) -> np.ndarray:
        """Run depth backbone to get latent vector.

        Parameters
        ----------
        depth_image : np.ndarray
            Depth image with shape matching backbone input requirements.

        Returns
        -------
        np.ndarray
            Depth latent vector of shape (1, depth_latent_dim).
        """
        input_feed = {self.depth_backbone_input_name: depth_image.astype(np.float32)}
        outputs = self.depth_backbone_session.run([self.depth_backbone_output_name], input_feed)
        return outputs[0]

    def _get_depth_image(self) -> np.ndarray:
        """Read depth image from shared memory and prepare for backbone.

        Returns
        -------
        np.ndarray
            Depth image shaped to match the backbone's expected input.
        """
        # Read from shared memory: shape (num_cameras, 1, H, W)
        depth_images = self.depth_img_array.copy()

        # Use first (and only) camera for single D435i setup
        # Shape: (1, H, W) -> squeeze channel dim -> (H, W)
        depth_frame = depth_images[0, 0]  # (H, W)

        # Update depth frame buffer
        self.depth_frame_buffer.append(depth_frame.copy())

        # Pad buffer if not yet full
        buffer_len = self.depth_buffer_len
        frames = list(self.depth_frame_buffer)
        if len(frames) < buffer_len:
            missing = buffer_len - len(frames)
            frames = [frames[0]] * missing + frames
        frames = frames[-buffer_len:]

        # Stack frames: (buffer_len, H, W)
        stacked = np.stack(frames, axis=0)  # (buffer_len, H, W)

        # Reshape to match the backbone's exact expected input shape
        return stacked

    def get_current_obs_buffer_dict(self, robot_state_data):
        """Build observation buffer with proprioceptive data.

        If joint reordering is enabled (``motion_dof_names`` differs from ``dof_names``),
        ``dof_pos`` and ``dof_vel`` are reordered from robot order to the model's
        expected motion-data order.  ``actions`` is already stored in model order
        (the raw ONNX output from the previous step), so it needs no reordering here.
        """
        current_obs_buffer_dict = super().get_current_obs_buffer_dict(robot_state_data)

        if self._real2model_index is not None:
            current_obs_buffer_dict["dof_pos"] = current_obs_buffer_dict["dof_pos"][:, self._real2model_index]
            current_obs_buffer_dict["dof_vel"] = current_obs_buffer_dict["dof_vel"][:, self._real2model_index]

        # Add depth image placeholder (actual processing happens in prepare_obs_for_rl)
        # This is needed so parse_current_obs_dict doesn't fail on missing "cam_depth"
        camera_config = self.config.camera
        if camera_config is not None:
            h = camera_config.props.resized_height
            w = camera_config.props.resized_width
            current_obs_buffer_dict["cam_depth"] = np.zeros((1, h * w))

        return current_obs_buffer_dict

    def prepare_obs_for_rl(self, robot_state_data):
        """Prepare observations for RL inference.

        1. Build proprioceptive observation vector (projected_gravity, ang_vel, dof_pos, dof_vel, actions)
        2. Append velocity command (one-hot) — matches training where command obs is concatenated to policy obs
        3. Read depth image from shared memory
        4. Run depth_backbone ONNX -> depth latent
        5. Concatenate: [proprioceptive, velocity_command, depth_latent]
        """
        # Build proprioceptive observations using group history
        self._prepare_group_observations(robot_state_data)

        # Get the actor_obs (proprioceptive) from history buffers
        actor_obs = self.obs_buf_dict["actor_obs"]  # (1, prop_dim)

        # Get depth image and run backbone
        depth_image = self._get_depth_image()  # (1, buffer_len, H, W)
        depth_latent = self._run_depth_backbone(depth_image)  # (1, latent_dim)

        # Concatenate proprioceptive + velocity command + depth latent
        # This matches the training order: obs = cat([policy_obs, command], dim=-1)
        # then depth_latent is appended by the student model forward pass.
        parts = [actor_obs]
        if self.velocity_command_dim > 0:
            parts.append(self.velocity_command)
        parts.append(depth_latent)
        obs = np.concatenate(parts, axis=1).astype(np.float32)

        input_feed = {"obs": obs}
        if "time_step" in self.onnx_input_names:
            input_feed["time_step"] = np.zeros((1, 1), dtype=np.float32)
        return input_feed

    def rl_inference(self, robot_state_data):
        """Perform RL inference with two-model architecture.

        When joint reordering is enabled, the raw ONNX action output is kept in
        model (motion-data) order for ``last_policy_action`` so that it feeds back
        correctly as the ``actions`` observation on the next step.  The returned
        ``scaled_policy_action`` is reordered to robot order so that it can be
        added to ``default_dof_angles`` and sent to the motors.
        """
        obs_dict = self.prepare_obs_for_rl(robot_state_data)

        if self.config.task.print_observations:
            self._print_observations(obs_dict)

        # Run student model
        outputs = self.policy(obs_dict)

        # Extract actions (first output) — in model's joint order
        policy_action = outputs[0]

        # Store raw action in model order (fed back as the "actions" observation)
        self.last_policy_action = policy_action.copy()

        # Reorder actions from model order to robot order for motor commands
        if self._model2real_index is not None:
            policy_action = policy_action[:, self._model2real_index]

        self.scaled_policy_action = policy_action * self.policy_action_scale

        return self.scaled_policy_action

    def process_joystick_input(self):
        """Process joystick input and map to velocity command one-hot."""
        prev_raw_keys = getattr(self, '_prev_raw_keys', 0)

        super().process_joystick_input()

        wc_msg = self.interface.get_joystick_msg()
        raw_keys = getattr(wc_msg, "keys", 0) if wc_msg else 0
        self._prev_raw_keys = raw_keys

        _L1 = 2
        _X = 1024
        _Y = 2048
        l1_held = bool(raw_keys & _L1)
        y_rising = bool(raw_keys & _Y) and not bool(prev_raw_keys & _Y)

        if l1_held and y_rising:
            self.speed_mode_high = not self.speed_mode_high
            logger.info(f"Speed mode switched to: {'HIGH' if self.speed_mode_high else 'LOW'}")

        if self.speed_mode_high:
            x_held = bool(raw_keys & _X)
            self.speed_mode = 2 if x_held else 1
        else:
            self.speed_mode = 0

        self._update_velocity_command_from_joystick()

    def _update_velocity_command_from_joystick(self):
        """Map left joystick axes to a discrete velocity command via angle sectors.

        Only active while L1 is held. Reads stick values directly from the raw
        joystick message to bypass the base_interface gate (which suppresses
        sticks when any button is pressed).

        Uses the left stick exclusively:
          - ly → linear_x  (forward/backward)
          - lx → angular_z (yaw/turning, negated)

        Mirrors the C++ MotionTrackingController::cmd_vel_callback logic:
        1. If L1 is not held, do nothing (keep previous command).
        2. Deadzone on hypot(linear_x, angular_z) < 0.05 -> stand.
        3. Compute a_deg = atan2(linear_x, -angular_z / 1.5) * 180 / pi.
        4. Map angle sectors to discrete command codes:
             (-45, 45)    -> right_45
             [45, 135)    -> forward
             [135, 225)   -> left_45
             [225, ...) or <= -45 -> back
        5. Select command code table based on current speed_mode.
        """
        if self.velocity_command_dim == 0:
            return

        if not self.key_states.get("L1", False):
            return

        # Read raw stick values directly from the joystick message,
        # since base_interface suppresses sticks when buttons are held.
        wc_msg = self.interface.get_joystick_msg()
        if wc_msg is None:
            return

        lx = getattr(wc_msg, "lx", 0.0)
        ly = getattr(wc_msg, "ly", 0.0)
        linear_x = ly if abs(ly) > 0.1 else 0.0
        angular_z = -lx if abs(lx) > 0.1 else 0.0

        norm_xz = math.hypot(linear_x, angular_z)
        codes = self.CMD_CODES[self.speed_mode]

        if norm_xz < self.JOYSTICK_DEADZONE:
            cmd_idx = codes["stand"]
        else:
            a_deg = math.atan2(linear_x, -angular_z / 1.5) * 180.0 / math.pi

            if -45.0 < a_deg < 45.0:
                cmd_idx = codes["right_45"]
            elif 45.0 <= a_deg < 135.0:
                cmd_idx = codes["forward"]
            elif 135.0 <= a_deg < 225.0:
                cmd_idx = codes["left_45"]
            else:  # a_deg >= 225 or a_deg <= -45
                cmd_idx = codes["back"]

        if cmd_idx != self.active_velocity_command_idx:
            self.set_velocity_command(cmd_idx)

    # Keyboard key -> discrete velocity command name mapping.
    # Unlike LocomotionPolicy (continuous lin_vel/ang_vel), this policy uses
    # discrete one-hot velocity commands selected from CMD_CODES.
    KEY_TO_CMD = {
        "w": "forward",
        "s": "back",
        "a": "left_45",
        "d": "right_45",
        "q": "left_90",
        "e": "right_90",
        "z": "stand",
    }

    def handle_keyboard_button(self, keycode):
        """Handle keyboard button presses for depth distillation.

        Overrides LocomotionPolicy's continuous velocity handlers (W/A/S/D/Q/E)
        with discrete velocity command selection, matching the joystick behavior
        in process_joystick_input.
        """
        # Base policy handles: ]/p (start), o (stop), i (init), v/b/f/g/r (kp), 1-9 (policy switch)
        # Skip LocomotionPolicy's continuous velocity handlers — they don't apply here.
        BasePolicy.handle_keyboard_button(self, keycode)

        if keycode in self.KEY_TO_CMD:
            codes = self.CMD_CODES[self.speed_mode]
            cmd_idx = codes[self.KEY_TO_CMD[keycode]]
            if cmd_idx != self.active_velocity_command_idx:
                self.set_velocity_command(cmd_idx)
        elif keycode == "=":
            # Cycle speed mode: low -> high -> madmax -> low
            self.speed_mode = (self.speed_mode + 1) % 3
            self.speed_mode_high = self.speed_mode > 0
            label = ["LOW", "HIGH", "MADMAX"][self.speed_mode]
            logger.info(f"Speed mode: {label}")
        elif keycode == "h":
            self._handle_enter_stiff_hold()

        self._print_control_status()

    def _print_control_status(self):
        """Print current control status with discrete velocity command info."""
        super()._print_control_status()
        label = ["LOW", "HIGH", "MADMAX"][self.speed_mode]
        # Reverse-lookup active command name from CMD_CODES
        cmd_name = "?"
        for name, idx in self.CMD_CODES[self.speed_mode].items():
            if idx == self.active_velocity_command_idx:
                cmd_name = name
                break
        mode = "stiff_hold" if self._stiff_hold_active else "damping" if self._damping_mode_active else "policy" if self.use_policy_action else "idle"
        print(f"Command: {cmd_name} (idx={self.active_velocity_command_idx}) | Speed: {label} | Mode: {mode}")

    def handle_joystick_button(self, cur_key):
        """Handle joystick button presses."""
        if cur_key == "start":
            self._handle_enter_stiff_hold()
        elif cur_key == "Y":
            # Toggle speed mode (low <-> high), matching C++ Y-button behavior
            self.speed_mode_high = not self.speed_mode_high
            logger.info(f"Speed mode switched to: {'HIGH' if self.speed_mode_high else 'LOW'}")
        else:
            super().handle_joystick_button(cur_key)


    def _get_manual_command(self, robot_state_data):
        """Return manual command when policy is not active.

        Stiff hold: high Kp/Kd at a fixed standing pose (on boot).
        Damping mode: Kp=0, Kd>0 so joints resist motion without tracking a position.
        """
        if self._stiff_hold_active:
            # On first call of a new stiff hold activation, capture current joint positions
            if self._stiff_blend_start_q is None:
                self._stiff_blend_start_q = robot_state_data[:, 7 : 7 + self.num_dofs].copy()

            # Compute interpolation alpha with smoothstep easing (slow start and end)
            t = min(self._stiff_blend_count / self._stiff_blend_duration, 1.0)
            alpha = t * t * (3.0 - 2.0 * t)

            # Interpolate joint positions from current to stiff target
            q = (1 - alpha) * self._stiff_blend_start_q + alpha * self._stiff_hold_q

            # Interpolate KP/KD from zero to stiff gains (robot starts soft, stiffens up)
            kp = alpha * self._stiff_hold_kp
            kd = alpha * self._stiff_hold_kd

            self._stiff_blend_count += 1

            return {
                "q": q,
                "kp": kp,
                "kd": kd,
            }
        if self._damping_mode_active:
            dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
            return {
                "q": dof_pos,
                "kp": np.zeros_like(self._stiff_hold_kp),
                "kd": self._stiff_hold_kd,
            }
        return None

    def _handle_enter_stiff_hold(self):
        """Enter stiff hold: hold the default standing pose with high Kp/Kd."""
        self.use_policy_action = False
        self.get_ready_state = False
        self._stiff_hold_active = True
        self._damping_mode_active = False
        self._stiff_blend_count = 0
        self._stiff_blend_start_q = None
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0
        self.depth_frame_buffer.clear()
        self.set_velocity_command(self.CMD_CODES[0]["stand"])
        logger.info("Entering stiff hold mode")

    def _handle_start_policy(self):
        """Handle start policy action."""
        super()._handle_start_policy()
        self._stiff_hold_active = False
        self._damping_mode_active = False
        logger.info("Depth distillation policy started")

    def _handle_stop_policy(self):
        """Enter damping mode: Kp=0, Kd>0 so joints resist motion without position tracking.

        Does NOT call super() because BasePolicy._handle_stop_policy sets
        no_action=1, which forces Kp=0 AND Kd=0 in the command sender,
        bypassing _get_manual_command overrides entirely.
        """
        self.use_policy_action = False
        self.get_ready_state = False
        self._stiff_hold_active = False
        self._damping_mode_active = True
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0
        self.depth_frame_buffer.clear()
        self.set_velocity_command(self.CMD_CODES[0]["stand"])
        logger.info("Entering damping mode (Kp=0, Kd>0)")

    def set_velocity_command(self, idx: int):
        """Set the active velocity command as a one-hot vector.

        Parameters
        ----------
        idx : int
            Index of the active command class (0 to velocity_command_dim-1).
            Use -1 to clear (all zeros).
        """
        if self.velocity_command_dim == 0:
            return
        self.velocity_command[:] = 0.0
        if 0 <= idx < self.velocity_command_dim:
            self.velocity_command[0, idx] = 1.0
            self.active_velocity_command_idx = idx
            logger.info(f"Velocity command set to index {idx} (one-hot dim {self.velocity_command_dim})")
        else:
            self.active_velocity_command_idx = self.CMD_CODES[0]["stand"]
            logger.info("Velocity command cleared (back to stand)")
