import json
import os
import sys
import time
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import pinocchio as pin
from defusedxml import ElementTree
from loguru import logger
from termcolor import colored

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_types.robot import RobotConfig
from holosoma_inference.policies import BasePolicy
from holosoma_inference.utils.clock import ClockSub
from holosoma_inference.utils.math.misc import get_index_of_a_in_b
from holosoma_inference.utils.math.quat import (
    matrix_from_quat,
    quat_apply,
    quat_inverse,
    quat_mul,
    quat_rotate_inverse,
    quat_to_rpy,
    rpy_to_quat,
    subtract_frame_transforms,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from holosoma_inference.utils.sim_state import SimStateSub


class PinocchioRobot:
    def __init__(self, robot_cfg: RobotConfig, urdf_text: str):
        # create pinocchio robot
        xml_text = self._create_xml_from_urdf(urdf_text)
        self.robot_model = pin.buildModelFromXML(xml_text, pin.JointModelFreeFlyer())
        self.robot_data = self.robot_model.createData()

        # get joint names in pinocchio robot and real robot
        joint_names_in_real_robot = robot_cfg.dof_names
        joint_names_in_pinocchio_robot = [
            name for name in self.robot_model.names if name not in ["universe", "root_joint"]
        ]
        assert len(joint_names_in_pinocchio_robot) == len(joint_names_in_real_robot), (
            "The number of joints in the pinocchio robot and the real robot are not the same"
        )
        self.real2pinocchio_index = get_index_of_a_in_b(joint_names_in_pinocchio_robot, joint_names_in_real_robot)

        # get ref body frame id in pinocchio robot
        self.ref_body_frame_id = self.robot_model.getFrameId(robot_cfg.motion["body_name_ref"][0])

    def fk_and_get_ref_body_orientation_in_world(self, configuration: np.ndarray) -> np.ndarray:
        # forward kinematics
        pin.framesForwardKinematics(self.robot_model, self.robot_data, configuration)

        # get ref body pose in world
        ref_body_pose_in_world = self.robot_data.oMf[self.ref_body_frame_id]
        quaternion = pin.Quaternion(ref_body_pose_in_world.rotation)  # (4, )

        return np.expand_dims(quaternion.coeffs(), axis=0)  # xyzw, (1, 4)

    def fk_ref_body_pose_with_root_identity(self, dof_pos_in_real: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        root_pos = np.zeros(3, dtype=np.float64)
        root_ori_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        dof_pos_in_pinocchio = dof_pos_in_real.reshape(-1)[self.real2pinocchio_index]

        configuration = np.concatenate([root_pos, root_ori_xyzw, dof_pos_in_pinocchio], axis=0)
        pin.framesForwardKinematics(self.robot_model, self.robot_data, configuration)

        ref_body_pose = self.robot_data.oMf[self.ref_body_frame_id]
        ref_pos = np.asarray(ref_body_pose.translation, dtype=np.float32).reshape(1, 3)
        ref_quat_xyzw = np.asarray(pin.Quaternion(ref_body_pose.rotation).coeffs(), dtype=np.float32).reshape(1, 4)
        return ref_pos, xyzw_to_wxyz(ref_quat_xyzw)

    @staticmethod
    def _create_xml_from_urdf(urdf_text: str) -> str:
        """Strip visuals/collisions from URDF text and return XML text."""
        root = ElementTree.fromstring(urdf_text)

        def _is_visual_or_collision(tag: str) -> bool:
            # Handle optional XML namespaces by only checking the suffix after '}'.
            return tag.split("}")[-1] in {"visual", "collision"}

        for parent in root.iter():
            for child in list(parent):
                if _is_visual_or_collision(child.tag):
                    parent.remove(child)

        xml_text = ElementTree.tostring(root, encoding="unicode")
        if not xml_text.lstrip().startswith("<?xml"):
            xml_text = '<?xml version="1.0"?>\n' + xml_text
        return xml_text


class WholeBodyTrackingPolicy(BasePolicy):
    def __init__(self, config: InferenceConfig):
        # initialize timestep
        self.motion_timestep = 0
        self.motion_clip_progressing = False
        self.motion_start_timestep = None
        self.motion_command_t = None
        self.ref_quat_xyzw_t = None
        self.motion_ref_pos_xyz_t = None
        self.motion_command_0 = None
        self.ref_quat_xyzw_0 = None
        self.motion_ref_pos_xyz_0 = None
        self._depth_img_shm = None
        self._depth_img_array = None
        self._motion_root_pos_w = None
        self._motion_root_quat_wxyz = None
        self._motion_root_command_origin_xy = None
        self._motion_joint_pos = None
        self._motion_joint_vel = None
        self._motion_object_pos_w = None
        self._contact_aware_carry_window = None
        self._contact_aware_window_mode = "rel_z"
        self._contact_aware_peak_height_alpha = 0.91
        self._contact_aware_peak_height_smoothing_steps = 5
        self._latest_sim_state: dict | None = None
        self._sim_state_sub: SimStateSub | None = None
        self._logged_sim_ref_from_sim_state = False
        self._policy_debug_path = os.environ.get("HOLOSOMA_POLICY_DEBUG_INPUT_PATH", "").strip()
        policy_command_status_path = os.environ.get(
            "HOLOSOMA_POLICY_COMMAND_STATUS_PATH",
            "/tmp/holosoma_policy_command_status.json",
        ).strip()
        self._policy_command_status_path = Path(policy_command_status_path) if policy_command_status_path else None
        self._policy_command_status_next_time = 0.0
        self._policy_command_status_period = 0.05
        self._logged_policy_command_status_error = False
        try:
            self._policy_debug_limit = int(os.environ.get("HOLOSOMA_POLICY_DEBUG_INPUT_LIMIT", "200") or "200")
        except ValueError:
            self._policy_debug_limit = 200
        self._policy_debug_count = 0
        self._policy_debug_file = None
        self._last_policy_inference_clock_ms: int | None = None
        self._use_motion_data_as_q_target = os.environ.get("HOLOSOMA_USE_MOTION_DATA_AS_Q_TARGET", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._logged_motion_data_q_target = False
        self._force_zero_sparse_root_command = os.environ.get(
            "HOLOSOMA_FORCE_ZERO_SPARSE_ROOT_COMMAND", os.environ.get("HOLOSOMA_FORCE_MANUAL_SPARSE_ROOT_COMMAND", "")
        ).strip().lower() in {"1", "true", "yes", "on"}
        try:
            self._pickup_button_command = float(os.environ.get("HOLOSOMA_POLICY_PICKUP_BUTTON", "1") or "1")
        except ValueError:
            self._pickup_button_command = 1.0
        try:
            self._drop_button_command = float(os.environ.get("HOLOSOMA_POLICY_DROP_BUTTON", "0") or "0")
        except ValueError:
            self._drop_button_command = 0.0
        # What a fresh launch starts with; restored whenever the policy is stopped
        # so that a restart sees the same inputs a first start does.
        self._pickup_button_startup = self._pickup_button_command
        self._drop_button_startup = self._drop_button_command
        self._pickup_button_key_down = False
        self._drop_button_key_down = False
        self._logged_missing_drop_button_key = False
        self._logged_zero_sparse_root_command = False
        self._logged_motion_local_sparse_root_command = False
        self._manual_sparse_root_command_offset = np.zeros((1, 3), dtype=np.float32)
        self._joystick_sparse_root_command_offset = np.zeros((1, 3), dtype=np.float32)
        self._external_sparse_root_command_mode = False
        self._policy_returns_reference_outputs = True
        self._policy_action_output_name = "actions"
        try:
            self._motion_index_offset = int(os.environ.get("HOLOSOMA_POLICY_MOTION_INDEX_OFFSET", "0") or "0")
        except ValueError:
            self._motion_index_offset = 0

        # Calculate timestep interval from rl_rate (e.g., 50Hz = 20ms intervals)
        self.timestep_interval_ms = 1000.0 / config.task.rl_rate

        # Initialize clock subscriber for synchronization
        self.clock_sub = ClockSub()
        self.clock_sub.start()
        self._last_clock_reading: int | None = None

        # Read use_sim_time from config
        self.use_sim_time = config.task.use_sim_time

        self._stiff_hold_active = True
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0

        super().__init__(config)
        if config.task.use_sim_state:
            self._sim_state_sub = SimStateSub(port=config.task.sim_state_port)
            self._sim_state_sub.start()
        if self._motion_index_offset != 0:
            logger.info("Using motion sequence index offset: {}", self._motion_index_offset)

        # Load stiff startup parameters from robot config
        if config.robot.stiff_startup_pos is not None:
            self._stiff_hold_q = np.array(config.robot.stiff_startup_pos, dtype=np.float32).reshape(1, -1)
        else:
            # Fallback to default_dof_angles if not specified
            self._stiff_hold_q = np.array(config.robot.default_dof_angles, dtype=np.float32).reshape(1, -1)

        if config.robot.stiff_startup_kp is not None:
            self._stiff_hold_kp = np.array(config.robot.stiff_startup_kp, dtype=np.float32)
        else:
            raise ValueError("Robot config must specify stiff_startup_kp for WBT policy")

        if config.robot.stiff_startup_kd is not None:
            self._stiff_hold_kd = np.array(config.robot.stiff_startup_kd, dtype=np.float32)
        else:
            raise ValueError("Robot config must specify stiff_startup_kd for WBT policy")

        if self._stiff_hold_q.shape[1] != self.num_dofs:
            raise ValueError("Stiff startup pose dimension mismatch with robot DOFs")

        # Prompt user before entering stiff mode (only if stdin is available)
        def _show_warning():
            logger.warning(
                colored(
                    "⚠️  Non-interactive mode detected - cannot prompt for stiff mode confirmation!",
                    "red",
                    attrs=["bold"],
                )
            )

        if config.task.auto_start_policy:
            logger.info("Auto-start policy enabled; skipping stiff hold confirmation prompt")
        elif sys.stdin.isatty():
            logger.info(colored("\n⚠️  Ready to enter stiff hold mode", "yellow", attrs=["bold"]))
            logger.info(colored("Press Enter to continue...", "yellow"))
            try:
                input()
                logger.info(colored("✓ Entering stiff hold mode", "green"))
            except EOFError:
                # [drockyd] seems like in some cases, input() will raise EOFError even in interactive mode.
                _show_warning()
        else:
            _show_warning()

    def _get_latest_sim_state(self) -> dict | None:
        if self._sim_state_sub is None:
            return self._latest_sim_state
        state = self._sim_state_sub.get_state()
        if state is not None:
            self._latest_sim_state = state
        return self._latest_sim_state

    def _get_sim_root_state(self) -> np.ndarray | None:
        state = self._get_latest_sim_state()
        if not state:
            return None
        root_state = state.get("robot_root_state")
        if root_state is None:
            return None
        root_state_np = np.asarray(root_state, dtype=np.float32).reshape(1, -1)
        if root_state_np.shape[1] < 13:
            return None
        return root_state_np[:, :13]

    def _get_sim_ref_state(self) -> np.ndarray | None:
        state = self._get_latest_sim_state()
        if not state:
            return None
        ref_state = state.get("robot_ref_state")
        if ref_state is None:
            return None
        ref_state_np = np.asarray(ref_state, dtype=np.float32).reshape(1, -1)
        if ref_state_np.shape[1] < 13:
            return None
        return ref_state_np[:, :13]

    def _augment_robot_state_with_sim_state(self, robot_state_data: np.ndarray | None) -> np.ndarray | None:
        if robot_state_data is None:
            return None
        sim_root_state = self._get_sim_root_state()
        if sim_root_state is None:
            return robot_state_data

        augmented = np.array(robot_state_data, dtype=np.float32, copy=True)
        root_quat_wxyz = xyzw_to_wxyz(sim_root_state[:, 3:7])
        augmented[:, :3] = sim_root_state[:, :3]
        augmented[:, 3:7] = root_quat_wxyz
        augmented[:, 7 + self.num_dofs : 7 + self.num_dofs + 3] = quat_rotate_inverse(
            root_quat_wxyz,
            sim_root_state[:, 7:10],
        )
        augmented[:, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6] = quat_rotate_inverse(
            root_quat_wxyz,
            sim_root_state[:, 10:13],
        )

        state = self._get_latest_sim_state()
        if state:
            dof_pos = state.get("robot_dof_pos")
            dof_vel = state.get("robot_dof_vel")
            if dof_pos is not None:
                dof_pos_np = np.asarray(dof_pos, dtype=np.float32).reshape(1, -1)
                if dof_pos_np.shape[1] >= self.num_dofs:
                    augmented[:, 7 : 7 + self.num_dofs] = dof_pos_np[:, : self.num_dofs]
            if dof_vel is not None:
                dof_vel_np = np.asarray(dof_vel, dtype=np.float32).reshape(1, -1)
                if dof_vel_np.shape[1] >= self.num_dofs:
                    augmented[:, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs] = dof_vel_np[
                        :, : self.num_dofs
                    ]
        return augmented

    def _get_ref_body_pose_in_world(self, robot_state_data):
        if bool(getattr(self.config.task, "prefer_sim_ref_from_sim_state", False)):
            sim_ref_state = self._get_sim_ref_state()
            if sim_ref_state is not None:
                if not self._logged_sim_ref_from_sim_state:
                    logger.info("Using simulator-measured ref-body pose from sim state")
                    self._logged_sim_ref_from_sim_state = True
                return sim_ref_state[:, :3], xyzw_to_wxyz(sim_ref_state[:, 3:7])

        # Create configuration for pinocchio robot
        # Note:
        # 1. pinocchio quaternion is in xyzw format, robot_state_data is in wxyz format
        # 2. joint sequences in pinocchio robot and real robot are different

        # free base pos, does not matter
        root_pos = robot_state_data[0, :3]

        # free base ori, wxyz -> xyzw
        root_ori_xyzw = wxyz_to_xyzw(robot_state_data[:, 3:7])[0]

        # dof pos in real robot -> pinocchio robot
        num_dofs = self.num_dofs
        dof_pos_in_real = robot_state_data[0, 7 : 7 + num_dofs]
        dof_pos_in_pinocchio = dof_pos_in_real[self.pinocchio_robot.real2pinocchio_index]

        configuration = np.concatenate([root_pos, root_ori_xyzw, dof_pos_in_pinocchio], axis=0)

        ref_ori_xyzw = self.pinocchio_robot.fk_and_get_ref_body_orientation_in_world(configuration)
        return np.zeros((1, 3), dtype=np.float32), xyzw_to_wxyz(ref_ori_xyzw)

    def _get_ref_body_orientation_in_world(self, robot_state_data):
        return self._get_ref_body_pose_in_world(robot_state_data)[1]

    def setup_policy(self, model_path):
        # The policy net runs in ~0.2 ms; onnxruntime's default thread pool (one
        # spinning thread per core) only starves whatever else shares the machine,
        # notably the depth server. HOLOSOMA_ORT_THREADS (default 2) caps it.
        so = onnxruntime.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get("HOLOSOMA_ORT_THREADS", "2"))
        so.inter_op_num_threads = 1
        self.onnx_policy_session = onnxruntime.InferenceSession(model_path, so)
        self.onnx_input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        self.onnx_output_names = [out.name for out in self.onnx_policy_session.get_outputs()]
        self._policy_action_output_name = self._resolve_action_output_name()

        # Extract KP/KD from ONNX metadata (same as base class)
        onnx_model = onnx.load(model_path)
        metadata = {}
        for prop in onnx_model.metadata_props:
            metadata[prop.key] = json.loads(prop.value)

        self._deployment_audit = None
        if os.environ.get("HOLOSOMA_DEPLOYMENT_AUDIT_DIR"):
            from holosoma.utils.deployment_audit import create_deployment_audit, sha256_file

            self._deployment_audit = create_deployment_audit("policy", {
                "model_path": str(model_path), "model_sha256": sha256_file(model_path),
                "onnx_input_names": self.onnx_input_names,
                "onnx_output_names": self.onnx_output_names,
                "onnx_metadata": metadata, "robot_config": self.config.robot,
                "observation_config": self.config.observation, "camera_config": self.config.camera,
                "task_config": self.config.task,
                "semantics": "successful_inference_inputs_not_proof_of_task_success",
            })

        # Extract URDF text from ONNX metadata
        assert "robot_urdf" in metadata, "Robot urdf text not found in ONNX metadata"
        self.pinocchio_robot = PinocchioRobot(self.config.robot, metadata["robot_urdf"])
        self._load_sparse_root_motion_file()

        self.onnx_kp = np.array(metadata["kp"]) if "kp" in metadata else None
        self.onnx_kd = np.array(metadata["kd"]) if "kd" in metadata else None

        if self.onnx_kp is not None:
            logger.info(f"Loaded KP/KD from ONNX metadata: {Path(model_path).name}")

        self._configure_contact_aware_window(metadata)
        self._set_policy_action_scales_from_metadata(metadata)
        self._policy_returns_reference_outputs = True

        self._external_sparse_root_command_mode = (
            self._motion_root_pos_w is None
            and (
                "sparse_target_root_trajectory_command" in self.obs_dims
                or "sparse_target_root_trajectory_command_contact_aware" in self.obs_dims
            )
            and "motion_command" not in self.obs_dims
        )
        if self._external_sparse_root_command_mode:
            logger.info(
                "Using external sparse-root command mode: command comes from zero/manual/joystick input; "
                "reference-motion outputs are not used."
            )
            self._policy_returns_reference_outputs = False
            self.motion_command_t = np.zeros((1, self.num_dofs * 2), dtype=np.float32)
            self.ref_quat_xyzw_t = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
            self.motion_ref_pos_xyz_t = np.zeros((1, 3), dtype=np.float32)
            self.motion_command_0 = self.motion_command_t.copy()
            self.ref_quat_xyzw_0 = self.ref_quat_xyzw_t.copy()
            self.motion_ref_pos_xyz_0 = self.motion_ref_pos_xyz_t.copy()

            def policy_act(input_feed):
                return self.onnx_policy_session.run([self._policy_action_output_name], input_feed)[0]

            self.policy = policy_act
            return

        # get initial command and ref quat xyzw
        input_feed = self._make_initial_input_feed()
        init_output_names = [
            name for name in ("joint_pos", "joint_vel", "ref_quat_xyzw", "ref_pos_xyz") if name in self.onnx_output_names
        ]
        missing_reference_outputs = {"joint_pos", "joint_vel", "ref_quat_xyzw"} - set(self.onnx_output_names)
        if missing_reference_outputs:
            has_sparse_root_obs = (
                "sparse_target_root_trajectory_command" in self.obs_dims
                or "sparse_target_root_trajectory_command_contact_aware" in self.obs_dims
            )
            if has_sparse_root_obs and "motion_command" not in self.obs_dims:
                logger.info(
                    "Using action-only ONNX with sparse-root command from motion/manual input; "
                    "reference-motion outputs are not required."
                )
                self._policy_returns_reference_outputs = False
                self.motion_command_t = np.zeros((1, self.num_dofs * 2), dtype=np.float32)
                self.ref_quat_xyzw_t = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
                self.motion_ref_pos_xyz_t = np.zeros((1, 3), dtype=np.float32)
                self.motion_command_0 = self.motion_command_t.copy()
                self.ref_quat_xyzw_0 = self.ref_quat_xyzw_t.copy()
                self.motion_ref_pos_xyz_0 = self.motion_ref_pos_xyz_t.copy()

                def policy_act(input_feed):
                    return self.onnx_policy_session.run([self._policy_action_output_name], input_feed)[0]

                self.policy = policy_act
                return
            raise RuntimeError(
                "This ONNX is missing tracking reference outputs "
                f"{sorted(missing_reference_outputs)}. If this is a root-pos/action-only policy, run it through "
                "mj_ro.sh external-root handling without --task.motion-file."
            )
        outputs = self.onnx_policy_session.run(init_output_names, input_feed)
        output_map = dict(zip(init_output_names, outputs, strict=True))

        # motion_command_t/ref_quat_xyzw_t will be used in get_current_obs_buffer_dict
        self.motion_command_t = np.concatenate([output_map["joint_pos"], output_map["joint_vel"]], axis=1)
        self.ref_quat_xyzw_t = output_map["ref_quat_xyzw"]
        self.motion_ref_pos_xyz_t = output_map.get("ref_pos_xyz")
        # duplicate, will be used in _get_init_target and _handle_stop_policy
        self.motion_command_0 = self.motion_command_t.copy()
        self.ref_quat_xyzw_0 = self.ref_quat_xyzw_t.copy()
        self.motion_ref_pos_xyz_0 = None if self.motion_ref_pos_xyz_t is None else self.motion_ref_pos_xyz_t.copy()
        if self._motion_joint_pos is not None and self._motion_joint_vel is not None:
            self.motion_command_t = np.concatenate(
                [self._motion_joint_pos[:1], self._motion_joint_vel[:1]], axis=1
            )
            self.motion_command_0 = self.motion_command_t.copy()

        def policy_act(input_feed):
            policy_output_names = [
                name
                for name in (self._policy_action_output_name, "joint_pos", "joint_vel", "ref_quat_xyzw", "ref_pos_xyz")
                if name in self.onnx_output_names
            ]
            output = self.onnx_policy_session.run(policy_output_names, input_feed)
            output_map = dict(zip(policy_output_names, output, strict=True))
            action = output_map[self._policy_action_output_name]
            motion_command = np.concatenate([output_map["joint_pos"], output_map["joint_vel"]], axis=1)
            ref_quat_xyzw = output_map["ref_quat_xyzw"]
            ref_pos_xyz = output_map.get("ref_pos_xyz")
            return action, motion_command, ref_quat_xyzw, ref_pos_xyz

        self.policy = policy_act

    def _resolve_action_output_name(self) -> str:
        for name in ("actions", "action"):
            if name in self.onnx_output_names:
                return name
        raise ValueError(
            "WBT ONNX policy must expose an action output named 'actions' or 'action'; "
            f"got outputs={self.onnx_output_names}"
        )

    @staticmethod
    def _extract_motion_config(metadata: dict) -> dict | None:
        exp_cfg = metadata.get("experiment_config")
        if not isinstance(exp_cfg, dict):
            return None
        motion_cfg = (
            exp_cfg.get("command", {})
            .get("setup_terms", {})
            .get("motion_command", {})
            .get("params", {})
            .get("motion_config", {})
        )
        return motion_cfg if isinstance(motion_cfg, dict) else None

    def _configure_contact_aware_window(self, metadata: dict) -> None:
        motion_cfg = self._extract_motion_config(metadata)
        if not motion_cfg:
            return
        self._contact_aware_window_mode = (
            str(motion_cfg.get("contact_aware_carry_window_mode", "rel_z")).strip().lower().replace("-", "_")
        )
        try:
            self._contact_aware_peak_height_alpha = float(
                motion_cfg.get("contact_aware_peak_height_alpha", self._contact_aware_peak_height_alpha)
            )
        except (TypeError, ValueError):
            self._contact_aware_peak_height_alpha = 0.91
        try:
            self._contact_aware_peak_height_smoothing_steps = int(
                motion_cfg.get(
                    "contact_aware_peak_height_smoothing_steps",
                    self._contact_aware_peak_height_smoothing_steps,
                )
            )
        except (TypeError, ValueError):
            self._contact_aware_peak_height_smoothing_steps = 5

    def _make_initial_input_feed(self) -> dict[str, np.ndarray]:
        """Create zero inputs for WBT model bootstrap outputs."""
        input_feed: dict[str, np.ndarray] = {}
        for input_meta in self.onnx_policy_session.get_inputs():
            name = input_meta.name
            if name == "obs":
                input_feed[name] = self._initial_actor_obs()
            elif name == "time_step":
                input_feed[name] = np.zeros((1, 1), dtype=np.float32)
            elif name == "perception_obs":
                input_feed[name] = self._zeros_for_input(input_meta)
            else:
                input_feed[name] = self._zeros_for_input(input_meta)
        return input_feed

    def _initial_actor_obs(self) -> np.ndarray:
        if "actor_obs" in self.obs_buf_dict:
            return self.obs_buf_dict["actor_obs"].copy()

        parts = [self.obs_buf_dict[group].copy() for group in self._actor_obs_group_names()]
        if not parts:
            raise ValueError("WBT policy needs actor_obs or split actor observation groups.")
        return np.concatenate(parts, axis=1)

    def _actor_obs_group_names(self) -> list[str]:
        return [group for group in self.obs_dict if group.startswith("actor_obs") and group != "actor_obs"]

    @staticmethod
    def _zeros_for_input(input_meta) -> np.ndarray:
        shape = []
        for dim in input_meta.shape:
            shape.append(dim if isinstance(dim, int) and dim > 0 else 1)
        return np.zeros(tuple(shape), dtype=np.float32)

    def _load_sparse_root_motion_file(self) -> None:
        motion_file = str(getattr(self.config.task, "motion_file", "") or "")
        if not motion_file:
            return

        path = self._resolve_motion_file_path(motion_file)
        with np.load(path, allow_pickle=True) as data:
            body_names = data["body_names"].tolist()
            joint_names = data["joint_names"].tolist()
            joint_indexes = [joint_names.index(name) for name in self.dof_names]
            root_index = body_names.index("pelvis") if "pelvis" in body_names else 0

            self._motion_root_pos_w = np.asarray(data["body_pos_w"][:, root_index, :], dtype=np.float32)
            self._motion_root_quat_wxyz = np.asarray(data["body_quat_w"][:, root_index, :], dtype=np.float32)
            self._motion_root_command_origin_xy = self._motion_root_pos_w[:1, :2].copy()
            self._motion_joint_pos = np.asarray(data["joint_pos"][:, 7:], dtype=np.float32)[:, joint_indexes]
            self._motion_joint_vel = np.asarray(data["joint_vel"][:, 6:], dtype=np.float32)[:, joint_indexes]
            if "object_pos_w" in data:
                self._motion_object_pos_w = np.asarray(data["object_pos_w"], dtype=np.float32)

        logger.info("[WBT] Loaded sparse-root motion file: {} ({} frames)", path, self._motion_root_pos_w.shape[0])

    @staticmethod
    def _resolve_motion_file_path(motion_file: str) -> Path:
        path = Path(motion_file).expanduser()
        if path.is_file():
            return path

        for base in (Path.cwd(), *Path(__file__).resolve().parents):
            candidate = base / motion_file
            if candidate.is_file():
                return candidate

        raise FileNotFoundError(f"Motion file not found: {motion_file}")

    def _motion_frame_index(self) -> int:
        if self._motion_root_pos_w is None:
            return 0
        idx = int(self.motion_timestep) + int(self._motion_index_offset)
        return min(max(idx, 0), self._motion_root_pos_w.shape[0] - 1)

    def wait_for_motion_initial_state(
        self,
        timeout_s: float = 5.0,
        yaw_tolerance_rad: float = 0.05,
        joint_tolerance_rad: float = 0.08,
    ) -> None:
        if os.environ.get("HOLOSOMA_MJ_MOTION_INIT", "").strip().lower() not in {"1", "true", "yes", "on"}:
            return
        if self._motion_root_quat_wxyz is None:
            return

        expected_yaw = quat_to_rpy(self._motion_root_quat_wxyz[0])[2]
        expected_q = getattr(
            self,
            "_stiff_hold_q",
            np.asarray(self.default_dof_angles, dtype=np.float32).reshape(1, -1),
        )
        deadline = time.monotonic() + timeout_s
        last_yaw = None
        last_joint_error = None
        while time.monotonic() < deadline:
            robot_state_data = self.interface.get_low_state()
            if robot_state_data is None:
                time.sleep(0.02)
                continue
            sim_state_ready = not self.config.task.use_sim_state or self._get_latest_sim_state() is not None
            current_yaw = quat_to_rpy(robot_state_data[0, 3:7])[2]
            dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
            joint_error = float(np.max(np.abs(dof_pos - expected_q)))
            last_yaw = current_yaw
            last_joint_error = joint_error
            yaw_error = (current_yaw - expected_yaw + np.pi) % (2 * np.pi) - np.pi
            if sim_state_ready and abs(float(yaw_error)) <= yaw_tolerance_rad and joint_error <= joint_tolerance_rad:
                logger.info(
                    "Matched motion-init low state: yaw current={:.1f} deg expected={:.1f} deg joint_max_err={:.3f}",
                    float(np.degrees(current_yaw)),
                    float(np.degrees(expected_yaw)),
                    joint_error,
                )
                return
            time.sleep(0.02)

        last_yaw_str = "none" if last_yaw is None else f"{np.degrees(last_yaw):.1f}"
        raise RuntimeError(
            "Timed out waiting for motion-init low state yaw: "
            f"last={last_yaw_str} deg, expected={np.degrees(expected_yaw):.1f} deg, "
            f"joint_max_err={last_joint_error}"
        )

    def _capture_policy_state(self):
        state = super()._capture_policy_state()
        state.update(
            {
                "motion_command_0": self.motion_command_0.copy(),
                "ref_quat_xyzw_0": self.ref_quat_xyzw_0.copy(),
                "motion_ref_pos_xyz_0": None
                if self.motion_ref_pos_xyz_0 is None
                else self.motion_ref_pos_xyz_0.copy(),
                "policy_action_output_name": self._policy_action_output_name,
                "policy_returns_reference_outputs": self._policy_returns_reference_outputs,
            }
        )
        return state

    def _restore_policy_state(self, state):
        super()._restore_policy_state(state)
        self.motion_command_0 = state["motion_command_0"].copy()
        self.ref_quat_xyzw_0 = state["ref_quat_xyzw_0"].copy()
        self.motion_ref_pos_xyz_0 = (
            None if state["motion_ref_pos_xyz_0"] is None else state["motion_ref_pos_xyz_0"].copy()
        )
        self._policy_action_output_name = state.get("policy_action_output_name", "actions")
        self._policy_returns_reference_outputs = state.get("policy_returns_reference_outputs", True)
        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None
        self._last_clock_reading = None
        self._last_policy_inference_clock_ms = None
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0

    def _on_policy_switched(self, model_path: str):
        super()._on_policy_switched(model_path)
        self.motion_command_t = self.motion_command_0.copy()
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self.motion_ref_pos_xyz_t = (
            None if self.motion_ref_pos_xyz_0 is None else self.motion_ref_pos_xyz_0.copy()
        )
        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None
        self._last_clock_reading = None
        self._last_policy_inference_clock_ms = None
        self._stiff_hold_active = True
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0

    def get_init_target(self, robot_state_data):
        """Get initialization target joint positions."""
        dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
        if self.get_ready_state:
            target_dof_pos = self._stiff_hold_q

            q_target = dof_pos + (target_dof_pos - dof_pos) * (self.init_count / 500)
            self.init_count += 1
            return q_target
        return dof_pos

    def get_current_obs_buffer_dict(self, robot_state_data):
        robot_state_data = self._augment_robot_state_with_sim_state(robot_state_data)
        current_obs_buffer_dict = {}
        required_terms = {term for terms in self.obs_dict.values() for term in terms}

        if "motion_command" in required_terms:
            current_obs_buffer_dict["motion_command"] = self.motion_command_t

        sparse_terms = {
            "sparse_target_root_trajectory_command",
            "sparse_target_root_trajectory_command_contact_aware",
        }
        if required_terms & sparse_terms:
            sparse_root_command = self._get_sparse_target_root_trajectory_command(robot_state_data)
            if "sparse_target_root_trajectory_command" in required_terms:
                current_obs_buffer_dict["sparse_target_root_trajectory_command"] = sparse_root_command
            if "sparse_target_root_trajectory_command_contact_aware" in required_terms:
                if self._external_sparse_root_command_mode:
                    current_obs_buffer_dict["sparse_target_root_trajectory_command_contact_aware"] = sparse_root_command
                else:
                    current_obs_buffer_dict["sparse_target_root_trajectory_command_contact_aware"] = (
                        self._get_sparse_target_root_trajectory_command_contact_aware(sparse_root_command)
                    )
            self._write_sparse_root_command_status(current_obs_buffer_dict)

        if "motion_ref_ori_b" in required_terms:
            motion_ref_ori = xyzw_to_wxyz(self.ref_quat_xyzw_t)  # wxyz
            motion_ref_ori = self._remove_yaw_offset(motion_ref_ori, self.motion_yaw_offset)

            robot_ref_ori = self._get_ref_body_orientation_in_world(robot_state_data)  #  wxyz
            robot_ref_ori = self._remove_yaw_offset(robot_ref_ori, self.robot_yaw_offset)

            motion_ref_ori_b = matrix_from_quat(subtract_frame_transforms(robot_ref_ori, motion_ref_ori))
            current_obs_buffer_dict["motion_ref_ori_b"] = motion_ref_ori_b[..., :2].reshape(1, -1)

        if "base_lin_vel" in required_terms:
            current_obs_buffer_dict["base_lin_vel"] = robot_state_data[:, 7 + self.num_dofs : 7 + self.num_dofs + 3]

        if "base_ang_vel" in required_terms:
            current_obs_buffer_dict["base_ang_vel"] = robot_state_data[
                :, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6
            ]

        if "dof_pos" in required_terms:
            current_obs_buffer_dict["dof_pos"] = robot_state_data[:, 7 : 7 + self.num_dofs] - self.default_dof_angles

        if "dof_vel" in required_terms:
            current_obs_buffer_dict["dof_vel"] = robot_state_data[
                :, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs
            ]

        if "actions" in required_terms:
            current_obs_buffer_dict["actions"] = self.last_policy_action
        if "pickup_button" in required_terms:
            current_obs_buffer_dict["pickup_button"] = np.array([[self._pickup_button_command]], dtype=np.float32)
        if "drop_button" in required_terms:
            current_obs_buffer_dict["drop_button"] = np.array([[self._drop_button_command]], dtype=np.float32)
        if "cam_depth" in required_terms:
            current_obs_buffer_dict["cam_depth"] = self._get_depth_image_obs()
        if self._policy_debug_path:
            self._last_current_obs_buffer_dict = {
                key: np.asarray(value, dtype=np.float32).copy()
                for key, value in current_obs_buffer_dict.items()
                if key
                in {
                    "sparse_target_root_trajectory_command",
                    "sparse_target_root_trajectory_command_contact_aware",
                    "base_lin_vel",
                    "base_ang_vel",
                    "dof_pos",
                    "dof_vel",
                    "actions",
                    "pickup_button",
                    "drop_button",
                    "cam_depth",
                }
            }

        return current_obs_buffer_dict

    def _write_sparse_root_command_status(self, current_obs_buffer_dict: dict[str, np.ndarray]) -> None:
        if self._policy_command_status_path is None:
            return

        now = time.time()
        if now < self._policy_command_status_next_time:
            return
        self._policy_command_status_next_time = now + self._policy_command_status_period

        term = None
        if "sparse_target_root_trajectory_command_contact_aware" in current_obs_buffer_dict:
            term = "sparse_target_root_trajectory_command_contact_aware"
        elif "sparse_target_root_trajectory_command" in current_obs_buffer_dict:
            term = "sparse_target_root_trajectory_command"
        if term is None:
            return

        command = np.asarray(current_obs_buffer_dict[term], dtype=np.float32).reshape(-1)[:3]
        payload = {
            "timestamp": now,
            "term": term,
            "command": [float(value) for value in command],
            "manual_offset": [float(value) for value in self._manual_sparse_root_command_offset.reshape(-1)[:3]],
            "joystick_offset": [float(value) for value in self._joystick_sparse_root_command_offset.reshape(-1)[:3]],
            "external_sparse_root_command_mode": bool(self._external_sparse_root_command_mode),
            "force_zero_sparse_root_command": bool(self._force_zero_sparse_root_command),
            "motion_clip_progressing": bool(self.motion_clip_progressing),
            "motion_timestep": int(self.motion_timestep),
        }
        if "pickup_button" in self.obs_dims:
            payload["pickup_button"] = float(self._pickup_button_command)
        if "drop_button" in self.obs_dims:
            payload["drop_button"] = float(self._drop_button_command)

        try:
            self._policy_command_status_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._policy_command_status_path.with_name(f".{self._policy_command_status_path.name}.tmp")
            tmp_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            tmp_path.replace(self._policy_command_status_path)
        except OSError as exc:
            if not self._logged_policy_command_status_error:
                logger.warning("Failed to write policy command status HUD file: {}", exc)
                self._logged_policy_command_status_error = True

    def _get_sparse_target_root_trajectory_command(self, robot_state_data) -> np.ndarray:
        if self._force_zero_sparse_root_command:
            if not self._logged_zero_sparse_root_command:
                logger.info("Using zero sparse root command.")
                self._logged_zero_sparse_root_command = True
            return np.zeros((1, 3), dtype=np.float32)

        external_sparse_command = self._manual_sparse_root_command_offset + self._joystick_sparse_root_command_offset
        if self._external_sparse_root_command_mode:
            return external_sparse_command.astype(np.float32, copy=False)

        target_pos, target_quat_wxyz = self._get_target_root_pose()
        if target_pos is None or target_quat_wxyz is None:
            return np.zeros((1, 3), dtype=np.float32)

        robot_root_xy = robot_state_data[:, :2]
        target_xy = target_pos[:, :2].astype(np.float32, copy=True)
        if (
            self._motion_root_command_origin_xy is not None
            and not self.config.task.use_sim_state
            and np.linalg.norm(robot_root_xy) < 1e-5
        ):
            if not self._logged_motion_local_sparse_root_command:
                logger.info("Using motion-local sparse root XY command because low-state root XY is zero.")
                self._logged_motion_local_sparse_root_command = True
            target_xy = target_xy - self._motion_root_command_origin_xy

        delta_xy_world = target_xy - robot_root_xy

        robot_yaw = quat_to_rpy(robot_state_data[0, 3:7])[2]
        target_yaw = quat_to_rpy(target_quat_wxyz[0])[2]

        c = np.cos(-robot_yaw)
        s = np.sin(-robot_yaw)
        delta_x_body = c * delta_xy_world[:, 0] - s * delta_xy_world[:, 1]
        delta_y_body = s * delta_xy_world[:, 0] + c * delta_xy_world[:, 1]
        yaw_error = (target_yaw - robot_yaw + np.pi) % (2 * np.pi) - np.pi

        return np.array([[delta_x_body[0], delta_y_body[0], yaw_error]], dtype=np.float32)

    def _get_sparse_target_root_trajectory_command_contact_aware(self, base_command: np.ndarray) -> np.ndarray:
        if self._motion_object_pos_w is None or self._motion_root_pos_w is None:
            return base_command
        carry_start, carry_end = self._get_contact_aware_carry_window()
        idx = self._motion_frame_index()
        if carry_start <= idx < carry_end:
            return base_command
        return np.zeros_like(base_command)

    def _get_contact_aware_carry_window(self) -> tuple[int, int]:
        if self._contact_aware_carry_window is not None:
            return self._contact_aware_carry_window
        if self._motion_object_pos_w is None or self._motion_root_pos_w is None:
            self._contact_aware_carry_window = (0, 0)
            return self._contact_aware_carry_window

        if self._contact_aware_window_mode == "peak_height":
            height = self._smooth_1d_edge_padded(
                self._motion_object_pos_w[:, 2],
                self._contact_aware_peak_height_smoothing_steps,
            )
            h_min = float(np.min(height))
            h_range = max(float(np.max(height) - h_min), 0.0)
            alpha = max(0.0, min(float(self._contact_aware_peak_height_alpha), 1.0))
            threshold = h_min + h_range * alpha
            high_mask = height >= threshold

            carry_start = self._first_sustained_true_index(high_mask, 5)
            if carry_start is None:
                high_indices = np.flatnonzero(high_mask)
                carry_start = int(high_indices[0]) if high_indices.size else int(np.argmax(height))

            peak_step = int(np.argmax(height))
            carry_end = self._first_sustained_true_index_from(
                ~high_mask,
                5,
                min(peak_step + 1, height.shape[0]),
            )
            if carry_end is None:
                carry_end = height.shape[0]
        else:
            rel_z = self._motion_object_pos_w[:, 2] - self._motion_root_pos_w[:, 2]
            z_min = float(np.min(rel_z))
            z_range = max(float(np.max(rel_z) - z_min), 0.0)
            threshold = z_min + max(0.10, z_range * 0.35)
            lifted_mask = rel_z >= threshold

            carry_start = self._first_sustained_true_index(lifted_mask, 5)
            if carry_start is None:
                lifted_indices = np.flatnonzero(lifted_mask)
                carry_start = int(lifted_indices[0]) if lifted_indices.size else int(np.argmax(rel_z))

            carry_end = self._first_sustained_true_index_from(
                ~lifted_mask,
                5,
                min(int(carry_start) + 1, rel_z.shape[0]),
            )
            if carry_end is None:
                carry_end = rel_z.shape[0]

        self._contact_aware_carry_window = (int(carry_start), int(carry_end))
        logger.info(
            "[WBT] Contact-aware sparse root active window: [{}, {}) mode={}",
            self._contact_aware_carry_window[0],
            self._contact_aware_carry_window[1],
            self._contact_aware_window_mode,
        )
        return self._contact_aware_carry_window

    @staticmethod
    def _smooth_1d_edge_padded(values: np.ndarray, window_steps: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        window_steps = max(1, int(window_steps))
        if values.size == 0 or window_steps <= 1:
            return values
        left_pad = window_steps // 2
        right_pad = window_steps - 1 - left_pad
        padded = np.concatenate(
            [
                np.repeat(values[:1], left_pad),
                values,
                np.repeat(values[-1:], right_pad),
            ]
        )
        kernel = np.full((window_steps,), 1.0 / float(window_steps), dtype=np.float32)
        return np.convolve(padded, kernel, mode="valid")

    @staticmethod
    def _first_sustained_true_index(mask: np.ndarray, steps: int) -> int | None:
        return WholeBodyTrackingPolicy._first_sustained_true_index_from(mask, steps, 0)

    @staticmethod
    def _first_sustained_true_index_from(mask: np.ndarray, steps: int, start_idx: int) -> int | None:
        if steps <= 1:
            indices = np.flatnonzero(mask[start_idx:])
            return int(indices[0] + start_idx) if indices.size else None

        run = 0
        run_start = start_idx
        for idx in range(start_idx, mask.shape[0]):
            if bool(mask[idx]):
                if run == 0:
                    run_start = idx
                run += 1
                if run >= steps:
                    return run_start
            else:
                run = 0
        return None

    def _get_target_root_pose(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self._motion_root_pos_w is not None and self._motion_root_quat_wxyz is not None:
            idx = self._motion_frame_index()
            return self._motion_root_pos_w[idx : idx + 1], self._motion_root_quat_wxyz[idx : idx + 1]

        if self.motion_ref_pos_xyz_t is None or self.ref_quat_xyzw_t is None or self.motion_command_t is None:
            return None, None

        ref_rel_pos, ref_rel_quat_wxyz = self.pinocchio_robot.fk_ref_body_pose_with_root_identity(
            self.motion_command_t[:, : self.num_dofs]
        )
        target_ref_pos = self.motion_ref_pos_xyz_t[:, :3].astype(np.float32, copy=False)
        target_ref_quat_wxyz = xyzw_to_wxyz(self.ref_quat_xyzw_t).astype(np.float32, copy=False)
        root_quat_wxyz = quat_mul(target_ref_quat_wxyz, quat_inverse(ref_rel_quat_wxyz)).astype(
            np.float32, copy=False
        )
        root_pos = target_ref_pos - quat_apply(root_quat_wxyz, ref_rel_pos)
        return root_pos.astype(np.float32, copy=False), root_quat_wxyz

    def _get_depth_image_obs(self) -> np.ndarray:
        if "cam_depth" not in self.obs_dims:
            return np.zeros((1, 0), dtype=np.float32)

        camera_config = self.config.camera
        if camera_config is None:
            raise ValueError("perception_obs requires a camera configuration.")

        expected_shape = (
            len(camera_config.poses),
            1,
            camera_config.props.resized_height,
            camera_config.props.resized_width,
        )

        if self._depth_img_array is None:
            try:
                self._depth_img_shm = shared_memory.SharedMemory(name="depth_img_shm")
                resource_tracker.unregister(self._depth_img_shm._name, "shared_memory")
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "perception_obs requires shared memory 'depth_img_shm'. Start the MuJoCo image server first."
                ) from exc
            self._depth_img_array = np.ndarray(expected_shape, dtype=np.float32, buffer=self._depth_img_shm.buf)
            logger.info("[WBT] Depth shared memory attached: shape={}", expected_shape)

        if getattr(self, "_deployment_audit", None) is not None:
            self._audit_depth_read_monotonic = time.monotonic()
        flattened = self._depth_img_array.copy().reshape(1, -1).astype(np.float32, copy=False)
        expected_dim = self.obs_dims["cam_depth"]
        if flattened.shape[1] != expected_dim:
            raise ValueError(
                f"Depth observation shape mismatch: got {flattened.shape[1]} values, expected {expected_dim}."
            )
        return flattened

    def prepare_obs_for_rl(self, robot_state_data):
        group_outputs = self._prepare_group_observations(robot_state_data)

        if "actor_obs" in group_outputs:
            actor_obs = group_outputs["actor_obs"].astype(np.float32, copy=False)
        else:
            parts = [group_outputs[group].astype(np.float32, copy=False) for group in self._actor_obs_group_names()]
            if not parts:
                raise KeyError("WBT policy needs actor_obs or split actor observation groups.")
            actor_obs = np.concatenate(parts, axis=1).astype(np.float32, copy=False)

        obs = {"obs": actor_obs, "actor_obs": actor_obs}
        if "perception_obs" in group_outputs:
            obs["perception_obs"] = group_outputs["perception_obs"].astype(np.float32, copy=False)
        return obs

    def rl_inference(self, robot_state_data):
        # prepare obs, run policy inference
        if not self.motion_clip_progressing:
            # Keep motion index pinned at the start while waiting to trigger the clip.
            self.motion_timestep = 0
            self.motion_start_timestep = None
            self._last_clock_reading = None
            self._last_policy_inference_clock_ms = None
        elif self.use_sim_time:
            current_clock = self.clock_sub.get_clock()
            if (
                self._last_policy_inference_clock_ms is not None
                and current_clock - self._last_policy_inference_clock_ms < self.timestep_interval_ms
            ):
                return self.scaled_policy_action
            self._last_policy_inference_clock_ms = current_clock

        audit = getattr(self, "_deployment_audit", None)
        audit_started = time.monotonic() if audit is not None else None
        obs = self.prepare_obs_for_rl(robot_state_data)
        input_feed = {}
        for name in self.onnx_input_names:
            if name == "time_step":
                input_feed[name] = np.array([[self._motion_frame_index()]], dtype=np.float32)
            elif name == "obs":
                input_feed[name] = obs["obs"]
            else:
                input_feed[name] = obs[name]
        if not self._policy_returns_reference_outputs:
            policy_action = self.policy(input_feed)
        else:
            policy_action, self.motion_command_t, self.ref_quat_xyzw_t, self.motion_ref_pos_xyz_t = self.policy(
                input_feed
            )

        # store last policy action
        self.last_policy_action = policy_action.copy()
        # scale policy action
        self.scaled_policy_action = policy_action * self.policy_action_scales
        if self._use_motion_data_as_q_target and self._motion_joint_pos is not None:
            target_joint_pos = self._motion_joint_pos[self._motion_frame_index() : self._motion_frame_index() + 1]
            self.scaled_policy_action = target_joint_pos.astype(np.float32, copy=False) - self.default_dof_angles
            if not self._logged_motion_data_q_target:
                logger.info("Using motion .npz joint_pos directly as MuJoCo q_target for diagnostic rollout.")
                self._logged_motion_data_q_target = True
        self._write_policy_debug(input_feed, obs, policy_action, self.scaled_policy_action, robot_state_data)
        if audit is not None:
            self._record_deployment_evidence(input_feed, policy_action, robot_state_data, audit_started)

        # update motion timestep
        if self.motion_clip_progressing:
            if self.use_sim_time:
                self._update_clock()
            else:
                self.motion_timestep += 1
        return self.scaled_policy_action

    def _record_deployment_evidence(self, input_feed, policy_action, robot_state_data, started):
        # This is the requested policy target, not an acknowledgement from a motor.
        # Logging never clips actions, clears buttons, or changes the depth path.
        self._deployment_audit.record({
            **{f"input__{name}": value for name, value in input_feed.items()},
            "policy_action": policy_action, "scaled_policy_action": self.scaled_policy_action,
            "requested_q_target": self.scaled_policy_action + self.default_dof_angles,
            "robot_state": robot_state_data, "action_scales": self.policy_action_scales,
            "motor_kp": np.asarray(self.robot_config.motor_kp) * self.interface.kp_level,
            "motor_kd": np.asarray(self.robot_config.motor_kd) * self.interface.kd_level,
            "manual_command": self._manual_sparse_root_command_offset,
            "joystick_command": self._joystick_sparse_root_command_offset,
        }, {
            "inference_started_monotonic": started,
            "inference_finished_monotonic": time.monotonic(),
            "depth_read_monotonic": getattr(self, "_audit_depth_read_monotonic", None),
            "drop_button": self._drop_button_command,
            "use_policy_action": self.use_policy_action, "get_ready_state": self.get_ready_state,
            "motion_clip_progressing": self.motion_clip_progressing,
            "motion_frame_index": self._motion_frame_index(),
            "depth_shm_name": "depth_img_shm",
            "training_depth_profile": None,
        })

    def close_deployment_audit(self):
        audit = getattr(self, "_deployment_audit", None)
        if audit is not None:
            audit.close()

    @staticmethod
    def _array_stats(value: np.ndarray) -> dict:
        arr = np.asarray(value, dtype=np.float32)
        flat = arr.reshape(-1)
        finite = np.isfinite(flat)
        if not bool(finite.all()):
            flat = flat[finite]
        if flat.size == 0:
            return {"shape": list(arr.shape), "finite": bool(finite.all()), "count": int(arr.size)}
        return {
            "shape": list(arr.shape),
            "finite": bool(finite.all()),
            "count": int(arr.size),
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "absmax": float(np.max(np.abs(flat))),
            "nonzero": int(np.count_nonzero(np.abs(flat) > 1e-6)),
        }

    def _write_policy_debug(
        self,
        input_feed: dict[str, np.ndarray],
        obs: dict[str, np.ndarray],
        policy_action: np.ndarray,
        scaled_policy_action: np.ndarray,
        robot_state_data: np.ndarray,
    ) -> None:
        if not self._policy_debug_path or self._policy_debug_count >= self._policy_debug_limit:
            return

        try:
            if self._policy_debug_file is None:
                debug_path = Path(self._policy_debug_path)
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                self._policy_debug_file = debug_path.open("a", encoding="utf-8", buffering=1)

            q_actual = np.asarray(robot_state_data[:, 7 : 7 + self.num_dofs], dtype=np.float32)
            q_target = np.asarray(scaled_policy_action, dtype=np.float32) + self.default_dof_angles
            terms = getattr(self, "_last_current_obs_buffer_dict", {})
            payload = {
                "step": int(self._policy_debug_count),
                "motion_timestep": int(self.motion_timestep),
                "motion_frame_index": int(self._motion_frame_index()),
                "clock_ms": self.clock_sub.get_clock(),
                "motion_clip_progressing": bool(self.motion_clip_progressing),
                "input": {name: self._array_stats(value) for name, value in input_feed.items()},
                "obs": {name: self._array_stats(value) for name, value in obs.items()},
                "terms": {name: self._array_stats(value) for name, value in terms.items()},
                "policy_action": self._array_stats(policy_action),
                "scaled_policy_action": self._array_stats(scaled_policy_action),
                "q_target": self._array_stats(q_target),
                "q_actual": self._array_stats(q_actual),
                "q_target_first": q_target.reshape(-1)[:8].astype(float).tolist(),
                "q_actual_first": q_actual.reshape(-1)[:8].astype(float).tolist(),
            }
            for name in (
                "sparse_target_root_trajectory_command",
                "sparse_target_root_trajectory_command_contact_aware",
                "base_lin_vel",
                "base_ang_vel",
            ):
                if name in terms:
                    payload[name] = np.asarray(terms[name]).reshape(-1).astype(float).tolist()

            depth = terms.get("cam_depth")
            if depth is not None and depth.size:
                flat = np.asarray(depth, dtype=np.float32).reshape(-1)
                payload["cam_depth_quantiles"] = [
                    float(value) for value in np.quantile(flat, [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
                ]

            self._policy_debug_file.write(json.dumps(payload, sort_keys=True) + "\n")
            self._policy_debug_count += 1
        except Exception as exc:
            logger.warning("Failed to write policy debug trace: {}", exc)
            self._policy_debug_path = ""

    def _get_manual_command(self, robot_state_data):
        # TODO: instead of adding kp/kd_override in def _set_motor_command,
        # just use the motor_kp/motor_kd when calling it in _fill_motor_commands
        if not self._stiff_hold_active:
            return None
        return {
            "q": self._stiff_hold_q.copy(),
            "kp": self._stiff_hold_kp,
            "kd": self._stiff_hold_kd,
        }

    def _reset_policy_io_state(self, reason: str) -> None:
        """Forget what the policy last did and reset operator toggles to startup values.

        On the robot, restarting after a stop fed the network the last action from
        before the stop (|a| 3.2 vs 0 on a fresh launch) and a drop_button that had
        been toggled on minutes earlier. Neither is what a start should look like.
        """
        self.last_policy_action.fill(0.0)
        self.scaled_policy_action.fill(0.0)
        if (
            self._pickup_button_command != self._pickup_button_startup
            or self._drop_button_command != self._drop_button_startup
        ):
            self.logger.info(
                colored(
                    f"{reason}: pickup_button -> {self._pickup_button_startup:.0f}, "
                    f"drop_button -> {self._drop_button_startup:.0f} (startup values)",
                    "blue",
                )
            )
        self._pickup_button_command = self._pickup_button_startup
        self._drop_button_command = self._drop_button_startup

    def _handle_init_state(self):
        super()._handle_init_state()
        self._reset_policy_io_state("init state")

    def _handle_start_policy(self):
        super()._handle_start_policy()
        self._stiff_hold_active = False
        # A restart must look like a first start to the network: no stale previous
        # action, and history filled from the robot's current state rather than
        # from whatever was in the buffers when it was stopped.
        self.last_policy_action.fill(0.0)
        self.scaled_policy_action.fill(0.0)
        try:
            self.warm_start_observation_history()
        except Exception as exc:  # never let a warm-start problem block starting
            self.logger.warning(f"observation history warm-start skipped: {exc}")
        self._capture_robot_yaw_offset()
        self._capture_motion_yaw_offset(self.ref_quat_xyzw_0)

    def _update_clock(self):
        # Use synchronized clock with motion-relative timing
        current_clock = self.clock_sub.get_clock()
        if self.motion_start_timestep is None:
            # Motion just started; anchor to the first received clock tick.
            self.motion_start_timestep = current_clock
        elif self._last_clock_reading is not None and current_clock < self._last_clock_reading:
            # Simulator clock jumped backwards (e.g., reset). Re-anchor start time while preserving progress.
            offset_ms = round(self.motion_timestep * self.timestep_interval_ms)
            self.logger.warning("Clock sync returned earlier timestamp; adjusting motion timing anchor.")
            self.motion_start_timestep = current_clock - offset_ms
        self._last_clock_reading = current_clock
        elapsed_ms = current_clock - self.motion_start_timestep
        if self.motion_timestep == 0 and int(elapsed_ms // self.timestep_interval_ms) > 1:
            self.logger.warning(
                "Still at the beginning but the clock jumped ahead: elapsed_ms={elapsed_ms}, self.timestep_interval_ms="
                "{timestep_interval_ms}, self.motion_timestep={motion_timestep}. "
                "Re-anchoring to the current timestamp so the motion always starts from frame 0.",
                elapsed_ms=elapsed_ms,
                timestep_interval_ms=self.timestep_interval_ms,
                motion_timestep=self.motion_timestep,
            )
            # Still at the beginning but the clock jumped ahead (e.g., due to waiting before start).
            # Re-anchor to the current timestamp so the motion always starts from frame 0.
            self.motion_start_timestep = current_clock
            self._last_clock_reading = current_clock
            self.motion_timestep = 0
            return
        previous_motion_timestep = self.motion_timestep
        self.motion_timestep = int(elapsed_ms // self.timestep_interval_ms)
        if self.motion_timestep != previous_motion_timestep:
            self.logger.info(
                "Motion timestep advanced from {previous_motion_timestep} to {motion_timestep}",
                previous_motion_timestep=previous_motion_timestep,
                motion_timestep=self.motion_timestep,
            )

    def _handle_stop_policy(self):
        """Handle stop policy action."""
        self.use_policy_action = False
        self.get_ready_state = False
        self._stiff_hold_active = True
        self.logger.info("Actions set to stiff startup command")
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0

        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None  # Reset motion start time
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self.motion_command_t = self.motion_command_0.copy()
        self._last_clock_reading = None
        self._last_policy_inference_clock_ms = None
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0
        self._reset_policy_io_state("policy stopped")

    def _handle_start_motion_clip(self):
        """Handle start motion clip action."""
        self.clock_sub.reset_origin()
        self.motion_clip_progressing = True
        # Capture motion-specific start timestep for policy-level timing control
        self.motion_start_timestep = None  # will be set in rl_inference
        self.motion_timestep = 0  # Reset to start from beginning of motion
        self._last_clock_reading = None
        self._last_policy_inference_clock_ms = None
        self.logger.info(colored("Starting motion clip", "blue"))

    def _handle_sparse_root_keyboard_command(self, keycode: str) -> bool:
        step = 0.025
        if keycode == "w":
            self._manual_sparse_root_command_offset[0, 0] += step
        elif keycode == "s":
            self._manual_sparse_root_command_offset[0, 0] -= step
        elif keycode == "a":
            self._manual_sparse_root_command_offset[0, 1] += step
        elif keycode == "d":
            self._manual_sparse_root_command_offset[0, 1] -= step
        elif keycode == "q":
            self._manual_sparse_root_command_offset[0, 2] -= step
        elif keycode == "e":
            self._manual_sparse_root_command_offset[0, 2] += step
        elif keycode == "z":
            self._manual_sparse_root_command_offset.fill(0.0)
        else:
            return False

        self.logger.info(
            colored(
                "Sparse root command offset: x={:.2f}, y={:.2f}, yaw={:.2f}".format(
                    float(self._manual_sparse_root_command_offset[0, 0]),
                    float(self._manual_sparse_root_command_offset[0, 1]),
                    float(self._manual_sparse_root_command_offset[0, 2]),
                ),
                "blue",
            )
        )
        return True

    def _handle_pickup_button_keyboard_command(self, keycode: str) -> bool:
        if keycode != "f" or "pickup_button" not in self.obs_dims:
            return False
        if self._pickup_button_key_down:
            return True

        self._pickup_button_key_down = True
        self._pickup_button_command = 0.0 if self._pickup_button_command >= 0.5 else 1.0
        self.logger.info(colored(f"Pickup button command: {self._pickup_button_command:.0f}", "blue"))
        return True

    def _handle_drop_button_keyboard_command(self, keycode: str) -> bool:
        if keycode != "g":
            return False
        if "drop_button" not in self.obs_dims:
            if not self._drop_button_key_down and not self._logged_missing_drop_button_key:
                self.logger.warning("Active policy has no drop_button observation; ignoring g.")
                self._logged_missing_drop_button_key = True
            self._drop_button_key_down = True
            return True
        if self._drop_button_key_down:
            return True

        self._drop_button_key_down = True
        self._drop_button_command = 0.0 if self._drop_button_command >= 0.5 else 1.0
        self.logger.info(colored(f"Drop button command: {self._drop_button_command:.0f}", "blue"))
        return True

    def _handle_drop_button_joystick_command(self, cur_key: str) -> bool:
        if cur_key != "X":
            return False
        if "drop_button" not in self.obs_dims:
            if not self._logged_missing_drop_button_key:
                self.logger.warning("Active policy has no drop_button observation; ignoring X.")
                self._logged_missing_drop_button_key = True
            return True

        self._drop_button_command = 0.0 if self._drop_button_command >= 0.5 else 1.0
        self.logger.info(colored(f"Drop button command: {self._drop_button_command:.0f}", "blue"))
        return True

    def handle_keyboard_release(self, keycode):
        if keycode == "f":
            self._pickup_button_key_down = False
        if keycode == "g":
            self._drop_button_key_down = False
        super().handle_keyboard_release(keycode)

    def _update_sparse_root_joystick_command(self) -> None:
        wc_msg = self.interface.get_joystick_msg()
        if wc_msg is None:
            self._joystick_sparse_root_command_offset.fill(0.0)
            return
        if getattr(wc_msg, "keys", 0) != 0:
            self._joystick_sparse_root_command_offset.fill(0.0)
            return

        deadband = 0.05
        xy_scale = 0.07
        yaw_scale = 0.1

        def apply_deadband(value: float) -> float:
            return value if abs(value) > deadband else 0.0

        lx = apply_deadband(float(getattr(wc_msg, "lx", 0.0)))
        ly = apply_deadband(float(getattr(wc_msg, "ly", 0.0)))
        rx = apply_deadband(float(getattr(wc_msg, "rx", 0.0)))

        self._joystick_sparse_root_command_offset[0, 0] = np.sign(ly) * xy_scale if ly else 0.0
        self._joystick_sparse_root_command_offset[0, 1] = -lx * xy_scale
        self._joystick_sparse_root_command_offset[0, 2] = -rx * yaw_scale

    def process_joystick_input(self):
        super().process_joystick_input()
        self._update_sparse_root_joystick_command()

    def handle_keyboard_button(self, keycode):
        """Add WBT keyboard controls for motion clips and sparse root command."""
        if keycode in {"m", "s"} and not self.motion_clip_progressing:
            self.clock_sub.reset_origin()
            self._handle_start_motion_clip()
        elif self._handle_pickup_button_keyboard_command(keycode):
            pass
        elif self._handle_drop_button_keyboard_command(keycode):
            pass
        elif self._handle_sparse_root_keyboard_command(keycode):
            pass
        else:
            super().handle_keyboard_button(keycode)

    def handle_joystick_button(self, cur_key):
        """Handle joystick button presses for WBT-specific controls."""
        if cur_key == "start":
            # Start playing motion clip
            self._handle_start_motion_clip()
        elif self._handle_drop_button_joystick_command(cur_key):
            pass
        else:
            # Delegate all other buttons to base class
            super().handle_joystick_button(cur_key)
        super()._print_control_status()

    def _capture_robot_yaw_offset(self):
        """Capture robot yaw when policy starts to use as reference offset."""
        robot_state_data = self.interface.get_low_state()
        if robot_state_data is None:
            self.robot_yaw_offset = 0.0
            self.logger.warning("Unable to capture robot yaw offset - missing robot state.")
            return

        robot_ref_ori = self._get_ref_body_orientation_in_world(robot_state_data)  # wxyz
        yaw = self._quat_yaw(robot_ref_ori)
        self.robot_yaw_offset = yaw
        self.logger.info(colored(f"Robot yaw offset captured at {np.degrees(yaw):.1f} deg", "blue"))

    def _capture_motion_yaw_offset(self, ref_quat_xyzw_0: np.ndarray) -> float:
        """Capture motion yaw when policy starts to use as reference offset."""
        self.motion_yaw_offset = self._quat_yaw(xyzw_to_wxyz(ref_quat_xyzw_0))
        self.logger.info(colored(f"Motion yaw offset captured at {np.degrees(self.motion_yaw_offset):.1f} deg", "blue"))

    def _remove_yaw_offset(self, quat_wxyz: np.ndarray, yaw_offset: float) -> np.ndarray:
        """Remove stored yaw offset from robot orientation quaternion."""
        if abs(yaw_offset) < 1e-6:
            return quat_wxyz
        yaw_quat = rpy_to_quat((0.0, 0.0, -yaw_offset)).reshape(1, 4)
        yaw_quat = np.broadcast_to(yaw_quat, quat_wxyz.shape)
        return quat_mul(yaw_quat, quat_wxyz)

    @staticmethod
    def _quat_yaw(quat_wxyz: np.ndarray) -> float:
        """Extract yaw angle from quaternion array of shape (1, 4)."""
        quat_flat = quat_wxyz.reshape(-1, 4)[0]
        _, _, yaw = quat_to_rpy(quat_flat)
        return float(yaw)
