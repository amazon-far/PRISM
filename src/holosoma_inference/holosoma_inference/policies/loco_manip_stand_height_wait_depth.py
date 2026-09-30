from multiprocessing import resource_tracker, shared_memory
import threading
import time
from collections import deque

import numpy as np
from termcolor import colored

from .locomotion import LocomotionPolicy


class LocoManipStandHeightWaitDepthPolicy(LocomotionPolicy):
    """
    Locomotion-manipulation policy with base height control, waist commands,
    and depth camera integration. Compatible with holosoma inference system.

    Inherits from LocomotionPolicy to maintain compatibility with holosoma
    while adding upper body control, RNN support, and depth camera integration.
    """

    def __init__(self, config):
        # Initialize parent locomotion policy
        super().__init__(config)

        # init default upper body positions
        self._init_upper_body_pos()

        # Initialize depth image client
        self._init_depth_shm()

        # Initialize pre-fixed speed command
        self._init_fixed_forward_speed_command()

    def _initialize_history_state(self):
        # KEEP
        """Initialize history state for observation buffers."""
        # TODO: to incorporate history length into the base policy
        super()._initialize_history_state()
        self.obs_history_buffers_per_group = {}
        for group in self.obs_terms_sorted:
            self.obs_history_buffers_per_group[group] = deque(maxlen=self.history_length_dict[group])

    def _init_upper_body_pos(self):
        # KEEP
        """Initialize upper body controller with threading support."""
        # Reference upper DOF positions
        robot_config = self.config.robot

        self.upper_dof_indices = [robot_config.dof_names.index(dof) for dof in robot_config.dof_names_upper_body]

        # all zeros for old default positions; beyondmimic has non-zero default upper body positions
        self.ref_upper_dof_pos = np.array([robot_config.default_dof_angles[i] for i in self.upper_dof_indices]).reshape(
            1, -1
        )

    def _init_depth_shm(self):
        # KEEP
        """Initialize depth image client using shared memory."""
        # Create shared memory for depth images
        camera_config = self.config.camera
        img_shape = [camera_config.props.resized_height, camera_config.props.resized_width]
        channels = 1  # depth camera is always 1 channel
        num_cameras = len(camera_config.poses)
        expected_shape = [num_cameras, channels, img_shape[0], img_shape[1]]
        print(f"expected_shape: {expected_shape}")

        # TDDO: enable this after debugging
        self.depth_img_shm = shared_memory.SharedMemory(name="depth_img_shm")
        resource_tracker.unregister(self.depth_img_shm._name, "shared_memory")
        self.depth_img_array = np.ndarray(expected_shape, dtype=np.float32, buffer=self.depth_img_shm.buf)
        print(f"[Policy] Depth image client initialized with shared memory: {self.depth_img_shm.name}")
        # self.depth_img_array = np.zeros(expected_shape, dtype=np.float32)

    def _init_fixed_forward_speed_command(self):
        # KEEP
        """Initialize pre-fixed speed command."""
        # default forward speed is 0.5 m/s; only influence the forward speed.
        self.fixed_forward_speed = 0.5
        self.fixed_forward_speed_enabled = False


    def get_current_obs_buffer_dict(self, robot_state_data):
        # KEEP
        current_obs_buffer_dict = super().get_current_obs_buffer_dict(robot_state_data)

        current_obs_buffer_dict["command_base_height"] = self.base_height_command
        current_obs_buffer_dict["command_waist_dofs"] = self.waist_dofs_command
        current_obs_buffer_dict["ref_upper_dof_pos"] = self.ref_upper_dof_pos

        # Get depth image from shared memory
        depth_images = self.depth_img_array.copy()  # Shape: (2, 1, 27, 48)

        # Visualize the two depth images
        # self.visualize_depth_images(depth_images)

        current_obs_buffer_dict["cam_depth"] = depth_images.reshape(1, -1)
        current_obs_buffer_dict["command_lin_vel_obs"] = self.lin_vel_command

        return current_obs_buffer_dict

    def _update_obs_history(self, current_obs_dict: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        # KEEP
        # TODO: to incorporate history length into the base policy, unify training code/inference code.
        """update obs_history"""

        # [a_t-2, b_t-2, c_t-2, ...,
        #  a_t-1, b_t-1, c_t-1, ...,
        #  ...
        #  a_t, b_t, c_t, ...
        # ]

        for group, term_dict in current_obs_dict.items():
            # get the observations for the group
            obs_list_per_group: list[np.ndarray] = []
            for term in self.obs_terms_sorted[group]:
                obs = np.asarray(term_dict[term], dtype=np.float32, order="C")
                if obs.ndim == 1:
                    obs = obs.reshape(1, -1)
                obs_list_per_group.append(obs.copy())
            obs_per_group = np.concatenate(obs_list_per_group, axis=1)
            # update the buffer
            if self.obs_interval_counters % self.obs_intervals[group] == 0:
                self.obs_history_buffers_per_group[group].append(obs_per_group)

            history_len = self.history_length_dict.get(group, 1)
            history = list(self.obs_history_buffers_per_group[group])
            if len(history) < history_len:
                missing = history_len - len(history)
                # padding with the oldest observation
                history = [history[0]] * missing + history
            self.obs_buf_dict[group] = np.stack(history, axis=1).reshape(1, -1)

        self.obs_interval_counters += 1

    def prepare_obs_for_rl(self, robot_state_data):
        """Prepare group observations for RL inference."""
        # super().prepare_obs_for_rl(robot_state_data)
        self._prepare_group_observations(robot_state_data)

        actor_obs_lower_body = np.concatenate(
            [
                self.obs_buf_dict["actor_obs"],
                self.obs_buf_dict["perception_obs"],
                self.obs_buf_dict["command_lin_vel_obs"],
            ],
            axis=1,
        )

        actor_obs_upper_body = self.obs_buf_dict["actor_obs"]
        return {
            "actor_obs_lower_body": actor_obs_lower_body.astype(np.float32),
            "actor_obs_upper_body": actor_obs_upper_body.astype(np.float32),
        }

    def process_joystick_input(self):
        """Process joystick input and update commands using interface."""
        super().process_joystick_input()

        if self.fixed_forward_speed_enabled:
            self.lin_vel_command[0, 0] = self.fixed_forward_speed


    def handle_joystick_button(self, cur_key: str):
        """Handle joystick button presses for locomotion."""
        # Call parent handler for common commands
        super().handle_joystick_button(cur_key)
        if cur_key == "L2":
            self.fixed_forward_speed_enabled = True
            self.fixed_forward_speed = 0.5
        elif cur_key == "R2":
            self.fixed_forward_speed_enabled = True
            self.fixed_forward_speed = -0.6
        elif cur_key == "R2+up":
            self.fixed_forward_speed += 0.1
        elif cur_key == "R2+down":
            self.fixed_forward_speed -= 0.1

    def _handle_stand_command(self):
        """Handle stand command toggle."""
        super()._handle_stand_command()
        if self.stand_command[0, 0] == 0:
            self.fixed_forward_speed_enabled = False
            self.fixed_forward_speed = 0.0
