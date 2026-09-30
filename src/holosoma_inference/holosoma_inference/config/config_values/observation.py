"""Default observation configurations for holosoma_inference.

This module provides pre-configured observation spaces for different
robot types and tasks, converted from the original YAML configurations.
"""

from __future__ import annotations

from importlib.metadata import entry_points

from holosoma_inference.config.config_types.observation import ObservationConfig

# =============================================================================
# Locomotion Observation Configurations
# =============================================================================

loco_g1_29dof = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "base_ang_vel",
            "projected_gravity",
            "command_lin_vel",
            "command_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
            "sin_phase",
            "cos_phase",
        ]
    },
    obs_dims={
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "command_lin_vel": 2,
        "command_ang_vel": 1,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "sin_phase": 2,
        "cos_phase": 2,
    },
    obs_scales={
        "base_lin_vel": 2.0,
        "base_ang_vel": 0.25,
        "projected_gravity": 1.0,
        "command_lin_vel": 1.0,
        "command_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "actions": 1.0,
        "sin_phase": 1.0,
        "cos_phase": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)

loco_t1_29dof = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "base_ang_vel",
            "projected_gravity",
            "command_lin_vel",
            "command_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
            "sin_phase",
            "cos_phase",
        ]
    },
    obs_dims={
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "command_lin_vel": 2,
        "command_ang_vel": 1,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "sin_phase": 2,
        "cos_phase": 2,
    },
    obs_scales={
        "base_lin_vel": 1.0,  # T1 uses 1.0 (vs G1's 2.0)
        "base_ang_vel": 1.0,  # T1 uses 1.0 (vs G1's 0.25)
        "projected_gravity": 1.0,
        "command_lin_vel": 1.0,
        "command_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.1,  # T1 uses 0.1 (vs G1's 0.05)
        "actions": 1.0,
        "sin_phase": 1.0,
        "cos_phase": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)


# =============================================================================
# WBT (Whole Body Tracking) Observation Configurations
# =============================================================================

wbt = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "motion_command",
            "motion_ref_ori_b",
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
        ]
    },
    obs_dims={
        "motion_command": 58,
        "motion_ref_pos_b": 3,
        "motion_ref_ori_b": 6,
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "actions": 1.0,
        "motion_command": 1.0,
        "motion_ref_pos_b": 1.0,
        "motion_ref_ori_b": 1.0,
        "base_lin_vel": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "robot_body_pos_b": 1.0,
        "robot_body_ori_b": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)


loco_manip_stand_height_waist = ObservationConfig(
    obs_intervals={
        "actor_obs": 1,
        "perception_obs": 5,
        "command_lin_vel_obs": 1,
    },
    obs_dict={
        "actor_obs": [
            "base_ang_vel",  # checked
            "projected_gravity",  # checked
            "command_lin_vel",  # checked
            "command_ang_vel",  # checked
            "command_stand",  # checked
            "command_base_height",
            "command_waist_dofs",
            "ref_upper_dof_pos",
            "dof_pos",  # checked
            "dof_vel",  # checked
            "actions",  # checked
        ],
        "perception_obs": [
            "cam_depth",
        ],
        "command_lin_vel_obs": [
            "command_lin_vel",
        ],
    },
    obs_dims={
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "command_lin_vel": 2,
        "command_ang_vel": 1,
        "command_stand": 1,
        "command_base_height": 1,
        "command_waist_dofs": 3,
        "ref_upper_dof_pos": 14,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        # cam_depth: ${eval:${robot.cameras.props.resized_width} * ${robot.cameras.props.resized_height} * ${len:${robot.cameras.poses}}}
        "cam_depth": 2592,  # 2 cameras, 48 width, 27 height
    },
    obs_scales={
        "base_ang_vel": 0.25,
        "projected_gravity": 1.0,
        "command_lin_vel": 1.0,
        "command_ang_vel": 1.0,
        "command_stand": 1.0,
        "command_base_height": 2.0,
        "command_waist_dofs": 1.0,
        "ref_upper_dof_pos": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "history": 1.0,
        "actions": 1.0,
        "cam_depth": 1.0,
    },
    history_length_dict={
        "actor_obs": 5,
        "perception_obs": 2,
        "command_lin_vel_obs": 1,
    },
    # Note: obs_intervals from YAML not included as it's not present in other Python configs
    # Original intervals were: actor_state_obs: 1, perception_obs: ${eval:${task.policy.config.rl_rate} / ${robot.cameras.props.frame_rate}}, command_lin_vel_obs: 1
)


# =============================================================================
# WBT Distillation (far-tracking depth distillation) Observation Configurations
# =============================================================================

wbt_distillation_g1 = ObservationConfig(
    obs_intervals={
        "actor_obs": 1,
        "depth_obs": 1,  # FREQ_RATIO=1 in far-tracking
    },
    obs_dict={
        "actor_obs": [
            "projected_gravity",
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
        ],
        "depth_obs": [
            "cam_depth",
        ],
    },
    obs_dims={
        "projected_gravity": 3,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        # cam_depth: resized_height * resized_width = 58 * 87 = 5046
        "cam_depth": 5046,
        # velocity_command: one-hot vector from the "command" obs group in training.
        # Not part of actor_obs — concatenated separately by DepthDistillationPolicy.
        "velocity_command": 15,
    },
    obs_scales={
        "projected_gravity": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "actions": 1.0,
        "cam_depth": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
        "depth_obs": 3,
    },
)


wbt_object_perception_g1 = ObservationConfig(
    obs_intervals={
        "actor_obs_root_contact_aware": 1,
        "actor_obs_proprio": 1,
        "perception_obs": 1,
    },
    obs_dict={
        "actor_obs_root_contact_aware": [
            "sparse_target_root_trajectory_command_contact_aware",
        ],
        "actor_obs_proprio": [
            "base_lin_vel",
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
        ],
        "perception_obs": [
            "cam_depth",
        ],
    },
    obs_dims={
        "sparse_target_root_trajectory_command_contact_aware": 3,
        "base_ang_vel": 3,
        "base_lin_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        # camera_depth_d435i after far-tracking warp resize: 58 * 87
        "cam_depth": 5046,
    },
    obs_scales={
        "sparse_target_root_trajectory_command_contact_aware": 1.0,
        "base_ang_vel": 1.0,
        "base_lin_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "cam_depth": 1.0,
    },
    history_length_dict={
        "actor_obs_root_contact_aware": 1,
        "actor_obs_proprio": 5,
        "perception_obs": 1,
    },
)


wbt_object_perception_no_linvel_g1 = ObservationConfig(
    obs_intervals={
        "actor_obs_root": 1,
        "actor_obs_proprio_no_linvel": 1,
        "perception_obs": 1,
    },
    obs_dict={
        "actor_obs_root": [
            "sparse_target_root_trajectory_command",
        ],
        "actor_obs_proprio_no_linvel": [
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
        ],
        "perception_obs": [
            "cam_depth",
        ],
    },
    obs_dims={
        "sparse_target_root_trajectory_command": 3,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "cam_depth": 5046,
    },
    obs_scales={
        "sparse_target_root_trajectory_command": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "cam_depth": 1.0,
    },
    history_length_dict={
        "actor_obs_root": 1,
        "actor_obs_proprio_no_linvel": 5,
        "perception_obs": 1,
    },
)


wbt_object_perception_contact_aware_actions_no_linvel_g1 = ObservationConfig(
    obs_intervals={
        "actor_obs_root_contact_aware": 1,
        "actor_obs_proprio_with_actions_no_linvel": 1,
        "perception_obs": 1,
    },
    obs_dict={
        "actor_obs_root_contact_aware": [
            "sparse_target_root_trajectory_command_contact_aware",
        ],
        "actor_obs_proprio_with_actions_no_linvel": [
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
        ],
        "perception_obs": [
            "cam_depth",
        ],
    },
    obs_dims={
        "sparse_target_root_trajectory_command_contact_aware": 3,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "cam_depth": 5046,
    },
    obs_scales={
        "sparse_target_root_trajectory_command_contact_aware": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "actions": 1.0,
        "cam_depth": 1.0,
    },
    history_length_dict={
        "actor_obs_root_contact_aware": 1,
        "actor_obs_proprio_with_actions_no_linvel": 5,
        "perception_obs": 1,
    },
)


wbt_object_perception_contact_aware_actions_no_linvel_h1_g1 = ObservationConfig(
    obs_intervals={
        "actor_obs_root_contact_aware": 1,
        "actor_obs_proprio_with_actions_no_linvel": 1,
        "perception_obs": 1,
    },
    obs_dict={
        "actor_obs_root_contact_aware": [
            "sparse_target_root_trajectory_command_contact_aware",
        ],
        "actor_obs_proprio_with_actions_no_linvel": [
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
        ],
        "perception_obs": [
            "cam_depth",
        ],
    },
    obs_dims={
        "sparse_target_root_trajectory_command_contact_aware": 3,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "cam_depth": 5046,
    },
    obs_scales={
        "sparse_target_root_trajectory_command_contact_aware": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "actions": 1.0,
        "cam_depth": 1.0,
    },
    history_length_dict={
        "actor_obs_root_contact_aware": 1,
        "actor_obs_proprio_with_actions_no_linvel": 1,
        "perception_obs": 1,
    },
)


wbt_object_perception_contact_aware_drop_button_actions_no_linvel_h1_g1 = ObservationConfig(
    obs_intervals={
        "actor_obs_root_contact_aware": 1,
        "actor_obs_drop_button": 1,
        "actor_obs_proprio_with_actions_no_linvel": 1,
        "perception_obs": 1,
    },
    obs_dict={
        "actor_obs_root_contact_aware": [
            "sparse_target_root_trajectory_command_contact_aware",
        ],
        "actor_obs_drop_button": [
            "drop_button",
        ],
        "actor_obs_proprio_with_actions_no_linvel": [
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
        ],
        "perception_obs": [
            "cam_depth",
        ],
    },
    obs_dims={
        "sparse_target_root_trajectory_command_contact_aware": 3,
        "drop_button": 1,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "cam_depth": 5046,
    },
    obs_scales={
        "sparse_target_root_trajectory_command_contact_aware": 1.0,
        "drop_button": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "actions": 1.0,
        "cam_depth": 1.0,
    },
    history_length_dict={
        "actor_obs_root_contact_aware": 1,
        "actor_obs_drop_button": 1,
        "actor_obs_proprio_with_actions_no_linvel": 1,
        "perception_obs": 1,
    },
)


wbt_object_perception_contact_aware_pickup_drop_button_actions_no_linvel_h1_g1 = ObservationConfig(
    obs_intervals={
        "actor_obs_root_contact_aware": 1,
        "actor_obs_pickup_button": 1,
        "actor_obs_drop_button": 1,
        "actor_obs_proprio_with_actions_no_linvel": 1,
        "perception_obs": 1,
    },
    obs_dict={
        "actor_obs_root_contact_aware": [
            "sparse_target_root_trajectory_command_contact_aware",
        ],
        "actor_obs_pickup_button": [
            "pickup_button",
        ],
        "actor_obs_drop_button": [
            "drop_button",
        ],
        "actor_obs_proprio_with_actions_no_linvel": [
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
        ],
        "perception_obs": [
            "cam_depth",
        ],
    },
    obs_dims={
        "sparse_target_root_trajectory_command_contact_aware": 3,
        "pickup_button": 1,
        "drop_button": 1,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "cam_depth": 5046,
    },
    obs_scales={
        "sparse_target_root_trajectory_command_contact_aware": 1.0,
        "pickup_button": 1.0,
        "drop_button": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "actions": 1.0,
        "cam_depth": 1.0,
    },
    history_length_dict={
        "actor_obs_root_contact_aware": 1,
        "actor_obs_pickup_button": 1,
        "actor_obs_drop_button": 1,
        "actor_obs_proprio_with_actions_no_linvel": 1,
        "perception_obs": 1,
    },
)


wbt_object_perception_actions_no_linvel_h1_g1 = ObservationConfig(
    obs_intervals={
        "actor_obs_root": 1,
        "actor_obs_proprio_with_actions_no_linvel": 1,
        "perception_obs": 1,
    },
    obs_dict={
        "actor_obs_root": [
            "sparse_target_root_trajectory_command",
        ],
        "actor_obs_proprio_with_actions_no_linvel": [
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
        ],
        "perception_obs": [
            "cam_depth",
        ],
    },
    obs_dims={
        "sparse_target_root_trajectory_command": 3,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "cam_depth": 5046,
    },
    obs_scales={
        "sparse_target_root_trajectory_command": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "actions": 1.0,
        "cam_depth": 1.0,
    },
    history_length_dict={
        "actor_obs_root": 1,
        "actor_obs_proprio_with_actions_no_linvel": 1,
        "perception_obs": 1,
    },
)


wbt_object_perception_actions_h1_g1 = ObservationConfig(
    obs_intervals={
        "actor_obs_root": 1,
        "actor_obs_proprio_with_actions": 1,
        "perception_obs": 1,
    },
    obs_dict={
        "actor_obs_root": [
            "sparse_target_root_trajectory_command",
        ],
        "actor_obs_proprio_with_actions": [
            "base_lin_vel",
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
        ],
        "perception_obs": [
            "cam_depth",
        ],
    },
    obs_dims={
        "sparse_target_root_trajectory_command": 3,
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "cam_depth": 5046,
    },
    obs_scales={
        "sparse_target_root_trajectory_command": 1.0,
        "base_lin_vel": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "actions": 1.0,
        "cam_depth": 1.0,
    },
    history_length_dict={
        "actor_obs_root": 1,
        "actor_obs_proprio_with_actions": 1,
        "perception_obs": 1,
    },
)


wbt_object_perception_actions_no_linvel_h5_g1 = ObservationConfig(
    obs_intervals={
        "actor_obs_root": 1,
        "actor_obs_proprio_with_actions_no_linvel": 1,
        "perception_obs": 1,
    },
    obs_dict={
        "actor_obs_root": [
            "sparse_target_root_trajectory_command",
        ],
        "actor_obs_proprio_with_actions_no_linvel": [
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
        ],
        "perception_obs": [
            "cam_depth",
        ],
    },
    obs_dims={
        "sparse_target_root_trajectory_command": 3,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "cam_depth": 5046,
    },
    obs_scales={
        "sparse_target_root_trajectory_command": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "actions": 1.0,
        "cam_depth": 1.0,
    },
    history_length_dict={
        "actor_obs_root": 1,
        "actor_obs_proprio_with_actions_no_linvel": 5,
        "perception_obs": 1,
    },
)


# =============================================================================
# Blind Fall Recovery Observation Configurations
# =============================================================================

blind_fall_recovery_g1 = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "projected_gravity",
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
        ],
    },
    obs_dims={
        "projected_gravity": 3,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "projected_gravity": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "actions": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)


# =============================================================================
# Default Configurations Dictionary
# =============================================================================

DEFAULTS = {
    "loco-g1-29dof": loco_g1_29dof,
    "loco-t1-29dof": loco_t1_29dof,
    "wbt": wbt,
    "loco-manip-stand-height-waist": loco_manip_stand_height_waist,
    "wbt-distillation-g1": wbt_distillation_g1,
    "wbt-object-perception-g1": wbt_object_perception_g1,
    "wbt-object-perception-no-linvel-g1": wbt_object_perception_no_linvel_g1,
    "wbt-object-perception-contact-aware-actions-no-linvel-g1": (
        wbt_object_perception_contact_aware_actions_no_linvel_g1
    ),
    "wbt-object-perception-contact-aware-actions-no-linvel-h1-g1": (
        wbt_object_perception_contact_aware_actions_no_linvel_h1_g1
    ),
    "wbt-object-perception-actions-no-linvel-h1-g1": wbt_object_perception_actions_no_linvel_h1_g1,
    "wbt-object-perception-actions-h1-g1": wbt_object_perception_actions_h1_g1,
    "wbt-object-perception-actions-no-linvel-h5-g1": wbt_object_perception_actions_no_linvel_h5_g1,
    "blind-fall-recovery-g1": blind_fall_recovery_g1,
}
"""Dictionary of all available observation configurations.

Keys use hyphen-case naming convention for CLI compatibility.
"""

# Auto-discover observation configs from installed extensions
for ep in entry_points(group="holosoma.config.observation"):
    DEFAULTS[ep.name] = ep.load()
