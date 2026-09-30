"""Default task configurations for holosoma_inference."""

from __future__ import annotations

from holosoma_inference.config.config_types.task import TaskConfig

# Locomotion task
locomotion = TaskConfig(
    model_path="",  # Must be provided by user
    rl_rate=50,
    policy_action_scale=0.25,
    use_phase=True,
    gait_period=1.0,
    desired_base_height=0.75,
    residual_upper_body_action=False,
    domain_id=0,
    interface="lo",
    use_joystick=False,
    joystick_type="xbox",
    joystick_device=0,
    use_ros=False,
    wandb_download_dir="/tmp",
)

# Whole-body tracking task
wbt = TaskConfig(
    model_path="",  # Must be provided by user
    rl_rate=50,
    policy_action_scale=1.0,
    use_phase=False,
    gait_period=1.0,
    desired_base_height=0.75,
    residual_upper_body_action=False,
    domain_id=0,
    interface="lo",
    use_joystick=False,
    joystick_type="xbox",
    joystick_device=0,
    use_ros=False,
    wandb_download_dir="/tmp",
)

loco_manip_stand_height_waist = TaskConfig(
    model_path="",  # Supply an ONNX path.
    rl_rate=50,
    policy_action_scale=0.25,
    use_phase=True,
    gait_period=1.0,
    desired_base_height=0.75,
    residual_upper_body_action=False,
    interface="lo",
    use_joystick=False,
    joystick_type="xbox",
    joystick_device=0,
    use_ros=False,
    wandb_download_dir="/tmp",
)

# Depth distillation task (far-tracking two-model architecture)
wbt_distillation = TaskConfig(
    model_path=["", ""],  # Supply depth-backbone and student ONNX paths.
    rl_rate=50,
    policy_action_scale=1.0,  # read from ONNX metadata action_scale if available
    use_phase=False,
    gait_period=1.0,
    desired_base_height=0.75,
    residual_upper_body_action=False,
    domain_id=0,
    interface="lo",
    use_joystick=False,
    joystick_type="xbox",
    joystick_device=0,
    use_ros=False,
    wandb_download_dir="/tmp",
)

# Blind fall recovery task (proprioceptive-only, single model)
blind_fall_recovery = TaskConfig(
    model_path="",  # student.onnx - must be provided by user
    policy_type="blind_fall_recovery",
    rl_rate=50,
    policy_action_scale=1.0,
    use_phase=False,
    gait_period=1.0,
    desired_base_height=0.75,
    residual_upper_body_action=False,
    domain_id=0,
    interface="lo",
    use_joystick=False,
    joystick_type="xbox",
    joystick_device=0,
    use_ros=False,
    wandb_download_dir="/tmp",
)

DEFAULTS = {
    "locomotion": locomotion,
    "wbt": wbt,
    "loco-manip-stand-height-waist": loco_manip_stand_height_waist,
    "wbt-distillation": wbt_distillation,
    "blind-fall-recovery": blind_fall_recovery,
}
