#!/usr/bin/env python3
"""
Policy Runner Script with Tyro Configuration

This script uses Tyro configuration system to run different policy types.

Usage:
    python run_policy.py inference:g1-29dof-loco --task.model-path path/to/model.onnx
    python run_policy.py inference:g1-29dof-loco --task.model-path wandb://project/run/model.onnx
    python run_policy.py inference:g1-29dof-loco --task.model-path https://wandb-url/files/model.onnx
"""

import sys
import traceback

import tyro
from loguru import logger

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_values.inference import AnnotatedInferenceConfig
from holosoma_inference.config.utils import TYRO_CONFIG
from holosoma_inference.policies.blind_fall_recovery import BlindFallRecoveryPolicy
from holosoma_inference.policies.depth_distillation import DepthDistillationPolicy
from holosoma_inference.policies.loco_manip_stand_height_wait_depth import LocoManipStandHeightWaitDepthPolicy
from holosoma_inference.policies.locomotion import LocomotionPolicy
from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy
from holosoma_inference.utils.misc import restore_terminal_settings

import numpy as np

import os

DEBUG = os.getenv("DEBUG", "False").lower() == "true"


def _print_control_guide(policy_class, use_joystick: bool, config: InferenceConfig):
    """Print control guide for users."""
    is_blind_fall_recovery = issubclass(policy_class, BlindFallRecoveryPolicy)
    is_depth_distillation = issubclass(policy_class, DepthDistillationPolicy)
    is_wbt = policy_class.__name__ == "WholeBodyTrackingPolicy"
    has_pickup_button = "pickup_button" in config.observation.obs_dims
    has_drop_button = "drop_button" in config.observation.obs_dims

    logger.info("=" * 80)
    logger.info("POLICY CONTROLS")
    logger.info("=" * 80)
    logger.info("")

    if use_joystick:
        logger.info("Using JOYSTICK control mode")
        logger.info("")
        logger.info("General Controls:")
        logger.info("  A button       - Start the policy")
        logger.info("  B button       - Stop the policy")
        logger.info("  Y button       - Set robot to default pose")
        logger.info("  L1+R1 (LB+RB)  - Kill controller program")

        if is_blind_fall_recovery:
            logger.info("")
            logger.info("Blind Fall Recovery Controls:")
            logger.info("  Start button   - Enter stiff hold mode")
            logger.info("  R1 + Right stick Y - Push forward/backward (waist pitch torque)")
            logger.info("  R1 + Right stick X - Push left/right (waist roll torque)")
            logger.info("  Release R1         - Zero all push torques")
        elif is_depth_distillation:
            logger.info("")
            logger.info("Depth Distillation Controls:")
            logger.info("  Start button   - Enter stiff hold mode")
            logger.info("  L1 + Left stick - Select direction (forward/back/left/right)")
            logger.info("  L1 + Y button  - Toggle speed mode (low/high)")
            logger.info("  L1 + X button  - Madmax speed (while in high mode)")
        elif is_wbt:
            logger.info("")
            logger.info("Whole-Body Tracking Controls:")
            logger.info("  Start button   - Start motion clip")
            if has_drop_button:
                logger.info("  X button       - Toggle drop button command")
            logger.info("  Left stick     - Relative sparse root XY command")
            logger.info("  Right stick X  - Relative sparse root yaw command")
        else:
            logger.info("")
            logger.info("Locomotion Controls:")
            logger.info("  Start button   - Switch walking/standing mode")
            logger.info("  Left stick     - Adjust linear velocity (forward/backward/left/right)")
            logger.info("  Right stick    - Adjust angular velocity (turn left/right)")
    else:
        logger.info("Using KEYBOARD control mode")
        logger.info("")
        logger.info("IMPORTANT: Make sure THIS TERMINAL is active to receive keyboard input!")
        logger.info("All commands below must be entered in THIS terminal window.")
        logger.info("")
        logger.info("General Controls:")
        logger.info("  ] or p - Start the policy")
        logger.info("  o      - Stop the policy")
        logger.info("  i      - Set robot to default pose")

        if is_blind_fall_recovery:
            logger.info("")
            logger.info("Push Injection (hold key to push on torso):")
            logger.info("  J (hold)   - Push forward    ↑")
            logger.info("  K (hold)   - Push backward   ↓")
            logger.info("  H (hold)   - Push left       ←")
            logger.info("  L (hold)   - Push right      →")
            logger.info("  U / M      - Increase / decrease push magnitude")
            logger.info("  N          - Reset push magnitude to default")

        if is_wbt:
            logger.info("")
            logger.info("Whole-Body Tracking Controls:")
            logger.info("  s or m - Start motion clip")
            if has_pickup_button:
                logger.info("  f       - Toggle pickup button command")
            if has_drop_button:
                logger.info("  g       - Toggle drop button command")
            logger.info("  w/s/a/d - Adjust sparse root XY command")
            logger.info("  q/e     - Adjust sparse root yaw command")
            logger.info("  z       - Reset sparse root command offset")
        elif is_depth_distillation:
            logger.info("")
            logger.info("Depth Distillation Controls:")
            logger.info("  w          - Forward")
            logger.info("  s          - Backward")
            logger.info("  a/d        - Turn left/right (45 deg)")
            logger.info("  q/e        - Turn left/right (90 deg)")
            logger.info("  z          - Stand (stop)")
            logger.info("  =          - Cycle speed mode (low/high/madmax)")
            logger.info("  h          - Enter stiff hold mode")
        else:
            logger.info("")
            logger.info("Locomotion Controls:")
            logger.info("  =          - Switch walking/standing mode")
            logger.info("  w/s        - Increase/decrease forward velocity")
            logger.info("  a/d        - Increase/decrease lateral velocity")
            logger.info("  q/e        - Increase/decrease angular velocity (turn left/right)")
            logger.info("  z          - Set all velocities to zero")

    logger.info("")
    logger.info("MuJoCo Simulator Controls (ONLY in MuJoCo window, NOT this terminal!):")
    logger.info("  7/8        - Decrease/increase elastic band length")
    logger.info("  9          - Toggle elastic band enable/disable")
    logger.info("  BACKSPACE  - Reset simulation")

    logger.info("")
    logger.info("=" * 80)
    logger.info("Press the appropriate button/key to begin!")
    logger.info("=" * 80)
    logger.info("")


def _select_policy_class(config: InferenceConfig):
    """Select the appropriate policy class based on configuration.

    Selection rules (in priority order):
    1. If task.policy_type == "blind_fall_recovery" -> BlindFallRecoveryPolicy
    2. If task.model_path is a list of 2 paths and depth_obs is in observation groups
       -> DepthDistillationPolicy
    3. If actor_obs contains "motion_command" -> WholeBodyTrackingPolicy
    4. If observation groups contain "perception_obs" (dual depth cameras)
       -> LocoManipStandHeightWaitDepthPolicy
    5. Otherwise -> LocomotionPolicy
    """
    # Explicit policy_type takes priority
    policy_type = getattr(config.task, "policy_type", None)
    if policy_type == "blind_fall_recovery":
        return BlindFallRecoveryPolicy

    obs_dict = config.observation.obs_dict
    actor_obs = obs_dict.get("actor_obs", [])
    model_path = config.task.model_path

    # Check for depth distillation: two model paths + depth_obs group
    has_depth_obs = "depth_obs" in obs_dict
    has_two_models = isinstance(model_path, list) and len(model_path) == 2
    if has_depth_obs and has_two_models:
        return DepthDistillationPolicy

    has_sparse_root_obs = any(group.startswith("actor_obs_root") for group in obs_dict)
    has_proprio_obs = any(group.startswith("actor_obs_proprio") for group in obs_dict)
    if has_sparse_root_obs and has_proprio_obs:
        return WholeBodyTrackingPolicy

    # Check for WBT policy
    if "motion_command" in actor_obs:
        return WholeBodyTrackingPolicy

    # Check for loco-manip depth policy
    if "perception_obs" in obs_dict:
        return LocoManipStandHeightWaitDepthPolicy

    return LocomotionPolicy


def run_policy(config: InferenceConfig):
    """Run policy with Tyro configuration."""
    logger.info("Starting Policy with Tyro configuration...")
    logger.info(f"Robot: {config.robot.robot_type}")
    logger.info(f"Observation groups: {list(config.observation.obs_dict.keys())}")
    logger.info(f"RL Rate: {config.task.rl_rate} Hz")
    logger.info(f"Model path: {config.task.model_path}")

    # Determine policy class based on observation and task configuration
    policy_class = _select_policy_class(config)
    logger.info(f"Using {policy_class.__name__}")
    policy: LocomotionPolicy | WholeBodyTrackingPolicy = policy_class(config=config)

    logger.info("Policy initialized successfully!")
    _print_control_guide(policy_class, config.task.use_joystick, config)

    if config.task.auto_start_policy:
        if hasattr(policy, "wait_for_motion_initial_state"):
            policy.wait_for_motion_initial_state()
        policy.warm_start_observation_history()
        policy._handle_start_policy()
    if config.task.auto_start_motion_clip:
        if not hasattr(policy, "_handle_start_motion_clip"):
            raise ValueError("auto_start_motion_clip is only supported by WBT policies.")
        policy._handle_start_motion_clip()

    if DEBUG:

        policy._handle_start_policy()

        # To avoid prininting,
        policy.interface.send_low_command(
            policy.cmd_q,
            policy.cmd_dq,
            policy.cmd_tau,
            np.zeros(policy.num_dofs),
        )
        if hasattr(policy, 'phase_dt'):
            policy.update_phase_time()
        for i in range(10):
            print(f"iteration {i}")
            policy.policy_action()
    else:
        try:
            policy.run()
        except Exception as e:
            logger.error(f"Error running policy: {e}")
            traceback.print_exc()
            sys.exit(1)
        finally:
            restore_terminal_settings()
            if hasattr(policy, "close_deployment_audit"):
                policy.close_deployment_audit()
    logger.info("Policy execution completed!")


def main():
    config = tyro.cli(
        AnnotatedInferenceConfig,
        config=TYRO_CONFIG,
    )
    run_policy(config)


if __name__ == "__main__":
    main()
