"""Default inference configurations for holosoma_inference."""

from dataclasses import replace
from importlib.metadata import entry_points

import tyro
from typing_extensions import Annotated

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_values import camera, observation, robot, task

g1_29dof_loco = InferenceConfig(
    robot=robot.g1_29dof,
    observation=observation.loco_g1_29dof,
    task=task.locomotion,
)

t1_29dof_loco = InferenceConfig(
    robot=robot.t1_29dof,
    observation=observation.loco_t1_29dof,
    task=task.locomotion,
)

# fmt: off
g1_29dof_wbt = InferenceConfig(
    robot=replace(
        robot.g1_29dof,
        stiff_startup_pos=(
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # left leg
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # right leg
            0.0, 0.0, 0.0,                          # waist
            0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,      # left arm
            0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,     # right arm
        ),
        stiff_startup_kp=(
            350.0, 200.0, 200.0, 300.0, 300.0, 150.0,
            350.0, 200.0, 200.0, 300.0, 300.0, 150.0,
            200.0, 200.0, 200.0,
            40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
            40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
        ),
        stiff_startup_kd=(
            5.0, 5.0, 5.0, 10.0, 5.0, 5.0,
            5.0, 5.0, 5.0, 10.0, 5.0, 5.0,
            5.0, 5.0, 5.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
        ),
    ),
# fmt: on
    observation=observation.wbt,
    task=task.wbt,
)

g1_29dof_loco_manip_stand_height_waist = InferenceConfig(
    robot=robot.g1_29dof_loco_manip_stand_height_waist,
    observation=observation.loco_manip_stand_height_waist,
    task=task.loco_manip_stand_height_waist,
    camera=camera.dual_depth_cameras,
)

g1_wbt_distillation = InferenceConfig(
    robot=replace(
        robot.g1_29dof_wbt_distillation,
        stiff_startup_pos=(
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # left leg
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # right leg
            0.0, 0.0, 0.0,                          # waist
            0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,      # left arm
            0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,     # right arm
        ),
        stiff_startup_kp=(
            350.0, 200.0, 200.0, 300.0, 300.0, 150.0,
            350.0, 200.0, 200.0, 300.0, 300.0, 150.0,
            200.0, 200.0, 200.0,
            40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
            40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
        ),
        stiff_startup_kd=(
            5.0, 5.0, 5.0, 10.0, 5.0, 5.0,
            5.0, 5.0, 5.0, 10.0, 5.0, 5.0,
            5.0, 5.0, 5.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
        ),
    ),
    observation=observation.wbt_distillation_g1,
    task=task.wbt_distillation,
    camera=camera.single_zed2i_depth,
)

g1_root_pos_object_perception = InferenceConfig(
    robot=g1_wbt_distillation.robot,
    observation=observation.wbt_object_perception_g1,
    task=task.wbt_distillation,
    camera=camera.single_d435i_depth,
)

g1_root_pos_object_perception_no_linvel = InferenceConfig(
    robot=g1_wbt_distillation.robot,
    observation=observation.wbt_object_perception_no_linvel_g1,
    task=task.wbt_distillation,
    camera=camera.single_d435i_depth,
)

g1_root_pos_contact_aware_actions_no_linvel = InferenceConfig(
    robot=g1_wbt_distillation.robot,
    observation=observation.wbt_object_perception_contact_aware_actions_no_linvel_g1,
    task=task.wbt_distillation,
    camera=camera.single_d435i_depth,
)

g1_root_pos_contact_aware_actions_no_linvel_h1 = InferenceConfig(
    robot=g1_wbt_distillation.robot,
    observation=observation.wbt_object_perception_contact_aware_actions_no_linvel_h1_g1,
    task=task.wbt_distillation,
    camera=camera.single_d435i_depth,
)

g1_root_pos_contact_aware_drop_button_actions_no_linvel_h1 = InferenceConfig(
    robot=g1_wbt_distillation.robot,
    observation=observation.wbt_object_perception_contact_aware_drop_button_actions_no_linvel_h1_g1,
    task=task.wbt_distillation,
    camera=camera.single_d435i_depth,
)

g1_root_pos_contact_aware_pickup_drop_button_actions_no_linvel_h1 = InferenceConfig(
    robot=g1_wbt_distillation.robot,
    observation=observation.wbt_object_perception_contact_aware_pickup_drop_button_actions_no_linvel_h1_g1,
    task=task.wbt_distillation,
    camera=camera.single_d435i_depth,
)

g1_root_pos_actions_no_linvel_h1 = InferenceConfig(
    robot=g1_wbt_distillation.robot,
    observation=observation.wbt_object_perception_actions_no_linvel_h1_g1,
    task=task.wbt_distillation,
    camera=camera.single_d435i_depth,
)

g1_root_pos_actions_h1 = InferenceConfig(
    robot=g1_wbt_distillation.robot,
    observation=observation.wbt_object_perception_actions_h1_g1,
    task=task.wbt_distillation,
    camera=camera.single_d435i_depth,
)

g1_root_pos_actions_no_linvel_h5 = InferenceConfig(
    robot=g1_wbt_distillation.robot,
    observation=observation.wbt_object_perception_actions_no_linvel_h5_g1,
    task=task.wbt_distillation,
    camera=camera.single_d435i_depth,
)

# fmt: off
g1_blind_fall_recovery = InferenceConfig(
    robot=replace(
        robot.g1_29dof,
        stiff_startup_pos=(
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # left leg
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # right leg
            0.0, 0.0, 0.0,                          # waist
            0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,      # left arm
            0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,     # right arm
        ),
        stiff_startup_kp=(
            350.0, 200.0, 200.0, 300.0, 300.0, 150.0,
            350.0, 200.0, 200.0, 300.0, 300.0, 150.0,
            200.0, 200.0, 200.0,
            40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
            40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
        ),
        stiff_startup_kd=(
            5.0, 5.0, 5.0, 10.0, 5.0, 5.0,
            5.0, 5.0, 5.0, 10.0, 5.0, 5.0,
            5.0, 5.0, 5.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
        ),
    ),
# fmt: on
    observation=observation.blind_fall_recovery_g1,
    task=task.blind_fall_recovery,
)

DEFAULTS = {
    "g1-29dof-loco": g1_29dof_loco,
    "t1-29dof-loco": t1_29dof_loco,
    "g1-29dof-wbt": g1_29dof_wbt,
    "g1-29dof-loco-manip-stand-height-waist": g1_29dof_loco_manip_stand_height_waist,
    "g1-wbt-distillation": g1_wbt_distillation,
    "g1-root_pos-object-perception": g1_root_pos_object_perception,
    "g1-root_pos-object-perception-no-linvel": g1_root_pos_object_perception_no_linvel,
    "g1-root_pos-contact-aware-actions-no-linvel": g1_root_pos_contact_aware_actions_no_linvel,
    "g1-root_pos-contact-aware-actions-no-linvel-h1": g1_root_pos_contact_aware_actions_no_linvel_h1,
    "g1-root_pos-contact-aware-drop-button-actions-no-linvel-h1": (
        g1_root_pos_contact_aware_drop_button_actions_no_linvel_h1
    ),
    "g1-root_pos-contact-aware-pickup-drop-button-actions-no-linvel-h1": (
        g1_root_pos_contact_aware_pickup_drop_button_actions_no_linvel_h1
    ),
    "g1-root_pos-actions-no-linvel-h1": g1_root_pos_actions_no_linvel_h1,
    "g1-root_pos-actions-h1": g1_root_pos_actions_h1,
    "g1-root_pos-actions-no-linvel-h5": g1_root_pos_actions_no_linvel_h5,
    "g1-blind-fall-recovery": g1_blind_fall_recovery,
}

# Auto-discover inference configs from installed extensions
for ep in entry_points(group="holosoma.config.inference"):
    DEFAULTS[ep.name] = ep.load()

AnnotatedInferenceConfig = Annotated[
    InferenceConfig,
    tyro.conf.arg(
        constructor=tyro.extras.subcommand_type_from_defaults({f"inference:{k}": v for k, v in DEFAULTS.items()})
        ),
]
