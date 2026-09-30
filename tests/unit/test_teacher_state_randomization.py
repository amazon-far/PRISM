from __future__ import annotations

from types import SimpleNamespace

import pytest

from holosoma.config_types.command import NoiseToInitialPoseConfig
from holosoma.config_types.randomization import RandomizationTermCfg
from holosoma.config_values.wbt.g1.command import init_pose_config
from holosoma.config_values.wbt.g1.randomization import (
    g1_29dof_wbt_randomization_w_object_with_action_delay,
    g1_29dof_wbt_randomization_w_object_teacher_state_robust,
    g1_29dof_wbt_randomization_w_object_teacher_state_robust_with_camera,
)
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.randomization.terms.locomotion import (
    MotionRelativeResetRandomizerState,
)


def _state(params: dict) -> MotionRelativeResetRandomizerState:
    cfg = RandomizationTermCfg(func="unused:test", params=params)
    return MotionRelativeResetRandomizerState(cfg, SimpleNamespace())


def _assert_range_contains(
    outer: list[float],
    inner: list[float],
    *,
    strict: bool = True,
) -> None:
    assert outer[0] <= inner[0]
    assert outer[1] >= inner[1]
    if strict:
        assert outer[0] < inner[0] or outer[1] > inner[1]


def test_teacher_preset_randomizes_states_and_physics_without_actuator_or_perception_terms() -> None:
    cfg = g1_29dof_wbt_randomization_w_object_teacher_state_robust

    assert set(cfg.setup_terms) == {
        "motion_relative_reset_randomizer_state",
        "push_randomizer_state",
        "randomize_robot_rigid_body_material_startup",
        "randomize_base_com_startup",
        "mass_randomizer",
        "randomize_object_rigid_body_material_startup",
        "randomize_object_rigid_body_mass_inertia_scale_startup",
    }
    assert set(cfg.reset_terms) == {
        "push_randomizer_state",
        "randomize_push_schedule",
    }
    assert set(cfg.step_terms) == {"push_randomizer_state", "apply_pushes"}

    serialized = repr(cfg).lower()
    for forbidden in (
        "action_delay",
        "actuator_randomizer",
        "pd_gain",
        "rfi",
        "torque",
        "camera",
        "dof_pos_bias",
    ):
        assert forbidden not in serialized

    state = _state(cfg.setup_terms["motion_relative_reset_randomizer_state"].params)
    assert state.noise_config.dof_pos == pytest.approx(0.20)
    assert state.noise_config.dof_vel == pytest.approx(0.35)
    assert state.noise_config.object_pos[2] == 0.0


def test_teacher_camera_preset_only_adds_student_camera_randomization() -> None:
    teacher = g1_29dof_wbt_randomization_w_object_teacher_state_robust
    teacher_camera = (
        g1_29dof_wbt_randomization_w_object_teacher_state_robust_with_camera
    )
    student = g1_29dof_wbt_randomization_w_object_with_action_delay

    assert {
        name: term
        for name, term in teacher_camera.setup_terms.items()
        if name != "setup_camera_raycast_randomization"
    } == teacher.setup_terms
    assert {
        name: term
        for name, term in teacher_camera.reset_terms.items()
        if name != "randomize_camera_raycast"
    } == teacher.reset_terms
    assert teacher_camera.step_terms == teacher.step_terms
    assert (
        teacher_camera.setup_terms["setup_camera_raycast_randomization"]
        == student.setup_terms["setup_camera_raycast_randomization"]
    )
    assert (
        teacher_camera.reset_terms["randomize_camera_raycast"]
        == student.reset_terms["randomize_camera_raycast"]
    )

    serialized = repr(teacher_camera).lower()
    for forbidden in ("action_delay", "actuator_randomizer", "pd_gain", "rfi", "torque"):
        assert forbidden not in serialized


def test_teacher_reset_state_support_covers_student() -> None:
    teacher = g1_29dof_wbt_randomization_w_object_teacher_state_robust
    teacher_state = _state(
        teacher.setup_terms["motion_relative_reset_randomizer_state"].params
    ).noise_config

    assert teacher_state.overall_noise_scale == init_pose_config.overall_noise_scale
    assert teacher_state.dof_pos > init_pose_config.dof_pos
    assert teacher_state.dof_vel >= init_pose_config.dof_vel
    for field_name in (
        "root_pos",
        "root_rot",
        "root_lin_vel",
        "root_ang_vel",
        "object_pos",
    ):
        teacher_half_ranges = getattr(teacher_state, field_name)
        student_half_ranges = getattr(init_pose_config, field_name)
        assert all(
            teacher_range >= student_range
            for teacher_range, student_range in zip(
                teacher_half_ranges,
                student_half_ranges,
                strict=True,
            )
        )


def test_teacher_push_distribution_covers_the_student_distribution() -> None:
    teacher = g1_29dof_wbt_randomization_w_object_teacher_state_robust
    student = g1_29dof_wbt_randomization_w_object_with_action_delay

    teacher_push = teacher.setup_terms["push_randomizer_state"].params
    student_push = student.setup_terms["push_randomizer_state"].params

    _assert_range_contains(
        teacher_push["push_interval_s"],
        student_push["push_interval_s"],
    )
    assert all(
        teacher_limit > student_limit
        for teacher_limit, student_limit in zip(
            teacher_push["max_push_vel"],
            student_push["max_push_vel"],
            strict=True,
        )
    )


def test_teacher_physical_parameter_support_covers_student() -> None:
    teacher = g1_29dof_wbt_randomization_w_object_teacher_state_robust
    student = g1_29dof_wbt_randomization_w_object_with_action_delay

    teacher_mass = teacher.setup_terms["mass_randomizer"].params
    student_mass = student.setup_terms["mass_randomizer"].params
    assert teacher_mass["enable_link_mass"] is student_mass["enable_link_mass"] is True
    assert teacher_mass["enable_base_mass"] is student_mass["enable_base_mass"] is True
    _assert_range_contains(
        teacher_mass["link_mass_range"],
        student_mass["link_mass_range"],
    )
    _assert_range_contains(
        teacher_mass["added_mass_range"],
        student_mass["added_mass_range"],
    )

    teacher_com = teacher.setup_terms["randomize_base_com_startup"].params
    student_com = student.setup_terms["randomize_base_com_startup"].params
    assert teacher_com["enabled"] is student_com["enabled"] is True
    for axis in ("x", "y", "z"):
        _assert_range_contains(
            teacher_com["base_com_range"][axis],
            student_com["base_com_range"][axis],
        )

    teacher_robot_material = teacher.setup_terms[
        "randomize_robot_rigid_body_material_startup"
    ].params
    student_robot_material = student.setup_terms[
        "randomize_robot_rigid_body_material_startup"
    ].params
    for range_name in (
        "static_friction_range",
        "dynamic_friction_range",
        "restitution_range",
    ):
        _assert_range_contains(
            teacher_robot_material[range_name],
            student_robot_material[range_name],
        )

    teacher_object_mass = teacher.setup_terms[
        "randomize_object_rigid_body_mass_inertia_scale_startup"
    ].params
    student_object_mass = student.setup_terms[
        "randomize_object_rigid_body_mass_inertia_scale_startup"
    ].params
    _assert_range_contains(
        teacher_object_mass["mass_scale_distribution_params"],
        student_object_mass["mass_scale_distribution_params"],
    )

    teacher_object_material = teacher.setup_terms[
        "randomize_object_rigid_body_material_startup"
    ].params
    student_object_material = student.setup_terms[
        "randomize_object_rigid_body_material_startup"
    ].params
    _assert_range_contains(
        teacher_object_material["static_friction_range"],
        student_object_material["static_friction_range"],
    )
    _assert_range_contains(
        teacher_object_material["dynamic_friction_ratio_range"],
        student_object_material["dynamic_friction_ratio_range"],
    )
    _assert_range_contains(
        teacher_object_material["restitution_range"],
        student_object_material["restitution_range"],
        strict=False,
    )


def test_student_uses_nominal_actuation_and_joint_calibration() -> None:
    student = g1_29dof_wbt_randomization_w_object_with_action_delay

    action_delay = student.setup_terms["setup_action_delay_buffers"].params
    assert action_delay["enabled"] is False
    assert action_delay["ctrl_delay_step_range"] == [0, 0]

    actuator = student.setup_terms["actuator_randomizer_state"].params
    assert actuator["enable_pd_gain"] is False
    assert actuator["kp_range"] == [1.0, 1.0]
    assert actuator["kd_range"] == [1.0, 1.0]
    assert actuator["enable_rfi_lim"] is False
    assert actuator["rfi_lim_range"] == [1.0, 1.0]

    torque_rfi = student.setup_terms["setup_torque_rfi"].params
    assert torque_rfi["enabled"] is False
    assert torque_rfi["rfi_lim"] == 0.0

    dof_pos_bias = student.setup_terms["setup_dof_pos_bias"].params
    assert dof_pos_bias["enabled"] is False
    assert dof_pos_bias["dof_pos_bias_range"] == [0.0, 0.0]


def test_student_object_mass_and_inertia_scale_contract() -> None:
    student = g1_29dof_wbt_randomization_w_object_with_action_delay
    term = student.setup_terms["randomize_object_rigid_body_mass_inertia_scale_startup"]

    assert term.params["mass_scale_distribution_params"] == [0.25, 4.0]
    assert term.params["mass_scale_distribution"] == "log_uniform"


def test_student_object_friction_uses_coupled_ratio_contract() -> None:
    student = g1_29dof_wbt_randomization_w_object_with_action_delay
    term = student.setup_terms["randomize_object_rigid_body_material_startup"]

    assert term.params["static_friction_range"] == [0.1, 0.7]
    assert term.params["dynamic_friction_ratio_range"] == [0.7, 0.99]
    assert "dynamic_friction_range" not in term.params


def test_motion_command_uses_final_writer_randomization_state() -> None:
    configured = _state(
        {
            "overall_noise_scale": 1.0,
            "dof_pos": 0.2,
            "dof_vel": 0.35,
            "root_pos": [0.08, 0.08, 0.025],
            "root_rot": [0.15, 0.15, 0.3],
            "root_lin_vel": [0.2, 0.2, 0.1],
            "root_ang_vel": [0.25, 0.25, 0.35],
            "object_pos": [0.08, 0.08, 0.0],
        }
    )
    manager = SimpleNamespace(
        get_state=lambda name: configured
        if name == "motion_relative_reset_randomizer_state"
        else None
    )
    command = object.__new__(MotionCommand)
    command._env = SimpleNamespace(randomization_manager=manager)
    command.init_pose_cfg = NoiseToInitialPoseConfig(dof_pos=0.1)

    assert command._effective_initial_pose_noise_config() is configured.noise_config


def test_motion_command_falls_back_to_command_noise_without_state_preset() -> None:
    command = object.__new__(MotionCommand)
    command._env = SimpleNamespace(randomization_manager=None)
    command.init_pose_cfg = NoiseToInitialPoseConfig(dof_pos=0.1, dof_vel=0.05)

    assert command._effective_initial_pose_noise_config() is command.init_pose_cfg


@pytest.mark.parametrize(
    "params",
    (
        {"dof_pos": -0.1},
        {"dof_vel": float("nan")},
        {"root_pos": [0.1, 0.1]},
        {"object_pos": [0.1, -0.1, 0.0]},
        {"action_delay": 1},
    ),
)
def test_motion_relative_reset_randomizer_rejects_invalid_config(params: dict) -> None:
    with pytest.raises(ValueError):
        _state(params)
