from holosoma.config_values.wbt.g1.observation import (
    g1_29dof_wbt_observation_w_object,
    g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd,
    g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd_teacher_linvel,
    g1_29dof_wbt_observation_w_object_teacher_linvel,
)


def test_teacher_linvel_adds_exact_privileged_actor_term() -> None:
    ordinary_terms = g1_29dof_wbt_observation_w_object.groups["actor_obs"].terms
    teacher_terms = g1_29dof_wbt_observation_w_object_teacher_linvel.groups["actor_obs"].terms

    assert "base_lin_vel" not in ordinary_terms
    assert list(teacher_terms) == [*ordinary_terms, "base_lin_vel"]
    term = teacher_terms["base_lin_vel"]
    assert term.func == "holosoma.managers.observation.terms.wbt:base_lin_vel"
    assert term.scale == 1.0
    assert term.noise == 0.0


def test_distill_linvel_variant_changes_only_teacher_query_group() -> None:
    ordinary = g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd
    teacher_linvel = g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd_teacher_linvel

    assert set(teacher_linvel.groups) == set(ordinary.groups)
    assert "base_lin_vel" in teacher_linvel.groups["actor_obs"].terms
    for group_name in ordinary.groups:
        if group_name != "actor_obs":
            assert teacher_linvel.groups[group_name] == ordinary.groups[group_name]
