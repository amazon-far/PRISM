from __future__ import annotations

from holosoma.config_values.wbt.g1.command import init_pose_config, motion_config_w_object_generalist


def test_g1_wbt_reset_velocity_noise_matches_success_teacher_profile() -> None:
    assert init_pose_config.root_lin_vel == [0.1, 0.1, 0.05]
    assert init_pose_config.root_ang_vel == [0.1, 0.1, 0.1]


def test_object_generalist_uses_uniform_within_clip_resets_without_timestep_zero_freeze() -> None:
    assert motion_config_w_object_generalist.use_adaptive_timesteps_sampler is False
    assert motion_config_w_object_generalist.freeze_at_timestep_zero_prob == 0.0
