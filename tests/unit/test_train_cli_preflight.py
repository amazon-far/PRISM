from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any, Callable, cast

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT_MODULE = runpy.run_path(str(REPO_ROOT / "scripts" / "validate_train_cli.py"))
PREFLIGHT = PREFLIGHT_MODULE["parse_and_validate_train_cli"]
parse_and_validate = cast(Callable[..., Any], PREFLIGHT)
validate_unique_long_options = cast(
    Callable[[list[str]], None],
    PREFLIGHT_MODULE["_validate_unique_long_options"],
)
strict_bool = cast(Callable[[str, object], bool], PREFLIGHT_MODULE["_strict_bool"])
EXP = "exp:g1-29dof-wbt-w-object-distill-sparse-root-cmd-r2s-rollout-ref"


def test_episodic_train_cli_parses_motion_end_contract_and_late_flags() -> None:
    config = parse_and_validate(
        [
            EXP,
            "perception:camera_depth_d435i",
            "termination:g1_29dof_wbt_generalist",
            "logger:disabled",
            "--training.num-envs=64",
            "--termination.terms.bad-tracking.params.bad-ref-pos-threshold=0.73",
        ],
        expected_motion_end_mode="episodic",
    )

    assert set(config.termination.terms) == {"timeout", "bad_tracking", "motion_ends"}
    assert config.termination.terms["motion_ends"].func.endswith(":motion_ends")
    assert config.termination.terms["motion_ends"].is_timeout is False
    assert config.termination.terms["bad_tracking"].params["bad_ref_pos_threshold"] == pytest.approx(0.73)
    assert config.training.num_envs == 64


def test_continuing_train_cli_parses_without_motion_end() -> None:
    config = parse_and_validate(
        [
            EXP,
            "termination:g1_29dof_wbt_distill",
            "logger:disabled",
            "--training.num-envs=64",
        ],
        expected_motion_end_mode="continuing",
    )

    assert set(config.termination.terms) == {"timeout", "bad_tracking"}


@pytest.mark.parametrize(
    "duplicate_args",
    [
        ["--training.num-envs=64", "--training.num-envs=64"],
        ["--training.num-envs=64", "--training.num-envs", "32"],
        [
            "--algo.config.distill.ppo-start-coeff=0.9",
            "--algo.config.distill.ppo_start_coeff=0.1",
        ],
    ],
)
def test_duplicate_long_options_fail_closed_before_tyro_parse(duplicate_args: list[str]) -> None:
    with pytest.raises(ValueError, match="Duplicate train CLI long option"):
        parse_and_validate(
            [
                EXP,
                "logger:disabled",
                *duplicate_args,
            ]
        )


def test_duplicate_guard_does_not_merge_distinct_namespaces_or_no_prefix() -> None:
    validate_unique_long_options(
        [
            "--actor.enabled=True",
            "--critic.enabled=True",
            "--feature=True",
            "--no-feature=True",
        ]
    )


@pytest.mark.parametrize(
    ("termination_profile", "expected_mode", "message"),
    [
        ("g1_29dof_wbt_distill", "episodic", "requires termination term 'motion_ends'"),
        ("g1_29dof_wbt_generalist", "continuing", "requires motion_ends to be absent"),
    ],
)
def test_motion_end_mode_mismatch_fails_closed(
    termination_profile: str,
    expected_mode: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_and_validate(
            [
                EXP,
                f"termination:{termination_profile}",
                "logger:disabled",
            ],
            expected_motion_end_mode=expected_mode,
        )


def _camera_wbt_args(*overrides: str) -> list[str]:
    return [
        EXP,
        "perception:camera_depth_d435i",
        "termination:g1_29dof_wbt_generalist",
        "logger:disabled",
        *overrides,
    ]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (["--algo.config.actor-learning-rate=nan"], "actor_learning_rate must be finite"),
        (["--algo.config.critic-learning-rate=-1"], "critic_learning_rate must be finite and > 0"),
        (["--algo.config.schedule=Adaptive"], "schedule must be exactly 'adaptive' or 'fixed'"),
        (["--algo.config.desired-kl=nan"], "desired_kl must be finite"),
        (["--algo.config.desired-kl=0"], "desired_kl must be finite and > 0"),
        (
            [
                "--algo.config.actor-learning-rate=0.001",
                "--algo.config.min-actor-learning-rate=0.002",
            ],
            "actor_learning_rate must be >=",
        ),
        (
            [
                "--algo.config.critic-learning-rate=0.001",
                "--algo.config.max-critic-learning-rate=0.0005",
            ],
            "critic_learning_rate must be <=",
        ),
        (["--algo.config.entropy-coef=nan"], "entropy_coef must be finite"),
        (["--algo.config.entropy-coef=-1"], "entropy_coef must be finite and >= 0"),
        (["--algo.config.init-noise-std=0"], "init_noise_std must be finite and > 0"),
        (
            ["--algo.config.module-dict.actor.min-noise-std=inf"],
            "actor.min_noise_std must be finite",
        ),
        (
            [
                "--algo.config.init-noise-std=0.01",
                "--algo.config.module-dict.actor.min-noise-std=0.1",
            ],
            "init_noise_std must be >=",
        ),
        (["--algo.config.distill.ppo-start-coeff=nan"], "ppo_start_coeff must be finite"),
        (["--algo.config.distill.ppo-target-coeff=2"], "ppo_target_coeff must be a finite probability"),
        (
            [
                "--algo.config.distill.ppo-start-coeff=0.8",
                "--algo.config.distill.ppo-target-coeff=0.7",
            ],
            "ppo_start_coeff must be <=",
        ),
        (["--algo.config.distill.dagger-loss-coef=-1"], "dagger_loss_coef must be finite and >= 0"),
        (["--algo.config.distill.ppo-start-noise-std=-1"], "ppo_start_noise_std must be finite and > 0"),
        (
            ["--algo.config.distill.ppo-start-noise-std-until-coeff=inf"],
            "ppo_start_noise_std_until_coeff must be finite",
        ),
    ],
)
def test_effective_ppo_numerics_fail_closed(overrides: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_and_validate(_camera_wbt_args(*overrides))


def _fixed_bc_guard_args(overrides: dict[str, object] | None = None) -> list[str]:
    values: dict[str, object] = {
        "--algo.config.num-learning-iterations": 20,
        "--algo.config.distill.enabled": True,
        "--algo.config.distill.mode": "dagger",
        "--algo.config.distill.policy-to-clone": "teacher.pt",
        "--algo.config.distill.bc-loss-coef": 1.0,
        "--algo.config.distill.ppo-start-epoch": 3,
        "--algo.config.distill.dagger-end-epoch": 10,
        "--algo.config.distill.ppo-start-coeff": 0.0,
        "--algo.config.distill.ppo-target-coeff": 0.7,
        "--algo.config.distill.ppo-schedule-step-epochs": 1,
        "--algo.config.distill.fixed-bc-eval-num-samples": 4096,
        "--algo.config.distill.fixed-bc-eval-log-interval": 1,
        "--algo.config.distill.fixed-bc-guard-enabled": True,
        "--algo.config.distill.fixed-bc-guard-reference-end-epoch": 2,
        "--algo.config.distill.fixed-bc-guard-max-reference-ratio": 2.0,
        "--algo.config.distill.fixed-bc-guard-absolute-max-mu-mse": 0.16,
        "--algo.config.distill.fixed-bc-guard-start-epoch": 10,
        "--algo.config.distill.fixed-bc-guard-consecutive-evals": 3,
    }
    values.update(overrides or {})
    return _camera_wbt_args(*(f"{name}={value}" for name, value in values.items()))


def test_fixed_bc_guard_valid_contract_survives_preflight() -> None:
    config = parse_and_validate(_fixed_bc_guard_args())

    distill = config.algo.config.distill
    assert distill.fixed_bc_guard_enabled is True
    assert distill.fixed_bc_guard_reference_end_epoch == 2
    assert distill.fixed_bc_guard_start_epoch == 10
    assert distill.fixed_bc_guard_consecutive_evals == 3


def _dagger_replay_args(overrides: dict[str, object] | None = None) -> list[str]:
    values: dict[str, object] = {
        "--algo.config.num-learning-iterations": 1000,
        "--algo.config.distill.enabled": True,
        "--algo.config.distill.mode": "dagger",
        "--algo.config.distill.policy-to-clone": "teacher.pt",
        "--algo.config.distill.bc-loss-coef": 1.0,
        "--algo.config.distill.switch-to-rl-after": -1,
        "--algo.config.distill.ppo-start-epoch": 0,
        "--algo.config.distill.dagger-end-epoch": 700,
        "--algo.config.distill.ppo-start-coeff": 0.0,
        "--algo.config.distill.ppo-target-coeff": 0.0,
        "--algo.config.distill.ppo-schedule-step-epochs": 100,
        "--algo.config.distill.dagger-loss-coef": 1.0,
        "--algo.config.distill.dagger-match-std": False,
        "--algo.config.distill.dagger-replay-enabled": True,
        "--algo.config.distill.dagger-replay-capacity": 512,
        "--algo.config.distill.dagger-replay-batch-size": 96,
        "--algo.config.distill.dagger-replay-fraction": 0.5,
        "--algo.config.distill.dagger-replay-seed": 17,
        "--algo.config.distill.fixed-bc-eval-num-samples": 4096,
        "--algo.config.distill.fixed-bc-eval-log-interval": 100,
        "--algo.config.distill.fixed-bc-guard-enabled": True,
        "--algo.config.distill.fixed-bc-guard-reference-end-epoch": 200,
        "--algo.config.distill.fixed-bc-guard-max-reference-ratio": 2.0,
        "--algo.config.distill.fixed-bc-guard-absolute-max-mu-mse": 0.16,
        "--algo.config.distill.fixed-bc-guard-start-epoch": 700,
        "--algo.config.distill.fixed-bc-guard-consecutive-evals": 3,
    }
    values.update(overrides or {})
    return _camera_wbt_args(*(f"{name}={value}" for name, value in values.items()))


def test_dagger_replay_r28_zero_ppo_schedule_survives_preflight() -> None:
    config = parse_and_validate(_dagger_replay_args())

    distill = config.algo.config.distill
    assert distill.dagger_replay_enabled is True
    assert distill.dagger_replay_capacity == 512
    assert distill.dagger_replay_batch_size == 96
    assert distill.dagger_replay_fraction == pytest.approx(0.5)
    assert distill.dagger_replay_seed == 17
    assert distill.ppo_start_epoch == 0
    assert distill.dagger_end_epoch == 700
    assert distill.ppo_start_coeff == pytest.approx(0.0)
    assert distill.ppo_target_coeff == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"--algo.config.distill.enabled": False},
            "requires enabled DAgger distillation",
        ),
        (
            {"--algo.config.distill.mode": "mse"},
            "requires enabled DAgger distillation",
        ),
        (
            {
                "--algo.config.distill.ppo-start-epoch": -1,
                "--algo.config.distill.dagger-end-epoch": -1,
                "--algo.config.distill.ppo-schedule-step-epochs": 0,
            },
            "requires an explicit PPO/DAgger schedule",
        ),
        (
            {"--algo.config.distill.ppo-target-coeff": 0.1},
            "operational float32 PPO to remain exactly zero",
        ),
        (
            {"--algo.config.distill.bc-loss-coef": 0.9},
            "requires bc_loss_coef=1.0",
        ),
        (
            {"--algo.config.distill.switch-to-rl-after": 1},
            "cannot be combined with switch_to_rl_after",
        ),
        (
            {"--algo.config.distill.dagger-match-std": True},
            "requires dagger_match_std=False",
        ),
        (
            {"--algo.config.distill.fixed-bc-guard-enabled": False,
             "--algo.config.distill.fixed-bc-guard-start-epoch": -1},
            "requires an enabled, non-empty fixed-BC guard",
        ),
        (
            {"--algo.config.distill.fixed-bc-eval-num-samples": 0},
            "requires an enabled, non-empty fixed-BC guard",
        ),
        (
            {"--algo.config.distill.ppo-target-coeff": "1e-50"},
            "rounds to zero in the float32 PPO actor loss graph",
        ),
    ],
)
def test_dagger_replay_invalid_scientific_contract_fails_preflight(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_and_validate(_dagger_replay_args(overrides))


def test_fixed_bc_guard_reference_at_zero_coeff_schedule_start_is_pure_bc() -> None:
    config = parse_and_validate(
        _fixed_bc_guard_args(
            {
                "--algo.config.distill.fixed-bc-guard-reference-end-epoch": 3,
            }
        )
    )

    assert config.algo.config.distill.ppo_start_coeff == pytest.approx(0.0)


@pytest.mark.parametrize("value", [1, "True", None])
def test_fixed_bc_guard_bool_validation_is_strict(value: object) -> None:
    with pytest.raises(ValueError, match="fixed_bc_guard_enabled must be a boolean"):
        strict_bool("algo.config.distill.fixed_bc_guard_enabled", value)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"--algo.config.distill.fixed-bc-guard-max-reference-ratio": "nan"},
            "fixed_bc_guard_max_reference_ratio must be finite",
        ),
        (
            {"--algo.config.distill.fixed-bc-guard-max-reference-ratio": 0.5},
            "fixed_bc_guard_max_reference_ratio must be >= 1.0",
        ),
        (
            {"--algo.config.distill.fixed-bc-guard-absolute-max-mu-mse": 0},
            "fixed_bc_guard_absolute_max_mu_mse must be finite and > 0",
        ),
        (
            {"--algo.config.distill.fixed-bc-guard-reference-end-epoch": -1},
            "fixed_bc_guard_reference_end_epoch must be >= 0",
        ),
        (
            {"--algo.config.distill.fixed-bc-guard-start-epoch": -1},
            "fixed_bc_guard_start_epoch must be >= 0",
        ),
        (
            {"--algo.config.distill.fixed-bc-guard-consecutive-evals": 0},
            "fixed_bc_guard_consecutive_evals must be > 0",
        ),
        (
            {"--algo.config.distill.fixed-bc-eval-num-samples": 0},
            "requires fixed_bc_eval_num_samples > 0",
        ),
        (
            {"--algo.config.distill.fixed-bc-eval-log-interval": 0},
            "fixed_bc_eval_log_interval must be > 0",
        ),
        (
            {"--algo.config.distill.enabled": False},
            "requires enabled DAgger distillation",
        ),
        (
            {"--algo.config.distill.mode": "mse"},
            "requires enabled DAgger distillation",
        ),
        (
            {
                "--algo.config.distill.ppo-start-epoch": -1,
                "--algo.config.distill.dagger-end-epoch": -1,
                "--algo.config.distill.ppo-schedule-step-epochs": 0,
            },
            "requires a valid PPO/DAgger schedule",
        ),
        (
            {"--algo.config.distill.fixed-bc-guard-reference-end-epoch": 11},
            "fixed_bc_guard_start_epoch must be >= fixed_bc_guard_reference_end_epoch",
        ),
        (
            {
                "--algo.config.distill.fixed-bc-guard-reference-end-epoch": 3,
                "--algo.config.distill.ppo-start-coeff": 0.1,
            },
            "reference period must remain pure BC",
        ),
        (
            {"--algo.config.distill.fixed-bc-guard-start-epoch": 9},
            "fixed_bc_guard_start_epoch must be >= dagger_end_epoch",
        ),
        (
            {
                "--algo.config.distill.ppo-start-epoch": 7,
                "--algo.config.distill.fixed-bc-guard-reference-end-epoch": 5,
                "--algo.config.distill.fixed-bc-eval-log-interval": 2,
            },
            "fixed_bc_guard_reference_end_epoch must coincide",
        ),
        (
            {
                "--algo.config.distill.ppo-start-epoch": 5,
                "--algo.config.distill.fixed-bc-guard-reference-end-epoch": 4,
                "--algo.config.distill.fixed-bc-guard-start-epoch": 11,
                "--algo.config.distill.fixed-bc-eval-log-interval": 2,
            },
            "fixed_bc_guard_start_epoch must coincide",
        ),
        (
            {
                "--algo.config.distill.fixed-bc-guard-reference-end-epoch": 2,
                "--algo.config.distill.fixed-bc-eval-log-interval": 2,
            },
            "reference period must contain at least three expected evaluations",
        ),
        (
            {"--algo.config.distill.fixed-bc-guard-start-epoch": 20},
            "fixed_bc_guard_start_epoch must be below algo.config.num_learning_iterations",
        ),
        (
            {"--algo.config.distill.fixed-bc-guard-start-epoch": 18},
            "enough scheduled evaluations",
        ),
        (
            {
                "--algo.config.distill.ppo-start-epoch": 21,
                "--algo.config.distill.dagger-end-epoch": 22,
                "--algo.config.distill.fixed-bc-guard-reference-end-epoch": 20,
                "--algo.config.distill.fixed-bc-guard-start-epoch": 22,
            },
            "fixed_bc_guard_reference_end_epoch must be below algo.config.num_learning_iterations",
        ),
        (
            {
                "--algo.config.distill.fixed-bc-guard-enabled": False,
                "--algo.config.distill.fixed-bc-guard-start-epoch": 0,
            },
            "Disabled fixed-BC guard requires .*fixed_bc_guard_start_epoch=-1",
        ),
    ],
)
def test_fixed_bc_guard_invalid_contract_fails_at_cli_preflight(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_and_validate(_fixed_bc_guard_args(overrides))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (["--perception.camera-pitch-deg=nan"], "camera_pitch_deg must be finite"),
        (["--perception.camera-far=-1"], "camera_far must be finite and > 0"),
        (["--perception.max-distance=inf"], "max_distance must be finite"),
        (["--perception.camera-warp-hole-prob=2"], "camera_warp_hole_prob must be a finite probability"),
        (
            ["--perception.camera-warp-additive-noise-std=-1"],
            "camera_warp_additive_noise_std must be finite and >= 0",
        ),
        (
            ["--perception.camera-warp-depth-offset-std=nan"],
            "camera_warp_depth_offset_std must be finite",
        ),
        (
            ["--perception.camera-far=2", "--perception.max-distance=3"],
            "max_distance must be within",
        ),
    ],
)
def test_effective_camera_numerics_fail_closed(overrides: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_and_validate(_camera_wbt_args(*overrides))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            ["--randomization.setup-terms.push-randomizer-state.params.push-interval-s=[2,-1]"],
            "push_interval_s must satisfy",
        ),
        (
            ["--randomization.setup-terms.push-randomizer-state.params.push-interval-s=[1]"],
            "push_interval_s must contain exactly 2",
        ),
        (
            ["--randomization.setup-terms.push-randomizer-state.params.push-interval-s=[1,1e309]"],
            r"push_interval_s\[1\] must be finite",
        ),
        (
            ["--randomization.setup-terms.push-randomizer-state.params.max-push-vel=[1,-1,1,1,1,1]"],
            "max_push_vel must be non-empty",
        ),
        (
            ["--randomization.setup-terms.push-randomizer-state.params.max-push-vel=[1,1]"],
            "must contain exactly 6 values",
        ),
    ],
)
def test_effective_wbt_push_numerics_fail_closed(overrides: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_and_validate(_camera_wbt_args(*overrides))


@pytest.mark.parametrize(
    ("start_iter", "end_iter", "total_iterations", "message"),
    [
        (6, 5, 10, "start_at_timestep_zero_prob_start_iter must be <="),
        (0, 6, 5, "start_at_timestep_zero_prob_end_iter must be <="),
    ],
)
def test_effective_reset_curriculum_bounds_fail_closed(
    start_iter: int,
    end_iter: int,
    total_iterations: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_and_validate(
            _camera_wbt_args(
                f"--algo.config.num-learning-iterations={total_iterations}",
                "--command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob=0.2",
                "--command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-end=1.0",
                "--command.setup-terms.motion-command.params.motion-config."
                f"start-at-timestep-zero-prob-start-iter={start_iter}",
                "--command.setup-terms.motion-command.params.motion-config."
                f"start-at-timestep-zero-prob-end-iter={end_iter}",
            )
        )


def test_nondefault_scientific_values_survive_final_parse_exactly_once() -> None:
    config = parse_and_validate(
        _camera_wbt_args(
            "--algo.config.num-learning-iterations=11",
            "--algo.config.actor-learning-rate=0.000321",
            "--algo.config.critic-learning-rate=0.000654",
            "--algo.config.schedule=fixed",
            "--algo.config.desired-kl=0.023",
            "--algo.config.min-actor-learning-rate=0.000123",
            "--algo.config.max-actor-learning-rate=0.000456",
            "--algo.config.min-critic-learning-rate=0.000234",
            "--algo.config.max-critic-learning-rate=0.000765",
            "--algo.config.entropy-coef=0.004",
            "--algo.config.init-noise-std=0.23",
            "--algo.config.module-dict.actor.min-noise-std=0.017",
            "--algo.config.distill.ppo-start-epoch=1",
            "--algo.config.distill.dagger-end-epoch=9",
            "--algo.config.distill.ppo-start-coeff=0.2",
            "--algo.config.distill.ppo-target-coeff=0.8",
            "--algo.config.distill.ppo-schedule-step-epochs=2",
            "--algo.config.distill.dagger-loss-coef=1.7",
            "--algo.config.distill.dagger-match-std=True",
            "--algo.config.distill.ppo-start-noise-std=0.31",
            "--algo.config.distill.ppo-start-noise-std-until-coeff=0.25",
            "--perception.camera-pitch-deg=11.25",
            "--perception.camera-far=4.0",
            "--perception.max-distance=3.5",
            "--perception.camera-warp-hole-prob=0.13",
            "--perception.camera-warp-additive-noise-std=0.02",
            "--perception.camera-warp-depth-offset-std=0.01",
            "--randomization.setup-terms.push-randomizer-state.params.push-interval-s=[0.75,1.25]",
            "--randomization.setup-terms.push-randomizer-state.params.max-push-vel=[0.11,0.22,0.33,0.44,0.55,0.66]",
            "--command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob=0.15",
            "--command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-end=0.85",
            "--command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-start-iter=0",
            "--command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-end-iter=11",
            "--command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob=0.05",
            "--command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-end=0.25",
            "--command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-start-iter=1",
            "--command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-end-iter=10",
        )
    )

    ppo = config.algo.config
    assert ppo.actor_learning_rate == pytest.approx(0.000321)
    assert ppo.critic_learning_rate == pytest.approx(0.000654)
    assert ppo.schedule == "fixed"
    assert ppo.desired_kl == pytest.approx(0.023)
    assert ppo.min_actor_learning_rate == pytest.approx(0.000123)
    assert ppo.max_actor_learning_rate == pytest.approx(0.000456)
    assert ppo.min_critic_learning_rate == pytest.approx(0.000234)
    assert ppo.max_critic_learning_rate == pytest.approx(0.000765)
    assert ppo.entropy_coef == pytest.approx(0.004)
    assert ppo.init_noise_std == pytest.approx(0.23)
    assert ppo.module_dict.actor.min_noise_std == pytest.approx(0.017)
    assert ppo.distill.ppo_start_coeff == pytest.approx(0.2)
    assert ppo.distill.ppo_target_coeff == pytest.approx(0.8)
    assert ppo.distill.dagger_match_std is True
    assert ppo.distill.ppo_start_noise_std == pytest.approx(0.31)
    assert config.perception.camera_pitch_deg == pytest.approx(11.25)
    assert config.perception.camera_warp_hole_prob == pytest.approx(0.13)
    push_params = config.randomization.setup_terms["push_randomizer_state"].params
    assert push_params["push_interval_s"] == pytest.approx([0.75, 1.25])
    assert push_params["max_push_vel"] == pytest.approx([0.11, 0.22, 0.33, 0.44, 0.55, 0.66])
    motion_config = config.command.setup_terms["motion_command"].params["motion_config"]
    assert motion_config.start_at_timestep_zero_prob == pytest.approx(0.15)
    assert motion_config.start_at_timestep_zero_prob_end == pytest.approx(0.85)
    assert motion_config.start_at_timestep_zero_prob_start_iter == 0
    assert motion_config.start_at_timestep_zero_prob_end_iter == 11
    assert motion_config.freeze_at_timestep_zero_prob == pytest.approx(0.05)
    assert motion_config.freeze_at_timestep_zero_prob_end == pytest.approx(0.25)
    assert motion_config.freeze_at_timestep_zero_prob_start_iter == 1
    assert motion_config.freeze_at_timestep_zero_prob_end_iter == 10
