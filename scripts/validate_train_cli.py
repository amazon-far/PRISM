#!/usr/bin/env python3
"""Parse a complete Holosoma training CLI without importing a simulator."""

from __future__ import annotations

import argparse
import math
import numbers
import struct
import sys
from collections.abc import Sequence
from pathlib import Path



import tyro  # noqa: E402

from holosoma.config_types.experiment import ExperimentConfig  # noqa: E402
from holosoma.config_values.experiment import AnnotatedExperimentConfig  # noqa: E402
from holosoma.utils.tyro_utils import TYRO_CONIFG  # noqa: E402


MOTION_END_FUNC = "holosoma.managers.termination.terms.wbt:motion_ends"
CONTACT_REGION_BODY_NAMES = {
    "left_wrist": "left_wrist_yaw_link",
    "right_wrist": "right_wrist_yaw_link",
    "left_elbow": "left_elbow_link",
    "right_elbow": "right_elbow_link",
    "left_wrist_roll": "left_wrist_roll_link",
    "right_wrist_roll": "right_wrist_roll_link",
    "left_wrist_pitch": "left_wrist_pitch_link",
    "right_wrist_pitch": "right_wrist_pitch_link",
    "torso": "torso_link",
}
CONTACT_REGION_ALIASES = {"left_palm": "left_wrist", "right_palm": "right_wrist"}


def _finite_real(name: str, value: object) -> float:
    """Return a strict finite scalar or raise an actionable config error."""

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real number, got {value!r}.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return parsed


def _positive_real(name: str, value: object) -> float:
    parsed = _finite_real(name, value)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}.")
    return parsed


def _nonnegative_real(name: str, value: object) -> float:
    parsed = _finite_real(name, value)
    if parsed < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}.")
    return parsed


def _probability(name: str, value: object) -> float:
    parsed = _finite_real(name, value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be a finite probability in [0, 1], got {value!r}.")
    return parsed


def _strict_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    return int(value)


def _strict_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean, got {value!r}.")
    return value


def _scheduled_ppo_coeff(
    *,
    current_epoch: int,
    start_epoch: int,
    end_epoch: int,
    start_coeff: float,
    target_coeff: float,
    step_epochs: int,
) -> float:
    """Mirror PPO's scheduled blend coefficient without importing Torch/Isaac."""

    if current_epoch < start_epoch:
        return 0.0
    if current_epoch >= end_epoch:
        return target_coeff

    total_epochs = max(1, end_epoch - start_epoch)
    ppo_epochs = max(0, current_epoch - start_epoch)
    coeff_span = target_coeff - start_coeff
    if step_epochs > 0:
        total_steps = max(1, (total_epochs + step_epochs - 1) // step_epochs)
        completed_steps = max(0, ppo_epochs // step_epochs)
        progress = min(float(completed_steps) / float(total_steps), 1.0)
        return start_coeff + progress * coeff_span

    progress = min(float(ppo_epochs) / float(total_epochs), 1.0)
    return start_coeff + progress * coeff_span


def _operational_float32_loss_weight(value: float) -> float:
    """Mirror the scalar materialization used by PPO's float32 actor graph."""

    parsed = float(value)
    try:
        return float(struct.unpack("!f", struct.pack("!f", parsed))[0])
    except OverflowError:
        return math.copysign(float("inf"), parsed)


def _operational_ppo_coefficient(value: float) -> float:
    """Clamp a configured PPO coefficient and materialize it as float32."""

    return _operational_float32_loss_weight(max(0.0, min(1.0, float(value))))


def _validate_dagger_replay_config(
    distill: object,
    *,
    schedule_enabled: bool,
    start_coeff: float,
    target_coeff: float,
) -> None:
    """Reject replay configurations that could leak off-policy rows into PPO."""

    replay_enabled = _strict_bool(
        "algo.config.distill.dagger_replay_enabled",
        getattr(distill, "dagger_replay_enabled"),
    )
    if not replay_enabled:
        return

    distill_enabled = _strict_bool(
        "algo.config.distill.enabled",
        getattr(distill, "enabled"),
    )
    distill_mode = str(getattr(distill, "mode")).strip().lower()
    if not distill_enabled or distill_mode != "dagger":
        raise ValueError(
            "algo.config.distill.dagger_replay_enabled requires enabled DAgger distillation."
        )

    if not schedule_enabled:
        raise ValueError(
            "algo.config.distill.dagger_replay_enabled requires an explicit PPO/DAgger schedule "
            "whose operational PPO coefficient remains exactly zero for the entire target."
        )
    operational_endpoints = {
        "ppo_start_coeff": _operational_ppo_coefficient(start_coeff),
        "ppo_target_coeff": _operational_ppo_coefficient(target_coeff),
    }
    if any(value != 0.0 for value in operational_endpoints.values()):
        raise ValueError(
            "algo.config.distill.dagger_replay_enabled requires operational float32 PPO to "
            "remain exactly zero for the entire target; got "
            + ", ".join(f"{name}={value}" for name, value in operational_endpoints.items())
            + "."
        )

    raw_bc_loss_coef = getattr(distill, "bc_loss_coef")
    if raw_bc_loss_coef is None:
        raw_bc_loss_coef = getattr(distill, "loss_coef")
    bc_loss_coef = _finite_real(
        "algo.config.distill.bc_loss_coef",
        raw_bc_loss_coef,
    )
    if bc_loss_coef != 1.0:
        raise ValueError(
            "algo.config.distill.dagger_replay_enabled requires bc_loss_coef=1.0 as the "
            f"pure-BC schedule sentinel, got {bc_loss_coef}."
        )

    switch_to_rl_after = _strict_int(
        "algo.config.distill.switch_to_rl_after",
        getattr(distill, "switch_to_rl_after"),
    )
    if switch_to_rl_after > 0:
        raise ValueError(
            "algo.config.distill.dagger_replay_enabled cannot be combined with "
            "switch_to_rl_after; replay is a pure-BC objective."
        )

    dagger_match_std = _strict_bool(
        "algo.config.distill.dagger_match_std",
        getattr(distill, "dagger_match_std"),
    )
    if dagger_match_std:
        raise ValueError(
            "algo.config.distill.dagger_replay_enabled requires dagger_match_std=False because "
            "the replay schema contains authenticated teacher actions but no teacher std."
        )

    fixed_guard_enabled = _strict_bool(
        "algo.config.distill.fixed_bc_guard_enabled",
        getattr(distill, "fixed_bc_guard_enabled"),
    )
    fixed_num_samples = _strict_int(
        "algo.config.distill.fixed_bc_eval_num_samples",
        getattr(distill, "fixed_bc_eval_num_samples"),
    )
    if not fixed_guard_enabled or fixed_num_samples <= 0:
        raise ValueError(
            "algo.config.distill.dagger_replay_enabled requires an enabled, non-empty fixed-BC "
            "guard so replay collection starts only after a disjoint immutable evaluation set exists."
        )


def _validate_fixed_bc_guard_config(
    algo_config: object,
    distill: object,
    *,
    schedule_enabled: bool,
    start_epoch: int,
    end_epoch: int,
    start_coeff: float,
    target_coeff: float,
    step_epochs: int,
) -> None:
    """Fail before simulator startup when the fixed-BC guard cannot be sound."""

    guard_enabled = _strict_bool(
        "algo.config.distill.fixed_bc_guard_enabled",
        getattr(distill, "fixed_bc_guard_enabled"),
    )
    distill_enabled = _strict_bool(
        "algo.config.distill.enabled",
        getattr(distill, "enabled"),
    )
    distill_mode = str(getattr(distill, "mode")).strip().lower()

    num_samples = _strict_int(
        "algo.config.distill.fixed_bc_eval_num_samples",
        getattr(distill, "fixed_bc_eval_num_samples"),
    )
    if num_samples < 0:
        raise ValueError(
            "algo.config.distill.fixed_bc_eval_num_samples must be >= 0, "
            f"got {num_samples}."
        )
    log_interval = _strict_int(
        "algo.config.distill.fixed_bc_eval_log_interval",
        getattr(distill, "fixed_bc_eval_log_interval"),
    )
    if log_interval <= 0:
        raise ValueError(
            "algo.config.distill.fixed_bc_eval_log_interval must be > 0, "
            f"got {log_interval}."
        )
    effective_num_samples = num_samples if distill_mode == "dagger" else 0
    if effective_num_samples == 0 and log_interval != 1:
        raise ValueError(
            "algo.config.distill.fixed_bc_eval_log_interval is only consumed when the effective "
            "DAgger fixed_bc_eval_num_samples budget is positive."
        )

    reference_end = _strict_int(
        "algo.config.distill.fixed_bc_guard_reference_end_epoch",
        getattr(distill, "fixed_bc_guard_reference_end_epoch"),
    )
    if reference_end < 0:
        raise ValueError(
            "algo.config.distill.fixed_bc_guard_reference_end_epoch must be >= 0, "
            f"got {reference_end}."
        )
    max_reference_ratio = _positive_real(
        "algo.config.distill.fixed_bc_guard_max_reference_ratio",
        getattr(distill, "fixed_bc_guard_max_reference_ratio"),
    )
    if max_reference_ratio < 1.0:
        raise ValueError(
            "algo.config.distill.fixed_bc_guard_max_reference_ratio must be >= 1.0 so the "
            "post-reference ceiling cannot be tighter than the observed reference minimum."
        )
    _positive_real(
        "algo.config.distill.fixed_bc_guard_absolute_max_mu_mse",
        getattr(distill, "fixed_bc_guard_absolute_max_mu_mse"),
    )
    guard_start = _strict_int(
        "algo.config.distill.fixed_bc_guard_start_epoch",
        getattr(distill, "fixed_bc_guard_start_epoch"),
    )
    consecutive_evals = _strict_int(
        "algo.config.distill.fixed_bc_guard_consecutive_evals",
        getattr(distill, "fixed_bc_guard_consecutive_evals"),
    )
    if consecutive_evals <= 0:
        raise ValueError(
            "algo.config.distill.fixed_bc_guard_consecutive_evals must be > 0, "
            f"got {consecutive_evals}."
        )

    if not guard_enabled:
        if guard_start != -1:
            raise ValueError(
                "Disabled fixed-BC guard requires "
                "algo.config.distill.fixed_bc_guard_start_epoch=-1, "
                f"got {guard_start}."
            )
        return

    if guard_start < 0:
        raise ValueError(
            "algo.config.distill.fixed_bc_guard_start_epoch must be >= 0 when the guard is enabled, "
            f"got {guard_start}."
        )
    if distill_mode != "dagger" or not distill_enabled:
        raise ValueError(
            "algo.config.distill.fixed_bc_guard_enabled requires enabled DAgger distillation."
        )
    if effective_num_samples <= 0:
        raise ValueError(
            "algo.config.distill.fixed_bc_guard_enabled requires fixed_bc_eval_num_samples > 0."
        )
    if not schedule_enabled:
        raise ValueError(
            "algo.config.distill.fixed_bc_guard_enabled requires a valid PPO/DAgger schedule."
        )
    if guard_start < reference_end:
        raise ValueError(
            "algo.config.distill.fixed_bc_guard_start_epoch must be >= "
            "fixed_bc_guard_reference_end_epoch; got "
            f"{guard_start} < {reference_end}."
        )

    reference_ppo_coeff = _scheduled_ppo_coeff(
        current_epoch=reference_end,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        start_coeff=start_coeff,
        target_coeff=target_coeff,
        step_epochs=step_epochs,
    )
    if reference_ppo_coeff != 0.0:
        raise ValueError(
            "algo.config.distill fixed-BC guard reference period must remain pure BC through "
            "fixed_bc_guard_reference_end_epoch; the configured PPO/DAgger schedule has "
            f"ppo_coeff={reference_ppo_coeff} at that iteration."
        )
    if guard_start < end_epoch:
        raise ValueError(
            "algo.config.distill.fixed_bc_guard_start_epoch must be >= dagger_end_epoch so "
            "post-reference exceedances are evaluated only after the configured PPO ramp completes."
        )
    if reference_end % log_interval != 0:
        raise ValueError(
            "algo.config.distill.fixed_bc_guard_reference_end_epoch must coincide with a "
            "fixed-BC evaluation iteration."
        )
    if guard_start % log_interval != 0:
        raise ValueError(
            "algo.config.distill.fixed_bc_guard_start_epoch must coincide with a fixed-BC "
            "evaluation iteration."
        )
    expected_reference_evals = reference_end // log_interval + 1
    if expected_reference_evals < 3:
        raise ValueError(
            "algo.config.distill fixed-BC guard reference period must contain at least three "
            f"expected evaluations; got {expected_reference_evals}."
        )

    total_iterations = _strict_int(
        "algo.config.num_learning_iterations",
        getattr(algo_config, "num_learning_iterations"),
    )
    if reference_end >= total_iterations:
        raise ValueError(
            "algo.config.distill.fixed_bc_guard_reference_end_epoch must be below "
            "algo.config.num_learning_iterations."
        )
    if guard_start >= total_iterations:
        raise ValueError(
            "algo.config.distill.fixed_bc_guard_start_epoch must be below "
            "algo.config.num_learning_iterations."
        )
    final_possible_trip_iteration = guard_start + (consecutive_evals - 1) * log_interval
    if final_possible_trip_iteration >= total_iterations:
        raise ValueError(
            "algo.config.distill fixed-BC guard must have enough scheduled evaluations to "
            "reach fixed_bc_guard_consecutive_evals before "
            "algo.config.num_learning_iterations; last required evaluation="
            f"{final_possible_trip_iteration}, run end={total_iterations}."
        )


def _validate_ppo_scientific_config(config: ExperimentConfig) -> None:
    algo_config = config.algo.config
    if not hasattr(algo_config, "distill"):
        return

    actor_lr = _positive_real("algo.config.actor_learning_rate", algo_config.actor_learning_rate)
    critic_lr = _positive_real("algo.config.critic_learning_rate", algo_config.critic_learning_rate)
    lr_schedule = str(getattr(algo_config, "schedule"))
    if lr_schedule not in {"adaptive", "fixed"}:
        raise ValueError(
            "algo.config.schedule must be exactly 'adaptive' or 'fixed', "
            f"got {lr_schedule!r}."
        )
    desired_kl_raw = getattr(algo_config, "desired_kl", None)
    if desired_kl_raw is None:
        if lr_schedule == "adaptive":
            raise ValueError("algo.config.schedule='adaptive' requires a positive algo.config.desired_kl.")
    else:
        _positive_real("algo.config.desired_kl", desired_kl_raw)
    _nonnegative_real("algo.config.entropy_coef", algo_config.entropy_coef)
    init_noise_std = _positive_real("algo.config.init_noise_std", algo_config.init_noise_std)

    for optimizer_name, initial_lr in (("actor", actor_lr), ("critic", critic_lr)):
        minimum_raw = getattr(algo_config, f"min_{optimizer_name}_learning_rate", None)
        maximum_raw = getattr(algo_config, f"max_{optimizer_name}_learning_rate", None)
        minimum = None
        maximum = None
        if minimum_raw is not None:
            minimum = _positive_real(
                f"algo.config.min_{optimizer_name}_learning_rate",
                minimum_raw,
            )
        if maximum_raw is not None:
            maximum = _positive_real(
                f"algo.config.max_{optimizer_name}_learning_rate",
                maximum_raw,
            )
        if minimum is not None and initial_lr < minimum:
            raise ValueError(
                f"algo.config.{optimizer_name}_learning_rate must be >= "
                f"algo.config.min_{optimizer_name}_learning_rate, got {initial_lr} < {minimum}."
            )
        if maximum is not None and initial_lr > maximum:
            raise ValueError(
                f"algo.config.{optimizer_name}_learning_rate must be <= "
                f"algo.config.max_{optimizer_name}_learning_rate, got {initial_lr} > {maximum}."
            )
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(
                f"algo.config.min_{optimizer_name}_learning_rate must be <= "
                f"algo.config.max_{optimizer_name}_learning_rate, got {minimum} > {maximum}."
            )

    actor_module = algo_config.module_dict.actor
    actor_min_noise: float | None = None
    actor_max_noise: float | None = None
    if actor_module.min_noise_std is not None:
        actor_min_noise = _positive_real(
            "algo.config.module_dict.actor.min_noise_std",
            actor_module.min_noise_std,
        )
        if init_noise_std < actor_min_noise:
            raise ValueError(
                "algo.config.init_noise_std must be >= "
                "algo.config.module_dict.actor.min_noise_std, "
                f"got {init_noise_std} < {actor_min_noise}."
            )
    if actor_module.max_noise_std is not None:
        actor_max_noise = _positive_real(
            "algo.config.module_dict.actor.max_noise_std",
            actor_module.max_noise_std,
        )
        if init_noise_std > actor_max_noise:
            raise ValueError(
                "algo.config.init_noise_std must be <= "
                "algo.config.module_dict.actor.max_noise_std, "
                f"got {init_noise_std} > {actor_max_noise}."
            )
    if actor_min_noise is not None and actor_max_noise is not None and actor_min_noise > actor_max_noise:
        raise ValueError(
            "algo.config.module_dict.actor.min_noise_std must be <= "
            "algo.config.module_dict.actor.max_noise_std, "
            f"got {actor_min_noise} > {actor_max_noise}."
        )

    distill = algo_config.distill
    start_coeff = _probability("algo.config.distill.ppo_start_coeff", distill.ppo_start_coeff)
    target_coeff = _probability("algo.config.distill.ppo_target_coeff", distill.ppo_target_coeff)
    if start_coeff > target_coeff:
        raise ValueError(
            "algo.config.distill.ppo_start_coeff must be <= "
            "algo.config.distill.ppo_target_coeff, "
            f"got {start_coeff} > {target_coeff}."
        )
    for coefficient_name, coefficient in (
        ("ppo_start_coeff", start_coeff),
        ("ppo_target_coeff", target_coeff),
    ):
        operational_coefficient = _operational_ppo_coefficient(coefficient)
        if coefficient > 0.0 and operational_coefficient <= 0.0:
            raise ValueError(
                f"algo.config.distill.{coefficient_name} is positive as a Python scalar but "
                "rounds to zero in the float32 PPO actor loss graph: "
                f"Python={coefficient}, float32={operational_coefficient}."
            )
    start_epoch = _strict_int("algo.config.distill.ppo_start_epoch", distill.ppo_start_epoch)
    end_epoch = _strict_int("algo.config.distill.dagger_end_epoch", distill.dagger_end_epoch)
    schedule_disabled = start_epoch == -1 and end_epoch == -1
    schedule_enabled = start_epoch >= 0 and end_epoch > start_epoch
    if not schedule_disabled and not schedule_enabled:
        raise ValueError(
            "algo.config.distill PPO/DAgger schedule must either use -1/-1 or satisfy "
            f"0 <= ppo_start_epoch < dagger_end_epoch, got {start_epoch}, {end_epoch}."
        )
    step_epochs = _strict_int(
        "algo.config.distill.ppo_schedule_step_epochs",
        distill.ppo_schedule_step_epochs,
    )
    if step_epochs < 0:
        raise ValueError(
            "algo.config.distill.ppo_schedule_step_epochs must be >= 0, "
            f"got {step_epochs}."
        )
    if step_epochs > 0 and not schedule_enabled:
        raise ValueError(
            "algo.config.distill.ppo_schedule_step_epochs requires an enabled "
            "ppo_start_epoch/dagger_end_epoch schedule."
        )
    dagger_loss_coef = _nonnegative_real(
        "algo.config.distill.dagger_loss_coef",
        distill.dagger_loss_coef,
    )
    if schedule_enabled and dagger_loss_coef <= 0.0:
        raise ValueError(
            "An enabled PPO/DAgger schedule requires "
            "algo.config.distill.dagger_loss_coef > 0."
        )
    noise_until_coeff = _probability(
        "algo.config.distill.ppo_start_noise_std_until_coeff",
        distill.ppo_start_noise_std_until_coeff,
    )
    if distill.ppo_start_noise_std is not None:
        ppo_start_noise = _positive_real(
            "algo.config.distill.ppo_start_noise_std",
            distill.ppo_start_noise_std,
        )
        if not schedule_enabled:
            raise ValueError(
                "algo.config.distill.ppo_start_noise_std requires an enabled "
                "ppo_start_epoch/dagger_end_epoch schedule."
            )
        if actor_min_noise is not None and ppo_start_noise < actor_min_noise:
            raise ValueError(
                "algo.config.distill.ppo_start_noise_std must be >= "
                "algo.config.module_dict.actor.min_noise_std, "
                f"got {ppo_start_noise} < {actor_min_noise}."
            )
        if actor_max_noise is not None and ppo_start_noise > actor_max_noise:
            raise ValueError(
                "algo.config.distill.ppo_start_noise_std must be <= "
                "algo.config.module_dict.actor.max_noise_std, "
                f"got {ppo_start_noise} > {actor_max_noise}."
            )
        if step_epochs == 0 and start_coeff > noise_until_coeff:
            raise ValueError(
                "A linear PPO/DAgger schedule starts above "
                "algo.config.distill.ppo_start_noise_std_until_coeff, so its noise cap would never apply."
            )

    _validate_dagger_replay_config(
        distill,
        schedule_enabled=schedule_enabled,
        start_coeff=start_coeff,
        target_coeff=target_coeff,
    )
    _validate_fixed_bc_guard_config(
        algo_config,
        distill,
        schedule_enabled=schedule_enabled,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        start_coeff=start_coeff,
        target_coeff=target_coeff,
        step_epochs=step_epochs,
    )


def _validate_camera_scientific_config(config: ExperimentConfig) -> None:
    perception = config.perception
    if not perception.enabled or perception.output_mode != "camera_depth":
        return

    _finite_real("perception.camera_pitch_deg", perception.camera_pitch_deg)
    near = _positive_real("perception.camera_near", perception.camera_near)
    far = _positive_real("perception.camera_far", perception.camera_far)
    max_distance = _positive_real("perception.max_distance", perception.max_distance)
    if near >= far:
        raise ValueError(
            f"perception.camera_near must be < perception.camera_far, got {near} >= {far}."
        )
    if max_distance < near or max_distance > far:
        raise ValueError(
            "perception.max_distance must be within [camera_near, camera_far], "
            f"got near={near}, max_distance={max_distance}, far={far}."
        )
    _probability("perception.camera_warp_hole_prob", perception.camera_warp_hole_prob)
    _nonnegative_real(
        "perception.camera_warp_additive_noise_std",
        perception.camera_warp_additive_noise_std,
    )
    _nonnegative_real(
        "perception.camera_warp_depth_offset_std",
        perception.camera_warp_depth_offset_std,
    )


def _numeric_sequence(name: str, value: object) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a numeric sequence, got {value!r}.")
    return [_finite_real(f"{name}[{index}]", item) for index, item in enumerate(value)]


def _validate_push_scientific_config(config: ExperimentConfig) -> None:
    if config.randomization is None:
        return
    term = config.randomization.setup_terms.get("push_randomizer_state")
    if term is None:
        return
    params = term.params
    interval = _numeric_sequence(
        "randomization.push_randomizer_state.push_interval_s",
        params.get("push_interval_s"),
    )
    if len(interval) != 2:
        raise ValueError(
            "randomization.push_randomizer_state.push_interval_s must contain exactly 2 values, "
            f"got {interval!r}."
        )
    if interval[0] <= 0.0 or interval[1] <= 0.0 or interval[0] > interval[1]:
        raise ValueError(
            "randomization.push_randomizer_state.push_interval_s must satisfy "
            f"0 < low <= high, got {interval!r}."
        )
    max_push_vel = _numeric_sequence(
        "randomization.push_randomizer_state.max_push_vel",
        params.get("max_push_vel"),
    )
    if not max_push_vel or any(value < 0.0 for value in max_push_vel):
        raise ValueError(
            "randomization.push_randomizer_state.max_push_vel must be non-empty, finite, "
            f"and non-negative, got {max_push_vel!r}."
        )
    if (".wbt." in config.env_class or "WholeBodyTracking" in config.env_class) and len(max_push_vel) != 6:
        raise ValueError(
            "Whole-body tracking randomization.push_randomizer_state.max_push_vel must contain "
            f"exactly 6 values, got {max_push_vel!r}."
        )


def _validate_motion_reset_curricula(config: ExperimentConfig) -> None:
    if config.command is None:
        return
    term = config.command.setup_terms.get("motion_command")
    if term is None:
        return
    motion_config = term.params.get("motion_config")
    if motion_config is None:
        return
    total_iterations = getattr(config.algo.config, "num_learning_iterations", None)
    if total_iterations is not None:
        total_iterations = _strict_int("algo.config.num_learning_iterations", total_iterations)

    for prefix in ("start_at_timestep_zero_prob", "freeze_at_timestep_zero_prob"):
        _probability(f"command.motion_config.{prefix}", getattr(motion_config, prefix))
        end_name = f"{prefix}_end"
        start_iter_name = f"{prefix}_start_iter"
        end_iter_name = f"{prefix}_end_iter"
        end_value = getattr(motion_config, end_name)
        start_iter_value = getattr(motion_config, start_iter_name)
        end_iter_value = getattr(motion_config, end_iter_name)
        provided = (
            end_value is not None,
            start_iter_value is not None,
            end_iter_value is not None,
        )
        if any(provided) and not all(provided):
            raise ValueError(
                f"command.motion_config.{prefix} schedule must set {end_name}, "
                f"{start_iter_name}, and {end_iter_name} together."
            )
        if not all(provided):
            continue
        _probability(f"command.motion_config.{end_name}", end_value)
        start_iter = _strict_int(
            f"command.motion_config.{start_iter_name}",
            start_iter_value,
        )
        end_iter = _strict_int(
            f"command.motion_config.{end_iter_name}",
            end_iter_value,
        )
        if start_iter < 0 or end_iter < 0:
            raise ValueError(
                f"command.motion_config.{prefix} schedule iterations must be non-negative, "
                f"got {start_iter}, {end_iter}."
            )
        if start_iter > end_iter:
            raise ValueError(
                f"command.motion_config.{start_iter_name} must be <= {end_iter_name}, "
                f"got {start_iter} > {end_iter}."
            )
        if total_iterations is not None and end_iter > total_iterations:
            raise ValueError(
                f"command.motion_config.{end_iter_name} must be <= "
                f"algo.config.num_learning_iterations, got {end_iter} > {total_iterations}."
            )


def _validate_scientific_train_config(config: ExperimentConfig) -> None:
    """Validate the effective parsed config, after every launcher override."""

    _validate_ppo_scientific_config(config)
    _validate_camera_scientific_config(config)
    _validate_push_scientific_config(config)
    _validate_motion_reset_curricula(config)


def _validate_unique_long_options(train_args: Sequence[str]) -> None:
    """Reject ambiguous Tyro overrides before parsing the training command.

    Tyro accepts both ``--field=value`` and ``--field value`` and treats
    underscores and hyphens as equivalent spellings.  A later occurrence can
    therefore silently replace a launcher-owned value while the launcher's
    provenance still records the earlier value.  Scientific launches must
    have one unambiguous source for every long option.
    """

    occurrences: dict[str, list[tuple[int, str]]] = {}
    for index, raw_arg in enumerate(train_args, start=1):
        if raw_arg == "--" or not raw_arg.startswith("--"):
            continue
        option_spelling = raw_arg.split("=", 1)[0]
        canonical_name = option_spelling[2:].replace("_", "-")
        if not canonical_name:
            continue
        occurrences.setdefault(canonical_name, []).append((index, option_spelling))

    duplicates = {name: values for name, values in occurrences.items() if len(values) > 1}
    if not duplicates:
        return

    details = []
    for canonical_name, values in duplicates.items():
        locations = ", ".join(f"argv[{index}]={spelling}" for index, spelling in values)
        details.append(f"--{canonical_name} ({locations})")
    raise ValueError(
        "Duplicate train CLI long option(s) are forbidden because a later Tyro value can silently "
        "override launcher provenance: " + "; ".join(details)
    )


def _normalized_contact_regions(raw_names: object, field_name: str) -> list[str]:
    names = [raw_names] if isinstance(raw_names, str) else list(raw_names)  # type: ignore[arg-type]
    normalized = [CONTACT_REGION_ALIASES.get(str(name), str(name)) for name in names]
    if not normalized or len(normalized) != len(set(normalized)):
        raise ValueError(f"offline_contact_guidance {field_name} must be non-empty and unique: {normalized!r}")
    unsupported = sorted(set(normalized) - set(CONTACT_REGION_BODY_NAMES))
    if unsupported:
        raise ValueError(f"offline_contact_guidance {field_name} has unsupported regions: {unsupported}")
    return normalized


def _validate_offline_contact_body_map(config: ExperimentConfig) -> None:
    reward_cfg = config.reward
    if reward_cfg is None:
        return
    term = reward_cfg.terms.get("offline_contact_guidance")
    if term is None or float(term.weight) == 0.0:
        return

    params = term.params
    required_regions: list[str] = []
    if float(params.get("wrist_weight", params.get("target_weight", 0.0))) != 0.0:
        required_regions.extend(
            _normalized_contact_regions(
                params.get("wrist_region_names", ["left_wrist", "right_wrist"]),
                "wrist_region_names",
            )
        )
    if float(params.get("contact_weight", 1.0)) != 0.0:
        required_regions.extend(
            _normalized_contact_regions(
                params.get("contact_region_names", params.get("region_names", ["left_wrist", "right_wrist"])),
                "contact_region_names",
            )
        )

    body_names = set(config.robot.body_names)
    missing_bodies = sorted(
        {
            CONTACT_REGION_BODY_NAMES[region_name]
            for region_name in required_regions
            if CONTACT_REGION_BODY_NAMES[region_name] not in body_names
        }
    )
    if missing_bodies:
        raise ValueError(
            "offline_contact_guidance targets regions unavailable in the parsed robot body map: "
            f"{missing_bodies}"
        )


def parse_and_validate_train_cli(
    train_args: Sequence[str],
    *,
    expected_motion_end_mode: str = "",
) -> ExperimentConfig:
    """Parse the exact train argv and validate the requested motion-end contract."""

    _validate_unique_long_options(train_args)
    config = tyro.cli(AnnotatedExperimentConfig, args=list(train_args), config=TYRO_CONIFG)
    normalized_mode = expected_motion_end_mode.strip().lower().replace("-", "_")
    if normalized_mode not in {"", "episodic", "continuing"}:
        raise ValueError(
            "expected_motion_end_mode must be empty, episodic, or continuing; "
            f"got {expected_motion_end_mode!r}"
        )

    terms = {} if config.termination is None else config.termination.terms
    motion_end_term = terms.get("motion_ends")
    if normalized_mode == "episodic":
        if motion_end_term is None:
            raise ValueError(
                "STUDENT_MOTION_END_MODE=episodic requires termination term 'motion_ends'; "
                f"parsed terms={sorted(terms)}"
            )
        if motion_end_term.func != MOTION_END_FUNC or motion_end_term.is_timeout:
            raise ValueError(
                "Invalid episodic motion_ends termination contract: "
                f"func={motion_end_term.func!r} is_timeout={motion_end_term.is_timeout!r}"
            )
    elif normalized_mode == "continuing" and motion_end_term is not None:
        raise ValueError(
            "STUDENT_MOTION_END_MODE=continuing requires motion_ends to be absent; "
            f"parsed terms={sorted(terms)}"
        )

    _validate_offline_contact_body_map(config)
    _validate_scientific_train_config(config)

    return config


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if "--" not in raw_args:
        raise SystemExit("usage: validate_train_cli.py [--expected-motion-end-mode MODE] -- <train argv...>")
    separator = raw_args.index("--")
    preflight_args = raw_args[:separator]
    train_args = raw_args[separator + 1 :]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-motion-end-mode",
        default="",
        choices=("", "episodic", "continuing"),
    )
    parsed = parser.parse_args(preflight_args)
    if not train_args:
        raise SystemExit("No train CLI arguments were provided after --")

    config = parse_and_validate_train_cli(
        train_args,
        expected_motion_end_mode=parsed.expected_motion_end_mode,
    )
    terms = [] if config.termination is None else sorted(config.termination.terms)
    print(
        "[INFO] train_cli_preflight_ok "
        f"expected_motion_end_mode={parsed.expected_motion_end_mode or '<unset>'} "
        f"termination_terms={terms}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
