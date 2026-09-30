from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from holosoma.managers.command.terms.wbt import (
    MAX_MOTION_TRANSITION_STEPS,
    MotionCommand,
    canonical_motion_transition_contract,
    motion_transition_contract_sha256,
)
from holosoma.utils.simulator_config import SimulatorType


_MISSING = object()
_GLOBAL_RUNTIME_SOURCE = {
    "version": 1,
    "source_clip_count": 30,
    "source_semantics": "global_multi_clip_runtime",
}
_NATIVE_SINGLE_SOURCE = {
    "version": 1,
    "source_clip_count": 1,
    "source_semantics": "single_clip_static",
}


def _rank_local_single_clip_command(
    *,
    global_clip_count: int,
    motion_transition_source: object = _MISSING,
) -> MotionCommand:
    command = object.__new__(MotionCommand)
    command.multi_clip = False
    command._rank_local_shard_metadata = {"global_clip_count": global_clip_count}
    command.motion = SimpleNamespace(
        num_clips=1,
        clip_ids=["clip_a"],
        clip_offsets=torch.tensor([0], dtype=torch.long),
        clip_lengths=torch.tensor([319], dtype=torch.long),
        time_step_total=319,
    )
    if motion_transition_source is not _MISSING:
        command.motion.motion_transition_source = motion_transition_source
        # New rank-local artifacts bind the exact same lineage record in both
        # the motion object-map root (loaded above) and rank metadata.
        command._rank_local_shard_metadata["motion_transition_source"] = (
            motion_transition_source
        )
    command.motion_cfg = SimpleNamespace(
        enable_default_pose_prepend=True,
        default_pose_prepend_duration_s=0.2,
        enable_default_pose_append=False,
        default_pose_append_duration_s=0.0,
    )
    command.device = "cpu"
    command.num_envs = 2
    command._env = SimpleNamespace(
        dt=0.02,
        simulator=SimpleNamespace(get_simulator_type=lambda: SimulatorType.ISAACSIM),
    )
    command._build_default_pose_state_robot_order = lambda _motion_idx: {
        "joint_pos": torch.zeros(29),
        "joint_vel": torch.zeros(29),
        "body_pos": torch.zeros(2, 3),
        "body_quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
        "body_lin_vel": torch.zeros(2, 3),
        "body_ang_vel": torch.zeros(2, 3),
        "object_pos": torch.zeros(3),
        "object_quat": torch.tensor([0.0, 0.0, 0.0, 1.0]),
        "object_lin_vel": torch.zeros(3),
    }
    return command


def test_filtered_single_clip_preserves_source_global_motion_clock() -> None:
    # The active optimization objective contains one clip.  Its authenticated
    # source lineage, independently, records that transition behavior came
    # from the original 30-clip bank.
    command = _rank_local_single_clip_command(
        global_clip_count=1,
        motion_transition_source=_GLOBAL_RUNTIME_SOURCE,
    )
    original_lengths = command.motion.clip_lengths.clone()
    original_offsets = command.motion.clip_offsets.clone()
    original_total = command.motion.time_step_total

    def _forbid_static_splice(*_args, **_kwargs) -> None:
        raise AssertionError("rank-local shard must not splice a transition into the motion")

    command._add_transition_to_motion = _forbid_static_splice
    command._maybe_add_default_pose_transition(prepend=True)
    command._maybe_add_default_pose_transition(prepend=False)

    assert torch.equal(command.motion.clip_lengths, original_lengths)
    assert torch.equal(command.motion.clip_offsets, original_offsets)
    assert command.motion.time_step_total == original_total
    assert command._uses_global_multi_clip_transition_semantics()


def test_filtered_single_clip_enables_runtime_prepend_and_bc_mask() -> None:
    command = _rank_local_single_clip_command(
        global_clip_count=1,
        motion_transition_source=_GLOBAL_RUNTIME_SOURCE,
    )

    command._configure_runtime_default_pose_prepend()
    command._activate_runtime_default_pose_prepend(torch.tensor([0], dtype=torch.long))

    assert command._runtime_default_pose_prepend_enabled is True
    assert command._runtime_default_pose_prepend_steps == 10
    assert command.get_runtime_default_pose_prepend_mask().tolist() == [True, False]
    assert command.motion.clip_lengths.tolist() == [319]


def test_native_single_clip_source_keeps_static_transition_semantics() -> None:
    command = _rank_local_single_clip_command(
        global_clip_count=1,
        motion_transition_source=_NATIVE_SINGLE_SOURCE,
    )
    state = {"sentinel": torch.tensor(1)}
    command._build_default_pose_state = Mock(return_value=state)
    command._add_transition_to_motion = Mock()

    assert not command._uses_global_multi_clip_transition_semantics()
    command._maybe_add_default_pose_transition(prepend=True)
    command._add_transition_to_motion.assert_called_once_with(state, 10, prepend=True)


def test_global_multi_clip_contract_records_runtime_prepend_and_no_append() -> None:
    command = _rank_local_single_clip_command(
        global_clip_count=1,
        motion_transition_source=_GLOBAL_RUNTIME_SOURCE,
    )
    command.motion_cfg.enable_default_pose_append = True
    command.motion_cfg.default_pose_append_duration_s = 0.2

    command._configure_runtime_default_pose_prepend()
    contract = command.get_motion_transition_contract()

    assert contract == {
        "version": 1,
        "control_dt_s": 0.02,
        "source_semantics": "global_multi_clip_runtime",
        "prepend": {"implementation": "runtime_hold", "applied": True, "steps": 10},
        "append": {"implementation": "none", "applied": False, "steps": 0},
    }
    assert len(motion_transition_contract_sha256(contract)) == 64


def test_standalone_contract_records_both_actual_static_splices() -> None:
    command = _rank_local_single_clip_command(
        global_clip_count=1,
        motion_transition_source=_NATIVE_SINGLE_SOURCE,
    )
    command.motion_cfg.enable_default_pose_append = True
    command.motion_cfg.default_pose_append_duration_s = 0.2
    command._build_default_pose_state = Mock(return_value={"sentinel": torch.tensor(1)})
    command._add_transition_to_motion = Mock()

    command._maybe_add_default_pose_transition(prepend=True)
    command._maybe_add_default_pose_transition(prepend=False)

    assert command.get_motion_transition_contract() == {
        "version": 1,
        "control_dt_s": 0.02,
        "source_semantics": "single_clip_static",
        "prepend": {"implementation": "static_splice", "applied": True, "steps": 10},
        "append": {"implementation": "static_splice", "applied": True, "steps": 10},
    }


@pytest.mark.parametrize("runtime", [False, True])
def test_transition_setup_rejects_more_than_deployment_safe_step_limit(runtime: bool) -> None:
    command = _rank_local_single_clip_command(
        global_clip_count=1,
        motion_transition_source=(
            _GLOBAL_RUNTIME_SOURCE if runtime else _NATIVE_SINGLE_SOURCE
        ),
    )
    duration = (MAX_MOTION_TRANSITION_STEPS + 1) * command._env.dt
    command.motion_cfg.default_pose_prepend_duration_s = duration

    with pytest.raises(ValueError, match="deployment-safe maximum"):
        if runtime:
            command._configure_runtime_default_pose_prepend()
        else:
            command._maybe_add_default_pose_transition(prepend=True)


def test_explicit_transition_source_does_not_change_single_clip_ddp_objective() -> None:
    command = _rank_local_single_clip_command(
        global_clip_count=1,
        motion_transition_source=_GLOBAL_RUNTIME_SOURCE,
    )
    command.clip_weighting_strategy = "uniform_clip"
    command._rank_local_shard_metadata.update(
        {
            "world_size": 8,
            "clip_cover_counts": {"clip_a": 8},
            "distributed_loss_weight": 1.0,
        }
    )

    command._configure_rank_local_clip_weighting()

    assert command._rank_local_shard_metadata["global_clip_count"] == 1
    assert command.motion.motion_transition_source["source_clip_count"] == 30
    assert command.distributed_loss_weight == pytest.approx(1.0)
    assert command._rank_local_inverse_cover_weights.tolist() == pytest.approx([1.0 / 8.0])


def test_legacy_rank_shard_without_transition_source_keeps_global_count_fallback() -> None:
    command = _rank_local_single_clip_command(global_clip_count=30)

    assert not hasattr(command.motion, "motion_transition_source")
    assert command._uses_global_multi_clip_transition_semantics()


def test_motion_transition_contract_schema_rejects_oversized_or_inconsistent_phase() -> None:
    contract = {
        "version": 1,
        "control_dt_s": 0.02,
        "source_semantics": "single_clip_static",
        "prepend": {"implementation": "none", "applied": False, "steps": 0},
        "append": {
            "implementation": "static_splice",
            "applied": True,
            "steps": MAX_MOTION_TRANSITION_STEPS + 1,
        },
    }
    with pytest.raises(ValueError, match=r"append\.steps"):
        canonical_motion_transition_contract(contract)

    contract["append"] = {"implementation": "static_splice", "applied": True, "steps": 1}
    with pytest.raises(ValueError, match=r"append\.steps"):
        canonical_motion_transition_contract(contract)

    contract["append"] = {"implementation": "static_splice", "applied": False, "steps": 0}
    with pytest.raises(ValueError, match="internally inconsistent"):
        canonical_motion_transition_contract(contract)


def _fixed_clip_distribution_command(
    clip_ids: list[int],
    *,
    num_clips: int,
    strategy: str = "uniform_clip",
    inverse_cover_weights: list[float] | None = None,
) -> MotionCommand:
    command = object.__new__(MotionCommand)
    command.device = "cpu"
    command.motion = SimpleNamespace(num_clips=num_clips)
    command._fixed_clip_ids = torch.tensor(clip_ids, dtype=torch.long)
    command.clip_weighting_strategy = strategy
    command._rank_local_inverse_cover_weights = (
        None
        if inverse_cover_weights is None
        else torch.tensor(inverse_cover_weights, dtype=torch.float32)
    )
    command._env = SimpleNamespace(is_evaluating=False)
    return command


def test_fixed_object_layout_accepts_exact_uniform_clip_distribution() -> None:
    command = _fixed_clip_distribution_command([0, 1, 2, 0, 1, 2], num_clips=3)

    command._validate_fixed_env_clip_sampling_distribution()


def test_fixed_object_layout_rejects_topology_dependent_clip_distribution() -> None:
    command = _fixed_clip_distribution_command([0, 1, 2, 0], num_clips=3)

    with pytest.raises(ValueError, match="topology-dependent objective"):
        command._validate_fixed_env_clip_sampling_distribution()


def test_fixed_object_layout_honors_rank_local_inverse_cover_mass() -> None:
    command = _fixed_clip_distribution_command(
        [0, 0, 1],
        num_clips=2,
        inverse_cover_weights=[1.0, 0.5],
    )

    command._validate_fixed_env_clip_sampling_distribution()


def test_fixed_object_layout_rejects_ignored_nonuniform_clip_strategy() -> None:
    command = _fixed_clip_distribution_command(
        [0, 1, 0, 1],
        num_clips=2,
        strategy="success_rate_adaptive",
    )

    with pytest.raises(ValueError, match="cannot honor clip_weighting_strategy"):
        command._validate_fixed_env_clip_sampling_distribution()
