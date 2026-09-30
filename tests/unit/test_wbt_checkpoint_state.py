from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.multiprocessing as mp

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.ppo.ppo import PPO
from holosoma.config_types.curriculum import CurriculumManagerCfg, CurriculumTermCfg
from holosoma.config_types.reward import RewardTermCfg
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.terms.wbt import AdaptiveTimestepsSampler, MotionCommand
from holosoma.managers.curriculum.terms.locomotion import (
    AverageEpisodeLengthTracker,
    PenaltyCurriculum,
    WObjectDifficultyCurriculum,
    configure_reward_penalty,
    update_reward_penalty,
)
from holosoma.managers.curriculum.manager import CurriculumManager
from holosoma.managers.observation.terms.wbt import clip_phase, drop_button


def _make_motion_command() -> MotionCommand:
    command = object.__new__(MotionCommand)
    command.device = "cpu"
    command.motion = SimpleNamespace(
        clip_ids=["clip_a", "clip_b"],
        clip_lengths=torch.tensor([62, 95], dtype=torch.long),
    )
    command.num_future_steps = 2
    command.clip_weighting_strategy = "uniform_clip"
    command._training_iteration = 17
    command.distributed_loss_weight = 1.0
    command.adaptive_timesteps_sampler = AdaptiveTimestepsSampler(
        None,
        "cpu",
        env_fps=30,
        clip_lengths=command.motion.clip_lengths,
        valid_start_counts=torch.tensor([60, 93]),
    )
    command.use_adaptive_timesteps_sampler = True
    command._clip_success_counts = None
    command._clip_total_counts = None
    command._raw_clip_sampling_weights = torch.tensor([0.25, 0.75])
    command._clip_sampling_weights = command._raw_clip_sampling_weights.clone()
    command.multi_clip = True
    command._clean_noisy_clip_curriculum_enabled = False
    command._clean_clip_mask = None
    command._contact_prior_active = True
    command._contact_prior_available = True
    region_names = command.contact_prior_region_names()
    command._contact_prior_force_body_names_by_region = {
        name: [f"{name}_force_body"] for name in region_names
    }
    command._contact_prior_position_body_names_by_region = {
        name: [f"{name}_position_body"] for name in region_names
    }
    num_clips = len(command.motion.clip_ids)
    num_phases = 2
    num_regions = len(region_names)
    command._contact_prior_total_count = torch.zeros((num_clips, num_phases), dtype=torch.float32)
    command._contact_prior_contact_sum = torch.zeros(
        (num_clips, num_phases, num_regions), dtype=torch.float32
    )
    command._contact_prior_force_mean = torch.zeros(
        (num_clips, num_phases, num_regions), dtype=torch.float32
    )
    command._contact_prior_force_count = torch.zeros(
        (num_clips, num_phases, num_regions), dtype=torch.float32
    )
    command._contact_prior_position_mean = torch.zeros(
        (num_clips, num_phases, num_regions, 3), dtype=torch.float32
    )
    command._contact_prior_position_count = torch.zeros(
        (num_clips, num_phases, num_regions), dtype=torch.float32
    )
    return command


_CONTACT_PRIOR_TENSOR_NAMES = (
    "total_count",
    "contact_sum",
    "force_mean",
    "force_count",
    "position_mean",
    "position_count",
)


def _set_nonzero_contact_prior(command: MotionCommand) -> None:
    command._contact_prior_total_count[0, 1] = 5.0
    command._contact_prior_contact_sum[0, 1, 0] = 3.0
    command._contact_prior_contact_sum[0, 1, 1] = 2.0
    command._contact_prior_force_count.copy_(command._contact_prior_contact_sum)
    command._contact_prior_position_count.copy_(command._contact_prior_contact_sum)
    command._contact_prior_force_mean[0, 1, 0] = 8.5
    command._contact_prior_force_mean[0, 1, 1] = 3.25
    command._contact_prior_position_mean[0, 1, 0] = torch.tensor([0.2, -0.4, 0.6])
    command._contact_prior_position_mean[0, 1, 1] = torch.tensor([-0.1, 0.3, -0.5])


def _contact_prior_live_tensors(command: MotionCommand) -> dict[str, torch.Tensor]:
    return {
        name: getattr(command, f"_contact_prior_{name}").detach().clone()
        for name in _CONTACT_PRIOR_TENSOR_NAMES
    }


def test_motion_command_contact_prior_checkpoint_nonzero_round_trip_and_canonical_reset() -> None:
    source = _make_motion_command()
    _set_nonzero_contact_prior(source)

    state = source.get_checkpoint_state()
    assert state["version"] == 3
    assert state["contact_prior"]["active"] is True
    assert state["contact_prior"]["available"] is True
    assert state["contact_prior"]["schema"]["clip_ids"] == ["clip_a", "clip_b"]
    assert state["contact_prior"]["schema"]["phase_names"] == [
        "before_pickup_anchor",
        "after_pickup_anchor",
    ]
    assert state["contact_prior"]["schema"]["region_names"] == list(source.contact_prior_region_names())

    expected = {name: state["contact_prior"][name].clone() for name in _CONTACT_PRIOR_TENSOR_NAMES}
    # Model init_buffers() at the canonical checkpoint boundary.
    for name in _CONTACT_PRIOR_TENSOR_NAMES:
        getattr(source, f"_contact_prior_{name}").zero_()
    source.load_checkpoint_state(state)

    for name, expected_tensor in expected.items():
        assert torch.equal(getattr(source, f"_contact_prior_{name}"), expected_tensor)

    # CPU snapshots must not alias the restored live tensors either.
    source._contact_prior_total_count.zero_()
    assert state["contact_prior"]["total_count"][0, 1].item() == 5.0


def test_motion_command_contact_prior_checkpoint_rejects_missing_and_bad_values_atomically() -> None:
    source = _make_motion_command()
    _set_nonzero_contact_prior(source)
    state = source.get_checkpoint_state()

    restored = _make_motion_command()
    _set_nonzero_contact_prior(restored)
    restored._contact_prior_total_count.add_(2.0)
    before = _contact_prior_live_tensors(restored)

    missing = copy.deepcopy(state)
    missing["contact_prior"].pop("force_count")
    with pytest.raises(ValueError, match="contact_prior keys.*missing=.*force_count"):
        restored.load_checkpoint_state(missing)
    for name, expected in before.items():
        assert torch.equal(getattr(restored, f"_contact_prior_{name}"), expected)

    wrong_shape = copy.deepcopy(state)
    wrong_shape["contact_prior"]["contact_sum"] = wrong_shape["contact_prior"]["contact_sum"][..., :-1]
    with pytest.raises(ValueError, match=r"contact_prior\.contact_sum shape"):
        restored.load_checkpoint_state(wrong_shape)
    for name, expected in before.items():
        assert torch.equal(getattr(restored, f"_contact_prior_{name}"), expected)

    wrong_dtype = copy.deepcopy(state)
    wrong_dtype["contact_prior"]["force_mean"] = wrong_dtype["contact_prior"]["force_mean"].double()
    with pytest.raises(ValueError, match=r"contact_prior\.force_mean dtype"):
        restored.load_checkpoint_state(wrong_dtype)
    for name, expected in before.items():
        assert torch.equal(getattr(restored, f"_contact_prior_{name}"), expected)

    nonfinite = copy.deepcopy(state)
    nonfinite["contact_prior"]["position_mean"][0, 1, 0, 0] = float("nan")
    with pytest.raises(ValueError, match=r"contact_prior\.position_mean contains NaN"):
        restored.load_checkpoint_state(nonfinite)
    for name, expected in before.items():
        assert torch.equal(getattr(restored, f"_contact_prior_{name}"), expected)

    impossible_count = copy.deepcopy(state)
    impossible_count["contact_prior"]["contact_sum"][0, 1, 0] = 6.0
    impossible_count["contact_prior"]["force_count"][0, 1, 0] = 6.0
    impossible_count["contact_prior"]["position_count"][0, 1, 0] = 6.0
    with pytest.raises(ValueError, match="contact_sum cannot exceed total_count"):
        restored.load_checkpoint_state(impossible_count)
    for name, expected in before.items():
        assert torch.equal(getattr(restored, f"_contact_prior_{name}"), expected)

    fractional_count = copy.deepcopy(state)
    fractional_count["contact_prior"]["total_count"][0, 1] = 5.5
    with pytest.raises(ValueError, match="total_count must contain integer counts"):
        restored.load_checkpoint_state(fractional_count)
    for name, expected in before.items():
        assert torch.equal(getattr(restored, f"_contact_prior_{name}"), expected)


def test_motion_command_contact_prior_checkpoint_binds_runtime_and_axis_schema() -> None:
    state = _make_motion_command().get_checkpoint_state()

    unavailable = copy.deepcopy(state)
    unavailable["contact_prior"]["available"] = False
    with pytest.raises(ValueError, match="activation differs"):
        _make_motion_command().load_checkpoint_state(unavailable)

    wrong_phase = copy.deepcopy(state)
    wrong_phase["contact_prior"]["schema"]["phase_names"].reverse()
    with pytest.raises(ValueError, match="clip/phase/region schema differs"):
        _make_motion_command().load_checkpoint_state(wrong_phase)

    wrong_region_mapping = copy.deepcopy(state)
    wrong_region_mapping["contact_prior"]["schema"]["force_body_names_by_region"]["left_wrist"] = [
        "different_body"
    ]
    with pytest.raises(ValueError, match="clip/phase/region schema differs"):
        _make_motion_command().load_checkpoint_state(wrong_region_mapping)


@pytest.mark.parametrize("legacy_version", [1, 2])
def test_motion_command_legacy_checkpoint_requires_inactive_contact_prior(legacy_version: int) -> None:
    legacy = _make_motion_command().get_checkpoint_state()
    legacy["version"] = legacy_version
    legacy.pop("contact_prior")

    with pytest.raises(ValueError, match="predates online contact-prior state.*exact resume is impossible"):
        _make_motion_command().load_checkpoint_state(legacy)

    inactive = _make_motion_command()
    inactive._contact_prior_active = False
    inactive._contact_prior_available = False
    inactive.load_checkpoint_state(legacy)
    assert inactive._training_iteration == 17


def test_motion_command_checkpoint_round_trip_and_clip_identity_guard() -> None:
    source = _make_motion_command()
    source.adaptive_timesteps_sampler.bin_failed_count[0, 1] = 4.0
    source.adaptive_timesteps_sampler.bin_exposure_count[0, 1] = 5.0
    state = source.get_checkpoint_state()

    restored = _make_motion_command()
    restored.adaptive_timesteps_sampler.bin_failed_count.zero_()
    restored._raw_clip_sampling_weights.fill_(0.5)
    restored.load_checkpoint_state(state)

    assert restored._training_iteration == 17
    assert restored.adaptive_timesteps_sampler.bin_failed_count[0, 1].item() == 4.0
    assert torch.equal(restored._raw_clip_sampling_weights, torch.tensor([0.25, 0.75]))

    incompatible = _make_motion_command()
    incompatible.motion.clip_ids = ["clip_b", "clip_a"]
    with pytest.raises(ValueError, match="different clip shard/order"):
        incompatible.load_checkpoint_state(state)


def test_motion_command_checkpoint_rejects_partial_adaptive_state() -> None:
    source = _make_motion_command()
    state = source.get_checkpoint_state()

    missing_sampler = dict(state)
    missing_sampler.pop("adaptive_timesteps_sampler")
    with pytest.raises(ValueError, match="missing adaptive_timesteps_sampler"):
        _make_motion_command().load_checkpoint_state(missing_sampler)

    missing_clip_weights = dict(state)
    missing_clip_weights.pop("raw_clip_sampling_weights")
    with pytest.raises(ValueError, match="raw_clip_sampling_weights"):
        _make_motion_command().load_checkpoint_state(missing_clip_weights)


def test_motion_command_checkpoint_validation_is_atomic() -> None:
    source = _make_motion_command()
    source.adaptive_timesteps_sampler.bin_failed_count[0, 1] = 2.0
    source.adaptive_timesteps_sampler.bin_exposure_count[0, 1] = 3.0
    state = source.get_checkpoint_state()
    state = {
        key: (
            {
                nested_key: nested_value.clone() if isinstance(nested_value, torch.Tensor) else nested_value
                for nested_key, nested_value in value.items()
            }
            if key == "adaptive_timesteps_sampler"
            else value.clone() if isinstance(value, torch.Tensor) else value
        )
        for key, value in state.items()
    }
    state["raw_clip_sampling_weights"] = torch.tensor([float("nan"), 0.0])

    restored = _make_motion_command()
    sampler_before = restored.adaptive_timesteps_sampler.bin_failed_count.clone()
    weights_before = restored._raw_clip_sampling_weights.clone()
    iteration_before = restored._training_iteration

    with pytest.raises(ValueError, match="raw_clip_sampling_weights contains NaN"):
        restored.load_checkpoint_state(state)

    assert torch.equal(restored.adaptive_timesteps_sampler.bin_failed_count, sampler_before)
    assert torch.equal(restored._raw_clip_sampling_weights, weights_before)
    assert restored._training_iteration == iteration_before


def test_motion_command_cpu_checkpoint_snapshot_does_not_alias_live_curriculum() -> None:
    command = _make_motion_command()
    command._clip_success_counts = torch.tensor([3.0, 5.0])
    command._clip_total_counts = torch.tensor([7.0, 11.0])
    command.adaptive_timesteps_sampler.current_bin_failed_count[0, 0] = 13.0
    command.adaptive_timesteps_sampler.bin_exposure_count[0, 0] = 17.0

    snapshot = command.get_checkpoint_state()

    command._clip_success_counts.zero_()
    command._clip_total_counts.zero_()
    command._raw_clip_sampling_weights.zero_()
    command.adaptive_timesteps_sampler.current_bin_failed_count.zero_()
    command.adaptive_timesteps_sampler.bin_exposure_count.zero_()

    assert torch.equal(snapshot["clip_success_counts"], torch.tensor([3.0, 5.0]))
    assert torch.equal(snapshot["clip_total_counts"], torch.tensor([7.0, 11.0]))
    assert torch.equal(snapshot["raw_clip_sampling_weights"], torch.tensor([0.25, 0.75]))
    assert snapshot["adaptive_timesteps_sampler"]["current_bin_failed_count"][0, 0].item() == 13.0
    assert snapshot["adaptive_timesteps_sampler"]["bin_exposure_count"][0, 0].item() == 17.0


class _FakeMotionCommand:
    def __init__(self) -> None:
        self.curriculum_value = 7

    def get_checkpoint_state(self):
        return {"curriculum_value": self.curriculum_value}

    def init_buffers(self):
        self.curriculum_value = 0

    def load_checkpoint_state(self, state):
        self.curriculum_value = int(state["curriculum_value"])

    def validate_checkpoint_state(self, state):
        if not isinstance(state, dict) or "curriculum_value" not in state:
            raise ValueError("invalid fake motion state")


class _CheckpointPerceptionStub:
    def __init__(self, value: float, *, exact_resume_supported: bool = True) -> None:
        self.enabled = True
        self.value = torch.tensor(value)
        self.canonical_reset_count = 0
        self.exact_resume_supported = exact_resume_supported

    def persistent_checkpoint_state_required(self) -> bool:
        return True

    def get_persistent_checkpoint_state(self):
        return {"value": self.value.detach().to("cpu").clone()}

    def validate_persistent_checkpoint_state(self, state) -> None:
        if (
            not isinstance(state, dict)
            or set(state) != {"value"}
            or not isinstance(state["value"], torch.Tensor)
            or state["value"].numel() != 1
            or not bool(torch.isfinite(state["value"]).all().item())
        ):
            raise ValueError("invalid perception stub state")

    def load_persistent_checkpoint_state(self, state) -> None:
        self.validate_persistent_checkpoint_state(state)
        self.value.copy_(state["value"])

    def reset_canonical_rollout_state(self) -> None:
        self.canonical_reset_count += 1

    def validate_exact_resume_supported(self) -> None:
        if not self.exact_resume_supported:
            raise RuntimeError("stub backend is not exactly resumable")


def test_wbt_initial_reset_all_does_not_erase_restored_curriculum(monkeypatch) -> None:
    command = _FakeMotionCommand()
    env = object.__new__(WholeBodyTrackingManager)
    env.command_manager = SimpleNamespace(get_state=lambda name: command)
    tracker = SimpleNamespace(
        state_dict=lambda: {"average_episode_length": 0.0},
        validate_state_dict=lambda state: None,
        load_state_dict=lambda state: None,
    )
    env.curriculum_manager = SimpleNamespace(
        get_term=lambda name: tracker if name == "average_episode_tracker" else None,
        iter_terms=lambda: iter(()),
    )

    def fake_base_reset_all(self):
        # Model the real zero-action warm-up mutating adaptive statistics.
        command.curriculum_value = 3
        return "observations"

    monkeypatch.setattr(BaseTask, "reset_all", fake_base_reset_all)

    result = env.reset_all()

    assert result == "observations"
    assert command.curriculum_value == 7


def _make_wobject_curriculum(
    *,
    enabled: bool = True,
    initial_lambda: float = 0.4,
) -> tuple[WObjectDifficultyCurriculum, SimpleNamespace]:
    env = SimpleNamespace(
        device="cpu",
        num_envs=3,
        time_out_buf=torch.tensor([True, False, False]),
        termination_manager=SimpleNamespace(
            get_last_term_result=lambda name: torch.tensor([False, True, False]),
        ),
        is_evaluating=False,
        log_dict={"motion/error_object_ref_pos": torch.tensor([0.0, 0.0, 1.0])},
    )
    cfg = SimpleNamespace(
        params={
            "enabled": enabled,
            "initial_lambda": initial_lambda,
            "lambda_step_up": 0.1,
            "lambda_step_down": 0.2,
            "early_termination_threshold": 0.3,
            "similarity_metric_key": "motion/error_object_ref_pos",
            "similarity_sigma": 0.5,
            "similarity_threshold": 0.6,
            "assist_beta_max": 1.0,
        }
    )
    term = WObjectDifficultyCurriculum(cfg, env)
    term.setup()
    return term, env


def _curriculum_manager_with(term: WObjectDifficultyCurriculum):
    tracker = AverageEpisodeLengthTracker(SimpleNamespace(params={}), term.env)
    tracker.setup()
    terms = {
        "average_episode_tracker": tracker,
        "w_object_difficulty_curriculum": term,
    }
    return SimpleNamespace(
        iter_terms=lambda: iter(terms.items()),
        get_term=lambda name: terms.get(name),
    )


class _PenaltyRewardManager:
    def __init__(self) -> None:
        self.active_terms = ["action_rate"]
        self._configs = {
            "action_rate": RewardTermCfg(
                func="unused",
                weight=-2.0,
                tags=["penalty_curriculum"],
            )
        }

    def get_term_cfg(self, name: str) -> RewardTermCfg:
        return self._configs[name]

    def set_term_cfg(self, name: str, cfg: RewardTermCfg) -> None:
        self._configs[name] = cfg


def _penalty_env() -> SimpleNamespace:
    return SimpleNamespace(
        reward_manager=_PenaltyRewardManager(),
        average_episode_length=0.0,
        log_dict={},
    )


def test_penalty_curriculum_empty_reset_does_not_change_objective() -> None:
    env = _penalty_env()
    term = PenaltyCurriculum(
        SimpleNamespace(
            params={
                "enabled": True,
                "initial_scale": 0.5,
                "min_scale": 0.0,
                "max_scale": 1.0,
                "level_down_threshold": 10.0,
                "level_up_threshold": 20.0,
                "degree": 0.1,
            }
        ),
        env,
    )
    term.setup()

    term.reset(torch.empty(0, dtype=torch.long))

    assert term.current_scale == pytest.approx(0.5)
    assert env.reward_penalty_scale == pytest.approx(0.5)
    assert env.reward_manager.get_term_cfg("action_rate").weight == pytest.approx(-1.0)

    term.reset(torch.tensor([0], dtype=torch.long))

    assert term.current_scale == pytest.approx(0.45)
    assert env.reward_penalty_scale == pytest.approx(0.45)
    assert env.reward_manager.get_term_cfg("action_rate").weight == pytest.approx(-0.9)


def test_legacy_penalty_curriculum_empty_reset_does_not_change_objective() -> None:
    env = _penalty_env()
    configure_reward_penalty(
        env,
        initial_scale=0.5,
        min_scale=0.0,
        max_scale=1.0,
        level_down_threshold=10.0,
        level_up_threshold=20.0,
        degree=0.1,
    )

    update_reward_penalty(env, torch.empty(0, dtype=torch.long))

    assert env._curriculum_penalty_cfg["current_scale"] == pytest.approx(0.5)
    assert env.reward_penalty_scale == pytest.approx(0.5)
    assert env.reward_manager.get_term_cfg("action_rate").weight == pytest.approx(-1.0)

    update_reward_penalty(env, torch.tensor([0], dtype=torch.long))

    assert env._curriculum_penalty_cfg["current_scale"] == pytest.approx(0.45)
    assert env.reward_penalty_scale == pytest.approx(0.45)
    assert env.reward_manager.get_term_cfg("action_rate").weight == pytest.approx(-0.9)


def _base_algo_with_penalty_curriculum(*, world_size: int, degree: float) -> BaseAlgo:
    penalty_term = SimpleNamespace(enabled=True, degree=degree)
    algo = object.__new__(BaseAlgo)
    algo.gpu_world_size = world_size
    algo.env = SimpleNamespace(
        use_reward_penalty_curriculum=True,
        curriculum_manager=SimpleNamespace(
            get_term=lambda name: penalty_term if name == "penalty_curriculum" else None,
        ),
    )
    return algo


def test_distributed_adaptive_penalty_curriculum_fails_closed() -> None:
    algo = _base_algo_with_penalty_curriculum(world_size=2, degree=0.00025)

    with pytest.raises(RuntimeError, match="ranks optimize different objectives"):
        algo._validate_distributed_penalty_curriculum_contract()


def test_static_or_single_rank_penalty_curriculum_is_allowed() -> None:
    _base_algo_with_penalty_curriculum(
        world_size=2,
        degree=0.0,
    )._validate_distributed_penalty_curriculum_contract()
    _base_algo_with_penalty_curriculum(
        world_size=1,
        degree=0.00025,
    )._validate_distributed_penalty_curriculum_contract()


def test_locomotion_curriculum_sync_updates_authoritative_penalty_and_reward_weights(
    monkeypatch,
) -> None:
    env = object.__new__(LeggedRobotLocomotionManager)
    env.device = "cpu"
    env.num_envs = 2
    env.reward_manager = _PenaltyRewardManager()
    env.log_dict = {}
    tracker = AverageEpisodeLengthTracker(SimpleNamespace(params={}), env)
    tracker.setup()
    penalty = PenaltyCurriculum(
        SimpleNamespace(
            params={
                "enabled": True,
                "initial_scale": 0.5,
                "min_scale": 0.0,
                "max_scale": 1.0,
                "level_down_threshold": 10.0,
                "level_up_threshold": 20.0,
                "degree": 0.0,
            }
        ),
        env,
    )
    terms = {
        "average_episode_tracker": tracker,
        "penalty_curriculum": penalty,
    }
    env.curriculum_manager = SimpleNamespace(get_term=lambda name: terms.get(name))
    penalty.setup()

    synchronized_values = iter((25.0, 0.25))

    def fake_broadcast(tensor, *, src, group) -> None:
        assert src == 0
        assert group == "test-group"
        tensor.fill_(next(synchronized_values))

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "broadcast", fake_broadcast)

    env.synchronize_curriculum_state(
        device="cpu",
        world_size=2,
        process_group="test-group",
    )

    assert tracker.get_average().item() == pytest.approx(25.0)
    assert penalty.current_scale == pytest.approx(0.25)
    assert env.reward_penalty_scale == pytest.approx(0.25)
    assert env.reward_manager.get_term_cfg("action_rate").weight == pytest.approx(-0.5)


def test_locomotion_curriculum_sync_requires_initialized_process_group(monkeypatch) -> None:
    env = object.__new__(LeggedRobotLocomotionManager)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="requires an initialized process group"):
        env.synchronize_curriculum_state(device="cpu", world_size=2)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"enabled": "false"}, "explicit boolean"),
        ({"tag": ""}, "non-empty string"),
        ({"initial_scale": float("nan")}, "must be finite"),
        ({"min_scale": 0.6}, "0 <= min_scale"),
        ({"level_down_threshold": 30.0}, "thresholds must satisfy"),
        ({"degree": 1.0}, "0 <= degree < 1"),
    ],
)
def test_penalty_curriculum_rejects_invalid_parameters(override, message) -> None:
    params = {
        "enabled": True,
        "tag": "penalty_curriculum",
        "initial_scale": 0.5,
        "min_scale": 0.0,
        "max_scale": 1.0,
        "level_down_threshold": 10.0,
        "level_up_threshold": 20.0,
        "degree": 0.1,
    }
    params.update(override)

    with pytest.raises(ValueError, match=message):
        PenaltyCurriculum(SimpleNamespace(params=params), _penalty_env())


@pytest.mark.parametrize("horizon", [True, 0, -1.0, float("nan"), float("inf")])
def test_average_episode_tracker_rejects_invalid_horizon(horizon) -> None:
    env = SimpleNamespace(device="cpu", num_envs=2, BASE_NUM_ENVS=2)

    with pytest.raises(ValueError, match="finite positive real"):
        AverageEpisodeLengthTracker(
            SimpleNamespace(params={"num_compute_average_epl": horizon}),
            env,
        )


@pytest.mark.parametrize(("term_params", "expected"), [({}, 123), ({"num_compute_average_epl": 77}, 77)])
def test_curriculum_manager_global_params_reach_terms_with_term_override(
    monkeypatch,
    term_params,
    expected,
) -> None:
    monkeypatch.setattr(
        "holosoma.managers.curriculum.manager.resolve_callable",
        lambda _path, *, context: AverageEpisodeLengthTracker,
    )
    cfg = CurriculumManagerCfg(
        params={"num_compute_average_epl": 123},
        setup_terms={
            "average_episode_tracker": CurriculumTermCfg(
                func="test:tracker",
                params=term_params,
            )
        },
    )
    env = SimpleNamespace(device="cpu", num_envs=2, BASE_NUM_ENVS=2)

    manager = CurriculumManager(cfg, env, "cpu")

    assert manager.get_term("average_episode_tracker").num_compute_average_epl == expected


def test_enabled_penalty_curriculum_requires_matching_reward_term_atomically() -> None:
    env = _penalty_env()
    initial_weight = env.reward_manager.get_term_cfg("action_rate").weight
    term = PenaltyCurriculum(
        SimpleNamespace(
            params={
                "enabled": True,
                "tag": "missing-tag",
                "initial_scale": 0.5,
                "min_scale": 0.0,
                "max_scale": 1.0,
                "level_down_threshold": 10.0,
                "level_up_threshold": 20.0,
                "degree": 0.1,
            }
        ),
        env,
    )

    with pytest.raises(ValueError, match="matches no active reward terms"):
        term.setup()

    assert env.reward_manager.get_term_cfg("action_rate").weight == initial_weight
    assert not hasattr(env, "use_reward_penalty_curriculum")


def test_legacy_penalty_configuration_reuses_strict_parameter_contract() -> None:
    env = _penalty_env()
    initial_weight = env.reward_manager.get_term_cfg("action_rate").weight

    with pytest.raises(ValueError, match="explicit boolean"):
        configure_reward_penalty(env, enabled="false")

    assert env.reward_manager.get_term_cfg("action_rate").weight == initial_weight
    assert not hasattr(env, "use_reward_penalty_curriculum")


def test_penalty_state_restore_realigns_reward_mirror_and_log() -> None:
    env = _penalty_env()
    term = PenaltyCurriculum(
        SimpleNamespace(
            params={
                "enabled": True,
                "initial_scale": 0.5,
                "min_scale": 0.0,
                "max_scale": 1.0,
                "level_down_threshold": 10.0,
                "level_up_threshold": 20.0,
                "degree": 0.1,
            }
        ),
        env,
    )
    term.setup()
    term._apply_scale(0.25)
    checkpoint_state = term.state_dict()
    term._apply_scale(0.75)

    term.load_state_dict(checkpoint_state)

    assert term.current_scale == pytest.approx(0.25)
    assert env.reward_penalty_scale == pytest.approx(0.25)
    assert env.log_dict["penalty_scale"].item() == pytest.approx(0.25)
    assert env.reward_manager.get_term_cfg("action_rate").weight == pytest.approx(-0.5)


def test_locomotion_canonical_restore_consumes_checkpoint_load_suppression() -> None:
    tracker_env = SimpleNamespace(device="cpu", num_envs=2, BASE_NUM_ENVS=2)
    tracker = AverageEpisodeLengthTracker(SimpleNamespace(params={}), tracker_env)
    tracker.setup()
    tracker.average_episode_length.fill_(42.0)

    env = object.__new__(LeggedRobotLocomotionManager)
    env.curriculum_manager = SimpleNamespace(
        get_term=lambda name: tracker if name == "average_episode_tracker" else None,
    )
    checkpoint_state = env.get_checkpoint_state()

    tracker.average_episode_length.zero_()
    env.load_checkpoint_state(checkpoint_state)
    assert tracker._suppress_next_update is True

    # Model reset_all_at_checkpoint_boundary(): snapshot the just-loaded
    # adaptive state, consume the one-shot guard during the forced reset, then
    # restore the snapshot without resurrecting that guard.
    boundary_snapshot = env.get_checkpoint_state()
    tracker.update(torch.tensor([0]), torch.tensor([100.0]))
    assert tracker.get_average().item() == pytest.approx(42.0)
    env._restore_checkpoint_state_after_canonical_reset(boundary_snapshot)

    assert tracker._suppress_next_update is False
    tracker.update(torch.tensor([0]), torch.tensor([100.0]))
    assert tracker.get_average().item() > 42.0


def _gloo_wobject_sync_worker(rank: int, world_size: int, init_file: str, output_dir: str) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        term, _ = _make_wobject_curriculum()
        term._pending_early_termination_count = 1 if rank == 0 else 0
        term._pending_motion_end_count = 0 if rank == 0 else 1
        term._pending_episode_count = 2
        term._pending_similarity_error_sum = 0.2 if rank == 0 else 0.0
        term._pending_similarity_count = 2
        term.synchronize_state(
            device="cpu",
            world_size=world_size,
            process_group=torch.distributed.group.WORLD,
        )
        torch.save(term.get_checkpoint_state(), f"{output_dir}/wobject_rank_{rank}.pt")
    finally:
        torch.distributed.destroy_process_group()


def test_wobject_curriculum_applies_sufficient_statistics_only_at_rollout_boundary() -> None:
    term, _ = _make_wobject_curriculum()

    term.reset(torch.tensor([0, 1]))

    # Timeout + clean motion end means no early termination.  The curriculum
    # scalar must remain fixed throughout the rollout rather than changing at
    # a rank-local reset boundary.
    assert term.lambda_value == pytest.approx(0.4)
    assert term._pending_episode_count == 2
    term.synchronize_state(device="cpu", world_size=1)

    assert term.lambda_value == pytest.approx(0.5)
    assert term._last_early_termination_rate == pytest.approx(0.0)
    assert term._last_motion_end_rate == pytest.approx(0.5)
    assert term._last_similarity == pytest.approx(1.0)
    assert term._pending_episode_count == 0


def test_wobject_curriculum_ddp_reduces_global_sufficient_statistics(monkeypatch) -> None:
    term, _ = _make_wobject_curriculum()
    term._pending_early_termination_count = 1
    term._pending_motion_end_count = 0
    term._pending_episode_count = 2
    term._pending_similarity_error_sum = 0.2
    term._pending_similarity_count = 2
    remote_statistics = torch.tensor([0.0, 1.0, 2.0, 0.0, 2.0], dtype=torch.float64)

    def fake_all_reduce(tensor, op=None, group=None):
        if op == torch.distributed.ReduceOp.SUM:
            tensor.add_(remote_statistics)
        elif op not in (torch.distributed.ReduceOp.MIN, torch.distributed.ReduceOp.MAX):
            raise AssertionError(f"unexpected reduction op {op}")

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    term.synchronize_state(device="cpu", world_size=2)

    # Global early rate is 1/4 and mean error is 0.2/4, so the shared
    # curriculum advances even though the local rank alone would fail the
    # early-termination threshold.
    assert term.lambda_value == pytest.approx(0.5)
    assert term._last_early_termination_rate == pytest.approx(0.25)
    assert term._last_motion_end_rate == pytest.approx(0.25)
    assert term._last_similarity == pytest.approx(torch.exp(torch.tensor(-0.1)).item())
    assert term._pending_episode_count == 0
    assert term._pending_similarity_error_sum == 0.0


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed is unavailable")
def test_wobject_curriculum_real_two_rank_gloo_produces_identical_state(tmp_path) -> None:
    init_file = tmp_path / "wobject_gloo_init"
    mp.start_processes(
        _gloo_wobject_sync_worker,
        args=(2, str(init_file), str(tmp_path)),
        nprocs=2,
        join=True,
        start_method="fork",
    )

    rank0 = torch.load(tmp_path / "wobject_rank_0.pt", weights_only=True)
    rank1 = torch.load(tmp_path / "wobject_rank_1.pt", weights_only=True)
    assert rank0 == rank1
    assert rank0["lambda_value"] == pytest.approx(0.5)
    assert rank0["last_early_termination_rate"] == pytest.approx(0.25)
    assert rank0["last_motion_end_rate"] == pytest.approx(0.25)
    assert rank0["pending_episode_count"] == 0


def test_wobject_curriculum_ddp_rejects_divergent_lambda(monkeypatch) -> None:
    term, _ = _make_wobject_curriculum()
    term._pending_episode_count = 1
    term._pending_similarity_count = 1
    calls = 0

    def fake_all_reduce(tensor, op=None, group=None):
        nonlocal calls
        calls += 1
        if tensor.dtype == torch.int32:
            return
        if op == torch.distributed.ReduceOp.MIN:
            tensor.fill_(0.2)
        elif op == torch.distributed.ReduceOp.MAX:
            tensor.fill_(0.4)

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    with pytest.raises(RuntimeError, match="lambda differs across ranks"):
        term.synchronize_state(device="cpu", world_size=2)

    # One validity handshake precedes the lambda min/max reductions so a
    # corrupt local rank cannot throw early and strand its peers.
    assert calls == 3
    assert term.lambda_value == pytest.approx(0.4)
    assert term._pending_episode_count == 1


def test_wobject_checkpoint_round_trip_preserves_pending_boundary_state_atomically() -> None:
    source, _ = _make_wobject_curriculum(initial_lambda=0.3)
    source._last_early_termination_rate = 0.25
    source._last_motion_end_rate = 0.5
    source._last_similarity = 0.75
    source._pending_early_termination_count = 1
    source._pending_motion_end_count = 2
    source._pending_episode_count = 4
    source._pending_similarity_error_sum = 0.6
    source._pending_similarity_count = 4
    state = source.get_checkpoint_state()

    restored, _ = _make_wobject_curriculum(initial_lambda=0.3)
    restored.load_checkpoint_state(state)
    assert restored.get_checkpoint_state() == state

    corrupt = dict(state)
    corrupt["lambda_value"] = float("nan")
    before = restored.get_checkpoint_state()
    with pytest.raises(ValueError, match="lambda_value"):
        restored.load_checkpoint_state(corrupt)
    assert restored.get_checkpoint_state() == before


def test_wbt_initial_reset_all_does_not_inject_wobject_statistics(monkeypatch) -> None:
    term, _ = _make_wobject_curriculum()
    term._pending_early_termination_count = 1
    term._pending_episode_count = 3
    term._pending_similarity_error_sum = 0.3
    term._pending_similarity_count = 3
    expected = term.get_checkpoint_state()
    command = _FakeMotionCommand()
    env = object.__new__(WholeBodyTrackingManager)
    env.command_manager = SimpleNamespace(get_state=lambda name: command)
    env.curriculum_manager = _curriculum_manager_with(term)

    def fake_base_reset_all(self):
        term.lambda_value = 0.9
        term._pending_early_termination_count = 8
        term._pending_episode_count = 9
        term._pending_similarity_error_sum = 4.0
        term._pending_similarity_count = 9
        return "observations"

    monkeypatch.setattr(BaseTask, "reset_all", fake_base_reset_all)

    assert env.reset_all() == "observations"
    assert term.get_checkpoint_state() == expected


def test_wbt_checkpoint_v4_round_trip_and_legacy_wobject_guard() -> None:
    term, _ = _make_wobject_curriculum()
    term._pending_episode_count = 2
    term._pending_similarity_count = 2
    command = _FakeMotionCommand()
    env = object.__new__(WholeBodyTrackingManager)
    env.command_manager = SimpleNamespace(get_state=lambda name: command)
    env.curriculum_manager = _curriculum_manager_with(term)

    state = env.get_checkpoint_state()
    assert state["version"] == 4
    assert state["curriculum_terms"]["average_episode_tracker"]["version"] == 1
    assert state["curriculum_terms"]["w_object_difficulty_curriculum"]["pending_episode_count"] == 2

    term.lambda_value = 0.9
    term._pending_episode_count = 0
    term._pending_similarity_count = 0
    env.load_checkpoint_state(state)
    assert term.lambda_value == pytest.approx(0.4)
    assert term._pending_episode_count == 2

    legacy = {"version": 1, "motion_command": command.get_checkpoint_state()}
    with pytest.raises(ValueError, match="exact resume is impossible"):
        env.validate_checkpoint_state(legacy)

    disabled_term, _ = _make_wobject_curriculum(enabled=False)
    env.curriculum_manager = _curriculum_manager_with(disabled_term)
    env.validate_checkpoint_state(legacy)


def test_wbt_checkpoint_restores_unique_perception_topology_but_post_boundary_does_not_rewind() -> None:
    term, _ = _make_wobject_curriculum()
    command = _FakeMotionCommand()
    actor_perception = _CheckpointPerceptionStub(1.25)
    teacher_perception = _CheckpointPerceptionStub(2.5)
    env = object.__new__(WholeBodyTrackingManager)
    env.command_manager = SimpleNamespace(get_state=lambda name: command)
    env.curriculum_manager = _curriculum_manager_with(term)
    env.perception_manager = actor_perception
    env.teacher_perception_manager = teacher_perception
    env.critic_perception_manager = actor_perception

    state = env.get_checkpoint_state()

    assert state["perception_managers"]["role_owners"] == {
        "actor": "actor",
        "teacher": "teacher",
        "critic": "actor",
    }
    assert set(state["perception_managers"]["states"]) == {"actor", "teacher"}

    actor_perception.value.fill_(9.0)
    teacher_perception.value.fill_(8.0)
    env.load_checkpoint_state(state)
    assert actor_perception.value.item() == pytest.approx(1.25)
    assert teacher_perception.value.item() == pytest.approx(2.5)

    actor_perception.value.fill_(7.0)
    teacher_perception.value.fill_(6.0)
    env._restore_checkpoint_state_after_canonical_reset(state)
    assert actor_perception.value.item() == pytest.approx(7.0)
    assert teacher_perception.value.item() == pytest.approx(6.0)

    legacy = dict(state)
    legacy["version"] = 3
    legacy.pop("perception_managers")
    with pytest.raises(RuntimeError, match="exact resume is impossible"):
        env.validate_checkpoint_state(legacy)


def test_base_task_canonicalizes_each_shared_perception_manager_once() -> None:
    actor_perception = _CheckpointPerceptionStub(1.0)
    teacher_perception = _CheckpointPerceptionStub(2.0)
    env = object.__new__(BaseTask)
    env.perception_manager = actor_perception
    env.teacher_perception_manager = teacher_perception
    env.critic_perception_manager = actor_perception
    env.get_checkpoint_state = lambda: {}
    env.reset_all = lambda: "observations"
    env._restore_checkpoint_state_after_canonical_reset = lambda state: None

    assert env.reset_all_at_checkpoint_boundary() == "observations"
    assert actor_perception.canonical_reset_count == 1
    assert teacher_perception.canonical_reset_count == 1


def test_base_task_rejects_unique_rendered_managers_on_one_camera_target() -> None:
    def rendered_manager():
        return SimpleNamespace(
            enabled=True,
            _simulator_backend="isaacsim",
            _rendered_camera_env_id=0,
            _uses_rendered_camera=lambda: True,
        )

    env = object.__new__(BaseTask)
    env.perception_manager = rendered_manager()
    env.teacher_perception_manager = rendered_manager()
    env.critic_perception_manager = None

    with pytest.raises(RuntimeError, match="overwrite each other"):
        env._validate_rendered_perception_topology()

    env.teacher_perception_manager = env.perception_manager
    env._validate_rendered_perception_topology()


def test_base_task_refuses_to_publish_nonresumable_perception_state() -> None:
    env = object.__new__(BaseTask)
    env.perception_manager = _CheckpointPerceptionStub(
        1.0,
        exact_resume_supported=False,
    )
    env.teacher_perception_manager = None
    env.critic_perception_manager = env.perception_manager

    with pytest.raises(RuntimeError, match="not exactly resumable"):
        env._get_perception_checkpoint_state()


def test_locomotion_checkpoint_restores_perception_only_on_full_load() -> None:
    tracker_env = SimpleNamespace(device="cpu", num_envs=2, BASE_NUM_ENVS=2)
    tracker = AverageEpisodeLengthTracker(SimpleNamespace(params={}), tracker_env)
    tracker.setup()
    perception = _CheckpointPerceptionStub(3.5)
    env = object.__new__(LeggedRobotLocomotionManager)
    env.curriculum_manager = SimpleNamespace(
        get_term=lambda name: tracker if name == "average_episode_tracker" else None,
    )
    env.perception_manager = perception
    env.teacher_perception_manager = None
    env.critic_perception_manager = perception

    state = env.get_checkpoint_state()
    assert state["version"] == 3
    perception.value.fill_(8.0)
    env.load_checkpoint_state(state)
    assert perception.value.item() == pytest.approx(3.5)

    perception.value.fill_(6.0)
    env._restore_checkpoint_state_after_canonical_reset(state)
    assert perception.value.item() == pytest.approx(6.0)

    legacy = dict(state)
    legacy["version"] = 2
    legacy.pop("perception_managers")
    with pytest.raises(RuntimeError, match="exact resume is impossible"):
        env.validate_checkpoint_state(legacy)


def test_enabled_wobject_curriculum_enables_wbt_sync_without_adaptive_sampler() -> None:
    term, _ = _make_wobject_curriculum()
    command = SimpleNamespace(
        adaptive_timesteps_sampler=None,
        _clip_success_counts=None,
        _clip_total_counts=None,
    )
    env = object.__new__(WholeBodyTrackingManager)
    env.command_manager = SimpleNamespace(get_state=lambda name: command)
    env.curriculum_manager = _curriculum_manager_with(term)

    assert env.curriculum_state_sync_enabled is True
    assert env.curriculum_state_checkpoint_required is True

    disabled_term, _ = _make_wobject_curriculum(enabled=False)
    env.curriculum_manager = _curriculum_manager_with(disabled_term)
    assert env.curriculum_state_sync_enabled is False
    assert env.curriculum_state_checkpoint_required is False


def _timeout_preview_env(*, time_step: int, clip_length: int) -> tuple[WholeBodyTrackingManager, SimpleNamespace]:
    command = SimpleNamespace(
        time_steps=torch.tensor([time_step, 3], dtype=torch.long),
        current_clip_lengths=torch.tensor([clip_length, 20], dtype=torch.long),
        _disable_clip_end_reset=False,
        _runtime_default_pose_prepend_active=torch.tensor([False, False]),
        _current_freeze_at_timestep_zero_prob=lambda: 0.0,
        future_target_poses=torch.tensor([[time_step * 10.0], [30.0]]),
    )

    def update_future_targets():
        command.future_target_poses[:, 0] = command.time_steps.to(dtype=torch.float32) * 10.0

    command._update_future_target_poses = update_future_targets
    observation_manager = SimpleNamespace(
        cfg=SimpleNamespace(groups={}, clip_observations=1000.0),
        active_group_names=None,
        compute=lambda **_: {
            "critic_obs": torch.stack(
                (command.time_steps.to(dtype=torch.float32), command.future_target_poses[:, 0]),
                dim=-1,
            )
        },
    )
    env = object.__new__(WholeBodyTrackingManager)
    env.time_out_buf = torch.tensor([True, False])
    env.command_manager = SimpleNamespace(get_state=lambda name: command)
    env.observation_manager = observation_manager
    env.log_dict = {}
    return env, command


def test_wbt_timeout_final_observation_previews_and_restores_next_command() -> None:
    env, command = _timeout_preview_env(time_step=4, clip_length=10)
    original_time_steps = command.time_steps.clone()
    original_future_targets = command.future_target_poses.clone()

    final_obs = env._compute_final_observations()

    assert torch.equal(final_obs["critic_obs"][0], torch.tensor([5.0, 50.0]))
    assert torch.equal(final_obs["critic_obs"][1], torch.tensor([3.0, 30.0]))
    assert torch.equal(command.time_steps, original_time_steps)
    assert torch.equal(command.future_target_poses, original_future_targets)
    assert torch.equal(env.time_out_buf, torch.tensor([True, False]))


def test_wbt_timeout_preview_updates_real_phase_and_drop_button_terms() -> None:
    command = object.__new__(MotionCommand)
    command.motion_cfg = SimpleNamespace(hybrid_stage2_enabled=False)
    command.device = "cpu"
    command.num_envs = 1
    command.time_steps = torch.tensor([4], dtype=torch.long)
    command.clip_ids = torch.tensor([0], dtype=torch.long)
    command.motion = SimpleNamespace(
        has_object=True,
        clip_lengths=torch.tensor([10], dtype=torch.long),
    )
    command.manual_control_enabled = False
    command.manual_drop_button_override_enabled = False
    command.manual_drop_button = None
    command._disable_clip_end_reset = False
    command._runtime_default_pose_prepend_active = torch.tensor([False])
    command._current_freeze_at_timestep_zero_prob = lambda: 0.0
    command.future_target_poses = None
    command._update_future_target_poses = lambda: None
    command._get_contact_aware_button_window_by_clip = lambda: torch.tensor([[2, 5]])

    env = object.__new__(WholeBodyTrackingManager)
    env.device = "cpu"
    env.num_envs = 1
    env.time_out_buf = torch.tensor([True])
    env.command_manager = SimpleNamespace(get_state=lambda name: command)
    env.log_dict = {}
    env.observation_manager = SimpleNamespace(
        cfg=SimpleNamespace(groups={}, clip_observations=1000.0),
        active_group_names=None,
    )
    env.observation_manager.compute = lambda **_: {"critic_obs": torch.cat((clip_phase(env), drop_button(env)), dim=-1)}

    final_obs = env._compute_final_observations()

    assert final_obs["critic_obs"][0, 0].item() == pytest.approx(5.0 / 9.0)
    assert final_obs["critic_obs"][0, 1].item() == 1.0
    assert command.time_steps.item() == 4
    assert clip_phase(env)[0, 0].item() == pytest.approx(4.0 / 9.0)
    assert drop_button(env)[0, 0].item() == 0.0


def test_wbt_timeout_clip_rollover_disables_invalid_bootstrap() -> None:
    env, command = _timeout_preview_env(time_step=9, clip_length=10)

    final_obs = env._compute_final_observations()

    assert torch.equal(final_obs["critic_obs"][0], torch.tensor([9.0, 90.0]))
    assert torch.equal(command.time_steps, torch.tensor([9, 3]))
    assert torch.equal(env.time_out_buf, torch.tensor([False, False]))
    assert env.log_dict["termination/timeout_bootstrap_rejected_frac"].item() == 0.5


def test_base_algo_detects_explicit_wbt_curriculum_sync_capability() -> None:
    algo = object.__new__(BaseAlgo)
    algo.env = SimpleNamespace(
        use_reward_penalty_curriculum=False,
        use_domain_rand_scale_curriculum=False,
        curriculum_state_sync_enabled=True,
    )

    assert algo.has_curricula_enabled() is True


def test_wbt_curriculum_sync_includes_failure_and_exposure_ema(monkeypatch) -> None:
    sampler = SimpleNamespace(
        current_bin_failed_count=torch.tensor([[1.0]]),
        bin_failed_count=torch.tensor([[2.0]]),
        current_bin_exposure_count=torch.tensor([[3.0]]),
        bin_exposure_count=torch.tensor([[4.0]]),
    )
    command = SimpleNamespace(
        _rank_local_shard_metadata=None,
        adaptive_timesteps_sampler=sampler,
        _clip_success_counts=None,
        _clip_total_counts=None,
        clip_weighting_strategy="uniform_clip",
    )
    env = object.__new__(WholeBodyTrackingManager)
    env.command_manager = SimpleNamespace(get_state=lambda name: command)
    reduced_values: list[float] = []

    def fake_all_reduce(tensor, op=None, group=None):
        reduced_values.append(float(tensor.item()))
        tensor.mul_(2.0)

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    env.synchronize_curriculum_state(device="cpu", world_size=2)

    assert reduced_values == [1.0, 2.0, 3.0, 4.0]
    assert sampler.current_bin_exposure_count.item() == 3.0
    assert sampler.bin_exposure_count.item() == 4.0


def _rank_local_sampler() -> SimpleNamespace:
    return SimpleNamespace(
        num_bins_per_clip=torch.tensor([2, 1], dtype=torch.long),
        current_bin_failed_count=torch.tensor([[2.0, 4.0, 97.0], [10.0, 98.0, 99.0]]),
        bin_failed_count=torch.tensor([[6.0, 8.0, 87.0], [20.0, 88.0, 89.0]]),
        current_bin_exposure_count=torch.tensor([[12.0, 14.0, 77.0], [30.0, 78.0, 79.0]]),
        bin_exposure_count=torch.tensor([[16.0, 18.0, 67.0], [40.0, 68.0, 69.0]]),
    )


def _rank_local_command(sampler: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        motion=SimpleNamespace(clip_ids=["duplicated", "unique_local"]),
        adaptive_timesteps_sampler=sampler,
        _rank_local_shard_metadata={
            "rank": 0,
            "world_size": 2,
            "clip_cover_counts": {"duplicated": 2, "unique_local": 1},
        },
        _clip_success_counts=None,
        _clip_total_counts=None,
    )


def test_rank_local_adaptive_sampler_enables_collective_on_every_rank() -> None:
    command = _rank_local_command(_rank_local_sampler())
    env = object.__new__(WholeBodyTrackingManager)
    env.command_manager = SimpleNamespace(get_state=lambda name: command)

    # Even a rank whose local clips are all unique must enter the collective;
    # otherwise ranks that own duplicated clips deadlock in all_gather_object.
    assert env.curriculum_state_sync_enabled is True
    command._rank_local_shard_metadata["clip_cover_counts"] = {
        "duplicated": 1,
        "unique_local": 1,
    }
    assert env.curriculum_state_sync_enabled is True


def test_ppo_routes_rank_local_as_object_sync_through_gloo() -> None:
    group = object()
    sync = Mock()
    command = SimpleNamespace(_rank_local_shard_metadata={"rank": 0})
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_state=lambda name: command),
        synchronize_curriculum_state=sync,
    )
    ppo = object.__new__(PPO)
    ppo.gpu_world_size = 8
    ppo._curriculum_state_sync_enabled = Mock(return_value=True)
    ppo._unwrap_env = Mock(return_value=env)
    ppo._setup_gloo_barrier_group = Mock(return_value=group)

    ppo._synchronize_curriculum_metrics()

    ppo._setup_gloo_barrier_group.assert_called_once_with()
    sync.assert_called_once_with(device="cpu", world_size=8, process_group=group)


def test_rank_local_sync_averages_only_duplicate_valid_bins(monkeypatch) -> None:
    sampler = _rank_local_sampler()
    command = _rank_local_command(sampler)
    env = object.__new__(WholeBodyTrackingManager)
    env.command_manager = SimpleNamespace(get_state=lambda name: command)
    group = object()

    local_before = {
        name: getattr(sampler, name).clone()
        for name in (
            "current_bin_failed_count",
            "bin_failed_count",
            "current_bin_exposure_count",
            "bin_exposure_count",
        )
    }
    remote_rows = {
        "current_bin_failed_count": torch.tensor([10.0, 12.0]),
        "bin_failed_count": torch.tensor([14.0, 16.0]),
        "current_bin_exposure_count": torch.tensor([20.0, 22.0]),
        "bin_exposure_count": torch.tensor([24.0, 26.0]),
    }

    def fake_all_gather_object(outputs, local_payload, group=None):
        assert group is globals_group
        outputs[0] = local_payload
        outputs[1] = {
            "rank": 1,
            "clip_cover_counts": {"duplicated": 2},
            "clips": {
                "duplicated": {
                    "cover_count": 2,
                    "valid_bins": 2,
                    "states": remote_rows,
                }
            },
        }

    # Give the closure a stable identity without relying on equality semantics.
    globals_group = group
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group=None: "gloo")
    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)

    env.synchronize_curriculum_state(device="cpu", world_size=2, process_group=group)

    for name, remote_row in remote_rows.items():
        target = getattr(sampler, name)
        expected_duplicate = (local_before[name][0, :2] + remote_row) / 2.0
        assert torch.equal(target[0, :2], expected_duplicate)
        # Padding in the duplicated row is not a real AS bin and must not move.
        assert target[0, 2].item() == local_before[name][0, 2].item()
        # A rank-local clip with global cover count one must never be mixed.
        assert torch.equal(target[1], local_before[name][1])


def test_rank_local_sync_rejects_missing_duplicate_owner(monkeypatch) -> None:
    command = _rank_local_command(_rank_local_sampler())
    env = object.__new__(WholeBodyTrackingManager)
    env.command_manager = SimpleNamespace(get_state=lambda name: command)

    def fake_all_gather_object(outputs, local_payload, group=None):
        outputs[0] = local_payload
        outputs[1] = {"rank": 1, "clip_cover_counts": {}, "clips": {}}

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group=None: "gloo")
    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)

    with pytest.raises(RuntimeError, match="coverage differs"):
        env.synchronize_curriculum_state(device="cpu", world_size=2)


def test_rank_local_sync_rejects_duplicate_clip_mislabeled_unique(monkeypatch) -> None:
    command = _rank_local_command(_rank_local_sampler())
    command._rank_local_shard_metadata["clip_cover_counts"]["duplicated"] = 1
    env = object.__new__(WholeBodyTrackingManager)
    env.command_manager = SimpleNamespace(get_state=lambda name: command)

    def fake_all_gather_object(outputs, local_payload, group=None):
        outputs[0] = local_payload
        outputs[1] = {
            "rank": 1,
            "clip_cover_counts": {"duplicated": 1},
            "clips": {},
        }

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group=None: "gloo")
    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)

    with pytest.raises(RuntimeError, match="coverage differs"):
        env.synchronize_curriculum_state(device="cpu", world_size=2)


def test_resumed_multi_gpu_learn_syncs_as_before_first_reset() -> None:
    events: list[str] = []
    ppo = object.__new__(PPO)
    ppo.current_learning_iteration = 8
    ppo.config = SimpleNamespace(num_learning_iterations=10)
    ppo.is_multi_gpu = True
    ppo._train_mode = Mock()
    ppo._sync_training_curriculum_state = Mock(side_effect=lambda **_: events.append("schedule"))
    ppo._curriculum_state_sync_enabled = Mock(return_value=True)
    ppo._synchronize_curriculum_metrics = Mock(side_effect=lambda: events.append("as_sync"))

    def stop_after_reset():
        events.append("reset")
        raise RuntimeError("stop after observing resumed first-reset order")

    ppo.env = SimpleNamespace(reset_all=stop_after_reset)

    with pytest.raises(RuntimeError, match="resumed first-reset order"):
        ppo.learn()

    assert events == ["schedule", "as_sync", "reset"]


def test_single_gpu_learn_finalizes_curriculum_at_same_rollout_boundary() -> None:
    events: list[str] = []
    ppo = object.__new__(PPO)
    ppo.current_learning_iteration = 8
    ppo.config = SimpleNamespace(num_learning_iterations=10)
    ppo.is_multi_gpu = False
    ppo._train_mode = Mock()
    ppo._sync_training_curriculum_state = Mock(side_effect=lambda **_: events.append("schedule"))
    ppo._curriculum_state_sync_enabled = Mock(return_value=True)
    ppo._synchronize_curriculum_metrics = Mock(side_effect=lambda: events.append("curriculum_sync"))

    def stop_after_reset():
        events.append("reset")
        raise RuntimeError("stop after observing single-rank first-reset order")

    ppo.env = SimpleNamespace(reset_all=stop_after_reset)

    with pytest.raises(RuntimeError, match="single-rank first-reset order"):
        ppo.learn()

    assert events == ["schedule", "curriculum_sync", "reset"]
