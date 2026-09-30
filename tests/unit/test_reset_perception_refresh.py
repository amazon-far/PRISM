from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.perception.manager import (
    PerceptionManager,
    _InfiniteFractalPerlin3D,
    _validated_rank_local_perlin_seed,
)


class _IndexedUpdateRecorder:
    def __init__(self, num_envs: int) -> None:
        self.counts = torch.zeros(num_envs, dtype=torch.long)
        self.calls: list[torch.Tensor | None] = []

    def update(self, env_ids: torch.Tensor | None = None) -> None:
        copied_ids = None if env_ids is None else env_ids.detach().cpu().clone()
        self.calls.append(copied_ids)
        if copied_ids is None:
            self.counts += 1
        else:
            self.counts[copied_ids] += 1


class _LegacyIndexedUpdateRecorder(_IndexedUpdateRecorder):
    @staticmethod
    def uses_legacy_full_reset_refresh() -> bool:
        return True


class _ResetRecorder:
    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def reset(self, env_ids: torch.Tensor) -> None:
        self.calls.append(env_ids.detach().cpu().clone())


class _RefreshSimulatorStub:
    def __init__(self, num_envs: int) -> None:
        self.base_quat = torch.arange(num_envs * 4, dtype=torch.float32).view(num_envs, 4)
        self.all_root_states = object()
        self.dof_state = object()
        self.calls: list[tuple[str, torch.Tensor | None]] = []

    def _record(self, name: str, env_ids: torch.Tensor | None = None) -> None:
        copied_ids = None if env_ids is None else env_ids.detach().cpu().clone()
        self.calls.append((name, copied_ids))

    def set_actor_root_state_tensor(self, env_ids: torch.Tensor, states: object) -> None:
        assert states is self.all_root_states
        self._record("root", env_ids)

    def set_dof_state_tensor(self, env_ids: torch.Tensor, states: object | None = None) -> None:
        assert states is None
        self._record("dof", env_ids)

    def clear_contact_forces_history(self, env_ids: torch.Tensor) -> None:
        self._record("contacts", env_ids)

    def refresh_sim_tensors(self) -> None:
        self._record("refresh")


@pytest.mark.parametrize(
    "manager_class",
    [WholeBodyTrackingManager, LeggedRobotLocomotionManager],
)
def test_reset_refresh_updates_only_reset_env_perception(manager_class: type) -> None:
    num_envs = 4
    reset_env_ids = torch.tensor([1, 3], dtype=torch.long)
    simulator = _RefreshSimulatorStub(num_envs)
    perception = _IndexedUpdateRecorder(num_envs)

    env = object.__new__(manager_class)
    env.simulator = simulator
    env.base_quat = torch.full((num_envs, 4), -1.0)
    env.need_to_refresh_envs = torch.ones(num_envs, dtype=torch.bool)
    env.perception_manager = perception
    # Alias all roles deliberately: one physical stream must be refreshed once.
    env.teacher_perception_manager = perception
    env.critic_perception_manager = perception
    if manager_class is LeggedRobotLocomotionManager:
        env.terrain_manager = SimpleNamespace(update_heights=lambda env_ids=None: None)

    # Model the ordinary all-environment update earlier in the same control step.
    env._pre_compute_observations_callback()
    before_partial_base_quat = env.base_quat.clone()
    simulator.base_quat += 1000.0

    env._refresh_envs_after_reset(reset_env_ids)

    assert perception.calls[0] is None
    assert torch.equal(perception.calls[1], reset_env_ids)
    assert torch.equal(perception.counts, torch.tensor([1, 2, 1, 2]))
    assert torch.equal(env.base_quat[reset_env_ids], simulator.base_quat[reset_env_ids])
    non_reset_env_ids = torch.tensor([0, 2], dtype=torch.long)
    assert torch.equal(
        env.base_quat[non_reset_env_ids],
        before_partial_base_quat[non_reset_env_ids],
    )
    assert not env.need_to_refresh_envs[reset_env_ids].any()
    assert env.need_to_refresh_envs[non_reset_env_ids].all()
    for name in ("root", "dof", "contacts"):
        matching_calls = [env_ids for call_name, env_ids in simulator.calls if call_name == name]
        assert len(matching_calls) == 1
        assert torch.equal(matching_calls[0], reset_env_ids)


def test_base_reset_resets_shared_perception_manager_once() -> None:
    reset_env_ids = torch.tensor([0, 2], dtype=torch.long)
    perception = _ResetRecorder()
    env = object.__new__(BaseTask)
    env.simulator = SimpleNamespace()
    env.observation_manager = SimpleNamespace(reset=lambda env_ids: None)
    env.perception_manager = perception
    env.teacher_perception_manager = perception
    env.critic_perception_manager = perception
    env._pending_episode_lengths = torch.zeros(3, dtype=torch.long)
    env._pending_episode_update_mask = torch.zeros(3, dtype=torch.bool)
    env.episode_length_buf = torch.arange(3, dtype=torch.long)
    env.randomization_manager = None
    env.action_manager = None
    env.command_manager = None
    env.curriculum_manager = None
    env.termination_manager = None
    env.reset_manager = SimpleNamespace(reset_scene=lambda env_ids: None)
    env._finalize_depth_logging_if_needed = lambda: None
    env._finalize_startup_depth_video_if_needed = lambda env_ids: None
    env._start_depth_logging_if_needed = lambda: None
    env._reset_envs_idx_impl = lambda env_ids, target_states, target_buf: None

    env.reset_envs_idx(reset_env_ids)

    assert len(perception.calls) == 1
    assert torch.equal(perception.calls[0], reset_env_ids)
    assert env._reset_refresh_pending is True


def test_legacy_reset_refresh_advances_full_vectorized_stream() -> None:
    perception = _LegacyIndexedUpdateRecorder(num_envs=4)
    env = object.__new__(BaseTask)
    env.perception_manager = perception
    env.teacher_perception_manager = perception
    env.critic_perception_manager = perception

    env._pre_compute_observations_callback(torch.tensor([1], dtype=torch.long))

    assert perception.calls == [None]
    assert torch.equal(perception.counts, torch.ones(4, dtype=torch.long))


def _make_camera_update_stub(*, freq_ratio: int = 3) -> PerceptionManager:
    manager = object.__new__(PerceptionManager)
    manager.enabled = True
    manager.num_envs = 3
    manager.device = "cpu"
    manager.cfg = SimpleNamespace(max_distance=10.0)
    manager.env = SimpleNamespace(dt=0.02)
    manager._debug_update_counter = 0
    manager._update_interval = 0.0
    manager._time_since_update = 0.0
    manager._camera_warp_preprocess = True
    manager._camera_warp_freq_ratio = freq_ratio
    manager._camera_obs_step_counter = 0
    manager._camera_depth = torch.full((manager.num_envs, 1, 1), -1.0)
    manager._camera_depth_obs = torch.full_like(manager._camera_depth, -1.0)
    manager._test_source_depth = torch.zeros_like(manager._camera_depth)
    manager._test_refresh_calls: list[tuple[torch.Tensor, bool]] = []

    manager._log_camera_randomization_state_once = lambda: None
    manager._uses_rendered_camera = lambda: False
    manager._uses_pytorch3d = lambda: False
    manager._uses_camera_far_tracking = lambda: True
    manager._uses_camera_scandots = lambda: False
    manager._uses_camera_raycast = lambda: False
    manager._prepare_camera_depth_for_observation = lambda depth, env_ids=None: depth
    manager._maybe_dump_camera_debug = lambda **kwargs: None
    manager._maybe_log_runtime_camera_alignment = lambda: None

    def compute_depth(env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return manager._test_source_depth.clone()
        return manager._test_source_depth[env_ids].clone()

    def update_observation(
        idx,
        depth: torch.Tensor,
        *,
        refresh: bool,
        advance_temporal_noise: bool = True,
    ) -> None:
        del advance_temporal_noise
        env_ids = manager._normalize_env_ids(idx)
        manager._test_refresh_calls.append((env_ids.detach().cpu().clone(), refresh))
        if refresh:
            manager._camera_depth_obs[env_ids] = depth

    manager._compute_far_tracking_camera_depth = compute_depth
    manager._update_camera_depth_observation = update_observation
    return manager


def test_partial_perception_update_preserves_global_camera_refresh_cadence() -> None:
    manager = _make_camera_update_stub(freq_ratio=3)

    manager._test_source_depth[:, 0, 0] = torch.tensor([10.0, 20.0, 30.0])
    manager.update()
    assert manager._camera_obs_step_counter == 1
    assert manager._test_refresh_calls[-1][1] is True

    # Reset env 1 between full control-frame updates.  Its fresh observation is
    # installed immediately, while the other environments retain their stream.
    manager._test_source_depth[:, 0, 0] = torch.tensor([11.0, 21.0, 31.0])
    manager.update(torch.tensor([1], dtype=torch.long))
    assert manager._camera_obs_step_counter == 1
    assert torch.equal(manager._test_refresh_calls[-1][0], torch.tensor([1]))
    assert manager._test_refresh_calls[-1][1] is True
    assert torch.equal(
        manager._camera_depth_obs[:, 0, 0],
        torch.tensor([10.0, 21.0, 30.0]),
    )
    assert torch.equal(
        manager._camera_depth[:, 0, 0],
        torch.tensor([10.0, 21.0, 30.0]),
    )

    # The partial reset is not a global camera tick.  With ratio 3, subsequent
    # full updates must still consume phases 1, 2, 0: False, False, True.
    for value in (40.0, 50.0, 60.0):
        manager._test_source_depth.fill_(value)
        manager.update()

    full_refresh_flags = [
        refresh
        for env_ids, refresh in manager._test_refresh_calls
        if env_ids.numel() == manager.num_envs
    ]
    assert full_refresh_flags == [True, False, False, True]
    assert manager._camera_obs_step_counter == 4
    assert torch.equal(
        manager._camera_depth_obs[:, 0, 0],
        torch.full((manager.num_envs,), 60.0),
    )


def test_full_camera_pipeline_advances_hole_clock_but_partial_reset_does_not() -> None:
    manager = _make_camera_update_stub(freq_ratio=1)
    manager.cfg = SimpleNamespace(camera_near=0.1, max_distance=10.0)
    manager._camera_depth = torch.ones((manager.num_envs, 4, 4))
    manager._camera_depth_obs = torch.zeros_like(manager._camera_depth)
    manager._heightmap = torch.zeros((manager.num_envs, 1, 1))
    manager._ray_hits_world = torch.zeros((manager.num_envs, 1, 3))
    manager._test_source_depth = torch.ones_like(manager._camera_depth)
    manager._camera_obs_height = 4
    manager._camera_obs_width = 4
    manager._camera_obs_fill_value = 10.0
    manager._camera_warp_buffer_len = 2
    manager._camera_warp_latency_frame = 0
    manager._camera_warp_latency_frame_range = None
    manager._camera_depth_buffer = torch.zeros(
        (manager.num_envs, manager._camera_warp_buffer_len, 4, 4)
    )
    manager._camera_depth_buffer_ready = torch.zeros(manager.num_envs, dtype=torch.bool)
    manager._camera_warp_crop_top = 0
    manager._camera_warp_crop_bottom = 0
    manager._camera_warp_crop_left = 0
    manager._camera_warp_crop_right = 0
    manager._camera_warp_min_valid_depth = 0.0
    manager._camera_warp_edge_noise = False
    manager._camera_warp_enable_holes = True
    manager._camera_warp_hole_prob = 0.2
    manager._camera_warp_hole_generator = _InfiniteFractalPerlin3D(
        shape=(4, 4),
        batch_size=manager.num_envs,
        resolutions=[(2, 2)],
        periods=[8],
        factors=[1.0],
        device="cpu",
    )
    manager._camera_warp_hole_frame_stats = None
    manager._camera_warp_additive_noise_std = 0.0
    manager._camera_warp_depth_offset_std = 0.0
    manager._camera_warp_normalize = False
    # Exercise the real buffer/preprocessing path rather than the cadence-only
    # recorder installed by ``_make_camera_update_stub``.
    manager._update_camera_depth_observation = PerceptionManager._update_camera_depth_observation.__get__(manager)

    manager.update()
    assert manager._camera_warp_hole_generator.frame_idx == 1
    surviving_buffers = manager._camera_depth_buffer[[0, 2]].clone()
    surviving_observations = manager._camera_depth_obs[[0, 2]].clone()

    reset_env_ids = torch.tensor([1], dtype=torch.long)
    manager.reset(reset_env_ids)
    manager.update(reset_env_ids)

    assert manager._camera_warp_hole_generator.frame_idx == 1
    assert torch.equal(manager._camera_depth_buffer[[0, 2]], surviving_buffers)
    assert torch.equal(manager._camera_depth_obs[[0, 2]], surviving_observations)
    assert manager._camera_depth_buffer_ready[reset_env_ids].all()


def test_partial_hole_frame_reuses_latest_full_tick_without_advancing() -> None:
    generator = _InfiniteFractalPerlin3D(
        shape=(4, 4),
        batch_size=3,
        resolutions=[(2, 2)],
        periods=[8],
        factors=[1.0],
        device="cpu",
    )

    full_frame = generator.generate_frame()
    assert generator.frame_idx == 1
    partial_frame = generator.generate_frame(
        frame_index=generator.frame_idx - 1,
        env_ids=torch.tensor([1], dtype=torch.long),
    )

    assert generator.frame_idx == 1
    assert torch.equal(partial_frame, full_frame[1:2])
    generator.generate_frame()
    assert generator.frame_idx == 2


def test_rank_local_perlin_seed_is_reproducible_distinct_and_rng_isolated() -> None:
    kwargs = {
        "shape": (4, 4),
        "batch_size": 3,
        "resolutions": [(2, 2)],
        "periods": [8],
        "factors": [1.0],
        "device": "cpu",
        "seed_semantics": "rank_local_v2",
    }
    torch.manual_seed(991)
    global_rng_before = torch.get_rng_state().clone()

    rank_zero = _InfiniteFractalPerlin3D(**kwargs, effective_seed=17)
    rank_zero_recreated = _InfiniteFractalPerlin3D(**kwargs, effective_seed=17)
    rank_one = _InfiniteFractalPerlin3D(**kwargs, effective_seed=18)
    rank_zero_frames = [rank_zero.generate_frame() for _ in range(3)]
    recreated_frames = [rank_zero_recreated.generate_frame() for _ in range(3)]
    rank_one_frames = [rank_one.generate_frame() for _ in range(3)]

    assert all(
        torch.equal(left, right)
        for left, right in zip(rank_zero_frames, recreated_frames, strict=True)
    )
    # At z=0 this implementation evaluates exactly on a zero-valued lattice
    # boundary for every seed; subsequent temporal frames must diverge.
    assert all(
        not torch.equal(left, right)
        for left, right in zip(rank_zero_frames[1:], rank_one_frames[1:], strict=True)
    )
    assert torch.equal(torch.get_rng_state(), global_rng_before)


def test_rank_local_perlin_seed_is_derived_and_checked_against_live_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "104")
    monkeypatch.setenv("RANK", "3")
    env = SimpleNamespace(training_config=SimpleNamespace(seed=17))
    original_rng_state = torch.get_rng_state().clone()
    try:
        torch.manual_seed(20)
        assert _validated_rank_local_perlin_seed(env) == 20

        torch.manual_seed(21)
        with pytest.raises(RuntimeError, match="seed contract mismatch"):
            _validated_rank_local_perlin_seed(env)
    finally:
        torch.set_rng_state(original_rng_state)


def test_rank_local_perlin_production_geometry_differs_on_first_frame() -> None:
    kwargs = {
        "shape": (64, 96),
        "batch_size": 2,
        "resolutions": [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32)],
        "periods": [32, 16, 8, 4, 2],
        "factors": [1.0],
        "device": "cpu",
        "seed_semantics": "rank_local_v2",
        "octave_profile": "legacy_single_octave_v1",
    }
    rank_zero = _InfiniteFractalPerlin3D(**kwargs, effective_seed=17)
    rank_one = _InfiniteFractalPerlin3D(**kwargs, effective_seed=18)

    assert not torch.equal(
        rank_zero.generate_frame(frame_index=0),
        rank_one.generate_frame(frame_index=0),
    )


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"seed_semantics": "rank_local_v2", "effective_seed": None}, "effective Perlin seed"),
        ({"seed_semantics": "unknown", "effective_seed": None}, "seed semantics"),
        ({"periods": [0]}, "positive integers"),
        ({"factors": [float("nan")]}, "finite real"),
        ({"resolutions": [(2, 2), (4, 4)]}, "equal lengths"),
    ],
)
def test_perlin_generator_rejects_ambiguous_or_invalid_schema(overrides, error: str) -> None:
    kwargs = {
        "shape": (4, 4),
        "batch_size": 3,
        "resolutions": [(2, 2)],
        "periods": [8],
        "factors": [1.0],
        "device": "cpu",
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=error):
        _InfiniteFractalPerlin3D(**kwargs)


def test_legacy_single_octave_profile_requires_upstream_layout() -> None:
    with pytest.raises(ValueError, match="5-candidate/1-active"):
        _InfiniteFractalPerlin3D(
            shape=(4, 4),
            batch_size=3,
            resolutions=[(2, 2)],
            periods=[8],
            factors=[1.0],
            device="cpu",
            octave_profile="legacy_single_octave_v1",
        )


def test_legacy_single_octave_profile_matches_explicit_octave_zero_across_period_boundary() -> None:
    legacy = _InfiniteFractalPerlin3D(
        shape=(64, 96),
        batch_size=2,
        resolutions=[(2, 2), (4, 4), (8, 8), (16, 16), (32, 32)],
        periods=[32, 16, 8, 4, 2],
        factors=[1.0],
        device="cpu",
        octave_profile="legacy_single_octave_v1",
    )
    explicit_octave_zero = _InfiniteFractalPerlin3D(
        shape=(64, 96),
        batch_size=2,
        resolutions=[(2, 2)],
        periods=[32],
        factors=[1.0],
        device="cpu",
    )

    for frame_index in (0, 1, 31, 32, 33):
        assert torch.equal(
            legacy.generate_frame(frame_index=frame_index),
            explicit_octave_zero.generate_frame(frame_index=frame_index),
        )
    assert all(not cache for cache in legacy.gradient_cache[1:])


def test_partial_hole_mask_matches_same_env_from_latest_full_tick() -> None:
    class _DeterministicHoleGenerator:
        def __init__(self) -> None:
            self.frame_idx = 0
            self.frame = torch.stack(
                (
                    torch.tensor(
                        [
                            [-10.0, -10.0, -10.0, -10.0],
                            [-10.0, 10.0, -10.0, -10.0],
                            [-10.0, -10.0, -10.0, -10.0],
                            [-10.0, -10.0, -10.0, -10.0],
                        ]
                    ),
                    torch.arange(16, dtype=torch.float32).view(4, 4) / 15.0,
                    torch.full((4, 4), -5.0),
                )
            )

        def generate_frame(
            self,
            *,
            frame_index: int | None = None,
            env_ids: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if frame_index is None:
                self.frame_idx += 1
            if env_ids is None:
                return self.frame.clone()
            return self.frame[env_ids].clone()

    manager = object.__new__(PerceptionManager)
    manager._camera_warp_hole_prob = 0.5
    manager._camera_warp_hole_generator = _DeterministicHoleGenerator()
    depth = torch.ones((3, 4, 4), dtype=torch.float32)

    full = manager._apply_warp_hole_noise(
        depth.clone(),
        max_depth=10.0,
    )
    assert manager._camera_warp_hole_generator.frame_idx == 1

    # This derived extrema cache is intentionally not checkpointed.  Model a
    # fresh-process load and require deterministic reconstruction from the
    # authenticated frame index before any new full tick.
    manager._camera_warp_hole_frame_stats = None
    reset_env_ids = torch.tensor([1], dtype=torch.long)
    partial = manager._apply_warp_hole_noise(
        depth[reset_env_ids].clone(),
        max_depth=10.0,
        env_ids=reset_env_ids,
        advance_frame=False,
    )

    assert manager._camera_warp_hole_generator.frame_idx == 1
    assert torch.equal(partial, full[reset_env_ids])


def test_partial_hole_refresh_at_frame_zero_rebuilds_without_advancing() -> None:
    manager = object.__new__(PerceptionManager)
    manager._camera_warp_hole_prob = 0.2
    manager._camera_warp_hole_generator = _InfiniteFractalPerlin3D(
        shape=(4, 4),
        batch_size=3,
        resolutions=[(2, 2)],
        periods=[8],
        factors=[1.0],
        device="cpu",
    )
    manager._camera_warp_hole_frame_stats = None
    depth = torch.ones((3, 4, 4), dtype=torch.float32)
    reset_env_ids = torch.tensor([2], dtype=torch.long)

    partial = manager._apply_warp_hole_noise(
        depth[reset_env_ids].clone(),
        max_depth=10.0,
        env_ids=reset_env_ids,
        advance_frame=False,
    )
    assert manager._camera_warp_hole_generator.frame_idx == 0

    full = manager._apply_warp_hole_noise(
        depth.clone(),
        max_depth=10.0,
    )
    assert manager._camera_warp_hole_generator.frame_idx == 1
    assert torch.equal(partial, full[reset_env_ids])
