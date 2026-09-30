from __future__ import annotations

from types import SimpleNamespace

import torch

from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.curriculum.terms.locomotion import WObjectDifficultyCurriculum
from holosoma.managers.perception.manager import PerceptionManager
from holosoma.utils.simulator_config import SimulatorType


def _states(num_envs: int, offset: float = 0.0) -> torch.Tensor:
    values = torch.arange(num_envs * 13, dtype=torch.float32).view(num_envs, 13) + offset
    values[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    return values


class _CountingStateAdapter:
    def __init__(self, states: torch.Tensor) -> None:
        self.states = states.clone()
        self.calls: list[tuple[str, torch.Tensor]] = []

    def get_object_states(self, object_name: str, env_ids: torch.Tensor) -> torch.Tensor:
        self.calls.append((object_name, env_ids.detach().cpu().clone()))
        return self.states[env_ids].clone()


class _UnexpectedRootStateRead:
    def __getitem__(self, _indices):
        raise AssertionError("single-object IsaacSim must bypass AllRootStatesProxy")


class _SingleObjectIsaacSim:
    def __init__(self, states: torch.Tensor) -> None:
        self._state_adapter = _CountingStateAdapter(states)
        self.all_root_states = _UnexpectedRootStateRead()
        self.write_calls: list[tuple[list[str], torch.Tensor, torch.Tensor]] = []

    @staticmethod
    def get_simulator_type() -> SimulatorType:
        return SimulatorType.ISAACSIM

    def set_actor_states(self, names: list[str], env_ids: torch.Tensor, states: torch.Tensor) -> None:
        self.write_calls.append((list(names), env_ids.clone(), states.clone()))
        assert names == ["box"]
        self._state_adapter.states[env_ids] = states


def _single_object_command(states: torch.Tensor) -> MotionCommand:
    command = object.__new__(MotionCommand)
    command.device = "cpu"
    command.num_envs = states.shape[0]
    command.motion = SimpleNamespace(has_object=True)
    command._env = SimpleNamespace(simulator=_SingleObjectIsaacSim(states))
    command._multi_object_enabled = False
    command._sim_object_names = ["box"]
    command.object_name = "box"
    command.object_indices_in_simulator = torch.arange(states.shape[0], dtype=torch.long)
    command._object_indices_matrix = None
    command._clip_object_ids = torch.zeros(1, dtype=torch.long)
    command.clip_ids = torch.zeros(states.shape[0], dtype=torch.long)
    command._simulator_object_state_snapshot = torch.empty_like(states)
    command._simulator_object_state_snapshot_ready = False
    return command


def test_four_object_properties_share_one_backend_read_per_control_step() -> None:
    initial = _states(4)
    command = _single_object_command(initial)
    adapter = command._env.simulator._state_adapter

    command.refresh_simulator_object_state_snapshot()

    torch.testing.assert_close(command.simulator_object_pos_w, initial[:, :3])
    torch.testing.assert_close(command.simulator_object_quat_w, initial[:, 3:7])
    torch.testing.assert_close(command.simulator_object_lin_vel_w, initial[:, 7:10])
    torch.testing.assert_close(command.simulator_object_ang_vel_w, initial[:, 10:13])
    assert len(adapter.calls) == 1
    assert adapter.calls[0][0] == "box"
    assert torch.equal(adapter.calls[0][1], torch.arange(4))

    # The next control step takes one new authoritative snapshot, then all
    # properties share it again.
    next_states = _states(4, offset=1000.0)
    adapter.states.copy_(next_states)
    command.refresh_simulator_object_state_snapshot()
    torch.testing.assert_close(command.simulator_object_pos_w, next_states[:, :3])
    torch.testing.assert_close(command.simulator_object_ang_vel_w, next_states[:, 10:13])
    assert len(adapter.calls) == 2


def test_subset_reset_refresh_preserves_survivors_and_write_updates_cache_immediately() -> None:
    initial = _states(4)
    command = _single_object_command(initial)
    adapter = command._env.simulator._state_adapter
    command.refresh_simulator_object_state_snapshot()

    backend_after_reset = _states(4, offset=2000.0)
    adapter.states.copy_(backend_after_reset)
    reset_env_ids = torch.tensor([1, 3], dtype=torch.long)
    command.refresh_simulator_object_state_snapshot(reset_env_ids)

    expected = initial.clone()
    expected[reset_env_ids] = backend_after_reset[reset_env_ids]
    torch.testing.assert_close(command.simulator_object_state_snapshot, expected)
    assert len(adapter.calls) == 2
    assert torch.equal(adapter.calls[-1][1], reset_env_ids)

    written = _states(1, offset=9000.0)
    command._set_simulator_object_states(torch.tensor([2]), written)
    torch.testing.assert_close(command.simulator_object_state_snapshot[2], written[0])
    # A write updates the already-valid snapshot directly; reading all four
    # component properties must not round-trip through the backend.
    _ = command.simulator_object_pos_w
    _ = command.simulator_object_quat_w
    _ = command.simulator_object_lin_vel_w
    _ = command.simulator_object_ang_vel_w
    assert len(adapter.calls) == 2


class _CountingRootStates:
    def __init__(self, states: torch.Tensor) -> None:
        self.states = states
        self.calls: list[torch.Tensor] = []

    def __getitem__(self, key):
        actor_indices, column_slice = key
        self.calls.append(actor_indices.detach().cpu().clone())
        return self.states[actor_indices, column_slice]


class _MultiObjectSimulator:
    def __init__(self, states: torch.Tensor, indices_by_name: dict[str, torch.Tensor]) -> None:
        self.all_root_states = _CountingRootStates(states)
        self.indices_by_name = indices_by_name

    @staticmethod
    def get_simulator_type() -> SimulatorType:
        return SimulatorType.ISAACSIM

    def set_actor_states(self, names: list[str], env_ids: torch.Tensor, states: torch.Tensor) -> None:
        rows_per_object = env_ids.numel()
        for object_index, name in enumerate(names):
            source = states[object_index * rows_per_object : (object_index + 1) * rows_per_object]
            self.all_root_states.states[self.indices_by_name[name][env_ids]] = source


def test_non_isaacsim_backends_keep_generic_tensor_contract() -> None:
    expected = _states(3)
    for simulator_type in (SimulatorType.MUJOCO, SimulatorType.ISAACGYM):
        root_states = _CountingRootStates(expected.clone())
        simulator = SimpleNamespace(
            get_simulator_type=lambda simulator_type=simulator_type: simulator_type,
            all_root_states=root_states,
        )
        command = _single_object_command(expected)
        command._env = SimpleNamespace(simulator=simulator)

        command.refresh_simulator_object_state_snapshot()

        torch.testing.assert_close(command.simulator_object_state_snapshot, expected)
        assert len(root_states.calls) == 1


def _multi_object_command() -> tuple[MotionCommand, _MultiObjectSimulator]:
    num_envs = 4
    flat_states = _states(num_envs * 3)
    indices_by_name = {
        "box": torch.tensor([1, 4, 7, 10], dtype=torch.long),
        "crate": torch.tensor([2, 5, 8, 11], dtype=torch.long),
    }
    simulator = _MultiObjectSimulator(flat_states, indices_by_name)
    command = object.__new__(MotionCommand)
    command.device = "cpu"
    command.num_envs = num_envs
    command.motion = SimpleNamespace(has_object=True)
    command._env = SimpleNamespace(simulator=simulator)
    command._multi_object_enabled = True
    command._sim_object_names = ["box", "crate"]
    command.object_name = "box"
    command.object_indices_in_simulator = indices_by_name["box"]
    command._object_indices_matrix = torch.stack([indices_by_name["box"], indices_by_name["crate"]])
    command._clip_object_ids = torch.tensor([0, 1], dtype=torch.long)
    command.clip_ids = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    command._simulator_object_state_snapshot = torch.empty((num_envs, 13))
    command._simulator_object_state_snapshot_ready = False
    return command, simulator


def test_multi_object_snapshot_uses_active_mapping_once_and_tracks_clip_rollover_write() -> None:
    command, simulator = _multi_object_command()
    active_indices = torch.tensor([1, 5, 8, 10], dtype=torch.long)

    command.refresh_simulator_object_state_snapshot()

    torch.testing.assert_close(
        command.simulator_object_state_snapshot,
        simulator.all_root_states.states[active_indices],
    )
    assert len(simulator.all_root_states.calls) == 1
    assert torch.equal(simulator.all_root_states.calls[0], active_indices)
    _ = command.simulator_object_pos_w
    _ = command.simulator_object_quat_w
    _ = command.simulator_object_lin_vel_w
    _ = command.simulator_object_ang_vel_w
    assert len(simulator.all_root_states.calls) == 1

    # Env 0 rolls from the box clip to the crate clip.  The reset write parks
    # the old object, writes the newly active one, and immediately updates the
    # active-state snapshot without depending on a later proxy read.
    command.clip_ids[0] = 1
    rollover_state = _states(1, offset=12000.0)
    command._set_simulator_object_states(torch.tensor([0]), rollover_state)
    torch.testing.assert_close(command.simulator_object_state_snapshot[0], rollover_state[0])
    torch.testing.assert_close(simulator.all_root_states.states[2], rollover_state[0])
    assert simulator.all_root_states.states[1, 2] == -100.0
    assert len(simulator.all_root_states.calls) == 1


def test_wbt_callback_refreshes_snapshot_before_perception_for_full_and_subset_updates() -> None:
    calls: list[torch.Tensor | None] = []
    perception_observed_call_counts: list[int] = []

    class _Command:
        def refresh_simulator_object_state_snapshot(self, env_ids=None) -> None:
            calls.append(None if env_ids is None else env_ids.clone())

    class _Perception:
        @staticmethod
        def uses_legacy_full_reset_refresh() -> bool:
            return False

        def update(self, _env_ids=None) -> None:
            perception_observed_call_counts.append(len(calls))

    command = _Command()
    perception = _Perception()
    env = object.__new__(WholeBodyTrackingManager)
    env.simulator = SimpleNamespace(base_quat=torch.zeros((3, 4)))
    env.base_quat = torch.empty((3, 4))
    env.command_manager = SimpleNamespace(get_state=lambda _name: command)
    env.perception_manager = perception
    env.teacher_perception_manager = perception
    env.critic_perception_manager = perception

    env._pre_compute_observations_callback()
    reset_ids = torch.tensor([2], dtype=torch.long)
    env._pre_compute_observations_callback(reset_ids)

    assert calls[0] is None
    assert torch.equal(calls[1], reset_ids)
    assert perception_observed_call_counts == [1, 2]


def test_object_assist_updates_shared_snapshot_after_deferred_backend_write() -> None:
    initial = torch.zeros((2, 13), dtype=torch.float32)
    initial[:, 6] = 1.0

    class _DeferredRootStates:
        def __init__(self) -> None:
            self.written: torch.Tensor | None = None

        def __getitem__(self, _key):
            raise AssertionError("object assist should consume the shared snapshot")

        def __setitem__(self, _key, values: torch.Tensor) -> None:
            self.written = values.clone()

    root_states = _DeferredRootStates()
    command = SimpleNamespace(
        motion=SimpleNamespace(has_object=True),
        simulator_object_state_snapshot=initial.clone(),
        object_pos_w=torch.tensor([[0.2, 0.0, 0.0], [0.0, 0.2, 0.0]]),
        object_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
        object_lin_vel_w=torch.zeros((2, 3)),
        _get_active_object_indices=lambda: torch.tensor([0, 1], dtype=torch.long),
    )

    def update_snapshot(env_ids: torch.Tensor, states: torch.Tensor) -> None:
        command.simulator_object_state_snapshot[env_ids] = states

    command._update_simulator_object_state_snapshot = update_snapshot
    env = SimpleNamespace(
        device="cpu",
        dt=0.1,
        is_evaluating=False,
        log_dict={},
        simulator=SimpleNamespace(all_root_states=root_states),
        command_manager=SimpleNamespace(get_state=lambda _name: command),
    )
    term = WObjectDifficultyCurriculum(
        SimpleNamespace(params={"enabled": True, "initial_lambda": 0.0}),
        env,
    )
    term.setup()

    term.step()

    assert root_states.written is not None
    torch.testing.assert_close(command.simulator_object_state_snapshot, root_states.written)
    assert torch.count_nonzero(command.simulator_object_state_snapshot[:, 7:10]) > 0


def test_far_tracking_reuses_snapshot_only_for_exact_single_active_object_mapping() -> None:
    snapshot = _states(3)
    command = SimpleNamespace(
        motion=SimpleNamespace(has_object=True),
        _multi_object_enabled=False,
        _sim_object_names=["box"],
        object_name="box",
        simulator_object_state_snapshot=snapshot,
    )
    manager = object.__new__(PerceptionManager)
    manager.num_envs = 3
    manager._far_tracking_object_names = ["box"]
    manager.env = SimpleNamespace(command_manager=SimpleNamespace(get_state=lambda _name: command))

    assert manager._shared_wbt_active_object_states() is snapshot

    command._multi_object_enabled = True
    assert manager._shared_wbt_active_object_states() is None
    command._multi_object_enabled = False
    manager._far_tracking_object_names = ["crate"]
    assert manager._shared_wbt_active_object_states() is None
