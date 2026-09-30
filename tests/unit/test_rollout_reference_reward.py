from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import holosoma.managers.reward.terms.wbt as reward_wbt


class _DummyMotion:
    def __init__(self, clip_ids: list[str]):
        self.clip_ids = clip_ids
        self.num_clips = len(clip_ids)
        self.clip_lengths = torch.full((len(clip_ids),), 3, dtype=torch.long)
        self.has_object = True


class _DummyMotionCommand:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.num_envs = 2
        self.motion = _DummyMotion(["box_10"])
        self.motion_cfg = SimpleNamespace(
            body_names_to_track=["pelvis", "torso_link"],
            body_name_ref=["torso_link"],
        )
        self.ref_body_index = 1
        self.clip_ids = torch.tensor([0, 0], dtype=torch.long)
        self.time_steps = torch.tensor([1, 1], dtype=torch.long)
        identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)

        self.robot_body_pos_w = torch.tensor(
            [
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.2]],
                [[0.3, 0.0, 1.0], [0.0, 0.0, 1.2]],
            ],
            dtype=torch.float32,
        )
        self.robot_body_quat_w = identity_quat.view(1, 1, 4).repeat(self.num_envs, 2, 1)
        self.robot_body_lin_vel_w = torch.zeros((self.num_envs, 2, 3), dtype=torch.float32)
        self.robot_body_ang_vel_w = torch.zeros((self.num_envs, 2, 3), dtype=torch.float32)

        self.robot_root_pos_w = self.robot_body_pos_w[:, 0, :]
        self.robot_root_quat_w = identity_quat.view(1, 4).repeat(self.num_envs, 1)
        self.robot_ref_pos_w = self.robot_body_pos_w[:, self.ref_body_index, :]
        self.robot_ref_quat_w = identity_quat.view(1, 4).repeat(self.num_envs, 1)

        self.simulator_object_pos_w = torch.tensor(
            [
                [1.0, 0.0, 0.5],
                [1.25, 0.0, 0.5],
            ],
            dtype=torch.float32,
        )
        self.simulator_object_quat_w = identity_quat.view(1, 4).repeat(self.num_envs, 1)

    def _get_env_offsets(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        offsets = torch.zeros((self.num_envs, 3), dtype=torch.float32)
        if env_ids is None:
            return offsets
        return offsets.index_select(0, env_ids.to(dtype=torch.long))


def _write_rollout_reference(export_root: Path) -> None:
    clip_dir = export_root / "clips" / "0000_box_10"
    clip_dir.mkdir(parents=True, exist_ok=True)

    valid_steps = np.asarray([True, True, True], dtype=np.bool_)
    body_pos_local = np.asarray(
        [
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.2]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.2]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.2]],
        ],
        dtype=np.float32,
    )
    identity_quat = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    body_quat_w = np.broadcast_to(identity_quat, (3, 2, 4)).copy()
    zeros_body = np.zeros((3, 2, 3), dtype=np.float32)
    zeros_vec = np.zeros((3, 3), dtype=np.float32)
    ref_pos_local = np.asarray(
        [
            [0.0, 0.0, 1.2],
            [0.0, 0.0, 1.2],
            [0.0, 0.0, 1.2],
        ],
        dtype=np.float32,
    )
    root_pos_local = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    object_pos_local = np.asarray(
        [
            [1.0, 0.0, 0.5],
            [1.0, 0.0, 0.5],
            [1.0, 0.0, 0.5],
        ],
        dtype=np.float32,
    )
    object_quat_w = np.broadcast_to(identity_quat, (3, 4)).copy()

    np.savez_compressed(
        clip_dir / "teacher_rollout_reference.npz",
        clip_id=np.asarray("box_10"),
        clip_index=np.asarray(0, dtype=np.int32),
        tracked_body_names=np.asarray(["pelvis", "torso_link"]),
        ref_body_name=np.asarray("torso_link"),
        valid_steps=valid_steps,
        body_pos_local=body_pos_local,
        body_quat_w=body_quat_w,
        body_lin_vel_w=zeros_body,
        body_ang_vel_w=zeros_body,
        ref_pos_local=ref_pos_local,
        ref_quat_w=np.broadcast_to(identity_quat, (3, 4)).copy(),
        ref_lin_vel_w=zeros_vec,
        ref_ang_vel_w=zeros_vec,
        root_pos_local=root_pos_local,
        root_quat_w=np.broadcast_to(identity_quat, (3, 4)).copy(),
        root_lin_vel_w=zeros_vec,
        root_ang_vel_w=zeros_vec,
        object_pos_local=object_pos_local,
        object_quat_w=object_quat_w,
        object_lin_vel_w=zeros_vec,
        object_ang_vel_w=zeros_vec,
    )


def _build_env(tmp_path: Path) -> tuple[SimpleNamespace, _DummyMotionCommand, Path]:
    export_root = tmp_path / "rollout_reference_export"
    _write_rollout_reference(export_root)
    motion_command = _DummyMotionCommand()
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        episode_length_buf=torch.ones((2,), dtype=torch.long),
        command_manager=SimpleNamespace(get_state=lambda name: motion_command if name == "motion_command" else None),
    )
    return env, motion_command, export_root


def _set_rollout_valid_steps(export_root: Path, valid_steps: list[bool]) -> None:
    rollout_path = export_root / "clips" / "0000_box_10" / "teacher_rollout_reference.npz"
    with np.load(rollout_path, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    payload["valid_steps"] = np.asarray(valid_steps, dtype=np.bool_)
    np.savez_compressed(rollout_path, **payload)


def _enable_episodic_motion_end(env: SimpleNamespace) -> None:
    env.termination_manager = SimpleNamespace(
        _term_names=["motion_ends"],
        _term_cfgs=[SimpleNamespace(func="holosoma.managers.termination.terms.wbt:motion_ends")],
    )


def test_rollout_reference_object_reward_uses_exported_rollout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, motion_command, export_root = _build_env(tmp_path)
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)

    reward = reward_wbt.object_global_ref_position_error_exp(
        env,
        sigma=0.1,
        rollout_reference_root=str(export_root),
    )

    assert reward.shape == (2,)
    assert reward[0].item() == pytest.approx(1.0, rel=1e-5, abs=1e-5)
    assert reward[1].item() < 0.01


def test_rollout_reference_rejects_duplicate_clip_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, motion_command, export_root = _build_env(tmp_path)
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)
    source = export_root / "clips" / "0000_box_10"
    shutil.copytree(source, export_root / "clips" / "box_10")

    with pytest.raises(RuntimeError, match="multiple directories for an active clip"):
        reward_wbt.object_global_ref_position_error_exp(
            env,
            sigma=0.1,
            rollout_reference_root=str(export_root),
        )


def test_rollout_reference_allows_only_unreachable_final_frame_to_be_invalid_in_episodic_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, motion_command, export_root = _build_env(tmp_path)
    _enable_episodic_motion_end(env)
    _set_rollout_valid_steps(export_root, [True, True, False])
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)

    reward = reward_wbt.object_global_ref_position_error_exp(
        env,
        sigma=0.1,
        rollout_reference_root=str(export_root),
    )
    assert reward[0].item() == pytest.approx(1.0, rel=1e-5, abs=1e-5)


def test_rollout_reference_requires_final_frame_in_continuing_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, motion_command, export_root = _build_env(tmp_path)
    _set_rollout_valid_steps(export_root, [True, True, False])
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)

    with pytest.raises(RuntimeError, match=r"first_invalid_step=2.*motion_end_mode=continuing"):
        reward_wbt.object_global_ref_position_error_exp(
            env,
            sigma=0.1,
            rollout_reference_root=str(export_root),
        )


def test_rollout_reference_rejects_invalid_reward_bearing_frame_in_episodic_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, motion_command, export_root = _build_env(tmp_path)
    _enable_episodic_motion_end(env)
    _set_rollout_valid_steps(export_root, [True, False, False])
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)

    with pytest.raises(RuntimeError, match=r"first_invalid_step=1.*motion_end_mode=episodic"):
        reward_wbt.object_global_ref_position_error_exp(
            env,
            sigma=0.1,
            rollout_reference_root=str(export_root),
        )


def test_rollout_reference_reward_fails_when_configured_root_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, motion_command, _ = _build_env(tmp_path)
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)

    with pytest.raises(FileNotFoundError, match="Refusing to silently replace"):
        reward_wbt.object_global_ref_position_error_exp(
            env,
            sigma=0.1,
            rollout_reference_root=str(tmp_path / "missing"),
        )


def test_rollout_reference_reward_fails_when_an_active_clip_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, motion_command, export_root = _build_env(tmp_path)
    motion_command.motion.clip_ids = ["box_10", "box_11"]
    motion_command.motion.num_clips = 2
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)

    with pytest.raises(RuntimeError, match=r"Missing or invalid clip ids.*box_11"):
        reward_wbt.object_global_ref_position_error_exp(
            env,
            sigma=0.1,
            rollout_reference_root=str(export_root),
        )


def test_rollout_reference_global_ref_reward_uses_exported_rollout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, motion_command, export_root = _build_env(tmp_path)
    motion_command.robot_ref_pos_w = torch.tensor(
        [
            [0.0, 0.0, 1.2],
            [0.2, 0.0, 1.2],
        ],
        dtype=torch.float32,
    )
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)

    reward = reward_wbt.motion_global_ref_position_error_exp(
        env,
        sigma=0.1,
        rollout_reference_root=str(export_root),
    )

    assert reward.shape == (2,)
    assert reward[0].item() == pytest.approx(1.0, rel=1e-5, abs=1e-5)
    assert reward[1].item() < 0.05


def test_rollout_reference_relative_body_reward_uses_exported_rollout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, motion_command, export_root = _build_env(tmp_path)
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)

    reward = reward_wbt.motion_relative_body_position_error_exp(
        env,
        sigma=0.1,
        rollout_reference_root=str(export_root),
    )

    assert reward.shape == (2,)
    assert reward[0].item() == pytest.approx(1.0, rel=1e-5, abs=1e-5)
    assert reward[1].item() < 0.2


def test_rollout_reference_global_body_lin_vel_reward_uses_exported_rollout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, motion_command, export_root = _build_env(tmp_path)
    motion_command.robot_body_lin_vel_w[1, 0, 0] = 1.0
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)

    reward = reward_wbt.motion_global_body_lin_vel(
        env,
        sigma=0.1,
        rollout_reference_root=str(export_root),
    )

    assert reward.shape == (2,)
    assert reward[0].item() == pytest.approx(1.0, rel=1e-5, abs=1e-5)
    assert reward[1].item() < 0.1


def test_legacy_reward_alias_matches_unified_motion_reward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, motion_command, export_root = _build_env(tmp_path)
    motion_command.robot_ref_pos_w = torch.tensor(
        [
            [0.0, 0.0, 1.2],
            [0.2, 0.0, 1.2],
        ],
        dtype=torch.float32,
    )
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)

    unified = reward_wbt.motion_global_ref_position_error_exp(
        env,
        sigma=0.1,
        rollout_reference_root=str(export_root),
    )
    legacy_alias = reward_wbt.teacher_rollout_global_ref_position_error_exp(
        env,
        sigma=0.1,
        rollout_reference_root=str(export_root),
    )

    torch.testing.assert_close(legacy_alias, unified, rtol=0.0, atol=0.0)


def test_rollout_reference_sample_is_cached_within_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, motion_command, export_root = _build_env(tmp_path)
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)

    gather_call_count = 0
    original_gather = reward_wbt._gather_clip_timestep_values

    def _counting_gather(values: torch.Tensor, clip_indices: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        nonlocal gather_call_count
        gather_call_count += 1
        return original_gather(values, clip_indices, time_steps)

    monkeypatch.setattr(reward_wbt, "_gather_clip_timestep_values", _counting_gather)

    reward_wbt.motion_global_ref_position_error_exp(
        env,
        sigma=0.1,
        rollout_reference_root=str(export_root),
    )
    first_call_gathers = gather_call_count
    assert first_call_gathers > 0

    reward_wbt.motion_global_ref_orientation_error_exp(
        env,
        sigma=0.1,
        rollout_reference_root=str(export_root),
    )
    assert gather_call_count == first_call_gathers

    motion_command.time_steps += 1
    reward_wbt.motion_global_ref_orientation_error_exp(
        env,
        sigma=0.1,
        rollout_reference_root=str(export_root),
    )
    assert gather_call_count > first_call_gathers


def test_rollout_reference_relative_targets_are_cached_within_reward_compute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, motion_command, export_root = _build_env(tmp_path)
    env._reward_compute_counter = 1
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)

    quat_apply_call_count = 0
    original_quat_apply = reward_wbt.quat_apply_broadcast_left

    def _counting_quat_apply(quat: torch.Tensor, vec: torch.Tensor, *, w_last: bool = True) -> torch.Tensor:
        nonlocal quat_apply_call_count
        quat_apply_call_count += 1
        return original_quat_apply(quat, vec, w_last=w_last)

    monkeypatch.setattr(reward_wbt, "quat_apply_broadcast_left", _counting_quat_apply)

    reward_wbt.motion_relative_body_position_error_exp(
        env,
        sigma=0.1,
        rollout_reference_root=str(export_root),
    )
    first_call_quat_apply_count = quat_apply_call_count
    assert first_call_quat_apply_count > 0

    reward_wbt.motion_relative_body_orientation_error_exp(
        env,
        sigma=0.1,
        rollout_reference_root=str(export_root),
    )
    assert quat_apply_call_count == first_call_quat_apply_count

    env._reward_compute_counter += 1
    reward_wbt.motion_relative_body_orientation_error_exp(
        env,
        sigma=0.1,
        rollout_reference_root=str(export_root),
    )
    assert quat_apply_call_count > first_call_quat_apply_count
