from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from holosoma.config_types.reward import RewardTermCfg
from holosoma.config_values.wbt.g1 import reward as reward_config
import holosoma.managers.reward.terms.wbt as reward_wbt


class _DummyMotion:
    def __init__(self, clip_ids: list[str]):
        self.clip_ids = clip_ids
        self.num_clips = len(clip_ids)
        self.has_object = True
        self.fps = 50.0


class _DummyMotionCommand:
    def __init__(
        self,
        *,
        clip_names: list[str],
        active_clip_indices: torch.Tensor,
        time_steps: torch.Tensor,
        body_names: list[str],
        body_positions_obj: torch.Tensor,
        body_force_magnitudes: torch.Tensor,
        pickup_steps_by_clip: torch.Tensor | None = None,
    ):
        self.motion = _DummyMotion(clip_names)
        self.clip_ids = active_clip_indices
        self.time_steps = time_steps
        self.num_envs = int(active_clip_indices.shape[0])
        self.device = str(active_clip_indices.device)
        self._body_names = list(body_names)
        self._body_positions_obj = body_positions_obj
        self._body_force_magnitudes = body_force_magnitudes
        self._body_name_to_index = {name: idx for idx, name in enumerate(self._body_names)}
        self.body_positions_call_count = 0
        self.force_history_call_count = 0
        if pickup_steps_by_clip is None:
            pickup_steps_by_clip = torch.zeros((len(clip_names),), dtype=torch.long, device=active_clip_indices.device)
        self._pickup_steps_by_clip = pickup_steps_by_clip

    def _body_positions_in_object_frame(self, body_indices: torch.Tensor) -> torch.Tensor:
        self.body_positions_call_count += 1
        return self._body_positions_obj.index_select(1, body_indices.to(dtype=torch.long))

    def get_body_object_contact_force_history(self, body_names: list[str]) -> torch.Tensor:
        self.force_history_call_count += 1
        indices = torch.tensor(
            [self._body_name_to_index[name] for name in body_names],
            device=self._body_positions_obj.device,
            dtype=torch.long,
        )
        magnitudes = self._body_force_magnitudes.index_select(1, indices)
        forces = torch.zeros(
            (self.num_envs, 1, len(body_names), 3),
            device=self._body_positions_obj.device,
            dtype=torch.float32,
        )
        forces[..., 0] = magnitudes.unsqueeze(1)
        return forces

    def _get_clip_pickup_steps_by_clip(self) -> torch.Tensor:
        return self._pickup_steps_by_clip


def _write_contact_clip(
    export_root: Path,
    *,
    clip_dir_name: str,
    clip_id: str,
    left_points: np.ndarray,
    right_points: np.ndarray,
    stable_contact_success: bool = True,
    write_metadata: bool = True,
    left_contact_interval: tuple[int, int] | None = None,
    right_contact_interval: tuple[int, int] | None = None,
) -> None:
    clip_dir = export_root / "clips" / clip_dir_name
    clip_dir.mkdir(parents=True, exist_ok=True)
    if write_metadata:
        metadata = {
            "clip_id": clip_id,
            "stable_contact_success": stable_contact_success,
        }
        (clip_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    np.save(clip_dir / "left_wrist_contact_points.npy", left_points.astype(np.float32))
    np.save(clip_dir / "right_wrist_contact_points.npy", right_points.astype(np.float32))
    for region_name in (
        "left_elbow",
        "right_elbow",
        "left_wrist_roll",
        "right_wrist_roll",
        "left_wrist_pitch",
        "right_wrist_pitch",
    ):
        np.save(clip_dir / f"{region_name}_contact_points.npy", np.zeros((0, 3), dtype=np.float32))
    np.save(clip_dir / "torso_contact_points.npy", np.zeros((0, 3), dtype=np.float32))
    if left_contact_interval is not None:
        np.save(clip_dir / "left_wrist_contact_interval_steps.npy", np.asarray(left_contact_interval, dtype=np.int32))
    if right_contact_interval is not None:
        np.save(clip_dir / "right_wrist_contact_interval_steps.npy", np.asarray(right_contact_interval, dtype=np.int32))


def _build_test_env(
    tmp_path: Path,
    *,
    left_force: float,
    right_force: float,
    time_steps: torch.Tensor | None = None,
    pickup_steps_by_clip: torch.Tensor | None = None,
    left_contact_interval: tuple[int, int] | None = None,
    right_contact_interval: tuple[int, int] | None = None,
):
    export_root = tmp_path / "teacher_box_contacts"
    _write_contact_clip(
        export_root,
        clip_dir_name="0000_box_10",
        clip_id="box_10",
        left_points=np.asarray([[0.20, 0.00, 0.00]], dtype=np.float32),
        right_points=np.asarray([[-0.20, 0.00, 0.00]], dtype=np.float32),
        left_contact_interval=left_contact_interval,
        right_contact_interval=right_contact_interval,
    )

    body_names = [
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_wrist_roll_link",
        "right_wrist_roll_link",
        "left_wrist_pitch_link",
        "right_wrist_pitch_link",
        "torso_link",
    ]
    body_positions_obj = torch.zeros((2, len(body_names), 3), dtype=torch.float32)
    body_positions_obj[0, body_names.index("left_wrist_yaw_link")] = torch.tensor([0.20, 0.00, 0.00])
    body_positions_obj[0, body_names.index("right_wrist_yaw_link")] = torch.tensor([-0.20, 0.00, 0.00])
    body_positions_obj[1, body_names.index("left_wrist_yaw_link")] = torch.tensor([0.55, 0.00, 0.00])
    body_positions_obj[1, body_names.index("right_wrist_yaw_link")] = torch.tensor([-0.55, 0.00, 0.00])

    body_force_magnitudes = torch.zeros((2, len(body_names)), dtype=torch.float32)
    body_force_magnitudes[0, body_names.index("left_wrist_yaw_link")] = left_force
    body_force_magnitudes[0, body_names.index("right_wrist_yaw_link")] = right_force
    body_force_magnitudes[1, body_names.index("left_wrist_yaw_link")] = left_force
    body_force_magnitudes[1, body_names.index("right_wrist_yaw_link")] = right_force

    motion_command = _DummyMotionCommand(
        clip_names=["box_10", "box_11"],
        active_clip_indices=torch.tensor([0, 1], dtype=torch.long),
        time_steps=torch.tensor([0, 0], dtype=torch.long) if time_steps is None else time_steps,
        body_names=body_names,
        body_positions_obj=body_positions_obj,
        body_force_magnitudes=body_force_magnitudes,
        pickup_steps_by_clip=pickup_steps_by_clip,
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        simulator=SimpleNamespace(body_names=body_names),
        command_manager=SimpleNamespace(get_state=lambda name: motion_command if name == "motion_command" else None),
    )
    return env, export_root


def test_offline_contact_guidance_uses_clip_specific_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    env, export_root = _build_test_env(tmp_path, left_force=40.0, right_force=40.0)
    motion_command = env.command_manager.get_state("motion_command")
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)
    term = reward_wbt.OfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:OfflineContactPointGuidance",
            params={
                "contact_export_root": str(export_root),
                "region_names": ["left_wrist", "right_wrist"],
                "position_sigma": 0.05,
                "force_threshold": 25.0,
                "force_sigma": 10.0,
            },
            weight=1.0,
        ),
        env,
    )

    reward = term(env)
    assert reward.shape == (2,)
    assert reward[0].item() == pytest.approx(1.0, rel=1e-5, abs=1e-5)
    assert reward[1].item() == pytest.approx(0.0, rel=1e-5, abs=1e-5)


def test_offline_contact_guidance_rejects_duplicate_clip_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, export_root = _build_test_env(tmp_path, left_force=40.0, right_force=40.0)
    motion_command = env.command_manager.get_state("motion_command")
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)
    _write_contact_clip(
        export_root,
        clip_dir_name="box_10",
        clip_id="box_10",
        left_points=np.asarray([[0.20, 0.00, 0.00]], dtype=np.float32),
        right_points=np.asarray([[-0.20, 0.00, 0.00]], dtype=np.float32),
    )
    term = reward_wbt.OfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:OfflineContactPointGuidance",
            params={
                "contact_export_root": str(export_root),
                "region_names": ["left_wrist", "right_wrist"],
            },
            weight=1.0,
        ),
        env,
    )

    with pytest.raises(RuntimeError, match="Multiple offline-contact directories"):
        term(env)


def test_offline_contact_guidance_skips_nonobject_metadata_for_inactive_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, export_root = _build_test_env(tmp_path, left_force=40.0, right_force=40.0)
    inactive_dir = export_root / "clips" / "unrelated_clip"
    inactive_dir.mkdir()
    (inactive_dir / "metadata.json").write_text("[]", encoding="utf-8")
    motion_command = env.command_manager.get_state("motion_command")
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)
    term = reward_wbt.OfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:OfflineContactPointGuidance",
            params={
                "contact_export_root": str(export_root),
                "region_names": ["left_wrist", "right_wrist"],
            },
            weight=1.0,
        ),
        env,
    )

    reward = term(env)

    assert torch.isfinite(reward).all()


def test_offline_contact_guidance_rejects_nonobject_metadata_for_active_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, export_root = _build_test_env(tmp_path, left_force=40.0, right_force=40.0)
    (export_root / "clips" / "0000_box_10" / "metadata.json").write_text(
        "[]",
        encoding="utf-8",
    )
    motion_command = env.command_manager.get_state("motion_command")
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)
    term = reward_wbt.OfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:OfflineContactPointGuidance",
            params={
                "contact_export_root": str(export_root),
                "region_names": ["left_wrist", "right_wrist"],
            },
            weight=1.0,
        ),
        env,
    )

    with pytest.raises(RuntimeError, match="active clip must be a JSON object"):
        term(env)


def test_offline_contact_guidance_fails_when_configured_root_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, _ = _build_test_env(tmp_path, left_force=40.0, right_force=40.0)
    motion_command = env.command_manager.get_state("motion_command")
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)
    term = reward_wbt.OfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:OfflineContactPointGuidance",
            params={
                "contact_export_root": str(tmp_path / "missing"),
                "region_names": ["left_wrist", "right_wrist"],
            },
            weight=1.0,
        ),
        env,
    )

    with pytest.raises(FileNotFoundError, match="configured contact export root"):
        term(env)


def test_offline_contact_guidance_required_coverage_rejects_missing_clip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, export_root = _build_test_env(tmp_path, left_force=40.0, right_force=40.0)
    motion_command = env.command_manager.get_state("motion_command")
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)
    monkeypatch.setenv("HOLOSOMA_REQUIRE_CONTACT_TARGET_COVERAGE", "1")
    term = reward_wbt.OfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:OfflineContactPointGuidance",
            params={
                "contact_export_root": str(export_root),
                "region_names": ["left_wrist", "right_wrist"],
            },
            weight=1.0,
        ),
        env,
    )

    with pytest.raises(RuntimeError, match=r"only 1/2 clips.*box_11"):
        term(env)


def test_offline_contact_guidance_fails_on_non_finite_matching_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, export_root = _build_test_env(tmp_path, left_force=40.0, right_force=40.0)
    np.save(
        export_root / "clips" / "0000_box_10" / "left_wrist_contact_points.npy",
        np.asarray([[np.nan, 0.0, 0.0]], dtype=np.float32),
    )
    motion_command = env.command_manager.get_state("motion_command")
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)
    term = reward_wbt.OfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:OfflineContactPointGuidance",
            params={
                "contact_export_root": str(export_root),
                "region_names": ["left_wrist", "right_wrist"],
            },
            weight=1.0,
        ),
        env,
    )

    with pytest.raises(RuntimeError, match="contains NaN or Inf"):
        term(env)


def test_offline_contact_guidance_accepts_legacy_palm_region_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, export_root = _build_test_env(tmp_path, left_force=40.0, right_force=40.0)
    motion_command = env.command_manager.get_state("motion_command")
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)
    term = reward_wbt.OfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:OfflineContactPointGuidance",
            params={
                "contact_export_root": str(export_root),
                "region_names": ["left_palm", "right_palm"],
                "position_sigma": 0.05,
                "force_threshold": 25.0,
                "force_sigma": 10.0,
            },
            weight=1.0,
        ),
        env,
    )

    assert term.region_names == ["left_wrist", "right_wrist"]
    reward = term(env)
    assert reward[0].item() == pytest.approx(1.0, rel=1e-5, abs=1e-5)


def test_offline_contact_guidance_soft_force_term_saturates_below_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, export_root = _build_test_env(tmp_path, left_force=15.0, right_force=15.0)
    motion_command = env.command_manager.get_state("motion_command")
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)
    term = reward_wbt.OfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:OfflineContactPointGuidance",
            params={
                "contact_export_root": str(export_root),
                "region_names": ["left_wrist", "right_wrist"],
                "position_sigma": 0.05,
                "force_threshold": 25.0,
                "force_sigma": 10.0,
                "force_gate_mode": "soft",
            },
            weight=1.0,
        ),
        env,
    )

    reward = term(env)
    assert reward[0].item() == pytest.approx(float(np.exp(-1.0)), rel=1e-5, abs=1e-5)


def test_offline_contact_guidance_default_force_gate_requires_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, export_root = _build_test_env(tmp_path, left_force=15.0, right_force=15.0)
    motion_command = env.command_manager.get_state("motion_command")
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)
    term = reward_wbt.OfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:OfflineContactPointGuidance",
            params={
                "contact_export_root": str(export_root),
                "region_names": ["left_wrist", "right_wrist"],
                "position_sigma": 0.05,
                "force_threshold": 25.0,
                "force_sigma": 10.0,
            },
            weight=1.0,
        ),
        env,
    )

    reward = term(env)
    assert reward[0].item() == pytest.approx(0.0, rel=1e-5, abs=1e-5)


def test_offline_contact_guidance_supports_outputs_clips_wrist_reach_without_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, export_root = _build_test_env(tmp_path, left_force=0.0, right_force=0.0)
    clip_root = export_root / "clips"
    for child in list(clip_root.iterdir()):
        if child.is_dir():
            for file_path in child.iterdir():
                if file_path.name == "metadata.json":
                    file_path.unlink()

    motion_command = env.command_manager.get_state("motion_command")
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)
    term = reward_wbt.OfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:OfflineContactPointGuidance",
            params={
                "contact_export_root": str(export_root),
                "region_names": ["left_wrist", "right_wrist"],
                "position_sigma": 0.05,
                "use_force_term": False,
            },
            weight=1.0,
        ),
        env,
    )

    reward = term(env)
    assert reward[0].item() == pytest.approx(1.0, rel=1e-5, abs=1e-5)


def test_offline_contact_guidance_binary_force_gate_requires_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, export_root = _build_test_env(tmp_path, left_force=40.0, right_force=10.0)
    motion_command = env.command_manager.get_state("motion_command")
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)
    term = reward_wbt.OfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:OfflineContactPointGuidance",
            params={
                "contact_export_root": str(export_root),
                "region_names": ["left_wrist", "right_wrist"],
                "position_sigma": 0.05,
                "force_threshold": 25.0,
                "use_force_term": True,
                "force_gate_mode": "binary",
            },
            weight=1.0,
        ),
        env,
    )

    reward = term(env)
    assert reward[0].item() == pytest.approx(0.5, rel=1e-5, abs=1e-5)


def test_fused_offline_contact_guidance_matches_two_terms_with_shared_measurements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, export_root = _build_test_env(tmp_path, left_force=40.0, right_force=10.0)
    motion_command = env.command_manager.get_state("motion_command")
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)

    common_params = {
        "contact_export_root": str(export_root),
        "region_names": ["left_wrist", "right_wrist"],
        "position_sigma": 0.05,
        "force_threshold": 25.0,
        "force_gate_mode": "binary",
    }
    wrist_term = reward_wbt.OfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:OfflineContactPointGuidance",
            params={**common_params, "use_force_term": False},
            weight=1.0,
        ),
        env,
    )
    contact_term = reward_wbt.OfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:OfflineContactPointGuidance",
            params={**common_params, "use_force_term": True},
            weight=1.0,
        ),
        env,
    )
    fused_term = reward_wbt.FusedOfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:FusedOfflineContactPointGuidance",
            params={
                "contact_export_root": str(export_root),
                "wrist_region_names": ["left_wrist", "right_wrist"],
                "contact_region_names": ["left_wrist", "right_wrist"],
                "position_sigma": 0.05,
                "force_threshold": 25.0,
                "force_gate_mode": "binary",
                "wrist_weight": 3.0,
                "contact_weight": 4.0,
            },
            weight=1.0,
        ),
        env,
    )

    motion_command.body_positions_call_count = 0
    motion_command.force_history_call_count = 0
    expected = 3.0 * wrist_term(env) + 4.0 * contact_term(env)
    separate_body_position_calls = motion_command.body_positions_call_count
    separate_force_history_calls = motion_command.force_history_call_count

    motion_command.body_positions_call_count = 0
    motion_command.force_history_call_count = 0
    fused_reward = fused_term(env)

    torch.testing.assert_close(fused_reward, expected)
    assert motion_command.body_positions_call_count < separate_body_position_calls
    assert motion_command.force_history_call_count < separate_force_history_calls


def test_fused_offline_contact_defaults_keep_wrist_position_separate_from_all_contact_regions() -> None:
    expected_contact_regions = [
        "left_wrist",
        "right_wrist",
        "left_elbow",
        "right_elbow",
        "left_wrist_roll",
        "right_wrist_roll",
        "left_wrist_pitch",
        "right_wrist_pitch",
        "torso",
    ]
    presets = [
        reward_config.g1_29dof_wbt_reward_w_object_generalist_offline_contact_guidance,
        reward_config.g1_29dof_wbt_reward_w_object_r2s_contact_guidance,
        reward_config.g1_29dof_wbt_reward_w_object_r2s_rollout_reference_guidance,
    ]
    for preset in presets:
        params = preset.terms["offline_contact_guidance"].params
        assert params["wrist_region_names"] == ["left_wrist", "right_wrist"]
        assert params["contact_region_names"] == expected_contact_regions


def test_contact_active_mean_does_not_grow_with_region_count() -> None:
    min_distance = torch.zeros((2, 9), dtype=torch.float32)
    current_force = torch.full((2, 9), 2.0, dtype=torch.float32)
    active_regions = torch.zeros((2, 9), dtype=torch.bool)
    active_regions[0, :2] = True
    active_regions[1, :] = True

    reward = reward_wbt.OfflineContactPointGuidance._compute_guidance_reward_from_min_distance(
        min_distance=min_distance,
        current_force=current_force,
        active_regions=active_regions,
        position_sigma=0.05,
        use_force_term=True,
        force_threshold=1.0,
        force_sigma=10.0,
        force_gate_mode="binary",
    )

    torch.testing.assert_close(reward, torch.ones(2))


def test_fused_secondary_pitch_targets_affect_only_force_gated_contact_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, export_root = _build_test_env(tmp_path, left_force=0.0, right_force=0.0)
    clip_dir = export_root / "clips" / "0000_box_10"
    for region_name in ("left_wrist", "right_wrist"):
        np.save(clip_dir / f"{region_name}_contact_points.npy", np.zeros((0, 3), dtype=np.float32))
    for region_name in ("left_wrist_pitch", "right_wrist_pitch"):
        np.save(clip_dir / f"{region_name}_contact_points.npy", np.zeros((1, 3), dtype=np.float32))

    motion_command = env.command_manager.get_state("motion_command")
    for body_name in ("left_wrist_pitch_link", "right_wrist_pitch_link"):
        body_index = motion_command._body_name_to_index[body_name]
        motion_command._body_force_magnitudes[0, body_index] = 2.0
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)

    term = reward_wbt.FusedOfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:FusedOfflineContactPointGuidance",
            params={
                "contact_export_root": str(export_root),
                "wrist_region_names": ["left_wrist", "right_wrist"],
                "contact_region_names": reward_config._OFFLINE_CONTACT_GUIDANCE_REGION_NAMES,
                "position_sigma": 0.05,
                "force_threshold": 1.0,
                "force_gate_mode": "binary",
                "wrist_weight": 3.0,
                "contact_weight": 4.0,
            },
            weight=1.0,
        ),
        env,
    )

    # No wrist-yaw targets are active, so the wrist-position component is 0.
    # Both pitch targets are perfect and force-gated, so the contact mean is 1
    # and the total is exactly contact_weight, not wrist+contact weight.
    assert term(env)[0].item() == pytest.approx(4.0)


def test_offline_contact_guidance_uses_contact_schedule_mask_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, export_root = _build_test_env(
        tmp_path,
        left_force=40.0,
        right_force=40.0,
        time_steps=torch.tensor([0, 0], dtype=torch.long),
        left_contact_interval=(1, 3),
        right_contact_interval=(1, 3),
    )
    motion_command = env.command_manager.get_state("motion_command")
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)
    term = reward_wbt.OfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:OfflineContactPointGuidance",
            params={
                "contact_export_root": str(export_root),
                "region_names": ["left_wrist", "right_wrist"],
                "position_sigma": 0.05,
                "use_force_term": False,
                "use_contact_schedule": True,
                "contact_schedule_missing_mode": "always_on",
            },
            weight=1.0,
        ),
        env,
    )

    reward = term(env)
    assert reward[0].item() == pytest.approx(0.0, rel=1e-5, abs=1e-5)

    motion_command.time_steps[0] = 1
    reward = term(env)
    assert reward[0].item() == pytest.approx(1.0, rel=1e-5, abs=1e-5)


def test_offline_contact_guidance_falls_back_to_reference_pickup_gate_when_schedule_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, export_root = _build_test_env(
        tmp_path,
        left_force=40.0,
        right_force=40.0,
        time_steps=torch.tensor([0, 0], dtype=torch.long),
        pickup_steps_by_clip=torch.tensor([2, 0], dtype=torch.long),
    )
    motion_command = env.command_manager.get_state("motion_command")
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)
    term = reward_wbt.OfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:OfflineContactPointGuidance",
            params={
                "contact_export_root": str(export_root),
                "region_names": ["left_wrist", "right_wrist"],
                "position_sigma": 0.05,
                "use_force_term": False,
                "use_contact_schedule": True,
                "contact_schedule_missing_mode": "after_pickup",
            },
            weight=1.0,
        ),
        env,
    )

    reward = term(env)
    assert reward[0].item() == pytest.approx(0.0, rel=1e-5, abs=1e-5)

    motion_command.time_steps[0] = 2
    reward = term(env)
    assert reward[0].item() == pytest.approx(1.0, rel=1e-5, abs=1e-5)


def test_offline_contact_schedule_hot_path_uses_cached_host_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, export_root = _build_test_env(
        tmp_path,
        left_force=40.0,
        right_force=40.0,
        time_steps=torch.tensor([2, 0], dtype=torch.long),
        pickup_steps_by_clip=torch.tensor([2, 0], dtype=torch.long),
        left_contact_interval=(1, 3),
    )
    motion_command = env.command_manager.get_state("motion_command")
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)
    term = reward_wbt.OfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:OfflineContactPointGuidance",
            params={
                "contact_export_root": str(export_root),
                "region_names": ["left_wrist", "right_wrist"],
                "position_sigma": 0.05,
                "use_force_term": False,
                "use_contact_schedule": True,
                "contact_schedule_missing_mode": "after_pickup",
            },
            weight=1.0,
        ),
        env,
    )

    # First evaluation performs the one-time bank load and host metadata
    # validation.  Subsequent reward evaluations must not rediscover that
    # immutable state through Tensor.item(), which synchronizes CUDA tensors.
    first_reward = term(env)
    item_calls = 0
    original_item = torch.Tensor.item

    def _count_item_calls(tensor: torch.Tensor):
        nonlocal item_calls
        item_calls += 1
        return original_item(tensor)

    monkeypatch.setattr(torch.Tensor, "item", _count_item_calls)
    second_reward = term(env)

    assert item_calls == 0
    torch.testing.assert_close(second_reward, first_reward, rtol=0.0, atol=0.0)


def test_offline_contact_guidance_relaxes_contact_interval_by_configured_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, export_root = _build_test_env(
        tmp_path,
        left_force=40.0,
        right_force=40.0,
        time_steps=torch.tensor([4, 4], dtype=torch.long),
        left_contact_interval=(10, 20),
        right_contact_interval=(10, 20),
    )
    motion_command = env.command_manager.get_state("motion_command")
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)
    term = reward_wbt.OfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:OfflineContactPointGuidance",
            params={
                "contact_export_root": str(export_root),
                "region_names": ["left_wrist", "right_wrist"],
                "position_sigma": 0.05,
                "use_force_term": False,
                "use_contact_schedule": True,
                "contact_schedule_relax_steps": 5,
                "contact_schedule_missing_mode": "always_on",
            },
            weight=1.0,
        ),
        env,
    )

    reward = term(env)
    assert reward[0].item() == pytest.approx(0.0, rel=1e-5, abs=1e-5)

    motion_command.time_steps[0] = 5
    reward = term(env)
    assert reward[0].item() == pytest.approx(1.0, rel=1e-5, abs=1e-5)

    motion_command.time_steps[0] = 24
    reward = term(env)
    assert reward[0].item() == pytest.approx(1.0, rel=1e-5, abs=1e-5)

    motion_command.time_steps[0] = 25
    reward = term(env)
    assert reward[0].item() == pytest.approx(0.0, rel=1e-5, abs=1e-5)


def test_offline_contact_schedule_converts_runtime_prepend_to_motion_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, export_root = _build_test_env(
        tmp_path,
        left_force=40.0,
        right_force=40.0,
        time_steps=torch.tensor([1, 0], dtype=torch.long),
        left_contact_interval=(11, 13),
        right_contact_interval=(11, 13),
    )
    motion_command = env.command_manager.get_state("motion_command")
    motion_command.motion_cfg = SimpleNamespace(contact_interval_runtime_prepend_compensation=True)
    motion_command._runtime_default_pose_prepend_enabled = True
    motion_command._runtime_default_pose_prepend_steps = 10
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)
    term = reward_wbt.OfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:OfflineContactPointGuidance",
            params={
                "contact_export_root": str(export_root),
                "region_names": ["left_wrist", "right_wrist"],
                "position_sigma": 0.05,
                "use_force_term": False,
                "use_contact_schedule": True,
            },
            weight=1.0,
        ),
        env,
    )

    reward = term(env)
    assert reward[0].item() == pytest.approx(1.0, rel=1e-5, abs=1e-5)


def test_offline_contact_schedule_preserves_legacy_runtime_prepend_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, export_root = _build_test_env(
        tmp_path,
        left_force=40.0,
        right_force=40.0,
        time_steps=torch.tensor([11, 0], dtype=torch.long),
        left_contact_interval=(11, 13),
        right_contact_interval=(11, 13),
    )
    motion_command = env.command_manager.get_state("motion_command")
    motion_command.motion_cfg = SimpleNamespace(contact_interval_runtime_prepend_compensation=False)
    motion_command._runtime_default_pose_prepend_enabled = True
    motion_command._runtime_default_pose_prepend_steps = 10
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)
    term = reward_wbt.OfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:OfflineContactPointGuidance",
            params={
                "contact_export_root": str(export_root),
                "region_names": ["left_wrist", "right_wrist"],
                "position_sigma": 0.05,
                "use_force_term": False,
                "use_contact_schedule": True,
            },
            weight=1.0,
        ),
        env,
    )

    reward = term(env)
    assert reward[0].item() == pytest.approx(1.0, rel=1e-5, abs=1e-5)
