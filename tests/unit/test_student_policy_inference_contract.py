from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

import holosoma_inference.policies.wbt as inference_wbt_module
import holosoma_inference.utils.policy_contract as inference_policy_contract_module

from holosoma.agents.ppo.ppo import (
    _precomputed_turn_then_forward_deployment_contract,
    _rolling_reference_delta_deployment_contract,
)
from holosoma.eval_agent import _infer_inference_config
from holosoma.managers.command.terms.wbt import (
    motion_transition_contract_sha256 as training_motion_transition_contract_sha256,
)
from holosoma.config_values.wbt.g1 import observation as training_observation_values
from holosoma.config_values.wbt.g1.observation import (
    g1_29dof_wbt_observation_w_object,
    g1_29dof_wbt_observation_w_object_legacy,
    g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd,
)
from holosoma.config_values.wbt.g1.experiment import (
    g1_29dof_wbt_w_object_distill_sparse_root_cmd,
    g1_29dof_wbt_w_object_generalist,
    g1_29dof_wbt_w_object_generalist_legacy_obs,
)
from holosoma.utils.contact_intervals import (
    convert_contact_interval_timebase as _training_convert_contact_interval_timebase,
)
from holosoma_inference.config.config_types.observation import ObservationConfig, ObservationTermDescriptor
from holosoma_inference.config.config_values import observation as observation_values
from holosoma_inference.config.config_values import task as inference_task_values
from holosoma_inference.config.config_values.inference import (
    g1_29dof_wbt_object_as_contact_aware_depth_distill,
    g1_29dof_wbt_object_as_contact_aware_history5_depth_distill,
    g1_29dof_wbt_object_as_depth_distill,
    g1_29dof_wbt_object_contact_aware_depth_distill,
    g1_29dof_wbt_object_contact_aware_drop_button_depth_distill,
    g1_29dof_wbt_object_contact_aware_dual_button_depth_distill,
    g1_29dof_wbt_object_generalist,
    g1_29dof_wbt_object_velocity_generalist,
    g1_29dof_wbt_w_object_history1,
    g1_29dof_wbt_w_object_legacy,
)
from holosoma_inference.policies.wbt import (
    MotionData,
    WholeBodyTrackingPolicy,
    _CONTACT_WINDOW_OBSERVATION_TERMS,
    _convert_contact_interval_timebase,
    _extract_control_dt_from_metadata,
    _infer_contact_export_clip_id,
    _load_contact_interval_from_dir,
    _map_source_window_to_materialized_timeline,
    _select_primary_contact_interval,
    _validated_contact_aware_carry_window_config,
    _validated_runtime_motion_transition_settings,
)
from holosoma_inference.policies.base import BasePolicy
from holosoma_inference.policies import base as base_policy_module
from holosoma_inference.tools.patch_motion_onnx import (
    _maybe_apply_training_motion_transitions as _patch_motion_transitions,
)
from holosoma_inference.utils.policy_contract import (
    PolicyContractError,
    actor_perception_input_name_from_metadata,
    motion_transition_contract_from_metadata,
    perception_observation_contract_sha256_from_metadata,
    validate_onnx_policy_contract,
)


@pytest.mark.parametrize(
    ("directory_name", "expected"),
    [
        ("0034_noscale__any_barrel_12", "noscale__any_barrel_12"),
        ("noscale__any_barrel_12", "noscale__any_barrel_12"),
        ("box_10", "box_10"),
    ],
)
def test_inference_contact_directory_parser_only_strips_numeric_export_prefix(
    directory_name: str,
    expected: str,
) -> None:
    assert _infer_contact_export_clip_id(directory_name) == expected


def _drop_button_metadata(
    cfg=g1_29dof_wbt_object_contact_aware_drop_button_depth_distill,
) -> dict:
    terms_by_group = cfg.observation.obs_dict
    uses_perception = bool(getattr(cfg.task, "use_split_perception_obs", False))
    return {
        "dof_names": list(cfg.robot.dof_names),
        "kp": [10.0] * len(cfg.robot.dof_names),
        "kd": [1.0] * len(cfg.robot.dof_names),
        "experiment_config": {
            "robot": {
                "init_state": {
                    "default_joint_angles": dict(
                        zip(cfg.robot.dof_names, cfg.robot.default_dof_angles, strict=True)
                    )
                },
                "control": {
                    "control_type": "P",
                    "action_scale": 0.25,
                    "action_clip_value": 100.0,
                    "clip_actions": True,
                    "action_scales_by_effort_limit_over_p_gain": False,
                },
            },
            "action": {
                "terms": {
                    "joint_control": {
                        "func": "holosoma.managers.action.terms.joint_control:JointPositionActionTerm",
                        "params": {},
                        "scale": 1.0,
                        "clip": None,
                    }
                }
            },
            "algo": {
                "config": {
                    "module_dict": {
                        "actor": {
                            "input_dim": list(terms_by_group),
                            "layer_config": {
                                "perception_input_name": "perception_obs" if uses_perception else "",
                                "perception_input_height": 58 if uses_perception else None,
                                "perception_input_width": 87 if uses_perception else None,
                            },
                        }
                    }
                }
            },
            "observation": {
                "clip_observations": cfg.observation.clip_observations,
                "groups": {
                    group: {
                        "terms": {
                            term: {
                                "func": cfg.observation.term_descriptors[term].func,
                                "params": dict(cfg.observation.term_descriptors[term].params),
                                "scale": cfg.observation.obs_scales[term],
                                "noise": cfg.observation.term_descriptors[term].noise,
                                "clip": cfg.observation.term_descriptors[term].clip,
                            }
                            for term in terms
                        },
                        "history_length": cfg.observation.history_length_dict[group],
                        "concatenate": cfg.observation.group_concatenate[group],
                        "enable_noise": cfg.observation.group_enable_noise[group],
                    }
                    for group, terms in terms_by_group.items()
                }
            },
        },
    }


def _precomputed_turn_then_forward_metadata(
    *,
    zero_root_command_when_drop_active: bool = False,
) -> dict:
    metadata = _drop_button_metadata()
    motion_config = {
        "contact_aware_sparse_root_command_mode": (
            "precomputed_turn_then_forward"
        )
    }
    if zero_root_command_when_drop_active:
        motion_config["zero_root_command_when_drop_active"] = True
    metadata["experiment_config"]["command"] = {
        "setup_terms": {
            "motion_command": {
                "params": {
                    "motion_config": motion_config
                }
            }
        }
    }
    contract = (
        inference_policy_contract_module._expected_precomputed_turn_then_forward_contract(
            zero_root_command_when_drop_active=(
                zero_root_command_when_drop_active
            )
        )
    )
    contract_payload = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    contract_sha256 = hashlib.sha256(contract_payload).hexdigest()
    metadata["precomputed_turn_then_forward_deployment_contract"] = contract
    metadata["precomputed_turn_then_forward_deployment_contract_sha256"] = (
        contract_sha256
    )
    metadata["iteration"] = 9
    metadata["onnx_validation_contract"] = {
        "version": 1,
        "checker": "onnx.checker.check_model",
        "runtime": "onnxruntime_cpu",
        "pytorch_vs_ort": True,
        "input_names": ["actor_obs", "perception_obs"],
        "output_names": ["action"],
        "probe_rows": 6,
        "rtol": 1.0e-4,
        "atol": 1.0e-5,
        "max_abs_error": 1.0e-7,
        "max_rel_error": 2.0e-6,
        "completed_iteration": 9,
        "actor_graph_semantics": (
            "raw_actor_observation_plus_authenticated_external_observation_adapter"
        ),
        "precomputed_command_contract_sha256": contract_sha256,
        "rolling_reference_delta_contract_sha256": None,
    }
    return metadata


def _rolling_reference_delta_metadata(
    *,
    lookahead_motion_frames: int = 30,
    zero_yaw_threshold_deg: float = 0.0,
    zero_root_command_when_drop_active: bool = True,
) -> dict:
    metadata = _precomputed_turn_then_forward_metadata(
        zero_root_command_when_drop_active=zero_root_command_when_drop_active
    )
    motion_config = metadata["experiment_config"]["command"]["setup_terms"][
        "motion_command"
    ]["params"]["motion_config"]
    motion_config["contact_aware_sparse_root_command_mode"] = (
        "rolling_reference_delta"
    )
    motion_config["contact_aware_sparse_root_segment_steps"] = (
        lookahead_motion_frames
    )
    motion_config["contact_aware_sparse_root_zero_yaw_threshold_deg"] = (
        zero_yaw_threshold_deg
    )
    metadata.pop("precomputed_turn_then_forward_deployment_contract")
    metadata.pop("precomputed_turn_then_forward_deployment_contract_sha256")
    contract, contract_sha256 = _rolling_reference_delta_deployment_contract(
        lookahead_motion_frames=lookahead_motion_frames,
        zero_yaw_threshold_deg=zero_yaw_threshold_deg,
        zero_root_command_when_drop_active=(
            zero_root_command_when_drop_active
        ),
    )
    metadata["rolling_reference_delta_deployment_contract"] = contract
    metadata["rolling_reference_delta_deployment_contract_sha256"] = (
        contract_sha256
    )
    metadata["onnx_validation_contract"][
        "precomputed_command_contract_sha256"
    ] = None
    metadata["onnx_validation_contract"][
        "rolling_reference_delta_contract_sha256"
    ] = contract_sha256
    return metadata


def _motion_transition_contract_metadata(
    *,
    source_semantics: str,
    prepend_implementation: str,
    prepend_steps: int,
    append_implementation: str,
    append_steps: int,
) -> dict:
    contract = {
        "version": 1,
        "control_dt_s": 0.02,
        "source_semantics": source_semantics,
        "prepend": {
            "implementation": prepend_implementation,
            "applied": prepend_steps > 0,
            "steps": prepend_steps,
        },
        "append": {
            "implementation": append_implementation,
            "applied": append_steps > 0,
            "steps": append_steps,
        },
    }
    digest = hashlib.sha256(
        json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "motion_transition_contract": contract,
        "motion_transition_contract_sha256": digest,
    }


def _attach_onnx_metadata(model, metadata: dict) -> None:
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = json.dumps(value)


def test_onnx_session_and_metadata_use_one_hashed_byte_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def artifact(slot: int) -> bytes:
        actor_obs = helper.make_tensor_value_info("actor_obs", TensorProto.FLOAT, [1, 1])
        action = helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 1])
        node = helper.make_node("Identity", ["actor_obs"], ["action"])
        model = helper.make_model(helper.make_graph([node], "immutable", [actor_obs], [action]))
        _attach_onnx_metadata(model, {"slot": slot})
        return model.SerializeToString()

    original_payload = artifact(1)
    replacement_payload = artifact(2)
    model_path = tmp_path / "policy.onnx"
    model_path.write_bytes(original_payload)
    real_loader = onnx.load_model_from_string
    session_payloads: list[bytes] = []

    def swap_path_after_parse(payload: bytes):
        model_path.write_bytes(replacement_payload)
        return real_loader(payload)

    class _Session:
        def __init__(self, payload: bytes):
            session_payloads.append(payload)

    monkeypatch.setattr(base_policy_module.onnx, "load_model_from_string", swap_path_after_parse)
    monkeypatch.setattr(base_policy_module.onnxruntime, "InferenceSession", _Session)
    policy = object.__new__(BasePolicy)

    session, metadata = policy._load_onnx_session_and_metadata(str(model_path))

    assert isinstance(session, _Session)
    assert metadata == {"slot": 1}
    assert session_payloads == [original_payload]
    assert policy._onnx_artifact_sha256 == hashlib.sha256(original_payload).hexdigest()
    assert model_path.read_bytes() == replacement_payload


def test_onnx_loader_rejects_duplicate_metadata_keys_before_session_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor_obs = helper.make_tensor_value_info("actor_obs", TensorProto.FLOAT, [1, 1])
    action = helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 1])
    node = helper.make_node("Identity", ["actor_obs"], ["action"])
    model = helper.make_model(helper.make_graph([node], "duplicate_metadata", [actor_obs], [action]))
    for slot in (1, 2):
        entry = model.metadata_props.add()
        entry.key = "experiment_config"
        entry.value = json.dumps({"slot": slot})
    model_path = tmp_path / "duplicate-metadata.onnx"
    model_path.write_bytes(model.SerializeToString())

    session_calls: list[bytes] = []
    monkeypatch.setattr(
        base_policy_module.onnxruntime,
        "InferenceSession",
        lambda payload: session_calls.append(payload),
    )

    with pytest.raises(ValueError, match="duplicate metadata key 'experiment_config'"):
        object.__new__(BasePolicy)._load_onnx_session_and_metadata(str(model_path))

    assert session_calls == []


@pytest.mark.parametrize("raw_value", ["NaN", "Infinity", "-Infinity", '{"value": NaN}'])
def test_onnx_loader_rejects_nonfinite_json_metadata_before_session_creation(
    raw_value: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor_obs = helper.make_tensor_value_info("actor_obs", TensorProto.FLOAT, [1, 1])
    action = helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 1])
    node = helper.make_node("Identity", ["actor_obs"], ["action"])
    model = helper.make_model(helper.make_graph([node], "nonfinite_metadata", [actor_obs], [action]))
    entry = model.metadata_props.add()
    entry.key = "iteration"
    entry.value = raw_value
    model_path = tmp_path / "nonfinite-metadata.onnx"
    model_path.write_bytes(model.SerializeToString())

    session_calls: list[bytes] = []
    monkeypatch.setattr(
        base_policy_module.onnxruntime,
        "InferenceSession",
        lambda payload: session_calls.append(payload),
    )

    with pytest.raises(ValueError, match="not strict finite JSON"):
        object.__new__(BasePolicy)._load_onnx_session_and_metadata(str(model_path))

    assert session_calls == []


def _export_metadata_from_training_config(config) -> dict:
    return {
        "experiment_config": config.to_serializable_dict(),
        "dof_names": list(config.robot.dof_names),
        "kp": [1.0] * len(config.robot.dof_names),
        "kd": [1.0] * len(config.robot.dof_names),
    }


def _as_contact_aware_training_config(*, proprio_history: int):
    """Mirror the effective post-Tyro config exported by the AS launcher."""

    base = g1_29dof_wbt_w_object_distill_sparse_root_cmd
    groups = dict(base.observation.groups)
    groups["actor_obs_proprio_with_actions_no_linvel"] = replace(
        groups["actor_obs_proprio_with_actions_no_linvel"],
        history_length=proprio_history,
    )
    groups["critic_proprio_history"] = replace(
        groups["critic_proprio_history"],
        history_length=proprio_history,
    )
    actor = base.algo.config.module_dict.actor
    actor = replace(
        actor,
        type="MLPPerceptionEncoder",
        input_dim=[
            "actor_obs_root_contact_aware",
            "actor_obs_proprio_with_actions_no_linvel",
        ],
        layer_config=replace(
            actor.layer_config,
            perception_input_name="perception_obs",
            perception_input_height=58,
            perception_input_width=87,
        ),
    )
    return replace(
        base,
        observation=replace(base.observation, groups=groups),
        algo=replace(
            base.algo,
            config=replace(
                base.algo.config,
                module_dict=replace(base.algo.config.module_dict, actor=actor),
            ),
        ),
    )


def test_drop_button_onnx_contract_accepts_exact_94_plus_5046_layout() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    assert validate_onnx_policy_contract(
        metadata=_drop_button_metadata(),
        input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
        output_shapes={"action": [1, 29]},
        observation=cfg.observation,
        runtime_dof_names=cfg.robot.dof_names,
        runtime_default_dof_angles=cfg.robot.default_dof_angles,
    )


def test_precomputed_policy_requires_and_accepts_exact_export_parity_contract() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill

    assert validate_onnx_policy_contract(
        metadata=_precomputed_turn_then_forward_metadata(),
        input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
        output_shapes={"action": [1, 29]},
        observation=cfg.observation,
        runtime_dof_names=cfg.robot.dof_names,
        runtime_default_dof_angles=cfg.robot.default_dof_angles,
    )


def test_rolling_reference_delta_policy_requires_and_accepts_exact_contract() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill

    assert validate_onnx_policy_contract(
        metadata=_rolling_reference_delta_metadata(),
        input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
        output_shapes={"action": [1, 29]},
        observation=cfg.observation,
        runtime_dof_names=cfg.robot.dof_names,
        runtime_default_dof_angles=cfg.robot.default_dof_angles,
    )


@pytest.mark.parametrize("zero_root_command_when_drop_active", [False, True])
def test_training_and_inference_rolling_reference_delta_contracts_are_byte_exact(
    zero_root_command_when_drop_active: bool,
) -> None:
    training_contract, training_digest = (
        _rolling_reference_delta_deployment_contract(
            lookahead_motion_frames=30,
            zero_yaw_threshold_deg=0.0,
            zero_root_command_when_drop_active=(
                zero_root_command_when_drop_active
            ),
        )
    )
    inference_contract = (
        inference_policy_contract_module._expected_rolling_reference_delta_contract(
            lookahead_motion_frames=30,
            zero_yaw_threshold_deg=0.0,
            zero_root_command_when_drop_active=(
                zero_root_command_when_drop_active
            ),
        )
    )
    payload = json.dumps(
        inference_contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")

    assert training_contract == inference_contract
    assert training_digest == hashlib.sha256(payload).hexdigest()


def test_rolling_reference_delta_policy_rejects_missing_adapter_contract() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _rolling_reference_delta_metadata()
    metadata.pop("rolling_reference_delta_deployment_contract")
    metadata.pop("rolling_reference_delta_deployment_contract_sha256")

    with pytest.raises(PolicyContractError, match="missing its deployment adapter"):
        validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


@pytest.mark.parametrize("zero_root_command_when_drop_active", [False, True])
def test_training_and_inference_precomputed_deployment_contracts_are_byte_exact(
    zero_root_command_when_drop_active: bool,
) -> None:
    training_contract, training_digest = (
        _precomputed_turn_then_forward_deployment_contract(
            zero_root_command_when_drop_active=(
                zero_root_command_when_drop_active
            )
        )
    )
    inference_contract = (
        inference_policy_contract_module._expected_precomputed_turn_then_forward_contract(
            zero_root_command_when_drop_active=(
                zero_root_command_when_drop_active
            )
        )
    )
    payload = json.dumps(
        inference_contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")

    assert training_contract == inference_contract
    assert training_digest == hashlib.sha256(payload).hexdigest()
    assert training_contract["version"] == (
        2 if zero_root_command_when_drop_active else 1
    )


def test_precomputed_drop_exclusive_policy_accepts_exact_v2_contract() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill

    assert validate_onnx_policy_contract(
        metadata=_precomputed_turn_then_forward_metadata(
            zero_root_command_when_drop_active=True
        ),
        input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
        output_shapes={"action": [1, 29]},
        observation=cfg.observation,
        runtime_dof_names=cfg.robot.dof_names,
        runtime_default_dof_angles=cfg.robot.default_dof_angles,
    )


def test_precomputed_drop_exclusive_policy_rejects_v1_adapter_contract() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _precomputed_turn_then_forward_metadata(
        zero_root_command_when_drop_active=True
    )
    legacy_contract = (
        inference_policy_contract_module._expected_precomputed_turn_then_forward_contract()
    )
    payload = json.dumps(
        legacy_contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    metadata["precomputed_turn_then_forward_deployment_contract"] = legacy_contract
    metadata["precomputed_turn_then_forward_deployment_contract_sha256"] = (
        hashlib.sha256(payload).hexdigest()
    )

    with pytest.raises(PolicyContractError, match="does not match"):
        validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("runtime", "unverified", "field 'runtime'"),
        ("pytorch_vs_ort", False, "field 'pytorch_vs_ort'"),
        ("probe_rows", 5, "probe_rows must be exactly 6"),
        ("completed_iteration", 8, "must equal the policy metadata iteration"),
        ("input_names", ["actor_obs"], "input names do not match"),
    ],
)
def test_precomputed_policy_rejects_mutated_export_parity_contract(
    field: str,
    value: object,
    message: str,
) -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _precomputed_turn_then_forward_metadata()
    metadata["onnx_validation_contract"][field] = value

    with pytest.raises(PolicyContractError, match=message):
        validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


def test_precomputed_policy_rejects_missing_export_parity_contract() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _precomputed_turn_then_forward_metadata()
    del metadata["onnx_validation_contract"]

    with pytest.raises(PolicyContractError, match="onnx_validation_contract must be a mapping"):
        validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


@pytest.mark.parametrize(
    ("cfg", "actor_obs_dim"),
    [
        (g1_29dof_wbt_object_as_depth_distill, 96),
        (g1_29dof_wbt_object_as_contact_aware_depth_distill, 93),
        (g1_29dof_wbt_object_as_contact_aware_history5_depth_distill, 453),
    ],
)
def test_generic_as_onnx_contract_accepts_exact_actor_layout(cfg, actor_obs_dim: int) -> None:
    assert validate_onnx_policy_contract(
        metadata=_drop_button_metadata(cfg),
        input_shapes={"actor_obs": [1, actor_obs_dim], "perception_obs": [1, 58 * 87]},
        output_shapes={"action": [1, 29]},
        observation=cfg.observation,
        runtime_dof_names=cfg.robot.dof_names,
        runtime_default_dof_angles=cfg.robot.default_dof_angles,
    )


@pytest.mark.parametrize("action_shape", [[29], [1], [1, "action_dim"], [1, 29, 1]])
def test_policy_contract_rejects_non_matrix_or_dynamic_action_features(action_shape) -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill

    with pytest.raises(PolicyContractError, match="rank 2|static positive feature dimension"):
        validate_onnx_policy_contract(
            metadata=_drop_button_metadata(cfg),
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": action_shape},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


def test_policy_contract_rejects_non_float32_graph_input_before_runtime() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill

    with pytest.raises(PolicyContractError, match="must have type tensor\\(float\\)"):
        validate_onnx_policy_contract(
            metadata=_drop_button_metadata(),
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            input_types={"actor_obs": "tensor(double)", "perception_obs": "tensor(float)"},
            output_types={"action": "tensor(float)"},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


@pytest.mark.parametrize("bad_batch", [2, True, 1.0, np.int64(2), ""])
def test_policy_contract_rejects_invalid_or_non_runtime_batch_dimensions(bad_batch) -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill

    with pytest.raises(PolicyContractError, match="batch dimension"):
        validate_onnx_policy_contract(
            metadata=_drop_button_metadata(),
            input_shapes={"actor_obs": [bad_batch, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


@pytest.mark.parametrize("batch", [1, np.int64(1), "batch", None])
def test_policy_contract_accepts_runtime_or_dynamic_batch_dimensions(batch) -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill

    assert validate_onnx_policy_contract(
        metadata=_drop_button_metadata(),
        input_shapes={"actor_obs": [batch, 94], "perception_obs": [batch, 58 * 87]},
        output_shapes={"action": [batch, 29]},
        observation=cfg.observation,
        runtime_dof_names=cfg.robot.dof_names,
        runtime_default_dof_angles=cfg.robot.default_dof_angles,
    )


@pytest.mark.parametrize(
    ("shape_target", "bad_shape", "message"),
    [
        ("time_step", [1, 2], "time_step input must have feature dimension 1"),
        ("joint_pos", [1, 28], "motion output 'joint_pos' dimension"),
        ("joint_vel", [29], "must have rank 2"),
        ("ref_pos_xyz", [2, 3], "fixed batch dimension 2"),
        ("ref_quat_xyzw", [1, 3], "motion output 'ref_quat_xyzw' dimension"),
    ],
)
def test_policy_contract_rejects_malformed_legacy_motion_tensor_shapes(
    shape_target: str,
    bad_shape: list[object],
    message: str,
) -> None:
    cfg = g1_29dof_wbt_w_object_history1
    input_shapes: dict[str, list[object]] = {
        "actor_obs": [1, 175],
        "time_step": [1, 1],
    }
    output_shapes: dict[str, list[object]] = {
        "action": [1, 29],
        "joint_pos": [1, 29],
        "joint_vel": [1, 29],
        "ref_pos_xyz": [1, 3],
        "ref_quat_xyzw": [1, 4],
    }
    target_shapes = input_shapes if shape_target == "time_step" else output_shapes
    target_shapes[shape_target] = bad_shape

    with pytest.raises(PolicyContractError, match=message):
        validate_onnx_policy_contract(
            metadata=_drop_button_metadata(cfg),
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


def test_policy_contract_rejects_embedded_motion_outputs_without_time_step() -> None:
    cfg = g1_29dof_wbt_w_object_history1

    with pytest.raises(PolicyContractError, match="motion outputs require a time_step input"):
        validate_onnx_policy_contract(
            metadata=_drop_button_metadata(cfg),
            input_shapes={"actor_obs": [1, 175]},
            output_shapes={
                "action": [1, 29],
                "joint_pos": [1, 29],
                "joint_vel": [1, 29],
                "ref_pos_xyz": [1, 3],
                "ref_quat_xyzw": [1, 4],
            },
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


def test_policy_contract_rejects_motion_graph_for_sparse_root_student() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill

    with pytest.raises(PolicyContractError, match="motion-command observation contract"):
        validate_onnx_policy_contract(
            metadata=_drop_button_metadata(),
            input_shapes={
                "actor_obs": [1, 94],
                "perception_obs": [1, 58 * 87],
                "time_step": [1, 1],
            },
            output_shapes={
                "action": [1, 29],
                "joint_pos": [1, 29],
                "joint_vel": [1, 29],
                "ref_quat_xyzw": [1, 4],
            },
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


def test_dual_button_onnx_contract_accepts_exact_95_plus_5046_layout() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_dual_button_depth_distill
    assert validate_onnx_policy_contract(
        metadata=_drop_button_metadata(cfg),
        input_shapes={"actor_obs": [1, 95], "perception_obs": [1, 58 * 87]},
        output_shapes={"action": [1, 29]},
        observation=cfg.observation,
        runtime_dof_names=cfg.robot.dof_names,
        runtime_default_dof_angles=cfg.robot.default_dof_angles,
    )


def test_student_contract_supports_authenticated_custom_perception_input_name() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _drop_button_metadata()
    actor = metadata["experiment_config"]["algo"]["config"]["module_dict"]["actor"]
    actor["layer_config"]["perception_input_name"] = "depth_features"

    assert actor_perception_input_name_from_metadata(metadata) == "depth_features"
    assert validate_onnx_policy_contract(
        metadata=metadata,
        input_shapes={"actor_obs": [1, 94], "depth_features": [1, 58 * 87]},
        output_shapes={"action": [1, 29]},
        observation=cfg.observation,
        runtime_dof_names=cfg.robot.dof_names,
        runtime_default_dof_angles=cfg.robot.default_dof_angles,
    )


def test_student_contract_rejects_perception_input_name_collision() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _drop_button_metadata()
    actor = metadata["experiment_config"]["algo"]["config"]["module_dict"]["actor"]
    actor["layer_config"].update(
        {
            "perception_input_name": "actor_obs",
            "perception_input_height": 1,
            "perception_input_width": 94,
        }
    )

    with pytest.raises(PolicyContractError, match="collides with a reserved"):
        validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={"actor_obs": [1, 94]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("perception_input_height", None),
        ("perception_input_width", "87"),
        ("perception_input_height", True),
        ("perception_input_width", 0),
    ],
)
def test_student_contract_rejects_missing_or_malformed_perception_geometry(field: str, value: object) -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _drop_button_metadata()
    actor = metadata["experiment_config"]["algo"]["config"]["module_dict"]["actor"]
    actor["layer_config"][field] = value

    with pytest.raises(PolicyContractError, match="positive integer"):
        validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


def _attach_effective_perception_contract(metadata: dict) -> str:
    contract = {
        "version": 2,
        "camera_source": "far_tracking_warp",
        "camera_obs_shape": [58, 87],
        "normalize": True,
        "producer_tick_dt": 0.02,
        "producer_lifecycle": {
            "reset_refresh_semantics": "targeted_v2",
            "ordinary_manager_update_calls_per_control_tick": 1,
            "initialization_control_ticks_before_first_reset_output": 1,
            "initialization_ordinary_manager_update_calls_before_first_reset_output": 1,
            "reset_output_republished_until_physics_advances": True,
            "reset_output_scope": "reset_env_subset",
            "hole_clock_advances_on_reset_refresh": False,
            "camera_frequency_phase_advances_on_reset_refresh": False,
            "camera_producer_reset_refresh_consumes_process_global_rng": True,
            "future_noise_sample_path_peer_reset_coupled": True,
            "batch_size_invariant_sample_path": False,
            "stochastic_equivalence": "distribution_only",
            "seed_replay_scope": "same_execution_trace_only",
        },
        "camera_reset_randomization": None,
        "camera_setup_randomization": None,
        "camera_ray_correction_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "hole_generator_schema": None,
        "training_geometry_support": {
            "version": 1,
            "camera_source": "far_tracking_warp",
            "training_rank_count": 2,
            "robot_mesh_bindings": [
                {
                    "slot_name": "torso_link",
                    "mesh": {
                        "suffix": ".stl",
                        "size_bytes": 3,
                        "sha256": "11" * 32,
                    },
                    "tracking_body_name": "torso_link",
                    "fixed_position_xyz": [0.0, 0.0, 0.0],
                    "fixed_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            ],
            "object_mesh_support": [
                {
                    "source_name": "object",
                    "mesh": {
                        "suffix": ".obj",
                        "size_bytes": 7,
                        "sha256": "22" * 32,
                    },
                    "training_active_env_count": 16,
                }
            ],
        },
    }
    digest = hashlib.sha256(
        json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    metadata["perception_observation_contract"] = contract
    metadata["perception_observation_contract_sha256"] = digest
    return digest


def _refresh_effective_perception_contract_digest(metadata: dict) -> None:
    contract = metadata["perception_observation_contract"]
    metadata["perception_observation_contract_sha256"] = hashlib.sha256(
        json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_student_contract_authenticates_effective_perception_contract_digest() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _drop_button_metadata()
    _attach_effective_perception_contract(metadata)

    assert validate_onnx_policy_contract(
        metadata=metadata,
        input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
        output_shapes={"action": [1, 29]},
        observation=cfg.observation,
        runtime_dof_names=cfg.robot.dof_names,
        runtime_default_dof_angles=cfg.robot.default_dof_angles,
    )

    metadata["perception_observation_contract_sha256"] = "00" * 32
    with pytest.raises(PolicyContractError, match="does not match"):
        validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


def test_student_contract_requires_training_geometry_support() -> None:
    metadata = _drop_button_metadata()
    _attach_effective_perception_contract(metadata)
    del metadata["perception_observation_contract"]["training_geometry_support"]
    _refresh_effective_perception_contract_digest(metadata)

    with pytest.raises(PolicyContractError, match="training_geometry_support"):
        perception_observation_contract_sha256_from_metadata(metadata)


def _attach_rank_local_hole_schema(metadata: dict) -> dict:
    contract = metadata["perception_observation_contract"]
    contract["hole_generator_schema"] = {
        "shape": [64, 96],
        "resolutions": [[2, 2], [4, 4], [8, 8], [16, 16], [32, 32]],
        "periods": [32, 16, 8, 4, 2],
        "factors": [1.0],
        "normalization_scope": "reference_batch",
        "reference_batch_size": 64,
        "seed_semantics": "rank_local_v2",
        "effective_seed": 42,
        "gradient_seed_mixer": "sha256_u63_be_v1",
        "octave_profile": "legacy_single_octave_v1",
    }
    _refresh_effective_perception_contract_digest(metadata)
    return contract["hole_generator_schema"]


def test_student_contract_accepts_complete_rank_local_hole_seed_schema() -> None:
    metadata = _drop_button_metadata()
    _attach_effective_perception_contract(metadata)
    _attach_rank_local_hole_schema(metadata)

    assert perception_observation_contract_sha256_from_metadata(metadata)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda schema: schema.pop("effective_seed"), "all present or all absent"),
        (lambda schema: schema.update(effective_seed=True), "effective_seed"),
        (lambda schema: schema.update(gradient_seed_mixer="unknown"), "gradient_seed_mixer"),
        (lambda schema: schema.update(octave_profile="unknown"), "octave_profile"),
    ],
)
def test_student_contract_rejects_invalid_rank_local_hole_seed_schema(
    mutation,
    error: str,
) -> None:
    metadata = _drop_button_metadata()
    _attach_effective_perception_contract(metadata)
    schema = _attach_rank_local_hole_schema(metadata)
    mutation(schema)
    _refresh_effective_perception_contract_digest(metadata)

    with pytest.raises(PolicyContractError, match=error):
        perception_observation_contract_sha256_from_metadata(metadata)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda support: support.update({"unexpected": True}),
            "must contain exactly",
        ),
        (
            lambda support: support["object_mesh_support"][0]["mesh"].update(
                {"suffix": "OBJ"}
            ),
            "lowercase file suffix",
        ),
        (
            lambda support: support["object_mesh_support"][0]["mesh"].update(
                {"sha256": "AA" * 32}
            ),
            "lowercase hexadecimal",
        ),
        (
            lambda support: support.update({"training_rank_count": True}),
            "training_rank_count must be a positive integer",
        ),
        (
            lambda support: support["robot_mesh_bindings"].append(
                support["robot_mesh_bindings"][0].copy()
            ),
            "canonical sorted unique",
        ),
    ],
)
def test_student_contract_rejects_malformed_training_geometry_support(
    mutation,
    error: str,
) -> None:
    metadata = _drop_button_metadata()
    _attach_effective_perception_contract(metadata)
    support = metadata["perception_observation_contract"]["training_geometry_support"]
    mutation(support)
    _refresh_effective_perception_contract_digest(metadata)

    with pytest.raises(PolicyContractError, match=error):
        perception_observation_contract_sha256_from_metadata(metadata)


def test_student_contract_rejects_unsorted_training_geometry_support() -> None:
    metadata = _drop_button_metadata()
    _attach_effective_perception_contract(metadata)
    support = metadata["perception_observation_contract"]["training_geometry_support"]
    support["object_mesh_support"].append(
        {
            "source_name": "another_object",
            "mesh": {
                "suffix": ".obj",
                "size_bytes": 5,
                "sha256": "00" * 32,
            },
            "training_active_env_count": 4,
        }
    )
    _refresh_effective_perception_contract_digest(metadata)

    with pytest.raises(PolicyContractError, match="canonical sorted unique"):
        perception_observation_contract_sha256_from_metadata(metadata)


def test_student_contract_rejects_nonunit_training_geometry_quaternion() -> None:
    metadata = _drop_button_metadata()
    _attach_effective_perception_contract(metadata)
    support = metadata["perception_observation_contract"]["training_geometry_support"]
    support["robot_mesh_bindings"][0]["fixed_quaternion_xyzw"] = [0.0, 0.0, 0.0, 2.0]
    _refresh_effective_perception_contract_digest(metadata)

    with pytest.raises(PolicyContractError, match="unit quaternion"):
        perception_observation_contract_sha256_from_metadata(metadata)


def test_student_contract_rejects_non_far_tracking_geometry_entries() -> None:
    metadata = _drop_button_metadata()
    _attach_effective_perception_contract(metadata)
    contract = metadata["perception_observation_contract"]
    contract["camera_source"] = "rendered"
    contract["training_geometry_support"]["camera_source"] = "rendered"
    _refresh_effective_perception_contract_digest(metadata)

    with pytest.raises(PolicyContractError, match="mesh lists must both be empty"):
        perception_observation_contract_sha256_from_metadata(metadata)


def test_student_contract_rejects_legacy_perception_contract_without_lifecycle() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _drop_button_metadata()
    _attach_effective_perception_contract(metadata)
    contract = metadata["perception_observation_contract"]
    contract["version"] = 1
    metadata["perception_observation_contract_sha256"] = hashlib.sha256(
        json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(PolicyContractError, match="legacy v1 artifacts"):
        validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


def test_student_contract_rejects_false_exact_sample_path_claim() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _drop_button_metadata()
    _attach_effective_perception_contract(metadata)
    contract = metadata["perception_observation_contract"]
    contract["producer_lifecycle"]["stochastic_equivalence"] = "exact_sample_path"
    metadata["perception_observation_contract_sha256"] = hashlib.sha256(
        json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(PolicyContractError, match="authenticates distributions"):
        validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


def test_student_contract_binds_authenticated_camera_shape_to_actor_geometry() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _drop_button_metadata()
    _attach_effective_perception_contract(metadata)
    contract = metadata["perception_observation_contract"]
    contract["camera_obs_shape"] = [29, 174]  # Same flattened width, different image geometry.
    metadata["perception_observation_contract_sha256"] = hashlib.sha256(
        json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(PolicyContractError, match="camera_obs_shape does not match"):
        validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


def test_split_perception_zmq_requires_matching_effective_contract() -> None:
    expected_digest = "11" * 32
    policy = object.__new__(BasePolicy)
    policy._perception_contract_sha256 = expected_digest
    policy._perception_obs_shm_sub = None
    policy._perception_obs_sub = SimpleNamespace(
        get_payload=lambda: {
            "perception_contract_sha256": expected_digest,
            "episode_generation": 7,
            "perception_obs": [1.0, 2.0],
        }
    )
    np.testing.assert_array_equal(
        policy._get_split_perception_obs(2, target_episode_generation=7),
        np.array([[1.0, 2.0]], dtype=np.float32),
    )

    policy._perception_obs_sub = SimpleNamespace(
        get_payload=lambda: {
            "perception_contract_sha256": "22" * 32,
            "episode_generation": 7,
            "perception_obs": [1.0, 2.0],
        }
    )
    with pytest.raises(RuntimeError, match="does not match"):
        policy._get_split_perception_obs(2, target_episode_generation=7)


def test_split_perception_requires_pinned_episode_identity() -> None:
    policy = object.__new__(BasePolicy)
    policy._perception_contract_sha256 = "11" * 32
    policy._perception_obs_shm_sub = None
    policy._perception_obs_sub = SimpleNamespace(get_payload=lambda: None)

    with pytest.raises(RuntimeError, match="pinned simulator episode_generation"):
        policy._get_split_perception_obs(2)


def test_split_perception_rejects_artifact_without_effective_contract() -> None:
    policy = object.__new__(BasePolicy)
    policy._perception_contract_sha256 = None
    policy._perception_obs_shm_sub = None
    policy._perception_obs_sub = None

    with pytest.raises(RuntimeError, match="no authenticated effective producer contract"):
        policy._get_split_perception_obs(2)


def test_student_contract_rejects_undeclared_extra_input() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _drop_button_metadata()

    with pytest.raises(PolicyContractError, match="not declared"):
        validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={
                "actor_obs": [1, 94],
                "perception_obs": [1, 58 * 87],
                "stale_depth": [1, 58 * 87],
            },
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


def _set_defm_metadata(metadata: dict, *, normalize: object = False) -> None:
    actor = metadata["experiment_config"]["algo"]["config"]["module_dict"]["actor"]
    actor["layer_config"]["perception_encoder_type"] = "defm_efficientnet_b2"
    metadata["experiment_config"]["perception"] = {
        "enabled": True,
        "output_mode": "camera_depth",
        "encoder_type": "defm_efficientnet_b2",
        "camera_warp_normalize": normalize,
    }


def test_defm_onnx_contract_accepts_explicit_metric_depth_semantics() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _drop_button_metadata()
    _set_defm_metadata(metadata, normalize=False)

    assert validate_onnx_policy_contract(
        metadata=metadata,
        input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
        output_shapes={"action": [1, 29]},
        observation=cfg.observation,
        runtime_dof_names=cfg.robot.dof_names,
        runtime_default_dof_angles=cfg.robot.default_dof_angles,
    )


@pytest.mark.parametrize("normalize", [True, None])
def test_defm_onnx_contract_rejects_nonmetric_or_missing_depth_semantics(normalize: object) -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _drop_button_metadata()
    _set_defm_metadata(metadata, normalize=normalize)

    with pytest.raises(PolicyContractError, match="metric depth in meters"):
        validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


def test_defm_onnx_contract_rejects_encoder_metadata_disagreement() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _drop_button_metadata()
    _set_defm_metadata(metadata, normalize=False)
    metadata["experiment_config"]["perception"]["encoder_type"] = "defm_vit_s14"

    with pytest.raises(PolicyContractError, match="must declare the same encoder"):
        validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


def _saved_sparse_root_term(metadata: dict) -> dict:
    return metadata["experiment_config"]["observation"]["groups"][
        "actor_obs_root_contact_aware"
    ]["terms"]["sparse_target_root_trajectory_command_contact_aware"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "func",
            "holosoma.managers.observation.terms.wbt:sparse_target_root_trajectory_command",
            "Runtime func",
        ),
        ("params", {"reference_frame": "torso"}, "Runtime params"),
        ("noise", "disabled", "must be numeric"),
        ("clip", [-1.0, 1.0], "Runtime clip"),
    ],
)
def test_active_sparse_root_contract_rejects_same_dim_semantic_drift(field, value, message) -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _drop_button_metadata()
    _saved_sparse_root_term(metadata)[field] = value

    with pytest.raises(PolicyContractError, match=message):
        validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


def test_active_sparse_root_contract_rejects_group_concatenate_drift() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _drop_button_metadata()
    metadata["experiment_config"]["observation"]["groups"][
        "actor_obs_root_contact_aware"
    ]["concatenate"] = False

    with pytest.raises(PolicyContractError, match="concatenate setting"):
        validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


@pytest.mark.parametrize("missing_field", ["func", "params", "scale", "noise", "clip"])
def test_complete_metadata_rejects_missing_sparse_root_term_fields(missing_field) -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _drop_button_metadata()
    del _saved_sparse_root_term(metadata)[missing_field]

    with pytest.raises(PolicyContractError, match="incomplete"):
        validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


@pytest.mark.parametrize("missing_field", ["terms", "history_length", "concatenate", "enable_noise"])
def test_complete_metadata_rejects_missing_actor_group_fields(missing_field) -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _drop_button_metadata()
    saved_group = metadata["experiment_config"]["observation"]["groups"][
        "actor_obs_root_contact_aware"
    ]
    del saved_group[missing_field]

    with pytest.raises(PolicyContractError, match=missing_field):
        validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


def test_partial_experiment_metadata_is_not_treated_as_legacy() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    with pytest.raises(PolicyContractError, match="algo.config.module_dict.actor"):
        validate_onnx_policy_contract(
            metadata={"experiment_config": {}},
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


def test_all_inference_presets_have_complete_canonical_observation_descriptors() -> None:
    for name, config in observation_values.DEFAULTS.items():
        terms = {term for group_terms in config.obs_dict.values() for term in group_terms}
        assert set(config.term_descriptors) == terms, name
        assert set(config.group_concatenate) == set(config.obs_dict), name
        assert set(config.group_enable_noise) == set(config.obs_dict), name
        assert all(config.group_concatenate.values()), name


def test_inference_term_clip_order_matches_training_history_and_global_clip() -> None:
    policy = object.__new__(BasePolicy)
    policy.config = SimpleNamespace(
        observation=ObservationConfig(
            obs_dict={"actor_obs": ["signal"]},
            obs_dims={"signal": 1},
            obs_scales={"signal": 2.0},
            history_length_dict={"actor_obs": 2},
            clip_observations=2.0,
            term_descriptors={
                "signal": ObservationTermDescriptor(
                    func="tests:signal",
                    clip=(-3.0, 3.0),
                )
            },
            group_concatenate={"actor_obs": True},
            group_enable_noise={"actor_obs": False},
        )
    )
    policy._init_obs_config()

    first = policy._update_obs_history(
        policy.parse_current_obs_dict(
            {"signal": np.array([[2.0]], dtype=np.float32)}
        )
    )
    second = policy._update_obs_history(
        policy.parse_current_obs_dict(
            {"signal": np.array([[0.5]], dtype=np.float32)}
        )
    )

    # 2 * 2 -> term-clipped to 3, then stored in history; the group clip is
    # applied only after history flattening.  On the next frame history is
    # old-to-new: [3, 0.5 * 2], before the final +/-2 clip.
    np.testing.assert_allclose(first["actor_obs"], [[0.0, 2.0]])
    np.testing.assert_allclose(second["actor_obs"], [[2.0, 1.0]])


@pytest.mark.parametrize(
    "cfg",
    [
        g1_29dof_wbt_object_as_depth_distill,
        g1_29dof_wbt_object_as_contact_aware_depth_distill,
        g1_29dof_wbt_object_contact_aware_drop_button_depth_distill,
        g1_29dof_wbt_object_contact_aware_dual_button_depth_distill,
    ],
)
def test_active_as_student_observation_descriptors_match_training_source_of_truth(cfg) -> None:
    """Prevent the deployment contract itself from drifting from AS training."""

    training = g1_29dof_wbt_observation_w_object_distill_sparse_root_cmd
    runtime = cfg.observation
    for group_name, runtime_terms in runtime.obs_dict.items():
        training_group = training.groups[group_name]
        assert sorted(runtime_terms) == sorted(training_group.terms), group_name
        assert runtime.history_length_dict[group_name] == training_group.history_length, group_name
        assert runtime.group_concatenate[group_name] is training_group.concatenate, group_name
        assert runtime.group_enable_noise[group_name] is training_group.enable_noise, group_name
        for term_name in runtime_terms:
            training_term = training_group.terms[term_name]
            descriptor = runtime.term_descriptors[term_name]
            assert descriptor.func == training_term.func, term_name
            assert descriptor.params == training_term.params, term_name
            assert descriptor.noise == training_term.noise, term_name
            assert descriptor.clip == training_term.clip, term_name
            assert runtime.obs_scales[term_name] == training_term.scale, term_name


def test_inference_contact_window_consumers_match_training_source_of_truth() -> None:
    training_consumer_terms = set().union(
        training_observation_values.object_distill_sparse_root_cmd_terms_contact_aware,
        training_observation_values.object_distill_drop_button_terms,
        training_observation_values.object_distill_pickup_button_terms,
    )

    assert _CONTACT_WINDOW_OBSERVATION_TERMS == frozenset(training_consumer_terms)


def test_as_contact_aware_history5_observation_matches_effective_training_contract() -> None:
    training = _as_contact_aware_training_config(proprio_history=5)
    runtime = g1_29dof_wbt_object_as_contact_aware_history5_depth_distill.observation
    actor_groups = training.algo.config.module_dict.actor.input_dim
    assert actor_groups == list(runtime.obs_dict)

    total_dim = 0
    for group_name, runtime_terms in runtime.obs_dict.items():
        training_group = training.observation.groups[group_name]
        assert sorted(runtime_terms) == sorted(training_group.terms)
        assert runtime.history_length_dict[group_name] == training_group.history_length
        assert runtime.group_concatenate[group_name] is training_group.concatenate
        assert runtime.group_enable_noise[group_name] is training_group.enable_noise
        frame_dim = sum(runtime.obs_dims[term] for term in runtime_terms)
        total_dim += frame_dim * runtime.history_length_dict[group_name]
        for term_name in runtime_terms:
            training_term = training_group.terms[term_name]
            descriptor = runtime.term_descriptors[term_name]
            assert descriptor.func == training_term.func
            assert descriptor.params == training_term.params
            assert descriptor.noise == training_term.noise
            assert descriptor.clip == training_term.clip
            assert runtime.obs_scales[term_name] == training_term.scale
    assert total_dim == 453


def test_full_exported_as_history5_contract_validates_and_auto_selects() -> None:
    training = _as_contact_aware_training_config(proprio_history=5)
    runtime = g1_29dof_wbt_object_as_contact_aware_history5_depth_distill
    metadata = _export_metadata_from_training_config(training)
    assert validate_onnx_policy_contract(
        metadata=metadata,
        input_shapes={"actor_obs": [1, 453], "perception_obs": [1, 58 * 87]},
        output_shapes={"action": [1, 29]},
        observation=runtime.observation,
        runtime_dof_names=runtime.robot.dof_names,
        runtime_default_dof_angles=runtime.robot.default_dof_angles,
        runtime_motor_effort_limits=runtime.robot.motor_effort_limit,
        runtime_joint2motor=runtime.robot.joint2motor,
    )

    actor_obs = helper.make_tensor_value_info("actor_obs", TensorProto.FLOAT, [1, 453])
    perception_obs = helper.make_tensor_value_info("perception_obs", TensorProto.FLOAT, [1, 58 * 87])
    action = helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 29])
    model = helper.make_model(helper.make_graph([], "contract", [actor_obs, perception_obs], [action]))
    _attach_onnx_metadata(model, metadata)
    script = _load_inference_config_script()
    assert script.infer_inference_config(model) == (
        "g1-29dof-wbt-object-as-contact-aware-history5-depth-distill"
    )


def test_object_linear_velocity_contract_pins_correct_v2_function() -> None:
    descriptor = observation_values.wbt_object_velocity_generalist.term_descriptors["obj_lin_vel_b"]
    assert descriptor.func == "holosoma.managers.observation.terms.wbt:obj_lin_vel_b_v2"


@pytest.mark.parametrize(
    ("runtime", "training"),
    [
        (observation_values.wbt_w_object, g1_29dof_wbt_observation_w_object),
        (observation_values.wbt_w_object_legacy, g1_29dof_wbt_observation_w_object_legacy),
    ],
)
def test_object_actor_presets_match_direct_training_source(runtime, training) -> None:
    training_group = training.groups["actor_obs"]
    assert sorted(runtime.obs_dict["actor_obs"]) == sorted(training_group.terms)
    assert runtime.history_length_dict["actor_obs"] == training_group.history_length == 5
    assert runtime.group_concatenate["actor_obs"] is training_group.concatenate is True
    assert runtime.group_enable_noise["actor_obs"] is training_group.enable_noise is True
    for term_name, training_term in training_group.terms.items():
        descriptor = runtime.term_descriptors[term_name]
        assert descriptor.func == training_term.func
        assert descriptor.params == training_term.params
        assert descriptor.noise == training_term.noise
        assert descriptor.clip == training_term.clip
        assert runtime.obs_scales[term_name] == training_term.scale


@pytest.mark.parametrize(
    ("training_config", "runtime_config"),
    [
        (g1_29dof_wbt_w_object_generalist, g1_29dof_wbt_object_generalist),
        (g1_29dof_wbt_w_object_generalist_legacy_obs, g1_29dof_wbt_w_object_legacy),
    ],
)
def test_full_exported_object_training_contract_validates_at_inference(
    training_config,
    runtime_config,
) -> None:
    assert validate_onnx_policy_contract(
        metadata=_export_metadata_from_training_config(training_config),
        input_shapes={"actor_obs": [1, 875]},
        output_shapes={"action": [1, 29]},
        observation=runtime_config.observation,
        runtime_dof_names=runtime_config.robot.dof_names,
        runtime_default_dof_angles=runtime_config.robot.default_dof_angles,
        runtime_motor_effort_limits=runtime_config.robot.motor_effort_limit,
        runtime_joint2motor=runtime_config.robot.joint2motor,
    )


def _recurrent_lstm_contract(hidden_dim: int = 256, num_layers: int = 1) -> tuple[dict, str]:
    contract = {
        "version": 1,
        "kind": "lstm",
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "dtype": "float32",
        "state_input_names": ["hidden_state", "cell_state"],
        "state_output_names": ["hidden_state_out", "cell_state_out"],
        "state_shape": [num_layers, "batch", hidden_dim],
        "state_batch_axis": 1,
        "step_semantics": "state_before_observation_to_state_after_observation",
        "reset_semantics": "zero_after_done_before_next_observation",
        "deployment_reset_events": [
            "episode_reset",
            "policy_start",
            "policy_stop",
            "policy_switch",
        ],
    }
    payload = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return contract, hashlib.sha256(payload).hexdigest()


def test_full_policy_lstm_contract_validates_explicit_state_graph() -> None:
    base = g1_29dof_wbt_w_object_generalist
    actor = base.algo.config.module_dict.actor
    lstm_actor = replace(
        actor,
        type="LSTM",
        layer_config=replace(
            actor.layer_config,
            lstm_hidden_dim=256,
            lstm_num_layers=1,
        ),
    )
    training_config = replace(
        base,
        algo=replace(
            base.algo,
            config=replace(
                base.algo.config,
                module_dict=replace(base.algo.config.module_dict, actor=lstm_actor),
            ),
        ),
    )
    metadata = _export_metadata_from_training_config(training_config)
    contract, digest = _recurrent_lstm_contract()
    metadata["recurrent_policy_contract"] = contract
    metadata["recurrent_policy_contract_sha256"] = digest

    assert validate_onnx_policy_contract(
        metadata=metadata,
        input_shapes={
            "actor_obs": ["batch", 875],
            "hidden_state": [1, "batch", 256],
            "cell_state": [1, "batch", 256],
        },
        output_shapes={
            "action": ["batch", 29],
            "hidden_state_out": [1, "batch", 256],
            "cell_state_out": [1, "batch", 256],
        },
        observation=g1_29dof_wbt_object_generalist.observation,
        runtime_dof_names=g1_29dof_wbt_object_generalist.robot.dof_names,
        runtime_default_dof_angles=g1_29dof_wbt_object_generalist.robot.default_dof_angles,
        runtime_motor_effort_limits=g1_29dof_wbt_object_generalist.robot.motor_effort_limit,
        runtime_joint2motor=g1_29dof_wbt_object_generalist.robot.joint2motor,
    )

    with pytest.raises(PolicyContractError, match="static shape"):
        validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={
                "actor_obs": ["batch", 875],
                "hidden_state": [1, "batch", 128],
                "cell_state": [1, "batch", 256],
            },
            output_shapes={
                "action": ["batch", 29],
                "hidden_state_out": [1, "batch", 256],
                "cell_state_out": [1, "batch", 256],
            },
            observation=g1_29dof_wbt_object_generalist.observation,
            runtime_dof_names=g1_29dof_wbt_object_generalist.robot.dof_names,
            runtime_default_dof_angles=g1_29dof_wbt_object_generalist.robot.default_dof_angles,
        )


@pytest.mark.parametrize(
    ("training_config", "wrong_runtime"),
    [
        (g1_29dof_wbt_w_object_generalist, g1_29dof_wbt_w_object_legacy),
        (g1_29dof_wbt_w_object_generalist_legacy_obs, g1_29dof_wbt_object_generalist),
    ],
)
def test_same_shape_current_legacy_cross_selection_is_rejected(training_config, wrong_runtime) -> None:
    with pytest.raises(PolicyContractError, match="terms for actor group"):
        validate_onnx_policy_contract(
            metadata=_export_metadata_from_training_config(training_config),
            input_shapes={"actor_obs": [1, 875]},
            output_shapes={"action": [1, 29]},
            observation=wrong_runtime.observation,
            runtime_dof_names=wrong_runtime.robot.dof_names,
            runtime_default_dof_angles=wrong_runtime.robot.default_dof_angles,
            runtime_motor_effort_limits=wrong_runtime.robot.motor_effort_limit,
            runtime_joint2motor=wrong_runtime.robot.joint2motor,
        )


def test_current_and_legacy_object_contracts_are_same_shape_but_semantically_distinct() -> None:
    current = observation_values.wbt_w_object
    legacy = observation_values.wbt_w_object_legacy
    current_dim = sum(current.obs_dims[name] for name in current.obs_dict["actor_obs"])
    legacy_dim = sum(legacy.obs_dims[name] for name in legacy.obs_dict["actor_obs"])
    assert current_dim == legacy_dim == 175
    assert current_dim * current.history_length_dict["actor_obs"] == 875
    assert legacy_dim * legacy.history_length_dict["actor_obs"] == 875
    assert set(current.obs_dict["actor_obs"]) != set(legacy.obs_dict["actor_obs"])


def test_velocity_generalist_contract_is_explicit_v2_history_one_layout() -> None:
    runtime = observation_values.wbt_object_velocity_generalist
    expected_terms = dict(training_observation_values.actor_obs_w_object_legacy_terms)
    expected_terms["obj_lin_vel_b"] = training_observation_values.critic_obs_w_object_terms[
        "obj_lin_vel_b"
    ]
    expected_terms["obj_ang_vel_b"] = (
        training_observation_values.critic_obs_w_object_command_privileged_terms["obj_ang_vel_b"]
    )
    assert sorted(runtime.obs_dict["actor_obs"]) == sorted(expected_terms)
    assert runtime.history_length_dict["actor_obs"] == 1
    assert sum(runtime.obs_dims[name] for name in runtime.obs_dict["actor_obs"]) == 181
    for term_name, training_term in expected_terms.items():
        descriptor = runtime.term_descriptors[term_name]
        assert descriptor.func == training_term.func
        assert descriptor.params == training_term.params
        assert descriptor.noise == training_term.noise
        assert descriptor.clip == training_term.clip
        assert runtime.obs_scales[term_name] == training_term.scale


@pytest.mark.parametrize(
    ("cfg", "obs_dim"),
    [
        (g1_29dof_wbt_object_generalist, 875),
        (g1_29dof_wbt_w_object_legacy, 875),
        (g1_29dof_wbt_object_velocity_generalist, 181),
    ],
)
def test_object_artifact_without_metadata_fails_closed(cfg, obs_dim) -> None:
    with pytest.raises(PolicyContractError, match="semantically ambiguous"):
        validate_onnx_policy_contract(
            metadata={},
            input_shapes={"actor_obs": [1, obs_dim]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


def test_object_observation_numeric_layout_matches_sorted_training_terms() -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)
    identity_wxyz = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    half_sqrt = np.float32(np.sqrt(0.5))
    target_quat_wxyz = np.array([[half_sqrt, 0.0, 0.0, half_sqrt]], dtype=np.float32)
    current_quat_xyzw = np.array([[half_sqrt, 0.0, 0.0, half_sqrt]], dtype=np.float32)
    target_pos = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
    current_pos = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    object_size = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
    target_rot6d = np.array([[0.0, -1.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    current_rot6d = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    current_lin_vel = np.array([[7.0, 8.0, 9.0]], dtype=np.float32)
    current_ang_vel = np.array([[10.0, 11.0, 12.0]], dtype=np.float32)

    policy._motion_data = SimpleNamespace(
        has_object=True,
        object_pos_w=target_pos,
        object_quat_w=target_quat_wxyz,
        object_size=object_size,
    )
    policy._motion_align_quat_wxyz = None
    policy._motion_align_pos = None
    policy._maybe_update_motion_alignment = lambda _: None
    policy._get_motion_index = lambda: 0
    policy._get_observation_reference_pose_in_world = lambda _: (
        np.zeros((1, 3), dtype=np.float32),
        identity_wxyz,
    )
    policy.config = SimpleNamespace(task=SimpleNamespace(sim_object_name="object"))
    policy._get_sim_actor_state = lambda _: np.concatenate(
        [
            current_pos,
            current_quat_xyzw,
            current_lin_vel,
            current_ang_vel,
        ],
        axis=1,
    )
    policy.num_dofs = 29
    policy.default_dof_angles = np.zeros((1, 29), dtype=np.float32)
    policy.motion_command_t = np.arange(58, dtype=np.float32).reshape(1, -1) + 100.0
    policy.last_policy_action = np.arange(29, dtype=np.float32).reshape(1, -1) + 200.0
    policy._get_motion_ref_ori_b = lambda _: np.arange(6, dtype=np.float32).reshape(1, -1) + 300.0
    policy._get_base_ang_vel_obs = lambda _: np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    robot_state = np.zeros((1, 7 + 29 + 6 + 29), dtype=np.float32)
    robot_state[:, 7 : 7 + 29] = np.arange(29, dtype=np.float32) + 400.0
    robot_state[:, 7 + 29 + 6 :] = np.arange(29, dtype=np.float32) + 500.0
    buffer = policy._get_object_generalist_obs_buffer_dict(robot_state)

    expected = {
        "actions": policy.last_policy_action,
        "base_ang_vel": np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        "dof_pos": robot_state[:, 7 : 7 + 29],
        "dof_vel": robot_state[:, 7 + 29 + 6 :],
        "motion_command": policy.motion_command_t,
        "motion_ref_ori_b": np.arange(6, dtype=np.float32).reshape(1, -1) + 300.0,
        "obj_ori_b": current_rot6d,
        "obj_pos_b": current_pos,
        "obj_size": object_size,
        "obj_target_ori_b": target_rot6d,
        "obj_target_pos_b": target_pos,
        "obj_target_pose_size_b": np.concatenate([target_pos, target_rot6d, object_size], axis=1),
        "obj_lin_vel_b": current_lin_vel,
        "obj_ang_vel_b": current_ang_vel,
    }
    for name, value in expected.items():
        np.testing.assert_allclose(buffer[name], value, rtol=0.0, atol=1.0e-6)

    current_order = sorted(g1_29dof_wbt_observation_w_object.groups["actor_obs"].terms)
    legacy_order = sorted(g1_29dof_wbt_observation_w_object_legacy.groups["actor_obs"].terms)
    current_frame = np.concatenate([buffer[name] for name in current_order], axis=1)
    legacy_frame = np.concatenate([buffer[name] for name in legacy_order], axis=1)
    expected_current = np.concatenate([expected[name] for name in current_order], axis=1)
    expected_legacy = np.concatenate([expected[name] for name in legacy_order], axis=1)
    np.testing.assert_array_equal(current_frame, expected_current)
    np.testing.assert_array_equal(legacy_frame, expected_legacy)
    assert current_frame.shape == legacy_frame.shape == (1, 175)
    assert not np.array_equal(current_frame, legacy_frame)


def test_object_observation_rejects_missing_current_object_state() -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._motion_data = SimpleNamespace(
        has_object=True,
        object_pos_w=np.zeros((1, 3), dtype=np.float32),
        object_quat_w=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        object_size=np.ones((1, 3), dtype=np.float32),
    )
    policy._maybe_update_motion_alignment = lambda _: None
    policy._get_motion_index = lambda: 0
    policy._get_observation_reference_pose_in_world = lambda _: (
        np.zeros((1, 3), dtype=np.float32),
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    policy._motion_align_quat_wxyz = None
    policy.config = SimpleNamespace(task=SimpleNamespace(sim_object_name="object"))
    policy._get_sim_actor_state = lambda _: None

    with pytest.raises(RuntimeError, match="require a valid current object state"):
        policy._get_object_generalist_obs_buffer_dict(np.zeros((1, 71), dtype=np.float32))


def test_legacy_onnx_without_metadata_rejects_95_to_94_contract_mismatch() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    with pytest.raises(PolicyContractError, match="Refusing to pad or truncate"):
        validate_onnx_policy_contract(
            metadata={},
            input_shapes={"actor_obs": [1, 95], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


def test_legacy_student_onnx_without_metadata_rejects_shape_only_dual_contract() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_dual_button_depth_distill
    with pytest.raises(PolicyContractError, match="require complete experiment_config metadata"):
        validate_onnx_policy_contract(
            metadata={},
            input_shapes={"actor_obs": [1, 95], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


def test_drop_button_onnx_contract_rejects_same_terms_in_wrong_group_order() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    reversed_groups = dict(reversed(list(cfg.observation.obs_dict.items())))
    wrong_observation = ObservationConfig(
        obs_dict=reversed_groups,
        obs_dims=dict(cfg.observation.obs_dims),
        obs_scales=dict(cfg.observation.obs_scales),
        history_length_dict=dict(cfg.observation.history_length_dict),
    )
    with pytest.raises(PolicyContractError, match="group order"):
        validate_onnx_policy_contract(
            metadata=_drop_button_metadata(),
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=wrong_observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


def test_drop_button_onnx_contract_rejects_shifted_default_joint_angles() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    shifted_defaults = list(cfg.robot.default_dof_angles)
    shifted_defaults[0] += 0.1
    with pytest.raises(PolicyContractError, match="default joint angles"):
        validate_onnx_policy_contract(
            metadata=_drop_button_metadata(),
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=shifted_defaults,
        )


def test_policy_contract_requires_dof_names_in_complete_metadata() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _drop_button_metadata()
    metadata.pop("dof_names")
    with pytest.raises(PolicyContractError, match="dof_names"):
        validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


def test_policy_contract_rejects_duplicate_metadata_dof_names() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _drop_button_metadata()
    metadata["dof_names"][1] = metadata["dof_names"][0]
    with pytest.raises(PolicyContractError, match="must be unique"):
        validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


def test_policy_contract_requires_complete_default_joint_angle_mapping() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _drop_button_metadata()
    missing_name = metadata["dof_names"][0]
    del metadata["experiment_config"]["robot"]["init_state"]["default_joint_angles"][missing_name]
    with pytest.raises(PolicyContractError, match="mapping is incomplete"):
        validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


def test_policy_contract_rejects_nonfinite_default_joint_angles() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _drop_button_metadata()
    first_name = metadata["dof_names"][0]
    metadata["experiment_config"]["robot"]["init_state"]["default_joint_angles"][first_name] = float("nan")
    with pytest.raises(PolicyContractError, match="must all be finite"):
        validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


@pytest.mark.parametrize(("key", "value"), [("kp", None), ("kd", [1.0, -1.0] + [1.0] * 27)])
def test_policy_contract_requires_valid_dof_ordered_pd_gains(key, value) -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _drop_button_metadata()
    if value is None:
        metadata.pop(key)
    else:
        metadata[key] = value

    with pytest.raises(PolicyContractError, match=key.upper()):
        validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


def test_policy_contract_rejects_global_observation_clip_drift() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    wrong_observation = replace(cfg.observation, clip_observations=1.0)
    with pytest.raises(PolicyContractError, match="global observation clip"):
        validate_onnx_policy_contract(
            metadata=_drop_button_metadata(),
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=wrong_observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


def test_policy_contract_requires_named_action_output() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    with pytest.raises(PolicyContractError, match="no supported action output"):
        validate_onnx_policy_contract(
            metadata=_drop_button_metadata(),
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"latent": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


def test_policy_contract_rejects_ambiguous_action_outputs() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    with pytest.raises(PolicyContractError, match="exactly one supported action output"):
        validate_onnx_policy_contract(
            metadata=_drop_button_metadata(),
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29], "actions": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("control_type", "V"), "position-control"),
        (("clip_actions", False), "clip_actions=true"),
        (("action_clip_value", float("nan")), "finite and > 0"),
        (("action_scale", -0.1), "finite and > 0"),
    ],
)
def test_policy_contract_rejects_unsupported_control_semantics(mutation, message) -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _drop_button_metadata()
    key, value = mutation
    metadata["experiment_config"]["robot"]["control"][key] = value

    with pytest.raises(PolicyContractError, match=message):
        validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


def test_policy_contract_rejects_non_position_action_term() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _drop_button_metadata()
    metadata["experiment_config"]["action"]["terms"]["joint_control"]["func"] = "pkg:TorqueActionTerm"

    with pytest.raises(PolicyContractError, match="JointPositionActionTerm"):
        validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "func",
            "different.physics:JointPositionActionTerm",
            "exact action implementation",
        ),
        ("params", {"stiffness": 1.0}, "explicit empty params"),
    ],
)
def test_policy_contract_rejects_lookalike_or_parameterized_action_term(
    field: str,
    value: object,
    message: str,
) -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _drop_button_metadata()
    metadata["experiment_config"]["action"]["terms"]["joint_control"][field] = value

    with pytest.raises(PolicyContractError, match=message):
        validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


def _per_joint_scale_metadata() -> dict:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _drop_button_metadata()
    metadata["kp"] = [10.0] * len(cfg.robot.dof_names)
    robot = metadata["experiment_config"]["robot"]
    robot["dof_effort_limit_list"] = [float(value) for value in cfg.robot.motor_effort_limit]
    robot["control"] = {
        "control_type": "P",
        "action_scale": 0.25,
        "action_clip_value": 100.0,
        "clip_actions": True,
        "action_scales_by_effort_limit_over_p_gain": True,
    }
    return metadata


def test_policy_contract_rejects_runtime_effort_drift_for_per_joint_scaling() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    runtime_effort = list(cfg.robot.motor_effort_limit)
    runtime_effort[0] += 1.0
    with pytest.raises(PolicyContractError, match="change the training action-scale contract"):
        validate_onnx_policy_contract(
            metadata=_per_joint_scale_metadata(),
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
            runtime_motor_effort_limits=runtime_effort,
            runtime_joint2motor=cfg.robot.joint2motor,
        )


def test_policy_contract_accepts_exact_runtime_effort_mapping() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    assert validate_onnx_policy_contract(
        metadata=_per_joint_scale_metadata(),
        input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
        output_shapes={"action": [1, 29]},
        observation=cfg.observation,
        runtime_dof_names=cfg.robot.dof_names,
        runtime_default_dof_angles=cfg.robot.default_dof_angles,
        runtime_motor_effort_limits=cfg.robot.motor_effort_limit,
        runtime_joint2motor=cfg.robot.joint2motor,
    )


def test_inference_observation_history_applies_training_global_clip() -> None:
    policy = object.__new__(BasePolicy)
    policy.obs_config = ObservationConfig(
        obs_dict={"actor_obs": ["signal"]},
        obs_dims={"signal": 1},
        obs_scales={"signal": 2.0},
        history_length_dict={"actor_obs": 1},
        clip_observations=3.0,
    )
    policy.obs_dict = policy.obs_config.obs_dict
    policy.obs_dims = policy.obs_config.obs_dims
    policy.history_length_dict = policy.obs_config.history_length_dict
    policy.observation_clip = policy.obs_config.clip_observations
    policy._initialize_history_state()

    # `_update_obs_history` receives post-scale terms from
    # `parse_current_obs_dict`; 2.0 * scale 2.0 is therefore 4.0 here.
    result = policy._update_obs_history({"actor_obs": {"signal": np.array([[4.0]], dtype=np.float32)}})

    assert result["actor_obs"].tolist() == [[3.0]]


def test_onnx_feed_clips_actor_and_perception_but_not_time_index() -> None:
    policy = object.__new__(BasePolicy)
    policy.obs_dict = {"actor_obs_root": ["signal"]}
    policy.observation_clip = 3.0
    policy._obs_input_name = "actor_obs"
    policy._perception_obs_input_name = "depth_features"
    policy._onnx_metadata = {
        "experiment_config": {
            "algo": {
                "config": {
                    "module_dict": {
                        "actor": {"layer_config": {"perception_input_name": "depth_features"}}
                    }
                }
            }
        }
    }

    result = policy._prepare_policy_input_feed(
        {
            "actor_obs": np.array([[4.0, -5.0]], dtype=np.float32),
            "depth_features": np.array([[9.0]], dtype=np.float32),
            "time_step": np.array([[1000.0]], dtype=np.float32),
        }
    )

    assert result["actor_obs"].tolist() == [[3.0, -3.0]]
    assert result["depth_features"].tolist() == [[3.0]]
    assert result["time_step"].tolist() == [[1000.0]]


@pytest.mark.parametrize(
    ("input_name", "bad_value"),
    [
        ("actor_obs", float("nan")),
        ("depth_features", float("inf")),
        ("time_step", float("-inf")),
    ],
)
def test_onnx_feed_rejects_nonfinite_values(input_name: str, bad_value: float) -> None:
    policy = object.__new__(BasePolicy)
    policy.obs_dict = {"actor_obs_root": ["signal"]}
    policy.observation_clip = 3.0
    policy._obs_input_name = "actor_obs"
    policy._perception_obs_input_name = "depth_features"
    policy._onnx_metadata = {}
    feed = {
        "actor_obs": np.zeros((1, 2), dtype=np.float32),
        "depth_features": np.zeros((1, 1), dtype=np.float32),
        "time_step": np.zeros((1, 1), dtype=np.float32),
    }
    feed[input_name][0, 0] = bad_value

    with pytest.raises(FloatingPointError, match="non-finite"):
        policy._prepare_policy_input_feed(feed)


def test_nonfinite_observation_term_is_rejected_before_history_is_poisoned() -> None:
    policy = object.__new__(BasePolicy)
    policy.obs_config = ObservationConfig(
        obs_dict={"actor_obs": ["signal"]},
        obs_dims={"signal": 1},
        obs_scales={"signal": 1.0},
        history_length_dict={"actor_obs": 2},
    )
    policy.obs_dict = policy.obs_config.obs_dict
    policy.obs_dims = policy.obs_config.obs_dims
    policy.history_length_dict = policy.obs_config.history_length_dict
    policy.observation_clip = policy.obs_config.clip_observations
    policy._initialize_history_state()

    with pytest.raises(FloatingPointError, match="actor_obs.signal"):
        policy._update_obs_history(
            {"actor_obs": {"signal": np.array([[float("nan")]], dtype=np.float32)}}
        )

    assert len(policy.obs_history_buffers["actor_obs"]["signal"]) == 0


def test_robot_state_validation_rejects_nonfinite_consumed_state() -> None:
    policy = object.__new__(BasePolicy)
    policy.num_dofs = 2
    state = np.zeros((1, 7 + 2 + 6 + 2), dtype=np.float32)
    state[0, 3] = 1.0
    state[0, 7] = 0.1
    assert policy._has_valid_robot_state(state)

    for bad_index in (3, 7, state.shape[1] - 1):
        bad_state = state.copy()
        bad_state[0, bad_index] = float("nan")
        assert not policy._has_valid_robot_state(bad_state)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_policy_action_output_rejects_nonfinite_before_scaling(bad_value: float) -> None:
    policy = object.__new__(BasePolicy)
    policy.num_dofs = 1
    policy.prepare_obs_for_rl = lambda _state: {"actor_obs": np.zeros((1, 1), dtype=np.float32)}
    policy.policy = lambda _obs: np.array([[bad_value]], dtype=np.float32)
    policy.policy_action_clip = 1.0
    policy.policy_action_scales = np.ones((1, 1), dtype=np.float32)

    with pytest.raises(FloatingPointError, match="policy action output"):
        policy.rl_inference(np.zeros((1, 1), dtype=np.float32))


def test_policy_action_history_keeps_raw_output_while_control_uses_action_clip() -> None:
    policy = object.__new__(BasePolicy)
    policy.num_dofs = 2
    policy.prepare_obs_for_rl = lambda _state: {"actor_obs": np.zeros((1, 1), dtype=np.float32)}
    policy.policy = lambda _obs: np.array([[2.0, -3.0]], dtype=np.float32)
    policy.policy_action_clip = 1.0
    policy.policy_action_scales = np.array([[0.5, 2.0]], dtype=np.float32)

    control_action = policy.rl_inference(np.zeros((1, 1), dtype=np.float32))

    # Match training: ActionManager.action is raw, while JointPositionActionTerm
    # clips its separate processed action before applying action scales.
    np.testing.assert_allclose(policy.last_policy_action, [[2.0, -3.0]])
    np.testing.assert_allclose(control_action, [[0.5, -2.0]])


def test_policy_action_output_rejects_broadcastable_wrong_shape_before_control() -> None:
    policy = object.__new__(BasePolicy)
    policy.num_dofs = 2
    policy.prepare_obs_for_rl = lambda _state: {"actor_obs": np.zeros((1, 1), dtype=np.float32)}
    policy.policy = lambda _obs: np.array([0.5], dtype=np.float32)
    policy.policy_action_clip = 1.0
    policy.policy_action_scales = np.array([[1.0, 2.0]], dtype=np.float32)
    policy.last_policy_action = np.array([[7.0, 8.0]], dtype=np.float32)
    policy.scaled_policy_action = np.array([[9.0, 10.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="refusing NumPy broadcasting"):
        policy.rl_inference(np.zeros((1, 1), dtype=np.float32))

    np.testing.assert_allclose(policy.last_policy_action, [[7.0, 8.0]])
    np.testing.assert_allclose(policy.scaled_policy_action, [[9.0, 10.0]])


def test_policy_action_scale_failure_does_not_partially_advance_action_history() -> None:
    policy = object.__new__(BasePolicy)
    policy.num_dofs = 2
    policy.policy_action_clip = 1.0
    policy.policy_action_scales = np.array([[1.0]], dtype=np.float32)
    policy.last_policy_action = np.array([[7.0, 8.0]], dtype=np.float32)
    policy.scaled_policy_action = np.array([[9.0, 10.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="Policy action scales must have shape"):
        policy._update_policy_action_state(
            np.array([[0.25, -0.5]], dtype=np.float32),
            label="policy action output",
        )

    np.testing.assert_allclose(policy.last_policy_action, [[7.0, 8.0]])
    np.testing.assert_allclose(policy.scaled_policy_action, [[9.0, 10.0]])


def test_wbt_sim_state_getters_reject_nonfinite_actor_state() -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._sim_state_sub = None
    actor_state = np.zeros(13, dtype=np.float32)
    actor_state[7] = float("nan")
    policy._latest_sim_state = {"actors": {"object": actor_state}}

    with pytest.raises(FloatingPointError, match="simulator actor 'object' state"):
        policy._get_sim_actor_state("object")


def test_dof_ordered_onnx_gains_are_mapped_to_motor_order() -> None:
    policy = object.__new__(BasePolicy)
    policy.num_dofs = 2
    policy.robot_config = SimpleNamespace(num_motors=2, joint2motor=(1, 0))

    motor_values = policy._joint_values_to_motor_order([10.0, 20.0], "KP")

    assert motor_values.tolist() == [20.0, 10.0]


class _RecordingPolicySession:
    def __init__(self, marker: float):
        self.marker = float(marker)
        self.calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def run(self, output_names, input_feed):
        self.calls.append((tuple(output_names), tuple(input_feed)))
        return [
            np.full((1, 2), self.marker + output_index, dtype=np.float32)
            for output_index, _ in enumerate(output_names)
        ]


def _capture_wbt_policy_slot(
    policy: WholeBodyTrackingPolicy,
    *,
    slot: int,
    obs_input_name: str,
    action_output_name: str,
    output_fetch: list[str],
) -> tuple[dict, _RecordingPolicySession]:
    session = _RecordingPolicySession(float(slot * 10))
    policy.onnx_policy_session = session
    policy.onnx_input_names = [obs_input_name, "time_step", "perception_obs"]
    policy.onnx_output_names = list(output_fetch)
    policy._obs_input_name = obs_input_name
    policy._time_step_input_name = "time_step" if slot == 1 else None
    policy._perception_obs_input_name = "perception_obs" if slot == 1 else None
    policy._action_output_name = action_output_name
    policy._onnx_output_fetch = list(output_fetch)
    policy._motion_output_names = set(output_fetch) - {action_output_name}
    policy._onnx_obs_dim = 90 + slot
    policy._has_policy_contract = True
    policy._onnx_metadata = {"slot": slot}
    policy._perception_contract_sha256 = "11" * 32
    policy.onnx_kp = np.array([slot + 1.0, slot + 2.0], dtype=np.float32)
    policy.onnx_kd = np.array([slot + 0.1, slot + 0.2], dtype=np.float32)
    policy.policy_action_scale = slot / 4.0
    policy.policy_action_clip = 10.0 - slot
    policy.policy_action_scales = np.array([[slot / 10.0, slot / 5.0]], dtype=np.float32)
    policy.pinocchio_robot = SimpleNamespace(slot=slot)
    policy._motion_data = SimpleNamespace(slot=slot)
    policy._motion_cfg = {"slot": slot}
    policy._motion_body_names = (f"body_{slot}",)
    policy._motion_transition_prepend_steps = slot
    policy._contact_aware_carry_window = (slot, slot + 10)
    policy._contact_aware_button_window = (slot + 1, slot + 11)
    policy._training_freeze_zero_prob = slot / 10.0
    policy._training_freeze_zero_extra_holds = slot + 2
    policy._motion_alignment_enabled = bool(slot % 2)
    policy.motion_command_0 = np.full((1, 4), slot, dtype=np.float32)
    policy.ref_quat_xyzw_0 = np.full((1, 4), slot + 0.5, dtype=np.float32)
    policy.ref_pos_xyz_t = np.full((1, 3), slot + 1.5, dtype=np.float32)

    def policy_callable(input_feed):
        prepared_feed = policy._prepare_policy_input_feed(input_feed)
        outputs = policy.onnx_policy_session.run(policy._onnx_output_fetch, prepared_feed)
        return dict(zip(policy._onnx_output_fetch, outputs))

    policy.policy = policy_callable
    return policy._capture_policy_state(), session


def test_wbt_multi_policy_switch_restores_complete_slot_state() -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._prepare_policy_input_feed = lambda input_feed: dict(input_feed)
    policy._reset_observation_history_state = lambda: None
    policy._on_policy_switched = lambda _model_path: None
    policy.last_policy_action = np.ones((1, 2), dtype=np.float32)
    policy.scaled_policy_action = np.ones((1, 2), dtype=np.float32)

    first_state, first_session = _capture_wbt_policy_slot(
        policy,
        slot=1,
        obs_input_name="obs",
        action_output_name="action",
        output_fetch=["action", "joint_pos"],
    )
    second_state, second_session = _capture_wbt_policy_slot(
        policy,
        slot=2,
        obs_input_name="actor_obs",
        action_output_name="actions",
        output_fetch=["actions", "joint_vel", "ref_quat_xyzw"],
    )
    policy.model_paths = ["first.onnx", "second.onnx"]
    policy._policy_states = [first_state, second_state]

    policy._activate_policy(0, announce=False)
    first_outputs = policy.policy({policy._obs_input_name: np.ones((1, 91), dtype=np.float32)})
    assert policy.onnx_policy_session is first_session
    assert policy._obs_input_name == "obs"
    assert policy._time_step_input_name == "time_step"
    assert policy._perception_obs_input_name == "perception_obs"
    assert policy._action_output_name == "action"
    assert policy._onnx_output_fetch == ["action", "joint_pos"]
    assert tuple(first_outputs) == ("action", "joint_pos")
    assert first_session.calls == [(('action', 'joint_pos'), ('obs',))]
    assert policy.policy_action_scale == pytest.approx(0.25)
    assert policy.policy_action_clip == pytest.approx(9.0)
    np.testing.assert_allclose(policy.policy_action_scales, [[0.1, 0.2]])
    np.testing.assert_allclose(policy.onnx_kp, [2.0, 3.0])
    np.testing.assert_allclose(policy.onnx_kd, [1.1, 1.2])
    assert policy._motion_data.slot == 1
    assert policy._motion_cfg == {"slot": 1}
    assert policy._motion_body_names == ("body_1",)
    assert policy._contact_aware_carry_window == (1, 11)
    assert policy._contact_aware_button_window == (2, 12)
    assert policy._perception_contract_sha256 == "11" * 32
    np.testing.assert_allclose(policy.motion_command_0, np.full((1, 4), 1.0))
    np.testing.assert_allclose(policy.ref_quat_xyzw_0, np.full((1, 4), 1.5))
    np.testing.assert_allclose(policy.ref_pos_xyz_t, np.full((1, 3), 2.5))
    assert not np.any(policy.last_policy_action)
    assert not np.any(policy.scaled_policy_action)

    policy._activate_policy(1, announce=False)
    second_outputs = policy.policy({policy._obs_input_name: np.ones((1, 92), dtype=np.float32)})
    assert policy.onnx_policy_session is second_session
    assert policy._obs_input_name == "actor_obs"
    assert policy._time_step_input_name is None
    assert policy._perception_obs_input_name is None
    assert policy._action_output_name == "actions"
    assert policy._onnx_output_fetch == ["actions", "joint_vel", "ref_quat_xyzw"]
    assert tuple(second_outputs) == ("actions", "joint_vel", "ref_quat_xyzw")
    assert second_session.calls == [
        (("actions", "joint_vel", "ref_quat_xyzw"), ("actor_obs",))
    ]
    assert policy.policy_action_scale == pytest.approx(0.5)
    assert policy.policy_action_clip == pytest.approx(8.0)
    np.testing.assert_allclose(policy.policy_action_scales, [[0.2, 0.4]])
    np.testing.assert_allclose(policy.onnx_kp, [3.0, 4.0])
    np.testing.assert_allclose(policy.onnx_kd, [2.1, 2.2])
    assert policy._motion_data.slot == 2
    assert policy._motion_cfg == {"slot": 2}
    assert policy._motion_body_names == ("body_2",)
    assert policy._contact_aware_carry_window == (2, 12)
    assert policy._contact_aware_button_window == (3, 13)
    np.testing.assert_allclose(policy.motion_command_0, np.full((1, 4), 2.0))
    np.testing.assert_allclose(policy.ref_quat_xyzw_0, np.full((1, 4), 2.5))
    np.testing.assert_allclose(policy.ref_pos_xyz_t, np.full((1, 3), 3.5))

    policy._activate_policy(0, announce=False)
    assert policy.onnx_policy_session is first_session
    assert policy._onnx_output_fetch == ["action", "joint_pos"]
    assert policy._motion_data.slot == 1


def test_wbt_multi_policy_rejects_any_legacy_contract_slot() -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)

    with pytest.raises(ValueError, match="complete serialized contracts"):
        policy._validate_policy_state_collection(
            [
                {"has_policy_contract": True},
                {"has_policy_contract": False},
            ]
        )


def test_policy_contract_rejects_non_boolean_contact_timebase_version() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _drop_button_metadata()
    metadata["experiment_config"]["command"] = {
        "setup_terms": {
            "motion_command": {
                "params": {
                    "motion_config": {
                        "contact_interval_runtime_prepend_compensation": "true",
                    }
                }
            }
        }
    }
    with pytest.raises(PolicyContractError, match="must be boolean"):
        validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("contact_aware_button_window_mode", "typo", "must be exactly"),
        ("contact_aware_button_window_mode", 1, "must be exactly"),
        ("contact_aware_carry_window_mode", "typo", "must be exactly"),
        ("contact_aware_carry_window_mode", 1, "must be exactly"),
        ("contact_aware_peak_height_alpha", True, "finite real number"),
        ("contact_aware_peak_height_alpha", "0.91", "finite real number"),
        ("contact_aware_peak_height_alpha", float("nan"), "finite real number"),
        ("contact_aware_peak_height_alpha", float("inf"), "finite real number"),
        ("contact_aware_peak_height_alpha", -0.01, "finite real number"),
        ("contact_aware_peak_height_alpha", 1.01, "finite real number"),
        ("contact_aware_peak_height_smoothing_steps", True, "integer in"),
        ("contact_aware_peak_height_smoothing_steps", 5.0, "integer in"),
        ("contact_aware_peak_height_smoothing_steps", "5", "integer in"),
        ("contact_aware_peak_height_smoothing_steps", 0, "integer in"),
        ("contact_aware_peak_height_smoothing_steps", 4097, "integer in"),
    ],
)
def test_policy_contract_rejects_invalid_contact_aware_carry_window_metadata(
    field: str,
    value: object,
    message: str,
) -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _drop_button_metadata()
    metadata["experiment_config"]["command"] = {
        "setup_terms": {
            "motion_command": {
                "params": {
                    "motion_config": {
                        "contact_aware_carry_window_mode": "peak_height",
                        field: value,
                    }
                }
            }
        }
    }

    with pytest.raises(PolicyContractError, match=message):
        validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


@pytest.mark.parametrize("alpha", [0.0, 1.0])
@pytest.mark.parametrize("smoothing_steps", [1, 4096])
def test_policy_contract_accepts_contact_aware_carry_window_boundaries(
    alpha: float,
    smoothing_steps: int,
) -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _drop_button_metadata()
    metadata["experiment_config"]["command"] = {
        "setup_terms": {
            "motion_command": {
                "params": {
                    "motion_config": {
                        "contact_aware_carry_window_mode": "peak_height",
                        "contact_aware_peak_height_alpha": alpha,
                        "contact_aware_peak_height_smoothing_steps": smoothing_steps,
                    }
                }
            }
        }
    }

    assert validate_onnx_policy_contract(
        metadata=metadata,
        input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
        output_shapes={"action": [1, 29]},
        observation=cfg.observation,
        runtime_dof_names=cfg.robot.dof_names,
        runtime_default_dof_angles=cfg.robot.default_dof_angles,
    )


@pytest.mark.parametrize(
    "motion_cfg",
    [
        {"contact_aware_carry_window_mode": "typo"},
        {
            "contact_aware_carry_window_mode": "peak_height",
            "contact_aware_peak_height_alpha": float("nan"),
        },
        {
            "contact_aware_carry_window_mode": "peak_height",
            "contact_aware_peak_height_smoothing_steps": 1_000_000_000,
        },
    ],
)
def test_carry_window_consumer_rejects_invalid_metadata_after_contract_bypass(
    motion_cfg: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _validated_contact_aware_carry_window_config(motion_cfg)


def test_motion_transition_contract_rejects_missing_or_digest_drift() -> None:
    with pytest.raises(PolicyContractError, match="missing motion_transition_contract"):
        motion_transition_contract_from_metadata({}, required=True)

    metadata = _motion_transition_contract_metadata(
        source_semantics="global_multi_clip_runtime",
        prepend_implementation="runtime_hold",
        prepend_steps=2,
        append_implementation="none",
        append_steps=0,
    )
    metadata["motion_transition_contract"]["prepend"]["steps"] = 3
    with pytest.raises(PolicyContractError, match="SHA-256 does not match"):
        motion_transition_contract_from_metadata(metadata, required=True)


def test_training_and_inference_motion_transition_contract_digest_schemas_match() -> None:
    metadata = _motion_transition_contract_metadata(
        source_semantics="global_multi_clip_runtime",
        prepend_implementation="runtime_hold",
        prepend_steps=2,
        append_implementation="none",
        append_steps=0,
    )
    contract = metadata["motion_transition_contract"]
    assert metadata["motion_transition_contract_sha256"] == (
        training_motion_transition_contract_sha256(contract)
    )
    assert motion_transition_contract_from_metadata(metadata, required=True) == contract


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda contract: contract["append"].update(
                {"implementation": "static_splice", "applied": True, "steps": 2}
            ),
            "Global multi-clip training",
        ),
        (
            lambda contract: contract["prepend"].update(
                {"implementation": "runtime_hold", "applied": True, "steps": 4097}
            ),
            "steps in",
        ),
        (
            lambda contract: contract["prepend"].update(
                {"implementation": "runtime_hold", "applied": True, "steps": 1}
            ),
            "steps in",
        ),
        (
            lambda contract: contract["prepend"].update(
                {"implementation": "none", "applied": False, "steps": 1}
            ),
            "Inactive motion transition",
        ),
    ],
)
def test_motion_transition_contract_rejects_invalid_effective_semantics(
    mutation,
    message: str,
) -> None:
    metadata = _motion_transition_contract_metadata(
        source_semantics="global_multi_clip_runtime",
        prepend_implementation="runtime_hold",
        prepend_steps=2,
        append_implementation="none",
        append_steps=0,
    )
    mutation(metadata["motion_transition_contract"])
    contract = metadata["motion_transition_contract"]
    metadata["motion_transition_contract_sha256"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    with pytest.raises(PolicyContractError, match=message):
        motion_transition_contract_from_metadata(metadata, required=True)


def _transition_application_fixture(
    *,
    source_semantics: str,
) -> tuple[WholeBodyTrackingPolicy, dict]:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy.config = SimpleNamespace(
        task=SimpleNamespace(apply_training_motion_transitions=True),
        robot=SimpleNamespace(dof_names=["j0", "j1"]),
    )
    policy._motion_data = SimpleNamespace(
        joint_pos=np.asarray([[2.0, 2.0], [4.0, 4.0]], dtype=np.float32),
        joint_vel=np.zeros((2, 2), dtype=np.float32),
        root_pos_w=np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32),
        root_quat_w=np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=np.float32),
        ref_pos_w=np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32),
        ref_quat_w=np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=np.float32),
        has_object=False,
        frame_count=2,
    )
    policy.pinocchio_robot = SimpleNamespace(
        real2pinocchio_index=np.asarray([0, 1]),
        fk_and_get_ref_body_pose_in_world=lambda configuration: (
            np.asarray(configuration[:3], dtype=np.float32),
            np.asarray(configuration[3:7], dtype=np.float32),
        ),
    )
    global_semantics = source_semantics == "global_multi_clip_runtime"
    metadata = {
        "dof_names": ["j0", "j1"],
        "robot_urdf": "<robot name='fixture'/>",
        "experiment_config": {
            "command": {
                "setup_terms": {
                    "motion_command": {
                        "params": {
                            "motion_config": {
                                "enable_default_pose_prepend": True,
                                "default_pose_prepend_duration_s": 0.04,
                                "enable_default_pose_append": True,
                                "default_pose_append_duration_s": 0.04,
                            }
                        }
                    }
                }
            },
            "robot": {
                "init_state": {
                    "pos": [0.0, 0.0, 1.0],
                    "rot": [0.0, 0.0, 0.0, 1.0],
                    "default_joint_angles": {"j0": 0.0, "j1": 0.0},
                }
            },
            "simulator": {
                "_target_": "holosoma.simulator.isaacsim.isaacsim.IsaacSim",
                "config": {
                    "name": "isaacsim",
                    "sim": {"fps": 50, "control_decimation": 1},
                },
            },
        },
    }
    metadata.update(
        _motion_transition_contract_metadata(
            source_semantics=source_semantics,
            prepend_implementation="runtime_hold" if global_semantics else "static_splice",
            prepend_steps=2,
            append_implementation="none" if global_semantics else "static_splice",
            append_steps=0 if global_semantics else 2,
        )
    )
    return policy, metadata


def test_global_multi_clip_inference_materializes_runtime_prepend_but_not_requested_append() -> None:
    policy, metadata = _transition_application_fixture(
        source_semantics="global_multi_clip_runtime"
    )

    applied_prepend = policy._maybe_apply_training_motion_transitions_to_motion_data(
        metadata,
        "torso_link",
    )

    assert applied_prepend == 2
    assert policy._motion_data.frame_count == 4
    np.testing.assert_allclose(policy._motion_data.joint_pos[:, 0], [0.0, 1.0, 2.0, 4.0])


def test_canonical_wbt_config_applies_authenticated_training_transitions_by_default() -> None:
    assert inference_task_values.wbt.apply_training_motion_transitions is True
    assert inference_task_values.locomotion.apply_training_motion_transitions is False


def test_direct_wbt_rejects_silently_unapplied_authenticated_transition() -> None:
    policy, metadata = _transition_application_fixture(
        source_semantics="global_multi_clip_runtime"
    )
    policy.config.task.apply_training_motion_transitions = False

    with pytest.raises(RuntimeError, match="non-equivalent raw timeline"):
        policy._maybe_apply_training_motion_transitions_to_motion_data(
            metadata,
            "torso_link",
        )


def test_direct_wbt_unapplied_transition_requires_explicit_diagnostic_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _policy, metadata = _transition_application_fixture(
        source_semantics="global_multi_clip_runtime"
    )
    monkeypatch.setenv("HOLOSOMA_ALLOW_UNAPPLIED_TRAINING_MOTION_TRANSITIONS", "1")

    settings = _validated_runtime_motion_transition_settings(
        metadata,
        apply_training_motion_transitions=False,
    )

    assert settings["prepend"]["applied"] is True


def test_authenticated_applied_transition_is_a_sha_bound_external_motion_dependency() -> None:
    policy, metadata = _transition_application_fixture(
        source_semantics="global_multi_clip_runtime"
    )
    policy._uses_motion_command = True
    policy._uses_videomimic = False
    policy._uses_object_mocap_distill = False
    policy._uses_object_generalist = False
    policy._uses_legacy_object_obs = False
    policy._uses_sparse_root_command = False
    policy._motion_output_names = {
        "action",
        "joint_pos",
        "joint_vel",
        "ref_quat_xyzw",
    }
    policy._effective_motion_transition_settings = (
        _validated_runtime_motion_transition_settings(
            metadata,
            apply_training_motion_transitions=True,
        )
    )

    assert policy._will_apply_authenticated_motion_transition() is True
    assert policy._policy_requires_motion_data_for_setup() is True

    metadata["motion_transition_contract"]["prepend"]["steps"] += 1
    with pytest.raises(PolicyContractError, match="SHA-256"):
        _validated_runtime_motion_transition_settings(
            metadata,
            apply_training_motion_transitions=True,
        )


def test_single_clip_inference_materializes_both_authenticated_static_transitions() -> None:
    policy, metadata = _transition_application_fixture(source_semantics="single_clip_static")

    applied_prepend = policy._maybe_apply_training_motion_transitions_to_motion_data(
        metadata,
        "torso_link",
    )

    assert applied_prepend == 2
    assert policy._motion_data.frame_count == 6
    np.testing.assert_allclose(
        policy._motion_data.joint_pos[:, 0],
        [0.0, 1.0, 2.0, 4.0, 2.0, 0.0],
    )


def test_motion_onnx_patcher_uses_global_effective_contract_and_skips_requested_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _policy, metadata = _transition_application_fixture(
        source_semantics="global_multi_clip_runtime"
    )
    motion = {
        "joint_pos": np.asarray([[2.0, 2.0], [4.0, 4.0]], dtype=np.float32),
        "joint_vel": np.zeros((2, 2), dtype=np.float32),
        "root_pos_w": np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32),
        "root_quat_wxyz": np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=np.float32),
        "ref_pos_xyz": np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32),
        "ref_quat_xyzw": np.asarray([[0.0, 0.0, 0.0, 1.0]] * 2, dtype=np.float32),
    }

    class _FakePinocchioRobot:
        def __init__(self, *_args, **_kwargs) -> None:
            self.real2pinocchio_index = np.asarray([0, 1])

        @staticmethod
        def fk_and_get_ref_body_pose_in_world(configuration):
            return (
                np.asarray(configuration[:3], dtype=np.float32),
                np.asarray(configuration[3:7], dtype=np.float32),
            )

    monkeypatch.setattr(inference_wbt_module, "PinocchioRobot", _FakePinocchioRobot)
    _patch_motion_transitions(
        motion,
        metadata,
        dof_names=["j0", "j1"],
        ref_body_name="torso_link",
    )

    assert motion["joint_pos"].shape[0] == 4
    np.testing.assert_allclose(motion["joint_pos"][:, 0], [0.0, 1.0, 2.0, 4.0])


def test_policy_contract_rejects_unimplemented_t1_aligned_sparse_root_mode() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _drop_button_metadata()
    metadata["experiment_config"]["command"] = {
        "setup_terms": {
            "motion_command": {
                "params": {
                    "motion_config": {
                        "contact_aware_sparse_root_command_mode": "t1_aligned_segment",
                    }
                }
            }
        }
    }
    with pytest.raises(PolicyContractError, match="only tracking_error/default"):
        validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
            output_shapes={"action": [1, 29]},
            observation=cfg.observation,
            runtime_dof_names=cfg.robot.dof_names,
            runtime_default_dof_angles=cfg.robot.default_dof_angles,
        )


@pytest.mark.parametrize("mode", ["tracking_error", "default", "robot-tracking-error"])
def test_policy_contract_accepts_equivalent_tracking_error_sparse_root_modes(mode) -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _drop_button_metadata()
    metadata["experiment_config"]["command"] = {
        "setup_terms": {
            "motion_command": {
                "params": {"motion_config": {"contact_aware_sparse_root_command_mode": mode}}
            }
        }
    }
    assert validate_onnx_policy_contract(
        metadata=metadata,
        input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
        output_shapes={"action": [1, 29]},
        observation=cfg.observation,
        runtime_dof_names=cfg.robot.dof_names,
        runtime_default_dof_angles=cfg.robot.default_dof_angles,
    )


def test_policy_contract_allows_explicitly_absent_command_config() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_drop_button_depth_distill
    metadata = _drop_button_metadata()
    metadata["experiment_config"]["command"] = None
    assert validate_onnx_policy_contract(
        metadata=metadata,
        input_shapes={"actor_obs": [1, 94], "perception_obs": [1, 58 * 87]},
        output_shapes={"action": [1, 29]},
        observation=cfg.observation,
        runtime_dof_names=cfg.robot.dof_names,
        runtime_default_dof_angles=cfg.robot.default_dof_angles,
    )


def test_contact_sidecar_selection_matches_training_carry_region_union(tmp_path: Path) -> None:
    assert _select_primary_contact_interval(
        {
            "left_wrist": np.array([77, 275]),
            "right_wrist": np.array([80, 244]),
            "torso": np.array([40, 300]),
        }
    ) == (40, 300)

    np.save(tmp_path / "left_wrist_contact_interval_steps.npy", np.array([77, 275]))
    np.save(tmp_path / "right_wrist_contact_interval_steps.npy", np.array([80, 244]))
    assert _load_contact_interval_from_dir(tmp_path) == (77, 275)


def _write_minimal_motion_npz(
    path: Path,
    *,
    fps: object = np.array([50.0], dtype=np.float32),
    include_fps: bool = True,
    overrides: dict[str, object] | None = None,
) -> None:
    body_quat_w = np.zeros((2, 2, 4), dtype=np.float32)
    body_quat_w[..., 0] = 1.0
    payload = {
        "body_names": np.array(["pelvis", "torso_link"]),
        "joint_names": np.array(["j0", "j1"]),
        "joint_pos": np.zeros((2, 2), dtype=np.float32),
        "joint_vel": np.zeros((2, 2), dtype=np.float32),
        "body_pos_w": np.zeros((2, 2, 3), dtype=np.float32),
        "body_quat_w": body_quat_w,
    }
    if include_fps:
        payload["fps"] = np.asarray(fps)
    if overrides:
        payload.update(overrides)
    np.savez(path, **payload)


def test_motion_data_reads_required_scalar_fps(tmp_path: Path) -> None:
    motion_path = tmp_path / "motion.npz"
    _write_minimal_motion_npz(motion_path, fps=np.array([50.0], dtype=np.float32))

    motion = MotionData(motion_path, ["j0", "j1"], "torso_link")

    assert motion.fps == pytest.approx(50.0)


def test_motion_data_preserves_wxyz_and_runtime_converts_to_xyzw(tmp_path: Path) -> None:
    motion_path = tmp_path / "motion.npz"
    body_quat_w = np.zeros((2, 2, 4), dtype=np.float32)
    body_quat_w[..., 0] = 1.0
    half_sqrt = np.float32(np.sqrt(0.5))
    body_quat_w[:, 1, :] = np.array([half_sqrt, 0.0, 0.0, half_sqrt], dtype=np.float32)
    _write_minimal_motion_npz(motion_path, overrides={"body_quat_w": body_quat_w})
    motion = MotionData(motion_path, ["j0", "j1"], "torso_link")
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._motion_data = motion

    outputs = policy._get_motion_outputs_from_motion_data(0)

    assert outputs is not None
    np.testing.assert_allclose(
        outputs["ref_quat_xyzw"],
        np.array([[0.0, 0.0, half_sqrt, half_sqrt]], dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("fps", "include_fps", "error_match"),
    [
        (None, False, "missing required scalar fps"),
        (np.array([30.0, 50.0]), True, "exactly one value"),
        (np.array([0.0]), True, "finite and positive"),
        (np.array([np.nan]), True, "finite and positive"),
    ],
)
def test_motion_data_rejects_invalid_fps(
    tmp_path: Path,
    fps: object,
    include_fps: bool,
    error_match: str,
) -> None:
    motion_path = tmp_path / "motion.npz"
    _write_minimal_motion_npz(motion_path, fps=fps, include_fps=include_fps)

    with pytest.raises(ValueError, match=error_match):
        MotionData(motion_path, ["j0", "j1"], "torso_link")


@pytest.mark.parametrize(
    ("overrides", "error_match"),
    [
        (
            {"joint_pos": np.array([[0.0, 0.0], [np.nan, 0.0]], dtype=np.float32)},
            "joint_pos.*non-finite",
        ),
        ({"joint_vel": np.zeros((1, 2), dtype=np.float32)}, "frame-count mismatch"),
        ({"body_pos_w": np.zeros((2, 2, 2), dtype=np.float32)}, "Unexpected body_pos_w shape"),
        ({"joint_pos": np.zeros((0, 2), dtype=np.float32)}, "at least one frame"),
        ({"object_pos_w": np.zeros((2, 3), dtype=np.float32)}, "provide object_pos_w and object_quat_w together"),
        (
            {
                "object_pos_w": np.zeros((2, 2), dtype=np.float32),
                "object_quat_w": np.zeros((2, 4), dtype=np.float32),
            },
            "Unexpected object_pos_w shape",
        ),
    ],
)
def test_motion_data_rejects_malformed_or_nonfinite_arrays(
    tmp_path: Path,
    overrides: dict[str, object],
    error_match: str,
) -> None:
    motion_path = tmp_path / "motion.npz"
    _write_minimal_motion_npz(motion_path, overrides=overrides)

    with pytest.raises(ValueError, match=error_match):
        MotionData(motion_path, ["j0", "j1"], "torso_link")


@pytest.mark.parametrize(
    ("overrides", "error_match"),
    [
        ({"joint_pos": np.zeros((2, 2), dtype=np.int32)}, "joint_pos.*real floating dtype"),
        ({"body_names": np.array(["pelvis", " torso_link"])}, "empty, padded, or NUL"),
        ({"joint_names": np.array(["j0", "j0"])}, "duplicate names"),
        ({"body_names": np.array([b"pelvis", b"\xff"])}, "non-UTF-8"),
        ({"body_quat_w": np.zeros((2, 2, 4), dtype=np.float32)}, "unit WXYZ quaternions"),
    ],
)
def test_motion_data_rejects_ambiguous_or_noncanonical_payloads(
    tmp_path: Path,
    overrides: dict[str, object],
    error_match: str,
) -> None:
    motion_path = tmp_path / "motion.npz"
    _write_minimal_motion_npz(motion_path, overrides=overrides)

    with pytest.raises(ValueError, match=error_match):
        MotionData(motion_path, ["j0", "j1"], "torso_link")


@pytest.mark.parametrize(
    ("object_fields", "error_match"),
    [
        ({"object_scale": np.ones(3, dtype=np.float32)}, "scale and size are not interchangeable"),
        ({"object_size": np.ones(3, dtype=np.int32)}, "object_size.*real floating dtype"),
        ({"object_size": np.array([-1.0, 1.0, 1.0], dtype=np.float32)}, "strictly positive"),
        ({"object_quat_w": np.zeros((2, 4), dtype=np.float32)}, "unit WXYZ quaternions"),
    ],
)
def test_motion_data_rejects_invalid_object_physical_contract(
    tmp_path: Path,
    object_fields: dict[str, object],
    error_match: str,
) -> None:
    motion_path = tmp_path / "motion.npz"
    object_quat_w = np.zeros((2, 4), dtype=np.float32)
    object_quat_w[:, 0] = 1.0
    overrides: dict[str, object] = {
        "object_pos_w": np.zeros((2, 3), dtype=np.float32),
        "object_quat_w": object_quat_w,
    }
    overrides.update(object_fields)
    _write_minimal_motion_npz(motion_path, overrides=overrides)

    with pytest.raises(ValueError, match=error_match):
        MotionData(motion_path, ["j0", "j1"], "torso_link")


def _runtime_timebase_metadata(*, simulator_fps: float = 500.0, control_decimation: float = 10.0) -> dict:
    return {
        "experiment_config": {
            "simulator": {
                "config": {
                    "sim": {
                        "fps": simulator_fps,
                        "control_decimation": control_decimation,
                    }
                }
            }
        }
    }


def test_wbt_runtime_motion_timebase_accepts_exact_training_rate() -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy.rl_rate = 50.0
    policy._motion_data = SimpleNamespace(fps=50.0)

    policy._validate_runtime_motion_timebase(_runtime_timebase_metadata())


@pytest.mark.parametrize(
    "metadata",
    [
        {"experiment_config": {"simulator": []}},
        {"experiment_config": {"simulator": {"config": []}}},
        {"experiment_config": {"simulator": {"config": {"sim": []}}}},
        _runtime_timebase_metadata(simulator_fps=True),
        _runtime_timebase_metadata(control_decimation=float("inf")),
        {"experiment_config": {"simulator": {"config": {"sim": {"fps": 500.0}}}}},
    ],
)
def test_extract_control_dt_rejects_malformed_or_partial_metadata(metadata: dict) -> None:
    with pytest.raises(ValueError):
        _extract_control_dt_from_metadata(metadata)


@pytest.mark.parametrize(
    ("motion_fps", "runtime_fps", "metadata", "message"),
    [
        (30.0, 50.0, _runtime_timebase_metadata(), "Motion FPS must match"),
        (50.0, 40.0, _runtime_timebase_metadata(), "Motion FPS must match"),
        (None, 40.0, _runtime_timebase_metadata(), "serialized training control frequency"),
    ],
)
def test_wbt_runtime_motion_timebase_rejects_rate_drift(
    motion_fps: float | None,
    runtime_fps: float,
    metadata: dict,
    message: str,
) -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy.rl_rate = runtime_fps
    policy._motion_data = None if motion_fps is None else SimpleNamespace(fps=motion_fps)

    with pytest.raises(ValueError, match=message):
        policy._validate_runtime_motion_timebase(metadata)


@pytest.mark.parametrize(
    "metadata",
    [
        {"contact_interval_fps": 30},
        {"fps": 30},
        {"contact_interval_fps": 30, "fps": 60},
    ],
)
def test_inference_contact_timebase_conversion_matches_training(
    metadata: dict[str, object],
) -> None:
    interval = (29, 166)
    expected = _training_convert_contact_interval_timebase(
        interval,
        metadata=metadata,
        motion_fps=50.0,
    )

    assert expected == (49, 277)
    assert _convert_contact_interval_timebase(
        interval,
        metadata=metadata,
        motion_fps=50.0,
    ) == expected


@pytest.mark.parametrize(
    ("metadata", "motion_fps"),
    [
        ({"fps": 0}, 50.0),
        ({"fps": 30}, 0.0),
        ({"fps": "invalid"}, 50.0),
        ({"fps": 30}, None),
    ],
)
def test_inference_contact_timebase_rejects_invalid_fps(
    metadata: dict[str, object],
    motion_fps: float | None,
) -> None:
    with pytest.raises(ValueError, match="Contact interval FPS"):
        _convert_contact_interval_timebase(
            (29, 166),
            metadata=metadata,
            motion_fps=motion_fps,
        )


def test_inference_drop_button_uses_sidecar_t2_instead_of_kinematic_t2() -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._manual_sparse_root_command_sub = None
    policy._motion_data = SimpleNamespace(has_object=True)
    policy._contact_aware_button_window = (77, 275)
    policy._contact_aware_carry_window = (125, 191)

    policy._get_motion_index = lambda: 274
    assert policy._get_drop_button().tolist() == [[0.0]]
    policy._get_motion_index = lambda: 275
    assert policy._get_drop_button().tolist() == [[1.0]]


def test_inference_pickup_button_uses_sidecar_t1() -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._motion_data = SimpleNamespace(has_object=True)
    policy._contact_aware_button_window = (77, 275)
    policy._contact_aware_carry_window = (125, 191)

    policy._get_motion_index = lambda: 76
    assert policy._get_pickup_button().tolist() == [[1.0]]
    policy._get_motion_index = lambda: 77
    assert policy._get_pickup_button().tolist() == [[0.0]]


@pytest.mark.parametrize(
    ("motion_index", "external_value", "expected"),
    [
        (76, 0.49, [[0.0]]),
        (77, 0.50, [[1.0]]),
    ],
)
def test_inference_external_pickup_button_overrides_automatic_sidecar(
    motion_index: int,
    external_value: float,
    expected: list[list[float]],
) -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._manual_sparse_root_command_sub = SimpleNamespace(
        get_payload=lambda: {"pickup_button": external_value}
    )
    policy._motion_data = SimpleNamespace(has_object=True)
    policy._contact_aware_button_window = (77, 275)
    policy._contact_aware_carry_window = (125, 191)
    policy._get_motion_index = lambda: motion_index

    assert policy._get_pickup_button().tolist() == expected


@pytest.mark.parametrize("external_value", [float("nan"), [0.0, 1.0], "not-a-number"])
def test_inference_malformed_external_pickup_button_falls_back_to_sidecar(
    external_value: object,
) -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._manual_sparse_root_command_sub = SimpleNamespace(
        get_payload=lambda: {"pickup_button": external_value}
    )
    policy._motion_data = SimpleNamespace(has_object=True)
    policy._contact_aware_button_window = (77, 275)
    policy._contact_aware_carry_window = (125, 191)
    policy._get_motion_index = lambda: 76

    # Invalid manual data must not inject a bit; retain the automatic pre-t1 value.
    assert policy._get_pickup_button().tolist() == [[1.0]]


def test_manual_root_publisher_pickup_and_drop_payloads_are_optional_and_symmetric() -> None:
    from holosoma_inference.utils.sim_control import ManualRootCommandPub

    published: list[dict[str, object]] = []
    publisher = object.__new__(ManualRootCommandPub)
    publisher.enabled = True
    publisher.socket = SimpleNamespace(
        send_string=lambda payload, _flags: published.append(json.loads(payload))
    )

    publisher.publish(
        enabled=True,
        mode="manual",
        command=(0.0, 0.0, 0.0),
        pickup_button=True,
        drop_button=False,
    )
    assert published[-1]["pickup_button"] == 1.0
    assert published[-1]["drop_button"] == 0.0

    publisher.publish(enabled=False, mode="motion", command=(0.0, 0.0, 0.0))
    assert "pickup_button" not in published[-1]
    assert "drop_button" not in published[-1]


def test_pickup_only_observation_is_recognized_as_wbt() -> None:
    from holosoma_inference.run_policy import _is_wbt_observation

    assert _is_wbt_observation({"actor_obs_pickup_button": ["pickup_button"]}) is True


def test_viser_manual_pickup_reset_restores_pre_t1_value() -> None:
    from holosoma.utils.safe_torch_import import torch
    from holosoma.utils.viser_live import ViserLiveViewer

    motion_command = SimpleNamespace(
        manual_pickup_button=torch.zeros((2, 1), dtype=torch.float32),
        manual_pickup_button_override_enabled=False,
    )
    pickup_checkbox = SimpleNamespace(value=False)
    viewer = object.__new__(ViserLiveViewer)
    viewer._get_motion_command = lambda: motion_command
    viewer._pickup_button_gui_enabled = True
    viewer._pickup_button_cb = pickup_checkbox
    viewer._pickup_button_status = None

    viewer._reset_manual_pickup_button(reset_gui_toggle=True)

    assert motion_command.manual_pickup_button.tolist() == [[1.0], [1.0]]
    assert motion_command.manual_pickup_button_override_enabled is True
    assert pickup_checkbox.value is True


def test_viser_disabled_pickup_gui_preserves_automatic_sidecar_mode() -> None:
    from holosoma.utils.safe_torch_import import torch
    from holosoma.utils.viser_live import ViserLiveViewer

    motion_command = SimpleNamespace(
        manual_pickup_button=torch.ones((1, 1), dtype=torch.float32),
        manual_pickup_button_override_enabled=True,
    )
    viewer = object.__new__(ViserLiveViewer)
    viewer._get_motion_command = lambda: motion_command
    viewer._pickup_button_gui_enabled = False

    viewer._update_manual_pickup_button()

    assert motion_command.manual_pickup_button_override_enabled is False
    assert motion_command.manual_pickup_button.tolist() == [[1.0]]


def _button_window_policy(
    contact_root: Path,
    *,
    applied_prepend_steps: int,
    compensated_in_training: bool,
    frame_count: int = 400,
    has_object: bool = True,
    uses_contact_window_observation: bool = True,
    use_adaptive_timesteps_sampler: bool = True,
    uniform_t1_window_sampling_enabled: bool = True,
    source_semantics: str = "global_multi_clip_runtime",
) -> WholeBodyTrackingPolicy:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._motion_data = SimpleNamespace(
        motion_path=Path("/portable/assets/clip.npz"),
        fps=50.0,
        frame_count=frame_count,
        has_object=has_object,
    )
    policy._motion_cfg = {
        "use_adaptive_timesteps_sampler": use_adaptive_timesteps_sampler,
        "uniform_t1_window_sampling_enabled": uniform_t1_window_sampling_enabled,
        "adaptive_sampling_contact_interval_root": str(contact_root),
        "contact_interval_runtime_prepend_compensation": compensated_in_training,
    }
    policy._uses_contact_window_observation = uses_contact_window_observation
    policy._motion_transition_prepend_steps = applied_prepend_steps
    policy._effective_motion_transition_settings = {
        "source_semantics": source_semantics,
    }
    return policy


def test_current_runtime_prepend_sidecar_is_not_offset_twice(tmp_path: Path) -> None:
    contact_root = tmp_path / "contacts"
    clip_dir = contact_root / "0000_clip"
    clip_dir.mkdir(parents=True)
    (clip_dir / "metadata.json").write_text(json.dumps({"clip_id": "clip"}), encoding="utf-8")
    (clip_dir / "contact_intervals.json").write_text(
        json.dumps({"left_wrist": [64, 328]}),
        encoding="utf-8",
    )

    policy = _button_window_policy(
        contact_root,
        applied_prepend_steps=10,
        compensated_in_training=True,
    )
    # Training subtracts the ten frozen runtime-prepend steps to obtain motion
    # time.  Inference materializes those ten frames, so the effective static
    # motion index is the original exported rollout index again.
    assert policy._load_contact_aware_button_window(tmp_path / "model.onnx") == (64, 328)


@pytest.mark.parametrize(
    ("compensated_in_training", "expected"),
    [
        (True, (49, 277)),
        (False, (59, 287)),
    ],
)
def test_inference_contact_sidecar_converts_fps_before_prepend_compensation(
    tmp_path: Path,
    compensated_in_training: bool,
    expected: tuple[int, int],
) -> None:
    contact_root = tmp_path / "contacts"
    clip_dir = contact_root / "0000_clip"
    clip_dir.mkdir(parents=True)
    (clip_dir / "metadata.json").write_text(
        json.dumps({"clip_id": "clip", "fps": 30}),
        encoding="utf-8",
    )
    (clip_dir / "contact_intervals.json").write_text(
        json.dumps({"left_wrist": [29, 166]}),
        encoding="utf-8",
    )

    policy = _button_window_policy(
        contact_root,
        applied_prepend_steps=10,
        compensated_in_training=compensated_in_training,
    )

    assert policy._load_contact_aware_button_window(tmp_path / "model.onnx") == expected


@pytest.mark.parametrize(
    "fps_metadata",
    [
        {"contact_interval_fps": 0},
        {"fps": "invalid"},
    ],
)
def test_inference_contact_sidecar_fails_closed_on_invalid_export_fps(
    tmp_path: Path,
    fps_metadata: dict[str, object],
) -> None:
    contact_root = tmp_path / "contacts"
    clip_dir = contact_root / "0000_clip"
    clip_dir.mkdir(parents=True)
    (clip_dir / "metadata.json").write_text(
        json.dumps({"clip_id": "clip", **fps_metadata}),
        encoding="utf-8",
    )
    (clip_dir / "contact_intervals.json").write_text(
        json.dumps({"left_wrist": [29, 166]}),
        encoding="utf-8",
    )
    policy = _button_window_policy(
        contact_root,
        applied_prepend_steps=10,
        compensated_in_training=True,
        use_adaptive_timesteps_sampler=False,
        uniform_t1_window_sampling_enabled=False,
    )

    with pytest.raises(ValueError, match="Contact interval FPS"):
        policy._load_contact_aware_button_window(tmp_path / "model.onnx")


@pytest.mark.parametrize("clip_id", ["noscale__any_barrel_12", "2024_box_10"])
def test_inference_contact_sidecar_accepts_exact_clip_directory_with_underscores(
    tmp_path: Path,
    clip_id: str,
) -> None:
    contact_root = tmp_path / "contacts"
    clip_dir = contact_root / clip_id
    clip_dir.mkdir(parents=True)
    np.save(clip_dir / "left_wrist_contact_interval_steps.npy", np.array([11, 29]))

    policy = _button_window_policy(
        contact_root,
        applied_prepend_steps=0,
        compensated_in_training=True,
    )
    policy._motion_data = SimpleNamespace(
        motion_path=Path("/portable/assets") / f"{clip_id}.npz",
        fps=50.0,
        frame_count=400,
        has_object=True,
    )
    assert policy._load_contact_aware_button_window(tmp_path / "model.onnx") == (11, 29)


def test_inference_contact_sidecar_rejects_duplicate_clip_directories(tmp_path: Path) -> None:
    contact_root = tmp_path / "contacts"
    for directory_name in ("clip", "0000_clip"):
        clip_dir = contact_root / directory_name
        clip_dir.mkdir(parents=True)
        np.save(clip_dir / "left_wrist_contact_interval_steps.npy", np.array([11, 29]))

    policy = _button_window_policy(
        contact_root,
        applied_prepend_steps=0,
        compensated_in_training=True,
    )
    with pytest.raises(RuntimeError, match="Multiple training contact directories"):
        policy._load_contact_aware_button_window(tmp_path / "model.onnx")


def test_legacy_no_prepend_sidecar_keeps_raw_unreachable_t2(tmp_path: Path) -> None:
    contact_root = tmp_path / "contacts"
    clip_dir = contact_root / "0000_clip"
    clip_dir.mkdir(parents=True)
    np.save(clip_dir / "left_wrist_contact_interval_steps.npy", np.array([77, 328]))

    policy = _button_window_policy(
        contact_root,
        applied_prepend_steps=0,
        compensated_in_training=False,
        frame_count=100,
    )
    # The old swl checkpoints were trained with the raw value, including clips
    # whose t2 exceeded the reachable motion index.  Do not silently clamp it.
    assert policy._load_contact_aware_button_window(tmp_path / "model.onnx") == (77, 328)


def test_legacy_runtime_prepend_sidecar_reproduces_old_training_timebase(tmp_path: Path) -> None:
    contact_root = tmp_path / "contacts"
    clip_dir = contact_root / "0000_clip"
    clip_dir.mkdir(parents=True)
    (clip_dir / "contact_intervals.json").write_text(
        json.dumps({"left_wrist": [64, 328]}),
        encoding="utf-8",
    )

    policy = _button_window_policy(
        contact_root,
        applied_prepend_steps=10,
        compensated_in_training=False,
        frame_count=100,
    )
    # Old u7-style training compared raw rollout t2 against motion time.  Once
    # inference materializes the frozen warmup, add S exactly once to preserve
    # the wall-clock label transition learned by that checkpoint.
    assert policy._load_contact_aware_button_window(tmp_path / "model.onnx") == (74, 338)


def test_compensated_sidecar_clamped_zero_includes_entire_materialized_prefix(
    tmp_path: Path,
) -> None:
    contact_root = tmp_path / "contacts"
    clip_dir = contact_root / "0000_clip"
    clip_dir.mkdir(parents=True)
    np.save(clip_dir / "left_wrist_contact_interval_steps.npy", np.array([3, 25]))
    policy = _button_window_policy(
        contact_root,
        applied_prepend_steps=10,
        compensated_in_training=True,
        frame_count=40,
    )

    assert policy._load_contact_aware_button_window(tmp_path / "model.onnx") == (0, 25)


def test_legacy_zero_start_sidecar_includes_entire_materialized_prefix(
    tmp_path: Path,
) -> None:
    contact_root = tmp_path / "contacts"
    clip_dir = contact_root / "0000_clip"
    clip_dir.mkdir(parents=True)
    np.save(clip_dir / "left_wrist_contact_interval_steps.npy", np.array([0, 20]))
    policy = _button_window_policy(
        contact_root,
        applied_prepend_steps=10,
        compensated_in_training=False,
        frame_count=40,
    )

    assert policy._load_contact_aware_button_window(tmp_path / "model.onnx") == (0, 30)


def test_compensated_sidecar_rejects_interval_from_different_length_clip(
    tmp_path: Path,
) -> None:
    contact_root = tmp_path / "contacts"
    clip_dir = contact_root / "0000_clip"
    clip_dir.mkdir(parents=True)
    (clip_dir / "metadata.json").write_text(json.dumps({"clip_id": "clip"}), encoding="utf-8")
    np.save(clip_dir / "left_wrist_contact_interval_steps.npy", np.array([11, 29]))
    policy = _button_window_policy(
        contact_root,
        applied_prepend_steps=0,
        compensated_in_training=True,
        frame_count=20,
    )

    with pytest.raises(ValueError, match="outside the active inference motion range"):
        policy._load_contact_aware_button_window(tmp_path / "model.onnx")


def test_compensated_sidecar_accepts_exact_active_motion_boundaries(tmp_path: Path) -> None:
    contact_root = tmp_path / "contacts"
    clip_dir = contact_root / "0000_clip"
    clip_dir.mkdir(parents=True)
    (clip_dir / "metadata.json").write_text(json.dumps({"clip_id": "clip"}), encoding="utf-8")
    np.save(clip_dir / "left_wrist_contact_interval_steps.npy", np.array([0, 30]))
    policy = _button_window_policy(
        contact_root,
        applied_prepend_steps=0,
        compensated_in_training=True,
        frame_count=30,
    )

    assert policy._load_contact_aware_button_window(tmp_path / "model.onnx") == (0, 30)


def test_configured_training_sidecar_is_fail_closed_when_unavailable(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing_contacts"
    policy = _button_window_policy(
        missing_root,
        applied_prepend_steps=10,
        compensated_in_training=True,
    )

    with pytest.raises(FileNotFoundError, match="change the student observation contract"):
        policy._load_contact_aware_button_window(tmp_path / "model.onnx")


def test_robot_only_motion_skips_configured_sidecar_before_root_resolution(tmp_path: Path) -> None:
    policy = _button_window_policy(
        tmp_path / "missing_contacts",
        applied_prepend_steps=0,
        compensated_in_training=True,
        has_object=False,
    )

    assert policy._load_contact_aware_button_window(tmp_path / "model.onnx") is None


def test_contact_observation_loads_sidecar_when_sampling_features_are_disabled(
    tmp_path: Path,
) -> None:
    contact_root = tmp_path / "contacts"
    clip_dir = contact_root / "0000_clip"
    clip_dir.mkdir(parents=True)
    (clip_dir / "metadata.json").write_text(json.dumps({"clip_id": "clip"}), encoding="utf-8")
    np.save(clip_dir / "left_wrist_contact_interval_steps.npy", np.array([11, 29]))
    policy = _button_window_policy(
        contact_root,
        applied_prepend_steps=0,
        compensated_in_training=True,
        use_adaptive_timesteps_sampler=False,
        uniform_t1_window_sampling_enabled=False,
    )

    assert policy._load_contact_aware_button_window(tmp_path / "model.onnx") == (11, 29)


def test_empty_contact_root_environment_override_keeps_configured_training_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contact_root = tmp_path / "contacts"
    clip_dir = contact_root / "0000_clip"
    clip_dir.mkdir(parents=True)
    (clip_dir / "metadata.json").write_text(json.dumps({"clip_id": "clip"}), encoding="utf-8")
    np.save(clip_dir / "left_wrist_contact_interval_steps.npy", np.array([11, 29]))
    policy = _button_window_policy(
        contact_root,
        applied_prepend_steps=0,
        compensated_in_training=True,
        use_adaptive_timesteps_sampler=False,
        uniform_t1_window_sampling_enabled=False,
    )
    monkeypatch.setenv("HOLOSOMA_CONTACT_INTERVAL_ROOT", "   ")

    assert policy._load_contact_aware_button_window(tmp_path / "model.onnx") == (11, 29)


def test_contact_observation_fails_closed_when_sampling_features_are_disabled(
    tmp_path: Path,
) -> None:
    policy = _button_window_policy(
        tmp_path / "missing_contacts",
        applied_prepend_steps=0,
        compensated_in_training=True,
        use_adaptive_timesteps_sampler=False,
        uniform_t1_window_sampling_enabled=False,
    )

    with pytest.raises(FileNotFoundError, match="change the student observation contract"):
        policy._load_contact_aware_button_window(tmp_path / "model.onnx")


def test_non_contact_observation_keeps_legacy_behavior_when_sampling_features_are_disabled(
    tmp_path: Path,
) -> None:
    policy = _button_window_policy(
        tmp_path / "missing_contacts",
        applied_prepend_steps=0,
        compensated_in_training=True,
        uses_contact_window_observation=False,
        use_adaptive_timesteps_sampler=False,
        uniform_t1_window_sampling_enabled=False,
    )

    assert policy._load_contact_aware_button_window(tmp_path / "model.onnx") is None


def test_rel_z_root_window_uses_same_sidecar_t2_release_cap() -> None:
    total_steps = 100
    root_pos = np.zeros((total_steps, 3), dtype=np.float32)
    object_pos = np.zeros((total_steps, 3), dtype=np.float32)
    object_pos[5:, 2] = 0.5
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._motion_data = SimpleNamespace(
        has_object=True,
        object_pos_w=object_pos,
        root_pos_w=root_pos,
        frame_count=total_steps,
    )
    policy._motion_cfg = {"contact_aware_carry_window_mode": "rel_z"}
    policy._contact_aware_carry_window = None
    policy._contact_aware_contact_window = (5, 50)
    # A kinematic button end must not replace the sidecar release cap used by
    # the independently configured rel-z root command.
    policy._contact_aware_button_window = (5, 90)

    carry_start, carry_end = policy._get_contact_aware_carry_window()

    assert carry_start == 5
    assert carry_end == 20


def test_global_runtime_prepend_kinematic_window_uses_original_source_clip() -> None:
    prepend_steps = 10
    source_steps = 20
    root_pos = np.zeros((prepend_steps + source_steps, 3), dtype=np.float32)
    object_pos = np.zeros_like(root_pos)
    # A materialized default-pose interpolation can cross the rel-z threshold,
    # but training computed the kinematic window from the unmodified clip and
    # held its time_step at zero throughout this prefix.
    object_pos[:prepend_steps, 2] = 1.0
    object_pos[prepend_steps + 5 :, 2] = 1.0
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._motion_data = SimpleNamespace(
        has_object=True,
        object_pos_w=object_pos,
        root_pos_w=root_pos,
        frame_count=prepend_steps + source_steps,
    )
    policy._motion_cfg = {"contact_aware_carry_window_mode": "rel_z"}
    policy._contact_aware_carry_window = None
    policy._contact_aware_button_window = None
    policy._motion_transition_prepend_steps = prepend_steps
    policy._effective_motion_transition_settings = {
        "source_semantics": "global_multi_clip_runtime",
    }

    assert policy._get_contact_aware_carry_window() == (15, 30)


def test_zero_source_window_maps_across_runtime_prepend_boundary() -> None:
    assert _map_source_window_to_materialized_timeline(
        (0, 7),
        source_semantics="global_multi_clip_runtime",
        prepend_steps=10,
    ) == (0, 17)
    assert _map_source_window_to_materialized_timeline(
        (3, 7),
        source_semantics="global_multi_clip_runtime",
        prepend_steps=10,
    ) == (13, 17)


def test_zero_start_contact_sidecar_includes_entire_materialized_runtime_prepend(
    tmp_path: Path,
) -> None:
    contact_root = tmp_path / "contacts"
    clip_dir = contact_root / "0000_clip"
    clip_dir.mkdir(parents=True)
    (clip_dir / "metadata.json").write_text(
        json.dumps({"clip_id": "clip", "fps": 50}),
        encoding="utf-8",
    )
    (clip_dir / "contact_intervals.json").write_text(
        json.dumps({"left_wrist": [0, 17]}),
        encoding="utf-8",
    )
    policy = _button_window_policy(
        contact_root,
        applied_prepend_steps=10,
        compensated_in_training=True,
    )

    # Training maps the physical interval to source time [0, 7).  Runtime
    # holds source time at zero during every materialized prepend frame, so the
    # equivalent half-open materialized interval is [0, 17), not [10, 17).
    assert policy._load_contact_aware_button_window(tmp_path / "model.onnx") == (0, 17)


def test_non_sim_timeline_consumes_last_frame_before_end_reset() -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy.config = SimpleNamespace(
        task=SimpleNamespace(restart_sim_on_motion_end=True)
    )
    policy._motion_data = SimpleNamespace(frame_count=4)
    policy._motion_index_offset = 0
    policy.motion_clip_progressing = True
    policy.motion_timestep = 0
    policy.use_sim_time = False
    policy._motion_end_reset_requested = False
    policy._disable_motion_end_sim_reset = True
    policy.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)

    consumed = []
    for _ in range(policy._motion_data.frame_count):
        motion_index = policy._get_motion_index()
        consumed.append(motion_index)
        policy._advance_motion_after_policy_step(motion_index)

    assert consumed == [0, 1, 2, 3]
    assert policy.motion_timestep == 3
    assert policy._motion_end_reset_requested is True


def test_provenanced_embedded_timeline_consumes_and_holds_final_frame_without_motion_data() -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy.config = SimpleNamespace(
        task=SimpleNamespace(restart_sim_on_motion_end=True)
    )
    policy._motion_data = None
    policy._embedded_motion_frame_count = 4
    policy._motion_output_names = {
        "action",
        "joint_pos",
        "joint_vel",
        "ref_quat_xyzw",
    }
    policy._motion_index_offset = 0
    policy.motion_clip_progressing = True
    policy.motion_timestep = 0
    policy.use_sim_time = False
    policy._motion_end_reset_requested = False
    policy._motion_end_reset_episode_generation = None
    policy._disable_motion_end_sim_reset = True
    policy.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)

    consumed = []
    for _ in range(4):
        motion_index = policy._get_motion_index()
        consumed.append(motion_index)
        policy._advance_motion_after_policy_step(motion_index)

    assert consumed == [0, 1, 2, 3]
    assert policy.motion_timestep == 3
    assert policy._get_motion_index() == 3
    assert policy._motion_end_reset_requested is True


def test_provenanced_embedded_timeline_requests_reset_and_waits_for_ack_without_motion_data() -> None:
    generation = [21]
    reset_reasons: list[str] = []
    restarted: list[bool] = []
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy.config = SimpleNamespace(
        task=SimpleNamespace(restart_sim_on_motion_end=True)
    )
    policy._motion_data = None
    policy._embedded_motion_frame_count = 4
    policy._motion_output_names = {
        "action",
        "joint_pos",
        "joint_vel",
        "ref_quat_xyzw",
    }
    policy._motion_index_offset = 0
    policy.motion_clip_progressing = True
    policy.motion_timestep = 4
    policy._motion_end_reset_requested = False
    policy._motion_end_reset_episode_generation = None
    policy._disable_motion_end_sim_reset = False
    policy.interface = SimpleNamespace(
        _sim_control_pub=SimpleNamespace(
            request_reset=lambda reason: reset_reasons.append(reason) or True
        )
    )
    policy._get_control_tick_episode_generation = lambda: generation[0]
    policy.logger = SimpleNamespace(
        error=lambda *_args, **_kwargs: None,
        info=lambda *_args, **_kwargs: None,
    )
    policy._handle_start_motion_clip = lambda: restarted.append(True)

    policy._maybe_restart_sim_at_motion_end(consumed_motion_index=3)
    assert reset_reasons == ["motion_end"]
    assert policy.motion_timestep == 3
    assert restarted == []

    policy._maybe_complete_motion_end_reset()
    assert restarted == []
    generation[0] = 22
    policy._maybe_complete_motion_end_reset()
    assert restarted == [True]


def test_legacy_embedded_timeline_without_provenance_retains_graph_clamp_behavior() -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._motion_data = None
    policy._embedded_motion_frame_count = None
    policy._motion_output_names = {
        "action",
        "joint_pos",
        "joint_vel",
        "ref_quat_xyzw",
    }
    policy._motion_index_offset = 0
    policy.motion_timestep = 99

    assert policy._active_motion_frame_count() is None
    assert policy._get_motion_index() == 99


def test_missing_reset_channel_holds_last_frame_without_replaying_clip() -> None:
    messages: list[str] = []
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy.config = SimpleNamespace(
        task=SimpleNamespace(restart_sim_on_motion_end=True)
    )
    policy._motion_data = SimpleNamespace(frame_count=4)
    policy._motion_index_offset = 0
    policy.motion_clip_progressing = True
    policy.motion_timestep = 4
    policy._motion_end_reset_requested = False
    policy._motion_end_reset_episode_generation = None
    policy._disable_motion_end_sim_reset = False
    policy.interface = SimpleNamespace()
    policy._get_control_tick_episode_generation = lambda: 7
    policy.logger = SimpleNamespace(
        error=lambda message, *_args: messages.append(message),
        info=lambda *_args, **_kwargs: None,
    )
    restarted: list[bool] = []
    policy._handle_start_motion_clip = lambda: restarted.append(True)

    policy._maybe_restart_sim_at_motion_end(consumed_motion_index=3)

    assert policy.motion_timestep == 3
    assert policy._motion_end_reset_requested is True
    assert restarted == []
    assert any("reset channel is unavailable" in message for message in messages)


def test_motion_end_restart_waits_for_episode_generation_acknowledgement() -> None:
    generation = [11]
    reset_reasons: list[str] = []
    restarted: list[bool] = []
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy.config = SimpleNamespace(
        task=SimpleNamespace(restart_sim_on_motion_end=True)
    )
    policy._motion_data = SimpleNamespace(frame_count=4)
    policy._motion_index_offset = 0
    policy.motion_timestep = 4
    policy._motion_end_reset_requested = False
    policy._motion_end_reset_episode_generation = None
    policy._disable_motion_end_sim_reset = False
    policy.interface = SimpleNamespace(
        _sim_control_pub=SimpleNamespace(
            request_reset=lambda reason: reset_reasons.append(reason) or True
        )
    )
    policy._get_control_tick_episode_generation = lambda: generation[0]
    policy.logger = SimpleNamespace(
        error=lambda *_args, **_kwargs: None,
        info=lambda *_args, **_kwargs: None,
    )
    policy._handle_start_motion_clip = lambda: restarted.append(True)

    policy._maybe_restart_sim_at_motion_end(consumed_motion_index=3)
    policy._maybe_complete_motion_end_reset()
    assert reset_reasons == ["motion_end"]
    assert restarted == []

    generation[0] = 12
    policy._maybe_complete_motion_end_reset()
    assert restarted == [True]


@pytest.mark.parametrize("reset_sent", [False, True])
def test_track_trigger_web_reports_actual_reset_publish_result(reset_sent: bool) -> None:
    from holosoma.mj_track_trigger_web import TrackTriggerState

    state = object.__new__(TrackTriggerState)
    state.policy_pub = None
    state.control_pub = SimpleNamespace(
        request_reset=lambda reason: reset_sent,
    )
    state.snapshot = lambda: {}

    response = state.request_reset("unit-test", delay_s=0.0)

    assert response["ok"] is reset_sent
    assert response["sent"] is reset_sent


@pytest.mark.parametrize("reset_sent", [False, True])
def test_command_web_reports_actual_reset_publish_result(
    monkeypatch: pytest.MonkeyPatch,
    reset_sent: bool,
) -> None:
    from holosoma.mj_command_web import CommandState

    calls: list[tuple[str, str | None]] = []
    state = object.__new__(CommandState)
    state.lock = threading.Lock()
    state.reset_to_default_pose = False
    state.control_pub = SimpleNamespace(
        request_reset=lambda reason, motion_init_mode=None: (
            calls.append((reason, motion_init_mode)) or reset_sent
        ),
    )
    state.snapshot = lambda: {}
    monkeypatch.setenv("SIM_MOTION_INIT_MODE", "raw_motion_grounded")

    response = state.request_reset("unit-test")

    assert calls == [("unit-test", "raw_motion_grounded")]
    assert response["ok"] is reset_sent
    assert response["sent"] is reset_sent


def _load_inference_config_script():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "mj_infer_inference_config.py"
    spec = importlib.util.spec_from_file_location("mj_infer_inference_config", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mj_config_tensor_shape_parser_preserves_zero_and_unset_dimensions() -> None:
    zero_batch = helper.make_tensor_value_info("zero", TensorProto.FLOAT, [0, 3])
    dynamic_batch = helper.make_tensor_value_info("dynamic", TensorProto.FLOAT, [None, 3])
    named_batch = helper.make_tensor_value_info("named", TensorProto.FLOAT, ["batch", 3])

    script = _load_inference_config_script()
    assert script._tensor_shapes([zero_batch, dynamic_batch, named_batch]) == {
        "zero": [0, 3],
        "dynamic": [None, 3],
        "named": ["batch", 3],
    }


@pytest.mark.parametrize("obs_input_name", ["actor_obs", "obs"])
def test_mj_config_inference_recognizes_drop_button_contract(obs_input_name: str) -> None:
    actor_obs = helper.make_tensor_value_info(obs_input_name, TensorProto.FLOAT, [1, 94])
    perception_obs = helper.make_tensor_value_info("perception_obs", TensorProto.FLOAT, [1, 58 * 87])
    action = helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 29])
    model = helper.make_model(helper.make_graph([], "contract", [actor_obs, perception_obs], [action]))
    _attach_onnx_metadata(model, _drop_button_metadata())

    script = _load_inference_config_script()
    assert script.infer_inference_config(model) == (
        "g1-29dof-wbt-object-contact-aware-drop-button-depth-distill"
    )


def test_mj_config_inference_recognizes_exact_dual_button_contract() -> None:
    cfg = g1_29dof_wbt_object_contact_aware_dual_button_depth_distill
    actor_obs = helper.make_tensor_value_info("actor_obs", TensorProto.FLOAT, [1, 95])
    perception_obs = helper.make_tensor_value_info("perception_obs", TensorProto.FLOAT, [1, 58 * 87])
    action = helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 29])
    model = helper.make_model(helper.make_graph([], "contract", [actor_obs, perception_obs], [action]))
    _attach_onnx_metadata(model, _drop_button_metadata(cfg))

    script = _load_inference_config_script()
    assert script.infer_inference_config(model) == (
        "g1-29dof-wbt-object-contact-aware-dual-button-depth-distill"
    )


@pytest.mark.parametrize(
    ("cfg", "actor_obs_dim", "expected"),
    [
        (
            g1_29dof_wbt_object_as_depth_distill,
            96,
            "g1-29dof-wbt-object-as-depth-distill",
        ),
        (
            g1_29dof_wbt_object_as_contact_aware_depth_distill,
            93,
            "g1-29dof-wbt-object-as-contact-aware-depth-distill",
        ),
        (
            g1_29dof_wbt_object_as_contact_aware_history5_depth_distill,
            453,
            "g1-29dof-wbt-object-as-contact-aware-history5-depth-distill",
        ),
        (
            g1_29dof_wbt_object_contact_aware_depth_distill,
            96,
            "g1-29dof-wbt-object-contact-aware-depth-distill",
        ),
    ],
)
def test_mj_config_inference_recognizes_generic_as_contract(
    cfg,
    actor_obs_dim: int,
    expected: str,
) -> None:
    actor_obs = helper.make_tensor_value_info("actor_obs", TensorProto.FLOAT, [1, actor_obs_dim])
    perception_obs = helper.make_tensor_value_info("perception_obs", TensorProto.FLOAT, [1, 58 * 87])
    action = helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 29])
    model = helper.make_model(helper.make_graph([], "contract", [actor_obs, perception_obs], [action]))
    _attach_onnx_metadata(model, _drop_button_metadata(cfg))

    script = _load_inference_config_script()
    assert script.infer_inference_config(model) == expected


def test_mj_config_inference_does_not_conflate_same_shape_as_and_legacy_contact_contracts() -> None:
    cfg = g1_29dof_wbt_object_as_depth_distill
    actor_obs = helper.make_tensor_value_info("actor_obs", TensorProto.FLOAT, [1, 96])
    perception_obs = helper.make_tensor_value_info("perception_obs", TensorProto.FLOAT, [1, 58 * 87])
    action = helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 29])
    model = helper.make_model(helper.make_graph([], "contract", [actor_obs, perception_obs], [action]))
    _attach_onnx_metadata(model, _drop_button_metadata(cfg))

    script = _load_inference_config_script()
    inferred = script.infer_inference_config(model)
    assert inferred == "g1-29dof-wbt-object-as-depth-distill"
    assert inferred != "g1-29dof-wbt-object-contact-aware-depth-distill"


def test_mj_config_inference_rejects_95_dim_with_single_button_terms() -> None:
    actor_obs = helper.make_tensor_value_info("actor_obs", TensorProto.FLOAT, [1, 95])
    perception_obs = helper.make_tensor_value_info("perception_obs", TensorProto.FLOAT, [1, 58 * 87])
    action = helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 29])
    model = helper.make_model(helper.make_graph([], "contract", [actor_obs, perception_obs], [action]))
    _attach_onnx_metadata(model, _drop_button_metadata())

    script = _load_inference_config_script()
    with pytest.raises(ValueError, match="No inference preset matches"):
        script.infer_inference_config(model)


@pytest.mark.parametrize(
    ("cfg", "obs_dim", "expected"),
    [
        (g1_29dof_wbt_object_generalist, 875, "g1-29dof-wbt-object-generalist"),
        (g1_29dof_wbt_w_object_history1, 175, "g1-29dof-wbt-w-object-history1"),
        (g1_29dof_wbt_w_object_legacy, 875, "g1-29dof-wbt-w-object-legacy"),
        (
            g1_29dof_wbt_object_velocity_generalist,
            181,
            "g1-29dof-wbt-object-velocity-generalist",
        ),
    ],
)
def test_mj_config_inference_uses_full_metadata_for_object_contract(cfg, obs_dim, expected) -> None:
    actor_obs = helper.make_tensor_value_info("actor_obs", TensorProto.FLOAT, [1, obs_dim])
    action = helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 29])
    model = helper.make_model(helper.make_graph([], "contract", [actor_obs], [action]))
    _attach_onnx_metadata(model, _drop_button_metadata(cfg))

    script = _load_inference_config_script()
    assert script.infer_inference_config(model) == expected


def test_mj_config_inference_rejects_shape_only_875_object_artifact() -> None:
    actor_obs = helper.make_tensor_value_info("actor_obs", TensorProto.FLOAT, [1, 875])
    action = helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 29])
    model = helper.make_model(helper.make_graph([], "contract", [actor_obs], [action]))

    script = _load_inference_config_script()
    with pytest.raises(ValueError, match="requires complete experiment_config metadata"):
        script.infer_inference_config(model)


def test_eval_agent_selects_exact_drop_button_inference_preset() -> None:
    actor = SimpleNamespace(
        input_dim=[
            "actor_obs_root_contact_aware",
            "actor_obs_drop_button",
            "actor_obs_proprio_with_actions_no_linvel",
        ],
        layer_config=SimpleNamespace(perception_input_name="perception_obs"),
    )
    config = SimpleNamespace(
        env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
        command=None,
        robot=SimpleNamespace(asset=SimpleNamespace(robot_type="g1_29dof")),
        algo=SimpleNamespace(config=SimpleNamespace(module_dict=SimpleNamespace(actor=actor))),
    )
    assert _infer_inference_config(config) == (
        "inference:g1-29dof-wbt-object-contact-aware-drop-button-depth-distill",
        True,
    )


def test_eval_agent_selects_exact_dual_button_inference_preset() -> None:
    actor = SimpleNamespace(
        input_dim=[
            "actor_obs_root_contact_aware",
            "actor_obs_pickup_button",
            "actor_obs_drop_button",
            "actor_obs_proprio_with_actions_no_linvel",
        ],
        layer_config=SimpleNamespace(perception_input_name="perception_obs"),
    )
    config = SimpleNamespace(
        env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
        command=None,
        robot=SimpleNamespace(asset=SimpleNamespace(robot_type="g1_29dof")),
        algo=SimpleNamespace(config=SimpleNamespace(module_dict=SimpleNamespace(actor=actor))),
    )
    assert _infer_inference_config(config) == (
        "inference:g1-29dof-wbt-object-contact-aware-dual-button-depth-distill",
        True,
    )


@pytest.mark.parametrize(
    ("actor_inputs", "histories", "expected"),
    [
        (
            ["actor_obs_root", "actor_obs_proprio", "actor_obs_actions"],
            None,
            "inference:g1-29dof-wbt-object-as-depth-distill",
        ),
        (
            ["actor_obs_root_contact_aware", "actor_obs_proprio_with_actions_no_linvel"],
            (1, 1),
            "inference:g1-29dof-wbt-object-as-contact-aware-depth-distill",
        ),
        (
            ["actor_obs_root_contact_aware", "actor_obs_proprio_with_actions_no_linvel"],
            (1, 5),
            "inference:g1-29dof-wbt-object-as-contact-aware-history5-depth-distill",
        ),
        (
            ["actor_obs_root_contact_aware", "actor_obs_proprio", "actor_obs_actions"],
            None,
            "inference:g1-29dof-wbt-object-contact-aware-depth-distill",
        ),
    ],
)
def test_eval_agent_selects_exact_generic_as_inference_preset(
    actor_inputs,
    histories,
    expected,
) -> None:
    actor = SimpleNamespace(
        input_dim=actor_inputs,
        layer_config=SimpleNamespace(perception_input_name="perception_obs"),
    )
    observation = None
    if histories is not None:
        observation = SimpleNamespace(
            groups={
                group_name: SimpleNamespace(history_length=history)
                for group_name, history in zip(actor_inputs, histories, strict=True)
            }
        )
    config = SimpleNamespace(
        env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
        command=None,
        robot=SimpleNamespace(asset=SimpleNamespace(robot_type="g1_29dof")),
        algo=SimpleNamespace(config=SimpleNamespace(module_dict=SimpleNamespace(actor=actor))),
        observation=observation,
    )
    assert _infer_inference_config(config) == (expected, True)


@pytest.mark.parametrize("proprio_history", [2, 3, 4, 6])
def test_eval_agent_rejects_unsupported_as_contact_history(proprio_history: int) -> None:
    actor_inputs = [
        "actor_obs_root_contact_aware",
        "actor_obs_proprio_with_actions_no_linvel",
    ]
    actor = SimpleNamespace(
        input_dim=actor_inputs,
        layer_config=SimpleNamespace(perception_input_name="perception_obs"),
    )
    config = SimpleNamespace(
        env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
        command=None,
        robot=SimpleNamespace(asset=SimpleNamespace(robot_type="g1_29dof")),
        algo=SimpleNamespace(config=SimpleNamespace(module_dict=SimpleNamespace(actor=actor))),
        observation=SimpleNamespace(
            groups={
                actor_inputs[0]: SimpleNamespace(history_length=1),
                actor_inputs[1]: SimpleNamespace(history_length=proprio_history),
            }
        ),
    )
    with pytest.raises(ValueError, match="No safe inference preset.*history contract"):
        _infer_inference_config(config)


def test_eval_agent_requires_history_metadata_for_ambiguous_as_contact_groups() -> None:
    actor = SimpleNamespace(
        input_dim=[
            "actor_obs_root_contact_aware",
            "actor_obs_proprio_with_actions_no_linvel",
        ],
        layer_config=SimpleNamespace(perception_input_name="perception_obs"),
    )
    config = SimpleNamespace(
        env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
        command=None,
        robot=SimpleNamespace(asset=SimpleNamespace(robot_type="g1_29dof")),
        algo=SimpleNamespace(config=SimpleNamespace(module_dict=SimpleNamespace(actor=actor))),
    )
    with pytest.raises(ValueError, match="missing serialized observation groups"):
        _infer_inference_config(config)


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (
            g1_29dof_wbt_w_object_generalist,
            "inference:g1-29dof-wbt-object-generalist",
        ),
        (
            g1_29dof_wbt_w_object_generalist_legacy_obs,
            "inference:g1-29dof-wbt-w-object-legacy",
        ),
    ],
)
def test_eval_agent_selects_object_preset_from_exact_training_group(config, expected) -> None:
    assert _infer_inference_config(config) == (expected, True)


def test_eval_agent_selects_explicit_velocity_generalist_contract() -> None:
    velocity_terms = dict(training_observation_values.actor_obs_w_object_legacy_terms)
    velocity_terms["obj_lin_vel_b"] = training_observation_values.critic_obs_w_object_terms[
        "obj_lin_vel_b"
    ]
    velocity_terms["obj_ang_vel_b"] = (
        training_observation_values.critic_obs_w_object_command_privileged_terms["obj_ang_vel_b"]
    )
    actor_group = dataclasses.replace(
        training_observation_values.actor_obs_w_object_legacy,
        history_length=1,
        terms=velocity_terms,
    )
    actor = SimpleNamespace(
        input_dim=["actor_obs"],
        layer_config=SimpleNamespace(perception_input_name=""),
    )
    config = SimpleNamespace(
        env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
        command=None,
        robot=SimpleNamespace(asset=SimpleNamespace(robot_type="g1_29dof")),
        algo=SimpleNamespace(config=SimpleNamespace(module_dict=SimpleNamespace(actor=actor))),
        observation=SimpleNamespace(groups={"actor_obs": actor_group}),
    )
    assert _infer_inference_config(config) == (
        "inference:g1-29dof-wbt-object-velocity-generalist",
        True,
    )


def test_eval_agent_selects_current_object_history1_contract() -> None:
    actor_group = dataclasses.replace(
        training_observation_values.g1_29dof_wbt_observation_w_object.groups["actor_obs"],
        history_length=1,
    )
    actor = SimpleNamespace(
        input_dim=["actor_obs"],
        layer_config=SimpleNamespace(perception_input_name=""),
    )
    config = SimpleNamespace(
        env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
        command=None,
        robot=SimpleNamespace(asset=SimpleNamespace(robot_type="g1_29dof")),
        algo=SimpleNamespace(config=SimpleNamespace(module_dict=SimpleNamespace(actor=actor))),
        observation=SimpleNamespace(groups={"actor_obs": actor_group}),
    )
    assert _infer_inference_config(config) == (
        "inference:g1-29dof-wbt-w-object-history1",
        True,
    )
