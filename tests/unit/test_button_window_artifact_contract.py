from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from holosoma.managers.command.terms.wbt import (
    _kinematic_lift_window_from_rel_z,
)
from holosoma_inference.utils.button_window_contract import (
    EMBEDDED_BUTTON_WINDOW_CONTRACT_KEY,
    EMBEDDED_BUTTON_WINDOW_CONTRACT_SHA256_KEY,
    build_kinematic_button_window_contract,
    embedded_button_window_contract_from_metadata,
    kinematic_lift_window_from_rel_z_np,
    validated_contact_aware_button_window_mode,
)
from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy


@pytest.mark.parametrize(
    "rel_z",
    [
        [0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.6, -0.6, -0.45, -0.3, -0.1, -0.1, -0.1, -0.1, -0.1, -0.5, -0.5, -0.5, -0.5, -0.5],
        [0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2],
    ],
)
def test_numpy_and_training_torch_kinematic_windows_are_bit_exact(rel_z):
    values = np.asarray(rel_z, dtype=np.float32)

    assert kinematic_lift_window_from_rel_z_np(values) == (
        _kinematic_lift_window_from_rel_z(
            torch.from_numpy(values.copy()),
            require_sustained_lift=True,
        )
    )


def test_button_kinematic_window_rejects_no_lift_and_rank_drift():
    flat = np.zeros((10,), dtype=np.float32)
    with pytest.raises(ValueError, match="never reaches"):
        kinematic_lift_window_from_rel_z_np(flat)
    with pytest.raises(ValueError, match="never reaches"):
        _kinematic_lift_window_from_rel_z(
            torch.from_numpy(flat.copy()),
            require_sustained_lift=True,
        )
    with pytest.raises(ValueError, match="rank 1"):
        kinematic_lift_window_from_rel_z_np(np.zeros((2, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="rank 1"):
        _kinematic_lift_window_from_rel_z(torch.zeros((2, 5), dtype=torch.float32))


def test_global_runtime_contract_preserves_t1_zero_over_prepend():
    contract, digest = build_kinematic_button_window_contract(
        clip_id="clip",
        source_motion_sha256="a" * 64,
        source_motion_size=100,
        source_frame_count=20,
        motion_fps=50.0,
        source_window=(0, 12),
        motion_transition_contract_sha256="b" * 64,
        source_semantics="global_multi_clip_runtime",
        effective_prepend_steps=10,
        effective_append_steps=0,
    )

    assert contract["source_window"] == [0, 12]
    assert contract["materialized_window"] == [0, 22]
    metadata = {
        EMBEDDED_BUTTON_WINDOW_CONTRACT_KEY: contract,
        EMBEDDED_BUTTON_WINDOW_CONTRACT_SHA256_KEY: digest,
    }
    assert embedded_button_window_contract_from_metadata(metadata) == contract


def test_inference_global_runtime_window_matches_training_source_clock_plus_prepend():
    source_rel_z = np.asarray(
        [0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    source_window = kinematic_lift_window_from_rel_z_np(source_rel_z)
    contract, digest = build_kinematic_button_window_contract(
        clip_id="clip",
        source_motion_sha256="a" * 64,
        source_motion_size=100,
        source_frame_count=source_rel_z.size,
        motion_fps=50.0,
        source_window=source_window,
        motion_transition_contract_sha256="b" * 64,
        source_semantics="global_multi_clip_runtime",
        effective_prepend_steps=10,
        effective_append_steps=0,
    )
    materialized_rel_z = np.concatenate(
        [np.zeros((10,), dtype=np.float32), source_rel_z]
    )
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._motion_data = SimpleNamespace(
        has_object=True,
        object_pos_w=np.stack(
            [np.zeros_like(materialized_rel_z), np.zeros_like(materialized_rel_z), materialized_rel_z],
            axis=1,
        ),
        root_pos_w=np.zeros((materialized_rel_z.size, 3), dtype=np.float32),
        source_frame_count=source_rel_z.size,
        frame_count=materialized_rel_z.size,
        motion_path=Path("clip.npz"),
        source_sha256="a" * 64,
        source_size=100,
        fps=50.0,
    )
    policy._effective_motion_transition_settings = {
        "source_semantics": "global_multi_clip_runtime",
        "contract_sha256": "b" * 64,
    }
    policy._motion_transition_prepend_steps = 10
    policy._onnx_metadata = {
        EMBEDDED_BUTTON_WINDOW_CONTRACT_KEY: contract,
        EMBEDDED_BUTTON_WINDOW_CONTRACT_SHA256_KEY: digest,
    }

    assert source_window == (2, 8)
    assert policy._load_kinematic_button_window() == (12, 18)


def test_inference_kinematic_window_requires_digest_bound_contract():
    source_rel_z = np.asarray(
        [0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._motion_data = SimpleNamespace(
        has_object=True,
        object_pos_w=np.stack(
            [np.zeros_like(source_rel_z), np.zeros_like(source_rel_z), source_rel_z],
            axis=1,
        ),
        root_pos_w=np.zeros((source_rel_z.size, 3), dtype=np.float32),
        source_frame_count=source_rel_z.size,
        frame_count=source_rel_z.size,
        motion_path=Path("clip.npz"),
        source_sha256="a" * 64,
        source_size=100,
        fps=50.0,
    )
    policy._effective_motion_transition_settings = {
        "source_semantics": "single_clip_static",
        "contract_sha256": "b" * 64,
    }
    policy._motion_transition_prepend_steps = 0
    policy._onnx_metadata = {}

    with pytest.raises(RuntimeError, match="require a digest-bound"):
        policy._load_kinematic_button_window()


def test_inference_kinematic_window_checks_contract_before_object_availability():
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._motion_data = SimpleNamespace(
        has_object=False,
        object_pos_w=None,
    )
    policy._onnx_metadata = {}

    with pytest.raises(RuntimeError, match="require a digest-bound"):
        policy._load_kinematic_button_window()


@pytest.mark.parametrize(
    "metadata",
    [
        {EMBEDDED_BUTTON_WINDOW_CONTRACT_KEY: {}},
        {EMBEDDED_BUTTON_WINDOW_CONTRACT_SHA256_KEY: "a" * 64},
    ],
)
def test_inference_kinematic_window_rejects_partial_contract(metadata):
    source_rel_z = np.asarray(
        [0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._motion_data = SimpleNamespace(
        has_object=True,
        object_pos_w=np.stack(
            [np.zeros_like(source_rel_z), np.zeros_like(source_rel_z), source_rel_z],
            axis=1,
        ),
        root_pos_w=np.zeros((source_rel_z.size, 3), dtype=np.float32),
        source_frame_count=source_rel_z.size,
        frame_count=source_rel_z.size,
    )
    policy._effective_motion_transition_settings = {
        "source_semantics": "single_clip_static",
    }
    policy._motion_transition_prepend_steps = 0
    policy._onnx_metadata = metadata

    with pytest.raises(ValueError, match="must appear together"):
        policy._load_kinematic_button_window()


def test_inference_kinematic_window_rejects_active_motion_identity_mismatch():
    source_rel_z = np.asarray(
        [0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    source_window = kinematic_lift_window_from_rel_z_np(source_rel_z)
    contract, digest = build_kinematic_button_window_contract(
        clip_id="clip",
        source_motion_sha256="a" * 64,
        source_motion_size=100,
        source_frame_count=source_rel_z.size,
        motion_fps=50.0,
        source_window=source_window,
        motion_transition_contract_sha256="b" * 64,
        source_semantics="single_clip_static",
        effective_prepend_steps=0,
        effective_append_steps=0,
    )
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._motion_data = SimpleNamespace(
        has_object=True,
        object_pos_w=np.stack(
            [np.zeros_like(source_rel_z), np.zeros_like(source_rel_z), source_rel_z],
            axis=1,
        ),
        root_pos_w=np.zeros((source_rel_z.size, 3), dtype=np.float32),
        source_frame_count=source_rel_z.size,
        frame_count=source_rel_z.size,
        motion_path=Path("clip.npz"),
        source_sha256="c" * 64,
        source_size=100,
        fps=50.0,
    )
    policy._effective_motion_transition_settings = {
        "source_semantics": "single_clip_static",
        "contract_sha256": "b" * 64,
    }
    policy._motion_transition_prepend_steps = 0
    policy._onnx_metadata = {
        EMBEDDED_BUTTON_WINDOW_CONTRACT_KEY: contract,
        EMBEDDED_BUTTON_WINDOW_CONTRACT_SHA256_KEY: digest,
    }

    with pytest.raises(RuntimeError, match="source_motion_sha256 does not match"):
        policy._load_kinematic_button_window()


def test_static_splice_contract_requires_recomputed_materialized_window():
    kwargs = {
        "clip_id": "clip",
        "source_motion_sha256": "a" * 64,
        "source_motion_size": 100,
        "source_frame_count": 20,
        "motion_fps": 50.0,
        "source_window": (2, 12),
        "motion_transition_contract_sha256": "b" * 64,
        "source_semantics": "single_clip_static",
        "effective_prepend_steps": 10,
        "effective_append_steps": 10,
    }
    with pytest.raises(ValueError, match="recomputed"):
        build_kinematic_button_window_contract(**kwargs)

    contract, _ = build_kinematic_button_window_contract(
        **kwargs,
        # A static interpolation can itself satisfy the lift threshold; this
        # is intentionally not inferred by adding ten to the source t1.
        materialized_window=(0, 24),
    )
    assert contract["materialized_window"] == [0, 24]


def test_embedded_contract_rejects_digest_or_integer_drift():
    contract, digest = build_kinematic_button_window_contract(
        clip_id="clip",
        source_motion_sha256="a" * 64,
        source_motion_size=100,
        source_frame_count=20,
        motion_fps=50.0,
        source_window=(2, 12),
        motion_transition_contract_sha256="b" * 64,
        source_semantics="global_multi_clip_runtime",
        effective_prepend_steps=10,
        effective_append_steps=0,
    )
    metadata = {
        EMBEDDED_BUTTON_WINDOW_CONTRACT_KEY: copy.deepcopy(contract),
        EMBEDDED_BUTTON_WINDOW_CONTRACT_SHA256_KEY: digest,
    }
    metadata[EMBEDDED_BUTTON_WINDOW_CONTRACT_KEY]["source_window"][0] = 3
    metadata[EMBEDDED_BUTTON_WINDOW_CONTRACT_KEY]["materialized_window"][0] = 13

    with pytest.raises(ValueError, match="digest mismatch"):
        embedded_button_window_contract_from_metadata(metadata)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("version", True, "version"),
        ("source_semantics", ["global_multi_clip_runtime"], "source_semantics"),
    ],
)
def test_embedded_contract_rejects_bool_or_unhashable_schema_corruption(
    field,
    value,
    message,
):
    contract, digest = build_kinematic_button_window_contract(
        clip_id="clip",
        source_motion_sha256="a" * 64,
        source_motion_size=100,
        source_frame_count=20,
        motion_fps=50.0,
        source_window=(2, 12),
        motion_transition_contract_sha256="b" * 64,
        source_semantics="global_multi_clip_runtime",
        effective_prepend_steps=10,
        effective_append_steps=0,
    )
    contract[field] = value
    metadata = {
        EMBEDDED_BUTTON_WINDOW_CONTRACT_KEY: contract,
        EMBEDDED_BUTTON_WINDOW_CONTRACT_SHA256_KEY: digest,
    }

    with pytest.raises(ValueError, match=message):
        embedded_button_window_contract_from_metadata(metadata)


@pytest.mark.parametrize("value", ["KINEMATIC_LIFT", "unknown", 1, None])
def test_button_window_mode_validation_is_exact(value):
    with pytest.raises(ValueError, match="must be exactly"):
        validated_contact_aware_button_window_mode(
            {"contact_aware_button_window_mode": value}
        )


def test_button_window_mode_legacy_default_is_contact_interval():
    assert validated_contact_aware_button_window_mode({}) == "contact_interval"


def test_inference_motion_config_prefers_equal_canonical_nested_metadata():
    nested = {
        "contact_aware_button_window_mode": "kinematic_lift",
        "motion_file": "motion.npz",
    }
    metadata = {
        "motion_config": copy.deepcopy(nested),
        "experiment_config": {
            "command": {
                "setup_terms": {
                    "motion_command": {"params": {"motion_config": nested}}
                }
            }
        },
    }

    assert WholeBodyTrackingPolicy._extract_motion_config(metadata) is nested


def test_inference_motion_config_rejects_top_level_nested_drift():
    metadata = {
        "motion_config": {"contact_aware_button_window_mode": "contact_interval"},
        "experiment_config": {
            "command": {
                "setup_terms": {
                    "motion_command": {
                        "params": {
                            "motion_config": {
                                "contact_aware_button_window_mode": "kinematic_lift"
                            }
                        }
                    }
                }
            }
        },
    }

    with pytest.raises(ValueError, match="metadata disagree"):
        WholeBodyTrackingPolicy._extract_motion_config(metadata)


def test_inference_motion_config_preserves_legacy_top_level_only_metadata():
    legacy = {"contact_aware_button_window_mode": "contact_interval"}

    assert WholeBodyTrackingPolicy._extract_motion_config(
        {"motion_config": legacy}
    ) is legacy
