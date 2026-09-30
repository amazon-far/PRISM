#!/usr/bin/env python3
"""Select a MuJoCo inference preset by authenticating the full ONNX contract."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping

import onnx

from holosoma_inference.config.config_values import inference as inference_values
from holosoma_inference.utils.policy_contract import PolicyContractError, validate_onnx_policy_contract


# Deliberately omit aliases whose observation contracts are identical.  A
# unique semantic match maps to one canonical launcher name.
_CANONICAL_CANDIDATES = (
    "g1-29dof-wbt",
    "g1-29dof-wbt-object-generalist",
    "g1-29dof-wbt-w-object-history1",
    "g1-29dof-wbt-w-object-legacy",
    "g1-29dof-wbt-object-velocity-generalist",
    "g1-29dof-wbt-object-distill",
    "g1-29dof-wbt-object-as-depth-distill",
    "g1-29dof-wbt-object-as-contact-aware-depth-distill",
    "g1-29dof-wbt-object-as-contact-aware-history5-depth-distill",
    "g1-29dof-wbt-object-contact-aware-depth-distill",
    "g1-29dof-wbt-object-contact-aware-drop-button-depth-distill",
    "g1-29dof-wbt-object-contact-aware-dual-button-depth-distill",
    "g1-29dof-wbt-object-mocap-distill",
)


def _load_metadata(model: onnx.ModelProto) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for prop in model.metadata_props:
        try:
            metadata[prop.key] = json.loads(prop.value)
        except (TypeError, ValueError):
            metadata[prop.key] = prop.value
    experiment = metadata.get("experiment_config")
    if not isinstance(experiment, Mapping):
        raise ValueError(
            "Automatic inference-preset selection requires complete experiment_config metadata; "
            "shape-only selection is unsafe."
        )
    return metadata


def _tensor_shapes(values) -> dict[str, list[object]]:
    def dimension_value(dim) -> object:
        kind = dim.WhichOneof("value")
        if kind == "dim_value":
            return dim.dim_value
        if kind == "dim_param":
            return dim.dim_param
        return None

    return {
        value.name: [dimension_value(dim) for dim in value.type.tensor_type.shape.dim]
        for value in values
    }


def _tensor_types(values) -> dict[str, str]:
    return {
        value.name: (
            "tensor(float)"
            if value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
            else f"tensor({onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type).lower()})"
        )
        for value in values
    }


def infer_inference_config(model: onnx.ModelProto) -> str:
    metadata = _load_metadata(model)
    input_shapes = _tensor_shapes(model.graph.input)
    output_shapes = _tensor_shapes(model.graph.output)
    input_types = _tensor_types(model.graph.input)
    output_types = _tensor_types(model.graph.output)

    matches: list[str] = []
    failures: dict[str, str] = {}
    for preset_name in _CANONICAL_CANDIDATES:
        preset = inference_values.DEFAULTS[preset_name]
        try:
            validate_onnx_policy_contract(
                metadata=metadata,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                input_types=input_types,
                output_types=output_types,
                observation=preset.observation,
                runtime_dof_names=preset.robot.dof_names,
                runtime_default_dof_angles=preset.robot.default_dof_angles,
                runtime_motor_effort_limits=preset.robot.motor_effort_limit,
                runtime_joint2motor=preset.robot.joint2motor,
            )
        except PolicyContractError as exc:
            failures[preset_name] = str(exc)
        else:
            matches.append(preset_name)

    if len(matches) != 1:
        actor = (
            metadata.get("experiment_config", {})
            .get("algo", {})
            .get("config", {})
            .get("module_dict", {})
            .get("actor", {})
        )
        actor_groups = actor.get("input_dim") if isinstance(actor, Mapping) else None
        if not matches:
            summaries = "; ".join(
                f"{name}: {failures[name]}" for name in _CANONICAL_CANDIDATES if name in failures
            )
            raise ValueError(
                "No inference preset matches the complete ONNX policy contract: "
                f"actor_input_dim={actor_groups!r}, inputs={input_shapes}. Candidate failures: {summaries}"
            )
        raise ValueError(
            "Inference-preset selection is ambiguous even after full metadata validation: "
            f"matches={matches}, actor_input_dim={actor_groups!r}, inputs={input_shapes}."
        )
    return matches[0]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} POLICY.onnx")
    model = onnx.load(sys.argv[1])
    try:
        inferred = infer_inference_config(model)
    except ValueError as exc:
        raise SystemExit(f"{exc} (model={sys.argv[1]})") from exc
    print(inferred)


if __name__ == "__main__":
    main()
