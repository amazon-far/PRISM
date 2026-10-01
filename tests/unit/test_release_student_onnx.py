"""Verify the bundled deployment model without opening camera or robot interfaces."""
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import onnx

from holosoma_inference.config.config_values.inference import (
    g1_root_pos_contact_aware_drop_button_actions_no_linvel_h1 as preset,
)
from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "_ckpts/student_28000.onnx"


def test_bundled_model_integrity_interface_and_public_metadata():
    entry = json.loads((ROOT / "_ckpts/manifest.json").read_text())["checkpoints"][0]
    sha = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    assert sha == entry["sha256"]
    assert MODEL.stat().st_size == entry["size_bytes"]
    assert (ROOT / "_ckpts/SHA256SUMS").read_text() == f"{sha}  {MODEL.name}\n"
    model = onnx.load(MODEL)
    onnx.checker.check_model(model, full_check=True)
    metadata = {item.key: json.loads(item.value) for item in model.metadata_props}
    assert len(metadata) == len(model.metadata_props)
    assert not {"wandb_run_path", "robot_urdf_path"} & metadata.keys()
    assert set(metadata["training_provenance"]) <= {
        "motion_shard_manifest_sha256", "contact_sidecar_manifest_sha256",
    }
    assert metadata["iteration"] == entry["completed_iteration"] == 27999
    assert metadata["release_checkpoint"]["checkpoint_step"] == entry["checkpoint_step"] == 28000
    assert metadata["release_checkpoint"]["sha256"] == entry["pytorch_checkpoint"]["sha256"]
    assert metadata["experiment_config"]["logger"]["type"] == "disabled"
    assert ET.fromstring(metadata["robot_urdf"]).tag == "robot"
    assert len(metadata["dof_names"]) == len(metadata["kp"]) == len(metadata["kd"]) == 29
    assert metadata["onnx_validation_contract"]["pytorch_vs_ort"] is True
    assert metadata["onnx_validation_contract"]["rtol"] == 1e-4
    # Inspect protobuf strings, including graph and URDF text, for private paths.
    text = str(model)
    assert not re.search(r"/(?:home|data|mnt|tmp|opt)/|wandb://|file://", text)
    for values, expected in ((model.graph.input, entry["inputs"]), (model.graph.output, entry["outputs"])):
        actual = {
            value.name: [d.dim_param if d.dim_param else d.dim_value for d in value.type.tensor_type.shape.dim]
            for value in values
        }
        assert actual == expected


def test_bundled_model_loads_in_the_real_deployment_adapter():
    policy = WholeBodyTrackingPolicy.__new__(WholeBodyTrackingPolicy)
    policy.config = preset
    policy._init_robot_config(preset.robot)
    policy._init_obs_config()
    policy.policy_action_scale = preset.task.policy_action_scale
    policy._motion_root_pos_w = None
    policy._contact_aware_peak_height_alpha = 0.91
    policy._contact_aware_peak_height_smoothing_steps = 5
    policy.setup_policy(str(MODEL))
    policy.interface = SimpleNamespace(update_config=lambda *args: None)
    policy._resolve_control_gains()
    assert policy._external_sparse_root_command_mode
    assert policy.num_dofs == 29
    inputs = {"actor_obs": np.zeros((1, 94), np.float32),
              "perception_obs": np.zeros((1, 5046), np.float32)}
    action = policy.policy(inputs)
    assert action.shape == (1, 29)
    assert np.isfinite(action).all()
