import json
from types import SimpleNamespace

import numpy as np

from holosoma.utils.deployment_audit import DeploymentAudit, create_deployment_audit
from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy


def test_recording_does_not_change_inputs_actions_commands_or_gains(tmp_path):
    policy = WholeBodyTrackingPolicy.__new__(WholeBodyTrackingPolicy)
    policy.motion_clip_progressing = False
    policy.onnx_input_names = ["actor_obs", "perception_obs"]
    policy._policy_returns_reference_outputs = False
    policy._use_motion_data_as_q_target = False
    actor = np.arange(94, dtype=np.float32).reshape(1, -1) / 100
    image = np.full((1, 5046), 0.25, np.float32)
    policy.prepare_obs_for_rl = lambda state: {"actor_obs": actor, "perception_obs": image}
    action = np.arange(29, dtype=np.float32).reshape(1, -1) / 20
    policy.policy = lambda inputs: action.copy()
    policy.policy_action_scales = np.full((1, 29), 0.25, np.float32)
    policy.default_dof_angles = np.full((1, 29), 0.1, np.float32)
    policy._write_policy_debug = lambda *args: None
    policy.robot_config = SimpleNamespace(motor_kp=[30.] * 29, motor_kd=[2.] * 29)
    policy.interface = SimpleNamespace(kp_level=0.75, kd_level=1.)
    policy._manual_sparse_root_command_offset = np.array([[0.15, 0., 0.]], np.float32)
    policy._joystick_sparse_root_command_offset = np.zeros((1, 3), np.float32)
    policy._drop_button_command = 0.
    policy.use_policy_action, policy.get_ready_state = True, False
    policy._motion_frame_index = lambda: 0
    state = np.zeros((1, 71), np.float32)
    without = policy.rl_inference(state).copy()
    policy._deployment_audit = DeploymentAudit(tmp_path / "policy", {})
    with_recording = policy.rl_inference(state)
    policy.close_deployment_audit()
    np.testing.assert_array_equal(with_recording, without)
    np.testing.assert_array_equal(policy.last_policy_action, action)
    with np.load(tmp_path / "policy/00000000.npz", allow_pickle=False) as record:
        np.testing.assert_array_equal(record["input__actor_obs"], actor)
        np.testing.assert_array_equal(record["input__perception_obs"], image)
        np.testing.assert_array_equal(record["policy_action"], action)
        np.testing.assert_array_equal(record["requested_q_target"], without + policy.default_dof_angles)
        np.testing.assert_array_equal(record["motor_kp"], [22.5] * 29)
        np.testing.assert_array_equal(record["manual_command"], policy._manual_sparse_root_command_offset)


def test_actual_deployment_config_and_identity_are_serializable(tmp_path, monkeypatch):
    from holosoma_inference.config.config_values.inference import (
        g1_root_pos_contact_aware_drop_button_actions_no_linvel_h1,
    )

    monkeypatch.setenv("HOLOSOMA_DEPLOYMENT_AUDIT_DIR", str(tmp_path))
    recorder = create_deployment_audit("policy", {
        "config": g1_root_pos_contact_aware_drop_button_actions_no_linvel_h1,
    })
    recorder.close()
    session = json.loads((recorder.directory / "session.json").read_text())
    assert len(session["identity"]["commit"]) == 40
    assert session["config"]["camera"]["props"]["resized_width"] == 87
