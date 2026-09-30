import importlib.util
import ast
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import pytest

from holosoma.utils.deployment_audit import DeploymentAudit, create_deployment_audit, sha256_file


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("depth_evidence", ROOT / "scripts/compare_real_depth_evidence.py")
compare_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_module)


def config():
    from dataclasses import asdict
    from holosoma.config_values.image_server import real_d435i

    return asdict(real_d435i)


def test_recorder_disabled_has_no_side_effects(tmp_path, monkeypatch):
    monkeypatch.delenv("HOLOSOMA_DEPLOYMENT_AUDIT_DIR", raising=False)
    assert create_deployment_audit("policy", {}) is None
    assert not list(tmp_path.iterdir())


def test_copies_losslessly_bounded_and_closed(tmp_path):
    recorder = DeploymentAudit(tmp_path / "capture", {}, every=2, limit=2)
    data = np.arange(5046, dtype=np.float32).reshape(1, -1)
    original = data.copy()
    recorder.record({"input__perception_obs": data}, {"drop_button": 0})
    data[:] = -100
    for _ in range(6):
        recorder.record({"input__perception_obs": data}, {})
    recorder.close()
    files = sorted(recorder.directory.glob("*.npz"))
    assert [file.name for file in files] == ["00000000.npz", "00000002.npz"]
    with np.load(files[0], allow_pickle=False) as record:
        np.testing.assert_array_equal(record["input__perception_obs"], original)
    manifest = json.loads((recorder.directory / "manifest.json").read_text())
    assert manifest["complete_sampled_window"]
    assert manifest["limit_reached"] and manifest["written"] == 2


def test_overflow_is_nonblocking_and_explicitly_incomplete(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = np.savez_compressed

    def slow_write(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        original(*args, **kwargs)

    monkeypatch.setattr(np, "savez_compressed", slow_write)
    recorder = DeploymentAudit(tmp_path / "capture", {}, queue_size=1)
    recorder.record({"a": [1]}, {})
    assert entered.wait(2)
    recorder.record({"a": [2]}, {})
    started = time.monotonic()
    recorder.record({"a": [3]}, {})
    assert time.monotonic() - started < 0.1
    assert recorder.dropped == 1
    release.set()
    recorder.close()
    manifest = json.loads((recorder.directory / "manifest.json").read_text())
    assert not manifest["complete_sampled_window"]
    assert manifest["dropped"] == 1


def test_write_error_does_not_change_or_block_inputs(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", fail)
    recorder = DeploymentAudit(tmp_path / "capture", {})
    data = np.ones((1, 29), np.float32)
    recorder.record({"action": data}, {})
    recorder.close()
    np.testing.assert_array_equal(data, np.ones((1, 29)))
    manifest = json.loads((recorder.directory / "manifest.json").read_text())
    assert "disk full" in manifest["error"]
    assert not manifest["complete_sampled_window"]


def test_may28_and_current_paths_are_not_equivalent():
    cfg = config()
    raw = np.zeros((480, 848), np.float32)
    np.testing.assert_array_equal(compare_module.depth_variant(raw, cfg, "may28"), -0.5)
    np.testing.assert_allclose(compare_module.depth_variant(raw, cfg, "current"), 0.5, atol=1e-7, rtol=0)
    raw = np.random.default_rng(42).uniform(0.3, 3, raw.shape).astype(np.float32)
    old = cv2.resize(raw[16:, 32:-32], (87, 58), cv2.INTER_CUBIC)
    linear = cv2.resize(raw[16:, 32:-32], (87, 58), interpolation=cv2.INTER_LINEAR)
    np.testing.assert_array_equal(old, linear)
    previous = compare_module.depth_variant(raw, cfg, "pre_sept17")
    current = compare_module.depth_variant(raw, cfg, "current")
    assert np.max(np.abs(previous - current)) > 0.01
    # Execute the actual pure method without importing legacy GPU/ZED backends.
    from types import SimpleNamespace

    tree = ast.parse((ROOT / "src/holosoma/holosoma/sensors/image_server.py").read_text())
    server = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ImageServer")
    method = next(node for node in server.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_resize_clip_expand_transpose")
    namespace = {"cv2": cv2, "np": np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "image_server.py", "exec"), namespace)
    actual = namespace[method.name](SimpleNamespace(cfg=SimpleNamespace(**cfg)), raw)
    np.testing.assert_array_equal(actual.reshape(1, -1), compare_module.depth_variant(raw, cfg, "may28"))


def make_evidence(tmp_path):
    import onnx
    from onnx import TensorProto, helper

    model_path = tmp_path / "tiny.onnx"
    graph = helper.make_graph([
        helper.make_node("ReduceMean", ["perception_obs"], ["mean"], axes=[1], keepdims=1),
        helper.make_node("Add", ["actor_obs", "mean"], ["actions"]),
    ], "depth_probe", [helper.make_tensor_value_info("actor_obs", TensorProto.FLOAT, [1, 29]),
                       helper.make_tensor_value_info("perception_obs", TensorProto.FLOAT, [1, 5046])],
       [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 29])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    onnx.save(model, model_path)
    common = {"identity": {"hostname": "test", "boot_id": "test"}}
    policy = DeploymentAudit(tmp_path / "policy", {**common, "role": "policy",
                             "model_sha256": sha256_file(model_path)})
    depth = DeploymentAudit(tmp_path / "depth", {**common, "role": "depth",
                            "image_server_config": config(), "training_depth_profile": None})
    raw = np.ones((1, 480, 848), np.float32)
    raw[:, :, :400] = 0
    processed = compare_module.depth_variant(raw[0], config(), "current")
    depth.record({"raw_depth": raw, "processed_depth": processed}, {
        "capture": {"d435": {"sensor_age_at_receive_ms": 20, "received_monotonic": 10.02}},
    })
    policy.record({"input__actor_obs": np.zeros((1, 29), np.float32),
                   "input__perception_obs": processed,
                   "policy_action": np.full((1, 29), processed.mean(), np.float32),
                   "action_scales": np.full((1, 29), 0.25, np.float32)}, {
                       "use_policy_action": True, "get_ready_state": False, "drop_button": 0,
                       "depth_read_monotonic": 10.12,
                   })
    policy.close()
    depth.close()
    return policy.directory, depth.directory, model_path


def test_offline_replay_parity_and_depth_ablation(tmp_path):
    paths = make_evidence(tmp_path)
    report = compare_module.compare(*paths)
    assert report["matched_frames"] == 1
    assert report["mean_action_rms"]["current"] < 1e-6
    assert report["mean_action_rms"]["may28"] > 0.1
    assert abs(report["rows"][0]["depth_age_ms"] - 120) < 1e-5
    paths[2].write_bytes(b"wrong model")
    with pytest.raises(ValueError, match="SHA256"):
        compare_module.compare(*paths)


@pytest.mark.parametrize("change", ["stale", "drop", "parity", "boot"])
def test_offline_rejects_unverifiable_evidence(tmp_path, change):
    policy, depth, model = make_evidence(tmp_path)
    if change == "boot":
        metadata_path = depth / "session.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["identity"]["boot_id"] = "another_boot"
        metadata_path.write_text(json.dumps(metadata))
    else:
        path = policy / "00000000.npz"
        arrays, metadata = compare_module.load_record(path)
        if change == "stale":
            metadata["depth_read_monotonic"] = 11.0
        elif change == "drop":
            metadata["drop_button"] = 1
        else:
            arrays["policy_action"] += 1
        np.savez(path, **arrays, metadata_json=np.asarray(json.dumps(metadata)))
    with pytest.raises(ValueError):
        compare_module.compare(policy, depth, model)
