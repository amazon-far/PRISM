"""Offline same-checkpoint depth ablation. Never opens a robot/camera interface."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np

from holosoma.utils.deployment_audit import depth_digest, sha256_file


MAY28_COMMIT = "c416c75ac5bd09c3449336ba3de4b466874ff728"


def depth_variant(raw, cfg, variant):
    """The three legacy real_d435i paths, with fixed historical OpenCV semantics."""
    near, far = cfg["near_clip"], cfg["far_clip"]
    value = np.asarray(raw, dtype=np.float32).copy()
    if variant != "may28":
        value = np.where((value == 0) | ~np.isfinite(value) | (value > far), far, value)
        value = value.clip(near, far)
    value = value[cfg["crop_y_start"]:cfg["crop_y_end"], cfg["crop_x_start"]:cfg["crop_x_end"]]
    # May's third positional argument was dst, not interpolation: effective linear.
    interpolation = cv2.INTER_CUBIC if variant == "current" else cv2.INTER_LINEAR
    value = cv2.resize(value, (cfg["resized_width"], cfg["resized_height"]), interpolation=interpolation)
    value = value.clip(near, far)
    minimum = near if cfg["min_valid_depth"] is None else cfg["min_valid_depth"]
    value[value < minimum] = far
    return ((value - near) / (far - near) - 0.5).reshape(1, -1)


def load_record(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        arrays = {name: archive[name] for name in archive.files if name != "metadata_json"}
    return arrays, metadata


def compare(policy_dir, depth_dir, model, max_frames=200, recorded_preprocess="current"):
    import onnxruntime as ort

    policy = json.loads((policy_dir / "session.json").read_text())
    depth = json.loads((depth_dir / "session.json").read_text())
    if policy["role"] != "policy" or depth["role"] != "depth":
        raise ValueError("Expected policy and depth evidence directories")
    if sha256_file(model) != policy["model_sha256"]:
        raise ValueError("Model SHA256 differs from recorded policy")
    if any(policy["identity"][key] != depth["identity"][key] for key in ("hostname", "boot_id")):
        raise ValueError("Depth and policy must come from the same host boot")
    if depth.get("training_depth_profile") is not None:
        raise ValueError("This May-vs-current ablation requires the legacy real_d435i path, not real_training")
    cfg = depth["image_server_config"]
    expected = {"camera_type": "realsense", "depth_source": "depth", "near_clip": 0.3, "far_clip": 3.0,
                "crop_y_start": 16, "crop_y_end": None, "crop_x_start": 32, "crop_x_end": -32,
                "resized_width": 87, "resized_height": 58, "min_valid_depth": None}
    if any(cfg[key] != value for key, value in expected.items()):
        raise ValueError("Evidence does not use the historical CORL real_d435i preprocessing configuration")

    index = defaultdict(list)
    for path in sorted(depth_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata_json"]))
            if len(meta["capture"]) != 1:
                raise ValueError("This comparison requires one depth camera")
            capture = next(iter(meta["capture"].values()))
            age = capture["sensor_age_at_receive_ms"]
            if age is None or not np.isfinite(age) or age < 0:
                continue
            captured = capture["received_monotonic"] - age / 1000.0
            index[depth_digest(data["processed_depth"])].append((captured, path))

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    output = next((item.name for item in session.get_outputs() if item.name in ("actions", "action")), None)
    if output is None:
        raise ValueError("No action output")
    rows = []
    skipped = defaultdict(int)
    for path in sorted(policy_dir.glob("*.npz")):
        values, meta = load_record(path)
        if not meta["use_policy_action"] or meta["get_ready_state"] or meta["drop_button"] >= 0.5:
            skipped["not_active_carry_attempt"] += 1
            continue
        inputs = {key.removeprefix("input__"): value for key, value in values.items() if key.startswith("input__")}
        if "perception_obs" not in inputs or any(not np.isfinite(value).all() for value in inputs.values()):
            skipped["missing_or_nonfinite_input"] += 1
            continue
        read_time = meta["depth_read_monotonic"]
        candidates = [(timestamp, raw_path) for timestamp, raw_path in index[depth_digest(inputs["perception_obs"])]]
        candidates = [(timestamp, raw_path) for timestamp, raw_path in candidates
                      if read_time is not None and 0 <= read_time - timestamp <= 0.25]
        if not candidates:
            skipped["no_exact_fresh_sampled_raw_frame"] += 1
            continue
        if len(candidates) != 1:
            skipped["ambiguous_identical_depth_samples"] += 1
            continue
        captured, raw_path = max(candidates, key=lambda item: item[0])
        native, _ = load_record(raw_path)
        raw = native["raw_depth"]
        if raw.shape != (1, 480, 848):
            raise ValueError(f"Unexpected native raw shape: {raw.shape}")
        baseline = session.run([output], inputs)[0]
        parity_error = float(np.max(np.abs(baseline - values["policy_action"])))
        if parity_error > 1e-3:
            raise ValueError(f"Recorded input/action replay parity failed: {path} maxabs={parity_error}")
        variants = {name: depth_variant(raw[0], cfg, name) for name in ("may28", "pre_sept17", "current")}
        processing_error = float(np.max(np.abs(variants[recorded_preprocess] - inputs["perception_obs"])))
        if processing_error > 2e-5:
            raise ValueError(f"Raw-to-recorded depth parity failed: {processing_error}")
        row = {"policy_record": path.name, "raw_record": raw_path.name,
               "depth_age_ms": 1000 * (read_time - captured), "action_replay_maxabs": parity_error,
               "raw_zero_fraction": float(np.mean(raw == 0)), "variants": {}}
        for name, alternative in variants.items():
            if not np.isfinite(alternative).all():
                raise ValueError(f"Nonfinite alternative depth: {name}")
            action = session.run([output], {**inputs, "perception_obs": alternative})[0]
            delta = action - baseline
            row["variants"][name] = {
                "depth_rms": float(np.sqrt(np.mean((alternative - inputs["perception_obs"]) ** 2))),
                "action_rms": float(np.sqrt(np.mean(delta ** 2))),
                "action_maxabs": float(np.max(np.abs(delta))),
                "q_target_delta_max_rad": float(np.max(np.abs(delta * values["action_scales"]))),
            }
        rows.append(row)
        if len(rows) >= max_frames:
            break
    if not rows:
        raise ValueError(f"No active, exact-matched, fresh frames; cannot test hypothesis. Skipped: {dict(skipped)}")
    return {
        "semantics": "offline_one_step_sensitivity_not_closed_loop_success",
        "recorded_preprocess": recorded_preprocess,
        "may28_commit": MAY28_COMMIT, "checkpoint_sha256": policy["model_sha256"],
        "policy_manifest": json.loads((policy_dir / "manifest.json").read_text()),
        "depth_manifest": json.loads((depth_dir / "manifest.json").read_text()),
        "matched_frames": len(rows), "skipped": dict(skipped),
        "mean_action_rms": {name: float(np.mean([row["variants"][name]["action_rms"] for row in rows]))
                            for name in ("may28", "pre_sept17", "current")},
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-evidence", required=True, type=Path)
    parser.add_argument("--depth-evidence", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--recorded-preprocess", choices=("may28", "pre_sept17", "current"), required=True)
    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error("--max-frames must be positive")
    result = compare(args.policy_evidence, args.depth_evidence, args.model, args.max_frames, args.recorded_preprocess)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
