from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_decoupled_root_command_bank.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("source_digest_field", ["payload_digest", "source_digest"])
def test_portable_bank_preserves_assets_contacts_and_source_arrays(
    tmp_path: Path, source_digest_field: str
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "_mesh_assets").mkdir()
    (source / "_single_slot_urdfs").mkdir()
    (source / "contacts" / "clips" / "0000_box_turn").mkdir(parents=True)
    mesh = source / "_mesh_assets" / "box.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="ascii")
    urdf = source / "_single_slot_urdfs" / "box_turn.urdf"
    urdf.write_text(
        "<robot name='box'><link name='baseLink'>"
        "<visual><geometry><mesh filename='../_mesh_assets/box.obj'/></geometry></visual>"
        "<collision><geometry><mesh filename='../_mesh_assets/box.obj'/></geometry></collision>"
        "</link></robot>\n",
        encoding="ascii",
    )
    contact = source / "contacts" / "clips" / "0000_box_turn" / "drop_interval.npy"
    np.save(contact, np.asarray([20, 29], dtype=np.int64))
    object_map = {
        "clips": {
            "box_turn": {
                "object_name": "box_turn",
                "object_urdf_path": "_single_slot_urdfs/box_turn.urdf",
            }
        }
    }
    (source / "_clip_object_urdf_map.json").write_text(
        json.dumps(object_map, sort_keys=True) + "\n", encoding="ascii"
    )

    frames = 30
    body_pos = np.zeros((frames, 1, 3), dtype=np.float32)
    body_pos[:, 0, 2] = 0.8
    body_pos[5:18, 0, 0] = np.linspace(0.0, 1.0, 13)
    body_pos[18:, 0, 0] = 1.0
    body_pos[18:, 0, 1] = np.linspace(0.0, 1.0, 12)
    body_quat = np.zeros((frames, 1, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    body_quat[18:, 0, 0] = np.cos(np.pi / 4.0)
    body_quat[18:, 0, 3] = np.sin(np.pi / 4.0)
    object_pos = body_pos[:, 0].copy()
    object_pos[:5, 2] = 0.0
    object_pos[5:, 2] = 0.4
    arrays = {
        "fps": np.asarray(50.0, dtype=np.float32),
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "object_pos_w": object_pos,
        "drop_label": np.arange(frames, dtype=np.int32),
    }
    np.savez_compressed(source / "box_turn.npz", **arrays)
    source_manifest = {
        "clip_count": 1,
        source_digest_field: "fixture-payload-digest",
    }
    source_manifest_path = source / "manifest.json"
    source_manifest_path.write_text(
        json.dumps(source_manifest, sort_keys=True) + "\n", encoding="ascii"
    )
    source_manifest_sha = _sha256(source_manifest_path)

    output = tmp_path / "derived"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source",
            str(source),
            "--output",
            str(output),
            "--copy-portable-source-tree",
            "--expected-clip-count",
            "1",
            "--expected-category-counts-json",
            '{"box":1,"ball":0,"barrel":0,"bin":0}',
            "--expected-source-payload-digest",
            "fixture-payload-digest",
            "--expected-source-manifest-sha256",
            source_manifest_sha,
            "--smoothing-steps",
            "1",
            "--rdp-epsilon-m",
            "0.005",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert _sha256(output / "_mesh_assets" / "box.obj") == _sha256(mesh)
    assert _sha256(output / "_single_slot_urdfs" / "box_turn.urdf") == _sha256(urdf)
    assert _sha256(output / "contacts" / "clips" / "0000_box_turn" / "drop_interval.npy") == _sha256(
        contact
    )
    with np.load(output / "box_turn.npz", allow_pickle=False) as derived:
        for key, expected in arrays.items():
            np.testing.assert_array_equal(derived[key], expected)
        command = derived["policy_command_xy_yaw"]
        assert np.all(command[:, 1] == 0.0)
        assert not np.any((command[:, 0] != 0.0) & (command[:, 2] != 0.0))
    manifest = json.loads((output / "manifest.json").read_text(encoding="ascii"))
    assert manifest["training_behavior"]["drop_button_and_contact_labels_unchanged"] is True
    assert manifest["source_payload_digest"] == "fixture-payload-digest"
    source_paths = [record["path"] for record in manifest["source_records"]]
    generated_paths = [record["path"] for record in manifest["generated_records"]]
    assert len(source_paths) == len(set(source_paths))
    assert len(generated_paths) == len(set(generated_paths))
