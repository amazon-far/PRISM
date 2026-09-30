from __future__ import annotations

import argparse
import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh


def _load_builder_module():
    script = Path(__file__).resolve().parents[2] / "scripts" / "build_mesh_physics_object_bank.py"
    spec = importlib.util.spec_from_file_location("build_mesh_physics_object_bank", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_category_mass_contract() -> None:
    builder = _load_builder_module()
    expected = {
        "ball": 0.5,
        "bin": 1.0,
        "barrel": 1.5,
        "box": 1.0,
    }
    for category, mass_kg in expected.items():
        prior = builder.DEFAULT_MASS_PRIORS[category]
        assert prior == {"default": mass_kg, "min": mass_kg, "max": mass_kg}
        assert builder._base_mass_for_category(category, builder.DEFAULT_MASS_PRIORS) == mass_kg


def test_source_mass_mode_changes_only_inertial_physics(tmp_path: Path) -> None:
    builder = _load_builder_module()
    input_bank = tmp_path / "input"
    input_bank.mkdir()

    mesh = trimesh.creation.box(extents=[0.4, 0.2, 0.1])
    expected_com = np.asarray([0.03, -0.02, 0.01], dtype=np.float64)
    mesh.apply_translation(expected_com)
    mesh_path = input_bank / "offset_box.obj"
    mesh.export(mesh_path)
    collision_mesh = trimesh.creation.box(extents=[1.2, 0.8, 0.6])
    collision_mesh_path = input_bank / "collision_proxy.obj"
    collision_mesh.export(collision_mesh_path)

    urdf_path = input_bank / "offset_box.urdf"
    urdf_path.write_text(
        """<?xml version="1.0"?>
<robot name="offset_box">
  <link name="baseLink">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="offset_box.obj" scale="1 1 1"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="collision_proxy.obj" scale="1 1 1"/></geometry>
    </collision>
  </link>
</robot>
""",
        encoding="utf-8",
    )
    object_map = input_bank / "_clip_object_urdf_map.json"
    object_map.write_text(
        json.dumps(
            {
                "clips": {
                    "box_test": {
                        "object_name": "offset_box",
                        "object_urdf_path": str(urdf_path),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (input_bank / "_scientific_stale_view").mkdir()
    (input_bank / "_single_slot_motion_bank").mkdir()
    (input_bank / "nfs_package_manifest.json").write_text("stale", encoding="utf-8")
    (input_bank / "realmesh_rollout_manifest.json").write_text(
        '{"teacher_checkpoint": "model_05000.pt"}\n',
        encoding="utf-8",
    )

    output_bank = tmp_path / "output"
    builder.build_bank(
        argparse.Namespace(
            input_bank=str(input_bank),
            object_map=str(object_map),
            output_bank=str(output_bank),
            mass_mode="source_urdf",
            mass_priors_json="",
            asset_mode="symlink",
            non_watertight_fallback="error",
            overwrite=False,
        )
    )

    generated = ET.parse(output_bank / "_mesh_physics_urdfs" / "offset_box.urdf").getroot()
    assert float(generated.find(".//inertial/mass").get("value")) == 0.1
    actual_com = np.fromstring(generated.find(".//inertial/origin").get("xyz"), sep=" ")
    np.testing.assert_allclose(actual_com, expected_com, atol=1.0e-9)

    for tag in ("visual", "collision"):
        origin = generated.find(f".//{tag}/origin")
        assert origin.get("xyz") == "0 0 0"
        assert origin.get("rpy") == "0 0 0"

    inertia = generated.find(".//inertial/inertia")
    actual_diag = np.asarray([float(inertia.get(key)) for key in ("ixx", "iyy", "izz")])
    expected_diag = 0.1 / 12.0 * np.asarray([0.2**2 + 0.1**2, 0.4**2 + 0.1**2, 0.4**2 + 0.2**2])
    np.testing.assert_allclose(actual_diag, expected_diag, rtol=1.0e-7, atol=1.0e-12)

    manifest = json.loads((output_bank / "_mesh_physics_manifest.json").read_text(encoding="utf-8"))
    assert manifest["mass_mode"] == "source_urdf"
    assert manifest["mass_priors"] == {}
    clip = json.loads((output_bank / "_clip_object_urdf_map.json").read_text(encoding="utf-8"))["clips"][
        "box_test"
    ]
    assert clip["mesh_physics_mass_mode"] == "source_urdf"
    assert clip["mesh_physics_base_mass_kg"] == 0.1
    report = (output_bank / "_mesh_physics_report.csv").read_text(encoding="utf-8")
    assert ",visual," in report
    assert not (output_bank / "_scientific_stale_view").exists()
    assert not (output_bank / "_single_slot_motion_bank").exists()
    assert not (output_bank / "nfs_package_manifest.json").exists()
    rollout_manifest = output_bank / "realmesh_rollout_manifest.json"
    assert rollout_manifest.is_file()
    assert not rollout_manifest.is_symlink()
    assert json.loads(rollout_manifest.read_text(encoding="utf-8")) == {
        "teacher_checkpoint": "model_05000.pt"
    }
