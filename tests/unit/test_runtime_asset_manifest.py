from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

import holosoma.utils.runtime_asset_manifest as runtime_asset_manifest_module
from holosoma.config_types.terrain import MeshType, SpawnCfg, TerrainManagerCfg, TerrainTermCfg
from holosoma.utils.runtime_asset_manifest import (
    build_urdf_asset_manifest,
    build_runtime_asset_manifest,
    finalize_runtime_asset_provenance,
    object_urdf_conversion_cache_key,
    persist_runtime_asset_manifest,
    runtime_asset_manifest_sha256,
)
from holosoma.utils.training_provenance import (
    ENV_NAME,
    EXECUTION_RUNTIME_KEY,
    RUNTIME_ASSET_DIGEST_KEY,
    RUNTIME_ASSET_MANIFEST_KEY,
    RUNTIME_ASSET_PHASE_FINAL,
    RUNTIME_ASSET_PHASE_KEY,
    RUNTIME_ASSET_PHASE_PENDING,
    SEMANTIC_ENVIRONMENT_FIELDS,
    SEMANTIC_ENVIRONMENT_KEY,
    disabled_checkpoint_sha256,
    pending_runtime_asset_manifest_sha256,
    validate_training_provenance,
)


def _write_assets(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    asset_root = tmp_path / "robots"
    robot_dir = asset_root / "g1"
    mesh_dir = robot_dir / "meshes"
    texture_dir = robot_dir / "textures"
    mesh_dir.mkdir(parents=True)
    texture_dir.mkdir()
    ordinary_mesh = mesh_dir / "body.STL"
    camera_mesh = mesh_dir / "d435_self.STL"
    texture = texture_dir / "albedo.png"
    ordinary_mesh.write_bytes(b"solid body\nendsolid body\n")
    camera_mesh.write_bytes(b"solid camera\nendsolid camera\n")
    texture.write_bytes(b"not-a-real-png-but-content-addressed")
    urdf = robot_dir / "robot.urdf"
    urdf.write_text(
        """<?xml version='1.0'?>
<robot name='test'>
  <link name='base'>
    <visual>
      <geometry><mesh filename='meshes/body.STL'/></geometry>
      <material name='paint'><texture filename='textures/albedo.png'/></material>
    </visual>
  </link>
</robot>
""",
        encoding="utf-8",
    )
    return asset_root, urdf, ordinary_mesh, camera_mesh


def _plane_terrain() -> TerrainManagerCfg:
    return TerrainManagerCfg(
        terrain_term=TerrainTermCfg(
            func="holosoma.managers.terrain.terms.locomotion:TerrainLocomotion",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.37,
            mesh_type=MeshType.PLANE,
            terrain_config={},
            spawn=SpawnCfg(),
        )
    )


def _config(tmp_path: Path):
    asset_root, urdf, ordinary_mesh, camera_mesh = _write_assets(tmp_path)
    config = SimpleNamespace(
        simulator=SimpleNamespace(_target_="holosoma.simulator.isaacsim.isaacsim.IsaacSim"),
        robot=SimpleNamespace(
            asset=SimpleNamespace(
                asset_root=str(asset_root),
                urdf_file="g1/robot.urdf",
                usd_file=None,
                robot_type="test_robot",
            ),
            object=SimpleNamespace(
                enabled=True,
                object_urdf_path="object-map.json",
                scale=[1.0],
            ),
        ),
        perception=SimpleNamespace(
            enabled=True,
            output_mode="camera_depth",
            camera_source="far_tracking_warp",
            camera_include_robot_mesh=True,
            camera_mesh_file_map={"base": "d435_self.STL"},
        ),
        terrain=_plane_terrain(),
    )
    return config, urdf, ordinary_mesh, camera_mesh


def _digest(config) -> str:
    return runtime_asset_manifest_sha256(build_runtime_asset_manifest(config))


@pytest.mark.parametrize(
    "asset_kind",
    ["urdf", "urdf_mesh", "urdf_texture", "d435_explicit_mesh"],
)
def test_runtime_asset_digest_changes_for_every_active_asset(tmp_path, asset_kind):
    config, urdf, ordinary_mesh, camera_mesh = _config(tmp_path)
    first = _digest(config)

    if asset_kind == "urdf":
        urdf.write_text(urdf.read_text(encoding="utf-8") + "<!-- changed -->\n", encoding="utf-8")
    elif asset_kind == "urdf_mesh":
        ordinary_mesh.write_bytes(ordinary_mesh.read_bytes() + b"mesh-change")
    elif asset_kind == "urdf_texture":
        texture = urdf.parent / "textures" / "albedo.png"
        texture.write_bytes(texture.read_bytes() + b"texture-change")
    else:
        camera_mesh.write_bytes(camera_mesh.read_bytes() + b"camera-change")

    assert _digest(config) != first


def test_runtime_asset_manifest_fails_closed_when_explicit_camera_mesh_is_missing(tmp_path):
    config, _urdf, _ordinary_mesh, camera_mesh = _config(tmp_path)
    camera_mesh.unlink()

    with pytest.raises(FileNotFoundError, match=r"camera_mesh_file_map\['base'\].*does not exist"):
        build_runtime_asset_manifest(config)


def test_defm_runtime_manifest_closes_source_and_pretrained_weight_bytes(tmp_path, monkeypatch):
    config, _urdf, _ordinary_mesh, _camera_mesh = _config(tmp_path)
    source_root = tmp_path / "defm"
    model_factory = source_root / "defm" / "model_factory.py"
    model_factory.parent.mkdir(parents=True)
    model_factory.write_text("MODEL_VERSION = 1\n", encoding="utf-8")
    config_path = source_root / "defm" / "configs" / "defm_vit_s14.yaml"
    config_path.parent.mkdir()
    config_path.write_text("arch: vit_small\n", encoding="utf-8")
    weight_path = tmp_path / "defm_vit_s14.pth"
    weight_path.write_bytes(b"pinned-defm-weight")
    weight_sha256 = hashlib.sha256(weight_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        runtime_asset_manifest_module,
        "_resolve_defm_source_root",
        lambda _environ: source_root,
    )
    config.perception = SimpleNamespace(
        enabled=True,
        output_mode="camera_depth",
        camera_source="rendered",
        camera_include_robot_mesh=False,
        camera_warp_normalize=False,
        encoder_type="defm_vit_s14",
        encoder_pretrained=True,
        encoder_pretrained_path=str(weight_path),
        encoder_pretrained_sha256=weight_sha256,
        inject_into_critic_modules=False,
    )
    config.training = SimpleNamespace(checkpoint=None, policy_init_checkpoint=None)

    manifest = build_runtime_asset_manifest(config)

    defm = manifest["perception"]["defm"]
    assert defm["pretrained_checkpoint_consumed"] is True
    assert defm["pretrained_checkpoint"]["sha256"] == weight_sha256
    assert {record["reference"] for record in defm["source_files"]} == {
        "submodules/defm/defm/configs/defm_vit_s14.yaml",
        "submodules/defm/defm/model_factory.py",
    }

    weight_path.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="pretrained checkpoint SHA256 mismatch"):
        build_runtime_asset_manifest(config)


def test_defm_python_distribution_manifest_closes_every_declared_file(tmp_path, monkeypatch):
    package_file = tmp_path / "fake_pkg" / "runtime.py"
    package_file.parent.mkdir()
    package_file.write_text("VALUE = 1\n", encoding="utf-8")
    empty_marker = tmp_path / "fake_pkg-1.0.dist-info" / "REQUESTED"
    empty_marker.parent.mkdir()
    empty_marker.write_bytes(b"")

    class FakeDistribution:
        version = "1.0"
        metadata = {"Name": "fake-pkg"}
        files = [Path("fake_pkg/runtime.py"), Path("fake_pkg-1.0.dist-info/REQUESTED")]

        @staticmethod
        def locate_file(relative_path):
            return tmp_path / relative_path

    monkeypatch.setattr(
        runtime_asset_manifest_module.importlib.metadata,
        "distribution",
        lambda name: FakeDistribution() if name == "fake-pkg" else None,
    )

    first = runtime_asset_manifest_module._distribution_content_manifest("fake-pkg")
    assert first["file_count"] == 2
    assert first["total_size"] == package_file.stat().st_size

    package_file.write_text("VALUE = 2\n", encoding="utf-8")
    second = runtime_asset_manifest_module._distribution_content_manifest("fake-pkg")
    assert second["content_manifest_sha256"] != first["content_manifest_sha256"]


def test_defm_full_resume_does_not_require_unconsumed_external_pretrain(tmp_path, monkeypatch):
    config, _urdf, _ordinary_mesh, _camera_mesh = _config(tmp_path)
    source_root = tmp_path / "defm"
    model_factory = source_root / "defm" / "model_factory.py"
    model_factory.parent.mkdir(parents=True)
    model_factory.write_text("MODEL_VERSION = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        runtime_asset_manifest_module,
        "_resolve_defm_source_root",
        lambda _environ: source_root,
    )
    declared_sha256 = "a" * 64
    config.perception = SimpleNamespace(
        enabled=True,
        output_mode="camera_depth",
        camera_source="rendered",
        camera_include_robot_mesh=False,
        camera_warp_normalize=False,
        encoder_type="defm_vit_s14",
        encoder_pretrained=True,
        encoder_pretrained_path=str(tmp_path / "intentionally-absent.pth"),
        encoder_pretrained_sha256=declared_sha256,
        inject_into_critic_modules=False,
    )
    config.training = SimpleNamespace(checkpoint="resume.pt", policy_init_checkpoint=None)

    manifest = build_runtime_asset_manifest(config)

    defm = manifest["perception"]["defm"]
    assert defm["pretrained_checkpoint_consumed"] is False
    assert defm["pretrained_checkpoint"] == {
        "reference": str(tmp_path / "intentionally-absent.pth"),
        "size": None,
        "sha256": declared_sha256,
    }


def test_defm_policy_init_closes_fresh_critic_role_and_all_perception_managers(
    tmp_path,
    monkeypatch,
):
    from holosoma.config_values import perception as perception_values

    config, _urdf, _ordinary_mesh, _camera_mesh = _config(tmp_path)
    source_root = tmp_path / "defm"
    model_factory = source_root / "defm" / "model_factory.py"
    model_factory.parent.mkdir(parents=True)
    model_factory.write_text("MODEL_VERSION = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        runtime_asset_manifest_module,
        "_resolve_defm_source_root",
        lambda _environ: source_root,
    )

    actor_missing_path = tmp_path / "actor-restored-from-policy-init.pth"
    actor_sha256 = "a" * 64
    critic_weight_path = tmp_path / "critic-fresh-init.pth"
    critic_weight_path.write_bytes(b"authenticated-critic-defm")
    critic_sha256 = hashlib.sha256(critic_weight_path.read_bytes()).hexdigest()

    actor_manager = SimpleNamespace(
        enabled=True,
        output_mode="camera_depth",
        camera_source="rendered",
        camera_include_robot_mesh=False,
        camera_warp_normalize=False,
        encoder_type="defm_vit_s14",
        encoder_pretrained=True,
        encoder_pretrained_path=str(actor_missing_path),
        encoder_pretrained_sha256=actor_sha256,
        inject_into_policy_modules=True,
        inject_into_critic_modules=False,
    )
    critic_manager = SimpleNamespace(
        enabled=True,
        output_mode="camera_depth",
        camera_source="rendered",
        camera_include_robot_mesh=False,
        camera_warp_normalize=False,
        encoder_type="defm_efficientnet_b2",
        encoder_pretrained=True,
        encoder_pretrained_path=str(critic_weight_path),
        encoder_pretrained_sha256=critic_sha256,
    )
    monkeypatch.setitem(perception_values.DEFAULTS, "test_defm_critic", critic_manager)

    def layer(encoder_type, path, digest, input_name):
        return SimpleNamespace(
            perception_input_name=input_name,
            perception_encoder_type=encoder_type,
            perception_pretrained=True,
            perception_pretrained_path=str(path),
            perception_pretrained_sha256=digest,
        )

    config.perception = actor_manager
    config.training = SimpleNamespace(
        checkpoint=None,
        policy_init_checkpoint="authenticated-actor.pt",
    )
    config.algo = SimpleNamespace(
        config=SimpleNamespace(
            module_dict=SimpleNamespace(
                actor=SimpleNamespace(
                    layer_config=layer(
                        "defm_vit_s14",
                        actor_missing_path,
                        actor_sha256,
                        "perception_obs",
                    )
                ),
                critic=SimpleNamespace(
                    layer_config=layer(
                        "defm_efficientnet_b2",
                        critic_weight_path,
                        critic_sha256,
                        "critic_perception_obs",
                    )
                ),
            ),
            distill=SimpleNamespace(
                enabled=False,
                mode="mse",
                teacher_perception_preset=None,
                critic_perception_preset="test_defm_critic",
            ),
        )
    )

    manifest = build_runtime_asset_manifest(config)
    perception_manifest = manifest["perception"]
    actor = perception_manifest["policy_encoder_roles"]["actor"]
    critic = perception_manifest["policy_encoder_roles"]["critic"]

    assert actor["pretrained_checkpoint_consumed"] is False
    assert actor["pretrained_checkpoint"]["sha256"] == actor_sha256
    assert critic["pretrained_checkpoint_consumed"] is True
    assert critic["pretrained_checkpoint"]["sha256"] == critic_sha256
    assert perception_manifest["defm"] == actor
    assert perception_manifest["manager_roles"]["critic"]["encoder_type"] == "defm_efficientnet_b2"

    critic_weight_path.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="pretrained checkpoint SHA256 mismatch"):
        build_runtime_asset_manifest(config)


def test_plane_terrain_has_explicit_semantics_and_no_external_asset(tmp_path):
    config, _urdf, _ordinary_mesh, _camera_mesh = _config(tmp_path)
    manifest = build_runtime_asset_manifest(config)

    assert manifest["terrain"]["mesh_type"] == "plane"
    assert manifest["terrain"]["external_assets"] == []
    assert manifest["terrain"]["isaacsim_collision_semantics"] == {
        "terrain_type": "plane",
        "friction_combine_mode": "multiply",
        "restitution_combine_mode": "multiply",
        "static_friction": 1.0,
        "dynamic_friction": 1.0,
        "restitution": 0.0,
    }


def test_runtime_asset_manifest_rejects_unclosed_usd_and_nonplane_terrain(tmp_path):
    config, _urdf, _ordinary_mesh, _camera_mesh = _config(tmp_path)
    config.robot.asset.usd_file = "g1/robot.usd"
    with pytest.raises(ValueError, match="does not support.*USD"):
        build_runtime_asset_manifest(config)

    config.robot.asset.usd_file = None
    config.terrain = dataclasses.replace(
        config.terrain,
        terrain_term=dataclasses.replace(config.terrain.terrain_term, mesh_type=MeshType.TRIMESH),
    )
    with pytest.raises(ValueError, match="supports only analytic plane terrain"):
        build_runtime_asset_manifest(config)


@pytest.mark.parametrize("field", ["scene_files", "rigid_objects"])
def test_runtime_asset_manifest_fails_closed_for_unclosed_scene_assets(tmp_path, field):
    config, _urdf, _ordinary_mesh, _camera_mesh = _config(tmp_path)
    scene = SimpleNamespace(
        scene_files=[SimpleNamespace(usd_path="scene.usda")] if field == "scene_files" else [],
        rigid_objects=[SimpleNamespace(usd_path="object.usda")] if field == "rigid_objects" else [],
        replicate_physics=True,
        env_spacing=20.0,
    )
    config.simulator.config = SimpleNamespace(scene=scene)

    with pytest.raises(ValueError, match=field):
        build_runtime_asset_manifest(config)


def test_object_collider_and_spawn_semantics_are_content_bound(tmp_path):
    config, _urdf, _ordinary_mesh, _camera_mesh = _config(tmp_path)
    hull = build_runtime_asset_manifest(
        config,
        environ={
            "HOLOSOMA_OBJECT_COLLIDER_TYPE": "convex_hull",
            "HOLOSOMA_OBJECT_SPAWN_MODE": "urdf",
        },
    )
    decomposition = build_runtime_asset_manifest(
        config,
        environ={
            "HOLOSOMA_OBJECT_COLLIDER_TYPE": "convex_decomposition",
            "HOLOSOMA_OBJECT_SPAWN_MODE": "single-slot-multi-urdf",
        },
    )

    assert hull["object_loader"]["semantics"]["collider_type"] == "convex_hull"
    assert decomposition["object_loader"]["semantics"]["collider_type"] == "convex_decomposition"
    assert decomposition["object_loader"]["semantics"]["spawn_mode"] == "single_slot_multi_urdf"
    assert runtime_asset_manifest_sha256(hull) != runtime_asset_manifest_sha256(decomposition)


def test_object_urdf_conversion_cache_key_binds_source_bytes_and_converter_semantics(tmp_path):
    _config_value, urdf, ordinary_mesh, _camera_mesh = _config(tmp_path)
    first = object_urdf_conversion_cache_key(
        urdf,
        collider_type="convex_hull",
        object_scale=(1.0, 1.0, 1.0),
    )
    collider_changed = object_urdf_conversion_cache_key(
        urdf,
        collider_type="convex_decomposition",
        object_scale=(1.0, 1.0, 1.0),
    )
    ordinary_mesh.write_bytes(ordinary_mesh.read_bytes() + b"new-source-bytes")
    source_changed = object_urdf_conversion_cache_key(
        urdf,
        collider_type="convex_hull",
        object_scale=(1.0, 1.0, 1.0),
    )

    assert collider_changed != first
    assert source_changed != first


def _write_obj_urdf(tmp_path: Path, obj_bytes: bytes) -> Path:
    obj = tmp_path / "object.obj"
    obj.write_bytes(obj_bytes)
    urdf = tmp_path / "object.urdf"
    urdf.write_text(
        "<robot name='object'><link name='base'><visual><geometry>"
        "<mesh filename='object.obj'/></geometry></visual></link></robot>",
        encoding="utf-8",
    )
    return urdf


def _write_glb(path: Path, payload: dict) -> None:
    json_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    json_chunk = struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(json_chunk)) + json_chunk)


def _write_glb_urdf(tmp_path: Path, payload: dict) -> tuple[Path, Path]:
    glb = tmp_path / "object.glb"
    _write_glb(glb, payload)
    urdf = tmp_path / "object.urdf"
    urdf.write_text(
        "<robot name='object'><link name='base'><visual><geometry>"
        "<mesh filename='object.glb'/></geometry></visual></link></robot>",
        encoding="utf-8",
    )
    return urdf, glb


def test_obj_dependency_scan_parses_only_mtllib_lines_and_closes_quoted_assets(tmp_path, monkeypatch):
    texture = tmp_path / "texture one.png"
    texture.write_bytes(b"texture")
    material = tmp_path / "material one.mtl"
    material.write_text(
        'newmtl "dependency-scanner-does-not-parse-this\nmap_Pr "texture one.png"\n',
        encoding="utf-8",
    )
    geometry = b"v 0 0 0\n" * 50_000
    urdf = _write_obj_urdf(
        tmp_path,
        b"# non-directive bytes may be opaque: \xff\n"
        + geometry
        + b'MTLliB "material one.mtl"\r\nv 1 1 1\r\n',
    )

    parsed_obj_lines: list[str] = []
    parsed_material_lines: list[str] = []
    original_parse = runtime_asset_manifest_module._parse_shell_words

    def _record_parse(line: str, *, role: str) -> list[str]:
        if role == "test object.mesh[0]":
            parsed_obj_lines.append(line)
        if role == "test object.mesh[0].material":
            parsed_material_lines.append(line)
        return original_parse(line, role=role)

    monkeypatch.setattr(runtime_asset_manifest_module, "_parse_shell_words", _record_parse)
    manifest = build_urdf_asset_manifest(urdf, role="test object", require_mesh=True)

    assert parsed_obj_lines == ['MTLliB "material one.mtl"']
    assert parsed_material_lines == ['map_Pr "texture one.png"']
    dependencies = manifest["mesh_assets"][0]["dependencies"]
    assert dependencies[0]["reference"] == "material one.mtl"
    assert dependencies[0]["textures"][0]["reference"] == "texture one.png"


def test_obj_dependency_scan_rejects_empty_mtllib_directive(tmp_path):
    urdf = _write_obj_urdf(tmp_path, b"mtllib\r\nv 0 0 0\r\n")

    with pytest.raises(ValueError, match="empty mtllib directive"):
        build_urdf_asset_manifest(urdf, role="test object", require_mesh=True)


def test_obj_dependency_scan_rejects_invalid_utf8_in_mtllib_directive(tmp_path):
    urdf = _write_obj_urdf(tmp_path, b"mtllib material-\xff.mtl\nv 0 0 0\n")

    with pytest.raises(ValueError, match="mtllib directive is not valid UTF-8"):
        build_urdf_asset_manifest(urdf, role="test object", require_mesh=True)


def test_glb_dependency_scan_content_closes_external_buffer_and_image(tmp_path):
    (tmp_path / "buffer.bin").write_bytes(b"buffer")
    (tmp_path / "texture.png").write_bytes(b"texture")
    (tmp_path / "extension.bin").write_bytes(b"extension")
    urdf, _glb = _write_glb_urdf(
        tmp_path,
        {
            "asset": {"version": "2.0"},
            "buffers": [{"uri": "buffer.bin", "byteLength": 6}],
            "images": [{"uri": "texture.png"}],
            "extensions": {"VENDOR_external": {"uri": "extension.bin"}},
        },
    )

    manifest = build_urdf_asset_manifest(urdf, role="test object", require_mesh=True)

    dependencies = manifest["mesh_assets"][0]["dependencies"]
    assert {(record["kind"], record["reference"]) for record in dependencies} == {
        ("buffer", "buffer.bin"),
        ("texture", "texture.png"),
        ("external", "extension.bin"),
    }


def test_glb_dependency_scan_rejects_declared_length_mismatch(tmp_path):
    urdf, glb = _write_glb_urdf(tmp_path, {"asset": {"version": "2.0"}})
    glb.write_bytes(glb.read_bytes() + b"unexpected")

    with pytest.raises(ValueError, match="declared length does not match file size"):
        build_urdf_asset_manifest(urdf, role="test object", require_mesh=True)


def test_glb_dependency_scan_rejects_unaligned_chunk_lengths(tmp_path):
    urdf, glb = _write_glb_urdf(tmp_path, {"asset": {"version": "2.0"}})
    json_bytes = b"{}"
    json_chunk = struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    glb.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(json_chunk)) + json_chunk)

    with pytest.raises(ValueError, match="not 4-byte aligned"):
        build_urdf_asset_manifest(urdf, role="test object", require_mesh=True)


def test_collada_dependency_scan_rejects_external_document_reference(tmp_path):
    (tmp_path / "other.dae").write_text("<COLLADA/>", encoding="utf-8")
    (tmp_path / "object.dae").write_text(
        '<COLLADA><library_visual_scenes><visual_scene><node>'
        '<instance_geometry url="other.dae#geometry"/>'
        "</node></visual_scene></library_visual_scenes></COLLADA>",
        encoding="utf-8",
    )
    urdf = tmp_path / "object.urdf"
    urdf.write_text(
        "<robot name='object'><link name='base'><visual><geometry>"
        "<mesh filename='object.dae'/></geometry></visual></link></robot>",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported external document reference"):
        build_urdf_asset_manifest(urdf, role="test object", require_mesh=True)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("HOLOSOMA_OBJECT_COLLIDER_TYPE", "mystery-collider"),
        ("HOLOSOMA_OBJECT_SPAWN_MODE", "mystery-spawner"),
        ("HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS", "maybe"),
    ],
)
def test_object_loader_semantics_reject_unknown_environment_values(tmp_path, name, value):
    config, _urdf, _ordinary_mesh, _camera_mesh = _config(tmp_path)
    with pytest.raises(ValueError, match=name):
        build_runtime_asset_manifest(config, environ={name: value})


def test_far_tracking_implementation_is_fixed_and_external_override_is_ignored(tmp_path):
    config, _urdf, _ordinary_mesh, _camera_mesh = _config(tmp_path)
    first = build_runtime_asset_manifest(config, environ={})
    second = build_runtime_asset_manifest(
        config,
        environ={"HOLOSOMA_FAR_TRACKING_PKG_ROOT": str(tmp_path / "external")},
    )

    implementation = first["perception"]["far_tracking_implementation"]
    assert implementation["kind"] == "holosoma_bundled_ft_warp_sensors"
    assert implementation["selection"] == "fixed"
    assert implementation["sources"]
    assert runtime_asset_manifest_sha256(first) == runtime_asset_manifest_sha256(second)

def _pending_provenance() -> dict:
    return {
        "version": 2,
        "teacher_sha256": "1" * 64,
        "policy_init_enabled": False,
        "policy_init_sha256": disabled_checkpoint_sha256("policy_init"),
        "training_resume_enabled": False,
        "training_resume_sha256": disabled_checkpoint_sha256("training_resume"),
        "motion_shard_manifest_sha256": "2" * 64,
        "contact_sidecar_manifest_sha256": "3" * 64,
        "source_bundle_sha256": "4" * 64,
        "environment": {
            EXECUTION_RUNTIME_KEY: {
                SEMANTIC_ENVIRONMENT_KEY: {
                    name: None for name in SEMANTIC_ENVIRONMENT_FIELDS
                }
            }
        },
        RUNTIME_ASSET_PHASE_KEY: RUNTIME_ASSET_PHASE_PENDING,
        RUNTIME_ASSET_DIGEST_KEY: pending_runtime_asset_manifest_sha256(),
    }


def test_pending_provenance_is_finalized_before_strict_consumers(tmp_path):
    config, _urdf, _ordinary_mesh, _camera_mesh = _config(tmp_path)
    environ = {ENV_NAME: json.dumps(_pending_provenance())}

    with pytest.raises(ValueError, match="still pending"):
        validate_training_provenance(_pending_provenance(), require_finalized=True)

    finalized = finalize_runtime_asset_provenance(config, environ=environ)
    assert finalized is not None
    assert finalized[RUNTIME_ASSET_PHASE_KEY] == RUNTIME_ASSET_PHASE_FINAL
    assert finalized[RUNTIME_ASSET_DIGEST_KEY] == _digest(config)
    assert finalized[RUNTIME_ASSET_MANIFEST_KEY] == build_runtime_asset_manifest(config)
    assert validate_training_provenance(
        json.loads(environ[ENV_NAME]),
        require_finalized=True,
    ) == finalized


def test_runtime_asset_finalization_rejects_semantic_environment_changed_after_generation(
    tmp_path,
):
    config, _urdf, _ordinary_mesh, _camera_mesh = _config(tmp_path)
    environ = {
        ENV_NAME: json.dumps(_pending_provenance()),
        "HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS": "1",
    }

    with pytest.raises(
        ValueError,
        match=(
            "Training semantic environment changed after provenance generation: "
            "HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS"
        ),
    ):
        finalize_runtime_asset_provenance(config, environ=environ)


def test_finalized_manifest_is_persisted_and_tampering_is_rejected(tmp_path):
    config, _urdf, _ordinary_mesh, _camera_mesh = _config(tmp_path)
    environ = {ENV_NAME: json.dumps(_pending_provenance())}
    finalized = finalize_runtime_asset_provenance(config, environ=environ)
    assert finalized is not None

    destination = persist_runtime_asset_manifest(tmp_path / "run" / "runtime_asset_manifest.json", finalized)
    assert json.loads(destination.read_text(encoding="utf-8")) == finalized[RUNTIME_ASSET_MANIFEST_KEY]

    tampered = json.loads(json.dumps(finalized))
    tampered[RUNTIME_ASSET_MANIFEST_KEY]["robot"]["robot_type"] = "tampered"
    with pytest.raises(ValueError, match="manifest digest mismatch"):
        validate_training_provenance(tampered, require_finalized=True)


def test_refinalization_detects_asset_mutation_before_simulator_use(tmp_path):
    config, _urdf, ordinary_mesh, _camera_mesh = _config(tmp_path)
    environ = {ENV_NAME: json.dumps(_pending_provenance())}
    assert finalize_runtime_asset_provenance(config, environ=environ) is not None
    ordinary_mesh.write_bytes(ordinary_mesh.read_bytes() + b"mutated-after-preflight")

    with pytest.raises(ValueError, match="already-finalized.*does not match"):
        finalize_runtime_asset_provenance(config, environ=environ)


def test_legacy_v1_runtime_asset_claim_fails_closed(tmp_path):
    config, _urdf, _ordinary_mesh, _camera_mesh = _config(tmp_path)
    legacy = _pending_provenance()
    legacy["version"] = 1
    legacy.pop(RUNTIME_ASSET_PHASE_KEY)
    legacy.pop(RUNTIME_ASSET_DIGEST_KEY)

    with pytest.raises(ValueError, match="unsupported training provenance version 1"):
        finalize_runtime_asset_provenance(config, environ={ENV_NAME: json.dumps(legacy)})


def test_train_main_finalizes_effective_assets_before_every_preflight(monkeypatch):
    import holosoma.train_agent as train_agent

    raw = SimpleNamespace(
        training=SimpleNamespace(
            num_envs=8,
            seed=42,
            checkpoint=None,
            policy_init_checkpoint=None,
        ),
        curriculum=None,
    )
    observation_effective = SimpleNamespace(training=raw.training, curriculum=None)
    fully_effective = SimpleNamespace(training=raw.training, curriculum=None)
    calls: list[str] = []

    monkeypatch.setattr(train_agent.tyro, "cli", lambda *_args, **_kwargs: raw)

    def apply_observation(config):
        assert config is raw
        calls.append("observation")
        return observation_effective

    def apply_perception(config):
        assert config is observation_effective
        calls.append("perception")
        return fully_effective

    def finalize(config):
        assert config is fully_effective
        calls.append("finalize_assets")
        return None

    monkeypatch.setattr(train_agent, "apply_observation_overrides", apply_observation)
    monkeypatch.setattr(train_agent, "apply_perception_overrides", apply_perception)
    monkeypatch.setattr(train_agent, "finalize_runtime_asset_provenance", finalize)
    monkeypatch.setattr(
        train_agent,
        "_preflight_data_assets_before_sim",
        lambda: calls.append("data_assets"),
    )
    monkeypatch.setattr(
        train_agent,
        "_per_rank_env_count",
        lambda *_args: calls.append("env_count"),
    )
    monkeypatch.setattr(
        train_agent,
        "_preflight_cross_rank_provenance_before_sim",
        lambda: calls.append("cross_rank"),
    )
    monkeypatch.setattr(
        train_agent,
        "_preflight_checkpoint_lineage_before_sim",
        lambda config: calls.append("lineage"),
    )

    def policy_init(config):
        calls.append("policy_init")
        return config

    def resume(config):
        calls.append("resume")
        return config

    monkeypatch.setattr(train_agent, "_preflight_policy_init_before_sim", policy_init)
    monkeypatch.setattr(train_agent, "_preflight_training_resume_before_sim", resume)
    monkeypatch.setattr(train_agent, "train", lambda config: calls.append("train"))

    train_agent.main()

    assert calls == [
        "observation",
        "perception",
        "finalize_assets",
        "data_assets",
        "env_count",
        "cross_rank",
        "lineage",
        "policy_init",
        "resume",
        "train",
    ]


def test_training_context_revalidates_assets_before_simulator_import(monkeypatch):
    import holosoma.train_agent as train_agent

    config = SimpleNamespace(training=SimpleNamespace(seed=0))
    calls: list[str] = []
    monkeypatch.setattr(
        train_agent,
        "_effective_runtime_config",
        lambda candidate: calls.append("effective") or candidate,
    )
    monkeypatch.setattr(
        train_agent,
        "finalize_runtime_asset_provenance",
        lambda candidate: calls.append("verify") if candidate is config else None,
    )
    monkeypatch.setattr(
        train_agent,
        "_preflight_data_assets_before_sim",
        lambda: calls.append("data_assets"),
    )
    monkeypatch.setattr(
        train_agent,
        "_preflight_policy_init_before_sim",
        lambda candidate: calls.append("policy_init") or candidate,
    )
    monkeypatch.setattr(
        train_agent,
        "init_sim_imports",
        lambda candidate: calls.append("import") if candidate is config else None,
    )

    context = train_agent.TrainingContext(config)
    context.__enter__()

    assert calls == ["effective", "verify", "data_assets", "policy_init", "import"]


def test_direct_train_finalizes_the_effective_runtime_config_before_simulator(monkeypatch):
    import holosoma.train_agent as train_agent

    raw = SimpleNamespace(training=SimpleNamespace(seed=9))
    effective = SimpleNamespace(
        training=SimpleNamespace(
            seed=9,
            checkpoint=None,
            policy_init_checkpoint=None,
        )
    )
    calls: list[str] = []

    monkeypatch.setattr(
        train_agent,
        "_effective_runtime_config",
        lambda candidate: calls.append("effective") or effective,
    )
    monkeypatch.setattr(
        train_agent,
        "_current_rank_training_seed",
        lambda seed: calls.append(f"seed:{seed}"),
    )
    monkeypatch.setattr(
        train_agent,
        "_configure_defm_materialization_mode",
        lambda candidate: calls.append("defm") if candidate is effective else None,
    )

    class Finalized(Exception):
        pass

    def finalize(candidate):
        assert candidate is effective
        calls.append("finalize")
        raise Finalized

    monkeypatch.setattr(train_agent, "finalize_runtime_asset_provenance", finalize)

    with pytest.raises(Finalized):
        train_agent.train(raw)

    assert calls == ["effective", "seed:9", "defm", "finalize"]


@pytest.mark.parametrize(
    ("checkpoint", "policy_init_checkpoint", "expected_mode"),
    [
        (None, None, "fresh"),
        (None, "actor.pt", "policy_init"),
        ("resume.pt", None, "full_resume"),
    ],
)
def test_defm_materialization_mode_is_derived_from_authoritative_training_config(
    monkeypatch,
    checkpoint,
    policy_init_checkpoint,
    expected_mode,
):
    import holosoma.train_agent as train_agent

    monkeypatch.setenv("HOLOSOMA_DEFM_MATERIALIZATION_MODE", "ambient-value-must-not-win")
    config = SimpleNamespace(
        training=SimpleNamespace(
            checkpoint=checkpoint,
            policy_init_checkpoint=policy_init_checkpoint,
        )
    )

    assert train_agent._configure_defm_materialization_mode(config) == expected_mode
    assert os.environ["HOLOSOMA_DEFM_MATERIALIZATION_MODE"] == expected_mode


def test_defm_materialization_mode_rejects_conflicting_checkpoint_operations(monkeypatch):
    import holosoma.train_agent as train_agent

    monkeypatch.setenv("HOLOSOMA_DEFM_MATERIALIZATION_MODE", "sentinel")
    config = SimpleNamespace(
        training=SimpleNamespace(
            checkpoint="resume.pt",
            policy_init_checkpoint="actor.pt",
        )
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        train_agent._configure_defm_materialization_mode(config)
    assert os.environ["HOLOSOMA_DEFM_MATERIALIZATION_MODE"] == "sentinel"
