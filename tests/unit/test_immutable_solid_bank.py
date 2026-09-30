from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from prepare_immutable_solid_bank import prepare_immutable_solid_bank


def _write_fixture(root: Path, *, primitive: bool = False) -> tuple[Path, Path, Path]:
    bank = root / "source_bank"
    assets = root / "assets"
    contact = bank / "contact_export"
    bank.mkdir(parents=True)
    assets.mkdir(parents=True)
    contact.mkdir(parents=True)
    (bank / "box_clip.npz").write_bytes(b"box-motion-v1")
    (bank / "other_clip.npz").write_bytes(b"other-motion")
    (contact / "box_clip.json").write_text('{"t1": 10, "t2": 20}\n', encoding="utf-8")

    mesh = assets / "box.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    geometry = '<box size="1 1 1" />' if primitive else '<mesh filename="box.obj" />'
    urdf = assets / "box.urdf"
    urdf.write_text(
        f'<robot name="box"><link name="baseLink"><visual><geometry>{geometry}</geometry></visual></link></robot>',
        encoding="utf-8",
    )
    mug_mesh = assets / "mug.obj"
    mug_mesh.write_text(mesh.read_text(encoding="utf-8"), encoding="utf-8")
    mug_urdf = assets / "mug.urdf"
    mug_urdf.write_text(
        '<robot name="mug"><link name="baseLink"><visual><geometry>'
        '<mesh filename="mug.obj" />'
        '</geometry></visual></link></robot>',
        encoding="utf-8",
    )
    object_map = bank / "_clip_object_urdf_map.json"
    object_map.write_text(
        json.dumps(
            {
                "clips": {
                    "box_clip": {
                        "object_name": "box_01",
                        "object_urdf_path": str(urdf),
                        "object_mesh_path": str(mesh),
                    },
                    "other_clip": {
                        "object_name": "mug_01",
                        "object_urdf_path": str(mug_urdf),
                        "object_mesh_path": str(mug_mesh),
                    },
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return bank, object_map, contact


def _prepare(bank: Path, object_map: Path, contact: Path):
    return prepare_immutable_solid_bank(
        source_bank=bank,
        source_map=object_map,
        allowed_categories={"box"},
        contact_export_name="contact_export",
        clip_list_path=None,
        target_bank_name="filtered_solid",
        contact_root_override=contact,
    )


def _read_object_map(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_object_map(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _published_object_map(manifest: dict[str, object]) -> dict[str, object]:
    return _read_object_map(Path(str(manifest["output_root"])) / "_clip_object_urdf_map.json")


def _write_rollout_manifest(bank: Path, *, checkpoint_sha256: str) -> None:
    lineage = {
        "source_checkpoint_sha256": checkpoint_sha256,
        "checkpoint_source": "wandb://entity/project/run/model_05000.pt",
        "saved_wandb_path": "entity/project/run",
    }
    publication_payload = {"teacher_lineage": lineage, "fixture": True}
    publication_id = hashlib.sha256(
        json.dumps(
            publication_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    (bank / "realmesh_rollout_manifest.json").write_text(
        json.dumps(
            {
                "publication_id": publication_id,
                "publication_payload": publication_payload,
                "teacher_lineage": lineage,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_solid_bank_is_filtered_content_addressed_and_reusable(tmp_path: Path) -> None:
    bank, object_map, contact = _write_fixture(tmp_path)
    first = _prepare(bank, object_map, contact)
    output_root = Path(first["output_root"])

    assert output_root.name.endswith(first["source_digest"])
    assert first["selected_clip_count"] == 1
    assert (output_root / "box_clip.npz").is_file()
    assert not (output_root / "box_clip.npz").is_symlink()
    assert (output_root / "box_clip.npz").stat().st_mode & 0o222 == 0
    assert not (output_root / "other_clip.npz").exists()
    filtered = json.loads((output_root / "_clip_object_urdf_map.json").read_text(encoding="utf-8"))
    assert list(filtered["clips"]) == ["box_clip"]
    assert (output_root / "contact_export").is_dir()
    assert not (output_root / "contact_export").is_symlink()
    assert (output_root / "contact_export" / "box_clip.json").read_text(encoding="utf-8") == '{"t1": 10, "t2": 20}\n'
    assert (output_root / "contact_export" / "box_clip.json").stat().st_mode & 0o222 == 0
    assert (output_root / "_clip_object_urdf_map.json").stat().st_mode & 0o222 == 0
    assert (output_root / "manifest.json").stat().st_mode & 0o222 == 0

    second = _prepare(bank, object_map, contact)
    assert second == first
    assert Path(second["output_root"]) == output_root


def test_solid_bank_seals_authenticated_motion_generator_teacher(tmp_path: Path) -> None:
    bank, object_map, contact = _write_fixture(tmp_path)
    digest = "7" * 64
    _write_rollout_manifest(bank, checkpoint_sha256=digest)

    manifest = _prepare(bank, object_map, contact)

    source_identity = manifest["source_identity"]
    assert source_identity["motion_generator_teacher"] == {
        "version": 1,
        "checkpoint_sha256": digest,
        "checkpoint_source": "wandb://entity/project/run/model_05000.pt",
        "saved_wandb_path": "entity/project/run",
    }
    rollout_record = source_identity["source_rollout_manifest"]
    assert Path(rollout_record["path"]).name == "realmesh_rollout_manifest.json"
    assert len(rollout_record["sha256"]) == 64


def test_solid_bank_infers_multi_clip_transition_source_before_filtering(tmp_path: Path) -> None:
    bank, object_map, contact = _write_fixture(tmp_path)

    manifest = _prepare(bank, object_map, contact)

    assert manifest["selected_clip_count"] == 1
    assert _published_object_map(manifest)["motion_transition_source"] == {
        "version": 1,
        "source_clip_count": 2,
        "source_semantics": "global_multi_clip_runtime",
    }


def test_solid_bank_infers_native_single_clip_static_transition_source(tmp_path: Path) -> None:
    bank, object_map, contact = _write_fixture(tmp_path)
    (bank / "other_clip.npz").unlink()
    payload = _read_object_map(object_map)
    del payload["clips"]["other_clip"]  # type: ignore[index]
    _write_object_map(object_map, payload)

    manifest = _prepare(bank, object_map, contact)

    assert _published_object_map(manifest)["motion_transition_source"] == {
        "version": 1,
        "source_clip_count": 1,
        "source_semantics": "single_clip_static",
    }


def test_solid_bank_preserves_existing_transition_source_across_refiltering(tmp_path: Path) -> None:
    bank, object_map, contact = _write_fixture(tmp_path)
    transition_source = {
        "version": 1,
        "source_clip_count": 30,
        "source_semantics": "global_multi_clip_runtime",
    }
    payload = _read_object_map(object_map)
    payload["motion_transition_source"] = transition_source
    _write_object_map(object_map, payload)

    manifest = _prepare(bank, object_map, contact)

    assert manifest["selected_clip_count"] == 1
    assert _published_object_map(manifest)["motion_transition_source"] == transition_source


@pytest.mark.parametrize(
    "transition_source",
    [
        pytest.param("global_multi_clip_runtime", id="not-a-mapping"),
        pytest.param({}, id="missing-fields"),
        pytest.param(
            {
                "version": 2,
                "source_clip_count": 2,
                "source_semantics": "global_multi_clip_runtime",
            },
            id="unsupported-version",
        ),
        pytest.param(
            {
                "version": 1,
                "source_clip_count": True,
                "source_semantics": "global_multi_clip_runtime",
            },
            id="boolean-count",
        ),
        pytest.param(
            {
                "version": 1,
                "source_clip_count": 0,
                "source_semantics": "global_multi_clip_runtime",
            },
            id="non-positive-count",
        ),
        pytest.param(
            {
                "version": 1,
                "source_clip_count": 1,
                "source_semantics": "global_multi_clip_runtime",
            },
            id="single-count-with-global-semantics",
        ),
        pytest.param(
            {
                "version": 1,
                "source_clip_count": 2,
                "source_semantics": "single_clip_static",
            },
            id="multi-count-with-static-semantics",
        ),
        pytest.param(
            {
                "version": 1,
                "source_clip_count": 1,
                "source_semantics": "single_clip_static",
            },
            id="count-smaller-than-active-source-set",
        ),
        pytest.param(
            {
                "version": 1,
                "source_clip_count": 2,
                "source_semantics": "unknown",
            },
            id="unknown-semantics",
        ),
        pytest.param(
            {
                "version": 1,
                "source_clip_count": 2,
                "source_semantics": "global_multi_clip_runtime",
                "unexpected": "not part of the schema",
            },
            id="unexpected-field",
        ),
    ],
)
def test_solid_bank_rejects_malformed_transition_source(
    tmp_path: Path,
    transition_source: object,
) -> None:
    bank, object_map, contact = _write_fixture(tmp_path)
    payload = _read_object_map(object_map)
    payload["motion_transition_source"] = transition_source
    _write_object_map(object_map, payload)

    with pytest.raises(ValueError, match="motion_transition_source"):
        _prepare(bank, object_map, contact)


@pytest.mark.parametrize("orphan", ["map-entry", "motion-file"])
def test_solid_bank_requires_exact_source_npz_and_map_clip_sets(
    tmp_path: Path,
    orphan: str,
) -> None:
    bank, object_map, contact = _write_fixture(tmp_path)
    if orphan == "map-entry":
        (bank / "other_clip.npz").unlink()
    else:
        payload = _read_object_map(object_map)
        del payload["clips"]["other_clip"]  # type: ignore[index]
        _write_object_map(object_map, payload)

    with pytest.raises(ValueError):
        _prepare(bank, object_map, contact)


def test_contact_or_motion_byte_change_selects_new_solid_generation(tmp_path: Path) -> None:
    bank, object_map, contact = _write_fixture(tmp_path)
    first = _prepare(bank, object_map, contact)
    first_root = Path(first["output_root"])

    (contact / "box_clip.json").write_text('{"t1": 11, "t2": 21}\n', encoding="utf-8")
    second = _prepare(bank, object_map, contact)
    assert second["source_digest"] != first["source_digest"]
    assert Path(second["output_root"]) != first_root
    assert first_root.is_dir()
    assert (first_root / "contact_export" / "box_clip.json").read_text(encoding="utf-8") == '{"t1": 10, "t2": 20}\n'

    (bank / "box_clip.npz").write_bytes(b"box-motion-v2")
    third = _prepare(bank, object_map, contact)
    assert third["source_digest"] not in {first["source_digest"], second["source_digest"]}


def test_invalid_existing_content_addressed_generation_is_not_replaced(tmp_path: Path) -> None:
    bank, object_map, contact = _write_fixture(tmp_path)
    manifest = _prepare(bank, object_map, contact)
    output_root = Path(manifest["output_root"])
    (output_root / "manifest.json").chmod(0o644)
    (output_root / "manifest.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Refusing to replace invalid content"):
        _prepare(bank, object_map, contact)
    assert (output_root / "manifest.json").read_text(encoding="utf-8") == "{}\n"


def test_primitive_object_geometry_is_rejected(tmp_path: Path) -> None:
    bank, object_map, contact = _write_fixture(tmp_path, primitive=True)
    with pytest.raises(ValueError, match="primitive geometry"):
        _prepare(bank, object_map, contact)


def test_solid_bank_closes_and_absolutizes_all_mesh_metadata(tmp_path: Path) -> None:
    bank, object_map, contact = _write_fixture(tmp_path)
    payload = json.loads(object_map.read_text(encoding="utf-8"))
    mesh = tmp_path / "assets" / "box.obj"
    payload["clips"]["box_clip"].update(
        {
            "object_visual_mesh_path": "../assets/box.obj",
            "object_collision_mesh_paths": [str(mesh)],
        }
    )
    object_map.write_text(json.dumps(payload), encoding="utf-8")

    manifest = _prepare(bank, object_map, contact)
    output_root = Path(manifest["output_root"])
    filtered = json.loads((output_root / "_clip_object_urdf_map.json").read_text())
    entry = filtered["clips"]["box_clip"]
    assert Path(entry["object_visual_mesh_path"]) == mesh.resolve()
    assert Path(entry["object_collision_mesh_paths"][0]) == mesh.resolve()
    asset_paths = {record["path"] for record in manifest["source_identity"]["object_assets"]}
    assert str(mesh.resolve()) in asset_paths


def test_solid_contact_file_symlink_is_materialized(tmp_path: Path) -> None:
    bank, object_map, contact = _write_fixture(tmp_path)
    target = tmp_path / "external_contact.json"
    target.write_text('{"t1": 30, "t2": 40}\n', encoding="utf-8")
    link = contact / "linked.json"
    link.symlink_to(target)

    manifest = _prepare(bank, object_map, contact)
    published = Path(manifest["output_root"]) / "contact_export" / "linked.json"
    assert published.is_file()
    assert not published.is_symlink()
    assert published.read_bytes() == target.read_bytes()


def test_long_requested_bank_name_leaves_room_for_atomic_temp_and_lock_names(tmp_path: Path) -> None:
    bank, object_map, contact = _write_fixture(tmp_path)
    manifest = prepare_immutable_solid_bank(
        source_bank=bank,
        source_map=object_map,
        allowed_categories={"box"},
        contact_export_name="contact_export",
        clip_list_path=None,
        target_bank_name="x" * 500,
        contact_root_override=contact,
    )
    output_root = Path(manifest["output_root"])
    assert output_root.is_dir()
    assert len(output_root.name.encode("utf-8")) <= 220
