from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from prepare_as_rank_shards import compute_rank_shard_source_digest, prepare_rank_shards
from prepare_immutable_single_slot_bank import prepare_immutable_single_slot_bank


def _write_source_bank(root: Path, *, two_links: bool = False) -> tuple[Path, Path]:
    motion_dir = root / "motions"
    motion_dir.mkdir(parents=True)
    (motion_dir / "clip_a.npz").write_bytes(b"motion-a-v1")
    mesh = root / "assets" / "object.obj"
    mesh.parent.mkdir(parents=True)
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    extra_link = '<link name="unexpected" />' if two_links else ""
    urdf = root / "assets" / "object.urdf"
    urdf.write_text(
        "<robot name=\"object\">"
        "<link name=\"object_link\">"
        "<visual><geometry><mesh filename=\"object.obj\" /></geometry></visual>"
        "<collision><geometry><mesh filename=\"object.obj\" /></geometry></collision>"
        "</link>"
        f"{extra_link}"
        "</robot>\n",
        encoding="utf-8",
    )
    object_map = motion_dir / "source_object_map.json"
    object_map.write_text(
        json.dumps(
            {
                "dataset": "fixture",
                "motion_transition_source": {
                    "version": 1,
                    "source_clip_count": 2,
                    "source_semantics": "global_multi_clip_runtime",
                },
                "clips": {
                    "clip_a": {"object_urdf_path": str(urdf)},
                    # An inactive map entry must not leak into the published
                    # view consumed by MotionLoader.
                    "inactive_clip": {"object_urdf_path": str(urdf)},
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return motion_dir, object_map


def test_single_slot_bank_is_content_addressed_reusable_and_active_only(tmp_path: Path) -> None:
    motion_dir, object_map = _write_source_bank(tmp_path)
    output_base = tmp_path / "generated" / "single-slot"

    first = prepare_immutable_single_slot_bank(
        source_motion_dir=motion_dir,
        source_object_map=object_map,
        output_base=output_base,
    )
    output_root = Path(first["output_root"])
    assert output_root == output_base / "by-source" / first["view_digest"]
    assert len(first["view_digest"]) == 64
    assert (output_root / "clip_a.npz").is_file()
    assert not (output_root / "clip_a.npz").is_symlink()
    assert (output_root / "clip_a.npz").read_bytes() == b"motion-a-v1"
    assert (output_root / "clip_a.npz").stat().st_mode & 0o222 == 0
    published_map = json.loads((output_root / "_clip_object_urdf_map.json").read_text(encoding="utf-8"))
    assert list(published_map["clips"]) == ["clip_a"]
    transition_source = {
        "version": 1,
        "source_clip_count": 2,
        "source_semantics": "global_multi_clip_runtime",
    }
    assert published_map["motion_transition_source"] == transition_source
    assert first["motion_transition_source"] == transition_source
    canonical_urdf = next((output_root / "_single_slot_urdfs").glob("*.urdf"))
    assert 'name="baseLink"' in canonical_urdf.read_text(encoding="utf-8")
    assert canonical_urdf.stat().st_mode & 0o222 == 0
    assert (output_root / "_clip_object_urdf_map.json").stat().st_mode & 0o222 == 0
    assert (output_root / "manifest.json").stat().st_mode & 0o222 == 0
    assert output_root.stat().st_mode & 0o222 == 0
    assert (output_root / "_single_slot_urdfs").stat().st_mode & 0o222 == 0
    assert (output_root / "_rank_shards").is_dir()
    assert (output_root / "_rank_shards").stat().st_mode & 0o200

    second = prepare_immutable_single_slot_bank(
        source_motion_dir=motion_dir,
        source_object_map=object_map,
        output_base=output_base,
    )
    assert second == first
    assert Path(second["output_root"]) == output_root


def test_single_slot_bank_reuse_does_not_write_to_published_namespace(tmp_path: Path) -> None:
    motion_dir, object_map = _write_source_bank(tmp_path)
    output_base = tmp_path / "generated" / "single-slot"
    first = prepare_immutable_single_slot_bank(
        source_motion_dir=motion_dir,
        source_object_map=object_map,
        output_base=output_base,
    )

    by_source = output_base / "by-source"
    (Path(first["output_root"]) / "_rank_shards").chmod(0o555)
    by_source.chmod(0o555)
    try:
        second = prepare_immutable_single_slot_bank(
            source_motion_dir=motion_dir,
            source_object_map=object_map,
            output_base=output_base,
        )
    finally:
        by_source.chmod(0o755)

    assert second == first


def test_transition_source_change_publishes_a_new_single_slot_generation(tmp_path: Path) -> None:
    motion_dir, object_map = _write_source_bank(tmp_path)
    output_base = tmp_path / "generated"
    first = prepare_immutable_single_slot_bank(
        source_motion_dir=motion_dir,
        source_object_map=object_map,
        output_base=output_base,
    )
    first_root = Path(first["output_root"])

    payload = json.loads(object_map.read_text(encoding="utf-8"))
    payload["motion_transition_source"] = {
        "version": 1,
        "source_clip_count": 3,
        "source_semantics": "global_multi_clip_runtime",
    }
    object_map.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    second = prepare_immutable_single_slot_bank(
        source_motion_dir=motion_dir,
        source_object_map=object_map,
        output_base=output_base,
    )

    assert second["view_digest"] != first["view_digest"]
    assert Path(second["output_root"]) != first_root
    assert first_root.is_dir()
    first_map = json.loads((first_root / "_clip_object_urdf_map.json").read_text(encoding="utf-8"))
    second_map = json.loads(
        (Path(second["output_root"]) / "_clip_object_urdf_map.json").read_text(encoding="utf-8")
    )
    assert first_map["motion_transition_source"] == first["motion_transition_source"]
    assert second_map["motion_transition_source"] == second["motion_transition_source"]
    assert first["motion_transition_source"] != second["motion_transition_source"]


def test_single_slot_view_seals_motion_generator_teacher_lineage(tmp_path: Path) -> None:
    motion_dir, object_map = _write_source_bank(tmp_path)
    output_base = tmp_path / "generated"
    generator = {
        "version": 1,
        "checkpoint_sha256": "5" * 64,
        "checkpoint_source": "wandb://entity/project/run/model_05000.pt",
        "saved_wandb_path": "entity/project/run",
    }
    source_manifest = motion_dir / "manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "version": 5,
                "output_root": str(motion_dir.resolve()),
                "source_identity": {"motion_generator_teacher": generator},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    first = prepare_immutable_single_slot_bank(
        source_motion_dir=motion_dir,
        source_object_map=object_map,
        output_base=output_base,
    )
    assert first["motion_generator_teacher"] == generator
    assert first["source_lineage_manifest"]["path"] == "manifest.json"

    changed = dict(generator)
    changed["checkpoint_sha256"] = "6" * 64
    source_manifest.write_text(
        json.dumps(
            {
                "version": 5,
                "output_root": str(motion_dir.resolve()),
                "source_identity": {"motion_generator_teacher": changed},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    second = prepare_immutable_single_slot_bank(
        source_motion_dir=motion_dir,
        source_object_map=object_map,
        output_base=output_base,
    )
    assert second["view_digest"] != first["view_digest"]
    assert second["motion_generator_teacher"] == changed


def test_source_byte_change_publishes_new_generation_without_mutating_old(tmp_path: Path) -> None:
    motion_dir, object_map = _write_source_bank(tmp_path)
    output_base = tmp_path / "generated"
    first = prepare_immutable_single_slot_bank(
        source_motion_dir=motion_dir,
        source_object_map=object_map,
        output_base=output_base,
    )
    first_root = Path(first["output_root"])

    (motion_dir / "clip_a.npz").write_bytes(b"motion-a-v2")
    second = prepare_immutable_single_slot_bank(
        source_motion_dir=motion_dir,
        source_object_map=object_map,
        output_base=output_base,
    )

    assert second["view_digest"] != first["view_digest"]
    assert Path(second["output_root"]) != first_root
    assert first_root.is_dir()
    assert (first_root / "clip_a.npz").read_bytes() == b"motion-a-v1"
    assert (Path(second["output_root"]) / "clip_a.npz").read_bytes() == b"motion-a-v2"
    assert (first_root / "manifest.json").is_file()


def test_corrupt_published_generation_is_never_replaced_in_place(tmp_path: Path) -> None:
    motion_dir, object_map = _write_source_bank(tmp_path)
    output_base = tmp_path / "generated"
    manifest = prepare_immutable_single_slot_bank(
        source_motion_dir=motion_dir,
        source_object_map=object_map,
        output_base=output_base,
    )
    output_root = Path(manifest["output_root"])
    (output_root / "manifest.json").chmod(0o644)
    (output_root / "manifest.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="identity must never be mutated"):
        prepare_immutable_single_slot_bank(
            source_motion_dir=motion_dir,
            source_object_map=object_map,
            output_base=output_base,
        )
    assert (output_root / "manifest.json").read_text(encoding="utf-8") == "{}\n"


@pytest.mark.parametrize("invalid_mode", [0o757])
def test_reuse_rejects_invalid_rank_shards_permissions(
    tmp_path: Path,
    invalid_mode: int,
) -> None:
    motion_dir, object_map = _write_source_bank(tmp_path)
    output_base = tmp_path / "generated"
    manifest = prepare_immutable_single_slot_bank(
        source_motion_dir=motion_dir,
        source_object_map=object_map,
        output_base=output_base,
    )
    output_root = Path(manifest["output_root"])
    rank_shards_root = output_root / "_rank_shards"
    rank_shards_root.chmod(invalid_mode)

    try:
        with pytest.raises(ValueError, match="identity must never be mutated"):
            prepare_immutable_single_slot_bank(
                source_motion_dir=motion_dir,
                source_object_map=object_map,
                output_base=output_base,
            )
    finally:
        rank_shards_root.chmod(0o755)

    assert Path(manifest["output_root"]) == output_root


def test_rank_shards_publish_below_frozen_generation(tmp_path: Path) -> None:
    motion_dir, object_map = _write_source_bank(tmp_path)
    manifest = prepare_immutable_single_slot_bank(
        source_motion_dir=motion_dir,
        source_object_map=object_map,
        output_base=tmp_path / "generated",
    )
    output_root = Path(manifest["output_root"])
    published_object_map = output_root / "_clip_object_urdf_map.json"
    source_digest = compute_rank_shard_source_digest(
        motion_dir=output_root,
        object_map=published_object_map,
        world_size=2,
    )
    shard_root = output_root / "_rank_shards" / "by-source" / source_digest / "ws2"

    shard_manifest = prepare_rank_shards(
        motion_dir=output_root,
        object_map=published_object_map,
        output_root=shard_root,
        world_size=2,
        expected_source_digest=source_digest,
    )

    assert shard_manifest["world_size"] == 2
    assert (shard_root / "rank_0" / "clip_a.npz").is_symlink()
    assert (shard_root / "rank_1" / "clip_a.npz").is_symlink()
    assert output_root.stat().st_mode & 0o222 == 0
    assert (output_root / "_single_slot_urdfs").stat().st_mode & 0o222 == 0


def test_failed_canonicalization_does_not_publish_partial_generation(tmp_path: Path) -> None:
    motion_dir, object_map = _write_source_bank(tmp_path, two_links=True)
    output_base = tmp_path / "generated"

    with pytest.raises(ValueError, match="exactly one URDF link"):
        prepare_immutable_single_slot_bank(
            source_motion_dir=motion_dir,
            source_object_map=object_map,
            output_base=output_base,
        )

    published_parent = output_base / "by-source"
    assert not published_parent.exists() or not any(
        path.is_dir() and not path.name.startswith(".") for path in published_parent.iterdir()
    )


def test_single_slot_rewrites_all_declared_mesh_metadata_paths(tmp_path: Path) -> None:
    motion_dir, object_map = _write_source_bank(tmp_path)
    payload = json.loads(object_map.read_text(encoding="utf-8"))
    mesh = tmp_path / "assets" / "object.obj"
    payload["clips"]["clip_a"].update(
        {
            "object_visual_mesh_path": "../assets/object.obj",
            "object_collision_mesh_paths": [str(mesh)],
        }
    )
    object_map.write_text(json.dumps(payload), encoding="utf-8")

    manifest = prepare_immutable_single_slot_bank(
        source_motion_dir=motion_dir,
        source_object_map=object_map,
        output_base=tmp_path / "generated",
    )
    output_root = Path(manifest["output_root"])
    published = json.loads((output_root / "_clip_object_urdf_map.json").read_text())
    entry = published["clips"]["clip_a"]
    assert (output_root / entry["object_visual_mesh_path"]).resolve() == mesh.resolve()
    assert (output_root / entry["object_collision_mesh_paths"][0]).resolve() == mesh.resolve()
