from __future__ import annotations

import importlib.util
import hashlib
import json
import math
import os
import stat
import struct
import subprocess
import sys
from pathlib import Path

import pytest


def _load_prepare_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "prepare_as_rank_shards.py"
    spec = importlib.util.spec_from_file_location("prepare_as_rank_shards", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_glb(path: Path, payload: dict) -> None:
    json_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    json_chunk = struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(json_chunk)) + json_chunk)


def _write_object(root: Path, name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    mesh = root / f"{name}.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    urdf = root / f"{name}.urdf"
    urdf.write_text(
        f"""
<robot name="{name}">
  <link name="baseLink">
    <visual>
      <geometry><mesh filename="{mesh.name}"/></geometry>
    </visual>
    <collision>
      <geometry><mesh filename="{mesh.name}"/></geometry>
    </collision>
  </link>
</robot>
""".strip(),
        encoding="utf-8",
    )
    return urdf


def _write_motion_bank(
    tmp_path: Path,
    *,
    clip_count: int,
    unique_urdfs: int,
    motion_transition_source: dict | None = None,
) -> tuple[Path, Path]:
    motion_dir = tmp_path / "motion"
    motion_dir.mkdir()
    urdf_dir = motion_dir / "_single_slot_urdfs"
    urdfs = [_write_object(urdf_dir, f"obj_{idx}") for idx in range(unique_urdfs)]

    clips = {}
    for idx in range(clip_count):
        clip_id = f"clip_{idx:03d}"
        (motion_dir / f"{clip_id}.npz").write_bytes(b"placeholder")
        urdf = urdfs[idx % unique_urdfs]
        clips[clip_id] = {
            "object_name": urdf.stem,
            "object_urdf_path": str(urdf.relative_to(motion_dir)),
            "object_mesh_path": str((urdf.with_suffix(".obj")).relative_to(motion_dir)),
        }

    object_map_payload = {"clips": clips}
    if motion_transition_source is not None:
        object_map_payload["motion_transition_source"] = motion_transition_source
    object_map = motion_dir / "_clip_object_urdf_map.json"
    object_map.write_text(json.dumps(object_map_payload, indent=2), encoding="utf-8")
    return motion_dir, object_map


def _clip_counts(output_root: Path, world_size: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rank in range(world_size):
        rank_dir = output_root / f"rank_{rank}"
        payload = json.loads((rank_dir / "_clip_object_urdf_map.json").read_text(encoding="utf-8"))
        for clip_id, entry in payload["clips"].items():
            counts[clip_id] = counts.get(clip_id, 0) + 1
            assert (rank_dir / f"{clip_id}.npz").is_symlink()
            assert (rank_dir / entry["object_urdf_path"]).resolve().is_file()
            assert (rank_dir / entry["object_mesh_path"]).resolve().is_file()
    return counts


def _make_owner_writable_for_same_uid_tamper(path: Path) -> None:
    """Explicitly bypass the publication mode barrier for adversarial tests."""

    path.chmod(stat.S_IMODE(path.lstat().st_mode) | stat.S_IWUSR)


def _assert_rank_shard_tree_is_mode_sealed(
    output_root: Path,
    manifest: dict,
) -> None:
    assert stat.S_IMODE(output_root.stat().st_mode) == 0o555
    assert stat.S_IMODE((output_root / ".generated_by_prepare_as_rank_shards").stat().st_mode) == 0o444
    assert stat.S_IMODE((output_root / "manifest.json").stat().st_mode) == 0o444
    for shard in manifest["shards"]:
        rank_dir = output_root / f"rank_{shard['rank']}"
        assert stat.S_IMODE(rank_dir.stat().st_mode) == 0o555
        assert stat.S_IMODE((rank_dir / "_clip_object_urdf_map.json").stat().st_mode) == 0o444
        assert stat.S_IMODE((rank_dir / "clip_ids.txt").stat().st_mode) == 0o444


def test_prepare_rank_shards_partitions_object_groups(tmp_path: Path) -> None:
    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=6, unique_urdfs=6)
    output_root = tmp_path / "rank_shards" / "ws3"

    manifest = module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=output_root,
        world_size=3,
    )

    assert manifest["strategy"] == "object_closure"
    assert manifest["exact_clip_partition"] is True
    assert manifest["duplicated_to_fill_empty_ranks"] is False
    assert all(shard["clip_count"] > 0 for shard in manifest["shards"])
    assert manifest["version"] == 3
    for shard in manifest["shards"]:
        assert shard["clip_ids"] == sorted(shard["clip_ids"])
        assert len(shard["object_map_sha256"]) == 64
        assert [record["name"] for record in shard["npz_files"]] == [
            f"{clip_id}.npz" for clip_id in shard["clip_ids"]
        ]
        assert len(shard["npz_content_sha256"]) == 64
    assert _clip_counts(output_root, 3) == {f"clip_{idx:03d}": 1 for idx in range(6)}


def test_new_rank_shard_tree_is_mode_sealed(tmp_path: Path) -> None:
    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=3, unique_urdfs=2)
    output_root = tmp_path / "rank_shards" / "ws2"

    manifest = module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=output_root,
        world_size=2,
    )

    assert manifest["version"] == 3
    _assert_rank_shard_tree_is_mode_sealed(output_root, manifest)


def test_prepare_rank_shards_splits_by_clip_when_world_size_exceeds_unique_urdfs(tmp_path: Path) -> None:
    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=5, unique_urdfs=2)
    output_root = tmp_path / "rank_shards" / "ws4"

    manifest = module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=output_root,
        world_size=4,
    )

    assert manifest["strategy"] == "clip"
    assert manifest["exact_clip_partition"] is True
    assert manifest["duplicated_to_fill_empty_ranks"] is False
    assert all(shard["clip_count"] > 0 for shard in manifest["shards"])
    assert _clip_counts(output_root, 4) == {f"clip_{idx:03d}": 1 for idx in range(5)}


def test_rank_shards_can_require_clip_counts_that_divide_envs_per_rank(
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(
        tmp_path,
        clip_count=15,
        unique_urdfs=15,
    )
    output_root = tmp_path / "rank_shards" / "ws6_e8"

    manifest = module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=output_root,
        world_size=6,
        environments_per_rank=8,
    )

    clip_counts = [shard["clip_count"] for shard in manifest["shards"]]
    assert sorted(clip_counts) == [1, 2, 2, 2, 4, 4]
    assert all(8 % count == 0 for count in clip_counts)
    assert manifest["environments_per_rank"] == 8
    assert manifest["rank_clip_counts_divide_environments_per_rank"] is True
    assert manifest["exact_clip_partition"] is True
    assert _clip_counts(output_root, 6) == {
        f"clip_{idx:03d}": 1 for idx in range(15)
    }

    verified = module.validate_published_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=output_root,
        world_size=6,
        environments_per_rank=8,
        expected_source_digest=manifest["source_digest"],
    )
    assert verified == manifest


def test_prepare_reuses_sealed_rank_shards_without_writing_parent(tmp_path: Path) -> None:
    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=2, unique_urdfs=2)
    output_root = tmp_path / "rank_shards" / "by-source" / "identity" / "ws2"
    manifest = module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=output_root,
        world_size=2,
    )

    output_root.parent.chmod(0o555)
    try:
        reused = module.prepare_rank_shards(
            motion_dir=motion_dir,
            object_map=object_map,
            output_root=output_root,
            world_size=2,
            expected_source_digest=manifest["source_digest"],
        )
    finally:
        output_root.parent.chmod(0o755)

    assert reused == manifest


def test_rank_shard_inverse_cover_weights_restore_global_uniform_clip_mass(tmp_path: Path) -> None:
    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=5, unique_urdfs=2)
    output_root = tmp_path / "rank_shards" / "ws8"

    manifest = module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=output_root,
        world_size=8,
    )

    assert manifest["duplicated_to_fill_empty_ranks"] is True
    global_contributions = {f"clip_{idx:03d}": 0.0 for idx in range(5)}
    for rank in range(8):
        payload = json.loads(
            (output_root / f"rank_{rank}" / "_clip_object_urdf_map.json").read_text(encoding="utf-8")
        )
        metadata = payload["rank_local_shard"]
        assert metadata["global_clip_count"] == 5
        local_weights = {
            clip_id: 1.0 / float(cover_count)
            for clip_id, cover_count in metadata["clip_cover_counts"].items()
        }
        local_mass = sum(local_weights.values())
        expected_rank_scale = 8.0 * local_mass / 5.0
        assert math.isclose(metadata["distributed_loss_weight"], expected_rank_scale)
        for clip_id, weight in local_weights.items():
            local_probability = weight / local_mass
            global_contributions[clip_id] += expected_rank_scale * local_probability / 8.0

    assert all(math.isclose(value, 1.0 / 5.0) for value in global_contributions.values())


def test_rank_shards_duplicate_to_environment_compatible_nonempty_ranks(
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(
        tmp_path,
        clip_count=30,
        unique_urdfs=30,
    )
    output_root = tmp_path / "rank_shards" / "ws32_e1024"

    manifest = module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=output_root,
        world_size=32,
        environments_per_rank=1024,
    )

    assert manifest["duplicated_to_fill_empty_ranks"] is True
    assert manifest["exact_clip_partition"] is False
    assert manifest["environments_per_rank"] == 1024
    assert manifest["rank_clip_counts_divide_environments_per_rank"] is True
    assert [shard["clip_count"] for shard in manifest["shards"]] == [1] * 32
    assert sorted(manifest["clip_cover_counts"].values()) == [1] * 28 + [2] * 2

    global_contributions = {f"clip_{idx:03d}": 0.0 for idx in range(30)}
    for rank in range(32):
        payload = json.loads(
            (output_root / f"rank_{rank}" / "_clip_object_urdf_map.json").read_text(
                encoding="utf-8"
            )
        )
        metadata = payload["rank_local_shard"]
        clip_id, cover_count = next(iter(metadata["clip_cover_counts"].items()))
        expected_rank_scale = 32.0 / (30.0 * float(cover_count))
        assert math.isclose(metadata["inverse_cover_mass"], 1.0 / float(cover_count))
        assert math.isclose(metadata["distributed_loss_weight"], expected_rank_scale)
        global_contributions[clip_id] += expected_rank_scale / 32.0

    assert all(
        math.isclose(value, 1.0 / 30.0)
        for value in global_contributions.values()
    )

    verified = module.validate_published_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=output_root,
        world_size=32,
        environments_per_rank=1024,
        expected_source_digest=manifest["source_digest"],
    )
    assert verified == manifest


def test_rank_shards_preserve_source30_transition_semantics_for_active1_ws8(
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    transition_source = {
        "version": 1,
        "source_clip_count": 30,
        "source_semantics": "global_multi_clip_runtime",
    }
    motion_dir, object_map = _write_motion_bank(
        tmp_path,
        clip_count=1,
        unique_urdfs=1,
        motion_transition_source=transition_source,
    )
    output_root = tmp_path / "rank_shards" / "ws8"

    manifest = module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=output_root,
        world_size=8,
    )

    assert manifest["version"] == 3
    assert manifest["clip_count"] == 1
    assert manifest["motion_transition_source"] == transition_source
    assert manifest["clip_cover_counts"] == {"clip_000": 8}
    assert manifest["duplicated_to_fill_empty_ranks"] is True
    for rank in range(8):
        payload = json.loads(
            (output_root / f"rank_{rank}" / "_clip_object_urdf_map.json").read_text(
                encoding="utf-8"
            )
        )
        metadata = payload["rank_local_shard"]
        assert payload["motion_transition_source"] == transition_source
        assert metadata["motion_transition_source"] == transition_source
        assert metadata["global_clip_count"] == 1
        assert metadata["clip_cover_counts"] == {"clip_000": 8}
        assert math.isclose(metadata["inverse_cover_mass"], 1.0 / 8.0)
        assert math.isclose(metadata["distributed_loss_weight"], 1.0)


def test_rank_shards_infer_native_single_clip_static_transition_source(tmp_path: Path) -> None:
    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=1, unique_urdfs=1)
    output_root = tmp_path / "rank_shards" / "ws1"
    expected_transition_source = {
        "version": 1,
        "source_clip_count": 1,
        "source_semantics": "single_clip_static",
    }

    manifest = module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=output_root,
        world_size=1,
    )

    payload = json.loads(
        (output_root / "rank_0" / "_clip_object_urdf_map.json").read_text(encoding="utf-8")
    )
    assert manifest["motion_transition_source"] == expected_transition_source
    assert payload["motion_transition_source"] == expected_transition_source
    assert payload["rank_local_shard"]["motion_transition_source"] == expected_transition_source
    assert payload["rank_local_shard"]["global_clip_count"] == 1
    assert math.isclose(payload["rank_local_shard"]["distributed_loss_weight"], 1.0)


@pytest.mark.parametrize("tamper_target", ["manifest", "map_top_level", "rank_block"])
def test_rank_local_validator_rejects_transition_source_provenance_drift(
    monkeypatch,
    tmp_path: Path,
    tamper_target: str,
) -> None:
    from holosoma.utils.rank_local_shards import resolve_rank_local_motion_path

    module = _load_prepare_module()
    transition_source = {
        "version": 1,
        "source_clip_count": 30,
        "source_semantics": "global_multi_clip_runtime",
    }
    motion_dir, object_map = _write_motion_bank(
        tmp_path,
        clip_count=1,
        unique_urdfs=1,
        motion_transition_source=transition_source,
    )
    root = tmp_path / "rank_shards" / "ws8"
    module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=root,
        world_size=8,
    )

    rank = 0
    local_map = root / f"rank_{rank}" / "_clip_object_urdf_map.json"
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tamper_target == "manifest":
        manifest["motion_transition_source"]["source_clip_count"] = 29
    else:
        payload = json.loads(local_map.read_text(encoding="utf-8"))
        provenance = (
            payload["motion_transition_source"]
            if tamper_target == "map_top_level"
            else payload["rank_local_shard"]["motion_transition_source"]
        )
        provenance["source_clip_count"] = 29
        _make_owner_writable_for_same_uid_tamper(local_map)
        local_map.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        manifest["shards"][rank]["object_map_sha256"] = hashlib.sha256(
            local_map.read_bytes()
        ).hexdigest()
    _make_owner_writable_for_same_uid_tamper(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    monkeypatch.setenv("HOLOSOMA_RANK_LOCAL_MOTION_ROOT", str(root))
    monkeypatch.setenv("HOLOSOMA_RANK_LOCAL_SHARDING_ENABLED", "1")
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", "8")
    with pytest.raises(RuntimeError, match="motion transition provenance"):
        resolve_rank_local_motion_path(motion_dir)


def test_prepare_rank_shards_serializes_concurrent_publishers(tmp_path: Path) -> None:
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=5, unique_urdfs=2)
    output_root = tmp_path / "rank_shards" / "ws4"
    script = Path(__file__).resolve().parents[2] / "scripts" / "prepare_as_rank_shards.py"
    command = [
        sys.executable,
        str(script),
        "--motion-dir",
        str(motion_dir),
        "--object-map",
        str(object_map),
        "--output-root",
        str(output_root),
        "--world-size",
        "4",
    ]

    processes = [
        subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for _ in range(6)
    ]
    results = [process.communicate(timeout=30) for process in processes]
    assert all(process.returncode == 0 for process in processes), results
    assert all(stdout.strip() == str(output_root) for stdout, _ in results)
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["world_size"] == 4
    assert len(manifest["source_digest"]) == 64
    assert not list(output_root.parent.glob(f".{output_root.name}.tmp-*"))


def test_prepare_rank_shards_republishes_complete_sibling_tree_atomically(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=4, unique_urdfs=2)
    output_root = tmp_path / "rank_shards" / "ws2"
    first = module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=output_root,
        world_size=2,
    )
    open_old_manifest = (output_root / "manifest.json").open("rb")
    mesh = next((motion_dir / "_single_slot_urdfs").glob("*.obj"))
    mesh.write_text(mesh.read_text(encoding="utf-8") + "# source changed\n", encoding="utf-8")

    original_write = module._write_rank_shard
    observations: list[Path] = []

    def observe_write(**kwargs):
        write_root = kwargs["rank_dir"].parent
        observations.append(write_root)
        assert write_root.parent == output_root.parent
        assert write_root != output_root
        # The public path is never cleaned while the replacement is built.
        visible = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
        assert visible["source_digest"] == first["source_digest"]
        return original_write(**kwargs)

    monkeypatch.setattr(module, "_write_rank_shard", observe_write)
    second = module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=output_root,
        world_size=2,
    )
    assert observations
    assert second["source_digest"] != first["source_digest"]
    assert json.loads((output_root / "manifest.json").read_text())["source_digest"] == second["source_digest"]
    open_old_manifest.seek(0)
    assert json.load(open_old_manifest)["source_digest"] == first["source_digest"]
    open_old_manifest.close()
    assert not list(output_root.parent.glob(f".{output_root.name}.tmp-*"))


def test_prepare_rank_shards_rejects_source_drift_during_build(monkeypatch, tmp_path: Path) -> None:
    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=4, unique_urdfs=2)
    output_root = tmp_path / "rank_shards" / "ws2"
    first = module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=output_root,
        world_size=2,
    )
    meshes = sorted((motion_dir / "_single_slot_urdfs").glob("*.obj"))
    meshes[0].write_text(meshes[0].read_text(encoding="utf-8") + "# trigger rebuild\n", encoding="utf-8")

    original_write = module._write_rank_shard
    changed_during_build = False

    def drift_after_write(**kwargs):
        nonlocal changed_during_build
        result = original_write(**kwargs)
        if not changed_during_build:
            meshes[1].write_text(
                meshes[1].read_text(encoding="utf-8") + "# concurrent source mutation\n",
                encoding="utf-8",
            )
            changed_during_build = True
        return result

    monkeypatch.setattr(module, "_write_rank_shard", drift_after_write)
    with pytest.raises(RuntimeError, match="source changed while its replacement was being built"):
        module.prepare_rank_shards(
            motion_dir=motion_dir,
            object_map=object_map,
            output_root=output_root,
            world_size=2,
        )
    assert json.loads((output_root / "manifest.json").read_text())["source_digest"] == first["source_digest"]
    assert not list(output_root.parent.glob(f".{output_root.name}.tmp-*"))


def test_source_digest_rejects_asset_drift_between_closure_scan_and_hash(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=2, unique_urdfs=1)
    mesh = next((motion_dir / "_single_slot_urdfs").glob("*.obj"))
    original_source_file_record = module._source_file_record
    mutated = False

    def mutate_after_closure_scan(path):
        nonlocal mutated
        if not mutated and Path(path).resolve() == object_map.resolve():
            mesh.write_text(
                mesh.read_text(encoding="utf-8") + "# mutation between closure scan and hash\n",
                encoding="utf-8",
            )
            mutated = True
        return original_source_file_record(path)

    monkeypatch.setattr(module, "_source_file_record", mutate_after_closure_scan)
    with pytest.raises(RuntimeError, match="Rank-shard source changed"):
        module.compute_rank_shard_source_digest(
            motion_dir=motion_dir,
            object_map=object_map,
            world_size=2,
        )
    assert mutated


def test_source_digest_rejects_motion_listing_drift_during_scan(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=2, unique_urdfs=1)
    original_source_file_record = module._source_file_record
    mutated = False

    def add_motion_after_active_listing(path):
        nonlocal mutated
        if not mutated and Path(path).resolve() == object_map.resolve():
            (motion_dir / "concurrent_clip.npz").write_bytes(b"concurrent motion")
            mutated = True
        return original_source_file_record(path)

    monkeypatch.setattr(module, "_source_file_record", add_motion_after_active_listing)
    with pytest.raises(RuntimeError, match="motion-file listing changed"):
        module.compute_rank_shard_source_digest(
            motion_dir=motion_dir,
            object_map=object_map,
            world_size=2,
        )


def test_source_digest_accepts_stable_symlink_motion_root_and_detects_retarget(tmp_path: Path) -> None:
    module = _load_prepare_module()
    source_root = tmp_path / "source"
    source_root.mkdir()
    real_root, object_map = _write_motion_bank(source_root, clip_count=2, unique_urdfs=1)
    alias = tmp_path / "motion_alias"
    alias.symlink_to(real_root, target_is_directory=True)

    digest = module.compute_rank_shard_source_digest(
        motion_dir=alias,
        object_map=object_map,
        world_size=2,
    )
    assert len(digest) == 64

    replacement_root = tmp_path / "replacement"
    replacement_root.mkdir()
    for npz_path in real_root.glob("*.npz"):
        (replacement_root / npz_path.name).write_bytes(npz_path.read_bytes())

    guard = module._SourceScanGuard()
    guard.record_motion_listing(alias, list(alias.glob("*.npz")))
    guard.verify()
    alias.unlink()
    alias.symlink_to(replacement_root, target_is_directory=True)
    with pytest.raises(RuntimeError, match="motion-file listing changed"):
        guard.verify()


def test_source_digest_rejects_equal_size_asset_rewrite_with_restored_mtime(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=2, unique_urdfs=1)
    urdf = motion_dir / "_single_slot_urdfs" / "obj_0.urdf"
    alternate_mesh = urdf.with_name("obj_1.obj")
    alternate_mesh.write_bytes(urdf.with_suffix(".obj").read_bytes())
    original_source_file_record = module._source_file_record
    mutated = False

    def mutate_after_closure_scan(path):
        nonlocal mutated
        if not mutated and Path(path).resolve() == object_map.resolve():
            before = urdf.stat()
            payload = urdf.read_text(encoding="utf-8").replace("obj_0.obj", "obj_1.obj")
            assert len(payload.encode("utf-8")) == before.st_size
            urdf.write_text(payload, encoding="utf-8")
            os.utime(urdf, ns=(before.st_atime_ns, before.st_mtime_ns))
            after = urdf.stat()
            assert (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) == (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
            assert after.st_ctime_ns != before.st_ctime_ns
            mutated = True
        return original_source_file_record(path)

    monkeypatch.setattr(module, "_source_file_record", mutate_after_closure_scan)
    with pytest.raises(RuntimeError, match="Rank-shard source changed"):
        module.compute_rank_shard_source_digest(
            motion_dir=motion_dir,
            object_map=object_map,
            world_size=2,
        )


def test_source_digest_rejects_urdf_symlink_retarget_after_closure_scan(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=2, unique_urdfs=1)
    original_urdf = motion_dir / "_single_slot_urdfs" / "obj_0.urdf"
    replacement_urdf = motion_dir / "_single_slot_urdfs" / "obj_replacement.urdf"
    replacement_urdf.write_text(original_urdf.read_text(encoding="utf-8"), encoding="utf-8")
    urdf_alias = motion_dir / "_single_slot_urdfs" / "active.urdf"
    urdf_alias.symlink_to(original_urdf.name)

    payload = json.loads(object_map.read_text(encoding="utf-8"))
    for entry in payload["clips"].values():
        entry["object_urdf_path"] = str(urdf_alias)
    object_map.write_text(json.dumps(payload), encoding="utf-8")

    original_source_file_record = module._source_file_record
    retargeted = False

    def retarget_after_closure_scan(path):
        nonlocal retargeted
        if not retargeted and Path(path).resolve() == object_map.resolve():
            urdf_alias.unlink()
            urdf_alias.symlink_to(replacement_urdf.name)
            retargeted = True
        return original_source_file_record(path)

    monkeypatch.setattr(module, "_source_file_record", retarget_after_closure_scan)
    with pytest.raises(RuntimeError, match="Rank-shard source changed"):
        module.compute_rank_shard_source_digest(
            motion_dir=motion_dir,
            object_map=object_map,
            world_size=2,
        )
    assert retargeted


def test_expected_digest_mode_never_repairs_existing_identity_in_place(tmp_path: Path) -> None:
    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=4, unique_urdfs=2)
    output_root = tmp_path / "rank_shards" / "by-source" / "identity" / "ws2"
    source_digest = module.compute_rank_shard_source_digest(
        motion_dir=motion_dir,
        object_map=object_map,
        world_size=2,
    )
    module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=output_root,
        world_size=2,
        expected_source_digest=source_digest,
    )
    local_map = output_root / "rank_0" / "_clip_object_urdf_map.json"
    _make_owner_writable_for_same_uid_tamper(local_map)
    local_map.write_text(local_map.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    tampered = local_map.read_bytes()

    with pytest.raises(ValueError, match="must never repair or redirect"):
        module.prepare_rank_shards(
            motion_dir=motion_dir,
            object_map=object_map,
            output_root=output_root,
            world_size=2,
            expected_source_digest=source_digest,
        )
    assert local_map.read_bytes() == tampered


@pytest.mark.parametrize("expected_digest_mode", [False, True])
def test_prepare_rejects_final_output_symlink_without_writing_its_target(
    tmp_path: Path,
    expected_digest_mode: bool,
) -> None:
    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=1, unique_urdfs=1)
    output_parent = tmp_path / "rank_shards"
    output_parent.mkdir()
    output_root = output_parent / "ws1"
    redirected_target = tmp_path / "must-not-be-created"
    output_root.symlink_to(redirected_target, target_is_directory=True)
    source_digest = module.compute_rank_shard_source_digest(
        motion_dir=motion_dir,
        object_map=object_map,
        world_size=1,
    )

    kwargs = {}
    if expected_digest_mode:
        kwargs["expected_source_digest"] = source_digest
    with pytest.raises(ValueError, match="output root must not be a symlink"):
        module.prepare_rank_shards(
            motion_dir=motion_dir,
            object_map=object_map,
            output_root=output_root,
            world_size=1,
            **kwargs,
        )

    assert output_root.is_symlink()
    assert not redirected_target.exists()
    assert not (output_parent / ".ws1.lock").exists()


@pytest.mark.parametrize("expected_digest_mode", [False, True])
def test_prepare_rejects_symlinked_output_parent_without_writing_its_target(
    tmp_path: Path,
    expected_digest_mode: bool,
) -> None:
    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=1, unique_urdfs=1)
    redirected_parent = tmp_path / "must-stay-empty"
    redirected_parent.mkdir()
    lexical_parent = tmp_path / "rank_shards"
    lexical_parent.symlink_to(redirected_parent, target_is_directory=True)
    output_root = lexical_parent / "ws1"
    source_digest = module.compute_rank_shard_source_digest(
        motion_dir=motion_dir,
        object_map=object_map,
        world_size=1,
    )

    kwargs = {}
    if expected_digest_mode:
        kwargs["expected_source_digest"] = source_digest
    with pytest.raises(ValueError, match="output parent must not be a symlink"):
        module.prepare_rank_shards(
            motion_dir=motion_dir,
            object_map=object_map,
            output_root=output_root,
            world_size=1,
            **kwargs,
        )

    assert lexical_parent.is_symlink()
    assert list(redirected_parent.iterdir()) == []


def test_prepare_rejects_existing_non_directory_output_parent(tmp_path: Path) -> None:
    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=1, unique_urdfs=1)
    non_directory_parent = tmp_path / "not-a-directory"
    original_bytes = b"preserve-parent-bytes\n"
    non_directory_parent.write_bytes(original_bytes)

    with pytest.raises(ValueError, match="output parent is not a directory"):
        module.prepare_rank_shards(
            motion_dir=motion_dir,
            object_map=object_map,
            output_root=non_directory_parent / "ws1",
            world_size=1,
        )

    assert non_directory_parent.read_bytes() == original_bytes


def test_output_lock_revalidates_parent_after_creating_missing_components(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=1, unique_urdfs=1)
    output_root = tmp_path / "new-parent" / "nested" / "ws1"
    redirected_parent = tmp_path / "mkdir-race-target"
    redirected_parent.mkdir()
    displaced_parent = tmp_path / "mkdir-race-original"
    original_mkdir = Path.mkdir
    redirected = False

    def redirect_after_parent_creation(path: Path, *args, **kwargs):
        nonlocal redirected
        result = original_mkdir(path, *args, **kwargs)
        if path == output_root.parent and not redirected:
            path.rename(displaced_parent)
            path.symlink_to(redirected_parent, target_is_directory=True)
            redirected = True
        return result

    monkeypatch.setattr(Path, "mkdir", redirect_after_parent_creation)
    with pytest.raises(ValueError, match="output parent must not be a symlink"):
        module.prepare_rank_shards(
            motion_dir=motion_dir,
            object_map=object_map,
            output_root=output_root,
            world_size=1,
        )

    assert redirected
    assert list(redirected_parent.iterdir()) == []


def test_output_lock_revalidates_parent_after_lock_acquisition(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=1, unique_urdfs=1)
    output_parent = tmp_path / "rank-shards"
    output_parent.mkdir()
    output_root = output_parent / "ws1"
    redirected_parent = tmp_path / "flock-race-target"
    redirected_parent.mkdir()
    displaced_parent = tmp_path / "flock-race-original"
    original_flock = module.fcntl.flock
    redirected = False

    def redirect_after_lock(descriptor: int, operation: int):
        nonlocal redirected
        result = original_flock(descriptor, operation)
        if operation == module.fcntl.LOCK_EX and not redirected:
            output_parent.rename(displaced_parent)
            output_parent.symlink_to(redirected_parent, target_is_directory=True)
            redirected = True
        return result

    monkeypatch.setattr(module.fcntl, "flock", redirect_after_lock)
    with pytest.raises(ValueError, match="output parent must not be a symlink"):
        module.prepare_rank_shards(
            motion_dir=motion_dir,
            object_map=object_map,
            output_root=output_root,
            world_size=1,
        )

    assert redirected
    assert list(redirected_parent.iterdir()) == []


def test_validate_published_rank_shards_rejects_final_and_parent_symlink_aliases(
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=1, unique_urdfs=1)
    output_root = tmp_path / "rank_shards" / "ws1"
    manifest = module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=output_root,
        world_size=1,
    )
    published_bytes = {
        path.relative_to(output_root).as_posix(): path.read_bytes()
        for path in output_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    final_alias = tmp_path / "final-alias"
    final_alias.symlink_to(output_root, target_is_directory=True)
    parent_alias = tmp_path / "parent-alias"
    parent_alias.symlink_to(output_root.parent, target_is_directory=True)

    for alias in (final_alias, parent_alias / output_root.name):
        with pytest.raises(ValueError, match="must not be a symlink"):
            module.validate_published_rank_shards(
                motion_dir=motion_dir,
                object_map=object_map,
                output_root=alias,
                world_size=1,
                expected_source_digest=manifest["source_digest"],
            )

    assert published_bytes == {
        path.relative_to(output_root).as_posix(): path.read_bytes()
        for path in output_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def test_cli_rejects_relative_output_symlink_without_creating_target(tmp_path: Path) -> None:
    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=1, unique_urdfs=1)
    output_parent = tmp_path / "rank_shards"
    output_parent.mkdir()
    redirected_target = tmp_path / "cli-must-not-be-created"
    (output_parent / "ws1").symlink_to(redirected_target, target_is_directory=True)
    source_digest = module.compute_rank_shard_source_digest(
        motion_dir=motion_dir,
        object_map=object_map,
        world_size=1,
    )
    script = Path(__file__).resolve().parents[2] / "scripts" / "prepare_as_rank_shards.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--motion-dir",
            str(motion_dir),
            "--object-map",
            str(object_map),
            "--output-root",
            "rank_shards/ws1",
            "--world-size",
            "1",
            "--expected-source-digest",
            source_digest,
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "output root must not be a symlink" in result.stderr
    assert not redirected_target.exists()
    assert not (output_parent / ".ws1.lock").exists()


def test_validate_published_rank_shards_is_read_only_and_source_derived(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=2, unique_urdfs=2)
    output_root = tmp_path / "rank_shards" / "ws2"
    manifest = module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=output_root,
        world_size=2,
    )
    mtimes_before = {
        path.relative_to(output_root).as_posix(): path.lstat().st_mtime_ns
        for path in output_root.rglob("*")
    }

    def unexpected_mutation(*_args, **_kwargs):
        raise AssertionError("verify-only API reached a publication primitive")

    monkeypatch.setattr(module, "_output_lock", unexpected_mutation)
    monkeypatch.setattr(module, "_atomic_publish", unexpected_mutation)
    monkeypatch.setattr(module, "_seal_generated_tree", unexpected_mutation)
    verified = module.validate_published_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=output_root,
        world_size=2,
        expected_source_digest=manifest["source_digest"],
    )
    assert verified == manifest
    assert mtimes_before == {
        path.relative_to(output_root).as_posix(): path.lstat().st_mtime_ns
        for path in output_root.rglob("*")
    }

    missing_root = tmp_path / "verify-must-not-create" / "ws2"
    with pytest.raises(ValueError, match="does not exist"):
        module.validate_published_rank_shards(
            motion_dir=motion_dir,
            object_map=object_map,
            output_root=missing_root,
            world_size=2,
            expected_source_digest=manifest["source_digest"],
        )
    assert not missing_root.parent.exists()


def test_prepare_migrates_byte_current_v3_tree_to_sealed_modes_without_republish(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=2, unique_urdfs=2)
    output_root = tmp_path / "rank_shards" / "by-source" / "identity" / "ws2"
    source_digest = module.compute_rank_shard_source_digest(
        motion_dir=motion_dir,
        object_map=object_map,
        world_size=2,
    )
    manifest = module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=output_root,
        world_size=2,
        expected_source_digest=source_digest,
    )
    published_bytes = {
        path.relative_to(output_root).as_posix(): path.read_bytes()
        for path in output_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }

    for path in output_root.rglob("*"):
        if not path.is_symlink() and (path.is_file() or path.is_dir()):
            _make_owner_writable_for_same_uid_tamper(path)
    _make_owner_writable_for_same_uid_tamper(output_root)

    def unexpected_publish(*_args, **_kwargs):
        raise AssertionError("mode-only v3 migration attempted to republish bytes")

    monkeypatch.setattr(module, "_atomic_publish", unexpected_publish)
    migrated = module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=output_root,
        world_size=2,
        expected_source_digest=source_digest,
    )

    assert migrated == manifest
    assert published_bytes == {
        path.relative_to(output_root).as_posix(): path.read_bytes()
        for path in output_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    _assert_rank_shard_tree_is_mode_sealed(output_root, manifest)


@pytest.mark.parametrize(
    "writable_target",
    ["root", "marker", "manifest", "rank_dir", "object_map", "clip_ids"],
)
def test_validate_published_rank_shards_rejects_writable_tree_without_repair(
    monkeypatch,
    tmp_path: Path,
    writable_target: str,
) -> None:
    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=1, unique_urdfs=1)
    output_root = tmp_path / "rank_shards" / "ws1"
    manifest = module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=output_root,
        world_size=1,
    )
    rank_dir = output_root / "rank_0"
    targets = {
        "root": output_root,
        "marker": output_root / ".generated_by_prepare_as_rank_shards",
        "manifest": output_root / "manifest.json",
        "rank_dir": rank_dir,
        "object_map": rank_dir / "_clip_object_urdf_map.json",
        "clip_ids": rank_dir / "clip_ids.txt",
    }
    target = targets[writable_target]
    _make_owner_writable_for_same_uid_tamper(target)
    mode_before = stat.S_IMODE(target.lstat().st_mode)
    mtime_before = target.lstat().st_mtime_ns

    def unexpected_mutation(*_args, **_kwargs):
        raise AssertionError("verify-only API attempted to repair writable publication modes")

    monkeypatch.setattr(module, "_output_lock", unexpected_mutation)
    monkeypatch.setattr(module, "_atomic_publish", unexpected_mutation)
    monkeypatch.setattr(module, "_seal_generated_tree", unexpected_mutation)
    with pytest.raises(ValueError, match="source-derived deterministic plan"):
        module.validate_published_rank_shards(
            motion_dir=motion_dir,
            object_map=object_map,
            output_root=output_root,
            world_size=1,
            expected_source_digest=manifest["source_digest"],
        )

    assert stat.S_IMODE(target.lstat().st_mode) == mode_before
    assert target.lstat().st_mtime_ns == mtime_before


def test_expected_digest_mode_rejects_coherent_rank_map_and_manifest_asset_tamper(
    tmp_path: Path,
) -> None:
    from holosoma.utils.rank_local_shards import _validated_rank_local_shard_payload

    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=2, unique_urdfs=2)
    output_root = tmp_path / "rank_shards" / "by-source" / "identity" / "ws2"
    source_digest = module.compute_rank_shard_source_digest(
        motion_dir=motion_dir,
        object_map=object_map,
        world_size=2,
    )
    module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=output_root,
        world_size=2,
        expected_source_digest=source_digest,
    )

    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rank_zero = manifest["shards"][0]
    rank_one = manifest["shards"][1]
    rank_zero_map = output_root / "rank_0" / "_clip_object_urdf_map.json"
    rank_one_map = output_root / "rank_1" / "_clip_object_urdf_map.json"
    rank_zero_payload = json.loads(rank_zero_map.read_text(encoding="utf-8"))
    rank_one_payload = json.loads(rank_one_map.read_text(encoding="utf-8"))
    rank_zero_clip = rank_zero["clip_ids"][0]
    rank_one_clip = rank_one["clip_ids"][0]

    # Keep the assignment/NPZ internally consistent but redirect rank zero's
    # simulator object to rank one's valid assets.  The old reuse check trusted
    # a correspondingly edited manifest SHA and therefore accepted this.
    for field in ("object_urdf_path", "object_mesh_path"):
        rank_zero_payload["clips"][rank_zero_clip][field] = rank_one_payload["clips"][
            rank_one_clip
        ][field]
    _make_owner_writable_for_same_uid_tamper(rank_zero_map)
    rank_zero_map.write_text(
        json.dumps(rank_zero_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest["shards"][0]["object_map_sha256"] = hashlib.sha256(
        rank_zero_map.read_bytes()
    ).hexdigest()
    _make_owner_writable_for_same_uid_tamper(manifest_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    # This is a coherent self-signed tree under the legacy verifier: the map
    # and manifest agree and every redirected asset exists.
    validated_payload = _validated_rank_local_shard_payload(
        str(output_root / "rank_0"),
        strict=True,
    )
    assert validated_payload["clips"][rank_zero_clip]["object_urdf_path"] == (
        rank_one_payload["clips"][rank_one_clip]["object_urdf_path"]
    )
    with pytest.raises(ValueError, match="source-derived deterministic plan"):
        module.validate_published_rank_shards(
            motion_dir=motion_dir,
            object_map=object_map,
            output_root=output_root,
            world_size=2,
            expected_source_digest=source_digest,
        )

    with pytest.raises(ValueError, match="must never repair or redirect"):
        module.prepare_rank_shards(
            motion_dir=motion_dir,
            object_map=object_map,
            output_root=output_root,
            world_size=2,
            expected_source_digest=source_digest,
        )


def test_expected_digest_mode_rejects_coherent_rank_assignment_and_manifest_tamper(
    tmp_path: Path,
) -> None:
    from holosoma.utils.rank_local_shards import _validated_rank_local_shard_payload

    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=2, unique_urdfs=2)
    output_root = tmp_path / "rank_shards" / "by-source" / "identity" / "ws2"
    source_digest = module.compute_rank_shard_source_digest(
        motion_dir=motion_dir,
        object_map=object_map,
        world_size=2,
    )
    module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=output_root,
        world_size=2,
        expected_source_digest=source_digest,
    )

    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_shards = json.loads(json.dumps(manifest["shards"]))
    original_payloads = [
        json.loads(
            (output_root / f"rank_{rank}" / "_clip_object_urdf_map.json").read_text(
                encoding="utf-8"
            )
        )
        for rank in range(2)
    ]
    original_link_targets = [
        os.readlink(
            output_root
            / f"rank_{rank}"
            / f"{original_shards[rank]['clip_ids'][0]}.npz"
        )
        for rank in range(2)
    ]

    tampered_shards = []
    for rank in range(2):
        source_rank = 1 - rank
        rank_dir = output_root / f"rank_{rank}"
        old_clip = original_shards[rank]["clip_ids"][0]
        new_clip = original_shards[source_rank]["clip_ids"][0]
        _make_owner_writable_for_same_uid_tamper(rank_dir)
        (rank_dir / f"{old_clip}.npz").unlink()
        (rank_dir / f"{new_clip}.npz").symlink_to(original_link_targets[source_rank])

        payload = json.loads(json.dumps(original_payloads[source_rank]))
        payload["rank_local_shard"]["rank"] = rank
        local_map = rank_dir / "_clip_object_urdf_map.json"
        _make_owner_writable_for_same_uid_tamper(local_map)
        local_map.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        clip_ids_path = rank_dir / "clip_ids.txt"
        _make_owner_writable_for_same_uid_tamper(clip_ids_path)
        clip_ids_path.write_text(f"{new_clip}\n", encoding="utf-8")

        shard = json.loads(json.dumps(original_shards[source_rank]))
        shard["rank"] = rank
        shard["dir"] = str(rank_dir)
        shard["object_map_sha256"] = hashlib.sha256(local_map.read_bytes()).hexdigest()
        tampered_shards.append(shard)
    manifest["shards"] = tampered_shards
    _make_owner_writable_for_same_uid_tamper(manifest_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for rank in range(2):
        payload = _validated_rank_local_shard_payload(
            str(output_root / f"rank_{rank}"),
            strict=True,
        )
        assert sorted(payload["clips"]) == tampered_shards[rank]["clip_ids"]
    with pytest.raises(ValueError, match="source-derived deterministic plan"):
        module.validate_published_rank_shards(
            motion_dir=motion_dir,
            object_map=object_map,
            output_root=output_root,
            world_size=2,
            expected_source_digest=source_digest,
        )

    with pytest.raises(ValueError, match="must never repair or redirect"):
        module.prepare_rank_shards(
            motion_dir=motion_dir,
            object_map=object_map,
            output_root=output_root,
            world_size=2,
            expected_source_digest=source_digest,
        )


def test_current_fast_path_reverifies_source_guard_before_return(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=2, unique_urdfs=2)
    output_root = tmp_path / "rank_shards" / "ws2"
    module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=output_root,
        world_size=2,
    )
    mesh = next((motion_dir / "_single_slot_urdfs").glob("*.obj"))
    original_current = module._published_output_is_current
    mutated = False

    def mutate_after_current_check(*args, **kwargs):
        nonlocal mutated
        result = original_current(*args, **kwargs)
        if result[0] and not mutated:
            mesh.write_text(
                mesh.read_text(encoding="utf-8") + "# drift before current return\n",
                encoding="utf-8",
            )
            mutated = True
        return result

    monkeypatch.setattr(module, "_published_output_is_current", mutate_after_current_check)
    with pytest.raises(RuntimeError, match="Rank-shard source changed"):
        module.prepare_rank_shards(
            motion_dir=motion_dir,
            object_map=object_map,
            output_root=output_root,
            world_size=2,
        )
    assert mutated


@pytest.mark.parametrize(
    "tamper_target",
    ["root_extra", "rank_extra", "clip_list", "npz_link_target"],
)
def test_expected_plan_rejects_namespace_clip_list_and_link_target_drift(
    tmp_path: Path,
    tamper_target: str,
) -> None:
    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=1, unique_urdfs=1)
    output_root = tmp_path / "rank_shards" / "by-source" / "identity" / "ws1"
    source_digest = module.compute_rank_shard_source_digest(
        motion_dir=motion_dir,
        object_map=object_map,
        world_size=1,
    )
    manifest = module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=output_root,
        world_size=1,
        expected_source_digest=source_digest,
    )
    rank_dir = output_root / "rank_0"
    clip_id = manifest["shards"][0]["clip_ids"][0]
    if tamper_target == "root_extra":
        _make_owner_writable_for_same_uid_tamper(output_root)
        (output_root / "ignored.txt").write_text("extra\n", encoding="utf-8")
    elif tamper_target == "rank_extra":
        _make_owner_writable_for_same_uid_tamper(rank_dir)
        (rank_dir / "ignored.txt").write_text("extra\n", encoding="utf-8")
    elif tamper_target == "clip_list":
        clip_ids_path = rank_dir / "clip_ids.txt"
        _make_owner_writable_for_same_uid_tamper(clip_ids_path)
        clip_ids_path.write_text(f"{clip_id}\n{clip_id}\n", encoding="utf-8")
    else:
        alternative = tmp_path / "same-bytes-alternative.npz"
        alternative.write_bytes((motion_dir / f"{clip_id}.npz").read_bytes())
        link = rank_dir / f"{clip_id}.npz"
        _make_owner_writable_for_same_uid_tamper(rank_dir)
        link.unlink()
        link.symlink_to(os.path.relpath(alternative, rank_dir))
        # Content-only verification cannot distinguish this retarget.
        assert link.read_bytes() == (motion_dir / f"{clip_id}.npz").read_bytes()

    with pytest.raises(ValueError, match="must never repair or redirect"):
        module.prepare_rank_shards(
            motion_dir=motion_dir,
            object_map=object_map,
            output_root=output_root,
            world_size=1,
            expected_source_digest=source_digest,
        )


def test_rank_symlinks_target_frozen_single_slot_payload_not_mutable_source(tmp_path: Path) -> None:
    module = _load_prepare_module()
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from prepare_immutable_single_slot_bank import prepare_immutable_single_slot_bank

    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=2, unique_urdfs=2)
    original_bytes = (motion_dir / "clip_000.npz").read_bytes()
    single_manifest = prepare_immutable_single_slot_bank(
        source_motion_dir=motion_dir,
        source_object_map=object_map,
        output_base=motion_dir / "immutable",
    )
    immutable_motion_dir = Path(single_manifest["output_root"])
    immutable_map = immutable_motion_dir / "_clip_object_urdf_map.json"
    source_digest = module.compute_rank_shard_source_digest(
        motion_dir=immutable_motion_dir,
        object_map=immutable_map,
        world_size=2,
    )
    rank_root = immutable_motion_dir / "_rank_shards" / "by-source" / source_digest / "ws2"
    rank_manifest = module.prepare_rank_shards(
        motion_dir=immutable_motion_dir,
        object_map=immutable_map,
        output_root=rank_root,
        world_size=2,
        expected_source_digest=source_digest,
    )
    rank = next(
        shard["rank"] for shard in rank_manifest["shards"] if "clip_000" in shard["clip_ids"]
    )
    rank_link = rank_root / f"rank_{rank}" / "clip_000.npz"
    assert rank_link.is_symlink()
    assert rank_link.resolve() == immutable_motion_dir / "clip_000.npz"
    assert rank_link.resolve().stat().st_mode & 0o222 == 0

    (motion_dir / "clip_000.npz").write_bytes(b"mutable-source-v2")
    assert rank_link.read_bytes() == original_bytes


def test_rank_shard_source_digest_covers_every_multi_object_asset(tmp_path: Path) -> None:
    module = _load_prepare_module()
    motion_dir = tmp_path / "motion"
    motion_dir.mkdir()
    urdf_a = _write_object(motion_dir / "objects", "obj_a")
    urdf_b = _write_object(motion_dir / "objects", "obj_b")
    (motion_dir / "clip_pair.npz").write_bytes(b"placeholder")
    object_map = motion_dir / "_clip_object_urdf_map.json"
    object_map.write_text(
        json.dumps(
            {
                "clips": {
                    "clip_pair": {
                        "object_urdf_path": str(urdf_a.relative_to(motion_dir)),
                        "object_urdf_paths": [str(urdf_b.relative_to(motion_dir))],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    first = module.compute_rank_shard_source_digest(
        motion_dir=motion_dir,
        object_map=object_map,
        world_size=2,
    )
    urdf_b.with_suffix(".obj").write_text("v 9 9 9\n", encoding="utf-8")
    second = module.compute_rank_shard_source_digest(
        motion_dir=motion_dir,
        object_map=object_map,
        world_size=2,
    )
    assert first != second


def test_rank_shard_source_digest_covers_gltf_external_buffers(tmp_path: Path) -> None:
    module = _load_prepare_module()
    motion_dir = tmp_path / "motion"
    asset_dir = motion_dir / "objects"
    asset_dir.mkdir(parents=True)
    (motion_dir / "clip_gltf.npz").write_bytes(b"placeholder")
    buffer_path = asset_dir / "geometry.bin"
    buffer_path.write_bytes(b"gltf-buffer-v1")
    gltf_path = asset_dir / "object.gltf"
    gltf_path.write_text(
        json.dumps({"asset": {"version": "2.0"}, "buffers": [{"uri": buffer_path.name, "byteLength": 14}]}),
        encoding="utf-8",
    )
    urdf_path = asset_dir / "object.urdf"
    urdf_path.write_text(
        '<robot name="object"><link name="baseLink"><visual><geometry>'
        '<mesh filename="object.gltf" />'
        '</geometry></visual></link></robot>',
        encoding="utf-8",
    )
    object_map = motion_dir / "_clip_object_urdf_map.json"
    object_map.write_text(
        json.dumps({"clips": {"clip_gltf": {"object_urdf_path": "objects/object.urdf"}}}),
        encoding="utf-8",
    )

    first = module.compute_rank_shard_source_digest(
        motion_dir=motion_dir,
        object_map=object_map,
        world_size=1,
    )
    buffer_path.write_bytes(b"gltf-buffer-v2")
    second = module.compute_rank_shard_source_digest(
        motion_dir=motion_dir,
        object_map=object_map,
        world_size=1,
    )
    assert first != second


def test_rank_shard_source_digest_covers_glb_external_buffers(tmp_path: Path) -> None:
    module = _load_prepare_module()
    motion_dir = tmp_path / "motion"
    asset_dir = motion_dir / "objects"
    asset_dir.mkdir(parents=True)
    (motion_dir / "clip_glb.npz").write_bytes(b"placeholder")
    buffer_path = asset_dir / "geometry.bin"
    buffer_path.write_bytes(b"glb-buffer-v1")
    extension_path = asset_dir / "extension.bin"
    extension_path.write_bytes(b"extension-v1")
    _write_glb(
        asset_dir / "object.glb",
        {
            "asset": {"version": "2.0"},
            "buffers": [{"uri": buffer_path.name, "byteLength": 13}],
            "extensions": {"VENDOR_external": {"uri": extension_path.name}},
        },
    )
    (asset_dir / "object.urdf").write_text(
        '<robot name="object"><link name="baseLink"><visual><geometry>'
        '<mesh filename="object.glb" />'
        "</geometry></visual></link></robot>",
        encoding="utf-8",
    )
    object_map = motion_dir / "_clip_object_urdf_map.json"
    object_map.write_text(
        json.dumps({"clips": {"clip_glb": {"object_urdf_path": "objects/object.urdf"}}}),
        encoding="utf-8",
    )

    first = module.compute_rank_shard_source_digest(
        motion_dir=motion_dir,
        object_map=object_map,
        world_size=1,
    )
    extension_path.write_bytes(b"extension-v2")
    second = module.compute_rank_shard_source_digest(
        motion_dir=motion_dir,
        object_map=object_map,
        world_size=1,
    )

    assert first != second


def test_rank_shard_source_digest_covers_pbr_mtl_texture_directives(tmp_path: Path) -> None:
    module = _load_prepare_module()
    motion_dir = tmp_path / "motion"
    assets = motion_dir / "objects"
    assets.mkdir(parents=True)
    (motion_dir / "clip_obj.npz").write_bytes(b"placeholder")
    texture = assets / "roughness.bin"
    texture.write_bytes(b"roughness-v1")
    (assets / "material.mtl").write_text("newmtl mat\nmap_Pr roughness.bin\n", encoding="utf-8")
    (assets / "object.obj").write_bytes(
        b"mtllib material.mtl\r\nv 0 0 0\r\nv 1 0 0\r\nv 0 1 0\r\nf 1 2 3\r\n",
    )
    (assets / "object.urdf").write_text(
        '<robot name="object"><link name="baseLink"><visual><geometry>'
        '<mesh filename="object.obj" />'
        "</geometry></visual></link></robot>",
        encoding="utf-8",
    )
    object_map = motion_dir / "_clip_object_urdf_map.json"
    object_map.write_text(
        json.dumps({"clips": {"clip_obj": {"object_urdf_path": "objects/object.urdf"}}}),
        encoding="utf-8",
    )
    first = module.compute_rank_shard_source_digest(
        motion_dir=motion_dir,
        object_map=object_map,
        world_size=1,
    )
    texture.write_bytes(b"roughness-v2")
    second = module.compute_rank_shard_source_digest(
        motion_dir=motion_dir,
        object_map=object_map,
        world_size=1,
    )
    assert first != second


def test_rank_shard_source_digest_covers_extensionless_collada_texture(tmp_path: Path) -> None:
    module = _load_prepare_module()
    motion_dir = tmp_path / "motion"
    assets = motion_dir / "objects"
    assets.mkdir(parents=True)
    (motion_dir / "clip_dae.npz").write_bytes(b"placeholder")
    texture = assets / "texture_without_suffix"
    texture.write_bytes(b"texture-v1")
    (assets / "object.dae").write_text(
        '<COLLADA><library_images><image id="tex"><init_from>'
        "texture_without_suffix"
        "</init_from></image></library_images></COLLADA>",
        encoding="utf-8",
    )
    (assets / "object.urdf").write_text(
        '<robot name="object"><link name="baseLink"><visual><geometry>'
        '<mesh filename="object.dae" />'
        "</geometry></visual></link></robot>",
        encoding="utf-8",
    )
    object_map = motion_dir / "_clip_object_urdf_map.json"
    object_map.write_text(
        json.dumps({"clips": {"clip_dae": {"object_urdf_path": "objects/object.urdf"}}}),
        encoding="utf-8",
    )
    first = module.compute_rank_shard_source_digest(
        motion_dir=motion_dir,
        object_map=object_map,
        world_size=1,
    )
    texture.write_bytes(b"texture-v2")
    second = module.compute_rank_shard_source_digest(
        motion_dir=motion_dir,
        object_map=object_map,
        world_size=1,
    )
    assert first != second


def test_rank_shard_closure_rejects_external_collada_document_reference(tmp_path: Path) -> None:
    module = _load_prepare_module()
    mesh_path = tmp_path / "object.dae"
    mesh_path.write_text(
        '<COLLADA><library_visual_scenes><visual_scene><node>'
        '<instance_geometry url="other.dae#geometry"/>'
        "</node></visual_scene></library_visual_scenes></COLLADA>",
        encoding="utf-8",
    )
    (tmp_path / "other.dae").write_text("<COLLADA/>", encoding="utf-8")

    with pytest.raises(ValueError, match="external document references are unsupported"):
        module.collect_local_mesh_asset_paths(mesh_path)


def test_rank_shard_source_digest_rejects_unclosed_mesh_format(tmp_path: Path) -> None:
    module = _load_prepare_module()
    motion_dir = tmp_path / "motion"
    asset_dir = motion_dir / "objects"
    asset_dir.mkdir(parents=True)
    (motion_dir / "clip_unknown.npz").write_bytes(b"placeholder")
    (asset_dir / "object.usd").write_bytes(b"#usda 1.0")
    (asset_dir / "object.urdf").write_text(
        '<robot name="object"><link name="baseLink"><visual><geometry>'
        '<mesh filename="object.usd" />'
        '</geometry></visual></link></robot>',
        encoding="utf-8",
    )
    object_map = motion_dir / "_clip_object_urdf_map.json"
    object_map.write_text(
        json.dumps({"clips": {"clip_unknown": {"object_urdf_path": "objects/object.urdf"}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no implemented transitive dependency closure"):
        module.compute_rank_shard_source_digest(
            motion_dir=motion_dir,
            object_map=object_map,
            world_size=1,
        )


def test_prepare_rank_shards_keeps_multi_object_clip_closure(tmp_path: Path) -> None:
    module = _load_prepare_module()
    motion_dir = tmp_path / "motion"
    motion_dir.mkdir()
    urdf_dir = motion_dir / "objects"
    urdf_a = _write_object(urdf_dir, "obj_a")
    urdf_b = _write_object(urdf_dir, "obj_b")
    urdf_c = _write_object(urdf_dir, "obj_c")
    for clip_id in ("clip_pair", "clip_single"):
        (motion_dir / f"{clip_id}.npz").write_bytes(b"placeholder")
    object_map = motion_dir / "_clip_object_urdf_map.json"
    object_map.write_text(
        json.dumps(
            {
                "clips": {
                    "clip_pair": {
                        "object_urdf_paths": [
                            str(urdf_a.relative_to(motion_dir)),
                            str(urdf_b.relative_to(motion_dir)),
                        ],
                        "object_mesh_paths": [
                            str(urdf_a.with_suffix(".obj").relative_to(motion_dir)),
                            str(urdf_b.with_suffix(".obj").relative_to(motion_dir)),
                        ],
                    },
                    "clip_single": {
                        "object_urdf_path": str(urdf_c.relative_to(motion_dir)),
                        "object_mesh_path": str(urdf_c.with_suffix(".obj").relative_to(motion_dir)),
                    },
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "rank_shards" / "ws2"

    manifest = module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=output_root,
        world_size=2,
    )

    assert manifest["strategy"] == "object_closure"
    pair_rank = None
    for rank in range(2):
        payload = json.loads((output_root / f"rank_{rank}" / "_clip_object_urdf_map.json").read_text())
        if "clip_pair" in payload["clips"]:
            pair_rank = rank
            entry = payload["clips"]["clip_pair"]
            assert len(entry["object_urdf_paths"]) == 2
            assert len(entry["object_mesh_paths"]) == 2
            assert all((output_root / f"rank_{rank}" / path).resolve().is_file() for path in entry["object_urdf_paths"])
            assert all((output_root / f"rank_{rank}" / path).resolve().is_file() for path in entry["object_mesh_paths"])
    assert pair_rank is not None


def test_rank_local_path_resolver_uses_local_rank(monkeypatch, tmp_path: Path) -> None:
    from holosoma.utils.rank_local_shards import (
        current_rank_local_shard_metadata,
        resolve_rank_local_motion_path,
        resolve_rank_local_object_map,
    )

    root = tmp_path / "rank_shards" / "ws2"
    rank_dir = root / "rank_1"
    rank_dir.mkdir(parents=True)
    object_map = rank_dir / "_clip_object_urdf_map.json"
    object_map.write_text(
        '{"clips": {}, "rank_local_shard": {"rank": 1, "world_size": 2}}',
        encoding="utf-8",
    )
    base_motion = tmp_path / "motion"
    base_motion.mkdir()

    monkeypatch.setenv("HOLOSOMA_RANK_LOCAL_MOTION_ROOT", str(root))
    monkeypatch.setenv("HOLOSOMA_RANK_LOCAL_SHARDING_ENABLED", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")

    assert resolve_rank_local_motion_path(base_motion) == str(rank_dir)
    assert resolve_rank_local_object_map(base_motion / "_clip_object_urdf_map.json") == str(object_map)
    assert current_rank_local_shard_metadata() == {"rank": 1, "world_size": 2}


def test_rank_local_resolver_rejects_invalid_enable_flag(monkeypatch, tmp_path: Path) -> None:
    from holosoma.utils.rank_local_shards import current_rank_local_shard_dir

    root = tmp_path / "rank_shards"
    root.mkdir()
    monkeypatch.setenv("HOLOSOMA_RANK_LOCAL_MOTION_ROOT", str(root))
    monkeypatch.setenv("HOLOSOMA_RANK_LOCAL_SHARDING_ENABLED", "treu")

    with pytest.raises(RuntimeError, match="must be a boolean"):
        current_rank_local_shard_dir()


@pytest.mark.parametrize("world_size", ["not-an-int", "0", "-1"])
def test_rank_local_resolver_rejects_invalid_world_size(
    monkeypatch,
    tmp_path: Path,
    world_size: str,
) -> None:
    from holosoma.utils.rank_local_shards import current_rank_local_shard_dir

    root = tmp_path / "rank_shards"
    (root / "rank_0").mkdir(parents=True)
    monkeypatch.setenv("HOLOSOMA_RANK_LOCAL_MOTION_ROOT", str(root))
    monkeypatch.setenv("HOLOSOMA_RANK_LOCAL_SHARDING_ENABLED", "1")
    monkeypatch.setenv("WORLD_SIZE", world_size)
    monkeypatch.setenv("RANK", "0")

    with pytest.raises(RuntimeError, match="WORLD_SIZE"):
        current_rank_local_shard_dir()


def test_rank_local_metadata_rejects_stale_world_size(monkeypatch, tmp_path: Path) -> None:
    from holosoma.utils.rank_local_shards import current_rank_local_shard_metadata

    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=2, unique_urdfs=2)
    root = tmp_path / "rank_shards" / "ws8"
    module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=root,
        world_size=8,
    )

    monkeypatch.setenv("HOLOSOMA_RANK_LOCAL_MOTION_ROOT", str(root))
    monkeypatch.setenv("HOLOSOMA_RANK_LOCAL_SHARDING_ENABLED", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "4")

    try:
        current_rank_local_shard_metadata()
    except RuntimeError as exc:
        assert "world size does not match" in str(exc)
    else:
        raise AssertionError("stale rank-local shard metadata was accepted")


def test_rank_local_path_resolver_uses_global_rank_for_multinode(monkeypatch, tmp_path: Path) -> None:
    from holosoma.utils.rank_local_shards import resolve_rank_local_motion_path, resolve_rank_local_object_map

    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=2, unique_urdfs=2)
    root = tmp_path / "rank_shards" / "ws48"
    module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=root,
        world_size=48,
    )
    rank_dir = root / "rank_17"
    object_map = rank_dir / "_clip_object_urdf_map.json"

    monkeypatch.setenv("HOLOSOMA_RANK_LOCAL_MOTION_ROOT", str(root))
    monkeypatch.setenv("HOLOSOMA_RANK_LOCAL_SHARDING_ENABLED", "1")
    monkeypatch.setenv("WORLD_SIZE", "48")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "17")
    monkeypatch.setenv("LOCAL_RANK", "1")

    assert resolve_rank_local_motion_path(motion_dir) == str(rank_dir)
    assert resolve_rank_local_object_map(motion_dir / "_clip_object_urdf_map.json") == str(object_map)


@pytest.mark.parametrize("tamper_target", ["npz", "object_map"])
def test_scientific_rank_local_resolver_rejects_content_drift(
    monkeypatch,
    tmp_path: Path,
    tamper_target: str,
) -> None:
    from holosoma.utils.rank_local_shards import resolve_rank_local_motion_path

    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=2, unique_urdfs=2)
    root = tmp_path / "rank_shards" / "ws2"
    manifest = module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=root,
        world_size=2,
    )
    rank = 0
    rank_dir = root / f"rank_{rank}"
    if tamper_target == "npz":
        clip_id = manifest["shards"][rank]["clip_ids"][0]
        (motion_dir / f"{clip_id}.npz").write_bytes(b"changed after shard preparation")
        expected_error = "NPZ content records"
    else:
        local_map = rank_dir / "_clip_object_urdf_map.json"
        _make_owner_writable_for_same_uid_tamper(local_map)
        local_map.write_text(local_map.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        expected_error = "object-map content digest mismatch"

    monkeypatch.setenv("HOLOSOMA_RANK_LOCAL_MOTION_ROOT", str(root))
    monkeypatch.setenv("HOLOSOMA_RANK_LOCAL_SHARDING_ENABLED", "1")
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", "2")

    with pytest.raises(RuntimeError, match=expected_error):
        resolve_rank_local_motion_path(motion_dir)


def test_scientific_rank_local_resolver_rejects_legacy_manifest(monkeypatch, tmp_path: Path) -> None:
    from holosoma.utils.rank_local_shards import resolve_rank_local_motion_path

    root = tmp_path / "rank_shards" / "ws2"
    rank_dir = root / "rank_0"
    rank_dir.mkdir(parents=True)
    (rank_dir / "_clip_object_urdf_map.json").write_text(
        '{"clips": {}, "rank_local_shard": {"rank": 0, "world_size": 2}}',
        encoding="utf-8",
    )
    (root / "manifest.json").write_text('{"world_size": 2, "shards": []}', encoding="utf-8")
    base_motion = tmp_path / "motion"
    base_motion.mkdir()

    monkeypatch.setenv("HOLOSOMA_RANK_LOCAL_MOTION_ROOT", str(root))
    monkeypatch.setenv("HOLOSOMA_RANK_LOCAL_SHARDING_ENABLED", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")

    with pytest.raises(RuntimeError, match="rejects legacy manifests"):
        resolve_rank_local_motion_path(base_motion)


@pytest.mark.parametrize(
    ("field", "bad_value", "expected_error"),
    [
        ("inverse_cover_mass", float("nan"), "finite positive"),
        ("inverse_cover_mass", 0.0, "finite positive"),
        ("inverse_cover_mass", 9.0, "global clip-cover formula"),
        ("distributed_loss_weight", float("inf"), "finite positive"),
        ("distributed_loss_weight", -1.0, "finite positive"),
        ("distributed_loss_weight", 9.0, "world/global cover formula"),
    ],
)
def test_scientific_rank_local_resolver_recomputes_loss_weights(
    monkeypatch,
    tmp_path: Path,
    field: str,
    bad_value: float,
    expected_error: str,
) -> None:
    from holosoma.utils.rank_local_shards import resolve_rank_local_motion_path

    module = _load_prepare_module()
    motion_dir, object_map = _write_motion_bank(tmp_path, clip_count=2, unique_urdfs=2)
    root = tmp_path / "rank_shards" / "ws4"
    module.prepare_rank_shards(
        motion_dir=motion_dir,
        object_map=object_map,
        output_root=root,
        world_size=4,
    )
    rank = 0
    local_map = root / f"rank_{rank}" / "_clip_object_urdf_map.json"
    payload = json.loads(local_map.read_text(encoding="utf-8"))
    payload["rank_local_shard"][field] = bad_value
    _make_owner_writable_for_same_uid_tamper(local_map)
    local_map.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shards"][rank]["object_map_sha256"] = hashlib.sha256(local_map.read_bytes()).hexdigest()
    _make_owner_writable_for_same_uid_tamper(manifest_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    monkeypatch.setenv("HOLOSOMA_RANK_LOCAL_MOTION_ROOT", str(root))
    monkeypatch.setenv("HOLOSOMA_RANK_LOCAL_SHARDING_ENABLED", "1")
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", "4")
    with pytest.raises(RuntimeError, match=expected_error):
        resolve_rank_local_motion_path(motion_dir)


def test_object_env_assignment_preserves_per_clip_frequency() -> None:
    from holosoma.utils.rank_local_shards import build_clip_weighted_object_assignment

    assignment = build_clip_weighted_object_assignment(
        ["a.urdf", "b.urdf"],
        ["a.urdf", "a.urdf", "b.urdf"],
        num_envs=6,
    )

    assert assignment == ["a.urdf", "a.urdf", "b.urdf", "a.urdf", "a.urdf", "b.urdf"]
