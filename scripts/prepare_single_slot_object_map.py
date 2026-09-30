#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from prepare_as_rank_shards import (
    MOTION_TRANSITION_SOURCE_KEY,
    resolve_motion_transition_source,
)


_ENTRY_SINGLE_MESH_KEYS = (
    "object_mesh_path",
    "object_visual_mesh_path",
    "object_collision_mesh_path",
)
_ENTRY_MULTI_MESH_KEYS = (
    "object_mesh_paths",
    "object_visual_mesh_paths",
    "object_collision_mesh_paths",
)


def _resolve_path(raw: str, base_dir: Path) -> Path:
    path = Path(str(raw).strip()).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_.")
    return cleaned or "object"


def _relpath(path: Path, base_dir: Path) -> str:
    return os.path.relpath(path.resolve(), base_dir.resolve())


def _load_clip_map(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
        return {key: value for key, value in payload.items() if key != "clips"}, payload["clips"]
    if isinstance(payload, dict):
        return {}, payload
    raise ValueError(f"Invalid object map: {path}")


def _canonicalize_mesh_paths(root: ET.Element, *, src_urdf: Path, dst_urdf: Path) -> None:
    for mesh in root.findall(".//mesh"):
        filename = str(mesh.get("filename", "")).strip()
        if not filename:
            continue
        if filename.startswith(("package://", "http://", "https://", "file://")):
            continue
        mesh_path = _resolve_path(filename, src_urdf.parent)
        mesh.set("filename", _relpath(mesh_path, dst_urdf.parent))


def _canonicalize_one_link_urdf(src_urdf: Path, dst_urdf: Path) -> None:
    tree = ET.parse(src_urdf)
    root = tree.getroot()
    links = root.findall("link")
    if len(links) != 1:
        raise ValueError(
            f"single-slot canonicalization requires exactly one URDF link; "
            f"{src_urdf} has {len(links)}"
        )

    old_link_name = str(links[0].get("name", "")).strip()
    if not old_link_name:
        raise ValueError(f"single-slot canonicalization requires a named link: {src_urdf}")

    root.set("name", _safe_stem(src_urdf.stem))
    links[0].set("name", "baseLink")

    for joint in root.findall("joint"):
        for tag_name in ("parent", "child"):
            node = joint.find(tag_name)
            if node is not None and node.get("link") == old_link_name:
                node.set("link", "baseLink")

    for idx, visual in enumerate(links[0].findall("visual")):
        visual.set("name", f"visual_{idx:03d}")
    for idx, collision in enumerate(links[0].findall("collision")):
        collision.set("name", f"collision_{idx:03d}")

    _canonicalize_mesh_paths(root, src_urdf=src_urdf, dst_urdf=dst_urdf)

    dst_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst_urdf, encoding="utf-8", xml_declaration=True)


def prepare_single_slot_map(
    *,
    motion_dir: Path,
    object_map: Path,
    output_map: Path,
    active_clip_ids: set[str] | None = None,
) -> tuple[int, int]:
    metadata, clips = _load_clip_map(object_map)
    if not clips:
        raise ValueError(f"Object map has no clips: {object_map}")
    if active_clip_ids is not None:
        normalized_active = {str(clip_id) for clip_id in active_clip_ids}
        missing = sorted(normalized_active.difference(clips))
        if missing:
            raise ValueError(
                f"Object map is missing {len(missing)} active clip(s): {', '.join(missing[:10])}"
            )
        clips = {clip_id: clips[clip_id] for clip_id in sorted(normalized_active)}

    transition_source = resolve_motion_transition_source(
        metadata,
        active_clip_count=len(clips),
        infer_if_missing=True,
        role=f"single-slot source map {object_map} {MOTION_TRANSITION_SOURCE_KEY}",
    )
    assert transition_source is not None
    metadata = dict(metadata)
    metadata[MOTION_TRANSITION_SOURCE_KEY] = transition_source

    output_dir = output_map.parent / "_single_slot_urdfs"
    output_dir.mkdir(parents=True, exist_ok=True)

    canonical_by_src: dict[Path, str] = {}
    used_names: set[str] = set()
    converted_count = 0

    updated_clips: dict[str, Any] = {}
    for clip_id, raw_entry in sorted(clips.items()):
        entry = dict(raw_entry) if isinstance(raw_entry, dict) else {"object_urdf_path": str(raw_entry)}
        raw_urdf = str(entry.get("object_urdf_path", "")).strip()
        if not raw_urdf:
            raise ValueError(f"Clip '{clip_id}' has an empty object_urdf_path")

        src_urdf = _resolve_path(raw_urdf, object_map.parent)
        if not src_urdf.is_file():
            raise FileNotFoundError(f"Clip '{clip_id}' object URDF is missing: {src_urdf}")

        canonical_rel = canonical_by_src.get(src_urdf)
        if canonical_rel is None:
            stem = _safe_stem(src_urdf.stem)
            candidate = stem
            suffix = 1
            while candidate in used_names:
                suffix += 1
                candidate = f"{stem}_{suffix}"
            used_names.add(candidate)

            dst_urdf = output_dir / f"{candidate}.urdf"
            _canonicalize_one_link_urdf(src_urdf, dst_urdf)
            canonical_rel = _relpath(dst_urdf, motion_dir)
            canonical_by_src[src_urdf] = canonical_rel
            converted_count += 1

        entry["object_urdf_path"] = canonical_rel
        for key in _ENTRY_SINGLE_MESH_KEYS:
            raw_mesh = str(entry.get(key, "")).strip()
            if raw_mesh:
                mesh_path = _resolve_path(raw_mesh, object_map.parent)
                entry[key] = _relpath(mesh_path, motion_dir)
        for key in _ENTRY_MULTI_MESH_KEYS:
            raw_values = entry.get(key)
            if raw_values is None:
                continue
            if isinstance(raw_values, str):
                raw_values = [raw_values]
            if not isinstance(raw_values, (list, tuple)) or not raw_values:
                raise ValueError(f"Clip '{clip_id}' has invalid {key}; expected a non-empty path list")
            rewritten_meshes: list[str] = []
            for index, raw_mesh in enumerate(raw_values):
                normalized_mesh = str(raw_mesh).strip()
                if not normalized_mesh:
                    raise ValueError(f"Clip '{clip_id}' has an empty {key}[{index}] path")
                rewritten_meshes.append(
                    _relpath(_resolve_path(normalized_mesh, object_map.parent), motion_dir)
                )
            entry[key] = rewritten_meshes
        updated_clips[str(clip_id)] = entry

    output_payload = dict(metadata)
    output_payload["clips"] = updated_clips
    output_payload["single_slot_canonicalization"] = {
        "source_object_map": _relpath(object_map, motion_dir),
        "canonical_link_name": "baseLink",
        "unique_source_urdf_count": len(canonical_by_src),
        "notes": "Generated for IsaacLab single-slot multi-URDF spawning; mesh paths remain file-backed.",
    }
    output_map.write_text(json.dumps(output_payload, indent=2, sort_keys=True), encoding="utf-8")
    return len(updated_clips), converted_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-dir", required=True, type=Path)
    parser.add_argument("--object-map", required=True, type=Path)
    parser.add_argument("--output-map", required=True, type=Path)
    args = parser.parse_args()

    motion_dir = args.motion_dir.expanduser().resolve()
    object_map = args.object_map.expanduser().resolve()
    output_map = args.output_map.expanduser()
    if not output_map.is_absolute():
        output_map = (motion_dir / output_map).resolve() if len(output_map.parts) == 1 else output_map.resolve()
    else:
        output_map = output_map.resolve()

    try:
            clip_count, urdf_count = prepare_single_slot_map(
                motion_dir=motion_dir,
                object_map=object_map,
                output_map=output_map,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to prepare single-slot object map: {exc}", file=sys.stderr)
        return 2

    print(f"[INFO] Prepared single-slot object map: {output_map}", file=sys.stderr)
    print(f"[INFO]   clips={clip_count} canonical_urdfs={urdf_count}", file=sys.stderr)
    print(str(output_map))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
