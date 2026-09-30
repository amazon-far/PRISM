#!/usr/bin/env python3
"""Build an object bank whose URDF inertials come from object meshes.

The source bank is left untouched. The output bank symlinks/copies motion files
and object assets, rewrites clip->URDF entries to point at generated URDFs, and
emits a report with the mesh-derived COM and inertia values.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


DEFAULT_MASS_PRIORS: dict[str, dict[str, float]] = {
    # These are exact nominal URDF masses. Runtime density uncertainty is
    # represented separately by the coupled mass/inertia scale randomizer.
    "barrel": {"default": 1.5, "min": 1.5, "max": 1.5},
    "bin": {"default": 1.0, "min": 1.0, "max": 1.0},
    "box": {"default": 1.0, "min": 1.0, "max": 1.0},
    "ball": {"default": 0.5, "min": 0.5, "max": 0.5},
    "other": {"default": 1.0, "min": 0.1, "max": 5.0},
}


def _parse_vec3(raw: str | None, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    if not raw:
        return np.asarray(default, dtype=np.float64)
    parts = [float(item) for item in raw.split()]
    parts += [0.0] * max(0, 3 - len(parts))
    return np.asarray(parts[:3], dtype=np.float64)


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def _origin_transform(origin: ET.Element | None) -> np.ndarray:
    xyz = _parse_vec3(origin.get("xyz") if origin is not None else None)
    rpy = _parse_vec3(origin.get("rpy") if origin is not None else None)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rpy_to_matrix(rpy)
    transform[:3, 3] = xyz
    return transform


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_.")
    return cleaned or "object"


def _resolve_path(raw: str, base_dir: Path) -> Path:
    raw = str(raw).strip()
    if raw.startswith("file://"):
        raw = raw[len("file://") :]
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    if raw.startswith("package://"):
        raise ValueError(f"package:// mesh paths are not supported by this bank builder: {raw}")
    return (base_dir / path).resolve()


def _relpath(path: Path, base_dir: Path) -> str:
    return os.path.relpath(path.resolve(), base_dir.resolve())


def _load_clip_map(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
        return {key: value for key, value in payload.items() if key != "clips"}, payload["clips"]
    if isinstance(payload, dict):
        return {}, payload
    raise ValueError(f"Invalid object map: {path}")


def _category_for(clip_id: str, entry: Any) -> str:
    parts = [clip_id]
    if isinstance(entry, dict):
        for key in ("object_name", "object_urdf_path", "object_mesh_path", "object_category", "category", "object_type"):
            value = str(entry.get(key, "")).strip()
            if not value:
                continue
            if key.endswith("_path"):
                path = Path(value)
                parts.extend([path.name, path.stem])
            else:
                parts.append(value)
    else:
        path = Path(str(entry).strip())
        parts.extend([path.name, path.stem])
    raw = " ".join(parts).lower().replace("-", "_")
    if "barrel" in raw:
        return "barrel"
    if "bin" in raw or "trash" in raw or "basket" in raw:
        return "bin"
    if "ball" in raw or "sphere" in raw:
        return "ball"
    if "box" in raw or "cube" in raw or "largebox" in raw:
        return "box"
    return "other"


def _load_mass_priors(path: Path | None) -> dict[str, dict[str, float]]:
    priors = json.loads(json.dumps(DEFAULT_MASS_PRIORS))
    if path is None:
        return priors
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Mass priors must be a JSON object: {path}")
    for category, values in raw.items():
        if not isinstance(values, dict):
            raise ValueError(f"Mass prior for {category!r} must be an object")
        prior = priors.setdefault(str(category), {})
        for key in ("default", "min", "max"):
            if key in values:
                prior[key] = float(values[key])
    return priors


def _base_mass_for_category(category: str, priors: dict[str, dict[str, float]]) -> float:
    prior = priors.get(category, priors["other"])
    mass = float(prior.get("default", priors["other"]["default"]))
    min_mass = float(prior.get("min", 0.001))
    max_mass = float(prior.get("max", max(min_mass, mass)))
    return float(np.clip(mass, min_mass, max_mass))


def _source_urdf_mass(src_urdf: Path) -> float:
    root = ET.parse(src_urdf).getroot()
    current = _read_current_inertial(root)
    if "urdf_mass_kg" not in current:
        raise ValueError(f"Source URDF has no inertial mass to preserve: {src_urdf}")
    mass = float(current["urdf_mass_kg"])
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"Source URDF mass must be positive and finite, got {mass!r}: {src_urdf}")
    return mass


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No mesh geometry in {path}")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh load result {type(loaded).__name__} for {path}")
    return loaded


def _mesh_entries(root: ET.Element, src_urdf: Path, tag_name: str) -> list[tuple[trimesh.Trimesh, Path]]:
    entries: list[tuple[trimesh.Trimesh, Path]] = []
    for geom_parent in root.findall(f".//{tag_name}"):
        mesh_node = geom_parent.find("./geometry/mesh")
        if mesh_node is None:
            continue
        filename = str(mesh_node.get("filename", "")).strip()
        if not filename:
            continue
        mesh_path = _resolve_path(filename, src_urdf.parent)
        mesh = _load_mesh(mesh_path)
        scale = _parse_vec3(mesh_node.get("scale"), default=(1.0, 1.0, 1.0))
        scale_transform = np.eye(4, dtype=np.float64)
        scale_transform[0, 0] = scale[0]
        scale_transform[1, 1] = scale[1]
        scale_transform[2, 2] = scale[2]
        mesh.apply_transform(_origin_transform(geom_parent.find("./origin")) @ scale_transform)
        entries.append((mesh, mesh_path))
    return entries


def _combined_geometry(root: ET.Element, src_urdf: Path) -> tuple[trimesh.Trimesh, list[Path], str]:
    # Inertial properties follow the object's rendered/perceived geometry.
    # Collision geometry may be a simplified proxy and must not silently alter
    # the nominal COM or inertia contract.
    entries = _mesh_entries(root, src_urdf, "visual")
    source = "visual"
    if not entries:
        raise ValueError(f"No visual mesh geometry found for inertia computation: {src_urdf}")
    mesh = entries[0][0] if len(entries) == 1 else trimesh.util.concatenate([item[0] for item in entries])
    if mesh.volume < 0:
        mesh.invert()
    return mesh, [item[1] for item in entries], source


def _mesh_mass_properties(mesh: trimesh.Trimesh, fallback: str) -> tuple[np.ndarray, np.ndarray, float, str, bool]:
    work = mesh
    kind = "volume"
    if not work.is_watertight or abs(float(work.volume)) < 1.0e-12:
        if fallback == "error":
            raise ValueError("Mesh is not watertight or has near-zero volume")
        work = work.convex_hull
        kind = "convex_hull_fallback"
        if work.volume < 0:
            work.invert()
    volume = abs(float(work.volume))
    if volume < 1.0e-12:
        raise ValueError("Mesh volume is near zero after fallback")
    work.density = 1.0 / volume
    return (
        np.asarray(work.center_mass, dtype=np.float64),
        np.asarray(work.moment_inertia, dtype=np.float64),
        volume,
        kind,
        bool(mesh.is_watertight),
    )


def _ensure_child(parent: ET.Element, tag: str) -> ET.Element:
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    return child


def _set_float_attr(node: ET.Element, key: str, value: float) -> None:
    node.set(key, f"{float(value):.9g}")


def _rewrite_mesh_paths(root: ET.Element, *, src_urdf: Path, dst_urdf: Path) -> None:
    for mesh_node in root.findall(".//mesh"):
        filename = str(mesh_node.get("filename", "")).strip()
        if not filename or filename.startswith(("package://", "http://", "https://")):
            continue
        resolved = _resolve_path(filename, src_urdf.parent)
        mesh_node.set("filename", _relpath(resolved, dst_urdf.parent))


def _read_current_inertial(root: ET.Element) -> dict[str, float]:
    inertial = root.find(".//link/inertial")
    if inertial is None:
        return {}
    mass = inertial.find("mass")
    origin = inertial.find("origin")
    inertia = inertial.find("inertia")
    out: dict[str, float] = {}
    if mass is not None and mass.get("value") is not None:
        out["urdf_mass_kg"] = float(mass.get("value", "0"))
    if origin is not None:
        xyz = _parse_vec3(origin.get("xyz"))
        out["urdf_com_x_m"], out["urdf_com_y_m"], out["urdf_com_z_m"] = xyz.tolist()
    if inertia is not None:
        for key in ("ixx", "iyy", "izz", "ixy", "iyz", "ixz"):
            out[f"urdf_{key}"] = float(inertia.get(key, "0"))
    return out


def _write_mesh_physics_urdf(
    *,
    src_urdf: Path,
    dst_urdf: Path,
    category: str,
    base_mass: float,
    fallback: str,
) -> dict[str, Any]:
    tree = ET.parse(src_urdf)
    root = tree.getroot()
    links = root.findall("link")
    if len(links) != 1:
        raise ValueError(f"Expected a single-link object URDF, got {len(links)} links: {src_urdf}")

    current = _read_current_inertial(root)
    mesh, mesh_paths, geometry_source = _combined_geometry(root, src_urdf)
    com, inertia_per_kg, volume, com_kind, watertight = _mesh_mass_properties(mesh, fallback=fallback)
    inertia = inertia_per_kg * float(base_mass)
    eigvals = np.linalg.eigvalsh(inertia)
    if np.any(eigvals <= 0.0):
        raise ValueError(f"Mesh inertia is not positive definite for {src_urdf}: eigenvalues={eigvals}")

    inertial = _ensure_child(links[0], "inertial")
    mass_node = _ensure_child(inertial, "mass")
    origin_node = _ensure_child(inertial, "origin")
    inertia_node = _ensure_child(inertial, "inertia")
    _set_float_attr(mass_node, "value", base_mass)
    origin_node.set("xyz", " ".join(f"{float(v):.9g}" for v in com))
    origin_node.set("rpy", "0 0 0")
    _set_float_attr(inertia_node, "ixx", inertia[0, 0])
    _set_float_attr(inertia_node, "ixy", inertia[0, 1])
    _set_float_attr(inertia_node, "ixz", inertia[0, 2])
    _set_float_attr(inertia_node, "iyy", inertia[1, 1])
    _set_float_attr(inertia_node, "iyz", inertia[1, 2])
    _set_float_attr(inertia_node, "izz", inertia[2, 2])

    _rewrite_mesh_paths(root, src_urdf=src_urdf, dst_urdf=dst_urdf)
    dst_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst_urdf, encoding="utf-8", xml_declaration=True)

    report = {
        "source_urdf": str(src_urdf),
        "generated_urdf": str(dst_urdf),
        "category": category,
        "base_mass_kg": float(base_mass),
        "geometry_source": geometry_source,
        "com_kind": com_kind,
        "source_mesh_watertight": watertight,
        "mesh_volume_m3": volume,
        "mesh_com_x_m": float(com[0]),
        "mesh_com_y_m": float(com[1]),
        "mesh_com_z_m": float(com[2]),
        "mesh_ixx": float(inertia[0, 0]),
        "mesh_iyy": float(inertia[1, 1]),
        "mesh_izz": float(inertia[2, 2]),
        "mesh_ixy": float(inertia[0, 1]),
        "mesh_iyz": float(inertia[1, 2]),
        "mesh_ixz": float(inertia[0, 2]),
        "mesh_inertia_eig_min": float(eigvals[0]),
        "mesh_inertia_eig_mid": float(eigvals[1]),
        "mesh_inertia_eig_max": float(eigvals[2]),
        "mesh_paths": ";".join(str(path) for path in mesh_paths),
    }
    report.update(current)
    return report


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        return
    if mode == "copy":
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        return
    target = os.path.relpath(src.resolve(), dst.parent.resolve())
    dst.symlink_to(target, target_is_directory=src.is_dir())


def _materialize_bank_assets(input_bank: Path, output_bank: Path, mode: str) -> None:
    output_bank.mkdir(parents=True, exist_ok=True)
    skip_names = {
        ".generated_by_solid80_clean_nfs_packager",
        "_clip_object_urdf_map.json",
        "cp_corl_local_manifest.json",
        "clip_object_urdf_map.json",
        "_mesh_physics_manifest.json",
        "_mesh_physics_report.csv",
        "_mesh_physics_urdfs",
        "nfs_package_manifest.json",
    }
    # This manifest is a provenance snapshot consumed by
    # prepare_immutable_solid_bank.py.  It must remain a regular file even
    # when the bulk motion/object payload is materialized with symlinks;
    # otherwise the immutable publication gate correctly rejects the bank.
    copy_names = {"realmesh_rollout_manifest.json"}
    skip_prefixes = ("_scientific_", "_single_slot_")
    for child in sorted(input_bank.iterdir()):
        if child.name in skip_names or child.name.startswith(skip_prefixes):
            continue
        _link_or_copy(
            child,
            output_bank / child.name,
            "copy" if child.name in copy_names else mode,
        )


def _resolve_object_urdf(raw: str, object_map: Path) -> Path:
    path = Path(str(raw).strip()).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (object_map.parent / path).resolve()


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_bank(args: argparse.Namespace) -> None:
    input_bank = Path(args.input_bank).expanduser().resolve()
    object_map = Path(args.object_map).expanduser().resolve() if args.object_map else input_bank / "_clip_object_urdf_map.json"
    output_bank = Path(args.output_bank).expanduser().resolve()
    if not input_bank.is_dir():
        raise FileNotFoundError(f"Input bank does not exist: {input_bank}")
    if not object_map.is_file():
        raise FileNotFoundError(f"Object map does not exist: {object_map}")
    if output_bank.exists() and any(output_bank.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output bank is not empty; pass --overwrite: {output_bank}")
    if output_bank.exists() and args.overwrite:
        shutil.rmtree(output_bank)

    if args.mass_mode == "source_urdf" and args.mass_priors_json:
        raise ValueError("--mass-priors-json cannot be combined with --mass-mode=source_urdf")
    priors = (
        {}
        if args.mass_mode == "source_urdf"
        else _load_mass_priors(Path(args.mass_priors_json).expanduser().resolve() if args.mass_priors_json else None)
    )
    _materialize_bank_assets(input_bank, output_bank, args.asset_mode)

    metadata, clips = _load_clip_map(object_map)
    generated_dir = output_bank / "_mesh_physics_urdfs"
    generated_dir.mkdir(parents=True, exist_ok=True)

    generated_by_src: dict[Path, tuple[Path, dict[str, Any]]] = {}
    used_names: set[str] = set()
    updated_clips: dict[str, Any] = {}
    report_rows: list[dict[str, Any]] = []

    for clip_id, raw_entry in sorted(clips.items()):
        entry = dict(raw_entry) if isinstance(raw_entry, dict) else {"object_urdf_path": str(raw_entry)}
        raw_urdf = str(entry.get("object_urdf_path", "")).strip()
        if not raw_urdf:
            raise ValueError(f"Clip {clip_id!r} has no object_urdf_path")
        src_urdf = _resolve_object_urdf(raw_urdf, object_map)
        if not src_urdf.is_file():
            raise FileNotFoundError(f"Clip {clip_id!r} object URDF does not exist: {src_urdf}")

        cached = generated_by_src.get(src_urdf)
        category = _category_for(clip_id, entry)
        if args.mass_mode == "source_urdf":
            base_mass = _source_urdf_mass(src_urdf)
        else:
            base_mass = _base_mass_for_category(category, priors)
        if cached is None:
            stem_source = str(entry.get("object_name", "") or src_urdf.stem)
            stem = _safe_stem(stem_source)
            candidate = stem
            suffix = 1
            while candidate in used_names:
                suffix += 1
                candidate = f"{stem}_{suffix}"
            used_names.add(candidate)
            dst_urdf = generated_dir / f"{candidate}.urdf"
            report = _write_mesh_physics_urdf(
                src_urdf=src_urdf,
                dst_urdf=dst_urdf,
                category=category,
                base_mass=base_mass,
                fallback=args.non_watertight_fallback,
            )
            generated_by_src[src_urdf] = (dst_urdf, report)
            report_rows.append(report)
        else:
            dst_urdf, report = cached
            if abs(float(report["base_mass_kg"]) - base_mass) > 1.0e-9:
                raise ValueError(
                    f"Source URDF {src_urdf} appears in multiple categories with different base masses: "
                    f"{report['base_mass_kg']} vs {base_mass}"
                )

        entry["object_urdf_path"] = _relpath(dst_urdf, output_bank)
        entry["mesh_physics_source_urdf_path"] = str(src_urdf)
        entry["mesh_physics_category"] = category
        entry["mesh_physics_base_mass_kg"] = base_mass
        entry["mesh_physics_mass_mode"] = args.mass_mode
        updated_clips[clip_id] = entry

    output_payload: dict[str, Any] = dict(metadata)
    output_payload["clips"] = updated_clips
    output_payload["mesh_physics"] = {
        "source_bank": str(input_bank),
        "source_object_map": str(object_map),
        "mass_mode": args.mass_mode,
        "mass_priors": priors,
        "non_watertight_fallback": args.non_watertight_fallback,
        "unique_urdf_count": len(generated_by_src),
        "clip_count": len(updated_clips),
    }
    (output_bank / "_clip_object_urdf_map.json").write_text(json.dumps(output_payload, indent=2) + "\n", encoding="utf-8")
    (output_bank / "_mesh_physics_manifest.json").write_text(
        json.dumps(output_payload["mesh_physics"], indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(output_bank / "_mesh_physics_report.csv", report_rows)

    print(f"[INFO] source_bank={input_bank}")
    print(f"[INFO] output_bank={output_bank}")
    print(f"[INFO] clips={len(updated_clips)} unique_urdfs={len(generated_by_src)}")
    print(f"[INFO] object_map={output_bank / '_clip_object_urdf_map.json'}")
    print(f"[INFO] report={output_bank / '_mesh_physics_report.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-bank", required=True, help="Source motion/object bank directory.")
    parser.add_argument("--object-map", default="", help="Source clip-object map. Defaults to INPUT/_clip_object_urdf_map.json.")
    parser.add_argument("--output-bank", required=True, help="Output bank directory.")
    parser.add_argument(
        "--mass-mode",
        choices=("category_priors", "source_urdf"),
        default="category_priors",
        help=(
            "Choose category priors or preserve each source URDF mass while "
            "recomputing only its mesh-derived COM and inertia."
        ),
    )
    parser.add_argument("--mass-priors-json", default="", help="Optional category mass prior JSON override.")
    parser.add_argument(
        "--asset-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="How to materialize source bank assets in the output bank.",
    )
    parser.add_argument(
        "--non-watertight-fallback",
        choices=("convex_hull", "error"),
        default="convex_hull",
        help="Fallback for non-watertight meshes.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing output bank before writing.")
    return parser.parse_args()


def main() -> None:
    try:
        build_bank(parse_args())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
