#!/usr/bin/env python3
"""Prepare object meshes for the three application pipelines.

The ``normalize`` command converts an arbitrary exported mesh to metres and
moves its bounding-box bottom centre to the object origin.  The ``reorient``
command bakes a corrective rotation into an existing mesh and reapplies the
same origin convention.

Examples:
    uv run scripts/setup/prepare_object_mesh.py normalize \
        --object curved_structure --input raw_export.obj
    uv run scripts/setup/prepare_object_mesh.py reorient \
        --object curved_structure --euler -90 0 0
    uv run scripts/setup/prepare_object_mesh.py reorient \
        --object glass --world-target-quat 0.5 0.5 0.5 0.5
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh
from trimesh.transformations import euler_matrix, quaternion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from common import config  # noqa: E402
from common.math_utils import quaternion_to_rotation_matrix  # noqa: E402


def load_mesh(path: Path) -> trimesh.Trimesh:
    """Load a non-empty mesh, concatenating scene geometry when necessary."""
    loaded = trimesh.load(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        geometries = list(loaded.geometry.values())
        if not geometries:
            raise ValueError(f"No geometry found in {path}")
        loaded = trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type from {path}: {type(loaded)!r}")
    if loaded.vertices.size == 0 or loaded.faces.size == 0:
        raise ValueError(f"Mesh has no vertices or faces: {path}")
    return loaded


def bottom_center(bounds: np.ndarray) -> np.ndarray:
    return np.array([
        (bounds[0, 0] + bounds[1, 0]) / 2.0,
        (bounds[0, 1] + bounds[1, 1]) / 2.0,
        bounds[0, 2],
    ], dtype=float)


def mesh_path(object_name: str, value: Path) -> Path:
    if value.is_absolute():
        return value
    return PROJECT_ROOT / "data" / object_name / "mesh" / value


def recenter(mesh: trimesh.Trimesh) -> None:
    mesh.apply_translation(-bottom_center(mesh.bounds))


def normalize(args: argparse.Namespace) -> trimesh.Trimesh:
    mesh = load_mesh(mesh_path(args.object, args.input)).copy()
    mesh.apply_scale(args.scale)
    recenter(mesh)
    return mesh


def rotation_matrix(args: argparse.Namespace) -> np.ndarray:
    if args.euler is not None:
        ax, ay, az = (np.deg2rad(v) for v in args.euler)
        return euler_matrix(ax, ay, az, "sxyz")
    if args.quat is not None:
        return quaternion_matrix(np.asarray(args.quat, dtype=float))

    config.apply_object_placement(args.object)
    configured = quaternion_to_rotation_matrix(config.TARGET_OBJECT["rotation"])
    target = quaternion_to_rotation_matrix(
        np.asarray(args.world_target_quat, dtype=float),
    )
    transform = np.eye(4)
    transform[:3, :3] = configured.T @ target
    return transform


def reorient(args: argparse.Namespace) -> trimesh.Trimesh:
    mesh = load_mesh(mesh_path(args.object, args.input)).copy()
    mesh.apply_transform(rotation_matrix(args))
    recenter(mesh)
    return mesh


def write_result(args: argparse.Namespace, mesh: trimesh.Trimesh) -> None:
    input_path = mesh_path(args.object, args.input)
    output_path = mesh_path(args.object, args.output)
    print(f"[{args.object}] {args.command}")
    print(f"  input:       {input_path}")
    print(f"  out extents: {mesh.extents}")
    print(f"  out bounds:  {mesh.bounds}")

    if args.dry_run:
        print("  dry-run:     not written")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "reorient" and not args.no_backup and output_path.exists():
        backup = output_path.with_suffix(output_path.suffix + ".bak")
        shutil.copy2(output_path, backup)
        print(f"  backup:      {backup}")
    mesh.export(output_path)
    print(f"  written:     {output_path}")
    print(
        "  next: uv run scripts/setup/build_object_usd.py "
        f"--object {args.object} --force"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    normalize_parser = commands.add_parser(
        "normalize",
        help="scale an exported mesh and move its bbox bottom centre to the origin",
    )
    normalize_parser.add_argument("--object", required=True)
    normalize_parser.add_argument("--input", type=Path, required=True)
    normalize_parser.add_argument("--output", type=Path, default=Path("source.obj"))
    normalize_parser.add_argument("--scale", type=float, default=0.001)
    normalize_parser.add_argument("--dry-run", action="store_true")
    normalize_parser.set_defaults(handler=normalize)

    reorient_parser = commands.add_parser(
        "reorient",
        help="bake a corrective rotation and restore the object origin convention",
    )
    reorient_parser.add_argument("--object", required=True)
    reorient_parser.add_argument("--input", type=Path, default=Path("source.obj"))
    reorient_parser.add_argument("--output", type=Path, default=Path("source.obj"))
    rotation = reorient_parser.add_mutually_exclusive_group(required=True)
    rotation.add_argument(
        "--world-target-quat", type=float, nargs=4,
        metavar=("W", "X", "Y", "Z"),
    )
    rotation.add_argument("--euler", type=float, nargs=3, metavar=("X", "Y", "Z"))
    rotation.add_argument("--quat", type=float, nargs=4, metavar=("W", "X", "Y", "Z"))
    reorient_parser.add_argument("--no-backup", action="store_true")
    reorient_parser.add_argument("--dry-run", action="store_true")
    reorient_parser.set_defaults(handler=reorient)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "normalize" and args.scale <= 0.0:
        parser.error("--scale must be positive")
    write_result(args, args.handler(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
