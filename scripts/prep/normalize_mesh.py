#!/usr/bin/env python3
"""Normalize STEP-exported OBJ meshes for the pipeline.

The FreeCAD exports for the new CAD objects are in millimeters and may have
their vertex coordinates offset from the object origin. This script writes
meter-scale `source.obj` files with the bbox bottom center at the origin.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUTS = {
    "curved_structure": "Unnamed-CURVED_STRUCTURE.obj",
    "cylinder_sample": "Unnamed-CYLINDER_SAMPLE.obj",
    "square_structure": "Unnamed-SQUARE_STRUCTURE.obj",
}


def load_mesh(path: Path) -> trimesh.Trimesh:
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
    return np.array(
        [
            (bounds[0, 0] + bounds[1, 0]) / 2.0,
            (bounds[0, 1] + bounds[1, 1]) / 2.0,
            bounds[0, 2],
        ],
        dtype=float,
    )


def normalize_mesh(mesh: trimesh.Trimesh, scale: float) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.apply_scale(scale)
    mesh.apply_translation(-bottom_center(mesh.bounds))
    return mesh


def process_object(
    object_name: str,
    input_name: str,
    output_name: str,
    scale: float,
    dry_run: bool,
) -> None:
    mesh_dir = PROJECT_ROOT / "data" / object_name / "mesh"
    input_path = mesh_dir / input_name
    output_path = mesh_dir / output_name

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    raw = load_mesh(input_path)
    normalized = normalize_mesh(raw, scale)

    print(f"[{object_name}]")
    print(f"  input:       {input_path.relative_to(PROJECT_ROOT)}")
    print(f"  raw extents: {raw.extents}")
    print(f"  out extents: {normalized.extents}")
    print(f"  out bounds:  {normalized.bounds}")

    if dry_run:
        print("  dry-run:     not written")
        return

    normalized.export(output_path)
    print(f"  written:     {output_path.relative_to(PROJECT_ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scale STEP-exported OBJ files from mm to m, then move their bbox "
            "bottom center to the origin and write source.obj."
        )
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.001,
        help="Scale applied before recentering. Default: 0.001 for mm to m.",
    )
    parser.add_argument(
        "--input-name",
        help=(
            "Input OBJ filename used for every object. "
            "Example: rotated.obj after fixing orientation in Blender."
        ),
    )
    parser.add_argument(
        "--output-name",
        default="source.obj",
        help="Output OBJ filename in each mesh directory. Default: source.obj.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print normalized dimensions without writing source.obj.",
    )
    parser.add_argument(
        "objects",
        nargs="*",
        choices=sorted(DEFAULT_INPUTS),
        help="Objects to process. Default: all three new CAD objects.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    objects = args.objects or sorted(DEFAULT_INPUTS)

    for object_name in objects:
        input_name = args.input_name or DEFAULT_INPUTS[object_name]
        process_object(
            object_name=object_name,
            input_name=input_name,
            output_name=args.output_name,
            scale=args.scale,
            dry_run=args.dry_run,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
