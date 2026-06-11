#!/usr/bin/env python3
"""Bake a corrective rotation into a target object's source.obj.

STEP→OBJ exports often carry a weird baked orientation. normalize_mesh.py only
scales + recenters (no rotation), so the tilt survives all the way into Isaac.
This bakes a rotation into the mesh vertices and re-applies the project's
normalize convention (bbox bottom-center → origin), so viser, Isaac, and the
cuRobo planner all see the corrected orientation consistently.

Pick the rotation one of three ways:

  --world-target-quat W X Y Z   The world Orient you dialed in with the Isaac
                                viewport E-gizmo (read it via the Pipeline UI
                                "Log Object Pose" button, or Stage > Property).
                                Internally baked as config_rot⁻¹ ∘ target so that
                                — with config.TARGET_OBJECT['rotation'] left
                                UNCHANGED — the reloaded object appears exactly
                                as you posed it. Keeps generation's bottom-filter
                                consistent. THIS IS THE USUAL PATH.

  --euler X Y Z                 Direct mesh-local Euler rotation in degrees
                                (sxyz). Good for quick 90° flips / trial.

  --quat W X Y Z                Direct mesh-local quaternion.

After baking: re-run build_object_usd.py --force, then regenerate viewpoints.

Usage:
    uv run scripts/prep/reorient_mesh.py --object curved_structure --euler -90 0 0
    uv run scripts/prep/reorient_mesh.py --object glass --world-target-quat 0.5 0.5 0.5 0.5
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
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from common import config  # noqa: E402
from common.math_utils import quaternion_to_rotation_matrix  # noqa: E402


def bottom_center(bounds: np.ndarray) -> np.ndarray:
    """bbox XY-center + Z-min (matches normalize_mesh.bottom_center)."""
    return np.array([
        (bounds[0, 0] + bounds[1, 0]) / 2.0,
        (bounds[0, 1] + bounds[1, 1]) / 2.0,
        bounds[0, 2],
    ], dtype=float)


def resolve_bake_matrix(args) -> np.ndarray:
    """4x4 mesh-local rotation to bake (rotation only, applied about origin)."""
    if args.euler is not None:
        ax, ay, az = (np.deg2rad(v) for v in args.euler)
        return euler_matrix(ax, ay, az, "sxyz")
    if args.quat is not None:
        return quaternion_matrix(np.asarray(args.quat, dtype=float))  # (w,x,y,z)
    # world-target-quat: R_bake = config_rot⁻¹ @ target, so that
    # config_rot @ R_bake @ v == target @ v (config rotation stays as-is).
    config_rot = quaternion_to_rotation_matrix(config.TARGET_OBJECT["rotation"])
    target_rot = quaternion_to_rotation_matrix(np.asarray(args.world_target_quat, dtype=float))
    R = config_rot.T @ target_rot
    T = np.eye(4)
    T[:3, :3] = R
    return T


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bake a corrective rotation into source.obj")
    p.add_argument("--object", required=True, help="Object name (data/{object}/mesh/)")
    p.add_argument("--input", default="source.obj", help="Input OBJ (default: source.obj)")
    p.add_argument("--output", default="source.obj", help="Output OBJ (default: source.obj)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--world-target-quat", type=float, nargs=4, metavar=("W", "X", "Y", "Z"),
                   help="World Orient from the Isaac gizmo (baked as config_rot⁻¹ ∘ target)")
    g.add_argument("--euler", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   help="Direct mesh-local Euler rotation, degrees (sxyz)")
    g.add_argument("--quat", type=float, nargs=4, metavar=("W", "X", "Y", "Z"),
                   help="Direct mesh-local quaternion (w x y z)")
    p.add_argument("--no-backup", action="store_true", help="Skip writing <output>.bak")
    p.add_argument("--dry-run", action="store_true", help="Print result extents, do not write")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    mesh_dir = PROJECT_ROOT / "data" / args.object / "mesh"
    in_path = mesh_dir / args.input
    out_path = mesh_dir / args.output
    if not in_path.exists():
        sys.exit(f"input not found: {in_path}")

    loaded = trimesh.load(in_path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(list(loaded.geometry.values()))

    bake = resolve_bake_matrix(args)
    mesh = loaded.copy()
    mesh.apply_transform(bake)
    # Re-apply normalize convention so the object still sits on the table.
    mesh.apply_translation(-bottom_center(mesh.bounds))

    print(f"[{args.object}]")
    print(f"  in extents:  {loaded.extents}")
    print(f"  out extents: {mesh.extents}")
    print(f"  out bounds:  {mesh.bounds}")

    if args.dry_run:
        print("  dry-run:     not written")
        return 0

    if not args.no_backup and out_path.exists():
        bak = out_path.with_suffix(out_path.suffix + ".bak")
        shutil.copy2(out_path, bak)
        print(f"  backup:      {bak.relative_to(PROJECT_ROOT)}")
    mesh.export(out_path)
    print(f"  written:     {out_path.relative_to(PROJECT_ROOT)}")
    print("  next: uv run scripts/isaac/usd/build_object_usd.py --object "
          f"{args.object} --force   (then regenerate viewpoints)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
