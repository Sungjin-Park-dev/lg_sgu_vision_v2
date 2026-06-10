#!/usr/bin/env python3
"""Build a visual-only "ghost" copy of the robot USD.

Strips every UsdPhysics/PhysxSchema applied API from each prim so the
resulting file can be referenced into a stage alongside the original robot
without perturbing the original's PhysX scene initialization. Joint prims
keep their type (so PreviewPlayer can extract the FK chain at runtime) but
are marked `jointEnabled=False`. Every Gprim gets a cyan tint + 30%
opacity baked in.

This is a one-shot file builder — re-run it if the source USD changes.

Usage:
    uv run scripts/isaac/usd/build_ghost_usd.py
        # uses defaults: ur20_description/ur20_with_camera.usd
        #             -> ur20_description/ur20_with_camera_ghost.usd
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Source the camera-less inner robot — the wrapper "ur20_with_camera.usd"
# pulls a sensor payload from a non-local S3 URL that we don't need for the
# ghost (we only want the arm geometry).
DEFAULT_SOURCE = PROJECT_ROOT / "ur20_description" / "ur20" / "ur20.usd"
DEFAULT_OUTPUT = PROJECT_ROOT / "ur20_description" / "ur20_ghost.usd"

# Substrings that mark an applied schema as physics-related.
PHYSICS_NAME_PARTS = ("Physics", "Physx")
GHOST_TINT = (0.2, 0.8, 0.9)   # cyan, MoveIt-ghost vibe
GHOST_OPACITY = 0.30


def is_physics_schema(name: str) -> bool:
    return any(part in name for part in PHYSICS_NAME_PARTS)


def strip_physics_apis(prim: Usd.Prim) -> bool:
    """Remove physics-related applied API schemas from prim. Returns True if changed."""
    applied = list(prim.GetAppliedSchemas())
    if not applied:
        return False
    kept = [s for s in applied if not is_physics_schema(s)]
    if kept == applied:
        return False
    prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(kept))
    return True


def tint_gprim(prim: Usd.Prim) -> None:
    gp = UsdGeom.Gprim(prim)
    gp.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*GHOST_TINT)]))
    gp.CreateDisplayOpacityAttr(Vt.FloatArray([GHOST_OPACITY]))


def disable_joint(prim: Usd.Prim) -> None:
    UsdPhysics.Joint(prim).GetJointEnabledAttr().Set(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Source USD missing: {args.source}")

    # Load everything (payloads included) so internal references and visual
    # meshes show up in Traverse(). Default load policy already loads all
    # payloads, but make it explicit for clarity.
    src = Usd.Stage.Open(str(args.source), load=Usd.Stage.LoadAll)
    if src is None:
        raise RuntimeError(f"Cannot open source: {args.source}")

    # Flatten the source so every composed opinion lives on a single
    # in-memory layer we can freely mutate. Avoids the headache of
    # override-via-reference for API removal.
    flat = src.Flatten()
    stage = Usd.Stage.Open(flat)

    # Expand instancing so meshes that would otherwise live in a (read-only)
    # prototype tree become regular, editable prims under the main tree.
    # Isaac Sim's URDF importer aggressively instances mesh prims; without
    # this step Traverse() only reaches the Xform proxies, not the Meshes.
    n_expanded = 0
    for prim in list(stage.Traverse()):
        if prim.IsInstance():
            prim.SetInstanceable(False)
            n_expanded += 1
    if n_expanded:
        print(f"[ghost-usd] expanded {n_expanded} instance(s) into regular prims")

    n_stripped = n_tinted = n_jdisabled = 0
    type_counts: dict = {}
    for prim in stage.Traverse():
        t = prim.GetTypeName()
        type_counts[t] = type_counts.get(t, 0) + 1
        if strip_physics_apis(prim):
            n_stripped += 1
        if prim.IsA(UsdGeom.Gprim):
            tint_gprim(prim)
            n_tinted += 1
        if prim.IsA(UsdPhysics.Joint):
            disable_joint(prim)
            n_jdisabled += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    flat.Export(str(args.output))
    print(f"[ghost-usd] wrote {args.output}")
    print(f"           stripped APIs from {n_stripped} prims, "
          f"tinted {n_tinted} gprims, disabled {n_jdisabled} joints")
    print(f"[ghost-usd] prim types seen ({sum(type_counts.values())} total):")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"             {c:4d}  {t or '(empty/abstract)'}")


if __name__ == "__main__":
    main()
