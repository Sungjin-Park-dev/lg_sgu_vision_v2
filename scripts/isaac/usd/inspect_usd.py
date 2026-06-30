#!/usr/bin/env python3
"""Dump type counts + instance/prototype info for a USD file.

Helps figure out why a `Traverse()`-based pass missed prims (instancing,
unloaded payloads, classes, etc.).

Usage:
    uv run scripts/isaac/usd/inspect_usd.py
        # defaults to workcell/robot/ur20_with_camera.usd
    uv run scripts/isaac/usd/inspect_usd.py --path workcell/robot/ur20/ur20.usd
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from pxr import Usd, UsdGeom

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATH = PROJECT_ROOT / "workcell" / "robot" / "ur20_with_camera.usd"


def count_types(prims):
    c: Counter = Counter()
    for p in prims:
        c[p.GetTypeName() or "(empty)"] += 1
    return c


def dump(label, counter):
    print(f"=== {label} ({sum(counter.values())} prims) ===")
    for t, n in counter.most_common():
        print(f"  {n:4d}  {t}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", type=Path, default=DEFAULT_PATH)
    args = p.parse_args()

    stage = Usd.Stage.Open(str(args.path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise SystemExit(f"Cannot open {args.path}")

    print(f"Inspecting: {args.path}")
    # Stage frame metadata — the usual culprit when a converted asset spawns rotated.
    print(f"upAxis:         {UsdGeom.GetStageUpAxis(stage)}")
    print(f"metersPerUnit:  {UsdGeom.GetStageMetersPerUnit(stage)}")
    default = stage.GetDefaultPrim()
    print(f"Default prim: {default.GetPath() if default else '(none)'}")
    if default:
        xf = UsdGeom.Xformable(default)
        ops = xf.GetOrderedXformOps()
        if ops:
            print("  default prim xformOps:")
            for op in ops:
                print(f"    {op.GetOpName():28s} = {op.Get()}")
        else:
            print("  default prim xformOps: (none — clean)")
    print()

    # 0. Every prim carrying a non-identity xformOp — finds rotations hiding in
    #    child Xforms (the usual reason a converted asset spawns rotated).
    print("=== prims with xformOps ===")
    any_op = False
    for prim in stage.Traverse():
        xf = UsdGeom.Xformable(prim)
        if not xf:
            continue
        ops = xf.GetOrderedXformOps()
        if not ops:
            continue
        any_op = True
        print(f"  {prim.GetPath()}  [{prim.GetTypeName()}]")
        for op in ops:
            print(f"      {op.GetOpName():26s} = {op.Get()}")
    if not any_op:
        print("  (none anywhere — fully clean)")
    print()

    # 1. Regular traverse (excludes instance masters, inactive, abstract)
    dump("stage.Traverse()", count_types(stage.Traverse()))

    # 2. Traverse with instance proxies expanded
    pred = Usd.PrimAllPrimsPredicate
    dump("\nstage.TraverseAll() (incl. inactive/abstract)",
         count_types(stage.TraverseAll()))

    # 3. Prototype (instance-master) trees
    protos = stage.GetPrototypes()
    print(f"\n=== {len(protos)} prototype(s) ===")
    for proto in protos:
        proto_types = count_types(Usd.PrimRange(proto, pred))
        print(f"  Prototype: {proto.GetPath()}  ({sum(proto_types.values())} prims)")
        for t, n in proto_types.most_common():
            print(f"      {n:4d}  {t}")

    # 4. List any instance prims at the top of the regular tree
    instances = [p for p in stage.Traverse() if p.IsInstance()]
    if instances:
        print(f"\n=== {len(instances)} instance prim(s) ===")
        for inst in instances:
            proto = inst.GetPrototype()
            print(f"  {inst.GetPath()}  ->  {proto.GetPath() if proto else '(no proto)'}")


if __name__ == "__main__":
    main()
