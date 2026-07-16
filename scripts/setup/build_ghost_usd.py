#!/usr/bin/env python3
"""Build a visual-only "ghost" copy of the camera-equipped robot USD.

Strips every UsdPhysics/PhysxSchema applied API from each prim so the
resulting file can be referenced into a stage alongside the original robot
without perturbing the original's PhysX scene initialization. Joint prims
keep their type so PreviewPlayer can extract the FK chain at runtime, but
are marked `jointEnabled=False`.

This is a one-shot file builder — re-run it if the source USD changes.

Usage:
    uv run scripts/setup/build_ghost_usd.py
        # uses defaults: workcell/robot/ur20_with_camera.usd
        #             -> workcell/robot/ur20_with_camera_ghost.usd
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "workcell" / "robot" / "ur20_with_camera.usd"
DEFAULT_OUTPUT = PROJECT_ROOT / "workcell" / "robot" / "ur20_with_camera_ghost.usd"

PHYSICS_NAME_PARTS = ("Physics", "Physx")
GHOST_TINT = (0.2, 0.8, 0.9)
GHOST_OPACITY = 0.30
GHOST_MATERIAL_NAME = "GhostPreviewMaterial"


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


def define_ghost_material(stage: Usd.Stage) -> UsdShade.Material:
    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        roots = list(stage.GetPseudoRoot().GetChildren())
        if not roots:
            raise RuntimeError("Stage has no root prim")
        root = roots[0]

    material_path = root.GetPath().AppendPath(f"Looks/{GHOST_MATERIAL_NAME}")
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, material_path.AppendPath("PreviewSurface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*GHOST_TINT))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(GHOST_OPACITY)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def force_ghost_material(prim: Usd.Prim, material: UsdShade.Material) -> bool:
    if not (prim.IsA(UsdGeom.Gprim) or prim.GetTypeName() == "GeomSubset"):
        return False
    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
    binding_api.UnbindAllBindings()
    binding_api.Bind(material)
    if prim.IsA(UsdGeom.Gprim):
        tint_gprim(prim)
    return True


def hide_collision_visuals(prim: Usd.Prim) -> bool:
    path = str(prim.GetPath()).lower()
    if "/collisions" not in path and "/collision" not in path:
        return False
    if prim.IsA(UsdGeom.Imageable):
        UsdGeom.Imageable(prim).MakeInvisible()
        return True
    return False


def disable_joint(prim: Usd.Prim) -> None:
    UsdPhysics.Joint(prim).GetJointEnabledAttr().Set(False)


def _relative_asset_path(path: str, out_dir: Path) -> str | None:
    """Return a layer-anchored path for an absolute asset path."""
    if not path or not os.path.isabs(path):
        return None
    relative = os.path.relpath(path, out_dir.resolve())
    return relative if relative.startswith(".") else "./" + relative


def _list_editor_fields(editor) -> tuple[str, ...]:
    return (
        "explicitItems", "prependedItems", "appendedItems",
        "addedItems", "deletedItems", "orderedItems",
    )


def relativize_asset_paths(layer: Sdf.Layer, out_dir: Path) -> int:
    """Relativize sublayers, references, payloads, and attribute asset values."""
    changed = 0

    sublayers = []
    for path in layer.subLayerPaths:
        relative = _relative_asset_path(path, out_dir)
        sublayers.append(relative or path)
        changed += relative is not None
    layer.subLayerPaths = sublayers

    def rewrite_reference(reference: Sdf.Reference) -> Sdf.Reference:
        nonlocal changed
        relative = _relative_asset_path(reference.assetPath, out_dir)
        if relative is None:
            return reference
        changed += 1
        return Sdf.Reference(
            relative, reference.primPath, reference.layerOffset, reference.customData,
        )

    def rewrite_payload(payload: Sdf.Payload) -> Sdf.Payload:
        nonlocal changed
        relative = _relative_asset_path(payload.assetPath, out_dir)
        if relative is None:
            return payload
        changed += 1
        return Sdf.Payload(relative, payload.primPath, payload.layerOffset)

    def visit(spec: Sdf.PrimSpec) -> None:
        nonlocal changed
        spec.referenceList.ModifyItemEdits(rewrite_reference)
        spec.payloadList.ModifyItemEdits(rewrite_payload)

        for prop in spec.properties:
            value = getattr(prop, "default", None)
            if isinstance(value, Sdf.AssetPath):
                relative = _relative_asset_path(value.path, out_dir)
                if relative is not None:
                    prop.default = Sdf.AssetPath(relative)
                    changed += 1
            elif isinstance(value, Sdf.AssetPathArray):
                rewritten = []
                for asset in value:
                    relative = _relative_asset_path(asset.path, out_dir)
                    rewritten.append(Sdf.AssetPath(relative or asset.path))
                    changed += relative is not None
                prop.default = Sdf.AssetPathArray(rewritten)

        for child in spec.nameChildren:
            visit(child)

    for root in layer.rootPrims:
        visit(root)
    return int(changed)


def absolute_asset_paths(layer: Sdf.Layer) -> list[str]:
    """Collect absolute authored asset paths for a post-build portability check."""
    found = [path for path in layer.subLayerPaths if os.path.isabs(path)]

    def inspect_editor(editor) -> None:
        for field in _list_editor_fields(editor):
            for item in getattr(editor, field):
                path = getattr(item, "assetPath", "")
                if os.path.isabs(path):
                    found.append(path)

    def visit(spec: Sdf.PrimSpec) -> None:
        inspect_editor(spec.referenceList)
        inspect_editor(spec.payloadList)
        for prop in spec.properties:
            value = getattr(prop, "default", None)
            if isinstance(value, Sdf.AssetPath) and os.path.isabs(value.path):
                found.append(value.path)
            elif isinstance(value, Sdf.AssetPathArray):
                found.extend(asset.path for asset in value if os.path.isabs(asset.path))
        for child in spec.nameChildren:
            visit(child)

    for root in layer.rootPrims:
        visit(root)
    return sorted(set(found))


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
    source_label = _relative_asset_path(str(args.source.resolve()), args.output.parent)
    flat.documentation = f"Generated from composed layer {source_label}\n"
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

    ghost_material = define_ghost_material(stage)

    n_stripped = n_bound = n_hidden = n_jdisabled = 0
    type_counts: dict = {}
    for prim in stage.Traverse():
        t = prim.GetTypeName()
        type_counts[t] = type_counts.get(t, 0) + 1
        if strip_physics_apis(prim):
            n_stripped += 1
        if force_ghost_material(prim, ghost_material):
            n_bound += 1
        if hide_collision_visuals(prim):
            n_hidden += 1
        if prim.IsA(UsdPhysics.Joint):
            disable_joint(prim)
            n_jdisabled += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_relativized = relativize_asset_paths(flat, args.output.parent)
    remaining = absolute_asset_paths(flat)
    if remaining:
        raise RuntimeError(
            "Ghost USD still contains absolute asset paths:\n  "
            + "\n  ".join(remaining)
        )
    flat.Export(str(args.output))
    exported = Sdf.Layer.FindOrOpen(str(args.output))
    if exported is None:
        raise RuntimeError(f"Cannot reopen generated ghost USD: {args.output}")
    remaining = absolute_asset_paths(exported)
    if remaining:
        raise RuntimeError(
            "Exported ghost USD contains absolute asset paths:\n  "
            + "\n  ".join(remaining)
        )
    print(f"[ghost-usd] wrote {args.output}")
    print(f"           relativized {n_relativized} absolute asset path(s)")
    print(f"           stripped APIs from {n_stripped} prims, "
          f"bound ghost material to {n_bound} prims, "
          f"hid {n_hidden} collision visual prims, "
          f"disabled {n_jdisabled} joints")
    print(f"[ghost-usd] prim types seen ({sum(type_counts.values())} total):")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"             {c:4d}  {t or '(empty/abstract)'}")


if __name__ == "__main__":
    main()
