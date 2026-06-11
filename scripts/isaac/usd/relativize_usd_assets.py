#!/usr/bin/env python3
"""Rewrite absolute container asset paths baked into USDs into layer-relative paths.

USDs built inside a container bake absolute asset paths like
``/workspace/ur20_description/ur20/configuration/materials/textures/UR20_DIFF_8bit_2K.png``.
On the host the project lives elsewhere, so the referenced asset can't be found:

    [Error] [omni.rtx.materials] [UsdToMdl] ... parameter 'diffuse_texture':
    References an asset that can not be found: '/workspace/.../UR20_DIFF_8bit_2K.png'

This scans USD layers and rewrites any asset-valued attribute whose path starts
with the container prefix (default ``/workspace``) into a path **relative to the
layer that authored it**, so it resolves regardless of where the project is
checked out. Idempotent: paths already relative are left untouched.

Mapping assumption: the container mounted the project at ``--container-prefix``
with the same internal layout as the host ``--host-root`` (the project root), i.e.
``/workspace/ur20_description/...`` ↔ ``<project>/ur20_description/...``.

The USD crate (binary) file-format plugin is only registered inside a Kit app,
so this runs in a headless SimulationApp like build_object_usd.py / build_ghost_usd.py.

Usage:
    uv run scripts/isaac/usd/relativize_usd_assets.py                  # scan ur20_description/
    uv run scripts/isaac/usd/relativize_usd_assets.py --dry-run        # report only, no writes
    uv run scripts/isaac/usd/relativize_usd_assets.py --usd a.usd b.usd
    uv run scripts/isaac/usd/relativize_usd_assets.py --container-prefix /root
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def out(msg: str = "") -> None:
    """Print to the ORIGINAL stderr fd. Kit hijacks sys.stdout after the
    SimulationApp boots, so ordinary print() output gets swallowed; writing to
    sys.__stderr__ (fd 2) keeps the report visible in the terminal."""
    sys.__stderr__.write(msg + "\n")
    sys.__stderr__.flush()


def _iter_usd_files(scan_dir: Path):
    for p in sorted(scan_dir.rglob("*.usd")):
        if ".thumbs" in p.parts:
            continue
        yield p
    for ext in ("*.usda", "*.usdc"):
        for p in sorted(scan_dir.rglob(ext)):
            if ".thumbs" in p.parts:
                continue
            yield p


def _to_relative(container_path: str, prefix: str, host_root: Path, layer_dir: Path):
    """container_path under `prefix` → (relative-to-layer path, host abspath) or None."""
    norm = container_path.replace("\\", "/")
    pfx = prefix.rstrip("/") + "/"
    if not norm.startswith(pfx):
        return None
    rest = norm[len(pfx):]
    host_abs = (host_root / rest).resolve()
    rel = os.path.relpath(host_abs, layer_dir.resolve())
    if not rel.startswith((".", "/")):
        rel = "./" + rel  # force layer-anchored relative (not a search path)
    return rel, host_abs


def _fix_layer(Sdf, usd_path: Path, prefix: str, host_root: Path, dry_run: bool):
    layer = Sdf.Layer.FindOrOpen(str(usd_path))
    if layer is None:
        out(f"  [skip] cannot open {usd_path}")
        return 0
    layer_dir = usd_path.parent
    changes = []

    def visit(prim_spec):
        for prop in prim_spec.properties:
            if not isinstance(prop, Sdf.AttributeSpec):
                continue
            v = prop.default
            # Single asset value.
            if isinstance(v, Sdf.AssetPath) and v.path:
                conv = _to_relative(v.path, prefix, host_root, layer_dir)
                if conv:
                    rel, host_abs = conv
                    if not dry_run:
                        prop.default = Sdf.AssetPath(rel)
                    changes.append((str(prop.path), v.path, rel, host_abs.exists()))
            # Array of asset values.
            elif isinstance(v, Sdf.AssetPathArray) and len(v):
                new_items, touched = [], False
                for ap in v:
                    conv = _to_relative(ap.path, prefix, host_root, layer_dir) if ap.path else None
                    if conv:
                        rel, host_abs = conv
                        new_items.append(Sdf.AssetPath(rel))
                        touched = True
                        changes.append((str(prop.path), ap.path, rel, host_abs.exists()))
                    else:
                        new_items.append(ap)
                if touched and not dry_run:
                    prop.default = Sdf.AssetPathArray(new_items)
        for child in prim_spec.nameChildren:
            visit(child)

    for root in layer.rootPrims:
        visit(root)

    # Composition arcs (references / payloads / sublayers) are NOT attribute
    # values — handle them through the layer-level dependency API. This catches
    # e.g. `prepend references = @/workspace/.../ur20.usd@`.
    for dep in layer.GetCompositionAssetDependencies():
        conv = _to_relative(dep, prefix, host_root, layer_dir)
        if conv:
            rel, host_abs = conv
            if not dry_run:
                layer.UpdateCompositionAssetDependency(dep, rel)
            changes.append(("<composition arc>", dep, rel, host_abs.exists()))

    if not changes:
        return 0, True
    out(f"\n=== {usd_path.relative_to(PROJECT_ROOT)} ===  ({len(changes)} path(s))")
    for attr_path, old, new, exists in changes:
        flag = "" if exists else "  [!! target missing on host]"
        out(f"  {attr_path}\n    - {old}\n    + {new}{flag}")
    if dry_run:
        return len(changes), True
    # USDs built in a container are often root-owned + read-only; layer.Save()
    # would raise. Pre-check so we report a clear chown hint and keep going
    # instead of dying on the first file.
    if not os.access(str(usd_path), os.W_OK):
        out("  [!] NOT saved — file is read-only (root-owned?). Needs chown.")
        return len(changes), False
    layer.Save()
    out("  saved.")
    return len(changes), True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usd", nargs="+", type=Path, default=None,
                    help="USD file(s) to fix. Default: scan ur20_description/")
    ap.add_argument("--scan-dir", type=Path, default=PROJECT_ROOT / "ur20_description",
                    help="Directory to scan for *.usd when --usd is not given")
    ap.add_argument("--container-prefix", default="/workspace",
                    help="Absolute container prefix to strip (default: /workspace)")
    ap.add_argument("--host-root", type=Path, default=PROJECT_ROOT,
                    help="Host path the container prefix maps to (default: project root)")
    ap.add_argument("--dry-run", action="store_true", help="Report changes, write nothing")
    args = ap.parse_args()

    files = list(args.usd) if args.usd else list(_iter_usd_files(args.scan_dir))
    files = [f if f.is_absolute() else (PROJECT_ROOT / f) for f in files]
    if not files:
        sys.exit(f"No USD files found (scan-dir: {args.scan_dir})")

    # Kit app must exist before USD crate file-format plugins are registered.
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True})
    try:
        from pxr import Sdf
        total = 0
        unsaved = []
        for f in files:
            n, saved = _fix_layer(Sdf, f, args.container_prefix, args.host_root, args.dry_run)
            total += n
            if n and not saved:
                unsaved.append(f)
        verb = "would rewrite" if args.dry_run else "rewrote"
        out(f"\n[relativize] {verb} {total} asset path(s) across {len(files)} file(s).")
        if unsaved:
            out("\n[!] These files are read-only (root-owned, likely built in a container)\n"
                "    so the changes were NOT written. Grant ownership, then re-run:\n"
                "      sudo chown $USER:$USER \\\n        "
                + " \\\n        ".join(str(u) for u in unsaved))
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
