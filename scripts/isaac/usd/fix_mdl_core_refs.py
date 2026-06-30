#!/usr/bin/env python3
"""Rewrite fragile file-path references to Isaac *core* MDL modules into bare
search-path module names, so they resolve regardless of where the USD lives.

Some USDs (e.g. thor_table.usd) were saved with the OmniPBR core material
referenced by a layer-relative path into the venv, e.g.

    info:mdl:sourceAsset = @../.venv/lib/python3.12/site-packages/isaacsim/kit/mdl/core/Base/OmniPBR.mdl@

That ``../`` count is anchored to the USD's directory depth, so moving the USD
(e.g. ur20_description/ → workcell/environment/) makes it point at a nonexistent
path and the material falls back to red:

    [omni.rtx.materials] Unable to find SdrShaderNode for prim '/World/Table/Looks/.../Shader'
    USD_MDL: ... '../.venv/.../OmniPBR.mdl' is Invalid

Isaac core MDL modules (``.../mdl/core/...``) are already on the MDL module
search path, so the robust reference is just the bare filename (``OmniPBR.mdl``),
which resolves anywhere. This scans USD layers and rewrites any
``*:sourceAsset`` asset path that points at a core MDL into its basename.
Idempotent: bare references are left untouched.

The USD crate (binary) file-format plugin is only registered inside a Kit app,
so this runs in a headless SimulationApp like relativize_usd_assets.py.

Usage:
    uv run scripts/isaac/usd/fix_mdl_core_refs.py                  # scan workcell/
    uv run scripts/isaac/usd/fix_mdl_core_refs.py --dry-run        # report only, no writes
    uv run scripts/isaac/usd/fix_mdl_core_refs.py --usd workcell/environment/thor_table.usd
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def out(msg: str = "") -> None:
    """Print to the ORIGINAL stderr fd. Kit hijacks sys.stdout after the
    SimulationApp boots, so ordinary print() output gets swallowed."""
    sys.__stderr__.write(msg + "\n")
    sys.__stderr__.flush()


def _iter_usd_files(scan_dir: Path):
    for ext in ("*.usd", "*.usda", "*.usdc"):
        for p in sorted(scan_dir.rglob(ext)):
            if ".thumbs" in p.parts:
                continue
            yield p


def _to_search_path(asset_path: str):
    """A fragile file path to a core Isaac MDL → its bare module name, else None.

    Targets paths that reach a core MDL via the venv / a ``mdl/core/`` tree.
    A reference that is already a bare ``Foo.mdl`` (no separator) is left alone.
    """
    norm = asset_path.replace("\\", "/")
    if not norm.endswith(".mdl") or "/" not in norm:
        return None
    if "/mdl/core/" not in norm and ".venv/" not in norm and "site-packages/" not in norm:
        return None
    return os.path.basename(norm)


def _fix_layer(Sdf, usd_path: Path, dry_run: bool):
    layer = Sdf.Layer.FindOrOpen(str(usd_path))
    if layer is None:
        out(f"  [skip] cannot open {usd_path}")
        return 0, True
    changes = []

    def visit(prim_spec):
        for prop in prim_spec.properties:
            if not isinstance(prop, Sdf.AttributeSpec):
                continue
            v = prop.default
            if isinstance(v, Sdf.AssetPath) and v.path:
                bare = _to_search_path(v.path)
                if bare and bare != v.path:
                    if not dry_run:
                        prop.default = Sdf.AssetPath(bare)
                    changes.append((str(prop.path), v.path, bare))
            elif isinstance(v, Sdf.AssetPathArray) and len(v):
                new_items, touched = [], False
                for ap in v:
                    bare = _to_search_path(ap.path) if ap.path else None
                    if bare and bare != ap.path:
                        new_items.append(Sdf.AssetPath(bare))
                        touched = True
                        changes.append((str(prop.path), ap.path, bare))
                    else:
                        new_items.append(ap)
                if touched and not dry_run:
                    prop.default = Sdf.AssetPathArray(new_items)
        for child in prim_spec.nameChildren:
            visit(child)

    for root in layer.rootPrims:
        visit(root)

    if not changes:
        return 0, True
    out(f"\n=== {usd_path.relative_to(PROJECT_ROOT)} ===  ({len(changes)} path(s))")
    for attr_path, old, new in changes:
        out(f"  {attr_path}\n    - {old}\n    + {new}")
    if dry_run:
        return len(changes), True
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
                    help="USD file(s) to fix. Default: scan workcell/")
    ap.add_argument("--scan-dir", type=Path, default=PROJECT_ROOT / "workcell",
                    help="Directory to scan for *.usd when --usd is not given")
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
            n, saved = _fix_layer(Sdf, f, args.dry_run)
            total += n
            if n and not saved:
                unsaved.append(f)
        verb = "would rewrite" if args.dry_run else "rewrote"
        out(f"\n[fix-mdl] {verb} {total} core-MDL ref(s) across {len(files)} file(s).")
        if unsaved:
            out("\n[!] These files are read-only (root-owned, likely built in a container)\n"
                "    so the changes were NOT written. Grant ownership, then re-run:\n"
                "      sudo chown $USER:$USER \\\n        "
                + " \\\n        ".join(str(u) for u in unsaved))
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
