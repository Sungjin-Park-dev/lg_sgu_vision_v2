#!/usr/bin/env python3
"""Convert a target object's source.obj into source.usd (one-shot, offline).

The Pipeline UI references data/{object}/mesh/source.usd to place the target
object on the stage. Only `sample` ships a source.usd; every other object has
just source.obj. Run this once per object to produce its source.usd so it can
be selected in the Pipeline UI's Object dropdown.

This is kept OUT of the interactive UI on purpose: omni.kit.asset_converter is
an async Kit task, and running it inside the live simulation loop is fragile.
Here it gets its own headless Kit app and we pump simulation_app.update() until
the conversion future resolves.

Usage:
    uv run scripts/isaac/usd/build_object_usd.py --object curved_structure
    uv run scripts/isaac/usd/build_object_usd.py --object glass --force
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"


def _mesh_dir(object_name: str) -> Path:
    return DATA_ROOT / object_name / "mesh"


def main():
    p = argparse.ArgumentParser(description="Convert data/{object}/mesh/source.obj → source.usd")
    p.add_argument("--object", required=True, help="Object name (e.g. curved_structure)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite source.usd if it already exists")
    args = p.parse_args()

    mesh_dir = _mesh_dir(args.object)
    obj_path = mesh_dir / "source.obj"
    usd_path = mesh_dir / "source.usd"

    if not obj_path.exists():
        sys.exit(f"source.obj not found: {obj_path}")
    if usd_path.exists() and not args.force:
        print(f"[build-object-usd] {usd_path} already exists — pass --force to overwrite. Skipping.")
        return

    # Kit app must exist before importing omni.* / isaacsim.* modules.
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True})

    from isaacsim.core.utils import extensions
    extensions.enable_extension("omni.kit.asset_converter")
    simulation_app.update()

    import omni.kit.asset_converter as converter

    def _progress(current: float, total: float) -> None:
        if total:
            print(f"[build-object-usd] converting... {current / total * 100:5.1f}%", end="\r")

    async def _convert(in_path: str, out_path: str) -> None:
        ctx = converter.AssetConverterContext()
        # source.obj is already in meters (prep normalize_mesh mm→m), and the
        # Isaac stage runs at metersPerUnit=1.0. Without this the converter
        # stamps the default cm (metersPerUnit=0.01), so referencing the result
        # triggers "Mismatched units found on drag drop" and the prim loads at
        # the wrong scale / fails. Author meters to match the stage.
        ctx.use_meter_as_world_unit = True
        task = converter.get_instance().create_converter_task(in_path, out_path, _progress, ctx)
        ok = await task.wait_until_finished()
        if not ok:
            raise RuntimeError(
                f"asset conversion failed (status={task.get_status()}): {task.get_error_message()}"
            )

    print(f"[build-object-usd] {obj_path}  →  {usd_path}")
    fut = asyncio.ensure_future(_convert(str(obj_path), str(usd_path)))
    while not fut.done():
        simulation_app.update()
    try:
        fut.result()  # re-raise any conversion error
        # The converter labels the output upAxis=Y, but our geometry is authored
        # Z-up (matching the Isaac stage). Referencing a Y-up asset into the Z-up
        # stage makes Omniverse inject `xformOp:rotateX:unitsResolve=90` on the
        # prim when it's (re)loaded while the sim is playing → the object spawns
        # rotated 90° via the UI "Load Object" button. Force upAxis=Z (metadata
        # only, geometry unchanged) so no correction op is ever added.
        from pxr import Usd, UsdGeom
        fix_stage = Usd.Stage.Open(str(usd_path))
        UsdGeom.SetStageUpAxis(fix_stage, UsdGeom.Tokens.z)
        fix_stage.GetRootLayer().Save()
        print(f"\n[build-object-usd] wrote {usd_path} (upAxis=Z)")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
