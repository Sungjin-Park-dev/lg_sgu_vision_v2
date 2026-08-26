#!/usr/bin/env python3
"""Stamp a scratch decal onto a target object's source.usd as a normal map.

Visual-only: this rewrites ``data/{object}/mesh/source.usd`` (what Isaac's
``load_target_object`` references) and never touches ``source.obj``, so the
viewpoint / GLNS / cuRobo collision side of the pipeline is unaffected.

The scratch PNGs under ``ff/Scratches/`` are *already* tangent-space normal
maps (background exactly RGB(127,127,255) with alpha 0).  So instead of the
Blender ``RGB to BW -> Bump -> Cycles bake`` round-trip -- which throws the
normal directions away and re-derives them from luminance -- we alpha-composite
the decal straight onto a flat normal canvas.  No bake, no fidelity loss, and
the whole thing runs headless in a couple of seconds.

Runs in two stages.  The outer stage (this venv: numpy + PIL + pxr) measures
the mesh, composites the texture and verifies the result; it re-execs itself
inside ``blender -b`` for the mesh work, because bpy is not importable here and
PIL is not importable there.

Examples:
    uv run scripts/setup/apply_scratch_normal.py \
        --object cylinder_sample --scratch ff/Scratches/scratch_16.png
    uv run scripts/setup/apply_scratch_normal.py \
        --object cylinder_sample --scratch ff/Scratches/scratch_16.png \
        --length-mm 40 --u 0.25 --strength 0.6 --preview /tmp/scratch.png
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

try:  # inside `blender -b --python this_file`
    import bpy  # noqa: F401

    INSIDE_BLENDER = True
except ImportError:
    INSIDE_BLENDER = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"


# ============================================================================
# stage 2 -- inside Blender (numpy available, PIL is not)
# ============================================================================

def blender_stage(cfg: dict) -> None:
    import bmesh
    import numpy as np

    bpy.ops.wm.read_homefile(use_empty=True)

    # Blender's OBJ importer defaults to forward=-Z / up=Y, which rotates the
    # mesh 90 deg about X.  Our OBJs are authored Z-up in metres already, so
    # forward=Y / up=Z is the identity we need -- the exported USD extent must
    # come back equal to the OBJ bbox (the outer stage asserts exactly that).
    bpy.ops.wm.obj_import(
        filepath=cfg["obj"], forward_axis="Y", up_axis="Z",
        global_scale=1.0, validate_meshes=True,
    )
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"no mesh imported from {cfg['obj']}")

    obj = meshes[0]
    if len(meshes) > 1:  # multi-material source -> one object to unwrap
        for o in meshes:
            o.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.join()
    for o in bpy.context.scene.objects:
        o.select_set(o is obj)
    bpy.context.view_layer.objects.active = obj

    # STEP tessellations arrive with per-face split vertices (59k for 18k
    # unique on cylinder_sample).  Without merging, the round wall shades
    # faceted and the tangent frame the normal map rides on is garbage.
    before = len(obj.data.vertices)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    print(f"[scratch]   merge by distance: {before} -> {len(obj.data.vertices)} verts")

    # Smooth the curved wall, keep the shoulders/edges sharp.
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(cfg["smooth_deg"]))

    _author_cylindrical_uv(obj, cfg, np)
    _build_material(obj, cfg)

    usd_out = cfg["usd"]
    bpy.ops.wm.usd_export(
        filepath=usd_out,
        selected_objects_only=False,
        export_materials=True,
        generate_preview_surface=True,
        export_uvmaps=True,
        export_normals=True,
        export_textures_mode="KEEP",   # texture already lives next to the USD
        relative_paths=True,           # -> ./textures/scratch_normal.png
        convert_orientation=False,     # keep Z-up
        convert_scene_units="METERS",
        meters_per_unit=1.0,
        root_prim_path="/root",
        evaluation_mode="RENDER",
    )
    print(f"[scratch]   exported {usd_out}")

    if cfg.get("preview"):
        _render_preview(obj, cfg)


def _author_cylindrical_uv(obj, cfg: dict, np) -> None:
    """Write an isotropic cylindrical UV projection directly onto the loops.

    ``bpy.ops.uv.cylinder_project`` needs a VIEW_3D context and is fragile
    headless, so we compute the projection ourselves.  Both axes are divided by
    the same ``m_per_uv`` so that one UV unit is the same number of millimetres
    horizontally and vertically -- that is what keeps a square texture from
    stretching across a 144.5 x 81 mm wall.
    """
    me = obj.data
    m_per_uv = cfg["m_per_uv"]
    u_span = cfg["circumference"] / m_per_uv   # fraction of U the wrap occupies

    nv = len(me.vertices)
    co = np.empty(nv * 3, np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(nv, 3)

    nl = len(me.loops)
    lv = np.empty(nl, np.int32)
    me.loops.foreach_get("vertex_index", lv)
    p = co[lv]

    u = (np.arctan2(p[:, 1], p[:, 0]) / (2 * math.pi) + 0.5) * u_span
    v = (p[:, 2] - cfg["z_min"]) / m_per_uv

    npoly = len(me.polygons)
    ltot = np.empty(npoly, np.int32)
    me.polygons.foreach_get("loop_total", ltot)
    if np.all(ltot == 3) and nl == npoly * 3:
        U = u.reshape(npoly, 3)
        # A face straddling the +-pi seam spans nearly the whole U range; pull
        # its low-side corners forward by one wrap so it stays a small quad in
        # UV instead of wrapping the texture backwards across the face.
        wrap = (U.max(1) - U.min(1)) > 0.5 * u_span
        U[wrap] = np.where(U[wrap] < 0.5 * u_span, U[wrap] + u_span, U[wrap])
        u = U.reshape(-1)
        print(f"[scratch]   seam-fixed {int(wrap.sum())} faces")
    else:
        start = np.empty(npoly, np.int32)
        me.polygons.foreach_get("loop_start", start)
        fixed = 0
        for s, t in zip(start, ltot):
            sl = u[s:s + t]
            if sl.max() - sl.min() > 0.5 * u_span:
                u[s:s + t] = np.where(sl < 0.5 * u_span, sl + u_span, sl)
                fixed += 1
        print(f"[scratch]   seam-fixed {fixed} faces (ngon path)")

    layer = me.uv_layers.get("UVMap") or me.uv_layers.new(name="UVMap")
    layer.data.foreach_set("uv", np.stack([u, v], 1).astype(np.float32).ravel())
    me.update()
    print(f"[scratch]   UV: {nl} loops, u_span={u_span:.3f} v_max={v.max():.3f}")


def _build_material(obj, cfg: dict) -> None:
    """The simple, USD-safe chain: texture -> Normal Map -> Principled BSDF."""
    mat = bpy.data.materials.new(cfg["material_name"])
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*cfg["base_color"], 1.0)
    bsdf.inputs["Roughness"].default_value = cfg["roughness"]
    bsdf.inputs["Metallic"].default_value = 0.0

    img = bpy.data.images.load(cfg["texture"])
    img.colorspace_settings.name = "Non-Color"   # sRGB here would skew normals
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.location = (-600, 0)
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nmap.location = (-300, 0)
    nt.links.new(tex.outputs["Color"], nmap.inputs["Color"])
    nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

    obj.data.materials.clear()
    obj.data.materials.append(mat)


def _render_preview(obj, cfg: dict) -> None:
    """One raking-light Cycles frame, so the groove can be eyeballed offline.

    Head-on light hides a groove entirely -- the shading cue is the shadowed
    wall, so the key light has to come in from the side.
    """
    import mathutils

    scene = bpy.context.scene
    theta = (cfg["u"] / (cfg["circumference"] / cfg["m_per_uv"]) - 0.5) * 2 * math.pi
    z = cfg["z_min"] + cfg["v"] * cfg["m_per_uv"]
    target_at = mathutils.Vector((cfg["r_outer"] * math.cos(theta),
                                  cfg["r_outer"] * math.sin(theta), z))

    aim = bpy.data.objects.new("aim", None)
    scene.collection.objects.link(aim)
    aim.location = target_at

    def _aim(ob, az_off, elev, radius=None):
        """Point `ob` at the scratch; a radius of None means a directional sun."""
        a = theta + math.radians(az_off)
        if radius is not None:
            ob.location = (target_at.x + radius * math.cos(a) * math.cos(math.radians(elev)),
                           target_at.y + radius * math.sin(a) * math.cos(math.radians(elev)),
                           z + radius * math.sin(math.radians(elev)))
        else:
            ob.location = (target_at.x + math.cos(a), target_at.y + math.sin(a),
                           z + math.tan(math.radians(elev)))
        c = ob.constraints.new("TRACK_TO")
        c.target, c.track_axis, c.up_axis = aim, "TRACK_NEGATIVE_Z", "UP_Y"

    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 50
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    _aim(cam, 0.0, 0.0, radius=0.13)
    scene.camera = cam

    # A sun's irradiance does not fall off with distance, so the exposure is
    # predictable no matter how large the object is.  68 deg off the camera
    # axis is what makes the groove readable -- head-on light hides it.
    sun_data = bpy.data.lights.new("key", type="SUN")
    sun_data.energy, sun_data.angle = 2.5, math.radians(3)
    sun = bpy.data.objects.new("key", sun_data)
    scene.collection.objects.link(sun)
    _aim(sun, 68.0, 18.0)

    fill_data = bpy.data.lights.new("fill", type="SUN")
    fill_data.energy = 0.35
    fill = bpy.data.objects.new("fill", fill_data)
    scene.collection.objects.link(fill)
    _aim(fill, -55.0, 10.0)

    scene.render.engine = "CYCLES"        # CPU Cycles always works headless
    scene.cycles.samples = 48
    scene.render.resolution_x = scene.render.resolution_y = 900
    scene.render.filepath = cfg["preview"]
    bpy.ops.render.render(write_still=True)
    print(f"[scratch]   preview -> {cfg['preview']}")


# ============================================================================
# stage 1 -- outer process (numpy + PIL + pxr)
# ============================================================================

def measure_mesh(obj_path: Path) -> dict:
    """Find the dominant outward-facing cylindrical wall of the source mesh."""
    import numpy as np

    verts, faces = [], []
    with obj_path.open() as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append(line.split()[1:4])
            elif line.startswith("f "):
                faces.append([t.split("/")[0] for t in line.split()[1:4]])
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64) - 1

    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(n, axis=1) / 2
    nn = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    centre = tri.mean(1)
    radius = np.linalg.norm(centre[:, :2], axis=1)

    # lateral (not a cap) and facing away from the axis (not a bore wall)
    lateral = (np.abs(nn[:, 2]) < 0.3) & (
        (centre[:, 0] * nn[:, 0] + centre[:, 1] * nn[:, 1]) > 0
    )
    if not lateral.any():
        raise SystemExit(f"{obj_path}: no outward-facing lateral surface found")

    # the radius band holding the most area is the outer wall
    hist, edges = np.histogram(radius[lateral], bins=64, weights=area[lateral])
    b = int(hist.argmax())
    band = lateral & (radius >= edges[b]) & (radius <= edges[b + 1])
    r_outer = float(np.average(radius[band], weights=area[band]))

    circumference = 2 * math.pi * r_outer
    z_min, z_max = float(v[:, 2].min()), float(v[:, 2].max())
    m_per_uv = max(circumference, z_max - z_min)
    return {
        "r_outer": r_outer,
        "circumference": circumference,
        "z_min": z_min,
        "z_max": z_max,
        "wall_z": (float(centre[band, 2].min()), float(centre[band, 2].max())),
        "wall_area": float(area[band].sum()),
        "m_per_uv": m_per_uv,
        "bbox_min": v.min(0).tolist(),
        "bbox_max": v.max(0).tolist(),
    }


def composite_normal_map(scratch: Path, out: Path, *, size: int,
                         length_px: float, u: float, v: float,
                         strength: float) -> tuple[int, int]:
    """Alpha-composite an already-tangent-space scratch onto a flat canvas."""
    import numpy as np
    from PIL import Image

    src = Image.open(scratch).convert("RGBA")
    a_full = np.asarray(src)[:, :, 3]
    ys, xs = np.nonzero(a_full > 8)
    if len(xs) == 0:
        raise SystemExit(f"{scratch}: fully transparent, nothing to stamp")
    src = src.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))

    scale = length_px / max(src.width, src.height)
    dst = (max(1, round(src.width * scale)), max(1, round(src.height * scale)))
    stamp = np.asarray(src.resize(dst, Image.LANCZOS), dtype=np.float32)

    sn = stamp[:, :, :3] / 255.0 * 2.0 - 1.0
    sa = (stamp[:, :, 3] / 255.0)[:, :, None]
    # --strength is the analogue of the doc's Bump Strength: flatten the
    # tangential deviation, then rebuild z so the result stays unit length.
    sn[:, :, :2] *= strength
    sn[:, :, 2] = np.sqrt(np.clip(1.0 - (sn[:, :, :2] ** 2).sum(2), 0.0, 1.0))

    layer = np.zeros((size, size, 3), np.float32)
    alpha = np.zeros((size, size, 1), np.float32)
    h, w = stamp.shape[:2]
    y0 = int(round((1.0 - v) * size - h / 2))       # UV origin is bottom-left
    x0 = int(round(u * size - w / 2))
    row = np.arange(y0, y0 + h)
    col = np.arange(x0, x0 + w) % size              # U wraps around the cylinder
    keep = (row >= 0) & (row < size)
    idx = np.ix_(row[keep], col)
    layer[idx] = sn[keep]
    alpha[idx] = sa[keep]

    flat = np.zeros((size, size, 3), np.float32)
    flat[:, :, 2] = 1.0
    out_n = flat * (1.0 - alpha) + layer * alpha
    out_n /= np.maximum(np.linalg.norm(out_n, axis=2, keepdims=True), 1e-8)

    rgb = np.clip(np.rint((out_n * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, "RGB").save(out)
    return dst


def verify_usd(usd: Path, expect_bbox: tuple[list, list], reference: Path | None) -> None:
    """Fail loudly on the failure modes that are invisible until Isaac loads."""
    from pxr import Gf, Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(str(usd))
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z, "upAxis is not Z"
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0, "metersPerUnit is not 1"

    mesh = next((p for p in stage.Traverse() if p.GetTypeName() == "Mesh"), None)
    assert mesh is not None, "no Mesh prim in exported USD"
    g = UsdGeom.Mesh(mesh)

    ext = g.GetExtentAttr().Get()
    lo, hi = expect_bbox
    for got, want, tag in ((ext[0], lo, "min"), (ext[1], hi, "max")):
        assert Gf.IsClose(Gf.Vec3f(*got), Gf.Vec3f(*[float(x) for x in want]), 1e-4), (
            f"extent {tag} {tuple(got)} != OBJ bbox {tuple(want)} "
            f"-- axis or scale went wrong on import"
        )
    names = [pv.GetName() for pv in UsdGeom.PrimvarsAPI(mesh).GetPrimvars()]
    assert "primvars:st" in names, f"no UV primvar exported (got {names})"
    assert g.GetNormalsAttr().Get(), "no normals exported"

    shaders = {p.GetPath().name: UsdShade.Shader(p)
               for p in stage.Traverse() if p.GetTypeName() == "Shader"}
    surf = next((s for s in shaders.values()
                 if s.GetIdAttr().Get() == "UsdPreviewSurface"), None)
    assert surf is not None, "no UsdPreviewSurface shader"
    nrm = surf.GetInput("normal")
    assert nrm and nrm.HasConnectedSource(), "BSDF normal input is not connected"
    tex = UsdShade.Shader(nrm.GetConnectedSource()[0].GetPrim())
    assert tex.GetIdAttr().Get() == "UsdUVTexture", "normal source is not a texture"

    cs = tex.GetInput("sourceColorSpace").Get()
    assert cs == "raw", f"texture colorspace is {cs!r}, expected 'raw'"
    for key, want in (("scale", (2, 2, 2, 2)), ("bias", (-1, -1, -1, -1))):
        got = tex.GetInput(key).Get()
        assert got and tuple(got) == want, f"texture {key}={got}, expected {want}"

    asset = tex.GetInput("file").Get()
    assert asset and Path(asset.resolvedPath).exists(), (
        f"texture path does not resolve: {asset}"
    )
    st = tex.GetInput("st")
    assert st and st.HasConnectedSource(), "texture st input is not connected"

    print(f"[scratch] verified {usd}")
    print(f"[scratch]   extent {tuple(ext[0])} .. {tuple(ext[1])}")
    print(f"[scratch]   texture {asset.path}")

    if reference and reference.exists():
        ref = Usd.Stage.Open(str(reference))
        ref_ids = sorted(UsdShade.Shader(p).GetIdAttr().Get()
                         for p in ref.Traverse() if p.GetTypeName() == "Shader")
        got_ids = sorted(s.GetIdAttr().Get() for s in shaders.values())
        assert got_ids == ref_ids, (
            f"shader network {got_ids} differs from precedent {reference.name} {ref_ids}"
        )
        print(f"[scratch]   shader network matches {reference.name}: {got_ids}")


def stamp_provenance(usd: Path, params: dict) -> None:
    """ff/ is gitignored, so record what produced this USD inside the USD."""
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd))
    prim = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    data = dict(prim.GetCustomData())
    data["scratchNormal"] = {k: str(v) for k, v in params.items()}
    prim.SetCustomData(data)
    stage.GetRootLayer().Save()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Stamp a scratch normal map onto data/{object}/mesh/source.usd")
    p.add_argument("--object", required=True, help="Object name (e.g. cylinder_sample)")
    p.add_argument("--scratch", required=True, type=Path,
                   help="Scratch PNG (tangent-space normal map with alpha)")
    p.add_argument("--length-mm", type=float, default=20.0,
                   help="Scratch length along its long axis, in mm (default 20)")
    p.add_argument("--u", type=float, default=0.5,
                   help="Circumferential position, 0..1 (default 0.5)")
    p.add_argument("--v", default="center",
                   help="Height position 0..1, or 'center' of the wall (default)")
    p.add_argument("--strength", type=float, default=1.0,
                   help="Groove depth, 1.0 = source normals unchanged (default 1.0)")
    p.add_argument("--tex-size", type=int, default=2048)
    p.add_argument("--smooth-deg", type=float, default=30.0)
    p.add_argument("--roughness", type=float, default=0.5)
    p.add_argument("--preview", type=Path, help="Also render a raking-light preview PNG")
    p.add_argument("--force", action="store_true",
                   help="Refresh source_prev.usd from the current source.usd")
    p.add_argument("--blender", type=Path,
                   default=Path(shutil.which("blender") or "/usr/local/bin/blender"))
    args = p.parse_args()

    mesh_dir = DATA_ROOT / args.object / "mesh"
    obj_path = mesh_dir / "source.obj"
    usd_path = mesh_dir / "source.usd"
    prev_path = mesh_dir / "source_prev.usd"
    tex_path = mesh_dir / "textures" / "scratch_normal.png"

    if not obj_path.exists():
        sys.exit(f"source.obj not found: {obj_path}")
    if not args.scratch.exists():
        sys.exit(f"scratch PNG not found: {args.scratch}")
    if not args.blender.exists():
        sys.exit(f"blender not found: {args.blender} (pass --blender)")

    m = measure_mesh(obj_path)
    m_per_uv = m["m_per_uv"]
    if args.v == "center":
        v = (sum(m["wall_z"]) / 2 - m["z_min"]) / m_per_uv
    else:
        v = float(args.v)
    u = args.u % 1.0

    print(f"[scratch] {args.object}: outer wall r={m['r_outer'] * 1000:.1f}mm "
          f"circumference={m['circumference'] * 1000:.1f}mm "
          f"height={(m['z_max'] - m['z_min']) * 1000:.1f}mm "
          f"area={m['wall_area'] * 1e4:.1f}cm2")

    length_px = args.length_mm / 1000.0 / m_per_uv * args.tex_size
    dst = composite_normal_map(
        args.scratch, tex_path, size=args.tex_size, length_px=length_px,
        u=u, v=v, strength=args.strength,
    )
    print(f"[scratch] texture {tex_path.relative_to(PROJECT_ROOT)} "
          f"({args.tex_size}^2, stamp {dst[0]}x{dst[1]}px for {args.length_mm}mm) "
          f"at u={u:.3f} v={v:.3f} strength={args.strength}")

    if usd_path.exists() and (args.force or not prev_path.exists()):
        shutil.copy2(usd_path, prev_path)
        print(f"[scratch] backed up -> {prev_path.relative_to(PROJECT_ROOT)}")

    cfg = {
        "obj": str(obj_path), "usd": str(usd_path), "texture": str(tex_path),
        "material_name": f"{args.object}_scratch",
        "base_color": [0.9, 0.9, 0.9], "roughness": args.roughness,
        "smooth_deg": args.smooth_deg,
        "m_per_uv": m_per_uv, "circumference": m["circumference"],
        "r_outer": m["r_outer"], "z_min": m["z_min"],
        "u": u, "v": v,
        "preview": str(args.preview) if args.preview else None,
    }
    proc = subprocess.run(
        [str(args.blender), "-b", "--factory-startup", "--python", str(Path(__file__).resolve()),
         "--", json.dumps(cfg)],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("[scratch]") or "Error" in line or "Traceback" in line:
            print(line)
    if proc.returncode != 0:
        sys.exit(f"blender stage failed (rc={proc.returncode}):\n{proc.stdout[-4000:]}\n{proc.stderr[-2000:]}")

    verify_usd(usd_path, (m["bbox_min"], m["bbox_max"]),
               DATA_ROOT / "square_structure" / "mesh" / "source.usd")
    stamp_provenance(usd_path, {
        "scratch": args.scratch, "lengthMm": args.length_mm,
        "u": round(u, 4), "v": round(v, 4), "strength": args.strength,
        "texSize": args.tex_size,
    })
    print(f"[scratch] done -> {usd_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    if INSIDE_BLENDER:
        blender_stage(json.loads(sys.argv[sys.argv.index("--") + 1]))
    else:
        main()
