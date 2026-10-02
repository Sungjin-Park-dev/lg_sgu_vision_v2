#!/usr/bin/env python3
"""Stamp scratch decals onto a target object's source.usd as normal maps.

Visual-only: this rewrites ``data/{object}/mesh/source.usd`` (what Isaac's
``load_target_object`` references) and never touches ``source.obj``, so the
viewpoint / GLNS / cuRobo collision side of the pipeline is unaffected.

The scratch PNGs under ``ff/Scratches/`` are *already* tangent-space normal
maps (background exactly RGB(127,127,255) with alpha 0).  So instead of the
Blender ``RGB to BW -> Bump -> Cycles bake`` round-trip -- which throws the
normal directions away and re-derives them from luminance -- we alpha-composite
the decal straight onto a flat normal canvas.  No bake, no fidelity loss, and
the whole thing runs headless in a couple of seconds.

**Local decals, not a global unwrap.**  A scratch is 20-40 mm on parts that are
boxes, cylinders and freeform shells, so there is no one UV layout that suits
every object.  Instead each scratch gets its own small patch of faces, its own
texture and its own material, and the patch is mapped by projecting it along
the scratch's normal.  The patch stops at sharp edges and where the surface
turns away, so a decal never wraps around a corner.  Faces are free to extend
past the texture (``sample`` has walls made of two huge triangles) because the
texture clamps to its flat border.

Runs in two stages.  The outer stage (this venv: numpy + PIL + trimesh + pxr)
plans the placements, composites the textures and verifies the result; it
re-execs itself inside ``blender -b`` for the mesh work, because bpy is not
importable here and PIL is not importable there.

Examples:
    # three random scratches, reproducible from the seed
    uv run scripts/setup/apply_scratch_normal.py --object sample --random 3 --seed 0
    # one scratch at a chosen spot (object frame, mm), 30 deg in the tangent plane
    uv run scripts/setup/apply_scratch_normal.py --object cylinder_sample \
        --scratch ff/Scratches/scratch_16.png --at 23 0 40 --angle-deg 30 --length-mm 40
    # replay exactly what a previous run recorded
    uv run scripts/setup/apply_scratch_normal.py --object sample \
        --spec data/sample/mesh/scratches.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

try:  # inside `blender -b --python this_file`
    import bpy  # noqa: F401

    INSIDE_BLENDER = True
except ImportError:
    INSIDE_BLENDER = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
SCRATCH_DIR = PROJECT_ROOT / "ff" / "Scratches"

# The texture is a square window on the surface, this many scratch lengths wide.
# The margin is what the decal's clamped border is made of, so it has to be wide
# enough that a face reaching past the window only ever reads flat pixels.
SPAN_FACTOR = 1.6
# A random placement only demands smooth surface around the *scratch itself*,
# not around the whole window.  Demanding the window (1.2 x length) rules out
# small parts entirely -- on the 23mm-radius cylinder every spot is within that
# distance of a rim.  The patch still stops at creases, so a window that runs
# off an edge is simply clipped there.
SMOOTH_RADIUS_FACTOR = 0.6
# A decal lives on one side of the part.  Creases alone do not stop it: a
# rounded edge is smooth all the way over, so on square_structure a patch rode
# a fillet onto the neighbouring wall and grew to 2278 faces that no flattening
# could hold (area ratio 0.17-6.4).  Turning away this far from the scratch's
# own normal ends the patch.
MAX_TILT_DEG = 70.0


# ============================================================================
# stage 2 -- inside Blender (numpy available, PIL is not)
# ============================================================================

def blender_stage(cfg: dict) -> None:
    """Blender keeps running (and exits 0) after a --python script raises, so
    every failure in here has to become an explicit non-zero exit."""
    try:
        _blender_stage(cfg)
    except Exception as exc:                        # noqa: BLE001 -- see docstring
        import traceback

        traceback.print_exc()
        print(f"[scratch] FAILED: {exc}")
        sys.exit(1)


def _blender_stage(cfg: dict) -> None:
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

    # Smooth the curved walls, keep the shoulders/edges sharp.
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(cfg["smooth_deg"]))

    _ensure_base_material(obj, cfg)
    _cut_decal_boundaries(obj, cfg["scratches"], bmesh, np)
    _author_box_uv(obj, np)     # background UV: unused by the shader, but never degenerate

    claimed = {}
    for i, s in enumerate(cfg["scratches"]):
        patch = _grow_patch(obj, s, cfg["smooth_deg"], np)
        # A face holds one UV, so it can serve one decal.  Where two patches
        # meet, the first keeps the face and this one goes without it: the
        # overlap is margin, and yielding a little margin beats refusing to
        # place the scratch at all (three scratches cannot help but meet on a
        # 46mm-wide cylinder).
        clash = [f for f in patch if f in claimed]
        if clash:
            patch = [f for f in patch if f not in claimed]
            print(f"[scratch]   {s['material']}: yielded {len(clash)} face(s) "
                  f"to an earlier scratch")
        if not patch:
            raise RuntimeError(
                f"scratch {i} has no faces left -- it lands on top of scratch "
                f"{claimed[clash[0]]}")
        claimed.update({f: i for f in patch})
        slot = _build_scratch_material(obj, s, obj.data.polygons[patch[0]].material_index)
        _project_patch(obj, patch, s, slot, np)

    usd_out = cfg["usd"]
    bpy.ops.wm.usd_export(
        filepath=usd_out,
        selected_objects_only=False,
        export_materials=True,
        generate_preview_surface=True,
        export_uvmaps=True,
        export_normals=True,
        export_textures_mode="KEEP",   # textures already live next to the USD
        relative_paths=True,           # -> ./textures/scratch_0.png
        convert_orientation=False,     # keep Z-up
        convert_scene_units="METERS",
        meters_per_unit=1.0,
        root_prim_path="/root",
        evaluation_mode="RENDER",
    )
    # The outer stage exports to a staging file and swaps it in once verified;
    # name the file the user will actually get.
    print(f"[scratch]   exported {cfg.get('usd_label', usd_out)}")

    if cfg.get("preview_dir"):
        for i, s in enumerate(cfg["scratches"]):
            _render_preview(s, Path(cfg["preview_dir"]) / f"{cfg['object']}_scratch{i}.png",
                            cfg.get("preview_size") or (900, 900))


def _cut_decal_boundaries(obj, scratches: list, bmesh, np) -> None:
    """Slice the mesh along each decal's four side planes.

    A UV layer holds one coordinate per loop, so a face can carry one decal and
    no more.  These parts have faces big enough to reach two scratches at once
    (square_structure's wall is one triangle that spans 235mm, which two
    placements both landed on).  Cutting along the decal boundaries gives each
    decal its own faces; everything outside is untouched geometry.
    """
    if not scratches:
        return
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    before = len(bm.faces)
    for s in scratches:
        centre = np.asarray(s["center"], dtype=np.float64)
        along = np.asarray(s["direction"], dtype=np.float64)
        normal = np.asarray(s["normal"], dtype=np.float64)
        across = np.cross(normal, along)
        for axis, half in ((along, s["half_len_m"]), (across, s["half_wid_m"])):
            for sign in (1.0, -1.0):
                geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
                bmesh.ops.bisect_plane(
                    bm, geom=geom, dist=1e-9,
                    plane_co=(centre + axis * half * sign).tolist(),
                    plane_no=axis.tolist(),
                )
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    print(f"[scratch]   cut decal boundaries: {before} -> {len(obj.data.polygons)} faces")


def _ensure_base_material(obj, cfg: dict) -> None:
    """Keep whatever the OBJ's .mtl gave us; only invent one if there is none."""
    if obj.data.materials:
        return
    mat = bpy.data.materials.new(f"{cfg['object']}_base")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.9, 0.9, 0.9, 1.0)
    bsdf.inputs["Roughness"].default_value = cfg["roughness"]
    bsdf.inputs["Metallic"].default_value = 0.0
    obj.data.materials.append(mat)


def _author_box_uv(obj, np) -> None:
    """A plain box projection for every loop, so no face carries a degenerate UV.

    Only the scratch patches read a texture, so what this maps to does not
    matter -- but a zero-area UV triangle makes tangent frames undefined, and
    renderers differ on what they do with that.  A box projection is cheap and
    always non-degenerate.
    """
    me = obj.data
    nv = len(me.vertices)
    co = np.empty(nv * 3, np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(nv, 3)

    nl = len(me.loops)
    lv = np.empty(nl, np.int32)
    me.loops.foreach_get("vertex_index", lv)
    p = co[lv]

    npoly = len(me.polygons)
    pn = np.empty(npoly * 3, np.float64)
    me.polygons.foreach_get("normal", pn)
    pn = pn.reshape(npoly, 3)
    ltot = np.empty(npoly, np.int32)
    me.polygons.foreach_get("loop_total", ltot)
    axis = np.repeat(np.abs(pn).argmax(1), ltot)     # dominant normal axis per loop

    span = max(float(np.ptp(co, axis=0).max()), 1e-9)
    uv = np.empty((nl, 2), np.float64)
    for a, (iu, iv) in enumerate(((1, 2), (0, 2), (0, 1))):
        m = axis == a
        uv[m, 0] = p[m, iu] / span
        uv[m, 1] = p[m, iv] / span

    layer = me.uv_layers.get("UVMap") or me.uv_layers.new(name="UVMap")
    layer.data.foreach_set("uv", uv.astype(np.float32).ravel())
    me.update()


def _tri_box_overlap(a, b, c, half, np) -> bool:
    """Akenine-Moller separating-axis test: does a triangle touch a box at the origin?"""
    for i in range(3):                              # the box's own three axes
        lo = min(a[i], b[i], c[i])
        hi = max(a[i], b[i], c[i])
        if lo > half[i] or hi < -half[i]:
            return False

    n = np.cross(b - a, c - a)                      # the triangle's plane
    if abs(float(n @ a)) > float(np.abs(n) @ half):
        return False

    for e in (b - a, c - b, a - c):                 # edge x box-axis cross products
        for i in range(3):
            axis = np.cross(e, np.eye(3)[i])
            if not axis.any():
                continue
            proj = [float(axis @ v) for v in (a, b, c)]
            if min(proj) > float(np.abs(axis) @ half) or max(proj) < -float(np.abs(axis) @ half):
                return False
    return True


def _grow_patch(obj, s: dict, smooth_deg: float, np) -> list:
    """Faces that carry one decal: a flood fill that refuses to cross a crease.

    Growing by distance alone would let a patch wrap around a box corner, where
    a flattening has to tear and the decal comes out smeared (measured: UV/3D
    area ratio spread 0.80-1.70 across an edge, versus 1.00-1.02 within a face).

    The region is the scratch's own box, not the square texture window.  A
    scratch is long and thin (a 29mm one is 2.4mm wide), and everything outside
    it is flat margin that no face needs to sample.  Taking the whole window
    made the patch swallow the entire cylinder wall -- 4551 faces wrapped into a
    ring, which a conformal unwrap cannot flatten (area ratio 0.62-2.26).
    """
    me = obj.data
    centre = np.asarray(s["center"], dtype=np.float64)
    along = np.asarray(s["direction"], dtype=np.float64)
    normal = np.asarray(s["normal"], dtype=np.float64)
    across = np.cross(normal, along)
    half_len, half_wid = s["half_len_m"], s["half_wid_m"]
    limit = math.radians(smooth_deg)

    # Work in the scratch's own frame, where the region is an axis-aligned box.
    basis = np.stack([along, across, normal])
    half = np.array([half_len, half_wid, half_len])
    local = {}
    for p in me.polygons:
        pts = np.asarray([me.vertices[i].co[:] for i in p.vertices], dtype=np.float64)
        local[p.index] = (pts - centre) @ basis.T

    def near(fi):
        # Overlap, not "has a vertex inside".  STEP tessellates this cylinder
        # wall into triangles that run its full 81mm height, so a vertex test
        # rejects every face a 5mm-wide band actually crosses -- which left one
        # scratch sitting on a single face.
        pts = local[fi]
        return any(_tri_box_overlap(pts[0], pts[k], pts[k + 1], half, np)
                   for k in range(1, len(pts) - 1))

    # The seed is the face the centre actually sits on.  Nearest-centroid picks
    # the wrong one where faces are huge (sample's walls are two triangles).
    tiny = np.full(3, 1e-4)
    seed = next((p.index for p in me.polygons
                 if any(_tri_box_overlap(local[p.index][0], local[p.index][k],
                                         local[p.index][k + 1], tiny, np)
                        for k in range(1, len(local[p.index]) - 1))), None)
    if seed is None:
        seed = min(range(len(me.polygons)),
                   key=lambda i: float(np.linalg.norm(local[i], axis=1).min()))

    # edge -> the faces on it, so a flood fill can test the dihedral angle
    edge_faces = {}
    for p in me.polygons:
        for e in p.edge_keys:
            edge_faces.setdefault(e, []).append(p.index)

    face_normal = {p.index: np.asarray(p.normal[:], dtype=np.float64) for p in me.polygons}
    patch, stack = {seed}, [seed]
    while stack:
        fi = stack.pop()
        for e in me.polygons[fi].edge_keys:
            for fj in edge_faces.get(e, ()):
                if fj in patch or not near(fj):
                    continue
                cos = float(np.clip(face_normal[fi] @ face_normal[fj], -1.0, 1.0))
                if math.acos(cos) > limit:
                    continue                    # a crease: the decal stops here
                if float(face_normal[fj] @ normal) < math.cos(math.radians(MAX_TILT_DEG)):
                    continue                    # turned too far off the decal's own plane
                patch.add(fj)
                stack.append(fj)
    return sorted(patch)


def _build_scratch_material(obj, s: dict, base_slot: int) -> int:
    """The simple, USD-safe chain: texture -> Normal Map -> Principled BSDF.

    Built as a copy of the material the patch already wore, so the only thing
    that changes on those faces is the normal map.  A patch can swallow a whole
    wall triangle, and a fresh grey material there would show up as a repaint.
    """
    base = obj.data.materials[base_slot] if obj.data.materials else None
    mat = base.copy() if base else bpy.data.materials.new(s["material"])
    mat.name = s["material"]
    if not mat.use_nodes:
        mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None
                   ) or nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        bsdf.inputs["Base Color"].default_value = (*s["base_color"], 1.0)
        bsdf.inputs["Roughness"].default_value = s["roughness"]
        bsdf.inputs["Metallic"].default_value = 0.0

    img = bpy.data.images.load(s["texture"])
    img.colorspace_settings.name = "Non-Color"   # sRGB here would skew normals
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    # A patch face may reach past the decal window; EXTEND repeats the flat
    # border instead of tiling the scratch across the rest of the wall.
    tex.extension = "EXTEND"
    tex.location = (-600, 0)
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nmap.location = (-300, 0)
    nt.links.new(tex.outputs["Color"], nmap.inputs["Color"])
    nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

    obj.data.materials.append(mat)
    return len(obj.data.materials) - 1


def _project_patch(obj, patch: list, s: dict, slot: int, np) -> None:
    """Put the decal window on the patch by projecting along the scratch normal.

    A conformal unwrap was the obvious tool and the wrong one.  These meshes mix
    a 240mm wall triangle with 2mm rim slivers, and LSCM flattens that pair into
    UV areas 15x apart -- the decal came out smeared.  Projection has no such
    failure: it is exact on a flat face, it degrades gently and predictably on a
    curved one (foreshortening by cos of the tilt, ~6% at the ends of a 30mm
    scratch on this 23mm-radius cylinder), and a face reaching far outside the
    window simply lands outside [0,1], where the clamped texture is flat.
    """
    me = obj.data
    centre = np.asarray(s["center"], dtype=np.float64)
    along = np.asarray(s["direction"], dtype=np.float64)
    normal = np.asarray(s["normal"], dtype=np.float64)
    across = np.cross(normal, along)
    span = s["span_m"]

    layer = me.uv_layers.active
    ratios = []
    for fi in patch:
        p = me.polygons[fi]
        p.material_index = slot
        pts = [np.asarray(me.vertices[i].co[:], dtype=np.float64) for i in p.vertices]
        uvs = []
        for point, li in zip(pts, p.loop_indices):
            d = point - centre
            uv = (float(d @ along) / span + 0.5, float(d @ across) / span + 0.5)
            layer.data[li].uv = uv
            uvs.append(np.asarray(uv))
        for k in range(1, len(pts) - 1):
            t3 = 0.5 * np.linalg.norm(np.cross(pts[k] - pts[0], pts[k + 1] - pts[0]))
            tuv = 0.5 * abs(np.cross(uvs[k] - uvs[0], uvs[k + 1] - uvs[0]))
            if t3 > 1e-14:
                ratios.append(tuv * span ** 2 / t3)
    me.update()

    # The ratio is the foreshortening the projection costs: 1.0 where the
    # surface is parallel to the decal plane, cos(tilt) where it turns away.
    r = np.asarray(ratios)
    print(f"[scratch]   {s['material']}: {len(patch)} faces, "
          f"area ratio median={np.median(r):.3f} p5={np.percentile(r, 5):.3f} "
          f"p95={np.percentile(r, 95):.3f}")


def _render_preview(s: dict, out: Path, size=(900, 900)) -> None:
    """One raking-light Cycles frame, so the groove can be eyeballed offline.

    Head-on light hides a groove entirely -- the shading cue is the shadowed
    wall, so the key light has to come in from the side, and across the
    scratch rather than along it.
    """
    import mathutils

    scene = bpy.context.scene
    centre = mathutils.Vector(s["center"])
    n = mathutils.Vector(s["normal"]).normalized()
    along = mathutils.Vector(s["direction"]).normalized()
    across = n.cross(along).normalized()

    for ob in list(scene.objects):
        if ob.type in {"CAMERA", "LIGHT", "EMPTY"}:
            bpy.data.objects.remove(ob, do_unlink=True)

    def _track(ob):
        """Point -Z at the scratch by setting the rotation outright.

        A TRACK_TO constraint is the usual way and it does nothing here: in
        background mode the constraint never evaluates, so every camera kept
        staring down -Z and all but the top-facing scratches rendered black.
        """
        ob.rotation_euler = (centre - ob.location).to_track_quat("-Z", "Y").to_euler()

    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 50
    # These parts are centimetres across and the camera sits centimetres away,
    # well inside Blender's default 0.1m near clip -- which renders black.
    cam_data.clip_start, cam_data.clip_end = 1e-3, 100.0
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    # Frame the decal window, not the whole part.  A 50mm lens on Blender's
    # 36mm sensor sees ~40 deg, so the window fills the frame at ~1.4x its
    # width; 2x leaves a little context around it.
    # Blender fits the sensor to the longer side, so a wide frame sees *less*
    # vertically -- pull back by the aspect so a scratch running up the frame
    # still fits, and the extra width becomes context around it.
    reach = max(2.0 * s["span_m"], 0.03) * max(1.0, size[0] / max(size[1], 1))
    cam.location = centre + n * reach
    _track(cam)
    scene.camera = cam

    # A sun's irradiance does not fall off with distance, so the exposure is
    # predictable no matter how large the object is.  Grazing across the
    # scratch is what makes the groove readable.
    key_data = bpy.data.lights.new("key", type="SUN")
    key_data.energy, key_data.angle = 2.5, math.radians(3)
    key = bpy.data.objects.new("key", key_data)
    scene.collection.objects.link(key)
    key.location = centre + across * 0.95 + n * 0.32
    _track(key)

    fill_data = bpy.data.lights.new("fill", type="SUN")
    fill_data.energy = 0.35
    fill = bpy.data.objects.new("fill", fill_data)
    scene.collection.objects.link(fill)
    fill.location = centre - across * 0.8 + n * 0.55
    _track(fill)

    scene.render.engine = "CYCLES"        # CPU Cycles always works headless
    scene.cycles.samples = 48
    # square_structure's own material is nearly black, and a groove read by its
    # shadow disappears entirely at that exposure.
    scene.view_settings.exposure = 2.0
    scene.render.resolution_x, scene.render.resolution_y = int(size[0]), int(size[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(out)
    bpy.ops.render.render(write_still=True)
    print(f"[scratch]   preview -> {out}")


# ============================================================================
# stage 1 -- outer process (numpy + PIL + trimesh + pxr)
# ============================================================================
#
# Everything below is a library first and a CLI second: the Viewpoint Studio
# calls these functions directly, and ``main()`` is a thin argparse wrapper.
# Failures raise ``ScratchError`` rather than exiting, so a GUI can show the
# message and carry on; the CLI turns it back into ``sys.exit(message)``.

# What scratches.json records per scratch -- the ground truth and the --spec input.
SPEC_KEYS = ("png", "center", "normal", "direction", "length_mm", "strength")


class ScratchError(RuntimeError):
    """A scratch run that cannot go on.

    ``placed`` keeps whatever a random placement managed before it gave up, so
    a GUI can still show the partial plan.
    """

    def __init__(self, message: str, *, placed: list | None = None):
        super().__init__(message)
        self.placed = list(placed or [])


@dataclass(frozen=True)
class PlanOptions:
    """Where scratches go.  Exactly one of ``count``, ``at_mm``, ``spec`` is set."""

    count: int | None = None                   # random placement
    seed: int = 0
    at_mm: tuple | None = None                 # one scratch here (object frame, mm)
    scratch_png: Path | None = None            # the PNG for ``at_mm``
    angle_deg: float = 0.0
    length_mm: tuple = (20.0, 20.0)            # (lo, hi); equal values = fixed length
    strength: float = 1.0
    smooth_deg: float = 30.0
    spec: Path | None = None                   # replay a scratches.json


@dataclass(frozen=True)
class ApplyOptions:
    """How the planned scratches are stamped into source.usd."""

    tex_size: int = 1024
    smooth_deg: float = 30.0
    roughness: float = 0.5
    preview_dir: Path | None = None
    preview_size: tuple = (900, 900)
    force_backup: bool = False                 # refresh source_prev.usd from source.usd
    blender: Path | None = None                # None -> find_blender()
    timeout_s: float | None = None


@dataclass
class ApplyResult:
    usd: Path
    spec_path: Path
    backup: Path | None
    backup_written: bool
    n_clamped: int
    lines: list = field(default_factory=list)  # the "[scratch]" lines Blender printed


def find_blender(explicit: Path | None = None) -> Path | None:
    """The Blender binary, or None: --blender / $BLENDER / PATH / /usr/local/bin."""
    candidates = [explicit, os.environ.get("BLENDER"), shutil.which("blender"),
                  "/usr/local/bin/blender"]
    for c in candidates:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return Path(c)
    return None


def scratch_library() -> list[Path]:
    """The scratch normal maps random placement picks from (``ff/`` is gitignored)."""
    if not SCRATCH_DIR.is_dir():
        raise ScratchError(f"scratch library not found: {SCRATCH_DIR}")
    library = sorted(SCRATCH_DIR.glob("*.png"))
    if not library:
        raise ScratchError(f"no scratch PNGs under {SCRATCH_DIR}")
    return library


def load_surface(object_name: str, *, data_root: Path = DATA_ROOT):
    """The surface scratches may land on, and the full mesh they get stamped on.

    ``sample`` is an assembly: source.obj carries the fixture too, and only
    target.ply is the part under inspection.  A scratch belongs on the part.
    """
    import trimesh

    mesh_dir = data_root / object_name / "mesh"
    full = trimesh.load(str(mesh_dir / "source.obj"), force="mesh")
    target_path = next((mesh_dir / n for n in ("target.ply", "target.obj")
                        if (mesh_dir / n).exists()), None)
    target = trimesh.load(str(target_path), force="mesh") if target_path else full
    target.merge_vertices()
    return target, target_path


def load_spec(path: Path) -> list[dict]:
    """The scratches a previous run recorded."""
    spec = json.loads(Path(path).read_text())
    return [dict(s) for s in spec["scratches"]]


def spec_record(object_name: str, scratches: list) -> dict:
    return {"object": object_name,
            "scratches": [{k: s[k] for k in SPEC_KEYS} for s in scratches]}


def tangent_frame(normal, np):
    """A deterministic pair of tangents, so --angle-deg means the same thing twice."""
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(normal @ ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    t0 = np.cross(ref, normal)
    t0 /= np.linalg.norm(t0)
    return t0, np.cross(normal, t0)


def _footprint_is_smooth(mesh, centre, basis, half, smooth_deg: float, np) -> bool:
    """True if no crease runs through the scratch's own box.

    A ball around the centre asks far too much: a scratch is thin, so a crease
    10mm off to the side is irrelevant, and demanding a clear ball rejected 55%
    of square_structure's surface.  The box is the scratch itself.
    """
    local = np.abs((mesh.vertices - centre) @ basis.T)
    close = (local <= half).all(axis=1)
    faces = np.flatnonzero(close[mesh.faces].any(axis=1))
    if not len(faces):
        return False
    inside = np.zeros(len(mesh.faces), bool)
    inside[faces] = True
    pair = mesh.face_adjacency
    both = inside[pair[:, 0]] & inside[pair[:, 1]]
    return not bool((mesh.face_adjacency_angles[both] > math.radians(smooth_deg)).any())


def plan_scratches(object_name: str, opts: PlanOptions, *, surface=None,
                   data_root: Path = DATA_ROOT, log=print) -> list:
    """Where the scratches go: from a spec, at a point, or sampled at random.

    ``surface`` is ``load_surface``'s result, so a caller that plans often (the
    studio) does not reload the mesh every time.
    """
    import numpy as np

    if opts.spec:
        return load_spec(opts.spec)

    target, target_path = surface or load_surface(object_name, data_root=data_root)
    lo, hi = float(min(opts.length_mm)), float(max(opts.length_mm))

    if opts.at_mm is not None:
        return [place_at(target, np.asarray(opts.at_mm, dtype=np.float64) / 1000.0,
                         png=opts.scratch_png, angle_deg=opts.angle_deg,
                         length_mm=lo, strength=opts.strength)]

    return place_random(object_name, target, target_path, count=opts.count, seed=opts.seed,
                        length_mm=(lo, hi), strength=opts.strength,
                        smooth_deg=opts.smooth_deg, data_root=data_root, log=log)


def place_at(target, point_m, *, png, angle_deg: float, length_mm: float,
             strength: float) -> dict:
    """One scratch at the surface point nearest ``point_m`` (object frame, metres)."""
    import numpy as np
    import trimesh

    closest, _, face = trimesh.proximity.closest_point(target, [point_m])
    centre, normal = closest[0], target.face_normals[face[0]]
    t0, t1 = tangent_frame(normal, np)
    a = math.radians(angle_deg)
    return _placement(png, centre, normal, math.cos(a) * t0 + math.sin(a) * t1,
                      length_mm, strength, np)


def place_random(object_name: str, target, target_path, *, count: int, seed: int,
                 length_mm: tuple, strength: float, smooth_deg: float,
                 data_root: Path = DATA_ROOT, log=print) -> list:
    """``count`` scratches at random on ``target``, reproducible from ``seed``.

    The random draws happen in a fixed order -- keep it that way, or the same
    seed stops giving the same scratches (recorded scratches.json replay fine
    either way; it is the seed that would silently change meaning).
    """
    import numpy as np
    import trimesh

    library = scratch_library()
    lo, hi = length_mm

    rng = np.random.default_rng(seed)
    log(f"[scratch] placing {count} scratch(es) on "
        f"{(target_path or (data_root / object_name / 'mesh' / 'source.obj')).name} "
        f"({target.area * 1e4:.1f}cm2), seed={seed}")

    placed, tries = [], 0
    rejected = {"downward": 0, "crease": 0, "off the edge": 0, "too close": 0}
    while len(placed) < count and tries < 4000:
        tries += 1
        points, faces = trimesh.sample.sample_surface(target, 1, seed=int(rng.integers(1 << 31)))
        centre, normal = points[0], target.face_normals[faces[0]]
        # Same rule as the viewpoint bottom filter: a downward face is not
        # something the camera inspects, so do not put a defect there.
        if float(normal[2]) < -math.cos(math.radians(80.0)):
            rejected["downward"] += 1
            continue
        length = float(rng.uniform(lo, hi))
        # Far enough apart that the scratches themselves do not overlap.  The
        # decal *windows* may overlap -- they are mostly flat margin.
        if any(np.linalg.norm(centre - np.asarray(p["center"]))
               < 0.6 * (length + p["length_mm"]) / 1000.0 for p in placed):
            rejected["too close"] += 1
            continue
        t0, t1 = tangent_frame(normal, np)
        a = float(rng.uniform(0.0, 2 * math.pi))
        direction = math.cos(a) * t0 + math.sin(a) * t1
        half = np.array([length / 1000.0 * SMOOTH_RADIUS_FACTOR,
                         max(length / 1000.0 * 0.06, 0.002),
                         length / 1000.0 * SMOOTH_RADIUS_FACTOR])
        if not _footprint_is_smooth(target, centre,
                                    np.stack([direction, np.cross(normal, direction), normal]),
                                    half, smooth_deg, np):
            rejected["crease"] += 1
            continue
        # Both ends have to land on the part.  Without this a scratch sampled
        # near a rim hangs half of itself off the edge into thin air.  The test
        # is a ray inward from each end, not a distance to the surface: on a
        # curved wall the tangent-plane end legitimately floats above it (5.6mm
        # on this 23mm-radius cylinder), but it still has material beneath it.
        ends = np.array([centre + direction * (length / 2000.0),
                         centre - direction * (length / 2000.0)]) + normal * 1e-3
        hit, ray_id = target.ray.intersects_location(
            ends, np.repeat(-normal[None], 2, axis=0), multiple_hits=False)[:2]
        if len(set(ray_id.tolist())) < 2 or (
                np.linalg.norm(hit - ends[ray_id], axis=1) > length / 1000.0).any():
            rejected["off the edge"] += 1
            continue
        png = library[int(rng.integers(len(library)))]
        placed.append(_placement(png, centre, normal,
                                 math.cos(a) * t0 + math.sin(a) * t1,
                                 length, strength, np))
    if len(placed) < count:
        why = ", ".join(f"{k} {v}" for k, v in rejected.items())
        raise ScratchError(f"only placed {len(placed)}/{count} scratches in {tries} tries "
                           f"(rejected: {why}) -- try a shorter --length-mm", placed=placed)
    return placed


def _placement(png, centre, normal, direction, length_mm: float, strength: float, np) -> dict:
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction - normal * float(normal @ direction)     # keep it tangent
    direction = direction / np.linalg.norm(direction)
    return {
        "png": str(Path(png)),
        "center": [float(x) for x in centre],
        "normal": [float(x) for x in normal],
        "direction": [float(x) for x in direction],
        "length_mm": float(length_mm),
        "strength": float(strength),
    }


def composite_normal_array(scratch: Path, *, size: int, length_px: float, strength: float):
    """The decal texture as arrays: ``(rgb uint8 (size,size,3), alpha (size,size), stamp_wh)``.

    One scratch per texture, centred, long axis along +u.  The border stays
    flat, which is what the clamped wrap reads for faces that reach past the
    decal window.  ``alpha`` is the scratch's own coverage -- the studio uses it
    to shade a preview decal.
    """
    import numpy as np
    from PIL import Image

    src = Image.open(scratch).convert("RGBA")
    a_full = np.asarray(src)[:, :, 3]
    ys, xs = np.nonzero(a_full > 8)
    if len(xs) == 0:
        raise ScratchError(f"{scratch}: fully transparent, nothing to stamp")
    src = src.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    if src.height > src.width:      # the long axis is the scratch's own axis
        src = src.transpose(Image.ROTATE_90)

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
    y0 = int(round(size / 2 - h / 2))
    x0 = int(round(size / 2 - w / 2))
    row = np.arange(y0, y0 + h)
    col = np.arange(x0, x0 + w)
    keep_r = (row >= 0) & (row < size)
    keep_c = (col >= 0) & (col < size)
    idx = np.ix_(row[keep_r], col[keep_c])
    layer[idx] = sn[np.ix_(keep_r, keep_c)]
    alpha[idx] = sa[np.ix_(keep_r, keep_c)]

    flat = np.zeros((size, size, 3), np.float32)
    flat[:, :, 2] = 1.0
    out_n = flat * (1.0 - alpha) + layer * alpha
    out_n /= np.maximum(np.linalg.norm(out_n, axis=2, keepdims=True), 1e-8)

    rgb = np.clip(np.rint((out_n * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)
    return rgb, alpha[:, :, 0], dst


def composite_normal_map(scratch: Path, out: Path, *, size: int,
                         length_px: float, strength: float) -> tuple[int, int]:
    """``composite_normal_array`` written to ``out`` as the decal's normal map."""
    from PIL import Image

    rgb, _, dst = composite_normal_array(scratch, size=size, length_px=length_px,
                                         strength=strength)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, "RGB").save(out)
    return dst


def prepare_scratches(object_name: str, scratches: list, *, tex_size: int, roughness: float,
                      data_root: Path = DATA_ROOT, log=print) -> list:
    """Write each scratch's texture and return copies carrying what Blender needs.

    The inputs are left alone -- they are the spec (``SPEC_KEYS``) a caller may
    keep showing or save again.
    """
    mesh_dir = data_root / object_name / "mesh"
    out = []
    for i, src in enumerate(scratches):
        s = dict(src)
        if not Path(s["png"]).exists():
            raise ScratchError(f"scratch PNG not found: {s['png']}")
        s["span_m"] = s["length_mm"] / 1000.0 * SPAN_FACTOR
        s["texture"] = str(mesh_dir / "textures" / f"scratch_{i}.png")
        s["material"] = f"{object_name}_scratch{i}"
        s["base_color"] = [0.9, 0.9, 0.9]
        s["roughness"] = roughness
        dst = composite_normal_map(
            Path(s["png"]), Path(s["texture"]), size=tex_size,
            length_px=tex_size / SPAN_FACTOR, strength=s["strength"])
        # How much surface the decal actually needs: the stamp's own box in
        # metres, plus a margin.  The rest of the window is flat.
        stamp_w = dst[0] / tex_size * s["span_m"]
        stamp_h = dst[1] / tex_size * s["span_m"]
        s["half_len_m"] = stamp_w * 0.5 * 1.25
        s["half_wid_m"] = max(stamp_h * 0.5 * 1.5, stamp_w * 0.06)
        c = s["center"]
        log(f"[scratch] {i}: {Path(s['png']).name} {s['length_mm']:.0f}mm at "
            f"({c[0] * 1000:.0f}, {c[1] * 1000:.0f}, {c[2] * 1000:.0f})mm "
            f"-> {Path(s['texture']).name} ({tex_size}^2, stamp {dst[0]}x{dst[1]}px "
            f"in a {s['span_m'] * 1000:.0f}mm window)")
        out.append(s)
    return out


def run_blender(cfg: dict, *, blender: Path, on_line=print, timeout_s: float | None = None) -> list:
    """Run the Blender stage, passing its ``[scratch]`` lines to ``on_line`` as they come.

    stdout and stderr stay apart: only stdout is filtered for progress, exactly
    as before, and stderr (Blender warnings, the traceback ``blender_stage``
    prints) only surfaces in the failure message.
    """
    lines, tail = [], []
    with tempfile.TemporaryFile(mode="w+") as err:
        proc = subprocess.Popen(
            [str(blender), "-b", "--factory-startup", "--python", str(Path(__file__).resolve()),
             "--", json.dumps(cfg)],
            cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1)
        timer = None
        timed_out = threading.Event()
        if timeout_s is not None:
            def _kill():
                timed_out.set()
                proc.kill()
            timer = threading.Timer(timeout_s, _kill)
            timer.start()
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                tail.append(line)
                if len(tail) > 400:
                    del tail[:200]
                if line.startswith("[scratch]") or "Error" in line or "Traceback" in line:
                    lines.append(line)
                    on_line(line)
            rc = proc.wait()
        finally:
            if timer is not None:
                timer.cancel()
        if timed_out.is_set():
            raise ScratchError(f"blender stage timed out after {timeout_s:g}s")
        if rc != 0:
            err.seek(0)
            raise ScratchError(f"blender stage failed (rc={rc}):\n"
                               f"{chr(10).join(tail)[-4000:]}\n{err.read()[-2000:]}")
    return lines


def obj_bbox(obj_path: Path) -> tuple[list, list]:
    """The OBJ's own bounds, read from the file the Blender stage imports."""
    import numpy as np

    verts = [line.split()[1:4] for line in obj_path.read_text().splitlines()
             if line.startswith("v ")]
    v = np.asarray(verts, dtype=np.float64)
    return v.min(0).tolist(), v.max(0).tolist()


def clamp_texture_wrap(usd: Path) -> int:
    """Force wrapS/wrapT to clamp on every normal texture.

    A patch face may reach past the decal window.  With the USD default (black
    or repeat, depending on the renderer) that shows up as a tiled scratch or a
    black wall; clamped, it reads the flat border and stays invisible.
    """
    from pxr import Sdf, Usd, UsdShade

    stage = Usd.Stage.Open(str(usd))
    n = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Shader":
            continue
        shader = UsdShade.Shader(prim)
        if shader.GetIdAttr().Get() != "UsdUVTexture":
            continue
        for key in ("wrapS", "wrapT"):
            shader.CreateInput(key, Sdf.ValueTypeNames.Token).Set("clamp")
        n += 1
    stage.GetRootLayer().Save()
    return n


def _check(ok, message: str) -> None:
    """``assert`` that survives ``python -O`` and reads as a ScratchError."""
    if not ok:
        raise ScratchError(message)


def verify_usd(usd: Path, expect_bbox: tuple[list, list], n_scratches: int, *,
               label: Path | None = None, log=print) -> None:
    """Fail loudly on the failure modes that are invisible until Isaac loads.

    ``label`` is the path to report -- the file being checked may be a staging
    copy that is about to replace it.
    """
    from pxr import Gf, Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(str(usd))
    _check(UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z, "upAxis is not Z")
    _check(UsdGeom.GetStageMetersPerUnit(stage) == 1.0, "metersPerUnit is not 1")

    mesh = next((p for p in stage.Traverse() if p.GetTypeName() == "Mesh"), None)
    _check(mesh is not None, "no Mesh prim in exported USD")
    g = UsdGeom.Mesh(mesh)

    ext = g.GetExtentAttr().Get()
    lo, hi = expect_bbox
    for got, want, tag in ((ext[0], lo, "min"), (ext[1], hi, "max")):
        _check(Gf.IsClose(Gf.Vec3f(*got), Gf.Vec3f(*[float(x) for x in want]), 1e-4),
               f"extent {tag} {tuple(got)} != OBJ bbox {tuple(want)} "
               f"-- axis or scale went wrong on import")
    names = [pv.GetName() for pv in UsdGeom.PrimvarsAPI(mesh).GetPrimvars()]
    _check("primvars:st" in names, f"no UV primvar exported (got {names})")
    _check(g.GetNormalsAttr().Get(), "no normals exported")

    surfaces = [UsdShade.Shader(p) for p in stage.Traverse()
                if p.GetTypeName() == "Shader"
                and UsdShade.Shader(p).GetIdAttr().Get() == "UsdPreviewSurface"]
    _check(surfaces, "no UsdPreviewSurface shader")

    textured = 0
    for surf in surfaces:
        nrm = surf.GetInput("normal")
        if not (nrm and nrm.HasConnectedSource()):
            continue                      # the base material carries no normal map
        tex = UsdShade.Shader(nrm.GetConnectedSource()[0].GetPrim())
        _check(tex.GetIdAttr().Get() == "UsdUVTexture", "normal source is not a texture")

        cs = tex.GetInput("sourceColorSpace").Get()
        _check(cs == "raw", f"texture colorspace is {cs!r}, expected 'raw'")
        for key, want in (("scale", (2, 2, 2, 2)), ("bias", (-1, -1, -1, -1))):
            got = tex.GetInput(key).Get()
            _check(got and tuple(got) == want, f"texture {key}={got}, expected {want}")
        for key in ("wrapS", "wrapT"):
            got = tex.GetInput(key).Get()
            _check(got == "clamp", f"texture {key}={got!r}, expected 'clamp'")

        asset = tex.GetInput("file").Get()
        _check(asset and Path(asset.resolvedPath).exists(),
               f"texture path does not resolve: {asset}")
        st = tex.GetInput("st")
        _check(st and st.HasConnectedSource(), "texture st input is not connected")
        textured += 1

    _check(textured == n_scratches,
           f"{textured} materials carry a normal map, expected {n_scratches}")
    log(f"[scratch] verified {label or usd}")
    log(f"[scratch]   extent {tuple(ext[0])} .. {tuple(ext[1])}")
    log(f"[scratch]   {textured} scratch material(s), UV primvar and normals present")


def stamp_provenance(usd: Path, params: dict) -> None:
    """ff/ is gitignored, so record what produced this USD inside the USD."""
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd))
    prim = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    data = dict(prim.GetCustomData())
    data["scratchNormal"] = {k: str(v) for k, v in params.items()}
    prim.SetCustomData(data)
    stage.GetRootLayer().Save()


def apply_scratches(object_name: str, scratches: list, opts: ApplyOptions | None = None, *,
                    data_root: Path = DATA_ROOT, log=print, on_line=None) -> ApplyResult:
    """Stamp ``scratches`` into ``data/{object}/mesh/source.usd``.

    Blender writes a staging file next to source.usd (so relative texture paths
    stay the same); it is verified and only then swapped in.  A failed or
    killed run leaves source.usd as it was -- before, Blender overwrote it in
    place and a failed check left a broken file behind.
    """
    opts = opts or ApplyOptions()
    on_line = on_line or log
    mesh_dir = data_root / object_name / "mesh"
    obj_path = mesh_dir / "source.obj"
    usd_path = mesh_dir / "source.usd"
    prev_path = mesh_dir / "source_prev.usd"
    spec_path = mesh_dir / "scratches.json"
    staging = mesh_dir / "source.staging.usd"

    if not obj_path.exists():
        raise ScratchError(f"source.obj not found: {obj_path}")
    blender = opts.blender or find_blender()
    if blender is None or not Path(blender).exists():
        raise ScratchError(f"blender not found: {blender} (pass --blender)")
    if not scratches:
        raise ScratchError("no scratches to apply")

    prepared = prepare_scratches(object_name, scratches, tex_size=opts.tex_size,
                                 roughness=opts.roughness, data_root=data_root, log=log)

    backup_written = False
    if usd_path.exists() and (opts.force_backup or not prev_path.exists()):
        shutil.copy2(usd_path, prev_path)
        backup_written = True
        log(f"[scratch] backed up -> {prev_path.relative_to(PROJECT_ROOT)}")

    cfg = {
        "object": object_name, "obj": str(obj_path),
        "usd": str(staging), "usd_label": str(usd_path),
        "smooth_deg": opts.smooth_deg, "roughness": opts.roughness,
        "scratches": prepared,
        "preview_dir": str(opts.preview_dir) if opts.preview_dir else None,
        "preview_size": [int(v) for v in opts.preview_size],
    }
    try:
        lines = run_blender(cfg, blender=Path(blender), on_line=on_line,
                            timeout_s=opts.timeout_s)
        n_clamped = clamp_texture_wrap(staging)
        log(f"[scratch] wrap=clamp on {n_clamped} texture(s)")
        verify_usd(staging, obj_bbox(obj_path), len(prepared), label=usd_path, log=log)
        stamp_provenance(staging, {"spec": spec_path.name, "count": len(prepared),
                                   "texSize": opts.tex_size})
        os.replace(staging, usd_path)
    finally:
        if staging.exists():
            staging.unlink()

    # The ground truth of where the defects are -- and the input that replays
    # this exact run with --spec.  Written only once the USD it describes is in place.
    spec_path.write_text(json.dumps(spec_record(object_name, prepared), indent=2) + "\n")
    log(f"[scratch] recorded -> {spec_path.relative_to(PROJECT_ROOT)}")
    log(f"[scratch] done -> {usd_path.relative_to(PROJECT_ROOT)}")
    return ApplyResult(usd=usd_path, spec_path=spec_path,
                       backup=prev_path if prev_path.exists() else None,
                       backup_written=backup_written, n_clamped=n_clamped, lines=lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Stamp scratch normal maps onto data/{object}/mesh/source.usd")
    p.add_argument("--object", required=True, help="Object name (e.g. cylinder_sample)")
    p.add_argument("--scratch", type=Path,
                   help="Scratch PNG for --at (tangent-space normal map with alpha)")
    p.add_argument("--random", type=int, metavar="N",
                   help="Place N scratches at random on the inspected surface")
    p.add_argument("--seed", type=int, default=0, help="Seed for --random (default 0)")
    p.add_argument("--spec", type=Path,
                   help="Replay a scratches.json written by an earlier run")
    p.add_argument("--at", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   help="Place one scratch here (object frame, mm; snapped to the surface)")
    p.add_argument("--angle-deg", type=float, default=0.0,
                   help="Direction in the tangent plane for --at (default 0)")
    p.add_argument("--length-mm", type=float, nargs="+", default=[20.0],
                   help="Scratch length in mm; two values = a random range (default 20)")
    p.add_argument("--strength", type=float, default=1.0,
                   help="Groove depth, 1.0 = source normals unchanged (default 1.0)")
    p.add_argument("--tex-size", type=int, default=1024,
                   help="Pixels per scratch texture (default 1024)")
    p.add_argument("--smooth-deg", type=float, default=30.0,
                   help="Above this dihedral angle an edge is a crease (default 30)")
    p.add_argument("--roughness", type=float, default=0.5)
    p.add_argument("--preview-dir", type=Path,
                   help="Also render one raking-light preview PNG per scratch")
    p.add_argument("--preview-size", default="900x900", metavar="WxH",
                   help="Preview resolution, e.g. 1600x800 (default 900x900)")
    p.add_argument("--force", action="store_true",
                   help="Refresh source_prev.usd from the current source.usd")
    p.add_argument("--blender", type=Path,
                   default=Path(shutil.which("blender") or "/usr/local/bin/blender"))
    args = p.parse_args()

    if sum(x is not None for x in (args.random, args.at, args.spec)) != 1:
        sys.exit("pick exactly one of --random N, --at X Y Z, --spec FILE")
    if args.at is not None and not args.scratch:
        sys.exit("--at needs --scratch PNG")
    if args.scratch and not args.scratch.exists():
        sys.exit(f"scratch PNG not found: {args.scratch}")

    obj_path = DATA_ROOT / args.object / "mesh" / "source.obj"
    if not obj_path.exists():
        sys.exit(f"source.obj not found: {obj_path}")
    if not args.blender.exists():
        sys.exit(f"blender not found: {args.blender} (pass --blender)")

    plan = PlanOptions(
        count=args.random, seed=args.seed,
        at_mm=tuple(args.at) if args.at is not None else None, scratch_png=args.scratch,
        angle_deg=args.angle_deg,
        length_mm=(float(min(args.length_mm)), float(max(args.length_mm))),
        strength=args.strength, smooth_deg=args.smooth_deg, spec=args.spec)
    apply = ApplyOptions(
        tex_size=args.tex_size, smooth_deg=args.smooth_deg, roughness=args.roughness,
        preview_dir=args.preview_dir,
        preview_size=tuple(int(v) for v in str(args.preview_size).lower().split("x")),
        force_backup=args.force, blender=args.blender)
    try:
        scratches = plan_scratches(args.object, plan)
        apply_scratches(args.object, scratches, apply)
    except ScratchError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    if INSIDE_BLENDER:
        blender_stage(json.loads(sys.argv[sys.argv.index("--") + 1]))
    else:
        main()
