#!/usr/bin/env python3
"""Bake a new camera CAD mesh into the mount frame -> camera.usdc + camera_body.obj.

The camera body is authored **pre-baked in the camera_mount frame** (meters): the USD
mesh carries no xformOps and the URDF attaches it to `camera_link` with an identity
visual/collision origin. This script is what enforces that convention when the CAD changes.

`camera_mount` (= URDF `camera_link`) is the flange rotated by CAMERA_MOUNT_RPY_DEG --
the clocking the camera is actually bolted on at. Until 2026-08-22 that was identity, so
"mount frame" and "flange frame" were the same thing and the old naming said flange.

Pipeline:
    camera.obj  (Blender export, meters, optical axis +Z; supplied with --source)
      -> rotate by R so the optical axis becomes mount +X
      -> quadric-decimate (open3d; trimesh's backend `fast_simplification` is absent)
      -> assert the bbox landed on the mount-frame convention
      -> write BOTH camera.usdc and camera_body.obj from the same float32 arrays

Both outputs come from one in-memory geometry on purpose: camera_body.obj is not an
intermediate, it is the URDF visual+collision mesh, the MorphIt sphere-fitting input and
what the viser IK tools render (ik_backend.RobotViz, via trajectory_studio). Generating
them separately guarantees they drift apart.

`camera.usdc` is overwritten in place -- ur20_with_camera.usd references the literal string
`./camera/camera.usdc`, so the filename is load-bearing. Commit 0c85601 is the precedent for
what happens when that reference breaks: the camera silently composes to an empty Xform.
Hence --verify, which asserts against the *composed* robot stage rather than the file we wrote.

Usage:
    uv run --no-sync scripts/setup/build_camera_mesh.py --source /path/to/camera.obj
    uv run --no-sync scripts/setup/build_camera_mesh.py --source /path/to/camera.obj --dry-run
    uv run --no-sync scripts/setup/build_camera_mesh.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAMERA_DIR = PROJECT_ROOT / "workcell" / "robot" / "camera"

OUT_USDC = CAMERA_DIR / "camera.usdc"
OUT_OBJ = CAMERA_DIR / "camera_body.obj"

ROBOT_USD = PROJECT_ROOT / "workcell" / "robot" / "ur20_with_camera.usd"
GHOST_USD = PROJECT_ROOT / "workcell" / "robot" / "ur20_with_camera_ghost.usd"

CUROBO_URDF = PROJECT_ROOT / "workcell" / "robot" / "ur20_with_camera_curobo.urdf"

FLANGE_PRIM = "/Root/UR20/wrist_3_link/flange"
MOUNT_PRIM = f"{FLANGE_PRIM}/camera_mount"
CAMERA_BODY_PRIM = f"{MOUNT_PRIM}/camera_body"
OPTICAL_PRIM = f"{MOUNT_PRIM}/camera_optical_frame"

# 카메라가 플랜지에 물린 clocking (deg). USD `camera_mount` 의 xformOp:rotateXYZ 와
# URDF `camera_mount_joint` 의 rpy 가 이 값의 두 사본이고, verify 가 셋을 맞춰본다.
# roll 만 쓴다: 툴 축(flange +X = 광축) 둘레라 광축 방향도 optical_frame 원점(0.141,0,0)도
# 움직이지 않는다 -- 몸체가 어느 쪽으로 뻗는지와 이미지 회전만 바뀐다. 실기 대조로 확정.
CAMERA_MOUNT_RPY_DEG = (-90.0, 0.0, 0.0)

# new-mesh (optical axis +Z) -> mount frame (optical axis +X). det = +1; no translation.
#   old_x = new_z ,  old_y = -new_x ,  old_z = -new_y      == USD rotateXYZ(-90, 0, -90)
# The sign pair is pinned by det: the (+new_y) variant is a mirror (det = -1).
R_NEW_TO_MOUNT = np.array([[0, 0, 1],
                            [-1, 0, 0],
                            [0, -1, 0]], dtype=np.float64)

# Mount-frame bbox the baked mesh must land on. x_max is the lens-barrel tip.
# x 는 툴 축이라 clocking(roll) 이 바뀌어도 그대로다. y/z 는 mount 프레임 기준이다.
EXPECT_LO = np.array([-0.001, -0.0949, -0.056])
EXPECT_HI = np.array([0.21877, 0.05, 0.056])
BBOX_TOL = 1e-4

# camera_optical_frame sits on the camera body's front face -- the datum the vendor measures
# their 250 mm working distance from. CAD (camera_asm_wo_light.stp, flange frame, mm):
#   flange 0 | PCB_2 122.7..124.3 | body_face 141.0 | lens barrel 157.5..218.770
#   | VIEW_1 (50x50 inspection plane) 391.0..392.0
# 391.0 - 141.0 = 250.0 exactly, so CAMERA_WORKING_DISTANCE_MM is now the vendor WD itself.
# Until 2026-07-27 this was 0.346 = the front face of LIGHT_1, an illumination box that was
# removed from the assembly; that left the frame floating 127 mm past the lens.
OPTICAL_FRAME_X = 0.141

DEFAULT_FACES = 130_000
CREASE_ANGLE_DEG = 30.0

# Single flat material, carried over from the CAD (MTL `color_adb5bd`, Kd 0.678431/0.709804/0.741176).
MATERIAL_NAME = "color_adb5bd_001"
SHADER_NAME = "Principled_BSDF"
DIFFUSE_COLOR = (0.678431, 0.709804, 0.741176)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_rotate(src: Path) -> trimesh.Trimesh:
    """Load the OBJ and bake the mount-frame rotation into the vertices."""
    raw = trimesh.load(src, process=False, force="mesh")
    # `f v//vn` splits every corner into its own vertex; process=True welds them back.
    mesh = trimesh.Trimesh(raw.vertices @ R_NEW_TO_MOUNT.T, raw.faces, process=True)
    print(f"  loaded {src.name}: {len(raw.vertices)} raw v -> {len(mesh.vertices)} welded v, "
          f"{len(mesh.faces)} f")
    return mesh


def decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Quadric decimation via open3d (trimesh's fast_simplification backend is not installed).

    boundary_weight pins the open boundaries -- this mesh is not watertight.
    """
    om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                   o3d.utility.Vector3iVector(mesh.faces))
    om = om.simplify_quadric_decimation(target_number_of_triangles=target_faces,
                                        boundary_weight=1000.0)
    dec = trimesh.Trimesh(np.asarray(om.vertices), np.asarray(om.triangles), process=False)
    print(f"  decimated {len(mesh.faces)} -> {len(dec.faces)} f, {len(dec.vertices)} v "
          f"(area {100 * dec.area / mesh.area:.2f}% of source)")
    return dec


def assert_mount_bbox(points: np.ndarray, what: str) -> None:
    lo, hi = points.min(axis=0), points.max(axis=0)
    np.testing.assert_allclose(lo, EXPECT_LO, atol=BBOX_TOL, err_msg=f"{what}: bbox min")
    np.testing.assert_allclose(hi, EXPECT_HI, atol=BBOX_TOL, err_msg=f"{what}: bbox max")
    print(f"  {what}: bbox OK  x[{lo[0]:+.5f}, {hi[0]:+.5f}]  "
          f"y[{lo[1]:+.5f}, {hi[1]:+.5f}]  z[{lo[2]:+.5f}, {hi[2]:+.5f}]")


def crease_normals(mesh: trimesh.Trimesh, angle_deg: float = CREASE_ANGLE_DEG) -> np.ndarray:
    """faceVarying normals: per corner, average incident face normals within the crease angle.

    Plain vertex-normal averaging smears the hard rim where the body meets the lens barrel;
    open3d's compute_vertex_normals() has the same problem. Flat per-face normals make
    decimation's irregular triangles read as noise on the barrel.
    """
    faces, fn, fa = mesh.faces, mesh.face_normals, mesh.area_faces
    vf = mesh.vertex_faces  # (V, maxdeg), -1 padded
    cos_thr = np.cos(np.radians(angle_deg))

    acc = np.zeros((len(faces), 3, 3), dtype=np.float64)
    for corner in range(3):
        vs = faces[:, corner]                       # (F,) vertex index of this corner
        for d in range(vf.shape[1]):                # walk the padded adjacency column-wise
            g = vf[vs, d]                           # (F,) neighbouring face, -1 if padding
            keep = (g >= 0) & ((fn[g] * fn).sum(axis=1) > cos_thr)
            acc[:, corner, :] += fn[g] * (fa[g] * keep)[:, None]

    n = acc.reshape(-1, 3)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    np.divide(n, norm, out=n, where=norm > 0)
    return n


def write_obj(path: Path, verts32: np.ndarray, faces: np.ndarray, src: Path, n_src_faces: int) -> None:
    """Match the old file's format exactly: no vn/vt/mtllib/usemtl, just `v` and `f a b c`.

    yourdfpy and MorphIt need neither normals nor materials; the URDF supplies `camera_grey`.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(f"# camera body mesh (camera_asm_wo_light) baked in mount frame (meters), "
                f"decimated {n_src_faces} -> {len(faces)} tris\n")
        source_label = os.path.relpath(src.resolve(), PROJECT_ROOT.resolve())
        f.write(f"# source: {source_label} (untracked)  sha256 {sha256(src)}\n")
        f.write(f"# generated by scripts/setup/{Path(__file__).name}\n")
        np.savetxt(f, verts32, fmt="v %.6f %.6f %.6f")
        np.savetxt(f, faces + 1, fmt="f %d %d %d")
    os.replace(tmp, path)
    print(f"  wrote {path.relative_to(PROJECT_ROOT)} ({path.stat().st_size / 1e6:.1f} MB)")


def write_usdc(path: Path, verts32: np.ndarray, faces: np.ndarray, normals_fv: np.ndarray) -> None:
    """Mirror the structure of the file being replaced: /root defaultPrim, meters, Z-up, no xformOps."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/root")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Xform.Define(stage, "/root/camera_body_geom")
    mesh = UsdGeom.Mesh.Define(stage, "/root/camera_body_geom/camera_body_mesh")

    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts32))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(
        np.full(len(faces), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(
        faces.astype(np.int32).ravel()))
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, verts32.min(axis=0))),
                                         Gf.Vec3f(*map(float, verts32.max(axis=0)))]))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals_fv.astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)

    UsdGeom.Scope.Define(stage, "/root/_materials")
    material = UsdShade.Material.Define(stage, f"/root/_materials/{MATERIAL_NAME}")
    shader = UsdShade.Shader.Define(stage, f"/root/_materials/{MATERIAL_NAME}/{SHADER_NAME}")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*DIFFUSE_COLOR))
    for name, value in (("roughness", 1.0), ("metallic", 0.0), ("specular", 0.0),
                        ("opacity", 1.0), ("ior", 1.5),
                        ("clearcoat", 0.0), ("clearcoatRoughness", 0.03)):
        shader.CreateInput(name, Sdf.ValueTypeNames.Float).Set(value)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    tmp = path.with_suffix(".tmp.usdc")  # .usdc extension selects the crate format
    stage.GetRootLayer().Export(str(tmp))
    os.replace(tmp, path)
    print(f"  wrote {path.relative_to(PROJECT_ROOT)} ({path.stat().st_size / 1e6:.1f} MB)")


def usd_mount_rpy_deg(stage) -> np.ndarray:
    """USD `camera_mount` 의 xformOp:rotateXYZ (deg)."""
    prim = stage.GetPrimAtPath(MOUNT_PRIM)
    assert prim and prim.IsValid(), f"{MOUNT_PRIM} missing"
    ops = {op.GetOpName(): op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()}
    op = ops.get("xformOp:rotateXYZ")
    assert op is not None, f"{MOUNT_PRIM}: xformOp:rotateXYZ missing"
    return np.asarray(op.Get(), dtype=np.float64)


def urdf_mount_rpy_deg() -> np.ndarray:
    """cuRobo URDF `camera_mount_joint` 의 origin rpy (deg)."""
    import xml.etree.ElementTree as ET

    root = ET.parse(CUROBO_URDF).getroot()
    joint = next((j for j in root.findall("joint")
                  if j.get("name") == "camera_mount_joint"), None)
    assert joint is not None, f"{CUROBO_URDF.name}: camera_mount_joint missing"
    # NB: `joint.find(...) or {}` 는 쓸 수 없다 -- 자식 없는 Element 는 falsy 라
    #     origin 을 찾아놓고도 조용히 버리고 "0 0 0" 을 읽는다.
    origin = joint.find("origin")
    rpy = "0 0 0" if origin is None else origin.get("rpy", "0 0 0")
    return np.rad2deg(np.asarray([float(v) for v in rpy.split()], dtype=np.float64))


def verify_composed(usd_path: Path) -> None:
    """Assert against the *composed* robot stage. This is the 0c85601 guard.

    A broken reference yields an empty Xform, not an error -- the camera just vanishes.
    Assert on raw points, not BBoxCache: ComputeRelativeBound pads by ~6e-5 m.
    """
    stage = Usd.Stage.Open(str(usd_path), load=Usd.Stage.LoadAll)
    body = stage.GetPrimAtPath(CAMERA_BODY_PRIM)
    assert body and body.IsValid(), f"{usd_path.name}: {CAMERA_BODY_PRIM} missing"

    meshes = [p for p in Usd.PrimRange(body) if p.IsA(UsdGeom.Mesh)]
    assert len(meshes) == 1, f"{usd_path.name}: expected 1 Mesh under camera_body, got {len(meshes)}"

    mesh = UsdGeom.Mesh(meshes[0])
    pts = np.asarray(mesh.GetPointsAttr().Get())
    idx = mesh.GetFaceVertexIndicesAttr().Get()
    assert len(pts) and idx, f"{usd_path.name}: camera_body mesh is empty"
    assert_mount_bbox(pts, f"{usd_path.name} composed mesh")

    # The mesh is pre-baked, so mesh-local == camera_mount-local exactly.
    xc = UsdGeom.XformCache()
    rel = (xc.GetLocalToWorldTransform(meshes[0])
           * xc.GetLocalToWorldTransform(stage.GetPrimAtPath(MOUNT_PRIM)).GetInverse())
    assert np.allclose(np.asarray(rel), np.eye(4), atol=1e-9), \
        f"{usd_path.name}: mesh->camera_mount is not identity:\n{np.asarray(rel)}"

    # clocking 은 이 스테이지와 URDF 두 곳에 산다. 한쪽만 고치면 계획(cuRobo/URDF)과
    # 화면(USD)이 서로 다른 로봇이 되는데, 어느 쪽도 에러를 내지 않는다 -- 그래서 여기서 맞춰본다.
    mount_rot = usd_mount_rpy_deg(stage)
    assert np.allclose(mount_rot, CAMERA_MOUNT_RPY_DEG, atol=1e-6), \
        f"{usd_path.name}: camera_mount rotateXYZ={mount_rot} != {CAMERA_MOUNT_RPY_DEG}"
    urdf_rot = urdf_mount_rpy_deg()
    assert np.allclose(urdf_rot, CAMERA_MOUNT_RPY_DEG, atol=1e-6), \
        (f"{CUROBO_URDF.name}: camera_mount_joint rpy={urdf_rot} deg != "
         f"{CAMERA_MOUNT_RPY_DEG} -- URDF 와 USD 의 clocking 이 어긋났다")

    optical = stage.GetPrimAtPath(OPTICAL_PRIM)
    assert optical and optical.IsValid(), f"{usd_path.name}: camera_optical_frame missing"
    tx = UsdGeom.Xformable(optical).GetOrderedXformOps()[0].Get().ExtractTranslation()
    assert np.allclose(list(tx), [OPTICAL_FRAME_X, 0, 0], atol=1e-9), \
        f"{usd_path.name}: camera_optical_frame moved to {tuple(tx)}"

    print(f"  {usd_path.name}: 1 Mesh, {len(pts)} pts, {len(idx) // 3} f, "
          f"mesh->mount identity, mount rpy=({', '.join(f'{v:g}' for v in mount_rot)}) deg == URDF, "
          f"optical_frame x={tx[0]}  OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, help="source OBJ (meters, +Z optical axis); required unless --verify-only")
    ap.add_argument("--faces", type=int, default=DEFAULT_FACES, help="decimation target triangle count")
    ap.add_argument("--dry-run", action="store_true", help="measure and assert, write nothing")
    ap.add_argument("--verify-only", action="store_true", help="only re-run the composed-stage assertions")
    ap.add_argument("--ghost", action="store_true", help="also verify the flattened ghost USD")
    args = ap.parse_args()

    if not args.verify_only:
        if args.source is None:
            ap.error("--source is required unless --verify-only is used")
        if not args.source.exists():
            ap.error(f"source OBJ not found: {args.source}")
        print(f"[1/4] Loading + rotating into mount frame")
        mesh = load_and_rotate(args.source)
        n_src_faces = len(mesh.faces)
        assert_mount_bbox(mesh.vertices, "baked (full res)")

        print(f"[2/4] Decimating to {args.faces} tris")
        dec = decimate(mesh, args.faces)
        assert_mount_bbox(dec.vertices, "decimated")

        print(f"[3/4] Writing outputs")
        verts32 = dec.vertices.astype(np.float32)
        assert_mount_bbox(verts32.astype(np.float64), "float32 points")
        normals_fv = crease_normals(dec)
        if args.dry_run:
            print("  --dry-run: nothing written")
            return
        write_obj(OUT_OBJ, verts32, dec.faces, args.source, n_src_faces)
        write_usdc(OUT_USDC, verts32, dec.faces, normals_fv)

    print(f"[4/4] Verifying composed stage(s)")
    verify_composed(ROBOT_USD)
    if args.ghost:
        verify_composed(GHOST_USD)
    print("OK")


if __name__ == "__main__":
    main()
