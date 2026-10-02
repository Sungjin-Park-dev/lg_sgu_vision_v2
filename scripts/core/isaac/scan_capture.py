#!/usr/bin/env python3
"""스캔 계획(plan.npz)의 관절 자세로 로봇을 순간이동시키며 손목 depth 를 찍는다.

로봇을 실제로 움직이지 않는다 — 복원 품질만 볼 때는 궤적이 필요 없다. 자세마다
관절 상태와 PD 목표를 같은 값으로 박고(``set_start_pose`` 와 같은 방식), 몇 프레임
렌더한 뒤 depth·내부 파라미터·카메라 pose 를 같은 프로세스에서 읽는다. 그래서 이미지와
pose 의 시점이 어긋날 일이 없다.

스캔 카메라는 ``camera_optical_frame`` 아래 별도 prim 이다(검사 카메라는 50mm 시야라
스캔에 못 쓴다). 장착 위치·내부 파라미터는 계획의 camera(= sensor.ScanCameraSpec).

출력: data/{object}/scan/{name}/frames/frame_###.npz — 키 규약은 reconstruction/storage.py
(world_T_cam 은 Isaac 이 렌더한 실제 자세).

Usage:
    uv run scripts/core/isaac/scan_capture.py --object cylinder_sample
    uv run scripts/core/isaac/scan_capture.py --object cylinder_sample --gui --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from common import config as _config  # noqa: E402
from common import scene_config as _scene_config  # noqa: E402
from core.isaac import scene as urscene  # noqa: E402
from core.reconstruction import storage  # noqa: E402
from core.reconstruction.sensor import ScanCameraSpec  # noqa: E402

JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
SCAN_CAMERA_NAME = "ScanCamera"
# USD 카메라의 aperture 는 아무 값이어도 된다 — focal 과의 비만 내부 파라미터를 정한다.
H_APERTURE_MM = 20.955
# OpenCV 광학(+Z 전방, +Y 아래) → USD 카메라(−Z 전방, +Y 위): X 축 둘레 180°.
CV_T_USD = np.diag([1.0, -1.0, -1.0, 1.0])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--object", required=True)
    _scene_config.add_cli_argument(p)
    p.add_argument("--name", default="default", help="data/{object}/scan/{name}")
    p.add_argument("--gui", action="store_true", help="창을 띄운다 (기본 headless)")
    p.add_argument("--limit", type=int, default=None, help="앞에서 몇 장만 찍는다")
    p.add_argument("--settle", type=int, default=2,
                   help="자세를 바꾼 뒤 depth 를 보기 전에 무조건 넘기는 렌더 프레임 수")
    p.add_argument("--max-steps", type=int, default=60,
                   help="depth 가 새 자세로 바뀌어 멈출 때까지 기다리는 최대 렌더 프레임 수")
    return p.parse_args()


def _find_optical_frame(stage) -> str:
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if urscene._is_under_root(p, urscene.STAGE_PATH) and \
                prim.GetName() == urscene.CAMERA_OPTICAL_FRAME_NAME:
            return p
    raise RuntimeError(f"{urscene.CAMERA_OPTICAL_FRAME_NAME} prim not found under {urscene.STAGE_PATH}")


def add_scan_camera(spec: ScanCameraSpec) -> str:
    """camera_optical_frame 아래에 스캔 카메라 prim 을 만든다. prim 경로를 돌려준다."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    path = f"{_find_optical_frame(stage)}/{SCAN_CAMERA_NAME}"
    cam = UsdGeom.Camera.Define(stage, path)
    local = spec.tool_T_cam_matrix @ CV_T_USD          # 열벡터 규약
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d(*local.T.ravel().tolist()))   # USD 는 행벡터 규약
    # fx_px = W · focal / h_aperture  →  focal = h_aperture · fx / W. 정사각 픽셀이라 v_ap 는 비례.
    cam.GetHorizontalApertureAttr().Set(H_APERTURE_MM)
    cam.GetVerticalApertureAttr().Set(H_APERTURE_MM * spec.height / spec.width)
    cam.GetFocalLengthAttr().Set(H_APERTURE_MM * spec.fx / spec.width)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 5.0))
    return path


def wait_for_new_depth(sim, camera, prev_depth, settle: int, max_steps: int):
    """새 자세의 depth 가 나올 때까지 렌더한다. (depth, 렌더한 프레임 수) — 실패하면 (None, n).

    렌더는 비동기라 annotator 가 몇 프레임 전 화면을 준다. 몇 프레임을 넘겨야 하는지는
    고정돼 있지 않다(3장 시험에선 맞았던 고정 4프레임이 42장에선 한 자세씩 밀렸다). 그래서
    **직전 자세의 depth 와 다르고, 연속 두 번 같은** 값이 나올 때까지 기다린다 — 로봇은 순간이동
    뒤 멈춰 있고 depth 렌더는 결정적이라, 같은 값이 두 번 나오면 그게 지금 자세의 화면이다.
    """
    last = None
    for step in range(1, max_steps + 1):
        sim.step(render=True)
        if step <= settle:
            continue
        d = camera.get_depth()
        if d is None:
            continue
        d = np.asarray(d, dtype=np.float32)
        fresh = prev_depth is None or not np.array_equal(d, prev_depth)
        if fresh and last is not None and np.array_equal(d, last):
            return d, step
        last = d
    return None, max_steps


def _pose_matrix(pos, quat_wxyz) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    T = np.eye(4)
    w, x, y, z = (float(v) for v in quat_wxyz)
    T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def main():
    # app.close() 가 프로세스를 바로 끝내서, 파일로 리다이렉트된 stdout 의 블록 버퍼가 사라진다.
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    out = storage.scan_dir(args.object, args.name)
    plan = storage.load_plan(out)
    spec = plan.camera
    q_plan = plan.q[: args.limit] if args.limit else plan.q
    T_plan = plan.world_T_cam[: len(q_plan)]

    _scene_config.apply_cli(args, _config)
    app = urscene.start_sim(headless=not args.gui, enable_ros_bridge=False)

    from isaacsim.core.api import SimulationContext
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.sensors.camera import Camera

    sim = SimulationContext(stage_units_in_meters=1.0)
    urscene.load_workcell(urscene.DEFAULT_USD)
    app.update()
    urscene.load_target_object(args.object)
    app.update()
    art_root = urscene.find_articulation_root()
    cam_path = add_scan_camera(spec)
    app.update()

    camera = Camera(prim_path=cam_path, resolution=(spec.width, spec.height))
    sim.initialize_physics()
    sim.play()
    sim.step(render=True)
    camera.initialize(attach_rgb_annotator=False)
    camera.add_distance_to_image_plane_to_frame()

    art = SingleArticulation(prim_path=art_root)
    art.initialize()
    idx = np.array([art.get_dof_index(n) for n in JOINT_NAMES], dtype=np.int32)
    ctrl = art.get_articulation_controller()

    frames_dir = storage.clear_frames(out)

    K_isaac, prev_depth = None, None
    pose_err_mm, rot_err_deg, n_valid = [], [], []
    t0 = time.time()
    for i, q in enumerate(q_plan):
        art.set_joint_positions(q, joint_indices=idx)
        ctrl.apply_action(ArticulationAction(joint_positions=q, joint_indices=idx))
        depth, steps = wait_for_new_depth(sim, camera, prev_depth, args.settle, args.max_steps)
        if depth is None:
            print(f"[capture] {i:03d}: depth did not settle in {args.max_steps} steps — skipped")
            continue
        prev_depth = depth
        K_isaac = np.asarray(camera.get_intrinsics_matrix(), dtype=np.float64)
        pos, quat = camera.get_world_pose(camera_axes="ros")
        T = _pose_matrix(pos, quat)
        q_meas = np.asarray(art.get_joint_positions(joint_indices=idx), dtype=np.float64)

        storage.save_frame(out, i, depth=depth, K=K_isaac, world_T_cam=T, q_cmd=q, q_meas=q_meas)
        dp = 1000.0 * float(np.linalg.norm(T[:3, 3] - T_plan[i][:3, 3]))
        cos = (np.trace(T_plan[i][:3, :3].T @ T[:3, :3]) - 1.0) / 2.0
        dr = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        valid = int(np.isfinite(depth).sum())
        pose_err_mm.append(dp)
        rot_err_deg.append(dr)
        n_valid.append(valid)
        print(f"[capture] {i:03d}: {steps:2d} steps  valid {valid / depth.size * 100:5.1f}%  "
              f"pose vs plan {dp:6.2f} mm {dr:5.2f}°  "
              f"|q err| {np.degrees(np.max(np.abs(q_meas - q))):.3f}°")

    summary = {
        "n_frames": len(pose_err_mm), "seconds": round(time.time() - t0, 1),
        "K_isaac": None if K_isaac is None else K_isaac.tolist(),
        "K_spec": spec.K.tolist(),
        "pose_vs_plan_mm": {"max": max(pose_err_mm, default=0.0),
                            "median": float(np.median(pose_err_mm)) if pose_err_mm else 0.0},
        "rot_vs_plan_deg": {"max": max(rot_err_deg, default=0.0)},
    }
    (out / "capture.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[capture] {summary['n_frames']} frames in {summary['seconds']}s -> "
          f"{frames_dir.relative_to(PROJECT_ROOT)}")
    print(f"[capture] pose vs plan: max {summary['pose_vs_plan_mm']['max']:.2f} mm, "
          f"{summary['rot_vs_plan_deg']['max']:.2f}°")
    if K_isaac is not None and not np.allclose(K_isaac, spec.K, rtol=1e-3):
        print(f"[capture] ⚠️  K mismatch: isaac {K_isaac.round(1).tolist()} vs spec {spec.K.round(1).tolist()}")

    sim.stop()
    app.close()


if __name__ == "__main__":
    main()
