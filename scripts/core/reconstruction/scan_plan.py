#!/usr/bin/env python3
"""관심 영역(구) 둘레에 스캔 자세를 깔고 IK 로 거른다 → plan.npz.

물체가 **어떤 구 안에 있다**는 것만 안다고 가정한다. 카메라는 그 구의 중심을 바라보며
위쪽 구면 위를 한 바퀴 돈다(Fibonacci 분포). 광축 roll 은 시야를 거의 바꾸지 않으므로
roll 4가지를 같이 풀어 도달성을 늘리고, 방향마다 시작 자세에 가장 가까운 해 하나를 고른다.

충돌 월드: 씬 장애물 + **관심 영역 전체를 막힌 상자로** 넣는다. 아직 모르는 공간이므로
점유됐다고 본다.

자기 가림: IK 충돌 검사는 시야를 보지 않는다. 물체 너머에서 로봇 쪽을 돌아보는 자세는 팔이
접히며 전완부가 카메라 앞을 막는다(cylinder_sample 첫 스캔에서 42장 중 2장이 ROI 를 한 점도
못 봤다). 그래서 로봇 충돌구가 카메라 ↔ ROI 사이 원뿔에 걸리는 해는 버린다.

관심 영역을 안 주면 시뮬레이션 편의상 정답 메시(source.obj)의 배치 bbox 에서 만든다 —
도면 없는 상황에서는 사용자가 줘야 하는 값이다.

Usage:
    uv run scripts/core/reconstruction/scan_plan.py --object cylinder_sample
    uv run scripts/core/reconstruction/scan_plan.py --object sample \\
        --roi-center -0.73 -0.17 0.25 --roi-radius 0.15 --n-views 40
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from common import config, scene_config  # noqa: E402
from common.math_utils import quaternion_to_rotation_matrix  # noqa: E402
from core.reconstruction.sensor import DEFAULT_SCAN_CAMERA, look_at  # noqa: E402

ROLLS_DEG = (0.0, 90.0, 180.0, 270.0)


def scan_dir(object_name: str, name: str) -> Path:
    return config.DATA_ROOT / object_name / "scan" / name


def roi_from_ground_truth(object_name: str, margin_m: float = 0.02):
    """정답 메시의 배치 bbox 를 감싸는 구 — 시뮬레이션에서 관심 영역 기본값으로만 쓴다."""
    import trimesh

    mesh = trimesh.load(str(config.get_mesh_path(object_name, mesh_type="source")), force="mesh")
    R = quaternion_to_rotation_matrix(np.asarray(config.TARGET_OBJECT["rotation"], dtype=np.float64))
    world = mesh.vertices @ R.T + np.asarray(config.TARGET_OBJECT["position"], dtype=np.float64)
    lo, hi = world.min(axis=0), world.max(axis=0)
    return 0.5 * (lo + hi), 0.5 * float(np.linalg.norm(hi - lo)) + margin_m


def fibonacci_cap(n: int, min_elev_deg: float, max_elev_deg: float) -> np.ndarray:
    """위쪽 구면 띠 [min_elev, max_elev] 위의 고른 방향 n개 (단위벡터, (n, 3))."""
    z_lo, z_hi = math.sin(math.radians(min_elev_deg)), math.sin(math.radians(max_elev_deg))
    i = np.arange(n) + 0.5
    z = z_lo + (z_hi - z_lo) * i / n            # 띠 안에서 면적이 고르게 — z 를 균등하게
    phi = i * math.pi * (3.0 - math.sqrt(5.0))  # 황금각
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def collision_world(roi_center, roi_radius):
    from curobo.scene import Cuboid, Scene

    cuboids = []
    for obj in config.OBSTACLES:
        pos, quat, dims = scene_config.obstacle_obb(obj)
        cuboids.append(Cuboid(name=obj["name"], pose=[*pos.tolist(), *quat.tolist()],
                              dims=dims.tolist()))
    side = 2.0 * float(roi_radius)
    cuboids.append(Cuboid(name="scan_roi", pose=[*np.asarray(roi_center).tolist(), 1, 0, 0, 0],
                          dims=[side, side, side]))
    return Scene(cuboid=cuboids)


# 자기 가림 검사에서 뺄 링크 — 카메라 몸체는 스캔 카메라 뒤·옆에 있고, 충돌구가 렌즈 앞으로
# 조금 삐져나와 있어 넣으면 모든 해가 가려진다고 나온다.
OCCLUSION_IGNORE_LINKS = ("camera_link",)


def self_occluded(robot_cfg, q: np.ndarray, world_T_cam: np.ndarray, roi_center, roi_radius,
                  ignore_links=OCCLUSION_IGNORE_LINKS) -> np.ndarray:
    """로봇 충돌구가 카메라에서 ROI 구를 보는 원뿔을 가리는가 — (M,) bool.

    구가 가린다 = 카메라 앞에 있고(ROI 중심보다 가까움), 시선 방향 각도가 ROI 의 각반경과
    자기 각반경의 합보다 작다. 충돌구는 링크를 감싸는 근사라 실제보다 약간 보수적이다.
    """
    import torch
    from curobo.types import JointState

    from core.trajectory.robot import _get_kinematics

    kin = _get_kinematics(robot_cfg)
    kc = kin.kinematics_config
    keep = np.ones(int(kc.total_spheres), dtype=bool)
    for name in ignore_links:
        keep[kc.get_sphere_index_from_link_name(name).cpu().numpy()] = False

    out = np.zeros(len(q), dtype=bool)
    for s in range(0, len(q), 512):
        qb = torch.tensor(q[s:s + 512], device="cuda:0", dtype=torch.float32)
        st = kin.compute_kinematics(JointState.from_position(qb, joint_names=kin.joint_names))
        sph = st.robot_spheres.reshape(len(qb), -1, 4).cpu().numpy()[:, keep]     # (B, S, 4)
        cam_inv = np.linalg.inv(world_T_cam[s:s + 512])
        c = np.einsum("bij,bsj->bsi", cam_inv[:, :3, :3], sph[..., :3]) + cam_inv[:, None, :3, 3]
        r = sph[..., 3]
        dist_roi = np.linalg.norm(roi_center - world_T_cam[s:s + 512, :3, 3], axis=1)   # (B,)
        roi_half = np.arcsin(np.clip(roi_radius / dist_roi, 0.0, 1.0))[:, None]
        norm = np.linalg.norm(c, axis=2)
        inside = norm <= r                                       # 카메라가 구 안 — 무조건 가림
        ang = np.arccos(np.clip(c[..., 2] / np.maximum(norm, 1e-9), -1.0, 1.0))
        own = np.arcsin(np.clip(r / np.maximum(norm, 1e-9), 0.0, 1.0))
        ahead = (c[..., 2] + r > 0.0) & (norm - r < dist_roi[:, None])
        hit = (r > 0.0) & (inside | (ahead & (ang < roi_half + own)))
        out[s:s + 512] = hit.any(axis=1)
    return out


def greedy_order(q: np.ndarray, start: np.ndarray) -> np.ndarray:
    """관절 L∞ 최근접 순서. 순간이동 캡처에는 상관없지만 실물로 갈 때의 기본 순서다."""
    left = list(range(len(q)))
    order, cur = [], start
    while left:
        k = min(left, key=lambda j: float(np.max(np.abs(q[j] - cur))))
        order.append(k)
        left.remove(k)
        cur = q[k]
    return np.asarray(order, dtype=np.int64)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--object", required=True,
                   help="배치(씬 object_placements)와, ROI 기본값을 만들 정답 메시의 이름")
    scene_config.add_cli_argument(p)
    p.add_argument("--name", default="default", help="결과 폴더 data/{object}/scan/{name}")
    p.add_argument("--roi-center", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   help="관심 영역 구의 중심 (world, m). 없으면 정답 메시 bbox 에서")
    p.add_argument("--roi-radius", type=float, help="관심 영역 구의 반경 (m)")
    p.add_argument("--standoff", type=float, default=None,
                   help="카메라 ↔ ROI 중심 거리 (m). 기본 = 센서 최적 거리")
    p.add_argument("--n-views", type=int, default=48, help="구면 위 후보 방향 수")
    p.add_argument("--min-elev", type=float, default=15.0, help="최저 앙각 (deg)")
    p.add_argument("--max-elev", type=float, default=85.0, help="최고 앙각 (deg)")
    p.add_argument("--ik-seeds", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    scene_config.apply_cli(args, config)
    config.apply_object_placement(args.object)
    spec = DEFAULT_SCAN_CAMERA

    if args.roi_center is not None and args.roi_radius is not None:
        center, radius = np.asarray(args.roi_center, dtype=np.float64), float(args.roi_radius)
        roi_source = "cli"
    else:
        center, radius = roi_from_ground_truth(args.object)
        roi_source = "ground_truth_bbox"
        print(f"[plan] ROI from ground-truth bbox (sim convenience): "
              f"center={np.round(center, 3).tolist()} r={radius:.3f} m")

    standoff = float(args.standoff or spec.optimal_range_m)
    half_fov = math.radians(min(spec.hfov_deg, spec.vfov_deg)) / 2.0
    if standoff - radius < spec.min_range_m:
        print(f"[plan] ⚠️  standoff {standoff:.3f} − r {radius:.3f} < min range "
              f"{spec.min_range_m:.2f} m — 가까운 면은 측정 범위 밖")
    if radius > standoff * math.sin(half_fov):
        print(f"[plan] ⚠️  ROI (r={radius:.3f}) 가 한 장의 시야({standoff * math.sin(half_fov):.3f}) "
              f"보다 크다 — 프레임마다 일부만 담긴다")

    dirs = fibonacci_cap(args.n_views, args.min_elev, args.max_elev)
    cam_T = np.stack([look_at(center + standoff * d, center, roll)
                      for d in dirs for roll in ROLLS_DEG])                    # (n*R, 4, 4)
    tool_T = cam_T @ np.linalg.inv(spec.tool_T_cam_matrix)

    from core import trajectory as PT
    from core.trajectory.poses import rot_to_quat_batch

    robot_cfg = PT.resolve_robot_config(config.DEFAULT_ROBOT_CONFIG)
    world = collision_world(center, radius)
    sols, ok = PT.solve_ik_multi_seed(
        robot_cfg, world, tool_T[:, :3, 3], rot_to_quat_batch(tool_T[:, :3, :3]),
        num_seeds=args.ik_seeds, batch_size=16)

    # 성공한 해마다 자기 가림을 본다 — 같은 방향이라도 팔 모양(분기)에 따라 갈린다.
    k_idx, s_idx = np.nonzero(ok)
    occluded = np.zeros_like(ok)
    occluded[k_idx, s_idx] = self_occluded(robot_cfg, sols[k_idx, s_idx], cam_T[k_idx],
                                           center, radius)
    usable = ok & ~occluded
    n_dirs_ik = int(ok.reshape(len(dirs), -1).any(axis=1).sum())
    n_dirs_clear = int(usable.reshape(len(dirs), -1).any(axis=1).sum())
    print(f"[plan] self-occlusion: {int(occluded.sum())}/{int(ok.sum())} IK solutions blocked; "
          f"directions with IK {n_dirs_ik} -> with a clear view {n_dirs_clear}")

    start = np.asarray(config.ROBOT_START_STATE, dtype=np.float64)
    n_roll = len(ROLLS_DEG)
    picked_q, picked_T, picked_dir, picked_roll = [], [], [], []
    for i in range(len(dirs)):
        best = None
        for r in range(n_roll):
            k = i * n_roll + r
            for s in np.flatnonzero(usable[k]):
                cost = float(np.max(np.abs(sols[k, s] - start)))
                if best is None or cost < best[0]:
                    best = (cost, k, s, r)
        if best is None:
            continue
        _, k, s, r = best
        picked_q.append(sols[k, s])
        picked_T.append(cam_T[k])
        picked_dir.append(dirs[i])
        picked_roll.append(ROLLS_DEG[r])

    if not picked_q:
        sys.exit("[plan] 도달 가능한 스캔 자세가 하나도 없다 — ROI/standoff/앙각을 확인")
    q = np.asarray(picked_q)
    order = greedy_order(q, start)
    q = q[order]
    goal_T_cam = np.asarray(picked_T)[order]
    # 저장하는 카메라 자세는 IK 목표가 아니라 **해의 FK** 다 — cuRobo 는 목표에서 수 mm 떨어진
    # 해도 성공으로 준다(curved_structure 에서 4.6mm). 캡처가 이 값과 Isaac 렌더 자세를 비교한다.
    from scipy.spatial.transform import Rotation

    fk_pos, fk_quat_xyzw = PT.compute_fk(q, robot_cfg)
    tool_fk = np.tile(np.eye(4), (len(q), 1, 1))
    tool_fk[:, :3, :3] = Rotation.from_quat(fk_quat_xyzw).as_matrix()
    tool_fk[:, :3, 3] = fk_pos
    world_T_cam = tool_fk @ spec.tool_T_cam_matrix
    ik_err_mm = 1000.0 * np.linalg.norm(world_T_cam[:, :3, 3] - goal_T_cam[:, :3, 3], axis=1)
    view_dirs = np.asarray(picked_dir)[order]
    rolls = np.asarray(picked_roll)[order]

    out = scan_dir(args.object, args.name)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "plan.npz", q=q, world_T_cam=world_T_cam, goal_T_cam=goal_T_cam,
             view_dirs=view_dirs, roll_deg=rolls,
             roi_center=center, roi_radius=radius, standoff=standoff)
    (out / "plan.json").write_text(json.dumps({
        "object": args.object, "scene": config.ACTIVE_SCENE,
        "roi_center": center.tolist(), "roi_radius": radius, "roi_source": roi_source,
        "standoff": standoff, "n_candidates": int(len(dirs)), "n_with_ik": n_dirs_ik,
        "n_reachable": int(len(q)),
        "elev_deg": [args.min_elev, args.max_elev], "rolls_deg": list(ROLLS_DEG),
        "camera": spec.to_dict(),
    }, indent=2) + "\n")
    elev = np.degrees(np.arcsin(view_dirs[:, 2]))
    print(f"[plan] IK pose error vs goal: max {ik_err_mm.max():.2f} mm")
    print(f"[plan] {len(q)}/{len(dirs)} directions reachable "
          f"(elev {elev.min():.0f}–{elev.max():.0f}°) -> {out.relative_to(PROJECT_ROOT)}/plan.npz")


if __name__ == "__main__":
    main()
