"""관심 영역(구) 둘레의 스캔 자세 — 구면 배치 → cuRobo IK → 자기 가림 제거 → 촬영 순서.

물체가 **어떤 구 안에 있다**는 것만 안다고 가정한다. 카메라는 그 구의 중심을 바라보며 구면 위
(standoff 거리)에 놓인다. 광축 roll 은 시야를 거의 바꾸지 않으므로 roll 4가지를 같이 풀어 도달성을
늘리고, 방향마다 시작 자세에 가장 가까운 해 하나를 고른다.

어느 방향에서 찍나 — **고정 패턴**(SCAN_PATTERN, 11장: 정수리 1 + 앙각 50° 4 + 앙각 25° 6)을 물체
방향(yaw)에 맞춰 돌려 쓴다. 매번 같은 방향이라 결과를 예측하고 비교하기 쉽다. 패턴 방향에 로봇이
못 닿으면(로봇 반대편 등) 위쪽 띠 후보(Fibonacci n_views 개) 중 **가장 가까운 닿는 방향**으로
바꾼다(SUBSTITUTE_MAX_DEG 이내). 그 안에도 없으면 그 장은 뺀다. 찍은 42장 중 고르게 고른 8~10장만
으로 완전성이 이미 포화됐다(cylinder 94.9%, curved 99.6%, sample 47.8%) — 11장은 넉넉한 쪽이다.
pattern=None 이면 패턴 없이 닿는 후보 방향 전부를 찍는다(실험용).

충돌 월드: 씬 장애물 + **관심 영역 전체를 막힌 상자로** 넣는다. 아직 모르는 공간이므로
점유됐다고 본다.

자기 가림: IK 충돌 검사는 시야를 보지 않는다. 물체 너머에서 로봇 쪽을 돌아보는 자세는 팔이
접히며 전완부가 카메라 앞을 막는다(cylinder_sample 첫 스캔에서 42장 중 2장이 ROI 를 한 점도
못 봤다). 그래서 로봇 충돌구가 카메라 ↔ ROI 사이 원뿔에 걸리는 해는 버린다.

파일을 읽고 쓰지 않는다 — ``plan_scan`` 은 ScanPlan 을 돌려주고 저장은 storage 가 한다.
cuRobo·torch 는 함수 안에서만 import 한다.
"""

from __future__ import annotations

import math

import numpy as np

from common import config, scene_config
from core.reconstruction.sensor import ScanCameraSpec, look_at
from core.reconstruction.storage import ScanPlan

ROLLS_DEG = (0.0, 90.0, 180.0, 270.0)
# 자기 가림 검사에서 뺄 링크 — 카메라 몸체는 스캔 카메라 뒤·옆에 있고, 충돌구가 렌즈 앞으로
# 조금 삐져나와 있어 넣으면 모든 해가 가려진다고 나온다.
OCCLUSION_IGNORE_LINKS = ("camera_link",)
# 고정 스캔 패턴 — (앙각 deg, (방위 deg, ...)). 방위는 물체 x축 기준이고 물체 yaw 를 따라 돈다.
# 정수리는 90° 대신 80°: 바로 위는 방위가 정해지지 않아 IK 가 불안정하다. 고리마다 방위를 엇갈려
# 위아래 장이 같은 쪽만 보지 않게 했다.
SCAN_PATTERN = (
    (80.0, (0.0,)),
    (50.0, (45.0, 135.0, 225.0, 315.0)),
    (25.0, (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)),
)
# 못 닿는 패턴 방향을 이 각도 안의 닿는 후보로 바꾼다. 넘으면 그 장은 뺀다.
SUBSTITUTE_MAX_DEG = 30.0


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


def pattern_dirs(pattern, yaw_deg: float = 0.0) -> np.ndarray:
    """(앙각, 방위들) 패턴 → 단위 방향 (P,3). 방위는 yaw_deg(물체 방향) 만큼 돌린다."""
    out = []
    for elev, azims in pattern:
        e = math.radians(elev)
        for az in azims:
            a = math.radians(az + yaw_deg)
            out.append((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))
    return np.asarray(out, dtype=np.float64).reshape(-1, 3)


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


def plan_scan(spec: ScanCameraSpec, roi_center, roi_radius: float, *, standoff: float | None = None,
              n_views: int = 48, min_elev_deg: float = 15.0, max_elev_deg: float = 85.0,
              ik_seeds: int = 32, pattern=SCAN_PATTERN, yaw_deg: float = 0.0,
              log=print) -> ScanPlan:
    """ROI 구 둘레 스캔 자세 → ScanPlan. 하나도 안 닿으면 ValueError.

    pattern: 고정 방향 패턴(모듈 docstring). None 이면 닿는 후보 방향 전부.
    yaw_deg: 패턴을 돌릴 물체 방향(z축 둘레, deg).
    n_views: 위쪽 띠(min/max_elev) 후보 수 — 패턴 방향을 대신할 풀이다.
    """
    from scipy.spatial.transform import Rotation

    from core import trajectory as PT
    from core.trajectory.poses import rot_to_quat_batch

    center = np.asarray(roi_center, dtype=np.float64)
    radius = float(roi_radius)
    standoff = float(standoff or spec.optimal_range_m)
    half_fov = math.radians(min(spec.hfov_deg, spec.vfov_deg)) / 2.0
    if standoff - radius < spec.min_range_m:
        log(f"[plan] ⚠️  standoff {standoff:.3f} − r {radius:.3f} < min range "
            f"{spec.min_range_m:.2f} m — 가까운 면은 측정 범위 밖")
    if radius > standoff * math.sin(half_fov):
        log(f"[plan] ⚠️  ROI (r={radius:.3f}) 가 한 장의 시야({standoff * math.sin(half_fov):.3f}) "
            f"보다 크다 — 프레임마다 일부만 담긴다")

    # 후보 = 패턴 방향(앞쪽 P개) + 대체용 띠 후보. 한 번의 IK 로 같이 푼다.
    pat = pattern_dirs(pattern, yaw_deg) if pattern else np.zeros((0, 3))
    dirs = np.concatenate([pat, fibonacci_cap(n_views, min_elev_deg, max_elev_deg)])
    cam_T = np.stack([look_at(center + standoff * d, center, roll)
                      for d in dirs for roll in ROLLS_DEG])                    # (n*R, 4, 4)
    tool_T = cam_T @ np.linalg.inv(spec.tool_T_cam_matrix)

    robot_cfg = PT.resolve_robot_config(config.DEFAULT_ROBOT_CONFIG)
    sols, ok = PT.solve_ik_multi_seed(
        robot_cfg, collision_world(center, radius), tool_T[:, :3, 3],
        rot_to_quat_batch(tool_T[:, :3, :3]), num_seeds=ik_seeds, batch_size=16)

    # 성공한 해마다 자기 가림을 본다 — 같은 방향이라도 팔 모양(분기)에 따라 갈린다.
    k_idx, s_idx = np.nonzero(ok)
    occluded = np.zeros_like(ok)
    occluded[k_idx, s_idx] = self_occluded(robot_cfg, sols[k_idx, s_idx], cam_T[k_idx],
                                           center, radius)
    usable = ok & ~occluded
    n_dirs_ik = int(ok.reshape(len(dirs), -1).any(axis=1).sum())
    n_dirs_clear = int(usable.reshape(len(dirs), -1).any(axis=1).sum())
    log(f"[plan] self-occlusion: {int(occluded.sum())}/{int(ok.sum())} IK solutions blocked; "
        f"directions with IK {n_dirs_ik} -> with a clear view {n_dirs_clear}")

    start = np.asarray(config.ROBOT_START_STATE, dtype=np.float64)
    n_roll = len(ROLLS_DEG)
    best = {}                                    # 방향 i → (q, goal_T, dir, roll) — 시작 자세에 가장 가까운 해
    for i in range(len(dirs)):
        top = None
        for r in range(n_roll):
            k = i * n_roll + r
            for s in np.flatnonzero(usable[k]):
                cost = float(np.max(np.abs(sols[k, s] - start)))
                if top is None or cost < top[0]:
                    top = (cost, k, s, r)
        if top is not None:
            _, k, s, r = top
            best[i] = (sols[k, s], cam_T[k], dirs[i], ROLLS_DEG[r])
    if not best:
        raise ValueError("도달 가능한 스캔 자세가 하나도 없다 — ROI/standoff/앙각을 확인")

    substituted, dropped = [], []
    if pattern:
        n_pat, used, picked = len(pat), set(), []
        pool = [i for i in best if i >= n_pat]
        for j in range(n_pat):
            if j in best:
                picked.append(best[j])
                continue
            free = [i for i in pool if i not in used]
            if free:
                ang = np.degrees(np.arccos(np.clip(dirs[free] @ dirs[j], -1.0, 1.0)))
                m = int(np.argmin(ang))
                if ang[m] <= SUBSTITUTE_MAX_DEG:
                    used.add(free[m])
                    picked.append(best[free[m]])
                    substituted.append([j, round(float(ang[m]), 1)])
                    continue
            dropped.append(j)
        log(f"[plan] pattern: {len(picked)}/{n_pat} views"
            + (f", {len(substituted)} moved to the nearest reachable direction "
               f"({', '.join(f'#{j} by {a:.0f}°' for j, a in substituted)})" if substituted else "")
            + (f" — ⚠️  {len(dropped)} dropped (nothing reachable within {SUBSTITUTE_MAX_DEG:.0f}°)"
               if dropped else ""))
        if not picked:
            raise ValueError("패턴 방향을 하나도 찍을 수 없다 — ROI/standoff 를 확인")
    else:
        picked = list(best.values())

    q = np.asarray([p[0] for p in picked])
    order = greedy_order(q, start)
    q = q[order]
    goal_T_cam = np.asarray([p[1] for p in picked])[order]
    # 저장하는 카메라 자세는 IK 목표가 아니라 **해의 FK** 다 — cuRobo 는 목표에서 수 mm 떨어진
    # 해도 성공으로 준다(curved_structure 에서 4.6mm). 캡처가 이 값과 Isaac 렌더 자세를 비교한다.
    fk_pos, fk_quat_xyzw = PT.compute_fk(q, robot_cfg)
    tool_fk = np.tile(np.eye(4), (len(q), 1, 1))
    tool_fk[:, :3, :3] = Rotation.from_quat(fk_quat_xyzw).as_matrix()
    tool_fk[:, :3, 3] = fk_pos
    world_T_cam = tool_fk @ spec.tool_T_cam_matrix
    ik_err_mm = 1000.0 * np.linalg.norm(world_T_cam[:, :3, 3] - goal_T_cam[:, :3, 3], axis=1)
    log(f"[plan] IK pose error vs goal: max {ik_err_mm.max():.2f} mm")

    return ScanPlan(
        q=q, world_T_cam=world_T_cam, goal_T_cam=goal_T_cam,
        view_dirs=np.asarray([p[2] for p in picked])[order],
        roll_deg=np.asarray([p[3] for p in picked])[order],
        roi_center=center, roi_radius=radius, standoff=standoff, camera=spec,
        meta={"n_candidates": int(len(dirs)), "n_with_ik": n_dirs_ik,
              "pattern": [[e, list(a)] for e, a in pattern] if pattern else None,
              "yaw_deg": float(yaw_deg), "substituted": substituted, "dropped": dropped,
              "elev_deg": [min_elev_deg, max_elev_deg], "rolls_deg": list(ROLLS_DEG)})
