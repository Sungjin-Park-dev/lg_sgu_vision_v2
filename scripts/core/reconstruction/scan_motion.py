"""스캔 자세 사이를 충돌-free 로 잇는 실행 궤적 — 시작 자세 → 장 0 → … → 장 N-1 → 시작 자세.

새 알고리즘은 없다. GLNS 궤적이 쓰는 transit 엔진을 그대로 쓴다(plan_move.py 와 같은 조립):
    build_reconfig_motion_planner → plan_reconfig_transits(direct → via-roll → via-home)
    → interpolate_and_resample → collision_gate_and_save(densify 충돌 게이트 + 시간)
모든 경계를 transit 으로 계획한다 — 스캔 장 사이는 수십~수백 도라 직선 보간 대상이 아니다.

충돌 월드는 계획과 같다(planning.collision_world): 씬 장애물 + **ROI 전체를 막힌 상자로**. 물체
모양을 모르므로 ROI 안으로는 아예 들어가지 않는다.

이동이 막히면 그 장의 **2π 동치**(손목 등을 반대로 감은 같은 자세 — FK 가 같아 찍히는 화면도 같다)
로 바꿔 다시 잇는다(_repair). 계획 DP 는 관절 거리만 보고 감는 방향을 고르는데, 그 방향이 충돌로
막혀 있을 때가 있다(sample: 손목 +212° 로는 못 가고 −148° 로는 바로 간다). 바뀐 관절값은 반환되어
계획에 다시 들어간다.

그래도 한 경계라도 못 이으면 궤적을 내지 않는다(MotionError) — interpolate_and_resample 은 이을 수 없는
경계에서 장을 조용히 버리는데, 스캔에서 장이 빠지면 그만큼 표면이 빈다. 계획 자체는 그대로
쓸 수 있다(촬영은 순간이동으로 한다).

출력: {scan}/trajectory.csv(+.npz) — 실행용 CSV 는 다른 궤적과 같은 스키마(publish.py 로 보낼 수
있다). 장마다의 행 번호는 npz meta 의 ``shot_rows`` (storage.load_motion).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.reconstruction import storage
from core.reconstruction.planning import WRAP_JOINTS, _wrap_variants, collision_world

# 플래너 warmup 버퍼는 배치 슬롯당 ~375MB — Isaac 이 같은 GPU 를 쓰는 동안 돈다(패널의 계획
# 서브프로세스). 경계가 12~14개라 4 면 3~4번에 나눠 푼다.
SCAN_MOTION_BATCH = 4
# 막힌 이동을 고칠 때 한 번에 시험할 2π 동치 수(가까운 순) — 하나 고치는 데 2이동 × 이만큼을 푼다.
REPAIR_CHUNK = 8


class MotionError(RuntimeError):
    """스캔 자세를 하나의 충돌-free 궤적으로 잇지 못했다."""


def plan_scan_motion(plan: storage.ScanPlan, start: np.ndarray, out_dir: Path, *,
                     batch_size: int = SCAN_MOTION_BATCH, log=print) -> tuple[dict, np.ndarray]:
    """start → 계획의 장들 → start 를 잇는 궤적을 계획해 out_dir/trajectory.csv(+npz) 로 저장.

    반환: (요약 dict — duration_s, n_waypoints, routes, repaired, 장 관절값 (N,6)). 장 관절값은
    막힌 이동을 고치느라 2π 동치로 바뀌었을 수 있다 — 호출자가 plan.q 에 다시 넣는다.
    실패하면 MotionError.
    """
    from common import config
    from core import trajectory as PT
    from core.trajectory import settings

    start = np.asarray(start, dtype=np.float64)
    selected = np.vstack([start[None], plan.q, start[None]])          # (N+2, 6)
    edges = np.arange(len(selected) - 1, dtype=np.int64)
    robot_cfg = PT.resolve_robot_config(config.DEFAULT_ROBOT_CONFIG)
    world = collision_world(plan.roi_center, plan.roi_radius)

    planner = PT.build_reconfig_motion_planner(robot_cfg, world, batch_size=batch_size)
    segments, stats = PT.plan_reconfig_transits(
        selected, edges, robot_cfg, world, lock_wrist3=False, enable_via_ladder=True,
        motion_planner=planner)
    routes = {}
    for st in stats:
        if st.get("success"):
            routes[st["route"]] = routes.get(st["route"], 0) + 1
    failed = [int(st["idx"]) for st in stats if not st.get("success")]
    repaired = []
    for e in failed:
        if e not in segments:                       # 앞 경계를 고치다 같이 풀렸을 수 있다
            node = _repair(selected, segments, e, robot_cfg, world, planner, log)
            if node is not None:
                repaired.append(node - 1)
    failed = [e for e in edges if int(e) not in segments]
    if failed:
        names = ", ".join(_edge_name(i, len(plan.q)) for i in failed)
        raise MotionError(f"{len(failed)}/{len(edges)} moves have no collision-free route ({names})")

    traj, _, dropped, _, kinds = PT.interpolate_and_resample(
        selected, segments, robot_cfg, mode=PT.RESAMPLE_MODE,
        reconfig_threshold_rad=np.deg2rad(PT.RECONFIG_THRESHOLD_DEG), world_scene=world)
    if dropped:                                   # 모든 경계에 transit 이 있으면 생기면 안 된다
        raise MotionError(f"resampling dropped poses {dropped}")
    # 장 = VIEWPOINT 행. 양 끝(시작 자세)도 selected 의 노드라 VIEWPOINT 로 찍혀 나온다 → 뺀다.
    rows = np.flatnonzero(kinds == PT.WAYPOINT_VIEWPOINT)
    shot_rows = rows[1:-1]
    if len(shot_rows) != len(plan.q) or not np.allclose(traj[shot_rows], selected[1:-1], atol=1e-4):
        raise MotionError(f"could not locate the {len(plan.q)} scan poses in the trajectory "
                          f"({len(rows)} viewpoint rows)")
    kinds = kinds.copy()
    kinds[[rows[0], rows[-1]]] = PT.WAYPOINT_PLANNED

    out_csv = Path(out_dir) / storage.MOTION_CSV
    gate = PT.collision_gate_and_save(
        traj, np.ones(len(traj), dtype=bool), robot_cfg=robot_cfg, world_config=world,
        out_csv=out_csv, kinds=kinds,
        meta={"source": "scan_motion", "shot_rows": shot_rows.tolist(),
              "start": start.tolist()})
    if not gate["collision_free"]:
        out_csv.unlink(missing_ok=True)
        raise MotionError(f"collision gate failed: {gate['n_collisions']} colliding samples")

    if repaired:
        routes["repaired"] = len(repaired)
    summary = {"duration_s": round(float(gate["total_time"]), 2),
               "n_waypoints": int(gate["n_waypoints"]), "routes": routes,
               "repaired_shots": repaired,
               "max_joint_vel_rad_s": float(settings.MAX_JOINT_VEL_RAD_S)}
    log(f"[motion] {len(edges)} moves ({', '.join(f'{v} {k}' for k, v in routes.items())}), "
        f"{summary['n_waypoints']} waypoints, {summary['duration_s']:.1f} s at "
        f"{np.degrees(settings.MAX_JOINT_VEL_RAD_S):.0f}°/s joint speed -> {out_csv.name}")
    return summary, selected[1:-1].copy()


def _repair(selected, segments, e, robot_cfg, world, planner, log):
    """막힌 이동 e(selected[e] → selected[e+1])를 한쪽 장의 2π 동치로 바꿔 다시 잇는다.

    바꿀 장 = 도착 장(마지막 이동이면 출발 장 — 시작 자세는 못 바꾼다). 그 장 앞뒤 두 이동이 모두
    direct 로 이어지는 동치 중 앞뒤 관절 거리 합이 가장 작은 것을 쓴다. selected·segments 를 고쳐
    쓰고, 바꾼 노드 번호를 돌려준다(실패하면 None).
    """
    from core import trajectory as PT

    last = len(selected) - 1
    node = e + 1 if e + 1 < last else e
    if node <= 0 or node >= last:
        return None
    prev_q, next_q = selected[node - 1], selected[node + 1]
    cur = selected[node]
    base = (cur + np.pi) % (2.0 * np.pi) - np.pi               # [-π, π] 로 되돌려 동치 전부를 만든다
    base = np.where(WRAP_JOINTS, base, cur)
    var = _wrap_variants(base[None])
    var = var[np.max(np.abs(var - cur), axis=1) > 1e-6]          # 지금 값은 이미 실패했다
    cost = np.abs(var - prev_q).max(axis=1) + np.abs(var - next_q).max(axis=1)
    var = var[np.argsort(cost)]
    for c0 in range(0, len(var), REPAIR_CHUNK):
        chunk = var[c0:c0 + REPAIR_CHUNK]
        rows = np.concatenate([np.stack([prev_q, v, v, next_q]) for v in chunk])
        idx = np.arange(0, len(rows), 2, dtype=np.int64)           # (prev→v), (v→next) 쌍
        segs, _ = PT.plan_reconfig_transits(
            rows, idx, robot_cfg, world, lock_wrist3=False, enable_via_ladder=False,
            motion_planner=planner)
        for k, v in enumerate(chunk):
            a, b = 4 * k, 4 * k + 2
            if a in segs and b in segs:
                log(f"[motion] repaired: shot #{node - 1} switched to an equivalent joint winding "
                    f"(Δ {np.degrees(v - cur).round(0).tolist()}°)")
                selected[node] = v
                segments[node - 1], segments[node] = segs[a], segs[b]
                return node
    log(f"[motion] could not repair the move into/out of shot #{node - 1}")
    return None


def _edge_name(i: int, n: int) -> str:
    a = "start" if i == 0 else f"#{i - 1}"
    b = "start" if i == n else f"#{i}"
    return f"{a}→{b}"
