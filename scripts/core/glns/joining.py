"""GLNS component joining, seam planning, and HOME transitions."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from core import trajectory as PT

class SeamFailure(RuntimeError):
    """An inter-component / HOME-bracket seam could not be bridged (incl. via-home)."""


def collision_gate_and_save(final_traj, final_is_transit, *, robot_cfg, world_config,
                             out_csv):
    """densify 충돌게이트 → 충돌-free 면 FK/시간/CSV/npz 저장(per-component·joined 공용)."""
    collision_traj = PT.densify_for_collision_check(final_traj)
    _, n_collisions = PT.batch_collision_check(collision_traj, robot_cfg, world_config)
    collision_free = n_collisions == 0

    total_time = transit_time = float("nan")
    if collision_free:
        ee_positions, ee_quaternions = PT.compute_fk(final_traj, robot_cfg)
        traj_times, time_stats = PT.compute_trajectory_times(
            final_traj, ee_positions, ee_quaternions,
            ee_speed_m_s=PT.EE_SPEED_MM_S / 1000.0,
            ee_angular_speed_rad_s=np.deg2rad(PT.EE_ANGULAR_SPEED_DEG_S),
            max_joint_vel_rad_s=PT.MAX_JOINT_VEL_RAD_S,
            min_segment_dt=PT.MIN_SEGMENT_DT_S,
            corner_angle_threshold_rad=np.deg2rad(PT.CORNER_ANGLE_THRESHOLD_DEG),
            corner_max_slowdown=PT.CORNER_MAX_SLOWDOWN,
            is_transit=final_is_transit,
        )
        total_time = float(time_stats["total_time"])
        transit_time = float(time_stats["transit_time"])
        PT.save_trajectory_csv(
            final_traj, ee_positions, ee_quaternions, str(out_csv), times=traj_times,
        )
        # trajectory_studio.py 의 dense 재생용 sidecar: CSV 에 없는 is_transit 마스크와
        # FK EE 위치를 함께 저장(실행용 CSV 스키마는 건드리지 않는다).
        np.savez(
            out_csv.with_suffix(".npz"),
            joints=np.asarray(final_traj, dtype=np.float64),
            ee_positions=np.asarray(ee_positions, dtype=np.float64),
            is_transit=np.asarray(final_is_transit, dtype=bool),
            times=np.asarray(traj_times, dtype=np.float64),
        )

    return {
        "n_collisions": int(n_collisions),
        "collision_free": collision_free,
        "total_time": total_time,
        "transit_time": transit_time,
        "n_waypoints": len(final_traj),
        "csv": str(out_csv) if collision_free else None,
    }


def _linf(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def choose_component_order(endpoints, home_q=None, *, strategy="optimized"):
    """방문 순서 + 성분별 방향 결정. 반환 [(성분_index, reversed_bool), ...].

    ``optimized``: component 간 seam 거리(joint L∞)만 최소화한다. HOME 접근/복귀는
    scan 경로와 독립적으로 계획하므로 순서 비용에 포함하지 않는다. 작은 K(≤6)는
    정확 brute-force, 큰 K는 모든 시작 component/방향을 비교하는 greedy를 쓴다.
    ``home_q``는 기존 호출자 호환을 위해 남겨두지만 최적화에 사용하지 않는다.
    ``fixed``: 입력 순서·원방향 그대로.
    """
    K = len(endpoints)
    if K <= 1 or strategy == "fixed":
        return [(i, False) for i in range(K)]
    ends = [(np.asarray(e0, np.float64), np.asarray(e1, np.float64)) for (e0, e1) in endpoints]
    if K <= 6:
        d_btw = [[[[_linf(ends[a][sa], ends[b][sb]) for sb in (0, 1)]
                   for b in range(K)] for sa in (0, 1)] for a in range(K)]
        best, best_cost = None, float("inf")
        for perm in itertools.permutations(range(K)):
            for bits in itertools.product((0, 1), repeat=K):
                cost = 0.0
                for j in range(K - 1):
                    a, b = perm[j], perm[j + 1]
                    # exit(a)=side(0 if reversed else 1), entry(b)=side(1 if reversed else 0)
                    cost += d_btw[a][0 if bits[a] else 1][b][1 if bits[b] else 0]
                if cost < best_cost - 1e-12:
                    best_cost, best = cost, (perm, bits)
        perm, bits = best
        return [(k, bool(bits[k])) for k in perm]
    # Larger K: remove HOME anchoring by trying every component/orientation as
    # the greedy chain's start and retaining the shortest viewpoint-only chain.
    best_order, best_cost = None, float("inf")
    for first in range(K):
        for first_rev in (False, True):
            order = [(first, first_rev)]
            remaining = set(range(K)) - {first}
            cur = ends[first][0 if first_rev else 1]
            cost = 0.0
            while remaining:
                bk, brev, bd = None, False, float("inf")
                for k in sorted(remaining):
                    for rev in (False, True):
                        d = _linf(cur, ends[k][1 if rev else 0])
                        if d < bd:
                            bd, bk, brev = d, k, rev
                order.append((bk, brev))
                cost += bd
                cur = ends[bk][0 if brev else 1]
                remaining.remove(bk)
            if cost < best_cost - 1e-12:
                best_order, best_cost = order, cost
    return best_order


def plan_seams_batched(pairs, *, robot_cfg, world_config, wd_m,
                        motion_planner=None,
                        enable_via_ladder=True):
    """모든 seam(q_from→q_to)을 한 번의 plan_reconfig_transits batch 로 계획.

    반환: pair 별 ``(seg|None, route|None)``. All six planned joints, including
    wrist_3, are preserved. warm BatchMotionPlanner 1회 build 로 모든 seam 처리.
    """
    if not pairs:
        return []
    seam_selected = np.stack([q for pair in pairs for q in pair])        # (2K, 6)
    reconfig_indices = np.arange(0, 2 * len(pairs), 2, dtype=np.int64)   # [0,2,4,...]
    transit_segments, transit_stats = PT.plan_reconfig_transits(
        seam_selected, reconfig_indices, robot_cfg, world_config,
        wd_m=wd_m, enable_via_ladder=enable_via_ladder,
        lock_wrist3=False, motion_planner=motion_planner,
    )
    routes = {s["idx"]: s.get("route") for s in transit_stats if s.get("success")}
    out = []
    for i in range(len(pairs)):
        idx = 2 * i
        seg, route = transit_segments.get(idx), routes.get(idx)
        out.append((seg, route))
    return out


def resample_seam(q_from, q_to, seam_wp, *, robot_cfg, world_config, reconfig_rad, spacing):
    """seam transit 을 성분 내 transit 과 동일 기준(sparse joint-L∞ + 충돌재검)으로 resample.

    seam 전체가 transit 이동이므로 마스크는 all-True 로 강제한다(interpolate_and_resample 은
    2-row 입력의 시작 노드를 scan 으로 타이핑하지만, seam 에는 scan 자세가 없다). 성분 사이
    seam 의 첫 점은 stitch dedup 으로 사라지지만, 맨 앞 HOME 브래킷의 첫 점은 안 사라진다.
    """
    sel = np.stack([np.asarray(q_from, np.float64), np.asarray(q_to, np.float64)])
    traj, _is_transit, _, _ = PT.interpolate_and_resample(
        sel, {0: seam_wp}, robot_cfg,
        mode=PT.RESAMPLE_MODE, spacing=spacing,
        reconfig_threshold_rad=reconfig_rad, world_scene=world_config,
    )
    return traj, np.ones(len(traj), dtype=bool)


def join_components(included, home_q, *, robot_cfg, world_config, wd_m,
                     spacing, reconfig_rad, enable_via_ladder, home_bracket,
                     order_strategy, out_csv, motion_planner=None):
    """충돌-free 성분들을 순서최적화 + seam transit + HOME 브래킷으로 한 궤적으로 stitch.

    seam(via-home 포함)이 하나라도 실패하면 ``SeamFailure`` — 성분을 조용히 드롭하지 않는다.
    """
    home = np.asarray(home_q, dtype=np.float64)
    order = choose_component_order([(c["entry"], c["exit"]) for c in included], home,
                          strategy=order_strategy)

    oriented = []
    for idx, rev in order:
        c = included[idx]
        traj, mask = c["final_traj"], c["final_is_transit"]
        if rev:
            traj, mask = traj[::-1].copy(), mask[::-1].copy()
        oriented.append({"cid": c["cid"], "traj": traj, "mask": mask,
                         "entry": traj[0], "exit": traj[-1]})

    # seam pairs(방문 순서): [front HOME?] inter-comp… [back HOME?]
    pairs, labels = [], []
    if home_bracket:
        pairs.append((home, oriented[0]["entry"]))
        labels.append(f"HOME→comp{oriented[0]['cid']}")
    for j in range(len(oriented) - 1):
        pairs.append((oriented[j]["exit"], oriented[j + 1]["entry"]))
        labels.append(f"comp{oriented[j]['cid']}→comp{oriented[j + 1]['cid']}")
    if home_bracket:
        pairs.append((oriented[-1]["exit"], home))
        labels.append(f"comp{oriented[-1]['cid']}→HOME")

    seam_results = plan_seams_batched(
        pairs, robot_cfg=robot_cfg, world_config=world_config, wd_m=wd_m,
        enable_via_ladder=enable_via_ladder, motion_planner=motion_planner,
    )
    for lbl, (seg, _route) in zip(labels, seam_results):
        if seg is None:
            raise SeamFailure(lbl)

    seam_trajs = [
        resample_seam(q_from, q_to, seg, robot_cfg=robot_cfg, world_config=world_config,
                       reconfig_rad=reconfig_rad, spacing=spacing)
        for (q_from, q_to), (seg, _route) in zip(pairs, seam_results)
    ]

    # 조각 stitch: [front?, traj0, seam01, traj1, …, trajK-1, back?]
    pieces, masks, si = [], [], 0
    if home_bracket:
        pieces.append(seam_trajs[si][0]); masks.append(seam_trajs[si][1]); si += 1
    for j, o in enumerate(oriented):
        pieces.append(o["traj"]); masks.append(o["mask"])
        if j < len(oriented) - 1:
            pieces.append(seam_trajs[si][0]); masks.append(seam_trajs[si][1]); si += 1
    if home_bracket:
        pieces.append(seam_trajs[si][0]); masks.append(seam_trajs[si][1]); si += 1

    joined_traj, joined_is_transit = PT.stitch_trajectory_pieces(pieces, masks)
    gate = collision_gate_and_save(
        joined_traj, joined_is_transit, robot_cfg=robot_cfg,
        world_config=world_config, out_csv=out_csv,
    )
    return {
        "order": [o["cid"] for o in oriented],
        "labels": labels,
        "seam_routes": [r for _seg, r in seam_results],
        "n_seams": len(pairs),
        "entry_exit_check": [(o["cid"], o["entry"], o["exit"]) for o in oriented],
        "gate": gate,
        "n_waypoints": len(joined_traj),
    }


def plan_home_transitions(scan_traj, home_q, *, robot_cfg, world_config, wd_m,
                           spacing, reconfig_rad, enable_via_ladder,
                           motion_planner, out_dir, transitions="both"):
    """Plan and save HOME approach/return independently from the scan trajectory."""
    scan = np.asarray(scan_traj, dtype=np.float64)
    if scan.ndim != 2 or scan.shape[1] != 6 or len(scan) < 2:
        raise ValueError("joined scan trajectory must have shape (N>=2, 6)")
    home = np.asarray(home_q, dtype=np.float64)
    specs = [
        ("approach", (home, scan[0]), "HOME→scan-start",
         "glns_trajectory_home_to_start"),
        ("return", (scan[-1], home), "scan-end→HOME",
         "glns_trajectory_end_to_home"),
    ]
    if transitions not in {"both", "approach", "return"}:
        raise ValueError(f"unknown HOME transition selection: {transitions}")
    selected = specs if transitions == "both" else [s for s in specs if s[0] == transitions]
    pairs = [s[1] for s in selected]
    planned = plan_seams_batched(
        pairs, robot_cfg=robot_cfg, world_config=world_config, wd_m=wd_m,
        enable_via_ladder=enable_via_ladder, motion_planner=motion_planner,
    )
    results = []
    for (_kind, pair, label, stem), (segment, route) in zip(selected, planned):
        out_csv = Path(out_dir) / f"{stem}.csv"
        if segment is None:
            out_csv.unlink(missing_ok=True)
            out_csv.with_suffix(".npz").unlink(missing_ok=True)
            results.append({"label": label, "route": None, "ok": False, "gate": None})
            continue
        traj, mask = resample_seam(
            pair[0], pair[1], segment, robot_cfg=robot_cfg, world_config=world_config,
            reconfig_rad=reconfig_rad, spacing=spacing,
        )
        gate = collision_gate_and_save(
            traj, mask, robot_cfg=robot_cfg, world_config=world_config, out_csv=out_csv,
        )
        results.append({
            "label": label, "route": route,
            "ok": bool(gate["collision_free"]), "gate": gate,
        })
    return results
