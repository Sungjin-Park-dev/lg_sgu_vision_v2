"""GLNS component joining, seam planning, and HOME transitions."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from core import trajectory as PT
from core.trajectory import collision_gate_and_save
from core.trajectory.periodic import periodic_joint_delta

class SeamFailure(RuntimeError):
    """An inter-component / HOME-bracket seam could not be bridged (incl. via-home)."""


def _linf(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def rank_component_orders(endpoints, *, strategy="optimized", limit=12):
    """seam 거리(joint L∞ 합)가 작은 순으로 (방문 순서+방향) 후보 목록. 각 원소 [(성분_index, reversed_bool), …].

    첫 원소가 최선 추정이지만 **추정일 뿐이다** — L∞ 는 계획 가능성을 모른다(실측 sample/75:
    추정 1위 seam 이 어떤 경로로도 안 풀리고, 2위(성분을 거꾸로 돌아 입구에서 넘기기)는 풀렸다).
    그래서 join_components 는 이 목록을 차례로 시도한다. 작은 K(≤6)는 전수 나열, 큰 K 는
    greedy 시작점·방향별 후보. ``fixed`` 는 입력 순서·원방향 하나뿐.
    """
    K = len(endpoints)
    if K <= 1 or strategy == "fixed":
        return [[(i, False) for i in range(K)]]
    ends = [(np.asarray(e0, np.float64), np.asarray(e1, np.float64)) for (e0, e1) in endpoints]
    scored = []
    if K <= 6:
        d_btw = [[[[_linf(ends[a][sa], ends[b][sb]) for sb in (0, 1)]
                   for b in range(K)] for sa in (0, 1)] for a in range(K)]
        for perm in itertools.permutations(range(K)):
            for bits in itertools.product((0, 1), repeat=K):
                cost = 0.0
                for j in range(K - 1):
                    a, b = perm[j], perm[j + 1]
                    # exit(a)=side(0 if reversed else 1), entry(b)=side(1 if reversed else 0)
                    cost += d_btw[a][0 if bits[a] else 1][b][1 if bits[b] else 0]
                scored.append((cost, [(k, bool(bits[k])) for k in perm]))
    else:
        # Larger K: greedy chain from every component/orientation as the start.
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
                scored.append((cost, order))
    scored.sort(key=lambda x: x[0])
    return [order for _cost, order in scored[:limit]]


def choose_component_order(endpoints, home_q=None, *, strategy="optimized"):
    """최선 추정 순서 하나 (rank_component_orders 의 첫 원소). ``home_q`` 는 호환용, 미사용."""
    return rank_component_orders(endpoints, strategy=strategy, limit=1)[0]


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
        # seam 은 성분 수-1 개뿐인데 base 회전의 대부분을 만든다 — 사다리 첫 성공이 아니라
        # 모든 단을 시도해 회전이 가장 적은 우회로를 고른다.
        pick_least_base_travel=True,
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
    traj, _is_transit, _, _, _kinds = PT.interpolate_and_resample(
        sel, {0: seam_wp}, robot_cfg,
        mode=PT.RESAMPLE_MODE, spacing=spacing,
        reconfig_threshold_rad=reconfig_rad, world_scene=world_config,
    )
    # seam 은 전부 모션 플래너가 만든 이동이다. 양 끝은 성분의 viewpoint 와 겹치는데,
    # stitch dedup 이 그 행을 성분 쪽 라벨로 남긴다.
    return (traj, np.ones(len(traj), dtype=bool),
            np.full(len(traj), PT.WAYPOINT_PLANNED, dtype=np.int8))


def join_components(included, home_q, *, robot_cfg, world_config, wd_m,
                     spacing, reconfig_rad, enable_via_ladder, home_bracket,
                     order_strategy, out_csv, motion_planner=None, meta=None):
    """충돌-free 성분들을 순서최적화 + seam transit + HOME 브래킷으로 한 궤적으로 stitch.

    seam(via-home 포함)이 하나라도 실패하면 ``SeamFailure`` — 성분을 조용히 드롭하지 않는다.

    ⚠ 성분은 ``solve`` 에서 **각자 독립적으로** 2π unwrap 된다. 그래서 성분 경계에서 같은
    자세가 360° 떨어져 보일 수 있다(실측 sample/74: comp000 출구 154.3°, comp001 입구
    -148.1°). 그것을 맞춰주는 정렬을 넣어봤으나 **두 물체 모두에서 악화**했다 — L∞ 로는
    가까워지지만 그 표현이 오히려 계획하기 어려운 쪽이라, direct 로 풀리던 seam 이
    via-home 으로 밀려났다(square_structure base3 563->740°). 고치려면 두 표현을 모두
    계획해 실제로 짧은 쪽을 채택해야 한다 — L∞ 는 경로 길이의 대리값이 못 된다.
    """
    home = np.asarray(home_q, dtype=np.float64)
    orders = rank_component_orders([(c["entry"], c["exit"]) for c in included],
                                   strategy=order_strategy)

    def _orient(order):
        out = []
        for idx, rev in order:
            c = included[idx]
            traj, mask = c["final_traj"], c["final_is_transit"]
            kinds_c = c.get("final_kinds")
            if kinds_c is None:
                kinds_c = np.full(len(traj), PT.WAYPOINT_INTERPOLATED, dtype=np.int8)
            kinds_c = np.asarray(kinds_c, dtype=np.int8)
            if rev:
                traj, mask = traj[::-1].copy(), mask[::-1].copy()
                kinds_c = kinds_c[::-1].copy()      # 라벨은 행에 붙어 있으므로 같이 뒤집는다
            out.append({"cid": c["cid"], "traj": traj, "mask": mask, "kinds": kinds_c,
                        "entry": traj[0], "exit": traj[-1], "rev": rev})
        return out

    def _pairs(oriented):
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
        return pairs, labels

    # 순서 추정(L∞)은 계획 가능성을 모른다. 1위가 막히면 다음 순서·방향으로 간다 — 같은
    # 끝점 쌍은 한 번만 계획한다(순서가 달라도 seam 은 끝점 쌍으로 정해진다).
    seam_cache: dict = {}

    def _key(q_from, q_to):
        return (tuple(np.round(q_from, 6)), tuple(np.round(q_to, 6)))

    chosen = None
    for attempt, order in enumerate(orders, start=1):
        oriented = _orient(order)
        pairs, labels = _pairs(oriented)
        todo = [(i, pr) for i, pr in enumerate(pairs) if _key(*pr) not in seam_cache]
        if todo:
            planned = plan_seams_batched(
                [pr for _i, pr in todo], robot_cfg=robot_cfg, world_config=world_config,
                wd_m=wd_m, enable_via_ladder=enable_via_ladder, motion_planner=motion_planner,
            )
            for (_i, pr), res in zip(todo, planned):
                seam_cache[_key(*pr)] = res
        seam_results = [seam_cache[_key(*pr)] for pr in pairs]
        failed = [lbl for lbl, (seg, _route) in zip(labels, seam_results) if seg is None]
        desc = " -> ".join(f"comp{o['cid']}{'(rev)' if o['rev'] else ''}" for o in oriented)
        if not failed:
            if attempt > 1:
                print(f"  order attempt {attempt}/{len(orders)}: {desc} - all seams planned")
            chosen = (oriented, pairs, labels, seam_results)
            break
        print(f"  order attempt {attempt}/{len(orders)}: {desc} - seam failed: "
              f"{', '.join(failed)}" + (" - trying the next order" if attempt < len(orders) else ""))
    if chosen is None:
        raise SeamFailure(f"no component order/direction could be bridged "
                          f"({len(orders)} tried, via-home included)")
    oriented, pairs, labels, seam_results = chosen

    seam_trajs = [
        resample_seam(q_from, q_to, seg, robot_cfg=robot_cfg, world_config=world_config,
                       reconfig_rad=reconfig_rad, spacing=spacing)
        for (q_from, q_to), (seg, _route) in zip(pairs, seam_results)
    ]

    # 조각 stitch: [front?, traj0, seam01, traj1, …, trajK-1, back?]
    pieces, masks, kinds, si = [], [], [], 0

    def _push_seam(idx):
        pieces.append(seam_trajs[idx][0]); masks.append(seam_trajs[idx][1])
        kinds.append(seam_trajs[idx][2])

    if home_bracket:
        _push_seam(si); si += 1
    for j, o in enumerate(oriented):
        pieces.append(o["traj"]); masks.append(o["mask"])
        kinds.append(o["kinds"])
        if j < len(oriented) - 1:
            _push_seam(si); si += 1
    if home_bracket:
        _push_seam(si); si += 1

    joined_traj, joined_is_transit, joined_kinds = PT.stitch_trajectory_pieces(
        pieces, masks, kinds=kinds)
    gate = collision_gate_and_save(
        joined_traj, joined_is_transit, robot_cfg=robot_cfg,
        world_config=world_config, out_csv=out_csv, meta=meta, kinds=joined_kinds,
    )
    return {
        "order": [o["cid"] for o in oriented],
        "order_desc": " -> ".join(f"comp{o['cid']}{'(rev)' if o['rev'] else ''}" for o in oriented),
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
         "trajectory_home_to_start"),
        ("return", (scan[-1], home), "scan-end→HOME",
         "trajectory_end_to_home"),
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
