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


def choose_component_order(endpoints, home_q=None, *, strategy="optimized"):
    """방문 순서 + 성분별 방향 결정. 반환 [(성분_index, reversed_bool), ...].

    ``optimized``: component 간 seam 거리(joint L∞)를 최소화한다. 작은 K(≤6)는 정확
    brute-force, 큰 K는 모든 시작 component/방향을 비교하는 greedy를 쓴다.
    ``fixed``: 입력 순서·원방향 그대로.

    ``home_q`` 는 기존 호출자 호환을 위해 남겨두며 최적화에 쓰지 않는다. 스캔의 진입
    자세를 현재 로봇 자세에 맞추는 일은 방문 순서를 바꾸지 않고 ``align_path_to_start``
    가 표현만 골라서 처리한다 — 검사 순서는 GLNS 가 정한 그대로 둔다.
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


def align_path_to_start(traj, start_q, robot_cfg):
    """이어붙인 궤적 전체를 ``start_q`` 에서 출발하도록 2π 등가로 다시 표현한다.

    첫 행은 start_q 에, 그 뒤 각 행은 **직전 행**에 가장 가까운 등가로 맞춘다. 첫 행만
    옮기면 둘째 행에서 튀기 때문이다. 자세는 하나도 안 바뀌고(2π 정수배 이동) 표현만 고른다.

    generate 의 **맨 마지막 후처리**다. 성분 순서·방향·seam 이 모두 확정되고 조각이 이어붙은
    뒤라, 여기서 고른 표현이 곧 저장되는 값이다. solve.py 의 성분별 unwrap 은 join 이 뒤집을
    수 있어 최종 표현을 정하지 못한다. 검사 순서(어느 viewpoint 를 언제 보는가)는 GLNS 가
    정한 그대로 두고 **관절값 표현만** 바꾼다.

    **안전장치**: 관절 한계 때문에 어떤 행이 못 따라오면 거기서 새 점프가 생긴다. 그래서
    정렬 뒤 인접 행 최대 변화량을 원본과 비교해, 나빠졌으면 **통째로 원본을 쓴다** —
    진입을 줄이려다 궤적 한가운데를 깨뜨리지 않는다.

    Returns: (표현이 바뀐 궤적, 적용했는지 여부)
    """
    from core.glns.candidates import _joint_limits_and_periods

    traj = np.asarray(traj, dtype=np.float64)
    # 어떤 경로로 끝나든 **반드시 한 줄은 찍는다.** 조용히 빠지면 "정렬이 안 돌았다" 와
    # "돌았는데 바꿀 게 없다" 를 구분할 수 없어서, 멀쩡한 동작을 버그로 의심하게 된다.
    if start_q is None:
        print("  Start alignment: skipped - no start pose given "
              "(pass --start-joints to align the trajectory to where the robot is)")
        return traj, False
    if len(traj) == 0:
        print("  Start alignment: skipped - the joined trajectory is empty")
        return traj, False
    try:
        lower, upper, periods = _joint_limits_and_periods(robot_cfg)
    except Exception as exc:  # noqa: BLE001 — 정렬 실패가 생성을 죽이면 안 된다
        print(f"  Start alignment: skipped - could not read joint limits ({exc})")
        return traj, False
    if not np.any(periods > 0.0):
        print("  Start alignment: skipped - no joint has a full extra turn of range, "
              "so there is no 2pi equivalent to choose")
        return traj, False

    out = np.empty_like(traj)
    ref = np.asarray(start_q, dtype=np.float64)
    try:
        for i in range(len(traj)):
            out[i] = PT.align_to_reference(traj[i], ref, periods, lower, upper)
            ref = out[i]
    except Exception as exc:  # noqa: BLE001 — 원본을 지키고 사실대로 알린다
        print(f"  Start alignment: failed, keeping the trajectory as solved ({exc})")
        return traj, False

    entry0 = float(np.max(np.abs(traj[0] - np.asarray(start_q, dtype=np.float64))))
    if np.array_equal(out, traj):
        # 침묵하면 "정렬이 안 돌았다"와 구분이 안 된다. 남은 거리는 2π 로 못 없애는
        # **진짜** 차이다(비주기 축이거나, 등가가 한계 밖이거나, 팔 자세 자체가 다르다).
        print(f"  Start alignment: nothing to change - the scan already starts at the "
              f"closest 2pi representation ({np.rad2deg(entry0):.1f} deg from the given "
              f"pose; that gap is real, use Plan/Move to Start)")
        return traj, False
    # 연속성이 나빠졌으면 채택하지 않는다.
    def worst_step(a):
        return float(np.max(np.abs(np.diff(a, axis=0)))) if len(a) > 1 else 0.0
    before, after = worst_step(traj), worst_step(out)
    if after > before + 1e-9:
        print(f"  start alignment rejected: it would grow the worst joint step "
              f"{np.rad2deg(before):.2f} -> {np.rad2deg(after):.2f} deg (limits block it)")
        return traj, False
    entry_before = float(np.max(np.abs(traj[0] - np.asarray(start_q, dtype=np.float64))))
    entry_after = float(np.max(np.abs(out[0] - np.asarray(start_q, dtype=np.float64))))
    per_joint = np.rad2deg(np.abs(out[0] - np.asarray(start_q, dtype=np.float64)))
    print(f"  Aligned the joined path to the start pose: entry travel "
          f"{np.rad2deg(entry_before):.1f} -> {np.rad2deg(entry_after):.1f} deg "
          f"(worst step unchanged at {np.rad2deg(after):.2f} deg)")
    print(f"    remaining per-joint gap [deg]: "
          f"{np.round(per_joint, 1).tolist()} - what is left cannot be removed by 2pi")
    return out, True


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
                     order_strategy, out_csv, motion_planner=None, meta=None,
                     start_q=None):
    # start_q: 이어붙인 궤적의 관절값 표현만 이 자세 기준으로 다시 고른다
    # (align_path_to_start). 방문 순서·방향은 건드리지 않는다.
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
    order = choose_component_order([(c["entry"], c["exit"]) for c in included], home,
                          strategy=order_strategy)

    oriented = []
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
        oriented.append({"cid": c["cid"], "traj": traj, "mask": mask, "kinds": kinds_c,
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
    # 방향·seam 이 확정된 뒤에 표현을 고른다 — 여기가 마지막 기회다.
    joined_traj, _aligned = align_path_to_start(joined_traj, start_q, robot_cfg)
    gate = collision_gate_and_save(
        joined_traj, joined_is_transit, robot_cfg=robot_cfg,
        world_config=world_config, out_csv=out_csv, meta=meta, kinds=joined_kinds,
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
