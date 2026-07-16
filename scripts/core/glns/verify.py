#!/usr/bin/env python3
"""Collision-aware verification of a GLNS-selected viewpoint/IK path.

``glns/solve.py`` 는 Delaunay 제약 하에 reconfiguration 을 최소화한 viewpoint/IK
순서를 성분별로 고르지만, **각 viewpoint 의 정적 자세 충돌**만 검사하고 viewpoint
사이의 **이동(motion)** 은 계획·충돌검사하지 않는다.

이 도구는 그 GLNS 결과(``glns_result_*.h5``)를 받아, **성분마다 독립적으로** GLNS 가 고른
joint 순서(``selected_joints``)를 ``trajectory/cli.py`` 의 Phase 4-6
(reconfig transit 계획 → densify 충돌검증 → uniform resample → FK/시간 → CSV)에 그대로
흘려보내 "충돌을 고려하면 이 경로가 실제로 실행 가능한가"를 확인한다.

두 도구는 같은 collision world / robot config / wrist_3 lock 값을 쓰므로(둘 다
``plan_trajectory`` 를 import), GLNS 에서 충돌-free 였던 자세는 여기서도 충돌-free 다 —
검증 대상은 오직 자세 사이의 이동이다.

plan_trajectory 는 일체 수정하지 않고 라이브러리로 재사용한다(solve_glns_path 와 동일 패턴).

``--join``(기본 on)이면 충돌-free 성분들을 하나의 연속 실행 궤적으로 잇는다: 방문 순서·방향을
viewpoint component 간 seam 거리(joint L∞)로만 최적화하고 ``_stitch_pieces``로
봉합한다. HOME 접근/복귀는 기본적으로 별도 계획하며, ``--home-bracket``을 명시한
경우에만 양 끝에 붙인다. seam 은 절대 조용히 드롭하지 않고
실패 시 hard-error(``glns_trajectory_joined.csv/.npz`` 미생성). 각 성분의 resample/drop 은
성분 내부로 한정돼(``interpolate_and_resample`` 의 "최장 run keep" 이 성분 경계를 넘지 못함).

실행:
    uv run --no-sync scripts/core/glns/verify.py \
        --result data/sample/ik/74/glns_result_YYYYMMDD_HHMMSS.h5 [--join] [--order optimized]

성분별 trajectory CSV 는 결과 h5 와 같은 디렉토리에 ``glns_trajectory_comp{cid}.csv`` 로
저장된다(DP 의 ``trajectory_*.csv`` 와 구분). 같은 자리에 ``glns_trajectory_comp{cid}.npz``
(joints/ee_positions/is_transit/times)도 저장돼 ``trajectory_studio.py`` 가 transit 포함 실제
motion 을 재생할 수 있다. ``--join`` 결과는 ``glns_trajectory_joined.csv/.npz`` (동일 스키마).
충돌이 검출된 성분은 CSV/npz 를 쓰지 않고 FAIL 로 보고한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from common import config  # noqa: E402
from core import trajectory as PT  # noqa: E402
from core.glns.joining import (  # noqa: E402
    SeamFailure,
    collision_gate_and_save,
    join_components,
    plan_home_transitions,
)
from core.glns.storage import read_result_hdf5  # noqa: E402


def _decode(value):
    """h5 attr(JSON 문자열/바이트/numpy 스칼라)을 파이썬 값으로 복원."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feed a GLNS-selected path through plan_trajectory's "
                    "collision-aware transit/verify/resample stage, per component.",
    )
    parser.add_argument("--result", type=Path, required=True,
                        help="GLNS result HDF5 (data/{object}/ik/{N}/glns_result_*.h5)")
    parser.add_argument("--object", default=None,
                        help="Object name override (default: read from result attrs)")
    parser.add_argument("--spacing", type=float, default=PT.DEFAULT_SPACING_M,
                        help=f"Scan resample spacing in meters (default: {PT.DEFAULT_SPACING_M})")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="CSV output dir (default: alongside the result h5)")
    parser.add_argument("--no-via", action="store_true",
                        help="via-roll/tilt/home 사다리 비활성 — direct(plan_cspace) 실패분은 "
                             "드롭(viewpoint skip). graph-direct 만으로 도는지 확인용")
    parser.add_argument("--join", action=argparse.BooleanOptionalAction, default=True,
                        help="충돌-free 성분들을 seam transit으로 하나의 연속 "
                             "scan 궤적(glns_trajectory_joined.csv)으로 연결 (default: on)")
    parser.add_argument("--order", choices=("optimized", "fixed"), default="optimized",
                        help="성분 방문 순서: optimized(seam 거리 최소) / fixed(id 순서). default optimized")
    parser.add_argument("--home-bracket", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="joined 궤적 양 끝에 HOME 접근/복귀를 붙임 "
                             "(default: off; use separate HOME transition planning)")
    parser.add_argument("--require-full-coverage", action="store_true",
                        help="fail a component and joined output if any viewpoint is skipped")
    parser.add_argument("--home-transitions-only", action="store_true",
                        help="plan HOME→scan-start and scan-end→HOME from an existing "
                             "glns_trajectory_joined.npz, without replanning the scan")
    parser.add_argument("--home-transition", choices=("both", "approach", "return"),
                        default="both",
                        help="with --home-transitions-only, plan both legs or only "
                             "HOME→start / end→HOME (default: both)")
    args = parser.parse_args()
    if not args.result.exists():
        parser.error(f"Result not found: {args.result}")
    if args.spacing <= 0.0:
        parser.error("--spacing must be > 0")
    return args


def _plan_and_resample_component(component, *, robot_cfg, world_config, reconfig_rad,
                                 wd_m, spacing, enable_via_ladder=True,
                                 motion_planner=None):
    """한 성분의 Phase 4-5(transit 계획 + resample). 파일 I/O·충돌게이트 없음.

    반환 dict 의 ``ok`` 가 False 면 안전 연속 구간이 viewpoint 2개 미만(`<2 safe-run`)이라
    검증 불가 — join 에서 이 성분을 제외하는 데 쓴다. `entry/exit` 는 resample 된 양 끝 자세.
    """
    selected = np.asarray(component["selected_joints"], dtype=np.float64)  # (M, 6)
    vp_order = np.asarray(component["viewpoint_order"], dtype=np.int64)     # (M,) 원본 인덱스
    M = len(selected)

    # reconfig 경계는 plan_trajectory main()/Phase 5(_build_runs) 와 동일하게 selected 의
    # 6-DoF L∞ 로 재산출한다(Phase 4 transit 대상과 Phase 5 run-building 이 일치해야 함).
    # GLNS strict r_any and continuous verifier both use all six joints. The
    # selected wrist_3 values are preserved through transit planning as well.
    jumps = np.max(np.abs(np.diff(selected, axis=0)), axis=1)              # (M-1,)
    is_reconfig = jumps > reconfig_rad
    reconfig_indices = np.where(is_reconfig)[0]

    gl_reconfig = component.get("is_reconfiguration")
    mismatch = 0
    if gl_reconfig is not None:
        gl_reconfig = np.asarray(gl_reconfig, dtype=bool)
        mismatch = int(np.sum(gl_reconfig != is_reconfig))
        if mismatch:
            print(f"    WARNING: GLNS is_reconfiguration 와 재산출 결과가 {mismatch}개 "
                  f"edge 에서 불일치 — wrist_3 lock/threshold 가정 확인 필요. "
                  f"재산출값으로 진행.")

    # 작은 jump는 원래 joint 직선 보간 대상이다. 이 중 충돌하는 edge도 viewpoint를
    # drop하기 전에 MotionGen fallback 대상으로 승격한다. reconfiguration edge와 합쳐
    # 한 번에 batch planning하므로 planner warmup/호출 비용도 공유한다.
    scan_edge_indices = np.where(~is_reconfig)[0]
    collision_fallback_indices = PT.find_colliding_interpolation_edges(
        selected, scan_edge_indices, robot_cfg, world_config,
    )
    if len(collision_fallback_indices):
        labels = [f"{int(vp_order[i])}→{int(vp_order[i + 1])}"
                  for i in collision_fallback_indices]
        print(f"    Scan interpolation collision: {len(collision_fallback_indices)} edge(s) "
              f"→ MotionGen fallback [{', '.join(labels)}]")

    # --- Phase 4: reconfig + colliding scan edge transit 계획(충돌회피 motion) ---
    motion_indices = np.union1d(reconfig_indices, collision_fallback_indices).astype(np.int64)
    transit_segments, transit_stats = {}, []
    if len(motion_indices) > 0:
        transit_segments, transit_stats = PT.plan_reconfig_transits(
            selected, motion_indices, robot_cfg, world_config,
            label_idx=vp_order, wd_m=wd_m, enable_via_ladder=enable_via_ladder,
            lock_wrist3=False, motion_planner=motion_planner,
        )
    n_transit_ok = len(transit_segments)
    n_transit_req = int(len(motion_indices))
    n_reconfig_req = int(len(reconfig_indices))
    n_collision_fallback_req = int(len(collision_fallback_indices))
    n_collision_fallback_ok = sum(
        int(i) in transit_segments for i in collision_fallback_indices
    )

    # --- Phase 5: transit 병합 + uniform resample (연속 scan edge 를 densify-충돌검증) ---
    try:
        final_traj, final_is_transit, skipped_vps, runs_info = PT.interpolate_and_resample(
            selected, transit_segments, robot_cfg,
            mode=PT.RESAMPLE_MODE, spacing=spacing,
            reconfig_threshold_rad=reconfig_rad, world_scene=world_config,
        )
    except RuntimeError as exc:   # <2 safe-run: 모든 인접 전이가 이을 수 없는 reconfig
        return {"ok": False, "error": str(exc), "M": M, "reconfig_mismatch": mismatch}

    skipped_orig = [int(vp_order[i]) for i in skipped_vps]
    return {
        "ok": True,
        "final_traj": final_traj,
        "final_is_transit": final_is_transit,
        "entry": np.asarray(final_traj[0], dtype=np.float64),
        "exit": np.asarray(final_traj[-1], dtype=np.float64),
        "M": M,
        "covered": int(runs_info["kept"][2]),
        "dropped": skipped_orig,
        "n_runs": len(runs_info["runs"]),
        "reconfig_req": n_reconfig_req,
        "transit_req": n_transit_req,
        "transit_ok": n_transit_ok,
        "collision_fallback_req": n_collision_fallback_req,
        "collision_fallback_ok": n_collision_fallback_ok,
        "reconfig_mismatch": mismatch,
    }


def _verify_component(component, *, robot_cfg, world_config, reconfig_rad, wd_m,
                      spacing, out_csv, enable_via_ladder=True,
                      require_full_coverage=False, motion_planner=None):
    """한 성분을 Phase 4-6 으로 검증(per-component CSV/npz 기록). 결과 dict 반환."""
    pr = _plan_and_resample_component(
        component, robot_cfg=robot_cfg, world_config=world_config,
        reconfig_rad=reconfig_rad, wd_m=wd_m,
        spacing=spacing, enable_via_ladder=enable_via_ladder,
        motion_planner=motion_planner,
    )
    if not pr["ok"]:
        return {
            "M": pr["M"], "covered": 0, "dropped": [], "n_runs": 0,
            "reconfig_req": 0, "transit_req": 0, "transit_ok": 0,
            "collision_fallback_req": 0, "collision_fallback_ok": 0,
            "n_collisions": 0,
            "collision_free": False, "total_time": float("nan"),
            "transit_time": float("nan"), "n_waypoints": 0,
            "reconfig_mismatch": pr.get("reconfig_mismatch", 0), "csv": None,
            "final_traj": None, "final_is_transit": None,
            "entry": None, "exit": None, "error": pr["error"],
        }
    if require_full_coverage and pr["dropped"]:
        Path(out_csv).unlink(missing_ok=True)
        Path(out_csv).with_suffix(".npz").unlink(missing_ok=True)
        return {
            "M": pr["M"], "covered": pr["covered"], "dropped": pr["dropped"],
            "n_runs": pr["n_runs"], "reconfig_req": pr["reconfig_req"],
            "transit_req": pr["transit_req"],
            "transit_ok": pr["transit_ok"], "n_collisions": 0,
            "collision_fallback_req": pr["collision_fallback_req"],
            "collision_fallback_ok": pr["collision_fallback_ok"],
            "collision_free": False, "total_time": float("nan"),
            "transit_time": float("nan"), "n_waypoints": len(pr["final_traj"]),
            "reconfig_mismatch": pr["reconfig_mismatch"], "csv": None,
            "final_traj": None, "final_is_transit": None,
            "entry": None, "exit": None,
            "error": "full coverage required but viewpoint(s) were skipped",
        }
    gate = collision_gate_and_save(
        pr["final_traj"], pr["final_is_transit"],
        robot_cfg=robot_cfg, world_config=world_config, out_csv=out_csv,
    )
    return {
        "M": pr["M"], "covered": pr["covered"], "dropped": pr["dropped"],
        "n_runs": pr["n_runs"], "reconfig_req": pr["reconfig_req"],
        "transit_req": pr["transit_req"],
        "transit_ok": pr["transit_ok"], "reconfig_mismatch": pr["reconfig_mismatch"],
        "collision_fallback_req": pr["collision_fallback_req"],
        "collision_fallback_ok": pr["collision_fallback_ok"],
        "final_traj": pr["final_traj"], "final_is_transit": pr["final_is_transit"],
        "entry": pr["entry"], "exit": pr["exit"], **gate,
    }


# =========================================================================
# Component joining: order → seam transits → stitch one continuous trajectory
# =========================================================================

def main() -> int:
    args = _parse_args()
    out_dir = args.output_dir if args.output_dir is not None else args.result.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("VERIFY GLNS TRAJECTORY (collision-aware, per component)")
    print("=" * 64)

    result = read_result_hdf5(args.result)
    meta = result["metadata"]
    object_name = args.object if args.object else _decode(meta["object"])
    object_position = np.asarray(_decode(meta["object_position"]), dtype=np.float64)
    object_quat = np.asarray(_decode(meta["object_quat_wxyz"]), dtype=np.float64)
    wd_m = float(_decode(meta["working_distance_m"]))
    reconfig_deg = float(_decode(meta["reconfig_threshold_deg"]))
    reconfig_rad = np.deg2rad(reconfig_deg)
    roll_augmented = bool(_decode(meta.get("roll_augmented", False)))

    # GLNS IK 가 풀린 바로 그 world 를 재현 — 결과에 박제된 배치를 config 에 주입한 뒤
    # plan_trajectory 의 build_collision_world 가 그 배치로 mesh 를 놓도록 한다(inspector 와 동일).
    config.TARGET_OBJECT["position"] = object_position
    config.TARGET_OBJECT["rotation"] = object_quat

    print(f"Result:   {args.result}")
    print(f"Object:   {object_name}  pos={object_position.tolist()}  quat(wxyz)={object_quat.tolist()}")
    print(f"WD:       {wd_m * 1000:.0f} mm   reconfig threshold: {reconfig_deg:.0f}°"
          f"   roll_augmented: {roll_augmented}")
    print(f"Output:   {out_dir}")
    print()

    robot_cfg = PT.resolve_robot_config(PT.ROBOT_CONFIG)
    world_config = PT.build_collision_world(object_name)
    home_q = np.asarray(config.ROBOT_START_STATE, dtype=np.float64)

    scan_joints = None
    if args.home_transitions_only:
        scan_npz = out_dir / "glns_trajectory_joined.npz"
        if not scan_npz.exists():
            print(f"HOME TRANSITIONS FAILED: scan trajectory not found: {scan_npz}")
            return 2
        if scan_npz.stat().st_mtime_ns < args.result.stat().st_mtime_ns:
            print("HOME TRANSITIONS FAILED: joined scan is older than the GLNS result; "
                  "run Plan scan motion first.")
            return 2
        with np.load(scan_npz) as scan_data:
            scan_joints = np.asarray(scan_data["joints"], dtype=np.float64)
    motion_planner = PT.build_reconfig_motion_planner(robot_cfg, world_config)

    if args.home_transitions_only:
        print("-" * 64)
        print("PLAN HOME TRANSITIONS (scan trajectory remains unchanged)")
        home_results = plan_home_transitions(
            scan_joints, home_q, robot_cfg=robot_cfg, world_config=world_config,
            wd_m=wd_m, spacing=args.spacing, reconfig_rad=reconfig_rad,
            enable_via_ladder=not args.no_via, motion_planner=motion_planner,
            out_dir=out_dir, transitions=args.home_transition,
        )
        all_home_ok = True
        for item in home_results:
            if not item["ok"]:
                all_home_ok = False
                print(f"  {item['label']}: FAILED (no collision-free route)")
                continue
            gate = item["gate"]
            print(f"  {item['label']}: OK [{item['route']}], "
                  f"{gate['n_waypoints']} waypoints, time={gate['total_time']:.1f}s")
            print(f"    CSV: {gate['csv']}")
        print("=" * 64)
        return 0 if all_home_ok else 2

    rows = []
    join_inputs = []   # 충돌-free 성분(joined 대상): final_traj/endpoints
    for component in result["components"]:
        cid = component["name"]
        status = component["status"]
        n_members = len(component["members"])
        print("-" * 64)
        print(f"[component {cid}] status={status}, {n_members} viewpoints")

        if status != "solved":
            print(f"    SKIP — {status}: {component.get('reason', '')}")
            rows.append((cid, status, n_members, None))
            continue
        if n_members < 2:
            print("    SKIP — fewer than 2 viewpoints (no path to verify)")
            rows.append((cid, "too_short", n_members, None))
            continue

        out_csv = out_dir / f"glns_trajectory_comp{cid}.csv"
        res = _verify_component(
            component, robot_cfg=robot_cfg, world_config=world_config,
            reconfig_rad=reconfig_rad, wd_m=wd_m,
            spacing=args.spacing, out_csv=out_csv, enable_via_ladder=not args.no_via,
            require_full_coverage=args.require_full_coverage,
            motion_planner=motion_planner,
        )
        rows.append((cid, "solved", n_members, res))

        if res.get("error"):
            print(f"    SKIP — {res['error']}")
            continue

        verdict = "OK (collision-free)" if res["collision_free"] else \
                  f"FAIL — {res['n_collisions']} colliding dense waypoints"
        drop_note = ""
        if res["dropped"]:
            split = f", split into {res['n_runs']} runs" if res["n_runs"] > 1 else ""
            drop_note = (f"\n    dropped {len(res['dropped'])} viewpoint(s) "
                         f"(scan-collision/transit-fail){split}, "
                         f"original idx: {res['dropped']}")
        time_note = (f", time={res['total_time']:.1f}s "
                     f"(scan={res['total_time'] - res['transit_time']:.1f}s, "
                     f"transit={res['transit_time']:.1f}s)") if res["collision_free"] else ""
        fallback_note = (f", scan-collision fallback "
                         f"{res['collision_fallback_ok']}/{res['collision_fallback_req']}"
                         if res["collision_fallback_req"] else "")
        print(f"    transit {res['transit_ok']}/{res['transit_req']} OK{fallback_note}, "
              f"coverage {res['covered']}/{res['M']}, "
              f"{res['n_waypoints']} waypoints{time_note}{drop_note}")
        print(f"    → {verdict}")
        if res["collision_free"]:
            print(f"    CSV: {res['csv']}")
            join_inputs.append({
                "cid": cid, "final_traj": res["final_traj"],
                "final_is_transit": res["final_is_transit"],
                "entry": res["entry"], "exit": res["exit"],
            })

    # --- 요약 ---
    print("=" * 64)
    print("SUMMARY")
    print("-" * 64)
    print(f"{'comp':>4} {'status':>10} {'vp':>4} {'cover':>6} {'drop':>5} "
          f"{'recfg':>6} {'transit':>8} {'coll':>5} {'time(s)':>8}")
    solved_total = 0
    solved_clean = 0
    any_dropped = False
    for cid, status, n_members, res in rows:
        if res is None:
            print(f"{cid:>4} {status:>10} {n_members:>4} {'-':>6} {'-':>5} "
                  f"{'-':>6} {'-':>8} {'-':>5} {'-':>8}")
            continue
        solved_total += 1
        if res["collision_free"]:
            solved_clean += 1
        if res["dropped"]:
            any_dropped = True
        coll = "0" if res["collision_free"] else str(res["n_collisions"])
        tstr = f"{res['total_time']:.1f}" if res["collision_free"] else "-"
        print(f"{cid:>4} {status:>10} {res['M']:>4} {res['covered']:>6} "
              f"{len(res['dropped']):>5} {res['reconfig_req']:>6} "
              f"{res['transit_ok']}/{res['transit_req']:<6} {coll:>5} {tstr:>8}")

    print("-" * 64)
    all_clean = solved_total > 0 and solved_clean == solved_total
    headline = "YES" if all_clean else "NO"
    print(f"All solved components collision-free: {headline} "
          f"({solved_clean}/{solved_total})")
    if any_dropped:
        print("NOTE: 일부 viewpoint 가 드롭됨 — 해당 성분의 GLNS 경로가 충돌-aware 이동에서 "
              "완전히 보존되지 못했다(연속 scan edge 충돌 또는 transit 계획 실패).")
    print("=" * 64)

    # --- 성분 연결: 하나의 연속 실행 궤적(glns_trajectory_joined.csv) ---
    if args.join:
        print("JOIN COMPONENTS → single continuous trajectory")
        print("-" * 64)
        if args.require_full_coverage and any_dropped:
            (out_dir / "glns_trajectory_joined.csv").unlink(missing_ok=True)
            (out_dir / "glns_trajectory_joined.npz").unlink(missing_ok=True)
            print("  FAIL — --require-full-coverage: skipped viewpoint detected; joined 미생성.")
            print("=" * 64)
            return 1
        if not join_inputs:
            print("  연결할 충돌-free 성분이 없음 — joined 미생성.")
            print("=" * 64)
        else:
            joined_csv = out_dir / "glns_trajectory_joined.csv"
            try:
                jr = join_components(
                    join_inputs, home_q, robot_cfg=robot_cfg, world_config=world_config,
                    wd_m=wd_m, spacing=args.spacing,
                    reconfig_rad=reconfig_rad, enable_via_ladder=not args.no_via,
                    home_bracket=args.home_bracket, order_strategy=args.order,
                    out_csv=joined_csv, motion_planner=motion_planner,
                )
            except SeamFailure as exc:
                print(f"  SEAM FAILED: {exc} — 가교 불가(via-home 포함). joined 미생성.")
                print("=" * 64)
                return 2
            hb = args.home_bracket
            seq = (["HOME"] if hb else []) + [f"comp{c}" for c in jr["order"]] + \
                  (["HOME"] if hb else [])
            print(f"  order({args.order}): {' → '.join(seq)}")
            print(f"  seams {jr['n_seams']}: routes={jr['seam_routes']}")
            g = jr["gate"]
            if g["collision_free"]:
                print(f"  → OK (collision-free), {g['n_waypoints']} waypoints, "
                      f"time={g['total_time']:.1f}s "
                      f"(scan={g['total_time'] - g['transit_time']:.1f}s, "
                      f"transit={g['transit_time']:.1f}s)")
                print(f"  CSV: {g['csv']}")
                print("=" * 64)
            else:
                print(f"  → FAIL — {g['n_collisions']} colliding dense waypoints; joined 미저장")
                print("=" * 64)
                return 1

    return 0 if all_clean else 1


if __name__ == "__main__":
    sys.exit(main())
