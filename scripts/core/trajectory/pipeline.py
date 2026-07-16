"""End-to-end DP trajectory planning pipeline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from common import config
from core.viewpoint import load_viewpoints_hdf5
from .ik import cluster_ik_solutions, solve_ik_multi_seed
from .motion import (
    densify_for_collision_check,
    find_colliding_interpolation_edges,
    interpolate_and_resample,
    plan_reconfig_transits,
)
from .poses import build_camera_poses, rot_to_quat_batch
from .robot import (
    _REUSE_HITS,
    _TIMINGS,
    _collision_sphere_buffer_summary,
    _tick,
    batch_collision_check,
    build_collision_world,
    compute_fk,
    resolve_robot_config,
)
from .selection import dp_optimal_path
from .settings import (
    CORNER_ANGLE_THRESHOLD_DEG,
    CORNER_MAX_SLOWDOWN,
    DEFAULT_SPACING_M,
    EE_ANGULAR_SPEED_DEG_S,
    EE_SPEED_MM_S,
    IK_BATCH_SIZE,
    MAX_JOINT_VEL_RAD_S,
    MIN_SEGMENT_DT_S,
    NUM_IK_SEEDS,
    RECONFIG_THRESHOLD_DEG,
    RESAMPLE_MODE,
    ROBOT_CONFIG,
    TRANSIT_RESAMPLE_SPACING_RAD,
)
from .storage import save_trajectory_csv
from .timing import compute_trajectory_times

def main():
    """CLI 진입점: 6단계 파이프라인(IK→대표해→DP→transit→resample/충돌→시간)을 실행해 궤적 CSV 저장."""
    parser = argparse.ArgumentParser(description="IK + DP + via-roll transit 기반 최적 궤적 생성")
    parser.add_argument("--object", type=str, required=True, help="Object name")
    parser.add_argument("--num-viewpoints", type=int, required=True, help="Number of viewpoints")
    parser.add_argument("--viewpoints", type=str, default=None,
                        help="Direct path to viewpoints.h5 (overrides --object/--num-viewpoints for loading)")
    parser.add_argument("--spacing", type=float, default=DEFAULT_SPACING_M,
                        help=f"EE arc-length resample spacing in meters (default: {DEFAULT_SPACING_M})")
    parser.add_argument("--output-suffix", type=str, default="dp",
                        help="Output file suffix (default: dp)")
    parser.add_argument("--object-position", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Override target object position in robot-base frame (meters). "
                             "If omitted, config.TARGET_OBJECT['position'] is used.")
    parser.add_argument("--object-quat", type=float, nargs=4, default=None,
                        metavar=("W", "X", "Y", "Z"),
                        help="Override target object orientation quaternion (w x y z). "
                             "If omitted, config.TARGET_OBJECT['rotation'] is used.")
    args = parser.parse_args()

    if args.spacing <= 0.0:
        parser.error("--spacing must be > 0")

    # 물체별 기본 배치(config.OBJECT_PLACEMENTS)를 먼저 반영. CLI override 가 그 뒤라 우선한다.
    if config.apply_object_placement(args.object):
        print(f"  Per-object placement '{args.object}': pos={config.TARGET_OBJECT['position']}, "
              f"quat={config.TARGET_OBJECT['rotation']}")

    # Object pose override (e.g. moved via the Isaac Sim viewport gizmo). Mutating
    # config.TARGET_OBJECT in place propagates to build_camera_poses (local→world EE
    # pose transform) and build_collision_world (mesh placement), which read it at
    # call time. Safe because this script runs as a one-shot subprocess.
    if args.object_position is not None:
        config.TARGET_OBJECT["position"] = np.array(args.object_position, dtype=np.float64)
        print(f"  Object position override (robot frame): {args.object_position}")
    if args.object_quat is not None:
        config.TARGET_OBJECT["rotation"] = np.array(args.object_quat, dtype=np.float64)
        print(f"  Object rotation override (w,x,y,z): {args.object_quat}")

    _t = time.time()
    # [1] Load viewpoints
    print("[1/6] Loading viewpoints...")
    h5_path = Path(args.viewpoints) if args.viewpoints \
        else config.get_viewpoint_path(args.object, args.num_viewpoints)
    viewpoint = load_viewpoints_hdf5(h5_path)
    positions = viewpoint.positions
    normals = viewpoint.normals
    path_order = viewpoint.path_order
    cluster_id = viewpoint.cluster_id
    wd_m = viewpoint.working_distance_m
    print(f"  Loaded from {h5_path}")
    print(f"  {len(positions)} viewpoints, working distance: {wd_m*1000:.1f} mm")

    # path_order 순서로 정렬 (cluster_id도 함께)
    if path_order is not None:
        sorted_idx = np.argsort(path_order)
        positions = positions[sorted_idx]
        normals = normals[sorted_idx]
        if cluster_id is not None:
            cluster_id = cluster_id[sorted_idx]

    # [2] Build camera poses
    print("[2/6] Building camera poses...")
    world_poses = build_camera_poses(positions, normals, wd_m)
    N = len(world_poses)

    positions_np = world_poses[:, :3, 3]
    quats_np = rot_to_quat_batch(world_poses[:, :3, :3])  # (w, x, y, z)
    print(f"  {N} camera poses built")
    _tick("load+poses", _t)

    # [3] Phase 1: Multi-seed IK
    print("[3/6] Phase 1 — Multi-seed IK...")
    _t = time.time()
    world_config = build_collision_world(args.object)
    robot_cfg = resolve_robot_config(ROBOT_CONFIG)
    print(f"  Robot YAML: urdf={robot_cfg['robot_cfg']['kinematics']['urdf_path']}")
    collision_buffer = _collision_sphere_buffer_summary(robot_cfg)
    if collision_buffer:
        print(f"  Collision sphere buffer: {collision_buffer} (from robot YAML)")
    _tick("build_world+robotcfg", _t)

    _t = time.time()
    all_solutions, all_success = solve_ik_multi_seed(
        robot_cfg, world_config, positions_np, quats_np,
        num_seeds=NUM_IK_SEEDS, batch_size=IK_BATCH_SIZE,
    )
    _tick("phase1_ik_total", _t)

    # [4] Phase 2 + 3: 대표해 추출 → DP
    print("[4/6] Phase 2 — per-viewpoint 대표해 (greedy dedup)...")
    representatives = cluster_ik_solutions(all_solutions, all_success)

    # wrist_3 고정을 DP '이전'에 적용 — wrist_3는 어차피 0으로 잠그며(검사 시 광축 roll
    # 무관), IK는 free wrist_3로 풀려 후보마다 wrist_3가 제각각이다. DP의 reconfig 비용은
    # 6-DoF L∞라, 5-DoF로는 연속인 해가 '버려질' wrist_3 차이 때문에 reconfig로 오판돼
    # DP가 엉뚱한 분기를 고른다. 미리 0으로 잠그면 6-DoF L∞ = 5-DoF L∞ 가 되어 DP가
    # 실제 최종 자세 기준으로 연속 해를 직접 고른다.
    wrist3_fixed = config.ROBOT_START_STATE[-1]
    for reps in representatives:
        if len(reps) > 0:
            reps[:, -1] = wrist3_fixed
    print(f"  Locked wrist_3 at {np.rad2deg(wrist3_fixed):.1f}° (pre-DP)")

    reconfig_rad = np.deg2rad(RECONFIG_THRESHOLD_DEG)

    # 충돌하는 대표 해 제거 (DP는 충돌을 안 보므로). 최종검사와 동일한 batch_collision_check로
    # wrist_3 잠금 후 자세를 검사 → 충돌 자세를 후보에서 빼 DP가 충돌-free만 고르게 한다.
    # 충돌-free가 0개가 된 viewpoint는 아래 empty-drop이 unreachable로 처리한다.
    _t = time.time()
    spans = []   # (vp, flat_start, count)
    flat = []
    for i, r in enumerate(representatives):
        if len(r) > 0:
            spans.append((i, sum(len(f) for f in flat), len(r)))
            flat.append(r)
    if flat:
        isc_flat, _ = batch_collision_check(
            np.concatenate(flat, axis=0), robot_cfg, world_config,
        )
        n_removed, n_emptied = 0, 0
        for vp, start, cnt in spans:
            free_mask = ~isc_flat[start:start + cnt]
            removed = cnt - int(free_mask.sum())
            if removed > 0:
                n_removed += removed
                representatives[vp] = representatives[vp][free_mask]
                if len(representatives[vp]) == 0:
                    n_emptied += 1
        if n_removed > 0:
            print(f"  Collision-filtered reps: removed {n_removed} colliding solutions "
                  f"(margin={config.COLLISION_MARGIN*1000:.0f}mm), "
                  f"{n_emptied} viewpoints emptied → unreachable")
        else:
            print(f"  Collision-filtered reps: 0 colliding (all candidates collision-free)")
    _tick("collision_filter_reps", _t)

    # 못 가는(empty) viewpoint 제거 — IK 해가 없거나 충돌 필터로 비워진 viewpoint는
    # carry-forward로 메우지 않고 경로에서 뺀다.
    # orig_idx: 남은 viewpoint의 '원본' 인덱스(드롭 후에도 로그를 원래 번호로 표기하기 위함).
    orig_idx = np.arange(len(representatives))
    keep = np.array([len(r) > 0 for r in representatives], dtype=bool)
    n_dropped_empty = int((~keep).sum())
    if n_dropped_empty > 0:
        dropped_list = orig_idx[~keep].tolist()
        print(f"  Dropping {n_dropped_empty} unreachable (empty) viewpoints "
              f"(no IK solution): {dropped_list}")
        representatives = [r for r, k in zip(representatives, keep) if k]
        all_solutions = all_solutions[keep]
        all_success = all_success[keep]
        if cluster_id is not None:
            cluster_id = cluster_id[keep]
        orig_idx = orig_idx[keep]
        if len(representatives) < 2:
            raise RuntimeError(
                f"도달 가능한 viewpoint가 {len(representatives)}개뿐입니다 — "
                "물체 배치/작업거리(WD)를 조정해 reachability를 높여야 합니다."
            )

    print("[5/6] Phase 3 — DP optimal path...")
    _t = time.time()
    selected, _, stats = dp_optimal_path(representatives, reconfig_rad)
    _tick("dp", _t)

    jumps = np.max(np.abs(np.diff(selected, axis=0)), axis=1)
    is_reconfig = jumps > reconfig_rad

    # 클러스터 간/내 reconfig 분석
    if cluster_id is not None:
        is_inter_cluster = cluster_id[:-1] != cluster_id[1:]

        n_inter = int(is_inter_cluster.sum())
        n_intra_transition = int((~is_inter_cluster).sum())
        rc_inter = int((is_reconfig & is_inter_cluster).sum())
        rc_intra = int((is_reconfig & ~is_inter_cluster).sum())

        print(f"\n  Reconfig analysis:")
        print(f"    Inter-cluster: {rc_inter}/{n_inter} transitions "
              f"({100 * rc_inter / max(n_inter, 1):.0f}%) — expected")
        print(f"    Intra-cluster: {rc_intra}/{n_intra_transition} transitions "
              f"({100 * rc_intra / max(n_intra_transition, 1):.0f}%) — should be 0")

        if rc_intra > 0:
            _jn = ["pan", "lift", "elbow", "w1", "w2", "w3"]
            intra_reconfig_idx = np.where(is_reconfig & ~is_inter_cluster)[0]
            for idx in intra_reconfig_idx:
                jump_deg = np.rad2deg(jumps[idx])
                cid = cluster_id[idx]
                # 어떤 joint가 튀는지 (per-joint |Δ| deg)
                dq_deg = np.rad2deg(np.abs(selected[idx + 1] - selected[idx]))
                worst = int(np.argmax(dq_deg))
                per_joint = " ".join(f"{n}={d:.0f}" for n, d in zip(_jn, dq_deg))
                # 연속 해가 IK pool에 있었는가? (wrist_3 잠금이므로 5-DoF L∞로 비교)
                def _min_pool_linf(vp, ref):
                    cand = all_solutions[vp][all_success[vp]]
                    if len(cand) == 0:
                        return None
                    return float(np.rad2deg(np.min(np.max(np.abs(cand[:, :5] - ref[:5]), axis=1))))
                d_next = _min_pool_linf(idx + 1, selected[idx])      # vp(idx+1) ~ selected[idx]
                d_prev = _min_pool_linf(idx, selected[idx + 1])      # vp(idx)   ~ selected[idx+1]
                thr_deg = np.rad2deg(reconfig_rad)
                def _verdict(d):
                    if d is None:
                        return "no-IK"
                    return f"{d:.0f}° ({'POOL-HAS-CONT' if d <= thr_deg else 'no-cont-in-pool'})"
                o0, o1 = int(orig_idx[idx]), int(orig_idx[idx + 1])
                print(f"      viewpoint {o0}→{o1} (cluster {cid}): jump {jump_deg:.1f}° "
                      f"[worst={_jn[worst]}]  Δ: {per_joint}")
                print(f"          pool-continuity: vp{o1}~sel[{o0}]={_verdict(d_next)}, "
                      f"vp{o0}~sel[{o1}]={_verdict(d_prev)}  (thr={thr_deg:.0f}°)")

    # Phase 4: BatchMotionPlanner transit at reconfig points and at otherwise-small
    # joint interpolation edges that collide.  The latter used to be discovered only
    # in Phase 5 and could split the run/drop viewpoints without trying MotionGen.
    reconfig_indices = np.where(is_reconfig)[0] if cluster_id is not None else np.array([], dtype=int)
    scan_edge_indices = np.where(~is_reconfig)[0]
    collision_fallback_indices = find_colliding_interpolation_edges(
        selected, scan_edge_indices, robot_cfg, world_config,
    )
    motion_indices = np.union1d(reconfig_indices, collision_fallback_indices).astype(np.int64)
    transit_segments = {}
    _t = time.time()
    if len(collision_fallback_indices):
        labels = [f"{int(orig_idx[i])}→{int(orig_idx[i + 1])}"
                  for i in collision_fallback_indices]
        print(f"\n[Phase 4] Scan interpolation collision: "
              f"{len(collision_fallback_indices)} edge(s) → MotionGen fallback "
              f"[{', '.join(labels)}]")
    if len(motion_indices) > 0:
        print(f"\n[Phase 4] BatchMotionPlanner transit for {len(motion_indices)} edges "
              f"({len(reconfig_indices)} reconfig + "
              f"{len(collision_fallback_indices)} scan-collision fallback)...")
        transit_segments, transit_stats = plan_reconfig_transits(
            selected, motion_indices, robot_cfg, world_config, label_idx=orig_idx, wd_m=wd_m,
        )
        # 안전망: transit planner가 중간에 wrist_3를 흔들었을 수 있으므로 강제 고정.
        # 단 via-roll/via-tilt 는 의도적으로 rolled 중간자세(wrist_3 가변)를 쓰므로 덮어쓰면
        # 가교가 깨진다 → scan config 가 양 끝인 direct/via-home route 에만 적용.
        _routes = {s["idx"]: s.get("route") for s in transit_stats if s.get("success")}
        for idx in transit_segments:
            if _routes.get(idx) in ("direct", "via-home"):
                transit_segments[idx][:, -1] = wrist3_fixed
    _tick("phase4_transit_total", _t)

    # Phase 5: Uniform resample + collision check
    print(f"\n[Phase 5] Interpolation + uniform resample (mode={RESAMPLE_MODE})...")
    _t = time.time()
    final_traj, final_is_transit, skipped_vps, runs_info = interpolate_and_resample(
        selected, transit_segments, robot_cfg,
        mode=RESAMPLE_MODE, spacing=args.spacing,
        reconfig_threshold_rad=reconfig_rad, world_scene=world_config,
    )
    _tick("phase5_resample", _t)
    skipped_orig = [int(orig_idx[i]) for i in skipped_vps]   # 원본 viewpoint 번호로 표기
    if len(runs_info["runs"]) > 1:
        kl = runs_info["kept"][2]
        print(
            f"  WARNING: 전이 불가(transit 실패/스캔 충돌)로 경로가 "
            f"{len(runs_info['runs'])}개 run으로 끊김 "
            f"→ 가장 긴 run ({kl}개 viewpoint) 채택, "
            f"viewpoint {len(skipped_vps)}개 드롭(원본 번호): {skipped_orig}"
        )
    elif skipped_vps:
        print(
            f"  WARNING: 전이 불가(transit 실패/스캔 충돌)로 아웃라이어 viewpoint "
            f"{len(skipped_vps)}개 건너뜀(원본 번호): {skipped_orig}"
        )
    if RESAMPLE_MODE == "ee":
        spacing_desc = f"EE spacing={args.spacing*1000:.1f} mm"
    else:
        spacing_desc = f"joint spacing={np.rad2deg(args.spacing):.2f}°"
    n_transit_wp = int(np.asarray(final_is_transit).sum())
    print(f"  Resampled: {len(final_traj)} waypoints ({spacing_desc}, "
          f"scan={len(final_traj) - n_transit_wp}, "
          f"transit={n_transit_wp} @ joint spacing {np.rad2deg(TRANSIT_RESAMPLE_SPACING_RAD):.1f}°)")

    # Collision check
    print("  Collision check...")
    _t = time.time()
    collision_traj = densify_for_collision_check(final_traj)
    if len(collision_traj) != len(final_traj):
        print(
            f"  Collision check densified: {len(final_traj)} → {len(collision_traj)} "
            f"waypoints (max joint step="
            f"{config.COLLISION_ADAPTIVE_MAX_JOINT_STEP_DEG:.3f}°"
            + (", excluding wrist_3 metric" if config.COLLISION_INTERP_EXCLUDE_LAST_JOINT else "")
            + ")"
        )
    _tick("phase5_densify", _t)
    _t = time.time()
    is_collision, n_collisions = batch_collision_check(
        collision_traj, robot_cfg, world_config,
    )
    _tick("phase5_collision_check", _t)
    if n_collisions > 0:
        collision_pct = 100 * n_collisions / len(collision_traj)
        raise RuntimeError(
            f"Collision validation failed: {n_collisions}/{len(collision_traj)} "
            f"dense waypoints in collision ({collision_pct:.1f}%). "
            "Refusing to save trajectory."
        )
    else:
        print(f"  No collisions detected ({len(collision_traj)} dense waypoints)")

    # FK + 저장
    _t = time.time()
    ee_positions, ee_quaternions = compute_fk(final_traj, robot_cfg)
    print(f"  Computed FK for {len(final_traj)} waypoints")

    traj_times, time_stats = compute_trajectory_times(
        final_traj, ee_positions, ee_quaternions,
        ee_speed_m_s=EE_SPEED_MM_S / 1000.0,
        ee_angular_speed_rad_s=np.deg2rad(EE_ANGULAR_SPEED_DEG_S),
        max_joint_vel_rad_s=MAX_JOINT_VEL_RAD_S,
        min_segment_dt=MIN_SEGMENT_DT_S,
        corner_angle_threshold_rad=np.deg2rad(CORNER_ANGLE_THRESHOLD_DEG),
        corner_max_slowdown=CORNER_MAX_SLOWDOWN,
        is_transit=final_is_transit,
    )
    scan_time = time_stats['total_time'] - time_stats['transit_time']
    print(f"  Time profile: total={time_stats['total_time']:.1f}s "
          f"(scan={scan_time:.1f}s, transit={time_stats['transit_time']:.1f}s "
          f"in {time_stats['n_transit_segments']} seg), "
          f"max scan EE={time_stats['max_linear_speed_mm_s']:.1f} mm/s, "
          f"max scan rot={time_stats['max_angular_speed_deg_s']:.1f} deg/s, "
          f"max joint={time_stats['max_joint_speed_rad_s']:.2f} rad/s, "
          f"corners={time_stats['n_slow_segments']} seg "
          f"(max angle={time_stats['max_corner_angle_deg']:.1f}°, "
          f"slowdown={time_stats['max_slowdown']:.2f}x)")
    _tick("fk+time", _t)

    _t = time.time()
    traj_dir = config.get_trajectory_path(args.object, args.num_viewpoints, "dummy").parent
    traj_dir.mkdir(parents=True, exist_ok=True)

    suffix = args.output_suffix
    spacing_str = f"{args.spacing:.3f}".replace(".", "")  # 0.010 → "0010", 0.050 → "0050"
    ee_speed_str = f"{EE_SPEED_MM_S:.0f}"
    ang_speed_str = f"{EE_ANGULAR_SPEED_DEG_S:.0f}"
    joint_vel_str = f"{MAX_JOINT_VEL_RAD_S:.2f}".replace(".", "p")
    tag = f"{suffix}_{RESAMPLE_MODE}_s{spacing_str}_eev{ee_speed_str}mms_av{ang_speed_str}dps_jv{joint_vel_str}"
    corner_thresh_str = f"{CORNER_ANGLE_THRESHOLD_DEG:.0f}"
    corner_slow_str = f"{CORNER_MAX_SLOWDOWN:.1f}".replace(".", "p")
    tag = f"{tag}_corner{corner_thresh_str}d_x{corner_slow_str}"

    csv_path = str(traj_dir / f"trajectory_{tag}.csv")
    save_trajectory_csv(
        final_traj, ee_positions, ee_quaternions, csv_path,
        times=traj_times,
    )
    # NPZ sidecar (dense playback): same {joints, ee_positions, is_transit, times}
    # schema verify_glns_trajectory emits, so trajectory_studio plays DP and GLNS
    # back identically (transit/scan coloring). CSV stays the publish artifact.
    npz_path = str(Path(csv_path).with_suffix(".npz"))
    np.savez(
        npz_path,
        joints=final_traj, ee_positions=ee_positions,
        is_transit=final_is_transit, times=traj_times,
    )
    print(f"  NPZ saved to {npz_path}")
    _tick("save", _t)

    n_transit_ok = len(transit_segments)
    n_collision_fallback_ok = sum(
        int(i) in transit_segments for i in collision_fallback_indices
    )
    covered = runs_info["kept"][2]
    print(f"\nDone. coverage={covered}/{N} viewpoints "
          f"(unreachable dropped={n_dropped_empty}, transit-split dropped={len(skipped_vps)}), "
          f"reconfigs={stats['n_reconfigs']} (inter={rc_inter}, intra={rc_intra}), "
          f"transit={n_transit_ok}/{len(motion_indices)} OK "
          f"(scan-collision fallback={n_collision_fallback_ok}/"
          f"{len(collision_fallback_indices)}), "
          f"collisions={n_collisions}, final={len(final_traj)} waypoints")

    # === Timing breakdown (생성시간 분해 — 레버 선택용) ===
    # phase1_ik_total = ik_build + ik_solve, phase4_transit_total = transit_build_warmup
    # + transit_direct + transit_pending (+ via_ik_build) 라 합산 중복 → 'sub' 표시로 제외 표기.
    # kin_build/cc_build(1x)는 다른 phase(collision_filter_reps·fk+time 등) 안에서 일어나므로 sub.
    _SUBS = {"ik_build", "ik_solve", "transit_build_warmup", "transit_direct",
             "transit_pending", "via_ik_build", "kin_build(1x)", "cc_build(1x)"}
    _total = sum(s for lbl, s in _TIMINGS if lbl not in _SUBS)
    print("\n=== Timing breakdown (top-level phases; sub = 중복분해, TOTAL 미포함) ===")
    for label, sec in sorted(_TIMINGS, key=lambda kv: kv[1], reverse=True):
        tag_sub = " (sub)" if label in _SUBS else ""
        pct = 100 * sec / _total if _total > 0 else 0.0
        print(f"  {label:<22} {sec:7.2f}s  {pct:5.1f}%{tag_sub}")
    print(f"  {'-' * 44}")
    print(f"  {'TOTAL (measured)':<22} {_total:7.2f}s")
    _cc = next((s for lbl, s in _TIMINGS if lbl == 'cc_build(1x)'), 0.0)
    _kn = next((s for lbl, s in _TIMINGS if lbl == 'kin_build(1x)'), 0.0)
    print(f"  Solver reuse: collision-checker {_REUSE_HITS['cc']} hits (~{_cc:.2f}s/build saved each), "
          f"kinematics {_REUSE_HITS['kin']} hits (~{_kn:.2f}s/build saved each)")
    print("  (이 합 vs `/usr/bin/time -v` wall 차이 = Python import + CUDA init 등 미계측 고정비)")


if __name__ == "__main__":
    main()
