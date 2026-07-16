"""Collision-free transit planning, interpolation, and resampling."""

from __future__ import annotations

import time

import numpy as np
import torch
from curobo.batch_motion_planner import BatchMotionPlanner
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.motion_planner import MotionPlannerCfg
from curobo.types import GoalToolPose, JointState, Pose
from scipy.spatial.transform import Rotation

from common import config
from .ik import normalize_joints
from .poses import rot_to_quat_batch
from .robot import _tick, batch_collision_check, compute_fk
from .settings import (
    BIG_BASE_RECONFIG_RAD,
    RECONFIG_THRESHOLD_DEG,
    ROLL_VARIANT_DEG,
    TILT_VARIANT_AZ,
    TILT_VARIANT_PHI,
    TRANSIT_BATCH_SIZE,
    TRANSIT_ENABLE_GRAPH_ATTEMPT,
    TRANSIT_FAIL_SKIP_MAX,
    TRANSIT_MAX_ATTEMPTS,
    TRANSIT_RESAMPLE_SPACING_RAD,
    VIA_ROLL_IK_SEEDS,
    VIA_ROLL_MAX_REPS,
)

def build_reconfig_motion_planner(robot_cfg, world_scene):
    """Build and warm one BatchMotionPlanner reusable across transit calls."""
    cache = {
        "obb": max(1, len(world_scene.cuboid)),
        "mesh": max(1, len(world_scene.mesh)),
    }
    _t_build = time.time()
    cfg = MotionPlannerCfg.create(
        robot=robot_cfg,
        collision_cache=cache,
        use_cuda_graph=False,
        max_batch_size=TRANSIT_BATCH_SIZE,
    )
    planner = BatchMotionPlanner(cfg)
    planner.update_world(world_scene)
    graph_on = TRANSIT_ENABLE_GRAPH_ATTEMPT <= TRANSIT_MAX_ATTEMPTS
    print(f"    Warming up BatchMotionPlanner (batch={TRANSIT_BATCH_SIZE}, "
          f"graph={'on' if graph_on else 'off'})...")
    planner.warmup(enable_graph=graph_on, num_warmup_iterations=2)
    _tick("transit_build_warmup", _t_build)
    return planner


def plan_reconfig_transits(
    selected, reconfig_indices, robot_cfg, world_scene, label_idx=None, wd_m=None,
    lock_wrist3=True, enable_via_ladder=True, motion_planner=None,
):
    """Reconfig 지점마다 BatchMotionPlanner joint-to-joint planning 수행.

    사다리: direct → via-roll → via-tilt → via-home. direct 는 전 경계를 한 배치로, via-roll/
    via-tilt 의 경계내 후보 leg(scan→변형·변형 bridge·변형→scan)도 한 배치로 GPU 병렬 계획해
    경계내 순차 탐색을 제거한다. via-roll/via-tilt 는 경계 양 끝 scan 자세를 광축 둘레로 roll/tilt
    한 중간자세를 경유해 가교한다(scan config 보존, ripple 없음).

    Args:
        selected: (N, 6) DP로 선택된 joint trajectory
        reconfig_indices: reconfig이 발생하는 transition 인덱스 배열
        robot_cfg: cuRobo robot config (dict)
        world_scene: Scene (충돌 세계)
        label_idx: (선택) filtered→원본 viewpoint 인덱스 매핑. 로그 표기용.
        wd_m: (선택) working distance [m]. via-tilt(orbit) 에만 필요.
        motion_planner: optional warmed BatchMotionPlanner to reuse across calls.

    Returns:
        transit_segments: dict {idx: (T, 6) transit trajectory} — 성공한 것만
        transit_stats: list of dicts
    """
    def _lbl(i):
        return int(label_idx[i]) if label_idx is not None else int(i)
    planner = (motion_planner if motion_planner is not None
               else build_reconfig_motion_planner(robot_cfg, world_scene))

    n_dof = selected.shape[-1]
    bs = planner.batch_size

    # HOME: 직접 transit 실패 시 경유할 안전 retract 자세. wrist_3 가 이미 lock 값과 동일.
    # (batch padding 의 빈 슬롯 채움에도 사용 — start==goal==home 은 자명 성공이라 무시된다.)
    home_q = np.asarray(config.ROBOT_START_STATE, dtype=np.float64)

    def _plan_chunk(starts, goals):
        """≤batch_size 개 (q_from,q_to) 를 한 번의 plan_cspace 로 GPU 병렬 계획. 정렬된 waypoints|None.

        batch_size 로 padding(빈 슬롯 start=goal=home → 자명 성공, 무시). plan_cspace 결과는
        seed 가 best 1 개로 collapse 된 (B,1,T,dof)/(B,1) → per-slot 추출.
        """
        K = len(starts)
        ns = np.tile(home_q, (bs, 1))
        ng = ns.copy()
        for slot in range(K):
            ns[slot] = starts[slot]
            ng[slot] = goals[slot]
        s = JointState.from_position(
            torch.tensor(ns, device="cuda:0", dtype=torch.float32),
            joint_names=planner.joint_names)
        g = JointState.from_position(
            torch.tensor(ng, device="cuda:0", dtype=torch.float32),
            joint_names=planner.joint_names)
        r = planner.plan_cspace(
            g, s, max_attempts=TRANSIT_MAX_ATTEMPTS,
            enable_graph_attempt=TRANSIT_ENABLE_GRAPH_ATTEMPT, success_ratio=1.0)
        out = [None] * K
        if r is None:
            return out
        succ = r.success                            # (bs, S)
        pos = r.interpolated_trajectory.position    # (bs, S, T, dof)
        last = r.interpolated_last_tstep            # (bs, S)
        for slot in range(K):
            ok_seeds = succ[slot].nonzero(as_tuple=False).ravel()
            if len(ok_seeds) == 0:
                continue
            j = int(ok_seeds[0])
            L = int(last[slot, j])
            wp = pos[slot, j, :L + 1, :n_dof].detach().cpu().numpy()
            if len(wp) >= 2:
                out[slot] = wp
        return out

    def _plan_batch(starts, goals):
        """임의 개수를 batch_size chunk 로 나눠 계획(early-stop 불필요한 direct/via-home 용)."""
        out = []
        for c0 in range(0, len(starts), bs):
            out.extend(_plan_chunk(starts[c0:c0 + bs], goals[c0:c0 + bs]))
        return out

    # ----- via-roll/via-tilt 인프라 (lazy: 직접 transit 실패가 생겨 처음 필요할 때만 빌드) -----
    # 카메라 광축 ≈ wrist_3 축 → wrist_3 = 광축 roll = 검사 무손실 redundant DOF. scan config
    # (selected[i]) 는 그대로 두고 transit 중간자세에만 roll/tilt 를 줘 direct 가교를 푼다.
    # 중간자세는 스캔되지 않으므로 시야 손실 없음. (compute_fk 로 endpoint 광축을 얻어 roll/tilt)
    _via = {"ready": False, "ik": None, "tool": None, "pose_of": None}
    _vcache = {}                       # (sel_idx, mode) -> [collision-free config, ...]

    def _ensure_via():
        """via-roll/tilt용 IK solver + endpoint FK pose를 최초 1회만 lazy 빌드(미사용 시 비용 0)."""
        if _via["ready"]:
            return
        _t_build = time.time()
        ep_set = sorted(set([int(i) for i in reconfig_indices]
                            + [int(i) + 1 for i in reconfig_indices]))
        ee_pos, ee_quat = compute_fk(selected[ep_set], robot_cfg)   # (M,3), (M,4 xyzw)
        _via["pose_of"] = {k: (Rotation.from_quat(ee_quat[j]).as_matrix(), ee_pos[j].copy())
                           for j, k in enumerate(ep_set)}
        ik_cache = {"obb": max(1, len(world_scene.cuboid)), "mesh": max(1, len(world_scene.mesh))}
        ik_cfg = InverseKinematicsCfg.create(
            robot=robot_cfg, scene_model={}, self_collision_check=True,
            num_seeds=VIA_ROLL_IK_SEEDS,
            max_batch_size=max(len(ROLL_VARIANT_DEG), len(TILT_VARIANT_PHI) * len(TILT_VARIANT_AZ)),
            use_cuda_graph=False, collision_cache=ik_cache)
        ik = InverseKinematics(ik_cfg)
        ik.update_world(world_scene)
        _via["ik"], _via["tool"] = ik, ik.tool_frames[0]
        _via["ready"] = True
        _tick("via_ik_build", _t_build)

    def _variant_configs(k, mode):
        """sel index k 의 변형(roll/tilt) IK 후보 — 광축 둘레 회전/기울임 타깃을 IK로 풀어
        collision-free 분기를 캐시해 반환. roll=위치 불변(시야 무손실), tilt=표면점 orbit(시야 ±φ)."""
        if (k, mode) in _vcache:
            return _vcache[(k, mode)]
        _ensure_via()
        R, p = _via["pose_of"][k]
        if mode == "roll":                                     # 광축(+z) 둘레 회전: 위치 불변
            targets = [(R @ Rotation.from_euler("z", deg, degrees=True).as_matrix(), p.copy())
                       for deg in ROLL_VARIANT_DEG]
        else:                                                  # 표면점 중심 orbit tilt: WD 유지, 광축 ±φ
            surf = p + R[:, 2] * wd_m
            targets = []
            for phi in TILT_VARIANT_PHI:
                for az in TILT_VARIANT_AZ:
                    u = R @ np.array([np.cos(np.deg2rad(az)), np.sin(np.deg2rad(az)), 0.0])
                    u = u / np.linalg.norm(u)
                    Rp = Rotation.from_rotvec(u * np.deg2rad(phi)).as_matrix() @ R
                    targets.append((Rp, surf - Rp[:, 2] * wd_m))
        Rs = np.stack([t[0] for t in targets])
        ps = np.stack([t[1] for t in targets])
        bp = torch.tensor(ps, device="cuda:0", dtype=torch.float32)
        bq = torch.tensor(rot_to_quat_batch(Rs), device="cuda:0", dtype=torch.float32)
        res = _via["ik"].solve_pose(
            GoalToolPose.from_poses({_via["tool"]: Pose(position=bp, quaternion=bq)}, num_goalset=1),
            return_seeds=VIA_ROLL_IK_SEEDS)
        sols = res.js_solution.position.detach().cpu().numpy()
        if sols.shape[-1] != n_dof:
            sols = sols[..., :n_dof]
        sols = normalize_joints(sols)
        flat = sols[res.success.detach().cpu().numpy()]        # (K,6) 성공 해만
        kept = []
        if len(flat):
            isc, _ = batch_collision_check(flat, robot_cfg, world_scene)
            tol = np.deg2rad(8.0)
            for s in flat[~isc]:
                if all(np.max(np.abs(s - q)) > tol for q in kept):
                    kept.append(s)
                if len(kept) >= VIA_ROLL_MAX_REPS:
                    break
        _vcache[(k, mode)] = kept
        return kept

    _w3lock = float(home_q[-1])   # scan/HOME wrist_3 잠금값 (= config.ROBOT_START_STATE[-1])

    def _transit_safe(wp):
        """transit 후보가 '최종 validation 과 동일 기준'(densify + COLLISION_MARGIN)으로 충돌-free 인지.
        plan_cspace 의 success 는 자기 trajopt 해상도·soft-cost tolerance 기준이라, 우리 densify 해상도·
        margin 에선 waypoint 사이를 스치며 관통할 수 있다(특히 물체 근처를 지나는 reconfig transit).
        그래서 채택 전에 같은 검사를 걸어 통과한 후보만 쓴다 → 최종 collision 검증이 transit 때문에
        실패하는 일을 없애고, batch 변동에도 '항상 유효한' transit 만 선택(plan_cspace success 맹신 제거)."""
        if wp is None or len(wp) < 2:
            return False
        _, ncol = batch_collision_check(densify_for_collision_check(wp), robot_cfg, world_scene)
        return ncol == 0

    def _via_variant(idx, mode):
        """변형 중간자세 경유 3-leg transit. 후보 leg 를 joint-closest 순으로 batch chunk 씩 풀고
        가교쌍이 나오면 즉시 멈춘다 — easy 경계는 1 chunk 로 끝나고(가까운 쌍이 바로 가교), hard
        경계만 더 깊이 탐색한다(short-circuit). 선택 규칙 = joint-최근접 가교쌍 중 **densify-검증
        (최종 collision 검증과 동일 기준) 통과**한 첫 쌍 — plan_cspace success 만으론 grazing 일 수 있어
        채택 전 `_transit_safe` 로 거른다(grazing 후보는 거부하고 다음 후보로).

        scan↔변형 leg(소수, ≤(MAX_REPS)*2)를 job 목록 앞에 둬 첫 chunk 에 모두 들어가게 한다
        (이후 어느 bridge chunk 든 leg 가 이미 있어 가교 판정 가능). VIA_ROLL_MAX_REPS 가 커져
        leg 수가 batch_size 를 넘으면 이 가정이 깨진다(현재 ≤12 ≪ 32).

        NOTE(2026-06-29): pending 경계들을 한 배치로 pool 하는 cross-boundary 배치를 시도했으나,
        plan_cspace 가 batch-composition 의존이라 같은 leg/bridge 문제가 섞인 배치에서 다른(유효하나
        다른) 궤적·성공집합을 내 transit 선택이 임의로 바뀌고 실행 cycle 이 흔들림(curved +25%) → 폐기,
        경계당 1 chunk 순차 유지. 자세히 [[plan-trajectory-cross-boundary-batch-rejected]].
        """
        qi, qj = selected[idx], selected[idx + 1]
        poolA = [qi] + _variant_configs(idx, mode)
        poolB = [qj] + _variant_configs(idx + 1, mode)
        if len(poolA) == 1 and len(poolB) == 1:        # 변형 후보 없음 → 가교 불가
            return None

        order = sorted(((float(np.max(np.abs(poolA[ia] - poolB[ib]))), ia, ib)
                        for ia in range(len(poolA)) for ib in range(len(poolB))
                        if not (ia == 0 and ib == 0)), key=lambda t: t[0])
        jobs = []
        for ia in range(1, len(poolA)):                # leg1: qi→A' (먼저 — 첫 chunk 에 전부)
            jobs.append((qi, poolA[ia], "leg1", ia))
        for ib in range(1, len(poolB)):                # leg3: B'→qj
            jobs.append((poolB[ib], qj, "leg3", ib))
        for _, ia, ib in order:                        # bridge: A'→B' (joint-closest 순)
            jobs.append((poolA[ia], poolB[ib], "bridge", (ia, ib)))

        leg1, leg3, bridge = {}, {}, {}
        tested = set()
        rejected = set()   # 미완성 or densify-검증 실패(grazing)한 가교쌍 — 다음 chunk 재검사 방지
        for c0 in range(0, len(jobs), bs):
            sub = jobs[c0:c0 + bs]
            res = _plan_chunk([j[0] for j in sub], [j[1] for j in sub])
            for j, wp in zip(sub, res):
                if j[2] == "leg1":
                    leg1[j[3]] = wp
                elif j[2] == "leg3":
                    leg3[j[3]] = wp
                else:
                    bridge[j[3]] = wp
                    tested.add(j[3])
            # 테스트된 가교쌍을 joint-최근접 순으로 보며, 3-leg 완성 + densify-검증 통과한 첫 쌍 채택.
            # leg 는 job 앞쪽이라 bridge 가 테스트될 땐 이미 테스트됨 → 미완성 쌍은 영구 거부(reject).
            for _, ia, ib in order:
                if (ia, ib) in rejected or (ia, ib) not in tested:
                    continue
                legm = bridge[(ia, ib)]                       # A'→B' (가교부)
                l1 = leg1.get(ia) if ia != 0 else None        # scan→A'
                l3 = leg3.get(ib) if ib != 0 else None        # B'→scan
                complete = (legm is not None
                            and (ia == 0 or l1 is not None)
                            and (ib == 0 or l3 is not None))
                if complete:
                    segs = [l1, legm[1:]] if ia != 0 else [legm]   # 중복 endpoint 제거
                    if ib != 0:
                        segs.append(l3[1:])
                    wp = np.concatenate(segs, axis=0)
                    if len(wp) >= 2 and _transit_safe(wp):    # ★ 최종 검증과 동일 기준 통과만 채택
                        return wp
                rejected.add((ia, ib))                        # 미완성 or grazing → 거부, 다음 후보로
        return None

    transit_segments = {}
    transit_stats = []

    def _record(idx, waypoints, route, dt, announce=True):
        transit_segments[idx] = waypoints
        max_step_deg = np.rad2deg(
            np.max(np.abs(np.diff(waypoints, axis=0)))
        ) if len(waypoints) > 1 else 0.0
        transit_stats.append({
            "idx": int(idx), "success": True, "route": route,
            "n_waypoints": len(waypoints), "time": dt,
            "max_step_deg": float(max_step_deg),
        })
        if announce:
            print(
                f"    {_lbl(idx)}→{_lbl(idx+1)}: OK [{route}] ({len(waypoints)} waypoints, "
                f"max_step={max_step_deg:.2f}°, {dt:.2f}s)"
            )

    recon = [int(i) for i in reconfig_indices]

    # Round 0: 전 경계 direct 를 한 배치로 (대부분 여기서 끝). 성공/실패만 분기.
    t0 = time.time()
    direct_wps = _plan_batch([selected[i] for i in recon],
                             [selected[i + 1] for i in recon])
    dt0 = time.time() - t0
    _tick("transit_direct", t0)
    pending = []
    for k, idx in enumerate(recon):
        wp = direct_wps[k]
        if wp is not None:
            wp = wp.copy()
            if lock_wrist3:
                wp[:, -1] = _w3lock             # main 이 direct 에 거는 wrist_3 lock 을 미리 적용해 검증
            if _transit_safe(wp):               # plan_cspace success 가 아니라 densify-검증으로 채택
                _record(idx, wp, "direct", dt0, announce=False)
                continue
        pending.append(idx)                     # 실패 or grazing → via-roll 사다리로
    print(f"    Direct batch: {len(recon) - len(pending)}/{len(recon)} ok ({dt0:.2f}s)")

    # Round 1+: 실패 경계만 사다리(via-roll → via-tilt → via-home), 각 경계내 후보는 _via_variant 가 배치 탐색.
    _t_pending = time.time()
    for idx in pending:
        t0 = time.time()
        waypoints = None
        route = None

        # 큰 base reconfig(팔 분기 flip)은 광축 roll/tilt 로 가교 불가 → 곧장 via-home 으로
        # 보내 헛된 roll/tilt attempt 를 생략(채택 route·실행 모션 동일, 생성시간만 단축).
        base_linf = float(np.max(np.abs(
            np.asarray(selected[idx])[:3] - np.asarray(selected[idx + 1])[:3])))
        skip_wrist_ladder = base_linf > BIG_BASE_RECONFIG_RAD

        if enable_via_ladder:                                      # via 사다리 off 시 direct 실패분은 드롭
            if not skip_wrist_ladder:
                waypoints = _via_variant(idx, "roll")              # 2) 광축 roll 중간자세 경유
                route = "via-roll" if waypoints is not None else None

                if waypoints is None and wd_m is not None:         # 3) tilt 중간자세 경유 (escalation)
                    waypoints = _via_variant(idx, "tilt")
                    if waypoints is not None:
                        route = "via-tilt"

            if waypoints is None:                                  # 4) HOME 경유 (big-base 는 곧장 여기)
                legs = _plan_batch([selected[idx], home_q],
                                   [home_q, selected[idx + 1]])
                if legs[0] is not None and legs[1] is not None:
                    cand = np.concatenate([legs[0], legs[1][1:]], axis=0)  # HOME 중복 제거
                    if lock_wrist3:
                        cand[:, -1] = _w3lock                          # via-home 도 wrist_3 lock 후 검증
                    if _transit_safe(cand):
                        waypoints = cand
                        route = "via-home"

        dt = time.time() - t0
        if waypoints is not None:
            _record(idx, waypoints, route, dt)
        else:
            transit_stats.append({"idx": int(idx), "success": False, "time": dt})
            print(f"    {_lbl(idx)}→{_lbl(idx+1)}: FAILED [genuinely-unbridgeable] ({dt:.2f}s)")
    _tick("transit_pending", _t_pending)

    n_ok = sum(1 for s in transit_stats if s["success"])
    by_route = []
    for nm in ("direct", "via-roll", "via-tilt", "via-home"):
        c = sum(1 for s in transit_stats if s.get("route") == nm)
        if c:
            by_route.append(f"{c} {nm}")
    print(f"  Transit planning: {n_ok}/{len(recon)} succeeded"
          + (f" ({', '.join(by_route)})" if by_route else ""))

    return transit_segments, transit_stats


# =========================================================================
# Phase 5: Uniform resample + collision check
# =========================================================================

def _resample_uniform(joints, spacing, robot_cfg=None):
    """waypoint를 균일 간격으로 재분할 → 상수 dt 재생 시 속도가 일정해진다.

    robot_cfg 가 주어지면 EE position arc-length(m), 아니면 joint-space L∞(max-joint, rad)
    기준으로 인접 출력 간 거리 ≈ spacing 이 되도록 분할한다. (scan은 EE, transit은 joint 기준)
    """
    if len(joints) < 2:
        return joints
    if robot_cfg is not None:
        ee_positions, _ = compute_fk(joints, robot_cfg)  # (M, 3)
        diffs = np.linalg.norm(np.diff(ee_positions, axis=0), axis=1)
    else:
        diffs = np.max(np.abs(np.diff(joints, axis=0)), axis=1)
    cum_len = np.concatenate([[0], np.cumsum(diffs)])
    total_len = cum_len[-1]
    if total_len < 1e-9:
        return joints
    n_out = max(2, int(np.ceil(total_len / spacing)) + 1)
    uniform_s = np.linspace(0, total_len, n_out)
    out = np.zeros((n_out, joints.shape[1]), dtype=np.float64)
    for j in range(joints.shape[1]):
        out[:, j] = np.interp(uniform_s, cum_len, joints[:, j])
    return out


def _build_runs(selected, transit_segments, reconfig_threshold_rad, max_skip,
                scan_free=None):
    """viewpoint 시퀀스를 '안전하게 연속인' run 들로 분할한다(호출부가 최장 run만 채택).

    run 내부 연결 규칙(두 viewpoint를 같은 run으로 이으려면 충돌-free 이동이 있어야 함):
        - 성공한 transit (consecutive, idx in transit_segments) → MotionGen 충돌-free → 유지
        - small jump (L∞ <= thr) **이고** 그 직선 스캔이 충돌-free → 유지
        - 위로 못 이으면 아웃라이어 viewpoint(i+1..j-1)를 최대 max_skip개까지 건너뛰어
          selected[i]와 thr 이내 **이고** 직선 스캔이 충돌-free인 j 에 재연결(run 유지)
    어떤 것으로도 못 이으면(클러스터 경계 / 스캔 관통 등) 그 지점에서 run 을 끊는다(cut).

    scan_free(a, b): 직선 joint 스캔 selected[a]→selected[b] 가 충돌-free인지 (None이면 항상 True,
    하위호환). small jump 이라도 곡면 측면에서 직선 보간이 물체를 관통할 수 있어 검사가 필요하다.

    Returns:
        runs: list[list[int]] — 각 run 의 viewpoint 인덱스(원본 순서)
        skipped: set[int] — run 내부에서 건너뛴 아웃라이어 인덱스
    """
    def _scan_ok(a, b):
        return scan_free is None or scan_free(a, b)

    N = len(selected)
    runs: list[list[int]] = []
    skipped: set[int] = set()
    cur = [0]
    i = 0
    while i < N - 1:
        # 성공 transit → 충돌-free 이동 보장 → 현재 run 유지
        if i in transit_segments:
            cur.append(i + 1)
            i += 1
            continue
        # small jump 이고 직선 스캔이 충돌-free → 유지
        jump = float(np.max(np.abs(selected[i + 1] - selected[i])))
        if jump <= reconfig_threshold_rad and _scan_ok(i, i + 1):
            cur.append(i + 1)
            i += 1
            continue

        # 못 이음(큰 reconfig 또는 스캔 관통) → 아웃라이어 skip 시도(transit 시작점은 건너뛰지 않음)
        def _connectable(a, b):
            return (b not in transit_segments
                    and float(np.max(np.abs(selected[b] - selected[a]))) <= reconfig_threshold_rad
                    and _scan_ok(a, b))

        j = i + 1
        n_dropped = 0
        while (j < N and n_dropped < max_skip and j not in transit_segments
               and not _connectable(i, j)):
            j += 1
            n_dropped += 1
        if j < N and _connectable(i, j):
            skipped.update(range(i + 1, j))
            cur.append(j)
            i = j
            continue

        # 이을 수 없는 경계 → run 을 끊고 i+1 부터 새 run 시작
        runs.append(cur)
        cur = [i + 1]
        i += 1

    if cur[-1] != N - 1:
        cur.append(N - 1)
    runs.append(cur)
    return runs, skipped


def _precompute_scan_free(selected, reconfig_threshold_rad, max_skip,
                          robot_cfg, world_scene):
    """직선 joint 스캔이 충돌-free인 (a,b) 쌍을 한 번의 batch collision check로 미리 계산.

    스캔으로 이어질 수 있는 후보(=jump ≤ thr, b-a ≤ max_skip+1)만 검사한다. 큰 jump는 어차피
    transit이 필요하므로 스캔 검사 불필요. 반환: 함수 scan_free(a,b)->bool (표에 없으면 True).
    """
    N = len(selected)
    pairs = []
    for a in range(N - 1):
        for b in range(a + 1, min(a + max_skip + 2, N)):
            if float(np.max(np.abs(selected[b] - selected[a]))) <= reconfig_threshold_rad:
                pairs.append((a, b))
    if not pairs:
        return lambda a, b: True

    dense_list, counts = [], []
    for (a, b) in pairs:
        d = densify_for_collision_check(np.stack([selected[a], selected[b]]))
        dense_list.append(d)
        counts.append(len(d))
    isc, _ = batch_collision_check(np.concatenate(dense_list, axis=0), robot_cfg, world_scene)

    free = {}
    off = 0
    for (a, b), n in zip(pairs, counts):
        free[(a, b)] = not bool(isc[off:off + n].any())
        off += n
    return lambda a, b: free.get((a, b), True)


def find_colliding_interpolation_edges(selected, edge_indices, robot_cfg, world_scene):
    """Return edges whose straight joint interpolation is in collision.

    Every requested ``i`` denotes ``selected[i] -> selected[i + 1]``.  All
    interpolations are densified and checked in one collision-check batch so
    callers can add only the failing scan edges to the MotionGen transit batch.
    """
    selected = np.asarray(selected, dtype=np.float64)
    edges = np.asarray(edge_indices, dtype=np.int64).reshape(-1)
    if len(edges) == 0:
        return np.zeros((0,), dtype=np.int64)
    if np.any(edges < 0) or np.any(edges >= len(selected) - 1):
        raise IndexError("interpolation edge index out of range")

    dense, counts = [], []
    for idx in edges:
        segment = densify_for_collision_check(
            np.stack([selected[int(idx)], selected[int(idx) + 1]])
        )
        dense.append(segment)
        counts.append(len(segment))
    is_collision, _ = batch_collision_check(
        np.concatenate(dense, axis=0), robot_cfg, world_scene,
    )

    colliding = []
    offset = 0
    for idx, count in zip(edges, counts):
        if bool(is_collision[offset:offset + count].any()):
            colliding.append(int(idx))
        offset += count
    return np.asarray(colliding, dtype=np.int64)


def _typed_segments_for_run(selected, transit_segments, run_idx, dense_step_rad):
    """한 run을 'scan' / 'transit' dense sub-path 들로 분할한다.

    인접 viewpoint 간 전이를 두 종류로 구분한다:
        - transit edge (nxt==cur+1 이고 cur in transit_segments) → MotionGen 재배치 경로.
          별도 'transit' 세그먼트로 떼어낸다(joint 속도로 빠르게 지나갈 구간).
        - 그 외(small jump / skip 재연결) → 직선 보간으로 잇는 실제 스캔 이동. 연속된
          스캔 이동은 하나의 'scan' 세그먼트로 모은다(EE arc-length로 resample할 구간).

    각 세그먼트는 시작/끝 viewpoint config 를 모두 포함하므로, 인접 세그먼트는 경계
    config 를 공유한다(stitch_trajectory_pieces 가 중복 제거).

    Returns:
        list[(kind, dense (K,6))], kind ∈ {"scan", "transit"}
    """
    segments = []
    scan_buf = [selected[run_idx[0]:run_idx[0] + 1]]   # 첫 viewpoint 로 시작
    for a in range(len(run_idx) - 1):
        cur = run_idx[a]
        nxt = run_idx[a + 1]
        if nxt == cur + 1 and cur in transit_segments:
            # 현재까지 모은 스캔 이동을 flush (selected[cur]에서 끝남)
            buf = np.concatenate(scan_buf, axis=0)
            if len(buf) >= 1:
                segments.append(("scan", buf))
            # transit 전체(양 끝 = selected[cur]≈transit[0], selected[nxt]≈transit[-1] 포함)
            segments.append(("transit", transit_segments[cur]))
            # 다음 스캔 버퍼는 selected[nxt]에서 새로 시작
            scan_buf = [selected[nxt:nxt + 1]]
        else:
            # selected[cur]→selected[nxt] 직선 보간의 내부점(양 끝 제외)을 dense하게 채운다
            q0, q1 = selected[cur], selected[nxt]
            n_steps = max(1, int(np.ceil(float(np.max(np.abs(q1 - q0))) / dense_step_rad)))
            if n_steps > 1:
                alphas = np.linspace(0.0, 1.0, n_steps + 1)[1:-1]
                scan_buf.append(q0[np.newaxis, :] + alphas[:, np.newaxis] * (q1 - q0)[np.newaxis, :])
            scan_buf.append(selected[nxt:nxt + 1])
    buf = np.concatenate(scan_buf, axis=0)
    if len(buf) >= 1:
        segments.append(("scan", buf))
    return segments


def stitch_trajectory_pieces(pieces, masks, dup_tol_rad=5e-3):
    """resample된 sub-path 들을 이어붙인다. 인접 piece 가 공유하는 경계 waypoint 는
    한 번만 남긴다(중복 제거). pieces/masks 는 길이가 같은 list."""
    out_j, out_m = [], []
    for p, m in zip(pieces, masks):
        if len(p) == 0:
            continue
        if out_j and len(p) >= 1 and \
                float(np.max(np.abs(p[0] - out_j[-1][-1]))) <= dup_tol_rad:
            p, m = p[1:], m[1:]
        if len(p) == 0:
            continue
        out_j.append(p)
        out_m.append(m)
    if not out_j:
        return np.zeros((0, 6), dtype=np.float64), np.zeros((0,), dtype=bool)
    return np.concatenate(out_j, axis=0), np.concatenate(out_m, axis=0)


def interpolate_and_resample(selected, transit_segments, robot_cfg,
                             mode="ee", spacing=0.01, dense_step_rad=0.02,
                             transit_spacing_rad=TRANSIT_RESAMPLE_SPACING_RAD,
                             reconfig_threshold_rad=np.deg2rad(RECONFIG_THRESHOLD_DEG),
                             max_skip=TRANSIT_FAIL_SKIP_MAX, world_scene=None):
    """DP 궤적 + transit을 합치고, 선택된 metric으로 uniform resample.

    transit 실패 reconfig 은 직선 보간하면 물체를 관통하므로 사용하지 않는다. 대신
    경로를 '안전 연속 run'들로 분할(_build_runs)하고 **가장 긴 run** 을 채택한다. 즉
    아웃라이어 viewpoint 는 건너뛰고, 이을 수 없는 경계에서는 작은 쪽을 통째로 드롭한다.

    Args:
        selected: (N, 6) DP 선택 궤적
        transit_segments: dict {idx: (T, 6)} 성공한 transit 경로
        robot_cfg: cuRobo robot config dict (mode='ee'일 때 FK 용)
        mode: "ee" (EE position arc-length, meters) | "joint" (cumulative L∞, radians)
        spacing: 스캔 구간 최종 spacing (mode에 따라 m 또는 rad)
        dense_step_rad: dense path 구성 시 joint-space L∞ step (radians)
        transit_spacing_rad: transit(재배치) 구간 joint-space resample 간격 (radians).
            transit은 검사가 아니므로 EE 기준이 아니라 joint 기준으로 sparse하게 재분할한다.
        reconfig_threshold_rad: 직선 보간 허용 임계값(이보다 크면 transit 필요)
        max_skip: transit 실패 시 run 내부에서 건너뛸 수 있는 최대 viewpoint 수

    Returns:
        resampled:  (M, 6) uniform-spaced trajectory (가장 긴 안전 run)
        is_transit: (M,) bool — 각 waypoint가 transit(재배치) 구간인지. compute_trajectory_times
                    가 이 구간을 joint 속도로만(검사 EE 속도 무시) 시간 매기는 데 쓴다.
        dropped:    list[int] — 채택 run 에 포함되지 않아 드롭된 viewpoint 인덱스
        runs_info:  dict — {"runs": [(start,end,len)...], "kept": (start,end,len)}
    """
    N = len(selected)
    # world_scene이 있으면 직선 스캔 충돌까지 고려해 run을 나눈다. 곡면 측면에서 인접
    # viewpoint 사이 직선 보간이 물체를 관통하면(via-home으로 진입은 됐어도) 스캔 자체가 불가 →
    # 그 구간을 끊어 정직하게 드롭(최종 batch 충돌검사 거부 대신 유효 궤적 저장).
    scan_free = (_precompute_scan_free(selected, reconfig_threshold_rad, max_skip,
                                       robot_cfg, world_scene)
                 if world_scene is not None else None)
    runs, _ = _build_runs(selected, transit_segments, reconfig_threshold_rad, max_skip,
                          scan_free=scan_free)
    kept = max(runs, key=len)
    kept_set = set(kept)
    dropped = sorted(set(range(N)) - kept_set)

    runs_info = {
        "runs": [(r[0], r[-1], len(r)) for r in runs],
        "kept": (kept[0], kept[-1], len(kept)),
    }

    if len(kept) < 2:
        raise RuntimeError(
            "안전하게 연속인 구간이 viewpoint 2개 미만입니다 — 모든 인접 전이가 "
            "이을 수 없는 reconfig. 물체 배치/작업거리 조정으로 reconfig를 줄여야 합니다."
        )

    if mode not in ("ee", "joint"):
        raise ValueError(f"Unknown resample mode: {mode!r} (expected 'ee' or 'joint')")

    # 스캔 / transit 세그먼트를 분리해 각각에 맞는 기준으로 resample 한다.
    #   scan    → EE arc-length(또는 joint, mode에 따라). 검사 spacing 그대로.
    #   transit → joint-space L∞ arc-length(sparse). EE 호 길이와 무관하게 재배치.
    segments = _typed_segments_for_run(selected, transit_segments, kept, dense_step_rad)
    pieces, masks = [], []
    for kind, dense in segments:
        if kind == "transit":
            rs = _resample_uniform(dense, transit_spacing_rad)
            # sparse joint resample은 코너에서 직선을 그어 MotionGen이 충돌-free로 보장한
            # 곡선 경로를 잘라낼 수 있다(특히 물체로 재진입하는 via-home 재접근 leg). 그러면
            # 최종 densify 충돌검사에서 관통이 잡혀 저장이 거부된다. world_scene이 주어지면
            # 이 transit piece를 검사해, 충돌이 생기면 MotionGen native 해상도를 그대로 쓴다
            # (densify가 모든 vertex를 지나 충돌-free 유지). 자유공간 transit은 sparse 유지.
            if world_scene is not None and len(rs) >= 2:
                isc, _ = batch_collision_check(
                    densify_for_collision_check(rs), robot_cfg, world_scene)
                if bool(isc.any()):
                    rs = dense
            mk = np.ones((len(rs),), dtype=bool)
        elif mode == "ee":
            rs = _resample_uniform(dense, spacing, robot_cfg)
            mk = np.zeros((len(rs),), dtype=bool)
        else:  # mode == "joint"
            rs = _resample_uniform(dense, spacing)
            mk = np.zeros((len(rs),), dtype=bool)
        pieces.append(rs)
        masks.append(mk)

    resampled, is_transit = stitch_trajectory_pieces(pieces, masks)
    return resampled, is_transit, dropped, runs_info


def densify_for_collision_check(trajectory: np.ndarray) -> np.ndarray:
    """Densify joint-space segments before collision validation."""
    if len(trajectory) < 2:
        return trajectory

    max_step_rad = np.deg2rad(config.COLLISION_ADAPTIVE_MAX_JOINT_STEP_DEG)
    if max_step_rad <= 0.0:
        raise ValueError("COLLISION_ADAPTIVE_MAX_JOINT_STEP_DEG must be > 0")

    metric = trajectory
    if config.COLLISION_INTERP_EXCLUDE_LAST_JOINT and trajectory.shape[1] > 1:
        metric = trajectory[:, :-1]

    segments = [trajectory[0:1]]
    for i in range(len(trajectory) - 1):
        q0 = trajectory[i]
        q1 = trajectory[i + 1]
        dist = float(np.max(np.abs(metric[i + 1] - metric[i])))
        n_steps = max(1, int(np.ceil(dist / max_step_rad)))
        alphas = np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float64)[1:]
        segments.append(q0[np.newaxis, :] + alphas[:, np.newaxis] * (q1 - q0)[np.newaxis, :])

    return np.concatenate(segments, axis=0)


# =========================================================================
# Time planning
# =========================================================================
