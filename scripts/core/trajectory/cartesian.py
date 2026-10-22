"""보간 구간을 Cartesian 직선으로 만든다 — viewpoint 사이를 카메라가 **직선으로** 지나간다.

예전에는 인접 viewpoint 의 관절값을 직선 보간했다(joint-space lerp). 관절이 직선이면 EE 는
곡선을 그리고, 그 곡선 위의 속도는 EE 속도로 시간을 매겨도 표면 위에서 고르지 않다. 여기서는
EE 위치를 lerp, 자세를 slerp 한 목표 포즈를 촘촘히 찍고, 관절 lerp 를 seed 로 IK 를 풀어
그 직선을 따르는 관절 경로를 만든다. 그러면 보간 구간의 EE 속도 노브가 말 그대로
"직선 위를 이 속도로" 가 된다(timing.py 의 보간 구간 규칙).

실패(IK 불능·관절 점프·충돌)한 edge 는 호출자(verify)가 **플래닝 구간**으로 승격한다 —
"점프로 인한 모션플래닝 부분은 플래닝 구간" 이라는 규칙이 여기서도 그대로다.

seed 가 관절 lerp 인 이유: 두 끝 자세가 reconfig 임계 안(같은 팔 분기)이라 그 사이 직선 위의
IK 해도 같은 분기에 있고, 관절 lerp 는 그 해에서 몇 mm 벗어난 근사다(5 mm 표본에서 실측
최대 14 mm). 여기서 감쇠 최소제곱(DLS) 뉴턴 반복으로 직선 위로 끌어오면 분기가 바뀌지 않는다.

cuRobo 의 IK 최적화기를 쓰지 않는 이유: 이 버전(v0.8)의 particle/LBFGS 파이프라인은 seed 를
국소 정밀화하는 도구가 아니다 — 정확한 해에 1e-3 rad 만 흔든 seed 에서도 절반이 수백 mm 로
발산했고(실측), 여러 seed 를 주면 성공하지만 다른 분기의 해를 돌려준다. 여기서는 cuRobo 의
배치 FK 만 쓰고(2000 자세 1 ms) 수치 자코비안으로 직접 푼다 — 표본 하나에 반복 몇 번이면 끝난다.
"""

from __future__ import annotations

import time

import numpy as np

from .robot import _get_kinematics, _tick, batch_collision_check, compute_fk

# 표본 간격. 5 mm / 2° 면 표본 사이 관절 lerp 가 직선에서 벗어나는 양이 실행기의 보간 오차
# (waypoint 10 mm 간격) 아래다. 촘촘할수록 FK 문제 수가 늘어난다(전부 한 배치라 선형).
CARTESIAN_STEP_M = 0.005
CARTESIAN_STEP_RAD = np.deg2rad(2.0)
# 인접 표본 사이 허용 관절 변화(L∞). 5 mm 에 0.25 rad 이면 이미 특이점 근처라 분기가 바뀐
# 것으로 본다 — 그 edge 는 플래닝으로 넘긴다.
CARTESIAN_MAX_JOINT_STEP_RAD = 0.25
# 수렴 판정. 검사 카메라의 직선성이 목적이라 0.5 mm / 0.01 rad. FK 는 float32 라 1e-6 m 정밀.
CARTESIAN_POSITION_TOL_M = 0.0005
CARTESIAN_ORIENTATION_TOL_RAD = 0.01
# DLS 반복: 감쇠 λ, 수치 미분 폭, 최대 반복.
_DLS_DAMPING = 0.02
_DLS_EPS = 1e-3
_DLS_MAX_ITER = 20


def _quat_xyzw_to_wxyz(q):
    q = np.asarray(q, dtype=np.float64)
    return q[..., [3, 0, 1, 2]]


def _slerp_wxyz(q0, q1, t):
    """(4,) wxyz 두 사분원 사이 slerp, t (K,) → (K, 4). 같은 반구로 맞춘 뒤 보간."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    t = np.asarray(t, dtype=np.float64)[:, None]
    if dot > 1.0 - 1e-9:
        out = q0[None, :] + t * (q1 - q0)[None, :]
        return out / np.linalg.norm(out, axis=1, keepdims=True)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    s = np.sin(theta)
    return (np.sin((1.0 - t) * theta) / s) * q0[None, :] + (np.sin(t * theta) / s) * q1[None, :]


def _quat_angle_wxyz(q0, q1):
    dot = abs(float(np.dot(q0, q1))) / max(np.linalg.norm(q0) * np.linalg.norm(q1), 1e-12)
    return 2.0 * float(np.arccos(np.clip(dot, -1.0, 1.0)))


def _quat_mul_wxyz(a, b):
    """(N,4) ⊗ (N,4) → (N,4), wxyz."""
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=1)


def _rotvec_between_wxyz(q_goal, q_cur):
    """q_cur → q_goal 로 가는 월드 프레임 회전벡터 (N,3): rotvec(q_goal ⊗ conj(q_cur))."""
    conj = q_cur * np.array([1.0, -1.0, -1.0, -1.0])
    e = _quat_mul_wxyz(q_goal, conj)
    e = np.where(e[:, :1] < 0.0, -e, e)                      # w ≥ 0 → 짧은 쪽 회전
    v = e[:, 1:]
    n = np.linalg.norm(v, axis=1)
    ang = 2.0 * np.arctan2(n, e[:, 0])
    scale = np.where(n > 1e-12, ang / np.maximum(n, 1e-12), 2.0)   # 소각: rotvec ≈ 2v
    return v * scale[:, None]


def _fk_wxyz(q, robot_cfg):
    p, q_xyzw = compute_fk(np.asarray(q, dtype=np.float64), robot_cfg)
    return np.asarray(p, dtype=np.float64), _quat_xyzw_to_wxyz(q_xyzw)


def _joint_limits(robot_cfg):
    kin = _get_kinematics(robot_cfg)
    pos = np.asarray(kin.get_joint_limits().position.cpu(), dtype=np.float64)
    if pos.shape[0] == 2:
        return pos[0], pos[1]
    return pos[:, 0], pos[:, 1]


def refine_on_line(seeds, goal_p, goal_q, robot_cfg, *, pos_tol=CARTESIAN_POSITION_TOL_M,
                   rot_tol=CARTESIAN_ORIENTATION_TOL_RAD, max_iter=_DLS_MAX_ITER,
                   damping=_DLS_DAMPING, eps=_DLS_EPS):
    """관절 seed (B,6) 를 목표 포즈 (B,3)/(B,4 wxyz) 위로 끌어온다 → (q (B,6), ok (B,)).

    감쇠 최소제곱 뉴턴: e = [Δp; rotvec], J 는 수치 미분(배치 FK 7회/반복),
    Δq = Jᵀ (J Jᵀ + λ²I)⁻¹ e. 수렴한 행은 더 건드리지 않는다. 관절 한계를 벗어난 행은 실패.
    """
    q = np.array(seeds, dtype=np.float64, copy=True)
    B, dof = q.shape
    ok = np.zeros(B, dtype=bool)
    lower, upper = _joint_limits(robot_cfg)
    lam2 = damping * damping
    for _ in range(max_iter):
        p, qw = _fk_wxyz(q, robot_cfg)
        e = np.concatenate([goal_p - p, _rotvec_between_wxyz(goal_q, qw)], axis=1)   # (B,6)
        ok = (np.linalg.norm(e[:, :3], axis=1) < pos_tol) & (np.linalg.norm(e[:, 3:], axis=1) < rot_tol)
        active = np.flatnonzero(~ok)
        if len(active) == 0:
            break
        qa = q[active]
        J = np.empty((len(active), 6, dof))
        for j in range(dof):
            qj = qa.copy()
            qj[:, j] += eps
            pj, qwj = _fk_wxyz(qj, robot_cfg)
            J[:, :3, j] = (pj - p[active]) / eps
            J[:, 3:, j] = _rotvec_between_wxyz(qwj, qw[active]) / eps
        JJt = J @ np.transpose(J, (0, 2, 1)) + lam2 * np.eye(6)[None]
        y = np.linalg.solve(JJt, e[active][..., None])[..., 0]          # (J Jᵀ + λ²I)⁻¹ e
        dq = np.einsum("bij,bj->bi", np.transpose(J, (0, 2, 1)), y)
        q[active] = qa + dq
    inside = np.all((q >= lower[None, :] - 1e-9) & (q <= upper[None, :] + 1e-9), axis=1)
    return q, ok & inside


def track_cartesian_scans(selected, pairs, robot_cfg, world_scene=None, *,
                          step_m=CARTESIAN_STEP_M, step_rad=CARTESIAN_STEP_RAD,
                          max_joint_step_rad=CARTESIAN_MAX_JOINT_STEP_RAD):
    """(a, b) 쌍마다 selected[a]→selected[b] 의 Cartesian 직선 관절 경로를 만든다.

    ``world_scene`` 은 여기서 쓰지 않는다(충돌은 colliding_scan_paths / run 분할이 본다) —
    호출 형태를 맞추려고 받는다.

    Returns:
        paths: dict {(a, b): (K, 6) 양 끝 포함 dense 경로 | None(실패)}
        stats: dict — n_pairs, n_ok, n_ik_fail, n_jump_fail, n_samples, seconds
    """
    selected = np.asarray(selected, dtype=np.float64)
    pairs = [(int(a), int(b)) for a, b in pairs]
    stats = {"n_pairs": len(pairs), "n_ok": 0, "n_ik_fail": 0, "n_jump_fail": 0,
             "n_samples": 0, "seconds": 0.0}
    if not pairs:
        return {}, stats
    t0 = time.time()
    ee_pos, ee_quat = _fk_wxyz(selected, robot_cfg)

    # 1) 모든 쌍의 내부 표본을 한 배열로 모은다(쌍별 slice 기록).
    goals_p, goals_q, seeds, slices = [], [], [], {}
    offset = 0
    for a, b in pairs:
        dp = float(np.linalg.norm(ee_pos[b] - ee_pos[a]))
        dth = _quat_angle_wxyz(ee_quat[a], ee_quat[b])
        n = max(1, int(np.ceil(max(dp / step_m, dth / step_rad))))
        if n <= 1:
            slices[(a, b)] = (0, 0)          # 표본 간격보다 짧다 — 내부점 없음
            continue
        t = np.arange(1, n) / n
        start = offset
        offset += len(t)
        goals_p.append(ee_pos[a][None, :] + t[:, None] * (ee_pos[b] - ee_pos[a])[None, :])
        goals_q.append(_slerp_wxyz(ee_quat[a], ee_quat[b], t))
        seeds.append(selected[a][None, :] + t[:, None] * (selected[b] - selected[a])[None, :])
        slices[(a, b)] = (start, start + len(t))
    paths = {}
    if not goals_p:
        for a, b in pairs:
            paths[(a, b)] = np.stack([selected[a], selected[b]])
        stats["n_ok"] = len(pairs)
        stats["seconds"] = time.time() - t0
        return paths, stats
    P = np.concatenate(goals_p, axis=0)
    Q = np.concatenate(goals_q, axis=0)
    S = np.concatenate(seeds, axis=0)
    stats["n_samples"] = int(len(P))

    # 2) 전 표본을 한 배치로 직선 위에 올린다.
    sol, ok = refine_on_line(S, P, Q, robot_cfg)
    _tick("cartesian_track", t0)

    # 3) 쌍별로 조립하고 연속성을 확인한다.
    for a, b in pairs:
        s, e = slices[(a, b)]
        if e > s and not ok[s:e].all():
            paths[(a, b)] = None
            stats["n_ik_fail"] += 1
            continue
        dense = np.concatenate([selected[a][None, :], sol[s:e], selected[b][None, :]], axis=0)
        if len(dense) > 1 and float(np.max(np.abs(np.diff(dense, axis=0)))) > max_joint_step_rad:
            paths[(a, b)] = None
            stats["n_jump_fail"] += 1
            continue
        paths[(a, b)] = dense
        stats["n_ok"] += 1
    stats["seconds"] = time.time() - t0
    return paths, stats


def colliding_scan_paths(paths, keys, robot_cfg, world_scene):
    """``keys`` 의 경로 중 densify 충돌검사에 걸리는 것 → set. None(추적 실패)은 충돌로 친다.

    한 번의 batch collision check 로 전부 본다(verify 가 실패 edge 를 플래닝으로 승격할 때).
    """
    from .motion import densify_for_collision_check   # 순환 import 회피(motion 이 이 모듈을 쓴다)

    bad = set()
    dense_list, counts, order = [], [], []
    for key in keys:
        dense = paths.get(tuple(key))
        if dense is None:
            bad.add(tuple(key))
            continue
        d = densify_for_collision_check(np.asarray(dense, dtype=np.float64))
        dense_list.append(d)
        counts.append(len(d))
        order.append(tuple(key))
    if dense_list:
        isc, _ = batch_collision_check(np.concatenate(dense_list, axis=0), robot_cfg, world_scene)
        off = 0
        for key, n in zip(order, counts):
            if bool(isc[off:off + n].any()):
                bad.add(key)
            off += n
    return bad
