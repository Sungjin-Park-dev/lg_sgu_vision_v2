"""궤적 시간매김 — CSV 의 time 열을 만든다.

구간(waypoint i-1 → i)의 종류가 어느 노브가 시간을 정하는지 가른다:

  보간 구간(스캔; viewpoint 사이, tilt 부채꼴)
      dt = max(min_dt, |Δp| / v_ee, Δθ / ω_ee)                 — EE 선속도·각속도
  플래닝 구간(transit; 재배치, Plan to Start/HOME)
      dt = max(min_dt, max_j |Δq_j| / (s_v · v_max_j))          — 관절별 속도 한계 × 속도 배율

관절별 한계(v_max_j, a_max_j)와 배율(s_v, s_a)의 뜻은 MoveIt 과 같다: 같은 joint_limits.yaml,
같은 scaling factor. 그 위에 관절 가속도 한계(s_a · a_max_j)를 경로-속도 프로파일로 건다 —
출발·정지는 0 에서, 꺾임과 구간 경계(플래닝→보간 이음새)는 실행기가 겪는 가속도 정의 그대로.
로봇 물리 한계(배율 100%)의 관절 속도는 모든 구간의 바닥이다 — 특이점 근처에서 보간 구간의
관절이 튀는 것을 막는 안전선이지 노브가 아니고, 스캔 속도에서는 닿지 않는다.
"""

import math

import numpy as np


def _quat_angle_xyzw(q0, q1):
    """Quaternion geodesic angle in radians. Input order: x, y, z, w."""
    q0 = q0 / max(np.linalg.norm(q0), 1e-12)
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    dot = abs(float(np.dot(q0, q1)))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def _joint_acc_per_node(dq, dt):
    """실행기가 보는 관절 가속도 — 구간 속도(dq/dt)의 노드 간 변화 / 인접 구간 평균 dt.

    실행기는 waypoint 사이를 직선 보간하므로 관절 속도는 구간마다 상수이고 노드에서 바뀐다.
    그 변화를 두 구간의 평균 시간으로 나눈 것이 로봇이 실제로 겪는 가속도다. (N-2, dof)
    """
    vj = dq / dt[:, None]
    return np.abs(np.diff(vj, axis=0)) / ((dt[:-1] + dt[1:]) / 2.0)[:, None]


def _limit_joint_acceleration(joints, dt_min, a_lim, max_iter=300):
    """관절별 가속도 상한을 만족하도록 구간 시간을 늘린다 → (dt, 노드 속도).

    경로 위 속도 프로파일이다. 구간 길이는 관절 L∞(가장 많이 도는 관절), 속도 상한은
    이미 계산된 최소 구간시간에서 온다(cap = 길이/dt_min). 노드 속도를 양 끝 0 에서
    출발해 전진 패스(v_k ≤ √(v_{k-1}² + 2aL)), 후진 패스로 깎으면 경로 방향의 가감속이
    한계 안에 든다. 구간 시간은 등가속 가정으로 2L/(v_i+v_j). dt_min 보다 짧아지는 일은
    없다(노드 속도가 cap 을 넘지 않으므로).

    관절별 한계는 구간마다 경로 가속도 상한 a_seg = min_j a_lim_j · L / |Δq_j| 로 접는다 —
    경로를 따라 등가속하면 관절 j 의 가속도는 그 |Δq_j|/L 배이므로 이것이 정확한 환산이다
    (한계가 전부 같으면 그 값 그대로).

    꺾임(인접 구간의 관절 방향이 다른 곳)은 경로 속도가 같아도 관절 속도 벡터가 꺾이며
    가속도가 튄다. 그것까지 _joint_acc_per_node 로 **실행기가 겪는 정의 그대로** 재서,
    넘는 노드의 속도를 √(초과비) 로 줄이고 패스를 다시 도는 고정점 반복으로 맞춘다 —
    보고되는 가속도와 강제하는 가속도가 같은 수다. 반복 한 번이 O(n) 이라 100 회는 공짜에
    가깝다(실측: 30 회면 0.8% 남고 100 회면 0.1% 허용치 안).
    """
    n = len(joints)
    a_lim = np.asarray(a_lim, dtype=np.float64)
    dq = np.diff(joints, axis=0)
    seg_len = np.max(np.abs(dq), axis=1)                             # (n-1,) L∞
    moving = seg_len > 1e-12
    cap = np.full(n - 1, np.inf)
    cap[moving] = seg_len[moving] / np.maximum(dt_min[moving], 1e-9)
    # 구간별 경로 가속도 상한(관절별 한계 → 경로 방향). 안 움직이는 구간은 아무 값이나 된다.
    with np.errstate(divide="ignore", invalid="ignore"):
        a_seg = np.min(np.where(np.abs(dq) > 1e-12,
                                a_lim[None, :] * seg_len[:, None] / np.abs(dq), np.inf), axis=1)
    a_seg[~moving] = float(a_lim.min())
    v = np.zeros(n)
    v[1:-1] = np.minimum(cap[:-1], cap[1:])                          # 노드 = 양옆 구간의 작은 cap

    def passes(v):
        for k in range(1, n):
            v[k] = min(v[k], math.sqrt(v[k - 1] ** 2 + 2.0 * a_seg[k - 1] * seg_len[k - 1]))
        for k in range(n - 2, -1, -1):
            v[k] = min(v[k], math.sqrt(v[k + 1] ** 2 + 2.0 * a_seg[k] * seg_len[k]))
        dt = np.array(dt_min, dtype=np.float64, copy=True)
        for i in np.flatnonzero(moving):
            vs = v[i] + v[i + 1]
            if vs > 1e-12:
                dt[i] = max(dt[i], 2.0 * seg_len[i] / vs)
        return dt

    dt = passes(v)
    for _ in range(max_iter):
        if n < 3:
            break
        ratio = np.max(_joint_acc_per_node(dq, dt) / a_lim[None, :], axis=1)   # (n-2,) 노드별 초과비
        bad = np.flatnonzero(ratio > 1.0 + 1e-3)
        if len(bad) == 0:
            break
        # 노드 k(=구간 k-1 과 k 사이)의 속도를 줄인다. 양옆 구간 평균 속도가 함께 내려가
        # 속도 변화는 줄고 평균 dt 는 늘어 초과비가 단조 감소한다.
        for k in bad:
            v[k + 1] /= math.sqrt(ratio[k])
        dt = passes(v)
    return dt, v


def _empty_stats():
    return {
        "total_time": 0.0, "transit_time": 0.0, "n_transit_segments": 0,
        "max_linear_speed_mm_s": 0.0, "max_angular_speed_deg_s": 0.0,
        "max_joint_speed_rad_s": 0.0, "max_joint_acc_rad_s2": 0.0,
        "max_joint_vel_ratio": 0.0, "max_joint_acc_ratio": 0.0,
    }


def compute_trajectory_times(joints, ee_positions, ee_quaternions, is_transit, *,
                             ee_speed_m_s, ee_angular_speed_rad_s,
                             joint_max_velocity, joint_max_acceleration,
                             joint_velocity_scaling, joint_acceleration_scaling,
                             min_segment_dt=0.02):
    """누적 time 열 → (times (N,), stats).

    Args:
        is_transit: (N,) bool — 플래닝(transit) 구간의 waypoint. 구간(i-1→i)은 양 끝 중 하나라도
            transit 이면 플래닝 구간으로 본다(스캔↔transit 진입/이탈 포함).
        ee_speed_m_s, ee_angular_speed_rad_s: 보간 구간의 EE 선속도·각속도. 0 이면 그 제한을 끈다.
        joint_max_velocity, joint_max_acceleration: (dof,) 로봇 한계(joint_limits.yaml, 100%).
        joint_velocity_scaling, joint_acceleration_scaling: (0, 1] — MoveIt 의 scaling factor 와
            같은 뜻. 속도 배율은 플래닝 구간에, 가속도 배율은 모든 노드에 걸린다.

    stats 의 ``max_joint_vel_ratio`` (플래닝 구간, 배율 적용 한계 대비)와 ``max_joint_acc_ratio``
    (전체 노드)가 1.0 이하면 한계를 지킨 것이다 — 강제할 때와 같은 정의로 잰다.
    """
    n = len(joints)
    joints = np.asarray(joints, dtype=np.float64)
    times = np.zeros((n,), dtype=np.float64)
    if n < 2:
        return times, _empty_stats()

    v_hard = np.asarray(joint_max_velocity, dtype=np.float64)          # 로봇 100%
    v_lim = v_hard * float(joint_velocity_scaling)                      # 플래닝 구간 한계
    a_lim = np.asarray(joint_max_acceleration, dtype=np.float64) * float(joint_acceleration_scaling)

    it = np.asarray(is_transit, dtype=bool)
    if len(it) != n:
        raise ValueError(f"is_transit has {len(it)} entries for {n} waypoints")
    seg_is_transit = it[1:] | it[:-1]

    dq = np.diff(joints, axis=0)                                        # (n-1, dof)
    abs_dq = np.abs(dq)
    lin_dist = np.linalg.norm(np.diff(ee_positions, axis=0), axis=1)
    ang_dist = np.array([_quat_angle_xyzw(ee_quaternions[i - 1], ee_quaternions[i])
                         for i in range(1, n)])

    # 1) 구간별 최소 시간 — 구간 종류에 따라 다른 쌍이 정한다. 로봇 100% 관절 속도는 모두의 바닥.
    seg_dt = np.full(n - 1, float(min_segment_dt))
    seg_dt = np.maximum(seg_dt, np.max(abs_dq / v_hard[None, :], axis=1))
    scan = ~seg_is_transit
    if ee_speed_m_s > 0.0:
        seg_dt[scan] = np.maximum(seg_dt[scan], lin_dist[scan] / ee_speed_m_s)
    if ee_angular_speed_rad_s > 0.0:
        seg_dt[scan] = np.maximum(seg_dt[scan], ang_dist[scan] / ee_angular_speed_rad_s)
    seg_dt[seg_is_transit] = np.maximum(
        seg_dt[seg_is_transit], np.max(abs_dq[seg_is_transit] / v_lim[None, :], axis=1))

    # 2) 가속도 제한 — 속도 제한이 정한 시간을 늘리기만 한다(시작·끝 정지 포함).
    seg_dt, _node_speed = _limit_joint_acceleration(joints, seg_dt, a_lim)

    times[1:] = np.cumsum(seg_dt)
    joint_speed = abs_dq / seg_dt[:, None]                              # (n-1, dof)
    linear_speed = lin_dist / seg_dt
    angular_speed = ang_dist / seg_dt
    # EE 선/각속도 최대는 보간 구간에서만 의미가 있다(플래닝 구간은 EE 가 자유공간을 빠르게
    # 휘둘러 EE 속도가 매우 커지지만 검사 속도가 아님).
    scan_lin = linear_speed[scan] if scan.any() else linear_speed
    scan_ang = angular_speed[scan] if scan.any() else angular_speed
    transit_ratio = (np.max(joint_speed[seg_is_transit] / v_lim[None, :])
                     if seg_is_transit.any() else 0.0)
    acc = _joint_acc_per_node(dq, seg_dt) if n >= 3 else np.zeros((0, joints.shape[1]))
    stats = {
        "total_time": float(times[-1]),
        "transit_time": float(seg_dt[seg_is_transit].sum()),
        "n_transit_segments": int(seg_is_transit.sum()),
        "max_linear_speed_mm_s": float(scan_lin.max() * 1000.0),
        "max_angular_speed_deg_s": float(np.rad2deg(scan_ang.max())),
        "max_joint_speed_rad_s": float(joint_speed.max()),
        "max_joint_acc_rad_s2": float(acc.max()) if len(acc) else 0.0,
        "max_joint_vel_ratio": float(transit_ratio),
        "max_joint_acc_ratio": float(np.max(acc / a_lim[None, :])) if len(acc) else 0.0,
    }
    return times, stats
