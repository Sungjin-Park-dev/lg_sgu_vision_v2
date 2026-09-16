"""Trajectory timing and corner slowdown."""

import numpy as np

def _quat_angle_xyzw(q0, q1):
    """Quaternion geodesic angle in radians. Input order: x, y, z, w."""
    q0 = q0 / max(np.linalg.norm(q0), 1e-12)
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    dot = abs(float(np.dot(q0, q1)))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def _corner_angles(points):
    """Polyline corner angles at waypoints. Returns length N, endpoints 0."""
    n = len(points)
    angles = np.zeros((n,), dtype=np.float64)
    if n < 3:
        return angles

    prev_vec = points[1:-1] - points[:-2]
    next_vec = points[2:] - points[1:-1]
    prev_norm = np.linalg.norm(prev_vec, axis=1)
    next_norm = np.linalg.norm(next_vec, axis=1)
    valid = (prev_norm > 1e-9) & (next_norm > 1e-9)
    if np.any(valid):
        u = prev_vec[valid] / prev_norm[valid, None]
        v = next_vec[valid] / next_norm[valid, None]
        cos_turn = np.sum(u * v, axis=1)
        angles[1:-1][valid] = np.arccos(np.clip(cos_turn, -1.0, 1.0))
    return angles


def _corner_slowdown_factors(ee_positions, joints,
                             threshold_rad=np.deg2rad(30.0),
                             max_slowdown=2.5):
    """Corner turn angle 기반 segment slowdown factor. Returns length N-1."""
    n = len(joints)
    if n < 2 or max_slowdown <= 1.0:
        return np.ones((max(n - 1, 0),), dtype=np.float64), {
            "n_slow_segments": 0,
            "max_corner_angle_deg": 0.0,
            "max_slowdown": 1.0,
        }

    ee_angles = _corner_angles(ee_positions)
    joint_angles = _corner_angles(joints)
    corner_angles = np.maximum(ee_angles, joint_angles)

    denom = max(np.pi - threshold_rad, 1e-9)
    wp_factor = np.ones((n,), dtype=np.float64)
    mask = corner_angles > threshold_rad
    if np.any(mask):
        alpha = np.clip((corner_angles[mask] - threshold_rad) / denom, 0.0, 1.0)
        wp_factor[mask] = 1.0 + alpha * (max_slowdown - 1.0)

    seg_factor = np.maximum(wp_factor[:-1], wp_factor[1:])
    stats = {
        "n_slow_segments": int((seg_factor > 1.001).sum()),
        "max_corner_angle_deg": float(np.rad2deg(corner_angles.max())),
        "max_slowdown": float(seg_factor.max()) if len(seg_factor) else 1.0,
    }
    return seg_factor, stats


def _joint_acc_per_node(dq, dt):
    """실행기가 보는 관절 가속도 — 구간 속도(dq/dt)의 노드 간 변화 / 인접 구간 평균 dt.

    실행기는 waypoint 사이를 직선 보간하므로 관절 속도는 구간마다 상수이고 노드에서 바뀐다.
    그 변화를 두 구간의 평균 시간으로 나눈 것이 로봇이 실제로 겪는 가속도다. (N-2, dof)
    """
    vj = dq / dt[:, None]
    return np.abs(np.diff(vj, axis=0)) / ((dt[:-1] + dt[1:]) / 2.0)[:, None]


def _limit_joint_acceleration(joints, dt_min, a_max, max_iter=30):
    """가속도 상한을 만족하도록 구간 시간을 늘린다 → (dt, 노드 속도).

    경로 위 속도 프로파일이다. 구간 길이는 관절 L∞(가장 많이 도는 관절), 속도 상한은
    이미 계산된 최소 구간시간에서 온다(cap = 길이/dt_min). 노드 속도를 양 끝 0 에서
    출발해 전진 패스(v_k ≤ √(v_{k-1}² + 2aL)), 후진 패스로 깎으면 경로 방향의 가감속이
    a_max 안에 든다. 구간 시간은 등가속 가정으로 2L/(v_i+v_j). dt_min 보다 짧아지는 일은
    없다(노드 속도가 cap 을 넘지 않으므로).

    꺾임(인접 구간의 관절 방향이 다른 곳)은 경로 속도가 같아도 관절 속도 벡터가 꺾이며
    가속도가 튄다. 그것까지 _joint_acc_per_node 로 **실행기가 겪는 정의 그대로** 재서,
    넘는 노드의 속도를 √(초과비) 로 줄이고 패스를 다시 도는 고정점 반복으로 맞춘다 —
    보고되는 가속도와 강제하는 가속도가 같은 수다. 예전 '코너 감속' 휴리스틱의 자리다.
    """
    import math
    n = len(joints)
    dq = np.diff(joints, axis=0)
    seg_len = np.max(np.abs(dq), axis=1)                             # (n-1,) L∞
    moving = seg_len > 1e-12
    cap = np.full(n - 1, np.inf)
    cap[moving] = seg_len[moving] / np.maximum(dt_min[moving], 1e-9)
    v = np.zeros(n)
    v[1:-1] = np.minimum(cap[:-1], cap[1:])                          # 노드 = 양옆 구간의 작은 cap

    def passes(v):
        for k in range(1, n):
            v[k] = min(v[k], math.sqrt(v[k - 1] ** 2 + 2.0 * a_max * seg_len[k - 1]))
        for k in range(n - 2, -1, -1):
            v[k] = min(v[k], math.sqrt(v[k + 1] ** 2 + 2.0 * a_max * seg_len[k]))
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
        ratio = np.max(_joint_acc_per_node(dq, dt), axis=1) / a_max   # (n-2,) 노드별 초과비
        bad = np.flatnonzero(ratio > 1.0 + 1e-3)
        if len(bad) == 0:
            break
        # 노드 k(=구간 k-1 과 k 사이)의 속도를 줄인다. 양옆 구간 평균 속도가 함께 내려가
        # 속도 변화는 줄고 평균 dt 는 늘어 초과비가 단조 감소한다.
        for k in bad:
            v[k + 1] /= math.sqrt(ratio[k])
        dt = passes(v)
    return dt, v


def compute_trajectory_times(joints, ee_positions, ee_quaternions,
                             ee_speed_m_s=0.08,
                             ee_angular_speed_rad_s=np.deg2rad(30.0),
                             max_joint_vel_rad_s=0.5,
                             min_segment_dt=0.05,
                             corner_angle_threshold_rad=np.deg2rad(30.0),
                             corner_max_slowdown=2.5,
                             is_transit=None,
                             max_joint_acc_rad_s2=0.0):
    """Continuous scan용 누적 time 생성.

    스캔 segment 시간은 EE 선속도, EE 각속도, joint 속도 제한을 모두 만족하는 최소 시간으로
    정한다. 단 transit(재배치) segment(is_transit로 표시)는 검사가 아니므로 EE 선속도/각속도/
    corner slowdown을 적용하지 않고 **joint 속도 한계로만** 시간을 매긴다. 그래야 base를 크게
    돌리며 팔 끝이 자유공간에 그리는 긴 호를 50mm/s로 기어가지 않고, joint 속도로 빠르게
    지나간다(예: 139° transit 105s → ~8s).
    """
    n = len(joints)
    times = np.zeros((n,), dtype=np.float64)
    if n < 2:
        return times, {
            "total_time": 0.0,
            "max_linear_speed_mm_s": 0.0,
            "max_angular_speed_deg_s": 0.0,
            "max_joint_speed_rad_s": 0.0,
            "transit_time": 0.0,
        }

    slowdown_factors, corner_stats = _corner_slowdown_factors(
        ee_positions, joints,
        threshold_rad=corner_angle_threshold_rad,
        max_slowdown=corner_max_slowdown,
    )

    # segment(i-1→i)는 양 끝 중 하나라도 transit이면 transit으로 본다(스캔↔transit 진입/이탈 포함).
    if is_transit is not None and len(is_transit) == n:
        it = np.asarray(is_transit, dtype=bool)
        seg_is_transit = it[1:] | it[:-1]
    else:
        seg_is_transit = np.zeros((n - 1,), dtype=bool)

    # 1) 구간별 최소 시간 — 속도 제한(EE 선/각속도·관절 속도)과 코너 감속.
    seg_dt = np.zeros((n - 1,), dtype=np.float64)
    for i in range(1, n):
        linear_dist = float(np.linalg.norm(ee_positions[i] - ee_positions[i - 1]))
        angular_dist = _quat_angle_xyzw(ee_quaternions[i - 1], ee_quaternions[i])
        joint_dist = float(np.max(np.abs(joints[i] - joints[i - 1])))

        if seg_is_transit[i - 1]:
            # 재배치: joint 속도 한계로만 (EE 속도/corner 무시)
            dt_candidates = [min_segment_dt]
            if max_joint_vel_rad_s > 0.0:
                dt_candidates.append(joint_dist / max_joint_vel_rad_s)
            dt = max(dt_candidates)
        else:
            dt_candidates = [min_segment_dt]
            if ee_speed_m_s > 0.0:
                dt_candidates.append(linear_dist / ee_speed_m_s)
            if ee_angular_speed_rad_s > 0.0:
                dt_candidates.append(angular_dist / ee_angular_speed_rad_s)
            if max_joint_vel_rad_s > 0.0:
                dt_candidates.append(joint_dist / max_joint_vel_rad_s)
            dt = max(dt_candidates) * slowdown_factors[i - 1]
        seg_dt[i - 1] = dt

    # 2) 가속도 제한 — 속도 제한이 정한 시간을 늘리기만 한다(시작·끝 정지 포함).
    node_speed = None
    if max_joint_acc_rad_s2 and max_joint_acc_rad_s2 > 0.0:
        seg_dt, node_speed = _limit_joint_acceleration(joints, seg_dt, float(max_joint_acc_rad_s2))

    times[1:] = np.cumsum(seg_dt)
    transit_time = float(seg_dt[seg_is_transit].sum())
    segment_dt = np.diff(times)
    linear_speed = np.linalg.norm(np.diff(ee_positions, axis=0), axis=1) / segment_dt
    angular_speed = np.array([
        _quat_angle_xyzw(ee_quaternions[i - 1], ee_quaternions[i]) / segment_dt[i - 1]
        for i in range(1, n)
    ])
    joint_speed = np.max(np.abs(np.diff(joints, axis=0)), axis=1) / segment_dt
    # EE 선/각속도 최대는 '스캔 구간'에서만 의미가 있다(transit은 EE가 자유공간을 빠르게 휘둘러
    # EE 속도가 매우 커지지만 검사 속도가 아님). joint 속도 최대는 전체에서 본다.
    scan_seg = ~seg_is_transit
    scan_lin = linear_speed[scan_seg] if scan_seg.any() else linear_speed
    scan_ang = angular_speed[scan_seg] if scan_seg.any() else angular_speed
    stats = {
        "total_time": float(times[-1]),
        "transit_time": float(transit_time),
        "n_transit_segments": int(seg_is_transit.sum()),
        "max_linear_speed_mm_s": float(scan_lin.max() * 1000.0),
        "max_angular_speed_deg_s": float(np.rad2deg(scan_ang.max())),
        "max_joint_speed_rad_s": float(joint_speed.max()),
        # 실행기가 겪는 관절 가속도 최대(구간 속도의 노드 간 변화 / 평균 dt). 제한을 켰으면
        # 이 값이 상한 이하다 — 강제할 때와 같은 정의로 잰다.
        "max_joint_acc_rad_s2": (
            float(np.max(_joint_acc_per_node(np.diff(joints, axis=0), segment_dt)))
            if n >= 3 else 0.0),
        **corner_stats,
    }
    return times, stats


# =========================================================================
# CSV output
# =========================================================================
