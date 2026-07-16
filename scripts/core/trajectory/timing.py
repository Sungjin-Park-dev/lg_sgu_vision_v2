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


def compute_trajectory_times(joints, ee_positions, ee_quaternions,
                             ee_speed_m_s=0.08,
                             ee_angular_speed_rad_s=np.deg2rad(30.0),
                             max_joint_vel_rad_s=0.5,
                             min_segment_dt=0.05,
                             corner_angle_threshold_rad=np.deg2rad(30.0),
                             corner_max_slowdown=2.5,
                             is_transit=None):
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

    transit_time = 0.0
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
            transit_time += dt
        else:
            dt_candidates = [min_segment_dt]
            if ee_speed_m_s > 0.0:
                dt_candidates.append(linear_dist / ee_speed_m_s)
            if ee_angular_speed_rad_s > 0.0:
                dt_candidates.append(angular_dist / ee_angular_speed_rad_s)
            if max_joint_vel_rad_s > 0.0:
                dt_candidates.append(joint_dist / max_joint_vel_rad_s)
            dt = max(dt_candidates) * slowdown_factors[i - 1]

        times[i] = times[i - 1] + dt

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
        **corner_stats,
    }
    return times, stats


# =========================================================================
# CSV output
# =========================================================================
