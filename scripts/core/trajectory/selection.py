"""Dynamic-programming selection of a continuous IK branch."""

import numpy as np

from .settings import RECONFIG_PENALTY

def dp_optimal_path(representatives, reconfig_threshold_rad=0.5):
    """DP로 최적 경로 선택. 1순위: reconfig 최소화, 2순위: joint distance 최소화.

    비용: edge_cost = is_reconfig * RECONFIG_PENALTY + l2_distance
    여기서 is_reconfig = (L-inf > reconfig_threshold_rad)

    Args:
        representatives: List[np.ndarray] — 각 원소 shape (K_i, 6)
        reconfig_threshold_rad: reconfig 판정 임계값 (rad)

    Returns:
        selected: (N, 6) 선택된 joint 해
        total_cost: float
        stats: dict
    """
    N = len(representatives)

    # carry-forward: 빈 viewpoint 처리
    carry_forward_count = 0
    for i in range(N):
        if len(representatives[i]) == 0:
            if i > 0 and len(representatives[i - 1]) > 0:
                representatives[i] = representatives[i - 1].copy()
                carry_forward_count += 1
            else:
                for j in range(i + 1, N):
                    if len(representatives[j]) > 0:
                        representatives[i] = representatives[j].copy()
                        carry_forward_count += 1
                        break

    K_0 = len(representatives[0])
    if K_0 == 0:
        raise RuntimeError("No IK solutions found for any viewpoint")

    # DP tables
    dp_cost = [None] * N
    dp_parent = [None] * N

    dp_cost[0] = np.zeros(K_0)
    dp_parent[0] = np.full(K_0, -1, dtype=int)

    for i in range(1, N):
        K_curr = len(representatives[i])

        dp_cost[i] = np.full(K_curr, np.inf)
        dp_parent[i] = np.full(K_curr, -1, dtype=int)

        # 비용 행렬 계산 (K_prev × K_curr)
        prev_reps = representatives[i - 1]  # (K_prev, 6)
        curr_reps = representatives[i]      # (K_curr, 6)
        diff = prev_reps[:, np.newaxis, :] - curr_reps[np.newaxis, :, :]  # (K_prev, K_curr, 6)
        l2_costs = np.linalg.norm(diff, axis=2)        # (K_prev, K_curr)
        linf = np.max(np.abs(diff), axis=2)             # (K_prev, K_curr)

        # reconfig 페널티: L-inf > threshold → +1000
        reconfig_mask = linf > reconfig_threshold_rad  # (K_prev, K_curr)
        edge_costs = reconfig_mask.astype(float) * RECONFIG_PENALTY + l2_costs

        # 벡터화 DP 갱신: total[k,j] = dp_cost[i-1][k] + edge_costs[k,j]
        total = dp_cost[i - 1][:, np.newaxis] + edge_costs  # (K_prev, K_curr)
        dp_cost[i] = total.min(axis=0)
        dp_parent[i] = total.argmin(axis=0)

    # Backtrack
    selected = np.zeros((N, 6))
    current = int(np.argmin(dp_cost[N - 1]))
    selected[N - 1] = representatives[N - 1][current]
    total_cost = dp_cost[N - 1][current]

    for i in range(N - 2, -1, -1):
        current = int(dp_parent[i + 1][current])
        selected[i] = representatives[i][current]

    # 통계
    jumps = np.max(np.abs(np.diff(selected, axis=0)), axis=1)
    n_reconfigs = int((jumps > reconfig_threshold_rad).sum()) if len(jumps) > 0 else 0
    stats = {
        "total_cost": float(total_cost),
        "n_reconfigs": n_reconfigs,
        "carry_forward": carry_forward_count,
        "max_jump_deg": float(np.rad2deg(np.max(jumps))) if len(jumps) > 0 else 0,
        "mean_jump_deg": float(np.rad2deg(np.mean(jumps))) if len(jumps) > 0 else 0,
    }

    print(f"  Phase 3 done: {n_reconfigs} reconfigs, "
          f"max_jump={stats['max_jump_deg']:.1f}°, "
          f"mean_jump={stats['mean_jump_deg']:.1f}°, "
          f"carry_forward={carry_forward_count}")

    return selected, total_cost, stats


# =========================================================================
# Phase 4: MotionGen transit at reconfig points
# =========================================================================
