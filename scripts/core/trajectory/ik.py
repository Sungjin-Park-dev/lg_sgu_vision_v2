"""Inverse-kinematics solving and representative selection."""

from __future__ import annotations

import time

import numpy as np
import torch
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import GoalToolPose, Pose

from common import config
from .robot import _tick
from .settings import DP_CANDIDATE_DEDUP_RAD, IK_RANDOM_SEED

def normalize_joints(q):
    """Joint angles를 [-π, π] 범위로 정규화. 형상 유지."""
    return ((q + np.pi) % (2 * np.pi)) - np.pi


# =========================================================================
# Phase 1: Multi-seed IK
# =========================================================================

def solve_ik_multi_seed(robot_cfg, world_scene, positions_np, quats_np,
                        num_seeds=100, batch_size=4,
                        random_seed=IK_RANDOM_SEED):
    """각 pose에 대해 num_seeds개 IK 해를 구한다.

    Args:
        robot_cfg: cuRobo robot config (dict)
        world_scene: 충돌 Scene
        positions_np: (N, 3) EE positions
        quats_np: (N, 4) EE quaternions (w, x, y, z)
        num_seeds: IK seed 수
        batch_size: GPU 배치 크기
        random_seed: cuRobo IK seed bank 생성 seed. 같은 값이면 pose/batch와
            무관하게 동일한 joint seed bank을 사용한다.

    Returns:
        all_solutions: (N, num_seeds, 6)
        all_success: (N, num_seeds) bool
    """
    cache = {
        "obb": max(1, len(world_scene.cuboid)),
        "mesh": max(1, len(world_scene.mesh)),
    }
    _t_build = time.time()
    cfg = InverseKinematicsCfg.create(
        robot=robot_cfg,
        scene_model={},
        self_collision_check=True,
        num_seeds=num_seeds,
        max_batch_size=batch_size,
        use_cuda_graph=False,
        random_seed=int(random_seed),
        collision_cache=cache,
    )
    ik = InverseKinematics(cfg)
    ik.update_world(world_scene)
    tool = ik.tool_frames[0]
    _tick("ik_build", _t_build)

    N = len(positions_np)
    n_dof = 6
    # cuRobo에 seed_config을 주지 않으면 내부 sampler의 현재 상태와
    # batch padding 순서에 따라 후보군이 달라질 수 있다. Solver가 가진 만큼의
    # 고정 Halton bank를 한 번만 만들고 모든 pose에 명시적으로 재사용한다.
    # 이렇게 하면 서로 다른 앱/프로세스도 동일한 IK 초기값에서 시작한다.
    seed_bank = ik.prepare_action_seeds(
        batch_size=1, num_seeds=num_seeds,
    ).reshape(1, num_seeds, n_dof)
    all_solutions = np.zeros((N, num_seeds, n_dof), dtype=np.float64)
    all_success = np.zeros((N, num_seeds), dtype=bool)

    n_batches = (N + batch_size - 1) // batch_size
    t0 = time.time()

    for b in range(n_batches):
        s = b * batch_size
        e = min(s + batch_size, N)

        bp = torch.tensor(positions_np[s:e], device="cuda:0", dtype=torch.float32)
        bq = torch.tensor(quats_np[s:e], device="cuda:0", dtype=torch.float32)
        goal = Pose(position=bp, quaternion=bq)
        batch_seed_config = seed_bank.repeat(e - s, 1, 1)

        result = ik.solve_pose(
            GoalToolPose.from_poses({tool: goal}, num_goalset=1),
            seed_config=batch_seed_config,
            return_seeds=num_seeds,
        )

        sol = result.js_solution.position.cpu().numpy()
        if sol.shape[-1] != n_dof:
            sol = sol[..., :n_dof]
        all_solutions[s:e] = sol
        all_success[s:e] = result.success.cpu().numpy()

        if (b + 1) % 50 == 0 or b == n_batches - 1:
            elapsed = time.time() - t0
            print(f"    IK batch {b+1}/{n_batches} ({elapsed:.1f}s)")

    _tick("ik_solve", t0)

    # [-π, π]로 정규화 — 2π 차이 oscillation 방지
    all_solutions = normalize_joints(all_solutions)

    total_success = all_success.sum()
    print(f"  Phase 1 done: {total_success}/{N * num_seeds} IK solutions "
          f"({total_success / (N * num_seeds) * 100:.1f}% success)")

    return all_solutions, all_success


# =========================================================================
# Phase 2: per-viewpoint 대표해 (성공 IK 해를 greedy near-duplicate 제거)
# =========================================================================

def cluster_ik_solutions(all_solutions, all_success):
    """viewpoint당 대표 해 추출 — 성공 IK 해에서 near-duplicate만 greedy 제거(모든 분기 보존).

    클러스터 medoid(클러스터당 1개)는 이웃과 연속인 분기를 버릴 수 있어, 대신 L∞ fine tolerance
    (DP_CANDIDATE_DEDUP_RAD)로 거의 동일한 seed만 제거하고 분기를 모두 후보로 남겨 DP가 직접 고른다.

    Args:
        all_solutions: (N, S, 6)
        all_success: (N, S) bool

    Returns:
        representatives: List[np.ndarray] — 각 원소 shape (K_i, 6)
    """
    N = all_solutions.shape[0]
    representatives = []
    total_reps = 0
    empty_count = 0

    for i in range(N):
        successful = all_solutions[i][all_success[i]]
        if len(successful) == 0:
            representatives.append(np.empty((0, 6)))
            empty_count += 1
            continue
        # greedy near-dup 제거: 이미 채택한 해와 L∞ > tol 인 해만 남긴다(분기 보존)
        kept = []
        for s in successful:
            if all(np.max(np.abs(s - k)) > DP_CANDIDATE_DEDUP_RAD for k in kept):
                kept.append(s)
        representatives.append(np.array(kept))
        total_reps += len(kept)

    avg_reps = total_reps / max(N - empty_count, 1)
    print(f"  Phase 2 done: {N} viewpoints → avg {avg_reps:.1f} representatives/viewpoint "
          f"(dedup={DP_CANDIDATE_DEDUP_RAD:.2f} rad, {empty_count} empty)")

    return representatives


# =========================================================================
# Phase 3: DP
# =========================================================================
