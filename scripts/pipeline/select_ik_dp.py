#!/usr/bin/env python3
"""
DBSCAN + DP + MotionGen 기반 최적 IK 궤적 생성

각 viewpoint에 대해 다수의 IK 해를 구하고, DP로 전역 최적 경로를 선택한 뒤,
reconfig 지점은 MotionGen으로 충돌회피 transit을 만들어 균일 spacing으로 resample한다.

단계:
    Phase 1: Multi-seed IK     — viewpoint당 num_seeds개 IK 해
    Phase 2: DBSCAN            — viewpoint당 대표 해 (medoid) 추출
    Phase 3: DP                — 최소 joint-space 비용 경로 선택
       ↓ wrist_3 잠금 (resample 균일성을 위해 metric에서 사실상 제외)
    Phase 4: MotionGen transit — reconfig 지점 충돌회피 joint-to-joint planning
    Phase 5: Uniform resample  — cumulative L2 arc-length로 균일 spacing + 충돌 검사

사용법:
    uv run scripts/pipeline/select_ik_dp.py --object sample --num-viewpoints 124 --viewpoints data/sample/viewpoint/124/viewpoints_coacd+dbscan.h5
"""

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.geom.types import Cuboid, Mesh as CuRoboMesh, WorldConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.types.robot import JointState as CuRoboJointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import config
from common.math_utils import quaternion_to_rotation_matrix, normalize_vectors


# =========================================================================
# Data loading & geometry
# =========================================================================

def load_viewpoints(h5_path: Path):
    """Load positions, normals, path_order, cluster_id, working_distance from HDF5."""
    if not h5_path.exists():
        raise FileNotFoundError(f"Viewpoints file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        grp = f["viewpoints"]
        positions = np.array(grp["positions"], dtype=np.float64)
        normals = np.array(grp["normals"], dtype=np.float64)
        path_order = np.array(grp["path_order"], dtype=np.int32) if "path_order" in grp else None
        cluster_id = np.array(grp["cluster_id"], dtype=np.int32) if "cluster_id" in grp else None

        wd_m = config.CAMERA_WORKING_DISTANCE_MM / 1000.0
        if "metadata" in f and "camera_spec" in f["metadata"]:
            cs = f["metadata"]["camera_spec"]
            if "working_distance_mm" in cs.attrs:
                wd_m = float(cs.attrs["working_distance_mm"]) / 1000.0

    return positions, normals, path_order, cluster_id, wd_m


def rot_to_quat_batch(R_batch: np.ndarray) -> np.ndarray:
    """Rotation matrices (N,3,3) → quaternions (N,4) as (w,x,y,z)."""
    batch_size = R_batch.shape[0]
    quats = np.zeros((batch_size, 4), dtype=np.float64)
    for i in range(batch_size):
        r = Rotation.from_matrix(R_batch[i])
        q_xyzw = r.as_quat()
        quats[i] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
    return quats


def build_camera_poses(positions, normals, working_distance_m):
    """Surface position + normal → camera 4x4 poses (N,4,4) in world frame."""
    safe_normals = normalize_vectors(normals)
    camera_positions = positions + safe_normals * working_distance_m
    approach = -safe_normals

    helper_z = np.array([0.0, 0.0, 1.0])
    helper_y = np.array([0.0, 1.0, 0.0])

    N = len(positions)
    local_poses = np.zeros((N, 4, 4), dtype=np.float64)

    for i in range(N):
        z_axis = approach[i] / np.linalg.norm(approach[i])
        helper = helper_z if abs(np.dot(z_axis, helper_z)) <= 0.99 else helper_y
        x_axis = np.cross(helper, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        local_poses[i, :3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
        local_poses[i, :3, 3] = camera_positions[i]
        local_poses[i, 3, 3] = 1.0

    target_world = np.eye(4, dtype=np.float64)
    target_world[:3, :3] = quaternion_to_rotation_matrix(config.TARGET_OBJECT["rotation"])
    target_world[:3, 3] = config.TARGET_OBJECT["position"]

    return np.einsum("ij,njk->nik", target_world, local_poses)


def build_collision_world(object_name: str):
    """Build cuRobo WorldConfig from config.py obstacles + target object mesh."""
    import trimesh

    cuboids = []
    for obj in [config.TABLE, config.ROBOT_MOUNT] + config.WALLS:
        cuboids.append(Cuboid(
            name=obj["name"],
            pose=[*obj["position"].tolist(), 1, 0, 0, 0],
            dims=obj["dimensions"].tolist(),
        ))

    meshes = []
    mesh_path = config.get_mesh_path(object_name, mesh_type="source")
    if mesh_path.exists():
        loaded = trimesh.load(str(mesh_path))
        if isinstance(loaded, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
        else:
            mesh = loaded
        pos = config.TARGET_OBJECT["position"]
        rot = config.TARGET_OBJECT["rotation"]
        meshes.append(CuRoboMesh(
            name="target_object",
            pose=[pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]],
            vertices=mesh.vertices.tolist(),
            faces=mesh.faces.flatten().tolist(),
        ))
        print(f"  Collision world: {len(cuboids)} cuboids + target mesh ({len(mesh.faces)} faces)")
    else:
        print(f"  Warning: Target mesh not found at {mesh_path}, skipping mesh collision")

    return WorldConfig(cuboid=cuboids, mesh=meshes if meshes else None)


def compute_fk(solutions, robot_cfg_file):
    """Compute FK for joint solutions. Returns (N,3) positions and (N,4) quats (x,y,z,w)."""
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_cfg_file))["robot_cfg"]
    ta = TensorDeviceType(device=torch.device("cuda:0"), dtype=torch.float32)

    rw_config = RobotWorldConfig.load_from_config(robot_cfg, WorldConfig(), tensor_args=ta)
    robot_world = RobotWorld(rw_config)

    q_batch = torch.tensor(solutions, device="cuda:0", dtype=torch.float32)
    state = robot_world.get_kinematics(q_batch)
    ee_positions = state.ee_position.cpu().numpy()
    ee_quat_wxyz = state.ee_quaternion.cpu().numpy()
    ee_quaternions = ee_quat_wxyz[:, [1, 2, 3, 0]]

    return ee_positions, ee_quaternions


# =========================================================================
# Joint angle normalization
# =========================================================================

def normalize_joints(q):
    """Joint angles를 [-π, π] 범위로 정규화. 형상 유지."""
    return ((q + np.pi) % (2 * np.pi)) - np.pi


# =========================================================================
# Phase 1: Multi-seed IK
# =========================================================================

def solve_ik_multi_seed(ik_solver, positions_np, quats_np, tensor_args,
                        num_seeds=100, batch_size=4):
    """각 pose에 대해 num_seeds개 IK 해를 구한다.

    Args:
        ik_solver: cuRobo IKSolver
        positions_np: (N, 3) EE positions
        quats_np: (N, 4) EE quaternions (w, x, y, z)
        tensor_args: TensorDeviceType
        num_seeds: IK seed 수
        batch_size: GPU 배치 크기

    Returns:
        all_solutions: (N, num_seeds, 6)
        all_success: (N, num_seeds) bool
    """
    N = len(positions_np)
    n_dof = 6
    all_solutions = np.zeros((N, num_seeds, n_dof), dtype=np.float64)
    all_success = np.zeros((N, num_seeds), dtype=bool)

    n_batches = (N + batch_size - 1) // batch_size
    t0 = time.time()

    for b in range(n_batches):
        s = b * batch_size
        e = min(s + batch_size, N)

        bp = torch.tensor(positions_np[s:e], device=tensor_args.device, dtype=tensor_args.dtype)
        bq = torch.tensor(quats_np[s:e], device=tensor_args.device, dtype=tensor_args.dtype)
        goal = Pose(position=bp, quaternion=bq)

        result = ik_solver.solve_batch(goal, num_seeds=num_seeds, return_seeds=num_seeds)

        all_solutions[s:e] = result.solution.cpu().numpy()
        all_success[s:e] = result.success.cpu().numpy()

        if (b + 1) % 50 == 0 or b == n_batches - 1:
            elapsed = time.time() - t0
            print(f"    IK batch {b+1}/{n_batches} ({elapsed:.1f}s)")

    # [-π, π]로 정규화 — 2π 차이 oscillation 방지
    all_solutions = normalize_joints(all_solutions)

    total_success = all_success.sum()
    print(f"  Phase 1 done: {total_success}/{N * num_seeds} IK solutions "
          f"({total_success / (N * num_seeds) * 100:.1f}% success)")

    return all_solutions, all_success


# =========================================================================
# Phase 2: DBSCAN clustering per viewpoint
# =========================================================================

def cluster_ik_solutions(all_solutions, all_success, eps=0.3, min_samples=1):
    """DBSCAN으로 viewpoint당 대표 해(medoid) 추출.

    Args:
        all_solutions: (N, S, 6)
        all_success: (N, S) bool
        eps: DBSCAN eps (radians)
        min_samples: DBSCAN min_samples

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

        if len(successful) == 1:
            representatives.append(successful.copy())
            total_reps += 1
            continue

        db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = db.fit_predict(successful)

        medoids = []
        for label in np.unique(labels):
            if label == -1:
                # noise → 각각 singleton으로 취급
                for p in successful[labels == -1]:
                    medoids.append(p)
            else:
                members = successful[labels == label]
                mean = members.mean(axis=0)
                dists = np.linalg.norm(members - mean, axis=1)
                medoids.append(members[np.argmin(dists)])

        representatives.append(np.array(medoids))
        total_reps += len(medoids)

    avg_reps = total_reps / max(N - empty_count, 1)
    print(f"  Phase 2 done: {N} viewpoints → avg {avg_reps:.1f} representatives/viewpoint "
          f"(eps={eps:.2f} rad, {empty_count} empty)")

    return representatives


# =========================================================================
# Phase 3: DP
# =========================================================================

RECONFIG_PENALTY = 1000.0


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
        K_prev = len(representatives[i - 1])
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

        for j in range(K_curr):
            for k in range(K_prev):
                c = dp_cost[i - 1][k] + edge_costs[k, j]
                if c < dp_cost[i][j]:
                    dp_cost[i][j] = c
                    dp_parent[i][j] = k

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

def plan_reconfig_transits(
    selected, reconfig_indices, robot_cfg_file, world_config, tensor_args,
):
    """Reconfig 지점마다 MotionGen joint-to-joint planning 수행.

    Args:
        selected: (N, 6) DP로 선택된 joint trajectory
        reconfig_indices: reconfig이 발생하는 transition 인덱스 배열
        robot_cfg_file: cuRobo robot config 파일명
        world_config: 충돌 세계 설정
        tensor_args: TensorDeviceType

    Returns:
        transit_segments: dict {idx: (T, 6) transit trajectory} — 성공한 것만
        transit_stats: list of dicts
    """
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg_file, world_config,
        tensor_args=tensor_args,
        interpolation_dt=0.02,
        use_cuda_graph=False,
    )
    motion_gen = MotionGen(motion_gen_config)
    print("    Warming up MotionGen...")
    motion_gen.warmup()

    plan_cfg = MotionGenPlanConfig(
        max_attempts=10,
        timeout=5.0,
        enable_graph=False,
        enable_opt=True,
        enable_finetune_trajopt=True,
    )

    transit_segments = {}
    transit_stats = []

    for idx in reconfig_indices:
        start_q = torch.tensor(
            selected[idx], device=tensor_args.device, dtype=tensor_args.dtype,
        ).unsqueeze(0)
        goal_q = torch.tensor(
            selected[idx + 1], device=tensor_args.device, dtype=tensor_args.dtype,
        ).unsqueeze(0)

        start_state = CuRoboJointState.from_position(start_q)
        goal_state = CuRoboJointState.from_position(goal_q)

        t0 = time.time()
        result = motion_gen.plan_single_js(start_state, goal_state, plan_cfg.clone())
        dt = time.time() - t0

        if result.success.item():
            traj = result.get_interpolated_plan()
            waypoints = traj.position.cpu().numpy()
            transit_segments[idx] = waypoints
            transit_stats.append({
                "idx": idx, "success": True,
                "n_waypoints": len(waypoints), "time": dt,
            })
            print(f"    {idx}→{idx+1}: OK ({len(waypoints)} waypoints, {dt:.2f}s)")
        else:
            transit_stats.append({
                "idx": idx, "success": False,
                "status": str(result.status), "time": dt,
            })
            print(f"    {idx}→{idx+1}: FAILED ({result.status}, {dt:.2f}s)")

    n_ok = sum(1 for s in transit_stats if s["success"])
    print(f"  Transit planning: {n_ok}/{len(reconfig_indices)} succeeded")

    return transit_segments, transit_stats


# =========================================================================
# Phase 5: Uniform resample + collision check
# =========================================================================

def interpolate_and_resample(selected, transit_segments, spacing_rad=0.02):
    """DP 궤적 + transit을 합치고, joint-space uniform spacing으로 resample.

    Non-reconfig 구간: joint-space linear interpolation
    Transit 구간: MotionGen의 dense 경로를 그대로 사용

    Args:
        selected: (N, 6) DP 선택 궤적
        transit_segments: dict {idx: (T, 6)} transit 경로
        spacing_rad: uniform spacing in joint-space L2 norm (radians)

    Returns:
        resampled: (M, 6) uniform-spaced trajectory
    """
    N = len(selected)

    # 1) 모든 구간을 dense path로 연결
    dense_segments = []
    for i in range(N - 1):
        dense_segments.append(selected[i:i+1])

        if i in transit_segments:
            # transit 구간: 이미 dense, 첫/끝 제외
            transit = transit_segments[i]
            if len(transit) > 2:
                dense_segments.append(transit[1:-1])
        else:
            # non-reconfig: linear interpolation
            q0, q1 = selected[i], selected[i + 1]
            dist = np.linalg.norm(q1 - q0)
            n_steps = max(1, int(np.ceil(dist / spacing_rad)))
            if n_steps > 1:
                alphas = np.linspace(0, 1, n_steps + 1)[1:-1]  # 양 끝 제외
                interp = q0[np.newaxis, :] + alphas[:, np.newaxis] * (q1 - q0)[np.newaxis, :]
                dense_segments.append(interp)

    dense_segments.append(selected[-1:])  # 마지막 점
    dense_path = np.concatenate(dense_segments, axis=0)

    # 2) Cumulative arc length 계산
    diffs = np.linalg.norm(np.diff(dense_path, axis=0), axis=1)
    cum_len = np.concatenate([[0], np.cumsum(diffs)])
    total_len = cum_len[-1]

    if total_len < 1e-9:
        return dense_path

    # 3) Uniform spacing으로 resample
    n_out = max(2, int(np.ceil(total_len / spacing_rad)) + 1)
    uniform_s = np.linspace(0, total_len, n_out)

    resampled = np.zeros((n_out, selected.shape[1]), dtype=np.float64)
    for j in range(selected.shape[1]):
        resampled[:, j] = np.interp(uniform_s, cum_len, dense_path[:, j])

    return resampled


def batch_collision_check(trajectory, robot_cfg_file, world_config, tensor_args):
    """전체 궤적에 대해 batch collision check 수행. Returns (is_collision, n_collisions)."""
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_cfg_file))["robot_cfg"]
    rw_config = RobotWorldConfig.load_from_config(robot_cfg, world_config, tensor_args=tensor_args)
    robot_world = RobotWorld(rw_config)

    q_tensor = torch.tensor(trajectory, device=tensor_args.device, dtype=tensor_args.dtype)
    d_world, d_self = robot_world.get_world_self_collision_distance_from_joints(q_tensor)

    # 음수 거리 = 충돌
    is_self_collision = (d_self < 0).any(dim=-1).cpu().numpy()
    is_world_collision = (d_world < 0).any(dim=-1).cpu().numpy()
    is_collision = is_self_collision | is_world_collision
    n_collisions = int(is_collision.sum())

    return is_collision, n_collisions


# =========================================================================
# CSV output
# =========================================================================

def save_trajectory_csv(solutions, ee_positions, ee_quaternions, output_path,
                        robot_name="ur20", dt=1.0):
    """Trajectory를 CSV로 저장. joint 컬럼에 robot_name prefix 추가."""
    import csv

    JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    header = ["time"] + [f"{robot_name}-{j}" for j in JOINT_NAMES] + [
        "target-POS_X", "target-POS_Y", "target-POS_Z",
        "target-ROT_X", "target-ROT_Y", "target-ROT_Z", "target-ROT_W",
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(solutions)):
            row = [i * dt] + solutions[i].tolist()
            row += ee_positions[i].tolist()
            row += ee_quaternions[i].tolist()
            writer.writerow(row)

    print(f"  CSV saved to {output_path} ({len(solutions)} waypoints)")


# =========================================================================
# Visualization
# =========================================================================

_JOINT_JUMP_THRESH_RAD = 0.5  # reconfig 판정 임계값

def _load_object_mesh_traces(object_name):
    """대상 메쉬를 world frame으로 변환하여 Plotly trace 반환."""
    import trimesh
    import plotly.graph_objects as go

    mesh_path = config.get_mesh_path(object_name, mesh_type="source")
    if not mesh_path.exists():
        return []

    loaded = trimesh.load(str(mesh_path))
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
    else:
        mesh = loaded

    from common.math_utils import quaternion_to_rotation_matrix
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quaternion_to_rotation_matrix(config.TARGET_OBJECT["rotation"])
    T[:3, 3] = config.TARGET_OBJECT["position"]

    verts = np.array(mesh.vertices)
    verts_h = np.c_[verts, np.ones(len(verts))]
    verts_w = (T @ verts_h.T).T[:, :3]
    faces = mesh.faces

    return [go.Mesh3d(
        x=verts_w[:, 0], y=verts_w[:, 1], z=verts_w[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color='lightgray', opacity=0.25,
        name='Mesh', hoverinfo='skip',
    )]


def _detect_reconfigs(joints):
    """연속 waypoint 간 max joint diff로 reconfig 판정. Returns (N-1,) bool."""
    diffs = np.max(np.abs(np.diff(joints, axis=0)), axis=1)
    return diffs > _JOINT_JUMP_THRESH_RAD


def visualize_static_html(object_name, joints, ee_positions, output_path):
    """Static trajectory: EE path + reconfig 강조 + 메쉬.

    Args:
        object_name: 대상 객체 이름
        joints: (N, 6) joint angles
        ee_positions: (N, 3) EE positions
        output_path: 출력 HTML 경로
    """
    import plotly.graph_objects as go

    N = len(ee_positions)
    reconfigs = _detect_reconfigs(joints)
    n_rc = int(reconfigs.sum())

    traces = _load_object_mesh_traces(object_name)

    # Normal / Reconfig 구간 분리
    norm_x, norm_y, norm_z = [], [], []
    rc_x, rc_y, rc_z = [], [], []

    for i in range(N - 1):
        seg_x = [ee_positions[i, 0], ee_positions[i + 1, 0], None]
        seg_y = [ee_positions[i, 1], ee_positions[i + 1, 1], None]
        seg_z = [ee_positions[i, 2], ee_positions[i + 1, 2], None]
        if reconfigs[i]:
            rc_x.extend(seg_x); rc_y.extend(seg_y); rc_z.extend(seg_z)
        else:
            norm_x.extend(seg_x); norm_y.extend(seg_y); norm_z.extend(seg_z)

    traces.append(go.Scatter3d(
        x=norm_x, y=norm_y, z=norm_z, mode='lines',
        line=dict(color='green', width=3),
        name=f'Normal ({N - 1 - n_rc} segments)',
    ))
    if n_rc > 0:
        traces.append(go.Scatter3d(
            x=rc_x, y=rc_y, z=rc_z, mode='lines',
            line=dict(color='red', width=4),
            name=f'Reconfig ({n_rc} segments)',
        ))

    # Step markers (색상 그라디언트)
    traces.append(go.Scatter3d(
        x=ee_positions[:, 0], y=ee_positions[:, 1], z=ee_positions[:, 2],
        mode='markers',
        marker=dict(size=3, color=np.arange(N), colorscale='Viridis',
                    colorbar=dict(title='Step', x=1.05), opacity=0.8),
        text=[f'Step {i}' for i in range(N)],
        hoverinfo='text',
        name='Poses',
    ))

    # Start / End
    traces.append(go.Scatter3d(
        x=[ee_positions[0, 0]], y=[ee_positions[0, 1]], z=[ee_positions[0, 2]],
        mode='markers', marker=dict(size=8, color='lime', symbol='diamond'),
        name='Start',
    ))
    traces.append(go.Scatter3d(
        x=[ee_positions[-1, 0]], y=[ee_positions[-1, 1]], z=[ee_positions[-1, 2]],
        mode='markers', marker=dict(size=8, color='orange', symbol='square'),
        name='End',
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f'Trajectory — {object_name} | {N} poses, {n_rc} reconfigs '
              f'({100 * n_rc / max(N - 1, 1):.1f}%)',
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
                   aspectmode='data'),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=80, b=0),
        width=1200, height=800,
    )
    fig.write_html(output_path)
    print(f"  Static HTML saved to {output_path}")


def visualize_animated_html(object_name, joints, ee_positions, output_path):
    """Animated trajectory: 슬라이더로 step별 경로 성장 애니메이션.

    Args:
        object_name: 대상 객체 이름
        joints: (N, 6) joint angles
        ee_positions: (N, 3) EE positions
        output_path: 출력 HTML 경로
    """
    import plotly.graph_objects as go

    N = len(ee_positions)
    reconfigs = _detect_reconfigs(joints)
    n_rc = int(reconfigs.sum())

    fig = go.Figure()

    # Trace 0: Mesh
    mesh_traces = _load_object_mesh_traces(object_name)
    for t in mesh_traces:
        fig.add_trace(t)
    n_fixed = len(mesh_traces)

    # Trace n_fixed+0: Full path (dim)
    fig.add_trace(go.Scatter3d(
        x=ee_positions[:, 0], y=ee_positions[:, 1], z=ee_positions[:, 2],
        mode='lines+markers',
        line=dict(color='lightgray', width=2),
        marker=dict(size=2, color='lightgray'),
        name='Full path',
    ))

    # Trace n_fixed+1: Current EE marker
    fig.add_trace(go.Scatter3d(
        x=[ee_positions[0, 0]], y=[ee_positions[0, 1]], z=[ee_positions[0, 2]],
        mode='markers', marker=dict(size=6, color='red'),
        name='Current pose',
    ))

    # Trace n_fixed+2: Normal path so far (green)
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[], mode='lines', line=dict(color='green', width=4),
        name='Normal',
    ))

    # Trace n_fixed+3: Reconfig path so far (red)
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[], mode='lines', line=dict(color='red', width=4),
        name='Reconfig',
    ))

    idx_marker = n_fixed + 1
    idx_norm = n_fixed + 2
    idx_rc = n_fixed + 3

    # Frames
    print(f"  Building {N} animation frames...")
    frames = []
    for step in range(N):
        rc_so_far = int(reconfigs[:max(step, 1)].sum()) if step > 0 else 0

        norm_x, norm_y, norm_z = [], [], []
        rc_x, rc_y, rc_z = [], [], []
        for i in range(step):
            seg = ([ee_positions[i, 0], ee_positions[i + 1, 0], None],
                   [ee_positions[i, 1], ee_positions[i + 1, 1], None],
                   [ee_positions[i, 2], ee_positions[i + 1, 2], None])
            if reconfigs[i]:
                rc_x.extend(seg[0]); rc_y.extend(seg[1]); rc_z.extend(seg[2])
            else:
                norm_x.extend(seg[0]); norm_y.extend(seg[1]); norm_z.extend(seg[2])

        frames.append(go.Frame(
            data=[
                go.Scatter3d(
                    x=[ee_positions[step, 0]], y=[ee_positions[step, 1]],
                    z=[ee_positions[step, 2]],
                    mode='markers', marker=dict(size=6, color='red'),
                ),
                go.Scatter3d(
                    x=norm_x, y=norm_y, z=norm_z,
                    mode='lines', line=dict(color='green', width=4),
                ),
                go.Scatter3d(
                    x=rc_x, y=rc_y, z=rc_z,
                    mode='lines', line=dict(color='red', width=4),
                ),
            ],
            traces=[idx_marker, idx_norm, idx_rc],
            name=str(step),
            layout=go.Layout(
                title_text=f'Step {step}/{N-1} | Reconfigs: {rc_so_far}',
            ),
        ))

    fig.frames = frames

    # Slider
    sliders = [dict(
        active=0,
        currentvalue=dict(prefix='Step: '),
        pad=dict(t=50),
        steps=[
            dict(args=[[str(s)], dict(frame=dict(duration=0, redraw=True),
                                       mode='immediate')],
                 label=str(s), method='animate')
            for s in range(N)
        ],
    )]

    # Play/Pause
    updatemenus = [dict(
        type='buttons', showactive=False,
        x=0.1, y=0, xanchor='right', yanchor='top',
        pad=dict(t=87, r=10),
        buttons=[
            dict(label='Play', method='animate',
                 args=[None, dict(frame=dict(duration=200, redraw=True),
                                  fromcurrent=True, transition=dict(duration=0))]),
            dict(label='Pause', method='animate',
                 args=[[None], dict(frame=dict(duration=0, redraw=True),
                                    mode='immediate', transition=dict(duration=0))]),
        ],
    )]

    fig.update_layout(
        title=f'Animation — {object_name} | {N} poses, {n_rc} reconfigs',
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
                   aspectmode='data'),
        sliders=sliders,
        updatemenus=updatemenus,
        width=1200, height=800,
    )
    fig.write_html(output_path)
    print(f"  Animated HTML saved to {output_path}")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="DBSCAN + DP 기반 최적 IK 해 선택")
    parser.add_argument("--object", type=str, required=True, help="Object name")
    parser.add_argument("--num-viewpoints", type=int, required=True, help="Number of viewpoints")
    parser.add_argument("--viewpoints", type=str, default=None,
                        help="Direct path to viewpoints.h5 (overrides --object/--num-viewpoints for loading)")
    parser.add_argument("--robot", type=str, default=config.DEFAULT_ROBOT_CONFIG,
                        help=f"Robot config (default: {config.DEFAULT_ROBOT_CONFIG})")
    parser.add_argument("--num-seeds", type=int, default=100,
                        help="IK seeds per viewpoint (default: 100)")
    parser.add_argument("--ik-batch-size", type=int, default=4,
                        help="GPU batch size for IK solving (default: 4)")
    parser.add_argument("--dbscan-eps", type=float, default=0.3,
                        help="DBSCAN eps in radians (default: 0.3)")
    parser.add_argument("--reconfig-threshold", type=float, default=29.0,
                        help="Reconfig threshold in degrees (default: 29.0)")
    parser.add_argument("--spacing", type=float, default=0.1,
                        help="Uniform resample spacing in radians (default: 0.1)")
    parser.add_argument("--output-suffix", type=str, default="dp",
                        help="Output file suffix (default: dp)")
    args = parser.parse_args()

    # [1] Load viewpoints
    print("[1/6] Loading viewpoints...")
    h5_path = Path(args.viewpoints) if args.viewpoints \
        else config.get_viewpoint_path(args.object, args.num_viewpoints)
    positions, normals, path_order, cluster_id, wd_m = load_viewpoints(h5_path)
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

    # [3] Phase 1: Multi-seed IK
    print("[3/6] Phase 1 — Multi-seed IK...")
    tensor_args = TensorDeviceType()
    world_config = build_collision_world(args.object)
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg, world_config,
        self_collision_check=True,
        tensor_args=tensor_args,
        use_cuda_graph=False,
    )
    ik_solver = IKSolver(ik_config)

    all_solutions, all_success = solve_ik_multi_seed(
        ik_solver, positions_np, quats_np, tensor_args,
        num_seeds=args.num_seeds, batch_size=args.ik_batch_size,
    )

    # [4] Phase 2 + 3: DBSCAN → DP
    print("[4/6] Phase 2 — DBSCAN clustering...")
    representatives = cluster_ik_solutions(
        all_solutions, all_success, eps=args.dbscan_eps,
    )

    print("[5/6] Phase 3 — DP optimal path...")
    reconfig_rad = np.deg2rad(args.reconfig_threshold)
    selected, _, stats = dp_optimal_path(representatives, reconfig_rad)

    # wrist_3 고정 — Phase 4/5 전체가 일관된 wrist_3로 동작하여
    # resample 후 인접 row의 L2 spacing이 균일해진다 (5-DoF L2 = 6-DoF L2).
    wrist3_fixed = config.ROBOT_START_STATE[-1]
    selected[:, -1] = wrist3_fixed
    print(f"  Locked wrist_3 at {np.rad2deg(wrist3_fixed):.1f}° (pre-transit)")

    # 클러스터 간/내 reconfig 분석
    if cluster_id is not None:
        jumps = np.max(np.abs(np.diff(selected, axis=0)), axis=1)
        is_reconfig = jumps > reconfig_rad
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
            intra_reconfig_idx = np.where(is_reconfig & ~is_inter_cluster)[0]
            for idx in intra_reconfig_idx:
                jump_deg = np.rad2deg(jumps[idx])
                cid = cluster_id[idx]
                print(f"      viewpoint {idx}→{idx+1} (cluster {cid}): "
                      f"jump {jump_deg:.1f}°")

    # Phase 4: MotionGen transit at reconfig points
    reconfig_indices = np.where(is_reconfig)[0] if cluster_id is not None else np.array([], dtype=int)
    transit_segments = {}
    if len(reconfig_indices) > 0:
        print(f"\n[Phase 4] MotionGen transit for {len(reconfig_indices)} reconfig points...")
        transit_segments, _ = plan_reconfig_transits(
            selected, reconfig_indices, args.robot, world_config, tensor_args,
        )
        # 안전망: MotionGen이 중간에 wrist_3를 흔들었을 수 있으므로 강제 고정
        for idx in transit_segments:
            transit_segments[idx][:, -1] = wrist3_fixed

    # Phase 5: Uniform resample + collision check
    print(f"\n[Phase 5] Interpolation + uniform resample...")
    final_traj = interpolate_and_resample(selected, transit_segments, spacing_rad=args.spacing)
    print(f"  Resampled: {len(final_traj)} waypoints (uniform spacing={args.spacing} rad)")

    # Collision check
    print("  Collision check...")
    is_collision, n_collisions = batch_collision_check(
        final_traj, args.robot, world_config, tensor_args,
    )
    if n_collisions > 0:
        collision_pct = 100 * n_collisions / len(final_traj)
        print(f"  WARNING: {n_collisions}/{len(final_traj)} waypoints in collision ({collision_pct:.1f}%)")
        final_traj = final_traj[~is_collision]
        # 균일성 복원: drop으로 생긴 gap을 cumulative L2 arc-length 재resample로 메움
        if len(final_traj) >= 2:
            diffs = np.linalg.norm(np.diff(final_traj, axis=0), axis=1)
            cum_len = np.concatenate([[0], np.cumsum(diffs)])
            total_len = cum_len[-1]
            if total_len > 1e-9:
                n_out = max(2, int(np.ceil(total_len / args.spacing)) + 1)
                uniform_s = np.linspace(0, total_len, n_out)
                new_traj = np.zeros((n_out, final_traj.shape[1]))
                for j in range(final_traj.shape[1]):
                    new_traj[:, j] = np.interp(uniform_s, cum_len, final_traj[:, j])
                final_traj = new_traj
        print(f"  Removed collisions → re-resampled to {len(final_traj)} waypoints")
    else:
        print(f"  No collisions detected ({len(final_traj)} waypoints)")

    # FK + 저장
    ee_positions, ee_quaternions = compute_fk(final_traj, args.robot)
    print(f"  Computed FK for {len(final_traj)} waypoints")

    traj_dir = config.get_trajectory_path(args.object, args.num_viewpoints, "dummy").parent
    traj_dir.mkdir(parents=True, exist_ok=True)

    suffix = args.output_suffix
    spacing_str = f"{args.spacing:.2f}".replace(".", "")  # 0.10 → "010"
    tag = f"{suffix}_s{spacing_str}"

    csv_path = str(traj_dir / f"trajectory_{tag}.csv")
    save_trajectory_csv(final_traj, ee_positions, ee_quaternions, csv_path)

    static_path = str(traj_dir / f"trajectory_{tag}.html")
    visualize_static_html(args.object, final_traj, ee_positions, static_path)

    anim_path = str(traj_dir / f"trajectory_{tag}_anim.html")
    visualize_animated_html(args.object, final_traj, ee_positions, anim_path)

    n_transit_ok = len(transit_segments)
    print(f"\nDone. reconfigs={stats['n_reconfigs']} "
          f"(inter={rc_inter}, intra={rc_intra}), "
          f"transit={n_transit_ok}/{len(reconfig_indices)} OK, "
          f"collisions={n_collisions}, "
          f"final={len(final_traj)} waypoints")


if __name__ == "__main__":
    main()
