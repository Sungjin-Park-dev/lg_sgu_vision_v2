#!/usr/bin/env python3
"""
로봇 모션 계획 및 궤적 생성

viewpoints.h5에 저장된 클러스터링된 뷰포인트를 기반으로 로봇 궤적을 생성한다.
클러스터 간: MotionGen 충돌 회피 이동, 클러스터 내: dense interpolation + seed-propagation IK.

출력:
    data/{object}/trajectory/{num}/trajectory.csv  — joint angles + EE pose
    data/{object}/trajectory/{num}/trajectory.html — 3D 시각화

사전 조건: generate_viewpoints.py 로 생성된 viewpoints.h5 필요

사용법:
    uv run scripts/pipeline/plan_motion.py --object sample --num-viewpoints 124

    # 보간 간격 변경 (기본: 2mm)
    uv run scripts/pipeline/plan_motion.py --object sample --num-viewpoints 124 --interp-spacing 5.0

ROS2 전송은 별도 스크립트:
    uv run scripts/pipeline/publish_trajectory.py --object sample --num-viewpoints 124
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

# cuRobo imports
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState as CuRoboJointState
from curobo.geom.types import Cuboid, Mesh as CuRoboMesh, WorldConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common import config
from common.math_utils import quaternion_to_rotation_matrix, normalize_vectors


# ==========================================================================
# Math utilities
# ==========================================================================

def rot_to_quat_batch(R_batch: np.ndarray) -> np.ndarray:
    batch_size = R_batch.shape[0]
    quats = np.zeros((batch_size, 4), dtype=np.float64)
    for i in range(batch_size):
        r = Rotation.from_matrix(R_batch[i])
        q_xyzw = r.as_quat()  # (x, y, z, w)
        quats[i] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]  # (w, x, y, z)
    return quats


# ==========================================================================
# Core logic
# ==========================================================================

def load_viewpoints(object_name: str, num_viewpoints: int):
    """Load positions, normals, path_order, row_index, cluster data from HDF5.

    Returns:
        (positions, normals, path_order, row_index, wd_m, cluster_id, cluster_order)
        cluster_id and cluster_order are None if not present (backward compat).
    """
    h5_path = config.get_viewpoint_path(object_name, num_viewpoints)
    if not h5_path.exists():
        raise FileNotFoundError(f"Viewpoints file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        grp = f["viewpoints"]
        positions = np.array(grp["positions"], dtype=np.float64)
        normals = np.array(grp["normals"], dtype=np.float64)
        path_order = np.array(grp["path_order"], dtype=np.int32) if "path_order" in grp else None
        row_index = np.array(grp["row_index"], dtype=np.int32) if "row_index" in grp else None
        cluster_id = np.array(grp["cluster_id"], dtype=np.int32) if "cluster_id" in grp else None
        cluster_order = np.array(grp["cluster_order"], dtype=np.int32) if "cluster_order" in grp else None

        # camera working distance
        wd_m = config.CAMERA_WORKING_DISTANCE_MM / 1000.0
        if "metadata" in f and "camera_spec" in f["metadata"]:
            cs = f["metadata"]["camera_spec"]
            if "working_distance_mm" in cs.attrs:
                wd_m = float(cs.attrs["working_distance_mm"]) / 1000.0

    return positions, normals, path_order, row_index, wd_m, cluster_id, cluster_order


def build_camera_poses(positions, normals, working_distance_m):
    """Compute camera 4x4 poses in local (object) frame.

    Returns:
        world_poses: (N, 4, 4) array of poses in world frame
        camera_positions_world: (N, 3) camera positions in world frame
    """
    safe_normals = normalize_vectors(normals)
    camera_positions = positions + safe_normals * working_distance_m
    approach = -safe_normals  # camera looks inward

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

    # Transform to world frame
    target_world = np.eye(4, dtype=np.float64)
    target_world[:3, :3] = quaternion_to_rotation_matrix(config.TARGET_OBJECT["rotation"])
    target_world[:3, 3] = config.TARGET_OBJECT["position"]

    world_poses = np.einsum("ij,njk->nik", target_world, local_poses)
    camera_positions_world = world_poses[:, :3, 3]

    return world_poses, camera_positions_world


def generate_dense_scan_path(positions, normals, path_order, interp_spacing_m):
    """path_order 순서로 연속 뷰포인트 쌍을 밀집 보간.

    Args:
        positions: (N, 3) surface positions
        normals: (N, 3) surface normals
        path_order: (N,) int ordering
        interp_spacing_m: interpolation spacing in meters

    Returns:
        dense_positions: (M, 3)
        dense_normals: (M, 3)
    """
    sorted_idx = np.argsort(path_order)
    ordered_pos = positions[sorted_idx]
    ordered_nrm = normals[sorted_idx]

    dense_positions = [ordered_pos[0]]
    dense_normals = [ordered_nrm[0]]

    for i in range(len(sorted_idx) - 1):
        p0, p1 = ordered_pos[i], ordered_pos[i + 1]
        n0, n1 = ordered_nrm[i], ordered_nrm[i + 1]

        dist = np.linalg.norm(p1 - p0)
        n_steps = max(1, int(np.ceil(dist / interp_spacing_m)))

        for j in range(1, n_steps + 1):
            alpha = j / n_steps
            p = p0 + alpha * (p1 - p0)
            n = n0 + alpha * (n1 - n0)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-6:
                n = n0 if alpha < 0.5 else n1
                n_norm = np.linalg.norm(n)
            n = n / max(n_norm, 1e-9)
            dense_positions.append(p)
            dense_normals.append(n)

    return np.array(dense_positions), np.array(dense_normals)


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

    # Target object mesh
    meshes = []
    mesh_path = config.get_mesh_path(object_name, mesh_type="source")
    if mesh_path.exists():
        loaded = trimesh.load(str(mesh_path))
        if isinstance(loaded, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
        else:
            mesh = loaded
        pos = config.TARGET_OBJECT["position"]
        rot = config.TARGET_OBJECT["rotation"]  # (w, x, y, z)
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


def _poses_from_world(world_poses, tensor_args):
    """Convert (N, 4, 4) world poses to cuRobo Pose (position + quaternion tensors)."""
    N = world_poses.shape[0]
    positions = world_poses[:, :3, 3]
    quats = rot_to_quat_batch(world_poses[:, :3, :3])
    pos_t = torch.tensor(positions, device=tensor_args.device, dtype=tensor_args.dtype)
    quat_t = torch.tensor(quats, device=tensor_args.device, dtype=tensor_args.dtype)
    return Pose(position=pos_t, quaternion=quat_t)


def plan_clustered_trajectory(
    robot_cfg_file,
    object_name,
    positions, normals, path_order,
    cluster_id, cluster_order,
    working_distance_m,
    interp_spacing_m=0.002,
):
    """Hybrid trajectory: MotionGen transit between clusters + dense scan IK within.

    Args:
        robot_cfg_file: cuRobo robot config filename
        object_name: object name for mesh collision loading
        positions: (N, 3) surface positions
        normals: (N, 3) surface normals
        path_order: (N,) global path ordering
        cluster_id: (N,) cluster assignment per viewpoint
        cluster_order: (K,) cluster visit order
        working_distance_m: camera working distance in meters
        interp_spacing_m: dense interpolation spacing in meters

    Returns:
        trajectory: (M, n_dof) concatenated joint trajectory
        cluster_stats: list of dicts with per-cluster info
    """
    import time

    tensor_args = TensorDeviceType()
    n_dof = len(config.ROBOT_START_STATE)

    # --- Setup MotionGen ---
    world_config = build_collision_world(object_name)
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg_file,
        world_config,
        tensor_args=tensor_args,
        interpolation_dt=0.02,
        use_cuda_graph=False,
    )
    motion_gen = MotionGen(motion_gen_config)
    print("  Warming up MotionGen...")
    motion_gen.warmup()

    plan_cfg = MotionGenPlanConfig(
        max_attempts=10,
        timeout=5.0,
        enable_graph=False,
        enable_opt=True,
        enable_finetune_trajopt=True,
    )

    # --- Setup IK solver (with collision world) ---
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_cfg_file))["robot_cfg"]
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_config,
        self_collision_check=True,
        tensor_args=tensor_args,
        use_cuda_graph=False,
    )
    ik_solver = IKSolver(ik_config)

    # --- Current state ---
    current_q = np.array(config.ROBOT_START_STATE, dtype=np.float64)
    all_traj_segments = []
    cluster_stats = []
    total_transit_ok = 0
    total_vp = 0
    total_vp_ok = 0

    K = len(cluster_order)

    for rank, cid in enumerate(cluster_order):
        mask = cluster_id == cid
        indices = np.where(mask)[0]
        indices_sorted = sorted(indices, key=lambda i: path_order[i])
        n_pts = len(indices_sorted)
        total_vp += n_pts

        cluster_pos = positions[indices_sorted]
        cluster_nrm = normals[indices_sorted]

        print(f"  Cluster {rank}/{K-1} (id={cid}, {n_pts} pts):")

        # B. MotionGen transit — 클러스터 내 순서대로 진입점을 시도
        transit_traj = None
        q_arrival = None
        entry_idx = 0  # 성공한 진입점 인덱스

        max_entry_tries = min(n_pts, 5)  # 최대 5개 진입점 시도
        for try_idx in range(max_entry_tries):
            target_world_poses, _ = build_camera_poses(
                cluster_pos[try_idx:try_idx+1], cluster_nrm[try_idx:try_idx+1],
                working_distance_m,
            )
            goal_pose = _poses_from_world(target_world_poses, tensor_args)

            start_q_t = torch.tensor(
                current_q, device=tensor_args.device, dtype=tensor_args.dtype,
            ).view(1, -1)
            current_state = CuRoboJointState.from_position(start_q_t)

            t0 = time.time()
            mg_result = motion_gen.plan_single(current_state, goal_pose, plan_cfg.clone())
            mg_dt = time.time() - t0

            if mg_result.success.item():
                traj = mg_result.get_interpolated_plan()
                transit_traj = traj.position.cpu().numpy()
                q_arrival = transit_traj[-1].copy()
                total_transit_ok += 1
                entry_idx = try_idx
                if try_idx == 0:
                    print(f"    MotionGen transit: OK ({len(transit_traj)} waypoints, {mg_dt:.2f}s)")
                else:
                    print(f"    MotionGen transit: OK at point {try_idx} ({len(transit_traj)} waypoints, {mg_dt:.2f}s)")
                break
            else:
                if try_idx == 0:
                    print(f"    MotionGen transit: FAILED (status: {mg_result.status}, {mg_dt:.2f}s)")
                else:
                    print(f"    MotionGen retry point {try_idx}: FAILED ({mg_result.status})")

        if transit_traj is None:
            print(f"    All {max_entry_tries} entry points failed — skipping cluster {cid}")
            cluster_stats.append({
                "cid": cid, "n_pts": n_pts,
                "transit_ok": False, "scan_ok": 0, "scan_total": n_pts,
            })
            continue

        if all_traj_segments:
            all_traj_segments.append(transit_traj[1:])
        else:
            all_traj_segments.append(transit_traj)

        current_q = q_arrival.copy()

        # C. Cluster-internal scan IK (entry_idx 이후부터 스캔)
        scan_pos = cluster_pos[entry_idx:]
        scan_nrm = cluster_nrm[entry_idx:]
        n_scan = len(scan_pos)

        if n_scan >= 2:
            scan_path_order = np.arange(n_scan, dtype=np.int32)
            dense_pos, dense_nrm = generate_dense_scan_path(
                scan_pos, scan_nrm, scan_path_order, interp_spacing_m,
            )
        else:
            dense_pos = scan_pos
            dense_nrm = scan_nrm

        dense_world_poses, _ = build_camera_poses(dense_pos, dense_nrm, working_distance_m)
        M = len(dense_world_poses)

        dense_positions_np = dense_world_poses[:, :3, 3]
        dense_quats_np = rot_to_quat_batch(dense_world_poses[:, :3, :3])

        num_seeds = 4
        batch_size = 32
        scan_solutions = np.zeros((M, n_dof), dtype=np.float64)
        scan_success = np.zeros(M, dtype=bool)
        prev_sol = q_arrival.copy()

        n_batches = (M + batch_size - 1) // batch_size
        for b in range(n_batches):
            s = b * batch_size
            e = min(s + batch_size, M)
            bn = e - s

            bp = torch.tensor(dense_positions_np[s:e], device=tensor_args.device, dtype=tensor_args.dtype)
            bq = torch.tensor(dense_quats_np[s:e], device=tensor_args.device, dtype=tensor_args.dtype)
            goal = Pose(position=bp, quaternion=bq)

            seed = torch.tensor(
                prev_sol, device=tensor_args.device, dtype=tensor_args.dtype,
            ).unsqueeze(0).unsqueeze(0).expand(bn, num_seeds, -1).contiguous()
            retract = torch.tensor(
                prev_sol, device=tensor_args.device, dtype=tensor_args.dtype,
            ).unsqueeze(0).expand(bn, -1).contiguous()

            result = ik_solver.solve_batch(
                goal, seed_config=seed, retract_config=retract, return_seeds=num_seeds,
            )

            succ_raw = result.success.cpu().numpy()
            succ_batch = succ_raw.any(axis=1) if succ_raw.ndim > 1 else succ_raw.astype(bool)
            sols_batch = result.solution.cpu().numpy()
            if sols_batch.ndim == 2:
                sols_batch = sols_batch[:, np.newaxis, :]

            MAX_JOINT_STEP_RAD = np.deg2rad(15)  # waypoint 간 최대 joint 변화 허용량

            for i in range(bn):
                idx = s + i
                if succ_batch[i]:
                    candidates = sols_batch[i]
                    diffs = np.max(np.abs(candidates - prev_sol), axis=1)
                    best = np.argmin(diffs)
                    if diffs[best] <= MAX_JOINT_STEP_RAD:
                        scan_success[idx] = True
                        scan_solutions[idx] = candidates[best]
                        prev_sol = candidates[best]
                    # else: IK 성공했지만 jump 초과 → 해당 waypoint skip

        scan_ok = int(scan_success.sum())
        total_vp_ok += min(n_scan, scan_ok)
        skipped_str = f", skipped first {entry_idx}" if entry_idx > 0 else ""
        print(f"    Scan IK: {scan_ok}/{M} dense waypoints OK "
              f"(from {n_scan}/{n_pts} viewpoints → {M} dense{skipped_str})")

        if scan_ok > 0:
            ok_indices = np.where(scan_success)[0]
            scan_traj = scan_solutions[ok_indices]
            if len(scan_traj) > 0:
                all_traj_segments.append(scan_traj[1:] if len(all_traj_segments) > 0 else scan_traj)
                current_q = scan_traj[-1].copy()

        cluster_stats.append({
            "cid": cid, "n_pts": n_pts,
            "transit_ok": True, "scan_ok": scan_ok, "scan_total": M,
        })

    if not all_traj_segments:
        raise RuntimeError("No trajectory segments planned successfully")

    trajectory = np.concatenate(all_traj_segments, axis=0)

    wrist3_fixed = config.ROBOT_START_STATE[-1]
    trajectory[:, -1] = wrist3_fixed
    print(f"  Fixed wrist_3 at {np.rad2deg(wrist3_fixed):.1f} deg")

    print(f"\n  Total: {total_vp} viewpoints, {total_transit_ok}/{K} transits OK")
    print(f"  Trajectory: {len(trajectory)} points")

    return trajectory, cluster_stats


def compute_fk(solutions, success, robot_cfg_file):
    """Compute FK for successful IK solutions.

    Returns:
        ee_positions: (N, 3) array — EE positions (NaN for failures)
        ee_quaternions: (N, 4) array — EE quaternions as (x, y, z, w) (NaN for failures)
    """
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_cfg_file))["robot_cfg"]
    world_config = WorldConfig()
    ta = TensorDeviceType(device=torch.device("cuda:0"), dtype=torch.float32)

    rw_config = RobotWorldConfig.load_from_config(robot_cfg, world_config, tensor_args=ta)
    robot_world = RobotWorld(rw_config)

    N = solutions.shape[0]
    ee_positions = np.full((N, 3), np.nan, dtype=np.float64)
    ee_quaternions = np.full((N, 4), np.nan, dtype=np.float64)

    mask = np.where(success)[0]
    if len(mask) == 0:
        return ee_positions, ee_quaternions

    q_batch = torch.tensor(solutions[mask], device="cuda:0", dtype=torch.float32)
    state = robot_world.get_kinematics(q_batch)
    ee_pos = state.ee_position.cpu().numpy()
    ee_quat_wxyz = state.ee_quaternion.cpu().numpy()  # cuRobo: (w, x, y, z)

    ee_positions[mask] = ee_pos
    # CSV 형식에 맞춰 (x, y, z, w)로 변환
    ee_quaternions[mask] = ee_quat_wxyz[:, [1, 2, 3, 0]]

    return ee_positions, ee_quaternions


def visualize_html(
    object_name,
    ee_positions,
    success,
    output_html: str,
):
    """Plotly HTML 3D visualization of EE trajectory."""
    import trimesh
    import plotly.graph_objects as go

    traces = []

    # --- Mesh (transparent) ---
    mesh_path = config.get_mesh_path(object_name, mesh_type="source")
    if mesh_path.exists():
        loaded = trimesh.load(str(mesh_path))
        if isinstance(loaded, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
        else:
            mesh = loaded

        # Transform to world frame
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = quaternion_to_rotation_matrix(config.TARGET_OBJECT["rotation"])
        T[:3, 3] = config.TARGET_OBJECT["position"]
        verts = np.array(mesh.vertices)
        verts_h = np.c_[verts, np.ones(len(verts))]
        verts_w = (T @ verts_h.T).T[:, :3]
        faces = mesh.faces

        traces.append(go.Mesh3d(
            x=verts_w[:, 0], y=verts_w[:, 1], z=verts_w[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color='lightgray', opacity=0.25,
            name='Mesh', hoverinfo='skip',
        ))

    # --- EE positions ---
    ee_ok = ee_positions[success]
    ee_fail = ee_positions[~success]

    traces.append(go.Scatter3d(
        x=ee_ok[:, 0], y=ee_ok[:, 1], z=ee_ok[:, 2],
        mode='markers',
        marker=dict(size=3, color='blue'),
        name=f'EE (OK: {success.sum()})',
        hoverinfo='text',
        text=[f'wp={i}' for i in np.where(success)[0]],
    ))

    if (~success).any():
        valid_fail = ee_fail[~np.isnan(ee_fail[:, 0])]
        if len(valid_fail) > 0:
            traces.append(go.Scatter3d(
                x=valid_fail[:, 0], y=valid_fail[:, 1], z=valid_fail[:, 2],
                mode='markers',
                marker=dict(size=3, color='red'),
                name=f'EE (fail: {(~success).sum()})',
            ))

    # --- EE path ---
    if len(ee_ok) > 1:
        traces.append(go.Scatter3d(
            x=ee_ok[:, 0], y=ee_ok[:, 1], z=ee_ok[:, 2],
            mode='lines',
            line=dict(color='blue', width=2),
            name='EE path', hoverinfo='skip',
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f'EE Trajectory — {object_name} ({success.sum()} waypoints)',
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
            aspectmode='data',
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=80, b=0),
    )
    fig.write_html(output_html)
    print(f"  HTML visualization saved to {output_html}")


def save_trajectory_csv(
    solutions: np.ndarray,
    ee_positions: np.ndarray,
    ee_quaternions: np.ndarray,
    output_path: str,
    dt: float = 1.0,
):
    """Trajectory를 CSV 파일로 저장.

    Args:
        solutions: (N, n_dof) joint trajectory
        ee_positions: (N, 3) EE positions
        ee_quaternions: (N, 4) EE quaternions (x, y, z, w)
        output_path: 출력 CSV 경로
        dt: waypoint 간 시간 간격 (초)
    """
    import csv

    JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    header = ["time"] + JOINT_NAMES + [
        "target-POS_X", "target-POS_Y", "target-POS_Z",
        "target-ROT_X", "target-ROT_Y", "target-ROT_Z", "target-ROT_W",
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(solutions)):
            t = i * dt
            row = [t] + solutions[i].tolist()
            row += ee_positions[i].tolist()
            row += ee_quaternions[i].tolist()
            writer.writerow(row)

    print(f"  CSV saved to {output_path} ({len(solutions)} waypoints)")



def main():
    parser = argparse.ArgumentParser(description="로봇 모션 계획 및 궤적 생성")
    parser.add_argument("--object", type=str, required=True, help="Object name")
    parser.add_argument("--num-viewpoints", type=int, required=True, help="Number of viewpoints")
    parser.add_argument("--robot", type=str, default=config.DEFAULT_ROBOT_CONFIG,
                        help=f"Robot config file (default: {config.DEFAULT_ROBOT_CONFIG})")
    parser.add_argument(
        "--interp-spacing", type=float, default=2.0,
        help="Cartesian interpolation spacing in mm (default: 2.0)",
    )
    args = parser.parse_args()

    # [1] Load viewpoints
    print("[1/3] Loading viewpoints...")
    positions, normals, path_order, row_index, wd_m, cluster_id, cluster_order = \
        load_viewpoints(args.object, args.num_viewpoints)

    if cluster_id is None or cluster_order is None:
        print("Error: viewpoints.h5에 클러스터 데이터가 없습니다.")
        print("  generate_viewpoints.py --cluster 로 먼저 생성하세요.")
        return

    n_clusters = len(cluster_order)
    print(f"  Loaded {len(positions)} viewpoints ({n_clusters} clusters, "
          f"working distance: {wd_m*1000:.1f} mm)")

    # [2] Plan trajectory
    print("[2/3] Planning trajectory...")
    interp_spacing_m = args.interp_spacing / 1000.0
    trajectory, cluster_stats = plan_clustered_trajectory(
        args.robot, args.object,
        positions, normals, path_order,
        cluster_id, cluster_order,
        wd_m,
        interp_spacing_m=interp_spacing_m,
    )

    solutions = trajectory
    success = np.ones(len(trajectory), dtype=bool)
    path_order = np.arange(len(trajectory), dtype=np.int32)

    # Fix wrist_3
    wrist3_fixed = config.ROBOT_START_STATE[-1]
    solutions[:, -1] = wrist3_fixed
    print(f"  Fixed wrist_3 at {np.rad2deg(wrist3_fixed):.1f} deg")

    # Report max joint jump
    ordered_idx = list(range(len(solutions)))
    if len(ordered_idx) > 1:
        max_jumps = np.max(np.abs(np.diff(solutions, axis=0)), axis=1)
        print(f"  Max joint jump: {np.rad2deg(np.max(max_jumps)):.1f} deg, "
              f"Mean: {np.rad2deg(np.mean(max_jumps)):.1f} deg")

    # [3] Save trajectory
    trajectory_dir = config.get_trajectory_path(args.object, args.num_viewpoints, "dummy").parent
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    print("[3/3] Computing FK...")
    ee_positions, ee_quaternions = compute_fk(solutions, success, args.robot)
    print(f"  Computed EE positions for {success.sum()} viewpoints")

    html_path = str(trajectory_dir / "trajectory.html")
    visualize_html(args.object, ee_positions, success, html_path)

    csv_path = str(trajectory_dir / "trajectory.csv")
    save_trajectory_csv(solutions, ee_positions, ee_quaternions, csv_path)


if __name__ == "__main__":
    main()
