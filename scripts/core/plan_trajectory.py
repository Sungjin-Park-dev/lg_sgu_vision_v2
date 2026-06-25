#!/usr/bin/env python3
"""
DBSCAN + DP + MotionPlanner 기반 최적 IK 궤적 생성 (cuRobo v0.8 API)

각 viewpoint에 대해 다수의 IK 해를 구하고, DP로 전역 최적 경로를 선택한 뒤,
reconfig 지점은 MotionPlanner로 충돌회피 transit을 만들어 균일 spacing으로 resample한다.

단계:
    Phase 1: Multi-seed IK         — viewpoint당 num_seeds개 IK 해
    Phase 2: DBSCAN                — viewpoint당 대표 해 (medoid) 추출
    Phase 3: DP                    — 최소 joint-space 비용 경로 선택
       ↓ wrist_3 잠금 (resample 균일성을 위해 metric에서 사실상 제외)
    Phase 4: MotionPlanner transit — reconfig 지점 충돌회피 joint-to-joint planning
    Phase 5: Uniform resample      — cumulative EE arc-length(m) spacing + 충돌 검사
    Phase 6: Time planning         — EE 선속도/각속도/joint 속도 제한 기반 continuous scan

사용법:
    uv run scripts/core/plan_trajectory.py --object sample --num-viewpoints 124 --viewpoints data/sample/viewpoint/124/viewpoints_coacd+dbscan.h5
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

from curobo.types import Pose, JointState, GoalToolPose
from curobo.scene import Scene, Cuboid, Mesh as CuRoboMesh
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import config
from common.math_utils import quaternion_to_rotation_matrix, normalize_vectors


# =========================================================================
# Pipeline defaults
# =========================================================================

ROBOT_CONFIG = config.DEFAULT_ROBOT_CONFIG
NUM_IK_SEEDS = 100
IK_BATCH_SIZE = 4
DBSCAN_EPS_RAD = 0.3
RECONFIG_THRESHOLD_DEG = 29.0

# 충돌검사에서 제외할 로봇 링크. base_link_inertia(로봇 베이스)는 base_link 에 고정이라
# 자세와 무관하게 항상 robot_mount(받침대) 윗면을 ~2cm 파고든다 → 모든 IK/충돌검사가
# 상시 충돌로 실패. 받침대 박스가 팔은 그대로 막아주고, base 는 자기 받침대만 닿으므로
# 충돌검사 자체가 무의미해 제외해도 실질 보호 손실이 없다.
COLLISION_EXCLUDE_LINKS = ("base_link_inertia",)

RESAMPLE_MODE = "ee"
DEFAULT_SPACING_M = 0.01

EE_SPEED_MM_S = 50.0
EE_ANGULAR_SPEED_DEG_S = 20.0
MAX_JOINT_VEL_RAD_S = 0.3
MIN_SEGMENT_DT_S = 0.05

CORNER_SLOWDOWN_ENABLED = True
CORNER_ANGLE_THRESHOLD_DEG = 30.0
CORNER_MAX_SLOWDOWN = 2.5


# =========================================================================
# Robot config resolution (absolute paths so cuRobo doesn't need symlinks)
# =========================================================================

def _resolve_robot_config(robot_filename: str):
    """Robot YAML 을 dict 로 로드하고 urdf_path/asset_root_path 를 절대경로로 패치.

    탐색 순서: 프로젝트 ur20_description/ → cuRobo content/configs/robot/.
    """
    import yaml
    from curobo.content import get_robot_configs_path

    candidates = [
        config.PROJECT_ROOT / "ur20_description" / robot_filename,
        Path(get_robot_configs_path()) / robot_filename,
    ]
    yaml_path = next((p for p in candidates if p.exists()), None)
    if yaml_path is None:
        raise FileNotFoundError(
            f"Robot config '{robot_filename}' not found in: "
            + ", ".join(str(p) for p in candidates)
        )

    import dataclasses
    from curobo._src.robot.loader.kinematics_loader_cfg import KinematicsLoaderCfg
    from curobo._src.robot.types.cspace_params import CSpaceParams

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    kin = cfg["robot_cfg"]["kinematics"]
    rel_urdf = kin.get("urdf_path", "")

    asset_search = [
        config.PROJECT_ROOT / "ur20_description",
        yaml_path.parent,
    ]
    asset_root = next(
        (p for p in asset_search if (p / Path(rel_urdf).name).exists()), None,
    )
    if asset_root is None:
        raise FileNotFoundError(
            f"Robot URDF '{Path(rel_urdf).name}' not found in: "
            + ", ".join(str(p) for p in asset_search)
        )

    kin["urdf_path"] = str(asset_root / Path(rel_urdf).name)
    kin["asset_root_path"] = str(asset_root)

    # Translate legacy ee_link → tool_frames; drop fields the new KinematicsLoaderCfg
    # doesn't accept (usd_*, link_names, ...).
    if "tool_frames" not in kin and "ee_link" in kin:
        ee = kin["ee_link"]
        kin["tool_frames"] = [ee] if isinstance(ee, str) else list(ee)

    # Translate legacy cspace.retract_config → default_joint_position; filter cspace
    # to fields CSpaceParams accepts.
    if isinstance(kin.get("cspace"), dict):
        cs = dict(kin["cspace"])
        if "default_joint_position" not in cs and "retract_config" in cs:
            cs["default_joint_position"] = cs["retract_config"]
        cs_allowed = {f.name for f in dataclasses.fields(CSpaceParams)}
        kin["cspace"] = {k: v for k, v in cs.items() if k in cs_allowed}

    # 고정 베이스 링크를 충돌검사에서 제외 (COLLISION_EXCLUDE_LINKS 참고).
    # collision_link_names 와 mesh_link_names 는 YAML anchor 로 같은 리스트일 수 있어
    # 각각 새 리스트로 다시 필터한다.
    for key in ("collision_link_names", "mesh_link_names"):
        if isinstance(kin.get(key), list):
            kin[key] = [l for l in kin[key] if l not in COLLISION_EXCLUDE_LINKS]
    if isinstance(kin.get("collision_spheres"), dict):
        kin["collision_spheres"] = {
            k: v for k, v in kin["collision_spheres"].items()
            if k not in COLLISION_EXCLUDE_LINKS
        }

    # RobotCfg.create() injects these into KinematicsLoaderCfg() explicitly; leaving
    # them in the kinematics dict raises "multiple values for keyword argument".
    injected = {"load_collision_spheres", "num_envs", "device_cfg"}
    allowed = {f.name for f in dataclasses.fields(KinematicsLoaderCfg)} - injected
    cfg["robot_cfg"]["kinematics"] = {k: v for k, v in kin.items() if k in allowed}
    return cfg


def _collision_sphere_buffer_summary(robot_cfg) -> str | None:
    buffer = robot_cfg["robot_cfg"]["kinematics"].get("collision_sphere_buffer", 0.0)
    if isinstance(buffer, dict):
        values = [float(v) for v in buffer.values()]
        values = [v for v in values if v > 0.0]
        if not values:
            return None
        if min(values) == max(values):
            return f"{values[0] * 1000:.1f} mm"
        return f"{min(values) * 1000:.1f}-{max(values) * 1000:.1f} mm"

    value = float(buffer or 0.0)
    if value <= 0.0:
        return None
    return f"{value * 1000:.1f} mm"


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
                h5_wd_mm = float(cs.attrs["working_distance_mm"])
                cfg_wd_mm = float(config.CAMERA_WORKING_DISTANCE_MM)
                if abs(h5_wd_mm - cfg_wd_mm) > 1e-6:
                    print(
                        f"  WARNING: viewpoints h5 working_distance_mm={h5_wd_mm:.1f}, "
                        f"current config={cfg_wd_mm:.1f}. Using h5 metadata."
                    )
                wd_m = h5_wd_mm / 1000.0

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

    return Scene(cuboid=cuboids, mesh=meshes if meshes else None)


def compute_fk(solutions, robot_cfg):
    """Compute FK for joint solutions. Returns (N,3) positions and (N,4) quats (x,y,z,w)."""
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(robot_cfg))
    q_batch = torch.tensor(solutions, device="cuda:0", dtype=torch.float32)
    js = JointState.from_position(q_batch, joint_names=kin.joint_names)
    state = kin.compute_kinematics(js)

    ee_pose = state.tool_poses.get_link_pose(kin.tool_frames[0])
    ee_positions = ee_pose.position.cpu().numpy()
    ee_quat_wxyz = ee_pose.quaternion.cpu().numpy()
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

def solve_ik_multi_seed(robot_cfg, world_scene, positions_np, quats_np,
                        num_seeds=100, batch_size=4):
    """각 pose에 대해 num_seeds개 IK 해를 구한다.

    Args:
        robot_cfg: cuRobo robot config (dict)
        world_scene: 충돌 Scene
        positions_np: (N, 3) EE positions
        quats_np: (N, 4) EE quaternions (w, x, y, z)
        num_seeds: IK seed 수
        batch_size: GPU 배치 크기

    Returns:
        all_solutions: (N, num_seeds, 6)
        all_success: (N, num_seeds) bool
    """
    cache = {
        "obb": max(1, len(world_scene.cuboid)),
        "mesh": max(1, len(world_scene.mesh)),
    }
    cfg = InverseKinematicsCfg.create(
        robot=robot_cfg,
        scene_model={},
        self_collision_check=True,
        num_seeds=num_seeds,
        max_batch_size=batch_size,
        use_cuda_graph=False,
        collision_cache=cache,
    )
    ik = InverseKinematics(cfg)
    ik.update_world(world_scene)
    tool = ik.tool_frames[0]

    N = len(positions_np)
    n_dof = 6
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

        result = ik.solve_pose(
            GoalToolPose.from_poses({tool: goal}, num_goalset=1),
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

def _single_joint_state_path(joint_state: JointState, n_dof: int) -> np.ndarray:
    """Extract the one planned path from cuRobo's batch/seed shaped JointState."""
    q = joint_state.position
    original_shape = tuple(q.shape)
    while q.ndim > 2 and q.shape[0] == 1:
        q = q.squeeze(0)
    if q.ndim != 2:
        raise RuntimeError(
            f"Expected a single trajectory shaped (T, dof), got "
            f"{original_shape} -> {tuple(q.shape)}"
        )
    waypoints = q.detach().cpu().numpy()
    if waypoints.shape[-1] != n_dof:
        waypoints = waypoints[..., :n_dof]
    return waypoints


def plan_reconfig_transits(
    selected, reconfig_indices, robot_cfg, world_scene,
):
    """Reconfig 지점마다 MotionPlanner joint-to-joint planning 수행.

    Args:
        selected: (N, 6) DP로 선택된 joint trajectory
        reconfig_indices: reconfig이 발생하는 transition 인덱스 배열
        robot_cfg: cuRobo robot config (dict)
        world_scene: Scene (충돌 세계)

    Returns:
        transit_segments: dict {idx: (T, 6) transit trajectory} — 성공한 것만
        transit_stats: list of dicts
    """
    cache = {
        "obb": max(1, len(world_scene.cuboid)),
        "mesh": max(1, len(world_scene.mesh)),
    }
    cfg = MotionPlannerCfg.create(
        robot=robot_cfg,
        collision_cache=cache,
        use_cuda_graph=False,
    )
    planner = MotionPlanner(cfg)
    planner.update_world(world_scene)
    print("    Warming up MotionPlanner...")
    planner.warmup(enable_graph=False, num_warmup_iterations=2)

    transit_segments = {}
    transit_stats = []

    for idx in reconfig_indices:
        start_q = torch.tensor(
            selected[idx], device="cuda:0", dtype=torch.float32,
        ).unsqueeze(0)
        goal_q = torch.tensor(
            selected[idx + 1], device="cuda:0", dtype=torch.float32,
        ).unsqueeze(0)

        start_state = JointState.from_position(start_q, joint_names=planner.joint_names)
        goal_state = JointState.from_position(goal_q, joint_names=planner.joint_names)

        t0 = time.time()
        result = planner.plan_cspace(goal_state, start_state, max_attempts=10)
        dt = time.time() - t0

        ok = result is not None and bool(result.success.any().item())
        if ok:
            traj = result.get_interpolated_plan()
            waypoints = _single_joint_state_path(traj, selected.shape[-1])
            if len(waypoints) < 2:
                transit_stats.append({
                    "idx": idx, "success": False,
                    "n_waypoints": len(waypoints), "time": dt,
                })
                print(
                    f"    {idx}→{idx+1}: FAILED "
                    f"(planner returned {len(waypoints)} waypoint, {dt:.2f}s)"
                )
                continue
            transit_segments[idx] = waypoints
            max_step_deg = np.rad2deg(
                np.max(np.abs(np.diff(waypoints, axis=0)))
            ) if len(waypoints) > 1 else 0.0
            transit_stats.append({
                "idx": idx, "success": True,
                "n_waypoints": len(waypoints), "time": dt,
                "max_step_deg": float(max_step_deg),
            })
            print(
                f"    {idx}→{idx+1}: OK ({len(waypoints)} waypoints, "
                f"max_step={max_step_deg:.2f}°, {dt:.2f}s)"
            )
        else:
            transit_stats.append({
                "idx": idx, "success": False, "time": dt,
            })
            print(f"    {idx}→{idx+1}: FAILED ({dt:.2f}s)")

    n_ok = sum(1 for s in transit_stats if s["success"])
    print(f"  Transit planning: {n_ok}/{len(reconfig_indices)} succeeded")

    return transit_segments, transit_stats


# =========================================================================
# Phase 5: Uniform resample + collision check
# =========================================================================

def _resample_uniform_ee(joints, robot_cfg, spacing_m):
    """EE position arc-length 기준 uniform resample.

    각 인접 waypoint 간 ||Δee_position|| ≈ spacing_m이 되도록 다시 분할한다.
    상수 dt 재생 시 EE 선속도가 dt당 spacing_m/dt로 일정해진다.
    """
    if len(joints) < 2:
        return joints
    ee_positions, _ = compute_fk(joints, robot_cfg)  # (M, 3)
    diffs = np.linalg.norm(np.diff(ee_positions, axis=0), axis=1)
    cum_len = np.concatenate([[0], np.cumsum(diffs)])
    total_len = cum_len[-1]
    if total_len < 1e-9:
        return joints
    n_out = max(2, int(np.ceil(total_len / spacing_m)) + 1)
    uniform_s = np.linspace(0, total_len, n_out)
    out = np.zeros((n_out, joints.shape[1]), dtype=np.float64)
    for j in range(joints.shape[1]):
        out[:, j] = np.interp(uniform_s, cum_len, joints[:, j])
    return out


def _resample_uniform_joint(joints, spacing_rad):
    """Joint-space cumulative L∞ (max-joint) 기준 uniform resample.

    각 인접 waypoint 간 max|Δq_j| ≈ spacing_rad이 되도록 다시 분할한다.
    상수 dt 재생 시 가장 빨리 움직이는 joint의 각속도가 dt당 spacing_rad/dt로 일정해진다.
    """
    if len(joints) < 2:
        return joints
    diffs = np.max(np.abs(np.diff(joints, axis=0)), axis=1)
    cum_len = np.concatenate([[0], np.cumsum(diffs)])
    total_len = cum_len[-1]
    if total_len < 1e-9:
        return joints
    n_out = max(2, int(np.ceil(total_len / spacing_rad)) + 1)
    uniform_s = np.linspace(0, total_len, n_out)
    out = np.zeros((n_out, joints.shape[1]), dtype=np.float64)
    for j in range(joints.shape[1]):
        out[:, j] = np.interp(uniform_s, cum_len, joints[:, j])
    return out


def interpolate_and_resample(selected, transit_segments, robot_cfg,
                             mode="ee", spacing=0.01, dense_step_rad=0.02):
    """DP 궤적 + transit을 합치고, 선택된 metric으로 uniform resample.

    Non-reconfig 구간: joint-space linear interpolation으로 dense화
    Transit 구간: MotionPlanner의 dense 경로를 그대로 사용
    최종 resample 단위: mode에 따라 EE position arc-length(m) 또는 joint L∞(rad)

    Args:
        selected: (N, 6) DP 선택 궤적
        transit_segments: dict {idx: (T, 6)} transit 경로
        robot_cfg: cuRobo robot config dict (mode='ee'일 때 FK 용)
        mode: "ee" (EE position arc-length, meters) | "joint" (cumulative L∞, radians)
        spacing: 최종 spacing (mode에 따라 m 또는 rad)
        dense_step_rad: dense path 구성 시 joint-space L∞ step (radians).

    Returns:
        resampled: (M, 6) uniform-spaced trajectory
    """
    N = len(selected)

    # 1) 모든 구간을 dense path로 연결 (joint-space)
    dense_segments = []
    for i in range(N - 1):
        dense_segments.append(selected[i:i+1])

        if i in transit_segments:
            # transit 구간: 이미 dense, 첫/끝 제외
            transit = transit_segments[i]
            if len(transit) > 2:
                dense_segments.append(transit[1:-1])
        else:
            # non-reconfig: joint-space linear interpolation
            q0, q1 = selected[i], selected[i + 1]
            dist = np.max(np.abs(q1 - q0))
            n_steps = max(1, int(np.ceil(dist / dense_step_rad)))
            if n_steps > 1:
                alphas = np.linspace(0, 1, n_steps + 1)[1:-1]  # 양 끝 제외
                interp = q0[np.newaxis, :] + alphas[:, np.newaxis] * (q1 - q0)[np.newaxis, :]
                dense_segments.append(interp)

    dense_segments.append(selected[-1:])  # 마지막 점
    dense_path = np.concatenate(dense_segments, axis=0)

    # 2) Mode별 uniform resample
    if mode == "ee":
        return _resample_uniform_ee(dense_path, robot_cfg, spacing)
    elif mode == "joint":
        return _resample_uniform_joint(dense_path, spacing)
    else:
        raise ValueError(f"Unknown resample mode: {mode!r} (expected 'ee' or 'joint')")


def batch_collision_check(trajectory, robot_cfg, world_scene):
    """전체 궤적에 대해 batch collision check 수행. Returns (is_collision, n_collisions)."""
    cfg = RobotCollisionCheckerCfg.load_from_config(
        robot_config=robot_cfg,
        scene_model=world_scene,
        n_cuboids=max(1, len(world_scene.cuboid)),
        n_meshes=max(1, len(world_scene.mesh)),
    )
    checker = RobotCollisionChecker(cfg)

    # NOTE: cuRobo v0.8 RobotSceneCollision.get_scene_self_collision_distance_from_joints
    # is buggy (passes a tensor where the underlying cost expects a KinematicsState).
    # Bypass: drive kinematics + collision costs directly with shape (batch, horizon=1, dof).
    q_tensor = torch.tensor(trajectory, device="cuda:0", dtype=torch.float32).unsqueeze(1)
    batch, horizon = q_tensor.shape[0], 1
    state = checker.get_kinematics(q_tensor)
    num_spheres = state.robot_spheres.shape[-2]
    checker.collision_cost.update_num_spheres(num_spheres, batch_size=batch, horizon=horizon)
    checker.self_collision_cost.setup_batch_tensors(batch, horizon)
    d_scene = checker.collision_cost.forward(state)
    d_self = checker.self_collision_cost.forward(state.robot_spheres)

    # 음수 거리 = 충돌. Cost shape may be (batch, horizon, num_spheres) or (batch, horizon).
    # Reduce trailing dims to per-row collision flag.
    d_scene_r = d_scene.view(batch, -1)
    d_self_r = d_self.view(batch, -1)
    is_world_collision = (d_scene_r < 0).any(dim=-1).cpu().numpy()
    is_self_collision = (d_self_r < 0).any(dim=-1).cpu().numpy()
    is_collision = is_self_collision | is_world_collision
    n_collisions = int(is_collision.sum())

    return is_collision, n_collisions


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
                             corner_slowdown_enabled=True,
                             corner_angle_threshold_rad=np.deg2rad(30.0),
                             corner_max_slowdown=2.5):
    """Continuous scan용 누적 time 생성.

    각 segment 시간은 EE 선속도, EE 각속도, joint 속도 제한을 모두 만족하는
    최소 시간으로 정한다.
    """
    n = len(joints)
    times = np.zeros((n,), dtype=np.float64)
    if n < 2:
        return times, {
            "total_time": 0.0,
            "max_linear_speed_mm_s": 0.0,
            "max_angular_speed_deg_s": 0.0,
            "max_joint_speed_rad_s": 0.0,
        }

    if corner_slowdown_enabled:
        slowdown_factors, corner_stats = _corner_slowdown_factors(
            ee_positions, joints,
            threshold_rad=corner_angle_threshold_rad,
            max_slowdown=corner_max_slowdown,
        )
    else:
        slowdown_factors = np.ones((n - 1,), dtype=np.float64)
        corner_stats = {
            "n_slow_segments": 0,
            "max_corner_angle_deg": 0.0,
            "max_slowdown": 1.0,
        }

    for i in range(1, n):
        linear_dist = float(np.linalg.norm(ee_positions[i] - ee_positions[i - 1]))
        angular_dist = _quat_angle_xyzw(ee_quaternions[i - 1], ee_quaternions[i])
        joint_dist = float(np.max(np.abs(joints[i] - joints[i - 1])))

        dt_candidates = [min_segment_dt]
        if ee_speed_m_s > 0.0:
            dt_candidates.append(linear_dist / ee_speed_m_s)
        if ee_angular_speed_rad_s > 0.0:
            dt_candidates.append(angular_dist / ee_angular_speed_rad_s)
        if max_joint_vel_rad_s > 0.0:
            dt_candidates.append(joint_dist / max_joint_vel_rad_s)

        times[i] = times[i - 1] + max(dt_candidates) * slowdown_factors[i - 1]

    segment_dt = np.diff(times)
    linear_speed = np.linalg.norm(np.diff(ee_positions, axis=0), axis=1) / segment_dt
    angular_speed = np.array([
        _quat_angle_xyzw(ee_quaternions[i - 1], ee_quaternions[i]) / segment_dt[i - 1]
        for i in range(1, n)
    ])
    joint_speed = np.max(np.abs(np.diff(joints, axis=0)), axis=1) / segment_dt
    stats = {
        "total_time": float(times[-1]),
        "max_linear_speed_mm_s": float(linear_speed.max() * 1000.0),
        "max_angular_speed_deg_s": float(np.rad2deg(angular_speed.max())),
        "max_joint_speed_rad_s": float(joint_speed.max()),
        **corner_stats,
    }
    return times, stats


# =========================================================================
# CSV output
# =========================================================================

def save_trajectory_csv(solutions, ee_positions, ee_quaternions, output_path,
                        robot_name="ur20", dt=1.0, times=None):
    """Trajectory를 CSV로 저장. joint 컬럼에 robot_name prefix 추가."""
    import csv
    import os
    import tempfile

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

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_path = Path(f.name)
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(len(solutions)):
                t = times[i] if times is not None else i * dt
                row = [float(t)] + solutions[i].tolist()
                row += ee_positions[i].tolist()
                row += ee_quaternions[i].tolist()
                writer.writerow(row)

        tmp_path.chmod(0o644)
        os.replace(tmp_path, output_path)
    except Exception:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
        raise

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
    world_config = build_collision_world(args.object)
    robot_cfg = _resolve_robot_config(ROBOT_CONFIG)
    print(f"  Robot YAML: urdf={robot_cfg['robot_cfg']['kinematics']['urdf_path']}")
    collision_buffer = _collision_sphere_buffer_summary(robot_cfg)
    if collision_buffer:
        print(f"  Collision sphere buffer: {collision_buffer} (from robot YAML)")

    all_solutions, all_success = solve_ik_multi_seed(
        robot_cfg, world_config, positions_np, quats_np,
        num_seeds=NUM_IK_SEEDS, batch_size=IK_BATCH_SIZE,
    )

    # [4] Phase 2 + 3: DBSCAN → DP
    print("[4/6] Phase 2 — DBSCAN clustering...")
    representatives = cluster_ik_solutions(
        all_solutions, all_success, eps=DBSCAN_EPS_RAD,
    )

    print("[5/6] Phase 3 — DP optimal path...")
    reconfig_rad = np.deg2rad(RECONFIG_THRESHOLD_DEG)
    selected, _, stats = dp_optimal_path(representatives, reconfig_rad)

    # wrist_3 고정 — Phase 4/5 전체가 일관된 wrist_3로 동작하여
    # |Δwrist_3|=0이 되므로 6-DoF L∞ = 5-DoF L∞ (다른 joint가 최대값을 결정).
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

    # Phase 4: MotionPlanner transit at reconfig points
    reconfig_indices = np.where(is_reconfig)[0] if cluster_id is not None else np.array([], dtype=int)
    transit_segments = {}
    if len(reconfig_indices) > 0:
        print(f"\n[Phase 4] MotionPlanner transit for {len(reconfig_indices)} reconfig points...")
        transit_segments, _ = plan_reconfig_transits(
            selected, reconfig_indices, robot_cfg, world_config,
        )
        # 안전망: MotionPlanner가 중간에 wrist_3를 흔들었을 수 있으므로 강제 고정
        for idx in transit_segments:
            transit_segments[idx][:, -1] = wrist3_fixed

    # Phase 5: Uniform resample + collision check
    print(f"\n[Phase 5] Interpolation + uniform resample (mode={RESAMPLE_MODE})...")
    final_traj = interpolate_and_resample(
        selected, transit_segments, robot_cfg,
        mode=RESAMPLE_MODE, spacing=args.spacing,
    )
    if RESAMPLE_MODE == "ee":
        spacing_desc = f"EE spacing={args.spacing*1000:.1f} mm"
    else:
        spacing_desc = f"joint spacing={np.rad2deg(args.spacing):.2f}°"
    print(f"  Resampled: {len(final_traj)} waypoints ({spacing_desc})")

    # Collision check
    print("  Collision check...")
    collision_traj = densify_for_collision_check(final_traj)
    if len(collision_traj) != len(final_traj):
        print(
            f"  Collision check densified: {len(final_traj)} → {len(collision_traj)} "
            f"waypoints (max joint step="
            f"{config.COLLISION_ADAPTIVE_MAX_JOINT_STEP_DEG:.3f}°"
            + (", excluding wrist_3 metric" if config.COLLISION_INTERP_EXCLUDE_LAST_JOINT else "")
            + ")"
        )
    is_collision, n_collisions = batch_collision_check(
        collision_traj, robot_cfg, world_config,
    )
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
    ee_positions, ee_quaternions = compute_fk(final_traj, robot_cfg)
    print(f"  Computed FK for {len(final_traj)} waypoints")

    traj_times, time_stats = compute_trajectory_times(
        final_traj, ee_positions, ee_quaternions,
        ee_speed_m_s=EE_SPEED_MM_S / 1000.0,
        ee_angular_speed_rad_s=np.deg2rad(EE_ANGULAR_SPEED_DEG_S),
        max_joint_vel_rad_s=MAX_JOINT_VEL_RAD_S,
        min_segment_dt=MIN_SEGMENT_DT_S,
        corner_slowdown_enabled=CORNER_SLOWDOWN_ENABLED,
        corner_angle_threshold_rad=np.deg2rad(CORNER_ANGLE_THRESHOLD_DEG),
        corner_max_slowdown=CORNER_MAX_SLOWDOWN,
    )
    print(f"  Time profile: total={time_stats['total_time']:.1f}s, "
          f"max EE={time_stats['max_linear_speed_mm_s']:.1f} mm/s, "
          f"max rot={time_stats['max_angular_speed_deg_s']:.1f} deg/s, "
          f"max joint={time_stats['max_joint_speed_rad_s']:.2f} rad/s, "
          f"corners={time_stats['n_slow_segments']} seg "
          f"(max angle={time_stats['max_corner_angle_deg']:.1f}°, "
          f"slowdown={time_stats['max_slowdown']:.2f}x)")

    traj_dir = config.get_trajectory_path(args.object, args.num_viewpoints, "dummy").parent
    traj_dir.mkdir(parents=True, exist_ok=True)

    suffix = args.output_suffix
    spacing_str = f"{args.spacing:.3f}".replace(".", "")  # 0.010 → "0010", 0.050 → "0050"
    ee_speed_str = f"{EE_SPEED_MM_S:.0f}"
    ang_speed_str = f"{EE_ANGULAR_SPEED_DEG_S:.0f}"
    joint_vel_str = f"{MAX_JOINT_VEL_RAD_S:.2f}".replace(".", "p")
    tag = f"{suffix}_{RESAMPLE_MODE}_s{spacing_str}_eev{ee_speed_str}mms_av{ang_speed_str}dps_jv{joint_vel_str}"
    if CORNER_SLOWDOWN_ENABLED:
        corner_thresh_str = f"{CORNER_ANGLE_THRESHOLD_DEG:.0f}"
        corner_slow_str = f"{CORNER_MAX_SLOWDOWN:.1f}".replace(".", "p")
        tag = f"{tag}_corner{corner_thresh_str}d_x{corner_slow_str}"

    csv_path = str(traj_dir / f"trajectory_{tag}.csv")
    save_trajectory_csv(
        final_traj, ee_positions, ee_quaternions, csv_path,
        times=traj_times,
    )

    # static_path = str(traj_dir / f"trajectory_{tag}.html")
    # visualize_static_html(args.object, final_traj, ee_positions, static_path)

    # anim_path = str(traj_dir / f"trajectory_{tag}_anim.html")
    # visualize_animated_html(args.object, final_traj, ee_positions, anim_path)

    n_transit_ok = len(transit_segments)
    print(f"\nDone. reconfigs={stats['n_reconfigs']} "
          f"(inter={rc_inter}, intra={rc_intra}), "
          f"transit={n_transit_ok}/{len(reconfig_indices)} OK, "
          f"collisions={n_collisions}, "
          f"final={len(final_traj)} waypoints")


if __name__ == "__main__":
    main()
