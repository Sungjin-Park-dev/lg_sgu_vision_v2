#!/usr/bin/env python3
"""
뷰포인트에서 충돌 없는 로봇 궤적 생성

비전 검사 파이프라인:
1. 뷰포인트 데이터 로드(표면 위치 + 법선 벡터)
2. 각 뷰포인트에 대한 IK 해 계산
3. 충돌 없는 IK 해만 필터링
4. GTSP로 방문 순서(투어) 최적화
5. 궤적 보간(interpolation) 및 충돌 검사
6. 충돌하는 구간은 cuRobo으로 재계획(replan)
    - 만약 motion planning에 실패하는 경우, 그 다음 viewpoint를 시도
    - 성공 시, 해당 viewpoint를 건너뛰고 검사
    - 실패 시, 가능한 viewpoint까지만 궤적 생성
7. 최종 충돌 없는 궤적을 CSV로 저장

사용법:
    omni_python scripts/2_generate_trajectory.py \
        --object sample \
        --num_viewpoints 163

경로는 자동으로 생성됩니다:
- 입력:  data/{object}/viewpoint/{num_viewpoints}/viewpoints.h5
- 출력:  data/{object}/trajectory/{num_viewpoints}/trajectory.csv
- 메쉬:  data/{object}/mesh/source.obj
"""

# ============================================================================
# Section 1: Imports & Configuration
# ============================================================================

# Standard library
import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Third-party
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

# CuRobo imports
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig, Mesh
from curobo.types.base import TensorDeviceType
from curobo.types.state import JointState
from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

# EAIK imports (external IK solver)
try:
    from eaik.IK_URDF import UrdfRobot
    EAIK_AVAILABLE = True
except ImportError:
    EAIK_AVAILABLE = False
    print("Warning: EAIK not available. IK computation will fail.")

# Optional visualization
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

# Import configuration from common
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import config

# EAIK transformation constant
CUROBO_TO_EAIK_TOOL = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# Global variable for robot kinematics model
KINEMATICS_MODEL = None


# ============================================================================
# Section 2: Math Utilities
# ============================================================================

def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize vectors to unit length

    Args:
        vectors: (N, 3) array of vectors or (3,) single vector

    Returns:
        normalized: Unit vectors with same shape as input
    """
    if vectors.size == 0:
        return vectors

    # Handle both 1D and 2D arrays
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        return vectors / np.maximum(norm, 1e-9)

    # 2D array: normalize each row
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    return vectors / norms


def offset_points_along_normals(
    points: np.ndarray,
    normals: np.ndarray,
    offset: float
) -> np.ndarray:
    """
    Offset points along their normals by a given distance

    Args:
        points: (N, 3) array of 3D points
        normals: (N, 3) array of normal vectors
        offset: Distance to offset along normals (meters)

    Returns:
        offset_points: (N, 3) array of offset points
    """
    if points.size == 0:
        return points

    if points.shape != normals.shape:
        raise ValueError(f"Points and normals must have same shape")

    safe_normals = normalize_vectors(normals)
    return points + safe_normals * offset


def transform_pose_to_world(
    local_pose: np.ndarray,
    object_world_pose: np.ndarray
) -> np.ndarray:
    """
    Transform local pose to world frame

    Args:
        local_pose: 4x4 pose matrix in object's local frame
        object_world_pose: 4x4 transformation of object in world frame

    Returns:
        4x4 pose matrix in world frame
    """
    if local_pose.shape != (4, 4):
        raise ValueError("local_pose must be 4x4")
    if object_world_pose.shape != (4, 4):
        raise ValueError("object_world_pose must be 4x4")

    return object_world_pose @ local_pose


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to 3x3 rotation matrix

    Args:
        q: Quaternion as (w, x, y, z)

    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = q
    return np.array([
        [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
    ], dtype=np.float64)


def rot_to_quat_batch(R_batch: np.ndarray) -> np.ndarray:
    """
    Convert batch of rotation matrices to quaternions

    Args:
        R_batch: (N, 3, 3) array of rotation matrices

    Returns:
        (N, 4) array of quaternions as (w, x, y, z)
    """
    batch_size = R_batch.shape[0]
    quats = np.zeros((batch_size, 4), dtype=np.float64)

    for i in range(batch_size):
        r = Rotation.from_matrix(R_batch[i])
        q_xyzw = r.as_quat()  # Returns (x, y, z, w)
        # Convert to (w, x, y, z)
        quats[i] = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

    return quats


def _quat_to_rot_matrix_torch(quats: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to rotation matrices using PyTorch

    Args:
        quats: (N, 4) tensor of quaternions as (w, x, y, z)

    Returns:
        (N, 3, 3) tensor of rotation matrices
    """
    w, x, y, z = quats.unbind(-1)
    two_s = 2.0 / (quats * quats).sum(dim=-1)

    o = torch.stack(
        (
            1 - two_s * (y * y + z * z),
            two_s * (x * y - z * w),
            two_s * (x * z + y * w),
            two_s * (x * y + z * w),
            1 - two_s * (x * x + z * z),
            two_s * (y * z - x * w),
            two_s * (x * z - y * w),
            two_s * (y * z + x * w),
            1 - two_s * (x * x + y * y),
        ),
        -1,
    )
    return o.view(quats.shape[:-1] + (3, 3))


# ============================================================================
# Section 3: Data I/O
# ============================================================================

def load_viewpoints_hdf5(hdf5_path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load viewpoints from HDF5 file

    Args:
        hdf5_path: Path to HDF5 file

    Returns:
        positions: (N, 3) array of surface positions (meters)
        normals: (N, 3) array of surface normals (unit vectors)
        metadata: Dictionary containing metadata and camera_spec
    """
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"Viewpoints file not found: {hdf5_path}")

    with h5py.File(hdf5_path, 'r') as f:
        if 'viewpoints' not in f:
            raise ValueError(f"Invalid HDF5 format: missing 'viewpoints' group")

        viewpoints_grp = f['viewpoints']

        if 'positions' not in viewpoints_grp or 'normals' not in viewpoints_grp:
            raise ValueError(f"Invalid HDF5 format: missing positions or normals")

        positions = np.array(viewpoints_grp['positions'], dtype=np.float64)
        normals = np.array(viewpoints_grp['normals'], dtype=np.float64)

        # Load metadata
        metadata = {}
        if 'metadata' in f:
            metadata_grp = f['metadata']
            for key in metadata_grp.attrs:
                metadata[key] = metadata_grp.attrs[key]

            # Load camera_spec if present
            if 'camera_spec' in metadata_grp:
                camera_spec = {}
                for key in metadata_grp['camera_spec'].attrs:
                    camera_spec[key] = metadata_grp['camera_spec'].attrs[key]
                metadata['camera_spec'] = camera_spec

    return positions, normals, metadata


def save_trajectory_csv(
    trajectory: np.ndarray,
    output_path: str,
    clusters: List[Dict],
    order: List[int],
    picked: Dict[int, int]
) -> Path:
    """
    Save trajectory to CSV file (includes interpolated and replanned configs)

    CSV format matches original pipeline:
    time, 6 joint angles, target position (x,y,z), target rotation (x,y,z,w as quaternion)

    Args:
        trajectory: (N, n_joints) array of ALL joint configurations
                   (includes waypoints + interpolated + replanned)
        output_path: Path to output CSV file
        clusters: Not used (kept for API compatibility)
        order: Not used (kept for API compatibility)
        picked: Not used (kept for API compatibility)

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "time",
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
        "target-POS_X", "target-POS_Y", "target-POS_Z",
        "target-ROT_X", "target-ROT_Y", "target-ROT_Z", "target-ROT_W",
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        # Save ALL configurations in trajectory (interpolated + replanned)
        for step, q in enumerate(trajectory):
            # Joint configuration (6 joints, pad with 0 if needed)
            if q.shape[0] >= 6:
                q6 = q[:6]
            else:
                q6 = list(q) + [0.0] * (6 - len(q))

            # Compute target position/rotation using FK
            R, pos = fk_single(q[:6], tool_z=0.0)
            Q_wxyz = rot_to_quat_batch(R[None, :])[0]
            rot_x, rot_y, rot_z, rot_w = Q_wxyz[1], Q_wxyz[2], Q_wxyz[3], Q_wxyz[0]

            row = [
                float(step),
                float(q6[0]), float(q6[1]), float(q6[2]),
                float(q6[3]), float(q6[4]), float(q6[5]),
                float(pos[0]), float(pos[1]), float(pos[2]),
                float(rot_x), float(rot_y), float(rot_z), float(rot_w),
            ]
            writer.writerow(row)

    return output_path


# ============================================================================
# Section 4: IK Computation
# ============================================================================

@dataclass
class Viewpoint:
    """Represents a camera viewpoint with IK solutions"""
    index: int
    local_pose: Optional[np.ndarray] = None  # 4x4 pose in object frame
    world_pose: Optional[np.ndarray] = None  # 4x4 pose in world frame
    all_ik_solutions: List[np.ndarray] = field(default_factory=list)
    safe_ik_solutions: List[np.ndarray] = field(default_factory=list)


def setup_collision_world(
    table: Dict,
    walls: List[Dict],
    robot_mount: Dict,
    target_object: Dict,
    target_mesh_path: Optional[str] = None,
) -> WorldConfig:
    """
    Setup collision world configuration with all obstacles

    Args:
        table: Table config dict with 'name', 'position', 'dimensions' keys
        walls: List of wall dicts with 'name', 'position', 'dimensions' keys
        robot_mount: Robot mount config dict with 'name', 'position', 'dimensions' keys
        target_object: Target object config dict with 'name', 'position', 'rotation' keys
        target_mesh_path: Optional path to target object mesh file for collision

    Returns:
        WorldConfig containing all configured obstacles
    """
    # Load base world config (table)
    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose = list(table["position"]) + [1, 0, 0, 0]
    world_cfg_table.cuboid[0].dims = list(table["dimensions"])
    world_cfg_table.cuboid[0].name = table["name"]

    # Add wall cuboids
    wall_cuboids = []
    for wall in walls:
        wall_dict = {
            "table": {
                "dims": wall["dimensions"].tolist(),
                "pose": list(wall["position"]) + [1, 0, 0, 0]
            }
        }
        wall_cfg = WorldConfig.from_dict({"cuboid": wall_dict})
        wall_cfg.cuboid[0].name = wall["name"]
        wall_cuboids.extend(wall_cfg.cuboid)

    # Add robot mount cuboid
    robot_mount_cuboid_dict = {
        "table": {
            "dims": robot_mount["dimensions"].tolist(),
            "pose": list(robot_mount["position"]) + [1, 0, 0, 0]
        }
    }
    robot_mount_cfg = WorldConfig.from_dict({"cuboid": robot_mount_cuboid_dict})
    robot_mount_cfg.cuboid[0].name = robot_mount["name"]

    # Combine all cuboids
    all_cuboids = (
        world_cfg_table.cuboid +
        wall_cuboids +
        robot_mount_cfg.cuboid
    )

    # Ensure all cuboid poses are consistent Python lists (avoid numpy array mixing)
    for cuboid in all_cuboids:
        if hasattr(cuboid, 'pose') and cuboid.pose is not None:
            cuboid.pose = list(cuboid.pose)
        if hasattr(cuboid, 'dims') and cuboid.dims is not None:
            cuboid.dims = list(cuboid.dims)

    # Add target object mesh if provided
    meshes = []
    if target_mesh_path is not None:
        target_mesh = Mesh(
            name=target_object["name"],
            file_path=target_mesh_path,
            pose=list(target_object["position"]) + list(target_object["rotation"]),
        )
        meshes.append(target_mesh)

    world_cfg = WorldConfig(cuboid=all_cuboids, mesh=meshes if meshes else None)

    return world_cfg


def compute_ik_eaik(
    world_matrices: np.ndarray,
    urdf_path: str = None
):
    """
    Compute IK solutions using EAIK (analytical IK solver)

    Input poses are expected to be in camera_optical_frame.
    This function transforms them to tool0/wrist3 frame before solving IK.

    Args:
        world_matrices: (N, 4, 4) array of world pose matrices (camera_optical_frame)
        urdf_path: Path to robot URDF file (default from config)

    Returns:
        List of EAIK solution objects
    """
    if urdf_path is None:
        urdf_path = config.DEFAULT_URDF_PATH

    if isinstance(world_matrices, torch.Tensor):
        mats_np = world_matrices.detach().cpu().numpy().copy()
    else:
        mats_np = np.asarray(world_matrices, dtype=np.float64).copy()

    if mats_np.ndim != 3 or mats_np.shape[1:] != (4, 4):
        raise ValueError("Expected poses shaped (batch, 4, 4)")

    # Transform from camera_optical_frame to tool0/wrist3
    # Move back along Z-axis by TOOL_TO_CAMERA_OPTICAL_OFFSET_M
    z_axis = mats_np[:, :3, 2]  # Z-axis direction for each pose
    mats_np[:, :3, 3] -= z_axis * config.TOOL_TO_CAMERA_OPTICAL_OFFSET_M

    # Load URDF robot
    bot = UrdfRobot(urdf_path)

    # Transform from CuRobo to EAIK tool frame
    curobo_to_eaik_tool = CUROBO_TO_EAIK_TOOL.astype(mats_np.dtype, copy=False)
    mats_eaik = mats_np @ curobo_to_eaik_tool

    # Compute IK solutions
    solutions = bot.IK_batched(mats_eaik)

    return solutions


def assign_ik_solutions_to_viewpoints(
    viewpoints: List[Viewpoint],
    eaik_results,
    indices: Optional[List[int]] = None
):
    """
    Assign EAIK IK solutions to viewpoints with constraint filtering
    """
    # Clear existing solutions
    for viewpoint in viewpoints:
        viewpoint.all_ik_solutions = []
        viewpoint.safe_ik_solutions = []

    if eaik_results is None:
        return

    try:
        eaik_list = list(eaik_results)
    except TypeError:
        eaik_list = [eaik_results]

    if indices is None:
        indices = list(range(len(viewpoints)))

    max_assignable = min(len(indices), len(eaik_list))

    for result_idx in range(max_assignable):
        viewpoint_idx = indices[result_idx]
        result = eaik_list[result_idx]

        if result is None:
            continue

        q_candidates = getattr(result, "Q", None)
        if q_candidates is None or len(q_candidates) == 0:
            continue

        valid_solutions = []
        
        # IK 해 후보들에 대해 필터링 수행
        for q in q_candidates:
            q_np = np.asarray(q, dtype=np.float64)
            
            # [NEW] 로봇 제약 조건 확인
            if config.ROBOT_HAS_CONSTRAINT:
                # 검사 대상: Index 0 ~ 4 (마지막 조인트 제외)
                # 시작 자세와의 차이(절댓값) 계산
                diff = np.abs(q_np[:5] - config.ROBOT_START_STATE[:5])
                
                # 하나라도 허용 범위를 초과하면 제외 (continue)
                if np.any(diff > config.MAX_JOINT_FROM_START_STATE):
                    continue
            
            valid_solutions.append(q_np)

        viewpoints[viewpoint_idx].all_ik_solutions = valid_solutions


def check_ik_solutions_collision(
    viewpoints: List[Viewpoint],
    ik_solver: IKSolver,
    debug: bool = False
):
    """
    Check which IK solutions are collision-free and update safe_ik_solutions

    Args:
        viewpoints: List of Viewpoint objects with all_ik_solutions
        ik_solver: IK solver with collision checker
        debug: Print detailed collision diagnostics
    """
    # Clear safe solutions
    for viewpoint in viewpoints:
        viewpoint.safe_ik_solutions = []

    # Batch all solutions
    batched_q: List[np.ndarray] = []
    index_map: List[Tuple[int, int]] = []  # (viewpoint_idx, solution_idx)

    for vp_idx, viewpoint in enumerate(viewpoints):
        for sol_idx, solution in enumerate(viewpoint.all_ik_solutions):
            batched_q.append(np.asarray(solution, dtype=np.float64))
            index_map.append((vp_idx, sol_idx))

    if not batched_q:
        return

    # Convert to tensor
    batched_array = np.stack(batched_q, axis=0)
    tensor_args = getattr(ik_solver, "tensor_args", TensorDeviceType())
    q_tensor = tensor_args.to_device(torch.from_numpy(batched_array))

    # Create joint state
    zeros = torch.zeros_like(q_tensor)
    joint_state = JointState(
        position=q_tensor,
        velocity=zeros,
        acceleration=zeros,
        jerk=zeros,
        joint_names=ik_solver.kinematics.joint_names,
    )

    # Check constraints
    metrics = ik_solver.check_constraints(joint_state)
    feasible = getattr(metrics, "feasible", None)

    if feasible is None:
        feasibility = torch.ones(len(index_map), dtype=torch.bool)
    else:
        feasibility = feasible.detach()
        if feasibility.is_cuda:
            feasibility = feasibility.cpu()
        feasibility = feasibility.flatten().to(dtype=torch.bool)

    # Debug output
    if debug:
        num_total = len(feasibility)
        num_feasible = int(feasibility.sum().item())
        print(f"\n  DEBUG: Collision check results:")
        print(f"    Total IK solutions: {num_total}")
        print(f"    Feasible: {num_feasible}")
        print(f"    Infeasible: {num_total - num_feasible}")

        # Print all available metrics attributes
        print(f"\n  DEBUG: Available metrics attributes:")
        for attr in dir(metrics):
            if not attr.startswith('_'):
                print(f"    - {attr}")

        # Check individual constraint types
        if hasattr(metrics, 'self_collision_distance'):
            self_coll = metrics.self_collision_distance
            if self_coll.is_cuda:
                self_coll = self_coll.cpu()
            min_dist = self_coll.min().item()
            print(f"    Min self-collision distance: {min_dist:.4f}")

        if hasattr(metrics, 'world_collision_distance'):
            world_coll = metrics.world_collision_distance
            if world_coll.is_cuda:
                world_coll = world_coll.cpu()
            min_dist = world_coll.min().item()
            print(f"    Min world collision distance: {min_dist:.4f}")

        # Sample a few infeasible solutions for detailed inspection
        if num_feasible == 0 and num_total > 0:
            print(f"\n  DEBUG: Inspecting first infeasible solution:")
            sample_q = batched_array[0]
            print(f"    Joint values (rad): {sample_q}")
            print(f"    Joint values (deg): {np.degrees(sample_q)}")

    # Update safe solutions
    for batch_idx, ((vp_idx, _), is_feasible) in enumerate(zip(index_map, feasibility)):
        if not bool(is_feasible):
            continue
        solution = batched_q[batch_idx]
        viewpoints[vp_idx].safe_ik_solutions.append(solution)


# ============================================================================
# Section 5: GTSP Optimization
# ============================================================================

def init_robot_model(robot_config_file: str):
    """
    Initialize CuRobo RobotWorld for FK computation

    Args:
        robot_config_file: Robot configuration YAML file name
    """
    global KINEMATICS_MODEL

    print(f"  Initializing robot kinematics model from: {robot_config_file}")

    # Load robot configuration
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_config_file))["robot_cfg"]

    # Setup kinematics-only configuration (no collision world needed for FK)
    world_config = WorldConfig()

    config_obj = RobotWorldConfig.load_from_config(
        robot_cfg,
        world_config,
        tensor_args=TensorDeviceType(device=torch.device("cuda:0"), dtype=torch.float32)
    )

    # Initialize model
    KINEMATICS_MODEL = RobotWorld(config_obj)
    print(f"  ✓ Robot kinematics model initialized")


def fk_single(q: np.ndarray, tool_z: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward kinematics for single configuration using CuRobo

    Args:
        q: Joint angles (6,) in radians
        tool_z: Tool offset along Z-axis (meters)

    Returns:
        R: 3x3 rotation matrix
        p: 3D position (meters)
    """
    global KINEMATICS_MODEL
    if KINEMATICS_MODEL is None:
        raise RuntimeError("Robot model not initialized. Call init_robot_model() first.")

    # Convert to torch tensor and add batch dimension
    q_tensor = torch.tensor(q, device='cuda:0', dtype=torch.float32).unsqueeze(0)

    # Compute FK using CuRobo
    state = KINEMATICS_MODEL.get_kinematics(q_tensor)

    # Extract position and quaternion
    ee_pos = state.ee_position[0]  # (3,)
    ee_quat = state.ee_quaternion[0]  # (4,) as (w, x, y, z)

    # Convert quaternion to rotation matrix
    ee_rot = _quat_to_rot_matrix_torch(ee_quat.unsqueeze(0))[0]  # (3, 3)

    # Apply tool offset
    if tool_z != 0.0:
        z_axis = ee_rot[:, 2]  # (3,)
        ee_pos = ee_pos + z_axis * tool_z

    # Convert to numpy
    R = ee_rot.cpu().numpy()
    p = ee_pos.cpu().numpy()

    return R, p

def fk_batch(qs: np.ndarray, tool_z: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batch forward kinematics using CuRobo

    Args:
        qs: (N, 6) array of joint angles
        tool_z: Tool offset along Z-axis (meters)

    Returns:
        R: (N, 3, 3) rotation matrices
        p: (N, 3) positions
    """
    global KINEMATICS_MODEL
    if KINEMATICS_MODEL is None:
        raise RuntimeError("Robot model not initialized. Call init_robot_model() first.")

    # Convert to torch tensor
    q_tensor = torch.tensor(qs, device='cuda:0', dtype=torch.float32)

    # Compute FK using CuRobo
    state = KINEMATICS_MODEL.get_kinematics(q_tensor)

    # Extract position and quaternion
    ee_pos = state.ee_position  # (N, 3)
    ee_quat = state.ee_quaternion  # (N, 4) as (w, x, y, z)

    # Convert quaternion to rotation matrix
    ee_rot = _quat_to_rot_matrix_torch(ee_quat)  # (N, 3, 3)

    # Apply tool offset
    if tool_z != 0.0:
        z_axis = ee_rot[:, :, 2]  # (N, 3)
        ee_pos = ee_pos + z_axis * tool_z

    # Convert to numpy
    R = ee_rot.cpu().numpy()
    p = ee_pos.cpu().numpy()

    return R, p


def build_neighbors_knn(target_coords: np.ndarray, k: int) -> List[List[int]]:
    """
    Build k-nearest neighbor graph

    Args:
        target_coords: (M, 3) array of target positions
        k: Number of nearest neighbors

    Returns:
        nbrs: List of M neighbor lists
    """
    M = target_coords.shape[0]
    nbrs = [[] for _ in range(M)]

    # Distance matrix
    dif = target_coords[:, None, :] - target_coords[None, :, :]
    dist = np.linalg.norm(dif, axis=-1)
    np.fill_diagonal(dist, np.inf)

    for i in range(M):
        idx = np.argsort(dist[i])[:max(1, min(k, M-1))]
        nbrs[i] = idx.tolist()

    return nbrs


def build_clusters_from_ik(
    viewpoints: List[Viewpoint],
    tool_z: float = 0.0
) -> Tuple[List[Dict], np.ndarray]:
    """
    Build clusters from viewpoints with IK solutions

    Args:
        viewpoints: List of Viewpoint objects with safe_ik_solutions
        tool_z: Tool offset for FK

    Returns:
        clusters: List of cluster dicts with q, R, p, Q, target, target_Q
        target_coords: (M, 3) array of target positions
    """
    clusters = []
    target_coords = []

    for vp in viewpoints:
        if len(vp.safe_ik_solutions) == 0:
            continue

        # Stack IK solutions
        q_list = vp.safe_ik_solutions
        q_array = np.stack(q_list, axis=0)

        # Compute FK for each IK solution
        R, p = fk_batch(q_array, tool_z)
        Q = rot_to_quat_batch(R)

        # Target pose from world_pose
        if vp.world_pose is not None:
            target = vp.world_pose[:3, 3]
            target_R = vp.world_pose[:3, :3]
            target_Q = rot_to_quat_batch(target_R[None, :])[0]
        else:
            target = p[0]  # Use first IK solution's position
            target_Q = Q[0]

        cluster = {
            "q": q_array,
            "R": R,
            "p": p,
            "Q": Q,
            "target": target,
            "target_Q": target_Q,
        }
        clusters.append(cluster)
        target_coords.append(target)

    # Ensure proper shape even when empty
    if target_coords:
        target_coords = np.array(target_coords, dtype=np.float64)
    else:
        target_coords = np.empty((0, 3), dtype=np.float64)

    return clusters, target_coords


def compute_motion_cost(
    q_a: np.ndarray, R_a: np.ndarray,
    q_b: np.ndarray, R_b: np.ndarray,
    lam_rot: float, tool_z: float
) -> float:
    """
    Compute motion cost between two configurations via midpoint

    Args:
        q_a, q_b: Joint configurations
        R_a, R_b: End-effector rotations
        lam_rot: Rotation cost weight
        tool_z: Tool offset

    Returns:
        Cost value
    """
    # Joint-space midpoint
    q_mid = 0.5 * (q_a + q_b)
    R_mid, p_mid = fk_single(q_mid, tool_z)
    _, p_a = fk_single(q_a, tool_z)
    _, p_b = fk_single(q_b, tool_z)

    # Position costs
    d1 = np.linalg.norm(p_a - p_mid)
    d2 = np.linalg.norm(p_mid - p_b)

    # Rotation costs (Z-axis angle)
    za = R_a[:, 2]
    zm = R_mid[:, 2]
    zb = R_b[:, 2]

    angle1 = np.arccos(np.clip(np.dot(za, zm), -1.0, 1.0))
    angle2 = np.arccos(np.clip(np.dot(zm, zb), -1.0, 1.0))

    return d1 + lam_rot * angle1 + d2 + lam_rot * angle2


def build_visit_order_robot_cost(
    clusters: List[Dict],
    nbrs: List[List[int]],
    lam_rot: float,
    tool_z: float
) -> List[int]:
    """
    Build visit order using greedy TSP with robot motion cost

    Args:
        clusters: List of cluster dictionaries
        nbrs: Neighbor lists
        lam_rot: Rotation cost weight
        tool_z: Tool offset

    Returns:
        order: Visit order (cluster indices)
    """
    M = len(clusters)
    if M == 0:
        return []

    # Start from cluster closest to origin
    target_coords = np.array([c["target"] for c in clusters])
    start = int(np.argmin(np.linalg.norm(target_coords, axis=1)))

    order = [start]
    visited = np.zeros(M, dtype=bool)
    visited[start] = True
    cur = start

    # Greedy selection
    for _ in range(M - 1):
        best_cost = np.inf
        best_j = -1

        # Try neighbors first
        cand = [v for v in nbrs[cur] if not visited[v]]

        if cand:
            # Compute cost to each neighbor (using first IK solution)
            for j in cand:
                cost = compute_motion_cost(
                    clusters[cur]["q"][0], clusters[cur]["R"][0],
                    clusters[j]["q"][0], clusters[j]["R"][0],
                    lam_rot, tool_z
                )
                if cost < best_cost:
                    best_cost, best_j = cost, j
        else:
            # Fallback: nearest unvisited by Euclidean distance
            dists = np.linalg.norm(target_coords[cur] - target_coords, axis=1)
            dists[visited] = np.inf
            best_j = int(np.argmin(dists))

        if best_j == -1:
            break

        visited[best_j] = True
        order.append(best_j)
        cur = best_j

    return order


def choose_ik_given_order(
    clusters: List[Dict],
    order: List[int],
    lam_rot: float,
    tool_z: float
) -> Tuple[Dict[int, int], float]:
    """
    Optimize IK selection for given visit order using DP

    Args:
        clusters: List of cluster dictionaries
        order: Visit order (cluster indices)
        lam_rot: Rotation cost weight
        tool_z: Tool offset

    Returns:
        picked: Dict mapping cluster index → IK index
        total_cost: Total trajectory cost
    """
    n = len(order)
    if n == 0:
        return {}, 0.0

    # Pre-compute cost matrices
    cost_mats = []
    for t in range(n - 1):
        ia, ib = order[t], order[t + 1]
        A, B = clusters[ia], clusters[ib]

        Sa, Sb = A["q"].shape[0], B["q"].shape[0]
        C = np.empty((Sa, Sb), dtype=np.float64)

        for i in range(Sa):
            for j in range(Sb):
                C[i, j] = compute_motion_cost(
                    A["q"][i], A["R"][i],
                    B["q"][j], B["R"][j],
                    lam_rot, tool_z
                )

        cost_mats.append(C)

    # DP
    Sa0 = clusters[order[0]]["q"].shape[0]
    dp = np.zeros(Sa0, dtype=np.float64)
    back = [np.full((Sa0,), -1, dtype=np.int32)]

    for t in range(n - 1):
        C = cost_mats[t]
        Sa, Sb = C.shape

        tmp = dp[:, None] + C
        dp = np.min(tmp, axis=0)
        arg = np.argmin(tmp, axis=0).astype(np.int32)
        back.append(arg)

    # Backtrack
    j_star = int(np.argmin(dp))
    total_cost = float(dp[j_star])

    picked_local = [None] * n
    picked_local[-1] = j_star
    for t in range(n - 2, -1, -1):
        i_star = int(back[t + 1][picked_local[t + 1]])
        picked_local[t] = i_star

    # Map to cluster indices
    picked = {}
    for t, cidx in enumerate(order):
        picked[cidx] = int(picked_local[t])

    return picked, total_cost


# ============================================================================
# Section 6: Collision Checking with Replanning
# ============================================================================

def generate_interpolated_path(
    start: np.ndarray,
    end: np.ndarray,
    num_steps: int
) -> List[np.ndarray]:
    """
    Generate linear interpolation between two configurations

    Args:
        start: Starting configuration
        end: Ending configuration
        num_steps: Number of intermediate steps

    Returns:
        path: List of interpolated configs (excludes start, includes end)
    """
    if num_steps <= 0:
        return [end]

    alphas = np.linspace(0.0, 1.0, num_steps + 1)[1:]
    path = [start + alpha * (end - start) for alpha in alphas]

    return path


def check_trajectory_with_skip_on_failure(
    trajectory: np.ndarray,
    order: List[int],
    picked: Dict[int, int],
    clusters: List[Dict],
    motion_gen: MotionGen,
    max_joint_step_deg: float = None
) -> Tuple[np.ndarray, List[int], Dict[int, int], Dict]:
    """
    Check trajectory collision with automatic waypoint skip on first failure

    If replanning fails for segment i→i+1:
    1. Try to skip waypoint i+1 and connect i→i+2 directly
    2. If successful: continue with remaining waypoints (excluding i+1)
    3. If failed: truncate trajectory up to waypoint i+1

    Args:
        trajectory: Initial trajectory waypoints (N, 6)
        order: Visit order (cluster indices)
        picked: IK selection {cluster_idx: ik_idx}
        clusters: Cluster data
        motion_gen: MotionGen for collision checking
        max_joint_step_deg: Maximum joint step for interpolation

    Returns:
        final_trajectory: Final trajectory (possibly with skipped waypoint)
        final_order: Updated order (possibly with skipped waypoint removed)
        final_picked: Updated picked dict
        stats: Statistics including 'skipped_waypoint_idx' if any
    """
    # First attempt: check original trajectory
    final_traj, stats = check_trajectory_collision(
        trajectory, motion_gen, max_joint_step_deg
    )

    # If no failures, return as is
    if not stats['failed_segments']:
        stats['skipped_waypoint_idx'] = None
        stats['skip_attempted'] = False
        return final_traj, order, picked, stats

    # Get first failed segment
    first_failed_seg = stats['failed_segments'][0]

    print(f"    First failure at segment {first_failed_seg}→{first_failed_seg+1}")

    # Check if we can skip (need at least one more waypoint after failure)
    if first_failed_seg + 2 >= len(trajectory):
        # Cannot skip - this is the last or second-to-last waypoint
        print(f"    Cannot skip: no waypoint after {first_failed_seg+1}")
        stats['skipped_waypoint_idx'] = None
        stats['skip_attempted'] = False
        stats['skip_successful'] = False
        stats['truncated_at'] = first_failed_seg + 1
        return final_traj, order, picked, stats

    # Try motion planning from waypoint[first_failed_seg] to waypoint[first_failed_seg+2]
    # to see if we can skip waypoint[first_failed_seg+1]
    print(f"    Attempting to skip waypoint {first_failed_seg+1}, connecting {first_failed_seg}→{first_failed_seg+2}...")

    start_config = trajectory[first_failed_seg]
    goal_config = trajectory[first_failed_seg + 2]

    # Try direct motion planning
    result = motion_gen.plan_single_js(
        start_state=JointState.from_position(
            torch.tensor(start_config, dtype=torch.float32, device=motion_gen.tensor_args.device).unsqueeze(0)
        ),
        goal_state=JointState.from_position(
            torch.tensor(goal_config, dtype=torch.float32, device=motion_gen.tensor_args.device).unsqueeze(0)
        ),
        plan_config=MotionGenPlanConfig(
            enable_opt=True,
            timeout=config.REPLAN_TIMEOUT,
            max_attempts=config.REPLAN_MAX_ATTEMPTS,
        )
    )

    if not result.success.item():
        # Skip failed - truncate trajectory at first_failed_seg+1
        print(f"    Skip failed - truncating trajectory at waypoint {first_failed_seg+1}")

        # Build truncated trajectory up to and including waypoint at first_failed_seg+1
        truncated_traj = []
        for i in range(first_failed_seg + 2):  # Include waypoints 0 to first_failed_seg+1
            truncated_traj.append(trajectory[i])
        truncated_traj = np.array(truncated_traj, dtype=np.float64)

        # Truncate order and picked
        truncated_order = order[:first_failed_seg + 2]
        truncated_picked = {cluster_idx: picked[cluster_idx] for cluster_idx in truncated_order}

        # Process truncated trajectory to get interpolated configs
        truncated_final, truncated_stats = check_trajectory_collision(
            truncated_traj, motion_gen, max_joint_step_deg
        )

        truncated_stats['skipped_waypoint_idx'] = None
        truncated_stats['skip_attempted'] = True
        truncated_stats['skip_successful'] = False
        truncated_stats['truncated_at'] = first_failed_seg + 1

        return truncated_final, truncated_order, truncated_picked, truncated_stats

    # Skip successful - rebuild trajectory excluding waypoint[first_failed_seg+1]
    print(f"    Skip successful - removing waypoint {first_failed_seg+1} from trajectory")

    # Build new trajectory without the skipped waypoint
    new_trajectory = []
    new_order = []
    for i, q in enumerate(trajectory):
        if i == first_failed_seg + 1:
            continue  # Skip this waypoint
        new_trajectory.append(q)
        new_order.append(order[i])

    new_trajectory = np.array(new_trajectory, dtype=np.float64)
    new_picked = {cluster_idx: picked[cluster_idx] for cluster_idx in new_order}

    # Re-check the new trajectory for collisions
    print(f"    Re-checking trajectory with skipped waypoint...")
    final_traj, new_stats = check_trajectory_collision(
        new_trajectory, motion_gen, max_joint_step_deg
    )

    # Update stats
    new_stats['skipped_waypoint_idx'] = first_failed_seg + 1
    new_stats['skip_attempted'] = True
    new_stats['skip_successful'] = True
    new_stats['original_failed_segments'] = stats['failed_segments']

    return final_traj, new_order, new_picked, new_stats


def check_trajectory_collision(
    trajectory: np.ndarray,
    motion_gen: MotionGen,
    max_joint_step_deg: float = None,
    replan_enabled: bool = None
) -> Tuple[np.ndarray, Dict]:
    """
    Check trajectory for collisions and optionally replan colliding segments

    Simplified pipeline:
    1. Adaptive interpolation
    2. Batch collision check
    3. Replan colliding segments
    4. Return final trajectory

    Args:
        trajectory: (N, n_joints) trajectory
        motion_gen: MotionGen for collision checking and replanning
        max_joint_step_deg: Max joint step for interpolation (default from config)
        replan_enabled: Enable replanning (default from config)

    Returns:
        final_trajectory: Collision-checked trajectory
        stats: Statistics dictionary
    """
    if max_joint_step_deg is None:
        max_joint_step_deg = config.COLLISION_ADAPTIVE_MAX_JOINT_STEP_DEG
    if replan_enabled is None:
        replan_enabled = config.REPLAN_ENABLED

    num_waypoints = len(trajectory)
    max_joint_step_rad = np.deg2rad(max_joint_step_deg)

    # Step 1: Adaptive interpolation
    interpolated_segments = []
    segment_indices = []

    for i in range(num_waypoints - 1):
        delta = np.abs(trajectory[i + 1] - trajectory[i])
        max_delta = np.max(delta[:-1])  # Exclude last joint
        num_steps = int(np.ceil(max_delta / max_joint_step_rad))

        if num_steps > 0:
            interp = generate_interpolated_path(
                trajectory[i], trajectory[i + 1], num_steps
            )
            interpolated_segments.extend(interp)
            segment_indices.extend([i] * len(interp))

    # Combine waypoints and interpolated points
    all_configs = list(trajectory) + interpolated_segments
    all_indices = list(range(num_waypoints)) + segment_indices

    # Step 2: Batch collision check
    q_array = np.stack(all_configs, axis=0)
    tensor_args = motion_gen.tensor_args
    q_tensor = tensor_args.to_device(torch.from_numpy(q_array.astype(np.float32)))

    zeros = torch.zeros_like(q_tensor)
    joint_state = JointState(
        position=q_tensor,
        velocity=zeros,
        acceleration=zeros,
        jerk=zeros,
        joint_names=motion_gen.kinematics.joint_names,
    )

    metrics = motion_gen.check_constraints(joint_state)
    feasible = getattr(metrics, "feasible", None)

    if feasible is None:
        feasibility = torch.ones(len(all_configs), dtype=torch.bool)
    else:
        feasibility = feasible.detach().cpu().flatten().to(dtype=torch.bool)

    # Find colliding segments
    collision_segments = set()
    for idx, is_feasible in enumerate(feasibility):
        if not bool(is_feasible):
            seg_idx = all_indices[idx]
            if seg_idx < num_waypoints - 1:
                collision_segments.add(seg_idx)

    collision_segments = sorted(collision_segments)

    # Step 3: Organize interpolated segments by waypoint index
    segment_configs = {}
    for seg_idx, interp_config in zip(segment_indices, interpolated_segments):
        if seg_idx not in segment_configs:
            segment_configs[seg_idx] = []
        segment_configs[seg_idx].append(interp_config)

    # Step 4: Replan colliding segments
    segment_plans = {}
    replan_attempts = 0
    replan_successes = 0
    failed_segments = []

    if replan_enabled and len(collision_segments) > 0:
        for seg_idx in collision_segments:
            start_config = trajectory[seg_idx]
            goal_config = trajectory[seg_idx + 1]

            # Try replanning
            replan_attempts += 1

            start_tensor = tensor_args.to_device(
                torch.from_numpy(start_config.astype(np.float32)).unsqueeze(0)
            )
            goal_tensor = tensor_args.to_device(
                torch.from_numpy(goal_config.astype(np.float32)).unsqueeze(0)
            )

            start_state = JointState.from_position(start_tensor)
            goal_state = JointState.from_position(goal_tensor)

            result = motion_gen.plan_single_js(
                start_state=start_state,
                goal_state=goal_state,
                plan_config=MotionGenPlanConfig(
                    enable_opt=True,
                    timeout=config.REPLAN_TIMEOUT,
                    max_attempts=config.REPLAN_MAX_ATTEMPTS,
                )
            )

            if result.success.item():
                replan_successes += 1
                # Store replanned trajectory
                replanned_traj = result.interpolated_plan.position.cpu().numpy()
                segment_plans[seg_idx] = replanned_traj
            else:
                failed_segments.append(seg_idx)

    # Step 5: Build final trajectory (interpolated + replanned)
    final_configs = []

    for i in range(num_waypoints - 1):
        # Add start waypoint (only for first segment)
        if i == 0:
            final_configs.append(trajectory[i])

        # Use replanned path if available, otherwise use original interpolated path
        if i in segment_plans:
            # Use replanned trajectory (skip first point to avoid duplication)
            replanned_traj = segment_plans[i]
            final_configs.extend(replanned_traj[1:])
        elif i in segment_configs:
            # Use original interpolated path (skip first point)
            final_configs.extend(segment_configs[i][1:])
        else:
            # No interpolation needed (direct connection)
            final_configs.append(trajectory[i + 1])

    final_trajectory = np.array(final_configs, dtype=np.float64)

    stats = {
        "num_waypoints": num_waypoints,
        "num_interpolated": len(final_trajectory) - num_waypoints,
        "num_collisions": len(collision_segments),
        "replan_attempts": replan_attempts,
        "replan_successes": replan_successes,
        "failed_segments": failed_segments,
    }

    return final_trajectory, stats


# ============================================================================
# Section 7: Visualization (Optional)
# ============================================================================

def visualize_trajectory(
    trajectory: np.ndarray,
    clusters: List[Dict],
    order: List[int],
    picked: Dict[int, int],
    mesh_path: Optional[str] = None
):
    """
    Visualize trajectory with Open3D

    Args:
        trajectory: Joint trajectory
        clusters: Cluster data
        order: Visit order
        picked: Selected IK indices
        mesh_path: Optional mesh file path
    """
    if not OPEN3D_AVAILABLE:
        print("Open3D not available - skipping visualization")
        return

    geometries = []

    # Add coordinate frame at origin
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    geometries.append(frame)

    # Add mesh if provided
    if mesh_path and os.path.exists(mesh_path):
        try:
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            # Transform to world pose
            T = np.eye(4)
            T[:3, :3] = quaternion_to_rotation_matrix(config.TARGET_OBJECT["rotation"])
            T[:3, 3] = config.TARGET_OBJECT["position"]
            mesh.transform(T)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([0.7, 0.7, 0.7])
            geometries.append(mesh)
        except:
            print(f"Failed to load mesh: {mesh_path}")

    # Add target positions
    target_positions = []
    for cidx in order:
        if cidx in picked:
            target_positions.append(clusters[cidx]["target"])

    if target_positions:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(target_positions))
        pcd.paint_uniform_color([0.0, 1.0, 0.0])
        geometries.append(pcd)

    # Add trajectory path
    if len(target_positions) > 1:
        lines = [[i, i + 1] for i in range(len(target_positions) - 1)]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.array(target_positions))
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([0.0, 0.0, 1.0])
        geometries.append(line_set)

    print("\nOpening visualization window...")
    print("  - Gray mesh: Object")
    print("  - Green points: Target positions")
    print("  - Blue lines: Trajectory path")
    print("  - Mouse to rotate/zoom, 'Q' to quit\n")

    o3d.visualization.draw_geometries(geometries)


# ============================================================================
# Section 8: Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate collision-free robot trajectory from viewpoints"
    )
    parser.add_argument(
        "--object",
        type=str,
        required=True,
        help="Object name for auto-path generation (e.g., 'sample', 'glass')"
    )
    parser.add_argument(
        "--num_viewpoints",
        type=int,
        required=True,
        help="Number of viewpoints"
    )
    parser.add_argument(
        "--knn",
        type=int,
        default=config.DEFAULT_KNN,
        help=f"Number of nearest neighbors (default: {config.DEFAULT_KNN})"
    )
    parser.add_argument(
        "--lambda-rot",
        type=float,
        default=config.DEFAULT_LAMBDA_ROT,
        help=f"Rotation cost weight (default: {config.DEFAULT_LAMBDA_ROT})"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show trajectory visualization"
    )
    parser.add_argument(
        "--robot",
        type=str,
        default=config.DEFAULT_ROBOT_CONFIG,
        help=f"Robot configuration file (default: {config.DEFAULT_ROBOT_CONFIG})"
    )

    args = parser.parse_args()

    # Resolve paths using auto-path generation
    input_path = str(config.get_viewpoint_path(args.object, args.num_viewpoints))
    output_path = str(config.get_trajectory_path(args.object, args.num_viewpoints, "trajectory.csv"))
    mesh_path = str(config.get_mesh_path(args.object, mesh_type="source"))

    # Start timer
    start_time = time.time()

    print("=" * 60)
    print("GENERATE TRAJECTORY (Step 2)")
    print("=" * 60)
    print(f"Object: {args.object}")
    print(f"Num viewpoints: {args.num_viewpoints}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Mesh:   {mesh_path}")
    print("=" * 60)
    print()

    # Validate input exists
    if not os.path.exists(input_path):
        print(f"Error: Input viewpoints not found: {input_path}")
        return 1

    # Step 1: Load viewpoints
    print("[1/7] Loading viewpoints...")
    positions, normals, metadata = load_viewpoints_hdf5(input_path)
    print(f"  ✓ Loaded {len(positions)} viewpoints")
    print()

    # Determine working distance
    working_distance_m = config.CAMERA_WORKING_DISTANCE_MM / 1000.0
    if 'camera_spec' in metadata and 'working_distance_mm' in metadata['camera_spec']:
        working_distance_m = metadata['camera_spec']['working_distance_mm'] / 1000.0

    # Create camera poses (offset along normals)
    camera_positions = offset_points_along_normals(positions, normals, working_distance_m)
    approach_normals = -normalize_vectors(normals)

    # Create viewpoints with local poses
    viewpoints = []
    helper_z = np.array([0.0, 0.0, 1.0])
    helper_y = np.array([0.0, 1.0, 0.0])

    for idx, (pos, normal) in enumerate(zip(camera_positions, approach_normals)):
        z_axis = normal / np.linalg.norm(normal)
        helper = helper_z if np.abs(np.dot(z_axis, helper_z)) <= 0.99 else helper_y
        x_axis = np.cross(helper, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
        pose[:3, 3] = pos

        viewpoints.append(Viewpoint(index=idx, local_pose=pose))

    # Step 2: Setup collision world
    print("[2/7] Setting up collision world...")

    # Use auto-resolved mesh path
    if not os.path.exists(mesh_path):
        print(f"  ⚠ Mesh file not found: {mesh_path}")
        print(f"  ⚠ Using default world without object mesh")
        target_mesh_path = None
    else:
        print(f"  Using mesh: {mesh_path}")
        target_mesh_path = mesh_path

    world_cfg = setup_collision_world(
        table=config.TABLE,
        walls=config.WALLS,
        robot_mount=config.ROBOT_MOUNT,
        target_object=config.TARGET_OBJECT,
        target_mesh_path=target_mesh_path,
    )
    print(f"  ✓ Collision world ready")
    print()

    # Step 3: Setup IK solver
    print("[3/7] Computing IK solutions...")

    # Initialize robot kinematics model for FK
    init_robot_model(args.robot)

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]
    tensor_args = TensorDeviceType()

    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        self_collision_check=True,
        collision_checker_type=CollisionCheckerType.MESH,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )
    ik_solver = IKSolver(ik_config)

    # Transform viewpoints to world frame
    target_world_pose = np.eye(4, dtype=np.float64)
    target_world_pose[:3, :3] = quaternion_to_rotation_matrix(config.TARGET_OBJECT["rotation"])
    target_world_pose[:3, 3] = config.TARGET_OBJECT["position"]

    for vp in viewpoints:
        vp.world_pose = transform_pose_to_world(vp.local_pose, target_world_pose)

    # Compute IK solutions
    world_mats = np.stack([vp.world_pose for vp in viewpoints], axis=0)
    ik_results = compute_ik_eaik(world_mats)
    assign_ik_solutions_to_viewpoints(viewpoints, ik_results)

    num_with_ik = sum(1 for vp in viewpoints if len(vp.all_ik_solutions) > 0)
    print(f"  ✓ Computed IK for {len(viewpoints)} poses")
    print(f"  ✓ Found {num_with_ik} viewpoints with valid IK")
    print()

    # Step 4: Check collision on IK solutions
    print("[4/7] Checking collision on IK solutions...")
    check_ik_solutions_collision(viewpoints, ik_solver, debug=False)

    num_with_safe = sum(1 for vp in viewpoints if len(vp.safe_ik_solutions) > 0)
    print(f"  ✓ Found {num_with_safe} viewpoints with collision-free IK")
    print()

    # Step 5: Build GTSP
    print("[5/7] Optimizing visit order (GTSP)...")

    clusters, target_coords = build_clusters_from_ik(viewpoints)
    print(f"  Built {len(clusters)} clusters")

    nbrs = build_neighbors_knn(target_coords, args.knn)
    print(f"  Built k-NN graph (k={args.knn})")

    order = build_visit_order_robot_cost(clusters, nbrs, args.lambda_rot, 0.0)
    print(f"  Visit order determined")

    picked, total_cost = choose_ik_given_order(clusters, order, args.lambda_rot, 0.0)
    print(f"  ✓ IK selection optimized (cost: {total_cost:.2f})")
    print()

    # Build initial trajectory
    initial_trajectory = []
    for cidx in order:
        if cidx in picked:
            ik_idx = picked[cidx]
            initial_trajectory.append(clusters[cidx]["q"][ik_idx])
    initial_trajectory = np.array(initial_trajectory, dtype=np.float64)

    # Step 6: Collision checking with replanning
    print("[6/7] Checking collision with replanning...")

    # Setup MotionGen for replanning
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args=tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        use_cuda_graph=True,
        interpolation_dt=config.REPLAN_INTERP_DT,
        trajopt_tsteps=config.REPLAN_TRAJOPT_TSTEPS,
    )
    motion_gen = MotionGen(motion_gen_config)

    final_trajectory, final_order, final_picked, stats = check_trajectory_with_skip_on_failure(
        initial_trajectory, order, picked, clusters, motion_gen
    )

    print(f"  Waypoints: {stats['num_waypoints']}")
    print(f"  Interpolated: {stats['num_interpolated']}")
    print(f"  Collisions found: {stats['num_collisions']}")

    # Show skip information if attempted
    if stats.get('skip_attempted', False):
        if stats.get('skip_successful', False):
            print(f"  ⚠ Skipped waypoint: {stats['skipped_waypoint_idx']} (successfully bypassed)")
            print(f"  ✓ Remaining waypoints: {len(final_order)}/{len(order)}")
        else:
            print(f"  ⚠ Skip attempt failed - trajectory truncated at waypoint {stats.get('truncated_at', 'unknown')}")
            print(f"  ✓ Saved waypoints: {len(final_order)}/{len(order)}")

    # Show replanning success rate
    if stats['replan_attempts'] > 0:
        success_rate = (stats['replan_successes'] / stats['replan_attempts']) * 100
        print(f"  Replanning: {stats['replan_successes']}/{stats['replan_attempts']} succeeded ({success_rate:.1f}%)")

        # Show failed segments if any
        if stats['failed_segments']:
            print(f"  ⚠ Failed segments remaining: {stats['failed_segments']}")
            print(f"    (Waypoint pairs: {[(s, s+1) for s in stats['failed_segments']]})")
    else:
        print(f"  Replanning: No segments required replanning")
    print()

    # Step 7: Save trajectory (always save, even if truncated/skipped)
    print("[7/7] Saving trajectory...")

    # Use final_order and final_picked (possibly modified by skip logic)
    saved_path = save_trajectory_csv(
        final_trajectory, output_path, clusters, final_order, final_picked
    )
    print(f"  ✓ Saved to {saved_path}")
    print()

    # Optional visualization
    if args.visualize:
        visualize_trajectory(
            final_trajectory, clusters, final_order, final_picked, target_mesh_path
        )

    elapsed = time.time() - start_time
    print("=" * 60)
    print("✓ Step 2 Complete!")
    print(f"Total time: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
