#!/usr/bin/env python3
"""
Roll/Pitch 변형을 포함한 Tilt 궤적 생성

파이프라인:
1. 기존 trajectory CSV에서 기준(base) 포즈 로드
2. 기준 포즈 주변에 roll/pitch tilt 웨이포인트 생성
3. IK 해 계산 및 필터링
4. 웨이포인트 방문 순서 최적화
5. 최종 궤적을 CSV로 저장

사용법:
    omni_python scripts/2-1_generate_tilt_trajectory.py \
        --object sample \
        --num_viewpoints 163 \
        --row-idx 10 \
        --input-file trajectory.csv \
        --output-file tilt_trajectory.csv \
        --roll-min -20 --roll-max 20 --roll-n 200 \
        --pitch-min -20 --pitch-max 20 --pitch-n 200

경로는 자동으로 생성됩니다:
- 입력:  data/{object}/trajectory/{num_viewpoints}/{input_file}
- 출력:  data/{object}/trajectory/{num_viewpoints}/{output_file}
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
import numpy as np
import torch

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

# EAIK imports
try:
    from eaik.IK_URDF import UrdfRobot
    EAIK_AVAILABLE = True
except ImportError:
    EAIK_AVAILABLE = False
    print("Warning: EAIK not available. IK computation will fail.")

# Import configuration
THIS_DIR = Path(__file__).parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

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


# ============================================================================
# Section 2: Math Utilities (from tilt.py)
# ============================================================================

def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Normalize vector to unit length

    Args:
        v: Vector or array of vectors
        eps: Epsilon for numerical stability

    Returns:
        Normalized vector
    """
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n


def quat_xyzw_to_rot(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [x, y, z, w] to 3x3 rotation matrix

    Args:
        q: (4,) quaternion in [x, y, z, w] format

    Returns:
        3x3 rotation matrix
    """
    x, y, z, w = q
    # Normalize
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-9:
        return np.eye(3)
    x /= n
    y /= n
    z /= n
    w /= n

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ]
    )
    return R


def rodrigues(axis: np.ndarray, theta: float) -> np.ndarray:
    """
    Rodrigues' rotation formula: rotation matrix from axis-angle

    Args:
        axis: (3,) unit vector axis of rotation
        theta: Rotation angle in radians

    Returns:
        3x3 rotation matrix
    """
    axis = normalize(axis)
    x, y, z = axis
    c = np.cos(theta)
    s = np.sin(theta)
    C = 1.0 - c

    R = np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )
    return R


# ============================================================================
# Section 3: Tilt Trajectory Generation (from tilt.py)
# ============================================================================

def build_initial_camera_frame(
    p: np.ndarray,
    c0: np.ndarray,
    R0: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build initial camera frame that looks at target point p

    Args:
        p: (3,) target point in world frame
        c0: (3,) initial camera position in world frame
        R0: (3,3) initial camera rotation in world frame

    Returns:
        x0, y0, z0: Camera local axes (world frame)
    """
    p = np.asarray(p, dtype=float)
    c0 = np.asarray(c0, dtype=float)

    # Camera z-axis points toward target
    z0 = normalize(p - c0)

    # Try to preserve initial camera y-axis (up direction)
    up_ref = R0[:, 1]

    # Handle parallel case
    if np.abs(np.dot(normalize(up_ref), z0)) > 0.99:
        up_ref = np.array([0.0, 0.0, 1.0], dtype=float)
        if np.abs(np.dot(normalize(up_ref), z0)) > 0.99:
            up_ref = np.array([0.0, 1.0, 0.0], dtype=float)

    # Compute x and y axes
    x0 = normalize(np.cross(up_ref, z0))
    y0 = np.cross(z0, x0)

    return x0, y0, z0


def generate_tilt_trajectory_from_pose(
    cam_pos: np.ndarray,
    cam_quat_xyzw: np.ndarray,
    target_pos: np.ndarray,
    axis_type: str = "roll",
    angle_min_deg: float = -20.0,
    angle_max_deg: float = 20.0,
    num_samples: int = 200,
) -> np.ndarray:
    """
    Generate tilt trajectory by rotating camera around one axis

    The camera rotates around the target point while maintaining view direction.

    Args:
        cam_pos: (3,) initial camera position in world frame
        cam_quat_xyzw: (4,) initial camera quaternion [x, y, z, w]
        target_pos: (3,) target point to look at
        axis_type: "roll" or "pitch"
            - "roll": tilt around camera x-axis
            - "pitch": tilt around camera y-axis
        angle_min_deg: Minimum tilt angle in degrees
        angle_max_deg: Maximum tilt angle in degrees
        num_samples: Number of waypoints to sample

    Returns:
        poses: (N, 4, 4) array of camera poses in world frame
    """
    cam_pos = np.asarray(cam_pos, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)
    R0 = quat_xyzw_to_rot(np.asarray(cam_quat_xyzw, dtype=float))

    # Build initial camera frame
    x0, y0, z0 = build_initial_camera_frame(target_pos, cam_pos, R0)

    # Initial position vector relative to target
    r0 = cam_pos - target_pos

    # Select rotation axis
    if axis_type == "roll":
        axis = x0  # Roll: rotate around x-axis
    elif axis_type == "pitch":
        axis = y0  # Pitch: rotate around y-axis
    else:
        raise ValueError("axis_type must be 'roll' or 'pitch'")

    # Up vector for orientation
    up_vec = y0

    # Generate angle samples
    angles_deg = np.linspace(angle_min_deg, angle_max_deg, num_samples)
    angles_rad = np.deg2rad(angles_deg)

    poses = []

    for theta in angles_rad:
        # 1) Rotate camera position around target
        R_axis = rodrigues(axis, theta)
        r = R_axis @ r0
        c = target_pos + r

        # 2) Compute camera orientation (looking at target)
        z_c = normalize(target_pos - c)

        # Handle parallel case
        if np.abs(np.dot(normalize(up_vec), z_c)) > 0.99:
            alt_up = np.array([0.0, 0.0, 1.0], dtype=float)
            if np.abs(np.dot(normalize(alt_up), z_c)) > 0.99:
                alt_up = np.array([0.0, 1.0, 0.0], dtype=float)
            up = alt_up
        else:
            up = up_vec

        # Compute camera axes
        x_c = normalize(np.cross(up, z_c))
        y_c = np.cross(z_c, x_c)

        R_c = np.column_stack([x_c, y_c, z_c])

        # Build homogeneous transformation
        T = np.eye(4)
        T[:3, :3] = R_c
        T[:3, 3] = c

        poses.append(T)

    poses = np.stack(poses, axis=0)
    return poses


def generate_roll_pitch_tilts(
    cam_pos: np.ndarray,
    cam_quat_xyzw: np.ndarray,
    target_pos: np.ndarray,
    roll_min_deg: float = -20.0,
    roll_max_deg: float = 20.0,
    n_roll: int = 200,
    pitch_min_deg: float = -20.0,
    pitch_max_deg: float = 20.0,
    n_pitch: int = 200,
) -> np.ndarray:
    """
    중심 → 상 → 중심 → 하 → 중심 → 좌 → 중심 → 우 → 중심
    이 순서를 하나의 연속된 궤적으로 생성.

    각 구간은:
      - pitch up:   0 -> +max -> 0
      - pitch down: 0 -> -min -> 0
      - roll left:  0 -> roll_min -> 0
      - roll right: 0 -> roll_max -> 0
    """

    # ---------- helper: 0 -> max -> 0 형태의 "왕복" 시퀀스 ----------
    def make_round_trip(axis_type: str, angle_min: float, angle_max: float, n: int) -> np.ndarray:
        """
        axis_type 축 기준으로 angle_min -> angle_max 를 따라가고
        다시 angle_max -> angle_min 으로 되돌아오는 왕복 궤적을 만든다.
        여기서는 항상 0 -> max -> 0 이라 angle_min=0 을 쓸 것.
        """
        # 가는 쪽 (0 -> max)
        go = generate_tilt_trajectory_from_pose(
            cam_pos, cam_quat_xyzw, target_pos,
            axis_type=axis_type,
            angle_min_deg=angle_min,
            angle_max_deg=angle_max,
            num_samples=n,
        )
        # 오는 쪽 (max -> 0)
        back = generate_tilt_trajectory_from_pose(
            cam_pos, cam_quat_xyzw, target_pos,
            axis_type=axis_type,
            angle_min_deg=angle_max,
            angle_max_deg=angle_min,
            num_samples=n,
        )

        # 마지막 포즈(=max 각도)는 go 끝과 back 시작이 중복이므로 back[1:]만 사용
        return np.concatenate([go, back[1:]], axis=0)

    # pitch: up (0 -> +max -> 0)
    up_cycle = make_round_trip(
        axis_type="pitch",
        angle_min=0.0,
        angle_max=pitch_max_deg,
        n=n_pitch,
    )

    # pitch: down (0 -> -|min| -> 0)
    down_cycle = make_round_trip(
        axis_type="pitch",
        angle_min=0.0,
        angle_max=pitch_min_deg,   # pitch_min_deg 는 음수여도 상관 없음
        n=n_pitch,
    )

    # roll: left (0 -> roll_min -> 0)
    left_cycle = make_round_trip(
        axis_type="roll",
        angle_min=0.0,
        angle_max=roll_min_deg,    # 보통 음수(좌측)
        n=n_roll,
    )

    # roll: right (0 -> roll_max -> 0)
    right_cycle = make_round_trip(
        axis_type="roll",
        angle_min=0.0,
        angle_max=roll_max_deg,    # 양수(우측)
        n=n_roll,
    )

    # up_cycle 끝도 center, down_cycle 시작도 center 이라서
    # 구간 사이의 첫 포즈는 하나씩만 유지하기 위해 [1:]로 시작 포즈만 제거
    poses_all = np.concatenate([
        up_cycle,                 # 중심 → 상 → 중심
        down_cycle[1:],           # (center 공유) → 하 → 중심
        left_cycle[1:],           # (center 공유) → 좌 → 중심
        right_cycle[1:],          # (center 공유) → 우 → 중심
    ], axis=0)

    return poses_all


# ============================================================================
# Section 4: Data I/O
# ============================================================================

def load_pose_from_csv(
    traj_path: str,
    row_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load pose from CSV trajectory file

    Args:
        traj_path: Path to trajectory CSV file
        row_idx: Row index to load (if -1, use middle row)

    Returns:
        position: (3,) position [px, py, pz]
        quaternion: (4,) quaternion [qx, qy, qz, qw] in xyzw format
    """
    with open(traj_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if row_idx < 0:
        row_idx = len(rows) // 2

    row = rows[row_idx]

    # Check column name compatibility
    if "target-POS_X" in row:
        px = float(row["target-POS_X"])
        py = float(row["target-POS_Y"])
        pz = float(row["target-POS_Z"])
        qx = float(row["target-ROT_X"])
        qy = float(row["target-ROT_Y"])
        qz = float(row["target-ROT_Z"])
        qw = float(row["target-ROT_W"])
    else:
        px = float(row["px"])
        py = float(row["py"])
        pz = float(row["pz"])
        qx = float(row["qx"])
        qy = float(row["qy"])
        qz = float(row["qz"])
        qw = float(row["qw"])

    position = np.array([px, py, pz], dtype=np.float64)
    quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)  # xyzw

    return position, quaternion


def save_trajectory_csv(
    path: str,
    trajectory: List[np.ndarray],
    poses_matrix: np.ndarray
):
    """
    Save trajectory to CSV file

    Args:
        path: Output CSV file path
        trajectory: List of (6,) joint configurations
        poses_matrix: (N, 4, 4) array of camera poses
    """
    headers = [
        "time",
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        "target-POS_X", "target-POS_Y", "target-POS_Z",
    ]

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for i, q in enumerate(trajectory):
            pos = poses_matrix[i, :3, 3]
            row = [float(i)] + q.tolist() + pos.tolist()
            w.writerow(row)

    print(f"  ✓ Saved trajectory to {path}")


# ============================================================================
# Section 5: IK Computation (reused from 2_generate_trajectory.py)
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
    Assign EAIK IK solutions to viewpoints

    Args:
        viewpoints: List of Viewpoint objects
        eaik_results: Results from EAIK solver
        indices: Indices of viewpoints that have results
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

        solutions = [np.asarray(q, dtype=np.float64) for q in q_candidates]
        viewpoints[viewpoint_idx].all_ik_solutions = solutions


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

    # Update safe solutions
    for batch_idx, ((vp_idx, _), is_feasible) in enumerate(zip(index_map, feasibility)):
        if not bool(is_feasible):
            continue
        solution = batched_q[batch_idx]
        viewpoints[vp_idx].safe_ik_solutions.append(solution)


# ============================================================================
# Section 6: DP Optimization (from solve_tilt_trajectory_eaik.py)
# ============================================================================

def run_dp_optimization(viewpoints: List[Viewpoint]) -> Optional[List[np.ndarray]]:
    """
    DP (Viterbi) algorithm: find optimal path through safe IK solutions

    Uses L2 norm in joint space as the cost metric.

    Args:
        viewpoints: List of Viewpoint objects with safe_ik_solutions

    Returns:
        optimal_trajectory: List of joint configurations forming optimal path
                          Returns None if no valid path exists
    """
    n_steps = len(viewpoints)
    if n_steps == 0:
        return None

    # Extract safe solutions from viewpoints
    solutions_list = []
    for vp in viewpoints:
        if len(vp.safe_ik_solutions) > 0:
            solutions_list.append(np.array(vp.safe_ik_solutions))
        else:
            solutions_list.append(np.empty((0, 6)))

    # Check first step
    if len(solutions_list[0]) == 0:
        print(f"  ✗ DP failed: Step 0 has no safe IK solutions")
        return None

    # Initialize DP table
    prev_costs = np.zeros(len(solutions_list[0]))
    paths = [[i] for i in range(len(solutions_list[0]))]

    # Forward pass
    for t in range(1, n_steps):
        curr_sols = solutions_list[t]
        prev_sols = solutions_list[t-1]

        if len(curr_sols) == 0:
            print(f"  ✗ DP failed: Step {t} has no safe IK solutions")
            return None

        n_curr = len(curr_sols)

        # Compute distance matrix using broadcasting
        diff = prev_sols[:, np.newaxis, :] - curr_sols[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)  # L2 norm

        # Compute total costs
        total_costs = prev_costs[:, np.newaxis] + dists

        # Find best parent for each current node
        best_prev_indices = np.argmin(total_costs, axis=0)  # (n_curr,)
        curr_costs = total_costs[best_prev_indices, np.arange(n_curr)]

        # Update paths
        new_paths = []
        for j in range(n_curr):
            parent_idx = best_prev_indices[j]
            new_paths.append(paths[parent_idx] + [j])

        prev_costs = curr_costs
        paths = new_paths

    # Backtrack to find optimal path
    best_final_idx = np.argmin(prev_costs)
    best_path_indices = paths[best_final_idx]

    # Build trajectory
    optimal_trajectory = []
    for t, sol_idx in enumerate(best_path_indices):
        optimal_trajectory.append(solutions_list[t][sol_idx])

    return optimal_trajectory


# ============================================================================
# Section 7: Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate Tilt Trajectory with Roll/Pitch Variations"
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
        "--input-file",
        type=str,
        default="trajectory.csv",
        help="Input trajectory filename (default: trajectory.csv)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="tilt_trajectory.csv",
        help="Output trajectory filename (default: tilt_trajectory.csv)"
    )
    parser.add_argument(
        "--row-idx",
        type=int,
        default=10,
        help="Row index to use as center pose (default: 10, -1 for middle)"
    )
    parser.add_argument(
        "--robot",
        type=str,
        default=config.DEFAULT_ROBOT_CONFIG,
        help=f"Robot config file (default: {config.DEFAULT_ROBOT_CONFIG})"
    )

    # Roll tilt parameters
    parser.add_argument("--roll-min", type=float, default=-20.0,
                       help="Minimum roll angle in degrees (default: -20)")
    parser.add_argument("--roll-max", type=float, default=20.0,
                       help="Maximum roll angle in degrees (default: 20)")
    parser.add_argument("--roll-n", type=int, default=200,
                       help="Number of roll samples (default: 200)")

    # Pitch tilt parameters
    parser.add_argument("--pitch-min", type=float, default=-20.0,
                       help="Minimum pitch angle in degrees (default: -20)")
    parser.add_argument("--pitch-max", type=float, default=20.0,
                       help="Maximum pitch angle in degrees (default: 20)")
    parser.add_argument("--pitch-n", type=int, default=200,
                       help="Number of pitch samples (default: 200)")

    args = parser.parse_args()

    # Resolve paths using auto-path generation
    input_path = str(config.get_trajectory_path(args.object, args.num_viewpoints, args.input_file))
    output_path = str(config.get_trajectory_path(args.object, args.num_viewpoints, args.output_file))

    # Start timer
    start_time = time.time()

    print("=" * 60)
    print("GENERATE TILT TRAJECTORY (Step 2-1)")
    print("=" * 60)
    print(f"Object: {args.object}")
    print(f"Num viewpoints: {args.num_viewpoints}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Row index: {args.row_idx}")
    print(f"Roll: [{args.roll_min}°, {args.roll_max}°] × {args.roll_n} samples")
    print(f"Pitch: [{args.pitch_min}°, {args.pitch_max}°] × {args.pitch_n} samples")
    print("=" * 60)
    print()

    # Validate input
    if not os.path.exists(input_path):
        print(f"✗ Error: Input trajectory not found: {input_path}")
        return 1

    # Step 1: Load pose & generate waypoints
    print("[1/5] Loading pose & generating tilt waypoints...")
    cam_pos, cam_quat = load_pose_from_csv(input_path, args.row_idx)

    # Compute target position (surface point in front of camera)
    wd_m = config.CAMERA_WORKING_DISTANCE_MM / 1000.0
    R0 = quat_xyzw_to_rot(cam_quat)
    target_pos = cam_pos + R0[:, 2] * wd_m

    # Generate tilt poses in natural order: center -> up -> center -> down -> center -> left -> center -> right -> center
    full_poses = generate_roll_pitch_tilts(
        cam_pos, cam_quat, target_pos,
        args.roll_min, args.roll_max, args.roll_n,
        args.pitch_min, args.pitch_max, args.pitch_n
    )
    print(f"  ✓ Generated {len(full_poses)} waypoints in natural sequence:")
    print(f"     center -> up -> center -> down -> center -> left -> center -> right -> center")
    print()

    # Step 2: Convert to Viewpoint objects
    viewpoints = []
    for i in range(len(full_poses)):
        vp = Viewpoint(index=i, world_pose=full_poses[i])
        viewpoints.append(vp)

    # Step 3: Compute EAIK (Analytical IK)
    print("[2/5] Computing IK solutions (EAIK)...")
    t0 = time.time()
    ik_results = compute_ik_eaik(full_poses)
    assign_ik_solutions_to_viewpoints(viewpoints, ik_results)

    num_with_ik = sum(1 for vp in viewpoints if len(vp.all_ik_solutions) > 0)
    print(f"  ✓ Computed IK for {len(viewpoints)} poses")
    print(f"  ✓ Found {num_with_ik} viewpoints with valid IK")
    print(f"  ✓ EAIK completed in {time.time()-t0:.2f}s")
    print()

    # Step 4: Setup collision checker
    print("[3/5] Setting up collision world...")

    # Setup world config
    world_cfg = setup_collision_world(
        table=config.TABLE,
        walls=config.WALLS,
        robot_mount=config.ROBOT_MOUNT,
        target_object=config.TARGET_OBJECT,
    )

    # Setup IK solver for collision checking
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]
    tensor_args = TensorDeviceType()

    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args=tensor_args,
        self_collision_check=True,
        collision_checker_type=CollisionCheckerType.MESH,
        use_cuda_graph=True,
    )
    ik_solver = IKSolver(ik_config)
    print(f"  ✓ Collision world ready")
    print()

    # Step 5: Check collisions
    print("[4/5] Checking collisions (CuRobo batch)...")
    t0 = time.time()
    check_ik_solutions_collision(viewpoints, ik_solver)

    num_with_safe = sum(1 for vp in viewpoints if len(vp.safe_ik_solutions) > 0)
    print(f"  ✓ Found {num_with_safe} viewpoints with collision-free IK")
    print(f"  ✓ Collision check completed in {time.time()-t0:.2f}s")
    print()

    # Step 6: DP optimization
    print("[5/5] Optimizing path (DP)...")
    optimal_traj = run_dp_optimization(viewpoints)

    if optimal_traj is None:
        print()
        print("=" * 60)
        print("✗ Failed to find valid trajectory")
        print("=" * 60)
        return 1

    print(f"  ✓ Found optimal path with {len(optimal_traj)} waypoints")
    print()

    # Step 7: Save trajectory
    print("Saving trajectory...")
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    save_trajectory_csv(output_path, optimal_traj, full_poses)
    print()

    elapsed = time.time() - start_time
    print("=" * 60)
    print("✓ Tilt Trajectory Generation Complete!")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Output: {output_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
