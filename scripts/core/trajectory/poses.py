"""Camera-pose construction helpers."""

import numpy as np
from scipy.spatial.transform import Rotation

from common import config
from common.math_utils import normalize_vectors, quaternion_to_rotation_matrix

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
