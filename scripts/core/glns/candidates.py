"""GLNS pose augmentation, IK candidate generation, and branch repair."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from core import trajectory as PT
from .problem import periodic_joint_delta

def _collision_filter_representatives(representatives, robot_cfg, world, metadata=None):
    offsets = []
    flat = []
    cursor = 0
    for viewpoint, reps in enumerate(representatives):
        if len(reps):
            offsets.append((viewpoint, cursor, len(reps)))
            flat.append(reps)
            cursor += len(reps)
    if not flat:
        return 0
    colliding, _ = PT.batch_collision_check(np.concatenate(flat, axis=0), robot_cfg, world)
    removed = 0
    for viewpoint, start, count in offsets:
        free = ~colliding[start:start + count]
        removed += count - int(free.sum())
        representatives[viewpoint] = representatives[viewpoint][free]
        if metadata is not None:
            metadata[viewpoint] = {
                key: np.asarray(value)[free] for key, value in metadata[viewpoint].items()
            }
    return removed
def _joint_limits_and_periods(robot_cfg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read active-joint limits and mark joints with at least two full turns as periodic."""
    kin = robot_cfg["robot_cfg"]["kinematics"]
    names = list(kin["cspace"]["joint_names"])
    limit_clip = float(kin["cspace"].get("position_limit_clip", 0.0))
    root = ET.parse(kin["urdf_path"]).getroot()
    joints = {node.attrib.get("name"): node for node in root.findall("joint")}
    lower, upper, periods = [], [], []
    for name in names:
        node = joints.get(name)
        limit = None if node is None else node.find("limit")
        if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
            raise ValueError(f"URDF joint limits missing for {name}")
        lo, hi = float(limit.attrib["lower"]), float(limit.attrib["upper"])
        lower.append(lo + limit_clip)
        upper.append(hi - limit_clip)
        # A ±π joint cannot cross its endpoint even though its pose is angularly
        # equivalent. Require a full extra revolution of available range.
        periods.append(2.0 * np.pi if hi - lo >= 4.0 * np.pi - 1e-6 else 0.0)
    return np.asarray(lower), np.asarray(upper), np.asarray(periods)


def _build_pose_variants(world_poses, wd_m, *, roll_augment=False, roll_step_deg=30.0,
                         tilt_augment=False, tilt_angles_deg=(5.0, 10.0),
                         tilt_azimuths=8):
    """Return flattened nominal + roll + tilt pose targets and their metadata."""
    records = []
    roll_angles = np.arange(roll_step_deg, 360.0 - 1e-9, roll_step_deg)
    azimuths = np.linspace(0.0, 360.0, int(tilt_azimuths), endpoint=False)
    for vp, pose in enumerate(np.asarray(world_poses, dtype=np.float64)):
        R, p = pose[:3, :3], pose[:3, 3]
        surf = p + R[:, 2] * wd_m
        records.append((vp, "nominal", 0.0, 0.0, np.nan, p.copy(), R.copy()))
        if roll_augment:
            for angle in roll_angles:
                Rp = R @ Rotation.from_euler("z", angle, degrees=True).as_matrix()
                records.append((vp, "roll", float(angle), 0.0, np.nan, p.copy(), Rp))
        if tilt_augment:
            for tilt in tilt_angles_deg:
                for az in azimuths:
                    axis = R @ np.array([np.cos(np.deg2rad(az)), np.sin(np.deg2rad(az)), 0.0])
                    Rp = Rotation.from_rotvec(axis * np.deg2rad(tilt)).as_matrix() @ R
                    pp = surf - Rp[:, 2] * wd_m
                    records.append((vp, "tilt", 0.0, float(tilt), float(az), pp, Rp))
    positions = np.stack([r[5] for r in records])
    rotations = np.stack([r[6] for r in records])
    return {
        "viewpoint": np.asarray([r[0] for r in records], dtype=np.int32),
        "variant": np.asarray([r[1] for r in records], dtype="U16"),
        "roll_deg": np.asarray([r[2] for r in records], dtype=np.float64),
        "tilt_deg": np.asarray([r[3] for r in records], dtype=np.float64),
        "tilt_azimuth_deg": np.asarray([r[4] for r in records], dtype=np.float64),
        "position": positions,
        "quaternion": PT.rot_to_quat_batch(rotations),
        "rotation": rotations,
    }


def _solve_pose_variant_candidates(targets, n_viewpoints, world, robot_cfg,
                                   num_seeds, batch_size, wrist3_fixed,
                                   lock_nominal_wrist3, joint_periods=None,
                                   ik_seed=PT.IK_RANDOM_SEED, dedup_rad=None):
    # dedup_rad: L∞ 관절 임계값(rad). None 이면 기본 CANDIDATE_DEDUP_RAD, 음수면 dedup 끔
    # (어떤 후보도 서로 임계 이내로 판정되지 않아 전부 남는다).
    thr = PT.CANDIDATE_DEDUP_RAD if dedup_rad is None else float(dedup_rad)
    sols, success = PT.solve_ik_multi_seed(
        robot_cfg, world, targets["position"], targets["quaternion"],
        num_seeds=num_seeds, batch_size=batch_size, random_seed=ik_seed,
    )
    representatives, metadata = [], []
    for vp in range(n_viewpoints):
        kept, fields = [], {key: [] for key in (
            "variant", "roll_deg", "tilt_deg", "tilt_azimuth_deg",
            "target_position", "target_quaternion",
        )}
        for pose_idx in np.flatnonzero(targets["viewpoint"] == vp):
            for q in sols[pose_idx][success[pose_idx]]:
                q = np.asarray(q, dtype=np.float64).copy()
                if lock_nominal_wrist3:
                    q[-1] = wrist3_fixed
                if any(np.max(np.abs(periodic_joint_delta(q - prior, joint_periods)))
                       <= thr for prior in kept):
                    continue
                kept.append(q)
                fields["variant"].append(targets["variant"][pose_idx])
                fields["roll_deg"].append(targets["roll_deg"][pose_idx])
                fields["tilt_deg"].append(targets["tilt_deg"][pose_idx])
                fields["tilt_azimuth_deg"].append(targets["tilt_azimuth_deg"][pose_idx])
                fields["target_position"].append(targets["position"][pose_idx])
                fields["target_quaternion"].append(targets["quaternion"][pose_idx])
        representatives.append(np.asarray(kept, dtype=np.float64).reshape(-1, 6))
        metadata.append({
            key: np.asarray(value, dtype=("U16" if key == "variant" else np.float64))
            for key, value in fields.items()
        })
    return representatives, metadata


# ============================================================================
# Post-GLNS branch repair (tilt outliers that force a catastrophic base reconfig)
# ============================================================================

_BASE_IDX = np.array([0, 1, 2])


def _tilt_cone_poses(world_pose, wd_m, tilt_deg, n_azimuth):
    """표면점 중심 orbit tilt 포즈들(광축 ±tilt_deg, n_azimuth 방위) → (positions, quats wxyz).

    plan_trajectory via-tilt 와 동일한 orbit: WD·표면점 유지, 광축만 기울여 새 팔 분기를 연다.
    (roll 은 wrist_3-decoupled 라 분기 불변 — tilt 만 분기를 바꾼다.)
    """
    R = np.asarray(world_pose[:3, :3], dtype=np.float64)
    p = np.asarray(world_pose[:3, 3], dtype=np.float64)
    surf = p + R[:, 2] * wd_m
    Rs, ps = [], []
    for az in np.linspace(0.0, 360.0, int(n_azimuth), endpoint=False):
        u = R @ np.array([np.cos(np.deg2rad(az)), np.sin(np.deg2rad(az)), 0.0])
        u = u / np.linalg.norm(u)
        Rp = Rotation.from_rotvec(u * np.deg2rad(tilt_deg)).as_matrix() @ R
        Rs.append(Rp)
        ps.append(surf - Rp[:, 2] * wd_m)
    return np.stack(ps), PT.rot_to_quat_batch(np.stack(Rs))


def _tilt_compatible_solution(viewpoint, neighbor_configs, *, world_poses, world, robot_cfg,
                              wd_m, wrist3_fixed, num_seeds, batch_size,
                              big_base_rad, tilt_magnitudes_deg, n_azimuth,
                              ik_seed=PT.IK_RANDOM_SEED):
    """outlier viewpoint 를 ±tilt 로 재-IK 해 이웃과 big-base reconfig 없는 collision-free 해 탐색.

    작은 tilt 부터 시도하고, base(어깨/팔꿈치) L∞ 가 모든 이웃에 대해 big_base_rad 미만인 해 중
    base 변화가 가장 작은 것을 고른다. 찾으면 (config (6,), used_tilt_deg), 없으면 (None, None).
    """
    for tilt_deg in tilt_magnitudes_deg:
        ps, quats = _tilt_cone_poses(world_poses[viewpoint], wd_m, tilt_deg, n_azimuth)
        sols, succ = PT.solve_ik_multi_seed(
            robot_cfg, world, ps, quats, num_seeds=num_seeds, batch_size=batch_size,
            random_seed=ik_seed)
        dof = sols.shape[2]
        reps = PT.cluster_ik_solutions(sols.reshape(1, -1, dof), succ.reshape(1, -1))[0]
        if not len(reps):
            continue
        reps = np.asarray(reps, dtype=np.float64)
        reps[:, -1] = wrist3_fixed                         # wrist_3 lock(스캔 컨벤션 동일)
        colliding, _ = PT.batch_collision_check(reps, robot_cfg, world)
        reps = reps[~np.asarray(colliding, dtype=bool)]
        best, best_worst = None, np.inf
        for r in reps:
            worst = max((float(np.max(np.abs(r[_BASE_IDX] - np.asarray(nb)[_BASE_IDX])))
                         for nb in neighbor_configs), default=0.0)
            if worst < big_base_rad and worst < best_worst:
                best, best_worst = r, worst
        if best is not None:
            return best, float(tilt_deg)
    return None, None
