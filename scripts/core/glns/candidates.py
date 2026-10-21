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


def pin_base_to_current(representatives, metadata, ref_base_rad, robot_cfg):
    """모든 IK 후보의 base 값을 **현재 base 에 가장 가까운 2π 등가**로 다시 표현한다.

    → (reps, meta, 재표현 수, 버린 수). IK 해는 [-π, π] 로 정규화돼 있어 저장된 숫자는 "얼마나
    돌아야 하는가" 를 말해주지 않는다(현재 -198° 바로 옆 방향이 +177° 로 저장된다). 각 해의
    두 표현(θ, θ±2π) 중 현재 base 에서 180° 안에 있는 쪽을 고르면 어떤 방향이든 가장 짧게 도는
    표현이 되고, 이후 "현재 자세에 맞춰 2π 재표현" 하는 단계가 필요 없다. 그 표현이 관절 한계
    밖이면(현재 base 의 정반대 방향 근처, 한계에 가까울 때만 생기는 좁은 띠) 반대쪽 등가를 쓴다
    — 그 지점만 멀리 돌지만 자세는 잃지 않는다.

    예전의 "관절 하한에 붙은 2π 구간" 고정은 경계(+6°)가 현재 자세와 무관한 곳에 있어서, base 가
    +20° 일 때 방향 0° 근처 지점이 버려지거나 334° 를 도는 표현이 나왔다 — 그래서 바꿨다.
    base 가 주기 관절이 아니면(범위 < 4π) 그대로. 입력은 건드리지 않는다(캐시 파일은 정규화된
    전체 해를 보존해야 로봇이 어디 있든 재사용된다).
    """
    lower, upper, periods = _joint_limits_and_periods(robot_cfg)
    if periods[0] <= 0.0:
        return list(representatives), list(metadata), 0, 0
    period = float(periods[0])
    ref = float(ref_base_rad)
    out_reps, out_meta, moved, dropped = [], [], 0, 0
    for reps, meta in zip(representatives, metadata):
        reps = np.array(reps, dtype=np.float64).reshape(-1, 6)   # 사본
        nearest = ref + (np.mod(reps[:, 0] - ref + period / 2.0, period) - period / 2.0)
        lo, hi = float(lower[0]) - 1e-9, float(upper[0]) + 1e-9
        # 가장 가까운 등가가 한계 밖이면(현재 base 의 정반대 근처) 반대쪽 등가로 간다 — 멀리
        # 돌더라도 자세를 잃는 것보다 낫다. 둘 다 밖이면(±2π 관절에선 없다) 버린다.
        other = np.where(nearest < lo, nearest + period, nearest - period)
        use_other = (nearest < lo) | (nearest > hi)
        chosen = np.where(use_other, other, nearest)
        keep = (chosen >= lo) & (chosen <= hi)
        moved += int(np.sum(np.abs(chosen[keep] - reps[keep, 0]) > 1e-9))
        dropped += int((~keep).sum())
        reps[:, 0] = chosen
        out_reps.append(reps[keep])
        out_meta.append({k: (np.asarray(v)[keep] if len(v) == len(keep) else v)
                         for k, v in meta.items()})
    return out_reps, out_meta, moved, dropped


def keep_front_reach(representatives, metadata, max_deg):
    """base 가 **그 지점 쪽을 보는** IK 해만 남긴다 → (reps, meta, 버린 수, 도달성을 잃은 viewpoint 수).

    UR 은 손목 위치 하나에 base 해가 둘이다: 지점 방향(atan2(y, x))을 보는 분기와, 반대쪽을
    보고 팔을 베이스 위로 넘겨 닿는 분기(base 가 방향에서 150° 이상 벗어남; 실측 sample/75 는
    두 무리가 30° 와 150° 로 갈려 중간이 비어 있다). 뒤로 넘기는 분기가 하나라도 뽑히면 base 가
    반대쪽으로 돌아가므로 후보에서 뺀다. 기준은 현재 base 가 아니라 **지점마다 자기 방향**이다 —
    물체가 로봇을 둘러싸고 있어도 각 지점은 자기 쪽을 보는 해를 갖는다. 어깨 오프셋 때문에
    정면 분기도 방향과 10~25° 어긋나므로 max_deg 는 90° 가 적당하다.
    """
    max_rad = np.deg2rad(float(max_deg))
    out_reps, out_meta, removed, lost = [], [], 0, 0
    for reps, meta in zip(representatives, metadata):
        reps = np.asarray(reps, dtype=np.float64).reshape(-1, 6)
        pos = np.asarray(meta["target_position"], dtype=np.float64).reshape(-1, 3)
        yaw = np.arctan2(pos[:, 1], pos[:, 0])
        diff = np.mod(reps[:, 0] - yaw + np.pi, 2.0 * np.pi) - np.pi
        keep = np.abs(diff) <= max_rad
        removed += int((~keep).sum())
        if len(reps) and not keep.any():
            lost += 1
        out_reps.append(reps[keep])
        out_meta.append({k: (np.asarray(v)[keep] if len(v) == len(keep) else v)
                         for k, v in meta.items()})
    return out_reps, out_meta, removed, lost


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
