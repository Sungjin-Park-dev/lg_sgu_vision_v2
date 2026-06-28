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

# (b) DP 후보 풍부화: DBSCAN medoid(클러스터당 1개)는 이웃과 연속인 해를 버릴 수 있다.
# enrich=True면 성공 IK 해 전체를 fine tolerance로 near-duplicate만 제거하고 모두 후보로
# 남겨 DP가 연속 분기를 직접 고르게 한다.
ENRICH_DP_CANDIDATES = True
DP_CANDIDATE_DEDUP_RAD = 0.08   # ~4.6°, 분기는 보존하며 거의 동일한 seed 만 제거

# (a) 연속성 IK seeding: reconfig 양끝/empty viewpoint를 이웃의 선택 자세로 warm-start하여
# IK 재시도 → random seed가 못 찾은 '이웃과 연속인' 분기를 찾거나 empty를 채운다.
# curved_structure 실측상 효과 0(empty 0개 채움, reconfig 불변)이라 기본 OFF. 다른 물체에서
# '연속 분기가 존재하는데 random seed가 놓친' 경우 도움될 수 있어 토글로 남겨둠.
ENABLE_CONTINUITY_SEEDING = False

# IK 해가 전혀 없는(empty) viewpoint를 carry-forward(이웃 자세 복제)로 메우지 않고 경로에서
# 제거한다. 복제는 그 표면점을 실제로 검사하지 않는 '가짜 커버리지'이므로, 도달 가능한
# viewpoint만 남겨 정직하게 검사한다.
DROP_UNREACHABLE_VIEWPOINTS = True

# DP는 충돌을 보지 않고 joint 비용만으로 대표 해를 고른다. 그런데 IK의 world(물체) 충돌은
# soft cost(activation 10mm)라 물체를 살짝 파고드는 해도 success로 돌아온다 → 한 viewpoint에
# 충돌/비충돌 분기가 섞여 있으면 DP가 더 싼 '충돌' 분기를 골라 최종 충돌검사에서 거부된다
# (sample/74에서 발생). 그래서 DP 이전에, 최종검사와 '동일한' hard 충돌검사로 각 viewpoint의
# 대표 해에서 충돌 자세를 제거한다. wrist_3 잠금 뒤에 검사해 실제 최종 자세와 일치시키고,
# 충돌-free 해가 0개가 된 viewpoint는 empty와 똑같이 unreachable로 드롭한다.
FILTER_COLLIDING_REPS = True

# Reconfig transit(충돌회피 joint-to-joint) 계획 강건화 (#1)
# plan_cspace는 timeout이 없고 max_attempts 회 재시도 후 실패하는 단순 루프라(성공 시 즉시 break),
# '실패 판정에 걸리는 시간'이 max_attempts에 거의 선형 비례한다(이 하드웨어에서 ~0.33s/attempt).
# 성공은 attempt 0~1(~0.37s)에 끝나므로 max_attempts를 줄여도 성공은 거의 영향 없고 실패 대기만 짧아진다.
# 15→8: 실패 대기 ~5s→~2.7s. 단 attempt 후반에야 풀리던 경계 transit 일부를 잃을 수 있음(→ 정직한 run-split).
TRANSIT_MAX_ATTEMPTS = 8            # plan_cspace 재시도 횟수 (cuRobo 기본 5). 실패 대기시간 ∝ 이 값
TRANSIT_ENABLE_GRAPH_ATTEMPT = 1    # 이 시도부터 PRM graph seeding 사용 (0=처음부터)

# transit이 끝내 실패한 reconfig를 직선 보간으로 메우면 카메라/팔이 물체를 관통한다(#2).
# 대신 아웃라이어 viewpoint를 최대 이 개수까지 건너뛰어(skip), 시작 자세와 다시
# reconfig_threshold 안에서 만나는 viewpoint에 재연결한다. 그 안에 재연결이 안 되면
# (클러스터 경계 등) 안전한 연속 경로가 없으므로 명확히 에러를 낸다.
TRANSIT_FAIL_SKIP_MAX = 5

# 직접(start→goal) transit 이 실패하면 안전한 HOME(=config.ROBOT_START_STATE)을 경유하는
# 2-leg(start→HOME, HOME→goal)로 재시도한다. 직접 큰 재구성(예: 정면→측면 78°)은 물체 옆 좁은
# 공간을 카메라가 쓸며 지나가야 해 자주 실패하지만, 각 leg는 '표면→자유공간(retract)'과
# '자유공간→표면(re-approach)' 문제라 훨씬 쉽고 빠르다(curved 47→48 실측: 직접 0/8 → via-home 8/8).
# 직접이 성공하면 미사용(fallback). 비용은 retract+reapproach 이동(joint-timed)만 추가되며 reconfig
# 는 궤적당 몇 개뿐이라, 못 가던 viewpoint(예: 측면 chunk)를 살리는 값으로 충분히 싸다.
TRANSIT_VIA_HOME = True

# via-roll: 직접/via-home 실패 시, 경계의 양 끝 scan 자세를 광축(camera 광축 ≈ wrist_3 축) 둘레로
# roll 한 '중간자세'를 경유해 direct 가교한다. wrist_3 lock 이 버린 redundant DOF(광축 roll =
# 검사 무손실)를 경계 transit 에서만 복원하는 것 — scan config(selected[i]) 는 보존하므로 검사
# 품질·스캔 일관성에 영향 없다(중간자세는 스캔되지 않음). 측정상 transit-fail 5/5 해소
# (scratchpad/freedom_bridge_probe.py: curved 좋은배치 5개 실패 경계 전부 roll 단독 가교).
# via-home detour 보다 우선(빠름), via-home 은 최후 fallback 으로 강등.
VIA_ROLL_ENABLE = True
ROLL_VARIANT_DEG = (45, 90, 135, 180, 225, 270, 315)   # 광축 둘레 roll 변형 각도
VIA_ROLL_IK_SEEDS = 40            # 변형 타깃당 IK seed 수 (collision-free 분기 확보)
VIA_ROLL_MAX_REPS = 6             # endpoint 당 변형 후보 상한(가까운 가교쌍이 cap 안에 들도록)
VIA_ROLL_PAIR_CAP = 20            # 경계당 (A',B') 후보쌍 MotionGen 시도 상한(가교쌍이 joint-far 일 수 있음)

# via-tilt: roll 로도 못 푼 경계를 표면점 중심 orbit tilt(광축 ±φ)로 escalation. 중간자세는
# 스캔되지 않으므로 시야 비스듬함은 무비용. roll 다음, via-home 앞. (wd_m 필요)
VIA_TILT_ENABLE = True
TILT_VARIANT_PHI = (15, 30)       # 광축 기울임 각도(도)
TILT_VARIANT_AZ = (0, 90, 180, 270)

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

# reconfig transit(재배치)은 검사 스캔이 아니라 단순 repositioning이다. 스캔과 똑같이 EE
# arc-length로 resample하고 EE 선속도(50mm/s)로 시간 매기면, base를 크게 돌릴 때 팔 끝이
# 자유공간에 그리는 긴 호(수 m)를 기어가느라 사이클의 대부분을 먹는다(curved/100에서 139°
# transit 1개가 5.25m 호 → 525 waypoint → 105s = 전체의 63%). 그래서 transit 구간은
# (1) joint-space L∞로 sparse하게 resample하고, (2) joint 속도 한계로만 시간을 매긴다
# (EE 선속도/각속도/corner slowdown 무시). 139° transit이 ~8s로 줄어든다.
TRANSIT_RESAMPLE_SPACING_RAD = 0.05   # ~2.9°, transit resample 간격(가장 빨리 도는 joint 기준)

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


def solve_ik_seeded(robot_cfg, world_scene, positions_np, quats_np,
                    target_idxs, anchors, num_seeds=NUM_IK_SEEDS):
    """target viewpoint들을 이웃의 선택 자세(anchor)로 warm-start하여 IK 재시도(#a).

    각 target은 batch item 으로, seed_config = anchor 를 num_seeds 개 복제해 강하게 편향.
    random multi-seed가 못 찾은 '이웃과 연속인' 분기를 찾거나 empty viewpoint를 채운다.

    Args:
        target_idxs: list[int] — 재시도할 viewpoint 인덱스
        anchors: (B, 6) — 각 target의 seed가 될 이웃 선택 자세

    Returns:
        sol: (B, num_seeds, 6) normalize된 IK 해
        success: (B, num_seeds) bool
    """
    B = len(target_idxs)
    cache = {
        "obb": max(1, len(world_scene.cuboid)),
        "mesh": max(1, len(world_scene.mesh)),
    }
    cfg = InverseKinematicsCfg.create(
        robot=robot_cfg,
        scene_model={},
        self_collision_check=True,
        num_seeds=num_seeds,
        max_batch_size=B,
        use_cuda_graph=False,
        collision_cache=cache,
    )
    ik = InverseKinematics(cfg)
    ik.update_world(world_scene)
    tool = ik.tool_frames[0]

    bp = torch.tensor(positions_np[target_idxs], device="cuda:0", dtype=torch.float32)
    bq = torch.tensor(quats_np[target_idxs], device="cuda:0", dtype=torch.float32)
    goal = Pose(position=bp, quaternion=bq)

    seed = torch.tensor(np.asarray(anchors), device="cuda:0", dtype=torch.float32)
    seed = seed.unsqueeze(1).repeat(1, num_seeds, 1)  # (B, num_seeds, 6)

    result = ik.solve_pose(
        GoalToolPose.from_poses({tool: goal}, num_goalset=1),
        seed_config=seed,
        return_seeds=num_seeds,
    )

    sol = result.js_solution.position.cpu().numpy()
    if sol.shape[-1] != 6:
        sol = sol[..., :6]
    success = result.success.cpu().numpy()
    return normalize_joints(sol), success


# =========================================================================
# Phase 2: DBSCAN clustering per viewpoint
# =========================================================================

def _greedy_dedup(sols, tol_rad):
    """L∞ 기준 greedy near-duplicate 제거. tol 안에 들어오는 해는 하나만 남긴다.

    클러스터링과 달리 medoid 로 합치지 않아 서로 다른 분기(branch)와 분기 내 다수 해를
    모두 보존한다 → DP 가 이웃과 연속인 해를 직접 고를 수 있다.
    """
    kept = []
    for s in sols:
        if all(np.max(np.abs(s - k)) > tol_rad for k in kept):
            kept.append(s)
    return np.array(kept)


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

        if ENRICH_DP_CANDIDATES:
            # (b) near-duplicate 만 제거하고 모든 분기를 후보로 유지
            reps = _greedy_dedup(successful, DP_CANDIDATE_DEDUP_RAD)
        else:
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
            reps = np.array(medoids)

        representatives.append(reps)
        total_reps += len(reps)

    avg_reps = total_reps / max(N - empty_count, 1)
    mode_desc = (f"enrich dedup={DP_CANDIDATE_DEDUP_RAD:.2f} rad"
                 if ENRICH_DP_CANDIDATES else f"DBSCAN medoid eps={eps:.2f} rad")
    print(f"  Phase 2 done: {N} viewpoints → avg {avg_reps:.1f} representatives/viewpoint "
          f"({mode_desc}, {empty_count} empty)")

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
    selected, reconfig_indices, robot_cfg, world_scene, label_idx=None, wd_m=None,
):
    """Reconfig 지점마다 MotionPlanner joint-to-joint planning 수행.

    사다리: direct → via-roll → via-tilt → via-home. via-roll/via-tilt 는 경계 양 끝 scan
    자세를 광축 둘레로 roll/tilt 한 중간자세를 경유해 가교한다(scan config 보존, ripple 없음).

    Args:
        selected: (N, 6) DP로 선택된 joint trajectory
        reconfig_indices: reconfig이 발생하는 transition 인덱스 배열
        robot_cfg: cuRobo robot config (dict)
        world_scene: Scene (충돌 세계)
        label_idx: (선택) filtered→원본 viewpoint 인덱스 매핑. 로그 표기용.
        wd_m: (선택) working distance [m]. via-tilt(orbit) 에만 필요.

    Returns:
        transit_segments: dict {idx: (T, 6) transit trajectory} — 성공한 것만
        transit_stats: list of dicts
    """
    def _lbl(i):
        return int(label_idx[i]) if label_idx is not None else int(i)
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
    # enable_graph=True 로 PRM roadmap 까지 워밍업한다. reconfig transit 은 자세를 크게
    # 뒤집는 hard 케이스라 trajopt 직선 시드만으로는 자주 실패 → graph seeding 이 핵심.
    print("    Warming up MotionPlanner (with graph planner)...")
    planner.warmup(enable_graph=True, num_warmup_iterations=2)

    n_dof = selected.shape[-1]

    def _plan_leg(q_from, q_to, max_attempts=TRANSIT_MAX_ATTEMPTS,
                  graph_attempt=TRANSIT_ENABLE_GRAPH_ATTEMPT):
        """단일 plan_cspace. 성공 시 waypoints (T,6), 실패/퇴화 시 None."""
        s = JointState.from_position(
            torch.tensor(q_from, device="cuda:0", dtype=torch.float32).unsqueeze(0),
            joint_names=planner.joint_names)
        g = JointState.from_position(
            torch.tensor(q_to, device="cuda:0", dtype=torch.float32).unsqueeze(0),
            joint_names=planner.joint_names)
        r = planner.plan_cspace(
            g, s, max_attempts=max_attempts, enable_graph_attempt=graph_attempt,
        )
        if r is None or not bool(r.success.any().item()):
            return None
        wp = _single_joint_state_path(r.get_interpolated_plan(), n_dof)
        return wp if len(wp) >= 2 else None

    # HOME: 직접 transit 실패 시 경유할 안전 retract 자세. wrist_3 가 이미 lock 값과 동일.
    home_q = np.asarray(config.ROBOT_START_STATE, dtype=np.float64)

    # ----- via-roll/via-tilt 인프라 (lazy: 직접 transit 실패가 생겨 처음 필요할 때만 빌드) -----
    # 카메라 광축 ≈ wrist_3 축 → wrist_3 = 광축 roll = 검사 무손실 redundant DOF. scan config
    # (selected[i]) 는 그대로 두고 transit 중간자세에만 roll/tilt 를 줘 direct 가교를 푼다.
    # 중간자세는 스캔되지 않으므로 시야 손실 없음. (compute_fk 로 endpoint 광축을 얻어 roll/tilt)
    _via = {"ready": False, "ik": None, "tool": None, "pose_of": None}
    _vcache = {}                       # (sel_idx, mode) -> [collision-free config, ...]

    def _ensure_via():
        if _via["ready"]:
            return
        ep_set = sorted(set([int(i) for i in reconfig_indices]
                            + [int(i) + 1 for i in reconfig_indices]))
        ee_pos, ee_quat = compute_fk(selected[ep_set], robot_cfg)   # (M,3), (M,4 xyzw)
        _via["pose_of"] = {k: (Rotation.from_quat(ee_quat[j]).as_matrix(), ee_pos[j].copy())
                           for j, k in enumerate(ep_set)}
        ik_cache = {"obb": max(1, len(world_scene.cuboid)), "mesh": max(1, len(world_scene.mesh))}
        ik_cfg = InverseKinematicsCfg.create(
            robot=robot_cfg, scene_model={}, self_collision_check=True,
            num_seeds=VIA_ROLL_IK_SEEDS,
            max_batch_size=max(len(ROLL_VARIANT_DEG), len(TILT_VARIANT_PHI) * len(TILT_VARIANT_AZ)),
            use_cuda_graph=False, collision_cache=ik_cache)
        ik = InverseKinematics(ik_cfg)
        ik.update_world(world_scene)
        _via["ik"], _via["tool"] = ik, ik.tool_frames[0]
        _via["ready"] = True

    def _roll_targets(R, p):
        return [(R @ Rotation.from_euler("z", deg, degrees=True).as_matrix(), p.copy())
                for deg in ROLL_VARIANT_DEG]

    def _tilt_targets(R, p):
        z = R[:, 2]
        surf = p + z * wd_m                                    # 표면점(orbit 중심)
        out = []
        for phi in TILT_VARIANT_PHI:
            for az in TILT_VARIANT_AZ:
                u = R @ np.array([np.cos(np.deg2rad(az)), np.sin(np.deg2rad(az)), 0.0])
                u = u / np.linalg.norm(u)
                Rp = Rotation.from_rotvec(u * np.deg2rad(phi)).as_matrix() @ R
                out.append((Rp, surf - Rp[:, 2] * wd_m))       # WD 유지하며 광축 ±φ
        return out

    def _variant_configs(k, mode):
        """sel index k 의 변형 IK 후보(collision-free, dedup). 캐시."""
        if (k, mode) in _vcache:
            return _vcache[(k, mode)]
        _ensure_via()
        R, p = _via["pose_of"][k]
        targets = _roll_targets(R, p) if mode == "roll" else _tilt_targets(R, p)
        Rs = np.stack([t[0] for t in targets])
        ps = np.stack([t[1] for t in targets])
        bp = torch.tensor(ps, device="cuda:0", dtype=torch.float32)
        bq = torch.tensor(rot_to_quat_batch(Rs), device="cuda:0", dtype=torch.float32)
        res = _via["ik"].solve_pose(
            GoalToolPose.from_poses({_via["tool"]: Pose(position=bp, quaternion=bq)}, num_goalset=1),
            return_seeds=VIA_ROLL_IK_SEEDS)
        sols = res.js_solution.position.detach().cpu().numpy()
        if sols.shape[-1] != n_dof:
            sols = sols[..., :n_dof]
        sols = normalize_joints(sols)
        flat = sols[res.success.detach().cpu().numpy()]        # (K,6) 성공 해만
        kept = []
        if len(flat):
            isc, _ = batch_collision_check(flat, robot_cfg, world_scene)
            tol = np.deg2rad(8.0)
            for s in flat[~isc]:
                if all(np.max(np.abs(s - q)) > tol for q in kept):
                    kept.append(s)
                if len(kept) >= VIA_ROLL_MAX_REPS:
                    break
        _vcache[(k, mode)] = kept
        return kept

    def _via_variant(idx, mode):
        """scan config 고정 채 변형 중간자세 경유 3-leg transit. 성공 시 waypoints, 아니면 None."""
        qi, qj = selected[idx], selected[idx + 1]
        poolA = [qi] + _variant_configs(idx, mode)
        poolB = [qj] + _variant_configs(idx + 1, mode)
        pairs = sorted(((float(np.max(np.abs(poolA[ia] - poolB[ib]))), ia, ib)
                        for ia in range(len(poolA)) for ib in range(len(poolB))),
                       key=lambda t: t[0])
        for n, (_, ia, ib) in enumerate(pairs):
            if n >= VIA_ROLL_PAIR_CAP:
                break
            a, b = poolA[ia], poolB[ib]
            leg1 = _plan_leg(qi, a) if ia != 0 else None       # scan→A' (roll, 보통 쉬움)
            if ia != 0 and leg1 is None:
                continue
            legm = _plan_leg(a, b)                              # A'→B' (가교부)
            if legm is None:
                continue
            leg3 = _plan_leg(b, qj) if ib != 0 else None        # B'→scan (roll)
            if ib != 0 and leg3 is None:
                continue
            segs = [leg1, legm[1:]] if ia != 0 else [legm]      # 중복 endpoint 제거
            if ib != 0:
                segs.append(leg3[1:])
            wp = np.concatenate(segs, axis=0)
            return wp if len(wp) >= 2 else None
        return None

    transit_segments = {}
    transit_stats = []

    for idx in reconfig_indices:
        t0 = time.time()
        waypoints = _plan_leg(selected[idx], selected[idx + 1])    # 1) 직접
        route = "direct"

        if waypoints is None and VIA_ROLL_ENABLE:                  # 2) 광축 roll 중간자세 경유
            waypoints = _via_variant(idx, "roll")
            if waypoints is not None:
                route = "via-roll"

        if waypoints is None and VIA_TILT_ENABLE and wd_m is not None:  # 3) tilt 중간자세 경유 (escalation)
            waypoints = _via_variant(idx, "tilt")
            if waypoints is not None:
                route = "via-tilt"

        if waypoints is None and TRANSIT_VIA_HOME:                  # 4) HOME 경유 (최후 fallback)
            leg1 = _plan_leg(selected[idx], home_q)
            leg2 = _plan_leg(home_q, selected[idx + 1])
            if leg1 is not None and leg2 is not None:
                waypoints = np.concatenate([leg1, leg2[1:]], axis=0)  # HOME 중복 제거
                route = "via-home"

        dt = time.time() - t0

        if waypoints is not None:
            transit_segments[idx] = waypoints
            max_step_deg = np.rad2deg(
                np.max(np.abs(np.diff(waypoints, axis=0)))
            ) if len(waypoints) > 1 else 0.0
            transit_stats.append({
                "idx": idx, "success": True, "route": route,
                "n_waypoints": len(waypoints), "time": dt,
                "max_step_deg": float(max_step_deg),
            })
            print(
                f"    {_lbl(idx)}→{_lbl(idx+1)}: OK [{route}] ({len(waypoints)} waypoints, "
                f"max_step={max_step_deg:.2f}°, {dt:.2f}s)"
            )
        else:
            transit_stats.append({
                "idx": idx, "success": False, "time": dt,
            })
            print(f"    {_lbl(idx)}→{_lbl(idx+1)}: FAILED [genuinely-unbridgeable] ({dt:.2f}s)")

    n_ok = sum(1 for s in transit_stats if s["success"])
    by_route = []
    for nm in ("via-roll", "via-tilt", "via-home"):
        c = sum(1 for s in transit_stats if s.get("route") == nm)
        if c:
            by_route.append(f"{c} {nm}")
    print(f"  Transit planning: {n_ok}/{len(reconfig_indices)} succeeded"
          + (f" ({', '.join(by_route)})" if by_route else ""))

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


def _linear_interior(q0, q1, dense_step_rad):
    """q0→q1 joint-space 직선 보간의 내부점(양 끝 제외). 없으면 None."""
    dist = float(np.max(np.abs(q1 - q0)))
    n_steps = max(1, int(np.ceil(dist / dense_step_rad)))
    if n_steps <= 1:
        return None
    alphas = np.linspace(0.0, 1.0, n_steps + 1)[1:-1]
    return q0[np.newaxis, :] + alphas[:, np.newaxis] * (q1 - q0)[np.newaxis, :]


def _build_runs(selected, transit_segments, reconfig_threshold_rad, max_skip,
                scan_free=None):
    """viewpoint 시퀀스를 '안전하게 연속인' run 들로 분할한다(#2).

    run 내부 연결 규칙(두 viewpoint를 같은 run으로 이으려면 충돌-free 이동이 있어야 함):
        - 성공한 transit (consecutive, idx in transit_segments) → MotionGen 충돌-free → 유지
        - small jump (L∞ <= thr) **이고** 그 직선 스캔이 충돌-free → 유지
        - 위로 못 이으면 아웃라이어 viewpoint(i+1..j-1)를 최대 max_skip개까지 건너뛰어
          selected[i]와 thr 이내 **이고** 직선 스캔이 충돌-free인 j 에 재연결(run 유지)
    어떤 것으로도 못 이으면(클러스터 경계 / 스캔 관통 등) 그 지점에서 run 을 끊는다(cut).

    scan_free(a, b): 직선 joint 스캔 selected[a]→selected[b] 가 충돌-free인지 (None이면 항상 True,
    하위호환). small jump 이라도 곡면 측면에서 직선 보간이 물체를 관통할 수 있어 검사가 필요하다.

    Returns:
        runs: list[list[int]] — 각 run 의 viewpoint 인덱스(원본 순서)
        skipped: set[int] — run 내부에서 건너뛴 아웃라이어 인덱스
    """
    def _scan_ok(a, b):
        return scan_free is None or scan_free(a, b)

    N = len(selected)
    runs: list[list[int]] = []
    skipped: set[int] = set()
    cur = [0]
    i = 0
    while i < N - 1:
        # 성공 transit → 충돌-free 이동 보장 → 현재 run 유지
        if i in transit_segments:
            cur.append(i + 1)
            i += 1
            continue
        # small jump 이고 직선 스캔이 충돌-free → 유지
        jump = float(np.max(np.abs(selected[i + 1] - selected[i])))
        if jump <= reconfig_threshold_rad and _scan_ok(i, i + 1):
            cur.append(i + 1)
            i += 1
            continue

        # 못 이음(큰 reconfig 또는 스캔 관통) → 아웃라이어 skip 시도(transit 시작점은 건너뛰지 않음)
        def _connectable(a, b):
            return (b not in transit_segments
                    and float(np.max(np.abs(selected[b] - selected[a]))) <= reconfig_threshold_rad
                    and _scan_ok(a, b))

        j = i + 1
        n_dropped = 0
        while (j < N and n_dropped < max_skip and j not in transit_segments
               and not _connectable(i, j)):
            j += 1
            n_dropped += 1
        if j < N and _connectable(i, j):
            skipped.update(range(i + 1, j))
            cur.append(j)
            i = j
            continue

        # 이을 수 없는 경계 → run 을 끊고 i+1 부터 새 run 시작
        runs.append(cur)
        cur = [i + 1]
        i += 1

    if cur[-1] != N - 1:
        cur.append(N - 1)
    runs.append(cur)
    return runs, skipped


def _precompute_scan_free(selected, reconfig_threshold_rad, max_skip,
                          robot_cfg, world_scene):
    """직선 joint 스캔이 충돌-free인 (a,b) 쌍을 한 번의 batch collision check로 미리 계산.

    스캔으로 이어질 수 있는 후보(=jump ≤ thr, b-a ≤ max_skip+1)만 검사한다. 큰 jump는 어차피
    transit이 필요하므로 스캔 검사 불필요. 반환: 함수 scan_free(a,b)->bool (표에 없으면 True).
    """
    N = len(selected)
    pairs = []
    for a in range(N - 1):
        for b in range(a + 1, min(a + max_skip + 2, N)):
            if float(np.max(np.abs(selected[b] - selected[a]))) <= reconfig_threshold_rad:
                pairs.append((a, b))
    if not pairs:
        return lambda a, b: True

    dense_list, counts = [], []
    for (a, b) in pairs:
        d = densify_for_collision_check(np.stack([selected[a], selected[b]]))
        dense_list.append(d)
        counts.append(len(d))
    isc, _ = batch_collision_check(np.concatenate(dense_list, axis=0), robot_cfg, world_scene)

    free = {}
    off = 0
    for (a, b), n in zip(pairs, counts):
        free[(a, b)] = not bool(isc[off:off + n].any())
        off += n
    return lambda a, b: free.get((a, b), True)


def _typed_segments_for_run(selected, transit_segments, run_idx, dense_step_rad):
    """한 run을 'scan' / 'transit' dense sub-path 들로 분할한다.

    인접 viewpoint 간 전이를 두 종류로 구분한다:
        - transit edge (nxt==cur+1 이고 cur in transit_segments) → MotionGen 재배치 경로.
          별도 'transit' 세그먼트로 떼어낸다(joint 속도로 빠르게 지나갈 구간).
        - 그 외(small jump / skip 재연결) → 직선 보간으로 잇는 실제 스캔 이동. 연속된
          스캔 이동은 하나의 'scan' 세그먼트로 모은다(EE arc-length로 resample할 구간).

    각 세그먼트는 시작/끝 viewpoint config 를 모두 포함하므로, 인접 세그먼트는 경계
    config 를 공유한다(_stitch_pieces 가 중복 제거).

    Returns:
        list[(kind, dense (K,6))], kind ∈ {"scan", "transit"}
    """
    segments = []
    scan_buf = [selected[run_idx[0]:run_idx[0] + 1]]   # 첫 viewpoint 로 시작
    for a in range(len(run_idx) - 1):
        cur = run_idx[a]
        nxt = run_idx[a + 1]
        if nxt == cur + 1 and cur in transit_segments:
            # 현재까지 모은 스캔 이동을 flush (selected[cur]에서 끝남)
            buf = np.concatenate(scan_buf, axis=0)
            if len(buf) >= 1:
                segments.append(("scan", buf))
            # transit 전체(양 끝 = selected[cur]≈transit[0], selected[nxt]≈transit[-1] 포함)
            segments.append(("transit", transit_segments[cur]))
            # 다음 스캔 버퍼는 selected[nxt]에서 새로 시작
            scan_buf = [selected[nxt:nxt + 1]]
        else:
            interior = _linear_interior(selected[cur], selected[nxt], dense_step_rad)
            if interior is not None:
                scan_buf.append(interior)
            scan_buf.append(selected[nxt:nxt + 1])
    buf = np.concatenate(scan_buf, axis=0)
    if len(buf) >= 1:
        segments.append(("scan", buf))
    return segments


def _stitch_pieces(pieces, masks, dup_tol_rad=5e-3):
    """resample된 sub-path 들을 이어붙인다. 인접 piece 가 공유하는 경계 waypoint 는
    한 번만 남긴다(중복 제거). pieces/masks 는 길이가 같은 list."""
    out_j, out_m = [], []
    for p, m in zip(pieces, masks):
        if len(p) == 0:
            continue
        if out_j and len(p) >= 1 and \
                float(np.max(np.abs(p[0] - out_j[-1][-1]))) <= dup_tol_rad:
            p, m = p[1:], m[1:]
        if len(p) == 0:
            continue
        out_j.append(p)
        out_m.append(m)
    if not out_j:
        return np.zeros((0, 6), dtype=np.float64), np.zeros((0,), dtype=bool)
    return np.concatenate(out_j, axis=0), np.concatenate(out_m, axis=0)


def interpolate_and_resample(selected, transit_segments, robot_cfg,
                             mode="ee", spacing=0.01, dense_step_rad=0.02,
                             transit_spacing_rad=TRANSIT_RESAMPLE_SPACING_RAD,
                             reconfig_threshold_rad=np.deg2rad(RECONFIG_THRESHOLD_DEG),
                             max_skip=TRANSIT_FAIL_SKIP_MAX, world_scene=None):
    """DP 궤적 + transit을 합치고, 선택된 metric으로 uniform resample.

    transit 실패 reconfig 은 직선 보간하면 물체를 관통하므로(#2) 사용하지 않는다. 대신
    경로를 '안전 연속 run'들로 분할(_build_runs)하고 **가장 긴 run** 을 채택한다. 즉
    아웃라이어 viewpoint 는 건너뛰고, 이을 수 없는 경계에서는 작은 쪽을 통째로 드롭한다.

    Args:
        selected: (N, 6) DP 선택 궤적
        transit_segments: dict {idx: (T, 6)} 성공한 transit 경로
        robot_cfg: cuRobo robot config dict (mode='ee'일 때 FK 용)
        mode: "ee" (EE position arc-length, meters) | "joint" (cumulative L∞, radians)
        spacing: 스캔 구간 최종 spacing (mode에 따라 m 또는 rad)
        dense_step_rad: dense path 구성 시 joint-space L∞ step (radians)
        transit_spacing_rad: transit(재배치) 구간 joint-space resample 간격 (radians).
            transit은 검사가 아니므로 EE 기준이 아니라 joint 기준으로 sparse하게 재분할한다.
        reconfig_threshold_rad: 직선 보간 허용 임계값(이보다 크면 transit 필요)
        max_skip: transit 실패 시 run 내부에서 건너뛸 수 있는 최대 viewpoint 수

    Returns:
        resampled:  (M, 6) uniform-spaced trajectory (가장 긴 안전 run)
        is_transit: (M,) bool — 각 waypoint가 transit(재배치) 구간인지. compute_trajectory_times
                    가 이 구간을 joint 속도로만(검사 EE 속도 무시) 시간 매기는 데 쓴다.
        dropped:    list[int] — 채택 run 에 포함되지 않아 드롭된 viewpoint 인덱스
        runs_info:  dict — {"runs": [(start,end,len)...], "kept": (start,end,len)}
    """
    N = len(selected)
    # world_scene이 있으면 직선 스캔 충돌까지 고려해 run을 나눈다. 곡면 측면에서 인접
    # viewpoint 사이 직선 보간이 물체를 관통하면(via-home으로 진입은 됐어도) 스캔 자체가 불가 →
    # 그 구간을 끊어 정직하게 드롭(최종 batch 충돌검사 거부 대신 유효 궤적 저장).
    scan_free = (_precompute_scan_free(selected, reconfig_threshold_rad, max_skip,
                                       robot_cfg, world_scene)
                 if world_scene is not None else None)
    runs, _ = _build_runs(selected, transit_segments, reconfig_threshold_rad, max_skip,
                          scan_free=scan_free)
    kept = max(runs, key=len)
    kept_set = set(kept)
    dropped = sorted(set(range(N)) - kept_set)

    runs_info = {
        "runs": [(r[0], r[-1], len(r)) for r in runs],
        "kept": (kept[0], kept[-1], len(kept)),
    }

    if len(kept) < 2:
        raise RuntimeError(
            "안전하게 연속인 구간이 viewpoint 2개 미만입니다 — 모든 인접 전이가 "
            "이을 수 없는 reconfig. transit 성공(#1) 또는 reconfig 감소(#3)가 필요합니다."
        )

    if mode not in ("ee", "joint"):
        raise ValueError(f"Unknown resample mode: {mode!r} (expected 'ee' or 'joint')")

    # 스캔 / transit 세그먼트를 분리해 각각에 맞는 기준으로 resample 한다.
    #   scan    → EE arc-length(또는 joint, mode에 따라). 검사 spacing 그대로.
    #   transit → joint-space L∞ arc-length(sparse). EE 호 길이와 무관하게 재배치.
    segments = _typed_segments_for_run(selected, transit_segments, kept, dense_step_rad)
    pieces, masks = [], []
    for kind, dense in segments:
        if kind == "transit":
            rs = _resample_uniform_joint(dense, transit_spacing_rad)
            # sparse joint resample은 코너에서 직선을 그어 MotionGen이 충돌-free로 보장한
            # 곡선 경로를 잘라낼 수 있다(특히 물체로 재진입하는 via-home 재접근 leg). 그러면
            # 최종 densify 충돌검사에서 관통이 잡혀 저장이 거부된다. world_scene이 주어지면
            # 이 transit piece를 검사해, 충돌이 생기면 MotionGen native 해상도를 그대로 쓴다
            # (densify가 모든 vertex를 지나 충돌-free 유지). 자유공간 transit은 sparse 유지.
            if world_scene is not None and len(rs) >= 2:
                isc, _ = batch_collision_check(
                    densify_for_collision_check(rs), robot_cfg, world_scene)
                if bool(isc.any()):
                    rs = dense
            mk = np.ones((len(rs),), dtype=bool)
        elif mode == "ee":
            rs = _resample_uniform_ee(dense, robot_cfg, spacing)
            mk = np.zeros((len(rs),), dtype=bool)
        else:  # mode == "joint"
            rs = _resample_uniform_joint(dense, spacing)
            mk = np.zeros((len(rs),), dtype=bool)
        pieces.append(rs)
        masks.append(mk)

    resampled, is_transit = _stitch_pieces(pieces, masks)
    return resampled, is_transit, dropped, runs_info


def batch_collision_check(trajectory, robot_cfg, world_scene):
    """전체 궤적에 대해 batch collision check 수행. Returns (is_collision, n_collisions)."""
    # collision_activation_distance 의 cuRobo 기본값은 0.2 m 이다. 그 경우 비용은 장애물
    # 20 cm 이내에서 양수가 되므로(카메라는 작업거리 46 mm 라 항상 그 안), cost > 0 검사가
    # 모든 waypoint 를 충돌로 판정해 버린다. 최종 검증에서는 실제 침투만 잡도록
    # activation distance 를 COLLISION_MARGIN(기본 0)으로 둔다 → cost > 0 ⇔ 실제 침투.
    cfg = RobotCollisionCheckerCfg.load_from_config(
        robot_config=robot_cfg,
        scene_model=world_scene,
        n_cuboids=max(1, len(world_scene.cuboid)),
        n_meshes=max(1, len(world_scene.mesh)),
        collision_activation_distance=float(config.COLLISION_MARGIN),
        self_collision_activation_distance=0.0,
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

    # cuRobo collision cost는 음수가 아니다: 0 = 안전, >0 = 충돌(또는 activation_distance
    # 이내 근접). cuRobo 본체 RobotSceneCollision.validate()도 "충돌 없음 ⇔ cost == 0.0"으로
    # 판정한다. 따라서 충돌은 cost > 0 으로 잡아야 한다. (과거 `< 0` 비교는 절대 참이 될 수
    # 없어 월드/자가 충돌 검사가 항상 무력화됐다.)
    # Cost shape may be (batch, horizon, num_spheres) or (batch, horizon); reduce trailing dims.
    COLLISION_COST_EPS = 1e-6  # float noise 방지용 임계값
    d_scene_r = d_scene.view(batch, -1)
    d_self_r = d_self.view(batch, -1)
    is_world_collision = (d_scene_r > COLLISION_COST_EPS).any(dim=-1).cpu().numpy()
    is_self_collision = (d_self_r > COLLISION_COST_EPS).any(dim=-1).cpu().numpy()
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
                             corner_max_slowdown=2.5,
                             is_transit=None):
    """Continuous scan용 누적 time 생성.

    스캔 segment 시간은 EE 선속도, EE 각속도, joint 속도 제한을 모두 만족하는 최소 시간으로
    정한다. 단 transit(재배치) segment(is_transit로 표시)는 검사가 아니므로 EE 선속도/각속도/
    corner slowdown을 적용하지 않고 **joint 속도 한계로만** 시간을 매긴다. 그래야 base를 크게
    돌리며 팔 끝이 자유공간에 그리는 긴 호를 50mm/s로 기어가지 않고, joint 속도로 빠르게
    지나간다(예: 139° transit 105s → ~8s).
    """
    n = len(joints)
    times = np.zeros((n,), dtype=np.float64)
    if n < 2:
        return times, {
            "total_time": 0.0,
            "max_linear_speed_mm_s": 0.0,
            "max_angular_speed_deg_s": 0.0,
            "max_joint_speed_rad_s": 0.0,
            "transit_time": 0.0,
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

    # segment(i-1→i)는 양 끝 중 하나라도 transit이면 transit으로 본다(스캔↔transit 진입/이탈 포함).
    if is_transit is not None and len(is_transit) == n:
        it = np.asarray(is_transit, dtype=bool)
        seg_is_transit = it[1:] | it[:-1]
    else:
        seg_is_transit = np.zeros((n - 1,), dtype=bool)

    transit_time = 0.0
    for i in range(1, n):
        linear_dist = float(np.linalg.norm(ee_positions[i] - ee_positions[i - 1]))
        angular_dist = _quat_angle_xyzw(ee_quaternions[i - 1], ee_quaternions[i])
        joint_dist = float(np.max(np.abs(joints[i] - joints[i - 1])))

        if seg_is_transit[i - 1]:
            # 재배치: joint 속도 한계로만 (EE 속도/corner 무시)
            dt_candidates = [min_segment_dt]
            if max_joint_vel_rad_s > 0.0:
                dt_candidates.append(joint_dist / max_joint_vel_rad_s)
            dt = max(dt_candidates)
            transit_time += dt
        else:
            dt_candidates = [min_segment_dt]
            if ee_speed_m_s > 0.0:
                dt_candidates.append(linear_dist / ee_speed_m_s)
            if ee_angular_speed_rad_s > 0.0:
                dt_candidates.append(angular_dist / ee_angular_speed_rad_s)
            if max_joint_vel_rad_s > 0.0:
                dt_candidates.append(joint_dist / max_joint_vel_rad_s)
            dt = max(dt_candidates) * slowdown_factors[i - 1]

        times[i] = times[i - 1] + dt

    segment_dt = np.diff(times)
    linear_speed = np.linalg.norm(np.diff(ee_positions, axis=0), axis=1) / segment_dt
    angular_speed = np.array([
        _quat_angle_xyzw(ee_quaternions[i - 1], ee_quaternions[i]) / segment_dt[i - 1]
        for i in range(1, n)
    ])
    joint_speed = np.max(np.abs(np.diff(joints, axis=0)), axis=1) / segment_dt
    # EE 선/각속도 최대는 '스캔 구간'에서만 의미가 있다(transit은 EE가 자유공간을 빠르게 휘둘러
    # EE 속도가 매우 커지지만 검사 속도가 아님). joint 속도 최대는 전체에서 본다.
    scan_seg = ~seg_is_transit
    scan_lin = linear_speed[scan_seg] if scan_seg.any() else linear_speed
    scan_ang = angular_speed[scan_seg] if scan_seg.any() else angular_speed
    stats = {
        "total_time": float(times[-1]),
        "transit_time": float(transit_time),
        "n_transit_segments": int(seg_is_transit.sum()),
        "max_linear_speed_mm_s": float(scan_lin.max() * 1000.0),
        "max_angular_speed_deg_s": float(np.rad2deg(scan_ang.max())),
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

    # 물체별 기본 배치(config.OBJECT_PLACEMENTS)를 먼저 반영. CLI override 가 그 뒤라 우선한다.
    if config.apply_object_placement(args.object):
        print(f"  Per-object placement '{args.object}': pos={config.TARGET_OBJECT['position']}, "
              f"quat={config.TARGET_OBJECT['rotation']}")

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

    # wrist_3 고정을 DP '이전'에 적용 — wrist_3는 어차피 0으로 잠그며(검사 시 광축 roll
    # 무관), IK는 free wrist_3로 풀려 후보마다 wrist_3가 제각각이다. DP의 reconfig 비용은
    # 6-DoF L∞라, 5-DoF로는 연속인 해가 '버려질' wrist_3 차이 때문에 reconfig로 오판돼
    # DP가 엉뚱한 분기를 고른다. 미리 0으로 잠그면 6-DoF L∞ = 5-DoF L∞ 가 되어 DP가
    # 실제 최종 자세 기준으로 연속 해를 직접 고른다.
    wrist3_fixed = config.ROBOT_START_STATE[-1]
    for reps in representatives:
        if len(reps) > 0:
            reps[:, -1] = wrist3_fixed
    print(f"  Locked wrist_3 at {np.rad2deg(wrist3_fixed):.1f}° (pre-DP)")

    reconfig_rad = np.deg2rad(RECONFIG_THRESHOLD_DEG)

    # (a) 연속성 IK seeding: 초기 DP로 reconfig/empty 지점을 찾아 이웃 자세로 IK 재시도.
    if ENABLE_CONTINUITY_SEEDING:
        empty_idxs = [i for i, r in enumerate(representatives) if len(r) == 0]
        reps_dp0 = [r.copy() for r in representatives]      # dp는 carry-forward로 입력을 변형
        selected0, _, _ = dp_optimal_path(reps_dp0, reconfig_rad)
        jumps0 = np.max(np.abs(np.diff(selected0, axis=0)), axis=1)
        recfg0 = np.where(jumps0 > reconfig_rad)[0]

        targets: dict[int, np.ndarray] = {}
        for idx in recfg0:                                  # reconfig 양끝을 서로의 자세로 seed
            targets[idx + 1] = selected0[idx]
            targets[idx] = selected0[idx + 1]
        for i in empty_idxs:                                # empty는 가까운 이웃 자세로 seed
            nb = i - 1 if i > 0 else min(i + 1, len(selected0) - 1)
            targets.setdefault(i, selected0[nb])

        if targets:
            tlist = sorted(targets)
            anchors = np.array([targets[i] for i in tlist])
            print(f"[5a/6] Continuity IK seeding — {len(tlist)} targets "
                  f"({len(recfg0)} reconfig, {len(empty_idxs)} empty)...")
            sol2, succ2 = solve_ik_seeded(
                robot_cfg, world_config, positions_np, quats_np, tlist, anchors,
            )
            sol2[..., -1] = wrist3_fixed
            added, filled = 0, 0
            for k, vp in enumerate(tlist):
                good = sol2[k][succ2[k]]
                if len(good) == 0:
                    continue
                if len(representatives[vp]) == 0:
                    filled += 1
                merged = (np.vstack([representatives[vp], good])
                          if len(representatives[vp]) > 0 else good)
                representatives[vp] = _greedy_dedup(merged, DP_CANDIDATE_DEDUP_RAD)
                added += len(good)
            print(f"  Continuity seeding: +{added} solutions, "
                  f"{filled}/{len(empty_idxs)} empty viewpoints filled")

    # 충돌하는 대표 해 제거 (DP는 충돌을 안 보므로). 최종검사와 동일한 batch_collision_check로
    # wrist_3 잠금 후 자세를 검사 → 충돌 자세를 후보에서 빼 DP가 충돌-free만 고르게 한다.
    # 충돌-free가 0개가 된 viewpoint는 아래 empty-drop이 unreachable로 처리한다.
    if FILTER_COLLIDING_REPS:
        spans = []   # (vp, flat_start, count)
        flat = []
        for i, r in enumerate(representatives):
            if len(r) > 0:
                spans.append((i, sum(len(f) for f in flat), len(r)))
                flat.append(r)
        if flat:
            isc_flat, _ = batch_collision_check(
                np.concatenate(flat, axis=0), robot_cfg, world_config,
            )
            n_removed, n_emptied = 0, 0
            for vp, start, cnt in spans:
                free_mask = ~isc_flat[start:start + cnt]
                removed = cnt - int(free_mask.sum())
                if removed > 0:
                    n_removed += removed
                    representatives[vp] = representatives[vp][free_mask]
                    if len(representatives[vp]) == 0:
                        n_emptied += 1
            if n_removed > 0:
                print(f"  Collision-filtered reps: removed {n_removed} colliding solutions "
                      f"(margin={config.COLLISION_MARGIN*1000:.0f}mm), "
                      f"{n_emptied} viewpoints emptied → unreachable")
            else:
                print(f"  Collision-filtered reps: 0 colliding (all candidates collision-free)")

    # 못 가는(empty) viewpoint 제거 — IK 해가 없거나 충돌 필터로 비워진 viewpoint는
    # carry-forward로 메우지 않고 경로에서 뺀다.
    # orig_idx: 남은 viewpoint의 '원본' 인덱스(드롭 후에도 로그를 원래 번호로 표기하기 위함).
    orig_idx = np.arange(len(representatives))
    n_dropped_empty = 0
    if DROP_UNREACHABLE_VIEWPOINTS:
        keep = np.array([len(r) > 0 for r in representatives], dtype=bool)
        n_dropped_empty = int((~keep).sum())
        if n_dropped_empty > 0:
            dropped_list = orig_idx[~keep].tolist()
            print(f"  Dropping {n_dropped_empty} unreachable (empty) viewpoints "
                  f"(no IK solution): {dropped_list}")
            representatives = [r for r, k in zip(representatives, keep) if k]
            all_solutions = all_solutions[keep]
            all_success = all_success[keep]
            if cluster_id is not None:
                cluster_id = cluster_id[keep]
            orig_idx = orig_idx[keep]
            if len(representatives) < 2:
                raise RuntimeError(
                    f"도달 가능한 viewpoint가 {len(representatives)}개뿐입니다 — "
                    "물체 배치/작업거리(WD)를 조정해 reachability를 높여야 합니다."
                )

    print("[5/6] Phase 3 — DP optimal path...")
    selected, _, stats = dp_optimal_path(representatives, reconfig_rad)

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
            _jn = ["pan", "lift", "elbow", "w1", "w2", "w3"]
            intra_reconfig_idx = np.where(is_reconfig & ~is_inter_cluster)[0]
            for idx in intra_reconfig_idx:
                jump_deg = np.rad2deg(jumps[idx])
                cid = cluster_id[idx]
                # 어떤 joint가 튀는지 (per-joint |Δ| deg)
                dq_deg = np.rad2deg(np.abs(selected[idx + 1] - selected[idx]))
                worst = int(np.argmax(dq_deg))
                per_joint = " ".join(f"{n}={d:.0f}" for n, d in zip(_jn, dq_deg))
                # 연속 해가 IK pool에 있었는가? (wrist_3 잠금이므로 5-DoF L∞로 비교)
                def _min_pool_linf(vp, ref):
                    cand = all_solutions[vp][all_success[vp]]
                    if len(cand) == 0:
                        return None
                    return float(np.rad2deg(np.min(np.max(np.abs(cand[:, :5] - ref[:5]), axis=1))))
                d_next = _min_pool_linf(idx + 1, selected[idx])      # vp(idx+1) ~ selected[idx]
                d_prev = _min_pool_linf(idx, selected[idx + 1])      # vp(idx)   ~ selected[idx+1]
                thr_deg = np.rad2deg(reconfig_rad)
                def _verdict(d):
                    if d is None:
                        return "no-IK"
                    return f"{d:.0f}° ({'POOL-HAS-CONT' if d <= thr_deg else 'no-cont-in-pool'})"
                o0, o1 = int(orig_idx[idx]), int(orig_idx[idx + 1])
                print(f"      viewpoint {o0}→{o1} (cluster {cid}): jump {jump_deg:.1f}° "
                      f"[worst={_jn[worst]}]  Δ: {per_joint}")
                print(f"          pool-continuity: vp{o1}~sel[{o0}]={_verdict(d_next)}, "
                      f"vp{o0}~sel[{o1}]={_verdict(d_prev)}  (thr={thr_deg:.0f}°)")

    # Phase 4: MotionPlanner transit at reconfig points
    reconfig_indices = np.where(is_reconfig)[0] if cluster_id is not None else np.array([], dtype=int)
    transit_segments = {}
    if len(reconfig_indices) > 0:
        print(f"\n[Phase 4] MotionPlanner transit for {len(reconfig_indices)} reconfig points...")
        transit_segments, transit_stats = plan_reconfig_transits(
            selected, reconfig_indices, robot_cfg, world_config, label_idx=orig_idx, wd_m=wd_m,
        )
        # 안전망: MotionPlanner가 중간에 wrist_3를 흔들었을 수 있으므로 강제 고정.
        # 단 via-roll/via-tilt 는 의도적으로 rolled 중간자세(wrist_3 가변)를 쓰므로 덮어쓰면
        # 가교가 깨진다 → scan config 가 양 끝인 direct/via-home route 에만 적용.
        _routes = {s["idx"]: s.get("route") for s in transit_stats if s.get("success")}
        for idx in transit_segments:
            if _routes.get(idx) in ("direct", "via-home"):
                transit_segments[idx][:, -1] = wrist3_fixed

    # Phase 5: Uniform resample + collision check
    print(f"\n[Phase 5] Interpolation + uniform resample (mode={RESAMPLE_MODE})...")
    final_traj, final_is_transit, skipped_vps, runs_info = interpolate_and_resample(
        selected, transit_segments, robot_cfg,
        mode=RESAMPLE_MODE, spacing=args.spacing,
        reconfig_threshold_rad=reconfig_rad, world_scene=world_config,
    )
    skipped_orig = [int(orig_idx[i]) for i in skipped_vps]   # 원본 viewpoint 번호로 표기
    if len(runs_info["runs"]) > 1:
        kl = runs_info["kept"][2]
        print(
            f"  WARNING: 전이 불가(transit 실패/스캔 충돌)로 경로가 "
            f"{len(runs_info['runs'])}개 run으로 끊김 "
            f"→ 가장 긴 run ({kl}개 viewpoint) 채택, "
            f"viewpoint {len(skipped_vps)}개 드롭(원본 번호): {skipped_orig}"
        )
    elif skipped_vps:
        print(
            f"  WARNING: 전이 불가(transit 실패/스캔 충돌)로 아웃라이어 viewpoint "
            f"{len(skipped_vps)}개 건너뜀(원본 번호): {skipped_orig}"
        )
    if RESAMPLE_MODE == "ee":
        spacing_desc = f"EE spacing={args.spacing*1000:.1f} mm"
    else:
        spacing_desc = f"joint spacing={np.rad2deg(args.spacing):.2f}°"
    n_transit_wp = int(np.asarray(final_is_transit).sum())
    print(f"  Resampled: {len(final_traj)} waypoints ({spacing_desc}, "
          f"scan={len(final_traj) - n_transit_wp}, "
          f"transit={n_transit_wp} @ joint spacing {np.rad2deg(TRANSIT_RESAMPLE_SPACING_RAD):.1f}°)")

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
        is_transit=final_is_transit,
    )
    scan_time = time_stats['total_time'] - time_stats['transit_time']
    print(f"  Time profile: total={time_stats['total_time']:.1f}s "
          f"(scan={scan_time:.1f}s, transit={time_stats['transit_time']:.1f}s "
          f"in {time_stats['n_transit_segments']} seg), "
          f"max scan EE={time_stats['max_linear_speed_mm_s']:.1f} mm/s, "
          f"max scan rot={time_stats['max_angular_speed_deg_s']:.1f} deg/s, "
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
    covered = runs_info["kept"][2]
    print(f"\nDone. coverage={covered}/{N} viewpoints "
          f"(unreachable dropped={n_dropped_empty}, transit-split dropped={len(skipped_vps)}), "
          f"reconfigs={stats['n_reconfigs']} (inter={rc_inter}, intra={rc_intra}), "
          f"transit={n_transit_ok}/{len(reconfig_indices)} OK, "
          f"collisions={n_collisions}, final={len(final_traj)} waypoints")


if __name__ == "__main__":
    main()
