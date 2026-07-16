"""Robot configuration, kinematics, world, and collision services."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.scene import Cuboid, Mesh as CuRoboMesh, Scene
from curobo.types import JointState

from common import config
from common.math_utils import quaternion_to_rotation_matrix
from .settings import COLLISION_EXCLUDE_LINKS

_TIMINGS = []  # [(label, seconds)]


def _tick(label, t0):
    """t0(=time.time() 시작) 이후 경과를 label 로 _TIMINGS 에 누적."""
    _TIMINGS.append((label, time.time() - t0))


# =========================================================================
# Solver/checker 재사용 — Kinematics·RobotCollisionChecker 를 run 당 1회만 빌드
# =========================================================================
# compute_fk·batch_collision_check 는 매 호출 cuRobo 객체를 새로 빌드했다. 한 run 내
# robot_cfg·world_scene 는 불변(main 에서 1개씩)이라 결과는 빌드 횟수와 무관 → id 캐시로
# 1회만 빌드해 반복 빌드비를 없앤다(궤적 보존, plan_cspace/IK 결과 불변). 캐시가 객체 참조를
# 들고 있어 id 재사용 위험 없음.

_KIN_CACHE = {}      # id(robot_cfg) -> Kinematics
_CC_CACHE = {}       # (id(robot_cfg), id(world_scene)) -> RobotCollisionChecker
_REUSE_HITS = {"kin": 0, "cc": 0}   # 캐시 히트(=절약된 빌드) 수


def _get_kinematics(robot_cfg):
    kin = _KIN_CACHE.get(id(robot_cfg))
    if kin is None:
        _t = time.time()
        kin = Kinematics(KinematicsCfg.from_robot_yaml_file(robot_cfg))
        _KIN_CACHE[id(robot_cfg)] = kin
        _tick("kin_build(1x)", _t)
    else:
        _REUSE_HITS["kin"] += 1
    return kin


def _get_collision_checker(robot_cfg, world_scene):
    key = (id(robot_cfg), id(world_scene))
    checker = _CC_CACHE.get(key)
    if checker is None:
        _t = time.time()
        cfg = RobotCollisionCheckerCfg.load_from_config(
            robot_config=robot_cfg,
            scene_model=world_scene,
            n_cuboids=max(1, len(world_scene.cuboid)),
            n_meshes=max(1, len(world_scene.mesh)),
            collision_activation_distance=float(config.COLLISION_MARGIN),
            self_collision_activation_distance=0.0,
        )
        checker = RobotCollisionChecker(cfg)
        _CC_CACHE[key] = checker
        _tick("cc_build(1x)", _t)
    else:
        _REUSE_HITS["cc"] += 1
    return checker
def resolve_robot_config(robot_filename: str):
    """Robot YAML 을 dict 로 로드하고 urdf_path/asset_root_path 를 절대경로로 패치.

    탐색 순서: 프로젝트 workcell/robot/ → cuRobo content/configs/robot/.
    """
    import yaml
    from curobo.content import get_robot_configs_path

    candidates = [
        config.PROJECT_ROOT / "workcell" / "robot" / robot_filename,
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
        config.PROJECT_ROOT / "workcell" / "robot",
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
    """robot_cfg의 collision_sphere_buffer를 진단용 한 줄 문자열로 요약(없으면 None)."""
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
def build_collision_world(object_name: str):
    """Build cuRobo WorldConfig from config.py obstacles + target object mesh."""
    import trimesh

    # TARGET_OBJECT may have been changed by a CLI override or a viewport gizmo.
    config.sync_support_to_target()
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
        shape = config.OBJECT_COLLISION_SHAPE.get(object_name)
        if shape == "box":
            # 소형 mesh 는 cuRobo mesh 충돌이 전부 오판 → mesh bbox 를 analytic Cuboid(obb) 로 대체.
            # object-local bbox 중심을 object pose 로 옮기고, cuboid 방향은 object rotation 을 그대로 쓴다.
            R = quaternion_to_rotation_matrix(np.asarray(rot, dtype=np.float64))
            world_center = np.asarray(pos, dtype=np.float64) + R @ mesh.bounds.mean(axis=0)
            cuboids.append(Cuboid(
                name="target_object",
                pose=[*world_center.tolist(), rot[0], rot[1], rot[2], rot[3]],
                dims=mesh.extents.tolist(),
            ))
            print(f"  Collision world: {len(cuboids)} cuboids "
                  f"(target as box proxy {np.round(mesh.extents, 3).tolist()} m — 소형 mesh 충돌 오판 회피)")
        else:
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
    kin = _get_kinematics(robot_cfg)            # run 당 1회만 빌드(재사용)
    q_batch = torch.tensor(solutions, device="cuda:0", dtype=torch.float32)
    js = JointState.from_position(q_batch, joint_names=kin.joint_names)
    state = kin.compute_kinematics(js)

    ee_pose = state.tool_poses.get_link_pose(kin.tool_frames[0])
    ee_positions = ee_pose.position.cpu().numpy()
    ee_quat_wxyz = ee_pose.quaternion.cpu().numpy()
    ee_quaternions = ee_quat_wxyz[:, [1, 2, 3, 0]]

    return ee_positions, ee_quaternions
def batch_collision_check(trajectory, robot_cfg, world_scene):
    """전체 궤적에 대해 batch collision check 수행. Returns (is_collision, n_collisions)."""
    # collision_activation_distance 의 cuRobo 기본값은 0.2 m 이다. 그 경우 비용은 장애물
    # 20 cm 이내에서 양수가 되므로(카메라는 작업거리 46 mm 라 항상 그 안), cost > 0 검사가
    # 모든 waypoint 를 충돌로 판정해 버린다. 최종 검증에서는 실제 침투만 잡도록
    # activation distance 를 COLLISION_MARGIN(기본 0)으로 둔다 → cost > 0 ⇔ 실제 침투.
    checker = _get_collision_checker(robot_cfg, world_scene)   # run 당 1회만 빌드(재사용)

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
