#!/usr/bin/env python3
"""검사 셀 씬(workcell/scenes/*.yaml) 의 스키마·로더·검증을 소유한다.

`config.py` 는 I/O 가 없는 심볼 테이블이고, 씬은 플래너(core/trajectory/robot.py)·
Isaac(core/isaac/scene.py)·viser(apps/trajectory_studio.py) 세 subsystem 이 공유한다.
그래서 **파일 포맷은 이 모듈이 소유**하고 config.py 는 소비자가 이미 import 하는 façade 로
남는다. 어떤 소비자도 YAML 을 직접 파싱하지 않는다.

의존성 주의: numpy 와 (함수 스코프) yaml 만 쓴다. torch/curobo/trimesh 를 모듈 레벨에서
import 하면 안 된다 — Isaac Sim 런타임이 cuRobo 를 피해서 이 모듈을 로드한다.

⚠️ cuRobo 0.8 은 sphere/cylinder/capsule 을 충돌에 넣지 않는다.
   curobo/_src/geom/data/data_scene.py 의 load_from_scene_cfg() 는 cuboid/mesh/voxel 만
   순회하고, add_obstacle() 도 Cuboid/Mesh/VoxelGrid 만 디스패치한다. 변환 함수
   SceneCfg.get_collision_check_world() 는 있지만 cuRobo 내부 어디서도 호출되지 않는다.
   즉 Scene(cylinder=[...]) 는 예외도 경고도 없이 충돌 기하가 0개가 된다 — 통과할 수 있는 벽.
   그래서 스키마는 네 타입을 표현하되, obstacle_obb() 가 **우리 손으로** OBB 로 바꾼다.
   플래너·viser·Isaac 이 모두 이 함수를 쓰므로 보이는 것과 푸는 것이 항상 일치한다.
"""
import re
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
SCENES_ROOT = PROJECT_ROOT / "workcell" / "scenes"

SCHEMA_VERSION = 1
DEFAULT_SCENE = "sim_default"

PRIMITIVE_TYPES = ("cuboid", "sphere", "cylinder", "capsule")

# 이름이 곧 역할이다. `role:` 키를 새로 만들지 않는 이유: support 는 이미
# config.sync_support_to_target() 과 core/isaac/scene.py 가 **이름으로** 찾고 있어서,
# 같은 일을 하는 두 번째 개념을 만들 이유가 없다.
ROLE_NAMES = ("table", "robot_mount", "support")

# 스테이지 전체를 이름으로 훑어 지우는 코드가 있는 스코프들(isaac_pipeline).
# 장애물이 이 이름을 쓰면 "Clear Collision Spheres" 같은 버튼에 조용히 지워진다.
RESERVED_PRIM_NAMES = ("CuRoboCollisionSpheres", "CameraFovPlane",
                       "CameraRangeRays", "Viewpoints")

_USD_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


# ── 경로 해석 ────────────────────────────────────────────────────────────────

def resolve_scene_path(name_or_path) -> Path:
    """씬 이름 또는 경로 → 실제 파일. robot.resolve_robot_config 와 같은 규약이다:
    후보를 순서대로 시도하고, 실패하면 **시도한 후보를 전부** 나열해 던진다."""
    if name_or_path is None:
        name_or_path = DEFAULT_SCENE
    raw = Path(str(name_or_path))
    candidates = []
    if raw.suffix in (".yaml", ".yml"):
        candidates.append(raw if raw.is_absolute() else PROJECT_ROOT / raw)
    else:
        candidates.append(SCENES_ROOT / f"{raw.name}.yaml")
        candidates.append(SCENES_ROOT / f"{raw.name}.yml")
    for path in candidates:
        if path.exists():
            return path
    tried = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"scene '{name_or_path}' not found. Tried:\n  {tried}\n"
        f"Available scenes: {', '.join(available_scenes()) or '(none)'}")


def available_scenes() -> list[str]:
    """workcell/scenes/ 에 있는 씬 이름들(에러 메시지·UI 용)."""
    if not SCENES_ROOT.is_dir():
        return []
    return sorted({p.stem for p in SCENES_ROOT.iterdir()
                   if p.suffix in (".yaml", ".yml")})


# ── 로드/검증 ────────────────────────────────────────────────────────────────

def _vec(value, n, where):
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (n,):
        raise ValueError(f"{where}: expected a list of length {n} (got {value!r})")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{where}: values must be finite (got {value!r})")
    return arr


def _quat(value, where):
    if value is None:
        return _IDENTITY_QUAT.copy()
    arr = _vec(value, 4, where)
    norm = float(np.linalg.norm(arr))
    # 손으로 잰 셀의 오타(0.7071 을 0.707 로 적는 류)를 여기서 잡는다.
    if abs(norm - 1.0) > 1e-6:
        raise ValueError(f"{where}: quaternion [w,x,y,z] must be unit-norm (norm={norm:.9f})")
    return arr


def _positive(value, where):
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{where}: must be positive (got {value!r})")
    return v


def _parse_obstacle(raw, idx):
    if not isinstance(raw, dict):
        raise ValueError(f"obstacles[{idx}]: must be a mapping (got {raw!r})")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"obstacles[{idx}]: name is required")
    where = f"obstacle '{name}'"

    # 이름은 그대로 USD prim 이름이 된다.
    if not _USD_NAME_RE.match(name):
        raise ValueError(
            f"{where}: must be a legal USD prim name (letters/digits/_, cannot start with a "
            f"digit) — core/isaac/scene.py creates it as /World/SceneObstacles/{{name}}")
    if name in RESERVED_PRIM_NAMES:
        raise ValueError(
            f"{where}: reserved prim name. isaac_pipeline deletes prims with this name "
            f"stage-wide, so the obstacle would silently disappear. Pick another name "
            f"(reserved: {', '.join(RESERVED_PRIM_NAMES)})")

    kind = raw.get("type", "cuboid")
    if kind not in PRIMITIVE_TYPES:
        raise ValueError(f"{where}: type must be one of {PRIMITIVE_TYPES} (got {kind!r})")

    obs = {
        "name": name,
        "type": kind,
        "position": _vec(raw.get("position", [0.0, 0.0, 0.0]), 3, f"{where}.position"),
        "rotation": _quat(raw.get("rotation"), f"{where}.rotation"),
        "isaac_visual": raw.get("isaac_visual", "primitive"),
    }
    if kind == "cuboid":
        if "dimensions" not in raw:
            raise ValueError(f"{where}: cuboid requires dimensions [x,y,z]")
        dims = _vec(raw["dimensions"], 3, f"{where}.dimensions")
        if np.any(dims <= 0.0):
            raise ValueError(f"{where}: dimensions must all be positive (got {dims.tolist()})")
        obs["dimensions"] = dims
    elif kind == "sphere":
        obs["radius"] = _positive(raw.get("radius"), f"{where}.radius")
    elif kind == "cylinder":
        obs["radius"] = _positive(raw.get("radius"), f"{where}.radius")
        obs["height"] = _positive(raw.get("height"), f"{where}.height")
    elif kind == "capsule":
        obs["radius"] = _positive(raw.get("radius"), f"{where}.radius")
        obs["base"] = _vec(raw.get("base", [0.0, 0.0, 0.0]), 3, f"{where}.base")
        obs["tip"] = _vec(raw.get("tip", [0.0, 0.0, 0.0]), 3, f"{where}.tip")
    return obs


def _parse_placement(raw, key):
    where = f"object_placements['{key}']"
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: must be a mapping")
    out = {"rotation": _quat(raw.get("rotation"), f"{where}.rotation")}
    if "position" in raw:
        out["position"] = _vec(raw["position"], 3, f"{where}.position")
    return out


def parse_scene(data: dict, source: str) -> dict:
    """이미 읽어들인 매핑(YAML 또는 h5 스냅샷)을 검증·정규화한다."""
    if not isinstance(data, dict):
        raise ValueError(f"{source}: top level must be a mapping")

    version = data.get("version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{source}: version must be {SCHEMA_VERSION} (got {version!r})")

    raw_target = data.get("target_object") or {}
    target = {
        "name": raw_target.get("name", "target_object"),
        "position": _vec(raw_target.get("position", [0.0, 0.0, 0.0]), 3,
                         "target_object.position"),
        "rotation": _quat(raw_target.get("rotation"), "target_object.rotation"),
    }

    raw_obstacles = data.get("obstacles")
    if not isinstance(raw_obstacles, list) or not raw_obstacles:
        raise ValueError(f"{source}: an obstacles list is required")
    obstacles = [_parse_obstacle(o, i) for i, o in enumerate(raw_obstacles)]

    names = [o["name"] for o in obstacles]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"{source}: duplicate obstacle names — {', '.join(dupes)}")
    for role in ROLE_NAMES:
        if names.count(role) != 1:
            raise ValueError(
                f"{source}: exactly one obstacle must be named '{role}' "
                f"(found {names.count(role)}). config.sync_support_to_target() and "
                f"core/isaac/scene.py look it up by this name")

    table = next(o for o in obstacles if o["name"] == "table")
    # sync_support_to_target() 은 table_top_z = position[2] + dimensions[2]/2 로 상면을
    # 구한다. z-yaw 는 상자의 **연직 방향 크기를 안 바꾸므로** 이 식이 그대로 맞다 —
    # 식을 깨는 것은 x/y 축 회전(기울이기)뿐이다. 셀 전체를 로봇 base Z 둘레로 돌리는 것은
    # 정당한 조작이라(로봇 기준 배치를 실제 셀에 맞추는 유일한 방법) z-yaw 는 허용한다.
    _, qx, qy, _ = (float(v) for v in table["rotation"])
    if table["type"] != "cuboid" or not (abs(qx) < 1e-9 and abs(qy) < 1e-9):
        raise ValueError(
            f"{source}: 'table' must be a cuboid whose rotation is a pure z-yaw "
            f"(quat x/y must be 0, got x={qx:g} y={qy:g}) — "
            f"config.sync_support_to_target() derives the top surface from "
            f"position[2] + dimensions[2]/2, which tilting would silently break")

    placements = {}
    for key, raw in (data.get("object_placements") or {}).items():
        placements[key] = _parse_placement(raw, key)

    return {
        "version": SCHEMA_VERSION,
        "name": data.get("name") or source,
        "target_object": target,
        "obstacles": obstacles,
        "object_placements": placements,
    }


def load_scene(name_or_path=None) -> dict:
    """씬 YAML → 검증·numpy 화된 dict. config 를 건드리지 않는다."""
    import yaml  # 리포 관례: yaml import 는 함수 스코프

    path = resolve_scene_path(name_or_path)
    with open(path) as fh:
        data = yaml.safe_load(fh)
    scene = parse_scene(data, str(path))
    scene["path"] = str(path)
    return scene


def load_snapshot(snap: dict) -> dict:
    """h5 metadata 에 박제된 스냅샷 → 씬 dict. 파일을 읽지 않는다(재현 전용)."""
    scene = parse_scene(snap, "scene snapshot")
    scene["path"] = None
    return scene


def snapshot(scene: dict) -> dict:
    """씬 dict → JSON 직렬화 가능한 스냅샷. glns/storage.py 가 자동으로 json.dumps 한다."""
    def _plain(d):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    return {
        "version": SCHEMA_VERSION,
        "name": scene["name"],
        "target_object": _plain(scene["target_object"]),
        "obstacles": [_plain(o) for o in scene["obstacles"]],
        "object_placements": {k: _plain(v) for k, v in scene["object_placements"].items()},
    }


# ── config 심볼 채우기 ───────────────────────────────────────────────────────

def apply_to(cfg, scene: dict) -> None:
    """config 심볼을 **in-place** 로 채운다. 절대 rebind 하지 않는다.

    config.sync_support_to_target() 은 WALLS 안의 support dict 를 제자리에서 바꾸고,
    build_collision_world 와 trajectory_studio 의 그리기 루프가 그 dict 참조를 들고 있다.
    새 dict 로 갈아끼우면 support 갱신이 조용히 사라진다. 그래서 clear()+update() / [:]= 만 쓴다.
    """
    cfg.TARGET_OBJECT.clear()
    cfg.TARGET_OBJECT.update({k: (v.copy() if isinstance(v, np.ndarray) else v)
                              for k, v in scene["target_object"].items()})
    # 미등록 물체를 되돌릴 기준값 — TARGET_OBJECT 와 달리 아무도 덮어쓰지 않는다.
    cfg._SCENE_TARGET_DEFAULT.clear()
    cfg._SCENE_TARGET_DEFAULT.update({k: (v.copy() if isinstance(v, np.ndarray) else v)
                                      for k, v in scene["target_object"].items()})

    cfg.OBJECT_PLACEMENTS.clear()
    cfg.OBJECT_PLACEMENTS.update({
        k: {kk: (vv.copy() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()}
        for k, v in scene["object_placements"].items()})

    obstacles = [{k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in o.items()}
                 for o in scene["obstacles"]]
    cfg.OBSTACLES[:] = obstacles

    table = next(o for o in cfg.OBSTACLES if o["name"] == "table")
    mount = next(o for o in cfg.OBSTACLES if o["name"] == "robot_mount")
    cfg.TABLE.clear();       cfg.TABLE.update(table)
    cfg.ROBOT_MOUNT.clear(); cfg.ROBOT_MOUNT.update(mount)
    # OBSTACLES 안의 원소를 별칭 dict 자체로 바꿔 identity 를 하나로 만든다 —
    # 그래야 TABLE 을 고친 코드와 OBSTACLES 를 읽는 코드가 같은 것을 본다.
    cfg.OBSTACLES[cfg.OBSTACLES.index(table)] = cfg.TABLE
    cfg.OBSTACLES[cfg.OBSTACLES.index(mount)] = cfg.ROBOT_MOUNT

    cfg.WALLS[:] = [o for o in cfg.OBSTACLES
                    if o is not cfg.TABLE and o is not cfg.ROBOT_MOUNT]

    # 별칭이 깨지면 계획 시점에 조용히 틀리는 대신 여기서 죽는다.
    assert isinstance(cfg.WALLS, list), "robot.py 의 `+ config.WALLS` 가 진짜 list 를 요구한다"
    assert cfg.TABLE is cfg.OBSTACLES[0], "TABLE 별칭이 OBSTACLES 와 분리됐다"
    support = next(w for w in cfg.WALLS if w["name"] == "support")
    assert any(o is support for o in cfg.OBSTACLES), "support 별칭이 OBSTACLES 와 분리됐다"

    cfg.ACTIVE_SCENE = scene["name"]


# ── 기하 변환 (플래너·viser·Isaac 공통) ──────────────────────────────────────

def _quat_to_matrix(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def obstacle_obb(obstacle: dict):
    """장애물 → (position(3,), quat_wxyz(4,), dims(3,)) OBB.

    cuRobo 가 sphere/cylinder/capsule 을 충돌에 넣지 않으므로(모듈 docstring 참고)
    우리가 여기서 OBB 로 바꾼다. 플래너·viser·Isaac 이 모두 이 함수를 쓰기 때문에
    화면에 보이는 것이 곧 플래너가 푸는 것이다.
    """
    kind = obstacle["type"]
    pos = np.asarray(obstacle["position"], dtype=np.float64)
    quat = np.asarray(obstacle.get("rotation", _IDENTITY_QUAT), dtype=np.float64)

    if kind == "cuboid":
        return pos.copy(), quat.copy(), np.asarray(obstacle["dimensions"], dtype=np.float64)
    if kind == "sphere":
        d = 2.0 * float(obstacle["radius"])
        return pos.copy(), _IDENTITY_QUAT.copy(), np.array([d, d, d], dtype=np.float64)
    if kind == "cylinder":
        r, h = float(obstacle["radius"]), float(obstacle["height"])
        return pos.copy(), quat.copy(), np.array([2 * r, 2 * r, h], dtype=np.float64)
    if kind == "capsule":
        r = float(obstacle["radius"])
        base = np.asarray(obstacle["base"], dtype=np.float64)
        tip = np.asarray(obstacle["tip"], dtype=np.float64)
        center = pos + _quat_to_matrix(quat) @ ((base + tip) / 2.0)
        length = float(np.linalg.norm(tip - base)) + 2 * r
        return center, quat.copy(), np.array([2 * r, 2 * r, length], dtype=np.float64)
    raise ValueError(f"unknown obstacle type: {kind!r}")


def obstacle_trimesh(obstacle: dict):
    """viser 용 (mesh, position, quat_wxyz). obstacle_obb 와 같은 OBB 를 그린다 —
    플래너가 푸는 것과 화면이 어긋나지 않게."""
    import trimesh  # 함수 스코프: Isaac 런타임에서 import 되지 않게

    pos, quat, dims = obstacle_obb(obstacle)
    return trimesh.creation.box(extents=dims), pos, quat


def obstacle_yaml_snippet(name, position, dimensions, rotation=None) -> str:
    """Isaac 에서 잰 값을 씬 YAML 에 붙여넣을 수 있는 조각으로. 파일은 쓰지 않는다."""
    quat = _IDENTITY_QUAT if rotation is None else np.asarray(rotation, dtype=np.float64)
    fmt = lambda v: "[" + ", ".join(f"{float(x):.6f}" for x in v) + "]"  # noqa: E731
    return (f"  - name: {name}\n"
            f"    type: cuboid\n"
            f"    position:   {fmt(position)}\n"
            f"    dimensions: {fmt(dimensions)}\n"
            f"    rotation:   {fmt(quat)}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def add_cli_argument(parser) -> None:
    """--scene 을 한 곳에서 정의한다 — 진입점마다 따로 쓰면 조용히 어긋난다."""
    parser.add_argument(
        "--scene", type=str, default=None,
        help=f"검사 셀 씬 이름 또는 YAML 경로 (기본 {DEFAULT_SCENE}). "
             f"사용 가능: {', '.join(available_scenes()) or '(없음)'}")


def apply_cli(args, config_module) -> str:
    """--scene 을 config 에 반영하고 활성 씬 이름을 돌려준다.

    주의: 반드시 config.apply_object_placement() **이전**에 불러야 한다 —
    물체 배치가 이제 씬 소유이기 때문이다.

    기본 씬은 config import 시점에 이미 로드돼 있으므로, --scene 이 없으면 재로드하지 않는다
    (재로드하면 CLI override 로 덮어쓴 TARGET_OBJECT 가 조용히 되돌아간다).
    """
    name = getattr(args, "scene", None)
    if name is not None:
        config_module.load_scene(name)
    print(f"  Scene: {config_module.ACTIVE_SCENE} ({config_module.ACTIVE_SCENE_PATH})")
    return config_module.ACTIVE_SCENE
