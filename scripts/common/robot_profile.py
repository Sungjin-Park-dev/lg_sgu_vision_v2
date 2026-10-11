#!/usr/bin/env python3
"""로봇 프로필(workcell/robot/profiles/*.yaml) 의 스키마·로더·검증을 소유한다.

로봇 하나 = 자산 묶음 하나다: cuRobo 설정(yml → URDF), cuMotion 용 XRDF, Isaac USD 와 미리보기
ghost USD, UR 드라이버 모델명, 시작 자세. 이것들이 전역 상수와 파일 곳곳의 사본으로 흩어져
있으면 로봇을 바꿀 때 한 군데가 조용히 UR20 으로 남는다. 그래서 **씬이 로봇을 고르고**
(scene YAML 의 `robot:`), 그 이름의 프로필이 나머지를 정한다 — `--scene` 이 이미 모든 진입점·
서브프로세스·h5 스냅샷을 타고 다니므로 로봇도 같이 따라간다.

자산은 옮기지 않고 **파일명으로** 가리킨다(전부 workcell/robot/ 기준). USD 폴더를 옮기면 코어
MDL 상대경로가 깨진다.

관절 이름은 프로필에 쓰지 않는다 — cuRobo yml 의 ``cspace.joint_names`` 가 소유하고, 여기서
읽어 채운다(사본을 두면 언젠가 어긋난다).

의존성: numpy 와 (함수 스코프) yaml 만. Isaac Sim 런타임도 config 를 통해 이 모듈을 로드한다.
"""
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
ROBOT_DIR = PROJECT_ROOT / "workcell" / "robot"
PROFILES_ROOT = ROBOT_DIR / "profiles"

SCHEMA_VERSION = 1
# 로봇이 기록되지 않은 것(robot 키 이전의 씬·h5 스냅샷·IK 저장소·스캔 계획)은 전부 이 로봇으로
# 만들어졌다. 씬에 robot 이 없을 때의 기본값도 같다.
DEFAULT_ROBOT = "ur20"

# 프로필이 가리키는 자산 — 키 → 설명. 전부 ROBOT_DIR 기준 파일명이고 존재해야 한다.
ASSET_KEYS = {
    "curobo_config": "cuRobo 로봇 설정 (urdf 는 이 yml 의 urdf_path)",
    "xrdf": "cuMotion(MoveIt) 용 로봇 정의",
    "usd": "Isaac Sim 실로봇 USD",
    "ghost_usd": "Isaac 미리보기 ghost USD (build_ghost_usd.py 산출물)",
}


def available_robots() -> list:
    return sorted(p.stem for p in PROFILES_ROOT.glob("*.yaml"))


def resolve_profile_path(name) -> Path:
    """프로필 **이름** → workcell/robot/profiles/{name}.yaml.

    경로는 받지 않는다(씬과 다르다) — 로봇은 이름으로 h5 스냅샷·IK 저장소·스캔 계획에 기록되고
    나중에 그 이름으로 다시 찾으므로, 이름만으로 늘 같은 파일이 나와야 한다.
    """
    if name is None:
        name = DEFAULT_ROBOT
    name = str(name)
    if "/" in name or name.endswith((".yaml", ".yml")):
        raise ValueError(f"robot must be a profile name in {PROFILES_ROOT} (got {name!r})")
    candidates = [PROFILES_ROOT / f"{name}.yaml", PROFILES_ROOT / f"{name}.yml"]
    for c in candidates:
        if c.is_file():
            return c
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"robot profile '{name}' not found. tried:\n  {tried}\n"
        f"available: {', '.join(available_robots()) or '(none)'}")


def _load_yaml(path: Path):
    import yaml  # 리포 관례: 함수 스코프

    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with open(path) as fh:
        return yaml.load(fh, Loader=loader)


def _curobo_joint_names(yml_path: Path) -> list:
    data = _load_yaml(yml_path) or {}
    kin = (data.get("robot_cfg") or data).get("kinematics") or {}
    names = (kin.get("cspace") or {}).get("joint_names")
    if not names:
        raise ValueError(f"{yml_path}: robot_cfg.kinematics.cspace.joint_names is missing")
    return [str(n) for n in names]


def load_profile(name=None, check_files: bool = True) -> dict:
    """프로필 YAML → 검증된 dict. config 를 건드리지 않는다.

    check_files=False 는 자산을 **만드는** 도구용이다(예: build_ghost_usd 가 아직 없는 ghost_usd
    를 쓸 자리). 파일 존재·관절 이름 검사를 건너뛴다 — 파이프라인은 항상 기본값(True)으로 읽는다.
    """
    path = resolve_profile_path(name)
    data = _load_yaml(path)
    src = str(path)
    if not isinstance(data, dict):
        raise ValueError(f"{src}: top level must be a mapping")
    if data.get("version") != SCHEMA_VERSION:
        raise ValueError(f"{src}: version must be {SCHEMA_VERSION} (got {data.get('version')!r})")
    name = str(data.get("name") or path.stem)
    if name != path.stem:
        raise ValueError(f"{src}: name '{name}' must match the file name '{path.stem}'")

    profile = {"name": name, "path": src, "ur_type": str(data.get("ur_type") or name)}
    for key, what in ASSET_KEYS.items():
        fname = data.get(key)
        if not fname:
            raise ValueError(f"{src}: '{key}' is required ({what})")
        if check_files and not (ROBOT_DIR / str(fname)).is_file():
            raise FileNotFoundError(f"{src}: {key} → {ROBOT_DIR / str(fname)} does not exist ({what})")
        profile[key] = str(fname)

    if not check_files:
        profile["joint_names"] = []
        profile["start_state_deg"] = [float(v) for v in data.get("start_state_deg") or []]
        return profile
    profile["joint_names"] = _curobo_joint_names(ROBOT_DIR / profile["curobo_config"])
    start = data.get("start_state_deg")
    if not isinstance(start, (list, tuple)) or len(start) != len(profile["joint_names"]):
        raise ValueError(
            f"{src}: start_state_deg must list {len(profile['joint_names'])} values "
            f"(one per joint: {', '.join(profile['joint_names'])})")
    profile["start_state_deg"] = [float(v) for v in start]
    return profile


def asset_path(profile: dict, key: str) -> Path:
    """프로필 자산의 절대경로."""
    return ROBOT_DIR / profile[key]


def apply_to(cfg, profile: dict) -> None:
    """config 의 로봇 심볼을 **제자리에서** 채운다(rebind 하지 않는다).

    ROBOT_JOINT_NAMES·ROBOT_START_STATE 는 모듈 수명 동안 같은 객체다 — 다른 모듈이 import 시점에
    별칭으로 붙잡아 둬도(예: isaac_pipeline.JOINT_NAMES) 씬이 고른 로봇을 따라간다.

    ROBOT_START_STATE 는 배열 내용만 바꾼다 — 길이가 다른 로봇(7축 등)은 지금 지원하지 않는다.
    """
    q = np.deg2rad(np.asarray(profile["start_state_deg"], dtype=np.float64))
    if q.shape != cfg.ROBOT_START_STATE.shape:
        raise ValueError(
            f"robot '{profile['name']}' has {len(q)} joints but the pipeline assumes "
            f"{len(cfg.ROBOT_START_STATE)}")
    cfg.ROBOT_JOINT_NAMES[:] = profile["joint_names"]
    cfg.ROBOT.clear()
    cfg.ROBOT.update({**profile, "joint_names": cfg.ROBOT_JOINT_NAMES})   # 같은 list 객체
    cfg.ROBOT_START_STATE[:] = q
