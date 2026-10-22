"""MoveIt/cuMotion 이 요구하는 형식으로 리포 자산을 파생시킨다 (launch 가 부른다).

원본은 전부 workcell/ 에 있고 Inspection(cuRobo·Isaac·viser)이 쓰는 것과 같은 파일이다.
여기서 만드는 것은 **파생물**이라 리포에 두지 않고 매번 /tmp 에 쓴다 — 원본을 고친 뒤
재생성을 잊어 MoveIt 만 옛 것을 보는 일이 구조적으로 불가능하게 하려고.
(기존 launch 가 xacro → /tmp/collated_ur20_urdf.urdf 로 하던 것과 같은 패턴이다.)
"""

from __future__ import annotations

import importlib.util
import os
import re

import yaml

MOVEIT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_SCRIPTS = os.path.dirname(MOVEIT_DIR)
PROJECT_ROOT = os.path.dirname(PROJECT_SCRIPTS)

CUROBO_URDF = os.path.join(PROJECT_ROOT, 'workcell', 'robot', 'ur20_with_camera_curobo.urdf')
CAMERA_MESH_DIR = os.path.join(PROJECT_ROOT, 'workcell', 'robot', 'camera')
SOURCE_XRDF = os.path.join(PROJECT_ROOT, 'workcell', 'robot', 'ur20_with_camera.xrdf')
WORLD_GEOMETRY_NAME = 'world_collision_model'


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _collision_exclude_links() -> tuple:
    """Inspection 이 월드충돌에서 빼는 링크 목록. **같은 목록을 공유하려고** 읽어온다.

    core.trajectory 패키지로 import 하면 __init__ 이 cuRobo 까지 끌어오므로 파일로 연다.
    """
    settings = os.path.join(PROJECT_SCRIPTS, 'core', 'trajectory', 'settings.py')
    try:
        return tuple(_load_module(settings, 'pt_settings').COLLISION_EXCLUDE_LINKS)
    except Exception:                      # noqa: BLE001 — 못 읽으면 알려진 기본값으로
        return ('base_link_inertia',)


def prepare_kinematics_urdf(output_file: str) -> str:
    """workcell 의 URDF 를 MoveIt 이 include 할 수 있게 손봐 /tmp 에 둔다.

    **기하는 건드리지 않는다** — 카메라 위치·clocking·관절은 그 파일이 소유한다.
    고치는 둘은 기하가 아니다:
      1. 실기 드라이버용 ros2_control 제거. Isaac 은 TopicBasedSystem 이 필요하고,
         두 블록이 남으면 controller_manager 가 양쪽 하드웨어를 다 올리려 한다.
      2. 카메라 메시 package:// → 절대경로. 컨테이너의 ur_description 에는
         meshes/camera 가 없어 그 경로가 풀리지 않는다(팔 메시는 정상이다).
    """
    source = open(CUROBO_URDF).read()
    kinematics = re.sub(r'<ros2_control\b.*?</ros2_control>', '', source, flags=re.S)
    kinematics = kinematics.replace(
        'package://ur_description/meshes/camera/', f'file://{CAMERA_MESH_DIR}/')
    with open(output_file, 'w') as f:
        f.write(kinematics)
    return output_file


def prepare_xrdf(output_file: str, source: str = None) -> str:
    """월드충돌용 스피어 집합을 파생시킨 xrdf 를 /tmp 에 둔다.

    ``base_link_inertia`` 스피어는 자세와 무관하게 ``robot_mount`` 상면을 ~16mm 파고든다.
    로봇 베이스 바닥과 기둥 상면이 **정확히 같은 평면**(볼트 체결면)이라, 평평한 면을 구로
    덮는 한 반드시 튀어나온다 — 스피어 59개로 촘촘히 맞춰도 안 없어진다. 그대로 두면
    cuMotion 이 모든 시작 자세를 "world collision" 으로 거부한다.

    Inspection 은 settings.COLLISION_EXCLUDE_LINKS 로 같은 링크를 이미 빼고 있다.
    그 목록을 읽어 써서 두 계획기가 같은 기준을 갖게 한다.

    **자기충돌에서는 빼지 않는다** — 팔이 자기 베이스에 닿는 것은 실제로 일어난다
    (무작위 자세의 4.2%, forearm 과 최대 118mm). xrdf 가 collision 과 self_collision 에
    서로 다른 geometry 집합을 가리킬 수 있어서 가능하다(cuMotion 로드로 실측 확인).
    """
    data = yaml.safe_load(open(source or SOURCE_XRDF))
    excluded = _collision_exclude_links()
    source_name = data['collision']['geometry']
    geometry = data['geometry'][source_name]
    nested = 'spheres' in geometry
    spheres = geometry['spheres'] if nested else geometry

    world = {k: v for k, v in spheres.items() if k not in excluded}
    data['geometry'][WORLD_GEOMETRY_NAME] = {'spheres': world} if nested else world
    data['collision']['geometry'] = WORLD_GEOMETRY_NAME
    buffer = data['collision'].get('buffer_distance')
    if isinstance(buffer, dict):
        data['collision']['buffer_distance'] = {
            k: v for k, v in buffer.items() if k not in excluded}
    # self_collision 은 원래 집합(source_name)을 계속 가리킨다 — 손대지 않는다.

    with open(output_file, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=True)
    return output_file


def build_collision_scene(scene_name: str, output_file: str) -> str:
    """씬 YAML → cuMotion 이 읽는 .scene (빈 이름이면 skip).

    이 파일이 없으면 MoveIt/cuMotion 은 **장애물 0개인 빈 월드**로 계획한다 — 그런데
    StaticPlanningSceneServer 가 조용히 넘어가서 그 사실이 드러나지 않는다.
    (다른 월드 입구인 nvblox ESDF 는 launch 가 의도적으로 꺼둔다.)
    """
    if not scene_name:
        return ''
    builder = os.path.join(PROJECT_SCRIPTS, 'setup', 'build_moveit_scene.py')
    return _load_module(builder, 'build_moveit_scene').write_scene_file(
        scene_name, output_file)
