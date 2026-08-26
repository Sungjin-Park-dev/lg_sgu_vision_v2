 #!/usr/bin/env python3
import sys

import numpy as np
from pathlib import Path

from common import scene_config

# 씬 로더가 이 모듈의 심볼을 in-place 로 갱신한다(scene_config.apply_to).
_self = sys.modules[__name__]

# ============================================================================
# 프로젝트 경로
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# ============================================================================
# 카메라 사양
# ============================================================================

# 용어·기준점 표준: docs/reference/camera-geometry.md (단일 진실원)

# FOV_footprint 가정 (mm). ⚠️ 실제 광학값 아님 — scene.py 의 "footprint 트릭"용 입력.
#   (focalLength=frame_standoff, aperture=이 값 으로 넣어 작업거리에서 프러스텀이 이 크기를 덮게 함)
#   실 센서(AR0820 8.08×4.55mm)와 다름. viewpoint col_spacing 계산에도 쓰임.
CAMERA_FOV_WIDTH_MM = 50.0
CAMERA_FOV_HEIGHT_MM = 50.0

# WD = frame_standoff (mm): optical_frame(=body_face) → object_plane 거리.
#   optical_frame 이 카메라 몸체 앞면에 있으므로 이 값이 곧 **벤더 공칭 WD** 다 (2026-07-27 정렬).
#   CAD(camera_asm_wo_light.stp) 실측: body_face=flange+141.0, VIEW_1(검사면)=flange+391.0
#   → 391.0 - 141.0 = 250.0. poses.py 가 이 값으로 viewpoint 를 표면에서 띄운다.
#   바꾸면 물체면이 실제로 이동 → viewpoint h5 재생성 + 도달성/충돌 재검증 필요.
CAMERA_WORKING_DISTANCE_MM = 250.0

# 카메라 뷰 유효 면적 (0.5 = 50% 중첩)
CAMERA_OVERLAP_RATIO = 0.5

# Isaac Sim 렌더/퍼블리시 해상도 — **FOV 종횡비에서 유도**한다.
# USD 카메라는 세로 화각을 렌더 해상도 비율에서 다시 계산한다(verticalAperture 는 사실상
# 무시). 그래서 해상도 비율이 FOV 비율과 다르면 퍼블리시된 이미지가 FOV_H 를 덮지 않는다 —
# 50×50 FOV 를 1024×750 으로 렌더하니 세로가 36.6mm 밖에 안 나왔다.
# 픽셀 수는 기존(1024×750)과 같게 유지한다 — 풀해상도는 렉이 걸려 다운샘플한 값이라
# 비율을 맞추자고 렌더 부하를 늘리면 그 튜닝을 깨뜨린다. 부하 조절은 이 예산으로 한다.
CAMERA_PUBLISH_PIXEL_BUDGET = 1024 * 750


def publish_resolution(fov_w_mm: float, fov_h_mm: float) -> tuple[int, int]:
    """FOV 비율은 맞추고 픽셀 수는 예산에 맞춘 (W, H). 8의 배수로 정렬한다."""
    aspect = float(fov_w_mm) / float(fov_h_mm)
    width = (CAMERA_PUBLISH_PIXEL_BUDGET * aspect) ** 0.5
    align = lambda v: max(8, int(round(v / 8.0)) * 8)
    w = align(width)
    return w, align(w / aspect)


CAMERA_PUBLISH_W, CAMERA_PUBLISH_H = publish_resolution(
    CAMERA_FOV_WIDTH_MM, CAMERA_FOV_HEIGHT_MM)

# Isaac Sim 검사 카메라 — ROS2 토픽/프레임
INSPECTION_CAMERA_FRAME_ID = "inspection_camera"
INSPECTION_CAMERA_RGB_TOPIC = "/inspection_camera/image_raw"
INSPECTION_CAMERA_DEPTH_TOPIC = "/inspection_camera/depth"
INSPECTION_CAMERA_INFO_TOPIC = "/inspection_camera/camera_info"

# MoveIt(cuMotion) 연동 — isaac_ros-dev 의 ur.ros2_control.xacro 와 토픽명이 일치해야 함.
# (TopicBasedSystem: joint_commands_topic=/isaac_joint_commands,
#  joint_states_topic=/isaac_joint_states)
MOVEIT_JOINT_COMMANDS_TOPIC = "/isaac_joint_commands"   # ROS→Isaac (MoveIt 위치 명령)
MOVEIT_JOINT_STATES_TOPIC = "/isaac_joint_states"       # Isaac→ROS (로봇 상태 피드백)

# 로봇 시작 자세 (radian). **한 값이 두 역할을 겸한다** — 나중에 나눌 여지가 있다:
#   셀 종속  : HOME 자세 (verify/motion 의 transit 브래킷, Return to HOME 목표, 스튜디오 초기 렌더)
#   알고리즘 : GLNS 의 reference_joints (IK 후보 선택 기준), [-1] = scan 구간 wrist_3 잠금값
# 현장에서 HOME 을 바꾸면 IK 후보 선택 기준까지 같이 바뀐다는 뜻이다. 지금은 값이 하나라
# 여기 두지만, 도달성이 예상과 다르게 움직이면 이 결합을 먼저 의심할 것.
# ROBOT_START_STATE = np.array([-1.67422354221344, -1.216842532157898, 1.6096495389938354, -2.0281713008880615, -1.5707969665527344, -0.031])
# ROBOT_START_STATE = np.array([-1.6007, -1.7271, -2.203, -0.808, 1.5951, -0.031])

# ROBOT_START_STATE = np.array([-2.0, -1.6, -1.8, -0.7, 1.8, -0.031])
# ROBOT_START_STATE = np.deg2rad([-270, -90, 60, -90, -90, 0])

# 실제 로봇 현재 자세 기준 (rad)
# ROBOT_START_STATE = np.array([1.5, -1.5, 2.0, -0.5, 1.5, 0.0])

# 실제 로봇 현재 자세 기준 1 (deg)
ROBOT_START_STATE = np.deg2rad([3.59, -111.84, -92.06, -66.09, 90.06, -99.61])

# 실제 로봇 현재 자세 기준 2 (deg)
# ROBOT_START_STATE = np.deg2rad([27.22, -114.96, -87.35, -67.68, 90.07, -75.98])


# 1.4753616491900843,-1.4261000792132776,2.299572706222534,-0.4354444742202759,1.4843419233905237,0.0,-0.15000295639038086,0.8933659791946411,0.222349613904953,-0.017051808536052704,-0.8432048559188843,-0.5372121930122375,0.010853501968085766

# ROBOT_START_STATE = np.deg2rad([-90, -120, -60, -90, 90, 0])


# ============================================================================
# 월드 설정 (robot base_link frame, 미터 단위)
# ============================================================================
#
# 값은 이 파일이 아니라 **씬 YAML(workcell/scenes/{name}.yaml)** 이 소유한다.
# 스키마·검증은 common/scene_config.py 가 소유하고, 이 모듈은 소비자가 이미 import 하는
# façade 로 남는다 — 어떤 소비자도 YAML 을 직접 파싱하지 않는다.
#
# 좌표계 주의:
# 위치/치수는 모두 robot base_link frame 기준 (cuRobo 충돌 입력용).
# **모든 좌표가 robot base_link frame 이다** — Isaac world 도 같다(2026-08-22).
# 원점 = 로봇 베이스 판(마운트 기둥 상면), 바닥은 z = -MOUNT_HEIGHT.
# 예외는 UR 컨트롤러의 'base' 프레임뿐이다(펜던트·RTDE·아루코). base_link 와 Z 축 180° 차이라
# 값을 그대로 넣으면 정반대에 놓인다 — 입력 경계에서 한 번만 변환한다
# (isaac_pipeline 의 Set Pose 프레임 토글).
#
# ⚠️ 아래 컨테이너들은 모듈 수명 동안 **같은 객체**로 유지된다. load_scene() 은 rebind 하지
#    않고 내용만 갈아끼운다(scene_config.apply_to 참고). sync_support_to_target() 이 WALLS
#    안의 support dict 를 제자리에서 바꾸고, build_collision_world 와 trajectory_studio 가
#    그 참조를 들고 있기 때문이다.

# 마운트 기둥 높이 (m) = 바닥 → 로봇 베이스 판. **씬 YAML 이 소유한다** —
# robot_mount 장애물의 dimensions[2] 가 곧 이 값이고, load_scene() 이 여기 채운다.
# 예전엔 여기 리터럴 0.805 가 있어 YAML 과 같은 숫자가 두 곳에 살았다: 한쪽만 고치면
# 기둥 상면이 로봇 베이스(z=0)와 조용히 어긋났다. 이제 출처가 하나다.
# (프레임 오프셋이 아니다 — 좌표계는 base_link 하나뿐이고 Isaac world 도 같다.
#  쓰이는 곳은 기둥의 치수로서다: mount USD 의 z 스케일, 바닥/환경을 내리는 양.)
MOUNT_HEIGHT = 0.0

DEFAULT_SCENE = scene_config.DEFAULT_SCENE
ACTIVE_SCENE = None          # 현재 로드된 씬 이름
ACTIVE_SCENE_PATH = None     # 그 씬의 파일 경로 (스냅샷 재현 시 None)

TARGET_OBJECT: dict = {}     # 대상 물체 pose (name/position/rotation)
# 씬이 준 물체 기본 pose 의 **불변 사본**. TARGET_OBJECT 는 배치/기즈모/CLI 가 계속
# 덮어쓰므로, 미등록 물체를 기본값으로 되돌리려면 별도 보관이 필요하다.
_SCENE_TARGET_DEFAULT: dict = {}
ENVIRONMENT: dict = {}       # 실험실 방 USD 의 pose (시각 전용, 충돌 월드 밖)
OBJECT_PLACEMENTS: dict = {}  # 물체별 배치 override
OBSTACLES: list = []         # 씬 순서 그대로의 전체 장애물 — 신규 정식 API
TABLE: dict = {}             # OBSTACLES 안의 그 dict 자체 (별칭)
ROBOT_MOUNT: dict = {}       # 〃
WALLS: list = []             # OBSTACLES - {table, robot_mount}. 기존 소비자 호환용 별칭


def load_scene(name_or_path=None) -> str:
    """씬 YAML 을 로드해 이 모듈의 월드 심볼을 갱신한다. 활성 씬 이름을 돌려준다."""
    global ACTIVE_SCENE_PATH
    scene = scene_config.load_scene(name_or_path)
    scene_config.apply_to(_self, scene)
    ACTIVE_SCENE_PATH = scene["path"]
    return ACTIVE_SCENE


def load_scene_snapshot(snap: dict) -> str:
    """h5 에 박제된 스냅샷으로 월드를 재현한다(파일을 읽지 않는다)."""
    global ACTIVE_SCENE_PATH
    scene = scene_config.load_snapshot(snap)
    scene_config.apply_to(_self, scene)
    ACTIVE_SCENE_PATH = None
    return ACTIVE_SCENE


def scene_snapshot() -> dict:
    """현재 월드를 JSON 직렬화 가능한 스냅샷으로. 해 h5 에 박제해 재현에 쓴다."""
    return scene_config.snapshot({
        "name": ACTIVE_SCENE,
        "target_object": TARGET_OBJECT,
        "environment": ENVIRONMENT,
        "obstacles": OBSTACLES,
        "object_placements": OBJECT_PLACEMENTS,
    })


# 물체별 충돌 형상 override (build_collision_world 가 참조).
# 씬이 아니라 **물체 자체의 속성**이라 셀을 바꿔도 안 변한다 — 그래서 씬 YAML 로 옮기지 않는다.
# cuRobo mesh 충돌은 최소 bbox 치수 ≲5cm 인 작은 메시를 **모든** 로봇 자세에 대해 충돌로 오판한다
# (0.5m 떨어진 home 자세조차 충돌 → IK 후보 전멸 → "No reachable viewpoints"). 해당 물체는
# mesh 대신 analytic primitive 로 충돌을 표현한다. "box" = mesh bbox 를 Cuboid(obb)로 — 모든 충돌
# consumer(IK/transit/verify)에서 확실히 반영. dims/center 는 mesh bbox 에서 자동 산출(메시 바뀌어도
# 추적). 표에 없는 물체(curved_structure/sample 등 충분히 큰 물체)는 기존대로 mesh 를 그대로 쓴다.
OBJECT_COLLISION_SHAPE = {
    "cylinder_sample": "box",  # Ø46×81mm — mesh 충돌 오판 회피용 bbox proxy
}

# 속이 빈(hollow) 물체 viewpoint 필터 override.
# 표면 샘플링은 안쪽 면까지 뽑아 viewpoint 가 공동 안에 생긴다(예: square_structure = 속 빈 상자).
# 여기 등록된 물체는 생성 후 convex-hull 법선 정렬 필터로 안쪽 껍데기 viewpoint 를 제거하고
# **바깥 껍데기만** 남긴다(위에서 안쪽 바닥을 내려다보는 것까지 제거). viewpoint_studio 와 CLI 가
# 참조해 ViewpointGenParams.filter_interior 를 켠다.
#   hull_align_min: 표면 법선 vs 최근접 convex-hull 바깥법선 정렬(cos) 임계. 미만이면 안쪽 면=제거.
# 주의: 오목한 '바깥' 형상(홈/계단)이 있는 물체엔 부적합 — box 류에만 opt-in.
OBJECT_FILTER_INTERIOR = {
    "square_structure": {"hull_align_min": 0.3},
}

# 물체별 기본 타깃 머티리얼 RGB ("R,G,B"). 지정 시 그 재질 면만 샘플링한다.
# 컨벤션: 초록(0,255,0) = 검사대상. 회색(170,163,158)은 비대상이라 제외.
# (source.obj usemtl 스왑으로 대상 평면을 초록으로 통일)
# ⚠️ 이 표를 안 보고 viewpoint 를 만들면 조용히 틀린 개수가 나온다 — sample 은 74 대신 161.
#    viewpoint_studio 와 viewpoint/cli.py 가 같은 표를 봐야 하는 이유다(예전엔 studio 에만 있어
#    CLI 는 사람이 --material-rgb 를 기억해 넘겨야 했다).
OBJECT_TARGET_MATERIAL = {
    "sample": "0,255,0",
}


def apply_object_placement(object_name):
    """object_name 의 배치를 TARGET_OBJECT/support 에 in-place 반영(robot frame).

    각 진입점에서 CLI override 전에 호출 → 다운스트림(build_camera_poses / build_collision_world /
    isaac scene)이 read 시점에 per-object 배치를 본다.

    표에 없는 물체는 **씬의 target_object 기본값으로 되돌리고** False 를 반환한다. 예전에는
    아무것도 안 하고 False 만 돌려줬는데, 그러면 TARGET_OBJECT 가 직전 물체 pose 로 오염된 채
    남는다 — studio 에서 물체를 갈아끼우면 미등록 물체가 앞 물체의 자리에 놓였다. 반환값 계약
    (표에 있었나?)은 그대로라 호출부 9곳의 로그 로직은 안 바뀐다.
    """
    p = OBJECT_PLACEMENTS.get(object_name)
    if p is None:
        TARGET_OBJECT["position"] = np.asarray(
            _SCENE_TARGET_DEFAULT["position"], dtype=np.float64).copy()
        TARGET_OBJECT["rotation"] = np.asarray(
            _SCENE_TARGET_DEFAULT["rotation"], dtype=np.float64).copy()
        sync_support_to_target()
        return False
    if "position" in p:
        TARGET_OBJECT["position"] = np.asarray(p["position"], dtype=np.float64).copy()
    TARGET_OBJECT["rotation"] = np.asarray(
        p.get("rotation", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64).copy()
    sync_support_to_target()
    return True


def sync_support_to_target():
    """Support가 테이블 상면과 물체 바닥 사이를 채우도록 배치한다."""
    support = next(w for w in WALLS if w["name"] == "support")
    table_top_z = float(TABLE["position"][2] + TABLE["dimensions"][2] / 2.0)
    object_bottom_z = float(TARGET_OBJECT["position"][2])
    height = object_bottom_z - table_top_z
    if height <= 0.0:
        raise ValueError(
            f"[scene '{ACTIVE_SCENE}'] Target object bottom z ({object_bottom_z:.4f}) must be "
            f"above table top z ({table_top_z:.4f}) — check the scene's "
            f"object_placements or table height"
        )

    support["position"] = np.array([
        TARGET_OBJECT["position"][0],
        TARGET_OBJECT["position"][1],
        table_top_z + height / 2.0,
    ], dtype=np.float64)
    support["dimensions"] = np.array([
        support["dimensions"][0],
        support["dimensions"][1],
        height,
    ], dtype=np.float64)
    return support


load_scene(DEFAULT_SCENE)


# ============================================================================
# 로봇 설정
# ============================================================================

# cuRobo와 EAIK에서 사용되는 로봇 설정 파일.
# URDF 경로는 여기 없다 — 이 yml 의 urdf_path 가 소유하고 robot.resolve_robot_config() 가
# workcell/robot/ 절대경로로 푼다(현재 ur20_with_camera_curobo.urdf).
DEFAULT_ROBOT_CONFIG = "ur20_with_camera.yml"

# mount_offset (m): flange → optical_frame 거리. 용어: docs/reference/camera-geometry.md
# optical_frame = 카메라 몸체 앞면(body_face). CAD 실측 flange+141.0mm.
# ⚠️ 하드웨어 상수 — 튜닝 대상이 아니다. 기하를 실제로 만드는 것은 URDF(camera_optical_joint)와
#    USD 지만, 이 상수는 **참고용 사본이 아니다**: 바로 아래 CAMERA_MIN_WORKING_DISTANCE_MM
#    (WD 검증 하한)과 CAMERA_NEAR_CLIP_M(렌더 near clip)이 여기서 파생된다. 셋이 어긋나면
#    검증과 렌더가 로봇과 다른 카메라를 가정하게 된다.
#    바꿀 때는 URDF·USD·이 상수를 함께 — 체크리스트는 docs/reference/robot-camera-assets.md.
TOOL_TO_CAMERA_OPTICAL_OFFSET_M = 0.141

# flange → 렌즈 배럴 끝 (CAD 실측, docs/reference/camera-geometry.md §A).
# 같은 값이 build_camera_mesh.EXPECT_HI[0] 와 inspect_camera_step.EXPECT 에도 있는데,
# 그 둘은 CAD 를 검증하는 독립 assert 라 검증 대상을 import 하면 의미를 잃는다 — 의도된 중복.
CAMERA_LENS_FRONT_OFFSET_M = 0.21877

# WD 하한 (mm): 이보다 작으면 검사면(= mount_offset + WD)이 렌즈 앞면보다 뒤 —
# 기하학적으로 불가능하다. 77.77mm = 배럴 길이. viewpoint 생성/로드에서 이 값으로 검증한다.
CAMERA_MIN_WORKING_DISTANCE_MM = (
    CAMERA_LENS_FRONT_OFFSET_M - TOOL_TO_CAMERA_OPTICAL_OFFSET_M
) * 1000.0

# 렌더 카메라 near clip (m). optical_frame 이 body_face 로 내려오면서 카메라 원점이
# **자기 렌즈 배럴 안**에 들어갔다 — 카메라 앞 77.8mm 까지가 배럴 내부다. near 를 그 너머로
# 두지 않으면 렌더 화면이 배럴로 가득 찬다(실제 카메라도 자기 배럴은 보지 못한다).
# 배럴 끝에 딱 맞추면 얇은 테두리가 남을 수 있어 2mm 여유를 준다.
CAMERA_NEAR_CLIP_M = (CAMERA_MIN_WORKING_DISTANCE_MM + 2.0) / 1000.0
CAMERA_FAR_CLIP_M = 5.0


def working_distance_error(wd_mm: float) -> str | None:
    """WD(mm)가 기하학적으로 불가능하면 사유 문자열, 정상이면 None.

    검사면은 flange 기준 ``mount_offset + WD`` 에 놓인다. 그게 렌즈 앞면보다 뒤라면 물체가
    렌즈 배럴 안에 있다는 뜻이라 어떤 배치로도 성립하지 않는다.

    "config 값과 다른가"를 보지 않는 이유: WD 는 카메라 스펙에 따라 조절하는 값이라
    기본값과 다른 것 자체는 결함이 아니다. 대신 물리적으로 불가능한 값만 잡는다
    (구 optical_frame 0.346 시절의 h5 는 WD 46mm 라 여기 걸린다).

    반환 문자열은 채널을 정하지 않는다 — 읽는 쪽이 print / parser.error / GUI 로 각자 쓴다.
    """
    if wd_mm > CAMERA_MIN_WORKING_DISTANCE_MM:
        return None
    object_plane_mm = (TOOL_TO_CAMERA_OPTICAL_OFFSET_M * 1000.0) + wd_mm
    return (
        f"working distance {wd_mm:.1f}mm 는 불가능하다 — 검사면이 flange+{object_plane_mm:.1f}mm 로 "
        f"렌즈 앞면(flange+{CAMERA_LENS_FRONT_OFFSET_M * 1000.0:.1f}mm)보다 뒤에 있다. "
        f"최소 {CAMERA_MIN_WORKING_DISTANCE_MM:.1f}mm. "
        f"optical_frame 이전(2026-07-27) 전에 만든 파일이면 재생성할 것 "
        f"(docs/reference/camera-geometry.md)."
    )


# ============================================================================
# 충돌 검사 파라미터
# ============================================================================

COLLISION_MARGIN = 0.0
# COLLISION_ADAPTIVE_MAX_JOINT_STEP_DEG = 0.05  # 1 step 당 최대 joint 변화량
COLLISION_ADAPTIVE_MAX_JOINT_STEP_DEG = 0.02  # 1 step 당 최대 joint 변화량
COLLISION_INTERP_EXCLUDE_LAST_JOINT = True # End-Effector 회전 무시


# ============================================================================
# 객체 기반 데이터 경로 헬퍼 함수
# ============================================================================

def get_mesh_path(object_name: str, filename: str = None, mesh_type: str = "target") -> Path:
    """
    객체 메시 파일 경로 반환

    Args:
        object_name: 객체 이름 (예: "glass", "phone")
        filename: 명시적 메시 파일명 (지정 시 mesh_type 무시)
        mesh_type: 메시 파일 유형 (기본값: "target")
            - "source": source.obj (충돌 검사용 전체 멀티 머티리얼 메시)
            - "target": target.ply (뷰포인트 샘플링용 검사 표면)

    Returns:
        메시 파일 경로: data/{object_name}/mesh/{filename}

    Examples:
        >>> get_mesh_path("glass")  # 기본값: 타겟 메시
        PosixPath('data/glass/mesh/target.ply')  # .ply가 없으면 target.obj

        >>> get_mesh_path("glass", mesh_type="source")  # 충돌용 전체 메시
        PosixPath('data/glass/mesh/source.obj')

        >>> get_mesh_path("glass", filename="custom.obj")  # 명시적 파일명
        PosixPath('data/glass/mesh/custom.obj')
    """
    if filename is None:
        # mesh_type에 따라 파일명 자동 결정
        if mesh_type == "source":
            filename = "source.obj"
        elif mesh_type == "target":
            # target.ply 우선 시도 (검사용 선호), target.obj로 폴백
            target_ply = DATA_ROOT / object_name / "mesh" / "target.ply"
            if target_ply.exists():
                return target_ply
            filename = "target.obj"
        else:
            raise ValueError(f"invalid mesh_type: '{mesh_type}'. Expected 'source' or 'target'")

    return DATA_ROOT / object_name / "mesh" / filename


def get_viewpoint_path(object_name: str, num_viewpoints: int, filename: str = "viewpoints.h5") -> Path:
    """
    뷰포인트 파일 경로 반환

    Args:
        object_name: 객체 이름 (예: "glass")
        num_viewpoints: 뷰포인트 개수
        filename: 파일명 (기본값: "viewpoints.h5")

    Returns:
        뷰포인트 경로: data/{object_name}/viewpoint/{num_viewpoints}/{filename}

    Example:
        >>> get_viewpoint_path("glass", 500)
        PosixPath('data/glass/viewpoint/500/viewpoints.h5')
    """
    return DATA_ROOT / object_name / "viewpoint" / str(num_viewpoints) / filename


def resolve_viewpoint_path(object_name: str, num_viewpoints: int) -> Path:
    """읽을 viewpoints h5 를 고른다 — writer 가 방법 접미사를 붙이기 때문.

    생성기는 ``viewpoints_{clustering_method}.h5`` 로 저장하므로 정규 이름
    ``viewpoints.h5`` 는 보통 존재하지 않는다. 있으면 그것을 쓰고, 없으면
    ``viewpoints*.h5`` 중 가장 최근 것을 쓴다. 어떤 방법으로 만든 파일이든 스키마가
    같으므로 읽는 쪽은 구분할 필요가 없다.

    Raises:
        FileNotFoundError: 후보가 하나도 없을 때 (탐색한 디렉토리를 함께 알린다).
    """
    directory = DATA_ROOT / object_name / "viewpoint" / str(num_viewpoints)
    canonical = directory / "viewpoints.h5"
    if canonical.exists():
        return canonical
    # 아래는 옛 파일(viewpoints_{clustering_method}.h5)을 위한 폴백이다. 지금 생성기는
    # 항상 정규 이름으로 쓰므로 후보가 하나뿐이고, mtime 이 다음 단계 입력을 정하는
    # 일은 생기지 않는다.
    candidates = sorted(directory.glob("viewpoints*.h5"))
    if not candidates:
        raise FileNotFoundError(
            f"No viewpoints*.h5 under {directory} — "
            f"generate them first (scripts/core/viewpoint/cli.py --object {object_name})."
        )
    chosen = max(candidates, key=lambda p: p.stat().st_mtime)
    # 옛 파일이 여럿 남아 있으면 mtime 이 정한다 — 조용히 고르면 어느 파일로 계획했는지
    # 모르는 채 진행하게 되므로 반드시 찍는다.
    if len(candidates) > 1:
        others = ", ".join(p.name for p in candidates if p != chosen)
        print(f"  viewpoints: {chosen.name} (newest) - other candidates in the same "
              f"folder: {others}")
    return chosen


def get_solution_path(object_name: str, num_viewpoints: int) -> Path:
    """GLNS 해 h5 — ``verify.py`` 의 입력이자 studio 시각화 소스.

    궤적 산출물과 같은 폴더에 둔다: 이 해로부터 궤적이 나오고, 둘의 수명이 같다.

        data/{object}/trajectory/{N}/solution.h5
    """
    return get_trajectory_path(object_name, num_viewpoints, "solution.h5")


def get_trajectory_artifact_path(object_name: str, num_viewpoints: int,
                                 role: str | None = None, suffix: str = ".csv") -> Path:
    """실행 궤적 산출물 경로.

    명명 규칙은 ``{역할}[_{세부}]`` 이고 백엔드 토큰은 없다 — 생산자가 GLNS 하나뿐이다.

        role=None                 → trajectory.csv        (스캔 궤적)
        role='home_to_start'      → trajectory_home_to_start.csv
        role='end_to_home'        → trajectory_end_to_home.csv
    """
    name = "trajectory" if not role else f"trajectory_{role}"
    return get_trajectory_path(object_name, num_viewpoints, f"{name}{suffix}")


def get_trajectory_path(object_name: str, num_viewpoints: int, filename: str = "gtsp.csv") -> Path:
    """
    궤적 파일 경로 반환

    Args:
        object_name: 객체 이름 (예: "glass")
        num_viewpoints: 뷰포인트 개수
        filename: 파일명 (기본값: "gtsp.csv", "gtsp_final.csv"도 가능)

    Returns:
        궤적 경로: data/{object_name}/trajectory/{num_viewpoints}/{filename}

    Example:
        >>> get_trajectory_path("glass", 500)
        PosixPath('data/glass/trajectory/500/gtsp.csv')
        >>> get_trajectory_path("glass", 500, "gtsp_final.csv")
        PosixPath('data/glass/trajectory/500/gtsp_final.csv')
    """
    return DATA_ROOT / object_name / "trajectory" / str(num_viewpoints) / filename
