 #!/usr/bin/env python3
import numpy as np
from pathlib import Path

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

# frame_standoff (mm): optical_frame → object_plane 거리. poses.py 가 이 값으로 viewpoint 를
#   표면에서 띄운다 → 바꾸면 전체 기하 재생성. ⚠️ 이름은 "working distance"지만 광학 WD 아님.
#   같은 배치: body_face→object=251mm ≈ 벤더 메일(2026-07) WD 250mm / lens_front→object=173mm.
#   (WD 기준점 78mm·재생성 여부는 파킹 — camera-geometry.md §미해결)
CAMERA_WORKING_DISTANCE_MM = 46.0

# 카메라 뷰 유효 면적 (0.5 = 50% 중첩)
CAMERA_OVERLAP_RATIO = 0.5

# 센서 해상도 (pixels) + 픽셀 크기 (mm)
# ⚠️ placeholder — 실제 AR0820 native 는 3848×2168 @ 2.1µm (8.08×4.55mm). camera-geometry.md 참고.
CAMERA_RESOLUTION_W = 4096
CAMERA_RESOLUTION_H = 3000
CAMERA_PIXEL_SIZE_MM = 0.010

# Isaac Sim 렌더/퍼블리시 해상도 — 풀해상도는 렉 걸려 다운샘플
CAMERA_PUBLISH_W = 1024
CAMERA_PUBLISH_H = 750

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

# 로봇 제약 여부
ROBOT_HAS_CONSTRAINT = True

# 로봇 시작 자세 (UR 시뮬레이터 기본 자세, radian)
# ROBOT_START_STATE = np.array([-1.67422354221344, -1.216842532157898, 1.6096495389938354, -2.0281713008880615, -1.5707969665527344, -0.031])
# ROBOT_START_STATE = np.array([-1.6007, -1.7271, -2.203, -0.808, 1.5951, -0.031])

# ROBOT_START_STATE = np.array([-2.0, -1.6, -1.8, -0.7, 1.8, -0.031])
# ROBOT_START_STATE = np.deg2rad([-270, -90, 60, -90, -90, 0])

# 실제 로봇 현재 자세 기준 (rad)
ROBOT_START_STATE = np.array([1.5, -1.5, 2.0, -0.5, 1.5, 0.0])

# 1.4753616491900843,-1.4261000792132776,2.299572706222534,-0.4354444742202759,1.4843419233905237,0.0,-0.15000295639038086,0.8933659791946411,0.222349613904953,-0.017051808536052704,-0.8432048559188843,-0.5372121930122375,0.010853501968085766

# ROBOT_START_STATE = np.deg2rad([-90, -120, -60, -90, 90, 0])

# 조인트 최대 움직임
MAX_JOINT_FROM_START_STATE = np.deg2rad(90)


# ============================================================================
# 월드 설정 (Isaac Sim 좌표계, 미터 단위)
# ============================================================================

# 좌표계 주의:
# 본 config의 위치/치수는 모두 robot base_link frame 기준 (cuRobo 충돌 입력용).
# Isaac Sim visual 은 world frame (floor=0) 이며, robot base = world z=MOUNT_HEIGHT(0.805m).
# 따라서 robot frame z = world z - 0.805.

# 대상 객체 설정 — visual: world (-0.1, 1.1, 0.795), robot frame z = 0.795-0.805 = -0.010
# rotation = identity: 물체 방향은 config가 아니라 **메시에 베이크**한다
# (setup/prepare_object_mesh.py reorient).
# 그래야 viser(로컬 프레임)와 Isaac(월드 프레임)이 회전 차이 없이 동일하게 보임.
# (이전 [0.7071,0,0,0.7071]은 순수 z-yaw였음 → identity로 바꿔도 bottom 필터는 불변, sample은 90° yaw만 풀림.)
TARGET_OBJECT = {
    "name": "target_object",
    "position": np.array([-0.1, 1.1, -0.010], dtype=np.float64),
    "rotation": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),  # 쿼터니언: w, x, y, z (identity)
}

# robot base_link 가 world z=MOUNT_HEIGHT 에 있다 → robot frame z = world z - MOUNT_HEIGHT.
MOUNT_HEIGHT = 0.805

# 물체별 배치 (robot base_link frame). 도달성/스캔성을 최대화하는 위치·방향으로 측정 결정.
# 표에 없는 물체는 TARGET_OBJECT 기본값을 그대로 쓴다. position 은 robot frame [x,y,z](m),
# rotation 은 쿼터니언 [w,x,y,z]. rotation 생략 시 identity. apply_object_placement() 로 반영.
# 주의: rotation 을 여기서 주면 mesh-bake 와 달리 viewer 마다 적용이 필요 — viewpoint_studio /
# Isaac scene 은 config rotation 을 반영하도록 맞춰져 있다(둘 다 동일 외형). z-yaw 는 bottom-filter
# 가 불변이라 기존 viewpoint h5 재생성 불필요(비-z 회전은 재생성 필요).
# 아래 값은 기존 GLNS 배치 스윕에서 얻은 min-reconfig best.
# 실제 joined motion reconfig = scan reconfig + seam(=solved component 수-1) 을 최소화(1~2개 미커버 허용).
# 전부 base reconfig=0. 각 물체별 placement_sweep/summary 참조. (coverage-우선 best 는 커밋 이력/summary 참고)
OBJECT_PLACEMENTS = {
    # curved: 99/100, base0, scan reconfig2, 1 component(seam0) → 실제 ~2. (coverage-best 는 100/100·reconfig6)
    "curved_structure": {
        # "position": np.array([-0.175, 0.725, 0.19], dtype=np.float64),
        "position": np.array([-0.15, 0.741, 0.19], dtype=np.float64),
        
        # "rotation": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "rotation": np.array([0.70710678, 0.0, 0.0, 0.70710678], dtype=np.float64),  # z-yaw 90°
    },
    # cylinder: 22/22, base0, reconfig2, 1 component. 근접 y 경계(0.95) → 더 가까이면 유리할 수 있음.
    "cylinder_sample": {
        # "position": np.array([-0.025, 0.95, -0.010], dtype=np.float64),
        "position": np.array([-0.15, 0.741, 0.19], dtype=np.float64),
        "rotation": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    },
    # sample: 72/74, base0, scan reconfig0, 3 component(seam2) → 실제 ~2 + z-yaw90(scan-collision 회피).
    "sample": {
        # "position": np.array([-0.1, 0.8, -0.010], dtype=np.float64),
        "position": np.array([-0.15, 0.741, 0.19], dtype=np.float64),
        "rotation": np.array([0.70710678, 0.0, 0.0, 0.70710678], dtype=np.float64),  # z-yaw 90°
    },
    # square: 70/71, base0, reconfig3, 1 component. 격자 전역 평평(reconfig3 바닥) → 근접 y/높은 z 확장 시 개선.
    "square_structure": {
        # "position": np.array([-0.175, 1.1, -0.010], dtype=np.float64),
        # "rotation": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        
        "position": np.array([-0.121, 0.743, 0.359], dtype=np.float64),
        "rotation": np.array([0.92387953, 0.0, 0.0, 0.38268343], dtype=np.float64),  # z-yaw 45°
    },
}

# 충돌 검사용 물체 형상 override (build_collision_world 가 참조).
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


def apply_object_placement(object_name):
    """object_name 의 배치를 TARGET_OBJECT/support 에 in-place 반영(robot frame).

    각 진입점에서 CLI override 전에 호출 → 다운스트림(build_camera_poses / build_collision_world /
    isaac scene)이 read 시점에 per-object 배치를 본다. 표에 없으면 기본값 유지하고 False 반환.
    """
    p = OBJECT_PLACEMENTS.get(object_name)
    if p is None:
        return False
    if "position" in p:
        TARGET_OBJECT["position"] = np.asarray(p["position"], dtype=np.float64).copy()
    TARGET_OBJECT["rotation"] = np.asarray(
        p.get("rotation", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64).copy()
    sync_support_to_target()
    return True


def target_object_world_position():
    """TARGET_OBJECT position(robot frame) → Isaac world frame(z += MOUNT_HEIGHT)."""
    return np.asarray(TARGET_OBJECT["position"], dtype=np.float64) + np.array([0.0, 0.0, MOUNT_HEIGHT])

# 테이블 직육면체 설정 — thor_table.usd 측정값 매칭
# visual: world center (-0.2, 1.1, 0.315), size 0.910×0.768×0.630
# robot frame z = 0.315 - 0.805 = -0.490
TABLE = {
    "name": "table",
    "position": np.array([-0.2, 0.7, -0.490], dtype=np.float64),
    "dimensions": np.array([0.910, 0.768, 0.630], dtype=np.float64),
}

# 벽(펜스) 직육면체 설정 - 작업 공간을 둘러싼 4개의 벽
WALLS = [
    {
        "name": "wall_front",
        "position": np.array([0.0, 1.6, 0.5], dtype=np.float64),
        "dimensions": np.array([2.2, 0.1, 3.0], dtype=np.float64),
    },
    {
        "name": "wall_back",
        "position": np.array([0.0, -1.0, 0.5], dtype=np.float64),
        "dimensions": np.array([2.2, 0.1, 3.0], dtype=np.float64),
    },
    {
        "name": "wall_left",
        "position": np.array([-1.0, 0.25, 0.5], dtype=np.float64),
        "dimensions": np.array([0.1, 2.7, 3.0], dtype=np.float64),
    },
    {
        "name": "wall_right",
        "position": np.array([1.0, 0.25, 0.5], dtype=np.float64),
        "dimensions": np.array([0.1, 2.7, 3.0], dtype=np.float64),
    },
    {
        # Target object 받침대. 위치와 높이는 apply_object_placement()에서 물체별로 갱신.
        "name": "support",
        "position": np.array([-0.1, 1.1, -0.0925], dtype=np.float64),
        "dimensions": np.array([0.05, 0.05, 0.165], dtype=np.float64),
    },
]


def sync_support_to_target():
    """Support가 테이블 상면과 물체 바닥 사이를 채우도록 배치한다."""
    support = next(w for w in WALLS if w["name"] == "support")
    table_top_z = float(TABLE["position"][2] + TABLE["dimensions"][2] / 2.0)
    object_bottom_z = float(TARGET_OBJECT["position"][2])
    height = object_bottom_z - table_top_z
    if height <= 0.0:
        raise ValueError(
            f"Target object bottom z ({object_bottom_z:.4f}) must be above "
            f"table top z ({table_top_z:.4f})"
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

# 로봇 마운트(베이스) 설정 — ur10_mount.usd visual 매칭
# visual: world center (0, 0, 0.4025), size 0.54×0.54×0.805 (XY 2배 스케일 적용 후)
# robot frame z = 0.4025 - 0.805 = -0.4025, top at robot frame z=0 (= robot base)
ROBOT_MOUNT = {
    "name": "robot_mount",
    "position": np.array([0.0, 0.0, -0.4025], dtype=np.float64),
    "dimensions": np.array([0.54, 0.54, 0.805], dtype=np.float64),
}


# ============================================================================
# 로봇 설정
# ============================================================================

# cuRobo와 EAIK에서 사용되는 로봇 설정 파일
DEFAULT_ROBOT_CONFIG = "ur20_with_camera.yml"
DEFAULT_URDF_PATH = "/curobo/src/curobo/content/assets/robot/ur_description/ur20_with_camera.urdf"

# mount_offset (m): flange → optical_frame 거리. 용어: docs/reference/camera-geometry.md
# ⚠️ optical_frame(0.346)은 실제 렌즈앞면(0.219)보다 127mm 앞 허공 — 낡은 기준(파킹).
TOOL_TO_CAMERA_OPTICAL_OFFSET_M = 0.346


# ============================================================================
# GTSP 최적화 기본값
# ============================================================================
DEFAULT_KNN = 30
DEFAULT_LAMBDA_ROT = 1.0

# ============================================================================
# 충돌 검사 파라미터
# ============================================================================

COLLISION_MARGIN = 0.0
COLLISION_ADAPTIVE_MAX_JOINT_STEP_DEG = 0.05  # 1 step 당 최대 joint 변화량
COLLISION_INTERP_EXCLUDE_LAST_JOINT = True # End-Effector 회전 무시


# ============================================================================
# 재계획 파라미터
# ============================================================================

REPLAN_ENABLED = True
REPLAN_MAX_ATTEMPTS = 60
REPLAN_TIMEOUT = 10.0  # 초
REPLAN_INTERP_DT = 0.005
REPLAN_TRAJOPT_TSTEPS = 32


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
            raise ValueError(f"잘못된 mesh_type: '{mesh_type}'. 'source' 또는 'target'이어야 합니다")

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


def get_ik_path(object_name: str, num_viewpoints: int, filename: str = "ik_solutions.h5") -> Path:
    """
    IK 솔루션 파일 경로 반환

    Args:
        object_name: 객체 이름 (예: "glass")
        num_viewpoints: 뷰포인트 개수
        filename: 파일명 (기본값: "ik_solutions.h5")

    Returns:
        IK 솔루션 경로: data/{object_name}/ik/{num_viewpoints}/{filename}

    Example:
        >>> get_ik_path("glass", 500)
        PosixPath('data/glass/ik/500/ik_solutions.h5')
    """
    return DATA_ROOT / object_name / "ik" / str(num_viewpoints) / filename


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
