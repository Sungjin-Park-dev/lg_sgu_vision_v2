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

# 카메라 시야각 (mm)
CAMERA_FOV_WIDTH_MM = 50.0
CAMERA_FOV_HEIGHT_MM = 50.0

# 작업 거리 (mm) - camera_optical_frame에서 검사 표면까지의 거리
CAMERA_WORKING_DISTANCE_MM = 250.0

# 카메라 뷰 유효 면적 (0.5 = 50% 중첩)
CAMERA_OVERLAP_RATIO = 0.7

# 센서 해상도 (pixels) + 픽셀 크기 (mm)
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
# rotation = identity: 물체 방향은 config가 아니라 **메시에 베이크**한다(prep/reorient_mesh.py).
# 그래야 viser(로컬 프레임)와 Isaac(월드 프레임)이 회전 차이 없이 동일하게 보임.
# (이전 [0.7071,0,0,0.7071]은 순수 z-yaw였음 → identity로 바꿔도 bottom 필터는 불변, sample은 90° yaw만 풀림.)
TARGET_OBJECT = {
    "name": "target_object",
    "position": np.array([-0.1, 1.1, -0.010], dtype=np.float64),
    "rotation": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),  # 쿼터니언: w, x, y, z (identity)
}

# 테이블 직육면체 설정 — thor_table.usd 측정값 매칭
# visual: world center (-0.2, 1.1, 0.315), size 0.910×0.768×0.630
# robot frame z = 0.315 - 0.805 = -0.490
TABLE = {
    "name": "table",
    "position": np.array([-0.2, 1.1, -0.490], dtype=np.float64),
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
        # Target object 받침대 — table top(world z=0.630)에서 object bottom(world z=0.795)까지
        # robot frame z = world 0.7125 - 0.805 = -0.0925, height 0.165m, 2×2cm 단면
        "name": "support",
        "position": np.array([-0.1, 1.1, -0.0925], dtype=np.float64),
        "dimensions": np.array([0.2, 0.3, 0.165], dtype=np.float64),
    },
]

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

# 툴 오프셋: flange/camera_mount에서 camera_optical_frame까지의 거리 (미터)
# End-Effector로부터 카메라 초점까지의 실제 거리로 변경해야 합니다.
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
