# config.py 파라미터

`common/config.py` — 전역 설정 파일

## 카메라 사양

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `CAMERA_FOV_WIDTH_MM` | 41.0 | 시야각 너비 (mm) |
| `CAMERA_FOV_HEIGHT_MM` | 30.0 | 시야각 높이 (mm) |
| `CAMERA_WORKING_DISTANCE_MM` | 110.0 | 작업 거리 (mm) |
| `CAMERA_OVERLAP_RATIO` | 0.5 | 뷰포인트 간 중첩률 (50%) |
| `TOOL_TO_CAMERA_OPTICAL_OFFSET_M` | 0.234 | tool0 → 카메라 초점 거리 (m) |

## 로봇 설정

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `ROBOT_START_STATE` | UR 시뮬 기본 자세 | 6-DOF radian, JOINT_NAMES 순서 |
| `ROBOT_HAS_CONSTRAINT` | True | 시작 자세 기준 제약 활성화 |
| `MAX_JOINT_FROM_START_STATE` | 90 deg | 시작 자세에서 최대 이동량 |
| `DEFAULT_ROBOT_CONFIG` | `ur20_with_camera.yml` | cuRobo 로봇 설정 파일 |

### ROBOT_START_STATE 순서

```python
# [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
ROBOT_START_STATE = np.array([-2.2030, -1.7271, -1.6007, -0.808, 1.5951, -0.031])
```

## 월드 설정 (Isaac Sim 좌표, 미터)

| 객체 | 위치 | 크기/회전 |
|------|------|----------|
| TARGET_OBJECT | [0.0, 0.5, 0.095] | quat [0.707, 0, 0, 0.707] |
| TABLE | [0.0, 1.34, -0.435] | 1.0 x 0.6 x 0.73 |
| ROBOT_MOUNT | [0.0, 0.0, -0.25] | 0.3 x 0.3 x 0.5 |
| WALLS | 4면 + support | 작업 공간 경계 |

## 충돌 검사

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `COLLISION_MARGIN` | 0.0 | 안전 여유 (m) |
| `COLLISION_ADAPTIVE_MAX_JOINT_STEP_DEG` | 0.05 | 보간 step 당 최대 joint 변화 |
| `COLLISION_INTERP_EXCLUDE_LAST_JOINT` | True | EE 회전 무시 |

## 재계획

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `REPLAN_MAX_ATTEMPTS` | 60 | 충돌 구간 당 최대 시도 |
| `REPLAN_TIMEOUT` | 10.0 | 시도 당 타임아웃 (초) |
| `REPLAN_INTERP_DT` | 0.005 | 보간 시간 간격 |
| `REPLAN_TRAJOPT_TSTEPS` | 32 | 궤적 최적화 스텝 수 |

## 경로 헬퍼 함수

```python
get_mesh_path(object_name, mesh_type="target")
# → data/{object}/mesh/target.ply (또는 .obj)

get_viewpoint_path(object_name, num_viewpoints)
# → data/{object}/viewpoint/{num}/viewpoints.h5

get_ik_path(object_name, num_viewpoints)
# → data/{object}/ik/{num}/ik_solutions.h5

get_trajectory_path(object_name, num_viewpoints)
# → data/{object}/trajectory/{num}/gtsp.csv
```
