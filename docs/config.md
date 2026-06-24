# config.py 파라미터

`common/config.py` — 전역 설정 파일. 실제 값 기준.

## 카메라 사양

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `CAMERA_FOV_WIDTH_MM` | 50.0 | 시야각 너비 (mm) |
| `CAMERA_FOV_HEIGHT_MM` | 50.0 | 시야각 높이 (mm) |
| `CAMERA_WORKING_DISTANCE_MM` | 250.0 | 작업 거리 (mm) |
| `CAMERA_OVERLAP_RATIO` | 0.7 | 뷰포인트 간 중첩률 |
| `TOOL_TO_CAMERA_OPTICAL_OFFSET_M` | 0.346 | tool0 → 카메라 초점 거리 (m) |

## 로봇 설정

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `ROBOT_START_STATE` | 아래 참조 | UR 시뮬 기본 자세 (radian) |
| `DEFAULT_ROBOT_CONFIG` | `"ur20_with_camera.yml"` | cuRobo 로봇 설정 파일 |
| `DEFAULT_URDF_PATH` | cuRobo assets 내 경로 | URDF 경로 |

```python
# [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
ROBOT_START_STATE = np.deg2rad([-270, -90, 60, -90, -90, 0])
```

## 월드 설정 (Isaac Sim 좌표, 미터)

| 객체 | 위치 | 크기/회전 |
|------|------|----------|
| `TARGET_OBJECT` | `[-0.1, 1.1, 0.095]` | quat `[0.7071, 0, 0, 0.7071]` |
| `TABLE` | `[-0.2, 1.1, -0.435]` | `1.0 × 0.6 × 0.73` |
| `ROBOT_MOUNT` | `[0.0, 0.0, -0.25]` | `0.3 × 0.3 × 0.5` |
| `WALLS` | 4면 벽 + `support` | 작업 공간 경계 |

`support`: TARGET_OBJECT 아래 지지대 `[-0.1, 1.1, -0.0525]`, `0.1 × 0.1 × 0.265`.

## GTSP 최적화

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `DEFAULT_KNN` | 30 | GTSP k-NN 그래프 크기 |
| `DEFAULT_LAMBDA_ROT` | 1.0 | 회전 비용 가중치 |

## 충돌 검사

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `COLLISION_MARGIN` | 0.0 | 안전 여유 (m) |
| `COLLISION_ADAPTIVE_MAX_JOINT_STEP_DEG` | 0.05 | 보간 step 당 최대 joint 변화 |
| `COLLISION_INTERP_EXCLUDE_LAST_JOINT` | True | EE 회전 무시 |

## 재계획

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `REPLAN_ENABLED` | True | 재계획 활성화 |
| `REPLAN_MAX_ATTEMPTS` | 60 | 충돌 구간 당 최대 시도 |
| `REPLAN_TIMEOUT` | 10.0 | 시도 당 타임아웃 (초) |
| `REPLAN_INTERP_DT` | 0.005 | 보간 시간 간격 |
| `REPLAN_TRAJOPT_TSTEPS` | 32 | 궤적 최적화 스텝 수 |

## 경로 헬퍼

```python
get_mesh_path(object_name, filename=None, mesh_type="target")
# mesh_type="target": target.ply 우선, 없으면 target.obj
# mesh_type="source": source.obj (충돌용 전체 멀티 머티리얼 메시)

get_viewpoint_path(object_name, num_viewpoints, filename="viewpoints.h5")
# → data/{object}/viewpoint/{num}/{filename}

get_ik_path(object_name, num_viewpoints, filename="ik_solutions.h5")
# → data/{object}/ik/{num}/{filename}

get_trajectory_path(object_name, num_viewpoints, filename="gtsp.csv")
# → data/{object}/trajectory/{num}/{filename}
```
