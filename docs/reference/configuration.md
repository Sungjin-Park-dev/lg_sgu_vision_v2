# 주요 설정값

공유 설정은 `scripts/common/config.py`, 궤적 기본값은 `scripts/core/trajectory/settings.py`에 있다. 문서보다 코드의 현재 값을 우선한다.

## 카메라

용어·기준점은 [camera-geometry.md](camera-geometry.md)를 단일 진실원으로 한다.

| 항목 | 값 | 코드 심볼 |
|---|---|---|
| FOV_footprint 가정 | 50 × 50 mm (트릭, 실광학 아님) | `CAMERA_FOV_WIDTH/HEIGHT_MM` |
| frame_standoff (optical_frame→object) | 46 mm | `CAMERA_WORKING_DISTANCE_MM` |
| WD (lens_front→object, 벤더공칭) | 250 mm (기준점 파킹) | — |
| mount_offset (flange→optical_frame) | 0.346 m | `TOOL_TO_CAMERA_OPTICAL_OFFSET_M` |
| overlap | 0.5 | `CAMERA_OVERLAP_RATIO` |
| 렌즈 / 센서 | MFA121-U50 f=50mm / AR0820 8.08×4.55mm | — |

## 로봇과 충돌

| 항목 | 기본값 |
|---|---|
| robot config | `ur20_with_camera.yml` |
| joint 순서 | shoulder pan, shoulder lift, elbow, wrist 1, wrist 2, wrist 3 |
| 시작 자세 | `ROBOT_START_STATE` |
| collision margin | 0 m |

물체 위치, 테이블, 벽과 support 형상도 `config.py`에서 정의한다. 설정을 바꾼 뒤에는 viewpoint pose, IK와 충돌 검사를 다시 실행한다.

경로 생성에는 `get_mesh_path`, `get_viewpoint_path`, `get_ik_path`, `get_trajectory_path` 헬퍼를 사용한다.
