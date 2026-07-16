# 주요 설정값

공유 설정은 `scripts/common/config.py`, 궤적 기본값은 `scripts/core/trajectory/settings.py`에 있다. 문서보다 코드의 현재 값을 우선한다.

## 카메라

| 항목 | 기본값 |
|---|---|
| FOV | 50 × 50 mm |
| working distance | 250 mm |
| overlap | 0.7 |
| tool0 → camera optical offset | 0.346 m |

## 로봇과 충돌

| 항목 | 기본값 |
|---|---|
| robot config | `ur20_with_camera.yml` |
| joint 순서 | shoulder pan, shoulder lift, elbow, wrist 1, wrist 2, wrist 3 |
| 시작 자세 | `ROBOT_START_STATE` |
| collision margin | 0 m |

물체 위치, 테이블, 벽과 support 형상도 `config.py`에서 정의한다. 설정을 바꾼 뒤에는 viewpoint pose, IK와 충돌 검사를 다시 실행한다.

경로 생성에는 `get_mesh_path`, `get_viewpoint_path`, `get_ik_path`, `get_trajectory_path` 헬퍼를 사용한다.
