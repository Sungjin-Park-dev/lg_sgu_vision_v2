# curobo_ws

UR5e 로봇을 이용한 비전 검사 궤적 생성 시스템. cuRobo(IK/충돌검사) + cuOpt(GTSP) 기반.

## 디렉토리 구조

```
scripts/           현재 사용 중인 스크립트
scripts/prev/      이전 버전 (2_generate_trajectory.py 등 full pipeline)
common/config.py   전역 설정 (카메라, 로봇, 월드, 경로 헬퍼)
common/math_utils.py  공용 수학 유틸리티
common/viz_utils.py   공용 시각화 유틸리티
data/{object}/     mesh/ viewpoint/ trajectory/ 구조
curobo/            NVIDIA cuRobo 라이브러리 (수정된 포크)
cuopt/             cuOpt TSP/GTSP 예제
```

## 파이프라인

```
generate_viewpoints.py → viewpoints.h5        (클러스터링 + GTSP 순서)
    ↓
plan_motion.py        → trajectory.csv       (IK + 충돌검사 + 재계획)
    ↓
publish_trajectory.py → ROS2 FollowJointTrajectory action
```

## 핵심 규칙

- **Joint 순서**: `[shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]`
  - `/joint_states` 토픽은 **알파벳 순** (elbow 먼저) — 코드는 이름 매칭으로 처리
- **좌표계**: Isaac Sim 기준, 미터 단위
- **로봇 설정**: `ur5e.yml` (IK용), `ur20_with_camera.yml` (config.py DEFAULT)
- **ROS2 컨트롤러**: `/scaled_joint_trajectory_controller/follow_joint_trajectory`

## 상세 문서

- [docs/architecture.md](docs/architecture.md) — 파이프라인 상세, 데이터 흐름
- [docs/running.md](docs/running.md) — 실행 방법 (UR 드라이버, 파이프라인, ROS2)
- [docs/config.md](docs/config.md) — config.py 파라미터 설명
- [docs/generate_viewpoints.md](docs/generate_viewpoints.md) — 뷰포인트 생성 + 클러스터링 + GTSP 상세
- [docs/logs/](docs/logs/) — 날짜별 작업 로그
