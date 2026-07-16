# 구조와 데이터 흐름

사용자는 `scripts/apps/`의 세 앱을 실행한다. 앱은 `core`를 import하거나 독립 프로세스로 호출한다.

```text
scripts/
├── apps/       사용자 GUI 3개
├── core/       viewpoint, trajectory, glns, isaac 엔진
├── common/     공유 설정과 수학 함수
├── setup/      mesh와 USD 준비 도구
├── moveit/     MoveIt sim/real 연동
└── julia/glns/ GLNS 런타임
```

## 처리 흐름

```text
Viewpoint Studio
  → viewpoints HDF5
  → Trajectory Studio 또는 Isaac Pipeline
  → IK + DP/GLNS + motion planning
  → trajectory CSV/NPZ
  → Isaac preview 또는 ROS2 controller
```

`core/viewpoint`는 mesh, sampling, clustering, ordering과 HDF5를 담당한다. `core/trajectory`는 로봇 모델, IK, DP, 충돌 회피와 timing을 담당하며, `core/glns`는 후보와 Delaunay 제약 경로를 최적화한다.

Inspection과 MoveIt은 같은 로봇에 서로 다른 명령을 보내므로 controller를 동시에 활성화하지 않는다. 현재 모드 조합은 [Isaac 실행 모드](../guides/isaac-modes.md)를 참고한다.
