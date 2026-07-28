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
  → GLNS solve (IK 후보 + 순서)
  → GLNS verify (motion planning + 충돌게이트)
  → trajectory CSV/NPZ
  → Isaac preview 또는 ROS2 controller
```

`core/viewpoint`는 mesh, sampling, clustering, ordering과 HDF5를 담당한다. `core/trajectory`는 로봇 모델, IK, 충돌 회피, timing 같은 공유 엔진을 담당하며, `core/glns`는 그것을 써서 후보 생성·Delaunay 제약 순서 최적화·궤적 검증을 수행한다.

Inspection과 MoveIt은 같은 로봇에 서로 다른 명령을 보내므로 controller를 동시에 활성화하지 않는다. 현재 모드 조합은 [Isaac 실행 모드](../guides/isaac-modes.md)를 참고한다.
