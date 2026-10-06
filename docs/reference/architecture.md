# 구조와 데이터 흐름

사용자는 `scripts/apps/`의 세 앱을 실행한다. 앱은 `core`를 import하거나 독립 프로세스로 호출한다.

```text
scripts/
├── apps/       사용자 GUI 3개
├── core/       viewpoint, trajectory, glns, reconstruction, isaac 엔진
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

`core/viewpoint`는 mesh, sampling, Delaunay 인접 그래프와 HDF5를 담당한다. `core/trajectory`는 로봇 모델, IK, 충돌 회피, timing 같은 공유 엔진을 담당하며, `core/glns`는 그것을 써서 후보 생성·Delaunay 제약 순서 최적화·궤적 검증을 수행한다.

Inspection과 MoveIt은 같은 로봇에 서로 다른 명령을 보내므로 controller를 동시에 활성화하지 않는다. 현재 모드 조합은 [Isaac 실행 모드](../guides/isaac-modes.md)를 참고한다.

## 도면 없는 물체 복원

CAD 가 없는 물체는 손목 depth 카메라로 찍어 메시를 만든다. 결과 메시가 위 흐름의 입력(viewpoint
생성)이 된다.

```text
Isaac Pipeline "Scan Object" 패널
  ROI 구(물체 놓는 위치 중심, 반경은 대략) → scan_plan.py(서브프로세스, cuRobo IK·자기 가림 제거)
  → 촬영(앱 안, CaptureSession — 관절 순간이동 + depth 렌더)
  → reconstruct.py(서브프로세스, 장애물 지우기 → (ICP) → TSDF → 메시)
      ↓ data/{object}/scan/{run}/  (plan · frames · recon{_tag}.ply/json)
Viewpoint Studio "Scan" 패널
  스캔 확인(남긴/지운 점·카메라·정답) · 파라미터 바꿔 재복원 · recon.obj 내보내기 → Generate
```

ROI 는 물체를 넉넉히 감싸기만 하면 된다 — 중심은 물체를 놓은 위치(ArUco 등으로 정한 물체 pose)를
쓰고, 넓어진 ROI 로 들어온 받침·테이블은 씬 장애물로 지워진다. 정답 형상은 쓰지 않는다.

`core/reconstruction` 이 엔진이다. `storage.py` 가 스캔 폴더 규약을 소유하고, 계획에는 촬영 당시의
물체 pose 와 씬 장애물 OBB 가 함께 남는다 — 복원과 Studio 가 지금 씬 설정이 아니라 그 값으로 점을
지운다. 정답 CAD 와의 비교, pose 오차 주입처럼 시뮬레이션에서만 의미 있는 것은 `sim.py` 에 모았고
실물 경로는 그것을 import 하지 않는다. 지금 `recon.obj` 는 정답 물체 좌표로 내보내는 평가용이며,
복원 메시를 새 물체로 등록하는 규약(바닥 중심 `source.obj` + 배치 + USD)은 아직 정하지 않았다.
