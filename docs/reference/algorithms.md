# 핵심 알고리즘

## Viewpoint 생성

물체 표면을 sampling하고 working distance만큼 떨어진 카메라 pose를 만든다. stage 1(Delaunay 표면 성분 또는 CoACD 볼록 파트)과 sub-clustering으로 검사 영역을 나누고 lawnmower 경로와 Delaunay 인접 그래프를 저장한다.

Sampling은 메시 표면 직접 FPS 하나뿐이다. PCA 평면에 격자를 깔고 투영하던 grid 모드는 곡면·측벽을 놓치고 속 빈 물체의 지붕을 잃어 2026-08-26 제거했다. 세부 실험 옵션은 내부 CLI에 남아 있다.

## GLNS 궤적

궤적 생성은 2단계다.

**1. solve** — 각 viewpoint의 nominal, roll, tilt pose와 IK branch를 후보로 만든다. Delaunay 연결을 허용 전이로 사용해 방문 순서와 joint branch를 함께 최적화한다(성분마다 open GTSP 하나). 결과는 `solution.h5`.

**2. verify** — 그 순서를 실제 이동으로 바꾼다. 재배치 구간과 충돌하는 scan 구간은 MotionGen으로 잇고, densify해 충돌을 다시 검사한 뒤 uniform resample과 timing을 적용한다. 성분들은 seam transit으로 이어 하나의 `trajectory.csv`가 된다.

viewpoint 1개짜리 성분(그래프상 고립 정점)도 join에 포함된다. `--require-full-coverage`는 joined가 전체 viewpoint를 덮지 못하면 실패시킨다.

세부 수치와 현재 기본값은 각 구현 모듈과 `--help`를 기준으로 한다.
