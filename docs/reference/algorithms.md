# 핵심 알고리즘

## Viewpoint 생성

물체 표면을 sampling하고 working distance만큼 떨어진 카메라 pose를 만든 뒤, 그 카메라 위치 위에 local-tangent Delaunay 인접 그래프를 얹어 저장한다. 산출물은 **기하 + 그래프** 둘뿐이다.

Sampling은 메시 표면 직접 FPS다. 면적 기준으로 목표 개수(`area / spacing²`)를 잡고 후보를 넉넉히 뽑은 뒤 farthest-point sampling으로 솎아내, 곡면과 측벽도 표면적 기준으로 고르게 덮는다. 이어 아래를 보는 면과 — 속 빈 물체로 등록된 경우 — 안쪽 껍데기를 걸러낸다.

방문 순서는 여기서 정하지 않는다. GLNS가 positions/normals/edges/working distance만 읽어 순서와 IK 자세를 함께 푼다.

## GLNS 궤적

궤적 생성은 2단계다.

**1. solve** — 각 viewpoint의 nominal, roll, tilt pose와 IK branch를 후보로 만든다. Delaunay 연결을 허용 전이로 사용해 방문 순서와 joint branch를 함께 최적화한다(성분마다 open GTSP 하나). 결과는 `solution.h5`.

**2. verify** — 그 순서를 실제 이동으로 바꾼다. 재배치 구간과 충돌하는 scan 구간은 MotionGen으로 잇고, densify해 충돌을 다시 검사한 뒤 uniform resample과 timing을 적용한다. 성분들은 seam transit으로 이어 하나의 `trajectory.csv`가 된다.

viewpoint 1개짜리 성분(그래프상 고립 정점)도 join에 포함된다. `--require-full-coverage`는 joined가 전체 viewpoint를 덮지 못하면 실패시킨다.

세부 수치와 현재 기본값은 각 구현 모듈과 `--help`를 기준으로 한다.
