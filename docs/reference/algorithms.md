# 핵심 알고리즘

## Viewpoint 생성

물체 표면을 sampling하고 working distance만큼 떨어진 카메라 pose를 만든다. CoACD와 sub-clustering으로 검사 영역을 나누고 lawnmower 경로와 Delaunay 인접 그래프를 저장한다.

Viewpoint Studio는 surface sampling을 기본으로 사용한다. grid와 세부 실험 옵션은 내부 CLI에 남아 있다.

## DP 궤적

각 viewpoint에서 multi-seed IK를 풀고 유사한 해를 줄인 뒤, 저장된 viewpoint 순서를 따라 joint 변화와 재배치를 최소화하는 후보를 선택한다. 필요한 구간은 MotionGen으로 연결하고 충돌 검사와 timing을 적용한다.

## GLNS 궤적

각 viewpoint의 nominal, roll, tilt pose와 IK branch를 후보로 만든다. Delaunay 연결을 허용 전이로 사용해 방문 순서와 joint branch를 함께 최적화한다. 이후 verify 단계가 scan motion, 충돌 회피 전환과 성분 연결을 생성한다.

세부 수치와 현재 기본값은 각 구현 모듈과 `--help`를 기준으로 한다.
