# 뷰포인트 만들기

Viewpoint Studio에서 물체 표면의 검사 위치와 카메라 방향을 생성하고 HDF5로 저장한다.

## 실행

```bash
uv run scripts/apps/viewpoint_studio.py
```

브라우저에서 `http://localhost:8080`에 접속한다.

## 작업 순서

1. `Object`에서 물체를 고른다.
2. `Camera spec`에서 FOV와 working distance를 정한다.
3. `Generate (...)`에서 overlap과 클러스터 설정을 조절한다.
4. `Generate`로 뷰포인트를 만든다.
5. 레이어와 `Playback`으로 분포와 순서를 확인한다.
6. `Save h5`로 저장한다.

## 주요 기능

| 영역 | 기능 |
|---|---|
| `Existing h5` | 저장된 뷰포인트 다시 열기. 그 파일의 카메라 스펙이 입력칸에 반영된다 |
| `Layers` | 메시, 경로, 전환, Delaunay 그래프 표시. `Color by`로 클러스터/그래프 성분 전환 |
| `Camera spec` | FOV 가로·세로(mm), working distance(mm) |
| `Generate (...)` | overlap, `Stage 1`, `Sub-cluster`, 생성·저장 |
| `Playback` | 최종 검사 순서 재생 |

## 카메라 스펙

세 값은 h5에 함께 저장되고, **그 h5를 읽는 IK·궤적·GLNS·Isaac이 전부 이 값을 쓴다.**
working distance는 뷰포인트를 표면에서 띄우는 거리라 바꾸면 로봇이 가는 위치가 달라진다
([카메라 기하](../reference/camera-geometry.md)).

- working distance에는 하한이 있다. 그보다 작으면 생성이 거부된다.
- overlap은 카메라 스펙이 아니라 샘플링 값이라 `Generate (...)`에 있다 — 표면 간격 = FOV × (1 − overlap).

## Stage 1

클러스터링은 2단계다. `Stage 1`이 큰 덩어리로 나누고 `Sub-cluster`가 그 안을 쪼갠다.

| Stage 1 | 특징 |
|---|---|
| `Delaunay` (기본) | 표면 인접 그래프의 연결 성분. 결정적 |
| `CoACD` | 볼록 분해 파트. 실행마다 결과가 달라질 수 있다 |

어느 쪽을 쓰든 h5 형식은 같고, 사용한 방법은 `metadata/clustering_method`에 남는다.

출력은 `data/{object}/viewpoint/{N}/viewpoints_{method}.h5`에 저장된다. 새 물체가 목록에 없다면 [자산 준비](../guides/prepare-object-assets.md)를 먼저 확인한다.
