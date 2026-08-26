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
3. `Generate viewpoints`에서 overlap과 그래프 파라미터를 조절한다.
4. `Generate`로 뷰포인트를 만든다.
5. `Display` 토글과 하단 한 줄로 점 분포와 그래프를 확인한다.
6. `Save h5`로 저장한다.

## 화면 구성

| 영역 | 기능 |
|---|---|
| `Object` | 물체 선택. 바꾸면 화면이 비워진다 |
| `Saved viewpoints` | 저장된 뷰포인트 다시 열기. 그 파일의 카메라 스펙이 입력칸에 반영된다 |
| `Camera spec` | FOV 가로·세로(mm), working distance(mm) |
| `Generate viewpoints` | overlap, 그래프 노브 3개, 생성·저장 |
| `Display` | 메시, 표면점, 카메라 위치, 그래프 간선 표시 |
| `Solver graph (hops)` | GLNS가 실제로 풀 그래프로 관점 전환 (즉시 반영) |

`Saved viewpoints`는 지금 화면의 출처를 가리킨다 — 생성만 하고 저장 전이면
`(generated · unsaved)`, 저장하면 그 파일이 선택된다.

## 카메라 스펙

세 값은 h5에 함께 저장되고, **그 h5를 읽는 IK·궤적·GLNS·Isaac이 전부 이 값을 쓴다.**
working distance는 뷰포인트를 표면에서 띄우는 거리라 바꾸면 로봇이 가는 위치가 달라진다
([카메라 기하](../reference/camera-geometry.md)).

- working distance에는 하한이 있다. 그보다 작으면 생성이 거부된다.
- overlap은 카메라 스펙이 아니라 샘플링 값이라 `Generate viewpoints`에 있다 —
  표면 간격 = min(FOV) × (1 − overlap).

## 생성 파라미터

네 개뿐이고, 넷 다 다음 단계를 실제로 바꾼다. 각 입력칸에 툴팁이 붙어 있다.

| 노브 | 무엇을 정하나 |
|---|---|
| `FOV overlap (%)` | 점의 **개수와 간격**. 올리면 촘촘해진다 |
| `Max edge length (×)` | 간선 길이 상한 — 주변 카메라 위치 간격의 배수 |
| `Max normal angle (°)` | 법선이 이보다 벌어진 두 점은 잇지 않는다. 90° = 반대편 면 차단 |
| `Neighbor search (k)` | 삼각분할 후보 이웃 수. 거의 건드릴 일이 없다 |

뒤 셋이 **Delaunay 인접 그래프**를 만든다. 이 그래프가 GLNS의 순서 제약이 되므로,
간선이 어떻게 이어지는지가 다음 단계의 이동 비용을 좌우한다.

## 하단 한 줄 읽기

```text
329 edges · 2 components · 0 isolated · GLNS: 852 (2-hop)
```

- `components`가 2 이상이면 물체가 그래프상 조각나 있다는 뜻이고, 조각 사이를 잇는
  transit이 생긴다. 대개는 정상이다 — 실제로 저장된 파일 다수가 2성분 이상이다.
- `isolated`는 아무 간선도 없는 점이다. GLNS가 제약 그래프에서 떨어뜨린다.
- `GLNS: N (h-hop)`은 GLNS가 실제로 푸는 간선 수다. 저장되는 간선은 항상 1-hop이고,
  `solve.py --delaunay-expand-hops`가 그것을 확장한다. `Solver graph (hops)` 슬라이더를
  그 값과 맞춰두면 화면과 다음 단계가 같은 그래프를 본다(양쪽 기본 2).

## 저장 형식

출력은 `data/{object}/viewpoint/{N}/viewpoints.h5`이고 **기하 + 그래프** 두 계층만
담는다([데이터 형식](../reference/data-formats.md)). 방문 순서는 담지 않는다 — GLNS가
순서와 IK 자세를 함께 풀기 때문이다.

새 물체가 목록에 없다면 [자산 준비](../guides/prepare-object-assets.md)를 먼저 확인한다.
