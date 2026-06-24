# generate_viewpoints.py

메시 표면에 뷰포인트를 생성하고, 클러스터링 + GTSP로 방문 순서를 최적화한다.
**배치(sampling)** 와 **클러스터 내부 순서(ordering)** 를 선택할 수 있다.

## 파이프라인

```
OBJ 로드 → (material RGB 필터)
  → [배치] grid: PCA 그리드 → 표면 스냅  |  surface: 표면 직접 균일 샘플링(FPS)
  → 법선/카메라 위치 → bottom-facing 필터 → 클러스터링
  → [순서] zigzag: 전역 PCA 행 정렬  |  graph: NN+2opt  |  lawnmower: 탄젠트 row sweep
  → GTSP 클러스터 순서 → HDF5 + 인터랙티브 HTML 저장
```

클러스터링은 항상 수행된다 (opt-out 없음). HTML 시각화는 항상 생성된다.

> 단계별 흐름도(mermaid): [generate_viewpoints_pipeline.md](generate_viewpoints_pipeline.md).

## 배치 모드 (`--sampling-mode`)

| 모드 | 설명 | 적합 |
|------|------|------|
| `grid` (기본) | PCA 평면에 균일 그리드 → 가장 가까운 표면점으로 스냅. | 평평하고 PCA 평면과 나란한 판재 |
| `surface` | 표면적 비례 후보점을 많이 뽑은 뒤 Farthest Point Sampling(FPS)으로 `target count = area/spacing²`개를 선택. 곡면·측벽도 **표면적 기준**으로 고르게 덮음. | 곡면·입체물 (cylinder/curved/box) |

> **왜 surface인가**: `grid`는 점 예산을 PCA **투영 면적**으로 잡고 closest-point 스냅에 의존해, PCA 평면에 수직인 곡면·측벽엔 점이 거의 안 생기고 윗면에 쏠린다. `surface`는 실제 표면적 기준으로 균일 분포를 뽑아 이 문제를 해결한다 (간격 = `--surface-spacing`, 기본 FOV 작은 축).

## 순서 모드 (`--ordering-mode`)

| 모드 | 설명 |
|------|------|
| `zigzag` (기본) | 전역 PCA 축 + 원본 grid `row_index` 기반 행 정렬 (아래 "클러스터 내부 정렬"). grid 배치에 적합. |
| `graph` | 클러스터 평균 법선의 탄젠트 평면에서 시작 극단점을 잡고, camera position 기준 nearest-neighbor + 2-opt open path를 만든다. tangent-정렬 baseline보다 길어지지 않도록 가드. |
| `lawnmower` | 표면점 기준 tangent plane에서 긴 축을 scan, 짧은 축을 row로 잡고 FOV-derived spacing으로 row를 나눠 serpentine/lawnmower 순서를 만든다. surface 배치에 적합. |

> 보통 `surface`+`lawnmower`, `grid`+`zigzag`를 쌍으로 쓴다 (`viewpoint_studio.py`는 `surface+lawnmower` 고정).

## CLI 옵션

### 뷰포인트 생성
| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--object` | (필수) | 오브젝트 이름 |
| `--material-rgb "R,G,B"` | None | 재질 RGB 필터 (예: `"0,255,0"`) |
| `--color-tolerance` | 5.0 | RGB 매칭 허용 오차 |
| `--row-spacing` | FOV_HEIGHT × (1-overlap) mm | [grid] 행 간격 |
| `--col-spacing` | FOV_WIDTH × (1-overlap) mm | [grid] 열 간격 |
| `--sampling-mode` | `grid` | `grid` \| `surface` (위 "배치 모드") |
| `--surface-spacing` | FOV 작은 축 mm | [surface] FPS 목표 표면 간격 |
| `--ordering-mode` | `zigzag` | `zigzag` \| `graph` \| `lawnmower` (위 "순서 모드") |
| `--no-filter-bottom` | off | 바닥향 뷰포인트 필터 비활성화 |
| `--bottom-angle` | 80.0 | 바닥향 필터 각도 (deg) |

### 클러스터링
| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--cluster-method` | `dbscan` | `dbscan` \| `coacd` \| `coacd+dbscan` \| `agglomerative` \| `coacd+agglomerative` |
| `--eps` | FOV_WIDTH mm | [dbscan] 이웃 반경 (mm) |
| `--min-samples` | 2 | [dbscan] 코어 포인트 최소 이웃 수 |
| `--target-size` | 12 | [agglomerative ward] 클러스터당 목표 점 개수 |
| `--max-span` | None | [agglomerative] 클러스터 최대 지름 mm (지정 시 complete-linkage, 지름 ≤ 값 보장) |
| `--normal-weight` | 0.0 | [dbscan/agglomerative] 법선 가중치 (m). 0이면 위치만 사용 |
| `--coacd-threshold` | 0.05 | [coacd] concavity threshold (낮을수록 파트 많아짐) |

### 비교/디버그
| 옵션 | 설명 |
|------|------|
| `--compare` | 선택된 방법의 파라미터 변형 비교 HTML 생성 |
| `--dry-run` | 통계만 출력, HDF5 저장 안 함 |

## 클러스터링 방법

| 방법 | 설명 |
|------|------|
| `dbscan` (기본) | 위치 (+ 선택적 법선) 기반 **밀도** 클러스터링. `--eps`, `--normal-weight`로 조절 |
| `coacd` | 메시를 convex 파트로 분해, 각 뷰포인트를 가장 가까운 파트에 할당 |
| `coacd+dbscan` | 1단계 CoACD로 파트 분리 → 2단계 각 파트 내 DBSCAN 세분화 |
| `agglomerative` | **공간 분할**. `--max-span D`(권장): complete-linkage로 **클러스터 지름 ≤ D 보장**(멀리 떨어진 점 묶임 방지). 또는 `--target-size`: Ward+개수 |
| `coacd+agglomerative` | 1단계 CoACD(표면 positions) → 2단계 각 파트 내 **카메라 위치** 기준 Agglomerative 세분화 |

> **클러스터링 기준 = 카메라 위치**: `coacd+dbscan`/`coacd+agglomerative`의 2단계는 표면 위치가
> 아니라 **camera_positions(표면+법선×working_distance)** 로 클러스터링한다. 렌더 마커·로봇 EE가
> 카메라 위치이고, 곡면에선 표면이 가까워도 카메라가 크게 벌어지기 때문(표면 60mm → 카메라 200mm+).

> **밀도 vs 공간 분할**: `surface` 샘플링은 밀도가 **균일**해서 DBSCAN(밀도 기반)이 자를 골짜기가 없어
> **"거대 클러스터 1개 + 싱글톤 다발"**로 깨진다(예: curved_structure 206vp → 36클러스터 중 18개 싱글톤).
> `agglomerative`는 **컴팩트·싱글톤 없는** 구역으로 나눈다. 권장 노브는 `--max-span D`:
> **complete linkage**라 모든 클러스터의 지름(내부 최대 점간 거리)이 D 이하로 보장돼,
> "멀리 떨어진 viewpoint가 한 클러스터로 묶이는" 문제를 원천 차단(D=60mm → 25클러스터, 지름 모두 ≤60mm).
> (대안: `--target-size`는 개수 기반 Ward — 평균 크기는 맞지만 지름은 제한 안 함.)
> **균일 표면엔 `coacd+agglomerative --max-span` 권장.**

## 클러스터 내부 정렬 (`--ordering-mode`)

**`zigzag` (기본)**: 전역 축(cam_axis1/cam_axis2)과 원본 grid `row_index`를 사용해 `reorder_zigzag()` 수행. 클러스터별 로컬 PCA는 생략 — `row_index_override`가 행 구분을 담당하므로 로컬 axis1이 불필요하고, axis2는 클러스터 간 열 방향 일관성을 위해 전역 값을 사용 (곡면 클러스터에서의 행 오분류 방지 — [archive/zigzag-row-assignment-issue.md](archive/zigzag-row-assignment-issue.md) 참조).

**`graph`**: `order_cluster_graph()` — 클러스터 평균 법선으로 탄젠트 2D 프레임을 만들고(PCA 아님), 가장 긴 tangent 방향의 극단에서 시작해 camera position 기준 nearest-neighbor + open-path 2-opt로 방문 순서를 푼다. 결과 경로가 tangent-정렬 baseline보다 길면 폴백한다.

**`lawnmower`**: `order_cluster_lawnmower()` — 표면점(`positions`)을 평균 법선 tangent plane에 투영하고, 그 2D 점들의 PCA로 긴 축(scan)과 짧은 축(row)을 잡는다. row 간격은 `min(row_spacing, col_spacing)`으로 양자화하고, row마다 scan 방향을 뒤집어 `→ ← → ←` 형태의 serpentine 경로를 만든다. 두 시작 방향 중 실제 `camera_positions` 경로 길이가 짧은 쪽을 선택한다. 반환은 `{sorted_indices, endpoint_a/b, normal_a/b}`로 zigzag와 동일 → GTSP 단계 그대로 동작.

## 클러스터 간 순서 (GTSP)

각 클러스터의 양 끝점(endpoint_a, endpoint_b) 중 어느 쪽으로 진입/퇴장할지를 동시에 결정해야 하므로, K개 클러스터 × 2방향 = **GTSP** (Generalized TSP).

**풀이**: Noon-Bean 변환 + 더미 노드
1. **노드 구성**: 클러스터 k마다 F_k(정방향), R_k(역방향) + 더미 D = 총 2K+1개
2. **Noon-Bean 변환**: GTSP → ATSP
   - 클러스터 내 사이클: F_k↔R_k 비용 0
   - 클러스터 간 간선: predecessor로 shift (`ATSP[R_k,F_l] = GTSP[F_k,F_l]`)
3. **더미 노드 D**: D↔모든 노드 비용 0 → open path (시작/끝점 자유)
4. **OR-Tools ATSP**: PATH_CHEAPEST_ARC + GUIDED_LOCAL_SEARCH (2초)
5. **디코딩**: 투어에서 D 제거 → 2개씩 쌍으로 묶어 진입 노드로 순서+방향 결정

**출력**: `cluster_order` (K,), `cluster_direction` (K,) — 0=Forward(a→b), 1=Reverse(b→a)

## 출력

- `data/{object}/viewpoint/{num}/viewpoints_{method}.h5` — 뷰포인트 + 클러스터 데이터
  (`{method}` = 클러스터 방법). h5 `metadata`에 `sampling_mode`/`ordering_mode` 기록,
  `method` 속성 = `"{sampling_mode}+{ordering_mode}"`.
- `data/{object}/viewpoint/{num}/viewpoints_{method}.html` — 인터랙티브 3D 시각화

HDF5 구조는 [architecture.md](architecture.md#viewpointsh5) 참조.

## HTML 시각화

`scripts/common/viewpoint_viz.py`의 `visualize_clusters_html()`로 생성 (내부 전용, 직접 실행 안 함).

**시각화 요소**:
- **메시**: 반투명 회색 배경
- **클러스터별 뷰포인트**: 고유 색상 마커 + 클러스터 내 경로 라인
- **클러스터 간 이동**: 회색 점선
- **CoACD 파트 메시**: 반투명 색상 오버레이 (coacd / coacd+dbscan 모드)

**`--compare` 모드**: 여러 파라미터 조합을 드롭다운으로 전환. 버튼 라벨에 경로 길이, 클러스터 수, 최적 대비 차이(%) 표시.

## 하류 호환성

`plan_trajectory.py`의 `load_viewpoints()`는 `cluster_id`, `cluster_order`를 **필수**로 읽는다. 두 데이터가 없으면 에러로 종료. 레거시 non-clustered 모드는 제거됨.
