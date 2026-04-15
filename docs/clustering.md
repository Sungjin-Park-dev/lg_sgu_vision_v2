# Viewpoint Clustering (경로 순서 최적화)

## 개요

`generate_viewpoints.py --cluster`는 뷰포인트를 공간 클러스터링하여 경로 순서를 최적화한다.

기본 PCA 지그재그는 곡면이 복잡한 물체에서 행 전환 시 불필요한 장거리 이동이 발생할 수 있다. 클러스터링을 통해 **카메라 위치 기준으로 가까운 뷰포인트끼리 묶고**, 클러스터 내부는 지그재그, 클러스터 간은 greedy NN으로 정렬하여 총 경로 길이를 줄인다.

## 파이프라인 위치

```
generate_viewpoints.py [--cluster] → viewpoints.h5
         ↓
plan_motion.py → IK + trajectory
```

## 사용법

```bash
# 기본 실행 (Ward 클러스터링)
uv run scripts/generate_viewpoints.py --object sample --cluster

# 통계만 확인, HDF5 수정 안 함
uv run scripts/generate_viewpoints.py --object sample --cluster --dry-run

# DBSCAN 방법 사용
uv run scripts/generate_viewpoints.py --object sample --cluster --cluster-method dbscan --eps 30

# Ward vs DBSCAN HTML 드롭다운 비교
uv run scripts/generate_viewpoints.py --object sample --cluster --compare

# 파라미터 조절하며 비교
uv run scripts/generate_viewpoints.py --object sample --cluster --compare --max-diameter 100 --eps 30

# 단일 방법 + HTML 시각화
uv run scripts/generate_viewpoints.py --object sample --cluster --html

# matplotlib 3D 시각화
uv run scripts/generate_viewpoints.py --object sample --cluster --visualize
```

## CLI 인자 (클러스터링 관련)

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--cluster` | - | 클러스터링 활성화 |
| `--cluster-method` | `ward` | 클러스터링 방법: `ward` 또는 `dbscan` |
| `--max-diameter` | 82mm (`FOV_WIDTH * 2`) | [ward] 클러스터 최대 직경 (mm) |
| `--eps` | 41mm (`FOV_WIDTH`) | [dbscan] 이웃 반경 (mm) |
| `--min-samples` | 2 | [dbscan] 코어 포인트 최소 이웃 수 |
| `--compare` | - | ward와 dbscan 결과를 HTML 드롭다운으로 비교 |
| `--html` | - | plotly HTML 시각화 저장 |
| `--visualize` | - | matplotlib 3D 시각화 |
| `--dry-run` | - | 통계만 출력, HDF5 수정 안 함 |

## 알고리즘

### 1. 카메라 위치 계산

```
camera_positions = positions + normals * working_distance_m
```

클러스터링은 표면 위치가 아닌 **카메라 위치** 기준으로 수행한다. 로봇이 실제 방문하는 좌표가 카메라 위치이므로 경로 최적화에 더 적합하다.

### 2. 클러스터링 방법

#### Ward (Agglomerative Hierarchical)

```python
Z = linkage(camera_positions, method='ward')
cluster_ids = fcluster(Z, t=max_diameter_m, criterion='distance')
```

- 분산을 최소화하는 방향으로 병합
- **클러스터 크기가 균일**하게 나뉨
- `--max-diameter` 값이 클수록 클러스터 수 감소

#### DBSCAN (Density-Based)

```python
db = DBSCAN(eps=eps_m, min_samples=min_samples)
labels = db.fit_predict(camera_positions)
```

- 밀도 기반 클러스터링, 클러스터 수 자동 결정
- 노이즈 포인트(-1)는 각각 별도 클러스터로 할당
- **클러스터 크기 편차가 큼** (밀집 영역에 큰 클러스터)

### 3. 클러스터 내부 정렬

각 클러스터별로 `compute_pca_axes()` → `reorder_zigzag()` 호출. 클러스터 크기 < 3이면 단순 순서 유지.

### 4. 클러스터 간 정렬

Greedy Nearest Neighbor on centroids. 시작 클러스터: 기존 `path_order[0]`에 해당하는 뷰포인트의 클러스터.

### 5. 글로벌 path_order 생성

클러스터 방문 순서 × 클러스터 내부 순서를 결합하여 `(N,)` path_order 배열 생성.

## 성능 비교 (sample 124 viewpoints)

### Ward — max-diameter별

| 직경(mm) | 클러스터 수 | 경로 개선 |
|---------|-----------|---------|
| 60 | 29 | 14.4% |
| **82** (기본) | **23** | **13.1%** |
| **100** | **20** | **14.0%** |
| 120 | 17 | 5.4% |
| 150 | 12 | -2.8% |

### DBSCAN — eps별

| eps(mm) | 클러스터 수 | 경로 개선 |
|---------|-----------|---------|
| 20 | 23 | 1.7% |
| **30** | **10** | **7.0%** |
| 41 (기본) | 9 | 5.4% |
| 60 | 7 | -9.3% |

### 결론

- **Ward가 전반적으로 우세**: 클러스터 크기가 균일하여 지그재그 효과가 잘 살아남
- **DBSCAN**은 크기 편차가 커서(min=1, max=60) 경로 최적화 효과 제한적
- 최적 구간: Ward `max-diameter` 60~100mm

## HDF5 변경사항

```
viewpoints/
  positions      (N, 3)  [기존 유지]
  normals        (N, 3)  [기존 유지]
  path_order     (N,)    [--cluster 시 덮어쓰기: 클러스터 기반 새 순서]
  row_index      (N,)    [기존 유지]
  cluster_id     (N,)    int32  [--cluster 시 추가: 클러스터 할당]
  cluster_order  (K,)    int32  [--cluster 시 추가: 클러스터 방문 순서]
  pca_center/axis1/axis2 [기존 유지]

metadata/
  clustering_method        str    "ward" 또는 "dbscan"
  cluster_max_diameter_mm  float  [ward 전용]
  dbscan_eps_mm            float  [dbscan 전용]
  dbscan_min_samples       int    [dbscan 전용]
  num_clusters             int
  clustered_path_length_mm float
  original_path_length_mm  float
  clustering_timestamp     str
```

## 하류 호환성

`plan_motion.py`의 `load_viewpoints()`:
- `path_order`를 optional로 읽음 → 덮어쓴 값을 그대로 사용
- `cluster_id`, `cluster_order`가 있으면 하이브리드 궤적 모드 사용
- 없으면 dense scan IK 모드 사용

## 시각화 출력

| 모드 | 출력 파일 |
|------|----------|
| `--html` / `--compare` | `data/{object}/viewpoint/{num}/viewpoints_clustered.html` |
| `--visualize` | matplotlib 창 (저장 없음) |

HTML 시각화 내용:
- 메시(물체) 반투명 표시
- 뷰포인트를 클러스터별 색상 구분 (고대비 팔레트)
- 클러스터 내부 경로: 실선 (클러스터 색상)
- 클러스터 간 전환: 점선 회색
- 클러스터 centroid: 검정 다이아몬드 마커
- `--compare` 시 드롭다운으로 ward/dbscan 전환, 경로 길이 비교 표시
