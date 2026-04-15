# generate_viewpoints.py

메시 표면에 PCA 그리드 기반 뷰포인트를 생성하고, 클러스터링으로 경로를 최적화한다.

## 파이프라인

```
OBJ 로드 → (재질 필터) → PCA 주축 계산 → 그리드 샘플링 → 표면 투영
→ zigzag 경로 → 클러스터링 → GTSP 클러스터 순서 → HDF5 + HTML 저장
```

## 주요 섹션

| 섹션 | 함수 | 설명 |
|------|------|------|
| Material/Mesh | `parse_mtl_file`, `parse_obj_material_usage`, `match_material_by_color`, `extract_target_mesh` | OBJ/MTL 파싱, RGB 기반 재질 필터링 |
| Path Utilities | `compute_path_length`, `reorder_zigzag`, `compute_pca_axes` | PCA 주축 계산, 지그재그 경로 정렬 |
| Viewpoint 생성 | `generate_grid_viewpoints` | PCA 축 기반 그리드 → 표면 투영 → 법선 계산 |
| Clustering | `cluster_dbscan`, `cluster_coacd`, `cluster_coacd_dbscan` | 3가지 클러스터링 방법 |
| GTSP 순서 | `order_clusters_gtsp`, `compute_cluster_internal_order`, `build_clustered_path_order` | 클러스터 간 방문 순서 + 내부 순서 최적화 |
| HDF5 저장 | `save_viewpoints_hdf5` | 뷰포인트, 경로, 클러스터 메타데이터 저장 |

## 클러스터링 방법

| 방법 | 설명 |
|------|------|
| `dbscan` (기본) | 위치 + 법선 기반 밀도 클러스터링. `--eps`, `--normal-weight`로 조절 |
| `coacd` | 메시를 convex 파트로 분해, 각 뷰포인트를 가장 가까운 파트에 할당 |
| `coacd+dbscan` | 1단계 CoACD로 파트 분리 → 2단계 각 파트 내 DBSCAN 세분화 |

## 클러스터 간 순서 결정 (GTSP)

클러스터 방문 순서와 각 클러스터의 내부 방향을 동시에 최적화한다.

### 문제 정의

각 클러스터는 PCA zigzag로 정렬된 양 끝점(endpoint_a, endpoint_b)을 가진다.
정방향(a→b)과 역방향(b→a) 중 어느 방향으로 진입/퇴장할지를 선택해야 하므로,
K개 클러스터 × 2방향 = **GTSP** (Generalized TSP) 문제가 된다.

### 풀이: Noon-Bean 변환 + 더미 노드

1. **노드 구성**: 클러스터 k마다 F_k(정방향), R_k(역방향) 2개 노드 + 더미 D = 총 2K+1개
2. **Noon-Bean 변환**: GTSP를 ATSP로 변환
   - 클러스터 내 사이클: F_k↔R_k 비용 0
   - 클러스터 간 간선: predecessor로 shift (`ATSP[R_k,F_l] = GTSP[F_k,F_l]`)
3. **더미 노드 D**: D↔모든 노드 비용 0 → open path (시작/끝점 자유)
4. **OR-Tools ATSP**: PATH_CHEAPEST_ARC + GUIDED_LOCAL_SEARCH (2초)
5. **디코딩**: 투어에서 D 제거 → 2개씩 쌍으로 묶어 진입 노드로 클러스터 순서+방향 결정

### 출력

- `cluster_order`: (K,) 클러스터 방문 순서
- `cluster_direction`: (K,) 0=Forward(a→b), 1=Reverse(b→a)

## CLI 옵션

```
--object NAME          오브젝트 이름 (필수)
--material-rgb "R,G,B" 재질 RGB 필터
--cluster-method       dbscan | coacd | coacd+dbscan
--eps                  [dbscan] 이웃 반경 (mm)
--normal-weight        [dbscan] 법선 가중치 (m)
--coacd-threshold      [coacd] concavity threshold
--compare              파라미터 변형 비교 HTML 생성
--dry-run              통계만 출력, 파일 저장 안 함
```

## 출력

- `data/{object}/viewpoint/{num}/viewpoints_{method}.h5` — 뷰포인트 데이터
- `data/{object}/viewpoint/{num}/viewpoints_{method}.html` — 인터랙티브 3D 시각화

## 시각화

HTML 시각화는 `scripts/visualize_viewpoints.py`에서 담당한다. 자세한 내용은 [visualize_viewpoints.md](visualize_viewpoints.md) 참조.
