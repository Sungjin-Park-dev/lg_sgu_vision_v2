# generate_viewpoints.py

메시 표면에 PCA 그리드 기반 뷰포인트를 생성하고, 클러스터링 + GTSP로 방문 순서를 최적화한다.

## 파이프라인

```
OBJ 로드 → (material RGB 필터) → PCA 주축 계산 → 그리드 샘플링 → 표면 투영/법선
→ bottom-facing 필터 → 클러스터링 → 클러스터 내부 zigzag → GTSP 클러스터 순서
→ HDF5 + 인터랙티브 HTML 저장
```

클러스터링은 항상 수행된다 (opt-out 없음). HTML 시각화는 항상 생성된다.

## CLI 옵션

### 뷰포인트 생성
| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--object` | (필수) | 오브젝트 이름 |
| `--material-rgb "R,G,B"` | None | 재질 RGB 필터 (예: `"170,163,158"`) |
| `--color-tolerance` | 5.0 | RGB 매칭 허용 오차 |
| `--row-spacing` | FOV_HEIGHT × (1-overlap) mm | 행 간격 |
| `--col-spacing` | FOV_WIDTH × (1-overlap) mm | 열 간격 |
| `--no-filter-bottom` | off | 바닥향 뷰포인트 필터 비활성화 |
| `--bottom-angle` | 80.0 | 바닥향 필터 각도 (deg) |

### 클러스터링
| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--cluster-method` | `dbscan` | `dbscan` \| `coacd` \| `coacd+dbscan` |
| `--eps` | FOV_WIDTH mm | [dbscan] 이웃 반경 (mm) |
| `--min-samples` | 2 | [dbscan] 코어 포인트 최소 이웃 수 |
| `--normal-weight` | 0.0 | [dbscan] 법선 가중치 (m). 0이면 위치만 사용 |
| `--coacd-threshold` | 0.05 | [coacd] concavity threshold (낮을수록 파트 많아짐) |

### 비교/디버그
| 옵션 | 설명 |
|------|------|
| `--compare` | 선택된 방법의 파라미터 변형 비교 HTML 생성 |
| `--dry-run` | 통계만 출력, HDF5 저장 안 함 |

## 클러스터링 방법

| 방법 | 설명 |
|------|------|
| `dbscan` (기본) | 위치 (+ 선택적 법선) 기반 밀도 클러스터링. `--eps`, `--normal-weight`로 조절 |
| `coacd` | 메시를 convex 파트로 분해, 각 뷰포인트를 가장 가까운 파트에 할당 |
| `coacd+dbscan` | 1단계 CoACD로 파트 분리 → 2단계 각 파트 내 DBSCAN 세분화 |

## 클러스터 내부 정렬

전역 축(cam_axis1/cam_axis2)과 원본 grid `row_index`를 사용해 `reorder_zigzag()` 수행. 클러스터별 로컬 PCA는 생략 — `row_index_override`가 행 구분을 담당하므로 로컬 axis1이 불필요하고, axis2는 클러스터 간 열 방향 일관성을 위해 전역 값을 사용 (곡면 클러스터에서의 행 오분류 방지 — [archive/zigzag-row-assignment-issue.md](archive/zigzag-row-assignment-issue.md) 참조).

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

`plan_motion.py`의 `load_viewpoints()`는 `cluster_id`, `cluster_order`를 **필수**로 읽는다. 두 데이터가 없으면 에러로 종료. 레거시 non-clustered 모드는 제거됨.
