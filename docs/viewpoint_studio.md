# viewpoint_studio.py (viser studio)

viser 기반 인터랙티브 스튜디오. 물체를 골라 **그 자리에서 viewpoint를 생성**하거나
(파라미터 튜닝), 기존 `viewpoints*.h5`를 불러와 본다. 정적 plotly
HTML(`core/viewpoint/visualization.py`)의 인터랙티브 대체 + 경로 순서 재생 + 실시간 재생성.

## 실행

```bash
uv run scripts/apps/viewpoint_studio.py --object curved_structure   # 초기 물체 선택
uv run scripts/apps/viewpoint_studio.py --viewpoints data/sample/viewpoint/124/viewpoints.h5
```

`http://localhost:8080` 접속. `Ctrl+C`로 종료.

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--object` | (첫 물체) | 시작 시 선택할 물체 |
| `--viewpoints` | None | 시작 시 불러올 h5 |
| `--data-root` | `data/` | 데이터 루트 |
| `--host` / `--port` | `0.0.0.0` / `8080` | 서버 바인드 |

## 두 가지 소스 (물체 중심)

- **Object** 드롭다운 — `data/*/mesh/source.obj` 가진 물체.
- **Existing h5** 드롭다운 — `data/{object}/viewpoint/*/viewpoints*.h5` 로드.
- **Generate** — 파라미터로 in-process 재생성 (아래).

## Generate (surface + coacd + sub-cluster)

`core.viewpoint`의 공개 import 시임을 직접 호출:
`load_meshes` → `prepare_grid` → `cluster_coacd` → `cluster_and_order`.

`viewpoint_studio.py`는 **surface 샘플링 전용**이다.

- `surface` — 표면 직접 균일 샘플링(FPS) + 순서 `lawnmower`(탄젠트 row sweep).
  곡면·측벽까지 표면적 기준으로 고르게 덮음 (PCA 투영 그리드의 곡면 누락 해결).
- `grid` 샘플링 코드는 `viewpoint/cli.py`/CLI에만 남겨둔다. 스튜디오 UI에서는 선택하지 않는다.
- surface 간격은 직접 mm로 입력하지 않고, 카메라 FOV와 overlap에서 계산한다.
- 클러스터 내부 순서는 표면점 기준 tangent plane에서 긴 축을 scan, 짧은 축을 row로 잡고
  `→ ← → ←` 형태로 훑는다. row 간격은 FOV-derived spacing을 사용하고, 두 시작 방향 중
  실제 camera position 경로가 짧은 쪽을 선택한다.

**Sub-cluster 드롭다운** (`agglomerative` 기본 / `dbscan`): CoACD 파트 내부 세분 방법.
- `agglomerative` — 공간 분할(**complete linkage + 거리 임계값**). 노브 = `max span (mm)` →
  **모든 클러스터의 카메라 위치 지름 ≤ 값 보장** = 멀리 떨어진 viewpoint가 한 클러스터로 묶이는
  것 원천 차단. (클러스터링은 **표면이 아니라 카메라 위치** 기준 — 렌더 마커·로봇 EE가 카메라
  위치이고, 곡면에선 표면이 가까워도 법선×working-distance 오프셋으로 카메라가 벌어지기 때문.)
- `dbscan` — 기존 밀도 클러스터링. 노브 = `eps`만 노출한다.
  내부 고정값은 `min_samples=2`, `normal_weight=0.0`.
- CoACD 분해는 둘이 공유(캐시 동일) → 토글하며 **나란히 비교** 가능.
- Sub-cluster 선택에 따라 관련 옵션만 표시한다: `agglomerative`는 `max span`,
  `dbscan`은 `eps`.

조절 슬라이더:
| 파라미터 | 범위 | 기본 | 비고 |
|----------|------|------|------|
| `FOV overlap (%)` | 20–90 | `config.CAMERA_OVERLAP_RATIO` | `surface_spacing = min(FOV_width, FOV_height) × (1 - overlap)`. 예: 50mm FOV, 70%면 15mm |
| `coacd_threshold` | 0.05–0.5 | 0.25 | |
| `max span (mm)` | 50–500 | 250 | **[agglomerative] 클러스터 최대 지름**. 작게 → 클러스터 많고 촘촘, 크게 → 적고 넓음. 지름 ≤ 이 값 보장 |
| `eps (mm)` | 5–80 | 1.5×surface_spacing | **[dbscan]** overlap을 바꾸면 자동으로 `1.5×surface_spacing`으로 따라감. 슬라이더는 유지 → 수동 조절 가능, 다음 overlap 변경 시 다시 기본값 |

> **개수 조절(overlap)**: overlap을 키우면 surface spacing이 작아지고 viewpoint 개수는 증가한다.
> 개수는 대략 `count ≈ area/(surface_spacing²)`라 제곱 반비례로 변한다.
> **클러스터 크기(max span)**: agglomerative에서 클러스터 최대 지름(mm). 크게 → 클러스터 적고 넓음,
> 작게 → 많고 촘촘. complete-linkage라 지름이 이 값을 절대 안 넘음(멀리 떨어진 점 묶임 방지).

`[Generate]` 클릭 → 백그라운드 스레드로 재생성(UI 안 멈춤) → 씬 갱신 + 상태 표시
(`N vp · K clusters · path … (xx% reduction)`).

**CoACD 캐싱**: `(object, surface_spacing, coacd_threshold)`별로 CoACD 결과를
캐시(= `--compare`의 `precomputed_coacd` 경로). **Sub-cluster(agglomerative↔dbscan)·그 노브
(`max span`/`eps`)만 바꾸면** CoACD 재실행 없이 **~2s**.
`coacd_threshold`/물체/overlap이 바뀌면 재실행. 배치(메시 로드+표면 샘플링+bottom
필터)도 `(object, surface_spacing)`별 1회 캐시.

`[Save h5]` → `data/{object}/viewpoint/{N}/viewpoints_{method}.h5` 기록(method=`coacd+{sub}`)
→ `trajectory/cli.py`가 바로 소비. (대상 디렉토리가 root 소유면 `PermissionError`
→ 상태에 안내. → [[root-owned-data-and-container-paths]] 참고용 메모.)

## 화면 구성 / 레이어

- **메시**(반투명) · **표면점**(working-distance 전 `positions`) · **클러스터 마커**(working-distance
  반영 후 `camera_positions`, 클러스터별 **고유 색** — 황금비 HSV로 K개 생성, 25개
  팔레트 순환과 달리 K>25여도 색 재사용 없음) · **클러스터 경로선** · **전이선**(회색)
  · **CoACD parts**(생성 결과에만, 반투명 색 파트).
- **Layers** 폴더 체크박스로 각 레이어 on/off.
- **Playback** 폴더: `Step` 스크럽(현재 뷰포인트 하이라이트 + "지나온 경로" 트레일),
  `Play`/`Speed (vp/s)` 자동 재생. Step 슬라이더는 씬 로드마다 `max=N-1`로 재생성.

## 스코프 / 한계

- **1단계 = CoACD 고정**. Sub-cluster는 `agglomerative`/`dbscan` 토글.
  CoACD-only나 CoACD 없이 `dbscan` 단독으로 보는 경우는 CLI 사용.
- **Sampling = surface / ordering = lawnmower 고정**. grid 샘플링이나 surface+zigzag 같은 교차 조합은
  CLI(`--sampling-mode`/`--ordering-mode`)에서만 사용.
- **material 필터 = 오브젝트별 하드코딩** (`OBJECT_TARGET_MATERIAL`). 등록된 오브젝트는 그 재질
  면만 샘플링하고, 미등록은 전체 메시. 예: `sample` → 초록(0,255,0) = 검사대상만(비대상 회색 제외).
  표시 메시는 전체(맥락용), viewpoint만 대상 영역에 생성. 다른 재질을 쓰려면 CLI `--material-rgb`.
- **bottom 필터**는 기본값 고정.

## h5 의존성 (Existing 로드)

`viewpoints/{positions, normals}` 필수. `cluster_id`/`cluster_order`/`path_order`는
있으면 사용, 없으면 단일 클러스터·생성 순서로 폴백. mesh 경로는
`config.get_mesh_path(object, "source")` 우선(h5의 `input_mesh` 절대경로는 컨테이너
경로일 수 있어 폴백), camera 오프셋은 `camera_spec/working_distance_mm`.

HDF5 구조: [architecture.md](architecture.md#viewpointsh5) · 생성 코어:
[generate_viewpoints.md](generate_viewpoints.md).
