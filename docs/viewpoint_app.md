# viewpoint_app.py (viser studio)

viser 기반 인터랙티브 스튜디오. 물체를 골라 **그 자리에서 viewpoint를 생성**하거나
(파라미터 튜닝), 기존 `viewpoints*.h5`를 불러와 본다. 정적 plotly
HTML(`common/viewpoint_viz.py`)의 인터랙티브 대체 + 경로 순서 재생 + 실시간 재생성.

## 실행

```bash
uv run scripts/viser/viewpoint_app.py --object curved_structure   # 초기 물체 선택
uv run scripts/viser/viewpoint_app.py --viewpoints data/sample/viewpoint/124/viewpoints.h5
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

## Generate (coacd + sub-cluster)

`generate_viewpoints.py`의 import 시임을 직접 호출:
`load_meshes` → `prepare_grid` → `cluster_coacd` → `cluster_and_order`.

**Sampling 드롭다운** (`surface` 기본 / `grid`):
- `surface` — 표면 직접 균일 샘플링(Poisson-disk) + 순서 `graph`(탄젠트 Delaunay/TSP).
  곡면·측벽까지 표면적 기준으로 고르게 덮음 (PCA 투영 그리드의 곡면 누락 해결).
- `grid` — 기존 PCA 그리드 투영 + 순서 `zigzag`. 비교/판재용.
- 순서 모드는 Sampling에 따라 자동 페어링(surface→graph, grid→zigzag).

**Sub-cluster 드롭다운** (`agglomerative` 기본 / `dbscan`): CoACD 파트 내부 세분 방법.
- `agglomerative` — 공간 분할(**complete linkage + 거리 임계값**). 노브 = `max span (mm)` →
  **모든 클러스터의 카메라 위치 지름 ≤ 값 보장** = 멀리 떨어진 viewpoint가 한 클러스터로 묶이는
  것 원천 차단. (클러스터링은 **표면이 아니라 카메라 위치** 기준 — 렌더 마커·로봇 EE가 카메라
  위치이고, 곡면에선 표면이 가까워도 법선×110mm 오프셋으로 카메라가 벌어지기 때문.)
- `dbscan` — 기존 밀도 클러스터링. 노브 = `eps`/`min_samples`. (균일 표면에선 거대1+싱글톤 다발)
- CoACD 분해는 둘이 공유(캐시 동일) → 토글하며 **나란히 비교** 가능.

조절 슬라이더:
| 파라미터 | 범위 | 기본 | 비고 |
|----------|------|------|------|
| `surface spacing (mm)` | 5–40 | 12 | **[surface] 뷰포인트 밀도** (12mm≈60% 오버랩). 클수록 개수↓ (`count ≈ area/(2·spacing²)`, 제곱 반비례라 조금만 키워도 확 줄어듦) |
| `coacd_threshold` | 0.05–0.5 | 0.25 | |
| `max span (mm)` | 20–150 | 80 | **[agglomerative] 클러스터 최대 지름**. 작게 → 클러스터 많고 촘촘, 크게 → 적고 넓음. 지름 ≤ 이 값 보장 |
| `eps (mm)` | 5–80 | 1.5×spacing | **[dbscan]** spacing 바꾸면 자동으로 `1.5×spacing`으로 따라감(surface). 슬라이더는 유지 → 수동 조절 가능, 다음 spacing 변경 시 다시 기본값. grid 모드에선 자동 추적 안 함 |
| `min_samples` | 1–5 | 2 | [dbscan] |
| `normal_weight` | 0.0–0.2 | 0.05 | [dbscan]; agglomerative distance 모드는 순수 위치 |

> **개수 조절(spacing)**: `surface spacing`을 키우면 개수↓ (curved_structure 기준 9mm→374, 12mm→206,
> 15mm→133, 20mm→80). 제곱 반비례라 1~2mm만 바꿔도 크게 변하니 조금씩 조절. grid 모드에선 무시됨.
> **클러스터 크기(max span)**: agglomerative에서 클러스터 최대 지름(mm). 크게 → 클러스터 적고 넓음,
> 작게 → 많고 촘촘. complete-linkage라 지름이 이 값을 절대 안 넘음(멀리 떨어진 점 묶임 방지).

`[Generate]` 클릭 → 백그라운드 스레드로 재생성(UI 안 멈춤) → 씬 갱신 + 상태 표시
(`N vp · K clusters · path … (xx% reduction)`).

**CoACD 캐싱**: `(object, sampling_mode, surface_spacing, coacd_threshold)`별로 CoACD 결과를
캐시(= `--compare`의 `precomputed_coacd` 경로). **Sub-cluster(agglomerative↔dbscan)·그 노브
(`target size`/`eps`/`min_samples`)·`normal_weight`만 바꾸면** CoACD 재실행 없이 **~2s**.
`coacd_threshold`/물체/Sampling/spacing이 바뀌면 재실행. 배치(메시 로드+그리드/표면 샘플링+bottom
필터)도 `(object, sampling_mode, spacing)`별 1회 캐시.

`[Save h5]` → `data/{object}/viewpoint/{N}/viewpoints_coacd+{sub}.h5` 기록(sub=선택한
sub-cluster) → `plan_trajectory.py`가 바로 소비. (대상 디렉토리가 root 소유면 `PermissionError`
→ 상태에 안내. → [[root-owned-data-and-container-paths]] 참고용 메모.)

## 화면 구성 / 레이어

- **메시**(반투명) · **클러스터 마커**(클러스터별 **고유 색** — 황금비 HSV로 K개 생성, 25개
  팔레트 순환과 달리 K>25여도 색 재사용 없음) · **클러스터 경로선** · **전이선**(회색)
  · **CoACD parts**(생성 결과에만, 반투명 색 파트).
- **Layers** 폴더 체크박스로 각 레이어 on/off.
- **Playback** 폴더: `Step` 스크럽(현재 뷰포인트 하이라이트 + "지나온 경로" 트레일),
  `Play`/`Speed (vp/s)` 자동 재생. Step 슬라이더는 씬 로드마다 `max=N-1`로 재생성.

## 스코프 / 한계

- **1단계 = CoACD 고정**. Sub-cluster는 `agglomerative`/`dbscan` 토글. coacd 단독·dbscan 단독
  (CoACD 없이)은 CLI 사용.
- **Sampling = surface/grid 토글** (순서는 자동 페어링). surface+zigzag 같은 교차 조합은
  CLI(`--sampling-mode`/`--ordering-mode`)로.
- **material 필터 제외** — 전체 메시로 생성. `sample`은 비대상(초록) 영역도 포함됨
  (정확히 하려면 CLI `--material-rgb "170,163,158"`, 또는 추후 material 드롭다운 추가).
- **간격(grid spacing / surface spacing)·bottom 필터**는 기본값 고정.

## h5 의존성 (Existing 로드)

`viewpoints/{positions, normals}` 필수. `cluster_id`/`cluster_order`/`path_order`는
있으면 사용, 없으면 단일 클러스터·생성 순서로 폴백. mesh 경로는
`config.get_mesh_path(object, "source")` 우선(h5의 `input_mesh` 절대경로는 컨테이너
경로일 수 있어 폴백), camera 오프셋은 `camera_spec/working_distance_mm`.

HDF5 구조: [architecture.md](architecture.md#viewpointsh5) · 생성 코어:
[generate_viewpoints.md](generate_viewpoints.md).
