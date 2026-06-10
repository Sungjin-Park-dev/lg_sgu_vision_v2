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

## Generate (coacd+dbscan)

`generate_viewpoints.py`의 import 시임을 직접 호출:
`load_meshes` → `prepare_grid` → `cluster_coacd` → `cluster_and_order`.

조절 슬라이더 (클러스터링만):
| 파라미터 | 범위 | 기본 |
|----------|------|------|
| `coacd_threshold` | 0.05–0.5 | 0.25 |
| `eps (mm)` | 5–80 | 20 |
| `normal_weight` | 0.0–0.2 | 0.05 |
| `min_samples` | 1–5 | 2 |

`[Generate]` 클릭 → 백그라운드 스레드로 재생성(UI 안 멈춤) → 씬 갱신 + 상태 표시
(`N vp · K clusters · path … (xx% reduction)`).

**CoACD 캐싱**: `(object, coacd_threshold)`별로 CoACD 결과를 캐시(= `--compare`의
`precomputed_coacd` 경로). `eps`/`normal_weight`/`min_samples`만 바꾸면 CoACD 재실행 없이
**~2s**(DBSCAN+GTSP). `coacd_threshold`/물체가 바뀌면 CoACD 재실행 **~6s**. 그리드(메시
로드+PCA 그리드+bottom 필터)도 물체별 1회 캐시.

`[Save h5]` → `data/{object}/viewpoint/{N}/viewpoints_coacd+dbscan.h5` 기록 →
`plan_trajectory.py`가 바로 소비. (대상 디렉토리가 root 소유면 `PermissionError` →
상태에 안내. → [[root-owned-data-and-container-paths]] 참고용 메모.)

## 화면 구성 / 레이어

- **메시**(반투명) · **클러스터 마커**(방문 순위 색) · **클러스터 경로선** · **전이선**(회색)
  · **CoACD parts**(생성 결과에만, 반투명 색 파트).
- **Layers** 폴더 체크박스로 각 레이어 on/off.
- **Playback** 폴더: `Step` 스크럽(현재 뷰포인트 하이라이트 + "지나온 경로" 트레일),
  `Play`/`Speed (vp/s)` 자동 재생. Step 슬라이더는 씬 로드마다 `max=N-1`로 재생성.

## 스코프 / 한계

- **method = `coacd+dbscan` 고정** (UI 단순화). dbscan/coacd 단독은 CLI 사용.
- **material 필터 제외** — 전체 메시로 생성. `sample`은 비대상(초록) 영역도 포함됨
  (정확히 하려면 CLI `--material-rgb "170,163,158"`, 또는 추후 material 드롭다운 추가).
- **그리드 간격·bottom 필터**는 기본값 고정.

## h5 의존성 (Existing 로드)

`viewpoints/{positions, normals}` 필수. `cluster_id`/`cluster_order`/`path_order`는
있으면 사용, 없으면 단일 클러스터·생성 순서로 폴백. mesh 경로는
`config.get_mesh_path(object, "source")` 우선(h5의 `input_mesh` 절대경로는 컨테이너
경로일 수 있어 폴백), camera 오프셋은 `camera_spec/working_distance_mm`.

HDF5 구조: [architecture.md](architecture.md#viewpointsh5) · 생성 코어:
[generate_viewpoints.md](generate_viewpoints.md).
