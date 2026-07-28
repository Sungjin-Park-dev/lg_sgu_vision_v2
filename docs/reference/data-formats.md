# 데이터 형식

모든 물체 데이터는 `data/{object}/` 아래 **세 폴더**에만 저장된다.

```text
mesh/                입력 자산 — source.obj, source.usd, material
viewpoint/{N}/       viewpoints_{clustering_method}.h5
trajectory/{N}/      solution.h5                       GLNS 해 (verify 의 입력)
                     trajectory.csv | .npz             스캔 궤적 (실행용)
                     trajectory_home_to_start.csv|npz  HOME 브래킷 (옵션)
                     trajectory_end_to_home.csv|npz
                     home_move_{approach,return}.csv|npz
```

`{N}`은 viewpoint 개수다. 파일명 규칙은 `{역할}[_{변형}][_{세부}]`이고, 변형 토큰은 여러 개가
실제로 공존할 때만 붙는다 — viewpoint는 클러스터링 방법이 여럿이라 붙고, 궤적은 생산자가
GLNS 하나뿐이라 안 붙는다.

## Viewpoint HDF5

클러스터링 방법과 무관하게 항상 같은 세 계층으로 저장된다.

1. **기하** — `viewpoints/positions`, `viewpoints/normals` `(N, 3)` 필수.
   `metadata/camera_spec`의 working distance와 `metadata/input_mesh`.
2. **표면 그래프** — `viewpoints/adjacency/edges` `(E, 2)`. 기하에서만 유도되므로
   어떤 클러스터링을 쓰든 동일하다. 연결성분은 저장하지 않고 필요할 때
   `components_from_edges(edges, N)`로 파생한다.
3. **그룹핑과 방문 순서** — `viewpoints/cluster_id` `(N,)`,
   `viewpoints/cluster_order` `(K,)`, `viewpoints/path_order` `(N,)`.
   어떤 방법이 만들었는지는 `metadata/clustering_method`에만 기록된다.

legacy 파일에는 `row_index`, `cluster_direction`, `pca_*`,
`adjacency/component_id`가 남아 있을 수 있다. 더 이상 쓰지도 읽지도 않으므로
무시되며, 기존 파일은 그대로 열린다.

파일명은 `viewpoints_{clustering_method}.h5` 형태다. 읽는 쪽은
`config.resolve_viewpoint_path(object, N)`로 고르면 된다 — 정규 이름
`viewpoints.h5`가 있으면 그것을, 없으면 가장 최근 `viewpoints*.h5`를 쓴다.

## GLNS 해 HDF5 (`solution.h5`)

`input`에는 도달 가능한 viewpoint와 Delaunay 그래프가, `components/{id}`에는 성분별 상태,
선택 순서(`viewpoint_order`)와 고른 joint 분기(`selected_joints`)가 저장된다.

이건 시각화용 부산물이 아니라 **`verify.py`의 유일한 입력**이다 — 어느 IK 분기를 어떤 순서로
쓸지, 그리고 IK를 푼 물체 배치가 여기 들어 있다. 물체·viewpoint 수당 하나이며 재solve하면
덮어쓴다.

## Trajectory CSV와 NPZ

CSV는 시간, UR20 6개 joint, target position과 quaternion을 담는 실행 파일이다. joint 값은 radian이다.

NPZ는 브라우저 재생용 sidecar다.

- `joints`
- `ee_positions`
- `is_transit`
- `times`
- `meta` — 생성 조건 JSON(물체 배치, WD, spacing, reconfig 임계, 커버리지). 파일명이
  파라미터를 담지 않으므로 여기에 남긴다.

성분별 중간 궤적은 저장하지 않는다. 실행 대상은 전 성분을 이은 `trajectory.csv` 하나다.
