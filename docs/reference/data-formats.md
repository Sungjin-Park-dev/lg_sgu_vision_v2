# 데이터 형식

모든 물체 데이터는 `data/{object}/` 아래에 저장된다.

```text
mesh/          source.obj, source.usd, target mesh
viewpoint/N/   viewpoints_*.h5
ik/N/          glns_result*.h5
trajectory/N/ trajectory*.csv, trajectory*.npz
```

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

## GLNS 결과 HDF5

`input`에는 도달 가능한 viewpoint와 Delaunay 그래프가, `components/{id}`에는 성분별 상태, 선택 순서와 joint 후보가 저장된다. 앱은 `status=solved`인 성분을 재생한다.

## Trajectory CSV와 NPZ

CSV는 시간, UR20 6개 joint, target position과 quaternion을 담는 실행 파일이다. joint 값은 radian이다.

NPZ는 브라우저 재생용 sidecar다.

- `joints`
- `ee_positions`
- `is_transit`
- `times`

GLNS 검증 결과는 성분별 `glns_trajectory_comp{id}`와 연결된 `glns_trajectory_joined` 파일로 저장된다.
