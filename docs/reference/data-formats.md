# 데이터 형식

모든 물체 데이터는 `data/{object}/` 아래에 저장된다.

```text
mesh/          source.obj, source.usd, target mesh
viewpoint/N/   viewpoints_*.h5
ik/N/          glns_result*.h5
trajectory/N/ trajectory*.csv, trajectory*.npz
```

## Viewpoint HDF5

`viewpoints/positions`와 `viewpoints/normals`는 `(N, 3)` 필수 데이터다. 다음 항목은 생성 방식이나 legacy 파일에 따라 없을 수 있다.

- `path_order`, `row_index`
- `cluster_id`, `cluster_order`, `cluster_direction`
- `adjacency/edges`, `adjacency/component_id`
- `metadata`의 입력 mesh와 camera working distance

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
