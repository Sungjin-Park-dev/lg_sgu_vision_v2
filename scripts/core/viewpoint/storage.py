"""Canonical viewpoint HDF5 loading and persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from common import config
from .models import ViewpointAdjacency, ViewpointData


def _decode_attr(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _optional_vector(group, name: str, *, length: int | None = None):
    if name not in group:
        return None
    value = np.asarray(group[name], dtype=np.int32)
    if value.ndim != 1 or (length is not None and value.shape != (length,)):
        expected = "(N,)" if length is not None else "one-dimensional"
        raise ValueError(f"viewpoints/{name} must be {expected}, got {value.shape}")
    return value


def load_viewpoints_hdf5(path: str | Path) -> ViewpointData:
    """Read one viewpoint file with consistent validation and legacy optionals."""
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Viewpoints file not found: {source_path}")

    with h5py.File(source_path, "r") as f:
        if "viewpoints" not in f:
            raise ValueError(f"{source_path} has no 'viewpoints' group")
        group = f["viewpoints"]
        missing = [name for name in ("positions", "normals") if name not in group]
        if missing:
            raise ValueError(
                f"{source_path} is missing required viewpoint datasets: {', '.join(missing)}"
            )

        positions = np.asarray(group["positions"], dtype=np.float64)
        normals = np.asarray(group["normals"], dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                f"viewpoints/positions must have shape (N, 3), got {positions.shape}"
            )
        if normals.shape != positions.shape:
            raise ValueError(
                f"viewpoints/normals shape {normals.shape} does not match "
                f"positions {positions.shape}"
            )
        count = len(positions)
        # 그룹핑 계층은 이 셋이 전부다. 어떤 클러스터링 방법이 만들었는지는
        # metadata/clustering_method 에만 남는다.
        path_order = _optional_vector(group, "path_order", length=count)
        cluster_id = _optional_vector(group, "cluster_id", length=count)
        cluster_order = _optional_vector(group, "cluster_order")

        adjacency = None
        if "adjacency" in group:
            adjacency_group = group["adjacency"]
            if "edges" not in adjacency_group:
                raise ValueError("viewpoints/adjacency exists but has no edges dataset")
            edges = np.asarray(adjacency_group["edges"], dtype=np.int32)
            if edges.ndim != 2 or edges.shape[1] != 2:
                raise ValueError(
                    f"viewpoints/adjacency/edges must have shape (E, 2), got {edges.shape}"
                )
            if len(edges) and (np.any(edges < 0) or np.any(edges >= count)):
                raise ValueError("viewpoints/adjacency/edges contains out-of-range indices")
            # 구버전 파일에 남아 있는 component_id 데이터셋은 읽지 않는다 —
            # 성분은 항상 edges 에서 파생한다(components_from_edges).
            attrs = adjacency_group.attrs
            stats = {
                key: (value.item() if isinstance(value, np.generic) else value)
                for key, value in attrs.items()
                if key in {
                    "num_edges", "num_components", "num_isolated",
                    "min_degree", "max_degree", "median_degree",
                    "median_edge_length_mm", "max_edge_length_mm",
                }
            }
            adjacency = ViewpointAdjacency(
                edges=edges,
                method=_decode_attr(attrs.get("method", "local_tangent_delaunay")),
                stats=stats,
                k_neighbors=int(attrs["k_neighbors"]) if "k_neighbors" in attrs else None,
                distance_factor=(
                    float(attrs["distance_factor"]) if "distance_factor" in attrs else None
                ),
                max_normal_angle_deg=(
                    float(attrs["max_normal_angle_deg"])
                    if "max_normal_angle_deg" in attrs else None
                ),
            )

        # 카메라 스펙의 출처 규칙: **config 는 새 h5 를 만들 때의 출발값, h5 는 만들어진 뒤의
        # 진실**이다. 여기서 config 로 되돌아가는 건 camera_spec 이 아예 없는 옛 파일뿐이고,
        # 그건 조용히 넘어가면 안 되므로 어느 키가 빠졌는지 찍어준다.
        input_mesh = None
        working_distance_m = float(config.CAMERA_WORKING_DISTANCE_MM) / 1000.0
        fov_width_m = float(config.CAMERA_FOV_WIDTH_MM) / 1000.0
        fov_height_m = float(config.CAMERA_FOV_HEIGHT_MM) / 1000.0
        missing = ["working_distance_mm", "fov_width_mm", "fov_height_mm"]
        if "metadata" in f:
            metadata = f["metadata"]
            if "input_mesh" in metadata.attrs:
                input_mesh = _decode_attr(metadata.attrs["input_mesh"])
            if "camera_spec" in metadata:
                cam_attrs = metadata["camera_spec"].attrs
                missing = [k for k in missing if k not in cam_attrs]
                if "working_distance_mm" in cam_attrs:
                    wd_mm = float(cam_attrs["working_distance_mm"])
                    # 기하학적으로 불가능한 WD 는 조용히 계획에 흘려보내지 않는다.
                    problem = config.working_distance_error(wd_mm)
                    if problem:
                        print(f"WARNING: {source_path.name}: {problem}")
                    working_distance_m = wd_mm / 1000.0
                if "fov_width_mm" in cam_attrs:
                    fov_width_m = float(cam_attrs["fov_width_mm"]) / 1000.0
                if "fov_height_mm" in cam_attrs:
                    fov_height_m = float(cam_attrs["fov_height_mm"]) / 1000.0
        if missing:
            print(
                f"WARNING: {source_path.name} has no camera_spec {', '.join(missing)} — "
                f"config defaults (FOV {config.CAMERA_FOV_WIDTH_MM:.0f}x"
                f"{config.CAMERA_FOV_HEIGHT_MM:.0f}mm, WD {config.CAMERA_WORKING_DISTANCE_MM:.0f}mm)"
                f" used instead. Regenerate the file if it was not built with that spec."
            )

    return ViewpointData(
        source_path=source_path,
        positions=positions,
        normals=normals,
        path_order=path_order,
        cluster_id=cluster_id,
        cluster_order=cluster_order,
        adjacency=adjacency,
        input_mesh=input_mesh,
        working_distance_m=working_distance_m,
        fov_width_m=fov_width_m,
        fov_height_m=fov_height_m,
    )


def _write_adjacency_group(viewpoints_grp, adjacency: dict, n_positions: int) -> None:
    """Write the canonical ``viewpoints/adjacency`` group (edges + attrs).

    Single source of truth for the adjacency schema — shared by ``save_viewpoints_hdf5``
    (full write) and ``write_adjacency_into_h5`` (in-place backfill into an existing file).
    The caller decides whether ``adjacency`` is present; this assumes it is.

    성분 라벨은 쓰지 않는다 — edges 와 어긋날 수 있는 파생값이라, 읽는 쪽이
    ``components_from_edges`` 로 그때그때 계산한다.
    """
    edges = np.asarray(adjacency['edges'], dtype=np.int32)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"adjacency edges must have shape (E, 2), got {edges.shape}")
    if len(edges):
        if np.any(edges < 0) or np.any(edges >= n_positions):
            raise ValueError("adjacency edges contain out-of-range viewpoint indices")
        if np.any(edges[:, 0] >= edges[:, 1]):
            raise ValueError("adjacency edges must be canonical undirected pairs (a < b)")
        if len(np.unique(edges, axis=0)) != len(edges):
            raise ValueError("adjacency edges contain duplicates")
    adjacency_grp = viewpoints_grp.create_group('adjacency')
    adjacency_grp.create_dataset('edges', data=edges)
    adjacency_grp.attrs['method'] = adjacency.get('method', 'local_tangent_delaunay')
    adjacency_grp.attrs['k_neighbors'] = int(adjacency['k_neighbors'])
    adjacency_grp.attrs['distance_factor'] = float(adjacency['distance_factor'])
    adjacency_grp.attrs['max_normal_angle_deg'] = float(adjacency['max_normal_angle_deg'])
    adjacency_grp.attrs['coordinate_space'] = 'camera_positions_object_local'
    adjacency_grp.attrs['edge_semantics'] = 'undirected_canonical'
    for key, value in adjacency.get('stats', {}).items():
        adjacency_grp.attrs[key] = value


def write_adjacency_into_h5(h5_path, adjacency: dict) -> Path:
    """Backfill/refresh ``viewpoints/adjacency`` in an EXISTING viewpoints h5, in place.

    Preserves all other datasets (positions/normals/cluster_id/path_order/...). Replaces any
    existing adjacency group so it is idempotent — which also strips the ``component_id``
    dataset older files carry, since the graph is now edges-only.
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "a") as f:
        if "viewpoints" not in f:
            raise ValueError(f"{h5_path} has no 'viewpoints' group")
        viewpoints_grp = f["viewpoints"]
        n_positions = int(viewpoints_grp["positions"].shape[0])
        if "adjacency" in viewpoints_grp:
            del viewpoints_grp["adjacency"]
        _write_adjacency_group(viewpoints_grp, adjacency, n_positions)
    return h5_path


def save_viewpoints_hdf5(
    positions: np.ndarray,
    normals: np.ndarray,
    output_path: str,
    metadata: Optional[dict] = None,
    camera_spec: Optional[dict] = None,
    path_order: Optional[np.ndarray] = None,
    cluster_id: Optional[np.ndarray] = None,
    cluster_order: Optional[np.ndarray] = None,
    cluster_metadata: Optional[dict] = None,
    adjacency: Optional[dict] = None,
) -> Path:
    """Save viewpoints to HDF5 file

    클러스터링 방법과 무관하게 항상 같은 세 계층을 쓴다: 기하(positions/normals +
    metadata/camera_spec), 표면 그래프(adjacency/edges), 그리고 선택된 그룹핑과 방문
    순서(cluster_id/cluster_order/path_order). 방법 이름은
    ``cluster_metadata['clustering_method']`` 로만 남는다.

    Args:
        cluster_id: (N,) int32 array — cluster assignment per viewpoint
        cluster_order: (K,) int32 array — cluster visit order
        cluster_metadata: dict with clustering parameters
        adjacency: build_local_delaunay_adjacency() 결과. viewpoints/adjacency 하위
            그룹으로 저장한다(edges + 파라미터/통계 attrs).
    """
    if positions.shape != normals.shape:
        raise ValueError(
            f"Positions and normals must have same shape, "
            f"got {positions.shape} and {normals.shape}"
        )
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"Positions must be (N, 3) array, got shape {positions.shape}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        viewpoints_grp = f.create_group('viewpoints')
        viewpoints_grp.create_dataset('positions', data=positions.astype(np.float32))
        viewpoints_grp.create_dataset('normals', data=normals.astype(np.float32))

        if path_order is not None:
            viewpoints_grp.create_dataset('path_order', data=path_order.astype(np.int32))

        if cluster_id is not None:
            viewpoints_grp.create_dataset('cluster_id', data=cluster_id.astype(np.int32))
        if cluster_order is not None:
            viewpoints_grp.create_dataset('cluster_order', data=cluster_order.astype(np.int32))

        if adjacency is not None:
            _write_adjacency_group(viewpoints_grp, adjacency, len(positions))

        metadata_grp = f.create_group('metadata')
        metadata_grp.attrs['num_viewpoints'] = len(positions)

        if metadata:
            for key, value in metadata.items():
                if key != 'camera_spec':
                    metadata_grp.attrs[key] = value

        if camera_spec:
            camera_spec_grp = metadata_grp.create_group('camera_spec')
            for key, value in camera_spec.items():
                camera_spec_grp.attrs[key] = value

        if cluster_metadata:
            for key, value in cluster_metadata.items():
                metadata_grp.attrs[key] = value

    print(f"  Saved {len(positions)} viewpoints to {output_path}")

    return output_path



# ============================================================================
# CLI Argument Parsing
# ============================================================================
