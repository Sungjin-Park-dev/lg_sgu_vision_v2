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
        path_order = _optional_vector(group, "path_order", length=count)
        row_index = _optional_vector(group, "row_index", length=count)
        cluster_id = _optional_vector(group, "cluster_id", length=count)
        cluster_order = _optional_vector(group, "cluster_order")
        cluster_direction = _optional_vector(group, "cluster_direction")
        if (
            cluster_order is not None
            and cluster_direction is not None
            and cluster_order.shape != cluster_direction.shape
        ):
            raise ValueError(
                "viewpoints/cluster_direction must match viewpoints/cluster_order"
            )

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
            component_id = _optional_vector(
                adjacency_group, "component_id", length=count,
            )
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
                component_id=component_id,
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

        input_mesh = None
        working_distance_m = float(config.CAMERA_WORKING_DISTANCE_MM) / 1000.0
        if "metadata" in f:
            metadata = f["metadata"]
            if "input_mesh" in metadata.attrs:
                input_mesh = _decode_attr(metadata.attrs["input_mesh"])
            if (
                "camera_spec" in metadata
                and "working_distance_mm" in metadata["camera_spec"].attrs
            ):
                working_distance_m = (
                    float(metadata["camera_spec"].attrs["working_distance_mm"]) / 1000.0
                )

    return ViewpointData(
        source_path=source_path,
        positions=positions,
        normals=normals,
        path_order=path_order,
        row_index=row_index,
        cluster_id=cluster_id,
        cluster_order=cluster_order,
        cluster_direction=cluster_direction,
        adjacency=adjacency,
        input_mesh=input_mesh,
        working_distance_m=working_distance_m,
    )


def _write_adjacency_group(viewpoints_grp, adjacency: dict, n_positions: int) -> None:
    """Write the canonical ``viewpoints/adjacency`` group (edges + component_id + attrs).

    Single source of truth for the adjacency schema — shared by ``save_viewpoints_hdf5``
    (full write) and ``write_adjacency_into_h5`` (in-place backfill into an existing file).
    The caller decides whether ``adjacency`` is present; this assumes it is.
    """
    edges = np.asarray(adjacency['edges'], dtype=np.int32)
    component_id = np.asarray(adjacency['component_id'], dtype=np.int32)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"adjacency edges must have shape (E, 2), got {edges.shape}")
    if component_id.shape != (n_positions,):
        raise ValueError(
            f"adjacency component_id must have shape ({n_positions},), "
            f"got {component_id.shape}"
        )
    if len(edges):
        if np.any(edges < 0) or np.any(edges >= n_positions):
            raise ValueError("adjacency edges contain out-of-range viewpoint indices")
        if np.any(edges[:, 0] >= edges[:, 1]):
            raise ValueError("adjacency edges must be canonical undirected pairs (a < b)")
        if len(np.unique(edges, axis=0)) != len(edges):
            raise ValueError("adjacency edges contain duplicates")
    adjacency_grp = viewpoints_grp.create_group('adjacency')
    adjacency_grp.create_dataset('edges', data=edges)
    adjacency_grp.create_dataset('component_id', data=component_id)
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
    existing adjacency group so it is idempotent. Used by viewpoint_studio's "Build + Save
    Delaunay" action to add the GLNS graph to older coacd-only files.
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
    pca_data: Optional[dict] = None,
    row_index: Optional[np.ndarray] = None,
    cluster_id: Optional[np.ndarray] = None,
    cluster_order: Optional[np.ndarray] = None,
    cluster_direction: Optional[np.ndarray] = None,
    cluster_metadata: Optional[dict] = None,
    adjacency: Optional[dict] = None,
) -> Path:
    """Save viewpoints to HDF5 file

    Args:
        pca_data: dict with 'center' (3,), 'axis1' (3,), 'axis2' (3,) arrays
        row_index: (N,) int32 array — row index per viewpoint
        cluster_id: (N,) int32 array — cluster assignment per viewpoint
        cluster_order: (K,) int32 array — cluster visit order
        cluster_direction: (K,) int32 array — 0=Forward, 1=Reverse per cluster
        cluster_metadata: dict with clustering parameters
        adjacency: build_local_delaunay_adjacency() 결과. 기존 reader와 호환되는
            viewpoints/adjacency 하위 그룹으로 저장한다.
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

        if row_index is not None:
            viewpoints_grp.create_dataset('row_index', data=row_index.astype(np.int32))

        if pca_data is not None:
            viewpoints_grp.create_dataset('pca_center', data=np.asarray(pca_data['center'], dtype=np.float32))
            viewpoints_grp.create_dataset('pca_axis1', data=np.asarray(pca_data['axis1'], dtype=np.float32))
            viewpoints_grp.create_dataset('pca_axis2', data=np.asarray(pca_data['axis2'], dtype=np.float32))

        if cluster_id is not None:
            viewpoints_grp.create_dataset('cluster_id', data=cluster_id.astype(np.int32))
        if cluster_order is not None:
            viewpoints_grp.create_dataset('cluster_order', data=cluster_order.astype(np.int32))
        if cluster_direction is not None:
            viewpoints_grp.create_dataset('cluster_direction', data=cluster_direction.astype(np.int32))

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
