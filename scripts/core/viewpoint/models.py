"""Data models shared by viewpoint generation and downstream consumers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


DEFAULT_DELAUNAY_NEIGHBORS = 12
DEFAULT_DELAUNAY_DISTANCE_FACTOR = 2.5
DEFAULT_DELAUNAY_MAX_NORMAL_ANGLE_DEG = 75.0


@dataclass(frozen=True)
class ViewpointAdjacency:
    """Optional local-surface adjacency stored below ``viewpoints/adjacency``.

    ``edges``가 그래프의 유일한 진실이다. 연결성분이 필요하면
    ``adjacency.components_from_edges(edges, n)``로 파생한다 — 저장하지 않는다.
    """

    edges: np.ndarray
    method: str
    stats: dict[str, object]
    k_neighbors: int | None = None
    distance_factor: float | None = None
    max_normal_angle_deg: float | None = None


@dataclass(frozen=True)
class ViewpointData:
    """Canonical in-memory representation of a viewpoint HDF5 file.

    세 계층으로 고정된다 — 기하(positions/normals/working_distance), 표면 그래프
    (adjacency), 선택된 그룹핑과 방문 순서(cluster_id/cluster_order/path_order).
    그룹핑을 만든 방법은 ``metadata/clustering_method``에만 기록되고, 어떤 방법이든
    이 세 계층의 모양은 동일하다.
    """

    source_path: Path
    positions: np.ndarray
    normals: np.ndarray
    path_order: np.ndarray | None
    cluster_id: np.ndarray | None
    cluster_order: np.ndarray | None
    adjacency: ViewpointAdjacency | None
    input_mesh: str | None
    working_distance_m: float
    # Camera-spec snapshot captured at generation time (metadata/camera_spec),
    # so preview/execute can default the FOV visualization to what the
    # viewpoints were actually planned with. Fall back to config when absent.
    fov_width_m: float
    fov_height_m: float

    @property
    def count(self) -> int:
        return int(len(self.positions))

    @property
    def visit_indices(self) -> np.ndarray:
        """Indices in stored visit order; legacy files fall back to HDF5 order."""
        if self.path_order is None:
            return np.arange(self.count, dtype=np.int32)
        return np.argsort(self.path_order, kind="stable").astype(np.int32)


@dataclass
class ViewpointGenParams:
    """Parameters for the importable viewpoint generation pipeline."""

    material_rgb: Optional[str] = None
    color_tolerance: float = 5.0
    row_spacing_mm: Optional[float] = None
    col_spacing_mm: Optional[float] = None
    filter_bottom: bool = True
    bottom_angle: float = 80.0
    filter_interior: bool = False
    interior_hull_align_min: float = 0.3
    cluster_method: str = "dbscan"
    eps_mm: Optional[float] = None
    min_samples: int = 2
    normal_weight: float = 0.0
    coacd_threshold: float = 0.05
    target_size: int = 12
    max_span_mm: Optional[float] = None
    sampling_mode: str = "grid"
    surface_spacing_mm: Optional[float] = None
    ordering_mode: str = "zigzag"
    build_delaunay: bool = True
    delaunay_neighbors: int = DEFAULT_DELAUNAY_NEIGHBORS
    delaunay_distance_factor: float = DEFAULT_DELAUNAY_DISTANCE_FACTOR
    delaunay_max_normal_angle_deg: float = DEFAULT_DELAUNAY_MAX_NORMAL_ANGLE_DEG


@dataclass
class ViewpointResult:
    """In-memory generation result; persistence remains the caller's choice."""

    positions: np.ndarray
    normals: np.ndarray
    camera_positions: np.ndarray
    path_order: np.ndarray
    cluster_id: np.ndarray
    cluster_order: np.ndarray
    coacd_parts: Optional[list]
    coacd_ids: Optional[np.ndarray]
    row_spacing_m: float
    col_spacing_m: float
    original_path_length_mm: float
    clustered_path_length_mm: float
    num_clusters: int
    cluster_meta: dict
    adjacency: Optional[dict]
    method: str
    label: str
