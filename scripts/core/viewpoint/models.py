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
    """Optional local-surface adjacency stored below ``viewpoints/adjacency``."""

    edges: np.ndarray
    component_id: np.ndarray | None
    method: str
    stats: dict[str, object]
    k_neighbors: int | None = None
    distance_factor: float | None = None
    max_normal_angle_deg: float | None = None


@dataclass(frozen=True)
class ViewpointData:
    """Canonical in-memory representation of a viewpoint HDF5 file."""

    source_path: Path
    positions: np.ndarray
    normals: np.ndarray
    path_order: np.ndarray | None
    row_index: np.ndarray | None
    cluster_id: np.ndarray | None
    cluster_order: np.ndarray | None
    cluster_direction: np.ndarray | None
    adjacency: ViewpointAdjacency | None
    input_mesh: str | None
    working_distance_m: float

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
    row_index: np.ndarray
    cluster_id: np.ndarray
    cluster_order: np.ndarray
    cluster_direction: np.ndarray
    coacd_parts: Optional[list]
    coacd_ids: Optional[np.ndarray]
    pca: dict
    row_spacing_m: float
    col_spacing_m: float
    original_path_length_mm: float
    clustered_path_length_mm: float
    num_clusters: int
    cluster_meta: dict
    adjacency: Optional[dict]
    method: str
    label: str
