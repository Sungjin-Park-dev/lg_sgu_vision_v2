"""Public viewpoint-generation and storage API."""

from .adjacency import build_local_delaunay_adjacency
from .clustering import cluster_coacd
from .mesh import load_meshes
from .models import (
    DEFAULT_DELAUNAY_DISTANCE_FACTOR,
    DEFAULT_DELAUNAY_MAX_NORMAL_ANGLE_DEG,
    DEFAULT_DELAUNAY_NEIGHBORS,
    ViewpointAdjacency,
    ViewpointData,
    ViewpointGenParams,
    ViewpointResult,
)
from .pipeline import cluster_and_order, generate_viewpoints_core, prepare_grid
from .sampling import compute_path_length
from .storage import (
    load_viewpoints_hdf5,
    save_viewpoints_hdf5,
    write_adjacency_into_h5,
)

__all__ = [
    "DEFAULT_DELAUNAY_DISTANCE_FACTOR",
    "DEFAULT_DELAUNAY_MAX_NORMAL_ANGLE_DEG",
    "DEFAULT_DELAUNAY_NEIGHBORS",
    "ViewpointAdjacency",
    "ViewpointData",
    "ViewpointGenParams",
    "ViewpointResult",
    "build_local_delaunay_adjacency",
    "cluster_and_order",
    "cluster_coacd",
    "compute_path_length",
    "generate_viewpoints_core",
    "load_meshes",
    "load_viewpoints_hdf5",
    "prepare_grid",
    "save_viewpoints_hdf5",
    "write_adjacency_into_h5",
]
