"""Public viewpoint-generation and storage API."""

from .adjacency import (
    build_local_delaunay_adjacency,
    canonical_edge_set,
    components_from_edges,
    cut_vertices,
    expand_edges_by_hops,
)
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
from .pipeline import generate_viewpoints_core, prepare_viewpoints
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
    "components_from_edges",
    "cut_vertices",
    "expand_edges_by_hops",
    "canonical_edge_set",
    "generate_viewpoints_core",
    "load_meshes",
    "load_viewpoints_hdf5",
    "prepare_viewpoints",
    "save_viewpoints_hdf5",
    "write_adjacency_into_h5",
]
