"""Public viewpoint-generation and storage API."""

from .adjacency import (
    build_local_delaunay_adjacency,
    canonical_edge_set,
    components_from_edges,
    cut_vertices,
    expand_edges_by_hops,
)
from .mesh import (
    ALIGN_MODES,
    DEFAULT_STEP_TOL_LINEAR,
    load_meshes,
    load_source_geometry,
    load_source_parts,
)
from .models import (
    DEFAULT_DELAUNAY_DISTANCE_FACTOR,
    DEFAULT_DELAUNAY_MAX_NORMAL_ANGLE_DEG,
    DEFAULT_DELAUNAY_NEIGHBORS,
    ViewpointAdjacency,
    ViewpointData,
    ViewpointGenParams,
    ViewpointResult,
)
from .pipeline import (
    append_viewpoints,
    finalize_viewpoints,
    generate_viewpoints_core,
    prepare_viewpoints,
    spacings_m,
    subset_viewpoints,
)
from . import select, visibility
from .storage import (
    load_viewpoints_hdf5,
    save_viewpoints_hdf5,
    write_adjacency_into_h5,
)

__all__ = [
    "ALIGN_MODES",
    "DEFAULT_DELAUNAY_DISTANCE_FACTOR",
    "DEFAULT_DELAUNAY_MAX_NORMAL_ANGLE_DEG",
    "DEFAULT_DELAUNAY_NEIGHBORS",
    "DEFAULT_STEP_TOL_LINEAR",
    "ViewpointAdjacency",
    "ViewpointData",
    "ViewpointGenParams",
    "ViewpointResult",
    "build_local_delaunay_adjacency",
    "canonical_edge_set",
    "components_from_edges",
    "cut_vertices",
    "expand_edges_by_hops",
    "append_viewpoints",
    "finalize_viewpoints",
    "generate_viewpoints_core",
    "load_meshes",
    "load_source_geometry",
    "load_source_parts",
    "load_viewpoints_hdf5",
    "prepare_viewpoints",
    "save_viewpoints_hdf5",
    "select",
    "spacings_m",
    "subset_viewpoints",
    "visibility",
    "write_adjacency_into_h5",
]
