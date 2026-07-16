"""Local tangent-plane Delaunay adjacency construction."""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay, QhullError, cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .models import (
    DEFAULT_DELAUNAY_DISTANCE_FACTOR,
    DEFAULT_DELAUNAY_MAX_NORMAL_ANGLE_DEG,
    DEFAULT_DELAUNAY_NEIGHBORS,
)

def _tangent_basis(mean_normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """평균 법선에 직교하는 2D 탄젠트 프레임 (u, v)를 수치 안정적으로 구성.

    PCA가 아니라 클러스터의 실제 평균 법선을 사용 (surface-intrinsic).
    평균 법선이 0에 가깝거나 NaN이면 표준 (e_x, e_y)로 폴백.
    """
    mn = np.asarray(mean_normal, dtype=np.float64)
    norm = np.linalg.norm(mn)
    if not np.isfinite(norm) or norm < 1e-9:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    n = mn / norm
    # n과 가장 덜 정렬된 좌표축을 기준으로 잡아 near-parallel cross 회피
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = a - np.dot(a, n) * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    return u, v


def build_local_delaunay_adjacency(
    camera_positions: np.ndarray,
    normals: np.ndarray,
    k_neighbors: int = DEFAULT_DELAUNAY_NEIGHBORS,
    distance_factor: float = DEFAULT_DELAUNAY_DISTANCE_FACTOR,
    max_normal_angle_deg: float = DEFAULT_DELAUNAY_MAX_NORMAL_ANGLE_DEG,
) -> dict:
    """곡면 viewpoint를 위한 로컬 탄젠트 Delaunay 인접 그래프를 만든다.

    전역 3D Delaunay는 물체 내부를 가로지르는 tetrahedron edge를 만들 수 있다. 대신 각
    viewpoint의 k-nearest neighborhood를 그 점의 tangent plane에 투영하고, 로컬 2D
    Delaunay에서 중심점에 incident한 edge만 합친다. 거리와 법선 각도 필터가 반대편 표면 및
    장거리 chord를 제거한다.

    반환 edge는 ``(min_index, max_index)`` 형태의 정렬된 무방향 edge이며 중복이 없다.
    ``component_id``는 향후 서로 다른 표면 성분을 잇는 명시적 bridge 후보 생성에 사용한다.
    """
    points = np.asarray(camera_positions, dtype=np.float64)
    nrms = np.asarray(normals, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"camera_positions must have shape (N, 3), got {points.shape}")
    if nrms.shape != points.shape:
        raise ValueError(f"normals must have shape {points.shape}, got {nrms.shape}")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(nrms)):
        raise ValueError("camera_positions and normals must contain only finite values")
    if k_neighbors < 3:
        raise ValueError("k_neighbors must be >= 3")
    if distance_factor <= 0.0:
        raise ValueError("distance_factor must be > 0")
    if not 0.0 < max_normal_angle_deg <= 180.0:
        raise ValueError("max_normal_angle_deg must be in (0, 180]")

    n_points = len(points)
    if n_points == 0:
        return {
            "edges": np.empty((0, 2), dtype=np.int32),
            "component_id": np.empty((0,), dtype=np.int32),
            "method": "local_tangent_delaunay",
            "k_neighbors": int(k_neighbors),
            "distance_factor": float(distance_factor),
            "max_normal_angle_deg": float(max_normal_angle_deg),
            "stats": {
                "num_edges": 0, "num_components": 0, "num_isolated": 0,
                "min_degree": 0, "median_degree": 0.0, "max_degree": 0,
                "median_edge_length_mm": 0.0, "max_edge_length_mm": 0.0,
            },
        }

    norm_len = np.linalg.norm(nrms, axis=1)
    if np.any(norm_len < 1e-9):
        bad = np.where(norm_len < 1e-9)[0].tolist()
        raise ValueError(f"zero-length normals at viewpoint indices: {bad[:10]}")
    unit_normals = nrms / norm_len[:, None]

    if n_points == 1:
        return {
            "edges": np.empty((0, 2), dtype=np.int32),
            "component_id": np.zeros((1,), dtype=np.int32),
            "method": "local_tangent_delaunay",
            "k_neighbors": int(k_neighbors),
            "distance_factor": float(distance_factor),
            "max_normal_angle_deg": float(max_normal_angle_deg),
            "stats": {
                "num_edges": 0, "num_components": 1, "num_isolated": 1,
                "min_degree": 0, "median_degree": 0.0, "max_degree": 0,
                "median_edge_length_mm": 0.0, "max_edge_length_mm": 0.0,
            },
        }

    query_k = min(int(k_neighbors), n_points - 1)
    tree = cKDTree(points)
    knn_dist, knn_idx = tree.query(points, k=query_k + 1)
    if knn_dist.ndim == 1:
        knn_dist = knn_dist[:, None]
        knn_idx = knn_idx[:, None]

    # 각 점의 local spacing: 가까운 최대 3개 이웃 거리의 median. 중복점 때문에 0만
    # 남는 경우에는 전체 데이터의 양의 최근접 거리로 폴백한다.
    positive_nn = knn_dist[:, 1:][knn_dist[:, 1:] > 1e-12]
    global_spacing = float(np.median(positive_nn)) if positive_nn.size else 1e-9
    local_spacing = np.empty(n_points, dtype=np.float64)
    for i in range(n_points):
        ds = knn_dist[i, 1:min(4, query_k + 1)]
        ds = ds[ds > 1e-12]
        local_spacing[i] = float(np.median(ds)) if ds.size else global_spacing

    cos_limit = float(np.cos(np.deg2rad(max_normal_angle_deg)))
    edges: set[tuple[int, int]] = set()

    def _passes_filters(i: int, j: int) -> bool:
        if i == j:
            return False
        distance = float(np.linalg.norm(points[i] - points[j]))
        max_distance = distance_factor * max(local_spacing[i], local_spacing[j])
        if distance > max_distance + 1e-12:
            return False
        return float(np.dot(unit_normals[i], unit_normals[j])) >= cos_limit - 1e-12

    for i in range(n_points):
        # cKDTree 결과에서 self 위치를 가정하지 않고, 중심점을 항상 local index 0으로 둔다.
        local_indices = [i]
        local_indices.extend(
            int(j) for j in knn_idx[i]
            if int(j) != i and int(j) not in local_indices
        )
        if len(local_indices) < 2:
            continue

        local_indices_arr = np.asarray(local_indices, dtype=np.int32)
        u, v = _tangent_basis(unit_normals[i])
        centered = points[local_indices_arr] - points[i]
        projected = np.column_stack([centered @ u, centered @ v])

        centered_2d = projected - projected.mean(axis=0)
        singular = np.linalg.svd(centered_2d, compute_uv=False)
        rank_2d = int(np.sum(singular > max(singular[0] if singular.size else 0.0, 1.0) * 1e-10))

        candidate_local: set[int] = set()
        if rank_2d >= 2 and len(local_indices) >= 3:
            try:
                options = "QJ" if len(local_indices) >= 4 else None
                triangulation = Delaunay(projected, qhull_options=options)
                for simplex in triangulation.simplices:
                    if 0 in simplex:
                        candidate_local.update(int(j) for j in simplex if int(j) != 0)
            except QhullError:
                # 수치적으로 퇴화한 neighborhood는 아래 1D 인접 규칙으로 처리한다.
                rank_2d = 1

        if rank_2d < 2:
            # 공선점의 1D Delaunay analogue: 주축 정렬에서 중심점의 직전·직후만 연결.
            if singular.size and singular[0] > 1e-12:
                _, _, vh = np.linalg.svd(centered_2d, full_matrices=False)
                coord = centered_2d @ vh[0]
            else:
                coord = np.linalg.norm(centered, axis=1)
            order = np.argsort(coord, kind="stable")
            center_rank = int(np.where(order == 0)[0][0])
            if center_rank > 0:
                candidate_local.add(int(order[center_rank - 1]))
            if center_rank + 1 < len(order):
                candidate_local.add(int(order[center_rank + 1]))

        for local_j in candidate_local:
            j = int(local_indices_arr[local_j])
            if _passes_filters(i, j):
                edges.add((min(i, j), max(i, j)))

    edge_array = (np.asarray(sorted(edges), dtype=np.int32).reshape(-1, 2)
                  if edges else np.empty((0, 2), dtype=np.int32))
    degree = np.zeros(n_points, dtype=np.int32)
    if len(edge_array):
        np.add.at(degree, edge_array[:, 0], 1)
        np.add.at(degree, edge_array[:, 1], 1)
        rows = np.concatenate([edge_array[:, 0], edge_array[:, 1]])
        cols = np.concatenate([edge_array[:, 1], edge_array[:, 0]])
        graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_points, n_points))
        num_components, component_id = connected_components(graph, directed=False)
        edge_lengths = np.linalg.norm(
            points[edge_array[:, 0]] - points[edge_array[:, 1]], axis=1,
        )
    else:
        num_components = n_points
        component_id = np.arange(n_points, dtype=np.int32)
        edge_lengths = np.empty((0,), dtype=np.float64)

    stats = {
        "num_edges": int(len(edge_array)),
        "num_components": int(num_components),
        "num_isolated": int(np.sum(degree == 0)),
        "min_degree": int(degree.min()),
        "median_degree": float(np.median(degree)),
        "max_degree": int(degree.max()),
        "median_edge_length_mm": float(np.median(edge_lengths) * 1000.0) if len(edge_lengths) else 0.0,
        "max_edge_length_mm": float(edge_lengths.max() * 1000.0) if len(edge_lengths) else 0.0,
    }
    return {
        "edges": edge_array,
        "component_id": np.asarray(component_id, dtype=np.int32),
        "method": "local_tangent_delaunay",
        "k_neighbors": int(k_neighbors),
        "distance_factor": float(distance_factor),
        "max_normal_angle_deg": float(max_normal_angle_deg),
        "stats": stats,
    }
