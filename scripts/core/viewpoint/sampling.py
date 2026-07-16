"""Viewpoint sampling and initial path construction."""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

def compute_path_length(positions: np.ndarray, path_order: np.ndarray) -> float:
    """경로 순서대로 연결했을 때 총 유클리드 거리 합"""
    sorted_idx = np.argsort(path_order)
    ordered = positions[sorted_idx]
    diffs = np.diff(ordered, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def reorder_zigzag(
    positions: np.ndarray,
    axis1: np.ndarray,
    axis2: np.ndarray,
    row_spacing_m: float,
    row_index_override: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """고정된 뷰포인트 집합에 대해 축 기준 지그재그 순서를 재할당한다.

    Args:
        positions: (N, 3) 뷰포인트 위치
        axis1: 행(슬라이스) 방향 단위 벡터
        axis2: 열(스캔) 방향 단위 벡터
        row_spacing_m: 행 구분 간격 (미터)
        row_index_override: (N,) 외부에서 제공하는 행 인덱스. None이면 양자화로 추정.

    Returns:
        path_order: (N,) 지그재그 경로 순서
        row_index: (N,) 행 인덱스
        n_rows: 행 개수
    """
    axis1 = axis1 / np.linalg.norm(axis1)
    axis2 = axis2 / np.linalg.norm(axis2)

    center = positions.mean(axis=0)
    centered = positions - center
    proj1 = centered @ axis1
    proj2 = centered @ axis2

    N = len(positions)
    if row_index_override is not None:
        row_index = row_index_override.copy()
    else:
        # 행 할당: proj1 값을 row_spacing 간격으로 양자화
        row_index = np.round((proj1 - proj1.min()) / max(row_spacing_m, 1e-9)).astype(np.int32)

    # 각 행 내에서 axis2 기준 정렬, 홀수 행은 역방향
    path_order = np.zeros(N, dtype=np.int32)
    order_idx = 0

    unique_rows = np.unique(row_index)
    for i, r in enumerate(unique_rows):
        mask = row_index == r
        indices = np.where(mask)[0]
        col_vals = proj2[indices]
        if i % 2 == 0:
            sorted_in_row = indices[np.argsort(col_vals)]
        else:
            sorted_in_row = indices[np.argsort(col_vals)[::-1]]
        for idx in sorted_in_row:
            path_order[idx] = order_idx
            order_idx += 1

    # 행 인덱스를 0부터 재할당
    _, row_index = np.unique(row_index, return_inverse=True)
    row_index = row_index.astype(np.int32)
    n_rows = len(unique_rows)

    return path_order, row_index, n_rows



def compute_pca_axes(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA로 주축 2개를 계산한다.

    지그재그 래스터에 최적화:
    - axis1 (행/슬라이스 방향) = 두 번째 주축 (짧은 축) → 행 수 최소화
    - axis2 (열/스캔 방향) = 첫 번째 주축 (긴 축) → 긴 연속 스캔

    Args:
        points: (N, 3) 표면 점들

    Returns:
        center: 점들의 중심
        axis1: 두 번째 주축 (행 방향, 짧은 축)
        axis2: 가장 긴 주축 (열 방향, 긴 축)
    """
    center = points.mean(axis=0)
    centered = points - center

    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # eigh는 오름차순 → 내림차순 정렬
    sorted_indices = np.argsort(eigenvalues)[::-1]
    longest = eigenvectors[:, sorted_indices[0]]
    second = eigenvectors[:, sorted_indices[1]]

    longest = longest / np.linalg.norm(longest)
    second = second / np.linalg.norm(second)

    # axis1 = 짧은 축 (행), axis2 = 긴 축 (열)
    axis1 = second
    axis2 = longest

    return center, axis1, axis2


# ============================================================================
# Grid Viewpoint Generation
# ============================================================================

def generate_grid_viewpoints(
    mesh: trimesh.Trimesh,
    row_spacing_m: float,
    col_spacing_m: float,
    axis1: Optional[np.ndarray] = None,
    axis2: Optional[np.ndarray] = None,
    center: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    그리드 뷰포인트를 생성한다.

    axis1/axis2/center가 None이면 PCA로 계산하고,
    값이 주어지면 그대로 사용한다.

    Args:
        mesh: 대상 메시
        row_spacing_m: axis1 방향 간격 (미터)
        col_spacing_m: axis2 방향 간격 (미터)
        axis1: 슬라이스 축 (None이면 PCA 사용)
        axis2: 슬라이스 내 정렬 축 (None이면 PCA 사용)
        center: 그리드 중심 (None이면 PCA 사용)

    Returns:
        positions: (N, 3) 그리드 포인트 위치 (메시 표면 위)
        normals: (N, 3) 각 포인트의 표면 법선
        path_order: (N,) 지그재그 경로 순서 인덱스
        row_index: (N,) 각 뷰포인트의 행 인덱스
        center: (3,) PCA/그리드 중심
        axis1: (3,) 행 축 (슬라이스 방향)
        axis2: (3,) 열 축 (스캔 방향)
    """
    # 1. 표면에서 충분한 점을 균일 샘플링 (시드 고정으로 재현성 보장)
    np.random.seed(42)
    num_sample_points = max(10000, len(mesh.faces))
    sample_points, _ = trimesh.sample.sample_surface(mesh, num_sample_points)

    # 2. 축 결정: 외부 주입 또는 PCA
    if axis1 is None or axis2 is None or center is None:
        center, axis1, axis2 = compute_pca_axes(sample_points)
        if verbose:
            print(f"  PCA center: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
            print(f"  PCA axis1 (row/slice): [{axis1[0]:.4f}, {axis1[1]:.4f}, {axis1[2]:.4f}]")
            print(f"  PCA axis2 (col): [{axis2[0]:.4f}, {axis2[1]:.4f}, {axis2[2]:.4f}]")
    else:
        axis1 = axis1 / np.linalg.norm(axis1)
        axis2 = axis2 / np.linalg.norm(axis2)
        if verbose:
            print(f"  Center: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
            print(f"  Axis1 (row/slice): [{axis1[0]:.4f}, {axis1[1]:.4f}, {axis1[2]:.4f}]")
            print(f"  Axis2 (col): [{axis2[0]:.4f}, {axis2[1]:.4f}, {axis2[2]:.4f}]")

    axis3 = np.cross(axis1, axis2)
    axis3 = axis3 / np.linalg.norm(axis3)

    # 3. PCA 좌표계로 변환
    centered = sample_points - center
    proj1 = centered @ axis1
    proj2 = centered @ axis2

    # 4. 그리드 범위 계산
    min1, max1 = proj1.min(), proj1.max()
    min2, max2 = proj2.min(), proj2.max()

    if verbose:
        print(f"  Range axis1: [{min1:.4f}, {max1:.4f}] ({max1 - min1:.4f} m)")
        print(f"  Range axis2: [{min2:.4f}, {max2:.4f}] ({max2 - min2:.4f} m)")

    mid1 = (min1 + max1) / 2.0
    mid2 = (min2 + max2) / 2.0
    half_extent1 = (max1 - min1) / 2.0
    half_extent2 = (max2 - min2) / 2.0

    n_rows = max(1, int(np.floor(2 * half_extent1 / row_spacing_m)) + 1)
    n_cols = max(1, int(np.floor(2 * half_extent2 / col_spacing_m)) + 1)

    row_coords = mid1 + np.linspace(-half_extent1, half_extent1, n_rows) if n_rows > 1 else np.array([mid1])
    col_coords = mid2 + np.linspace(-half_extent2, half_extent2, n_cols) if n_cols > 1 else np.array([mid2])

    if n_rows > 1:
        row_coords = np.arange(n_rows) * row_spacing_m
        row_coords = row_coords - row_coords.mean() + mid1
    if n_cols > 1:
        col_coords = np.arange(n_cols) * col_spacing_m
        col_coords = col_coords - col_coords.mean() + mid2

    if verbose:
        print(f"  Grid: {n_rows} rows x {n_cols} cols = {n_rows * n_cols} points")
        print(f"  Row spacing: {row_spacing_m * 1000:.1f} mm, Col spacing: {col_spacing_m * 1000:.1f} mm")

    # 5. 그리드 포인트 생성 (지그재그 순서)
    grid_points_3d = []
    path_order = []
    row_indices = []
    order_idx = 0

    for i, r in enumerate(row_coords):
        cols = col_coords if i % 2 == 0 else col_coords[::-1]
        for c in cols:
            point_3d = center + r * axis1 + c * axis2
            grid_points_3d.append(point_3d)
            path_order.append(order_idx)
            row_indices.append(i)
            order_idx += 1

    grid_points_3d = np.array(grid_points_3d)
    path_order = np.array(path_order, dtype=np.int32)
    row_index = np.array(row_indices, dtype=np.int32)

    # 6. 각 그리드 포인트에서 가장 가까운 메시 표면의 법선 조회
    closest_points, distances, face_indices = trimesh.proximity.closest_point(mesh, grid_points_3d)
    normals = mesh.face_normals[face_indices]

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    normals = normals / norms

    positions = closest_points.astype(np.float32)
    normals = normals.astype(np.float32)

    if verbose:
        print(f"  Max distance to surface: {distances.max():.6f} m")
        print(f"  Mean distance to surface: {distances.mean():.6f} m")

    return positions, normals, path_order, row_index, center, axis1, axis2


def _nn_path_length(points: np.ndarray) -> float:
    """Greedy nearest-neighbor 경로 길이 (미터). 클러스터링 전 baseline 보고용.

    PCA/그리드 구조에 의존하지 않는 단순 베이스라인. 점이 2개 미만이면 0.
    """
    n = len(points)
    if n < 2:
        return 0.0
    visited = np.zeros(n, dtype=bool)
    cur = 0
    visited[0] = True
    total = 0.0
    for _ in range(n - 1):
        d = np.linalg.norm(points - points[cur], axis=1)
        d[visited] = np.inf
        nxt = int(np.argmin(d))
        total += float(d[nxt])
        visited[nxt] = True
        cur = nxt
    return total


def filter_interior_viewpoints(
    mesh: trimesh.Trimesh,
    positions: np.ndarray,
    normals: np.ndarray,
    hull_align_min: float = 0.3,
    verbose: bool = True,
) -> np.ndarray:
    """속이 빈 물체의 '안쪽 껍데기' viewpoint 를 제거해 **바깥 껍데기만** 남긴다.

    각 표면점의 법선이, 그 점에서 가장 가까운 convex-hull 표면의 **바깥 법선**과 이루는 정렬
    (dot)이 hull_align_min 미만이면 안쪽 면(공동을 향함)으로 보고 제거한다. 바깥 껍데기 점은
    법선이 hull 바깥 법선과 정렬(≈+1)되고, 안쪽 껍데기 점은 반대(≈−1)라 깔끔히 갈린다.

    가시성/가림이 아니라 **껍데기 구조**로 판정하므로, 얕고 넓은 물체에서 '위에서 열린 틈으로
    내려다보이는 안쪽 바닥'까지 제거된다(순수 가림 필터로는 안 잡히는 것). 벽 두께와도 무관.
    볼록한 물체는 모든 점이 hull 과 정렬돼 아무것도 안 지운다.
    ※ 주의: 오목한 '바깥' 형상(예: 홈·계단)이 있는 물체는 그 면도 지울 수 있어 부적합 →
      config.OBJECT_FILTER_INTERIOR 로 **물체별 opt-in** 할 때만 쓴다.

    Args:
        mesh: 대상(전체) 메시 — convex hull 계산용.
        positions: (N,3) 표면점 좌표(mesh 로컬, hull 과 동일 프레임).
        normals: (N,3) 표면 법선.
    Returns:
        keep: (N,) bool — 남길(=바깥 껍데기) viewpoint.
    """
    n = len(positions)
    if n == 0:
        return np.ones(0, dtype=bool)
    hull = mesh.convex_hull
    _, _, tri = hull.nearest.on_surface(np.asarray(positions, dtype=np.float64))
    hull_normals = hull.face_normals[tri]                      # 가장 가까운 hull 면의 바깥 법선
    unit_n = normals / np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9, None)
    align = np.einsum("ij,ij->i", unit_n, hull_normals)
    keep = align >= hull_align_min
    if verbose:
        print(f"  Interior filter (outer-shell): removed {int((~keep).sum())}/{n} inner-skin "
              f"viewpoints (hull-normal align < {hull_align_min}); {int(keep.sum())} remain")
    return keep


def farthest_point_sample_indices(points: np.ndarray, count: int) -> np.ndarray:
    """Greedy farthest-point sampling over candidate points.

    The candidates are already sampled on the mesh surface. FPS then picks a
    deterministic subset that maximizes spacing in 3D Euclidean distance. This is
    not geodesic FPS, but is a strong practical improvement over pure random or
    weak rejection sampling for inspection viewpoint coverage.
    """
    n = len(points)
    if count >= n:
        return np.arange(n, dtype=np.int32)
    if count <= 0:
        return np.empty(0, dtype=np.int32)

    pts = np.asarray(points, dtype=np.float64)
    selected = np.empty(count, dtype=np.int32)

    centroid = pts.mean(axis=0)
    selected[0] = int(np.argmin(np.sum((pts - centroid) ** 2, axis=1)))

    min_dist2 = np.full(n, np.inf, dtype=np.float64)
    for i in range(1, count):
        last = pts[selected[i - 1]]
        diff = pts - last
        dist2 = np.einsum("ij,ij->i", diff, diff)
        min_dist2 = np.minimum(min_dist2, dist2)
        selected[i] = int(np.argmax(min_dist2))

    return selected


def generate_surface_viewpoints(
    mesh: trimesh.Trimesh,
    spacing_m: float,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """표면 직접 균일 샘플링(Farthest Point Sampling)으로 뷰포인트를 생성한다.

    PCA 평면 투영 그리드와 달리 메시 표면 위에서 직접 균일 분포를 뽑아,
    곡면·측벽도 표면적 기준으로 고르게 덮는다(평면 투영의 곡면 누락 문제 해결).

    Args:
        mesh: 대상 메시
        spacing_m: 목표 표면 간격(미터). 목표 개수는 area / spacing²로 계산한다.

    Returns:
        generate_grid_viewpoints와 동일한 7-튜플
        (positions, normals, path_order, row_index, center, axis1, axis2).
        path_order/row_index는 placeholder(cluster ordering이 별도로 대체).
    """
    count = max(16, int(mesh.area / max(spacing_m, 1e-6) ** 2))
    oversample_factor = 20
    candidate_count = max(count, count * oversample_factor)
    if verbose:
        print(f"Generating surface viewpoints (FPS over area-weighted candidates)...")
        print(f"  Surface area: {mesh.area:.6f} m2, target spacing: {spacing_m * 1000:.1f} mm")
        print(f"  Target count: {count}")
        print(f"  Candidate count: {candidate_count}")

    candidates, candidate_faces = trimesh.sample.sample_surface(mesh, candidate_count, seed=42)
    keep = farthest_point_sample_indices(candidates, count)
    samples = np.asarray(candidates[keep])
    face_indices = np.asarray(candidate_faces[keep])

    normals = mesh.face_normals[face_indices]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    normals = (normals / norms).astype(np.float32)

    positions = samples.astype(np.float32)
    N = len(positions)
    if verbose:
        print(f"  Generated: {N} viewpoints (target spacing ≈ {spacing_m * 1000:.1f} mm)")

    center, axis1, axis2 = compute_pca_axes(samples.astype(np.float64))
    path_order = np.arange(N, dtype=np.int32)   # placeholder (cluster ordering이 대체)
    row_index = np.zeros(N, dtype=np.int32)     # placeholder

    return positions, normals, path_order, row_index, center, axis1, axis2


# ============================================================================
# Clustering
# ============================================================================
