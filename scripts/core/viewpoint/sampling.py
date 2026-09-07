"""Viewpoint sampling and initial path construction."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import trimesh


# ============================================================================
# Grid Viewpoint Generation
# ============================================================================

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
) -> Tuple[np.ndarray, np.ndarray]:
    """표면 직접 균일 샘플링(Farthest Point Sampling)으로 뷰포인트를 생성한다.

    PCA 평면 투영 그리드와 달리 메시 표면 위에서 직접 균일 분포를 뽑아,
    곡면·측벽도 표면적 기준으로 고르게 덮는다(평면 투영의 곡면 누락 문제 해결).

    Args:
        mesh: 대상 메시
        spacing_m: 목표 표면 간격(미터). 목표 개수는 area / spacing²로 계산한다.

    Returns:
        (positions, normals) — 표면점과 그 점이 앉은 면의 단위 법선.
        방문 순서와 행 구조는 만들지 않는다: 순서는 cluster ordering 이 정하고,
        표면 FPS 에는 행 개념이 없다.
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

    return positions, normals


# ============================================================================
# Clustering
# ============================================================================
