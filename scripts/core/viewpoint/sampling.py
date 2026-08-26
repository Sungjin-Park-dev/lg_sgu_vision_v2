"""Viewpoint sampling and initial path construction."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import trimesh


def compute_path_length(positions: np.ndarray, path_order: np.ndarray) -> float:
    """경로 순서대로 연결했을 때 총 유클리드 거리 합"""
    sorted_idx = np.argsort(path_order)
    ordered = positions[sorted_idx]
    diffs = np.diff(ordered, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


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
