#!/usr/bin/env python3
"""
뷰포인트 생성 및 클러스터링 기반 경로 순서 최적화

메시 표면에 PCA 주축을 기반으로 그리드 형태의 뷰포인트를 생성한다.
선택적으로 공간 클러스터링을 수행하여 경로 순서를 최적화한다.

입력: OBJ 파일 (선택적으로 재질 RGB 색상 필터 적용 가능)
출력: 표면 위치, 법선 벡터, 경로 순서 인덱스가 포함된 HDF5 파일

사용법:
    # 기본: FOV 기반 자동 간격
    uv run scripts/core/generate_viewpoints.py --object sample

    # 재질 필터
    uv run scripts/core/generate_viewpoints.py --object sample --material-rgb "0,255,0"

    # 간격 수동 오버라이드 (mm)
    uv run scripts/core/generate_viewpoints.py --object sample --row-spacing 5.0 --col-spacing 5.0

    # 파라미터 변형 비교 (드롭다운 HTML)
    uv run scripts/core/generate_viewpoints.py --object sample --cluster-method coacd --compare

    # CoACD 기반 클러스터링 (메시 convex decomposition)
    uv run scripts/core/generate_viewpoints.py --object sample --cluster-method coacd

    uv run scripts/core/generate_viewpoints.py --object sample --material-rgb "0,255,0" --cluster-method dbscan
    uv run scripts/core/generate_viewpoints.py --object sample --material-rgb "0,255,0" --cluster-method coacd
    
    # Sample
    uv run scripts/core/generate_viewpoints.py --object sample --material-rgb "0,255,0" --cluster-method coacd+dbscan --normal-weight 0.05 --coacd-threshold 0.25 --compare
    uv run scripts/core/generate_viewpoints.py --object sample --material-rgb "0,255,0" --cluster-method coacd+dbscan --normal-weight 0.05 --coacd-threshold 0.25 --eps 20

    # Glass
    uv run scripts/core/generate_viewpoints.py --object glass --cluster-method coacd+dbscan --normal-weight 0.05 --coacd-threshold 0.25 --compare
"""

import os
import sys
import argparse
import time
import warnings
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass

import trimesh
import h5py
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import config
from common.math_utils import quaternion_to_rotation_matrix
from common.viewpoint_viz import visualize_clusters_html


# ============================================================================
# MTL/OBJ Parsing
# ============================================================================

def parse_mtl_file(mtl_path: str) -> Dict[str, Dict]:
    """Parse MTL file to extract material properties"""
    materials = {}
    current_material = None

    if not os.path.exists(mtl_path):
        print(f"Warning: MTL file not found: {mtl_path}")
        return materials

    with open(mtl_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) == 0:
                continue

            command = parts[0]

            if command == 'newmtl':
                if len(parts) >= 2:
                    current_material = ' '.join(parts[1:])
                    materials[current_material] = {}
            elif command == 'Kd' and current_material:
                if len(parts) >= 4:
                    try:
                        r = float(parts[1])
                        g = float(parts[2])
                        b = float(parts[3])
                        materials[current_material]['Kd'] = np.array([r, g, b], dtype=np.float64)
                    except ValueError:
                        pass

    return materials


def parse_obj_material_usage(obj_path: str) -> Tuple[Dict[int, str], str]:
    """Parse OBJ file to determine which material is used for each triangle"""
    triangle_materials = {}
    current_material = None
    face_index = 0
    mtl_file = None

    with open(obj_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) == 0:
                continue

            command = parts[0]

            if command == 'mtllib':
                if len(parts) >= 2:
                    mtl_filename = ' '.join(parts[1:])
                    obj_dir = os.path.dirname(obj_path)
                    mtl_file = os.path.join(obj_dir, mtl_filename)
            elif command == 'usemtl':
                if len(parts) >= 2:
                    current_material = ' '.join(parts[1:])
            elif command == 'f':
                if current_material:
                    triangle_materials[face_index] = current_material
                face_index += 1

    return triangle_materials, mtl_file


def rgb_to_kd(r: int, g: int, b: int) -> np.ndarray:
    """Convert RGB (0-255) to Kd (0.0-1.0)"""
    return np.array([r / 255.0, g / 255.0, b / 255.0], dtype=np.float64)


def kd_to_rgb(kd: np.ndarray) -> Tuple[int, int, int]:
    """Convert Kd (0.0-1.0) to RGB (0-255)"""
    r = int(round(kd[0] * 255))
    g = int(round(kd[1] * 255))
    b = int(round(kd[2] * 255))
    return (r, g, b)


def match_material_by_color(
    materials: Dict[str, Dict],
    target_rgb: Tuple[int, int, int],
    tolerance: float = 5.0
) -> List[str]:
    """Find materials matching target RGB color within tolerance"""
    target_kd = rgb_to_kd(*target_rgb)
    matched_materials = []

    for mat_name, mat_props in materials.items():
        if 'Kd' not in mat_props:
            continue

        mat_kd = mat_props['Kd']
        mat_rgb = kd_to_rgb(mat_kd)
        distance = np.sqrt(
            (mat_rgb[0] - target_rgb[0])**2 +
            (mat_rgb[1] - target_rgb[1])**2 +
            (mat_rgb[2] - target_rgb[2])**2
        )

        if distance <= tolerance:
            matched_materials.append(mat_name)

    return matched_materials


def extract_target_mesh(
    mesh: trimesh.Trimesh,
    triangle_materials: Dict[int, str],
    target_materials: List[str]
) -> trimesh.Trimesh:
    """Extract submesh containing only triangles with target materials"""
    target_face_indices = []

    for face_idx in range(len(mesh.faces)):
        if face_idx in triangle_materials:
            mat_name = triangle_materials[face_idx]
            if mat_name in target_materials:
                target_face_indices.append(face_idx)

    if len(target_face_indices) == 0:
        raise ValueError(
            f"No triangles found with target materials: {target_materials}\n"
            f"Available materials: {set(triangle_materials.values())}"
        )

    target_mesh = mesh.submesh([target_face_indices], append=True)
    return target_mesh


# ============================================================================
# Path & PCA Utilities
# ============================================================================

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

def cluster_dbscan(
    camera_positions: np.ndarray,
    normals: np.ndarray,
    eps_m: float,
    min_samples: int = 2,
    normal_weight: float = 0.0,
) -> np.ndarray:
    """DBSCAN 위치+법선 기반 클러스터링.

    position (x,y,z)와 법선 (nx,ny,nz)를 결합한 feature로 클러스터링한다.
    normal_weight로 법선의 비중을 조절한다.

    Args:
        camera_positions: (N, 3) 카메라 위치
        normals: (N, 3) 표면 법선 벡터 (단위 벡터)
        eps_m: 이웃 반경 (미터)
        min_samples: 코어 포인트 최소 이웃 수
        normal_weight: 법선 가중치 (미터 단위, 0이면 위치만 사용)

    Returns:
        cluster_ids: (N,) 0-based 클러스터 할당 (노이즈 포인트도 개별 클러스터로 할당)
    """
    if normal_weight > 0:
        features = np.hstack([camera_positions, normal_weight * normals])
    else:
        features = camera_positions
    db = DBSCAN(eps=eps_m, min_samples=min_samples)
    labels = db.fit_predict(features)
    next_id = labels.max() + 1 if labels.max() >= 0 else 0
    for i in range(len(labels)):
        if labels[i] == -1:
            labels[i] = next_id
            next_id += 1
    return labels


def cluster_agglomerative(
    positions: np.ndarray,
    normals: np.ndarray,
    target_size: int = 12,
    normal_weight: float = 0.0,
    n_neighbors: int = 8,
    max_span_mm: Optional[float] = None,
) -> np.ndarray:
    """Agglomerative 기반 공간 분할 클러스터링. 두 가지 노브 모드.

    균일 밀도 표면에서 DBSCAN이 '거대 클러스터 1개 + 싱글톤 다발'로 깨지는 문제를
    피해, 컴팩트한 구역으로 나눈다.

    - max_span_mm 지정(권장): **complete linkage + distance_threshold**.
      → 모든 클러스터의 지름(내부 최대 점간 거리) ≤ max_span 보장. 멀리 떨어진
      viewpoint가 한 클러스터로 묶이는 것을 원천 차단. 클러스터 수는 자동 결정.
      이 모드는 **순수 위치 거리** 기준(threshold가 mm로 직접 해석되도록 normal_weight 미적용).
    - max_span_mm None: Ward + kNN 연결성, n_clusters = round(N / target_size)
      (평균 크기 ≈ target_size; 지름은 제한 안 함).

    Args:
        positions: (N, 3) 위치
        normals: (N, 3) 표면 법선
        target_size: [ward 모드] 클러스터당 목표 점 개수
        normal_weight: [ward 모드] 법선 가중치 (feature에 결합)
        n_neighbors: [ward 모드] 연결성 그래프 kNN 수
        max_span_mm: [distance 모드] 클러스터 최대 지름 (mm)

    Returns:
        cluster_ids: (N,) 0-based 클러스터 할당
    """
    from sklearn.cluster import AgglomerativeClustering

    n = len(positions)
    if n < 2:
        return np.zeros(n, dtype=np.int32)

    # distance 모드: complete linkage로 클러스터 지름 ≤ max_span 보장 (순수 위치)
    if max_span_mm is not None:
        model = AgglomerativeClustering(
            n_clusters=None, distance_threshold=max_span_mm / 1000.0, linkage='complete',
        )
        labels = model.fit_predict(positions)
        return labels.astype(np.int32)

    # ward 모드: 개수 기반
    from sklearn.neighbors import kneighbors_graph
    if n < 3:
        return np.zeros(n, dtype=np.int32)
    k = max(1, int(round(n / max(target_size, 1))))
    if k <= 1:
        return np.zeros(n, dtype=np.int32)
    k = min(k, n)

    if normal_weight > 0:
        features = np.hstack([positions, normal_weight * normals])
    else:
        features = positions

    # 연결성 그래프는 위치(표면 인접)로, Ward 비용은 feature로.
    conn = kneighbors_graph(
        positions, n_neighbors=min(n_neighbors, n - 1), include_self=False,
    )
    with warnings.catch_warnings():
        # 연결성 그래프가 분리되면 sklearn이 트리를 완성하며 경고 → 무음 처리.
        warnings.simplefilter("ignore")
        model = AgglomerativeClustering(n_clusters=k, connectivity=conn, linkage='ward')
        labels = model.fit_predict(features)
    return labels.astype(np.int32)


def cluster_coacd(
    mesh: trimesh.Trimesh,
    positions: np.ndarray,
    threshold: float = 0.05,
) -> Tuple[np.ndarray, List]:
    """CoACD convex decomposition 기반 클러스터링.

    메시를 convex 파트로 분해한 뒤, 각 뷰포인트 표면 위치를
    가장 가까운 파트에 할당한다.

    Args:
        mesh: 대상 메시 (원본 target_mesh)
        positions: (N, 3) 뷰포인트 표면 위치
        threshold: CoACD concavity threshold (낮을수록 더 많은 파트)

    Returns:
        cluster_ids: (N,) 0-based 클러스터 할당
        part_meshes: List[trimesh.Trimesh] convex 파트 메시 목록
    """
    import coacd

    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(coacd_mesh, threshold=threshold)
    print(f"  CoACD: {len(parts)} convex parts")

    # 각 파트에 대해 모든 뷰포인트까지 거리 계산
    part_meshes = []
    distances = np.full((len(positions), len(parts)), np.inf)
    for j, (verts, faces) in enumerate(parts):
        part_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        part_meshes.append(part_mesh)
        _, dists, _ = trimesh.proximity.closest_point(part_mesh, positions)
        distances[:, j] = dists

    cluster_ids = np.argmin(distances, axis=1).astype(np.int32)

    # 빈 클러스터 제거: 0부터 연속 ID로 재매핑
    unique_ids = np.unique(cluster_ids)
    id_map = {old: new for new, old in enumerate(unique_ids)}
    cluster_ids = np.array([id_map[c] for c in cluster_ids], dtype=np.int32)
    part_meshes = [part_meshes[old] for old in unique_ids]

    return cluster_ids, part_meshes


def cluster_coacd_dbscan(
    mesh: trimesh.Trimesh,
    positions: np.ndarray,
    normals: np.ndarray,
    camera_positions: np.ndarray,
    coacd_threshold: float = 0.05,
    eps_m: float = 0.03,
    min_samples: int = 2,
    normal_weight: float = 0.0,
    precomputed_coacd: Optional[Tuple[np.ndarray, List]] = None,
) -> Tuple[np.ndarray, List, np.ndarray]:
    """CoACD → DBSCAN 2단계 클러스터링.

    1단계: CoACD로 메시를 convex 파트로 분해하여 뷰포인트를 파트별로 할당(표면 positions).
    2단계: 각 CoACD 파트 내에서 **camera_positions** 기준 DBSCAN으로 세분화
    (렌더·로봇 EE가 카메라 위치이므로 — 곡면에서 표면은 가까워도 카메라는 벌어짐).

    Args:
        mesh: 대상 메시
        positions: (N, 3) 뷰포인트 표면 위치 (CoACD 파트 할당용)
        normals: (N, 3) 표면 법선 벡터
        camera_positions: (N, 3) 카메라 위치 (DBSCAN 클러스터링 기준)
        coacd_threshold: CoACD concavity threshold
        eps_m: DBSCAN 이웃 반경 (미터)
        min_samples: DBSCAN 코어 포인트 최소 이웃 수
        normal_weight: DBSCAN 법선 가중치
        precomputed_coacd: (coacd_ids, part_meshes) 사전 계산된 CoACD 결과 (캐싱용)

    Returns:
        cluster_ids: (N,) 0-based 최종 클러스터 할당
        part_meshes: List[trimesh.Trimesh] convex 파트 메시 목록
        coacd_ids: (N,) 0-based CoACD 파트 할당 (시각화용)
    """
    # 1단계: CoACD
    if precomputed_coacd is not None:
        coacd_ids, part_meshes = precomputed_coacd
        t_coacd = 0.0
    else:
        t0 = time.perf_counter()
        coacd_ids, part_meshes = cluster_coacd(mesh, positions, coacd_threshold)
        t_coacd = time.perf_counter() - t0
    num_coacd_parts = len(np.unique(coacd_ids))
    print(f"  CoACD+DBSCAN: {num_coacd_parts} CoACD parts → DBSCAN sub-clustering...")

    # 2단계: 각 CoACD 파트 내에서 camera_positions 기준 DBSCAN
    t0 = time.perf_counter()
    final_ids = np.full(len(positions), -1, dtype=np.int32)
    next_cluster = 0
    total_sub_clusters = 0

    for part_id in np.unique(coacd_ids):
        mask = coacd_ids == part_id
        part_cam = camera_positions[mask]
        part_normals = normals[mask]
        indices = np.where(mask)[0]

        if len(part_cam) < min_samples:
            # 포인트가 너무 적으면 하나의 클러스터로
            final_ids[indices] = next_cluster
            next_cluster += 1
            total_sub_clusters += 1
        else:
            sub_ids = cluster_dbscan(
                part_cam, part_normals,
                eps_m, min_samples, normal_weight,
            )
            n_sub = len(np.unique(sub_ids))
            total_sub_clusters += n_sub
            for sub_id in np.unique(sub_ids):
                sub_mask = sub_ids == sub_id
                final_ids[indices[sub_mask]] = next_cluster
                next_cluster += 1
    t_dbscan = time.perf_counter() - t0

    print(f"  CoACD+DBSCAN: {num_coacd_parts} parts → {next_cluster} final clusters "
          f"(coacd={t_coacd:.3f}s, dbscan={t_dbscan:.3f}s)")
    return final_ids, part_meshes, coacd_ids


def cluster_coacd_agglomerative(
    mesh: trimesh.Trimesh,
    positions: np.ndarray,
    normals: np.ndarray,
    camera_positions: np.ndarray,
    coacd_threshold: float = 0.05,
    target_size: int = 12,
    normal_weight: float = 0.0,
    max_span_mm: Optional[float] = None,
    precomputed_coacd: Optional[Tuple[np.ndarray, List]] = None,
) -> Tuple[np.ndarray, List, np.ndarray]:
    """CoACD → Agglomerative 2단계 클러스터링 (DBSCAN 대체).

    1단계: CoACD로 convex 파트 분할(표면 positions). 2단계: 각 파트 내
    **camera_positions** 기준 Agglomerative 공간 분할 (렌더·로봇 EE가 카메라 위치이므로 —
    곡면에서 표면은 가까워도 카메라는 벌어짐). max_span은 카메라 위치 지름을 제한.

    Returns: (cluster_ids, part_meshes, coacd_ids) — cluster_coacd_dbscan과 동일 시그니처.
    """
    # 1단계: CoACD (캐시 재사용 가능 — dbscan과 동일 경로)
    if precomputed_coacd is not None:
        coacd_ids, part_meshes = precomputed_coacd
        t_coacd = 0.0
    else:
        t0 = time.perf_counter()
        coacd_ids, part_meshes = cluster_coacd(mesh, positions, coacd_threshold)
        t_coacd = time.perf_counter() - t0
    num_coacd_parts = len(np.unique(coacd_ids))
    print(f"  CoACD+Agglomerative: {num_coacd_parts} CoACD parts → Ward sub-clustering...")

    # 2단계: 각 CoACD 파트 내에서 camera_positions 기준 Agglomerative
    t0 = time.perf_counter()
    final_ids = np.full(len(positions), -1, dtype=np.int32)
    next_cluster = 0

    for part_id in np.unique(coacd_ids):
        mask = coacd_ids == part_id
        indices = np.where(mask)[0]
        sub_ids = cluster_agglomerative(
            camera_positions[mask], normals[mask], target_size, normal_weight,
            max_span_mm=max_span_mm,
        )
        for sub_id in np.unique(sub_ids):
            sub_mask = sub_ids == sub_id
            final_ids[indices[sub_mask]] = next_cluster
            next_cluster += 1
    t_aggl = time.perf_counter() - t0

    print(f"  CoACD+Agglomerative: {num_coacd_parts} parts → {next_cluster} final clusters "
          f"(coacd={t_coacd:.3f}s, aggl={t_aggl:.3f}s)")
    return final_ids, part_meshes, coacd_ids


def _edge_cost(pos_from, pos_to, nrm_from, nrm_to, normal_weight):
    """두 점 사이의 위치+법선 비용 (미터 단위)."""
    d = float(np.linalg.norm(pos_from - pos_to))
    if normal_weight > 0:
        d += float(np.linalg.norm(nrm_from - nrm_to)) * normal_weight
    return d


def _gtsp_bruteforce(unique_clusters, cluster_internal, normal_weight):
    """K ≤ 2일 때 전수 탐색으로 최적 순서+방향을 반환."""
    import itertools
    K = len(unique_clusters)
    best_cost = np.inf
    best_order = None
    best_dirs = None

    for perm in itertools.permutations(range(K)):
        for dirs in itertools.product([0, 1], repeat=K):
            cost = 0.0
            for step in range(K - 1):
                k, l = perm[step], perm[step + 1]
                cid_k, cid_l = unique_clusters[k], unique_clusters[l]
                ci_k, ci_l = cluster_internal[cid_k], cluster_internal[cid_l]
                # 퇴장점: F=endpoint_b, R=endpoint_a
                exit_pos = ci_k['endpoint_a'] if dirs[step] == 1 else ci_k['endpoint_b']
                exit_nrm = ci_k['normal_a'] if dirs[step] == 1 else ci_k['normal_b']
                # 진입점: F=endpoint_a, R=endpoint_b
                entry_pos = ci_l['endpoint_b'] if dirs[step + 1] == 1 else ci_l['endpoint_a']
                entry_nrm = ci_l['normal_b'] if dirs[step + 1] == 1 else ci_l['normal_a']
                cost += _edge_cost(exit_pos, entry_pos, exit_nrm, entry_nrm, normal_weight)
            if cost < best_cost:
                best_cost = cost
                best_order = [unique_clusters[p] for p in perm]
                best_dirs = list(dirs)

    return (np.array(best_order, dtype=np.int32),
            np.array(best_dirs, dtype=np.int32))


def _gtsp_greedy_nn(unique_clusters, cluster_internal, normal_weight):
    """양방향 고려 greedy nearest-neighbor fallback."""
    K = len(unique_clusters)
    visited = set()
    order = []
    directions = []

    # 시작: 모든 클러스터×방향 중 가장 낮은 "첫 진입 비용"은 없으므로 임의로 첫 번째
    current_cid = unique_clusters[0]
    current_dir = 0
    order.append(current_cid)
    directions.append(current_dir)
    visited.add(current_cid)

    for _ in range(K - 1):
        ci_cur = cluster_internal[current_cid]
        cur_exit = ci_cur['endpoint_a'] if current_dir == 1 else ci_cur['endpoint_b']
        cur_exit_n = ci_cur['normal_a'] if current_dir == 1 else ci_cur['normal_b']

        best_cost = np.inf
        best_cid = None
        best_dir = 0
        for cid in unique_clusters:
            if cid in visited:
                continue
            ci = cluster_internal[cid]
            for d in [0, 1]:
                entry = ci['endpoint_b'] if d == 1 else ci['endpoint_a']
                entry_n = ci['normal_b'] if d == 1 else ci['normal_a']
                c = _edge_cost(cur_exit, entry, cur_exit_n, entry_n, normal_weight)
                if c < best_cost:
                    best_cost = c
                    best_cid = cid
                    best_dir = d

        order.append(best_cid)
        directions.append(best_dir)
        visited.add(best_cid)
        current_cid = best_cid
        current_dir = best_dir

    return (np.array(order, dtype=np.int32),
            np.array(directions, dtype=np.int32))


def order_clusters_gtsp(
    cluster_ids: np.ndarray,
    camera_positions: np.ndarray,
    cluster_internal: dict,
    normal_weight: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """GTSP (Noon-Bean 변환 + 더미 노드)로 클러스터 방문 순서 및 방향 최적화.

    각 클러스터에 대해 정방향(F: a→b)과 역방향(R: b→a) 두 선택지를 두고,
    Noon-Bean 변환으로 GTSP를 ATSP로 변환하여 OR-Tools로 풀어
    최적 순서와 방향을 동시에 결정한다.
    더미 노드를 추가하여 open path(비순환 경로)를 생성한다.

    Args:
        cluster_ids: (N,) 0-based 클러스터 할당
        camera_positions: (N, 3)
        cluster_internal: compute_cluster_internal_order()의 결과
        normal_weight: 법선 가중치 (0이면 위치만 사용)

    Returns:
        cluster_order: (K,) 클러스터 방문 순서 (클러스터 ID 배열)
        cluster_direction: (K,) 각 클러스터의 방향 (0=Forward a→b, 1=Reverse b→a)
    """
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp

    unique_clusters = np.unique(cluster_ids)
    K = len(unique_clusters)

    if K <= 2:
        return _gtsp_bruteforce(unique_clusters, cluster_internal, normal_weight)

    # --- 엔드포인트 배열 준비 ---
    ep_a = np.array([cluster_internal[c]['endpoint_a'] for c in unique_clusters])
    ep_b = np.array([cluster_internal[c]['endpoint_b'] for c in unique_clusters])
    nr_a = np.array([cluster_internal[c]['normal_a'] for c in unique_clusters])
    nr_b = np.array([cluster_internal[c]['normal_b'] for c in unique_clusters])

    # --- Noon-Bean ATSP 거리행렬 구성 (2K+1 노드) ---
    # 노드 인덱싱: 2*k = F_k, 2*k+1 = R_k, 2*K = Dummy
    SCALE = 1_000_000
    INF = 10**15
    N_ATSP = 2 * K + 1
    D = 2 * K  # dummy node index

    atsp = np.full((N_ATSP, N_ATSP), INF, dtype=np.int64)

    for k in range(K):
        # 클러스터 내 사이클 (비용 0)
        atsp[2 * k, 2 * k + 1] = 0      # F_k → R_k
        atsp[2 * k + 1, 2 * k] = 0      # R_k → F_k

        for l in range(K):
            if k == l:
                continue

            # 원본 GTSP 비용 (퇴장점 → 진입점)
            # F_k→F_l: exit_b[k] → entry_a[l]
            ff = _edge_cost(ep_b[k], ep_a[l], nr_b[k], nr_a[l], normal_weight)
            # F_k→R_l: exit_b[k] → entry_b[l]
            fr = _edge_cost(ep_b[k], ep_b[l], nr_b[k], nr_b[l], normal_weight)
            # R_k→F_l: exit_a[k] → entry_a[l]
            rf = _edge_cost(ep_a[k], ep_a[l], nr_a[k], nr_a[l], normal_weight)
            # R_k→R_l: exit_a[k] → entry_b[l]
            rr = _edge_cost(ep_a[k], ep_b[l], nr_a[k], nr_b[l], normal_weight)

            # Noon-Bean: GTSP[X_k, Y_l] → ATSP[pred(X_k), Y_l]
            # pred(F_k) = R_k, pred(R_k) = F_k
            atsp[2 * k + 1, 2 * l]     = int(ff * SCALE)  # GTSP F_k→F_l → ATSP R_k→F_l
            atsp[2 * k + 1, 2 * l + 1] = int(fr * SCALE)  # GTSP F_k→R_l → ATSP R_k→R_l
            atsp[2 * k,     2 * l]     = int(rf * SCALE)   # GTSP R_k→F_l → ATSP F_k→F_l
            atsp[2 * k,     2 * l + 1] = int(rr * SCALE)   # GTSP R_k→R_l → ATSP F_k→R_l

    # 더미 노드: open path를 위해 비용 0
    atsp[D, :] = 0
    atsp[:, D] = 0
    atsp[D, D] = INF

    # --- OR-Tools ATSP ---
    manager = pywrapcp.RoutingIndexManager(N_ATSP, 1, D)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return atsp[from_node, to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = 2

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        # D를 제외한 투어 추출
        tour = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != D:
                tour.append(node)
            index = solution.Value(routing.NextVar(index))

        # 디코딩: 2개씩 쌍, 첫 번째가 진입 노드
        cluster_order = []
        cluster_direction = []
        for i in range(0, len(tour), 2):
            entry_node = tour[i]
            k = entry_node // 2
            direction = entry_node % 2  # 0=F, 1=R
            cluster_order.append(unique_clusters[k])
            cluster_direction.append(direction)

        return (np.array(cluster_order, dtype=np.int32),
                np.array(cluster_direction, dtype=np.int32))

    # Fallback: 양방향 greedy NN
    print("  Warning: GTSP solver failed, falling back to greedy NN")
    return _gtsp_greedy_nn(unique_clusters, cluster_internal, normal_weight)


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


def _two_opt_open(order: list, points: np.ndarray, max_passes: int) -> list:
    """Open-path 2-opt 개선(full). 양 끝점(order[0], order[-1])은 고정 → 열린 경로 유지.

    클러스터가 작으므로(호출측에서 n<=max_2opt_n 보장) O(n²) full 2-opt로 충분.
    """
    n = len(order)
    if n < 4:
        return order
    pos = points
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n - 1):
                a, b = order[i - 1], order[i]
                c, d = order[j], order[j + 1]
                before = np.linalg.norm(pos[a] - pos[b]) + np.linalg.norm(pos[c] - pos[d])
                after = np.linalg.norm(pos[a] - pos[c]) + np.linalg.norm(pos[b] - pos[d])
                if after + 1e-12 < before:
                    order[i:j + 1] = order[i:j + 1][::-1]
                    improved = True
    return order


def order_cluster_graph(
    camera_positions_sub: np.ndarray,
    normals_sub: np.ndarray,
    max_2opt_n: int = 120,
    max_2opt_passes: int = 30,
) -> list:
    """한 클러스터의 nearest-neighbor open-path 순서(로컬 인덱스 permutation).

    전역 PCA 평면 대신 클러스터 평균 법선의 탄젠트 평면을 사용해 시작 극단점과
    tangent-정렬 baseline을 잡고, 카메라 위치 기준 nearest-neighbor + 2-opt로
    경로를 만든다. 절대 tangent-정렬 baseline보다 길어지지 않도록 가드한다.

    Returns: list[int] — 방문 순서(permutation). sorted_indices = [idx[i] for i in perm].
    """
    P = np.asarray(camera_positions_sub, dtype=np.float64)
    n = len(P)
    if n <= 2:
        return list(range(n))
    if not np.all(np.isfinite(P)):
        return list(range(n))

    # 1. 평균 법선 탄젠트 프레임 → 2D 투영
    mean_n = np.asarray(normals_sub, dtype=np.float64).mean(axis=0)
    u, v = _tangent_basis(mean_n)
    Pc = P - P.mean(axis=0)
    P2 = np.column_stack([Pc @ u, Pc @ v])

    # 최장 탄젠트 extent 축 → 시작점(극단) + tangent-정렬 baseline
    spread = P2.max(axis=0) - P2.min(axis=0)
    major = 0 if spread[0] >= spread[1] else 1
    tangent_sorted = list(np.argsort(P2[:, major]))

    # 2. nearest-neighbor seed (극단점에서 시작) + 2-opt
    start = int(np.argmin(P2[:, major]))
    visited = np.zeros(n, dtype=bool)
    order = [start]
    visited[start] = True
    cur = start
    for _ in range(n - 1):
        d = np.linalg.norm(P - P[cur], axis=1)
        d[visited] = np.inf
        nxt = int(np.argmin(d))
        order.append(nxt)
        visited[nxt] = True
        cur = nxt

    if n <= max_2opt_n:
        order = _two_opt_open(order, P, max_2opt_passes)

    # 3. anti-explosion 가드: tangent-정렬 baseline보다 길면 폴백
    def _plen(seq: list) -> float:
        return float(np.sum(np.linalg.norm(np.diff(P[seq], axis=0), axis=1))) if len(seq) > 1 else 0.0

    if _plen(order) > _plen(tangent_sorted):
        return tangent_sorted
    return order


def order_cluster_lawnmower(
    surface_positions_sub: np.ndarray,
    camera_positions_sub: np.ndarray,
    normals_sub: np.ndarray,
    row_spacing_m: float,
) -> list:
    """한 클러스터를 tangent-plane lawnmower 패턴으로 정렬한다.

    row/scan 축은 표면점 기준으로 잡고, 두 가능한 시작 방향 중 실제 카메라
    위치 경로가 짧은 쪽을 선택한다. 따라서 coverage row는 surface 기준,
    이동 비용은 working-distance가 반영된 camera 기준이 된다.
    """
    S = np.asarray(surface_positions_sub, dtype=np.float64)
    C = np.asarray(camera_positions_sub, dtype=np.float64)
    n = len(S)
    if n <= 2:
        return list(range(n))
    if not np.all(np.isfinite(S)) or not np.all(np.isfinite(C)):
        return order_cluster_graph(camera_positions_sub, normals_sub)

    spacing = max(float(row_spacing_m), 1e-6)

    # 1. 평균 법선 tangent frame으로 표면점을 2D 투영
    mean_n = np.asarray(normals_sub, dtype=np.float64).mean(axis=0)
    u, v = _tangent_basis(mean_n)
    Sc = S - S.mean(axis=0)
    P2 = np.column_stack([Sc @ u, Sc @ v])
    if not np.all(np.isfinite(P2)):
        return order_cluster_graph(camera_positions_sub, normals_sub)

    # 2. tangent 2D 안에서 PCA: 긴 축=scan, 짧은 축=row
    P2c = P2 - P2.mean(axis=0)
    try:
        cov = np.cov(P2c, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        scan_axis = vecs[:, order[0]]
        row_axis = vecs[:, order[1]]
    except Exception:  # noqa: BLE001
        spread = P2.max(axis=0) - P2.min(axis=0)
        scan_axis = np.array([1.0, 0.0]) if spread[0] >= spread[1] else np.array([0.0, 1.0])
        row_axis = np.array([-scan_axis[1], scan_axis[0]])

    scan = P2c @ scan_axis
    row = P2c @ row_axis

    # 3. FOV-derived spacing으로 row binning
    row_span = float(row.max() - row.min())
    if row_span < spacing * 0.5:
        row_bins = np.zeros(n, dtype=np.int32)
    else:
        row_bins = np.floor((row - row.min()) / spacing + 0.5).astype(np.int32)

    rows = []
    for rb in np.unique(row_bins):
        idx = np.where(row_bins == rb)[0]
        if idx.size == 0:
            continue
        rows.append((float(row[idx].mean()), idx))
    rows.sort(key=lambda item: item[0])

    if not rows:
        return order_cluster_graph(camera_positions_sub, normals_sub)

    def _make(reverse_first: bool) -> list:
        out = []
        for r, (_, idx) in enumerate(rows):
            local = idx[np.argsort(scan[idx], kind="stable")]
            if (r % 2 == 1) ^ reverse_first:
                local = local[::-1]
            out.extend(int(i) for i in local)
        return out

    def _plen(seq: list) -> float:
        return float(np.sum(np.linalg.norm(np.diff(C[seq], axis=0), axis=1))) if len(seq) > 1 else 0.0

    forward = _make(False)
    reverse = _make(True)
    return reverse if _plen(reverse) < _plen(forward) else forward


def compute_cluster_internal_order(
    cluster_ids: np.ndarray,
    surface_positions: np.ndarray,
    camera_positions: np.ndarray,
    normals: np.ndarray,
    row_spacing_m: float,
    col_spacing_m: Optional[float] = None,
    grid_row_index: Optional[np.ndarray] = None,
    global_axis1: Optional[np.ndarray] = None,
    global_axis2: Optional[np.ndarray] = None,
    ordering_mode: str = 'zigzag',
) -> dict:
    """각 클러스터의 내부 방문 순서를 사전 계산.

    TSP 전에 호출하여 클러스터별 start/end point를 확보한다.

    grid_row_index가 주어지면 전역 축(global_axis1/axis2)을 사용하여
    클러스터별 PCA를 생략한다. row_index_override가 행 구분을 담당하므로
    로컬 PCA의 axis1은 불필요하고, axis2는 클러스터 간 열 방향 일관성을
    위해 이미 전역 값을 사용해야 하기 때문이다.

    Args:
        cluster_ids: (N,) 클러스터 ID
        surface_positions: (N, 3) 표면 위치
        camera_positions: (N, 3) 카메라 위치
        normals: (N, 3) 법선
        row_spacing_m: 행 간격 (미터)
        col_spacing_m: 열 간격 (미터). lawnmower row 간격은 min(row, col)을 사용.
        grid_row_index: (N,) 그리드 생성 시 할당된 원본 행 인덱스. None이면 양자화로 추정.
        global_axis1: (3,) 전역 행 방향 축. grid_row_index 사용 시 전달 (proj1 계산용, 행 구분에는 미사용).
        global_axis2: (3,) 전역 열 방향 축. grid_row_index 사용 시 행 내 정렬에 사용.

    Returns:
        dict[cluster_id] → {
            'sorted_indices': list[int],  # 내부 순서대로 정렬된 원본 인덱스
            'endpoint_a': (3,),           # 카메라 끝점 A
            'endpoint_b': (3,),           # 카메라 끝점 B
            'normal_a': (3,),             # 끝점 A 법선
            'normal_b': (3,),             # 끝점 B 법선
        }
    """
    unique_clusters = np.unique(cluster_ids)
    result = {}

    for cid in unique_clusters:
        mask = cluster_ids == cid
        indices = np.where(mask)[0]

        if len(indices) < 3:
            sorted_indices = list(indices)
        elif ordering_mode == 'lawnmower':
            spacing = min(row_spacing_m, col_spacing_m) if col_spacing_m is not None else row_spacing_m
            perm = order_cluster_lawnmower(
                surface_positions[indices], camera_positions[indices], normals[indices], spacing,
            )
            sorted_indices = [indices[i] for i in perm]
        elif ordering_mode == 'graph':
            # 평균 법선 tangent 기준 시작점 + camera-space NN + open-path 2-opt.
            # order_cluster_graph는 permutation을 직접 반환(argsort 불필요).
            perm = order_cluster_graph(camera_positions[indices], normals[indices])
            sorted_indices = [indices[i] for i in perm]
        else:
            cluster_cam = camera_positions[indices]

            if grid_row_index is not None:
                cluster_row_idx = grid_row_index[indices]
                axis1 = global_axis1 / np.linalg.norm(global_axis1)
                axis2 = global_axis2 / np.linalg.norm(global_axis2)
                local_order, _, _ = reorder_zigzag(
                    cluster_cam, axis1, axis2, row_spacing_m,
                    row_index_override=cluster_row_idx,
                )
            else:
                _, axis1, axis2 = compute_pca_axes(cluster_cam.astype(np.float64))
                local_order, _, _ = reorder_zigzag(cluster_cam, axis1, axis2, row_spacing_m)

            sorted_local = np.argsort(local_order)
            sorted_indices = [indices[i] for i in sorted_local]

        result[cid] = {
            'sorted_indices': sorted_indices,
            'endpoint_a': camera_positions[sorted_indices[0]],
            'endpoint_b': camera_positions[sorted_indices[-1]],
            'normal_a': normals[sorted_indices[0]],
            'normal_b': normals[sorted_indices[-1]],
        }

    return result


def build_clustered_path_order(
    cluster_ids: np.ndarray,
    cluster_order: np.ndarray,
    cluster_internal: dict,
    cluster_direction: Optional[np.ndarray] = None,
) -> np.ndarray:
    """클러스터 순서와 사전 계산된 내부 순서를 결합하여 글로벌 path_order 생성.

    Args:
        cluster_ids: (N,) 클러스터 할당
        cluster_order: (K,) 클러스터 방문 순서
        cluster_internal: compute_cluster_internal_order()의 결과
        cluster_direction: (K,) 각 클러스터의 방향 (0=Forward, 1=Reverse). None이면 전부 Forward.

    Returns:
        path_order: (N,) 글로벌 경로 순서
    """
    N = len(cluster_ids)
    path_order = np.zeros(N, dtype=np.int32)
    global_idx = 0

    for rank, cid in enumerate(cluster_order):
        indices = cluster_internal[cid]['sorted_indices']
        if cluster_direction is not None and cluster_direction[rank] == 1:
            indices = list(reversed(indices))
        for idx in indices:
            path_order[idx] = global_idx
            global_idx += 1

    return path_order


# ============================================================================
# HDF5 I/O
# ============================================================================

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
) -> Path:
    """Save viewpoints to HDF5 file

    Args:
        pca_data: dict with 'center' (3,), 'axis1' (3,), 'axis2' (3,) arrays
        row_index: (N,) int32 array — row index per viewpoint
        cluster_id: (N,) int32 array — cluster assignment per viewpoint
        cluster_order: (K,) int32 array — cluster visit order
        cluster_direction: (K,) int32 array — 0=Forward, 1=Reverse per cluster
        cluster_metadata: dict with clustering parameters
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

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='뷰포인트 생성 및 클러스터링 기반 경로 순서 최적화',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 기본: FOV 기반 자동 간격
  uv run scripts/core/generate_viewpoints.py --object sample

  # 재질 필터
  uv run scripts/core/generate_viewpoints.py --object sample --material-rgb "0,255,0"

  # DBSCAN 클러스터링
  uv run scripts/core/generate_viewpoints.py --object sample --material-rgb "0,255,0" --cluster-method dbscan

  # CoACD 클러스터링 (메시 convex decomposition)
  uv run scripts/core/generate_viewpoints.py --object sample --cluster-method coacd --coacd-threshold 0.05
        """,
    )

    # --- Viewpoint generation ---
    parser.add_argument('--object', type=str, required=True, help='오브젝트 이름')
    parser.add_argument('--material-rgb', type=str, default=None,
                        help='Target material RGB color as "R,G,B" (e.g., "0,255,0")')
    parser.add_argument('--color-tolerance', type=float, default=5.0,
                        help='RGB color matching tolerance (default: 5.0)')
    parser.add_argument('--row-spacing', type=float, default=None,
                        help='Row spacing in mm (default: CAMERA_FOV_HEIGHT_MM * (1-overlap))')
    parser.add_argument('--col-spacing', type=float, default=None,
                        help='Column spacing in mm (default: CAMERA_FOV_WIDTH_MM * (1-overlap))')
    parser.add_argument('--no-filter-bottom', action='store_true', default=False,
                        help='Disable bottom-face filtering')
    parser.add_argument('--bottom-angle', type=float, default=80.0,
                        help='Bottom filter angle in degrees (default: 80)')

    # --- Clustering ---
    parser.add_argument('--cluster-method', type=str, default='dbscan',
                        choices=['dbscan', 'coacd', 'coacd+dbscan',
                                 'agglomerative', 'coacd+agglomerative'],
                        help='클러스터링 방법 (기본: dbscan). agglomerative=Ward 공간분할(균등·싱글톤 없음)')
    parser.add_argument('--eps', type=float, default=None,
                        help=f'[dbscan] 이웃 반경 mm (기본: {config.CAMERA_FOV_WIDTH_MM:.0f}mm)')
    parser.add_argument('--min-samples', type=int, default=2,
                        help='[dbscan] 코어 포인트 최소 이웃 수 (기본: 2)')
    parser.add_argument('--target-size', type=int, default=12,
                        help='[agglomerative ward] 클러스터당 목표 점 개수 (기본: 12)')
    parser.add_argument('--max-span', type=float, default=None,
                        help='[agglomerative] 클러스터 최대 지름 mm. 지정 시 complete-linkage로 '
                             '지름 ≤ 값 보장(멀리 떨어진 점 묶임 방지)')
    parser.add_argument('--normal-weight', type=float, default=0.0,
                        help='[dbscan/agglomerative] 법선 가중치 (미터 단위, 0이면 위치만 사용, 기본: 0.0)')
    parser.add_argument('--coacd-threshold', type=float, default=0.05,
                        help='[coacd] concavity threshold (낮을수록 더 많은 파트, 기본: 0.05)')

    # --- Sampling / Ordering ---
    parser.add_argument('--sampling-mode', type=str, default='grid',
                        choices=['grid', 'surface'],
                        help='뷰포인트 배치: grid(PCA 평면 투영) | surface(표면 FPS, 곡면 커버리지). 기본: grid')
    parser.add_argument('--surface-spacing', type=float, default=None,
                        help='[surface] FPS 목표 표면 간격 mm (기본: FOV 작은 축)')
    parser.add_argument('--ordering-mode', type=str, default='zigzag',
                        choices=['zigzag', 'graph', 'lawnmower'],
                        help='클러스터 내부 순서: zigzag(전역 PCA) | graph(NN+2opt) | '
                             'lawnmower(탄젠트 row sweep). 기본: zigzag')

    # --- Comparison ---
    parser.add_argument('--compare', action='store_true',
                        help='선택된 방법의 파라미터 변형 비교 HTML 생성')

    # --- Visualization ---
    # --- Debug ---
    parser.add_argument('--dry-run', action='store_true', help='통계만 출력, HDF5 저장 안 함')

    args = parser.parse_args()

    # Validate RGB format
    if args.material_rgb is not None:
        try:
            rgb_parts = args.material_rgb.split(',')
            if len(rgb_parts) != 3:
                raise ValueError("RGB must have 3 components")
            r, g, b = map(int, rgb_parts)
            if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                raise ValueError("RGB values must be in range [0, 255]")
        except ValueError as e:
            parser.error(f"Invalid RGB format: {e}")

    return args


# ============================================================================
# Generation core (importable — no file IO / CLI)
# ============================================================================

@dataclass
class ViewpointGenParams:
    """generate_viewpoints_core 입력 파라미터 (CLI 플래그 미러)."""
    material_rgb: Optional[str] = None        # "R,G,B" 문자열 (load_meshes에서 파싱)
    color_tolerance: float = 5.0
    row_spacing_mm: Optional[float] = None
    col_spacing_mm: Optional[float] = None
    filter_bottom: bool = True
    bottom_angle: float = 80.0
    cluster_method: str = 'dbscan'
    eps_mm: Optional[float] = None
    min_samples: int = 2
    normal_weight: float = 0.0
    coacd_threshold: float = 0.05
    target_size: int = 12                     # [agglomerative ward] 클러스터당 목표 점 개수
    max_span_mm: Optional[float] = None       # [agglomerative] 클러스터 최대 지름 mm (지정 시 complete-linkage)
    sampling_mode: str = 'grid'               # 'grid'(PCA 그리드 투영) | 'surface'(FPS)
    surface_spacing_mm: Optional[float] = None  # surface 모드 FPS 목표 간격 (None이면 FOV 작은 축)
    ordering_mode: str = 'zigzag'             # 'zigzag' | 'graph' | 'lawnmower'


@dataclass
class ViewpointResult:
    """generate_viewpoints_core 결과 (in-memory; 저장은 호출측 책임)."""
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
    pca: dict                       # {'center', 'axis1', 'axis2'}
    row_spacing_m: float
    col_spacing_m: float
    original_path_length_mm: float
    clustered_path_length_mm: float
    num_clusters: int
    cluster_meta: dict
    method: str
    label: str


def load_meshes(object_name, material_rgb=None, color_tolerance=5.0):
    """소스 메시 로드 + (선택) 재질 RGB 필터로 타깃 메시 추출.

    Returns: (full_mesh, target_mesh, input_path)
    """
    input_path = str(config.get_mesh_path(object_name, mesh_type="source"))
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input mesh not found: {input_path}")

    print("Loading mesh...")
    loaded = trimesh.load(input_path)
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
    else:
        mesh = loaded
    print(f"  Loaded: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} triangles")
    print()

    if material_rgb:
        print("Parsing materials...")
        triangle_materials, mtl_file = parse_obj_material_usage(input_path)
        if mtl_file is None or not os.path.exists(mtl_file):
            raise FileNotFoundError("MTL file not found")

        materials = parse_mtl_file(mtl_file)
        print(f"  Found {len(materials)} materials:")
        for mat_name, mat_props in materials.items():
            if 'Kd' in mat_props:
                rgb = kd_to_rgb(mat_props['Kd'])
                print(f"    - {mat_name}: RGB{rgb}")
        print()

        print("Matching material...")
        target_rgb = tuple(map(int, material_rgb.split(',')))
        matched_materials = match_material_by_color(materials, target_rgb, color_tolerance)
        if len(matched_materials) == 0:
            raise ValueError(
                f"No materials matched RGB{target_rgb} within tolerance {color_tolerance}")
        print(f"  Matched: {matched_materials}")
        print()

        print("Extracting target mesh...")
        target_mesh = extract_target_mesh(mesh, triangle_materials, matched_materials)
        target_percentage = (len(target_mesh.faces) / len(mesh.faces)) * 100
        print(f"  Target: {len(target_mesh.faces):,} / {len(mesh.faces):,} triangles ({target_percentage:.1f}%)")
    else:
        print("Using entire mesh (no material filter)...")
        target_mesh = mesh
        print(f"  Triangles: {len(target_mesh.faces):,}")

    print(f"  Surface area: {target_mesh.area:.6f} m2")
    print()
    return mesh, target_mesh, input_path


def cluster_and_order(label, method, *, positions, normals, camera_positions, target_mesh,
                      row_spacing_m, col_spacing_m, grid_row_index, cam_axis1, cam_axis2,
                      original_path_length_mm, **kwargs):
    """한 가지 클러스터링 설정을 실행하고 결과 dict를 반환."""
    t_total_start = time.perf_counter()

    coacd_parts = None
    t0 = time.perf_counter()
    if method == 'dbscan':
        cids = cluster_dbscan(
            camera_positions, normals, kwargs['eps_m'],
            kwargs.get('min_samples', 2), kwargs.get('normal_weight', 0.0),
        )
    elif method == 'coacd':
        if kwargs.get('precomputed_coacd') is not None:
            cids, coacd_parts = kwargs['precomputed_coacd']
        else:
            cids, coacd_parts = cluster_coacd(target_mesh, positions, kwargs['threshold'])
    elif method == 'coacd+dbscan':
        cids, coacd_parts, coacd_ids = cluster_coacd_dbscan(
            target_mesh, positions, normals, camera_positions,
            coacd_threshold=kwargs['threshold'],
            eps_m=kwargs.get('eps_m', 0.03),
            min_samples=kwargs.get('min_samples', 2),
            normal_weight=kwargs.get('normal_weight', 0.0),
            precomputed_coacd=kwargs.get('precomputed_coacd'),
        )
    elif method == 'agglomerative':
        cids = cluster_agglomerative(
            camera_positions, normals,
            kwargs.get('target_size', 12), kwargs.get('normal_weight', 0.0),
            max_span_mm=kwargs.get('max_span_mm'),
        )
    elif method == 'coacd+agglomerative':
        cids, coacd_parts, coacd_ids = cluster_coacd_agglomerative(
            target_mesh, positions, normals, camera_positions,
            coacd_threshold=kwargs['threshold'],
            target_size=kwargs.get('target_size', 12),
            normal_weight=kwargs.get('normal_weight', 0.0),
            max_span_mm=kwargs.get('max_span_mm'),
            precomputed_coacd=kwargs.get('precomputed_coacd'),
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    t_cluster = time.perf_counter() - t0

    K = len(np.unique(cids))
    sizes = np.array([np.sum(cids == c) for c in np.unique(cids)])
    nw = kwargs.get('normal_weight', 0.0)

    t0 = time.perf_counter()
    c_internal = compute_cluster_internal_order(
        cids, positions, camera_positions, normals, row_spacing_m,
        col_spacing_m=col_spacing_m,
        grid_row_index=grid_row_index, global_axis1=cam_axis1, global_axis2=cam_axis2,
        ordering_mode=kwargs.get('ordering_mode', 'zigzag'),
    )
    c_order, c_direction = order_clusters_gtsp(cids, camera_positions, c_internal, nw)
    t_gtsp = time.perf_counter() - t0

    p_order = build_clustered_path_order(cids, c_order, c_internal, c_direction)
    pl_mm = compute_path_length(camera_positions, p_order) * 1000.0

    t_total = time.perf_counter() - t_total_start

    n_fwd = int(np.sum(c_direction == 0))
    n_rev = int(np.sum(c_direction == 1))
    print(f"  [{label}] {K} clusters "
          f"(sizes: {sizes.min()}-{sizes.max()}, mean={sizes.mean():.1f}) "
          f"path: {pl_mm:.1f} mm "
          f"({(1 - pl_mm / original_path_length_mm) * 100:.1f}% reduction) "
          f"[dir: {n_fwd}F/{n_rev}R]")
    print(f"    timing: cluster={t_cluster:.3f}s, GTSP={t_gtsp:.3f}s, total={t_total:.3f}s")

    result = {
        'cluster_ids': cids, 'cluster_order': c_order,
        'cluster_direction': c_direction,
        'path_order': p_order, 'path_length_mm': pl_mm,
        'num_clusters': K,
    }
    if coacd_parts is not None:
        result['coacd_parts'] = coacd_parts
    if method in ('coacd+dbscan', 'coacd+agglomerative'):
        result['coacd_ids'] = coacd_ids
    return result


def prepare_grid(target_mesh, params: ViewpointGenParams):
    """그리드 뷰포인트 생성 + 지그재그 정렬 + bottom 필터 (클러스터링 전 단계).

    Returns: dict — positions, normals, camera_positions, path_order, row_index,
        grid_row_index, cam_axis1, cam_axis2, row_spacing_m, col_spacing_m,
        original_path_length_mm, pca_center, pca_axis1, pca_axis2
    """
    # 3. spacing
    if params.row_spacing_mm:
        row_spacing_m = params.row_spacing_mm / 1000.0
    else:
        row_spacing_m = config.CAMERA_FOV_HEIGHT_MM / 1000.0 * (1.0 - config.CAMERA_OVERLAP_RATIO)
    if params.col_spacing_mm:
        col_spacing_m = params.col_spacing_mm / 1000.0
    else:
        col_spacing_m = config.CAMERA_FOV_WIDTH_MM / 1000.0 * (1.0 - config.CAMERA_OVERLAP_RATIO)

    print(f"  Row spacing (axis1): {row_spacing_m * 1000:.1f} mm")
    print(f"  Col spacing (axis2): {col_spacing_m * 1000:.1f} mm")
    print()

    wd_m = config.CAMERA_WORKING_DISTANCE_MM / 1000.0

    if params.sampling_mode == 'surface':
        # 4'. 표면 직접 균일 샘플링 (FPS) — 곡면 커버리지
        spacing_m = (params.surface_spacing_mm / 1000.0) if params.surface_spacing_mm \
            else min(row_spacing_m, col_spacing_m)
        positions, normals, path_order, row_index, pca_center, pca_axis1, pca_axis2 = \
            generate_surface_viewpoints(target_mesh, spacing_m)
        camera_positions = positions + normals * wd_m
        # 전역 zigzag 생략(cluster ordering이 담당). axis/row는 placeholder.
        grid_row_index = row_index.copy()
        cam_axis1, cam_axis2 = pca_axis1, pca_axis2
    else:
        # 4. grid viewpoints (PCA)
        print(f"Generating grid viewpoints (PCA)...")
        positions, normals, path_order, row_index, pca_center, pca_axis1, pca_axis2 = generate_grid_viewpoints(
            target_mesh, row_spacing_m, col_spacing_m,
        )
        print(f"  Generated: {len(positions)} viewpoints")

        # Camera positions
        camera_positions = positions + normals * wd_m

        # 5. PCA 축 기준 지그재그 정렬
        grid_row_index = row_index.copy()  # 그리드 생성 시 확정된 원본 행 인덱스 보존
        _, cam_axis1, cam_axis2 = compute_pca_axes(camera_positions.astype(np.float64))
        path_order, row_index, n_rows = reorder_zigzag(camera_positions, cam_axis1, cam_axis2, row_spacing_m)
        print(f"  Ordered with PCA-axis on camera positions: {n_rows} rows")

    # 7. Filter bottom-facing viewpoints
    if params.filter_bottom:
        R_obj = quaternion_to_rotation_matrix(config.TARGET_OBJECT["rotation"])
        world_normals = (R_obj @ normals.T).T
        cos_thresh = np.cos(np.deg2rad(params.bottom_angle))
        keep = (-world_normals[:, 2]) < cos_thresh

        n_removed = (~keep).sum()
        if n_removed > 0:
            positions = positions[keep]
            normals = normals[keep]
            camera_positions = camera_positions[keep]
            row_index = row_index[keep]
            grid_row_index = grid_row_index[keep]
            old_order = path_order[keep]
            path_order = np.argsort(np.argsort(old_order)).astype(np.int32)
            print(f"  Filtered {n_removed} bottom-facing viewpoints (within {params.bottom_angle}° from down)")
            print(f"  Remaining: {len(positions)} viewpoints")
        else:
            print(f"  No bottom-facing viewpoints to filter")

    # Path length (before clustering) — surface는 PCA 무관 NN baseline
    if params.sampling_mode == 'surface':
        original_path_length_mm = _nn_path_length(camera_positions) * 1000.0
    else:
        original_path_length_mm = compute_path_length(camera_positions, path_order) * 1000.0
    print(f"  Total path length: {original_path_length_mm:.1f} mm")
    print()

    return {
        'positions': positions, 'normals': normals, 'camera_positions': camera_positions,
        'path_order': path_order, 'row_index': row_index, 'grid_row_index': grid_row_index,
        'cam_axis1': cam_axis1, 'cam_axis2': cam_axis2,
        'row_spacing_m': row_spacing_m, 'col_spacing_m': col_spacing_m,
        'original_path_length_mm': original_path_length_mm,
        'pca_center': pca_center, 'pca_axis1': pca_axis1, 'pca_axis2': pca_axis2,
    }


def generate_viewpoints_core(target_mesh, params: ViewpointGenParams) -> ViewpointResult:
    """그리드 생성 → 클러스터링 → 경로 순서 최적화 (단일 method). 파일 IO 없음.

    Phase B(viser 실시간 재생성)가 호출할 import 시임.
    """
    grid = prepare_grid(target_mesh, params)
    common = dict(
        positions=grid['positions'], normals=grid['normals'],
        camera_positions=grid['camera_positions'], target_mesh=target_mesh,
        row_spacing_m=grid['row_spacing_m'], col_spacing_m=grid['col_spacing_m'],
        grid_row_index=grid['grid_row_index'],
        cam_axis1=grid['cam_axis1'], cam_axis2=grid['cam_axis2'],
        original_path_length_mm=grid['original_path_length_mm'],
        ordering_mode=params.ordering_mode,
    )

    method = params.cluster_method
    eps_mm = params.eps_mm if params.eps_mm else config.CAMERA_FOV_WIDTH_MM

    if method == 'dbscan':
        nw_str = f", normal_weight: {params.normal_weight}" if params.normal_weight > 0 else ""
        print(f"Clustering (DBSCAN, eps: {eps_mm:.0f} mm, min_samples: {params.min_samples}{nw_str})...")
        label = method
        result = cluster_and_order(
            label, 'dbscan', **common, eps_m=eps_mm / 1000.0,
            min_samples=params.min_samples, normal_weight=params.normal_weight,
        )
    elif method == 'coacd':
        print(f"Clustering (CoACD, threshold: {params.coacd_threshold})...")
        label = method
        result = cluster_and_order(
            label, 'coacd', **common, threshold=params.coacd_threshold,
            normal_weight=params.normal_weight,
        )
    elif method == 'coacd+dbscan':
        nw_str = f", normal_weight: {params.normal_weight}" if params.normal_weight > 0 else ""
        print(f"Clustering (CoACD+DBSCAN, coacd_threshold: {params.coacd_threshold}, "
              f"eps: {eps_mm:.0f} mm, min_samples: {params.min_samples}{nw_str})...")
        label = f"coacd+dbscan t={params.coacd_threshold} eps={eps_mm:.0f}mm"
        result = cluster_and_order(
            label, 'coacd+dbscan', **common, threshold=params.coacd_threshold,
            eps_m=eps_mm / 1000.0, min_samples=params.min_samples,
            normal_weight=params.normal_weight,
        )
    elif method == 'agglomerative':
        knob = (f"max_span: {params.max_span_mm:.0f}mm" if params.max_span_mm
                else f"target_size: {params.target_size}")
        print(f"Clustering (Agglomerative, {knob})...")
        label = method
        result = cluster_and_order(
            label, 'agglomerative', **common,
            target_size=params.target_size, normal_weight=params.normal_weight,
            max_span_mm=params.max_span_mm,
        )
    elif method == 'coacd+agglomerative':
        knob = (f"max_span={params.max_span_mm:.0f}mm" if params.max_span_mm
                else f"ts={params.target_size}")
        print(f"Clustering (CoACD+Agglomerative, coacd_threshold: {params.coacd_threshold}, {knob})...")
        label = f"coacd+agglomerative t={params.coacd_threshold} {knob}"
        result = cluster_and_order(
            label, 'coacd+agglomerative', **common, threshold=params.coacd_threshold,
            target_size=params.target_size, normal_weight=params.normal_weight,
            max_span_mm=params.max_span_mm,
        )
    else:
        raise ValueError(f"Unknown cluster_method: {method}")

    cluster_meta = {
        'clustering_method': method,
        'num_clusters': result['num_clusters'],
        'clustered_path_length_mm': result['path_length_mm'],
        'original_path_length_mm': grid['original_path_length_mm'],
        'clustering_timestamp': datetime.now().isoformat(),
    }
    if method == 'dbscan':
        cluster_meta['dbscan_eps_mm'] = eps_mm
        cluster_meta['dbscan_min_samples'] = params.min_samples
        cluster_meta['dbscan_normal_weight'] = params.normal_weight
    elif method == 'coacd':
        cluster_meta['coacd_threshold'] = params.coacd_threshold
    elif method == 'coacd+dbscan':
        cluster_meta['coacd_threshold'] = params.coacd_threshold
        cluster_meta['dbscan_eps_mm'] = eps_mm
        cluster_meta['dbscan_min_samples'] = params.min_samples
        cluster_meta['dbscan_normal_weight'] = params.normal_weight
    elif method == 'agglomerative':
        cluster_meta['normal_weight'] = params.normal_weight
        if params.max_span_mm:
            cluster_meta['max_span_mm'] = params.max_span_mm
        else:
            cluster_meta['target_size'] = params.target_size
    elif method == 'coacd+agglomerative':
        cluster_meta['coacd_threshold'] = params.coacd_threshold
        cluster_meta['normal_weight'] = params.normal_weight
        if params.max_span_mm:
            cluster_meta['max_span_mm'] = params.max_span_mm
        else:
            cluster_meta['target_size'] = params.target_size

    return ViewpointResult(
        positions=grid['positions'], normals=grid['normals'],
        camera_positions=grid['camera_positions'],
        path_order=result['path_order'], row_index=grid['row_index'],
        cluster_id=result['cluster_ids'], cluster_order=result['cluster_order'],
        cluster_direction=result['cluster_direction'],
        coacd_parts=result.get('coacd_parts'), coacd_ids=result.get('coacd_ids'),
        pca={'center': grid['pca_center'], 'axis1': grid['pca_axis1'], 'axis2': grid['pca_axis2']},
        row_spacing_m=grid['row_spacing_m'], col_spacing_m=grid['col_spacing_m'],
        original_path_length_mm=grid['original_path_length_mm'],
        clustered_path_length_mm=result['path_length_mm'],
        num_clusters=result['num_clusters'],
        cluster_meta=cluster_meta, method=method, label=label,
    )


# ============================================================================
# Main
# ============================================================================


def main():
    args = parse_arguments()

    # 물체별 배치를 반영(rotation 은 bottom-filter 판정에 사용 — line ~1776).
    if config.apply_object_placement(args.object):
        print(f"  Per-object placement '{args.object}': quat={config.TARGET_OBJECT['rotation']}")

    input_path = str(config.get_mesh_path(args.object, mesh_type="source"))

    print("=" * 60)
    print("GENERATE VIEWPOINTS")
    print("=" * 60)
    print(f"Object: {args.object}")
    print(f"Input:  {input_path}")
    if args.material_rgb:
        print(f"Target RGB: {args.material_rgb}")
    else:
        print(f"Target: entire mesh (no material filter)")
    print(f"Clustering: {args.cluster_method}")
    print()

    # 1-2. Load mesh + extract target mesh (material filter)
    try:
        mesh, target_mesh, input_path = load_meshes(
            args.object, args.material_rgb, args.color_tolerance,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return 1

    params = ViewpointGenParams(
        material_rgb=args.material_rgb,
        color_tolerance=args.color_tolerance,
        row_spacing_mm=args.row_spacing,
        col_spacing_mm=args.col_spacing,
        filter_bottom=not args.no_filter_bottom,
        bottom_angle=args.bottom_angle,
        cluster_method=args.cluster_method,
        eps_mm=args.eps,
        min_samples=args.min_samples,
        normal_weight=args.normal_weight,
        coacd_threshold=args.coacd_threshold,
        target_size=args.target_size,
        max_span_mm=args.max_span,
        sampling_mode=args.sampling_mode,
        surface_spacing_mm=args.surface_spacing,
        ordering_mode=args.ordering_mode,
    )

    method = args.cluster_method
    eps_mm = args.eps if args.eps else config.CAMERA_FOV_WIDTH_MM

    # ------------------------------------------------------------------
    # Compare mode: 파라미터 스윕 → 드롭다운 HTML
    # ------------------------------------------------------------------
    if args.compare:
        grid = prepare_grid(target_mesh, params)
        common = dict(
            positions=grid['positions'], normals=grid['normals'],
            camera_positions=grid['camera_positions'], target_mesh=target_mesh,
            row_spacing_m=grid['row_spacing_m'], col_spacing_m=grid['col_spacing_m'],
            grid_row_index=grid['grid_row_index'],
            cam_axis1=grid['cam_axis1'], cam_axis2=grid['cam_axis2'],
            original_path_length_mm=grid['original_path_length_mm'],
            ordering_mode=params.ordering_mode,
        )
        fov_w = config.CAMERA_FOV_WIDTH_MM
        compare_results = {}

        if method == 'dbscan':
            print("Comparing DBSCAN (eps variations)...")
            for factor in [0.5, 0.75, 1.0, 1.5, 2.0]:
                e_mm = fov_w * factor
                label = f"dbscan eps={e_mm:.0f}mm"
                compare_results[label] = cluster_and_order(
                    label, 'dbscan', **common, eps_m=e_mm / 1000.0,
                    min_samples=args.min_samples, normal_weight=args.normal_weight,
                )
        elif method == 'coacd':
            print("Comparing CoACD (threshold variations)...")
            for t in [0.1, 0.2, 0.25, 0.3]:
                label = f"coacd t={t}"
                compare_results[label] = cluster_and_order(
                    label, 'coacd', **common, threshold=t,
                    normal_weight=args.normal_weight,
                )
        elif method == 'coacd+dbscan':
            t = args.coacd_threshold
            print(f"Comparing CoACD+DBSCAN (coacd_threshold={t} fixed, eps variations)...")
            # CoACD 1회 실행 후 캐싱
            t0 = time.perf_counter()
            cached_coacd = cluster_coacd(target_mesh, grid['positions'], t)
            t_coacd = time.perf_counter() - t0
            print(f"  CoACD precomputed: {len(np.unique(cached_coacd[0]))} parts ({t_coacd:.3f}s)")
            for factor in [0.5, 0.75, 1.0, 1.5, 2.0]:
                e_mm = fov_w * factor
                label = f"coacd+dbscan t={t} eps={e_mm:.0f}mm"
                compare_results[label] = cluster_and_order(
                    label, 'coacd+dbscan', **common, threshold=t,
                    eps_m=e_mm / 1000.0, min_samples=args.min_samples,
                    normal_weight=args.normal_weight,
                    precomputed_coacd=cached_coacd,
                )
        elif method == 'agglomerative':
            if args.max_span:
                print("Comparing Agglomerative (max_span variations)...")
                for ms in [40, 60, 80, 120]:
                    label = f"agglomerative span={ms}mm"
                    compare_results[label] = cluster_and_order(
                        label, 'agglomerative', **common,
                        max_span_mm=ms, normal_weight=args.normal_weight,
                    )
            else:
                print("Comparing Agglomerative (target_size variations)...")
                for ts in [8, 12, 16, 24]:
                    label = f"agglomerative ts={ts}"
                    compare_results[label] = cluster_and_order(
                        label, 'agglomerative', **common,
                        target_size=ts, normal_weight=args.normal_weight,
                    )
        elif method == 'coacd+agglomerative':
            t = args.coacd_threshold
            t0 = time.perf_counter()
            cached_coacd = cluster_coacd(target_mesh, grid['positions'], t)
            t_coacd = time.perf_counter() - t0
            print(f"  CoACD precomputed: {len(np.unique(cached_coacd[0]))} parts ({t_coacd:.3f}s)")
            if args.max_span:
                print(f"Comparing CoACD+Agglomerative (coacd_threshold={t} fixed, max_span variations)...")
                for ms in [40, 60, 80, 120]:
                    label = f"coacd+agglomerative t={t} span={ms}mm"
                    compare_results[label] = cluster_and_order(
                        label, 'coacd+agglomerative', **common, threshold=t,
                        max_span_mm=ms, normal_weight=args.normal_weight,
                        precomputed_coacd=cached_coacd,
                    )
            else:
                print(f"Comparing CoACD+Agglomerative (coacd_threshold={t} fixed, target_size variations)...")
                for ts in [8, 12, 16, 24]:
                    label = f"coacd+agglomerative t={t} ts={ts}"
                    compare_results[label] = cluster_and_order(
                        label, 'coacd+agglomerative', **common, threshold=t,
                        target_size=ts, normal_weight=args.normal_weight,
                        precomputed_coacd=cached_coacd,
                    )

        print()

        # 비교 HTML 저장
        if not args.dry_run:
            html_path = str(config.get_viewpoint_path(
                args.object, len(grid['positions']), filename=f"compare_{method}.html",
            ))
            os.makedirs(os.path.dirname(html_path), exist_ok=True)
            visualize_clusters_html(
                mesh, grid['positions'], grid['camera_positions'],
                compare_results, grid['original_path_length_mm'],
                html_path,
            )

        print("Compare complete!")
        print("=" * 60)
        return 0

    # ------------------------------------------------------------------
    # Single mode: 생성 코어 호출 → 저장 → 시각화
    # ------------------------------------------------------------------
    res = generate_viewpoints_core(target_mesh, params)

    # 9. Save to HDF5
    if args.dry_run:
        print()
        print("[DRY RUN] HDF5 not modified.")
    else:
        output_path = str(config.get_viewpoint_path(
            args.object, len(res.positions), filename=f"viewpoints_{method}.h5",
        ))
        print(f"Output: {output_path}")

        print("Saving to HDF5...")
        camera_spec = {
            'fov_width_mm': config.CAMERA_FOV_WIDTH_MM,
            'fov_height_mm': config.CAMERA_FOV_HEIGHT_MM,
            'working_distance_mm': config.CAMERA_WORKING_DISTANCE_MM,
        }
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'input_mesh': str(input_path),
            'method': f"{params.sampling_mode}+{params.ordering_mode}",
            'sampling_mode': params.sampling_mode,
            'ordering_mode': params.ordering_mode,
            'surface_spacing_mm': params.surface_spacing_mm if params.surface_spacing_mm
                else min(res.row_spacing_m, res.col_spacing_m) * 1000.0,
            'row_spacing_mm': res.row_spacing_m * 1000.0,
            'col_spacing_mm': res.col_spacing_m * 1000.0,
            'total_path_length_mm': compute_path_length(res.camera_positions, res.path_order) * 1000.0,
        }
        pca_data = {
            'center': res.pca['center'], 'axis1': res.pca['axis1'], 'axis2': res.pca['axis2'],
        }
        save_viewpoints_hdf5(
            res.positions, res.normals, output_path, metadata, camera_spec,
            res.path_order, pca_data, res.row_index,
            cluster_id=res.cluster_id,
            cluster_order=res.cluster_order,
            cluster_direction=res.cluster_direction,
            cluster_metadata=res.cluster_meta,
        )
        print()

    print("Complete!")
    print("=" * 60)

    # 10. Visualization
    if not args.dry_run:
        html_path = str(Path(output_path).with_suffix('.html'))
        cluster_result = {
            res.label: {
                'cluster_ids': res.cluster_id, 'cluster_order': res.cluster_order,
                'path_order': res.path_order, 'path_length_mm': res.clustered_path_length_mm,
                'num_clusters': res.num_clusters,
            }
        }
        if res.coacd_parts is not None:
            cluster_result[res.label]['coacd_parts'] = res.coacd_parts
        if res.coacd_ids is not None:
            cluster_result[res.label]['coacd_ids'] = res.coacd_ids
        visualize_clusters_html(
            mesh, res.positions, res.camera_positions,
            cluster_result, res.original_path_length_mm,
            html_path,
        )

    return 0


if __name__ == '__main__':
    sys.exit(main())
