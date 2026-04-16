#!/usr/bin/env python3
"""
뷰포인트 생성 및 클러스터링 기반 경로 순서 최적화

메시 표면에 PCA 주축을 기반으로 그리드 형태의 뷰포인트를 생성한다.
선택적으로 공간 클러스터링을 수행하여 경로 순서를 최적화한다.

입력: OBJ 파일 (선택적으로 재질 RGB 색상 필터 적용 가능)
출력: 표면 위치, 법선 벡터, 경로 순서 인덱스가 포함된 HDF5 파일

사용법:
    # 기본: FOV 기반 자동 간격
    uv run scripts/pipeline/generate_viewpoints.py --object sample

    # 재질 필터
    uv run scripts/pipeline/generate_viewpoints.py --object sample --material-rgb "170,163,158"

    # 간격 수동 오버라이드 (mm)
    uv run scripts/pipeline/generate_viewpoints.py --object sample --row-spacing 5.0 --col-spacing 5.0

    # 파라미터 변형 비교 (드롭다운 HTML)
    uv run scripts/pipeline/generate_viewpoints.py --object sample --cluster-method coacd --compare

    # CoACD 기반 클러스터링 (메시 convex decomposition)
    uv run scripts/pipeline/generate_viewpoints.py --object sample --cluster-method coacd

    uv run scripts/pipeline/generate_viewpoints.py --object sample --material-rgb "170,163,158" --cluster-method dbscan
    uv run scripts/pipeline/generate_viewpoints.py --object sample --material-rgb "170,163,158" --cluster-method coacd
    # Sample
    uv run scripts/pipeline/generate_viewpoints.py --object sample --material-rgb "170,163,158" --cluster-method dbscan --normal-weight 0.05 --compare
    uv run scripts/pipeline/generate_viewpoints.py --object sample --material-rgb "170,163,158" --cluster-method coacd --normal-weight 0.05 --compare
    uv run scripts/pipeline/generate_viewpoints.py --object sample --material-rgb "170,163,158" --cluster-method coacd+dbscan --normal-weight 0.05 --coacd-threshold 0.25 --compare
    uv run scripts/pipeline/generate_viewpoints.py --object sample --material-rgb "170,163,158" --cluster-method coacd+dbscan --normal-weight 0.05 --coacd-threshold 0.25 --eps 31

    # Glass
    uv run scripts/pipeline/generate_viewpoints.py --object glass --cluster-method dbscan --normal-weight 0.05 --compare
    uv run scripts/pipeline/generate_viewpoints.py --object glass --cluster-method coacd --normal-weight 0.05 --compare
    uv run scripts/pipeline/generate_viewpoints.py --object glass --cluster-method coacd+dbscan --normal-weight 0.05 --coacd-threshold 0.25 --compare
"""

import os
import sys
import argparse
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import trimesh
import h5py
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import config
from common.math_utils import quaternion_to_rotation_matrix
from viz.visualize_viewpoints import visualize_clusters_html


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
    coacd_threshold: float = 0.05,
    eps_m: float = 0.03,
    min_samples: int = 2,
    normal_weight: float = 0.0,
    precomputed_coacd: Optional[Tuple[np.ndarray, List]] = None,
) -> Tuple[np.ndarray, List, np.ndarray]:
    """CoACD → DBSCAN 2단계 클러스터링.

    1단계: CoACD로 메시를 convex 파트로 분해하여 뷰포인트를 파트별로 할당.
    2단계: 각 CoACD 파트 내에서 DBSCAN으로 세분화.

    Args:
        mesh: 대상 메시
        positions: (N, 3) 뷰포인트 표면 위치
        normals: (N, 3) 표면 법선 벡터
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

    # 2단계: 각 CoACD 파트 내에서 DBSCAN
    # camera_positions는 positions + normals * working_distance이므로
    # 여기서는 positions (표면 위치) 기준으로 DBSCAN 적용
    t0 = time.perf_counter()
    final_ids = np.full(len(positions), -1, dtype=np.int32)
    next_cluster = 0
    total_sub_clusters = 0

    for part_id in np.unique(coacd_ids):
        mask = coacd_ids == part_id
        part_positions = positions[mask]
        part_normals = normals[mask]
        indices = np.where(mask)[0]

        if len(part_positions) < min_samples:
            # 포인트가 너무 적으면 하나의 클러스터로
            final_ids[indices] = next_cluster
            next_cluster += 1
            total_sub_clusters += 1
        else:
            sub_ids = cluster_dbscan(
                part_positions, part_normals,
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


def compute_cluster_internal_order(
    cluster_ids: np.ndarray,
    camera_positions: np.ndarray,
    normals: np.ndarray,
    row_spacing_m: float,
    grid_row_index: Optional[np.ndarray] = None,
    global_axis1: Optional[np.ndarray] = None,
    global_axis2: Optional[np.ndarray] = None,
) -> dict:
    """각 클러스터의 내부 zigzag 순서를 사전 계산.

    TSP 전에 호출하여 클러스터별 start/end point를 확보한다.

    grid_row_index가 주어지면 전역 축(global_axis1/axis2)을 사용하여
    클러스터별 PCA를 생략한다. row_index_override가 행 구분을 담당하므로
    로컬 PCA의 axis1은 불필요하고, axis2는 클러스터 간 열 방향 일관성을
    위해 이미 전역 값을 사용해야 하기 때문이다.

    Args:
        cluster_ids: (N,) 클러스터 ID
        camera_positions: (N, 3) 카메라 위치
        normals: (N, 3) 법선
        row_spacing_m: 행 간격 (미터)
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
  uv run scripts/pipeline/generate_viewpoints.py --object sample

  # 재질 필터
  uv run scripts/pipeline/generate_viewpoints.py --object sample --material-rgb "170,163,158"

  # DBSCAN 클러스터링
  uv run scripts/pipeline/generate_viewpoints.py --object sample --material-rgb "170,163,158" --cluster-method dbscan

  # CoACD 클러스터링 (메시 convex decomposition)
  uv run scripts/pipeline/generate_viewpoints.py --object sample --cluster-method coacd --coacd-threshold 0.05
        """,
    )

    # --- Viewpoint generation ---
    parser.add_argument('--object', type=str, required=True, help='오브젝트 이름')
    parser.add_argument('--material-rgb', type=str, default=None,
                        help='Target material RGB color as "R,G,B" (e.g., "170,163,158")')
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
                        choices=['dbscan', 'coacd', 'coacd+dbscan'],
                        help='클러스터링 방법 (기본: dbscan)')
    parser.add_argument('--eps', type=float, default=None,
                        help=f'[dbscan] 이웃 반경 mm (기본: {config.CAMERA_FOV_WIDTH_MM:.0f}mm)')
    parser.add_argument('--min-samples', type=int, default=2,
                        help='[dbscan] 코어 포인트 최소 이웃 수 (기본: 2)')
    parser.add_argument('--normal-weight', type=float, default=0.0,
                        help='[dbscan] 법선 가중치 (미터 단위, 0이면 위치만 사용, 기본: 0.0)')
    parser.add_argument('--coacd-threshold', type=float, default=0.05,
                        help='[coacd] concavity threshold (낮을수록 더 많은 파트, 기본: 0.05)')

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
# Main
# ============================================================================

def main():
    args = parse_arguments()

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

    # Validate input exists
    if not os.path.exists(input_path):
        print(f"Error: Input mesh not found: {input_path}")
        return 1

    # 1. Load OBJ
    print("Loading mesh...")
    loaded = trimesh.load(input_path)

    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
    else:
        mesh = loaded

    print(f"  Loaded: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} triangles")
    print()

    # 2. Determine target mesh
    if args.material_rgb:
        print("Parsing materials...")
        triangle_materials, mtl_file = parse_obj_material_usage(input_path)

        if mtl_file is None or not os.path.exists(mtl_file):
            print(f"Error: MTL file not found")
            return 1

        materials = parse_mtl_file(mtl_file)
        print(f"  Found {len(materials)} materials:")
        for mat_name, mat_props in materials.items():
            if 'Kd' in mat_props:
                rgb = kd_to_rgb(mat_props['Kd'])
                print(f"    - {mat_name}: RGB{rgb}")
        print()

        print("Matching material...")
        target_rgb = tuple(map(int, args.material_rgb.split(',')))
        matched_materials = match_material_by_color(materials, target_rgb, args.color_tolerance)

        if len(matched_materials) == 0:
            print(f"  Error: No materials matched RGB{target_rgb} within tolerance {args.color_tolerance}")
            return 1

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

    # 3. Calculate spacing
    if args.row_spacing:
        row_spacing_m = args.row_spacing / 1000.0
    else:
        row_spacing_m = config.CAMERA_FOV_HEIGHT_MM / 1000.0 * (1.0 - config.CAMERA_OVERLAP_RATIO)
    if args.col_spacing:
        col_spacing_m = args.col_spacing / 1000.0
    else:
        col_spacing_m = config.CAMERA_FOV_WIDTH_MM / 1000.0 * (1.0 - config.CAMERA_OVERLAP_RATIO)

    print(f"  Row spacing (axis1): {row_spacing_m * 1000:.1f} mm")
    print(f"  Col spacing (axis2): {col_spacing_m * 1000:.1f} mm")
    print()

    # 4. Generate viewpoints
    print(f"Generating grid viewpoints (PCA)...")
    positions, normals, path_order, row_index, pca_center, pca_axis1, pca_axis2 = generate_grid_viewpoints(
        target_mesh, row_spacing_m, col_spacing_m,
    )
    print(f"  Generated: {len(positions)} viewpoints")

    # Camera positions
    wd_m = config.CAMERA_WORKING_DISTANCE_MM / 1000.0
    camera_positions = positions + normals * wd_m

    # 5. PCA 축 기준 지그재그 정렬
    grid_row_index = row_index.copy()  # 그리드 생성 시 확정된 원본 행 인덱스 보존
    _, cam_axis1, cam_axis2 = compute_pca_axes(camera_positions.astype(np.float64))
    path_order, row_index, n_rows = reorder_zigzag(camera_positions, cam_axis1, cam_axis2, row_spacing_m)
    print(f"  Ordered with PCA-axis on camera positions: {n_rows} rows")

    # 7. Filter bottom-facing viewpoints
    if not args.no_filter_bottom:
        R_obj = quaternion_to_rotation_matrix(config.TARGET_OBJECT["rotation"])
        world_normals = (R_obj @ normals.T).T
        cos_thresh = np.cos(np.deg2rad(args.bottom_angle))
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
            print(f"  Filtered {n_removed} bottom-facing viewpoints (within {args.bottom_angle}° from down)")
            print(f"  Remaining: {len(positions)} viewpoints")
        else:
            print(f"  No bottom-facing viewpoints to filter")

    # Path length (before clustering)
    original_path_length_mm = compute_path_length(camera_positions, path_order) * 1000.0
    print(f"  Total path length: {original_path_length_mm:.1f} mm")
    print()

    # 8. Clustering helper
    base_path_order = path_order.copy()

    def run_clustering(label, method, **kwargs):
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
            cids, coacd_parts = cluster_coacd(target_mesh, positions, kwargs['threshold'])
        elif method == 'coacd+dbscan':
            cids, coacd_parts, coacd_ids = cluster_coacd_dbscan(
                target_mesh, positions, normals,
                coacd_threshold=kwargs['threshold'],
                eps_m=kwargs.get('eps_m', 0.03),
                min_samples=kwargs.get('min_samples', 2),
                normal_weight=kwargs.get('normal_weight', 0.0),
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
            cids, camera_positions, normals, row_spacing_m,
            grid_row_index=grid_row_index, global_axis1=cam_axis1, global_axis2=cam_axis2,
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
        if method == 'coacd+dbscan':
            result['coacd_ids'] = coacd_ids
        return result

    method = args.cluster_method
    eps_mm = args.eps if args.eps else config.CAMERA_FOV_WIDTH_MM

    if args.compare:
        # 파라미터 스윕 비교 모드
        fov_w = config.CAMERA_FOV_WIDTH_MM
        compare_results = {}

        if method == 'dbscan':
            print("Comparing DBSCAN (eps variations)...")
            for factor in [0.5, 0.75, 1.0, 1.5, 2.0]:
                e_mm = fov_w * factor
                label = f"dbscan eps={e_mm:.0f}mm"
                compare_results[label] = run_clustering(
                    label, 'dbscan', eps_m=e_mm / 1000.0,
                    min_samples=args.min_samples, normal_weight=args.normal_weight,
                )
        elif method == 'coacd':
            print("Comparing CoACD (threshold variations)...")
            for t in [0.1, 0.2, 0.25, 0.3]:
                label = f"coacd t={t}"
                compare_results[label] = run_clustering(
                    label, 'coacd', threshold=t,
                    normal_weight=args.normal_weight,
                )
        elif method == 'coacd+dbscan':
            t = args.coacd_threshold
            print(f"Comparing CoACD+DBSCAN (coacd_threshold={t} fixed, eps variations)...")
            # CoACD 1회 실행 후 캐싱
            t0 = time.perf_counter()
            cached_coacd = cluster_coacd(target_mesh, positions, t)
            t_coacd = time.perf_counter() - t0
            print(f"  CoACD precomputed: {len(np.unique(cached_coacd[0]))} parts ({t_coacd:.3f}s)")
            for factor in [0.5, 0.75, 1.0, 1.5, 2.0]:
                e_mm = fov_w * factor
                label = f"coacd+dbscan t={t} eps={e_mm:.0f}mm"
                compare_results[label] = run_clustering(
                    label, 'coacd+dbscan', threshold=t,
                    eps_m=e_mm / 1000.0, min_samples=args.min_samples,
                    normal_weight=args.normal_weight,
                    precomputed_coacd=cached_coacd,
                )

        print()

        # 비교 HTML 저장
        if not args.dry_run:
            html_path = str(config.get_viewpoint_path(
                args.object, len(positions), filename=f"compare_{method}.html",
            ))
            os.makedirs(os.path.dirname(html_path), exist_ok=True)
            visualize_clusters_html(
                mesh, positions, camera_positions,
                compare_results, original_path_length_mm,
                html_path,
            )

        print("Compare complete!")
        print("=" * 60)
        return 0

    # 단일 클러스터링 모드
    if method == 'dbscan':
        nw_str = f", normal_weight: {args.normal_weight}" if args.normal_weight > 0 else ""
        print(f"Clustering (DBSCAN, eps: {eps_mm:.0f} mm, min_samples: {args.min_samples}{nw_str})...")
        label = method
        result = run_clustering(
            label, 'dbscan', eps_m=eps_mm / 1000.0,
            min_samples=args.min_samples, normal_weight=args.normal_weight,
        )
    elif method == 'coacd':
        print(f"Clustering (CoACD, threshold: {args.coacd_threshold})...")
        label = method
        result = run_clustering(label, 'coacd', threshold=args.coacd_threshold,
                               normal_weight=args.normal_weight)
    elif method == 'coacd+dbscan':
        nw_str = f", normal_weight: {args.normal_weight}" if args.normal_weight > 0 else ""
        print(f"Clustering (CoACD+DBSCAN, coacd_threshold: {args.coacd_threshold}, "
              f"eps: {eps_mm:.0f} mm, min_samples: {args.min_samples}{nw_str})...")
        label = f"coacd+dbscan t={args.coacd_threshold} eps={eps_mm:.0f}mm"
        result = run_clustering(
            label, 'coacd+dbscan', threshold=args.coacd_threshold,
            eps_m=eps_mm / 1000.0, min_samples=args.min_samples,
            normal_weight=args.normal_weight,
        )

    cluster_ids = result['cluster_ids']
    cluster_order = result['cluster_order']
    cluster_direction = result['cluster_direction']
    path_order = result['path_order']
    clustered_path_length_mm = result['path_length_mm']
    K = result['num_clusters']

    cluster_meta = {
        'clustering_method': method,
        'num_clusters': K,
        'clustered_path_length_mm': clustered_path_length_mm,
        'original_path_length_mm': original_path_length_mm,
        'clustering_timestamp': datetime.now().isoformat(),
    }
    if method == 'dbscan':
        cluster_meta['dbscan_eps_mm'] = eps_mm
        cluster_meta['dbscan_min_samples'] = args.min_samples
        cluster_meta['dbscan_normal_weight'] = args.normal_weight
    elif method == 'coacd':
        cluster_meta['coacd_threshold'] = args.coacd_threshold
    elif method == 'coacd+dbscan':
        cluster_meta['coacd_threshold'] = args.coacd_threshold
        cluster_meta['dbscan_eps_mm'] = eps_mm
        cluster_meta['dbscan_min_samples'] = args.min_samples
        cluster_meta['dbscan_normal_weight'] = args.normal_weight

    # 9. Save to HDF5
    if args.dry_run:
        print()
        print("[DRY RUN] HDF5 not modified.")
    else:
        output_path = str(config.get_viewpoint_path(
            args.object, len(positions), filename=f"viewpoints_{method}.h5",
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
            'method': 'pca_grid',
            'row_spacing_mm': row_spacing_m * 1000.0,
            'col_spacing_mm': col_spacing_m * 1000.0,
            'total_path_length_mm': compute_path_length(camera_positions, path_order) * 1000.0,
        }
        pca_data = {'center': pca_center, 'axis1': pca_axis1, 'axis2': pca_axis2}
        save_viewpoints_hdf5(
            positions, normals, output_path, metadata, camera_spec,
            path_order, pca_data, row_index,
            cluster_id=cluster_ids,
            cluster_order=cluster_order,
            cluster_direction=cluster_direction,
            cluster_metadata=cluster_meta,
        )
        print()

    print("Complete!")
    print("=" * 60)

    # 10. Visualization
    if not args.dry_run:
        html_path = str(Path(output_path).with_suffix('.html'))
        cluster_result = {
            label: {
                'cluster_ids': cluster_ids, 'cluster_order': cluster_order,
                'path_order': path_order, 'path_length_mm': clustered_path_length_mm,
                'num_clusters': K,
            }
        }
        if 'coacd_parts' in result:
            cluster_result[label]['coacd_parts'] = result['coacd_parts']
        if 'coacd_ids' in result:
            cluster_result[label]['coacd_ids'] = result['coacd_ids']
        visualize_clusters_html(
            mesh, positions, camera_positions,
            cluster_result, original_path_length_mm,
            html_path,
        )

    return 0


if __name__ == '__main__':
    sys.exit(main())
