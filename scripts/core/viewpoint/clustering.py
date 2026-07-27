"""Spatial and mesh-based viewpoint clustering."""

from __future__ import annotations

import time
import warnings
from typing import List, Optional, Tuple

import numpy as np
import trimesh
from sklearn.cluster import DBSCAN

from .adjacency import components_from_edges

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


def cluster_delaunay(edges: np.ndarray, n_points: int) -> np.ndarray:
    """Delaunay 인접 그래프의 연결성분을 stage-1 그룹으로 쓴다.

    CoACD의 convex 파트와 같은 자리를 차지하지만, 볼록 분해가 아니라 **표면 연결성**이
    기준이다. 위상적으로 끊긴 면이 한 그룹에 섞이지 않고, CoACD의 비결정성도 없다.

    성분 수는 보통 매우 적으므로(물체 하나면 1개인 경우도 흔하다) 이것만으로는 스캔
    클러스터가 되지 않는다 — ``subcluster_within_parts``로 성분 내부를 반드시 세분화한다.
    """
    _, component_id = components_from_edges(edges, n_points)
    return component_id.astype(np.int32)


def subcluster_within_parts(
    part_ids: np.ndarray,
    normals: np.ndarray,
    camera_positions: np.ndarray,
    submethod: str,
    *,
    target_size: int = 12,
    normal_weight: float = 0.0,
    max_span_mm: Optional[float] = None,
    eps_m: float = 0.03,
    min_samples: int = 2,
) -> np.ndarray:
    """stage-1 파트 안에서만 세분화하고 전역 0-based 라벨로 이어 붙인다.

    기준은 표면 positions가 아니라 **camera_positions** — 렌더·로봇 EE가 카메라 위치이고,
    곡면에서는 표면이 가까워도 카메라는 벌어지기 때문이다.

    stage-1이 CoACD 파트든 Delaunay 성분이든 이 단계는 동일하다.

    Args:
        part_ids: (N,) stage-1 그룹 할당
        submethod: 'dbscan' | 'agglomerative'

    Returns:
        cluster_ids: (N,) 0-based 최종 클러스터 할당
    """
    if submethod not in ('dbscan', 'agglomerative'):
        raise ValueError(f"Unknown sub-clustering method: {submethod}")

    final_ids = np.full(len(part_ids), -1, dtype=np.int32)
    next_cluster = 0

    for part_id in np.unique(part_ids):
        mask = part_ids == part_id
        indices = np.where(mask)[0]

        if submethod == 'dbscan':
            if len(indices) < min_samples:
                # 포인트가 너무 적으면 하나의 클러스터로
                final_ids[indices] = next_cluster
                next_cluster += 1
                continue
            sub_ids = cluster_dbscan(
                camera_positions[mask], normals[mask],
                eps_m, min_samples, normal_weight,
            )
        else:
            sub_ids = cluster_agglomerative(
                camera_positions[mask], normals[mask], target_size, normal_weight,
                max_span_mm=max_span_mm,
            )

        for sub_id in np.unique(sub_ids):
            final_ids[indices[sub_ids == sub_id]] = next_cluster
            next_cluster += 1

    return final_ids


def cluster_delaunay_subclustered(
    edges: np.ndarray,
    normals: np.ndarray,
    camera_positions: np.ndarray,
    submethod: str,
    **subcluster_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Delaunay 성분 → sub-clustering 2단계 (CoACD 2단계 함수들과 같은 형태).

    Returns: (cluster_ids, component_ids) — component_ids는 시각화·로그용 stage-1 라벨.
    """
    n_points = len(camera_positions)
    t0 = time.perf_counter()
    component_ids = cluster_delaunay(edges, n_points)
    num_components = len(np.unique(component_ids))
    t_components = time.perf_counter() - t0

    print(f"  Delaunay+{submethod}: {num_components} surface components → sub-clustering...")
    t0 = time.perf_counter()
    final_ids = subcluster_within_parts(
        component_ids, normals, camera_positions, submethod, **subcluster_kwargs,
    )
    t_sub = time.perf_counter() - t0

    print(f"  Delaunay+{submethod}: {num_components} components → "
          f"{len(np.unique(final_ids))} final clusters "
          f"(components={t_components:.3f}s, {submethod}={t_sub:.3f}s)")
    return final_ids, component_ids


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
    final_ids = subcluster_within_parts(
        coacd_ids, normals, camera_positions, 'dbscan',
        eps_m=eps_m, min_samples=min_samples, normal_weight=normal_weight,
    )
    t_dbscan = time.perf_counter() - t0

    print(f"  CoACD+DBSCAN: {num_coacd_parts} parts → {len(np.unique(final_ids))} final clusters "
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
    final_ids = subcluster_within_parts(
        coacd_ids, normals, camera_positions, 'agglomerative',
        target_size=target_size, normal_weight=normal_weight, max_span_mm=max_span_mm,
    )
    t_aggl = time.perf_counter() - t0

    print(f"  CoACD+Agglomerative: {num_coacd_parts} parts → "
          f"{len(np.unique(final_ids))} final clusters "
          f"(coacd={t_coacd:.3f}s, aggl={t_aggl:.3f}s)")
    return final_ids, part_meshes, coacd_ids
