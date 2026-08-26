"""Importable viewpoint generation pipeline."""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np

from common import config
from common.math_utils import quaternion_to_rotation_matrix
from .adjacency import build_local_delaunay_adjacency
from .clustering import (
    cluster_agglomerative,
    cluster_coacd,
    cluster_coacd_agglomerative,
    cluster_coacd_dbscan,
    cluster_dbscan,
    cluster_delaunay_subclustered,
)
from .models import ViewpointGenParams, ViewpointResult
from .ordering import (
    build_clustered_path_order,
    compute_cluster_internal_order,
    order_clusters_gtsp,
)
from .sampling import (
    _nn_path_length,
    compute_path_length,
    filter_interior_viewpoints,
    generate_surface_viewpoints,
)

def cluster_and_order(label, method, *, positions, normals, camera_positions, target_mesh,
                      row_spacing_m, col_spacing_m, original_path_length_mm, **kwargs):
    """한 가지 클러스터링 설정을 실행하고 결과 dict를 반환.

    아래 dispatch만 방법마다 다르고, 그 뒤(내부 순서 → GTSP 클러스터 순서 → 전역
    path_order)는 임의의 ``cids``를 받는 공통 경로다. 새 클러스터링 방법은 ``cids``만
    만들면 되고, 저장 스키마도 그대로 유지된다.
    """
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
    elif method in ('delaunay+dbscan', 'delaunay+agglomerative'):
        edges = kwargs.get('adjacency_edges')
        if edges is None:
            raise ValueError(
                f"method={method} requires adjacency_edges "
                "(build_local_delaunay_adjacency must run before clustering)"
            )
        submethod = method.split('+', 1)[1]
        sub_kwargs = {'normal_weight': kwargs.get('normal_weight', 0.0)}
        if submethod == 'dbscan':
            sub_kwargs.update(
                eps_m=kwargs.get('eps_m', 0.03),
                min_samples=kwargs.get('min_samples', 2),
            )
        else:
            sub_kwargs.update(
                target_size=kwargs.get('target_size', 12),
                max_span_mm=kwargs.get('max_span_mm'),
            )
        cids, component_ids = cluster_delaunay_subclustered(
            edges, normals, camera_positions, submethod, **sub_kwargs,
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
        ordering_mode=kwargs.get('ordering_mode', 'lawnmower'),
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
    if method in ('delaunay+dbscan', 'delaunay+agglomerative'):
        result['component_ids'] = component_ids
    return result


def prepare_viewpoints(target_mesh, params: ViewpointGenParams):
    """표면 뷰포인트 생성 + bottom/interior 필터 (클러스터링 전 단계).

    샘플링은 메시 표면 직접 FPS 하나뿐이다. 예전에는 PCA 평면에 격자를 깔고
    ``closest_point`` 로 표면에 투영하는 grid 모드가 있었지만, 평면 투영이라 곡면·측벽을
    놓치고 속 빈 물체에서는 안쪽 면이 더 가까워 지붕을 통째로 잃었다. 저장된 h5 는 전부
    surface 로 만들어졌고 grid 산출물은 하나도 없어 2026-08-26 에 제거했다.

    Returns: dict — positions, normals, camera_positions,
        row_spacing_m, col_spacing_m, original_path_length_mm

    방문 순서는 만들지 않는다 — cluster_and_order 가 클러스터 내부 순서와 GTSP 클러스터
    순서에서 새로 만든다. row/col spacing 은 lawnmower 의 row 간격(min)과 h5 metadata 로
    쓰이므로 유도값이라도 반환한다.
    """
    # spacing — row/col 을 직접 준 게 없으면 FOV×(1-overlap) 로 유도한다.
    # FOV·overlap·WD 는 params 가 소유한다(__post_init__ 이 config 로 해소).
    if params.row_spacing_mm:
        row_spacing_m = params.row_spacing_mm / 1000.0
    else:
        row_spacing_m = params.fov_height_mm / 1000.0 * (1.0 - params.overlap_ratio)
    if params.col_spacing_mm:
        col_spacing_m = params.col_spacing_mm / 1000.0
    else:
        col_spacing_m = params.fov_width_mm / 1000.0 * (1.0 - params.overlap_ratio)

    print(f"  Row spacing (axis1): {row_spacing_m * 1000:.1f} mm")
    print(f"  Col spacing (axis2): {col_spacing_m * 1000:.1f} mm")
    print(f"  Working distance:    {params.working_distance_mm:.1f} mm "
          f"(FOV {params.fov_width_mm:.0f}×{params.fov_height_mm:.0f} mm)")
    print()

    # 이 한 줄이 곧 h5 → IK/궤적/GLNS 로 흘러가는 값이다(camera_positions 기준).
    wd_m = params.working_distance_mm / 1000.0

    # 표면 직접 균일 샘플링 (FPS) — 곡면 커버리지
    spacing_m = (params.surface_spacing_mm / 1000.0) if params.surface_spacing_mm \
        else min(row_spacing_m, col_spacing_m)
    positions, normals = generate_surface_viewpoints(target_mesh, spacing_m)
    camera_positions = positions + normals * wd_m

    # Filter bottom-facing viewpoints
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
            print(f"  Filtered {n_removed} bottom-facing viewpoints (within {params.bottom_angle}° from down)")
            print(f"  Remaining: {len(positions)} viewpoints")
        else:
            print(f"  No bottom-facing viewpoints to filter")

    # Filter inner-skin viewpoints → 바깥 껍데기만 (hollow parts: 안쪽 면 viewpoint 가 공동 안에 생김)
    if params.filter_interior:
        keep = filter_interior_viewpoints(
            target_mesh, positions, normals,
            hull_align_min=params.interior_hull_align_min)
        if (~keep).any():
            positions = positions[keep]
            normals = normals[keep]
            camera_positions = camera_positions[keep]

    # Path length (before clustering) — PCA 무관 NN baseline
    original_path_length_mm = _nn_path_length(camera_positions) * 1000.0
    print(f"  Total path length: {original_path_length_mm:.1f} mm")
    print()

    return {
        'positions': positions, 'normals': normals, 'camera_positions': camera_positions,
        'row_spacing_m': row_spacing_m, 'col_spacing_m': col_spacing_m,
        'original_path_length_mm': original_path_length_mm,
    }


def generate_viewpoints_core(target_mesh, params: ViewpointGenParams) -> ViewpointResult:
    """표면 샘플링 → 클러스터링 → 경로 순서 최적화 (단일 method). 파일 IO 없음.

    Phase B(viser 실시간 재생성)가 호출할 import 시임.
    """
    surface = prepare_viewpoints(target_mesh, params)
    adjacency = None
    if params.build_delaunay:
        print("Building local tangent Delaunay adjacency...")
        adjacency = build_local_delaunay_adjacency(
            surface['camera_positions'], surface['normals'],
            k_neighbors=params.delaunay_neighbors,
            distance_factor=params.delaunay_distance_factor,
            max_normal_angle_deg=params.delaunay_max_normal_angle_deg,
        )
        ds = adjacency['stats']
        print(
            f"  Delaunay: {ds['num_edges']} edges, {ds['num_components']} components, "
            f"{ds['num_isolated']} isolated, degree={ds['min_degree']}-"
            f"{ds['max_degree']} (median {ds['median_degree']:.1f}), "
            f"edge median/max={ds['median_edge_length_mm']:.1f}/"
            f"{ds['max_edge_length_mm']:.1f} mm"
        )
        if ds['num_isolated'] > 0:
            print("  WARNING: Delaunay graph has isolated viewpoints; future hard-constrained "
                  "routing will require parameter adjustment or explicit bridge edges.")

    common = dict(
        positions=surface['positions'], normals=surface['normals'],
        camera_positions=surface['camera_positions'], target_mesh=target_mesh,
        row_spacing_m=surface['row_spacing_m'], col_spacing_m=surface['col_spacing_m'],
        original_path_length_mm=surface['original_path_length_mm'],
        ordering_mode=params.ordering_mode,
        adjacency_edges=adjacency['edges'] if adjacency is not None else None,
    )

    method = params.cluster_method
    eps_mm = params.eps_mm if params.eps_mm else params.fov_width_mm

    if method.startswith('delaunay+') and adjacency is None:
        raise ValueError(
            f"cluster_method={method} needs the Delaunay adjacency graph as its stage-1 "
            "grouping, but build_delaunay is off (--no-delaunay)."
        )

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
    elif method == 'delaunay+dbscan':
        nw_str = f", normal_weight: {params.normal_weight}" if params.normal_weight > 0 else ""
        print(f"Clustering (Delaunay+DBSCAN, eps: {eps_mm:.0f} mm, "
              f"min_samples: {params.min_samples}{nw_str})...")
        label = f"delaunay+dbscan eps={eps_mm:.0f}mm"
        result = cluster_and_order(
            label, 'delaunay+dbscan', **common,
            eps_m=eps_mm / 1000.0, min_samples=params.min_samples,
            normal_weight=params.normal_weight,
        )
    elif method == 'delaunay+agglomerative':
        knob = (f"max_span={params.max_span_mm:.0f}mm" if params.max_span_mm
                else f"ts={params.target_size}")
        print(f"Clustering (Delaunay+Agglomerative, {knob})...")
        label = f"delaunay+agglomerative {knob}"
        result = cluster_and_order(
            label, 'delaunay+agglomerative', **common,
            target_size=params.target_size, normal_weight=params.normal_weight,
            max_span_mm=params.max_span_mm,
        )
    else:
        raise ValueError(f"Unknown cluster_method: {method}")

    cluster_meta = {
        'clustering_method': method,
        'num_clusters': result['num_clusters'],
        'clustered_path_length_mm': result['path_length_mm'],
        'original_path_length_mm': surface['original_path_length_mm'],
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
    elif method == 'delaunay+dbscan':
        cluster_meta['dbscan_eps_mm'] = eps_mm
        cluster_meta['dbscan_min_samples'] = params.min_samples
        cluster_meta['dbscan_normal_weight'] = params.normal_weight
    elif method == 'delaunay+agglomerative':
        cluster_meta['normal_weight'] = params.normal_weight
        if params.max_span_mm:
            cluster_meta['max_span_mm'] = params.max_span_mm
        else:
            cluster_meta['target_size'] = params.target_size

    return ViewpointResult(
        positions=surface['positions'], normals=surface['normals'],
        camera_positions=surface['camera_positions'],
        path_order=result['path_order'],
        cluster_id=result['cluster_ids'], cluster_order=result['cluster_order'],
        coacd_parts=result.get('coacd_parts'), coacd_ids=result.get('coacd_ids'),
        row_spacing_m=surface['row_spacing_m'], col_spacing_m=surface['col_spacing_m'],
        original_path_length_mm=surface['original_path_length_mm'],
        clustered_path_length_mm=result['path_length_mm'],
        num_clusters=result['num_clusters'],
        cluster_meta=cluster_meta, adjacency=adjacency, method=method, label=label,
    )


# ============================================================================
# Main
# ============================================================================
