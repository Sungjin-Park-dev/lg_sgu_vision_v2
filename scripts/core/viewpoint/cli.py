#!/usr/bin/env python3
"""CLI for viewpoint generation and clustering."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS_ROOT))

from common import config  # noqa: E402
from core.viewpoint import (  # noqa: E402
    DEFAULT_DELAUNAY_DISTANCE_FACTOR,
    DEFAULT_DELAUNAY_MAX_NORMAL_ANGLE_DEG,
    DEFAULT_DELAUNAY_NEIGHBORS,
    ViewpointGenParams,
    build_local_delaunay_adjacency,
    cluster_and_order,
    cluster_coacd,
    compute_path_length,
    generate_viewpoints_core,
    load_meshes,
    prepare_grid,
    save_viewpoints_hdf5,
)
from core.viewpoint.visualization import visualize_clusters_html  # noqa: E402

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='뷰포인트 생성 및 클러스터링 기반 경로 순서 최적화',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 기본: FOV 기반 자동 간격
  uv run scripts/core/viewpoint/cli.py --object sample

  # 재질 필터
  uv run scripts/core/viewpoint/cli.py --object sample --material-rgb "0,255,0"

  # DBSCAN 클러스터링
  uv run scripts/core/viewpoint/cli.py --object sample --material-rgb "0,255,0" --cluster-method dbscan

  # CoACD 클러스터링 (메시 convex decomposition)
  uv run scripts/core/viewpoint/cli.py --object sample --cluster-method coacd --coacd-threshold 0.05
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

    # --- Viewpoint adjacency (future GLNS constraint graph) ---
    parser.add_argument('--no-delaunay', action='store_true',
                        help='로컬 표면 Delaunay 인접 그래프 생성/저장을 비활성화')
    parser.add_argument('--delaunay-neighbors', type=int, default=DEFAULT_DELAUNAY_NEIGHBORS,
                        help=f'로컬 Delaunay kNN 크기 (기본: {DEFAULT_DELAUNAY_NEIGHBORS})')
    parser.add_argument('--delaunay-distance-factor', type=float,
                        default=DEFAULT_DELAUNAY_DISTANCE_FACTOR,
                        help='edge 최대 길이 / 로컬 spacing 비율 '
                             f'(기본: {DEFAULT_DELAUNAY_DISTANCE_FACTOR})')
    parser.add_argument('--delaunay-max-normal-angle', type=float,
                        default=DEFAULT_DELAUNAY_MAX_NORMAL_ANGLE_DEG,
                        help='인접 edge의 최대 법선 차이 deg '
                             f'(기본: {DEFAULT_DELAUNAY_MAX_NORMAL_ANGLE_DEG:.0f})')

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

    if args.delaunay_neighbors < 3:
        parser.error("--delaunay-neighbors must be >= 3")
    if args.delaunay_distance_factor <= 0.0:
        parser.error("--delaunay-distance-factor must be > 0")
    if not 0.0 < args.delaunay_max_normal_angle <= 180.0:
        parser.error("--delaunay-max-normal-angle must be in (0, 180]")

    return args

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

    _fi = config.OBJECT_FILTER_INTERIOR.get(args.object)  # hollow 물체만 opt-in (studio 와 동일)
    params = ViewpointGenParams(
        material_rgb=args.material_rgb,
        color_tolerance=args.color_tolerance,
        row_spacing_mm=args.row_spacing,
        col_spacing_mm=args.col_spacing,
        filter_bottom=not args.no_filter_bottom,
        bottom_angle=args.bottom_angle,
        filter_interior=_fi is not None,
        interior_hull_align_min=(_fi or {}).get("hull_align_min", 0.3),
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
        build_delaunay=not args.no_delaunay,
        delaunay_neighbors=args.delaunay_neighbors,
        delaunay_distance_factor=args.delaunay_distance_factor,
        delaunay_max_normal_angle_deg=args.delaunay_max_normal_angle,
    )

    method = args.cluster_method
    eps_mm = args.eps if args.eps else config.CAMERA_FOV_WIDTH_MM

    # ------------------------------------------------------------------
    # Compare mode: 파라미터 스윕 → 드롭다운 HTML
    # ------------------------------------------------------------------
    if args.compare:
        grid = prepare_grid(target_mesh, params)
        adjacency = None
        if params.build_delaunay:
            adjacency = build_local_delaunay_adjacency(
                grid['camera_positions'], grid['normals'],
                k_neighbors=params.delaunay_neighbors,
                distance_factor=params.delaunay_distance_factor,
                max_normal_angle_deg=params.delaunay_max_normal_angle_deg,
            )
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
                adjacency_edges=adjacency['edges'] if adjacency is not None else None,
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
            adjacency=res.adjacency,
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
            adjacency_edges=res.adjacency['edges'] if res.adjacency is not None else None,
        )

    return 0


if __name__ == '__main__':
    sys.exit(main())
