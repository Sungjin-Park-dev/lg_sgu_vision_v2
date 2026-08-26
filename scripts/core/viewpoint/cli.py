#!/usr/bin/env python3
"""CLI for surface viewpoint generation + Delaunay adjacency."""

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

from common import config, scene_config  # noqa: E402
from core.viewpoint import (  # noqa: E402
    DEFAULT_DELAUNAY_DISTANCE_FACTOR,
    DEFAULT_DELAUNAY_MAX_NORMAL_ANGLE_DEG,
    DEFAULT_DELAUNAY_NEIGHBORS,
    ViewpointGenParams,
    generate_viewpoints_core,
    load_meshes,
    save_viewpoints_hdf5,
)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='표면 뷰포인트 생성 + Delaunay 인접 그래프 (방문 순서는 GLNS 가 정한다)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 기본: FOV 기반 자동 간격
  uv run scripts/core/viewpoint/cli.py --object sample

  # 재질 필터
  uv run scripts/core/viewpoint/cli.py --object sample --material-rgb "0,255,0"

  # Delaunay 그래프 노브
  uv run scripts/core/viewpoint/cli.py --object sample --delaunay-max-normal-angle 90
        """,
    )

    # --- Viewpoint generation ---
    parser.add_argument('--object', type=str, required=True, help='오브젝트 이름')
    scene_config.add_cli_argument(parser)
    parser.add_argument('--material-rgb', type=str, default=None,
                        help='Target material RGB color as "R,G,B" (e.g., "0,255,0")')
    parser.add_argument('--color-tolerance', type=float, default=5.0,
                        help='RGB color matching tolerance (default: 5.0)')
    parser.add_argument('--row-spacing', type=float, default=None,
                        help='Row spacing in mm (default: FOV height * (1-overlap))')
    parser.add_argument('--col-spacing', type=float, default=None,
                        help='Column spacing in mm (default: FOV width * (1-overlap))')

    # --- Camera spec (h5 metadata/camera_spec 로 저장 → IK/궤적/GLNS/Isaac 이 이 값을 쓴다) ---
    parser.add_argument('--fov-width', type=float, default=None,
                        help=f'FOV width in mm (default: {config.CAMERA_FOV_WIDTH_MM:.0f})')
    parser.add_argument('--fov-height', type=float, default=None,
                        help=f'FOV height in mm (default: {config.CAMERA_FOV_HEIGHT_MM:.0f})')
    parser.add_argument('--overlap', type=float, default=None,
                        help='FOV overlap ratio 0~1 '
                             f'(default: {config.CAMERA_OVERLAP_RATIO}). --row/col-spacing 이 우선')
    parser.add_argument('--working-distance', type=float, default=None,
                        help='Working distance in mm — 카메라 끝(렌즈 배럴 앞면)에서 검사면까지 '
                             f'(default: {config.CAMERA_WORKING_DISTANCE_MM:g}, '
                             f'최소 {config.CAMERA_MIN_WORKING_DISTANCE_MM:.1f} 초과)')
    parser.add_argument('--no-filter-bottom', action='store_true', default=False,
                        help='Disable bottom-face filtering')
    parser.add_argument('--bottom-angle', type=float, default=80.0,
                        help='Bottom filter angle in degrees (default: 80)')

    # --- Clustering ---
    # --- Sampling / Ordering ---
    # 샘플링은 표면 FPS 하나뿐이다 — grid(PCA 평면 투영) 모드는 제거했다.
    parser.add_argument('--surface-spacing', type=float, default=None,
                        help='FPS 목표 표면 간격 mm (기본: FOV 작은 축)')
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

    # --- Output ---
    # solve.py --output / verify.py --output-dir 와 짝을 맞춘다. 이게 없으면 세 단계 중
    # viewpoint 만 리다이렉션이 안 돼서, data/ 를 건드리지 않는 실행(회귀 테스트 등)이 불가능하다.
    parser.add_argument('--output', type=Path, default=None,
                        help='출력 h5 경로 '
                             '(기본: data/{object}/viewpoint/{N}/viewpoints_{method}.h5)')

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

    for flag, value in (("--fov-width", args.fov_width), ("--fov-height", args.fov_height)):
        if value is not None and value <= 0.0:
            parser.error(f"{flag} must be > 0")
    if args.overlap is not None and not 0.0 <= args.overlap < 1.0:
        parser.error("--overlap must be in [0, 1)")
    if args.working_distance is not None:
        problem = config.working_distance_error(args.working_distance)
        if problem:
            parser.error(f"--working-distance: {problem}")

    return args

def main():
    args = parse_arguments()

    # 물체별 배치를 반영(rotation 은 bottom-filter 판정에 사용 — line ~1776).
    # 씬을 먼저 반영한다 — 물체 배치(object_placements)가 이제 씬 소유다.
    scene_config.apply_cli(args, config)
    if config.apply_object_placement(args.object):
        print(f"  Per-object placement '{args.object}': quat={config.TARGET_OBJECT['rotation']}")

    input_path = str(config.get_mesh_path(args.object, mesh_type="source"))

    # 재질 필터를 안 주면 물체별 기본값(config)에서 채운다 — 안 그러면 sample 이 조용히
    # 161개(전체 메시)로 나온다. 정답은 초록 재질만 74개다.
    rgb_source = "지정"
    if args.material_rgb is None:
        args.material_rgb = config.OBJECT_TARGET_MATERIAL.get(args.object)
        rgb_source = "config 기본값"

    print("=" * 60)
    print("GENERATE VIEWPOINTS")
    print("=" * 60)
    print(f"Object: {args.object}")
    print(f"Input:  {input_path}")
    if args.material_rgb:
        print(f"Target RGB: {args.material_rgb}  ({rgb_source})")
    else:
        print(f"Target: entire mesh (no material filter)")
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
        working_distance_mm=args.working_distance,
        fov_width_mm=args.fov_width,
        fov_height_mm=args.fov_height,
        overlap_ratio=args.overlap,
        filter_bottom=not args.no_filter_bottom,
        bottom_angle=args.bottom_angle,
        filter_interior=_fi is not None,
        interior_hull_align_min=(_fi or {}).get("hull_align_min", 0.3),
        surface_spacing_mm=args.surface_spacing,
        build_delaunay=not args.no_delaunay,
        delaunay_neighbors=args.delaunay_neighbors,
        delaunay_distance_factor=args.delaunay_distance_factor,
        delaunay_max_normal_angle_deg=args.delaunay_max_normal_angle,
    )

    # ------------------------------------------------------------------
    # 생성 코어 호출 → 저장
    # ------------------------------------------------------------------
    res = generate_viewpoints_core(target_mesh, params)

    # 9. Save to HDF5
    if args.dry_run:
        print()
        print("[DRY RUN] HDF5 not modified.")
    else:
        # 정규 이름으로 쓴다 — resolve_viewpoint_path 가 가장 먼저 찾는 이름이라
        # 같은 폴더에 후보가 여럿일 때의 mtime 자동선택 함정이 생기지 않는다.
        output_path = str(args.output) if args.output else str(config.get_viewpoint_path(
            args.object, len(res.positions), filename="viewpoints.h5",
        ))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        print(f"Output: {output_path}")

        print("Saving to HDF5...")
        # config 가 아니라 params 에서 — 안 그러면 --working-distance 120 으로 만든 h5 가
        # 250 이라고 주장하고, 그걸 읽는 IK/궤적/Isaac 이 전부 250 으로 계획한다.
        camera_spec = params.camera_spec
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'input_mesh': str(input_path),
            'method': 'surface',
            'sampling_mode': 'surface',
            'surface_spacing_mm': params.surface_spacing_mm if params.surface_spacing_mm
                else min(res.row_spacing_m, res.col_spacing_m) * 1000.0,
            'row_spacing_mm': res.row_spacing_m * 1000.0,
            'col_spacing_mm': res.col_spacing_m * 1000.0,
            # overlap 은 카메라 스펙이 아니라 샘플링 파라미터라 camera_spec 이 아닌 여기 —
            # viewpoint_studio 의 저장 경로와 같은 키/단위(0~1)를 쓴다.
            'overlap_ratio': params.overlap_ratio,
            # 방문 순서가 없으므로 '경로 길이' 도 없다. greedy NN 베이스라인만 남긴다 —
            # 실행 순서가 아니라 밀도 감각을 주는 수치다.
            'nn_path_length_mm': res.nn_path_length_mm,
        }
        save_viewpoints_hdf5(
            res.positions, res.normals, output_path, metadata, camera_spec,
            adjacency=res.adjacency,
        )
        print()

    print("Complete!")
    print("=" * 60)

    # 시각화는 viewpoint_studio(viser)가 h5 를 직접 읽어 담당한다.

    return 0


if __name__ == '__main__':
    sys.exit(main())
