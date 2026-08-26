"""Importable viewpoint generation pipeline.

표면 샘플링과 Delaunay 인접 그래프까지가 이 모듈의 전부다. 클러스터링과 방문 순서
(clustering.py / ordering.py)는 plan_trajectory 시절의 계획 단계였고, GLNS 로 대체되며
2026-08-26 에 제거했다 — GLNS 는 positions/normals/edges/WD 만 읽고 방문 순서와 IK
자세를 함께 푼다.
"""

from __future__ import annotations

import numpy as np

from common import config
from common.math_utils import quaternion_to_rotation_matrix
from .adjacency import build_local_delaunay_adjacency
from .models import ViewpointGenParams, ViewpointResult
from .sampling import (
    _nn_path_length,
    filter_interior_viewpoints,
    generate_surface_viewpoints,
)


def prepare_viewpoints(target_mesh, params: ViewpointGenParams):
    """표면 뷰포인트 생성 + bottom/interior 필터 (클러스터링 전 단계).

    샘플링은 메시 표면 직접 FPS 하나뿐이다. 예전에는 PCA 평면에 격자를 깔고
    ``closest_point`` 로 표면에 투영하는 grid 모드가 있었지만, 평면 투영이라 곡면·측벽을
    놓치고 속 빈 물체에서는 안쪽 면이 더 가까워 지붕을 통째로 잃었다. 저장된 h5 는 전부
    surface 로 만들어졌고 grid 산출물은 하나도 없어 2026-08-26 에 제거했다.

    Returns: dict — positions, normals, camera_positions,
        row_spacing_m, col_spacing_m, original_path_length_mm

    방문 순서는 만들지 않는다 — GLNS 가 IK 자세와 함께 푼다. row/col spacing 은
    h5 metadata 로 기록되므로 유도값이라도 반환한다.
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
    """표면 샘플링 → Delaunay 인접 그래프. 파일 IO 없음.

    방문 순서는 만들지 않는다. 예전에는 여기서 클러스터링(stage1+sub) → 클러스터 내부
    lawnmower → 클러스터 GTSP 로 ``path_order`` 를 만들었지만, 그 순서를 소비하던
    plan_trajectory 는 GLNS 로 대체되며 사라졌다. GLNS 는 positions/normals/edges/WD 만
    읽고 순서와 IK 자세를 함께 푼다 — 그래서 여기서 순서를 정하는 것은 무의미할 뿐 아니라,
    저장해두면 어느 쪽이 진짜 순서인지 두 답이 생긴다.
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
            print("  WARNING: Delaunay graph has isolated viewpoints; GLNS will drop them "
                  "from the constraint graph - loosen delaunay_max_normal_angle_deg or "
                  "distance_factor.")

    return ViewpointResult(
        positions=surface['positions'], normals=surface['normals'],
        camera_positions=surface['camera_positions'],
        row_spacing_m=surface['row_spacing_m'], col_spacing_m=surface['col_spacing_m'],
        nn_path_length_mm=surface['original_path_length_mm'],
        adjacency=adjacency,
    )
