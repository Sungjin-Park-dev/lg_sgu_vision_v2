"""Importable viewpoint generation pipeline.

표면 샘플링과 Delaunay 인접 그래프까지가 이 모듈의 전부다. 클러스터링과 방문 순서
(clustering.py / ordering.py)는 plan_trajectory 시절의 계획 단계였고, GLNS 로 대체되며
2026-08-26 에 제거했다 — GLNS 는 positions/normals/edges/WD 만 읽고 방문 순서와 IK
자세를 함께 푼다.
"""

from __future__ import annotations

import numpy as np

from common import config
from . import visibility
from .adjacency import build_local_delaunay_adjacency
from .models import ViewpointGenParams, ViewpointResult
from .sampling import _nn_path_length, generate_surface_viewpoints


def spacings_m(params: ViewpointGenParams) -> tuple[float, float]:
    """(row, col) 표면 간격(m). 직접 준 값이 없으면 FOV×(1-overlap) 로 유도한다."""
    row = params.row_spacing_mm / 1000.0 if params.row_spacing_mm \
        else params.fov_height_mm / 1000.0 * (1.0 - params.overlap_ratio)
    col = params.col_spacing_mm / 1000.0 if params.col_spacing_mm \
        else params.fov_width_mm / 1000.0 * (1.0 - params.overlap_ratio)
    return row, col


def finalize_viewpoints(positions, normals, params: ViewpointGenParams,
                        occluder_mesh=None, extras=None,
                        row_spacing_m=None, col_spacing_m=None) -> dict:
    """샘플러가 낸 점에 **공통 후처리**를 적용한다: 카메라 위치 + bottom/가림 필터.

    표면 FPS 든 CAD 면 격자든 여기를 지나야 한다 — 필터가 두 곳에 있으면 샘플러를 바꿨을 때
    조용히 다른 규칙이 적용된다.

    ``extras``: 점별 부가 배열 dict(예: face_id). 필터가 점을 지울 때 **같이** 지워진다 —
    따로 관리하면 필터 하나 추가할 때마다 어긋난다.

    ``occluder_mesh``: 가시성 필터의 가림체 — **어셈블리 전체**를 넘긴다(지그 포함).
    """
    if row_spacing_m is None or col_spacing_m is None:
        row_spacing_m, col_spacing_m = spacings_m(params)
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    extras = {k: np.asarray(v) for k, v in (extras or {}).items()}
    wd_m = params.working_distance_mm / 1000.0
    camera_positions = positions + normals * wd_m

    spec = visibility.SensorSpec.from_params(params)

    if params.filter_bottom:
        keep = visibility.facing_up(normals, config.TARGET_OBJECT["rotation"],
                                    params.bottom_angle)
        n_removed = int((~keep).sum())
        if n_removed > 0:
            positions, normals = positions[keep], normals[keep]
            camera_positions = camera_positions[keep]
            extras = {k: v[keep] for k, v in extras.items()}
            print(f"  Filtered {n_removed} bottom-facing viewpoints "
                  f"(within {params.bottom_angle}° from down)")
            print(f"  Remaining: {len(positions)} viewpoints")
        else:
            print("  No bottom-facing viewpoints to filter")

    if params.filter_occluded and occluder_mesh is not None and len(positions):
        # 자기 표면점을 볼 수 있는가 = 후보 적격성. 판정은 visibility 한 곳에서만 한다.
        keep = visibility.self_visible(positions, normals, occluder_mesh, spec)
        blocked = int((~keep).sum())
        print(f"  Occlusion filter: removed {blocked}/{len(positions)} viewpoints the camera "
              f"cannot see (blocked by the object or fixture); {int(keep.sum())} remain"
              if blocked else "  Occlusion filter: all viewpoints visible")
        if (~keep).any():
            positions, normals = positions[keep], normals[keep]
            camera_positions = camera_positions[keep]
            extras = {k: v[keep] for k, v in extras.items()}

    original_path_length_mm = _nn_path_length(camera_positions) * 1000.0
    print(f"  Total path length: {original_path_length_mm:.1f} mm")
    print()
    return {
        'positions': positions.astype(np.float32), 'normals': normals.astype(np.float32),
        'camera_positions': camera_positions.astype(np.float32),
        'row_spacing_m': row_spacing_m, 'col_spacing_m': col_spacing_m,
        'original_path_length_mm': original_path_length_mm,
        'extras': extras,
    }


def subset_viewpoints(surface: dict, indices) -> dict:
    """``finalize_viewpoints`` 결과에서 일부만 남긴다 — 선택 단계가 쓴다.

    positions/normals/camera_positions 와 ``extras`` 를 **한꺼번에** 자른다. 따로 자르면
    face_id 가 점과 어긋나고, 그 어긋남은 커버리지·h5 추적성에서야 드러난다.
    경로 길이는 남은 점 기준으로 다시 잰다(옛 값을 들고 있으면 거짓말이 된다).
    """
    indices = np.asarray(indices, dtype=int)
    out = dict(surface)
    for key in ("positions", "normals", "camera_positions"):
        out[key] = np.asarray(surface[key])[indices]
    out["extras"] = {k: np.asarray(v)[indices] for k, v in surface.get("extras", {}).items()}
    out["original_path_length_mm"] = _nn_path_length(
        np.asarray(out["camera_positions"], dtype=np.float64)) * 1000.0
    return out


def prepare_viewpoints(target_mesh, params: ViewpointGenParams, occluder_mesh=None):
    """표면 FPS 샘플링 + 공통 후처리(bottom/가림 필터).

    샘플링은 메시 표면 직접 FPS 하나뿐이다. 예전에는 PCA 평면에 격자를 깔고
    ``closest_point`` 로 표면에 투영하는 grid 모드가 있었지만, 평면 투영이라 곡면·측벽을
    놓치고 속 빈 물체에서는 안쪽 면이 더 가까워 지붕을 통째로 잃었다. 저장된 h5 는 전부
    surface 로 만들어졌고 grid 산출물은 하나도 없어 2026-08-26 에 제거했다.
    (CAD 면 위 격자는 ``core/viewpoint/brep.py`` 로 따로 산다 — 그쪽은 삼각형이 아니라
     B-rep 곡면을 쓴다.)

    Returns: dict — positions, normals, camera_positions,
        row_spacing_m, col_spacing_m, original_path_length_mm

    방문 순서는 만들지 않는다 — GLNS 가 IK 자세와 함께 푼다.
    """
    row_spacing_m, col_spacing_m = spacings_m(params)
    print(f"  Row spacing (axis1): {row_spacing_m * 1000:.1f} mm")
    print(f"  Col spacing (axis2): {col_spacing_m * 1000:.1f} mm")
    print(f"  Working distance:    {params.working_distance_mm:.1f} mm "
          f"(FOV {params.fov_width_mm:.0f}×{params.fov_height_mm:.0f} mm)")
    print()

    spacing_m = (params.surface_spacing_mm / 1000.0) if params.surface_spacing_mm \
        else min(row_spacing_m, col_spacing_m)
    positions, normals = generate_surface_viewpoints(target_mesh, spacing_m)
    return finalize_viewpoints(
        positions, normals, params,
        occluder_mesh=occluder_mesh if occluder_mesh is not None else target_mesh,
        row_spacing_m=row_spacing_m, col_spacing_m=col_spacing_m)


def generate_viewpoints_core(target_mesh, params: ViewpointGenParams,
                             occluder_mesh=None) -> ViewpointResult:
    """표면 샘플링 → Delaunay 인접 그래프. 파일 IO 없음.

    방문 순서는 만들지 않는다. 예전에는 여기서 클러스터링(stage1+sub) → 클러스터 내부
    lawnmower → 클러스터 GTSP 로 ``path_order`` 를 만들었지만, 그 순서를 소비하던
    plan_trajectory 는 GLNS 로 대체되며 사라졌다. GLNS 는 positions/normals/edges/WD 만
    읽고 순서와 IK 자세를 함께 푼다 — 그래서 여기서 순서를 정하는 것은 무의미할 뿐 아니라,
    저장해두면 어느 쪽이 진짜 순서인지 두 답이 생긴다.
    """
    surface = prepare_viewpoints(target_mesh, params, occluder_mesh=occluder_mesh)
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
