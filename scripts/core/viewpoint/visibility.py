#!/usr/bin/env python3
"""검사 가능성 판정 — "이 카메라가 이 표면점을 검사할 수 있는가" 의 **단일 진실원**.

같은 질문이 파이프라인의 세 자리에서 나온다:

  1. **후보 적격성** — 생성한 viewpoint 가 자기 표면점을 볼 수 있나 (가림 필터)
  2. **커버리지 평가** — 최종 집합이 표면의 어느 셀을 덮었나
  3. **선택(set cover)** — 각 후보가 덮는 셀 집합은 무엇인가

셋이 각자 판정을 구현하면 조용히 어긋난다. 실제로 그랬다 — 커버리지 쪽만 "가장 가까운
viewpoint 3개" 를 후보로 봐서, 유효 FOV 가 1mm 도 안 되는 필렛 면 주변에서 정작 그 셀을
덮어주는 큰 FOV viewpoint 가 밀려나 커버리지가 91.6% 로 잘못 나왔다(반경 질의로 고친 뒤
98.3%). 그래서 판정은 여기 한 곳에만 둔다.

판정 조건은 넷이고, 싼 것부터 본다:

  * **FOV**      — 표면점이 촬영 프레임 안인가 (viewpoint 의 표면점에서의 거리)
  * **입사각**    — 표면이 너무 기울어 보이지 않는가 (0 = 제한 없음)
  * **초점(DOF)** — 카메라에서의 거리가 심도 안인가 (0 = 제한 없음)
  * **가림**      — 카메라와 표면점 사이에 다른 면이 없는가 (광선)

카메라가 물리적으로 들어갈 자리가 있는지는 여기서 보지 않는다 — 그건 하류의 cuRobo 충돌
검사가 로봇+카메라 모델로 판정한다.

단위 규약: **기하는 미터, 센서 스펙은 mm**. 파이프라인 전체가 미터를 쓰고 카메라 스펙은
mm 로 적히기 때문이다. 변환은 이 모듈 안에서만 일어난다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# 셀 하나가 시험해 볼 후보 viewpoint 수 상한(가까운 순).
DEFAULT_CANDIDATE_CAP = 12


@dataclass(frozen=True)
class ViewFrames:
    """viewpoint 마다의 **촬영 사각형** — 두 축(단위벡터)과 축별 프레임 폭(mm).

    센서는 원이 아니라 사각형이다. 내접원으로 근사하면 간격 = FOV 로 깔았을 때 대각선에
    틈이 있는 것으로 계산된다(원으로 정사각 격자를 덮으려면 반경 ≥ 0.707×간격 이어야 하고,
    그래서 겹침 29.3% 미만이 전부 구멍으로 보였다). 축을 알면 그 왜곡이 사라지고, 덤으로
    비등방 FOV(50×40)도 축별로 쓸 수 있다.

    축은 CAD 면의 (u,v) 접선에서 온다 — 격자와 프레임이 같은 방향을 보므로 타일링이 정확하다.
    """

    axis_u: np.ndarray          # (N,3) 단위벡터
    axis_v: np.ndarray          # (N,3) 단위벡터, axis_u 와 직교
    fov_u_mm: np.ndarray        # (N,)
    fov_v_mm: np.ndarray        # (N,)

    def __len__(self) -> int:
        return len(self.axis_u)

    def __getitem__(self, index) -> "ViewFrames":
        return ViewFrames(self.axis_u[index], self.axis_v[index],
                          np.atleast_1d(self.fov_u_mm[index]),
                          np.atleast_1d(self.fov_v_mm[index]))

    @property
    def half_diagonal_mm(self) -> np.ndarray:
        """프레임을 감싸는 원의 반경 — 후보 탐색 반경으로 쓴다(놓치지 않으려면 외접원)."""
        return np.hypot(self.fov_u_mm, self.fov_v_mm) / 2.0

    @classmethod
    def from_extras(cls, extras: dict) -> Optional["ViewFrames"]:
        """샘플러가 실어 보낸 배열에서 만든다. 없으면 None(원 근사로 떨어진다)."""
        keys = ("frame_u", "frame_v", "fov_u_mm", "fov_v_mm")
        if not all(k in extras and len(np.asarray(extras[k])) for k in keys):
            return None
        return cls(np.asarray(extras["frame_u"], dtype=np.float64).reshape(-1, 3),
                   np.asarray(extras["frame_v"], dtype=np.float64).reshape(-1, 3),
                   np.asarray(extras["fov_u_mm"], dtype=np.float64).reshape(-1),
                   np.asarray(extras["fov_v_mm"], dtype=np.float64).reshape(-1))


@dataclass(frozen=True)
class SensorSpec:
    """판정에 필요한 센서 스펙. 0 은 '그 제한을 걸지 않음' 을 뜻한다.

    ``max_incidence_deg`` 와 ``depth_of_field_mm`` 이 0 이면 지금까지의 암묵적 가정
    (프레임 안에서 표면이 아무리 기울어도, 거리가 얼마든 검사 가능)이 그대로 유지된다.
    """

    working_distance_mm: float
    max_incidence_deg: float = 0.0
    depth_of_field_mm: float = 0.0
    occlusion_tolerance_mm: float = 1.0

    @property
    def working_distance_m(self) -> float:
        return float(self.working_distance_mm) / 1000.0

    @classmethod
    def from_params(cls, params) -> "SensorSpec":
        """``ViewpointGenParams`` 에서 뽑아온다 — 스펙의 출처를 한 줄로 묶어 둔다."""
        return cls(
            working_distance_mm=float(params.working_distance_mm),
            max_incidence_deg=float(params.max_incidence_deg or 0.0),
            depth_of_field_mm=float(params.depth_of_field_mm or 0.0),
            occlusion_tolerance_mm=float(params.occlusion_tolerance_mm),
        )


def as_rotation_matrix(rotation) -> Optional[np.ndarray]:
    """(3,3) 행렬 · (4,) 쿼터니언 [w,x,y,z] · None 을 모두 받는다."""
    if rotation is None:
        return None
    value = np.asarray(rotation, dtype=np.float64)
    if value.shape == (3, 3):
        return value
    if value.shape == (4,):
        from common.math_utils import quaternion_to_rotation_matrix
        return quaternion_to_rotation_matrix(value)
    raise ValueError(f"rotation must be (3,3) or (4,) quaternion, got {value.shape}")


def facing_up(normals, rotation=None, bottom_angle_deg: float = 0.0) -> np.ndarray:
    """월드 −z 에서 ``bottom_angle_deg`` 안쪽을 향하지 **않는** 점 = True.

    로봇이 물체 아래에서 올려다볼 수 없다는 도메인 제약이다. 이 규칙이 사는 유일한 곳 —
    예전에는 필터(pipeline)와 커버리지(brep) 두 곳에 같은 식이 따로 있었다.
    """
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    if bottom_angle_deg <= 0.0 or len(normals) == 0:
        return np.ones(len(normals), dtype=bool)
    matrix = as_rotation_matrix(rotation)
    world = normals if matrix is None else (matrix @ normals.T).T
    return (-world[:, 2]) < np.cos(np.deg2rad(bottom_angle_deg))


def _blocked(occluder, origins: np.ndarray, directions: np.ndarray,
             lengths: np.ndarray, tolerance_m: float) -> np.ndarray:
    """광선이 목표점보다 **앞에서** 무언가에 맞으면 True. 한 번에 묶어 쏜다."""
    if occluder is None or len(origins) == 0:
        return np.zeros(len(origins), dtype=bool)
    _tri, rays, locations = occluder.ray.intersects_id(
        ray_origins=origins, ray_directions=directions,
        return_locations=True, multiple_hits=False)
    nearest = np.full(len(origins), np.inf)
    if len(rays):
        rays = np.asarray(rays)
        hit = np.linalg.norm(np.asarray(locations) - origins[rays], axis=1)
        np.minimum.at(nearest, rays, hit)
    return nearest < (lengths - tolerance_m)


def sees(view_positions, view_normals, targets, target_normals,
         occluder, spec: SensorSpec, fov_mm=None,
         frames: Optional[ViewFrames] = None) -> np.ndarray:
    """**쌍 단위** 판정 — i번째 viewpoint 가 i번째 표면점을 검사 가능한 조건으로 보는가.

    카메라는 ``view_positions + view_normals × WD`` 에 있다(광축은 그 표면점의 법선).

    Args:
        view_positions/view_normals: (N,3) viewpoint 의 표면점과 법선 (미터)
        targets/target_normals: (N,3) 판정할 표면점과 그 법선 (미터)
        frames: (N,) 촬영 사각형. 주면 두 축에 투영해 판정한다(정확). 없으면 ``fov_mm``
            내접원으로 근사한다(FPS 처럼 프레임 방향이 없는 샘플러용).
        fov_mm: 프레임 폭(mm). 스칼라 또는 (N,) 배열. None 이면 FOV 검사를 건너뛴다
            (자기 점을 보는 경우처럼 거리가 0 인 상황).
    Returns:
        (N,) bool — 넷 다 통과했는가.
    """
    view_positions = np.asarray(view_positions, dtype=np.float64).reshape(-1, 3)
    view_normals = np.asarray(view_normals, dtype=np.float64).reshape(-1, 3)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1, 3)
    target_normals = np.asarray(target_normals, dtype=np.float64).reshape(-1, 3)
    n = len(targets)
    if n == 0:
        return np.zeros(0, dtype=bool)

    ok = np.ones(n, dtype=bool)

    # ① FOV — 표면점이 그 촬영의 프레임 안인가.
    if frames is not None:
        # 사각 프레임: 표면점을 두 축에 투영한다. 카메라 광축이 법선이라 이 투영이 곧
        # 이미지 평면 좌표다(작은 FOV 에서 원근 보정은 무시할 만하다).
        delta = targets - view_positions
        ok &= np.abs(np.einsum("ij,ij->i", delta, frames.axis_u)) \
            <= frames.fov_u_mm / 2.0 / 1000.0
        ok &= np.abs(np.einsum("ij,ij->i", delta, frames.axis_v)) \
            <= frames.fov_v_mm / 2.0 / 1000.0
    elif fov_mm is not None:
        radius_m = np.asarray(fov_mm, dtype=np.float64) / 2.0 / 1000.0
        ok &= np.linalg.norm(targets - view_positions, axis=1) <= radius_m

    cameras = view_positions + view_normals * spec.working_distance_m
    ray = targets - cameras
    length = np.linalg.norm(ray, axis=1)
    unit = ray / np.maximum(length, 1e-12)[:, None]

    # ② 입사각 — 표면이 시선에 대해 얼마나 기울어 보이는가.
    if spec.max_incidence_deg > 0.0 and ok.any():
        cos_incidence = np.einsum("ij,ij->i", -unit, target_normals)
        incidence = np.degrees(np.arccos(np.clip(cos_incidence, -1.0, 1.0)))
        ok &= incidence <= spec.max_incidence_deg

    # ③ 초점 — 카메라에서의 거리가 심도 안인가.
    if spec.depth_of_field_mm > 0.0 and ok.any():
        ok &= np.abs(length * 1000.0 - spec.working_distance_mm) <= spec.depth_of_field_mm

    # ④ 가림 — 가장 비싸므로 마지막에, 살아남은 것만.
    if occluder is not None and ok.any():
        idx = np.where(ok)[0]
        ok[idx] &= ~_blocked(occluder, cameras[idx], unit[idx], length[idx],
                             spec.occlusion_tolerance_mm / 1000.0)
    return ok


def self_visible(positions, normals, occluder, spec: SensorSpec) -> np.ndarray:
    """viewpoint 가 **자기 표면점**을 볼 수 있는가 — ``sees`` 의 특수 케이스.

    자기 점이므로 입사각 0°, 거리는 정확히 WD 라 ①②③ 은 자동으로 통과하고 **가림만** 남는다.
    법선 기반 필터(bottom)로는 못 잡는 '파인 곳·지그 뒤' 가 여기서 걸러진다.
    """
    return sees(positions, normals, positions, normals, occluder, spec, fov_mm=None)


def search_radius_m(fov_mm=None, frames: Optional[ViewFrames] = None) -> float:
    """후보 탐색 반경(m). 사각 프레임이면 **외접원**이라야 모서리 쪽을 안 놓친다."""
    if frames is not None and len(frames):
        return float(np.max(frames.half_diagonal_mm)) / 1000.0
    fov = np.asarray(fov_mm, dtype=np.float64)
    return float(fov.max() if fov.ndim else fov) / 2.0 / 1000.0


def pair_candidates(targets, view_positions, radius_m: float,
                    cap: int = DEFAULT_CANDIDATE_CAP) -> Tuple[np.ndarray, np.ndarray]:
    """표면점마다 시험해 볼 viewpoint 후보를 가까운 순으로 (idx, dist) 로 돌려준다.

    후보는 **반경**으로 고른다. 'k개 최근접' 은 viewpoint 밀도가 들쭉날쭉할 때 무너진다 —
    유효 FOV 가 1mm 도 안 되는 필렛 면은 점이 수백 개 몰려서 그 k 자리를 다 차지하고, 정작
    그 셀을 덮어주는 큰 FOV viewpoint 가 후보에서 밀려난다.

    Returns:
        idx: (T, k) int64 — 후보 viewpoint 인덱스. 빈 자리는 -1
        dist: (T, k) float — 표면점까지의 거리(미터). 빈 자리는 inf
    """
    from scipy.spatial import cKDTree

    targets = np.asarray(targets, dtype=np.float64).reshape(-1, 3)
    view_positions = np.asarray(view_positions, dtype=np.float64).reshape(-1, 3)
    if len(view_positions) == 0 or len(targets) == 0:
        return (np.full((len(targets), 1), -1, dtype=np.int64),
                np.full((len(targets), 1), np.inf))

    tree = cKDTree(view_positions)
    neighbours = tree.query_ball_point(targets, radius_m)
    width = max(1, min(int(cap), max((len(c) for c in neighbours), default=1)))
    idx = np.full((len(targets), width), -1, dtype=np.int64)
    dist = np.full((len(targets), width), np.inf)
    for row, candidates in enumerate(neighbours):
        if not candidates:
            continue
        candidates = np.asarray(candidates)
        d = np.linalg.norm(view_positions[candidates] - targets[row], axis=1)
        order = np.argsort(d)[:width]
        idx[row, :len(order)] = candidates[order]
        dist[row, :len(order)] = d[order]
    return idx, dist


def covered_by_any(targets, target_normals, view_positions, view_normals,
                   point_fov_mm, occluder, spec: SensorSpec,
                   mask=None, cap: int = DEFAULT_CANDIDATE_CAP,
                   frames: Optional[ViewFrames] = None) -> np.ndarray:
    """표면점마다 '하나라도 이 점을 검사할 수 있는 viewpoint 가 있는가'.

    커버리지 평가가 쓰는 형태다. 가까운 후보부터 라운드로 시험하고, 라운드마다 광선을 한 번에
    묶어 쏜다. ``mask`` 가 False 인 점은 아예 보지 않는다(예: 아래를 향해 검사 대상이 아닌 셀).
    """
    targets = np.asarray(targets, dtype=np.float64).reshape(-1, 3)
    n = len(targets)
    covered = np.zeros(n, dtype=bool)
    if n == 0 or len(view_positions) == 0:
        return covered
    point_fov_mm = np.asarray(point_fov_mm, dtype=np.float64).reshape(-1)
    todo_mask = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)

    idx, dist = pair_candidates(
        targets, view_positions, search_radius_m(point_fov_mm, frames), cap=cap)
    for rank in range(idx.shape[1]):
        pending = np.where(todo_mask & ~covered)[0]
        if not len(pending):
            break
        view = idx[pending, rank]
        alive = view >= 0
        pending, view = pending[alive], view[alive]
        if not len(pending):
            continue
        ok = sees(view_positions[view], view_normals[view],
                  targets[pending], target_normals[pending],
                  occluder, spec, fov_mm=point_fov_mm[view],
                  frames=frames[view] if frames is not None else None)
        covered[pending[ok]] = True
    return covered
