#!/usr/bin/env python3
"""선택 단계 — 후보 viewpoint 중 **무엇을 쓸 것인가**.

파이프라인은 세 단계다: **후보 생성**(brep/sampling) → **판정**(visibility) → **선택**(여기).
지금까지는 셋째가 없어서 "만든 것을 전부 쓴다" 였다. 격자는 겹침 50% 로 규칙적으로 깔리므로
중복이 있고, 실제로 curved_structure 는 105점 중 **67점만으로 커버리지 100% 가 유지**된다.

선택을 별도 단계로 두는 이유는 강제하지 않기 위해서다 — 기본값 ``all`` 은 지금까지의 동작
그대로이고, ``greedy`` 는 옵션이다. 커버 관계는 ``visibility`` 를 그대로 쓰므로 커버리지
리포트와 **같은 판정** 위에서 고른다(둘이 어긋날 수 없다).

set covering 은 NP-hard 지만 greedy 가 (1+ln n) 근사를 보장하고, 우리 규모(후보 10²,
셀 10³~10⁴)에서는 순식간이다.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from . import visibility

SELECTION_ALL = "all"          # 전부 사용 = 지금까지의 동작
SELECTION_GREEDY = "greedy"    # 면적 이득이 큰 순으로 최소 집합
SELECTION_MODES = (SELECTION_ALL, SELECTION_GREEDY)


def coverage_sets(cells, positions, normals, point_fov_mm, occluder,
                  spec: visibility.SensorSpec, mask=None,
                  frames: Optional[visibility.ViewFrames] = None) -> List[np.ndarray]:
    """viewpoint 마다 **자기가 덮는 셀 인덱스**. set covering 의 '집합' 들이다.

    셀 기준이 아니라 viewpoint 기준으로 훑는다 — 커버리지 평가와 방향만 반대이고 판정은
    같은 ``visibility.sees`` 다. 광선은 viewpoint 하나당 한 번에 묶어 쏜다.
    """
    from scipy.spatial import cKDTree

    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    point_fov_mm = np.asarray(point_fov_mm, dtype=np.float64).reshape(-1)
    keep = np.ones(len(cells.points), dtype=bool) if mask is None \
        else np.asarray(mask, dtype=bool)

    tree = cKDTree(cells.points)
    sets: List[np.ndarray] = []
    for k in range(len(positions)):
        # 사각 프레임이면 외접원 반경으로 후보를 모아야 모서리 쪽을 안 놓친다.
        radius_m = (float(frames.half_diagonal_mm[k]) / 1000.0 if frames is not None
                    else point_fov_mm[k] / 2.0 / 1000.0)
        candidates = np.asarray(
            tree.query_ball_point(positions[k], radius_m), dtype=int)
        candidates = candidates[keep[candidates]] if len(candidates) else candidates
        if not len(candidates):
            sets.append(np.zeros(0, dtype=int))
            continue
        frame_k = None
        if frames is not None:
            frame_k = visibility.ViewFrames(
                np.repeat(frames.axis_u[k][None, :], len(candidates), axis=0),
                np.repeat(frames.axis_v[k][None, :], len(candidates), axis=0),
                np.full(len(candidates), frames.fov_u_mm[k]),
                np.full(len(candidates), frames.fov_v_mm[k]))
        ok = visibility.sees(
            np.repeat(positions[k][None, :], len(candidates), axis=0),
            np.repeat(normals[k][None, :], len(candidates), axis=0),
            cells.points[candidates], cells.normals[candidates],
            occluder, spec, fov_mm=point_fov_mm[k], frames=frame_k)
        sets.append(candidates[ok])
    return sets


def greedy_cover(sets: Sequence[np.ndarray], areas, mask=None,
                 target_ratio: float = 1.0) -> np.ndarray:
    """면적 이득이 가장 큰 viewpoint 를 반복해 고른다. 고른 인덱스를 돌려준다.

    ``target_ratio`` 는 **검사 대상 면적 대비** 목표 커버리지(0~1)다. 후보 전체로도 그 값에
    못 미치면 더 보탤 것이 없을 때 멈춘다 — 즉 "가능한 만큼" 이 상한이다.
    """
    areas = np.asarray(areas, dtype=np.float64)
    keep = np.ones(len(areas), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    total = float(areas[keep].sum())
    if total <= 0.0 or not len(sets):
        return np.arange(len(sets), dtype=int)

    goal = float(np.clip(target_ratio, 0.0, 1.0)) * total
    covered = np.zeros(len(areas), dtype=bool)
    chosen: List[int] = []
    remaining = list(range(len(sets)))
    while remaining:
        gains = [float(areas[sets[i][~covered[sets[i]]]].sum()) if len(sets[i]) else 0.0
                 for i in remaining]
        best = int(np.argmax(gains))
        if gains[best] <= 1e-12:
            break                      # 더 보탤 면적이 없다
        pick = remaining.pop(best)
        covered[sets[pick]] = True
        chosen.append(pick)
        if float(areas[covered & keep].sum()) >= goal - 1e-12:
            break
    return np.asarray(sorted(chosen), dtype=int)


def select(mode: str, cells, positions, normals, point_fov_mm, occluder,
           spec: visibility.SensorSpec, mask=None, target_ratio: float = 1.0,
           frames: Optional[visibility.ViewFrames] = None,
           verbose: bool = True) -> Optional[np.ndarray]:
    """모드에 따라 쓸 viewpoint 인덱스를 고른다. ``all`` 이면 None(=전부)."""
    if mode != SELECTION_GREEDY or cells is None or len(cells.points) == 0:
        return None
    sets = coverage_sets(cells, positions, normals, point_fov_mm, occluder, spec,
                         mask=mask, frames=frames)
    chosen = greedy_cover(sets, cells.areas_cm2, mask=mask, target_ratio=target_ratio)
    if verbose:
        print(f"  Selection (greedy set cover): {len(chosen)}/{len(positions)} viewpoints kept")
    return chosen
