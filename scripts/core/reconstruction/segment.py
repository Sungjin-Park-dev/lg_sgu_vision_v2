"""depth 픽셀마다 "물체인가" 를 판정한다 — 복원(pipeline)과 Studio Scan 패널이 같은 판정을 쓴다.

판정 순서(먼저 걸린 이유로 분류):
  INVALID   depth 가 없음(inf·NaN) 또는 센서 측정 범위 밖
  OUTSIDE   world 로 펼친 점이 ROI 구 밖
  OBSTACLE  씬 장애물 OBB(+ 여유) 안 — 테이블·지지대·벽처럼 씬 YAML 이 아는 셀 기하
  OBJECT    나머지 = 물체

"아는 것을 지우면 모르는 것(물체)만 남는다" 는 가정이다. 씬 YAML 에 없는 지그는 물체로 남는다.
장애물은 계획에 남긴 촬영 당시의 OBB(ScanPlan.obstacles)를 받는다 — 없으면 지금 config 의 것.
"""

from __future__ import annotations

import numpy as np

from common import config, scene_config
from common.math_utils import quaternion_to_rotation_matrix
from core.reconstruction.sensor import pixel_world_points

OBJECT, INVALID, OUTSIDE, OBSTACLE = 0, 1, 2, 3
LABEL_NAMES = {OBJECT: "object", INVALID: "invalid", OUTSIDE: "outside ROI", OBSTACLE: "known obstacle"}


def _config_obstacles() -> np.ndarray:
    rows = [np.concatenate(scene_config.obstacle_obb(obj)) for obj in config.OBSTACLES]
    return np.asarray(rows, dtype=np.float64).reshape(-1, 10)


def inside_obstacles(points: np.ndarray, margin_m: float, obstacles=None) -> np.ndarray:
    """장애물 OBB(+ 여유) 안의 점 — (..., 3) → (...,) bool. 플래너가 보는 것과 같은 OBB.

    obstacles: (M,10) [pos3, quat_wxyz4, dims3]. None 이면 지금 config.OBSTACLES.
    """
    flat = points.reshape(-1, 3)
    hit = np.zeros(len(flat), dtype=bool)
    ok = np.flatnonzero(np.isfinite(flat).all(axis=1))
    boxes = _config_obstacles() if obstacles is None else np.asarray(obstacles, dtype=np.float64)
    for row in boxes:
        pos, quat, dims = row[:3], row[3:7], row[7:]
        R = quaternion_to_rotation_matrix(quat)
        local = (flat[ok] - pos) @ R                  # world → OBB 로컬
        hit[ok[(np.abs(local) <= dims / 2.0 + margin_m).all(axis=1)]] = True
    return hit.reshape(points.shape[:-1])


def classify_pixels(depth, K, world_T_cam, spec, roi_center, roi_radius: float,
                    obstacle_margin_m: float, stride: int = 1, obstacles=None):
    """→ (points (H',W',3), labels (H',W') int8). 라벨은 모듈 docstring 의 순서로 매긴다."""
    pts = pixel_world_points(depth, K, world_T_cam, stride)
    d = np.asarray(depth, dtype=np.float64)[::stride, ::stride]
    labels = np.full(d.shape, OBJECT, dtype=np.int8)
    valid = np.isfinite(d) & (d >= spec.min_range_m) & (d <= spec.max_range_m)
    labels[~valid] = INVALID
    outside = valid & (np.linalg.norm(pts - roi_center, axis=-1) > roi_radius)
    labels[outside] = OUTSIDE
    rest = labels == OBJECT
    obstacle = np.zeros_like(rest)
    obstacle[rest] = inside_obstacles(pts[rest], obstacle_margin_m, obstacles)
    labels[obstacle] = OBSTACLE
    return pts, labels
