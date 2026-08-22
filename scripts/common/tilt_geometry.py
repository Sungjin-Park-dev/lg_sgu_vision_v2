#!/usr/bin/env python3
"""Tilt 기하 — 한 viewpoint 를 표면점 둘레로 공전시키는 카메라 포즈들. numpy 만 쓴다.

CLI(`core/trajectory/tilt_motion.py`)와 Isaac UI 가 **같은 부채꼴**을 보여야 한다. UI 는
Isaac Sim 프로세스 안에서 도는데 `core.trajectory` 는 import 만으로 cuRobo/torch 를 끌고
오므로 거기서는 재사용할 수 없다 — 둘이 공유할 수 있는 유일한 층이 numpy-only 인 여기다.
한 벌만 두면 UI 가 보여주는 그림이 실제 생성될 궤적과 어긋날 수 없다.

**프레임**: 포즈는 ``object_pose`` 를 준 프레임 그대로 나온다. CLI 는 물체 배치
(config.TARGET_OBJECT)를, UI 는 스테이지에서 읽은 변환을 주는데 **둘이 같은 프레임**이다 —
Isaac world 원점이 robot base_link 이기 때문이다(2026-08-22 이전에는 z 가 0.805 어긋났다).
"""

from __future__ import annotations

import numpy as np

from .math_utils import normalize_vectors

# 왕복 순서. (축, 라벨, 각도 인자) — 축은 카메라 로컬 축이다:
#   pitch = 카메라 y축 둘레 공전 → 화면상 위/아래
#   roll  = 카메라 x축 둘레 공전 → 화면상 좌/우
TILT_LEGS = (
    ("pitch", "up", "pitch_max"),
    ("pitch", "down", "pitch_min"),
    ("roll", "left", "roll_min"),
    ("roll", "right", "roll_max"),
)

# 광축이 up 벡터와 거의 나란하면 카메라 x축이 정의되지 않는다. ±45° 이내 tilt 에서는 걸릴
# 일이 없지만 큰 각도 인자를 받을 수 있으므로 대체 up 을 둔다.
_PARALLEL_COS = 0.99


def _unit(v: np.ndarray) -> np.ndarray:
    return normalize_vectors(np.asarray(v, dtype=np.float64))


def _rodrigues(axis: np.ndarray, theta: float) -> np.ndarray:
    """축-각 → 3x3 회전행렬 (scipy 없이)."""
    x, y, z = _unit(axis)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=np.float64)


def camera_pose(position, normal, working_distance_m: float, object_pose) -> np.ndarray:
    """viewpoint(표면점 + 법선) → 카메라 4x4 포즈.

    `core.trajectory.poses.build_camera_poses` 와 같은 규약이다(그쪽이 스캔 파이프라인의
    정본이고, 여기는 그 한 개짜리 numpy-only 사본이다 — 규약을 바꾸면 양쪽을 같이 고칠 것).
    카메라는 법선 방향으로 WD 만큼 떠서 표면을 마주본다.
    """
    n = _unit(normal)
    z_axis = -n                                     # 광축은 표면을 향한다
    helper = (np.array([0.0, 0.0, 1.0])
              if abs(float(np.dot(z_axis, [0.0, 0.0, 1.0]))) <= _PARALLEL_COS
              else np.array([0.0, 1.0, 0.0]))
    x_axis = _unit(np.cross(helper, z_axis))
    y_axis = np.cross(z_axis, x_axis)

    local = np.eye(4, dtype=np.float64)
    local[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
    local[:3, 3] = np.asarray(position, dtype=np.float64) + n * float(working_distance_m)
    return np.asarray(object_pose, dtype=np.float64) @ local


def orbit_pose(cam_pos, R0, target, axis, theta_rad: float) -> np.ndarray:
    """target 둘레로 카메라를 axis 축 theta 만큼 공전시킨 4x4 포즈.

    위치는 회전시키고 자세는 '항상 target 을 본다'로 다시 만든다 — 그래서 공전 중에도
    주시점과 작업거리가 정확히 보존된다.
    """
    cam_pos = np.asarray(cam_pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    c = target + _rodrigues(axis, float(theta_rad)) @ (cam_pos - target)

    z_c = _unit(target - c)
    up = np.asarray(R0, dtype=np.float64)[:, 1]
    if abs(float(np.dot(_unit(up), z_c))) > _PARALLEL_COS:
        up = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(up, z_c))) > _PARALLEL_COS:
            up = np.array([0.0, 1.0, 0.0])
    x_c = _unit(np.cross(up, z_c))
    y_c = np.cross(z_c, x_c)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.column_stack([x_c, y_c, z_c])
    T[:3, 3] = c
    return T


def tilt_legs(center_pose, working_distance_m: float, *,
              pitch_min: float, pitch_max: float, pitch_n: int,
              roll_min: float, roll_max: float, roll_n: int):
    """중심 포즈에서 뻗는 네 방향(up/down/left/right)의 공전 포즈들.

    Returns:
        target: (3,) 주시점 = 표면점. 모든 시선이 여기로 모인다(부채꼴의 꼭짓점).
        legs: [(label, poses (k,4,4), angles_deg (k,)), ...] — **중심(0°)은 제외**하고,
            중심에서 먼 쪽으로 정렬된다. 각도가 0 이거나 샘플이 없으면 그 leg 는 빠진다.
    """
    center_pose = np.asarray(center_pose, dtype=np.float64)
    cam_pos = center_pose[:3, 3]
    R0 = center_pose[:3, :3]
    target = cam_pos + R0[:, 2] * float(working_distance_m)

    spec = {
        "pitch_max": (pitch_max, pitch_n), "pitch_min": (pitch_min, pitch_n),
        "roll_min": (roll_min, roll_n), "roll_max": (roll_max, roll_n),
    }
    legs = []
    for axis_name, label, key in TILT_LEGS:
        angle_deg, n_samples = spec[key]
        if abs(float(angle_deg)) < 1e-9 or int(n_samples) < 2:
            continue
        axis = R0[:, 0] if axis_name == "roll" else R0[:, 1]
        # 0° 는 중심 포즈 자신이라 뺀다 — leg 당 새 포즈는 n-1 개다.
        angles = np.linspace(0.0, float(angle_deg), int(n_samples))[1:]
        poses = np.stack([
            orbit_pose(cam_pos, R0, target, axis, np.deg2rad(a)) for a in angles
        ])
        legs.append((label, poses, angles))
    return target, legs
