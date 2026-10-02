"""스캔용 depth 카메라의 정의 — 내부 파라미터, 측정 범위, 손목 장착 위치.

카메라는 아직 정해지지 않았다. 기본값은 **Zivid 2+ M60 급의 근사치**다(공식 스펙으로
확인한 값이 아니다): 600mm 에서 대략 570×460mm 를 보고, 350~1100mm 를 잰다. 해상도는 2448×2048
을 2×2 로 줄인 1224×1024. 카메라가 정해지면 ``ScanCameraSpec`` 값만 바꾼다.

좌표 규약 — 이 모듈이 다루는 카메라 프레임은 **OpenCV 광학 프레임**이다(+Z 전방, +X 오른쪽,
+Y 아래). Open3D·ROS optical 과 같고, cuRobo 툴 프레임 ``camera_optical_frame`` 도 +Z 가
광축이다. USD 카메라(−Z 전방, +Y 위)로의 변환은 Isaac 쪽(``scan_capture``)에서만 한다.

numpy 만 쓴다 — Isaac 의 Python 과 cuRobo 쪽이 둘 다 import 한다.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass(frozen=True)
class ScanCameraSpec:
    width: int = 1224
    height: int = 1024
    hfov_deg: float = 50.8              # 600mm 에서 ~570mm 폭
    min_range_m: float = 0.35           # 이보다 가깝거나 먼 depth 는 측정 실패로 버린다
    max_range_m: float = 1.10
    optimal_range_m: float = 0.60
    # 툴 프레임(camera_optical_frame) → 스캔 카메라 광학 프레임, 4×4 row-major.
    # 기본은 단위행렬 = 검사 카메라와 같은 자리·같은 광축. 실제 장착이 정해지면 바꾼다.
    tool_T_cam: tuple = field(default_factory=lambda: tuple(np.eye(4).ravel().tolist()))

    @property
    def fx(self) -> float:
        return 0.5 * self.width / math.tan(math.radians(self.hfov_deg) / 2.0)

    @property
    def vfov_deg(self) -> float:
        # 정사각 픽셀 — fy = fx
        return math.degrees(2.0 * math.atan(0.5 * self.height / self.fx))

    @property
    def K(self) -> np.ndarray:
        # cx, cy = 해상도/2 — Isaac Camera.get_intrinsics_matrix() 와 같은 규약
        return np.array([[self.fx, 0.0, 0.5 * self.width],
                         [0.0, self.fx, 0.5 * self.height],
                         [0.0, 0.0, 1.0]])

    @property
    def tool_T_cam_matrix(self) -> np.ndarray:
        return np.asarray(self.tool_T_cam, dtype=np.float64).reshape(4, 4)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tool_T_cam"] = list(self.tool_T_cam)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ScanCameraSpec":
        d = dict(d)
        d["tool_T_cam"] = tuple(float(v) for v in d["tool_T_cam"])
        return cls(**d)


DEFAULT_SCAN_CAMERA = ScanCameraSpec()


def look_at(cam_pos, target, roll_deg: float = 0.0, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """``target`` 을 광축(+Z)으로 보는 카메라의 world 4×4 (OpenCV 프레임).

    roll 0 은 이미지 +X 가 수평(world up 에 직교)인 자세다. 광축 둘레 roll 은 카메라 시야를
    거의 바꾸지 않으면서 IK 에는 다른 문제를 주므로, 스캔 계획에서 도달성을 늘리는 데 쓴다.
    """
    p = np.asarray(cam_pos, dtype=np.float64)
    z = np.asarray(target, dtype=np.float64) - p
    z /= np.linalg.norm(z)
    up = np.asarray(up, dtype=np.float64)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:          # 바로 위/아래를 볼 때 — 다른 기준축
        x = np.cross(z, np.array([1.0, 0.0, 0.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    a = math.radians(roll_deg)
    x, y = math.cos(a) * x + math.sin(a) * y, -math.sin(a) * x + math.cos(a) * y
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, p
    return T


def backproject(depth: np.ndarray, K: np.ndarray, world_T_cam: np.ndarray,
                min_range_m: float = 0.0, max_range_m: float = np.inf,
                stride: int = 1) -> np.ndarray:
    """z-depth 이미지(H, W) → world 점 (N, 3). 범위 밖·inf·NaN 픽셀은 버린다.

    depth 는 **광축 방향 거리**(Isaac ``distance_to_image_plane``)여야 한다 — 광선 길이
    (``distance_to_camera``)를 넣으면 가장자리일수록 바깥으로 부푼다.
    """
    d = np.asarray(depth, dtype=np.float64)[::stride, ::stride]
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    u = u * stride
    v = v * stride
    ok = np.isfinite(d) & (d >= min_range_m) & (d <= max_range_m)
    z = d[ok]
    x = (u[ok] - K[0, 2]) * z / K[0, 0]
    y = (v[ok] - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], axis=1)
    return pts @ world_T_cam[:3, :3].T + world_T_cam[:3, 3]
