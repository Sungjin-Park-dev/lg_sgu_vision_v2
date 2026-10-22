"""실물 스캔의 촬영 기본 요소 — 로봇·카메라 종류와 무관하다(PiPER 전용은 piper.py).

강체 변환 검증, depth 단위 변환, 원자적 저장, URDF FK, 관절 이력(정지 판정·시간 동기화), 카메라
모델·프레임, 촬영 세션(ScanSession → data/{object}/scan/{run}, core.reconstruction.storage 규약).

All transforms map the named right-hand frame into the left-hand frame.
The scan world is the robot base frame; camera frames are OpenCV/ROS optical.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import re
import tempfile
import threading
import time

import numpy as np


def transform(value):
    T = np.asarray(value, dtype=np.float64)
    if (T.shape != (4, 4) or not np.isfinite(T).all()
            or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-5)):
        raise ValueError("잘못된 강체 변환 행렬입니다")
    return T.copy()


def rigid(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.asarray(t).reshape(3)
    return transform(T)


def depth_meters(array, encoding):
    a = np.asarray(array)
    if a.ndim != 2:
        raise ValueError("depth는 2차원 이미지여야 합니다")
    if encoding == "16UC1" and a.dtype == np.uint16:
        d = a.astype(np.float32) * 0.001  # RealSense ROS wrapper publishes mm.
    elif encoding == "32FC1" and a.dtype == np.float32:
        d = a.copy()
    else:
        raise ValueError(f"지원하지 않는 depth 인코딩/dtype: {encoding}/{a.dtype}")
    d[~np.isfinite(d) | (d <= 0)] = np.inf
    return d


def median_depth(frames):
    """멈춘 카메라의 depth 여러 장 → 픽셀별 중앙값 한 장(CameraFrame, 가운데 장의 시각).

    D405 같은 수동 스테레오는 장마다 잡음이 다르게 들어간다 — 픽셀마다 값이 흔들리고, 한두 장에서만
    튀거나 비는 픽셀이 있다. 과반 장에 값이 있는 픽셀만 그 값들의 중앙값으로 채우고 나머지는 무효(inf)로
    둔다(한 장에만 튄 값으로 구멍을 메우지 않는다). 매 장 똑같이 틀리는 오차(무늬 없는 면 등)는 못 고친다.
    """
    frames = list(frames)
    if not frames:
        raise ValueError("depth 프레임이 없습니다")
    model = frames[0].model
    if any(f.model is not model and f.model.record() != model.record() for f in frames):
        raise ValueError("depth 여러 장 사이에 카메라 설정이 변경되었습니다")
    if len(frames) == 1:
        return frames[0]
    stack = np.stack([np.where(np.isfinite(f.pixels), f.pixels, np.nan) for f in frames]).astype(np.float32)
    valid = np.isfinite(stack).sum(axis=0)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # 전부 NaN 인 픽셀 — 아래에서 inf 로
            d = np.nanmedian(stack, axis=0)
    d[valid < len(frames) // 2 + 1] = np.inf
    stamps = sorted(f.stamp for f in frames)
    return CameraFrame(stamps[len(stamps) // 2], d.astype(np.float32), model)


def atomic_json(path, value):
    atomic_bytes(Path(path), (json.dumps(value, ensure_ascii=False, indent=2,
                                        allow_nan=False) + "\n").encode())


def atomic_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        # mkstemp creates 0600. The container writes as root, so keep data readable
        # by the host user (viewer, Studio, regression) like np.savez/write_text do.
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "wb") as f:
            f.write(value)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class UrdfFK:
    """측정 관절값의 URDF FK(yourdfpy). hash 는 보정 identity 에 들어가는 URDF 바이트의 sha256."""
    def __init__(self, urdf, joint_names, base_frame, tool_frame, mismatch_message):
        from yourdfpy import URDF
        self.path = Path(urdf)
        self.hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.robot = URDF.load(str(self.path), load_meshes=False,
                               load_collision_meshes=False, build_collision_scene_graph=False)
        self.joint_names = tuple(joint_names)
        if tuple(self.robot.actuated_joint_names) != self.joint_names:
            raise ValueError(mismatch_message)
        self.base_frame, self.tool_frame = base_frame, tool_frame
        self.lock = threading.Lock()
        self.pose(np.zeros(len(self.joint_names)))

    def pose(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (len(self.joint_names),) or not np.isfinite(q).all():
            raise ValueError("유효한 측정 관절값이 필요합니다")
        with self.lock:
            # Measured teach poses must not be clamped or normalized to joint limits.
            self.robot.update_cfg(dict(zip(self.joint_names, q)))
            return transform(self.robot.get_transform(self.tool_frame, self.base_frame))


class JointHistory:
    def __init__(self, stable_s=0.5, step_rad=0.002, max_age_s=0.5, sync_s=0.1):
        self.samples = deque(maxlen=1000)
        self.lock = threading.Lock()
        self.stable_s, self.step_rad = stable_s, step_rad
        self.max_age_s, self.sync_s = max_age_s, sync_s

    def add(self, stamp, q):
        q = np.asarray(q, dtype=np.float64)
        if not np.isfinite(stamp) or q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("6개 관절의 유효한 측정값이 필요합니다")
        with self.lock:
            if self.samples and stamp < self.samples[-1][0]:
                self.samples.clear()  # Clock reset: accumulate a fresh stationary window.
            if not self.samples or stamp > self.samples[-1][0]:
                self.samples.append((float(stamp), q.copy()))

    def _snapshot(self, stamp, now=None):
        now = time.time() if now is None else now
        with self.lock:
            samples = list(self.samples)
        if not samples or now - samples[-1][0] > self.max_age_s:
            raise ValueError("관절 피드백이 없거나 오래되었습니다")
        if now - stamp > self.max_age_s or stamp - now > self.sync_s:
            raise ValueError("카메라 프레임이 오래되었거나 시계가 일치하지 않습니다")
        ts = np.asarray([s[0] for s in samples])
        qs = np.stack([s[1] for s in samples])
        return ts, qs

    def at_time(self, stamp, now=None):
        """Fresh, synchronized measured pose for visualization, including during movement."""
        return self._interpolate(*self._snapshot(stamp, now), stamp)

    def at_rest(self, stamp, now=None):
        ts, qs = self._snapshot(stamp, now)
        if ts[0] > stamp - self.stable_s:
            raise ValueError("정지 확인 중 — 팔을 0.5초 이상 멈추세요")
        window = qs[ts >= stamp - self.stable_s]
        if len(window) < 5 or np.max(np.ptp(window, axis=0)) > self.step_rad:
            raise ValueError("팔이 움직이고 있습니다 — 정지 후 촬영하세요")
        return self._interpolate(ts, qs, stamp)

    def _interpolate(self, ts, qs, stamp):
        closest = int(np.argmin(np.abs(ts - stamp)))
        if abs(ts[closest] - stamp) > self.sync_s:
            raise ValueError("카메라와 관절 피드백의 시간 차이가 큽니다")
        right = int(np.searchsorted(ts, stamp))
        q = qs[closest]
        if 0 < right < len(ts) and ts[right] - ts[right - 1] <= self.sync_s:
            alpha = (stamp - ts[right - 1]) / (ts[right] - ts[right - 1])
            q = qs[right - 1] * (1 - alpha) + qs[right] * alpha
        return q.copy(), float(abs(ts[closest] - stamp))


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    K: np.ndarray
    D: np.ndarray
    frame: str
    distortion_model: str = "plumb_bob"

    def record(self):
        return {"width": self.width, "height": self.height, "K": self.K.tolist(),
                "D": self.D.tolist(), "frame": self.frame,
                "distortion_model": self.distortion_model}


@dataclass
class CameraFrame:
    stamp: float
    pixels: np.ndarray
    model: CameraModel


def safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("이름은 영문·숫자로 시작하고 영문·숫자·_ . - 만 사용하세요")
    return value


class ScanSession:
    """Compatible measured dataset; no simulated scene or ground-truth pose."""
    def __init__(self, root, object_name, run_name, metadata, *, resume=False, robot="piper",
                 world_frame="base_link"):
        self.directory = Path(root) / safe_name(object_name) / "scan" / safe_name(run_name)
        if self.directory.exists() and not resume:
            raise FileExistsError(f"기존 run을 덮어쓰지 않습니다: {self.directory}")
        self.meta = {**metadata, "object": object_name, "robot": robot,
                     "source": metadata.get("source", f"{robot}_real"),
                     "capture_mode": "manual", "world_frame": world_frame}
        self.records = []
        self.profile = None
        self.calibration = None
        if resume:
            capture = json.loads((self.directory / "capture.json").read_text())
            if (capture["object"] != object_name or capture["robot"] != robot
                    or capture["source"] != self.meta["source"]):
                raise ValueError("촬영 원본의 물체·로봇·실제/가상 구분이 현재 설정과 다릅니다")
            for key in ("min_range_m", "max_range_m"):
                if capture[key] != metadata[key]:
                    raise ValueError("촬영 당시 depth 범위와 현재 설정이 다릅니다")
            names = [f"frame_{i:03d}.npz" for i in range(len(capture["frames"]))]
            if names != sorted(p.name for p in (self.directory/"frames").glob("frame_*.npz")):
                raise ValueError("촬영 기록과 원본 프레임 개수가 다릅니다. 원본을 확인하세요")
            if [r["index"] for r in capture["frames"]] != list(range(len(names))):
                raise ValueError("촬영 프레임 번호가 일치하지 않습니다")
            self.meta = {k:v for k,v in capture.items() if k not in ("camera", "calibration", "frames")}
            self.records = capture["frames"]
            self.profile = json.dumps(capture["camera"], sort_keys=True)
            self.calibration = json.dumps(capture["calibration"], sort_keys=True)

    def save(self, frame, q, base_T_cam, calibration, roi, radius, obstacles, names, detail):
        from core.reconstruction import storage
        from core.reconstruction.sensor import ScanCameraSpec
        from core.reconstruction.storage import ScanPlan
        if radius <= 0 or not np.isfinite(radius) or not np.isfinite(roi).all():
            raise ValueError("ROI 중심과 반경을 확인하세요")
        signature = json.dumps(frame.model.record(), sort_keys=True)
        cal_signature = json.dumps(calibration, sort_keys=True)
        if self.profile is not None and signature != self.profile:
            raise ValueError("촬영 중 카메라 설정이 변경되었습니다 — 새 run을 시작하세요")
        if self.calibration is not None and cal_signature != self.calibration:
            raise ValueError("촬영 중 보정을 변경할 수 없습니다 — 새 run을 시작하세요")
        q, T = np.asarray(q, dtype=float), transform(base_T_cam)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("측정 관절값이 잘못되었습니다")
        if frame.pixels.shape != (frame.model.height, frame.model.width):
            raise ValueError("depth와 CameraInfo의 크기가 다릅니다")
        index = len(self.records)
        path = self.directory / "frames" / f"frame_{index:03d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"부분 저장 파일이 있습니다: {path}. 새 run을 사용하세요")
        import io
        data = io.BytesIO()
        np.savez_compressed(data, depth=frame.pixels, K=frame.model.K, world_T_cam=T,
                            q_cmd=q, q_meas=q)
        atomic_bytes(path, data.getvalue())
        self.profile, self.calibration = signature, cal_signature
        self.records.append({"index": index, "stamp": frame.stamp, "q": q.tolist(),
                             "world_T_cam": T.tolist(), **detail})
        spec = ScanCameraSpec(width=frame.model.width, height=frame.model.height,
                              hfov_deg=float(np.degrees(2 * np.arctan(frame.model.width / (2 * frame.model.K[0, 0])))),
                              min_range_m=self.meta["min_range_m"], max_range_m=self.meta["max_range_m"],
                              optimal_range_m=0.2, tool_T_cam=tuple(np.asarray(calibration["tool_T_depth"]).ravel()))
        poses = np.asarray([r["world_T_cam"] for r in self.records])
        directions = poses[:, :3, 3] - np.asarray(roi)
        norm = np.linalg.norm(directions, axis=1)
        directions /= np.maximum(norm[:, None], 1e-9)
        plan = ScanPlan(q=np.asarray([r["q"] for r in self.records]), world_T_cam=poses,
                        goal_T_cam=poses.copy(), view_dirs=directions, roll_deg=np.zeros(len(poses)),
                        roi_center=np.asarray(roi), roi_radius=float(radius), standoff=float(np.mean(norm)),
                        camera=spec, meta=self.meta, obstacles=np.asarray(obstacles).reshape(-1, 10),
                        obstacle_names=list(names))
        storage.save_plan(self.directory, plan)
        atomic_json(self.directory / "capture.json", {**self.meta, "camera": frame.model.record(),
                                                      "calibration": calibration, "frames": self.records})
        return index
