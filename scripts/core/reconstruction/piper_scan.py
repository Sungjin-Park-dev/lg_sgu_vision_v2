"""Real PiPER scanning primitives. No robot control or simulation dependencies.

All transforms map the named right-hand frame into the left-hand frame.
The scan world is PiPER ``base_link``; camera frames are OpenCV/ROS optical.
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
from scipy.spatial.transform import Rotation


JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))


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


def atomic_json(path, value):
    atomic_bytes(Path(path), (json.dumps(value, ensure_ascii=False, indent=2,
                                        allow_nan=False) + "\n").encode())


def atomic_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(value)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


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


class PiperFK:
    def __init__(self, urdf, base_frame="base_link", tool_frame="link6"):
        from yourdfpy import URDF
        self.path = Path(urdf)
        self.hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.robot = URDF.load(str(self.path), load_meshes=False,
                               load_collision_meshes=False, build_collision_scene_graph=False)
        if tuple(self.robot.actuated_joint_names) != JOINT_NAMES:
            raise ValueError("PiPER joint1–joint6 URDF가 필요합니다")
        self.base_frame, self.tool_frame = base_frame, tool_frame
        self.lock = threading.Lock()
        self.pose(np.zeros(6))

    def pose(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("유효한 측정 관절값이 필요합니다")
        with self.lock:
            # Measured teach poses must not be clamped or normalized to UR limits.
            self.robot.update_cfg(dict(zip(JOINT_NAMES, q)))
            return transform(self.robot.get_transform(self.tool_frame, self.base_frame))


class PassivePiper:
    """SDK CAN listener with TX physically blocked at its comm object.

    Only connect/get_joint_angles/get_arm_status/disconnect are used. The guard
    also catches accidental future SDK writes, including during connection.
    """
    def __init__(self, channel, history, factory=None):
        if factory is None:
            from pyAgxArm import AgxArmFactory, create_agx_arm_config
            factory = lambda: AgxArmFactory.create_arm(
                create_agx_arm_config(robot="piper", comm="can", channel=channel))
        self.arm = factory()
        self.history = history
        self.error = "CAN 피드백 대기 중"
        self.status = None
        self.blocked_tx = 0
        self.stop_event = threading.Event()
        ctx = self.arm.get_context()
        comm = ctx.get_comm() or ctx.init_comm()
        comm.send = self._deny_send
        try:
            self.arm.connect()
        except BaseException:
            self.arm.disconnect()
            raise
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def _deny_send(self, *args, **kwargs):
        self.blocked_tx += 1
        self.error = "읽기 전용 연결에서 CAN 송신 시도가 차단되었습니다"
        raise RuntimeError(self.error)

    def _poll(self):
        while not self.stop_event.wait(0.01):
            try:
                # SDK merges three feedback frames; require all three to be fresh.
                parts = [getattr(self.arm._parser, n, None)
                         for n in ("joint_12", "joint_34", "joint_56")]
                if any(p is None for p in parts):
                    raise ValueError("6개 관절의 CAN 피드백을 기다리고 있습니다")
                stamps = [float(p.timestamp) for p in parts]
                if time.time() - min(stamps) > 0.5 or max(stamps) - min(stamps) > 0.1:
                    raise ValueError("CAN 관절 피드백이 오래되었거나 불완전합니다")
                value = self.arm.get_joint_angles()
                status = self.arm.get_arm_status()
                if value is None or status is None or time.time() - status.timestamp > 0.5:
                    raise ValueError("PiPER 상태 피드백 대기 중")
                self.status = {name: int(getattr(status.msg, name)) for name in
                               ("ctrl_mode", "arm_status", "teach_status", "motion_status")}
                if self.status["arm_status"] not in (0, 11):  # Normal or teaching record.
                    raise ValueError(f"PiPER 오류 상태: {self.status}")
                if self.blocked_tx:
                    raise ValueError("CAN 송신 시도가 발생했습니다 — 앱을 다시 시작하세요")
                self.history.add(float(value.timestamp), list(value.msg))
                self.error = ""
            except Exception as exc:
                self.error = str(exc)

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.arm.disconnect()


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


class CharucoBoard:
    def __init__(self):
        import cv2
        self.cv = cv2
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        if hasattr(cv2.aruco, "CharucoBoard_create"):
            self.board = cv2.aruco.CharucoBoard_create(6, 5, 0.020, 0.015, self.dictionary)
        else:
            self.board = cv2.aruco.CharucoBoard((6, 5), 0.020, 0.015, self.dictionary)
        self.spec = {"squares_x": 6, "squares_y": 5, "square_m": 0.020,
                     "marker_m": 0.015, "dictionary": "DICT_4X4_50",
                     "layout": "opencv_4.6_even_row_legacy"}
        if hasattr(self.board, "setLegacyPattern"):
            self.board.setLegacyPattern(True)  # Same physical board across OpenCV versions.

    def svg(self):
        cv = self.cv
        image = (self.board.draw((1200, 1000), marginSize=0, borderBits=1)
                 if hasattr(self.board, "draw")
                 else self.board.generateImage((1200, 1000), marginSize=0, borderBits=1))
        # Lossless raster embedded in SVG with explicit millimetres for printing.
        import base64
        data = base64.b64encode(cv.imencode(".png", image)[1]).decode()
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="140mm" height="130mm" '
                f'viewBox="0 0 140 130"><rect width="140" height="130" fill="white"/>'
                f'<image x="10" y="10" width="120" height="100" '
                f'href="data:image/png;base64,{data}"/>'
                f'<path d="M10 118h20m-20-2v4m20-4v4" stroke="black" stroke-width="0.3"/>'
                f'<text x="35" y="120" font-size="4">20 mm — print 100%</text></svg>')

    def detect(self, frame):
        cv = self.cv
        if frame.model.distortion_model not in ("plumb_bob", "rational_polynomial", "none"):
            raise ValueError(f"지원하지 않는 RGB 왜곡 모델: {frame.model.distortion_model}")
        gray = cv.cvtColor(frame.pixels, cv.COLOR_RGB2GRAY)
        if hasattr(cv.aruco, "interpolateCornersCharuco"):
            corners, ids, _ = cv.aruco.detectMarkers(gray, self.dictionary)
            if ids is None:
                raise ValueError("ChArUco 마커가 보이지 않습니다")
            count, corners_c, ids_c = cv.aruco.interpolateCornersCharuco(
                corners, ids, gray, self.board, cameraMatrix=frame.model.K,
                distCoeffs=frame.model.D)
        else:
            params = cv.aruco.CharucoParameters()
            params.cameraMatrix, params.distCoeffs = frame.model.K, frame.model.D
            corners_c, ids_c, _, _ = cv.aruco.CharucoDetector(self.board, params).detectBoard(gray)
            count = 0 if ids_c is None else len(ids_c)
        if count < 6:
            raise ValueError("ChArUco 코너가 6개 이상 보이게 해주세요")
        points = (self.board.chessboardCorners if hasattr(self.board, "chessboardCorners")
                  else self.board.getChessboardCorners())[ids_c.reshape(-1)]
        ok, rvec, tvec = cv.solvePnP(points, corners_c, frame.model.K, frame.model.D)
        if not ok or float(tvec[2, 0]) <= 0:
            raise ValueError("보드 자세 계산에 실패했습니다")
        projected = cv.projectPoints(points, rvec, tvec, frame.model.K, frame.model.D)[0]
        rms = float(np.sqrt(np.mean(np.sum((projected - corners_c) ** 2, axis=-1))))
        if rms > 2.0:
            raise ValueError(f"보드 재투영 오차가 큽니다: {rms:.2f}px")
        annotated = frame.pixels.copy()
        cv.aruco.drawDetectedCornersCharuco(annotated, corners_c, ids_c)
        return rigid(cv.Rodrigues(rvec)[0], tvec), rms, annotated


def board_spread(poses):
    poses = np.stack([transform(p) for p in poses])
    centre = np.mean(poses[:, :3, 3], axis=0)
    mean_R = Rotation.from_matrix(poses[:, :3, :3]).mean()
    translations = np.linalg.norm(poses[:, :3, 3] - centre, axis=1) * 1000
    rotations = (mean_R.inv() * Rotation.from_matrix(poses[:, :3, :3])).magnitude() * 180 / np.pi
    return {"translation_max_mm": float(translations.max()),
            "rotation_max_deg": float(rotations.max())}


class CalibrationQualityError(ValueError):
    def __init__(self, report, candidate, fixed):
        super().__init__(f"보정 오차가 기준을 넘었습니다: {report}")
        self.report, self.candidate, self.fixed = report, candidate, fixed


def calibration_limits(max_translation_mm, max_rotation_deg):
    if (not np.isfinite([max_translation_mm, max_rotation_deg]).all()
            or max_translation_mm <= 0 or not 0 < max_rotation_deg < 180):
        raise ValueError("보정 허용 오차는 유효한 양수여야 합니다")
    return {"max_translation_mm": float(max_translation_mm), "max_rotation_deg": float(max_rotation_deg)}


def solve_hand_eye(base_T_tools, color_T_boards, method=None, *, max_translation_mm=5., max_rotation_deg=1.):
    import cv2
    if len(base_T_tools) < 15 or len(base_T_tools) != len(color_T_boards):
        raise ValueError("보정 자세가 15개 이상 필요합니다")
    tools = np.stack([transform(p) for p in base_T_tools])
    boards = np.stack([transform(p) for p in color_T_boards])
    rotations = Rotation.from_matrix(tools[0, :3, :3].T @ tools[:, :3, :3]).as_rotvec()
    singular = np.linalg.svd(rotations, compute_uv=False)
    if singular[1] < 0.15 or np.max(np.linalg.norm(rotations, axis=1)) < np.deg2rad(20):
        raise ValueError("서로 다른 축으로 손목을 더 회전시켜주세요 — 회전 변화가 부족합니다")
    R, t = cv2.calibrateHandEye(list(tools[:, :3, :3]), list(tools[:, :3, 3]),
                               list(boards[:, :3, :3]), list(boards[:, :3, 3]),
                               method=cv2.CALIB_HAND_EYE_PARK if method is None else method)
    X = rigid(R, t)
    fixed = tools @ X @ boards
    limits = calibration_limits(max_translation_mm, max_rotation_deg)
    report = {**board_spread(fixed), "limits": limits}
    if report["translation_max_mm"] > max_translation_mm or report["rotation_max_deg"] > max_rotation_deg:
        raise CalibrationQualityError(report, X, fixed)
    return X, report, fixed


def validate_holdout(reference_poses, holdout_poses, *, max_translation_mm=5., max_rotation_deg=1.):
    limits = calibration_limits(max_translation_mm, max_rotation_deg)
    if len(holdout_poses) < 3:
        raise ValueError("별도의 검증 자세 3개가 필요합니다")
    ref = np.stack(reference_poses)
    centre = np.mean(ref[:, :3, 3], axis=0)
    mean_R = Rotation.from_matrix(ref[:, :3, :3]).mean()
    holds = np.stack([transform(T) for T in holdout_poses])
    report = {"translation_max_mm": float(np.max(np.linalg.norm(holds[:, :3, 3] - centre, axis=1)) * 1000),
              "rotation_max_deg": float(np.max((mean_R.inv() * Rotation.from_matrix(holds[:, :3, :3])).magnitude()) * 180 / np.pi),
              "limits": limits}
    if report["translation_max_mm"] > max_translation_mm or report["rotation_max_deg"] > max_rotation_deg:
        raise ValueError(f"검증 자세 오차가 기준을 넘었습니다: {report}")
    return report


def mean_board_pose(poses):
    poses = np.stack([transform(p) for p in poses])
    return rigid(Rotation.from_matrix(poses[:, :3, :3]).mean().as_matrix(),
                 np.mean(poses[:, :3, 3], axis=0))


def check_board_depth(frame, color_T_board, color_T_depth, *, board_size=(.12, .10),
                      border_m=.012, stride=3, min_range_m=.07, max_range_m=.50,
                      min_valid_fraction=.35, min_points=100,
                      max_offset_mm=3., max_p95_mm=5., max_normal_deg=3.):
    """Compare raw rectified depth against an independently RGB-detected board.

    Select pixels geometrically inside the projected board, never by residual.
    This tests plane-normal/offset agreement, not all six extrinsic degrees of
    freedom: translations along an infinite plane are unobservable here.
    Returned points/residuals include failures for honest diagnostic rendering.
    """
    import cv2
    depth_T_board = np.linalg.inv(transform(color_T_depth)) @ transform(color_T_board)
    width, height = board_size
    if not 0 < border_m < min(width, height)/2 or stride < 1 or min_points < 3:
        raise ValueError("잘못된 보드 검사 영역 설정")
    corners = np.array([[border_m, border_m, 0], [width-border_m, border_m, 0],
                        [width-border_m, height-border_m, 0], [border_m, height-border_m, 0]])
    camera_corners = corners @ depth_T_board[:3, :3].T + depth_T_board[:3, 3]
    if np.any(camera_corners[:, 2] <= 0):
        raise ValueError("보드가 depth 카메라 뒤에 있습니다")
    pixels = camera_corners @ frame.model.K.T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    mask = np.zeros(frame.pixels.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(pixels).astype(np.int32), 1)
    v, u = np.mgrid[:frame.model.height:stride, :frame.model.width:stride]
    selected = mask[::stride, ::stride].astype(bool)
    depth = frame.pixels[::stride, ::stride]
    valid = selected & np.isfinite(depth) & (depth >= min_range_m) & (depth <= max_range_m)
    projected_count, count = int(selected.sum()), int(valid.sum())
    fraction = count / max(projected_count, 1)
    K = frame.model.K
    points = np.stack([(u[valid]-K[0, 2])*depth[valid]/K[0, 0],
                       (v[valid]-K[1, 2])*depth[valid]/K[1, 1], depth[valid]], axis=-1)
    # Row-vector form of board_T_depth. z is signed distance to RGB board plane.
    board_points = (points - depth_T_board[:3, 3]) @ depth_T_board[:3, :3]
    residual_mm = board_points[:, 2] * 1000
    report = {"passed": False, "points": count, "projected_points": projected_count,
              "valid_fraction": float(fraction), "limits": {
                  "min_valid_fraction": min_valid_fraction, "min_points": min_points,
                  "max_offset_mm": max_offset_mm, "max_p95_mm": max_p95_mm,
                  "max_normal_deg": max_normal_deg}}
    reasons = []
    if count < min_points:
        reasons.append(f"유효 depth 점 부족: {count}/{min_points}")
    if fraction < min_valid_fraction:
        reasons.append(f"유효 depth 비율 부족: {fraction:.0%}")
    if count >= 3:
        _, singular, Vt = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
        normal = Vt[-1]
        expected = depth_T_board[:3, 2]
        angle = float(np.rad2deg(np.arccos(np.clip(abs(normal @ expected), 0, 1))))
        offset = float(np.median(residual_mm))
        p95 = float(np.percentile(np.abs(residual_mm), 95))
        report.update(offset_mm=offset, p95_abs_mm=p95,
                      rms_mm=float(np.sqrt(np.mean(residual_mm**2))), normal_error_deg=angle)
        if singular[1] < 1e-6:
            reasons.append("깊이점의 평면 방향을 계산할 수 없습니다")
        if abs(offset) > max_offset_mm:
            reasons.append(f"보드 평면 거리 오차: {offset:.2f} mm")
        if p95 > max_p95_mm:
            reasons.append(f"depth 오차 P95: {p95:.2f} mm")
        if angle > max_normal_deg:
            reasons.append(f"보드 평면 각도 오차: {angle:.2f}°")
    report["passed"], report["reasons"] = not reasons, reasons
    return report, points.astype(np.float32), residual_mm.astype(np.float32)


def safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("이름은 영문·숫자로 시작하고 영문·숫자·_ . - 만 사용하세요")
    return value


class ScanSession:
    """Compatible measured dataset; no simulated scene or ground-truth pose."""
    def __init__(self, root, object_name, run_name, metadata, *, resume=False):
        self.directory = Path(root) / safe_name(object_name) / "scan" / safe_name(run_name)
        if self.directory.exists() and not resume:
            raise FileExistsError(f"기존 run을 덮어쓰지 않습니다: {self.directory}")
        self.meta = {**metadata, "object": object_name, "robot": "piper",
                     "source": metadata.get("source", "piper_real"),
                     "capture_mode": "manual", "world_frame": "base_link"}
        self.records = []
        self.profile = None
        self.calibration = None
        if resume:
            capture = json.loads((self.directory / "capture.json").read_text())
            if (capture["object"] != object_name or capture["robot"] != "piper"
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
