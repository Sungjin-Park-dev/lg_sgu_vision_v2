"""손목 카메라 장착 보정 — ChArUco 보드 검출, hand-eye 계산, 별도 자세 검증, 보드 depth 평면 검사.

로봇 종류와 무관하다: base_T_tool(q) 는 호출자가 그 로봇의 FK 로 준다. OpenCV 는 함수 안에서만.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from core.realscan.capture import rigid, transform


def board_size(spec):
    """보드 바깥 크기 (가로, 세로) [m]."""
    return spec["squares_x"]*spec["square_m"], spec["squares_y"]*spec["square_m"]


def board_centre_frame(spec):
    """OpenCV 보드 좌표(모서리 원점, z 는 보드 안쪽) → 작업 좌표(보드 중심 원점, x 긴 변, z 위)."""
    return np.array([[1., 0, 0, spec["squares_x"]*spec["square_m"]/2],
                     [0, -1, 0, spec["squares_y"]*spec["square_m"]/2],
                     [0, 0, -1, 0], [0, 0, 0, 1]])


CALIBRATION_BOARD = {"squares_x": 6, "squares_y": 5, "square_m": 0.020,
                     "marker_m": 0.015, "dictionary": "DICT_4X4_50",
                     "layout": "opencv_4.6_even_row_legacy"}


class CharucoBoard:
    def __init__(self, spec=None):
        import cv2
        self.cv = cv2
        self.spec = dict(spec or CALIBRATION_BOARD)
        s = self.spec
        self.dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, s["dictionary"]))
        if (s["squares_x"] * s["squares_y"]) // 2 > self.dictionary.bytesList.shape[0]:
            raise ValueError(f"{s['dictionary']} 마커가 {s['squares_x']}×{s['squares_y']} 보드에 모자랍니다")
        size = (s["squares_x"], s["squares_y"])
        if hasattr(cv2.aruco, "CharucoBoard_create"):
            self.board = cv2.aruco.CharucoBoard_create(*size, s["square_m"], s["marker_m"], self.dictionary)
        else:
            self.board = cv2.aruco.CharucoBoard(size, s["square_m"], s["marker_m"], self.dictionary)
        if hasattr(self.board, "setLegacyPattern"):
            self.board.setLegacyPattern(True)  # Same physical board across OpenCV versions.

    def image(self, px_per_mm=10):
        """보드만 그린 흑백 영상 — 칸 한 변이 정수 픽셀이 되게 (흐림 없이 인쇄)."""
        s = self.spec
        size = tuple(round(s[k] * s["square_m"] * 1000 * px_per_mm) for k in ("squares_x", "squares_y"))
        return (self.board.draw(size, marginSize=0, borderBits=1) if hasattr(self.board, "draw")
                else self.board.generateImage(size, marginSize=0, borderBits=1))

    def svg(self):
        cv = self.cv
        s = self.spec
        w, h = (s[k] * s["square_m"] * 1000 for k in ("squares_x", "squares_y"))
        square = s["square_m"] * 1000
        # Lossless raster embedded in SVG with explicit millimetres for printing.
        import base64
        data = base64.b64encode(cv.imencode(".png", self.image())[1]).decode()
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w + 20:g}mm" height="{h + 30:g}mm" '
                f'viewBox="0 0 {w + 20:g} {h + 30:g}"><rect width="{w + 20:g}" height="{h + 30:g}" fill="white"/>'
                f'<image x="10" y="10" width="{w:g}" height="{h:g}" '
                f'href="data:image/png;base64,{data}"/>'
                f'<path d="M10 {h + 18:g}h{square:g}m-{square:g}-2v4m{square:g}-4v4" stroke="black" stroke-width="0.3"/>'
                f'<text x="{square + 15:g}" y="{h + 20:g}" font-size="4">{square:g} mm — print 100%</text></svg>')

    def detect(self, frame, min_corners=6):
        """RGB → (color_T_board, 재투영 RMS px, 표시 영상). 쓴 코너 수는 self.corners 에 남긴다."""
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
        self.corners = int(count)
        if count < min_corners:
            raise ValueError(f"ChArUco 코너가 {min_corners}개 이상 보이게 해주세요 (지금 {count}개)")
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
                      max_offset_mm=3., max_p95_mm=5., max_normal_deg=3., ignore_nearer_m=None):
    """Compare raw rectified depth against an independently RGB-detected board.

    Select pixels geometrically inside the projected board, never by residual.
    This tests plane-normal/offset agreement, not all six extrinsic degrees of
    freedom: translations along an infinite plane are unobservable here.
    Returned points/residuals include failures for honest diagnostic rendering.

    ignore_nearer_m: depth points more than this in front of the RGB board plane (towards the camera)
    cannot be the board — open gripper fingers or a hand in view. They are dropped and counted as
    occluded_points. Points behind the plane and small offsets are never dropped, so a depth bias still
    fails. Default None drops nothing (wrist calibration validation).
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
    occluded = 0
    if ignore_nearer_m is not None:
        front = residual_mm < -ignore_nearer_m * 1000   # board +z points away from the camera
        occluded = int(front.sum())
        points, residual_mm = points[~front], residual_mm[~front]
        count, projected_count = count - occluded, projected_count - occluded
        fraction = count / max(projected_count, 1)
    report = {"passed": False, "points": count, "projected_points": projected_count,
              "valid_fraction": float(fraction), "limits": {
                  "min_valid_fraction": min_valid_fraction, "min_points": min_points,
                  "max_offset_mm": max_offset_mm, "max_p95_mm": max_p95_mm,
                  "max_normal_deg": max_normal_deg}}
    if ignore_nearer_m is not None:
        report["occluded_points"] = occluded
        report["limits"]["ignore_nearer_m"] = ignore_nearer_m
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
