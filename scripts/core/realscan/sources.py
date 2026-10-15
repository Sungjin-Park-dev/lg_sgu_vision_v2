"""실물 스캔의 프레임 소스 — 앱(apps/piper_scan.py)이 같은 인터페이스(sample/preview/color_T_depth)로 쓴다.

  LiveSource             ROS RealSense 수신 + PiPER CAN 수신 전용(piper.PassivePiper). 로봇 제어 클라이언트 없음.
  DemoSource             합성 구·테이블 RGB-D — ROS/CAN 없이 화면·복원 확인
  CalibrationSimSource   가상 보드 영상으로 보정부터 복원까지 — 팔 물리·IK 는 흉내 내지 않는다

ROS 모듈은 LiveSource 안에서만 import 한다.
"""
from __future__ import annotations

import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from core.realscan.capture import CameraFrame, CameraModel, JointHistory, depth_meters, rigid
from core.realscan.piper import PassivePiper


def stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class LiveSource:
    def __init__(self, cfg, allow_controllers=False):
        """allow_controllers: PiPER 드라이버·MoveIt 이 켜져 있어도 연결한다(정해 둔 자세로 이동하며
        보기·찍기). 기본은 거부 — 수동 촬영은 사람이 팔을 잡고 있으니 다른 프로그램이 명령을 보낼 수
        없어야 한다. 이 앱은 어느 쪽이든 CAN 을 읽기만 한다."""
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import Image, CameraInfo
        from cv_bridge import CvBridge
        from tf2_ros import Buffer, TransformListener
        from realsense2_camera_msgs.srv import DeviceInfo

        self.cfg, self.lock = cfg, threading.RLock()
        self.allow_controllers = allow_controllers
        self.depth = self.color = None
        self.depth_model = self.color_model = None
        self.serial = ""
        self.device = {}
        self.camera_error = "카메라 토픽 대기 중"
        self.future = None
        self.closed = False
        self.history = JointHistory(**cfg["capture_guard"])
        self.rclpy, self.bridge = rclpy, CvBridge()
        # Let the app's KeyboardInterrupt/finally close CAN and Viser before ROS.
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self.node = Node("piper_scan_observer")
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self.node)
        topics = cfg["camera_topics"]
        self.subs = [
            self.node.create_subscription(CameraInfo, topics["depth_info"],
                                          lambda m: self._info(m, "depth"), qos_profile_sensor_data),
            self.node.create_subscription(CameraInfo, topics["color_info"],
                                          lambda m: self._info(m, "color"), qos_profile_sensor_data),
            self.node.create_subscription(Image, topics["depth"],
                                          lambda m: self._image(m, "depth"), qos_profile_sensor_data),
            self.node.create_subscription(Image, topics["color"],
                                          lambda m: self._image(m, "color"), qos_profile_sensor_data),
        ]
        self.device_client = self.node.create_client(DeviceInfo, topics["device_info"])
        self.timer = self.node.create_timer(1.0, self._device_info)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()
        self.arm = None
        try:
            time.sleep(1)  # DDS discovery before opening the passive CAN socket.
            self._check_other_controllers()
            self.arm = PassivePiper(cfg["can_channel"], self.history)
        except BaseException:
            self.close()
            raise

    def _spin(self):
        from rclpy.executors import ExternalShutdownException
        try:
            self.executor.spin()
        except ExternalShutdownException:
            pass

    def _check_other_controllers(self):
        if getattr(self, "allow_controllers", False):
            return
        names = self.node.get_node_names()
        if any(n.startswith("agx_arm_ctrl") or n == "move_group" for n in names):
            raise ValueError("기존 PiPER 드라이버와 MoveIt을 Ctrl+C로 종료한 뒤 스캔 앱을 실행하세요")

    def _device_info(self):
        from realsense2_camera_msgs.srv import DeviceInfo
        if self.future is None and self.device_client.service_is_ready():
            self.future = self.device_client.call_async(DeviceInfo.Request())
        if self.future is not None and self.future.done():
            try:
                response = self.future.result()
                with self.lock:
                    self.serial = response.serial_number
                    self.device = {k: getattr(response, k) for k in
                                   ("device_name", "serial_number", "firmware_version", "usb_type_descriptor")}
            except Exception as exc:
                self.camera_error = str(exc)
            self.future = None  # Refresh also detects a hot-swapped camera.

    def _info(self, msg, kind):
        try:
            K = np.array(msg.k).reshape(3, 3)
            D = np.array(msg.d)
            if kind == "depth":
                # image_rect_raw: use the rectified projection intrinsics.
                P = np.array(msg.p).reshape(3, 4)
                if P[0, 0] > 0:
                    K = P[:, :3]
                if not np.allclose(np.array(msg.r).reshape(3, 3), np.eye(3), atol=1e-5):
                    raise ValueError("depth CameraInfo의 비단위 rectification은 현재 지원하지 않습니다")
                D = np.zeros(5)
            if (K[0, 0] <= 0 or K[1, 1] <= 0 or not np.isfinite(K).all()
                    or msg.width <= 0 or msg.height <= 0 or not msg.header.frame_id):
                raise ValueError("잘못된 CameraInfo")
            model = CameraModel(msg.width, msg.height, K, D, msg.header.frame_id,
                                "none" if kind == "depth" else msg.distortion_model)
            with self.lock:
                setattr(self, f"{kind}_model", model)
        except Exception as exc:
            self.camera_error = str(exc)
            with self.lock:
                setattr(self, f"{kind}_model", None)
                setattr(self, kind, None)

    def _image(self, msg, kind):
        try:
            with self.lock:
                model = getattr(self, f"{kind}_model")
            if model is None:
                return
            if msg.header.frame_id != model.frame or (msg.width, msg.height) != (model.width, model.height):
                raise ValueError(f"{kind} 이미지와 CameraInfo가 일치하지 않습니다")
            if kind == "depth":
                pixels = depth_meters(self.bridge.imgmsg_to_cv2(msg, "passthrough"), msg.encoding)
            else:
                pixels = self.bridge.imgmsg_to_cv2(msg, "rgb8").copy()
            with self.lock:
                setattr(self, kind, CameraFrame(stamp_seconds(msg.header.stamp), pixels, model))
                self.camera_error = ""
        except Exception as exc:
            self.camera_error = str(exc)
            with self.lock:
                setattr(self, kind, None)

    def color_T_depth(self):
        from rclpy.time import Time
        from rclpy.duration import Duration
        with self.lock:
            color, depth = self.color_model, self.depth_model
        if color is None or depth is None:
            raise ValueError("RGB/depth CameraInfo 대기 중")
        if color.frame == depth.frame:
            return np.eye(4)
        tf = self.tf.lookup_transform(color.frame, depth.frame, Time(), timeout=Duration(seconds=0.1)).transform
        q = tf.rotation
        return rigid(Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(),
                     [tf.translation.x, tf.translation.y, tf.translation.z])

    def sample(self, kind):
        return self._sample(kind, stationary=True)

    def preview(self, kind="depth"):
        """Read a moving pose for display only; captures must use sample()."""
        return self._sample(kind, stationary=False)

    def _sample(self, kind, *, stationary):
        self._check_other_controllers()
        if self.arm.error:
            raise ValueError(self.arm.error)
        with self.lock:
            frame = getattr(self, kind)
            serial = self.serial
        if not serial:
            raise ValueError("RealSense device_info와 카메라 일련번호 대기 중")
        if frame is None:
            raise ValueError(self.camera_error or f"{kind} 프레임 대기 중")
        q, dt = (self.history.at_rest if stationary else self.history.at_time)(frame.stamp)
        return frame, q, dt

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.arm is not None:
            self.arm.close()
        self.executor.shutdown(timeout_sec=2)
        self.thread.join(timeout=2)
        self.node.destroy_node()
        self.rclpy.try_shutdown()


class DemoSource:
    """Deterministic sphere/table RGB-D for UI and reconstruction checks, no ROS/CAN."""
    def __init__(self):
        self.index = 0
        self.serial = "DEMO-NO-HARDWARE"
        self.device = {"device_name": "Synthetic sphere, no robot connected"}
        self.depth_model = CameraModel(160, 120, np.array([[145., 0, 80], [0, 145., 60], [0, 0, 1]]),
                                      np.zeros(5), "demo_depth_optical", "none")
        self.color_model = self.depth_model
        self.center = np.array([0.25, 0.0, 0.075])
        self.arm = None
        self._render()

    def _render(self):
        from core.reconstruction.sensor import look_at
        azimuth = np.deg2rad(self.index * 35)
        elevation = np.deg2rad(30 + (self.index % 3) * 15)
        direction = np.array([np.cos(azimuth) * np.cos(elevation),
                              np.sin(azimuth) * np.cos(elevation), np.sin(elevation)])
        self.T = look_at(self.center + direction * 0.2, self.center)
        model = self.depth_model
        v, u = np.mgrid[:model.height, :model.width]
        K = model.K
        rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1],
                         np.ones_like(u)], axis=-1) @ self.T[:3, :3].T
        oc = self.T[:3, 3] - self.center
        a = np.sum(rays * rays, axis=-1)
        b = 2 * np.sum(rays * oc, axis=-1)
        c = np.sum(oc * oc) - 0.04 ** 2
        disc = b * b - 4 * a * c
        sphere = np.where(disc >= 0, (-b - np.sqrt(np.maximum(0, disc))) / (2 * a), np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            table = -self.T[2, 3] / rays[..., 2]
        table[table <= 0] = np.inf
        d = np.minimum(sphere, table).astype(np.float32)
        d[(d <= 0) | (d > 0.5)] = np.inf
        rgb = np.full((model.height, model.width, 3), 180, dtype=np.uint8)
        rgb[sphere < table] = [220, 90, 70]
        self.depth = CameraFrame(time.time(), d, self.depth_model)
        self.color = CameraFrame(self.depth.stamp, rgb, self.color_model)

    def sample(self, kind):
        self._render()
        return getattr(self, kind), np.zeros(6), 0.0

    def preview(self, kind="depth"):
        return self.sample(kind)

    def color_T_depth(self):
        return np.eye(4)

    def advance(self):
        self.index += 1
        self._render()

    def close(self):
        pass


class CalibrationSimSource(DemoSource):
    """Geometric RGB-D simulation, not arm physics/IK. Never imports ROS or CAN.

    Render a fixed physical-size board from varied virtual wrist-camera poses.
    Known extrinsics are used only to render and assess calibration accuracy.
    The UI's detector/solver must estimate its own extrinsics from these images.
    """
    def __init__(self, depth_bias_mm=0.0):
        from core.realscan.calibration import CharucoBoard
        self.depth_bias_m = float(depth_bias_mm) / 1000
        self.index, self.phase = 0, "calibration"
        self.serial = "SIM-CALIBRATION-NO-HARDWARE"
        self.device = {"device_name": "Virtual wrist camera, no arm physics"}
        self.depth_model = CameraModel(640, 480, np.array([[580., 0, 320], [0, 580., 240], [0, 0, 1]]),
                                      np.zeros(5), "sim_camera_optical", "none")
        self.color_model = self.depth_model
        self.center = np.array([.25, 0., .075])
        self.known_X = rigid(Rotation.from_rotvec([.15, -.25, .12]).as_matrix(), [.035, .01, .05])
        # OpenCV board y points down the printed front; its +z points away
        # from a front-facing camera. Avoid rendering mirrored backside markers.
        board_R = np.diag([1., -1., -1.])
        self.base_T_board = rigid(board_R, self.center - board_R @ [.06, .05, 0])
        self.arm = None
        board = CharucoBoard().board
        self.board_image = (board.draw((1200, 1000), marginSize=0) if hasattr(board, "draw")
                            else board.generateImage((1200, 1000), marginSize=0))
        self._render()

    def _render(self):
        if self.phase == "scan":
            return super()._render()
        import cv2
        from core.reconstruction.sensor import look_at
        azimuth = np.deg2rad(self.index * 47)
        elevation = np.deg2rad(55 + (self.index % 4) * 8)
        direction = np.array([np.cos(azimuth)*np.cos(elevation),
                              np.sin(azimuth)*np.cos(elevation), np.sin(elevation)])
        self.T = look_at(self.center + direction * (.23 + .01*(self.index % 3)), self.center)
        self.T[:3, :3] = self.T[:3, :3] @ Rotation.from_rotvec(
            [0, 0, np.deg2rad(12*np.sin(self.index))]).as_matrix()
        camera_T_board = np.linalg.inv(self.T) @ self.base_T_board
        corners = np.array([[0, 0, 0], [.12, 0, 0], [.12, .10, 0], [0, .10, 0]], dtype=float)
        projected = cv2.projectPoints(corners, cv2.Rodrigues(camera_T_board[:3, :3])[0],
                                     camera_T_board[:3, 3], self.color_model.K, np.zeros(5))[0]
        H = cv2.getPerspectiveTransform(np.array([[0, 0], [1200, 0], [1200, 1000], [0, 1000]],
                                                dtype=np.float32), projected.reshape(4, 2).astype(np.float32))
        gray = cv2.warpPerspective(self.board_image, H, (640, 480),
                                   flags=cv2.INTER_LINEAR, borderValue=180)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        v, u = np.mgrid[:480, :640]
        rays = np.stack([(u-320)/580, (v-240)/580, np.ones_like(u)], axis=-1) @ self.T[:3, :3].T
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = (self.center[2]-self.T[2, 3]) / rays[..., 2]
        depth += self.depth_bias_m
        depth[(depth <= 0) | (depth > .5)] = np.inf
        self.depth = CameraFrame(time.time(), depth.astype(np.float32), self.depth_model)
        self.color = CameraFrame(self.depth.stamp, rgb, self.color_model)

    def tool_pose(self):
        return self.T @ np.linalg.inv(self.known_X)

    def start_scan(self):
        self.phase, self.index = "scan", 0
        self._render()

    def calibration_error(self, estimated):
        delta = np.linalg.inv(self.known_X) @ np.asarray(estimated)
        return {"translation_mm": float(np.linalg.norm(delta[:3, 3])*1000),
                "rotation_deg": float(np.rad2deg(Rotation.from_matrix(delta[:3, :3]).magnitude()))}
