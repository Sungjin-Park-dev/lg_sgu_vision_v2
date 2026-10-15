"""AgileX PiPER 전용 — 관절 이름, URDF FK(yourdfpy), CAN 수신 전용 SDK 연결. 로봇 제어 없음."""
from __future__ import annotations

from pathlib import Path
import hashlib
import threading
import time

import numpy as np

from core.realscan.capture import transform


JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))


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
