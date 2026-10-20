"""Universal Robots 전용 — 관절 이름, URDF FK, RTDE 관절값 수신(읽기 전용). 로봇 제어 없음.

RTDE 는 출력 레시피만 만든다(입력 레시피를 만들지 않으므로 이 연결로는 레지스터 하나 쓸 수 없다).
External Control 프로그램·UR 드라이버와 무관하게 돌아서, 펜던트 프리드라이브로 팔을 옮기는
수동 촬영에 쓴다. Summit 뒤의 UR 은 `ssh -N -L 30004:192.168.0.210:30004 robot@<Summit>` 터널로 닿는다.
"""
from __future__ import annotations

from collections import deque
import socket
import struct
import threading
import time

from core.realscan.capture import UrdfFK


JOINT_NAMES = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")

# RTDE 패킷 종류(프로토콜 2).
PROTOCOL_VERSION, SETUP_OUTPUTS, START, PAUSE, DATA, TEXT = (ord(c) for c in "VOSPUM")
VARIABLES = ("timestamp", "actual_q", "robot_mode", "safety_mode")
TYPE_FORMATS = {"DOUBLE": "d", "VECTOR6D": "6d", "VECTOR3D": "3d", "INT32": "i", "UINT32": "I",
                "UINT64": "Q", "VECTOR6INT32": "6i", "VECTOR6UINT32": "6I", "BOOL": "?", "UINT8": "B"}
# 팔에 전원이 들어가 관절값이 유효한 robot_mode: IDLE(브레이크 잠김)·BACKDRIVE·RUNNING.
POWERED_MODES = {5: "IDLE", 6: "BACKDRIVE", 7: "RUNNING"}
SAFETY_MODES = {1: "NORMAL", 2: "REDUCED", 3: "PROTECTIVE_STOP", 4: "RECOVERY", 5: "SAFEGUARD_STOP",
                6: "SYSTEM_EMERGENCY_STOP", 7: "ROBOT_EMERGENCY_STOP", 8: "VIOLATION", 9: "FAULT"}


class UrFK(UrdfFK):
    def __init__(self, urdf, base_frame="base_link", tool_frame="tool0"):
        super().__init__(urdf, JOINT_NAMES, base_frame, tool_frame,
                         "UR shoulder_pan_joint–wrist_3_joint URDF가 필요합니다")


def _recv_exact(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("RTDE 연결이 끊겼습니다")
        data += chunk
    return data


def _send(sock, kind, payload=b""):
    sock.sendall(struct.pack(">HB", len(payload) + 3, kind) + payload)


def _read(sock):
    size, kind = struct.unpack(">HB", _recv_exact(sock, 3))
    return kind, _recv_exact(sock, size - 3)


def _expect(sock, kind):
    while True:
        got, payload = _read(sock)
        if got == kind:
            return payload
        if got != TEXT:  # Controller log messages may arrive at any time.
            raise ValueError(f"RTDE 응답 종류가 다릅니다: {chr(got)!r} (기다린 것 {chr(kind)!r})")


class RtdeReader:
    """RTDE 출력(timestamp·actual_q·robot_mode·safety_mode)을 받아 JointHistory 에 넣는다.

    관절 시각: 컨트롤러 timestamp 에 '수신 시각 − timestamp' 의 최근 10초 최솟값을 더한다 — 무선 지연이
    튀어도 시각이 따라 튀지 않고, 카메라(같은 NTP 벽시계)와 비교할 수 있다.
    """
    def __init__(self, host, port, history, frequency=125.0, connect=socket.create_connection):
        self.host, self.port, self.history = host, int(port), history
        self.frequency, self.connect = float(frequency), connect
        self.error = "UR RTDE 연결 중"
        self.status = {}
        self.stop = threading.Event()
        self.sock = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _handshake(self, sock):
        _send(sock, PROTOCOL_VERSION, struct.pack(">H", 2))
        if not _expect(sock, PROTOCOL_VERSION)[0]:
            raise ValueError("컨트롤러가 RTDE 프로토콜 2를 받지 않습니다")
        _send(sock, SETUP_OUTPUTS, struct.pack(">d", self.frequency) + ",".join(VARIABLES).encode())
        reply = _expect(sock, SETUP_OUTPUTS)
        recipe, types = reply[0], reply[1:].decode().split(",")
        if len(types) != len(VARIABLES) or any(t not in TYPE_FORMATS for t in types):
            raise ValueError(f"RTDE 출력 설정 실패: {dict(zip(VARIABLES, types))}")
        _send(sock, START)
        if not _expect(sock, START)[0]:
            raise ValueError("컨트롤러가 RTDE 시작을 거부했습니다")
        return recipe, ">B" + "".join(TYPE_FORMATS[t] for t in types)

    def _run(self):
        while not self.stop.is_set():
            try:
                with self.connect((self.host, self.port), timeout=3.0) as sock:
                    self.sock = sock
                    recipe, fmt = self._handshake(sock)
                    offsets = deque(maxlen=int(self.frequency * 10))
                    while not self.stop.is_set():
                        kind, payload = _read(sock)
                        if kind != DATA:
                            continue
                        values = struct.unpack(fmt, payload)
                        if values[0] != recipe:
                            continue
                        controller_time, q = values[1], values[2:8]
                        robot_mode, safety_mode = values[8], values[9]
                        offsets.append(time.time() - controller_time)
                        self.status = {"robot_mode": POWERED_MODES.get(robot_mode, robot_mode),
                                       "safety_mode": SAFETY_MODES.get(safety_mode, safety_mode)}
                        if robot_mode not in POWERED_MODES:
                            self.error = f"UR 팔 전원이 꺼져 있습니다 (robot_mode {robot_mode})"
                            continue
                        self.history.add(controller_time + min(offsets), q)
                        self.error = ""
            except Exception as exc:
                if not self.stop.is_set():
                    self.error = f"UR RTDE({self.host}:{self.port}): {exc}"
                    self.stop.wait(1.0)
            finally:
                self.sock = None

    def close(self):
        self.stop.set()
        sock = self.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.thread.join(timeout=2)
