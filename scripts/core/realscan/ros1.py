"""ROS1(TCPROS) 수신 전용 최소 클라이언트 — ROS1 설치 없이 Summit(Melodic) 의 토픽·서비스를 읽는다.

master 에는 아무것도 등록하지 않는다: 발행 노드를 물어 그 노드에 직접 TCPROS 로 붙는다(rostopic 의
AnyMsg 처럼 md5 '*'). 발행·파라미터 쓰기는 없다. 메시지 해석은 스캔에 쓰는 타입만 —
sensor_msgs/Image·CameraInfo·CompressedImage, realsense2_camera/Extrinsics·DeviceInfo.
"""
from __future__ import annotations

import socket
import struct
import threading
import time
import xmlrpc.client

import numpy as np

CALLER_ID = "/realscan_reader"


class _Transport(xmlrpc.client.Transport):
    def __init__(self, timeout):
        super().__init__()
        self.timeout = timeout

    def make_connection(self, host):
        connection = super().make_connection(host)
        connection.timeout = self.timeout  # Per call; never the process-wide socket default.
        return connection


def _proxy(uri, timeout):
    return xmlrpc.client.ServerProxy(uri, transport=_Transport(timeout))


def _result(reply, what):
    code, message, value = reply
    if code != 1:
        raise ValueError(f"{what}: {message}")
    return value


def publisher_address(master, topic, timeout=3.0):
    pubs = dict(_result(_proxy(master, timeout).getSystemState(CALLER_ID), "ROS master")[0])
    if not pubs.get(topic):
        raise ValueError(f"{topic} 를 발행하는 노드가 없습니다")
    node = _result(_proxy(master, timeout).lookupNode(CALLER_ID, pubs[topic][0]), "lookupNode")
    protocol = _result(_proxy(node, timeout).requestTopic(CALLER_ID, topic, [["TCPROS"]]), "requestTopic")
    return protocol[1], int(protocol[2])


def _header_bytes(fields):
    body = b"".join(struct.pack("<I", len(f)) + f for f in (f"{k}={v}".encode() for k, v in fields.items()))
    return struct.pack("<I", len(body)) + body


def _recv_exact(sock, n):
    buffer = bytearray(n)
    view, got = memoryview(buffer), 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if not k:
            raise ConnectionError("연결이 끊겼습니다")
        got += k
    return bytes(buffer)


def _read_block(sock):
    return _recv_exact(sock, struct.unpack("<I", _recv_exact(sock, 4))[0])


def _parse_header(raw):
    fields, i = {}, 0
    while i < len(raw):
        n = struct.unpack_from("<I", raw, i)[0]
        key, _, value = raw[i + 4:i + 4 + n].decode(errors="replace").partition("=")
        fields[key] = value
        i += 4 + n
    return fields


def _subscribe(sock, topic):
    sock.sendall(_header_bytes({"callerid": CALLER_ID, "topic": topic, "md5sum": "*", "type": "*",
                                "tcp_nodelay": "1"}))
    reply = _parse_header(_read_block(sock))
    if "error" in reply:
        raise ValueError(f"{topic}: {reply['error']}")
    return reply


def fetch(master, topic, accept=None, timeout=5.0):
    """새 연결로 토픽을 받아 accept(msg) 가 참인 첫 메시지를 돌려준다(latched 토픽이면 즉시)."""
    host, port = publisher_address(master, topic, timeout)
    deadline = time.monotonic() + timeout
    with socket.create_connection((host, port), timeout=timeout) as sock:
        _subscribe(sock, topic)
        while True:
            message = _read_block(sock)
            if accept is None or accept(message):
                return message
            if time.monotonic() > deadline:
                raise TimeoutError(f"{topic}: 조건에 맞는 메시지가 {timeout:g}s 안에 오지 않았습니다")


def call_service(master, service, request=b"", timeout=3.0):
    uri = _result(_proxy(master, timeout).lookupService(CALLER_ID, service), "lookupService")
    host, port = uri.replace("rosrpc://", "").rstrip("/").rsplit(":", 1)
    with socket.create_connection((host, int(port)), timeout=timeout) as sock:  # md5 probe only
        sock.sendall(_header_bytes({"callerid": CALLER_ID, "service": service, "md5sum": "*", "probe": "1"}))
        md5 = _parse_header(_read_block(sock)).get("md5sum", "*")
    with socket.create_connection((host, int(port)), timeout=timeout) as sock:
        sock.sendall(_header_bytes({"callerid": CALLER_ID, "service": service, "md5sum": md5}))
        reply = _parse_header(_read_block(sock))
        if "error" in reply:
            raise ValueError(f"{service}: {reply['error']}")
        sock.sendall(struct.pack("<I", len(request)) + request)
        ok = _recv_exact(sock, 1)[0]
        payload = _read_block(sock)
    if not ok:
        raise ValueError(f"{service}: {payload.decode(errors='replace')}")
    return payload


class TopicStream:
    """토픽을 계속 받아 메시지마다 callback(bytes) 을 부른다. 끊기거나 발행이 멈추면 다시 붙는다."""
    def __init__(self, master, topic, callback, *, timeout=3.0, retry_s=1.0):
        self.master, self.topic, self.callback = master, topic, callback
        self.timeout, self.retry_s = timeout, retry_s
        self.error = f"{topic} 연결 중"
        self.stop = threading.Event()
        self.sock = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            try:
                host, port = publisher_address(self.master, self.topic, self.timeout)
                with socket.create_connection((host, port), timeout=self.timeout) as sock:
                    self.sock = sock
                    _subscribe(sock, self.topic)
                    while not self.stop.is_set():
                        message = _read_block(sock)
                        self.error = ""
                        self.callback(message)
            except Exception as exc:
                if not self.stop.is_set():
                    self.error = f"{self.topic}: {exc}"
                    self.stop.wait(self.retry_s)
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


class _Reader:
    def __init__(self, data):
        self.data, self.offset = data, 0

    def take(self, fmt):
        values = struct.unpack_from("<" + fmt, self.data, self.offset)
        self.offset += struct.calcsize("<" + fmt)
        return values

    def u32(self):
        return self.take("I")[0]

    def string(self):
        n = self.u32()
        value = self.data[self.offset:self.offset + n].decode(errors="replace")
        self.offset += n
        return value

    def blob(self):
        n = self.u32()
        start = self.offset
        self.offset += n
        return start, n

    def header(self):
        _, sec, nsec = self.take("III")
        return sec + nsec * 1e-9, self.string()


def message_stamp(message):
    """header 가 맨 앞인 메시지의 시각만 읽는다(픽셀은 건드리지 않음)."""
    return _Reader(message).header()[0]


IMAGE_DTYPES = {"16UC1": ("<u2", 1), "mono16": ("<u2", 1), "32FC1": ("<f4", 1),
                "rgb8": ("u1", 3), "bgr8": ("u1", 3), "mono8": ("u1", 1)}


def decode_image(message):
    """sensor_msgs/Image → (stamp, frame_id, encoding, ndarray). 버퍼를 복사해 돌려준다."""
    r = _Reader(message)
    stamp, frame = r.header()
    height, width = r.take("II")
    encoding = r.string()
    bigendian, step = r.take("BI")
    start, n = r.blob()
    if encoding not in IMAGE_DTYPES:
        raise ValueError(f"지원하지 않는 이미지 인코딩: {encoding}")
    if bigendian:
        raise ValueError("big-endian 이미지는 지원하지 않습니다")
    dtype, channels = IMAGE_DTYPES[encoding]
    item = np.dtype(dtype).itemsize
    if n != height * step or step < width * channels * item:
        raise ValueError("이미지 크기와 데이터 길이가 맞지 않습니다")
    rows = np.frombuffer(message, np.uint8, n, start).reshape(height, step)[:, :width * channels * item]
    pixels = rows.copy().view(dtype).reshape((height, width, channels) if channels > 1 else (height, width))
    return stamp, frame, encoding, pixels


def decode_compressed(message):
    """sensor_msgs/CompressedImage → (stamp, frame_id, format, bytes)."""
    r = _Reader(message)
    stamp, frame = r.header()
    fmt = r.string()
    start, n = r.blob()
    return stamp, frame, fmt, message[start:start + n]


def decode_camera_info(message):
    r = _Reader(message)
    stamp, frame = r.header()
    height, width = r.take("II")
    model = r.string()
    D = r.take(f"{r.u32()}d")
    K, R, P = r.take("9d"), r.take("9d"), r.take("12d")
    return {"stamp": stamp, "frame": frame, "width": width, "height": height, "distortion_model": model,
            "D": np.array(D), "K": np.array(K).reshape(3, 3), "R": np.array(R).reshape(3, 3),
            "P": np.array(P).reshape(3, 4)}


def decode_extrinsics(message):
    """realsense2_camera/Extrinsics → 4x4 (회전은 librealsense 의 column-major)."""
    r = _Reader(message)
    r.header()
    rotation, translation = r.take("9d"), r.take("3d")
    T = np.eye(4)
    T[:3, :3] = np.array(rotation).reshape(3, 3, order="F")
    T[:3, 3] = translation
    return T


DEVICE_INFO_FIELDS = ("device_name", "serial_number", "firmware_version", "usb_type_descriptor",
                      "firmware_update_id", "sensors", "physical_port")


def decode_device_info(payload):
    """realsense2_camera/DeviceInfo 응답(문자열만 차례로)."""
    r = _Reader(payload)
    info = {}
    for name in DEVICE_INFO_FIELDS:
        if r.offset >= len(payload):
            break
        info[name] = r.string()
    return info
