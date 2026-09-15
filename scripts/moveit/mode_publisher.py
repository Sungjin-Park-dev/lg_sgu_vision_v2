#!/usr/bin/env python3
"""앱의 모드를 ROS 로 계속 발행한다 — isaac_pipeline 이 ROS 를 건드리는 **유일한** 통로.

isaac_pipeline 은 ROS 를 import 하지 않는다(Isaac 번들 파이썬). 대신 이 프로세스를 하나
띄워 두고, 모드가 바뀌면 stdin 에 한 줄 쓴다. 이 프로세스는 마지막 모드를 ``--rate`` Hz 로
``/looksgood/mode`` 에 계속 발행한다. 스택 쪽 mode_gate 가 그걸 받아 집행한다.

stdin 이 닫히면(앱 종료) 발행을 멈추고 끝난다 → 게이트는 timeout 뒤 fail-safe 로 간다.
그래서 "앱이 살아서 이 모드를 원한다" 가 곧 발행이고, 발행이 멈추면 아무도 로봇을 못
움직인다.

    python3 scripts/moveit/mode_publisher.py --mode sim/inspection
    (stdin)  real/moveit
"""

from __future__ import annotations

import argparse
import sys
import threading

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

MODES = ("sim/moveit", "sim/inspection", "real/moveit", "real/inspection")


class ModePublisher(Node):
    def __init__(self, topic: str, mode: str, rate_hz: float) -> None:
        super().__init__("looksgood_mode_publisher")
        self._pub = self.create_publisher(String, topic, 10)
        self._mode = mode
        self._lock = threading.Lock()
        self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(f"publishing {topic} = {mode} at {rate_hz:g} Hz")

    def set_mode(self, mode: str) -> None:
        with self._lock:
            if mode != self._mode:
                self.get_logger().info(f"mode -> {mode}")
            self._mode = mode

    def _tick(self) -> None:
        with self._lock:
            mode = self._mode
        msg = String()
        msg.data = mode
        self._pub.publish(msg)


def _stdin_loop(node: ModePublisher, stop: threading.Event) -> None:
    for line in sys.stdin:
        mode = line.strip()
        if not mode:
            continue
        if mode not in MODES:
            node.get_logger().warning(f"ignoring unknown mode {mode!r}")
            continue
        node.set_mode(mode)
    stop.set()          # EOF = 앱이 갔다


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--topic", default="/looksgood/mode")
    p.add_argument("--rate", type=float, default=2.0)
    a = p.parse_args(argv)
    rclpy.init(args=None)
    node = ModePublisher(a.topic, a.mode, a.rate)
    stop = threading.Event()
    threading.Thread(target=_stdin_loop, args=(node, stop), daemon=True).start()
    try:
        while rclpy.ok() and not stop.is_set():
            rclpy.spin_once(node, timeout_sec=0.2)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass                    # SIGINT/SIGTERM — rclpy 가 이미 context 를 닫았다
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
