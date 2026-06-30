#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exit successfully after receiving a non-empty JointState message."""

import argparse
import sys
from typing import Optional

import rclpy
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointStateGate(Node):
    """Wait for the first useful JointState sample on a topic."""

    def __init__(self, topic_name: str) -> None:
        super().__init__('joint_state_gate')
        self._received_msg = False
        self._subscription = self.create_subscription(
            JointState, topic_name, self._joint_state_callback, 10
        )

    def _joint_state_callback(self, msg: JointState) -> None:
        if not msg.name:
            return

        self._received_msg = True
        stamp = f'{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}'
        self.get_logger().info(
            f'Received JointState with {len(msg.name)} joints at {stamp}.'
        )

    @property
    def received_msg(self) -> bool:
        return self._received_msg


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Wait for a JointState message before releasing launch gating.'
    )
    parser.add_argument('--topic', default='/joint_states')
    parser.add_argument('--timeout', type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    rclpy.init(args=None)

    node = JointStateGate(args.topic)
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    start_time = node.get_clock().now()
    timeout = Duration(seconds=args.timeout)

    try:
        while rclpy.ok() and not node.received_msg:
            executor.spin_once(timeout_sec=0.1)
            if node.get_clock().now() - start_time > timeout:
                node.get_logger().error(
                    f'Timed out waiting for JointState on {args.topic}.'
                )
                return 1
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
