#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Relay /isaac_joint_commands_raw -> /isaac_joint_commands, but ONLY while a
trajectory is actively executing on one of the watched controllers.

Why the gate: the TopicBasedSystem hardware continuously publishes the active
controller's command (which, when idle, is the controller's held/initial setpoint
— NOT necessarily the robot's pose). Forwarding that unconditionally makes the
Isaac robot jump to follow MoveIt the moment move_group/controllers come up. By
forwarding only while a FollowJointTrajectory goal is ACCEPTED/EXECUTING, the robot
stays put when idle (MoveIt reflects the robot, not vice-versa) and moves only on
an actual Execute / Publish.

Watches multiple controllers' action status so it works for both the MoveIt
controller (scaled_joint_trajectory_controller) and the Inspection controller
(joint_trajectory_controller); only one is active per pipeline mode.

No /isaac_joint_states liveness gating and no goal cancelling — those (earlier)
misread transient tick hitches as stops and cancelled legitimate trajectories.
"""

import argparse
import sys
from typing import Optional

from action_msgs.msg import GoalStatus, GoalStatusArray
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

_ACTIVE = {GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING}


class IsaacJointCommandRelay(Node):
    def __init__(self, input_topic: str, output_topic: str, status_topics: list) -> None:
        super().__init__('isaac_joint_command_relay')
        self._active = {t: False for t in status_topics}
        # Mode gate, set by the Isaac app via `ros2 param set`. The app is the only
        # thing that knows the run mode: in sim it IS the robot (forward commands), in
        # real it MIRRORS the real robot (Isaac must NOT be driven by these commands).
        # Gating HERE — at the single point that feeds /isaac_joint_commands — means a
        # command executed in real mode never reaches Isaac at all, so it can never be
        # buffered and replayed when sim is re-entered. Defaults True (sim).
        self.declare_parameter('forward_enabled', True)
        self._pub = self.create_publisher(JointState, output_topic, 10)
        self._cmd_sub = self.create_subscription(
            JointState, input_topic, self._command_callback, 10)
        for topic in status_topics:
            self.create_subscription(
                GoalStatusArray, topic,
                lambda msg, t=topic: self._status_callback(t, msg), 10)
        self.get_logger().info(
            f'Relaying {input_topic} -> {output_topic} only while forward_enabled and a '
            f'goal is active on {status_topics}.')

    def _status_callback(self, topic: str, msg: GoalStatusArray) -> None:
        self._active[topic] = any(g.status in _ACTIVE for g in msg.status_list)

    def _command_callback(self, msg: JointState) -> None:
        if not self.get_parameter('forward_enabled').get_parameter_value().bool_value:
            return  # real mode (or app disabled it) — discard, don't feed Isaac
        if any(self._active.values()):
            self._pub.publish(msg)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-topic', default='/isaac_joint_commands_raw')
    parser.add_argument('--output-topic', default='/isaac_joint_commands')
    parser.add_argument(
        '--status-topic', action='append', default=None,
        help='Controller action status topic to watch (repeatable). '
             'Defaults to scaled + joint_trajectory_controller.')
    # Accepted for backward-compat with the launch invocation; unused.
    parser.add_argument('--controller', default='')
    parser.add_argument('--state-topic', default='')
    parser.add_argument('--state-timeout', type=float, default=0.0)
    parser.add_argument('--resume-cooldown', type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    status_topics = args.status_topic or [
        '/scaled_joint_trajectory_controller/follow_joint_trajectory/_action/status',
        '/joint_trajectory_controller/follow_joint_trajectory/_action/status',
    ]
    rclpy.init(args=None)
    node = IsaacJointCommandRelay(args.input_topic, args.output_topic, status_topics)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
