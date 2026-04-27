#!/usr/bin/env python3
"""
로봇을 시작 자세(ROBOT_START_STATE)로 이동시키는 스크립트.

scaled_joint_trajectory_controller의 FollowJointTrajectory action을 사용하여
현재 위치에서 시작 자세까지 보간된 궤적을 전송한다.

사용법:
    uv run scripts/ros2/move_to_start.py
    uv run scripts/ros2/move_to_start.py --duration 5.0   # 5초에 걸쳐 이동
"""

import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import config

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

CONTROLLER_TOPIC = "/scaled_joint_trajectory_controller/follow_joint_trajectory"


class MoveToStartNode(Node):
    def __init__(self, target, duration, max_vel):
        super().__init__("move_to_start")
        self._target = target
        self._duration = duration
        self._max_vel = max_vel
        self._current_positions = None

        self._action_client = ActionClient(
            self, FollowJointTrajectory, CONTROLLER_TOPIC
        )
        self._js_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_state_cb, 10
        )
        self._timer = self.create_timer(1.0, self._on_startup)

    def _joint_state_cb(self, msg):
        positions = {}
        for name, pos in zip(msg.name, msg.position):
            positions[name] = pos
        if all(n in positions for n in JOINT_NAMES):
            self._current_positions = np.array([positions[n] for n in JOINT_NAMES])

    def _on_startup(self):
        if self._current_positions is None:
            self.get_logger().info("Waiting for /joint_states...", once=True)
            return

        self._timer.cancel()

        self.get_logger().info("Waiting for action server...")
        if not self._action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                f"Action server {CONTROLLER_TOPIC} not available. "
                "Is ur_robot_driver running?"
            )
            raise SystemExit(1)

        diff = np.abs(self._target - self._current_positions)
        max_diff = np.max(diff)
        self.get_logger().info(
            f"Current: [{', '.join(f'{v:.3f}' for v in self._current_positions)}]"
        )
        self.get_logger().info(
            f"Target:  [{', '.join(f'{v:.3f}' for v in self._target)}]"
        )
        self.get_logger().info(f"Max joint diff: {np.rad2deg(max_diff):.1f} deg")

        if max_diff < 0.01:
            self.get_logger().info("Already at start position.")
            raise SystemExit(0)

        self._send_trajectory(max_diff)

    def _send_trajectory(self, max_diff):
        MAX_STEP_RAD = 0.1

        # Interpolate
        q_from = self._current_positions
        q_to = self._target
        diff = q_to - q_from
        n_steps = max(int(np.ceil(np.max(np.abs(diff)) / MAX_STEP_RAD)), 1)

        # Compute total duration: max(user-specified, velocity-limited)
        vel_limited_duration = max_diff / self._max_vel
        total_duration = max(self._duration, vel_limited_duration)

        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES

        # t=0: current position
        pt0 = JointTrajectoryPoint()
        pt0.positions = q_from.tolist()
        pt0.velocities = [0.0] * len(JOINT_NAMES)
        pt0.time_from_start = Duration(sec=0, nanosec=0)
        traj.points.append(pt0)

        step_dt = total_duration / n_steps
        for i in range(1, n_steps + 1):
            alpha = i / n_steps
            q = (q_from + alpha * diff).tolist()
            t = i * step_dt
            pt = JointTrajectoryPoint()
            pt.positions = q
            pt.velocities = [0.0] * len(JOINT_NAMES)
            pt.time_from_start = Duration(
                sec=int(t), nanosec=int((t - int(t)) * 1e9)
            )
            traj.points.append(pt)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self.get_logger().info(
            f"Sending {len(traj.points)} points over {total_duration:.1f}s..."
        )

        future = self._action_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected!")
            raise SystemExit(1)
        self.get_logger().info("Goal accepted. Moving to start position...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _result_cb(self, future):
        result = future.result().result
        if result.error_code == 0:
            self.get_logger().info("Reached start position.")
        else:
            self.get_logger().error(f"Failed (error_code: {result.error_code})")
        raise SystemExit(0)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Move robot to start position")
    parser.add_argument(
        "--duration", type=float, default=5.0,
        help="Minimum duration for the motion (seconds, default: 5.0)",
    )
    parser.add_argument(
        "--max-vel", type=float, default=0.5,
        help="Max joint velocity (rad/s, default: 0.5)",
    )
    args = parser.parse_args()

    target = np.array(config.ROBOT_START_STATE, dtype=np.float64)

    print(f"Target (ROBOT_START_STATE): {np.rad2deg(target).astype(int).tolist()} deg")
    print(f"Duration: >= {args.duration}s, Max vel: {args.max_vel} rad/s")

    rclpy.init()
    node = MoveToStartNode(target, args.duration, args.max_vel)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
