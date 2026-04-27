#!/usr/bin/env python3
"""
저장된 trajectory CSV를 ROS2 FollowJointTrajectory action으로 전송

plan_motion.py가 생성한 trajectory.csv를 읽어서 로봇에 전송한다.

사용법:
    uv run scripts/pipeline/publish_trajectory.py --object sample --num-viewpoints 124
    uv run scripts/pipeline/publish_trajectory.py --csv data/sample/trajectory/124/trajectory.csv

    uv run scripts/pipeline/publish_trajectory.py --csv data/sample/trajectory/124/trajectory_dp_s010.csv
"""

import argparse
import csv
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


def load_trajectory_csv(csv_path: str) -> np.ndarray:
    """CSV에서 joint trajectory를 로드. 헤더에 prefix(예: 'ur20_')가 있어도 동작.

    Returns:
        solutions: (N, 6) joint angles in radians
    """
    solutions = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        col_map = {}
        for name in JOINT_NAMES:
            matches = [c for c in reader.fieldnames if c.endswith(name)]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one column ending with '{name}', "
                    f"found {matches} in {reader.fieldnames}"
                )
            col_map[name] = matches[0]

        for row in reader:
            q = [float(row[col_map[name]]) for name in JOINT_NAMES]
            solutions.append(q)
    return np.array(solutions, dtype=np.float64)


class TrajectoryPublisher(Node):
    def __init__(self, solutions: np.ndarray):
        super().__init__("publish_trajectory")

        self._action_client = ActionClient(
            self, FollowJointTrajectory, CONTROLLER_TOPIC
        )
        self._solutions = solutions
        self._current_joint_positions = None

        self._js_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_state_cb, 10
        )
        self._startup_timer = self.create_timer(1.0, self._on_startup)

    def _joint_state_cb(self, msg):
        positions = {}
        for name, pos in zip(msg.name, msg.position):
            positions[name] = pos
        if all(n in positions for n in JOINT_NAMES):
            self._current_joint_positions = [positions[n] for n in JOINT_NAMES]

    def _on_startup(self):
        if self._current_joint_positions is None:
            self.get_logger().info("Waiting for /joint_states...", once=True)
            return

        self._startup_timer.cancel()

        self.get_logger().info("Waiting for action server...")
        if not self._action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                f"Action server {CONTROLLER_TOPIC} not available. "
                "Is ur_robot_driver running?"
            )
            raise SystemExit(1)

        self._send_trajectory()

    def _send_trajectory(self):
        MAX_STEP_RAD = 0.1
        MAX_JOINT_VEL = 2.0

        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES

        # Interpolate between waypoints
        all_waypoints = []
        prev = np.array(self._current_joint_positions)

        for q in self._solutions:
            diff = q - prev
            max_diff = np.max(np.abs(diff))
            if max_diff <= MAX_STEP_RAD:
                interp_points = [q.tolist()]
            else:
                n_steps = int(np.ceil(max_diff / MAX_STEP_RAD))
                interp_points = []
                for s in range(1, n_steps + 1):
                    alpha = s / n_steps
                    interp_points.append((prev + alpha * diff).tolist())

            for wp in interp_points:
                wp_arr = np.array(wp)
                md = np.max(np.abs(wp_arr - prev))
                all_waypoints.append((wp, md))
                prev = wp_arr

        # First point: current position at t=0
        pt0 = JointTrajectoryPoint()
        pt0.positions = self._current_joint_positions
        pt0.velocities = [0.0] * len(JOINT_NAMES)
        pt0.time_from_start = Duration(sec=0, nanosec=0)
        traj.points.append(pt0)

        # Add interpolated waypoints with time proportional to displacement
        cumulative_t = 0.0
        for q, max_diff in all_waypoints:
            seg_dt = max(max_diff / MAX_JOINT_VEL, 0.05)
            cumulative_t += seg_dt
            pt = JointTrajectoryPoint()
            pt.positions = q
            pt.velocities = [0.0] * len(JOINT_NAMES)
            pt.time_from_start = Duration(
                sec=int(cumulative_t),
                nanosec=int((cumulative_t - int(cumulative_t)) * 1e9),
            )
            traj.points.append(pt)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self.get_logger().info(
            f"Sending trajectory with {len(traj.points)} points "
            f"({len(self._solutions)} waypoints interpolated to "
            f"{len(all_waypoints)} steps, total time: {cumulative_t:.1f}s, "
            f"max_vel: {MAX_JOINT_VEL} rad/s)..."
        )

        future = self._action_client.send_goal_async(
            goal, feedback_callback=self._feedback_cb
        )
        future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected by controller!")
            raise SystemExit(1)

        self.get_logger().info("Goal accepted. Executing trajectory...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _feedback_cb(self, feedback_msg):
        pass

    def _result_cb(self, future):
        result = future.result().result
        self.get_logger().info(f"Trajectory execution complete (error_code: {result.error_code})")
        raise SystemExit(0)


def main():
    parser = argparse.ArgumentParser(description="저장된 trajectory CSV를 ROS2로 전송")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", type=str, help="CSV 파일 경로 (직접 지정)")
    group.add_argument("--object", type=str, help="오브젝트 이름 (--num-viewpoints 필요)")
    parser.add_argument("--num-viewpoints", type=int, help="뷰포인트 수")
    args = parser.parse_args()

    if args.object:
        if args.num_viewpoints is None:
            parser.error("--object 사용 시 --num-viewpoints 필요")
        csv_path = str(config.get_trajectory_path(args.object, args.num_viewpoints, "trajectory.csv"))
    else:
        csv_path = args.csv

    if not Path(csv_path).exists():
        print(f"Error: CSV not found: {csv_path}")
        print("  plan_motion.py를 먼저 실행하세요.")
        return

    print(f"Loading trajectory from {csv_path}...")
    solutions = load_trajectory_csv(csv_path)
    print(f"  {len(solutions)} waypoints loaded")
    print(f"  Action server: {CONTROLLER_TOPIC}")

    rclpy.init()
    node = TrajectoryPublisher(solutions)
    try:
        print("  Spinning ROS2 node (Ctrl+C to stop)...")
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("  ROS2 shutdown complete.")


if __name__ == "__main__":
    main()
