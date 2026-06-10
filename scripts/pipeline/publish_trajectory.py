#!/usr/bin/env python3
"""
저장된 trajectory CSV를 ROS2 FollowJointTrajectory action으로 전송

plan_trajectory.py가 생성한 trajectory.csv의 time 컬럼을 보존해 로봇에 전송한다.

사용법:
    uv run scripts/pipeline/publish_trajectory.py --csv data/sample/trajectory/124/trajectory_dp_ee_s0010.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

CONTROLLER_TOPIC = "/scaled_joint_trajectory_controller/follow_joint_trajectory"
MAX_STEP_RAD = 0.1
APPROACH_MAX_JOINT_VEL_RAD_S = 0.5
MIN_APPROACH_TIME_S = 0.5


def load_trajectory_csv(csv_path: str):
    """CSV에서 joint trajectory를 로드. 헤더에 prefix(예: 'ur20_')가 있어도 동작.

    Returns:
        solutions: (N, 6) joint angles in radians
        times: (N,) time from trajectory start in seconds
    """
    solutions = []
    times = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        if "time" not in reader.fieldnames:
            raise ValueError(f"CSV must include a 'time' column: {reader.fieldnames}")
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
            times.append(float(row["time"]))
            q = [float(row[col_map[name]]) for name in JOINT_NAMES]
            solutions.append(q)

    solutions = np.array(solutions, dtype=np.float64)
    times = np.array(times, dtype=np.float64)
    if len(times) != len(solutions):
        raise ValueError("CSV time and joint row counts do not match")
    if len(solutions) == 0:
        raise ValueError("CSV contains no trajectory rows")
    if len(times) > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("CSV time column must be strictly increasing")
    return solutions, times


class TrajectoryPublisher(Node):
    def __init__(self, solutions: np.ndarray, times: np.ndarray):
        super().__init__("publish_trajectory")

        self._action_client = ActionClient(
            self, FollowJointTrajectory, CONTROLLER_TOPIC
        )
        self._solutions = solutions
        self._times = times
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
        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES

        csv_times = self._times - self._times[0]
        first_q = self._solutions[0]
        current_q = np.array(self._current_joint_positions)
        start_diff = np.max(np.abs(first_q - current_q))
        approach_time = 0.0
        points_with_time = []

        if start_diff > 1e-4:
            approach_time = max(
                start_diff / APPROACH_MAX_JOINT_VEL_RAD_S,
                MIN_APPROACH_TIME_S,
            )
            n_steps = int(np.ceil(start_diff / MAX_STEP_RAD))
            for s in range(1, n_steps + 1):
                alpha = s / n_steps
                q = current_q + alpha * (first_q - current_q)
                points_with_time.append((q.tolist(), alpha * approach_time))
            start_index = 1
        else:
            start_index = 1

        # Interpolate between CSV waypoints while preserving CSV timing.
        prev = first_q
        prev_time = approach_time
        for i in range(start_index, len(self._solutions)):
            q = self._solutions[i]
            target_time = approach_time + csv_times[i]
            diff = q - prev
            max_diff = np.max(np.abs(diff))
            if max_diff <= MAX_STEP_RAD:
                interp_points = [(q.tolist(), target_time)]
            else:
                n_steps = int(np.ceil(max_diff / MAX_STEP_RAD))
                interp_points = []
                for s in range(1, n_steps + 1):
                    alpha = s / n_steps
                    wp = (prev + alpha * diff).tolist()
                    t = prev_time + alpha * (target_time - prev_time)
                    interp_points.append((wp, t))

            points_with_time.extend(interp_points)
            prev = q
            prev_time = target_time

        # Build positions + times arrays first (pt0 + all interpolated waypoints).
        positions = [list(self._current_joint_positions)]
        times = [0.0]
        for q, t in points_with_time:
            positions.append(list(q))
            times.append(float(t))

        positions = np.array(positions)
        times = np.array(times)

        # Finite-difference velocities (central diff interior, 0 at endpoints).
        # 컨트롤러가 cubic Hermite spline로 보간하여 C¹ 연속 → reversal 부드럽게.
        velocities = np.zeros_like(positions)
        if len(times) >= 3:
            dt_pair = (times[2:] - times[:-2])[:, None]
            velocities[1:-1] = (positions[2:] - positions[:-2]) / dt_pair

        for i in range(len(positions)):
            pt = JointTrajectoryPoint()
            pt.positions = positions[i].tolist()
            pt.velocities = velocities[i].tolist()
            t = times[i]
            pt.time_from_start = Duration(
                sec=int(t),
                nanosec=int((t - int(t)) * 1e9),
            )
            traj.points.append(pt)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self.get_logger().info(
            f"Sending trajectory with {len(traj.points)} points "
            f"({len(self._solutions)} waypoints interpolated to "
            f"{len(points_with_time)} steps, total time: {times[-1]:.1f}s, "
            f"csv duration: {csv_times[-1]:.1f}s, "
            f"approach time: {approach_time:.1f}s)..."
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
    parser.add_argument("--csv", type=str, required=True, help="CSV 파일 경로")
    args = parser.parse_args()
    csv_path = args.csv

    if not Path(csv_path).exists():
        print(f"Error: CSV not found: {csv_path}")
        print("  plan_trajectory.py를 먼저 실행하세요.")
        return

    print(f"Loading trajectory from {csv_path}...")
    solutions, times = load_trajectory_csv(csv_path)
    print(f"  {len(solutions)} waypoints loaded")
    print(f"  CSV duration: {times[-1] - times[0]:.1f}s")
    print(f"  Action server: {CONTROLLER_TOPIC}")

    rclpy.init()
    node = TrajectoryPublisher(solutions, times=times)
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
