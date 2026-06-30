#!/usr/bin/env python3
"""
저장된 trajectory CSV를 로봇에 전송.

두 가지 대상(--target):
  controller : ros2_control 컨트롤러로 FollowJointTrajectory action 전송 (실로봇).
               trajectory.csv의 time 컬럼을 보존해 컨트롤러가 spline 보간하여 실행.
  isaac      : /isaac_joint_commands 로 JointState를 시간에 맞춰 직접 스트리밍 (Isaac sim).
               셸2(ros2_control 스택) 없이도 Isaac sim 로봇을 구동 — Inspection을 셸2와
               독립적으로 만들기 위함.

사용법:
    uv run scripts/core/publish_trajectory.py --csv <csv> --target isaac
    uv run scripts/core/publish_trajectory.py --csv <csv> --target controller
"""

import argparse
import csv
import time
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

# Inspection publishes to joint_trajectory_controller (kept separate from
# scaled_joint_trajectory_controller, which MoveIt uses) so the two are gated
# independently by pipeline mode.
CONTROLLER_TOPIC = "/joint_trajectory_controller/follow_joint_trajectory"
# Direct-to-Isaac (sim) topics — match scripts/common/config.py.
ISAAC_CMD_TOPIC = "/isaac_joint_commands"
ISAAC_STATE_TOPIC = "/isaac_joint_states"

MAX_STEP_RAD = 0.1
APPROACH_MAX_JOINT_VEL_RAD_S = 0.5
MIN_APPROACH_TIME_S = 0.5
# Fixed publish rate for direct-to-Isaac streaming (smooth position targets).
STREAM_HZ = 100.0


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


def build_interpolated_points(solutions: np.ndarray, csv_times: np.ndarray,
                              current_q):
    """현재 자세 → 첫 점(approach) + CSV 구간을 MAX_STEP_RAD 이하로 보간.

    CSV 타이밍을 보존한다. Returns (positions (M,6), times (M,)) — 첫 행은 현재 자세, t=0.
    controller/isaac 두 경로가 공유한다.
    """
    csv_times = csv_times - csv_times[0]
    first_q = solutions[0]
    current_q = np.array(current_q, dtype=np.float64)
    start_diff = np.max(np.abs(first_q - current_q))
    approach_time = 0.0
    points_with_time = []

    if start_diff > 1e-4:
        approach_time = max(
            start_diff / APPROACH_MAX_JOINT_VEL_RAD_S, MIN_APPROACH_TIME_S)
        n_steps = int(np.ceil(start_diff / MAX_STEP_RAD))
        for s in range(1, n_steps + 1):
            alpha = s / n_steps
            q = current_q + alpha * (first_q - current_q)
            points_with_time.append((q.tolist(), alpha * approach_time))

    prev = first_q
    prev_time = approach_time
    for i in range(1, len(solutions)):
        q = solutions[i]
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

    positions = [list(current_q)]
    times = [0.0]
    for q, t in points_with_time:
        positions.append(list(q))
        times.append(float(t))
    return np.array(positions), np.array(times)


class TrajectoryPublisher(Node):
    """--target controller: FollowJointTrajectory action으로 ros2_control 컨트롤러에 전송."""

    def __init__(self, solutions: np.ndarray, times: np.ndarray):
        super().__init__("publish_trajectory")
        self._action_client = ActionClient(self, FollowJointTrajectory, CONTROLLER_TOPIC)
        self._solutions = solutions
        self._times = times
        self._current_joint_positions = None
        self._js_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_state_cb, 10)
        self._startup_timer = self.create_timer(1.0, self._on_startup)

    def _joint_state_cb(self, msg):
        positions = {n: p for n, p in zip(msg.name, msg.position)}
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
                f"Action server {CONTROLLER_TOPIC} not available. Is the robot stack running?")
            raise SystemExit(1)
        self._send_trajectory()

    def _send_trajectory(self):
        positions, times = build_interpolated_points(
            self._solutions, self._times, self._current_joint_positions)

        # Finite-difference velocities (central diff interior, 0 at endpoints) so the
        # controller's cubic Hermite spline is C¹ continuous (smooth reversals). The
        # zero endpoints also satisfy controllers that reject nonzero end velocity.
        velocities = np.zeros_like(positions)
        if len(times) >= 3:
            dt_pair = (times[2:] - times[:-2])[:, None]
            velocities[1:-1] = (positions[2:] - positions[:-2]) / dt_pair

        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES
        for i in range(len(positions)):
            pt = JointTrajectoryPoint()
            pt.positions = positions[i].tolist()
            pt.velocities = velocities[i].tolist()
            t = times[i]
            pt.time_from_start = Duration(sec=int(t), nanosec=int((t - int(t)) * 1e9))
            traj.points.append(pt)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        self.get_logger().info(
            f"Sending trajectory with {len(traj.points)} points "
            f"(total time: {times[-1]:.1f}s) to {CONTROLLER_TOPIC}...")
        future = self._action_client.send_goal_async(goal, feedback_callback=self._feedback_cb)
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


class IsaacStreamPublisher(Node):
    """--target isaac: /isaac_joint_commands 로 JointState를 시간에 맞춰 직접 스트리밍.

    셸2 없이 Isaac sim 로봇(/MoveItGraph가 /isaac_joint_commands 구독)을 구동한다.
    현재 자세는 Isaac이 발행하는 /isaac_joint_states에서 읽는다.
    """

    def __init__(self, solutions: np.ndarray, times: np.ndarray):
        super().__init__("publish_trajectory_isaac")
        self._solutions = solutions
        self._times = times
        self._current = None
        self._pub = self.create_publisher(JointState, ISAAC_CMD_TOPIC, 10)
        self.create_subscription(JointState, ISAAC_STATE_TOPIC, self._state_cb, 10)

    def _state_cb(self, msg):
        positions = {n: p for n, p in zip(msg.name, msg.position)}
        if all(n in positions for n in JOINT_NAMES):
            self._current = [positions[n] for n in JOINT_NAMES]

    def run(self) -> int:
        # Wait for Isaac's current pose.
        t0 = time.monotonic()
        while self._current is None and time.monotonic() - t0 < 5.0:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._current is None:
            self.get_logger().error(
                f"No {ISAAC_STATE_TOPIC}; is Isaac running in sim mode and playing?")
            return 1

        positions, times = build_interpolated_points(
            self._solutions, self._times, self._current)

        # Resample to a fixed high rate so Isaac gets a SMOOTH stream of position
        # targets. Publishing only the coarse waypoints (<=0.1 rad apart) made the
        # ArticulationController jump target-to-target → visibly choppy. Linear
        # interpolation at STREAM_HZ removes that.
        total = float(times[-1])
        dt = 1.0 / STREAM_HZ
        ts = np.arange(0.0, total, dt) if total > 0 else np.array([0.0])
        ts = np.append(ts, total)  # ensure the exact final pose is sent
        resampled = np.column_stack(
            [np.interp(ts, times, positions[:, j]) for j in range(positions.shape[1])])
        self.get_logger().info(
            f"Streaming {len(resampled)} pts to {ISAAC_CMD_TOPIC} at {STREAM_HZ:.0f}Hz "
            f"over {total:.1f}s (no controller needed)...")

        start = time.monotonic()
        for i in range(len(resampled)):
            target = start + float(ts[i])
            now = time.monotonic()
            if now < target:
                time.sleep(target - now)
            msg = JointState()
            msg.name = JOINT_NAMES
            msg.position = [float(x) for x in resampled[i]]
            self._pub.publish(msg)
        self.get_logger().info("Streaming complete. Isaac holds final pose.")
        return 0


def main():
    parser = argparse.ArgumentParser(description="저장된 trajectory CSV를 로봇/Isaac에 전송")
    parser.add_argument("--csv", type=str, required=True, help="CSV 파일 경로")
    parser.add_argument("--target", choices=["controller", "isaac"], default="controller",
                        help="controller=실로봇 ros2_control, isaac=Isaac에 직접 스트리밍(sim)")
    args = parser.parse_args()
    csv_path = args.csv

    if not Path(csv_path).exists():
        print(f"Error: CSV not found: {csv_path}")
        print("  plan_trajectory.py를 먼저 실행하세요.")
        return

    print(f"Loading trajectory from {csv_path}...")
    solutions, times = load_trajectory_csv(csv_path)
    print(f"  {len(solutions)} waypoints loaded (CSV duration: {times[-1] - times[0]:.1f}s)")
    print(f"  target: {args.target}")

    rclpy.init()
    if args.target == "isaac":
        node = IsaacStreamPublisher(solutions, times=times)
        try:
            rc = node.run()
        except (KeyboardInterrupt, SystemExit):
            rc = 0
        finally:
            node.destroy_node()
            rclpy.shutdown()
        raise SystemExit(rc)

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
