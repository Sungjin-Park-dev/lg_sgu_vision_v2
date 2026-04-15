#!/usr/bin/env python3
"""
ROS2 토픽으로 검사 궤적을 moveit으로 전송
로봇 컨트롤은 moveit을 사용하여 이루어짐

사용법:
    # 전체 검사 궤적
    omni_python scripts/4_publish_trajectory.py \
        --object sample \
        --num_viewpoints 163

    # Tilt 궤적
    omni_python scripts/4_publish_trajectory.py \
        --object sample \
        --num_viewpoints 163 \
        --tilt

경로는 자동으로 생성됩니다:
- 전체 검사: data/{object}/trajectory/{num_viewpoints}/trajectory.csv
- Tilt: data/{object}/trajectory/{num_viewpoints}/tilt_trajectory.csv
"""

import csv
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

# Add parent directory to path for common module
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import config


class JointTrajectoryPublisher(Node):
    def __init__(self, csv_path: str, dt: float = 0.01):
        super().__init__('joint_trajectory_publisher')

        self.dt = dt  # Time step between points (seconds)

        # Load trajectory from CSV and auto-detect joint names from header
        self.trajectory_points, self.joint_names = self.load_csv(csv_path)
        self.get_logger().info(f'Loaded {len(self.trajectory_points)} points (dt={dt}s, total={len(self.trajectory_points)*dt:.2f}s)')
        self.get_logger().info(f'Joint names: {self.joint_names}')

        # Publisher with QoS settings to prevent message caching
        # VOLATILE: Don't store messages for late-joining subscribers
        # KEEP_LAST with depth=1: Only keep the most recent message
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,  # Don't cache messages
            history=HistoryPolicy.KEEP_LAST,
            depth=1  # Only keep latest message
        )

        self.pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_from_csv',
            qos_profile
        )

        # Publish once after a short delay to ensure connections are established
        self.publish_timer = self.create_timer(1.0, self.publish_trajectory)
        self.shutdown_timer = None
        self.published = False

    def load_csv(self, csv_path: str) -> tuple:
        """Load trajectory points from CSV file and auto-detect joint names.

        Returns:
            tuple: (points, joint_names) where points is list of joint positions
                   and joint_names is list of joint column names from CSV header
        """
        points = []
        joint_names = None

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)

            # Auto-detect joint columns from header (columns containing 'joint')
            if reader.fieldnames:
                joint_names = [col for col in reader.fieldnames if 'joint' in col.lower()]

            if not joint_names:
                raise ValueError(f"No joint columns found in CSV header: {reader.fieldnames}")

            # Read trajectory data
            for row in reader:
                positions = [float(row[col]) for col in joint_names]
                points.append(positions)

        return points, joint_names

    def publish_trajectory(self):
        """Publish the full trajectory."""
        if self.published:
            return

        # 🚀 subscriber 연결될 때까지 대기
        while self.pub.get_subscription_count() == 0:
            self.get_logger().info("아직 Subscriber 연결 안 됨. 기다리는 중...")
            time.sleep(1.0)

        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = self.joint_names

        # Convert all CSV points to JointTrajectoryPoints
        for i, positions in enumerate(self.trajectory_points):
            point = JointTrajectoryPoint()
            point.positions = positions
            point.velocities = [0.0] * len(self.joint_names)

            # time_from_start = index * dt
            t = i * self.dt
            sec = int(t)
            nanosec = int((t - sec) * 1e9)
            point.time_from_start = Duration(sec=sec, nanosec=nanosec)

            msg.points.append(point)

        self.pub.publish(msg)
        self.get_logger().info(f'Published trajectory with {len(msg.points)} points')
        self.published = True

        # Cancel publish timer to stop repeating
        if self.publish_timer:
            self.publish_timer.cancel()

        # Keep node alive for a moment then shutdown
        self.shutdown_timer = self.create_timer(2.0, self.shutdown_node)

    def shutdown_node(self):
        """Properly shutdown the node."""
        self.get_logger().info('Trajectory published. Shutting down...')

        # Cancel all timers
        if self.publish_timer:
            self.publish_timer.cancel()
        if self.shutdown_timer:
            self.shutdown_timer.cancel()

        # Destroy publisher
        self.destroy_publisher(self.pub)

        # Trigger shutdown
        raise SystemExit


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(description='Publish joint trajectory from CSV')
    parser.add_argument(
        '--object',
        type=str,
        required=True,
        help="Object name for auto-path generation (e.g., 'sample', 'glass')"
    )
    parser.add_argument(
        '--num_viewpoints',
        type=int,
        required=True,
        help='Number of viewpoints'
    )
    parser.add_argument(
        '--dt',
        type=float,
        default=0.01,
        help='Time step between trajectory points in seconds (default: 0.01)'
    )
    parser.add_argument(
        '--tilt',
        action='store_true',
        help='Use tilt trajectory (tilt_trajectory.csv) instead of regular trajectory',
        default=False
    )
    parser.add_argument(
        '--trajectory-file',
        type=str,
        default=None,
        help='Custom trajectory filename (overrides --tilt if specified)'
    )
    parsed_args, remaining = parser.parse_known_args()

    # Determine trajectory filename
    if parsed_args.trajectory_file is not None:
        # Custom filename specified
        trajectory_filename = parsed_args.trajectory_file
    elif parsed_args.tilt:
        # Tilt mode enabled
        trajectory_filename = "tilt_trajectory.csv"
    else:
        # Default trajectory
        trajectory_filename = "trajectory.csv"

    # Auto-generate trajectory path
    csv_path = str(config.get_trajectory_path(parsed_args.object, parsed_args.num_viewpoints, trajectory_filename))

    print(f"Object: {parsed_args.object}")
    print(f"Num viewpoints: {parsed_args.num_viewpoints}")
    print(f"Trajectory path: {csv_path}")

    rclpy.init(args=remaining)
    node = JointTrajectoryPublisher(csv_path, parsed_args.dt)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
