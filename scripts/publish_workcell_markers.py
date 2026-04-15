#!/usr/bin/env python3
"""
RViz에 워크셀 시각화 Marker 발행.

config.py에 정의된 물체, 테이블, 벽, 로봇 마운트를 RViz에 표시한다.
대상 물체는 실제 mesh 파일(source.obj)을 사용.

사용법:
    # 터미널에서 실행 (ur_robot_driver 실행 후)
    uv run scripts/publish_workcell_markers.py --object sample

    # RViz에서: Add → By topic → /workcell_markers → MarkerArray
"""

import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import config


class WorkcellMarkerPublisher(Node):
    def __init__(self, object_name):
        super().__init__("workcell_marker_publisher")
        self._object_name = object_name

        # Latched (transient local) so RViz picks up markers even if it starts later
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._pub = self.create_publisher(MarkerArray, "/workcell_markers", qos)

        # Publish once after short delay, then periodically (1 Hz) to keep alive
        self._timer = self.create_timer(1.0, self._publish)

    def _quat_wxyz_to_msg(self, q_wxyz):
        """Convert [w, x, y, z] to geometry_msgs Quaternion fields."""
        from geometry_msgs.msg import Quaternion
        return Quaternion(x=float(q_wxyz[1]), y=float(q_wxyz[2]),
                          z=float(q_wxyz[3]), w=float(q_wxyz[0]))

    def _make_cuboid(self, marker_id, name, position, dimensions, color_rgba):
        """Create a CUBE marker."""
        m = Marker()
        m.header.frame_id = "base_link"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "workcell"
        m.id = marker_id
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = float(position[0])
        m.pose.position.y = float(position[1])
        m.pose.position.z = float(position[2])
        m.pose.orientation.w = 1.0
        m.scale = Vector3(x=float(dimensions[0]),
                          y=float(dimensions[1]),
                          z=float(dimensions[2]))
        m.color = ColorRGBA(r=color_rgba[0], g=color_rgba[1],
                            b=color_rgba[2], a=color_rgba[3])
        m.lifetime.sec = 0  # persistent
        return m

    def _make_mesh_marker(self, marker_id, mesh_path, position, rotation_wxyz, color_rgba):
        """Create a MESH_RESOURCE marker."""
        m = Marker()
        m.header.frame_id = "base_link"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "workcell"
        m.id = marker_id
        m.type = Marker.MESH_RESOURCE
        m.action = Marker.ADD
        m.mesh_resource = f"file://{mesh_path}"
        m.pose.position.x = float(position[0])
        m.pose.position.y = float(position[1])
        m.pose.position.z = float(position[2])
        m.pose.orientation = self._quat_wxyz_to_msg(rotation_wxyz)
        m.scale = Vector3(x=1.0, y=1.0, z=1.0)  # mesh is already in meters
        m.color = ColorRGBA(r=color_rgba[0], g=color_rgba[1],
                            b=color_rgba[2], a=color_rgba[3])
        m.mesh_use_embedded_materials = True
        m.lifetime.sec = 0
        return m

    def _publish(self):
        markers = MarkerArray()
        marker_id = 0

        # --- Target object (mesh) ---
        mesh_path = config.get_mesh_path(self._object_name, mesh_type="source")
        if mesh_path.exists():
            markers.markers.append(self._make_mesh_marker(
                marker_id, str(mesh_path.resolve()),
                config.TARGET_OBJECT["position"],
                config.TARGET_OBJECT["rotation"],
                [0.8, 0.6, 0.4, 0.9],
            ))
        else:
            self.get_logger().warn(f"Mesh not found: {mesh_path}", once=True)
        marker_id += 1

        # --- Table ---
        markers.markers.append(self._make_cuboid(
            marker_id, config.TABLE["name"],
            config.TABLE["position"], config.TABLE["dimensions"],
            [0.6, 0.5, 0.4, 0.5],
        ))
        marker_id += 1

        # --- Walls ---
        for wall in config.WALLS:
            markers.markers.append(self._make_cuboid(
                marker_id, wall["name"],
                wall["position"], wall["dimensions"],
                [0.3, 0.3, 0.3, 0.2],
            ))
            marker_id += 1

        # --- Robot mount ---
        markers.markers.append(self._make_cuboid(
            marker_id, config.ROBOT_MOUNT["name"],
            config.ROBOT_MOUNT["position"], config.ROBOT_MOUNT["dimensions"],
            [0.4, 0.4, 0.4, 0.6],
        ))
        marker_id += 1

        self._pub.publish(markers)
        self.get_logger().info(
            f"Published {len(markers.markers)} markers to /workcell_markers",
            once=True,
        )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Publish workcell markers to RViz")
    parser.add_argument("--object", type=str, default="sample", help="Object name")
    args = parser.parse_args()

    rclpy.init()
    node = WorkcellMarkerPublisher(args.object)
    try:
        print(f"Publishing workcell markers (object: {args.object}). Ctrl+C to stop.")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
