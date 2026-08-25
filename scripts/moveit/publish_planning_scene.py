#!/usr/bin/env python3
"""씬 YAML 의 장애물을 MoveIt 플래닝 씬에 직접 넣는다 — cuMotion 없이.

왜 필요한가
-----------
장애물을 MoveIt 에 넣는 노드(``StaticPlanningSceneServer``)는 NVIDIA 의 launch 에서
cuMotion 플래너와 **같은 ComposableNodeContainer 안**에 산다. 그래서 cuMotion 이
못 뜨면 — 예를 들어 호스트 드라이버가 CUDA 13 을 못 받쳐서 — 장애물도 같이 사라진다.
계획기 하나가 빠지는 것으로 끝나지 않고 **로봇이 테이블을 못 보게** 된다.

이 노드는 그 결합을 끊는다. cuMotion 도, ``.scene`` 파일 포맷도 거치지 않고
``/apply_planning_scene`` 서비스로 CollisionObject 를 직접 넣는다. OMPL 만 쓰는
스택에서도 충돌 회피가 살아있다.

프레임: 씬 YAML 도 MoveIt 도 ``base_link`` 다(world 와 identity 로 묶여 있다 —
ur_camera.urdf.xacro). 좌표 변환이 없다.

주기적으로 다시 넣는 이유: ADD 는 같은 id 로 덮어쓰므로 몇 번을 보내도 결과가 같고,
move_group 이 늦게 뜨거나 재시작해도 씬이 저절로 복구된다.

사용법 (보통은 launch 가 띄운다):
    ros2 run ... 대신 직접:
    python3 scripts/moveit/publish_planning_scene.py --ros-args -p scene:=sim_default
"""

from __future__ import annotations

import os
import sys

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from common import config, scene_config  # noqa: E402

PLANNING_FRAME = "base_link"


def collision_objects(scene_name: str):
    """씬 YAML → CollisionObject 리스트. build_moveit_scene.build() 와 같은 소스를 읽는다."""
    config.load_scene(scene_name)
    config.sync_support_to_target()      # support 기둥은 물체 배치에서 파생된다
    objects = []
    for obstacle in config.OBSTACLES:
        pos, quat_wxyz, dims = scene_config.obstacle_obb(obstacle)
        w, x, y, z = (float(v) for v in quat_wxyz)

        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = (float(v) for v in pos)
        pose.orientation.x, pose.orientation.y = x, y
        pose.orientation.z, pose.orientation.w = z, w

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [float(v) for v in dims]

        obj = CollisionObject()
        obj.header.frame_id = PLANNING_FRAME
        obj.id = str(obstacle["name"])
        obj.primitives = [box]
        obj.primitive_poses = [pose]
        obj.operation = CollisionObject.ADD
        objects.append(obj)
    return objects


class PlanningScenePublisher(Node):

    def __init__(self):
        super().__init__("cell_planning_scene_publisher")
        self.declare_parameter("scene", "sim_default")
        self.declare_parameter("republish_period", 10.0)

        scene_name = self.get_parameter("scene").value
        if not scene_name:
            self.get_logger().warn("scene 이 비어 있다 — 장애물 없이 계획한다")
            self._objects = []
        else:
            self._objects = collision_objects(scene_name)
            names = ", ".join(o.id for o in self._objects)
            self.get_logger().info(
                f"씬 '{scene_name}': 장애물 {len(self._objects)}개 [{names}] "
                f"frame={PLANNING_FRAME}")

        self._client = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self._applied = False
        self._period = float(self.get_parameter("republish_period").value)
        self._ticks = 0
        # 1Hz 로 돌면서, 넣기 전에는 매초 재시도하고 넣은 뒤에는 period 마다만 다시 넣는다.
        # move_group 이 아직 없으면 조용히 넘어간다 — launch 순서에 의존하지 않는다.
        self._timer = self.create_timer(1.0, self._tick)

    def _tick(self):
        if not self._objects:
            return
        if self._applied:
            if self._period <= 0:
                return
            self._ticks += 1
            if self._ticks < self._period:
                return
            self._ticks = 0
        if not self._client.service_is_ready():
            if not self._applied:
                self.get_logger().info("/apply_planning_scene 대기 중…", once=True)
            return

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = self._objects
        future = self._client.call_async(ApplyPlanningScene.Request(scene=scene))
        future.add_done_callback(self._on_applied)

    def _on_applied(self, future):
        try:
            success = future.result().success
        except Exception as exc:                       # noqa: BLE001
            self.get_logger().error(f"apply_planning_scene 실패: {exc}")
            return
        if success and not self._applied:
            self.get_logger().info(
                f"장애물 {len(self._objects)}개를 플래닝 씬에 넣었다 (cuMotion 불필요)")
        elif not success:
            self.get_logger().error("apply_planning_scene 이 success=False 를 돌려줬다")
        self._applied = success
        # 한 번만 넣기로 했으면 타이머를 더 돌릴 이유가 없다.
        if success and self._period <= 0:
            self._timer.cancel()


def main():
    rclpy.init()
    node = PlanningScenePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
