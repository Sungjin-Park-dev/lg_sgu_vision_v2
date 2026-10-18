#!/usr/bin/env python3
"""PiPER 고정 자세 재방문 시험 — 정해 둔 관절값으로 MoveIt 계획·실행하고, 실제로 어디에 도착했는지 잰다.

촬영은 하지 않는다. 자동 촬영(스캔 앱) 전에 "정해 둔 자세로 보내면 그 자리에 다시 서는가"를 확인한다.
자세는 교시 촬영 run 의 관절값(capture.json)이나 YAML 목록에서 읽는다. 잰 것:
  관절 오차          목표 관절값 ↔ 정지 후 /feedback/joint_states
  카메라 자세 오차   같은 두 관절값의 FK @ tool_T_depth (손목 보정) — 스캔에서 실제로 중요한 값

제어 경로는 probe_piper_plan.py(agx_arm_ws)와 같다: /plan_kinematic_path → /execute_trajectory →
MoveIt mock 컨트롤러 → /control/joint_states → PiPER 드라이버. 물체는 충돌 모델에 없다 — 첫 실행은
저속으로, 비상정지를 손에 두고 본다. 충돌 장면은 테이블(--table-z)과 손목 D405 상자만 넣는다.

실행 전제(--execute, 실기): 안내서 5.1 CAN 제어 복구 → 5.2 드라이버(control_enabled:=false) →
5.4 MoveIt(follow:=true, auto_control_gate:=true). 스캔 앱과 함께 쓸 때(도착마다 보정·촬영 버튼)는
앱을 --allow-robot-control 로 켠다. 자세 사이에서 끝낼 때는 q — 움직이는 중 Ctrl+C 는 비상정지라 5.6 복구가 필요하다.

    # 컨테이너에서 (ROS + agx_arm_ws 를 source 한 뒤, .venv-piper 의 python)
    python scripts/core/realscan/pose_tour.py --from-run data/piper_test/scan/20261014_002105            # 계획만
    python scripts/core/realscan/pose_tour.py --from-run data/piper_test/scan/20261014_002105 --execute  # 자세마다 Enter
    # 하드웨어 없이 로직만: MoveIt demo follow:=false(드라이버 없음)를 띄우고 --mock
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from core.realscan.capture import atomic_json, transform  # noqa: E402
from core.realscan.piper import JOINT_NAMES, PiperFK  # noqa: E402

# 스캔 앱의 정지 판정(workcell/scans/piper_d405.yaml capture_guard)과 같은 기준
STABLE_S, STEP_RAD = 0.5, 0.002
# 실행 전 MoveIt mock 컨트롤러 ↔ 실제 팔 허용 차이 (probe_piper_plan.py 와 같다)
SYNC_RAD = 0.02
# 교시 관절값이 URDF 한계를 살짝 넘으면(관절 2·3 이 0 근처) 이만큼 안쪽으로 넣는다
LIMIT_MARGIN_RAD = 0.002
# 충돌 장면 — 테이블(윗면 z 는 --table-z), 손목 D405(42×42×23mm)+여유. 물체는 모델에 없다.
SCENE_TABLE, SCENE_CAMERA = "scan_table", "scan_d405"
CAMERA_BOX_M = (0.050, 0.050, 0.035)


def load_poses(args) -> tuple[np.ndarray, str]:
    if args.from_run:
        run = Path(args.from_run)
        run = run if run.is_absolute() else ROOT / run
        frames = json.loads((run / "capture.json").read_text())["frames"]
        return np.asarray([f["q"] for f in frames], dtype=np.float64), str(run.relative_to(ROOT))
    import yaml
    data = yaml.safe_load(Path(args.poses).read_text())
    return np.deg2rad(np.asarray(data["poses_deg"], dtype=np.float64)), str(args.poses)


def camera_error(fk, tool_T_depth, q_a, q_b) -> tuple[float, float]:
    """두 관절값의 depth 카메라 자세 차이 → (위치 mm, 회전 deg)."""
    A = fk.pose(q_a) @ tool_T_depth
    B = fk.pose(q_b) @ tool_T_depth
    D = np.linalg.inv(A) @ B
    return float(np.linalg.norm(A[:3, 3] - B[:3, 3]) * 1000), float(np.degrees(Rotation.from_matrix(D[:3, :3]).magnitude()))


class Tour:
    def __init__(self, feedback_topic, mock):
        import rclpy
        from moveit_msgs.action import ExecuteTrajectory
        from moveit_msgs.srv import GetMotionPlan
        from rclpy.action import ActionClient
        from rclpy.node import Node
        from sensor_msgs.msg import JointState

        from rclpy.signals import SignalHandlerOptions

        # rclpy 의 SIGINT 처리는 컨텍스트부터 닫아 실행 취소(cancel_goal)를 못 보낸다 — Ctrl+C 는 여기서 받는다.
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self.rclpy, self.mock = rclpy, mock
        self.node = Node("piper_pose_tour")
        self.history = deque(maxlen=400)          # (monotonic, q)
        self.control = None
        self.status = None
        self.node.create_subscription(JointState, feedback_topic, self._feedback, 10)
        if not mock:
            from agx_arm_msgs.msg import AgxArmStatus
            self.node.create_subscription(JointState, "/control/joint_states", self._control, 10)
            self.node.create_subscription(AgxArmStatus, "/feedback/arm_status",
                                          lambda m: setattr(self, "status", m), 10)
        self.moving = False                       # 실행을 보냈고 아직 정지를 확인하지 못했다
        self.sending = self.handle = None         # 진행 중인 MoveIt 실행 요청 / 실행
        self.plan_client = self.node.create_client(GetMotionPlan, "/plan_kinematic_path")
        self.exec_client = ActionClient(self.node, ExecuteTrajectory, "/execute_trajectory")
        self.GetMotionPlan, self.ExecuteTrajectory = GetMotionPlan, ExecuteTrajectory

    @staticmethod
    def _joints(msg):
        pos = dict(zip(msg.name, msg.position))
        if all(n in pos and np.isfinite(pos[n]) for n in JOINT_NAMES):
            return np.array([pos[n] for n in JOINT_NAMES])
        return None

    def _feedback(self, msg):
        q = self._joints(msg)
        if q is not None:
            self.history.append((time.monotonic(), q))

    def _control(self, msg):
        q = self._joints(msg)
        if q is not None:
            self.control = q

    def spin_until(self, cond, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            if cond():
                return True
        return False

    def current(self):
        # 서비스·액션을 기다리는 동안은 구독을 처리하지 않는다 — 읽기 전에 새 값을 받는다.
        self.spin_until(lambda: bool(self.history) and time.monotonic() - self.history[-1][0] < 0.1, 1.0)
        if not self.history or time.monotonic() - self.history[-1][0] > 0.5:
            raise RuntimeError("관절 피드백이 없거나 오래되었습니다")
        return self.history[-1][1].copy()

    def settled(self):
        now = time.monotonic()
        window = [q for t, q in self.history if t >= now - STABLE_S]
        if not self.history or self.history[0][0] > now - STABLE_S or len(window) < 5:
            return False
        return float(np.max(np.ptp(np.stack(window), axis=0))) <= STEP_RAD

    def preflight(self):
        if not self.spin_until(lambda: bool(self.history), 10):
            raise RuntimeError("관절 피드백이 들어오지 않습니다 — 드라이버/MoveIt·ROS_DOMAIN_ID 를 확인하세요")
        if not self.plan_client.wait_for_service(timeout_sec=5):
            raise RuntimeError("MoveIt 계획 서비스(/plan_kinematic_path)가 없습니다")

    def setup_scene(self, table_top_z, tool_T_depth):
        """충돌 장면: 테이블 상자 + 손목 D405 상자(link6 에 붙임). 물체는 넣지 않는다.

        MoveIt 은 장면을 기억한다 — 같은 id 를 지우고 다시 넣는다. RViz 에도 보인다.
        """
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene
        from moveit_msgs.srv import ApplyPlanningScene
        from shape_msgs.msg import SolidPrimitive

        def box(oid, frame, T, dims):
            o = CollisionObject()
            o.id, o.header.frame_id, o.operation = oid, frame, CollisionObject.ADD
            o.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(d) for d in dims])]
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = (float(v) for v in T[:3, 3])
            x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = (
                float(x), float(y), float(z), float(w))
            o.primitive_poses = [pose]
            return o

        client = self.node.create_client(ApplyPlanningScene, "/apply_planning_scene")
        if not client.wait_for_service(timeout_sec=5):
            raise RuntimeError("MoveIt 장면 서비스(/apply_planning_scene)가 없습니다")

        def apply(scene, required=True):
            fut = client.call_async(ApplyPlanningScene.Request(scene=scene))
            self.rclpy.spin_until_future_complete(self.node, fut, timeout_sec=5)
            ok = fut.done() and fut.result() is not None and fut.result().success
            if required and not ok:
                raise RuntimeError("MoveIt 충돌 장면 설정 실패")

        clear = PlanningScene(is_diff=True)
        o = CollisionObject(); o.id, o.header.frame_id, o.operation = SCENE_TABLE, "base_link", CollisionObject.REMOVE
        clear.world.collision_objects = [o]
        a = AttachedCollisionObject(link_name="link6")
        a.object.id, a.object.operation = SCENE_CAMERA, CollisionObject.REMOVE
        clear.robot_state.attached_collision_objects = [a]
        clear.robot_state.is_diff = True
        apply(clear, required=False)                      # 처음엔 지울 것이 없어 실패로 답한다

        scene = PlanningScene(is_diff=True)
        T = np.eye(4); T[2, 3] = table_top_z - 0.025
        scene.world.collision_objects = [box(SCENE_TABLE, "base_link", T, [1.5, 1.5, 0.05])]
        off = np.eye(4); off[2, 3] = -0.010              # depth 광학 프레임(z 앞): 몸통은 렌즈 뒤
        cam = AttachedCollisionObject(link_name="link6",
                                      object=box(SCENE_CAMERA, "link6", tool_T_depth @ off, CAMERA_BOX_M))
        cam.touch_links = ["link6", "link5"]
        scene.robot_state.attached_collision_objects = [cam]
        scene.robot_state.is_diff = True
        apply(scene)

    def check_can_execute(self):
        if self.mock:
            return
        self.spin_until(lambda: self.status is not None and self.control is not None, 3)
        s = self.status
        if s is None or s.ctrl_mode != 1 or s.arm_status != 0:
            raise RuntimeError(f"PiPER 가 정상 CAN 제어 상태가 아닙니다(교시 해제·5.1 복구 필요): {s}")
        if self.control is None:
            raise RuntimeError("MoveIt mock 컨트롤러 관절값(/control/joint_states)이 없습니다")
        gap = float(np.max(np.abs(self.control - self.current())))
        if gap > SYNC_RAD:
            raise RuntimeError(f"MoveIt mock 과 실제 팔이 {np.degrees(gap):.2f}° 어긋났습니다 — "
                               "실제 팔이 아직 도착 전이거나 MoveIt 초기값 동기화가 필요합니다")

    def plan(self, goal_q, scale):
        from moveit_msgs.msg import Constraints, JointConstraint
        req = self.GetMotionPlan.Request()
        m = req.motion_plan_request
        m.group_name, m.pipeline_id = "arm", "ompl"
        m.allowed_planning_time, m.num_planning_attempts = 5.0, 1
        m.max_velocity_scaling_factor = m.max_acceleration_scaling_factor = float(scale)
        m.start_state.is_diff = True                      # move_group 의 현재 상태(follow:=true 면 실제 팔)
        g = Constraints()
        for n, v in zip(JOINT_NAMES, goal_q):
            g.joint_constraints.append(JointConstraint(joint_name=n, position=float(v), tolerance_above=1e-3,
                                                       tolerance_below=1e-3, weight=1.0))
        m.goal_constraints = [g]
        fut = self.plan_client.call_async(req)
        self.rclpy.spin_until_future_complete(self.node, fut, timeout_sec=15)
        if not fut.done() or fut.result() is None:
            raise RuntimeError("MoveIt 계획 서비스 응답 없음")
        res = fut.result().motion_plan_response
        if res.error_code.val != 1 or not res.trajectory.joint_trajectory.points:
            raise RuntimeError(f"계획 실패 (MoveIt error_code {res.error_code.val})")
        return res

    def execute(self, trajectory, timeout):
        """MoveIt 실행 — 끝나면 돌아온다. 중간에 예외(Ctrl+C 포함)가 나면 호출자가 stop() 을 부른다."""
        if not self.exec_client.wait_for_server(timeout_sec=5):
            raise RuntimeError("MoveIt 실행 액션(/execute_trajectory)이 없습니다")
        self.moving = True
        self.sending = self.exec_client.send_goal_async(self.ExecuteTrajectory.Goal(trajectory=trajectory))
        self.rclpy.spin_until_future_complete(self.node, self.sending, timeout_sec=5)
        sent, self.sending = self.sending, None
        if not sent.done() or sent.result() is None or not sent.result().accepted:
            raise RuntimeError("MoveIt 이 실행 요청을 받지 않았습니다")
        self.handle = sent.result()
        result = self.handle.get_result_async()
        self.rclpy.spin_until_future_complete(self.node, result, timeout_sec=timeout)
        if not result.done():
            raise RuntimeError("MoveIt 실행 시간 초과")
        self.handle = None
        code = result.result().result.error_code.val
        if code != 1:
            raise RuntimeError(f"MoveIt 실행 실패 (error_code {code})")

    def stop(self):
        """실제 팔을 지금 자리에 세우고 MoveIt 실행을 멈춘다 — Ctrl+C·실행 중 오류.

        MoveIt 취소만으로는 부족하다: 실제 팔은 mock 이 이미 보낸 목표를 펌웨어 속도(speed_percent)로
        따라가므로, 저속이면 mock 보다 한참 뒤에서 계속 움직인다. 그래서 (1) 제어 게이트를 닫아 드라이버가
        mock 을 더는 따르지 않게 하고 (2) 드라이버 /emergency_stop 으로 지금 관절값에 세운 뒤
        (3) MoveIt 실행을 취소한다. 물리 비상정지가 여전히 1차 안전장치다.
        ⚠️ 구형 펌웨어(드라이버 is_switch_seamlessly=False)의 /emergency_stop 은 move_js(MIT 모드)로 세운다 —
        그 뒤 펌웨어가 move_j 를 받지 않으므로 안내서 5.6 복구(MoveIt·드라이버 재시작)가 필요하다.
        그래서 팔이 멈춰 있을 때(자세 사이 Enter 대기·계획 중)의 Ctrl+C 에서는 부르지 않는다.
        """
        from std_msgs.msg import String
        from std_srvs.srv import Empty, SetBool

        notes = []
        if not self.mock:
            for name, srv, req in (("/control_enable", SetBool, SetBool.Request(data=False)),
                                   ("/emergency_stop", Empty, Empty.Request())):
                client = self.node.create_client(srv, name)
                if client.wait_for_service(timeout_sec=1.0):
                    fut = client.call_async(req)
                    self.rclpy.spin_until_future_complete(self.node, fut, timeout_sec=2.0)
                    notes.append(f"{name} {'OK' if fut.done() else '응답 없음'}")
                else:
                    notes.append(f"{name} 없음")
        if self.sending is not None:                       # 요청을 보내는 중이었다 — 받아들여졌는지 본다
            self.rclpy.spin_until_future_complete(self.node, self.sending, timeout_sec=2.0)
            if self.sending.done() and self.sending.result() is not None and self.sending.result().accepted:
                self.handle = self.sending.result()
            self.sending = None
        if self.handle is not None:
            self.handle.cancel_goal_async()
            notes.append("MoveIt 실행 취소")
        pub = self.node.create_publisher(String, "/trajectory_execution_event", 10)
        self.spin_until(lambda: False, 0.3)               # 발행자 연결
        pub.publish(String(data="stop"))                  # MoveIt 의 모든 실행 중단
        self.spin_until(lambda: False, 0.5)
        self.handle = None
        return notes

    def close(self):
        self.node.destroy_node()
        self.rclpy.try_shutdown()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-run", help="교시 촬영 run 폴더 — capture.json 의 관절값을 자세로 쓴다")
    src.add_argument("--poses", help="YAML: poses_deg: [[j1..j6], ...]")
    p.add_argument("--only", help="이 번호만, 이 순서로 (예: 0,3,5)")
    p.add_argument("--execute", action="store_true", help="실제로 움직인다 (없으면 계획만)")
    p.add_argument("--yes", action="store_true", help="자세마다 Enter 확인을 건너뛴다")
    p.add_argument("--scale", type=float, default=0.1, help="MoveIt 속도·가속 비율 (드라이버 speed_percent 는 별도)")
    p.add_argument("--settle-timeout", type=float, default=180.0,
                   help="실행 후 실제 팔이 멈출 때까지 기다리는 최대 시간 [s] — speed_percent 1% 면 길게")
    p.add_argument("--table-z", type=float, default=-0.002,
                   help="충돌 장면의 테이블 윗면 높이 (base_link 기준 m) — 베이스가 테이블 위면 0 바로 아래")
    p.add_argument("--mock", action="store_true",
                   help="MoveIt demo follow:=false(드라이버 없음)로 로직만 시험 — 피드백 = /control/joint_states")
    p.add_argument("--config", default=str(ROOT / "workcell/scans/piper_d405.yaml"))
    p.add_argument("--calibration", default=str(ROOT / "workcell/calibration/piper_d405.yaml"))
    args = p.parse_args()
    if not 0 < args.scale <= 0.3:
        p.error("--scale 은 0~0.3")

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    fk = PiperFK(ROOT / cfg["urdf"], cfg["base_frame"], cfg["tool_frame"])
    tool_T_depth = transform(yaml.safe_load(Path(args.calibration).read_text())["tool_T_depth"])
    lower = np.array([fk.robot.joint_map[n].limit.lower for n in JOINT_NAMES])
    upper = np.array([fk.robot.joint_map[n].limit.upper for n in JOINT_NAMES])

    taught, source = load_poses(args)
    order = [int(i) for i in args.only.split(",")] if args.only else list(range(len(taught)))
    print(f"자세 {len(order)}개 ← {source} | {'실행' if args.execute else '계획만'}"
          f"{' (mock)' if args.mock else ''} | MoveIt 속도 비율 {args.scale}")

    tour = Tour("/control/joint_states" if args.mock else "/feedback/joint_states", args.mock)
    log = {"source": source, "execute": args.execute, "mock": args.mock, "scale": args.scale,
           "table_z": args.table_z,
           "started": datetime.now().isoformat(), "poses": []}
    out = ROOT / "output" / "piper_pose_tour" / f"{datetime.now():%Y%m%d_%H%M%S}.json"
    try:
        tour.preflight()
        tour.setup_scene(args.table_z, tool_T_depth)
        print(f"충돌 장면: 테이블 윗면 z={args.table_z:+.3f} m + 손목 D405 상자 (물체는 없음)")
        for k, i in enumerate(order):
            target = np.clip(taught[i], lower + LIMIT_MARGIN_RAD, upper - LIMIT_MARGIN_RAD)
            nudge = float(np.degrees(np.max(np.abs(target - taught[i]))))
            start = tour.current()
            rec = {"index": i, "taught_q": taught[i].tolist(), "target_q": target.tolist(),
                   "limit_nudge_deg": nudge, "start_q": start.tolist()}
            log["poses"].append(rec)
            res = tour.plan(target, args.scale)
            pts = res.trajectory.joint_trajectory.points
            dur = pts[-1].time_from_start.sec + pts[-1].time_from_start.nanosec * 1e-9
            rec.update(plan_s=float(res.planning_time), plan_points=len(pts), plan_duration_s=dur)
            print(f"[{k + 1}/{len(order)}] 자세 #{i}: 최대 관절 이동 {np.degrees(np.max(np.abs(target - start))):5.1f}° | "
                  f"계획 {len(pts)}점 {dur:.1f}s" + (f" | 한계 안으로 {nudge:.2f}°" if nudge > 1e-6 else ""))
            if not args.execute:
                continue
            if not args.yes:
                answer = input("    Enter = 이동 / s = 건너뜀 / q = 중단 > ").strip().lower()
                if answer == "q":
                    break
                if answer == "s":
                    rec["skipped"] = True
                    continue
            tour.check_can_execute()
            t0 = time.monotonic()
            tour.execute(res.trajectory, timeout=dur * 3 + 30)
            t_exec = time.monotonic() - t0
            if not tour.spin_until(tour.settled, args.settle_timeout):
                raise RuntimeError(f"{args.settle_timeout:.0f}s 안에 팔이 멈추지 않았습니다")
            tour.moving = False
            reached = tour.current()
            err = np.degrees(reached - target)
            cam_mm, cam_deg = camera_error(fk, tool_T_depth, reached, taught[i])
            rec.update(reached_q=reached.tolist(), joint_err_deg=err.tolist(), exec_s=t_exec,
                       settle_s=time.monotonic() - t0 - t_exec, camera_err_mm=cam_mm, camera_err_deg=cam_deg)
            print(f"    도착: 관절 오차 최대 {np.max(np.abs(err)):.3f}° ({' '.join(f'{e:+.2f}' for e in err)}) | "
                  f"교시 때 카메라 대비 {cam_mm:.2f} mm / {cam_deg:.3f}° | 실행 {t_exec:.1f}s + 정지 대기 "
                  f"{rec['settle_s']:.1f}s")
    except (KeyboardInterrupt, RuntimeError) as exc:
        reason = "Ctrl+C" if isinstance(exc, KeyboardInterrupt) else str(exc)
        log["aborted"] = reason
        print(f"\n중단: {reason}")
        if tour.moving:
            notes = tour.stop()
            log["stop"] = notes
            print(f"정지 처리: {', '.join(notes)} — 팔 상태를 눈으로 확인하세요. "
                  "계속하려면 안내서 5.6 복구(MoveIt·드라이버 재시작)가 필요합니다")
    finally:
        tour.close()
        done = [r for r in log["poses"] if "camera_err_mm" in r]
        if done:
            cam = np.array([r["camera_err_mm"] for r in done])
            print(f"\n도착한 자세 {len(done)}개: 카메라 위치 오차 중앙값 {np.median(cam):.2f} mm, 최대 {cam.max():.2f} mm")
        if args.execute:
            atomic_json(out, log)
            print(f"기록 → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
