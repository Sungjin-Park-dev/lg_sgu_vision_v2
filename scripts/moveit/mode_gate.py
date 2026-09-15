#!/usr/bin/env python3
"""LooksGood 모드 게이트 — ROS 스택 쪽의 **유일한** 모드 집행자.

앱(isaac_pipeline)이 ``/looksgood/mode`` 에 ``"<run>/<pipeline>"`` 을 계속 발행한다
(``sim/moveit`` · ``sim/inspection`` · ``real/moveit`` · ``real/inspection``). 이 노드는 그걸
받아 아래 둘을 **항상 실제 상태와 맞춰 둔다**. 설정이 아니라 유지다 — 스택이 앱보다 늦게
뜨든, 도중에 재시작되든, 다음 주기에 맞는다.

  1. 활성 컨트롤러:  moveit → scaled 만 활성 / inspection → jtc 만 활성 / 모드 없음 → 둘 다 비활성
     (ros2_control 은 비활성 컨트롤러로 오는 goal 을 거부한다 — 그것이 실로봇 쪽 "폐기"다.)
  2. Isaac 명령 릴레이: /isaac_joint_commands_raw → /isaac_joint_commands 를
     ``sim/moveit`` 이고 goal 이 활성일 때만 흘린다 (옛 isaac_joint_command_relay 를 흡수).

fail-safe: 모드가 ``--mode-timeout`` 동안 안 들리면 "모드 없음" 으로 본다 → 어떤 궤적
컨트롤러도 활성이 아니다 → 아무도 로봇을 못 움직인다. 앱 없이 MoveIt 만 쓰려면
``--default-mode sim/moveit`` 처럼 명시한다(launch 인자 ``default_mode``).

Isaac 쪽 게이트(그래프의 ArticulationController 노드 on/off)와 같은 원칙이다:
**항상 받고 → 모드를 보고 → 통과/폐기.**
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import Optional

import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from action_msgs.srv import CancelGoal
from controller_manager_msgs.srv import ListControllers, SwitchController
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

MODES = ("sim/moveit", "sim/inspection", "real/moveit", "real/inspection")
_GOAL_ACTIVE = {GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING}


def _wait(node: Node, future, timeout_s: float):
    """MultiThreadedExecutor 아래에서 콜백 안에서 future 를 기다린다(다른 스레드가 응답을 돌린다)."""
    deadline = time.monotonic() + timeout_s
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.02)
    return future.result() if future.done() else None


class ModeGate(Node):
    def __init__(self, a: argparse.Namespace) -> None:
        super().__init__("looksgood_mode_gate")
        self.moveit_ctrl = a.moveit_controller
        self.insp_ctrl = a.inspection_controller
        self.mode_timeout = float(a.mode_timeout)
        self.default_mode: Optional[str] = a.default_mode or None
        if self.default_mode is not None and self.default_mode not in MODES:
            raise SystemExit(f"--default-mode must be one of {MODES}, got {self.default_mode!r}")

        self._mode: Optional[str] = None
        self._mode_at: Optional[float] = None
        self._goal_active = {self.moveit_ctrl: False, self.insp_ctrl: False}
        self._cm_seen = False
        self._reconciling = threading.Lock()
        self._last_log = ""

        cb = ReentrantCallbackGroup()
        self.create_subscription(String, a.mode_topic, self._on_mode, 10, callback_group=cb)
        self._cmd_pub = self.create_publisher(JointState, a.output_topic, 10)
        self.create_subscription(JointState, a.input_topic, self._on_cmd, 10, callback_group=cb)
        for ctrl in (self.moveit_ctrl, self.insp_ctrl):
            self.create_subscription(
                GoalStatusArray, f"/{ctrl}/follow_joint_trajectory/_action/status",
                lambda m, c=ctrl: self._on_status(c, m), 10, callback_group=cb)
        self._state_pub = self.create_publisher(String, a.state_topic, 10)
        cm = a.controller_manager.rstrip("/")
        self._list_cli = self.create_client(ListControllers, f"{cm}/list_controllers",
                                            callback_group=cb)
        self._switch_cli = self.create_client(SwitchController, f"{cm}/switch_controller",
                                              callback_group=cb)
        self._cancel_cli = {
            ctrl: self.create_client(
                CancelGoal, f"/{ctrl}/follow_joint_trajectory/_action/cancel_goal",
                callback_group=cb)
            for ctrl in (self.moveit_ctrl, self.insp_ctrl)
        }
        self.create_timer(float(a.reconcile_period), self._reconcile, callback_group=cb)
        self.get_logger().info(
            f"mode gate up: listening {a.mode_topic} (timeout {self.mode_timeout}s, "
            f"default {self.default_mode or 'none = fail-safe'}); "
            f"moveit={self.moveit_ctrl} inspection={self.insp_ctrl}; "
            f"relay {a.input_topic} -> {a.output_topic} in sim/moveit only")

    # ── 입력 ────────────────────────────────────────────────────────────────
    def _on_mode(self, msg: String) -> None:
        mode = msg.data.strip()
        if mode not in MODES:
            self.get_logger().warning(f"ignoring unknown mode {mode!r}")
            return
        if mode != self._mode:
            self.get_logger().info(f"mode -> {mode}")
        self._mode, self._mode_at = mode, time.monotonic()

    def _on_status(self, ctrl: str, msg: GoalStatusArray) -> None:
        self._goal_active[ctrl] = any(g.status in _GOAL_ACTIVE for g in msg.status_list)

    def _on_cmd(self, msg: JointState) -> None:
        """릴레이: sim/moveit 이고 goal 이 활성일 때만 통과. 나머지는 폐기."""
        eff, _ = self._effective_mode()
        if eff == "sim/moveit" and any(self._goal_active.values()):
            self._cmd_pub.publish(msg)

    # ── 판단 ────────────────────────────────────────────────────────────────
    def _effective_mode(self) -> tuple[Optional[str], str]:
        """(지금 적용할 모드 또는 None, 출처). None 이면 fail-safe(둘 다 비활성)."""
        if self._mode is not None and self._mode_at is not None \
                and time.monotonic() - self._mode_at <= self.mode_timeout:
            return self._mode, "app"
        if self.default_mode is not None:
            return self.default_mode, "default"
        return None, "silent"

    def _wanted(self, eff: Optional[str]) -> tuple[tuple, tuple]:
        if eff is None:
            return (), (self.moveit_ctrl, self.insp_ctrl)
        if eff.endswith("/moveit"):
            return (self.moveit_ctrl,), (self.insp_ctrl,)
        return (self.insp_ctrl,), (self.moveit_ctrl,)

    # ── 집행 ────────────────────────────────────────────────────────────────
    def _log_once(self, text: str, level: str = "info") -> None:
        if text == self._last_log:
            return
        self._last_log = text
        getattr(self.get_logger(), level)(text)

    def _reconcile(self) -> None:
        if not self._reconciling.acquire(blocking=False):
            return
        try:
            self._reconcile_locked()
        finally:
            self._reconciling.release()

    def _reconcile_locked(self) -> None:
        eff, src = self._effective_mode()
        act, deact = self._wanted(eff)

        if not self._list_cli.service_is_ready():
            if self._cm_seen:
                self.get_logger().warning("controller_manager gone - waiting for it to return")
                self._cm_seen = False
            self._publish_state(eff, src, None, forward=False)
            return
        if not self._cm_seen:
            self._cm_seen = True
            self.get_logger().info("controller_manager detected - enforcing mode from now on")

        res = _wait(self, self._list_cli.call_async(ListControllers.Request()), 2.0)
        if res is None:
            self._log_once("list_controllers timed out", "warning")
            return
        state = {c.name: c.state for c in res.controller}
        missing = [c for c in (self.moveit_ctrl, self.insp_ctrl) if c not in state]
        if missing:
            self._log_once(f"waiting for controllers to load: {missing}")
            return

        to_activate = [c for c in act if state[c] != "active"]
        to_deactivate = [c for c in deact if state[c] == "active"]
        if not to_activate and not to_deactivate:
            self._publish_state(eff, src, state, forward=(eff == "sim/moveit"))
            return

        # 내리는 컨트롤러의 goal 을 먼저 취소한다. 비활성화 도중 goal 이 EXECUTING 에
        # 얼어붙으면 그 상태 토픽이 "활성" 으로 남아 릴레이가 계속 흘린다(옛 버그).
        for c in to_deactivate:
            self._cancel_goals(c)
        req = SwitchController.Request()
        req.activate_controllers = to_activate
        req.deactivate_controllers = to_deactivate
        req.strictness = SwitchController.Request.BEST_EFFORT
        req.activate_asap = True
        req.timeout.sec = 2
        res = _wait(self, self._switch_cli.call_async(req), 3.0)
        ok = bool(res is not None and res.ok)
        self.get_logger().info(
            f"mode={eff or 'none'} ({src}) -> +{to_activate} -{to_deactivate} "
            f"{'OK' if ok else 'FAILED'}")
        for c in to_activate:
            self._cancel_goals(c)          # 새로 켠 쪽에 남아 있던 옛 goal 도 치운다
        self._publish_state(eff, src, state, forward=(eff == "sim/moveit"))

    def _cancel_goals(self, ctrl: str) -> None:
        cli = self._cancel_cli[ctrl]
        if not cli.service_is_ready():
            return
        _wait(self, cli.call_async(CancelGoal.Request()), 1.0)   # goal_id 0 = 전부

    def _publish_state(self, eff, src, state, *, forward: bool) -> None:
        active = sorted(n for n, s in (state or {}).items() if s == "active"
                        and n in (self.moveit_ctrl, self.insp_ctrl))
        msg = String()
        msg.data = (f"mode={eff or 'none'} source={src} active={active} "
                    f"forward={'on' if forward else 'off'} "
                    f"cm={'up' if state is not None else 'down'}")
        self._state_pub.publish(msg)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode-topic", default="/looksgood/mode")
    p.add_argument("--state-topic", default="/looksgood/gate_state",
                   help="관측용: 지금 게이트가 보는 모드·활성 컨트롤러·릴레이 상태")
    p.add_argument("--mode-timeout", type=float, default=2.0,
                   help="이 시간 동안 모드가 안 들리면 fail-safe (둘 다 비활성)")
    p.add_argument("--default-mode", default="",
                   help=f"앱이 없을 때 쓸 모드 {MODES}. 비우면 fail-safe")
    p.add_argument("--reconcile-period", type=float, default=0.5)
    p.add_argument("--controller-manager", default="/controller_manager")
    p.add_argument("--moveit-controller", default="scaled_joint_trajectory_controller")
    p.add_argument("--inspection-controller", default="joint_trajectory_controller")
    p.add_argument("--input-topic", default="/isaac_joint_commands_raw")
    p.add_argument("--output-topic", default="/isaac_joint_commands")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    rclpy.init(args=None)
    node = ModeGate(args)
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass                    # SIGINT/SIGTERM — rclpy 가 이미 context 를 닫았다
    finally:
        ex.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
