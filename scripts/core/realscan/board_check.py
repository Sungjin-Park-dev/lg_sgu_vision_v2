#!/usr/bin/env python3
"""PiPER 관절 읽기값·손목 보정 점검 — 지금 자세에서 보드를 찍어 base_link 기준 보드 위치를 보정 때와 비교한다.

보드를 보정 때 그 자리에 두고, 보드가 보이는 자세(예: 보정 자세 하나를 pose_tour 로)에서 실행한다.
지금 관절값 FK @ tool_T_color @ (검출한 보드)가 보정 파일의 base_T_board 와 맞으면 관절 읽기값과
장착이 둘 다 정상이다. 2026-10-18: joint1 읽기값이 8.5° 틀어져 있어 회전 차이 −8.5° 로 나왔고,
전원을 껐다 켠 뒤 +0.5° 로 돌아왔다 — 전원을 켤 때마다 한 번 돌려 볼 것.

--scan-board 는 물체 스캔용 큰 보드(설정 scan_board)를 찾아 작업 좌표(보드 중심, x 긴 변, z 위)의 base 위치를
잰다 — 보드를 옮기거나 돌린 뒤 scan_poses.py --work-frame 에 넣을 값(--save-frame). 물체가 올라가 있어도 된다.

구독만 한다(/feedback/joint_states, RGB·CameraInfo). 드라이버가 켜져 있어야 한다.

    # 컨테이너에서 (ROS + agx_arm_ws 를 source 한 뒤, .venv-piper 의 python)
    python scripts/core/realscan/board_check.py
    python scripts/core/realscan/board_check.py --scan-board --save-frame output/piper_scan_poses/frame.json
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from core.realscan.calibration import CharucoBoard, board_centre_frame  # noqa: E402
from core.realscan.capture import CameraFrame, CameraModel, transform  # noqa: E402
from core.realscan.piper import JOINT_NAMES, PiperFK  # noqa: E402

# 이 이상이면 joint1 등 관절 읽기값이나 보드 자리를 의심한다 (보정 퍼짐 ~5mm·1°)
WARN_DEG, WARN_MM = 1.5, 10.0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default=str(ROOT / "workcell/scans/piper_d405.yaml"))
    p.add_argument("--calibration", default=str(ROOT / "workcell/calibration/piper_d405.yaml"))
    p.add_argument("--image", help="검출 표시 영상을 이 경로에 PNG 로 저장")
    p.add_argument("--scan-board", action="store_true",
                   help="설정의 scan_board 로 작업 좌표(보드 중심, z 위)의 base 위치를 잰다 (보정 보드 비교 대신)")
    p.add_argument("--save-frame", help="--scan-board: base_T_world 를 이 JSON 에 저장 — scan_poses.py --work-frame")
    args = p.parse_args()

    import cv2
    import rclpy
    import yaml
    from cv_bridge import CvBridge
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image, JointState

    cfg = yaml.safe_load(Path(args.config).read_text())
    cal = yaml.safe_load(Path(args.calibration).read_text())
    fk = PiperFK(ROOT / cfg["urdf"], cfg["base_frame"], cfg["tool_frame"])
    X, B0 = transform(cal["tool_T_color"]), transform(cal["base_T_board"])

    rclpy.init()
    node = rclpy.create_node("piper_board_check")
    bridge = CvBridge()
    st = {"q": [], "img": None, "model": None}

    def on_joint(m):
        pos = dict(zip(m.name, m.position))
        st["q"].append([pos[j] for j in JOINT_NAMES])

    def on_info(m):
        st["model"] = CameraModel(m.width, m.height, np.array(m.k).reshape(3, 3), np.array(m.d),
                                  m.header.frame_id, m.distortion_model)

    node.create_subscription(JointState, "/feedback/joint_states", on_joint, 50)
    node.create_subscription(CameraInfo, cfg["camera_topics"]["color_info"], on_info, qos_profile_sensor_data)
    node.create_subscription(Image, cfg["camera_topics"]["color"],
                             lambda m: st.__setitem__("img", bridge.imgmsg_to_cv2(m, "rgb8").copy()),
                             qos_profile_sensor_data)
    t0 = time.monotonic()
    while (st["img"] is None or st["model"] is None or len(st["q"]) < 60) and time.monotonic() - t0 < 10:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    rclpy.try_shutdown()
    if st["img"] is None or st["model"] is None or len(st["q"]) < 60:
        raise SystemExit("관절(/feedback/joint_states)·RGB 를 10초 안에 받지 못했습니다 — 드라이버·카메라를 확인하세요")

    qs = np.array(st["q"][-60:])
    motion = math.degrees(float(np.ptp(qs, axis=0).max()))
    q = qs.mean(axis=0)
    print(f"관절 [deg] {np.degrees(q).round(2).tolist()} (측정 중 움직임 {motion:.3f}°)")
    if motion > math.degrees(cfg["capture_guard"]["step_rad"]):     # 앱의 촬영 정지 판정과 같은 기준
        print("⚠️ 팔이 움직이는 중입니다 — 멈춘 뒤 다시 실행하세요")
    board = CharucoBoard(cfg["scan_board"]) if args.scan_board else CharucoBoard()
    C, rms, annotated = board.detect(CameraFrame(time.time(), st["img"], st["model"]))
    if args.image:
        cv2.imwrite(args.image, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    if args.scan_board:
        W = fk.pose(q) @ X @ C @ board_centre_frame(board.spec)
        print(f"스캔 보드 재투영 {rms:.2f}px · 코너 {board.corners}개")
        print(f"작업 좌표 원점(보드 중심) base 기준 {W[:3, 3].round(4).tolist()} m · 긴 변 방향 "
              f"{math.degrees(math.atan2(W[1, 0], W[0, 0])):.1f}° · 기울기 {math.degrees(math.acos(min(1., abs(W[2, 2])))):.1f}°")
        if args.save_frame:
            import json
            out = Path(args.save_frame)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"base_T_world": W.tolist(), "board": board.spec, "q": q.tolist(),
                                       "reprojection_px": rms, "corners": board.corners}, indent=1))
            print(f"→ {out}")
        return
    B = fk.pose(q) @ X @ C
    centre = np.array([0.06, 0.05, 0.0])            # 6×5칸 × 20mm 보드의 가운데
    c0, c1 = B0[:3, :3] @ centre + B0[:3, 3], B[:3, :3] @ centre + B[:3, 3]
    yaw = lambda M: math.degrees(math.atan2(M[1, 0], M[0, 0]))
    d_yaw = (yaw(B) - yaw(B0) + 180) % 360 - 180
    d_mm = (c1 - c0) * 1000
    # 로봇 축(joint1) 둘레 회전으로 보면 몇 도인가 — joint1 읽기값이 틀어지면 이 값과 d_yaw 가 같이 움직인다
    d_axis = (math.degrees(math.atan2(c1[1], c1[0]) - math.atan2(c0[1], c0[0])) + 180) % 360 - 180
    print(f"보드 재투영 {rms:.2f}px")
    print(f"보드 중심 지금 {c1.round(4).tolist()} | 보정 때 {c0.round(4).tolist()} | 차이 {d_mm.round(1).tolist()} mm")
    print(f"보드 방향 차이 {d_yaw:+.2f}° | 로봇 축 둘레 위치 차이 {d_axis:+.2f}°")
    if abs(d_yaw) > WARN_DEG or np.linalg.norm(d_mm) > WARN_MM:
        print("⚠️ 보정 때와 다릅니다. 보드를 같은 자리에 놓았다면 관절 읽기값(특히 joint1 — 두 값이 같이 "
              "어긋나면)이나 카메라 장착을 의심하세요. 전원을 껐다 켜고 다시 확인합니다.")
    else:
        print("정상: 관절 읽기값과 손목 보정이 보정 때와 맞습니다")


if __name__ == "__main__":
    main()
