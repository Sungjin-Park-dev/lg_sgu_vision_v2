#!/usr/bin/env python3
"""PiPER wrist D405: read-only hand-eye calibration, manual capture and TSDF viewer."""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from core.realscan.calibration import (  # noqa: E402
    CalibrationQualityError, CharucoBoard, board_spread, calibration_limits, check_board_depth,
    mean_board_pose, solve_hand_eye, validate_holdout,
)
from core.realscan.capture import ScanSession, atomic_bytes, atomic_json, transform  # noqa: E402
from core.realscan.piper import PiperFK  # noqa: E402
from core.realscan.sources import CalibrationSimSource, DemoSource, LiveSource  # noqa: E402
from core.reconstruction import storage  # noqa: E402
from core.reconstruction.sensor import pixel_world_points  # noqa: E402
from core.reconstruction.segment import OBJECT, classify_pixels  # noqa: E402


def read_config(path):
    import yaml
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg.get("version") != 1:
        raise ValueError("설정 파일 version은 1이어야 합니다")
    if not 0 < cfg["min_range_m"] < cfg["max_range_m"]:
        raise ValueError("카메라 측정 범위를 확인하세요")
    cfg["calibration_validation"] = calibration_limits(**cfg.get("calibration_validation",
        {"max_translation_mm": 5., "max_rotation_deg": 1.}))
    cfg["urdf"] = str(ROOT / cfg["urdf"])          # 상대경로는 리포 기준 (절대경로는 그대로)
    return cfg


def json_signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def quat(T):
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    return tuple(q[[3, 0, 1, 2]])


def image_markdown(image):
    import cv2
    ok, blob = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        return "미리보기 생성 실패"
    return "![미리보기](data:image/jpeg;base64," + base64.b64encode(blob).decode() + ")"


class ScanUI:
    def __init__(self, args, cfg, source, fk):
        import viser
        self.args, self.cfg, self.source, self.fk = args, cfg, source, fk
        self.sim = getattr(args, "sim_calibration", False)
        self.lock = threading.RLock()
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.busy = False
        self.board = CharucoBoard()
        self.samples, self.holdouts = [], []
        self.depth_checks, self.cal_handles, self.depth_handles = [], [], []
        self.candidate = self.calibration = None
        self.fixed_boards = []
        self.identity_locked = None
        self.last_annotated = None
        self.annotated_until = 0.0
        self.calibration_dir = ROOT / "workcell" / "calibration"
        self.calibration_path = self.calibration_dir / "piper_d405.yaml"
        if self.sim:
            self.calibration_dir /= "simulation"
            self.calibration_path = self.calibration_dir / "piper_d405.yaml"
        self.cal_record_dir = self.calibration_dir / "samples" / args.run
        self.log_text = "카메라와 관절 피드백을 확인하세요. 촬영 중에는 팔을 멈추세요."
        self.session = None
        self.loaded = Path(args.view_run).resolve() if args.view_run else None
        if not self.loaded:
            self.session = ScanSession(ROOT / "data", args.object, args.run,
                                       {"min_range_m": cfg["min_range_m"], "max_range_m": cfg["max_range_m"],
                                        "source": ("synthetic_calibration" if self.sim else
                                                   "synthetic_demo" if args.demo else "piper_real"),
                                        **({"pose_source": "virtual_camera_trajectory",
                                            "robot_kinematics": "not_simulated"} if self.sim else {})},
                                       resume=bool(getattr(args, "resume_scan", None)))
        self.mesh = None
        self.mesh_handle = None
        self.shot_handles = {}
        self.live_camera_handles = {}
        self.live_camera_valid = False
        self._roi_busy = False
        self.server = viser.ViserServer(host=args.host, port=args.port)
        self.server.gui.configure_theme(control_layout="collapsible", control_width="large")
        self.server.scene.set_up_direction("+z")
        self.server.scene.add_frame("/base_link", axes_length=0.05, axes_radius=0.001)

        @self.server.on_client_connect
        def connected(client):
            client.camera.position = (0.65, -0.50, 0.40)
            client.camera.look_at = (0.25, 0, 0.07)
            client.camera.up_direction = (0, 0, 1)

        g = self.server.gui
        g.add_markdown("# PiPER 가상 스캔\n가상 보정 → 촬영 → 복원" if self.sim else
                       "# PiPER 실제 스캔\n교시로 이동 → 정지 → 촬영 → 복원")
        if args.demo:
            g.add_markdown("**데모 데이터입니다. 로봇·카메라에 연결하지 않습니다.**")
        if getattr(args, "allow_robot_control", False):
            g.add_markdown("**드라이버·MoveIt 동시 실행 허용** — 팔이 다른 프로그램의 명령으로 움직일 수 "
                           "있습니다. 이 앱은 관절값을 읽기만 하며, 촬영은 정지 확인 후에만 됩니다.")
        if self.sim:
            g.add_markdown("**가상 보정 → 촬영 → 복원**\n카메라 기하 시뮬레이션입니다. "
                           "PiPER 물리·충돌 시뮬레이션은 포함하지 않습니다.")
        if self.loaded:
            g.add_markdown(f"**저장 결과 보기**\n`{self.loaded}`")
        self.status = g.add_markdown("연결 확인 중")
        self.operation = g.add_markdown(self.log_text)
        with g.add_folder("카메라 미리보기", expand_by_default=True):
            self.rgb_preview = g.add_markdown("RGB 대기 중", image_root=ROOT)
            self.depth_preview = g.add_markdown("depth 대기 중", image_root=ROOT)
            self.show_live = g.add_checkbox("현재 카메라 점군 보기", initial_value=True)
            self.show_camera = g.add_checkbox("현재 카메라 위치·촬영 방향 보기", initial_value=True)
            self.camera_status = g.add_markdown("카메라 위치 대기 중 — 주황색: 현재 / 파랑: 저장된 촬영 시점")
        with g.add_folder("1. 카메라 장착 보정", expand_by_default=True):
            g.add_markdown("출력한 보드를 고정하고 15개 이상의 다른 자세를 촬영하세요.\n"
                           "보정 계산 후 **새 자세 3개**를 검증용으로 촬영합니다.")
            self.board_button = g.add_button("보정 보드 SVG 받기")
            self.cal_capture = g.add_button("보정 자세 기록")
            self.cal_solve = g.add_button("15개 이상으로 보정 계산")
            self.hold_capture = g.add_button("별도 RGB·depth 검증 자세 기록")
            self.cal_accept = g.add_button("검증 후 보정 저장")
            self.cal_status = g.add_markdown("보정값 없음")
            limits = cfg["calibration_validation"]
            g.add_markdown(f"보정·별도 검증 허용 편차: **{limits['max_translation_mm']:g} mm / "
                           f"{limits['max_rotation_deg']:g}°**")
            self.sample_number = g.add_number("제외/복구할 자세 번호", initial_value=1, min=1, step=1)
            self.toggle_sample = g.add_button("선택 자세 제외/복구 (원본 유지)")
            self.show_calibration = g.add_checkbox("보드 겹침·depth 검증 3D 보기", initial_value=True)
            self.board_status = g.add_markdown("보정 계산 후 학습 보드(파랑)·검증 보드(초록)·기준 보드(노랑)를 표시합니다.")
            self.depth_status = g.add_markdown("검증 자세에서 보드를 계속 보여주세요. RGB 평면과 depth를 함께 확인합니다.")
            if self.sim:
                self.sim_depth_bias = g.add_number("가상 depth 거리 오차 [mm]",
                    initial_value=source.depth_bias_m*1000, min=-30., max=30., step=1.)
                self.sim_depth_bias.on_update(lambda _: setattr(
                    self.source, "depth_bias_m", float(self.sim_depth_bias.value)/1000))
        with g.add_folder("2. 물체 촬영", expand_by_default=True):
            self.object_name = g.add_text("물체 이름", initial_value=args.object)
            self.new_scan_button = g.add_button("새 촬영 시작 (기존 결과 유지)")
            self.roi = g.add_vector3("ROI 중심 [m]", initial_value=tuple(cfg["roi_center"]), step=0.005)
            self.radius = g.add_number("ROI 반경 [m]", initial_value=cfg["roi_radius"], min=0.005,
                                       max=0.5, step=0.005)
            self.roi_centre_button = g.add_button("화면 중앙 물체에 ROI 놓기")
            self.table_enabled = g.add_checkbox("테이블 제외 (높이 확인 후)", initial_value=False)
            self.table_z = g.add_number("바닥 제외 높이 z [m]", initial_value=0.0, step=0.001)
            self.support_enabled = g.add_checkbox("추가 영역 제외 (받침대·잡음)", initial_value=False)
            self.support_pos = g.add_vector3("제외 영역 중심 [m]", initial_value=(0.25, 0.0, 0.025), step=0.005)
            self.support_dims = g.add_vector3("제외 영역 크기 [m]", initial_value=(0.10, 0.10, 0.05),
                                              min=(0.001, 0.001, 0.001), step=0.005)
            self.capture_button = g.add_button("정지한 자세 촬영")
            self.capture_status = g.add_markdown("0장 — 권장 8–12개 시점")
        with g.add_folder("3. 복원", expand_by_default=True):
            self.voxel = g.add_number("TSDF voxel [mm]", initial_value=1.0, min=0.2, max=5.0, step=0.2)
            self.icp = g.add_checkbox("ICP 자세 보정", initial_value=False)
            self.recon_button = g.add_button("촬영 프레임으로 복원")
            self.recon_status = g.add_markdown("PLY·OBJ는 로봇 base_link 좌표로 저장됩니다.")
            self.show_clouds = g.add_checkbox("촬영 점군 보기", initial_value=True)
        self.roi_handle = self.server.scene.add_transform_controls("/roi_gizmo", scale=0.07,
                                                                   position=tuple(self.roi.value),
                                                                   disable_rotations=True)
        self.roi.on_update(lambda _: self._roi_changed())
        self.radius.on_update(lambda _: self._draw_roi())

        @self.roi_handle.on_update
        def moved(_):
            if not self._roi_busy:
                self.roi.value = tuple(self.roi_handle.position)

        self.board_button.on_click(lambda e: e.client.send_file_download("piper_charuco_20mm.svg",
                                                                         self.board.svg().encode()))
        self.cal_capture.on_click(lambda _: self._action(self._cal_capture))
        self.cal_solve.on_click(lambda _: self._action(self._cal_solve))
        self.hold_capture.on_click(lambda _: self._action(lambda: self._cal_capture(holdout=True)))
        self.cal_accept.on_click(lambda _: self._action(self._cal_accept))
        self.toggle_sample.on_click(lambda _: self._action(self._toggle_sample))
        self.capture_button.on_click(lambda _: self._action(self.capture))
        self.new_scan_button.on_click(lambda _: self._action(self._new_scan))
        self.roi_centre_button.on_click(lambda _: self._action(self._centre_roi))
        self.recon_button.on_click(lambda _: self._action(self._start_reconstruction))
        self.show_clouds.on_update(lambda _: self._cloud_visibility())
        self.show_live.on_update(lambda _: self._live_visibility())
        self.show_camera.on_update(lambda _: self._camera_visibility())
        self.show_calibration.on_update(lambda _: self._cal_visibility())
        self._draw_roi()
        if self.loaded:
            plan = storage.load_plan(self.loaded)
            self.roi.value, self.radius.value = tuple(plan.roi_center), plan.roi_radius
            self._draw_saved(self.loaded)
            self.capture_status.content = f"저장 프레임 {len(storage.load_frames(self.loaded))}장"
            mesh = storage.load_recon(self.loaded)
            if mesh is not None:
                self._draw_mesh(mesh)
        elif args.demo and not self.sim:
            self.calibration = {"validated": True, "demo": True, "identity": self.identity(),
                                "tool_T_depth": np.eye(4).tolist(), "tool_T_color": np.eye(4).tolist()}
            self.roi.value = tuple(source.center)
            self.radius.value = 0.06
            self.table_enabled.value = True
            self.cal_status.content = "데모용 단위 변환 — 실제 보정에 사용할 수 없습니다."
        elif not self.sim and self.calibration_path.exists():
            import yaml
            try:
                cal = yaml.safe_load(self.calibration_path.read_text())
                if cal.get("validated") is not True or cal.get("demo"):
                    raise ValueError("검증된 실제 보정값이 아닙니다")
                if cal.get("depth_validation", {}).get("passed") is not True:
                    raise ValueError("RGB·depth 검증이 없는 이전 보정값입니다. 새 보정을 진행하세요")
                transform(cal["tool_T_depth"])
                self.calibration = cal
                self.cal_status.content = "저장된 보정값 로드 — 카메라와 URDF 일치 확인 대기 중"
            except Exception as exc:
                self.cal_status.content = f"보정값 로드 실패: {exc}"
        if getattr(args, "resume_calibration", None):
            self._restore_samples(args.resume_calibration)
            self._action(self._cal_solve)
        if getattr(args, "resume_scan", None):
            self._restore_scan()
        self._controls(False)

    def _action(self, action):
        with self.lock:
            if self.busy:
                return
            try:
                action()
            except Exception as exc:
                self.operation.content = f"**확인 필요:** {exc}"

    def identity(self):
        if self.source is None:
            raise ValueError("저장 결과 보기에서는 촬영할 수 없습니다")
        if not self.source.serial or self.source.color_model is None or self.source.depth_model is None:
            raise ValueError("카메라 일련번호와 RGB/depth CameraInfo 대기 중")
        return {"serial": self.source.serial, "base_frame": self.cfg["base_frame"],
                "tool_frame": self.cfg["tool_frame"], "urdf_sha256": self.fk.hash if self.fk else "DEMO",
                "color": self.source.color_model.record(), "depth": self.source.depth_model.record(),
                "color_T_depth": self.source.color_T_depth().tolist()}

    def _tool_pose(self, q):
        if self.sim:
            return self.source.tool_pose()
        return self.source.T.copy() if self.args.demo else self.fk.pose(q)

    def _active_samples(self):
        return [s for s in self.samples if not s.get("excluded", False)]

    def _restore_scan(self):
        if self.calibration is None or json.dumps(self.calibration, sort_keys=True) != self.session.calibration:
            raise ValueError("현재 보정값과 촬영 당시 보정값이 다릅니다. 결과 보기 모드를 사용하세요")
        plan = storage.load_plan(self.session.directory)
        self.session.meta.update(plan.meta)
        self.roi.value, self.radius.value = tuple(plan.roi_center), float(plan.roi_radius)
        for name, row in zip(plan.obstacle_names, plan.obstacles):
            if not np.allclose(row[3:7], [1,0,0,0]):
                raise ValueError("현재 UI에서 복구할 수 없는 장애물 회전입니다. 결과 보기 모드를 사용하세요")
            if name == "table":
                self.table_enabled.value = True
                self.table_z.value = float(row[2]+row[9]/2)
            elif name == "support":
                self.support_enabled.value = True
                self.support_pos.value, self.support_dims.value = tuple(row[:3]), tuple(row[7:])
            else:
                raise ValueError("현재 UI에서 복구할 수 없는 장애물입니다")
        self._draw_saved(self.session.directory)
        self.capture_status.content = f"**촬영 원본 {len(self.session.records)}장 복구**\n`{self.session.directory}`"
        tag = "icp" if (self.session.directory / "recon_icp.ply").exists() else ""
        mesh = storage.load_recon(self.session.directory, tag=tag)
        if mesh is not None:
            self.icp.value = tag == "icp"
            self._draw_mesh(mesh)
            self.show_live.value = self.show_clouds.value = False
            self.recon_status.content = f"**저장 복원 결과 로드** · {len(mesh.faces)}개 면\n`{self.session.directory}`"
        self.operation.content = "원본 촬영과 ROI를 복구했습니다. 같은 물체·카메라 장착 상태에서 이어가세요."

    def _write_samples(self):
        atomic_json(self.cal_record_dir / "samples.json", {"identity": self.identity_locked,
            "board": self.board.spec, "train": self.samples, "holdout": self.holdouts})

    def _restore_samples(self, directory):
        path = Path(directory).resolve()
        expected = self.calibration_dir / "samples"
        if not path.is_relative_to(expected.resolve()):
            raise ValueError("보정 원본 폴더는 현재 실제/가상 보정 samples 안에 있어야 합니다")
        data = json.loads((path / "samples.json").read_text())
        if data["board"] != self.board.spec:
            raise ValueError("저장한 보드 규격이 현재와 다릅니다")
        deadline = time.monotonic()+5
        while True:
            try:
                current = self.identity()
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise ValueError("원본 복구를 위한 카메라 식별 정보 대기 시간 초과")
                time.sleep(.1)
        if json_signature(data["identity"]) != json_signature(current):
            raise ValueError("저장된 카메라·URDF 설정과 현재 설정이 다릅니다")
        for i, sample in enumerate(data["train"], 1):
            transform(sample["base_T_tool"])
            transform(sample["color_T_board"])
            if not (path / f"train_{i:03d}.npz").exists():
                raise ValueError(f"보정 원본 이미지 {i}번이 없습니다")
        self.samples = data["train"]
        self.holdouts = []  # A resumed solve requires new independent holdouts.
        self.candidate = self.calibration = None
        self.identity_locked, self.cal_record_dir = data["identity"], path
        check_path = path / "depth_checks.json"
        if check_path.exists():
            self.depth_checks = json.loads(check_path.read_text())["checks"]
        self.operation.content = f"보정 원본 {len(self.samples)}개 복구 완료. 별도 검증은 새로 기록하세요."

    def _toggle_sample(self):
        if self.session.records:
            raise ValueError("물체 촬영 후에는 보정 자세를 변경할 수 없습니다")
        number = int(self.sample_number.value)
        if not 1 <= number <= len(self.samples):
            raise ValueError("저장된 자세 번호를 선택하세요")
        sample = self.samples[number-1]
        sample["excluded"] = not sample.get("excluded", False)
        self.candidate = self.calibration = None
        self.holdouts.clear()
        self._write_samples()
        self.cal_status.content = (f"{number}번 {'제외' if sample['excluded'] else '복구'} · "
            f"사용 {len(self._active_samples())}/{len(self.samples)}개. 15개 이상으로 다시 계산하세요.")
        self.operation.content = "원본은 유지됩니다. 제외는 사용자가 직접 선택하며, 별도 검증을 다시 진행해야 합니다."

    def _cal_capture(self, holdout=False):
        if self.session.records:
            raise ValueError("이미 물체를 촬영했습니다. 보정을 바꾸려면 앱을 종료하고 새 run을 시작하세요")
        if holdout and self.candidate is None:
            raise ValueError("먼저 보정 계산을 실행하세요")
        frame, q, dt = self.source.sample("color")
        identity = self.identity()
        if json_signature(frame.model.record()) != json_signature(identity["color"]):
            raise ValueError("RGB 프레임과 현재 CameraInfo가 다릅니다")
        if self.identity_locked is not None and json_signature(identity) != json_signature(self.identity_locked):
            raise ValueError("보정 중 카메라/설정이 변경되었습니다. 앱을 다시 시작하세요")
        board_T, rms, annotated = self.board.detect(frame)
        self.last_annotated = annotated
        self.annotated_until = time.monotonic() + 2.0
        sample = {"stamp": frame.stamp, "q": q.tolist(), "base_T_tool": self._tool_pose(q).tolist(),
                  "color_T_board": board_T.tolist(), "reprojection_rms_px": rms, "sync_error_s": dt}
        if holdout:
            report = self._record_depth_check(frame, q, board_T, identity)
            if not report["passed"]:
                raise ValueError("RGB·depth 검증 실패: " + "; ".join(report["reasons"]) +
                                 " — 이 자세는 검증 개수에 포함하지 않았습니다")
            sample["depth_report"] = report
            self.holdouts.append(sample)
            label = "holdout"
            existing = [int(p.stem.split("_")[-1]) for p in self.cal_record_dir.glob("holdout_*.npz")]
            number = max(existing, default=0)+1
            sample["record_file"] = f"holdout_{number:03d}.npz"
        else:
            self.samples.append(sample)
            self.candidate = self.calibration = None
            self.holdouts.clear()
            for handle in reversed(self.cal_handles + self.depth_handles):
                handle.remove()
            self.cal_handles.clear()
            self.depth_handles.clear()
            self.board_status.content = "새 보정 자세 기록 중 — 보정 계산 후 겹침을 표시합니다."
            label, number = "train", len(self.samples)
        self.identity_locked = identity
        import io
        buf = io.BytesIO()
        np.savez_compressed(buf, color=frame.pixels, **{k: np.asarray(v) for k, v in sample.items()})
        atomic_bytes(self.cal_record_dir / f"{label}_{number:03d}.npz", buf.getvalue())
        self._write_samples()
        self.cal_status.content = f"보정 사용 {len(self._active_samples())}/{len(self.samples)}개 / 별도 검증 {len(self.holdouts)}개 · 재투영 {rms:.2f}px"
        if self.candidate is not None:
            self._draw_calibration()
        self.operation.content = "자세가 저장되었습니다. 다른 위치·회전 방향으로 옮긴 후 정지하세요."
        if self.sim:
            self.source.advance()
            self.operation.content = "가상 자세를 기록하고 다음 시점으로 이동했습니다."

    def _cal_solve(self):
        self.candidate = self.calibration = None
        self.holdouts.clear()
        active = self._active_samples()
        self.cal_status.content = f"보정 계산 중 · 사용 {len(active)}/{len(self.samples)}개"
        try:
            X, report, fixed = solve_hand_eye([s["base_T_tool"] for s in active],
                                              [s["color_T_board"] for s in active],
                                              **self.cfg["calibration_validation"])
        except ValueError as exc:
            self.cal_status.content = f"**보정 계산 실패**\n{exc}\n원본 {len(self.samples)}개는 유지됩니다."
            print(f"Calibration failed ({len(active)} samples): {exc}", flush=True)
            if isinstance(exc, CalibrationQualityError):
                self._draw_calibration(preview=exc.candidate)
                atomic_json(self.cal_record_dir / "solve_diagnostic.json",
                    {"validated": False, "report": exc.report, "tool_T_color": exc.candidate.tolist(),
                     "sample_numbers": [i+1 for i,s in enumerate(self.samples) if not s.get("excluded", False)]})
            raise
        self.candidate = {"tool_T_color": X.tolist(), "training_report": report}
        self.fixed_boards = list(fixed)
        self.holdouts.clear()
        self._draw_calibration()
        self.cal_status.content = f"보정 계산 완료: {report}\n별도의 새 자세 3개를 검증용으로 촬영하세요."

    def _cal_accept(self):
        if self.candidate is None:
            raise ValueError("먼저 보정 계산을 실행하세요")
        identity = self.identity()
        if json_signature(identity) != json_signature(self.identity_locked):
            raise ValueError("보정 중 카메라 설정이 변경되었습니다")
        X = transform(self.candidate["tool_T_color"])
        poses = [transform(s["base_T_tool"]) @ X @ transform(s["color_T_board"]) for s in self.holdouts]
        report = validate_holdout(self.fixed_boards, poses, **self.cfg["calibration_validation"])
        depths = [s.get("depth_report", {}) for s in self.holdouts]
        if any(d.get("passed") is not True for d in depths):
            raise ValueError("3개 이상의 별도 자세에서 RGB·depth 검증이 필요합니다")
        depth_summary = {"passed": True, "samples": len(depths),
                         "max_abs_offset_mm": max(abs(d["offset_mm"]) for d in depths),
                         "max_p95_abs_mm": max(d["p95_abs_mm"] for d in depths),
                         "max_normal_error_deg": max(d["normal_error_deg"] for d in depths),
                         "min_valid_fraction": min(d["valid_fraction"] for d in depths),
                         "limits": depths[0]["limits"]}
        self.calibration = {**self.candidate, "validated": True, "demo": self.args.demo,
                            "created_at": datetime.now().isoformat(), "identity": identity,
                            "tool_T_depth": (X @ transform(identity["color_T_depth"])).tolist(),
                            "board": self.board.spec, "holdout_report": report,
                            "base_T_board": mean_board_pose(self.fixed_boards).tolist(),
                            "depth_validation": depth_summary,
                            "sample_directory": str(self.cal_record_dir.relative_to(ROOT))}
        import yaml
        atomic_bytes(self.calibration_path, yaml.safe_dump(self.calibration, allow_unicode=True).encode())
        self.cal_status.content = f"**보정 저장 완료**\n검증: {report}"
        self.operation.content = "보드를 제거하고 물체를 놓으세요. 카메라 장착은 변경하지 마세요."
        self._draw_calibration()
        if self.sim:
            error = self.source.calibration_error(X)
            self.calibration["simulation_truth_error"] = error
            atomic_bytes(self.calibration_path, yaml.safe_dump(self.calibration, allow_unicode=True).encode())
            self.source.start_scan()
            self.roi.value, self.radius.value = tuple(self.source.center), .06
            self.table_enabled.value = True
            self.cal_status.content = f"**가상 보정 검증 완료**\n알려진 장착값 대비 오차: {error}"
            self.operation.content = "가상 보드를 구형 물체로 바꿨습니다. 촬영 버튼으로 시점을 기록하세요."

    def _check_calibration(self):
        if self.calibration is None or self.calibration.get("validated") is not True:
            raise ValueError("카메라 장착 보정과 별도 자세 검증이 먼저 필요합니다")
        if self.calibration.get("demo", False) != self.args.demo:
            raise ValueError("실제/데모 보정값이 맞지 않습니다")
        if (not self.args.demo or self.sim) and self.calibration.get("depth_validation", {}).get("passed") is not True:
            raise ValueError("RGB·depth 보드 검증이 먼저 필요합니다")
        if json_signature(self.calibration["identity"]) != json_signature(self.identity()):
            raise ValueError("보정 당시 카메라·내부 파라미터·URDF와 다릅니다. 다시 보정하세요")

    def _cal_visibility(self):
        for handle in self.cal_handles + self.depth_handles:
            handle.visible = self.show_calibration.value

    def _board_geometry(self, name, T, color, opacity=.04):
        corners = np.array([[0, 0, 0], [.12, 0, 0], [.12, .10, 0], [0, .10, 0]], dtype=np.float32)
        outline = np.concatenate([np.linspace(corners[i], corners[(i+1) % 4], 40) for i in range(4)])
        return [self.server.scene.add_frame(name, wxyz=quat(T), position=tuple(T[:3, 3]),
                                           axes_length=.015, axes_radius=.0005),
                self.server.scene.add_mesh_simple(name+"/plane", vertices=corners,
                    faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32), color=color, opacity=opacity),
                self.server.scene.add_point_cloud(name+"/outline", points=outline,
                                                  colors=color, point_size=.001)]

    def _draw_calibration(self, preview=None):
        X = transform(preview if preview is not None else (self.candidate or self.calibration)["tool_T_color"])
        for handle in reversed(self.cal_handles):
            handle.remove()
        self.cal_handles.clear()
        poses = [transform(s["base_T_tool"]) @ X @ transform(s["color_T_board"]) for s in self._active_samples()]
        if not poses:
            return
        reference = mean_board_pose(poses)
        self.cal_handles += self._board_geometry("/calibration/reference", reference, (255, 210, 40), .08)
        for group, samples, color in (("train", self.samples, (70, 150, 255)),
                                      ("holdout", self.holdouts, (70, 230, 120))):
            for i, sample in enumerate(samples):
                T = transform(sample["base_T_tool"]) @ X @ transform(sample["color_T_board"])
                self.cal_handles += self._board_geometry(f"/calibration/{group}_{i:03d}", T,
                    (130, 130, 130) if sample.get("excluded", False) else color)
        spread = board_spread(poses)
        rotation_mean = Rotation.from_matrix(np.stack(poses)[:, :3, :3]).mean()
        errors = np.rad2deg((rotation_mean.inv()*Rotation.from_matrix(np.stack(poses)[:, :3, :3])).magnitude())
        numbers = [i+1 for i,s in enumerate(self.samples) if not s.get("excluded", False)]
        rows = "\n".join(f"| {i} | {e:.2f} |" for i,e in zip(numbers, errors))
        self.board_status.content = (f"**{'미검증 진단용' if preview is not None else '보드 겹침'}** · 학습 사용 {len(poses)} / 검증 {len(self.holdouts)}\n"
            f"학습 최대 위치 편차 {spread['translation_max_mm']:.2f} mm · 회전 {spread['rotation_max_deg']:.2f}°\n"
            "파랑: 학습 / 초록: 별도 검증 / 노랑: 학습 평균 / 회색: 제외\n\n"
            "| 자세 번호 | 회전 편차 [°] |\n|---|---|\n"+rows)
        self._cal_visibility()

    def _record_depth_check(self, color, color_q, color_T_board, identity):
        depth, q, dt = self.source.sample("depth")
        if (json_signature(self.identity()) != json_signature(identity)
                or json_signature(depth.model.record()) != json_signature(identity["depth"])):
            raise ValueError("RGB·depth 기록 사이에 카메라 설정이 변경되었습니다")
        frame_dt = abs(color.stamp-depth.stamp)
        if frame_dt > self.cfg["capture_guard"]["sync_s"]:
            raise ValueError("RGB와 depth 프레임의 시간 차이가 큽니다. 정지한 상태에서 다시 기록하세요")
        if np.max(np.abs(q-color_q)) > self.cfg["capture_guard"]["step_rad"]:
            raise ValueError("RGB·depth 기록 사이에 팔이 움직였습니다")
        params = {**self.cfg.get("depth_validation", {}),
                  "min_range_m": self.cfg["min_range_m"], "max_range_m": self.cfg["max_range_m"]}
        report, points, residual = check_board_depth(depth, color_T_board, identity["color_T_depth"], **params)
        X = transform(self.candidate["tool_T_color"])
        world_T_depth = self._tool_pose(q) @ X @ transform(identity["color_T_depth"])
        world_T_board = self._tool_pose(color_q) @ X @ color_T_board
        world_points = points @ world_T_depth[:3, :3].T + world_T_depth[:3, 3]
        report.update(index=len(self.depth_checks), color_stamp=color.stamp, depth_stamp=depth.stamp,
                      color_depth_sync_s=frame_dt, depth_joint_sync_s=dt,
                      base_T_board=world_T_board.tolist(), world_T_depth=world_T_depth.tolist())
        self.depth_checks.append(report)
        import io
        buf = io.BytesIO()
        np.savez_compressed(buf, color=color.pixels, depth=depth.pixels, K=depth.model.K,
                            q=q, color_q=color_q, color_T_board=color_T_board,
                            color_T_depth=identity["color_T_depth"], tool_T_color=X,
                            world_T_depth=world_T_depth, residual_mm=residual)
        atomic_bytes(self.cal_record_dir / "depth_checks" / f"depth_{report['index']:03d}.npz", buf.getvalue())
        atomic_json(self.cal_record_dir / "depth_checks.json", {"identity": identity, "checks": self.depth_checks})
        for handle in reversed(self.depth_handles):
            handle.remove()
        self.depth_handles = self._board_geometry("/depth_check/rgb_board", world_T_board, (255, 180, 60), .1)
        if len(points):
            colors = np.tile(np.array([70, 230, 120], dtype=np.uint8), (len(points), 1))
            colors[np.abs(residual) > params.get("max_p95_mm", 5.)] = [255, 70, 70]
            self.depth_handles.append(self.server.scene.add_point_cloud("/depth_check/points",
                points=world_points.astype(np.float32), colors=colors, point_size=.0015))
        result = "통과" if report["passed"] else "실패"
        metrics = (f"거리 중앙값 {report['offset_mm']:.2f} mm · P95 {report['p95_abs_mm']:.2f} mm · "
                   f"평면 각도 {report['normal_error_deg']:.2f}°" if "offset_mm" in report else "유효점 부족")
        self.depth_status.content = (f"**RGB·depth {result}** · {metrics}\n"
            f"유효 depth {report['valid_fraction']:.0%} · {report['points']}점\n"
            "주황: RGB 보드 평면 / 초록: depth 점 / 빨강: 거리 기준 초과\n" + "; ".join(report["reasons"]))
        self._cal_visibility()
        return report

    def _obstacles(self):
        rows, names = [], []
        if self.table_enabled.value:
            x, y, _ = self.roi.value
            rows.append([x, y, self.table_z.value - 0.025, 1, 0, 0, 0, 1.5, 1.5, 0.05])
            names.append("table")
        if self.support_enabled.value:
            dims = self.support_dims.value
            if not np.isfinite(dims).all() or min(dims) <= 0:
                raise ValueError("받침대 크기는 양수여야 합니다")
            rows.append([*self.support_pos.value, 1, 0, 0, 0, *dims])
            names.append("support")
        return np.asarray(rows, dtype=float).reshape(-1, 10), names

    def _new_scan(self):
        if self.source is None or self.loaded is not None:
            raise ValueError("저장 결과 보기에서는 새 촬영을 시작할 수 없습니다")
        self._check_calibration()
        run = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        meta = {k:v for k,v in self.session.meta.items() if k in (
            "min_range_m", "max_range_m", "source", "pose_source", "robot_kinematics")}
        session = ScanSession(ROOT / "data", self.object_name.value, run, meta)
        for handle in self.shot_handles.values():
            handle.remove()
        self.shot_handles.clear()
        if self.mesh_handle is not None:
            self.mesh_handle.remove()
        self.mesh = self.mesh_handle = None
        self.session = session
        self.args.object, self.args.run = self.object_name.value, run
        self.args.resume_scan = self.args.resume_calibration = None
        self.cal_record_dir = self.calibration_dir / "samples" / run
        self.samples.clear(); self.holdouts.clear(); self.depth_checks.clear()
        self.candidate = None
        self.fixed_boards = []
        self.identity_locked = None
        self.last_annotated = None
        for handle in reversed(self.cal_handles + self.depth_handles):
            handle.remove()
        self.cal_handles.clear(); self.depth_handles.clear()
        self.show_live.value = self.show_clouds.value = True
        self.capture_status.content = f"**새 촬영 0장**\n`{session.directory}`"
        self.recon_status.content = "새 촬영 프레임으로 복원합니다. 이전 결과는 기존 폴더에 유지됩니다."
        self.cal_status.content = "검증된 보정값 재사용 — 카메라 장착이 같아야 합니다."
        self.board_status.content = "보정은 유지됩니다. 새 촬영에서 ROI와 제외 영역을 확인하세요."
        self.depth_status.content = "저장된 보정의 RGB·depth 검증 결과를 재사용합니다."
        self.operation.content = "새 촬영을 시작했습니다. 물체·위치·받침 높이가 바뀌면 ROI와 제외 영역을 조정하세요."
        self._controls(False)

    def capture(self):
        self._check_calibration()
        frame, q, dt = self.source.sample("depth")
        T = self._tool_pose(q) @ transform(self.calibration["tool_T_depth"])
        boxes, names = self._obstacles()
        index = self.session.save(frame, q, T, self.calibration, self.roi.value, self.radius.value,
                                  boxes, names, {"sync_error_s": dt, "device": self.source.device})
        self.capture_status.content = f"**{index + 1}장 저장** — 권장 8–12개 시점\n`{self.session.directory}`"
        self._draw_saved(self.session.directory)
        self.operation.content = "저장 완료. 겹치는 부분이 보이도록 다른 시점으로 옮겨 정지하세요."
        if self.args.demo:
            self.source.advance()

    def _centre_roi(self):
        self._check_calibration()
        frame, q, _ = self.source.sample("depth")
        h, w = frame.pixels.shape
        y, x = h // 2, w // 2
        region = frame.pixels[max(y-2, 0):y+3, max(x-2, 0):x+3]
        values = region[np.isfinite(region) & (region >= self.cfg["min_range_m"])
                        & (region <= self.cfg["max_range_m"])]
        if not len(values):
            raise ValueError("화면 중앙에 유효한 물체 depth가 없습니다")
        z = float(np.median(values))
        K = frame.model.K
        p = np.array([(x-K[0, 2])*z/K[0, 0], (y-K[1, 2])*z/K[1, 1], z])
        T = self._tool_pose(q) @ transform(self.calibration["tool_T_depth"])
        self.roi.value = tuple(T[:3, :3] @ p + T[:3, 3])

    def _roi_changed(self):
        self._roi_busy = True
        try:
            self.roi_handle.position = tuple(self.roi.value)
            self._draw_roi()
        finally:
            self._roi_busy = False

    def _draw_roi(self):
        import trimesh
        # Draw edges only: a translucent filled sphere can still occlude the object.
        mesh = trimesh.creation.icosphere(subdivisions=1, radius=float(self.radius.value))
        self.server.scene.add_mesh_simple("/roi", vertices=(mesh.vertices + self.roi.value).astype(np.float32),
                                          faces=mesh.faces.astype(np.uint32), color=(80, 160, 255),
                                          wireframe=True, opacity=0.35)

    def _draw_saved(self, directory):
        plan = storage.load_plan(directory)
        frames = storage.load_frames(directory)
        for i, frame in enumerate(frames):
            pts, labels = classify_pixels(frame["depth"], frame["K"], frame["world_T_cam"], plan.camera,
                                          plan.roi_center, plan.roi_radius, 0.0015, stride=3,
                                          obstacles=plan.obstacles)
            points = pts[labels == OBJECT].astype(np.float32)
            if len(points):
                color = tuple(int(v) for v in (np.array([80, 170, 230]) + i * np.array([47, 29, 13])) % 255)
                self.shot_handles[f"{i}/points"] = self.server.scene.add_point_cloud(f"/shots/{i}/points", points=points, colors=color,
                                                  point_size=0.0015, visible=self.show_clouds.value)
            T = frame["world_T_cam"]
            K = frame["K"]
            h, w = frame["depth"].shape
            self.shot_handles[f"{i}/camera"] = self.server.scene.add_camera_frustum(f"/shots/{i}/camera", fov=float(2*np.arctan(h/(2*K[1, 1]))),
                                                 aspect=w/h, scale=0.035, wxyz=quat(T),
                                                 position=tuple(T[:3, 3]), color=(60, 150, 230))

    def _cloud_visibility(self):
        # Rebuild saved clouds with the same segmentation as reconstruction.
        directory = self.loaded or (self.session.directory if self.session else None)
        if directory is not None and (directory / "plan.npz").exists():
            self._draw_saved(directory)

    def _live_visibility(self):
        if hasattr(self, "live_cloud"):
            self.live_cloud.visible = self.show_live.value

    def _camera_visibility(self):
        for handle in self.live_camera_handles.values():
            handle.visible = self.live_camera_valid and self.show_camera.value

    def _draw_current_camera(self, frame, T, dt):
        model, color = frame.model, (255, 145, 35)
        fov = float(2*np.arctan(model.height/(2*model.K[1, 1])))
        if not self.live_camera_handles:
            scene = self.server.scene
            self.live_camera_handles = {
                "pose": scene.add_frame("/live/camera", axes_length=.025, axes_radius=.0006),
                "frustum": scene.add_camera_frustum("/live/camera/frustum", fov=fov,
                    aspect=model.width/model.height, scale=.055, color=color),
                "direction": scene.add_spline_catmull_rom("/live/camera/direction",
                    positions=np.array([[0, 0, 0], [0, 0, .1], [0, 0, .2]], dtype=np.float32),
                    line_width=2., color=color, segments=2),
                "label": scene.add_label("/live/camera/label", text="현재 카메라 (depth)",
                    position=(.015, -.015, 0)),
            }
        pose = self.live_camera_handles["pose"]
        pose.wxyz, pose.position = quat(T), tuple(T[:3, 3])
        self.live_camera_handles["frustum"].fov = fov
        self.live_camera_handles["frustum"].aspect = model.width/model.height
        self.live_camera_valid = True
        self._camera_visibility()
        x, y, z = T[:3, 3]
        distance = np.linalg.norm(T[:3, 3] - np.asarray(self.roi.value))
        self.camera_status.content = (f"**현재 depth 카메라** · base_link 기준 [m]\n"
            f"x {x:.3f} / y {y:.3f} / z {z:.3f}\n"
            f"ROI 중심까지 {distance*100:.1f} cm · 시간 차이 {dt*1000:.1f} ms\n"
            "주황색: 현재 위치·촬영 방향 / 파랑: 저장된 촬영 시점")

    def _draw_mesh(self, mesh):
        self.mesh = mesh
        self.mesh_handle = self.server.scene.add_mesh_simple("/reconstruction", vertices=mesh.vertices.astype(np.float32),
                                          faces=mesh.faces.astype(np.uint32), color=(220, 190, 140), opacity=1.0)

    def _start_reconstruction(self):
        directory = self.loaded or self.session.directory
        if not (directory / "plan.npz").exists():
            raise ValueError("먼저 물체를 촬영하세요")
        plan = storage.load_plan(directory)
        if self.loaded is None:
            # All raw frames use the final ROI/known-obstacle snapshot chosen by the user.
            plan.roi_center, plan.roi_radius = np.asarray(self.roi.value), float(self.radius.value)
            plan.obstacles, plan.obstacle_names = self._obstacles()
            storage.save_plan(directory, plan)
        self.busy = True
        use_icp, voxel = self.icp.value, float(self.voxel.value)
        self._controls(False)
        self.recon_status.content = "복원 중 — 원본 프레임을 보존합니다."
        self.worker.submit(self._reconstruct, directory, use_icp, voxel)

    def _reconstruct(self, directory, use_icp, voxel):
        try:
            from core.reconstruction.pipeline import FusionParams, reconstruct, recon_report
            plan = storage.load_plan(directory)
            result = reconstruct(storage.load_frames(directory), plan,
                                 FusionParams(voxel_mm=voxel), icp=use_icp,
                                 log=lambda text: setattr(self.recon_status, "content", text))
            report = {**recon_report(FusionParams(voxel_mm=voxel), result, use_icp),
                      "world_frame": "base_link", "source": plan.meta.get("source", "unknown")}
            tag = "icp" if use_icp else ""
            path = storage.save_recon(directory, result.mesh, report, tag=tag)
            result.mesh.export(str(path.with_suffix(".obj")))
            import io
            buffer = io.BytesIO()
            np.savez_compressed(buffer, world_T_cam=result.poses)
            atomic_bytes(directory / f"{storage.recon_stem(tag)}_poses.npz", buffer.getvalue())
            with self.lock:
                self._draw_mesh(result.mesh)
                self.show_live.value = self.show_clouds.value = False
                self.recon_status.content = (f"**복원 완료** · {report['faces']}개 면\n"
                                             f"크기: {report['extents_mm']} mm\n`{path}`\n"
                                             "PLY·OBJ·통계·보정된 카메라 자세 저장")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.recon_status.content = f"**복원 실패:** {exc}\n원본 프레임은 유지되었습니다."
        finally:
            with self.lock:
                self.busy = False

    def _controls(self, ready):
        self.new_scan_button.disabled = self.busy or self.source is None or self.loaded is not None
        self.object_name.disabled = self.new_scan_button.disabled
        real_capture = self.source is not None and (not self.args.demo or self.sim) and not self.busy
        already_shot = bool(self.session and self.session.records)
        self.cal_capture.disabled = not real_capture or already_shot
        self.cal_solve.disabled = not real_capture or len(self._active_samples()) < 15 or already_shot
        self.toggle_sample.disabled = not real_capture or not self.samples or already_shot
        self.sample_number.disabled = self.toggle_sample.disabled
        self.hold_capture.disabled = not real_capture or self.candidate is None or already_shot
        self.cal_accept.disabled = not real_capture or len(self.holdouts) < 3 or already_shot
        self.capture_button.disabled = not ready or self.busy or self.loaded is not None
        self.roi_centre_button.disabled = self.capture_button.disabled
        directory = self.loaded or (self.session.directory if self.session else None)
        self.recon_button.disabled = self.busy or directory is None or not (directory / "plan.npz").exists()
        for field in (self.roi, self.radius, self.table_enabled, self.table_z, self.support_enabled,
                      self.support_pos, self.support_dims, self.voxel, self.icp):
            field.disabled = self.busy or (self.loaded is not None and field not in (self.voxel, self.icp))
        self.roi_handle.visible = not self.busy and self.loaded is None
        if self.sim:
            self.sim_depth_bias.disabled = self.busy or self.source.phase != "calibration"

    def update(self):
        import cv2
        with self.lock:
            if self.source is None:
                self.status.content = "저장 결과 보기 · 로봇 연결 없음"
                self.camera_status.content = "저장 결과 보기 — 현재 카메라 위치는 표시하지 않습니다."
                self._controls(False)
                return
            color = self.source.color
            depth = self.source.depth
            if color is not None:
                image = (self.last_annotated if self.last_annotated is not None
                         and time.monotonic() < self.annotated_until else color.pixels)
                self.rgb_preview.content = image_markdown(image)
            if depth is not None:
                d = depth.pixels
                scaled = np.clip((d - self.cfg["min_range_m"]) /
                                 (self.cfg["max_range_m"] - self.cfg["min_range_m"]), 0, 1)
                scaled = np.where(np.isfinite(d), scaled, 0)
                image = cv2.cvtColor(cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO),
                                     cv2.COLOR_BGR2RGB)
                image[~np.isfinite(d)] = 0
                self.depth_preview.content = image_markdown(image)
            try:
                frame, q, dt = self.source.preview("depth")
                self._check_calibration()
                T = self._tool_pose(q) @ transform(self.calibration["tool_T_depth"])
                self._draw_current_camera(frame, T, dt)
            except Exception as exc:
                self.live_camera_valid = False
                self._camera_visibility()
                self.camera_status.content = f"현재 카메라 위치 대기: {exc}"
            try:
                frame, q, dt = self.source.sample("depth")
                self._check_calibration()
                T = self._tool_pose(q) @ transform(self.calibration["tool_T_depth"])
                pts = pixel_world_points(frame.pixels, frame.model.K, T, stride=5)
                valid = np.isfinite(pts).all(axis=-1) & (frame.pixels[::5, ::5] >= self.cfg["min_range_m"]) & (frame.pixels[::5, ::5] <= self.cfg["max_range_m"])
                if np.any(valid):
                    self.live_cloud = self.server.scene.add_point_cloud(
                        "/live/points", points=pts[valid].astype(np.float32),
                        colors=(110, 130, 150), point_size=0.0015, visible=self.show_live.value)
                self.status.content = f"**촬영 가능** · {self.source.serial}\n시간 차이 {dt*1000:.1f}ms · base_link 좌표"
                self._controls(True)
            except Exception as exc:
                if hasattr(self, "live_cloud"):
                    self.live_cloud.visible = False
                self.status.content = f"촬영 대기: {exc}"
                self._controls(False)

    def close(self):
        self.worker.shutdown(wait=True)
        try:
            if self.source is not None:
                self.source.close()
        finally:
            self.server.stop()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "workcell/scans/piper_d405.yaml"))
    p.add_argument("--object", default="piper_test")
    p.add_argument("--run", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    p.add_argument("--port", type=int, default=8082)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--demo", action="store_true", help="로봇·ROS 없이 합성 데이터로 화면 확인")
    p.add_argument("--sim-calibration", action="store_true", help="가상 보드 검출부터 손목 보정·촬영·복원 시험")
    p.add_argument("--sim-auto", action="store_true", help="가상 보정 15+3개, 촬영 8개와 복원을 자동 실행")
    p.add_argument("--sim-depth-bias-mm", type=float, default=0., help="가상 보드 depth에 거리 오차 추가 [-30,30] mm")
    p.add_argument("--demo-frames", type=int, default=0, help="데모 시작 시 지정 개수의 합성 시점 저장")
    p.add_argument("--view-run", help="저장 폴더를 읽어 복원/보기, 로봇 연결 없음")
    p.add_argument("--resume-calibration", help="저장된 보정 원본 폴더를 복구하고 다시 계산")
    p.add_argument("--resume-scan", help="같은 물체·장착 상태에서 저장된 실제 촬영을 이어가기")
    p.add_argument("--allow-robot-control", action="store_true",
                   help="PiPER 드라이버·MoveIt 이 켜져 있어도 연결 — 정해 둔 자세로 이동(pose_tour)하며 "
                        "보기·찍기. 기본은 수동 촬영 안전을 위해 거부")
    args = p.parse_args()
    if args.resume_scan:
        if args.view_run or args.resume_calibration or args.demo or args.sim_calibration:
            p.error("실제 촬영 복구는 결과 보기/보정 복구/가상 모드와 함께 사용할 수 없습니다")
        path = Path(args.resume_scan).resolve()
        if not path.is_relative_to((ROOT/"data").resolve()) or path.parent.name != "scan":
            p.error("촬영 복구 경로는 프로젝트 data/<물체>/scan/<run>이어야 합니다")
        args.object, args.run = path.parent.parent.name, path.name
    if args.resume_calibration:
        if args.view_run or args.sim_auto or args.demo_frames:
            p.error("보정 복구는 결과 보기/자동 데모와 함께 사용할 수 없습니다")
        args.run = Path(args.resume_calibration).resolve().name
    if not np.isfinite(args.sim_depth_bias_mm) or abs(args.sim_depth_bias_mm) > 30:
        p.error("가상 depth 오차는 -30~30 mm 범위여야 합니다")
    if args.sim_depth_bias_mm and not args.sim_calibration:
        p.error("--sim-depth-bias-mm은 --sim-calibration과 함께 사용하세요")
    if args.sim_auto and not args.sim_calibration:
        p.error("--sim-auto는 --sim-calibration과 함께 사용하세요")
    if args.sim_calibration:
        args.demo = True
        if args.object == "piper_test":
            args.object = "piper_sim"
    if args.demo and args.object == "piper_test":
        args.object = "piper_demo"
    if args.demo and args.view_run:
        p.error("--demo와 --view-run은 함께 사용할 수 없습니다")
    if args.demo_frames and (not args.demo or args.demo_frames < 1):
        p.error("--demo-frames는 --demo와 양수 개수를 사용하세요")
    if args.demo_frames and args.sim_calibration:
        p.error("가상 보정 자동 실행에는 --demo-frames 대신 --sim-auto를 사용하세요")
    cfg = read_config(args.config)
    source = fk = ui = None
    try:
        if not args.view_run:
            if args.demo:
                source = CalibrationSimSource(args.sim_depth_bias_mm) if args.sim_calibration else DemoSource()
            else:
                fk = PiperFK(cfg["urdf"], cfg["base_frame"], cfg["tool_frame"])
                source = LiveSource(cfg, allow_controllers=args.allow_robot_control)
        ui = ScanUI(args, cfg, source, fk)
        if args.sim_auto:
            try:
                for _ in range(15):
                    ui._cal_capture()
                ui._cal_solve()
                for _ in range(3):
                    ui._cal_capture(holdout=True)
                ui._cal_accept()
                print("Simulation calibration:", ui.calibration["simulation_truth_error"], flush=True)
                print("RGB-depth validation:", ui.calibration["depth_validation"], flush=True)
                for _ in range(8):
                    ui.capture()
                ui._start_reconstruction()
            except ValueError as exc:
                ui.operation.content = f"**가상 검증 중단:** {exc}\n오차를 수정한 뒤 검증 자세를 다시 기록하세요."
                print("Simulation validation stopped:", exc, flush=True)
        for _ in range(args.demo_frames):
            ui.capture()
        print(f"PiPER Scan: http://{args.host}:{args.port} (Ctrl+C로 종료)", flush=True)
        while True:
            ui.update()
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        if ui is not None:
            ui.close()
        elif source is not None:
            source.close()


if __name__ == "__main__":
    main()
