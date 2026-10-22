#!/usr/bin/env python3
"""Headless checks for PiPER/UR real scanning. Never opens real CAN, camera, RTDE or ROS devices."""
from collections import deque
import json
from pathlib import Path
import socket
import struct
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from core.realscan.calibration import CharucoBoard, check_board_depth, solve_hand_eye, validate_holdout  # noqa: E402
from core.realscan.capture import (  # noqa: E402
    CameraFrame, CameraModel, JointHistory, ScanSession, atomic_bytes, atomic_json, depth_meters, median_depth, rigid,
    transform,
)
from core.realscan.piper import PassivePiper, PiperFK  # noqa: E402
from core.realscan.sources import CalibrationSimSource, DemoSource  # noqa: E402
from core.reconstruction import storage  # noqa: E402
from core.reconstruction.pipeline import FusionParams, reconstruct  # noqa: E402


class ScanChecks(unittest.TestCase):
    def test_icp_accepts_readonly_transform_buffers(self):
        from core.reconstruction.registration import _vee
        r=np.array([.1,-.2,.05]);t=np.array([.01,.02,.03])
        T=rigid(Rotation.from_rotvec(r).as_matrix(),t)
        T.setflags(write=False)
        np.testing.assert_allclose(_vee(T),np.concatenate([r,t]),atol=1e-12)
        self.assertFalse(T.flags.writeable)
        import open3d as o3d
        result=o3d.pipelines.registration.RegistrationResult()
        np.testing.assert_allclose(_vee(result.transformation),np.zeros(6),atol=1e-12)

    def test_configurable_rotation_tolerance_keeps_translation_gate(self):
        reference = [np.eye(4)]*15
        held = [rigid(Rotation.from_euler('z',1.5,degrees=True).as_matrix(),[0,0,0])]*3
        with self.assertRaises(ValueError):
            validate_holdout(reference,held)
        report=validate_holdout(reference,held,max_rotation_deg=2.)
        self.assertAlmostEqual(report["rotation_max_deg"],1.5)
        self.assertEqual(report["limits"]["max_rotation_deg"],2.)
        self.assertEqual(report["limits"]["max_translation_mm"],5.)
        shifted=[T.copy() for T in held]
        shifted[0][0,3]=.006
        with self.assertRaises(ValueError):
            validate_holdout(reference,shifted,max_rotation_deg=2.)
        with self.assertRaises(ValueError):
            validate_holdout(reference,held,max_rotation_deg=float('nan'))
    def test_calibration_failure_feedback_resume_and_explicit_exclusion(self):
        import socket
        from apps import piper_scan as app
        cfg = app.read_config(ROOT / "workcell/scans/piper_d405.yaml")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0));port = sock.getsockname()[1]
        args = SimpleNamespace(demo=True, sim_calibration=True, view_run=None,
            object="simulation", run="resume_check", host="127.0.0.1", port=port)
        with tempfile.TemporaryDirectory() as root, patch.object(app, "ROOT", Path(root)):
            ui = app.ScanUI(args, cfg, CalibrationSimSource(), None)
            try:
                for _ in range(15):
                    ui._cal_capture()
                original = (ui.cal_record_dir / "train_001.npz").read_bytes()
                altered = np.array(ui.samples[0]["color_T_board"])
                altered[:3,:3] = altered[:3,:3] @ Rotation.from_rotvec([.15,0,0]).as_matrix()
                ui.samples[0]["color_T_board"] = altered.tolist()
                ui._write_samples()
                ui._action(ui._cal_solve)
                self.assertIn("보정 계산 실패", ui.cal_status.content)
                self.assertIsNone(ui.candidate)
                self.assertIsNone(ui.calibration)
                self.assertGreater(len(ui.cal_handles), 40)
                self.assertTrue((ui.cal_record_dir/"solve_diagnostic.json").exists())
                directory = ui.cal_record_dir
            finally:
                ui.close()

            args.resume_calibration = str(directory)
            ui = app.ScanUI(args, cfg, CalibrationSimSource(), None)
            try:
                self.assertEqual(len(ui.samples),15)
                self.assertIn("보정 계산 실패", ui.cal_status.content)
                ui.sample_number.value = 1
                ui._toggle_sample()
                self.assertEqual(len(ui._active_samples()),14)
                ui._controls(False)
                self.assertTrue(ui.cal_solve.disabled)
                ui._cal_capture()  # A fresh valid pose replaces the explicitly excluded one.
                ui._cal_solve()
                self.assertIsNotNone(ui.candidate)
                self.assertEqual(len(ui.samples),16)
                self.assertEqual((directory/"train_001.npz").read_bytes(),original)
                self.assertTrue(ui.samples[0]["excluded"])
            finally:
                ui.close()

    def test_depth_plane_noise_bias_extrinsics_and_missing(self):
        source = CalibrationSimSource()
        source.index = 15
        source._render()
        board_T, _, _ = CharucoBoard().detect(source.color)
        report, points, residual = check_board_depth(source.depth, board_T, np.eye(4))
        self.assertTrue(report["passed"], report)
        self.assertGreater(len(points), 100)
        self.assertEqual(len(points), len(residual))
        rng = np.random.default_rng(3)
        noisy = source.depth.pixels + rng.normal(0, .001, source.depth.pixels.shape).astype(np.float32)
        noisy[rng.random(noisy.shape) < .1] = np.inf
        frame = CameraFrame(source.depth.stamp, noisy, source.depth_model)
        report, _, _ = check_board_depth(frame, board_T, np.eye(4))
        self.assertTrue(report["passed"], report)
        source.depth_bias_m = .01
        source._render()
        report, points, residual = check_board_depth(source.depth, board_T, np.eye(4))
        self.assertFalse(report["passed"])
        self.assertGreater(report["p95_abs_mm"], 8)
        self.assertEqual(report["points"], report["projected_points"])
        self.assertGreater(np.mean(np.abs(residual) > 5), .9)  # Bad points are never silently filtered.
        source.depth_bias_m = 0
        source._render()
        wrong_tf = rigid(np.eye(3), [0, 0, .02])
        report, _, _ = check_board_depth(source.depth, board_T, wrong_tf)
        self.assertFalse(report["passed"])
        # Open gripper fingers in front of the board: they tilt the plane fit unless dropped as occluders,
        # and dropping them must not hide a depth bias (points behind the plane are kept).
        fingers = source.depth.pixels.copy()
        fingers[180:300, 240:330] = .10
        frame = CameraFrame(source.depth.stamp, fingers, source.depth_model)
        report, _, _ = check_board_depth(frame, board_T, np.eye(4))
        self.assertFalse(report["passed"])
        report, _, _ = check_board_depth(frame, board_T, np.eye(4), ignore_nearer_m=.03)
        self.assertTrue(report["passed"], report)
        self.assertGreater(report["occluded_points"], 500)
        biased = CameraFrame(source.depth.stamp, np.where(fingers > .11, fingers + .01, fingers).astype(np.float32),
                             source.depth_model)
        report, _, _ = check_board_depth(biased, board_T, np.eye(4), ignore_nearer_m=.03)
        self.assertFalse(report["passed"])
        missing = CameraFrame(source.depth.stamp, np.full_like(source.depth.pixels, np.inf), source.depth_model)
        report, _, _ = check_board_depth(missing, board_T, np.eye(4))
        self.assertFalse(report["passed"])
        self.assertEqual(report["points"], 0)
        self.assertNotIn("offset_mm", report)

    def test_simulated_board_calibration_and_scan(self):
        import socket
        from apps import piper_scan as app
        cfg = app.read_config(ROOT / "workcell/scans/piper_d405.yaml")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        args = SimpleNamespace(demo=True, sim_calibration=True, view_run=None,
                               object="simulation", run="calibration_check", host="127.0.0.1", port=port)
        with tempfile.TemporaryDirectory() as root, patch.object(app, "ROOT", Path(root)):
            ui = app.ScanUI(args, cfg, CalibrationSimSource(), None)
            try:
                with self.assertRaises(ValueError):
                    ui.capture()
                for _ in range(15):
                    ui._cal_capture()
                ui._cal_solve()
                self.assertGreater(len(ui.cal_handles), 40)
                np.testing.assert_allclose(ui.cal_handles[0].position, ui.source.base_T_board[:3, 3], atol=.002)
                original_sample = ui.source.sample
                def unsynchronized(kind, near=None):
                    frame, q, dt = original_sample(kind, near)
                    if kind == "depth":
                        frame = CameraFrame(frame.stamp-.2, frame.pixels, frame.model)
                    return frame, q, dt
                with patch.object(ui.source, "sample", side_effect=unsynchronized):
                    with self.assertRaisesRegex(ValueError, "시간 차이"):
                        ui._cal_capture(holdout=True)
                self.assertEqual(len(ui.holdouts), 0)
                ui.source.depth_bias_m = .01
                with self.assertRaisesRegex(ValueError, "RGB·depth 검증 실패"):
                    ui._cal_capture(holdout=True)
                self.assertEqual(len(ui.holdouts), 0)
                self.assertFalse(ui.depth_checks[-1]["passed"])
                self.assertIsNone(ui.calibration)
                self.assertTrue((ui.cal_record_dir / "depth_checks/depth_000.npz").exists())
                self.assertIn("실패", ui.depth_status.content)
                self.assertGreater(len(ui.depth_handles), 3)
                ui.source.depth_bias_m = 0
                for _ in range(3):
                    ui._cal_capture(holdout=True)
                # A wrong hand-eye candidate must fail independent pose validation,
                # even when local RGB/depth plane checks passed.
                candidate = np.array(ui.candidate["tool_T_color"])
                shifted = candidate.copy()
                shifted[:3, 3] += [.04, 0, 0]
                ui.candidate["tool_T_color"] = shifted.tolist()
                ui._draw_calibration()
                with self.assertRaises(ValueError):
                    ui._cal_accept()
                self.assertIsNone(ui.calibration)
                ui.candidate["tool_T_color"] = candidate.tolist()
                ui._cal_accept()
                self.assertTrue(ui.calibration["demo"])
                self.assertTrue(ui.calibration["depth_validation"]["passed"])
                self.assertEqual(ui.calibration["depth_validation"]["samples"], 3)
                self.assertGreater(len(ui.depth_checks), 3)
                np.testing.assert_allclose(ui.calibration["base_T_board"], ui.source.base_T_board, atol=.003)
                ui.show_calibration.value = False
                ui._cal_visibility()
                self.assertTrue(all(not h.visible for h in ui.cal_handles + ui.depth_handles))
                self.assertLess(ui.calibration["simulation_truth_error"]["translation_mm"], 1)
                self.assertLess(ui.calibration["simulation_truth_error"]["rotation_deg"], .5)
                self.assertIn("simulation", ui.calibration_path.parts)
                self.assertFalse((Path(root)/"workcell/calibration/piper_d405.yaml").exists())
                self.assertEqual(ui.source.phase, "scan")
                # Board work frame: centre origin, z up. The virtual board centre is the object centre.
                ui.source.phase = "calibration"
                ui.source._render()
                ui._set_board_frame()
                self.assertTrue(ui.track_board.value)       # the board work frame defaults to tracking
                ui.track_board.value = False                # this part checks arm-FK capture in the board frame
                ui.source.start_scan()
                np.testing.assert_allclose(ui.base_T_world[:3, 3], ui.source.center, atol=.003)
                np.testing.assert_allclose(ui.base_T_world[:3, :3], np.eye(3), atol=.01)
                self.assertEqual(ui.session.meta["world_frame"], "board")
                ui.roi.value = (0., 0., 0.)
                for _ in range(8):
                    ui.capture()
                with self.assertRaisesRegex(ValueError, "바꿀 수 없습니다"):
                    ui._set_board_frame()
                plan = storage.load_plan(ui.session.directory)
                self.assertEqual(plan.meta["source"], "synthetic_calibration")
                self.assertEqual(plan.meta["world_frame"], "board")
                result = reconstruct(storage.load_frames(ui.session.directory), plan,
                                     FusionParams(smooth_iters=0), log=lambda _: None)
                self.assertGreater(len(result.mesh.faces), 100)
                np.testing.assert_allclose(result.mesh.extents[:2], [.08, .08], atol=.008)
                np.testing.assert_allclose(result.mesh.bounds.mean(axis=0)[:2], [0, 0], atol=.006)
                ui._reconstruct(ui.session.directory, False, 2.0)
                report = json.loads((ui.session.directory / "recon.json").read_text())
                self.assertEqual(report["world_frame"], "board")
                np.testing.assert_allclose(report["base_T_world"], ui.base_T_world)
                # A new run keeps the frame; settings from a saved run bring its frame back.
                scanned, frame = ui.session.directory, ui.base_T_world.copy()
                ui._new_scan()
                self.assertEqual(ui.session.meta["world_frame"], "board")
                np.testing.assert_allclose(ui.session.meta["base_T_world"], frame)
                ui._set_world("base_link", np.eye(4))
                self.assertNotIn("base_T_world", ui.session.meta)
                ui._settings_from(scanned)
                self.assertEqual(ui.world_frame, "board")
                self.assertTrue(ui.track_board.value)
                np.testing.assert_allclose(ui.base_T_world, frame)
                # Board tracking: the clamp slipped (8 mm, 1°) so arm-computed poses are wrong,
                # but poses measured from the board in each RGB frame still match the truth.
                ui.source.phase, ui.source.index = "calibration", 0
                ui.source._render()
                ui.source.known_X = ui.source.known_X @ rigid(Rotation.from_rotvec([0, .0175, 0]).as_matrix(),
                                                              [.008, 0, 0])
                ui.track_board.value = True
                world_T_base = np.linalg.inv(ui.source.base_T_board @ app.board_centre_frame(ui.scan_board.spec))
                truth = []
                for k in range(4):
                    truth.append(world_T_base @ ui.source.T)
                    if k < 3:
                        ui.capture()
                    else:                                    # pose_tour --capture → capture service
                        reply = ui._remote_capture(None, SimpleNamespace())
                        self.assertTrue(reply.success, reply.message)
                        self.assertIn("4장 저장 · 코너", reply.message)
                for frame_data, T in zip(storage.load_frames(ui.session.directory), truth):
                    np.testing.assert_allclose(frame_data["world_T_cam"][:3, 3], T[:3, 3], atol=.002)
                    R = frame_data["world_T_cam"][:3, :3].T @ T[:3, :3]
                    self.assertLess(np.rad2deg(Rotation.from_matrix(R).magnitude()), .5)
                records = json.loads((ui.session.directory / "capture.json").read_text())["frames"]
                self.assertTrue(all(r["camera_pose"] == "board" and r["robot_vs_board_mm"] > 6 for r in records))
                self.assertEqual(storage.load_plan(ui.session.directory).meta["camera_pose"], "board")
                self.assertIn("팔로 계산한 위치와 차이", ui.capture_status.content)
                ui.source.start_scan()                       # no board in view: nothing is saved
                with self.assertRaisesRegex(ValueError, "보드 추적"):
                    ui.capture()
                reply = ui._remote_capture(None, SimpleNamespace())
                self.assertFalse(reply.success)
                self.assertIn("보드 추적", reply.message)
                self.assertEqual(len(ui.session.records), 4)
                ui.track_board.value = False                 # one pose source per run
                with self.assertRaisesRegex(ValueError, "켜고"):
                    ui.capture()
                ui._controls(True)
                self.assertTrue(ui.track_board.disabled)
                ui._new_scan()
                self.assertTrue(ui.track_board.value)        # a new run in the board frame starts tracked
                ui._set_world("base_link", np.eye(4))        # tracking needs the board work frame
                ui._controls(True)
                self.assertFalse(ui.track_board.value)
                self.assertTrue(ui.track_board.disabled)
            finally:
                ui.close()

    def test_app_capture_and_background_reconstruction(self):
        import socket
        from urllib.request import urlopen
        from apps import piper_scan as app
        cfg = app.read_config(ROOT / "workcell/scans/piper_d405.yaml")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        args = SimpleNamespace(demo=True, view_run=None, object="demo", run="ui_check",
                               host="127.0.0.1", port=port)
        with tempfile.TemporaryDirectory() as root, patch.object(app, "ROOT", Path(root)):
            ui = app.ScanUI(args, cfg, DemoSource(), None)
            try:
                ui.update()
                self.assertFalse(ui.capture_button.disabled)
                np.testing.assert_allclose(ui.live_camera_handles["pose"].position, ui.source.T[:3,3])
                ui.show_live.value = False
                with patch.object(ui.source, "sample", side_effect=ValueError("팔이 움직이고 있습니다")), \
                     patch.object(ui.source, "preview", return_value=(ui.source.depth, np.zeros(6), 0.)):
                    ui.source.T[:3,3] += [.01,0,0]
                    ui.update()
                    self.assertTrue(ui.live_camera_valid)
                    self.assertTrue(ui.live_camera_handles["frustum"].visible)
                    np.testing.assert_allclose(ui.live_camera_handles["pose"].position, ui.source.T[:3,3])
                    self.assertTrue(ui.capture_button.disabled)
                    self.assertEqual(len(ui.session.records),0)
                with patch.object(ui.source, "preview", side_effect=ValueError("오래된 피드백")):
                    ui.update()
                    self.assertFalse(ui.live_camera_valid)
                    self.assertTrue(all(not h.visible for h in ui.live_camera_handles.values()))
                ui.show_camera.value = False
                ui.update()
                self.assertTrue(ui.live_camera_valid)
                self.assertTrue(all(not h.visible for h in ui.live_camera_handles.values()))
                ui.show_camera.value = True
                ui._camera_visibility()
                with urlopen(f"http://127.0.0.1:{port}", timeout=5) as response:
                    self.assertEqual(response.status, 200)
                # 자세마다 depth 를 설정 장수(depth_frames)만큼 받아 중앙값 — 그 사이 팔이 움직이면 저장 안 함
                self.assertEqual(cfg["depth_frames"], 5)
                original_sample, calls = ui.source.sample, []
                def drifting(kind, near=None):
                    frame, q, dt = original_sample(kind, near)
                    calls.append(kind)
                    return frame, q + (.01 if len(calls) > 1 else 0.), dt
                with patch.object(ui.source, "sample", side_effect=drifting):
                    with self.assertRaisesRegex(ValueError, "팔이 움직였습니다"):
                        ui.capture()
                self.assertEqual(len(ui.session.records), 0)
                for _ in range(8):
                    ui.capture()
                records = json.loads((ui.session.directory / "capture.json").read_text())["frames"]
                self.assertTrue(all(r["depth_frames"] == 5 for r in records))
                self.assertIn("depth 5장 중앙값", ui.capture_status.content)
                ui._start_reconstruction()
                self.assertTrue(ui.capture_button.disabled)
                deadline = time.monotonic() + 30
                while ui.busy and time.monotonic() < deadline:
                    time.sleep(.05)
                self.assertFalse(ui.busy)
                self.assertIsNotNone(ui.mesh, ui.recon_status.content)
                self.assertGreater(len(ui.mesh.faces), 100)
                self.assertTrue((ui.session.directory / "recon.obj").exists())
                self.assertTrue((ui.session.directory / "recon_poses.npz").exists())
                ui.update()
                self.assertFalse(ui.capture_button.disabled)
            finally:
                ui.close()
            original=(Path(root)/"data/demo/scan/ui_check/frames/frame_000.npz").read_bytes()
            args.resume_scan=str(Path(root)/"data/demo/scan/ui_check")
            ui=app.ScanUI(args,cfg,DemoSource(),None)
            try:
                self.assertEqual(len(ui.session.records),8)
                self.assertIsNotNone(ui.mesh)
                np.testing.assert_allclose(ui.roi.value,[.25,0,.075])
                self.assertTrue(ui.table_enabled.value)
                ui.capture()
                self.assertEqual(len(ui.session.records),9)
                self.assertEqual((ui.session.directory/"frames/frame_000.npz").read_bytes(),original)
                previous=ui.session.directory
                old_files={p.name:p.read_bytes() for p in (previous/"frames").glob('frame_*.npz')}
                roi=ui.roi.value;cal=app.json_signature(ui.calibration)
                ui.object_name.value="repeat"
                ui._new_scan()
                self.assertEqual(len(ui.session.records),0)
                self.assertIsNone(ui.mesh)
                self.assertIsNone(ui.mesh_handle)
                self.assertEqual(len(ui.shot_handles),0)
                self.assertEqual(ui.roi.value,roi)
                self.assertTrue(ui.table_enabled.value)
                self.assertEqual(app.json_signature(ui.calibration),cal)
                self.assertNotEqual(ui.cal_record_dir.name,"ui_check")
                ui.capture()
                self.assertEqual(len(ui.session.records),1)
                self.assertNotEqual(ui.session.directory,previous)
                for name,data in old_files.items():
                    self.assertEqual((previous/"frames"/name).read_bytes(),data)
                self.assertTrue((previous/"recon.obj").exists())
            finally:
                ui.close()

    def test_atomic_writes_are_readable_by_others(self):
        # Container (root) writes; the host user must still read frames and calibration.
        with tempfile.TemporaryDirectory() as root:
            for path, write in ((Path(root)/"a/frame.npz", lambda p: atomic_bytes(p, b"x")),
                                (Path(root)/"b/capture.json", lambda p: atomic_json(p, {"k": 1}))):
                write(path)
                self.assertEqual(path.stat().st_mode & 0o777, 0o644)
                write(path)  # Replacing an existing file keeps the same mode.
                self.assertEqual(path.stat().st_mode & 0o777, 0o644)
            self.assertEqual([p.name for p in Path(root).rglob(".writing-*")], [])

    def test_depth_units_and_invalid(self):
        image = np.array([[0, 70, 200, 500]], dtype=np.uint16)
        result = depth_meters(image, "16UC1")
        self.assertTrue(np.isinf(result[0, 0]))
        np.testing.assert_allclose(result[0, 1:], [.07, .2, .5])
        float_depth = np.array([[np.nan, np.inf, -1, 0, .2]], dtype=np.float32)
        self.assertTrue(np.isinf(depth_meters(float_depth, "32FC1")[0, :4]).all())
        self.assertAlmostEqual(float(depth_meters(float_depth, "32FC1")[0, -1]), .2)
        with self.assertRaises(ValueError):
            depth_meters(image, "rgb8")

    def test_median_depth_rejects_noise_fliers_and_lone_values(self):
        model = CameraModel(4, 3, np.array([[100., 0, 2], [0, 100., 1.5], [0, 0, 1]]), np.zeros(5), "d", "none")
        rng = np.random.default_rng(0)
        frames = []
        for k in range(5):
            d = (.2 + rng.normal(0, .001, (3, 4))).astype(np.float32)
            if k == 2:
                d[0, 0] = .25                       # 한 장에만 튄 값
            if k < 2:
                d[0, 1] = np.inf                    # 두 장에서 빈 픽셀 → 나머지 세 장으로 채움
            if k < 3:
                d[0, 2] = np.inf                    # 세 장에서 빈 픽셀 → 과반이 없으니 무효
            if k > 0:
                d[0, 3] = np.inf                    # 한 장에만 값 → 무효(한 장의 튄 값으로 메우지 않음)
            frames.append(CameraFrame(10. + k * .07, d, model))
        out = median_depth(frames)
        self.assertAlmostEqual(out.stamp, 10.14)    # 가운데 장의 시각
        self.assertLess(abs(out.pixels[0, 0] - .2), .003)
        self.assertTrue(np.isfinite(out.pixels[0, 1]))
        self.assertTrue(np.isinf(out.pixels[0, 2]) and np.isinf(out.pixels[0, 3]))
        rest = out.pixels[1:]
        self.assertLess(np.std(rest - .2), np.std(frames[0].pixels[1:] - .2))
        self.assertIs(median_depth(frames[:1]), frames[0])
        other = CameraModel(4, 3, np.eye(3) * 50, np.zeros(5), "d", "none")
        with self.assertRaisesRegex(ValueError, "카메라 설정"):
            median_depth([frames[0], CameraFrame(11., frames[1].pixels, other)])

    def test_stability_time_and_incomplete_feedback(self):
        hist = JointHistory()
        for stamp in np.arange(0, 1.01, .01):
            hist.add(stamp, np.ones(6) * stamp * .001)
        q, delta = hist.at_rest(.995, now=1.01)
        np.testing.assert_allclose(q, np.ones(6) * .000995)
        self.assertLess(delta, .01)
        with self.assertRaises(ValueError):
            hist.at_rest(.3, now=1.01)
        with self.assertRaises(ValueError):
            hist.at_rest(1, now=2)
        with self.assertRaises(ValueError):
            hist.add(1.02, [1, 2])
        hist.add(1.02, np.ones(6))
        with self.assertRaises(ValueError):
            hist.at_rest(1.02, now=1.02)

    def test_moving_camera_preview_does_not_relax_capture_guard(self):
        import threading
        from core.realscan.sources import LiveSource
        source = LiveSource.__new__(LiveSource)  # No ROS/CAN connection.
        source.lock = threading.RLock()
        source.arm = SimpleNamespace(error="")
        source.serial = "TEST"
        source._check_other_controllers = lambda: None
        source.history = JointHistory()
        now = time.time()
        for offset in np.arange(-.8, .001, .01):
            source.history.add(now+offset, np.full(6,offset))
        model = CameraModel(2,2,np.eye(3),np.zeros(5),"depth_optical")
        source.depth = CameraFrame(now-.125,np.ones((2,2),dtype=np.float32),model)
        frame, q, dt = source.preview()
        self.assertIs(frame,source.depth)
        np.testing.assert_allclose(q,np.full(6,-.125),atol=1e-6)
        self.assertLess(dt,.01)
        with self.assertRaisesRegex(ValueError,"움직이고"):
            source.sample("depth")
        source.depth = CameraFrame(now-2,source.depth.pixels,model)
        with self.assertRaisesRegex(ValueError,"오래"):
            source.preview()
        source.arm.error = "CAN 피드백 오류"
        with self.assertRaisesRegex(ValueError,"CAN"):
            source.preview()

    def test_controller_guard_is_opt_out_only(self):
        from core.realscan.sources import LiveSource
        source = LiveSource.__new__(LiveSource)  # No ROS/CAN connection.
        source.node = SimpleNamespace(get_node_names=lambda: ["agx_arm_ctrl_single_node", "move_group"])
        with self.assertRaisesRegex(ValueError, "MoveIt"):
            source._check_other_controllers()        # Default: manual scanning refuses robot control.
        source.allow_controllers = True
        source._check_other_controllers()            # --allow-robot-control: connect alongside them.

    def test_handeye_direction_and_holdout(self):
        rng = np.random.default_rng(2)
        X = rigid(Rotation.from_rotvec([.15, -.25, .12]).as_matrix(), [.035, .01, .05])
        B = rigid(Rotation.from_rotvec([.1, .05, -.12]).as_matrix(), [.25, .1, .05])
        tools = [rigid(Rotation.from_rotvec(rng.normal(size=3)*.5).as_matrix(),
                       rng.uniform([.05, -.1, .15], [.3, .2, .35])) for _ in range(18)]
        boards = [np.linalg.inv(T @ X) @ B for T in tools]
        solved, report, fixed = solve_hand_eye(tools[:15], boards[:15])
        np.testing.assert_allclose(solved, X, atol=1e-8)
        self.assertLess(report["translation_max_mm"], 1e-5)
        hold = [T @ solved @ C for T, C in zip(tools[15:], boards[15:])]
        validate_holdout(fixed, hold)
        wrong = [T.copy() for T in hold]
        wrong[0][0, 3] += .02
        with self.assertRaises(ValueError):
            validate_holdout(fixed, wrong)
        with self.assertRaises(ValueError):
            solve_hand_eye([np.eye(4)]*15, [np.eye(4)]*15)
        with self.assertRaises(ValueError):
            transform(np.zeros((4, 4)))

    def test_board_svg_and_pnp(self):
        board = CharucoBoard()
        self.assertIn('width="140mm"', board.svg())
        b = board.board
        image = b.draw((1200, 1000), marginSize=0) if hasattr(b, "draw") else b.generateImage((1200, 1000), marginSize=0)
        canvas = np.full((1200, 1400, 3), 255, dtype=np.uint8)
        canvas[100:1100, 100:1300] = image[..., None]
        model = CameraModel(1400, 1200, np.array([[2000., 0, 700], [0, 2000., 600], [0, 0, 1]]),
                            np.zeros(5), "color_optical")
        T, rms, _ = board.detect(CameraFrame(1, canvas, model))
        np.testing.assert_allclose(T[:3, 3], [-.06, -.05, .2], atol=.001)
        self.assertLess(rms, .2)

    def test_scan_board_detects_around_object(self):
        """스캔용 큰 보드(9×7, 25 mm): 가운데 물체와 그 뒤쪽이 가려도 남은 코너로 같은 자세가 나온다."""
        board = CharucoBoard({"squares_x": 9, "squares_y": 7, "square_m": .025, "marker_m": .018,
                              "dictionary": "DICT_4X4_50", "layout": "opencv_4.6_even_row_legacy"})
        self.assertIn('width="245mm"', board.svg())
        image = board.image()                                  # 10 px/mm → 칸 250 px
        self.assertEqual(image.shape, (1750, 2250))
        canvas = np.full((1950, 2450, 3), 255, dtype=np.uint8)
        canvas[100:1850, 100:2350] = image[..., None]
        model = CameraModel(2450, 1950, np.array([[2000., 0, 1225], [0, 2000., 975], [0, 0, 1]]),
                            np.zeros(5), "color_optical")
        T, _, _ = board.detect(CameraFrame(1, canvas, model))
        np.testing.assert_allclose(T[:3, 3], [-.1125, -.0875, .2], atol=.001)
        canvas[100:1225, 975:1475] = 90                        # 5 cm 큐브 + 뒤로 가려지는 띠
        T_occluded, rms, _ = board.detect(CameraFrame(1, canvas, model))
        np.testing.assert_allclose(T_occluded, T, atol=1e-3)
        self.assertLess(rms, .5)
        with self.assertRaisesRegex(ValueError, "모자랍니다"):
            CharucoBoard({**board.spec, "squares_x": 11, "squares_y": 10})

    def test_scan_poses_pull_views_in_and_keep_fingers_clear(self):
        """scan_poses: 시점 방향을 유지하고 물체 쪽으로 당기되, 손가락이 상자에 닿거나 코너가 모자라면 뺀다."""
        from core.realscan import scan_poses as sp
        from core.realscan.calib_poses import FastFK
        fk = PiperFK(ROOT / "workcell/robot/piper/piper_description.urdf", "base_link", "link6")
        ffk = FastFK(fk)
        mount = Rotation.from_matrix([[0.003, 0.709, 0.705], [-1.0, 0.010, -0.006], [-0.011, -0.705, 0.709]])
        tool_T_depth = rigid(mount.as_matrix(), [-0.092, 0.012, 0.063])   # 손목 D405 장착(보정값 근처)
        seed = np.radians([-37.2, 68.8, -39.4, 50.6, 57.7, -92.9])   # 보드를 위에서 보던 교시 자세
        base_T_world = rigid(np.eye(3), [0.27, 0.0, 0.0])
        centre = np.array([0, 0, 0.068])
        u, v, w = np.meshgrid(np.r_[-0.08:-0.04:5j, 0.06:0.10:5j], np.linspace(0, 0.07, 5), np.linspace(0.095, 0.12, 4))
        fingers = np.c_[u.ravel(), v.ravel(), w.ravel()]               # 벌린 손가락 두 덩어리(카메라 좌표)
        K = np.array([[650., 0, 640], [0, 650., 360], [0, 0, 1]])
        spec = {"squares_x": 9, "squares_y": 7, "square_m": 0.025}

        def plan(object_size, **kw):
            scene = sp.Scene(sp.board_corners(spec), [(centre, object_size), ([0, 0, 0.021], [0.04, 0.04, 0.042])],
                             fingers, None, K, (1280, 720), np.eye(4))
            lo, hi = ffk.lower + np.radians(3), ffk.upper - np.radians(3)
            args = dict(distances=[0.16, 0.18, 0.20, 0.22], raise_deg=[0], min_corners=10, finger_clear=0.02,
                        max_move_deg=60, max_reach=0.30, lo=lo, hi=hi)
            return sp.plan_views(ffk, [seed], base_T_world, tool_T_depth, scene, centre, **{**args, **kw})[0]

        r = plan([0.05, 0.05, 0.05])
        self.assertTrue(r["ok"], r)
        cam = np.linalg.inv(base_T_world) @ ffk.pose(r["q"]) @ tool_T_depth
        self.assertAlmostEqual(np.linalg.norm(cam[:3, 3] - centre), r["distance_m"], delta=0.001)
        along = cam[:3, 2] @ (centre - cam[:3, 3])                      # 광축이 물체 중심을 지난다
        np.testing.assert_allclose(cam[:3, 3] + along * cam[:3, 2], centre, atol=0.001)
        self.assertGreaterEqual(r["finger_clear_m"], 0.02)
        self.assertFalse(plan([0.30, 0.30, 0.05])["ok"])               # 손가락이 큰 물체 안으로
        self.assertFalse(plan([0.05, 0.05, 0.05], min_corners=60)["ok"])
        self.assertAlmostEqual(float(sp.box_distance([[0.1, 0, 0]], [0, 0, 0], [0.1, 0.1, 0.1])[0]), 0.05)
        ob = sp.obstacle_for_tour(base_T_world, [(centre, [0.05, 0.05, 0.05])], margin=0.01)
        np.testing.assert_allclose(ob, [0.27, 0, 0.068, 0.07, 0.07, 0.07], atol=1e-9)
        # 회전판: 방향을 직접 주고, 판을 90° 씩 돌린 4 경우 중 가장 나쁜 값으로 본다
        np.testing.assert_allclose(sp.view_directions([(0, 0)], base_T_world, centre)[0], [-1, 0, 0], atol=1e-9)
        top = sp.view_directions([(85, 0)], base_T_world, centre)       # 방위 0 = 물체에서 로봇 쪽
        long_box = [0.12, 0.04, 0.05]
        one = plan(long_box, directions=top, distances=[0.20])
        four = plan(long_box, directions=top, distances=[0.20], turns=4)
        self.assertTrue(four["ok"], four)
        self.assertAlmostEqual(four["elev_deg"], 85, delta=0.01)
        self.assertEqual(len(four["corners_by_turn"]), 4)
        self.assertEqual(four["corners"], min(four["corners_by_turn"]))
        self.assertLessEqual(four["finger_clear_m"], one["finger_clear_m"])
        ob = sp.obstacle_for_tour(base_T_world, [(centre, long_box)], margin=0, turns=4)
        np.testing.assert_allclose(ob[3:5], [0.12, 0.12], atol=1e-9)    # 돌린 긴 물체까지 감싼다

    def test_fk_measured_values_not_clamped(self):
        candidates = [ROOT / "workcell/robot/piper/piper_description.urdf",
                      Path("/workspace/agx_arm_ws/install/agx_arm_description/share/agx_arm_description/agx_arm_urdf/piper/urdf/piper_description.urdf"),
                      ROOT.parent / "agx_arm_ws/src/agx_arm_ros/src/agx_arm_description/agx_arm_urdf/piper/urdf/piper_description.urdf"]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            self.skipTest("Official PiPER URDF not mounted")
        fk = PiperFK(path)
        zero = fk.pose(np.zeros(6))
        np.testing.assert_allclose(zero[:3, 3], [.056135, 0, .213178], atol=1e-6)
        q = np.zeros(6)
        q[1] = -.035
        self.assertGreater(np.linalg.norm(fk.pose(q)[:3, 3] - zero[:3, 3]), .001)

    def test_passive_sdk_and_send_guard(self):
        class FakeComm:
            def __init__(self):
                self.sent = 0
            def send(self, msg):
                self.sent += 1
        class FakeArm:
            def __init__(self, write_on_connect=False):
                self.comm = FakeComm()
                self.closed = False
                self.write_on_connect = write_on_connect
                self._parser = SimpleNamespace(**{n: SimpleNamespace(timestamp=time.time())
                                                  for n in ("joint_12", "joint_34", "joint_56")})
            def get_context(self):
                return SimpleNamespace(get_comm=lambda: self.comm)
            def connect(self):
                if self.write_on_connect:
                    self.comm.send("unexpected")
            def disconnect(self):
                self.closed = True
            def get_joint_angles(self):
                for n in ("joint_12", "joint_34", "joint_56"):
                    getattr(self._parser, n).timestamp = time.time()
                return SimpleNamespace(timestamp=time.time(), msg=[.1]*6)
            def get_arm_status(self):
                return SimpleNamespace(timestamp=time.time(), msg=SimpleNamespace(
                    ctrl_mode=2, arm_status=11, teach_status=1, motion_status=1))
        arm = FakeArm()
        observer = PassivePiper("not-a-real-bus", JointHistory(), factory=lambda: arm)
        time.sleep(.04)
        self.assertEqual(observer.error, "")
        self.assertEqual(arm.comm.sent, 0)
        with self.assertRaises(RuntimeError):
            arm.comm.send("must not transmit")
        self.assertEqual(arm.comm.sent, 0)
        observer.close()
        self.assertTrue(arm.closed)
        unsafe = FakeArm(write_on_connect=True)
        with self.assertRaises(RuntimeError):
            PassivePiper("fake", JointHistory(), factory=lambda: unsafe)
        self.assertTrue(unsafe.closed)
        self.assertEqual(unsafe.comm.sent, 0)

    def test_dataset_roundtrip_reconstruction_and_profile_change(self):
        source = DemoSource()
        cal = {"tool_T_depth": np.eye(4).tolist(), "demo": True}
        with tempfile.TemporaryDirectory() as root:
            session = ScanSession(root, "test", "run", {"min_range_m": .07, "max_range_m": .5,
                                                       "source": "synthetic_demo"})
            for i in range(8):
                frame, q, _ = source.sample("depth")
                session.save(frame, q, source.T, cal, source.center, .06, [], [], {})
                source.advance()
            plan = storage.load_plan(session.directory)
            self.assertEqual(plan.meta["source"], "synthetic_demo")
            self.assertEqual(plan.obstacles.shape, (0, 10))
            frames = storage.load_frames(session.directory)
            result = reconstruct(frames, plan, FusionParams(smooth_iters=0), log=lambda _: None)
            self.assertGreater(len(result.mesh.faces), 100)
            np.testing.assert_allclose(result.mesh.extents[:2], [.08, .08], atol=.008)
            # Upper-hemisphere shots cannot reconstruct the unseen bottom cap.
            self.assertGreater(result.mesh.extents[2], .05)
            self.assertLess(result.mesh.extents[2], .085)
            aligned=reconstruct(frames,plan,FusionParams(smooth_iters=0),icp=True,log=lambda _:None)
            self.assertTrue(np.isfinite(aligned.poses).all())
            self.assertEqual(aligned.poses.shape,(8,4,4))
            self.assertIn("icp",aligned.stats)
            self.assertGreater(len(aligned.mesh.faces),100)
            storage.save_recon(session.directory, result.mesh, {"world_frame": "base_link"})
            self.assertGreater(len(storage.load_recon(session.directory).faces), 100)
            with self.assertRaises(FileExistsError):
                ScanSession(root, "test", "run", {})
            frame, q, _ = source.sample("depth")
            frame.model = CameraModel(160, 120, frame.model.K*2, np.zeros(5), "different")
            with self.assertRaises(ValueError):
                session.save(frame, q, source.T, cal, source.center, .06, [], [], {})
            # Old simulation metadata remains readable and retains its own robot.
            plan.meta = {"robot": "ur20"}
            storage.save_plan(session.directory, plan)
            self.assertEqual(storage.load_plan(session.directory).meta["robot"], "ur20")


def _ros_string(text):
    data = text.encode()
    return struct.pack("<I", len(data)) + data


def _ros_header(stamp, frame):
    sec = int(stamp)
    return struct.pack("<III", 0, sec, int(round((stamp - sec) * 1e9))) + _ros_string(frame)


def _ros_image(stamp, frame, encoding, pixels, pad=0):
    raw = np.ascontiguousarray(pixels).view(np.uint8).reshape(pixels.shape[0], -1)
    rows = np.zeros((raw.shape[0], raw.shape[1] + pad), np.uint8)
    rows[:, :raw.shape[1]] = raw
    return (_ros_header(stamp, frame) + struct.pack("<II", pixels.shape[0], pixels.shape[1])
            + _ros_string(encoding) + struct.pack("<BII", 0, rows.shape[1], rows.size) + rows.tobytes())


def _ros_camera_info(stamp, frame, width, height, K):
    P = [K[0], K[1], K[2], 0, K[3], K[4], K[5], 0, 0, 0, 1, 0]
    return (_ros_header(stamp, frame) + struct.pack("<II", height, width) + _ros_string("plumb_bob")
            + struct.pack("<I5d", 5, *[0.] * 5) + struct.pack("<9d", *K)
            + struct.pack("<9d", *np.eye(3).ravel()) + struct.pack("<12d", *P) + struct.pack("<6I?", *[0] * 6, False))


class FakeRos1:
    """127.0.0.1 XML-RPC master/node + TCPROS publisher/service. Records every XML-RPC call."""
    def __init__(self, topics, services):
        from xmlrpc.server import SimpleXMLRPCServer
        from core.realscan import ros1
        self.ros1, self.topics, self.services, self.calls = ros1, topics, services, []
        self.closed = threading.Event()
        self.tcp = socket.socket()
        self.tcp.bind(("127.0.0.1", 0))
        self.tcp.listen(8)
        port = self.tcp.getsockname()[1]
        self.rpc = SimpleXMLRPCServer(("127.0.0.1", 0), logRequests=False, allow_none=True)
        self.uri = f"http://127.0.0.1:{self.rpc.server_address[1]}/"
        answers = {
            "getSystemState": lambda c: [1, "", [[[t, ["/fake"]] for t in topics], [],
                                                 [[s, ["/fake"]] for s in services]]],
            "lookupNode": lambda c, n: [1, "", self.uri],
            "lookupService": lambda c, s: [1, "", f"rosrpc://127.0.0.1:{port}"],
            "requestTopic": lambda c, t, p: [1, "", ["TCPROS", "127.0.0.1", port]]}
        for name, answer in answers.items():
            self.rpc.register_function(lambda *a, name=name, answer=answer: (self.calls.append(name), answer(*a))[1],
                                       name)
        threading.Thread(target=self.rpc.serve_forever, daemon=True).start()
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                conn, _ = self.tcp.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        ros1 = self.ros1
        with conn:
            try:
                header = ros1._parse_header(ros1._read_block(conn))
                conn.sendall(ros1._header_bytes({"md5sum": "abc", "type": "fake/Type"}))
                if "service" in header:
                    if header.get("probe") != "1":
                        payload = self.services[header["service"]](ros1._read_block(conn))
                        conn.sendall(b"\x01" + struct.pack("<I", len(payload)) + payload)
                    return
                for message in self.topics[header["topic"]]():
                    conn.sendall(struct.pack("<I", len(message)) + message)
                    time.sleep(.01)
                self.closed.wait(5)  # A quiet publisher keeps the connection.
            except (OSError, ConnectionError):
                pass

    def close(self):
        self.closed.set()
        self.rpc.shutdown()
        self.rpc.server_close()
        self.tcp.close()


class FakeRtde:
    """RTDE controller on 127.0.0.1: handshake, a text message, then power-off and running data packets."""
    def __init__(self, powered_off=10, running=50):
        self.server = socket.socket()
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(1)
        self.port = self.server.getsockname()[1]
        self.kinds, self.frequency, self.variables = [], None, None
        self.q = np.array([.1, -1.2, 1.3, -.4, 1.5, .6])
        threading.Thread(target=self._serve, args=(powered_off, running), daemon=True).start()

    def _serve(self, powered_off, running):
        from core.realscan import ur
        conn, _ = self.server.accept()
        self.server.close()
        with conn:
            def reply(kind, payload):
                conn.sendall(struct.pack(">HB", len(payload) + 3, kind) + payload)
            for _ in range(3):
                kind, payload = ur._read(conn)
                self.kinds.append(chr(kind))
                if kind == ur.SETUP_OUTPUTS:
                    self.frequency = struct.unpack(">d", payload[:8])[0]
                    self.variables = payload[8:].decode().split(",")
                    reply(kind, b"\x01DOUBLE,VECTOR6D,INT32,INT32")
                else:
                    reply(kind, b"\x01")
            reply(ur.TEXT, b"\x05hello\x03ctl\x01")
            t = 1000.
            for i in range(powered_off + running):
                t += .008
                reply(ur.DATA, struct.pack(">Bd6dii", 1, t, *self.q, 3 if i < powered_off else 7, 1))
                time.sleep(.002)
            conn.settimeout(5)
            try:
                while True:
                    kind, _ = ur._read(conn)
                    self.kinds.append(chr(kind))
            except (OSError, ConnectionError):
                pass


class UrScanChecks(unittest.TestCase):
    def test_rtde_reader_sets_up_outputs_only_and_gates_on_power(self):
        from core.realscan.ur import RtdeReader
        fake = FakeRtde()
        history = JointHistory()
        reader = RtdeReader("127.0.0.1", fake.port, history, frequency=125)
        try:
            deadline = time.monotonic() + 3
            while len(history.samples) < 50 and time.monotonic() < deadline:
                time.sleep(.02)
            time.sleep(.05)
        finally:
            reader.close()
        self.assertEqual(len(history.samples), 50)  # The 10 power-off packets are not joint data.
        stamps = np.array([s[0] for s in history.samples])
        self.assertTrue(np.all(np.diff(stamps) > 0))
        self.assertLess(abs(time.time() - stamps[-1]), 1.0)
        np.testing.assert_allclose(history.samples[-1][1], fake.q)
        self.assertEqual(reader.status, {"robot_mode": "RUNNING", "safety_mode": "NORMAL"})
        self.assertEqual((fake.frequency, fake.variables), (125, ["timestamp", "actual_q", "robot_mode", "safety_mode"]))
        self.assertEqual(fake.kinds[:3], ["V", "O", "S"])
        self.assertNotIn("I", fake.kinds)  # No input recipe: this connection cannot write to the robot.

    def test_ros1_reader_decodes_and_never_registers(self):
        from core.realscan import ros1
        now = time.time()
        depth = np.arange(12, dtype=np.uint16).reshape(3, 4) * 100
        rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
        K = [400., 0, 2, 0, 401., 1.5, 0, 0, 1]
        R = Rotation.from_rotvec([.01, -.02, .03]).as_matrix()
        topics = {
            "/cam/depth": lambda: [_ros_image(now - 1, "depth_optical", "16UC1", depth),
                                   _ros_image(now + .5, "depth_optical", "16UC1", depth + 1)],
            "/cam/color": lambda: [_ros_image(now, "color_optical", "rgb8", rgb, pad=4)],
            "/cam/info": lambda: [_ros_camera_info(now, "depth_optical", 4, 3, K)],
            "/cam/extrinsics": lambda: [_ros_header(now, "depth") + struct.pack("<9d", *R.ravel(order="F"))
                                        + struct.pack("<3d", .015, 0, 0)]}
        services = {"/cam/device_info": lambda request: b"".join(map(_ros_string, ("d435", "123", "5.13", "3.2")))}
        fake = FakeRos1(topics, services)
        try:
            stamp, _, _, pixels = ros1.decode_image(
                ros1.fetch(fake.uri, "/cam/depth", accept=lambda m: ros1.message_stamp(m) >= now))
            self.assertAlmostEqual(stamp, now + .5, places=5)  # The older queued frame is skipped.
            np.testing.assert_array_equal(pixels, depth + 1)
            self.assertEqual(pixels.dtype, np.uint16)
            _, _, encoding, color = ros1.decode_image(ros1.fetch(fake.uri, "/cam/color"))
            self.assertEqual(encoding, "rgb8")
            np.testing.assert_array_equal(color, rgb)  # Row padding removed.
            info = ros1.decode_camera_info(ros1.fetch(fake.uri, "/cam/info"))
            np.testing.assert_allclose(info["K"].ravel(), K)
            self.assertEqual((info["width"], info["height"], info["frame"]), (4, 3, "depth_optical"))
            T = ros1.decode_extrinsics(ros1.fetch(fake.uri, "/cam/extrinsics"))
            np.testing.assert_allclose(T[:3, :3], R, atol=1e-12)  # librealsense column-major
            np.testing.assert_allclose(T[:3, 3], [.015, 0, 0])
            device = ros1.decode_device_info(ros1.call_service(fake.uri, "/cam/device_info"))
            self.assertEqual(device["serial_number"], "123")
            got = []
            stream = ros1.TopicStream(fake.uri, "/cam/depth", got.append)
            deadline = time.monotonic() + 3
            while len(got) < 2 and time.monotonic() < deadline:
                time.sleep(.02)
            stream.close()
            self.assertEqual((len(got), stream.error), (2, ""))
            with self.assertRaisesRegex(ValueError, "발행하는 노드가 없습니다"):
                ros1.fetch(fake.uri, "/not/published")
            self.assertLessEqual(set(fake.calls), {"getSystemState", "lookupNode", "lookupService", "requestTopic"})
        finally:
            fake.close()

    def test_ur_fk_is_factory_kinematics(self):
        import yaml
        from core.realscan.ur import UrFK
        fk = UrFK(ROOT / "workcell/scans/ur5e_summit.urdf")
        k = yaml.safe_load((ROOT / "workcell/scans/ur5e_summit_kinematics.yaml").read_text())["kinematics"]

        def chain(q):
            T = rigid(Rotation.from_euler("z", np.pi).as_matrix(), [0, 0, 0])  # base_link → base_link_inertia
            for name, angle in zip(("shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3"), q):
                d = k[name]
                T = (T @ rigid(Rotation.from_euler("xyz", [d["roll"], d["pitch"], d["yaw"]]).as_matrix(),
                               [d["x"], d["y"], d["z"]])
                     @ rigid(Rotation.from_euler("z", angle).as_matrix(), [0, 0, 0]))
            return T
        for q in np.random.default_rng(3).uniform(-3, 3, (20, 6)):
            np.testing.assert_allclose(fk.pose(q), chain(q), atol=1e-12)
        with self.assertRaisesRegex(ValueError, "UR"):
            UrFK(ROOT / "workcell/robot/piper/piper_description.urdf")

    def test_ur_source_pairs_depth_with_rgb_time(self):
        from core.realscan.sources import UrLiveSource
        source = UrLiveSource.__new__(UrLiveSource)  # No RTDE or ROS1 connection.
        source.lock, source.streams, source.camera_error = threading.RLock(), [], ""
        source.arm, source.serial, source.depth = SimpleNamespace(error=""), "123", None
        source.history = JointHistory(max_age_s=3.0)
        now = time.time()
        for i in range(200):
            source.history.add(now - 2 + i * .01, np.full(6, .3))
        model = CameraModel(4, 3, np.array([[400., 0, 2], [0, 400., 1.5], [0, 0, 1]]), np.zeros(5), "d", "none")
        source.recent_depth = deque((now - 1 + .1 * i, np.full((3, 4), 500 + i, np.uint16), "16UC1", model)
                                    for i in range(10))
        frame, q, _ = source.sample("depth", near=now - .52)
        self.assertAlmostEqual(frame.stamp, now - .5)
        self.assertAlmostEqual(float(frame.pixels[0, 0]), .505, places=6)
        np.testing.assert_allclose(q, .3)
        source.arm.error = "UR 팔 전원이 꺼져 있습니다 (robot_mode 3)"
        with self.assertRaisesRegex(ValueError, "전원"):
            source.sample("depth", near=now - .5)

    def test_app_takes_robot_and_calibration_file_from_rig_config(self):
        from apps import piper_scan as app
        cfg = app.read_config(ROOT / "workcell/scans/ur5e_d435f.yaml")
        self.assertEqual((cfg["robot"], cfg["label"], cfg["name"]), ("ur5e", "UR5e", "ur5e_d435f"))
        piper = app.read_config(ROOT / "workcell/scans/piper_d405.yaml")
        self.assertEqual((piper["robot"], piper["label"], piper["name"]), ("piper", "PiPER", "piper_d405"))
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        args = SimpleNamespace(demo=True, sim_calibration=False, view_run=None, object="ur_demo", run="r1",
                               host="127.0.0.1", port=port)
        with tempfile.TemporaryDirectory() as root, patch.object(app, "ROOT", Path(root)):
            ui = app.ScanUI(args, cfg, DemoSource(), None)
            try:
                self.assertEqual(ui.calibration_path.name, "ur5e_d435f.yaml")
                self.assertEqual((ui.session.meta["robot"], ui.session.meta["world_frame"]), ("ur5e", "base_link"))
                ui.capture()
                capture = json.loads((ui.session.directory / "capture.json").read_text())
                self.assertEqual(capture["robot"], "ur5e")
            finally:
                ui.close()
            with self.assertRaisesRegex(ValueError, "로봇"):
                ScanSession(Path(root) / "data", "ur_demo", "r1", {"min_range_m": .2, "max_range_m": .8,
                            "source": "synthetic_demo"}, resume=True, robot="piper")


if __name__ == "__main__":
    unittest.main(verbosity=2)
