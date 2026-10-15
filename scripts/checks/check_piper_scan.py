#!/usr/bin/env python3
"""Headless checks for PiPER real scanning. Never opens real CAN or camera devices."""
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from core.reconstruction.piper_scan import (  # noqa: E402
    CameraFrame, CameraModel, CharucoBoard, JointHistory, PassivePiper, PiperFK,
    ScanSession, check_board_depth, depth_meters, rigid, solve_hand_eye, transform, validate_holdout,
)
from core.reconstruction.piper_camera import CalibrationSimSource, DemoSource  # noqa: E402
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
                def unsynchronized(kind):
                    frame, q, dt = original_sample(kind)
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
                for _ in range(8):
                    ui.capture()
                plan = storage.load_plan(ui.session.directory)
                self.assertEqual(plan.meta["source"], "synthetic_calibration")
                result = reconstruct(storage.load_frames(ui.session.directory), plan,
                                     FusionParams(smooth_iters=0), log=lambda _: None)
                self.assertGreater(len(result.mesh.faces), 100)
                np.testing.assert_allclose(result.mesh.extents[:2], [.08, .08], atol=.008)
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
                for _ in range(8):
                    ui.capture()
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
        from core.reconstruction.piper_camera import LiveSource
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

    def test_fk_measured_values_not_clamped(self):
        candidates = [Path("/workspace/agx_arm_ws/install/agx_arm_description/share/agx_arm_description/agx_arm_urdf/piper/urdf/piper_description.urdf"),
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
