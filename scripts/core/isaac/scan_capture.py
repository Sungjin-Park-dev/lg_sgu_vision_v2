#!/usr/bin/env python3
"""스캔 계획의 관절 자세로 로봇을 옮기며 손목 depth 를 찍는다.

옮기는 방법은 둘이다. 계획에 장 사이 실행 궤적(scan_motion → trajectory.csv)이 있으면 **그 궤적을
따라** 움직인다 — 프레임마다 시간에 맞는 관절값을 박는 재생이다(speed 배속). 없으면(또는
motion=False) 자세마다 순간이동한다 — 복원 품질만 볼 때는 궤적이 필요 없다. 어느 쪽이든 장 자세에
닿으면 관절 상태와 PD 목표를 같은 값으로 박고(``set_start_pose`` 와 같은 방식), 새 depth 가 나오면
depth·내부 파라미터·카메라 pose 를 같은 프로세스에서 읽는다. 그래서 이미지와 pose 의 시점이
어긋날 일이 없다.

촬영은 프레임 단위 상태기계 ``CaptureSession`` 이다. 렌더 한 번마다 ``step()`` 을 한 번 부른다 —
아래 독립 실행은 ``sim.step(render=True)`` → ``session.step()`` 루프를 돌고, isaac_pipeline 은
``pump()`` 에서 매 프레임 ``step()`` 을 부른다(그 앱에는 async 가 없고, 버튼 콜백 안에서
``sim.step`` 을 부르면 UI 가 멈춘다). numpy 만 쓴다 — 앱의 번들 Python 이 scipy 를 피하기 때문이다.

스캔 카메라는 ``camera_optical_frame`` 아래 별도 prim 이다(검사 카메라는 50mm 시야라
스캔에 못 쓴다). 장착 위치·내부 파라미터는 계획의 camera(= sensor.ScanCameraSpec).

출력: data/{object}/scan/{name}/frames/frame_###.npz — 키 규약은 reconstruction/storage.py
(world_T_cam 은 Isaac 이 렌더한 실제 자세).

Usage:
    uv run scripts/core/isaac/scan_capture.py --object cylinder_sample
    uv run scripts/core/isaac/scan_capture.py --object cylinder_sample --gui --limit 5
    uv run scripts/core/isaac/scan_capture.py --object cylinder_sample --teleport
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from common import config as _config  # noqa: E402
from common import scene_config as _scene_config  # noqa: E402
from common.math_utils import quaternion_to_rotation_matrix  # noqa: E402
from core.isaac import scene as urscene  # noqa: E402
from core.reconstruction import storage  # noqa: E402
from core.reconstruction.sensor import ScanCameraSpec  # noqa: E402

JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
SCAN_CAMERA_NAME = "ScanCamera"
# USD 카메라의 aperture 는 아무 값이어도 된다 — focal 과의 비만 내부 파라미터를 정한다.
H_APERTURE_MM = 20.955
# OpenCV 광학(+Z 전방, +Y 아래) → USD 카메라(−Z 전방, +Y 위): X 축 둘레 180°.
CV_T_USD = np.diag([1.0, -1.0, -1.0, 1.0])


def _find_optical_frame(stage, root: str = urscene.STAGE_PATH) -> str:
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if urscene._is_under_root(p, root) and prim.GetName() == urscene.CAMERA_OPTICAL_FRAME_NAME:
            return p
    raise RuntimeError(f"{urscene.CAMERA_OPTICAL_FRAME_NAME} prim not found under {root}")


def add_scan_camera(spec: ScanCameraSpec) -> str:
    """camera_optical_frame 아래에 스캔 카메라 prim 을 만든다. prim 경로를 돌려준다."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    path = f"{_find_optical_frame(stage)}/{SCAN_CAMERA_NAME}"
    cam = UsdGeom.Camera.Define(stage, path)
    local = spec.tool_T_cam_matrix @ CV_T_USD          # 열벡터 규약
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d(*local.T.ravel().tolist()))   # USD 는 행벡터 규약
    # fx_px = W · focal / h_aperture  →  focal = h_aperture · fx / W. 정사각 픽셀이라 v_ap 는 비례.
    cam.GetHorizontalApertureAttr().Set(H_APERTURE_MM)
    cam.GetVerticalApertureAttr().Set(H_APERTURE_MM * spec.height / spec.width)
    cam.GetFocalLengthAttr().Set(H_APERTURE_MM * spec.fx / spec.width)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 5.0))
    return path


# "새 자세의 화면인가" 판정 — 픽셀이 이만큼 이상 바뀐 비율이 이보다 커야 새 화면으로 본다.
# 비트 단위로 "다르다" 만 보면 안 된다: 순간이동 직후 렌더러가 **직전 자세 화면을 미세하게 흔들린
# 채로**(물리 정지 판정 전의 µm 움직임) 다시 내보낼 때가 있고, 그 화면이 "다르다" 를 통과한 뒤 두 번
# 같게 나와 확정됐다 — cylinder_sample 에서 25번 자세 화면이 26번 pose 로 저장돼(평균 차 0.00mm)
# 물체 옆에 덩어리가 복원됐다. 연속한 스캔 자세는 방향이 크게 달라 진짜 새 화면이면 대부분이 바뀐다.
VIEW_CHANGE_TOL_M = 0.005
VIEW_CHANGE_MIN_FRAC = 0.01


def _view_changed(a: np.ndarray, b: np.ndarray) -> bool:
    """두 depth 가 의미 있게 다른 화면인가 — 유효/무효가 바뀌었거나 5mm 넘게 달라진 픽셀이 1% 넘나."""
    fa, fb = np.isfinite(a), np.isfinite(b)
    both = fa & fb
    changed = (fa != fb)
    changed[both] |= np.abs(a[both] - b[both]) > VIEW_CHANGE_TOL_M
    return float(changed.mean()) > VIEW_CHANGE_MIN_FRAC


def _pose_matrix(pos, quat_wxyz) -> np.ndarray:
    """Isaac 의 (pos, quat wxyz) → 4×4. 쿼터니언이 float32 로 오므로 float64 로 정규화부터 한다."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = quaternion_to_rotation_matrix(q / np.linalg.norm(q))
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


class CaptureSession:
    """프레임 단위 촬영. ``start()`` → 렌더마다 ``step()`` → ``close()``.

    새 자세의 depth 판정: 렌더는 비동기라 annotator 가 몇 프레임 전 화면을 준다. 몇 프레임을
    넘겨야 하는지는 고정돼 있지 않다(3장 시험에선 맞았던 고정 4프레임이 42장에선 한 자세씩
    밀렸다). 그래서 **직전 자세의 depth 와 의미 있게 다르고(_view_changed), 연속 두 번 같은** 값이
    나올 때까지 기다린다 —
    로봇은 순간이동 뒤 멈춰 있고 depth 렌더는 결정적이라, 같은 값이 두 번 나오면 그게 지금 자세의
    화면이다. ``steps`` 는 순간이동 뒤 렌더 횟수다(물리 엔진이 정지로 판정할 때까지 대부분 ~29 —
    headless 42장 20초).

    첫 자세도 같은 판정을 받도록, 순간이동 **전에** 지금(시작) 자세의 화면이 안정될 때까지 기다려
    그것을 "직전 자세 depth" 로 둔다(warm-up). 이게 없으면 카메라를 만들자마자 나오는 시작 자세
    화면이 비교 대상 없이 0번으로 저장되고, 그 여파로 1번도 4프레임 만에 엉뚱한 화면으로
    확정됐다(카메라를 미리 만들어 두던 옛 독립 실행은 타이밍상 우연히 피했다).
    """

    def __init__(self, out_dir: Path, plan: storage.ScanPlan, articulation_root: str, *,
                 settle: int = 2, max_steps: int = 60, limit: int | None = None,
                 motion: bool = True, speed: float = 1.0, log=print):
        self.out_dir = Path(out_dir)
        self.plan = plan
        self.articulation_root = articulation_root
        self.settle, self.max_steps = int(settle), int(max_steps)
        self.q_plan = plan.q[:limit] if limit else plan.q
        self.log = log
        self.index = 0
        self.saved = 0
        self.skipped: list[int] = []
        self.done = False
        self.summary: dict = {}
        self._cancelled = False
        self._closed = False
        self._warming = False
        self._steps = 0
        self._last = None          # 이 자세에서 직전 step 의 depth
        self._prev = None          # 마지막으로 저장한 자세의 depth
        self._camera = None
        self._cam_path = None
        self._art = self._ctrl = self._idx = None
        self._q0 = None
        self._K = None
        self._pose_err_mm: list[float] = []
        self._rot_err_deg: list[float] = []
        self._t0 = None
        # 궤적 재생 — (joints (M,6), times (M,), shot_rows (N,)) 또는 None(순간이동)
        self.motion = storage.load_motion(self.out_dir) if motion else None
        if self.motion is not None and not self._motion_matches():
            self.log("[capture] ⚠️  trajectory.csv does not match the plan's poses — teleporting instead")
            self.motion = None
        self.speed = max(float(speed), 1e-3)
        self._drive_to = None      # 재생 중이면 목표 행, 아니면 None
        self._t_play = 0.0         # 궤적 시간(s)
        self._t_wall = None

    def _motion_matches(self) -> bool:
        joints, times, rows = self.motion
        return (len(rows) == len(self.plan.q) and len(times) == len(joints)
                and bool(np.allclose(joints[rows], self.plan.q, atol=1e-4)))

    @property
    def progress(self) -> tuple[int, int]:
        return self.index, len(self.q_plan)

    def start(self) -> None:
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.sensors.camera import Camera

        spec = self.plan.camera
        self._cam_path = add_scan_camera(spec)
        self._camera = Camera(prim_path=self._cam_path, resolution=(spec.width, spec.height))
        self._camera.initialize(attach_rgb_annotator=False)
        self._camera.add_distance_to_image_plane_to_frame()

        # 매번 새로 bind 한다 — Stop/Play 가 physics view 를 다시 만들 수 있다(IsaacArticulationExecutor 와 같은 이유).
        self._art = SingleArticulation(prim_path=self.articulation_root)
        self._art.initialize()
        self._idx = np.array([self._art.get_dof_index(n) for n in JOINT_NAMES], dtype=np.int32)
        self._ctrl = self._art.get_articulation_controller()
        self._q0 = np.asarray(self._art.get_joint_positions(joint_indices=self._idx), dtype=np.float64)

        storage.clear_frames(self.out_dir)
        self._t0 = time.time()
        if self.motion is not None:
            q_start = self.motion[0][0]
            if np.max(np.abs(q_start - self._q0)) > 1e-3:
                # 궤적은 계획 당시 자세에서 시작한다. 그 사이 로봇이 움직였으면 거기로 먼저 옮긴다.
                self.log(f"[capture] robot is {np.degrees(np.max(np.abs(q_start - self._q0))):.1f}° "
                         f"away from the trajectory start — teleporting there first")
                self._teleport(q_start)
            self.log(f"[capture] following trajectory.csv ({self.motion[1][-1]:.1f} s at ×{self.speed:g})")
        if len(self.q_plan):
            self._warming = True               # 시작 자세 화면부터 잡는다 — step() 참고
            self._steps, self._last = 0, None
        else:
            self.done = True

    def _teleport(self, q) -> None:
        from isaacsim.core.utils.types import ArticulationAction

        self._art.set_joint_positions(q, joint_indices=self._idx)
        self._ctrl.apply_action(ArticulationAction(joint_positions=q, joint_indices=self._idx))
        self._steps = 0
        self._last = None

    def _set_q(self, q) -> None:
        from isaacsim.core.utils.types import ArticulationAction

        self._art.set_joint_positions(q, joint_indices=self._idx)
        self._ctrl.apply_action(ArticulationAction(joint_positions=q, joint_indices=self._idx))

    def _go_to_shot(self, i: int) -> None:
        """장 i 로 — 궤적이 있으면 재생을 시작하고, 없으면 순간이동."""
        if self.motion is None:
            self._teleport(self.q_plan[i])
            return
        self._drive(int(self.motion[2][i]))

    def _drive(self, row: int) -> None:
        self._drive_to = row
        self._t_wall = None

    def _drive_step(self) -> bool:
        """재생 한 프레임. True = 목표 행에 닿았다(그 자세로 박고 촬영 대기로 넘어간다)."""
        joints, times, _ = self.motion
        now = time.time()
        # 실제 경과 시간 × 배속. 한 프레임이 오래 걸려도(셰이더 컴파일 등) 0.1s 이상은 건너뛰지 않는다.
        dt = 0.0 if self._t_wall is None else min(now - self._t_wall, 0.1)
        self._t_wall = now
        t_goal = float(times[self._drive_to])
        self._t_play = min(self._t_play + dt * self.speed, t_goal)
        if self._t_play >= t_goal:
            self._teleport(joints[self._drive_to])
            self._drive_to = None
            return True
        self._set_q(np.array([np.interp(self._t_play, times, joints[:, j]) for j in range(joints.shape[1])]))
        return False

    def cancel(self) -> None:
        self._cancelled = True

    def step(self) -> bool:
        """렌더 한 번마다 부른다. True = 계속, False = 끝(또는 취소)."""
        if self.done or self._cancelled:
            return False
        if self._drive_to is not None:
            if not self._drive_step():
                return True
            if self.index >= len(self.q_plan):     # 마지막 장 뒤 시작 자세로 돌아왔다
                self.done = True
                return False
            return True                            # 장 자세에 닿았다 — 다음 프레임부터 촬영 대기
        self._steps += 1
        if self._steps <= self.settle:
            return True
        d = self._camera.get_depth()
        if self._warming:
            stable = False
            if d is not None:
                d = np.asarray(d, dtype=np.float32)
                stable = self._last is not None and np.array_equal(d, self._last)
                self._last = d
            if not stable and self._steps < self.max_steps:
                return True                    # 카메라가 아직 depth 를 안 주거나(None) 화면이 흔들린다
            # 이미 첫 자세에 서 있으면 화면이 안 바뀌어 "직전과 다름" 이 영영 안 걸린다 — 그땐 비교를 뺀다.
            at_first = np.allclose(self._q0, self.q_plan[0], atol=1e-6)
            self._prev = None if at_first else self._last
            self._warming = False
            self._go_to_shot(0)
            return True
        if d is not None:
            d = np.asarray(d, dtype=np.float32)
            fresh = self._prev is None or _view_changed(d, self._prev)
            if fresh and self._last is not None and np.array_equal(d, self._last):
                self._save(d)
                return self._advance()
            self._last = d
        if self._steps >= self.max_steps:
            self.log(f"[capture] {self.index:03d}: depth did not settle in {self.max_steps} steps — skipped")
            self.skipped.append(self.index)
            return self._advance()
        return True

    def _save(self, depth) -> None:
        i, q = self.index, self.q_plan[self.index]
        self._prev = depth
        self._K = np.asarray(self._camera.get_intrinsics_matrix(), dtype=np.float64)
        pos, quat = self._camera.get_world_pose(camera_axes="ros")
        T = _pose_matrix(pos, quat)
        q_meas = np.asarray(self._art.get_joint_positions(joint_indices=self._idx), dtype=np.float64)
        storage.save_frame(self.out_dir, i, depth=depth, K=self._K, world_T_cam=T, q_cmd=q, q_meas=q_meas)
        T_plan = self.plan.world_T_cam[i]
        dp = 1000.0 * float(np.linalg.norm(T[:3, 3] - T_plan[:3, 3]))
        cos = (np.trace(T_plan[:3, :3].T @ T[:3, :3]) - 1.0) / 2.0
        dr = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        self._pose_err_mm.append(dp)
        self._rot_err_deg.append(dr)
        self.saved += 1
        valid = int(np.isfinite(depth).sum())
        self.log(f"[capture] {i:03d}: {self._steps:2d} steps  valid {valid / depth.size * 100:5.1f}%  "
                 f"pose vs plan {dp:6.2f} mm {dr:5.2f}°  "
                 f"|q err| {np.degrees(np.max(np.abs(q_meas - q))):.3f}°")

    def _advance(self) -> bool:
        self.index += 1
        if self.index >= len(self.q_plan):
            if self.motion is not None and len(self.q_plan) == len(self.plan.q):
                self._drive(len(self.motion[0]) - 1)    # 궤적 끝(시작 자세)까지 마저 간다
                return True
            self.done = True
            return False
        self._go_to_shot(self.index)
        return True

    def close(self, restore: bool = True) -> dict:
        """마무리 — 여러 번 불러도 안전하다. restore: 촬영 전 관절값으로 순간이동해 되돌린다."""
        if self._closed:
            return self.summary
        self._closed = True
        if restore and self._q0 is not None and self._art is not None:
            try:
                self._teleport(self._q0)
            except Exception as exc:  # noqa: BLE001 — 정리는 끝까지 한다
                self.log(f"[capture] restore failed: {exc}")
        spec = self.plan.camera
        self.summary = {
            "complete": bool(self.done and not self._cancelled and not self.skipped),
            "n_planned": int(len(self.q_plan)), "n_frames": self.saved, "skipped": self.skipped,
            "seconds": round(time.time() - self._t0, 1) if self._t0 else 0.0,
            "moved_along_trajectory": self.motion is not None,
            "K_isaac": None if self._K is None else self._K.tolist(), "K_spec": spec.K.tolist(),
            "pose_vs_plan_mm": {"max": max(self._pose_err_mm, default=0.0),
                                "median": float(np.median(self._pose_err_mm)) if self._pose_err_mm else 0.0},
            "rot_vs_plan_deg": {"max": max(self._rot_err_deg, default=0.0)},
        }
        try:
            (self.out_dir / "capture.json").write_text(json.dumps(self.summary, indent=2) + "\n")
        except OSError as exc:
            self.log(f"[capture] capture.json 을 못 썼다: {exc}")
        if self._camera is not None:
            self._camera.destroy()                 # annotator·render product·구독을 모두 푼다
            self._camera = None
        if self._cam_path:
            from isaacsim.core.utils import prims

            if prims.is_prim_path_valid(self._cam_path):
                prims.delete_prim(self._cam_path)
        self.log(f"[capture] {self.saved}/{len(self.q_plan)} frames in {self.summary['seconds']}s"
                 + (" (cancelled)" if self._cancelled else "")
                 + (f", skipped {self.skipped}" if self.skipped else ""))
        self.log(f"[capture] pose vs plan: max {self.summary['pose_vs_plan_mm']['max']:.2f} mm, "
                 f"{self.summary['rot_vs_plan_deg']['max']:.2f}°")
        if self._K is not None and not np.allclose(self._K, spec.K, rtol=1e-3):
            self.log(f"[capture] ⚠️  K mismatch: isaac {self._K.round(1).tolist()} vs spec {spec.K.round(1).tolist()}")
        return self.summary


def place_object(pose) -> None:
    """물체를 계획에 남긴 pose (7,) 에 놓고 지지대·장애물 시각물을 거기 맞춘다(독립 실행용).

    load_target_object 는 씬 배치에 놓는다. 계획이 다른 자리에서 세워졌으면(Isaac 기즈모, 또는
    --object-position 으로 옮긴 실험) 찍는 장면이 계획과 어긋나므로, 계획의 pose 로 옮긴다.
    """
    import omni.usd
    from pxr import Gf, UsdGeom

    pose = np.asarray(pose, dtype=np.float64)
    _config.TARGET_OBJECT["position"] = pose[:3].copy()
    _config.TARGET_OBJECT["rotation"] = pose[3:7].copy()
    _config.sync_support_to_target()
    urscene.spawn_scene_obstacles(_config)
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(f"/World/{_config.TARGET_OBJECT['name']}")
    if not prim or not prim.IsValid():
        return
    M = Gf.Matrix4d()
    M.SetTransform(Gf.Rotation(Gf.Quatd(*(float(v) for v in pose[3:7]))),
                   Gf.Vec3d(*(float(v) for v in pose[:3])))
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(M)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--object", required=True)
    _scene_config.add_cli_argument(p)
    p.add_argument("--name", default="default", help="data/{object}/scan/{name}")
    p.add_argument("--gui", action="store_true", help="창을 띄운다 (기본 headless)")
    p.add_argument("--limit", type=int, default=None, help="앞에서 몇 장만 찍는다")
    p.add_argument("--settle", type=int, default=2,
                   help="자세를 바꾼 뒤 depth 를 보기 전에 무조건 넘기는 렌더 프레임 수")
    p.add_argument("--max-steps", type=int, default=60,
                   help="depth 가 새 자세로 바뀌어 멈출 때까지 기다리는 최대 렌더 프레임 수")
    p.add_argument("--teleport", action="store_true",
                   help="trajectory.csv 가 있어도 따라가지 않고 자세마다 순간이동한다")
    p.add_argument("--speed", type=float, default=1.0, help="궤적 재생 배속 (실제 시간 기준)")
    return p.parse_args()


def main():
    # app.close() 가 프로세스를 바로 끝내서, 파일로 리다이렉트된 stdout 의 블록 버퍼가 사라진다.
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    out = storage.scan_dir(args.object, args.name)
    plan = storage.load_plan(out)

    _scene_config.apply_cli(args, _config)
    app = urscene.start_sim(headless=not args.gui, enable_ros_bridge=False)

    from isaacsim.core.api import SimulationContext

    sim = SimulationContext(stage_units_in_meters=1.0)
    urscene.load_workcell(urscene.DEFAULT_USD)
    app.update()
    urscene.load_target_object(args.object)
    if plan.object_pose is not None:
        place_object(plan.object_pose)
    app.update()
    art_root = urscene.find_articulation_root()
    app.update()

    sim.initialize_physics()
    sim.play()
    sim.step(render=True)
    session = CaptureSession(out, plan, art_root, settle=args.settle, max_steps=args.max_steps,
                             limit=args.limit, motion=not args.teleport, speed=args.speed)
    session.start()
    while True:
        sim.step(render=True)
        if not session.step():
            break
    session.close(restore=False)
    print(f"[capture] -> {storage.frames_dir(out).relative_to(PROJECT_ROOT)}")

    sim.stop()
    app.close()


if __name__ == "__main__":
    main()
