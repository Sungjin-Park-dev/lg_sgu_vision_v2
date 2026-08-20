#!/usr/bin/env python3
"""
Omni UI panel for the trajectory pipeline inside Isaac Sim.

Boots Isaac Sim through the shared ``core.isaac.scene`` runtime, then opens
an Omni UI window with four panels:

    A) Load object (dropdown + native viewport gizmo move)
    B) GLNS trajectory parameters + [Generate Trajectory]  (subprocess)
    C) Ghost preview with Play/Pause/Stop/Slider            (in-process; sim, ROS-free)
    D) Execute trajectory on Isaac UR20 or real robot       (subprocess)

The pipeline scripts run as `uv run` subprocesses to keep Isaac Sim's bundled
Python isolated from cuRobo / rclpy. Stdout streams into a scrolling log.

Preview overlays a pre-built physics-free ghost UR20 with the camera attached
(built via scripts/setup/build_ghost_usd.py) at /World/UR20_preview and
poses each link by writing one xformOp per frame via FK. The real /World/UR20
articulation is never touched by preview.

Usage:
    uv run scripts/apps/isaac_pipeline.py --object sample
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Callable, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse the core Isaac scene loaders — same workcell, robot, camera.
from core.isaac import scene as urctl  # noqa: E402
from common import config as _cfg_module  # noqa: E402
from common import scene_config  # noqa: E402

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Must match scripts/core/trajectory/settings.py::IK_RANDOM_SEED. Kept local because
# this module is imported by Isaac Sim's bundled Python before the uv subprocess.
IK_RANDOM_SEED = 123
# Must match scripts/core/trajectory/settings.py::CANDIDATE_DEDUP_RAD (default dedup
# threshold for the Check-and-Save-IK / GLNS IK candidate stage).
CANDIDATE_DEDUP_RAD = 0.08

CSV_PATH_RE = re.compile(r"CSV saved to (\S+)")

GHOST_ROOT_PATH = "/World/UR20_preview"
# prim 이름 → 씬 YAML 의 장애물 이름(USD prim 규칙: 영문/숫자/_).
_SCENE_PRIM_NAME_RE = re.compile(r"[^A-Za-z0-9_]")
GHOST_USD_NAME = "ur20_with_camera_ghost.usd"

# Trajectory controllers, gated by pipeline mode (only one active at a time):
#   MoveIt (RViz move_group) → MOVEIT_CONTROLLER
#   Inspection (publish_trajectory) → INSPECTION_CONTROLLER
MOVEIT_CONTROLLER = "scaled_joint_trajectory_controller"
INSPECTION_CONTROLLER = "joint_trajectory_controller"

# Matches scene.load_target_object (/World/{config.TARGET_OBJECT['name']}).
TARGET_OBJECT_PRIM = "/World/target_object"
VIEWPOINTS_ROOT_PRIM = f"{TARGET_OBJECT_PRIM}/Viewpoints"
VIEWPOINTS_POINTS_PRIM = f"{VIEWPOINTS_ROOT_PRIM}/CameraPoints"
VIEWPOINT_POINT_WIDTH_M = 0.008
# Tilt 중심으로 고른 viewpoint 하나를 눈에 띄게 (크게 + 노랗게) 그리는 배율.
VIEWPOINT_HIGHLIGHT_SCALE = 3.0
COLLISION_SPHERES_SCOPE_NAME = "CuRoboCollisionSpheres"
FOV_PLANE_SCOPE_NAME = "CameraFovPlane"
FOV_PLANE_OUTLINE_WIDTH_M = 0.003
FOV_PLANE_CENTERLINE_WIDTH_M = 0.002
# Camera range: a grid of rays cast across the FOV from the optical origin,
# drawn to where they actually hit the target object (ray-mesh intersection).
CAMERA_RANGE_SCOPE_NAME = "CameraRangeRays"
CAMERA_RANGE_GRID = 7  # N×N rays across the FOV
CAMERA_RANGE_RAY_WIDTH_M = 0.0012
CAMERA_RANGE_HIT_WIDTH_M = 0.003
CAMERA_RANGE_UPDATE_DT = 0.05  # s between live re-casts (rays follow the camera)
# Tilt 부채꼴 시각화. 물체의 로컬 프레임에 그린다(viewpoint 점들과 같은 자리) — 그러면
# 기즈모로 물체를 옮겨도 USD 가 알아서 따라오고, robot/world 프레임 변환이 아예 필요 없다.
TILT_FAN_SCOPE_NAME = "TiltFan"
TILT_FAN_ARC_WIDTH_M = 0.0025
TILT_FAN_WAYPOINT_WIDTH_M = 0.005
TILT_FAN_RAY_WIDTH_M = 0.0010
TILT_FAN_CENTER_WIDTH_M = 0.014
# 시선을 waypoint 마다 그리면 부채꼴이 뭉개진다 — leg 당 이만큼만, 최대각은 반드시 포함.
TILT_FAN_RAYS_PER_LEG = 7
# leg 4개를 색으로 구분한다. 기존 씬 색과 뜻이 겹치지 않게 골랐다(viewpoint=cyan,
# FOV=주황/노랑, range=초록, hit=빨강).
TILT_LEG_COLORS = {
    "up": (1.00, 0.85, 0.10),
    "down": (0.60, 0.40, 1.00),
    "left": (0.20, 0.70, 1.00),
    "right": (1.00, 0.35, 0.60),
}
TILT_CENTER_COLOR = (1.00, 1.00, 1.00)

CAMERA_COLLISION_LINKS = {
    "tool0",
    "camera_cable_frame",
    "camera_frame_1",
    "camera_frame_2",
    "camera_link",
}

# Execute 패널의 HOME 이동. 각 leg 는 현재 자세 → 목표를 plan_move.py 로 계획해 실행한다.
HOME_TRANSITIONS = {
    "approach": "move to start",
    "return": "return to HOME",
}


def discover_objects() -> list[str]:
    """Object names that have data/{object}/mesh/source.obj — Object dropdown candidates.

    Mirrors viewpoint_studio.discover_objects. Listing by source.obj (not
    source.usd) shows every object; load_target_object reports which ones still
    need `build_object_usd.py` to produce a source.usd.
    """
    data_root = PROJECT_ROOT / "data"
    return [p.parent.parent.name for p in sorted(data_root.glob("*/mesh/source.obj"))]


# =============================================================================
# Preview ghost — references a pre-built physics-free ghost USD.
# PreviewPlayer poses link xforms via FK; we never touch PhysX at runtime.
# =============================================================================

@dataclass
class GhostJoint:
    """One revolute joint in the ghost's kinematic chain (parent → child)."""
    name: str
    parent_link_path: str
    child_link_path: str
    axis: np.ndarray              # 3-vec, unit
    T_joint_in_parent: np.ndarray # 4x4, joint origin expressed in parent link frame
    T_joint_in_child: np.ndarray  # 4x4, joint origin expressed in child  link frame


def _np_from_pos_quat(pos, quat_wxyz) -> np.ndarray:
    """4x4 numpy transform (column-vector convention) from (x,y,z) + (w,x,y,z)."""
    T = np.eye(4)
    T[:3, 3] = [pos[0], pos[1], pos[2]]
    w, x, y, z = quat_wxyz
    T[:3, :3] = np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x*x + y*y)],
    ])
    return T


def _axis_angle_4x4(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues 4x4 rotation about `axis` (unit) by `angle` (rad)."""
    c, s = float(np.cos(angle)), float(np.sin(angle))
    x, y, z = float(axis[0]), float(axis[1]), float(axis[2])
    R = np.array([
        [c + x*x*(1-c),     x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s,   c + y*y*(1-c),   y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s,   z*y*(1-c) + x*s, c + z*z*(1-c)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    return T


def _gf_to_np(gf_mat) -> np.ndarray:
    """Gf.Matrix4d (row-vector, translation in last row) → numpy 4x4 (column-vec)."""
    arr = np.array([[gf_mat[r][c] for c in range(4)] for r in range(4)],
                   dtype=np.float64)
    return arr.T


def _np_to_gf(np_mat: np.ndarray):
    """numpy 4x4 (column-vec) → Gf.Matrix4d (row-vector)."""
    from pxr import Gf
    M = np_mat.T
    return Gf.Matrix4d(
        float(M[0,0]), float(M[0,1]), float(M[0,2]), float(M[0,3]),
        float(M[1,0]), float(M[1,1]), float(M[1,2]), float(M[1,3]),
        float(M[2,0]), float(M[2,1]), float(M[2,2]), float(M[2,3]),
        float(M[3,0]), float(M[3,1]), float(M[3,2]), float(M[3,3]),
    )


def spawn_preview_ghost(usd_path: Path, ghost_root: str, position,
                        joint_order: list,
                        log: Callable[[str], None]):
    """Reference the pre-built physics-free ghost USD and extract its FK chain.

    The USD is already stripped (no rigid bodies, no articulation, no
    collisions — see build_ghost_usd.py), so this function only does
    USD-level work: reference, walk joint prims for chain info, hide.
    Returns (base_link_path, chain).
    """
    from isaacsim.core.utils import prims
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    prims.create_prim(
        ghost_root, "Xform",
        position=position,
        usd_path=str(usd_path),
    )

    stage = omni.usd.get_context().get_stage()

    # Joint prims are kept in the ghost USD (with jointEnabled=False) precisely
    # so we can read body0/body1/axis/localPose to build the FK chain.
    found: "dict[str, GhostJoint]" = {}
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if not p.startswith(ghost_root):
            continue
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        rj = UsdPhysics.RevoluteJoint(prim)
        pn = prim.GetName()
        match = next((n for n in joint_order if pn == n or pn.endswith(n)), None)
        if match is None:
            continue
        b0 = rj.GetBody0Rel().GetTargets()
        b1 = rj.GetBody1Rel().GetTargets()
        if not b0 or not b1:
            log(f"[ghost] joint {pn} missing body0/body1, skipping")
            continue
        axis_tok = rj.GetAxisAttr().Get() or "Z"
        axis = np.array({"X": [1., 0., 0.], "Y": [0., 1., 0.], "Z": [0., 0., 1.]}[axis_tok])
        p0 = rj.GetLocalPos0Attr().Get()
        r0 = rj.GetLocalRot0Attr().Get()
        p1 = rj.GetLocalPos1Attr().Get()
        r1 = rj.GetLocalRot1Attr().Get()
        pos0 = (float(p0[0]), float(p0[1]), float(p0[2])) if p0 else (0., 0., 0.)
        pos1 = (float(p1[0]), float(p1[1]), float(p1[2])) if p1 else (0., 0., 0.)
        rot0 = (float(r0.GetReal()), *(float(v) for v in r0.GetImaginary())) if r0 \
               else (1., 0., 0., 0.)
        rot1 = (float(r1.GetReal()), *(float(v) for v in r1.GetImaginary())) if r1 \
               else (1., 0., 0., 0.)
        found[match] = GhostJoint(
            name=match,
            parent_link_path=str(b0[0]),
            child_link_path=str(b1[0]),
            axis=axis,
            T_joint_in_parent=_np_from_pos_quat(pos0, rot0),
            T_joint_in_child=_np_from_pos_quat(pos1, rot1),
        )

    chain = [found[n] for n in joint_order if n in found]
    if len(chain) != len(joint_order):
        missing = [n for n in joint_order if n not in found]
        raise RuntimeError(f"[ghost] missing joints under {ghost_root}: {missing}")
    base_link_path = chain[0].parent_link_path

    UsdGeom.Imageable(stage.GetPrimAtPath(ghost_root)).MakeInvisible()
    log(f"[ghost] spawned at {ghost_root}: chain={len(chain)} joints rooted at "
        f"{base_link_path}, starting hidden")
    return base_link_path, chain


def set_ghost_visible(ghost_root: str, visible: bool) -> None:
    import omni.usd
    from pxr import UsdGeom
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(ghost_root)
    if not prim or not prim.IsValid():
        return
    img = UsdGeom.Imageable(prim)
    img.MakeVisible() if visible else img.MakeInvisible()


# =============================================================================
# Subprocess runner — threaded reader pushes lines to a Queue.
# =============================================================================

class SubprocessRunner:
    """Run a subprocess in the background and forward stdout lines to a queue.

    The Kit UI loop polls `pump()` each frame to drain the queue and call the
    line callback. Stderr is merged into stdout to preserve ordering.
    """

    # Cancel 은 프로세스 그룹에 SIGTERM 을 보낸다. 이만큼 지나도 안 죽으면 SIGKILL.
    KILL_GRACE_S = 3.0

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._queue: Queue = Queue()
        self._reader: Optional[threading.Thread] = None
        self._on_line: Optional[Callable[[str], None]] = None
        self._on_exit: Optional[Callable[[int], None]] = None
        self._done = True
        # Bumped on every start(). Queue items carry the generation that produced
        # them; pump() drops items from a superseded process so a stale __exit__
        # can't flip _done for a newly started one.
        self._gen = 0
        self._term_at: Optional[float] = None
        self._cancelled = False

    @property
    def running(self) -> bool:
        return not self._done

    @property
    def cancelled(self) -> bool:
        """마지막 실행이 사용자 Cancel 로 끝났는가 — 실패와 구분해 로그를 정직하게 쓰려고."""
        return self._cancelled

    def start(self, cmd, cwd, on_line, on_exit):
        # Supersede any in-flight process instead of raising. A prior
        # fire-and-forget run (e.g. a quick `ros2 param set`) may not have been
        # drained by pump() yet — this happens at startup where apply_mode and
        # apply_pipeline_mode both set the relay before the UI loop pumps.
        if self.running:
            self.terminate()

        env = os.environ.copy()
        # Kit's PYTHONHOME/PATH leaks into children and confuses `uv run`.
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)
        env["PYTHONUNBUFFERED"] = "1"

        self._gen += 1
        gen = self._gen
        self._on_line = on_line
        self._on_exit = on_exit
        self._done = False
        self._term_at = None
        self._cancelled = False
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, universal_newlines=True,
            # 자기 프로세스 그룹으로 띄운다 — Cancel 이 bash 뿐 아니라 그 자손(uv → python)
            # 까지 한 번에 잡아야 한다. `A && B` 셸은 bash 가 exec 하지 않으므로 bash 만
            # 죽이면 A 가 GPU 를 문 채 살아남고, stdout 파이프도 안 닫혀 reader 스레드가
            # 끝나지 않는다 → __exit__ 이 오지 않아 UI 가 영원히 잠긴다(실측 확인).
            start_new_session=True,
        )
        self._proc = proc
        self._reader = threading.Thread(
            target=self._read_loop, args=(gen, proc), daemon=True)
        self._reader.start()

    def _read_loop(self, gen, proc):
        try:
            for line in iter(proc.stdout.readline, ""):
                self._queue.put((gen, line.rstrip("\n")))
        finally:
            proc.stdout.close()
            rc = proc.wait()
            self._queue.put((gen, "__exit__", rc))

    def _signal_group(self, sig):
        """프로세스 그룹 전체에 시그널. 그룹이 없으면(경합) 프로세스 하나라도 잡는다."""
        if self._proc is None:
            return
        try:
            os.killpg(os.getpgid(self._proc.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                self._proc.send_signal(sig)
            except Exception:  # noqa: BLE001 — 이미 죽었으면 할 일 없음
                pass

    def terminate(self):
        """Cancel — 첫 호출은 그룹에 SIGTERM(정리 기회), 두 번째 호출은 즉시 SIGKILL."""
        if not self.running or self._proc is None:
            return
        self._cancelled = True
        if self._term_at is None:
            self._term_at = time.time()
            self._signal_group(signal.SIGTERM)
        else:
            self._term_at = None
            self._signal_group(signal.SIGKILL)

    def pump(self):
        """Drain the queue, call on_line / on_exit on the UI thread."""
        # SIGTERM 을 못 받는 자손(CUDA 커널 중 등)이 있으면 유예 뒤 강제 종료한다. 안 그러면
        # 파이프가 안 닫혀 __exit__ 이 오지 않고 UI 가 잠긴 채 남는다.
        if (self._term_at is not None and not self._done
                and time.time() - self._term_at > self.KILL_GRACE_S):
            self._term_at = None
            self._signal_group(signal.SIGKILL)
        if self._on_line is None:
            return
        try:
            while True:
                item = self._queue.get_nowait()
                gen, payload = item[0], item[1:]
                if gen != self._gen:
                    continue  # output from a superseded process
                if payload and payload[0] == "__exit__":
                    self._done = True
                    if self._on_exit is not None:
                        self._on_exit(int(payload[1]))
                else:
                    self._on_line(str(payload[0]))
        except Empty:
            return


# =============================================================================
# CSV loader (inlined — publish_trajectory.load_trajectory_csv pulls in rclpy
# at module import which is not available in Isaac Sim's bundled Python).
# =============================================================================

def load_trajectory_csv(csv_path: str):
    solutions, times = [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        if "time" not in reader.fieldnames:
            raise ValueError(f"CSV must include a 'time' column: {reader.fieldnames}")
        col_map = {}
        for name in JOINT_NAMES:
            matches = [c for c in reader.fieldnames if c.endswith(name)]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one column ending with '{name}', "
                    f"found {matches} in {reader.fieldnames}"
                )
            col_map[name] = matches[0]
        for row in reader:
            times.append(float(row["time"]))
            solutions.append([float(row[col_map[n]]) for n in JOINT_NAMES])
    solutions = np.array(solutions, dtype=np.float64)
    times = np.array(times, dtype=np.float64)
    if len(solutions) == 0:
        raise ValueError("CSV contains no trajectory rows")
    if len(times) > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("CSV time column must be strictly increasing")
    return solutions, times


# =============================================================================
# Direct Isaac executor — drives the real articulation in-process, without ROS.
# =============================================================================

class IsaacArticulationExecutor:
    """Apply CSV joint targets directly to Isaac's UR20 articulation each frame."""

    APPROACH_MAX_JOINT_VEL_RAD_S = 0.5
    MIN_APPROACH_TIME_S = 0.5

    def __init__(self, articulation_root: str, joint_names: list[str],
                 log: Callable[[str], None]):
        self._articulation_root = articulation_root
        self._joint_names = joint_names
        self._log = log
        self._articulation = None
        self._controller = None
        self._indices = None
        self._positions = np.zeros((0, len(joint_names)), dtype=np.float64)
        self._times = np.zeros(0, dtype=np.float64)
        self._elapsed = 0.0
        self._running = False
        self._on_done: Optional[Callable[[int], None]] = None

    @property
    def running(self) -> bool:
        return self._running

    def _initialize(self):
        from isaacsim.core.prims import SingleArticulation

        # Rebind on every execution because Stop/Play can recreate the physics view.
        art = SingleArticulation(prim_path=self._articulation_root)
        art.initialize()
        indices = np.array(
            [art.get_dof_index(name) for name in self._joint_names], dtype=np.int32,
        )
        if np.any(indices < 0):
            raise RuntimeError(f"UR20 joint lookup failed: {indices.tolist()}")
        self._articulation = art
        self._controller = art.get_articulation_controller()
        self._indices = indices

    def start(self, csv_path: str, on_done: Optional[Callable[[int], None]] = None) -> bool:
        try:
            solutions, csv_times = load_trajectory_csv(csv_path)
            current = self.current_joints()

            relative_times = csv_times - csv_times[0]
            start_diff = float(np.max(np.abs(solutions[0] - current)))
            if start_diff > 1e-4:
                approach_time = max(
                    start_diff / self.APPROACH_MAX_JOINT_VEL_RAD_S,
                    self.MIN_APPROACH_TIME_S,
                )
                self._positions = np.vstack([current, solutions])
                self._times = np.concatenate([[0.0], approach_time + relative_times])
            else:
                self._positions = solutions
                self._times = relative_times

            self._elapsed = 0.0
            self._running = True
            self._on_done = on_done
            self._apply_target(self._positions[0])
            self._log(
                f"[execute] Isaac in-process trajectory: {len(solutions)} CSV waypoints, "
                f"duration={self._times[-1]:.2f}s"
            )
            return True
        except Exception as exc:  # noqa: BLE001 — report runtime/Isaac API failures in UI
            self._log(f"[execute] Isaac articulation start failed: {exc}")
            self._running = False
            self._on_done = None
            return False

    def _apply_target(self, q: np.ndarray):
        from isaacsim.core.utils.types import ArticulationAction

        self._controller.apply_action(ArticulationAction(
            joint_positions=np.asarray(q, dtype=np.float64),
            joint_indices=self._indices,
        ))

    def current_joints(self) -> np.ndarray:
        """UR20 의 현재 관절값 (rad).

        **두 run-mode 모두 여기서 읽는다.** sim 은 Isaac 이 곧 로봇이고, real 은
        /RealRobotGraph 가 실로봇 /joint_states 를 이 articulation 으로 미러링한다
        (apply_mode 참고) — 그래서 별도 ROS 구독 없이 실제 자세를 얻는다.
        """
        self._initialize()
        q = np.asarray(
            self._articulation.get_joint_positions(joint_indices=self._indices),
            dtype=np.float64,
        )
        if q.shape != (len(self._joint_names),) or not np.all(np.isfinite(q)):
            raise RuntimeError(f"invalid current UR20 joint state: {q}")
        return q

    def step(self, dt: float):
        if not self._running:
            return
        self._elapsed = min(self._elapsed + max(float(dt), 0.0), float(self._times[-1]))
        q = np.array([
            np.interp(self._elapsed, self._times, self._positions[:, j])
            for j in range(self._positions.shape[1])
        ], dtype=np.float64)
        self._apply_target(q)
        if self._elapsed >= float(self._times[-1]):
            self._running = False
            callback, self._on_done = self._on_done, None
            self._log("[execute] Isaac UR20 trajectory complete")
            if callback is not None:
                callback(0)

    def cancel(self):
        if not self._running:
            return
        self._running = False
        self._on_done = None
        self._log("[execute] Isaac trajectory cancelled; holding last target")


# =============================================================================
# Preview player — applies CSV waypoints to the ghost via FK.
# =============================================================================

@dataclass
class PreviewState:
    solutions: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    times: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    duration: float = 0.0
    t: float = 0.0
    playing: bool = False
    dof_perm: Optional[list[int]] = None


class PreviewPlayer:
    """Animates the ghost UR20 by computing FK and writing each link's xformOp.

    The ghost USD is physics-free; this class only does USD-level pose
    writes. The real /World/UR20 articulation is never touched, so
    GraphTickSwitch and any external /joint_states publisher keep running
    independently.
    """

    def __init__(self, ghost_root_prim: str, base_link_path: str,
                 chain: "list[GhostJoint]", log: Callable[[str], None]):
        self._ghost_root = ghost_root_prim
        self._base_link_path = base_link_path
        self._chain = chain
        self._log = log
        self._state = PreviewState()

    def load(self, csv_path: str) -> bool:
        try:
            solutions, times = load_trajectory_csv(csv_path)
        except Exception as e:
            self._log(f"[preview] CSV load failed: {e}")
            return False
        # CSV columns are already in JOINT_NAMES order (load_trajectory_csv),
        # and the chain is built in that same order — no permutation needed.
        self._state = PreviewState(
            solutions=solutions,
            times=times,
            duration=float(times[-1] - times[0]),
            t=0.0,
            playing=False,
            dof_perm=None,
        )
        self._log(
            f"[preview] Loaded {len(solutions)} waypoints, "
            f"duration={self._state.duration:.2f}s"
        )
        set_ghost_visible(self._ghost_root, True)
        self._apply()
        return True

    @property
    def loaded(self) -> bool:
        return len(self._state.solutions) > 0

    @property
    def state(self) -> PreviewState:
        return self._state

    def play(self):
        if not self.loaded:
            return
        if self._state.t >= self._state.duration:
            self._state.t = 0.0
        self._state.playing = True

    def pause(self):
        self._state.playing = False

    def stop(self):
        self._state.playing = False
        self._state.t = 0.0
        set_ghost_visible(self._ghost_root, False)

    def seek(self, t: float):
        if not self.loaded:
            return
        self._state.t = float(np.clip(t, 0.0, self._state.duration))
        self._apply()

    def step(self, dt: float):
        if not self.loaded or not self._state.playing:
            return
        self._state.t += dt
        if self._state.t >= self._state.duration:
            self._state.t = self._state.duration
            self._state.playing = False
        self._apply()

    def _apply(self):
        if not self.loaded:
            return
        import omni.usd
        from pxr import UsdGeom

        sol, times = self._state.solutions, self._state.times
        t = self._state.t + times[0]
        q = np.array(
            [np.interp(t, times, sol[:, j]) for j in range(sol.shape[1])],
            dtype=np.float64,
        )

        stage = omni.usd.get_context().get_stage()
        base_prim = stage.GetPrimAtPath(self._base_link_path)
        if not base_prim.IsValid():
            self._log(f"[fk] base link invalid: {self._base_link_path}")
            return

        parent_world = _gf_to_np(
            UsdGeom.Xformable(base_prim).ComputeLocalToWorldTransform(0.0)
        )

        for j_idx, joint in enumerate(self._chain):
            angle = float(q[j_idx])
            # USD physics joint constraint:
            #   body0_world @ T_joint_in_parent @ R(axis, angle)
            #     == body1_world @ T_joint_in_child
            child_world = (
                parent_world
                @ joint.T_joint_in_parent
                @ _axis_angle_4x4(joint.axis, angle)
                @ np.linalg.inv(joint.T_joint_in_child)
            )

            child_prim = stage.GetPrimAtPath(joint.child_link_path)
            if not child_prim.IsValid():
                continue

            usd_parent = child_prim.GetParent()
            if usd_parent and usd_parent.IsValid():
                pw = _gf_to_np(
                    UsdGeom.Xformable(usd_parent).ComputeLocalToWorldTransform(0.0)
                )
                local_T = np.linalg.inv(pw) @ child_world
            else:
                local_T = child_world

            xform = UsdGeom.Xformable(child_prim)
            xform.ClearXformOpOrder()
            op = xform.AddTransformOp(opSuffix="ghostFK")
            op.Set(_np_to_gf(local_T))

            parent_world = child_world


# =============================================================================
# Action Graph enable/disable — probe both API surfaces.
# =============================================================================

class GraphTickSwitch:
    """그래프의 OnPlaybackTick 을 껐다 켜서 그래프가 관절을 쓰는 것을 막는다.

    /RealRobotGraph 와 /SimRobotGraph 에 하나씩 붙는다 — 둘 다 ArticulationController 를
    갖고 있어서 동시에 tick 하면 같은 관절을 두고 싸운다. 정확히 하나만 돌게 하는 게 이 스위치다.
    """

    def __init__(self, graph_path: str, log: Callable[[str], None]):
        self._graph_path = graph_path
        self._log = log
        self._mode: Optional[str] = None  # "node" | "evaluator" | None (no-op)

    def _probe(self) -> str:
        import omni.graph.core as og
        try:
            node = og.Controller.node(f"{self._graph_path}/OnPlaybackTick")
            if node is not None and hasattr(node, "set_disabled"):
                return "node"
        except Exception:
            pass
        try:
            attr = og.Controller.attribute(f"{self._graph_path}.evaluator:enabled")
            if attr is not None:
                return "evaluator"
        except Exception:
            pass
        return "noop"

    def set_active(self, active: bool):
        """active=False → graph stops driving joints; active=True → resume."""
        if self._mode is None:
            self._mode = self._probe()
            self._log(f"[graph] disable mode = {self._mode}")
        if self._mode == "noop":
            return
        import omni.graph.core as og
        try:
            if self._mode == "node":
                node = og.Controller.node(f"{self._graph_path}/OnPlaybackTick")
                node.set_disabled(not active)
            elif self._mode == "evaluator":
                og.Controller.set(
                    og.Controller.attribute(f"{self._graph_path}.evaluator:enabled"),
                    bool(active),
                )
        except Exception as e:
            self._log(f"[graph] toggle failed: {e}")


# =============================================================================
# Omni UI window
# =============================================================================

def clear_artic_commands(*graph_paths: str) -> None:
    """Empty the ArticulationController command inputs of the given graphs.

    Used whenever a graph (re)starts driving the robot — on pipeline-mode switch
    and on Stop/Play — so a stale retained /isaac_joint_commands (or /joint_states)
    value is not re-applied, which would snap the robot instead of leaving it at
    its current/reset pose. Mirrors start_isaac_sim_ur20.py's clear behavior.
    """
    import omni.graph.core as og
    for gp in graph_paths:
        for attr in ("jointNames", "positionCommand"):
            try:
                og.Controller.set(
                    og.Controller.attribute(f"{gp}/ArticulationController.inputs:{attr}"),
                    [],
                )
            except Exception:  # noqa: BLE001 — best effort
                pass


class PipelineWindow:
    """Four-panel Omni UI window: Load / Generate / Preview / Execute + Log."""

    LOG_MAX_LINES = 500

    def __init__(self, ghost_root_prim: str, base_link_path: str,
                 chain: "list[GhostJoint]", real_graph_path: str,
                 default_object: str, initial_mode: str = "sim",
                 sim_graph_path: str = "/SimRobotGraph",
                 initial_pipeline_mode: str = "inspection",
                 articulation_root: str = "",
                 scene: str = ""):
        import omni.ui as ui

        self._ui = ui
        self._mode = initial_mode  # "sim" (no live ROS) | "real" (ROS robot)
        # 활성 씬 이름 — 서브프로세스(check_ik/solve/plan_move)에 그대로 넘긴다.
        # 앱과 플래너가 다른 셀을 보면 조용히 틀린 궤적이 나온다.
        self._scene = scene or _cfg_module.ACTIVE_SCENE
        # Top-level mode: "inspection" (this whole UI) | "moveit" (MoveIt drives robot).
        self._pipeline_mode = initial_pipeline_mode
        self._log_lines: list[str] = []
        self._log_model = ui.SimpleStringModel("")
        self._csv_path_model = ui.SimpleStringModel("")
        self._h5_path_model = ui.SimpleStringModel("")

        # ONE editable camera spec (mm) shared by both inspection cameras. 프리뷰 고스트와
        # 실 로봇은 같은 물리 카메라의 두 표현이라, 스펙이 갈리면 프리뷰가 실제와 다른 화각을
        # 보여주는 거짓말이 된다. 편집하면 두 카메라의 USD intrinsic 에 같이 적용되고,
        # 공유 default = 로드된 뷰포인트를 만들 때 쓴 카메라 스펙(초기값 + Reset 대상). config
        # 기본에서 시작해 Show Viewpoints 가 h5 스냅샷으로 갱신한다. UI 입력값 자체는 카메라별로
        # 분리(테스트용) — preview/execute 에 서로 다른 값을 넣을 수 있다.
        from common import config as _cfg
        self._cam_spec_default = {
            "fov_w": float(_cfg.CAMERA_FOV_WIDTH_MM),
            "fov_h": float(_cfg.CAMERA_FOV_HEIGHT_MM),
            "wd": float(_cfg.CAMERA_WORKING_DISTANCE_MM),
        }
        # _real_graph_path 는 _apply_render_resolution 이 쓰므로 스펙 콜백보다 먼저 있어야 한다
        # (검사 카메라 렌더 프로덕트가 RealRobotGraph 안에 있다).
        self._real_graph_path = real_graph_path
        self._render_resolution = (_cfg.CAMERA_PUBLISH_W, _cfg.CAMERA_PUBLISH_H)
        #   preview -> InspectionCameraPreview (ghost, Preview panel)
        #   execute -> InspectionCamera        (real robot, Execute panel)
        # 각 타깃은 자기만의 스펙 모델을 갖는다(_make_cam_target).
        self._cam_targets = {
            "preview": self._make_cam_target("preview", "InspectionCameraPreview", GHOST_ROOT_PATH),
            "execute": self._make_cam_target("execute", "InspectionCamera", urctl.STAGE_PATH),
        }
        # Camera-range live update: cached object mesh (world) + throttle accum.
        self._range_trimesh = None
        self._range_accum = 0.0

        self._gen_runner = SubprocessRunner()
        self._ik_runner = SubprocessRunner()
        self._pub_runner = SubprocessRunner()
        self._ctrl_runner = SubprocessRunner()   # ros2 control switch/cancel calls
        self._relay_runner = SubprocessRunner()  # ros2 param set on the relay (mode gate)
        # Keep GraphTickSwitch around for the publish path. Preview no
        # longer needs it (ghost is a separate prim tree, not the real UR20),
        # so we leave the graph untouched during preview — the user-confirmed
        # stable original idle behavior is preserved.
        self._real_graph = GraphTickSwitch(real_graph_path, self._append_log)
        # Separate switch for the MoveIt bridge graph (/isaac_joint_commands).
        # Only one of (_real_graph, _sim_graph) ticks at a time — see apply_mode.
        self._sim_graph = GraphTickSwitch(sim_graph_path, self._append_log)
        self._sim_graph_path = sim_graph_path
        self._articulation_root = articulation_root
        self._mode_applied: Optional[str] = None  # last run mode actually applied
        self._preview = PreviewPlayer(
            ghost_root_prim, base_link_path, chain, self._append_log,
        )
        self._sim_executor = IsaacArticulationExecutor(
            articulation_root, JOINT_NAMES, self._append_log,
        )

        self._uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
        if not Path(self._uv).exists() and shutil.which("uv") is None:
            self._append_log(f"[warn] uv binary not found on PATH; falling back to: {self._uv}")

        # Mutable field models (created in _build).
        self._fields: dict = {}
        self._btn_generate = None
        self._btn_home_approach = None
        self._btn_home_return = None
        self._btn_cancel_gen = None
        self._btn_check_ik = None
        self._btn_cancel_ik = None
        self._btn_publish = None
        self._btn_cancel_pub = None
        self._btn_tilt_generate = None
        self._btn_tilt_cancel = None
        self._btn_tilt_fan = None
        self._tilt_fan_on = False
        # 장시간 작업이 도는 동안 유일하게 살아 있는 위젯(그 작업의 Cancel). None = 유휴.
        self._busy_cancel = None
        self._slider_model: Optional["ui.SimpleFloatModel"] = None
        self._slider: Optional["ui.FloatSlider"] = None
        self._updating_slider = False
        self._status_label: Optional["ui.Label"] = None
        self._mode_combo = None
        self._mode_label: Optional["ui.Label"] = None
        self._publish_hint_label: Optional["ui.Label"] = None
        self._pipeline_combo = None
        self._pipeline_label: Optional["ui.Label"] = None
        # Inspection widgets/frames locked (greyed) when pipeline mode = moveit.
        self._inspection_widgets: list = []
        self._inspection_frames: list = []

        self._window = ui.Window("Pipeline UI", width=520, height=820)
        self._default_object = default_object
        # Object currently loaded in the scene (gizmo target). Tracked so Generate
        # can validate the picked h5 against it and warn on mismatch.
        self._current_object = default_object
        self._objects = discover_objects()
        if default_object and default_object not in self._objects:
            self._objects.insert(0, default_object)
        if not self._objects:
            self._objects = [default_object or "sample"]
        self._object_combo = None
        self._build()

        # Dock into the right-hand panel instead of floating as a standalone
        # window. deferred_dock_in waits until the target panel exists in the
        # layout, then tabs this window alongside it. CURRENT_WINDOW_IS_ACTIVE
        # brings the Pipeline UI tab to the front. "Property" is the standard
        # bottom-right panel in the Isaac Sim default layout; change the title
        # (e.g. "Stage") to dock elsewhere.
        try:
            self._window.deferred_dock_in(
                "Property", ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE
            )
        except Exception as e:  # noqa: BLE001 — docking is cosmetic, never fatal
            self._append_log(f"[ui] dock failed ({e}); window stays floating")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build(self):
        ui = self._ui
        with self._window.frame:
            with ui.ScrollingFrame(
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            ):
                with ui.VStack(height=0, spacing=6):
                    self._build_panel_pipeline_mode()
                    self._build_panel_mode()
                    self._build_panel_object()
                    self._build_panel_generate()
                    self._build_panel_preview()
                    self._build_panel_publish()
                    # 워크플로(물체→생성→프리뷰→실행) 뒤, 로그 바로 위. 셀을 실측에 맞출 때만
                    # 쓰는 도구라 기본 접힘으로 눈에 안 띄게 둔다.
                    self._build_panel_scene()
                    self._build_log()

    def _lock(self, widget):
        """Register an interactive widget so it greys out in MoveIt mode. Returns it."""
        self._inspection_widgets.append(widget)
        return widget

    def _row(self, label: str, model, width: int = 180):
        ui = self._ui
        with ui.HStack(height=22, spacing=6):
            ui.Label(label, width=width)
            if isinstance(model, str):
                f = ui.StringField()
                f.model.set_value(model)
                return f.model
            elif isinstance(model, int):
                f = ui.IntField()
                f.model.set_value(model)
                return f.model
            elif isinstance(model, float):
                f = ui.FloatField()
                f.model.set_value(model)
                return f.model
            else:
                raise TypeError(type(model))

    def _checkbox_row(self, label: str, default: bool, width: int = 180):
        """라벨 + 실제 체크박스 한 줄. 모델을 반환한다(_get_field(key, bool)로 읽음)."""
        ui = self._ui
        with ui.HStack(height=22, spacing=6):
            ui.Label(label, width=width)
            cb = ui.CheckBox()
            cb.model.set_value(bool(default))
            return cb.model

    def _num_field(self, default, width: int = 70):
        """라벨 없는 좁은 숫자 입력 한 칸 (한 줄에 min/max/n 을 나란히 놓을 때). 모델 반환."""
        ui = self._ui
        f = ui.IntField(width=width) if isinstance(default, int) else ui.FloatField(width=width)
        f.model.set_value(default)
        self._lock(f)
        return f.model

    # ------------------------------------------------------------------
    # Pipeline mode panel (Inspection / MoveIt) — top-level selector
    # ------------------------------------------------------------------
    def _build_panel_pipeline_mode(self):
        ui = self._ui
        with ui.CollapsableFrame("Pipeline Mode", height=0, collapsed=False):
            with ui.VStack(spacing=4):
                with ui.HStack(height=26, spacing=8):
                    ui.Label("Pipeline", width=80)
                    idx = 1 if self._pipeline_mode == "moveit" else 0
                    self._pipeline_combo = ui.ComboBox(
                        idx, "Inspection", "MoveIt")
                    self._pipeline_combo.model.add_item_changed_fn(
                        self._on_pipeline_mode_changed)
                    self._pipeline_label = ui.Label(self._pipeline_text(), width=160)

    def _pipeline_text(self) -> str:
        if self._pipeline_mode == "moveit":
            return "MoveIt (Inspection locked)"
        return "Inspection"

    def _on_pipeline_mode_changed(self, *_):
        if self._pipeline_combo is None:
            return
        idx = self._pipeline_combo.model.get_item_value_model().get_value_as_int()
        self.apply_pipeline_mode("moveit" if idx == 1 else "inspection")

    def apply_pipeline_mode(self, mode: str):
        """Pipeline mode = command source + UI lock. It does NOT toggle any graph —
        the Run mode (sim/real) owns graph selection (see apply_mode). Both MoveIt
        (RViz move_group) and Inspection (Execute panel) ultimately command the same
        controller; pipeline mode only changes which tool the user drives with and
        locks the other.

        moveit     → Inspection panels greyed/locked (use RViz to drive the robot).
        inspection → Inspection panels active (Execute drives the current robot:
                     Isaac in sim, real robot in real).
        """
        self._pipeline_mode = mode
        self._set_inspection_ui_enabled(mode != "moveit")
        if mode == "inspection":
            self._sync_mode_ui()  # restore Execute-button state after unlock
        self._sync_pipeline_ui()
        # Activate exactly the controller for this mode so the OTHER source is
        # blocked at the controller level: MoveIt → scaled_joint_trajectory_controller,
        # Inspection → joint_trajectory_controller. In Inspection mode scaled is
        # deactivated, so MoveIt Execute is rejected ("controller not active").
        # (No-op in real mode if its stack uses the same controller names; best-effort
        # if the ROS stack isn't up yet.)
        if self._mode == "sim":
            # Inspection SIM executes directly through SingleArticulation and must
            # not require ROS or allow /SimRobotGraph to overwrite PD targets.
            # (그래서 sim × inspection 에서는 두 그래프 모두 꺼진다.)
            self._sim_graph.set_active(mode == "moveit")
            self._real_graph.set_active(False)
            if mode == "moveit":
                # Relay forwarding is owned by apply_mode (stays True throughout sim);
                # re-asserting it here is redundant and, at startup, double-starts the
                # relay runner before the UI loop has drained the first run.
                self._switch_controllers(MOVEIT_CONTROLLER, INSPECTION_CONTROLLER)
        elif mode == "moveit":
            self._switch_controllers(MOVEIT_CONTROLLER, INSPECTION_CONTROLLER)
        else:
            self._switch_controllers(INSPECTION_CONTROLLER, MOVEIT_CONTROLLER)
        self._append_log(
            f"[pipeline] -> {mode.upper()} :: "
            + ("MoveIt active, Inspection locked" if mode == "moveit"
               else "Inspection active, MoveIt blocked"))

    def _switch_controllers(self, activate: str, deactivate: str):
        """Activate one trajectory controller and deactivate the other, cancelling any
        lingering goal on BOTH around the switch (best-effort subprocess; the ROS
        stack lives in the other shell).

        Why cancel both: if a controller is deactivated mid-goal its action status
        freezes at EXECUTING, and the relay (which forwards /isaac_joint_commands
        only while a goal is active) then keeps forwarding the OTHER controller's
        idle hold command — a state→command→robot→state feedback loop that makes the
        robot shake. Cancelling the outgoing goal before the switch (its CANCELED
        status reaches the relay while still active) and the incoming controller's
        stale goal after the switch clears that, with no timers/deadlines."""
        if self._ctrl_runner.running:
            self._ctrl_runner.terminate()

        def cancel(ctrl: str) -> str:
            return (f"timeout 2 ros2 service call /{ctrl}/follow_joint_trajectory"
                    "/_action/cancel_goal action_msgs/srv/CancelGoal '{}' "
                    "2>/dev/null || true")

        shell_cmd = (
            "source /opt/ros/jazzy/setup.bash && "
            f"{cancel(deactivate)} ; "
            f"ros2 control switch_controllers --activate {activate} --deactivate {deactivate} ; "
            f"{cancel(activate)}"
        )
        self._append_log(f"[ctrl] switch: +{activate} -{deactivate} (cancel both)")
        self._ctrl_runner.start(
            ["bash", "-c", shell_cmd], cwd=PROJECT_ROOT,
            on_line=self._append_log,
            on_exit=lambda rc: self._append_log(f"[ctrl] switch exit={rc}"))

    def _ensure_inspection_controller_cmd(self) -> str:
        """Shell snippet that activates the inspection controller (jtc) and
        deactivates MoveIt's (scaled), so a real publish always reaches an ACTIVE
        controller.

        Prepended to every real controller-publish because the one-shot switch in
        apply_pipeline_mode/apply_mode can miss: a sim→real run-mode change does
        NOT switch controllers, and restarting the shell-2 UR stack resets it to
        scaled-active. Idempotent and best-effort ('|| true' keeps it non-fatal
        when already in that state), so it is safe to run before every send."""
        return (
            f"ros2 control switch_controllers "
            f"--activate {INSPECTION_CONTROLLER} --deactivate {MOVEIT_CONTROLLER} "
            f"|| true"
        )

    # Style overrides toggled with the lock so panels also *look* disabled
    # (this Isaac theme doesn't auto-grey on .enabled=False).
    _DIM_WIDGET_STYLE = {"color": 0xFF666666}            # dim text/foreground
    _DIM_FRAME_STYLE = {"CollapsableFrame": {"color": 0xFF666666},
                        "Label": {"color": 0xFF666666}}

    def _set_inspection_ui_enabled(self, on: bool):
        """Grey out / re-enable every Inspection widget and panel frame.

        Sets .enabled (blocks input) AND a dimmed style (visual cue); clearing
        the style with {} reverts to the theme default when re-enabled.
        """
        w_style = {} if on else self._DIM_WIDGET_STYLE
        f_style = {} if on else self._DIM_FRAME_STYLE
        for w in self._inspection_widgets:
            try:
                w.enabled = on
                w.style = w_style
            except Exception:  # noqa: BLE001 — best-effort, never fatal
                pass
        for f in self._inspection_frames:
            try:
                f.enabled = on
                f.style = f_style
            except Exception:  # noqa: BLE001
                pass

    def _set_busy(self, cancel_button=None):
        """장시간 작업이 도는 동안 Inspection 위젯을 전부 잠그고 그 작업의 Cancel 하나만 남긴다.

        IK 체크 / 스캔·틸트 생성 / 이동 / 실행이 모두 같은 스테이지 상태(물체 pose, 선택한
        h5·CSV)와 같은 러너를 공유한다. 도중에 다른 버튼이 눌리면 입력이 바뀐 채로 결과가
        돌아오거나 러너가 갈아엎힌다 — 그래서 하나가 돌면 나머지는 전부 잠근다.

        프레임(_inspection_frames)은 건드리지 않는다: 컨테이너를 비활성화하면 살려둔 Cancel
        버튼까지 같이 죽을 수 있다. 잠금은 위젯 단위로만 한다.
        """
        self._busy_cancel = cancel_button
        for widget in self._inspection_widgets:
            keep = widget is cancel_button
            try:
                widget.enabled = keep
                widget.style = {} if keep else self._DIM_WIDGET_STYLE
            except Exception:  # noqa: BLE001 — best-effort, never fatal
                pass

    def _clear_busy(self):
        """작업이 끝났다 — Inspection UI 를 되살리고 모드별 상태를 다시 반영한다."""
        self._busy_cancel = None
        if self._pipeline_mode != "inspection":
            return          # MoveIt 락이 이긴다 — 그쪽이 계속 잠근 상태로 두어야 한다
        self._set_inspection_ui_enabled(True)
        self._sync_mode_ui()

    def _sync_pipeline_ui(self):
        if self._pipeline_label is not None:
            self._pipeline_label.text = self._pipeline_text()
            self._pipeline_label.style = {
                "color": 0xFF3399FF if self._pipeline_mode == "moveit" else 0xFFCCCCCC
            }

    # ------------------------------------------------------------------
    # Mode panel (sim / real) + helpers
    # ------------------------------------------------------------------
    def _build_panel_mode(self):
        ui = self._ui
        # Run mode (sim/real) is a TOP-LEVEL axis like Pipeline mode — it must stay
        # selectable in BOTH MoveIt and Inspection. So it is NOT added to the
        # Inspection lock lists (_inspection_frames/_lock).
        frame = ui.CollapsableFrame("Run Mode", height=0, collapsed=False)
        with frame:
            with ui.VStack(spacing=4):
                with ui.HStack(height=26, spacing=8):
                    ui.Label("Run mode", width=80)
                    idx = 0 if self._mode == "sim" else 1
                    self._mode_combo = ui.ComboBox(
                        idx, "Simulation (Isaac only)", "Real (ROS robot)")
                    self._mode_combo.model.add_item_changed_fn(self._on_mode_changed)
                    # self._mode_label = ui.Label(self._mode_text(), width=120)
                # ui.Label("sim = Isaac only, no ROS — A Load, B Generate, C Preview.  "
                #          "real = all of that + D Publish to the robot & mirror /joint_states "
                #          "(needs ur_robot_driver).",
                #          height=40, word_wrap=True)

    def _mode_text(self) -> str:
        return f"{self._mode.upper()}"

    def _on_mode_changed(self, *_):
        if self._mode_combo is None:
            return
        idx = self._mode_combo.model.get_item_value_model().get_value_as_int()
        self.apply_mode("sim" if idx == 0 else "real")

    def apply_mode(self, mode: str):
        """Run mode = which robot drives the Isaac articulation:

          sim  → Isaac IS the robot: /SimRobotGraph drives from /isaac_joint_commands
                 and publishes /isaac_joint_states + /clock; /RealRobotGraph mirror OFF.
          real → Isaac MIRRORS the real robot: /RealRobotGraph ON (drive from real
                 /joint_states + cameras); /SimRobotGraph's driving + publishing nodes
                 OFF (so it neither moves Isaac nor feeds the twin loop).

        Cross-mode replay is stopped at the SOURCE: the relay (셸2) is the only thing
        that feeds /isaac_joint_commands, so the app sets its `forward_enabled`
        parameter — true in sim, false in real. In real mode the relay discards
        commands, so a MoveIt Execute done in real mode never reaches Isaac and there
        is nothing to replay when sim is re-entered (works even if the goal is still
        active). No rebuild / cancel / timing needed.

        REAL and MoveIt modes require the matching ROS stack; Inspection+SIM does not.
        """
        self._mode = mode
        if mode == "sim":
            clear_artic_commands(self._sim_graph_path)
            self._real_graph.set_active(False)      # /RealRobotGraph mirror off
            if self._pipeline_mode == "moveit":
                self._set_relay_forwarding(True)
                self._sim_graph.set_active(True)
                which = "/SimRobotGraph drives (live /isaac_joint_commands)"
            else:
                self._sim_graph.set_active(False)
                which = "in-process executor drives Isaac UR20 (ROS-free)"
        else:  # real
            self._set_relay_forwarding(False)  # relay discards → Isaac not driven by commands
            self._sim_graph.set_active(False)
            clear_artic_commands(self._real_graph_path)
            self._real_graph.set_active(True)       # /RealRobotGraph mirror on
            which = "/RealRobotGraph mirrors real /joint_states (twin)"
        self._mode_applied = mode
        self._sync_mode_ui()
        self._append_log(f"[run-mode] -> {mode.upper()} :: {which}")

    def _set_relay_forwarding(self, on: bool):
        """Tell the relay (셸2) whether to feed /isaac_joint_commands. This is the
        mode gate: in real mode the relay discards commands so they never reach
        Isaac (no buffering, no replay on sim re-entry). Best-effort subprocess."""
        if self._relay_runner.running:
            self._relay_runner.terminate()
        val = "true" if on else "false"
        cmd = ("source /opt/ros/jazzy/setup.bash && "
               f"timeout 3 ros2 param set /isaac_joint_command_relay forward_enabled {val} "
               "2>/dev/null || true")
        self._append_log(f"[relay] forward_enabled -> {val}")
        self._relay_runner.start(
            ["bash", "-c", cmd], cwd=PROJECT_ROOT, on_line=self._append_log,
            on_exit=lambda rc: self._append_log(f"[relay] param set exit={rc}"))

    def _sync_mode_ui(self):
        if self._mode_label is not None:
            self._mode_label.text = self._mode_text()
            self._mode_label.style = {
                "color": 0xFF33CC33 if self._mode == "real" else 0xFFFF6622
            }
        if self._publish_hint_label is not None:
            self._publish_hint_label.text = self._publish_hint_text()
            self._publish_hint_label.style = {
                "color": 0xFF33CC33 if self._mode == "real" else 0xFF2277EE
            }
        if self._btn_publish is not None:
            # Execute works in both sim (→ Isaac) and real (→ real robot); it is
            # available in Inspection pipeline mode (locked in MoveIt mode).
            self._btn_publish.text = (
                "Execute on Isaac UR20" if self._mode == "sim"
                else "Execute on Real Robot"
            )
            self._btn_publish.enabled = (
                self._pipeline_mode == "inspection"
                and not self._pub_runner.running and not self._sim_executor.running
            )
        # Unified labels across sim/real (sim naming is the standard). "Move to Start" 는
        # 스캔·틸트 공용이다 — 두 궤적이 같은 CSV 칸을 쓰므로 하는 일이 정확히 같다.
        if self._btn_home_approach is not None:
            self._btn_home_approach.text = "Move to Start"
        if self._btn_home_return is not None:
            self._btn_home_return.text = "Return to HOME"
        # _pub_runner 항이 빠져 있어 real 이동 중에 버튼이 되살아났다(기존 버그).
        home_enabled = (
            self._pipeline_mode == "inspection" and not self._gen_runner.running
            and not self._sim_executor.running and not self._pub_runner.running
        )
        for button in (self._btn_home_approach, self._btn_home_return):
            if button is not None:
                button.enabled = home_enabled

    def _build_panel_scene(self):
        """활성 셀(씬) 표시 + 뷰포트에서 잰 치수를 씬 YAML 조각으로 뽑는 보조 도구.

        일상 워크플로가 아니다 — 씬을 실측에 맞출 때만 쓴다. 그래서 워크플로 패널들 뒤,
        로그 바로 위에 **기본 접힘**으로 둔다.

        기즈모로 옮긴 결과는 YAML 에 적어야 계획에 반영된다(스테이지는 플래너의 진실원이
        아니다). 이 버튼은 그 숫자를 robot frame 으로 변환해 붙여넣기 좋게 찍어줄 뿐,
        파일은 쓰지 않는다.
        """
        ui = self._ui
        frame = ui.CollapsableFrame("Scene (obstacles)", height=0, collapsed=True)
        self._inspection_frames.append(frame)
        with frame:
            with ui.VStack(spacing=4):
                with ui.HStack(height=22, spacing=6):
                    ui.Label("Active scene", width=110)
                    ui.Label(f"{self._scene}  (workcell/scenes/{self._scene}.yaml)")
                with ui.HStack(height=28, spacing=6):
                    self._lock(ui.Button("Log Selected Prim as YAML",
                                         clicked_fn=self._on_log_selected_prim_yaml))

    def _on_log_selected_prim_yaml(self):
        """뷰포트에서 고른 prim 의 pose/치수를 씬 YAML 조각으로 로그에 찍는다(파일은 안 쓴다).

        파일을 쓰지 않는 이유: 씬 YAML 은 손으로 쓴 근거 주석을 달고 있어서 writer 가 그걸
        날린다. 실측값은 사람이 보고 넣는 게 맞다.
        """
        import omni.usd
        from pxr import Gf, Usd, UsdGeom

        try:
            selection = omni.usd.get_context().get_selection()
            paths = list(selection.get_selected_prim_paths())
        except Exception as exc:      # Kit 버전마다 selection API 가 다르다 — 죽지 않고 안내만
            self._append_log(f"[scene] selection API unavailable ({exc}) — check the Stage tree")
            return
        if not paths:
            self._append_log("[scene] select a prim in the viewport or Stage tree first.")
            return

        stage = omni.usd.get_context().get_stage()
        # BBoxCache 는 참조된 USD(테이블 등) 같은 임의 prim 에도 동작한다. Xformable 의
        # 행렬에서 ExtractRotationQuat 를 쓰면 스케일이 섞인 행렬에서 회전이 오염되므로
        # 쓰지 않는다 — 여기서는 bbox 행렬의 basis 행을 정규화해 회전과 크기를 분리한다.
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                  [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                                  useExtentsHint=True)
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                self._append_log(f"[scene] {path}: not a valid prim")
                continue
            bound = cache.ComputeWorldBound(prim)
            rng = bound.GetRange()
            if rng.IsEmpty():
                self._append_log(f"[scene] {path}: no bbox (Xform-only prim?)")
                continue

            m = bound.GetMatrix()
            # bbox 행렬의 basis 행 노름 = 축별 스케일. 그걸 빼내야 (a) 치수가 실제 크기가 되고
            # (b) 남은 정규직교 행렬에서 회전을 깨끗하게 뽑을 수 있다.
            basis = [np.array([m[r][0], m[r][1], m[r][2]], dtype=float) for r in range(3)]
            scales = [float(np.linalg.norm(b)) or 1.0 for b in basis]
            dims = [float(s) * k for s, k in zip(rng.GetSize(), scales)]
            center_world = m.Transform(rng.GetMidpoint())
            position = [float(center_world[0]), float(center_world[1]),
                        float(center_world[2]) - urctl.MOUNT_HEIGHT]

            rot = Gf.Matrix4d()
            for r, (b, s) in enumerate(zip(basis, scales)):
                rot.SetRow3(r, Gf.Vec3d(*(b / s)))
            quat = rot.ExtractRotationQuat().GetNormalized()
            rotation = [float(quat.GetReal()), *(float(v) for v in quat.GetImaginary())]

            name = _SCENE_PRIM_NAME_RE.sub("_", prim.GetName()).strip("_").lower() or "obstacle"
            snippet = scene_config.obstacle_yaml_snippet(name, position, dims, rotation)
            if np.any(np.asarray(dims) <= 0.0):
                self._append_log(f"[scene] {path}: zero-sized — prim may have no geometry")
            if urctl._is_under_root(str(path), urctl.STAGE_PATH) or \
                    urctl._is_under_root(str(path), GHOST_ROOT_PATH):
                self._append_log(f"[scene] note: {path} is part of the robot, not an obstacle")
            self._append_log(
                f"[scene] {path} → paste into obstacles: in workcell/scenes/{self._scene}.yaml\n"
                f"{snippet}")

    def _build_panel_object(self):
        """검사 대상 정의 — 물체, 그 물체의 viewpoint h5, 그리고 카메라 스펙.

        셋은 같은 것에 대한 설명이다: viewpoint h5 는 data/{object}/viewpoint/ 아래 살고,
        Show Viewpoints 는 그 물체 위에 점을 그리며(물체가 먼저 로드돼 있어야 한다),
        카메라 스펙은 그 h5 가 어떤 카메라로 계획됐는지를 말한다. 예전에는 h5 가 Generate 에,
        카메라 스펙이 Preview/Execute 에 흩어져 있어 이 선후관계가 UI 에 안 드러났다.
        """
        ui = self._ui
        frame = ui.CollapsableFrame("Load Object & Viewpoints", height=0)
        self._inspection_frames.append(frame)
        with frame:
            with ui.VStack(spacing=4):
                with ui.HStack(height=22, spacing=6):
                    ui.Label("Object", width=80)
                    default_idx = self._objects.index(self._default_object) \
                        if self._default_object in self._objects else 0
                    self._object_combo = self._lock(ui.ComboBox(default_idx, *self._objects))
                    self._lock(ui.Button("Load Object", width=110, clicked_fn=self._on_load_object))
                    self._lock(ui.Button("Log Pose", width=90, clicked_fn=self._on_log_object_pose))
                with ui.HStack(height=22, spacing=6):
                    ui.Label("Viewpoints (h5)", width=110)
                    self._lock(ui.StringField(model=self._h5_path_model))
                    self._lock(ui.Button("Browse...", width=80, clicked_fn=self._on_browse_h5))
                with ui.HStack(height=28, spacing=6):
                    self._lock(ui.Button("Show Viewpoints", clicked_fn=self._on_show_viewpoints))
                    self._lock(ui.Button("Clear Viewpoints", clicked_fn=self._on_clear_viewpoints))
                # 카메라 스펙 입력은 Preview / Execute 패널로 옮겼다. Show Viewpoints 는
                # 여전히 h5 스냅샷으로 그 공유 스펙을 채워준다(_sync_camera_spec_from_h5).

    def _build_panel_generate(self):
        ui = self._ui
        frame = ui.CollapsableFrame("Generate Trajectory", height=0)
        self._inspection_frames.append(frame)
        with frame:
            with ui.VStack(spacing=4):
                # ui.Label("Pick the object's viewpoints .h5 and Generate. Object name + viewpoint "
                #          "count are read from the h5 path; the object's live pose comes from the "
                #          "scene — load & place it in panel A first.",
                #          height=40, word_wrap=True)
                # viewpoint h5 선택은 위 "Load Object & Viewpoints" 로 옮겼다 —
                # 여기서는 그 h5(_h5_path_model)를 입력으로 쓰기만 한다.
                # IK 후보 옵션 — Check and Save IK 와 Generate 가 공유한다. 값이 같아야 Generate
                # 가 저장된 IK(data/{object}/ik/{N}/*.h5)를 그대로 재사용한다. 기본 펼침.
                with ui.CollapsableFrame("IK options", height=0, collapsed=False):
                    with ui.VStack(spacing=4):
                        self._fields["glns_roll_augment"] = self._checkbox_row("roll augment", True)
                        self._fields["glns_roll_step"] = self._row("    roll-step-deg", 30.0)
                        self._fields["glns_tilt_augment"] = self._checkbox_row("tilt augment", True)
                        self._fields["glns_tilt_angles"] = self._row("    tilt-angles-deg", "5 10")
                        self._fields["glns_tilt_azimuths"] = self._row("    tilt-azimuths", 8)
                        self._fields["glns_dedup"] = self._checkbox_row("dedup", True)
                        self._fields["glns_dedup_rad"] = self._row(
                            "    dedup-rad", float(CANDIDATE_DEDUP_RAD))
                        self._fields["glns_num_seeds"] = self._row("num-seeds", 32)
                        self._fields["glns_ik_batch_size"] = self._row("ik-batch-size", 128)
                with ui.HStack(height=28, spacing=6):
                    self._btn_check_ik = self._lock(ui.Button(
                        "Check and Save IK",
                        clicked_fn=self._on_check_ik_reachability,
                    ))
                    self._btn_cancel_ik = self._lock(ui.Button("Cancel IK Check", clicked_fn=self._on_cancel_ik))
                with ui.CollapsableFrame("Scan options (GLNS)", height=0, collapsed=True):
                    with ui.VStack(spacing=4):
                        self._fields["glns_hops"]     = self._row("--delaunay-expand-hops", 2)
                        self._fields["glns_max_candidates"] = self._row(
                            "--max-candidates-per-viewpoint", 32)
                with ui.HStack(height=28, spacing=6):
                    self._btn_generate = self._lock(ui.Button(
                        "Generate Scan Motion", clicked_fn=self._on_generate))
                    self._btn_cancel_gen = self._lock(ui.Button(
                        "Cancel", clicked_fn=self._on_cancel_generate))
                # Tilt: 스캔과 나란한 두 번째 궤적 생성기 — 모든 viewpoint 를 한 번씩 도는 대신
                # viewpoint 하나를 표면점 중심으로 공전한다(center→up→center→down→center→
                # left→center→right→center). 입력(물체 pose, viewpoints h5)과 산출물이 놓이는
                # 자리가 스캔과 같아서 같은 패널에 둔다. 재생/실행은 아래 공용 Preview/Execute.
                with ui.CollapsableFrame("Tilt options", height=0, collapsed=True):
                    with ui.VStack(spacing=4):
                        with ui.HStack(height=22, spacing=6):
                            ui.Label("center viewpoint idx", width=140)
                            field = ui.IntField()
                            field.model.set_value(0)
                            self._fields["tilt_index"] = field.model
                            self._lock(field)
                            self._lock(ui.Button("Highlight", width=90,
                                                 clicked_fn=self._on_highlight_tilt_viewpoint))
                        # pitch(down/up) = 카메라 y축 둘레 공전, roll(left/right) = x축 둘레.
                        # n 은 '중심 → 끝' 한쪽의 샘플 수(중심 포함)라 leg 당 새 포즈는 n-1 개다.
                        with ui.HStack(height=22, spacing=6):
                            ui.Label("pitch down/up deg", width=140)
                            self._fields["tilt_pitch_min"] = self._num_field(-20.0)
                            self._fields["tilt_pitch_max"] = self._num_field(20.0)
                            ui.Label("n", width=12)
                            self._fields["tilt_pitch_n"] = self._num_field(40, width=50)
                        with ui.HStack(height=22, spacing=6):
                            ui.Label("roll left/right deg", width=140)
                            self._fields["tilt_roll_min"] = self._num_field(-20.0)
                            self._fields["tilt_roll_max"] = self._num_field(20.0)
                            ui.Label("n", width=12)
                            self._fields["tilt_roll_n"] = self._num_field(40, width=50)
                        for key in ("tilt_index", "tilt_pitch_min", "tilt_pitch_max",
                                    "tilt_pitch_n", "tilt_roll_min", "tilt_roll_max",
                                    "tilt_roll_n"):
                            self._fields[key].add_value_changed_fn(self._refresh_tilt_fan)
                        self._fields["tilt_num_seeds"] = self._row("num-seeds", 32)
                        self._fields["tilt_batch_size"] = self._row("ik-batch-size", 128)
                        self._fields["tilt_clamp"] = self._checkbox_row(
                            "clamp unreachable angles", True)
                with ui.HStack(height=28, spacing=6):
                    self._btn_tilt_generate = self._lock(ui.Button(
                        "Generate Tilt Motion", clicked_fn=self._on_generate_tilt))
                    self._btn_tilt_cancel = self._lock(ui.Button(
                        "Cancel", clicked_fn=self._on_cancel_generate))
                with ui.HStack(height=28, spacing=6):
                    self._btn_tilt_fan = self._lock(ui.Button(
                        "Show Tilt Fan", clicked_fn=self._on_toggle_tilt_fan))
                    self._lock(ui.Button("Clear Tilt Fan",
                                         clicked_fn=self._on_clear_tilt_fan))

    def _build_panel_preview(self):
        ui = self._ui
        frame = ui.CollapsableFrame("Preview in Simulation", height=0)
        self._inspection_frames.append(frame)
        with frame:
            with ui.VStack(spacing=4):
                # ui.Label("Ghost playback inside Isaac — visual only, never touches the real "
                #          "robot or ROS. Available in both sim and real mode.",
                #          height=28, word_wrap=True)
                with ui.HStack(height=22, spacing=6):
                    ui.Label("CSV path", width=80)
                    self._lock(ui.StringField(model=self._csv_path_model))
                    self._lock(ui.Button("Browse...", width=80, clicked_fn=self._on_browse_csv))
                with ui.HStack(height=28, spacing=6):
                    self._lock(ui.Button("Load & Preview", clicked_fn=self._on_load_preview))
                    self._lock(ui.Button("Play", clicked_fn=self._on_play))
                    self._lock(ui.Button("Pause", clicked_fn=self._on_pause))
                    self._lock(ui.Button("Stop", clicked_fn=self._on_stop))
                with ui.HStack(height=28, spacing=6):
                    self._lock(ui.Button("Show Collision Spheres", clicked_fn=self._on_show_collision_spheres))
                    self._lock(ui.Button("Clear Collision Spheres", clicked_fn=self._on_clear_collision_spheres))
                # 고스트(InspectionCameraPreview) 시각화. 스펙 값 자체는 위
                # "Load Object & Viewpoints" 에서 두 카메라가 공유한다.
                self._build_camera_view_ui("preview")
                with ui.HStack(height=22, spacing=6):
                    ui.Label("t", width=20)
                    self._slider_model = ui.SimpleFloatModel(0.0)
                    self._slider = self._lock(ui.FloatSlider(self._slider_model, min=0.0, max=1.0))
                    self._slider.model.add_value_changed_fn(self._on_slider)
                self._status_label = ui.Label("t=0.00s / 0.00s  (no CSV)")

    def _build_panel_publish(self):
        ui = self._ui
        frame = ui.CollapsableFrame("Execute Trajectory", height=0)
        self._inspection_frames.append(frame)
        with frame:
            with ui.VStack(spacing=4):
                # self._publish_hint_label = ui.Label(self._publish_hint_text(),
                #                                      height=28, word_wrap=True)
                with ui.HStack(height=22, spacing=6):
                    ui.Label("CSV path", width=80)
                    self._lock(ui.StringField(model=self._csv_path_model))
                    self._lock(ui.Button("Browse...", width=80, clicked_fn=self._on_browse_csv))
                # 실 카메라(InspectionCamera, ROS render product) 시각화. 스펙은 공유.
                self._build_camera_view_ui("execute")
                with ui.HStack(height=28, spacing=6):
                    self._btn_home_approach = self._lock(ui.Button(
                        "Move to Start",
                        clicked_fn=lambda: self._on_plan_home_transition("approach")))
                    self._btn_home_return = self._lock(ui.Button(
                        "Return to HOME",
                        clicked_fn=lambda: self._on_plan_home_transition("return")))
                    self._btn_home_approach.enabled = True
                    self._btn_home_return.enabled = True
                with ui.HStack(height=28, spacing=6):
                    self._btn_publish = self._lock(ui.Button(
                        "Execute Selected CSV", clicked_fn=self._on_execute))
                    self._btn_cancel_pub = self._lock(ui.Button(
                        "Cancel", clicked_fn=self._on_cancel_execute))

    def _publish_hint_text(self) -> str:
        if self._mode == "real":
            return "● REAL mode — executes the CSV on the live robot."
        return "● SIM mode — executes the CSV on the Isaac UR20 articulation."

    def _build_log(self):
        ui = self._ui
        with ui.CollapsableFrame("Log", height=0):
            with ui.ScrollingFrame(height=260,
                                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                                    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON):
                ui.StringField(model=self._log_model, multiline=True, read_only=True)

    # ------------------------------------------------------------------
    # Log helpers
    # ------------------------------------------------------------------
    def _append_log(self, line: str):
        self._log_lines.append(line)
        if len(self._log_lines) > self.LOG_MAX_LINES:
            self._log_lines = self._log_lines[-self.LOG_MAX_LINES:]
        self._log_model.set_value("\n".join(self._log_lines))

    # ------------------------------------------------------------------
    # Generate panel callbacks
    # ------------------------------------------------------------------
    def _get_field(self, key, kind):
        m = self._fields[key]
        if kind is bool:
            return m.get_value_as_bool()
        if kind is str:
            return m.get_value_as_string()
        if kind is int:
            return m.get_value_as_int()
        if kind is float:
            return m.get_value_as_float()
        raise TypeError(kind)

    def _ik_batch(self) -> int:
        return max(1, int(self._get_field("glns_ik_batch_size", int)))

    def _ik_candidate_tokens(self) -> list:
        """roll/tilt/dedup/num-seeds/ik-seed CLI 토큰 — Check and Save IK 와 Generate 가
        공유한다. 두 쪽이 같은 값을 넘겨야 저장된 IK(ik_*.h5)가 그대로 재사용된다. IK 배치
        크기는 결과에 영향이 없고 스크립트별 플래그명이 달라(--batch-size vs --ik-batch-size)
        여기 넣지 않고 호출자가 붙인다."""
        toks: list[str] = []
        if self._get_field("glns_roll_augment", bool):
            toks.append("--roll-augment")
            toks += ["--roll-step-deg", str(float(self._get_field("glns_roll_step", float)))]
        if self._get_field("glns_tilt_augment", bool):
            toks.append("--tilt-augment")
            toks.append("--tilt-angles-deg")
            toks += [str(float(x)) for x in self._get_field("glns_tilt_angles", str).split()]
            toks += ["--tilt-azimuths", str(max(1, int(self._get_field("glns_tilt_azimuths", int))))]
        if not self._get_field("glns_dedup", bool):
            toks.append("--no-dedup")
        else:
            toks += ["--dedup-rad", str(float(self._get_field("glns_dedup_rad", float)))]
        toks += ["--num-seeds", str(max(1, int(self._get_field("glns_num_seeds", int))))]
        toks += ["--ik-seed", str(IK_RANDOM_SEED)]
        return toks

    def _on_load_object(self):
        """Swap /World/target_object to the dropdown selection at its default pose."""
        idx = self._object_combo.model.get_item_value_model().get_value_as_int()
        obj = self._objects[idx]
        usd_path = PROJECT_ROOT / "data" / obj / "mesh" / "source.usd"
        if not usd_path.exists():
            self._append_log(
                f"[object] '{obj}' has no source.usd - build it once, then retry:\n"
                f"  uv run scripts/setup/build_object_usd.py --object {obj}")
            return
        self._append_log(f"[object] loading '{obj}' ...")
        try:
            urctl.load_target_object(obj)
        except Exception as e:
            self._append_log(f"[object] load failed: {e}")
            return
        self._current_object = obj
        self._range_trimesh = None  # new geometry — force range re-extract
        self._append_log(
            f"[object] loaded '{obj}'. Move it with the viewport gizmo (W/E), then Generate.")

    def _on_log_object_pose(self):
        """Print the current object world orientation for prepare_object_mesh.py."""
        pose = self._read_object_world_pose()
        if pose is None:
            self._append_log("[object] no target prim on stage - Load Object first.")
            return
        (rx, ry, rz), (w, x, y, z) = pose
        obj = (self._current_object or "").strip() or "<name>"
        self._append_log(
            f"[object] world quat (w,x,y,z) = {w:.6f} {x:.6f} {y:.6f} {z:.6f}\n"
            f"[object] robot-frame pos = {rx:.4f} {ry:.4f} {rz:.4f}  "
            f"(world z = {rz + urctl.MOUNT_HEIGHT:.4f})\n"
            f"[object] bake upright: uv run scripts/setup/prepare_object_mesh.py "
            f"reorient --object {obj} "
            f"--world-target-quat {w:.6f} {x:.6f} {y:.6f} {z:.6f}")

    def _read_object_world_pose(self):
        """World pose of /World/target_object → (pos_robot (x,y,z), quat (w,x,y,z)) or None.

        Reads the live transform (gizmo edits included) and converts world→robot
        frame: x/y and rotation unchanged, z -= MOUNT_HEIGHT (config.py frame note).
        """
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(TARGET_OBJECT_PRIM)
        if not prim or not prim.IsValid():
            return None
        m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
        t = m.ExtractTranslation()
        q = m.ExtractRotationQuat()  # Gf.Quatd, (w,x,y,z) via GetReal/GetImaginary
        w = float(q.GetReal())
        x, y, z = (float(v) for v in q.GetImaginary())
        pos_robot = (float(t[0]), float(t[1]), float(t[2]) - urctl.MOUNT_HEIGHT)
        return pos_robot, (w, x, y, z)

    @staticmethod
    def _parse_h5_meta(h5_path: str):
        """Derive (object, num_viewpoints) from a standard viewpoints path
        ``data/{object}/viewpoint/{N}/file.h5``. Either may be None if the path
        is off-layout."""
        parts = Path(h5_path).parts
        obj = None
        num = None
        if "data" in parts:
            i = parts.index("data")
            if i + 1 < len(parts):
                obj = parts[i + 1]
        if "viewpoint" in parts:
            j = parts.index("viewpoint")
            if j + 1 < len(parts) and parts[j + 1].isdigit():
                num = int(parts[j + 1])
        return obj, num

    @staticmethod
    def _load_camera_viewpoint_points(h5_path: str):
        """Return camera viewpoint points in the target object's local frame."""
        from core.viewpoint.storage import load_viewpoints_hdf5

        viewpoint = load_viewpoints_hdf5(h5_path)
        positions = viewpoint.positions
        normals = viewpoint.normals
        wd_m = viewpoint.working_distance_m

        n = np.linalg.norm(normals, axis=1, keepdims=True)
        safe_normals = np.divide(
            normals, n,
            out=np.zeros_like(normals),
            where=n > 1e-12,
        )
        return positions + safe_normals * wd_m, wd_m

    def _draw_camera_viewpoint_points(self, points_local, colors=None, opacity: float = 0.9,
                                      highlight: Optional[int] = None):
        """뷰포인트 점들을 물체 로컬 좌표로 그린다.

        ``highlight`` 를 주면 그 인덱스 하나만 크고 노랗게 — tilt 중심으로 어느 점을 골랐는지
        스테이지에서 바로 보이게 한다(색 배열이 이미 있으면 그 점만 덮어쓴다).
        """
        import omni.usd
        from pxr import Gf, UsdGeom, Vt

        stage = omni.usd.get_context().get_stage()
        self._delete_viewpoint_points(log=False)

        widths = [VIEWPOINT_POINT_WIDTH_M] * len(points_local)
        if highlight is not None and 0 <= highlight < len(points_local):
            widths[highlight] = VIEWPOINT_POINT_WIDTH_M * VIEWPOINT_HIGHLIGHT_SCALE
            colors = (list(colors) if colors is not None
                      else [(0.0, 0.85, 1.0)] * len(points_local))
            colors[highlight] = (1.0, 0.85, 0.0)

        UsdGeom.Xform.Define(stage, VIEWPOINTS_ROOT_PRIM)
        points = UsdGeom.Points.Define(stage, VIEWPOINTS_POINTS_PRIM)
        points.CreatePointsAttr(Vt.Vec3fArray([
            Gf.Vec3f(float(p[0]), float(p[1]), float(p[2]))
            for p in points_local
        ]))
        points.CreateWidthsAttr(Vt.FloatArray(widths))

        if colors is None:
            points.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.0, 0.85, 1.0)]))
        elif len(colors) == len(points_local):
            color_primvar = points.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
            color_primvar.Set(Vt.Vec3fArray([
                Gf.Vec3f(float(c[0]), float(c[1]), float(c[2]))
                for c in colors
            ]))
        else:
            raise ValueError(
                f"color count {len(colors)} does not match point count {len(points_local)}"
            )

        points.CreateDisplayOpacityAttr(Vt.FloatArray([float(opacity)]))

    def _on_show_viewpoints(self):
        """Visualize camera viewpoints from the selected h5 as object-local USD points."""
        h5 = self._h5_path_model.get_value_as_string().strip()
        if not h5:
            self._append_log("[viewpoints] pick a viewpoints .h5 first (Browse...).")
            return
        if not Path(h5).exists():
            self._append_log(f"[viewpoints] h5 not found: {h5}")
            return

        import omni.usd

        stage = omni.usd.get_context().get_stage()
        target_prim = stage.GetPrimAtPath(TARGET_OBJECT_PRIM)
        if not target_prim or not target_prim.IsValid():
            self._append_log("[viewpoints] no target object on stage - Load Object first.")
            return

        try:
            points_local, wd_m = self._load_camera_viewpoint_points(h5)
        except Exception as e:
            self._append_log(f"[viewpoints] load failed: {e}")
            return
        # Adopt this viewpoint set's camera snapshot as the FOV/range default.
        # config 와 달라도 경고하지 않는다 — WD/FOV 는 viewpoint 생성 시 고르는 값이고
        # (viewpoint_studio / viewpoint cli), 스펙칸이 방금 그 값으로 맞춰졌다. 기하학적으로
        # 불가능한 WD 만 load_viewpoints_hdf5 가 잡는다(config.working_distance_error).
        self._sync_camera_spec_from_h5(h5)

        self._draw_camera_viewpoint_points(points_local)

        self._append_log(
            f"[viewpoints] displayed {len(points_local)} camera points under "
            f"{VIEWPOINTS_ROOT_PRIM} (working distance={wd_m * 1000:.1f} mm)")

    def _delete_viewpoint_points(self, log: bool):
        from isaacsim.core.utils import prims

        if prims.is_prim_path_valid(VIEWPOINTS_ROOT_PRIM):
            prims.delete_prim(VIEWPOINTS_ROOT_PRIM)
            if log:
                self._append_log(f"[viewpoints] cleared {VIEWPOINTS_ROOT_PRIM}")
        elif log:
            self._append_log("[viewpoints] nothing to clear")

    def _on_clear_viewpoints(self):
        self._delete_viewpoint_points(log=True)

    def _apply_ik_reachability_result(self, h5_path: str, result_path: Path):
        if not result_path.exists():
            self._append_log(f"[ik] result JSON not found: {result_path}")
            return

        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            counts = np.array(result["success_counts"], dtype=np.int32)
            points_local, wd_m = self._load_camera_viewpoint_points(h5_path)
        except Exception as e:
            self._append_log(f"[ik] result load failed: {e}")
            return

        if len(counts) != len(points_local):
            self._append_log(
                f"[ik] result/viewpoint count mismatch: {len(counts)} vs {len(points_local)}"
            )
            return

        colors = [
            (0.05, 0.95, 0.20) if count > 0 else (1.0, 0.05, 0.02)
            for count in counts
        ]
        try:
            self._draw_camera_viewpoint_points(points_local, colors=colors, opacity=0.95)
        except Exception as e:
            self._append_log(f"[ik] display failed: {e}")
            return

        reachable_count = int((counts > 0).sum())
        total = len(counts)
        self._append_log(
            f"[ik] reachability displayed under {VIEWPOINTS_ROOT_PRIM}: "
            f"{reachable_count}/{total} reachable "
            f"({100.0 * reachable_count / max(total, 1):.1f}%), "
            f"working distance={wd_m * 1000:.1f} mm"
        )

    def _on_check_ik_reachability(self):
        if self._ik_runner.running:
            self._append_log("[ik] check already running")
            return

        h5 = self._h5_path_model.get_value_as_string().strip()
        if not h5:
            self._append_log("[ik] pick a viewpoints .h5 first (Browse...).")
            return
        if not Path(h5).exists():
            self._append_log(f"[ik] h5 not found: {h5}")
            return

        obj, n_vp = self._parse_h5_meta(h5)
        if obj is None:
            obj = (self._current_object or "").strip()
            if not obj:
                self._append_log(
                    "[ik] couldn't read object from h5 path and no object is loaded."
                )
                return
            self._append_log(f"[ik] couldn't read object from h5 path; using '{obj}'.")
        if self._current_object and obj != self._current_object:
            self._append_log(
                f"[ik] WARNING: h5 object '{obj}' != loaded scene object "
                f"'{self._current_object}'. IK/collision mesh uses '{obj}'."
            )

        pose = self._read_object_world_pose()
        if pose is None:
            self._append_log(
                "[ik] no target object on stage - pick one in the Object dropdown "
                "and click 'Load Object' first."
            )
            return
        pos_robot, quat_wxyz = pose

        result_path = Path("/tmp") / (
            f"isaac_pipeline_ik_{os.getpid()}_{int(time.time() * 1000)}.json"
        )
        # 색칠용 JSON 은 임시. IK 후보는 data/{object}/ik/{N}/ 에 옵션(roll/tilt/dedup)별
        # 파일로 저장되고(check_ik 가 경로를 정한다), Generate 가 같은 옵션·물체 pose 면 재사용한다.
        cmd = [
            self._uv, "run", "scripts/core/trajectory/check_ik.py",
            "--object", obj,
            "--viewpoints", h5,
            "--output", str(result_path),
        ]
        if n_vp is not None:
            cmd += ["--num-viewpoints", str(n_vp)]
        cmd += self._ik_candidate_tokens()
        cmd += ["--batch-size", str(self._ik_batch())]
        cmd += ["--scene", self._scene]
        cmd += ["--object-position", *(f"{v:.6f}" for v in pos_robot)]
        cmd += ["--object-quat", *(f"{v:.6f}" for v in quat_wxyz)]

        self._set_busy(self._btn_cancel_ik)
        self._append_log("[ik] $ " + " ".join(cmd))

        def on_line(line: str):
            self._append_log(line)

        def on_exit(rc: int):
            cancelled = self._ik_runner.cancelled
            self._append_log("[ik] cancelled" if cancelled
                             else f"[ik] exit code = {rc}")
            self._clear_busy()
            if rc == 0 and not cancelled:
                self._apply_ik_reachability_result(h5, result_path)

        self._ik_runner.start(cmd, cwd=PROJECT_ROOT, on_line=on_line, on_exit=on_exit)

    def _on_cancel_ik(self):
        if self._ik_runner.running:
            self._append_log("[ik] terminating...")
            self._ik_runner.terminate()

    @staticmethod
    def _find_prim_by_name(stage, robot_root: str, prim_name: str):
        root = stage.GetPrimAtPath(robot_root)
        if not root or not root.IsValid():
            return None
        for prim in stage.Traverse():
            p = str(prim.GetPath())
            if f"/{COLLISION_SPHERES_SCOPE_NAME}/" in p:
                continue
            if f"/{FOV_PLANE_SCOPE_NAME}/" in p:
                continue
            # `+ "/"` so root="/World/UR20" does not also match the sibling
            # ghost "/World/UR20_preview".
            if (p == robot_root or p.startswith(robot_root + "/")) and prim.GetName() == prim_name:
                return prim
        return None

    def _make_cam_target(self, key, camera_name, root_path):
        """Per-camera state incl. its OWN spec models (UI 입력값은 카메라별 분리; 기본값만
        공유 default 에서 시작). 편집하면 이 카메라만 갱신한다(_on_camera_spec_changed(key))."""
        ui = self._ui
        d = self._cam_spec_default
        t = {
            "key": key,
            "camera": camera_name,        # InspectionCamera / InspectionCameraPreview
            "root": root_path,            # robot root the camera lives under
            "fov_w": ui.SimpleFloatModel(float(d["fov_w"])),
            "fov_h": ui.SimpleFloatModel(float(d["fov_h"])),
            "wd": ui.SimpleFloatModel(float(d["wd"])),
            "fov_on": False,
            "range_on": False,
            "btn_fov": None,
            "btn_range": None,
            "cam_prim_path": None,        # cached camera prim path
            "updating": False,            # suppress apply/redraw on batch set
        }
        for fld in ("fov_w", "fov_h", "wd"):
            t[fld].add_value_changed_fn(lambda *_a, _k=key: self._on_camera_spec_changed(_k))
        return t

    def _build_camera_view_ui(self, key):
        """카메라별 스펙 입력 + 시각화 토글. 입력값은 이 카메라만의 모델(분리, 테스트용),
        Reset 은 공유 default(로드된 뷰포인트 값)로 되돌린다."""
        ui = self._ui
        t = self._cam_targets[key]
        with ui.HStack(height=22, spacing=6):
            ui.Label("FOV W", width=44)
            self._lock(ui.FloatField(model=t["fov_w"], width=60))
            ui.Label("FOV H", width=44)
            self._lock(ui.FloatField(model=t["fov_h"], width=60))
            ui.Label("WD", width=24)
            self._lock(ui.FloatField(model=t["wd"], width=60))
            self._lock(ui.Button(
                "Reset", width=64, clicked_fn=lambda k=key: self._on_reset_camera_spec(k)))
        with ui.HStack(height=28, spacing=6):
            t["btn_fov"] = self._lock(ui.Button(
                "Show FOV", clicked_fn=lambda k=key: self._on_toggle_fov(k)))
            t["btn_range"] = self._lock(ui.Button(
                "Show Camera Range", clicked_fn=lambda k=key: self._on_toggle_range(k)))

    def _find_camera_prim(self, stage, key):
        """The UsdGeom.Camera prim for this target (cached), or None."""
        from pxr import UsdGeom
        t = self._cam_targets[key]
        p = t.get("cam_prim_path")
        if p:
            prim = stage.GetPrimAtPath(p)
            if prim and prim.IsValid() and prim.IsA(UsdGeom.Camera):
                return prim
        frame = self._find_prim_by_name(stage, t["root"], urctl.CAMERA_OPTICAL_FRAME_NAME)
        if frame is None or not frame.IsValid():
            return None
        cam_path = f"{frame.GetPath()}/{t['camera']}"
        prim = stage.GetPrimAtPath(cam_path)
        if prim and prim.IsValid() and prim.IsA(UsdGeom.Camera):
            t["cam_prim_path"] = cam_path
            return prim
        return None

    @staticmethod
    def _lock_camera_prim(prim):
        """Prevent viewport (mouse) navigation from moving this camera — control
        is UI-only. Omniverse honours the bool attr omni:kit:cameraLock."""
        from pxr import Sdf
        try:
            a = prim.GetAttribute("omni:kit:cameraLock")
            if not a or not a.IsValid():
                a = prim.CreateAttribute("omni:kit:cameraLock", Sdf.ValueTypeNames.Bool)
            if a.Get() is not True:
                a.Set(True)
        except Exception:  # noqa: BLE001
            pass

    def _lock_camera_prims(self):
        """Keep both inspection cameras mouse-locked every frame (idempotent)."""
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        for key in self._cam_targets:
            prim = self._find_camera_prim(stage, key)
            if prim is not None:
                self._lock_camera_prim(prim)

    @staticmethod
    def _get_viewport_windows():
        """All viewport windows, trying both module homes across Kit versions."""
        import importlib
        for mod in ("omni.kit.viewport.window", "omni.kit.viewport.utility"):
            try:
                fn = getattr(importlib.import_module(mod),
                             "get_viewport_window_instances", None)
                if fn is not None:
                    wins = list(fn())
                    if wins:
                        return wins
            except Exception:  # noqa: BLE001
                continue
        return []

    def _lock_inspection_viewports(self, verbose=False):
        """Keep every viewport showing InspectionCamera / InspectionCameraPreview
        at a 1:1 render aspect, so the square FOV is shown in full (letterboxed)
        rather than cropped by a rectangular window — matching the square range
        rays. Runs every frame; a no-op once square, re-squares on resize."""
        wins = self._get_viewport_windows()
        if verbose and not wins:
            self._append_log("[cam] no viewport windows found (API unavailable)")
        for win in wins:
            vp = getattr(win, "viewport_api", None)
            if vp is None:
                continue
            cp = getattr(vp, "camera_path", None)
            res = getattr(vp, "resolution", None)
            cp_s = str(cp) if cp is not None else ""
            is_insp = (cp_s.endswith("/InspectionCamera")
                       or cp_s.endswith("/InspectionCameraPreview"))
            if verbose:
                self._append_log(f"[cam] viewport cam={cp_s or '?'} res={res} insp={is_insp}")
            if not is_insp or not res or not res[0] or not res[1]:
                continue
            w, h = int(res[0]), int(res[1])
            if w != h:
                n = max(w, h)
                try:
                    vp.resolution = (n, n)
                    if verbose:
                        self._append_log(f"[cam] locked {cp_s} -> {n}x{n}")
                except Exception as e:  # noqa: BLE001
                    if verbose:
                        self._append_log(f"[cam] set resolution failed: {e}")

    def _delete_fov_plane(self, key, log):
        import omni.usd
        from isaacsim.core.utils import prims

        root = self._cam_targets[key]["root"]
        stage = omni.usd.get_context().get_stage()
        paths = [
            str(prim.GetPath())
            for prim in stage.Traverse()
            if prim.GetName() == FOV_PLANE_SCOPE_NAME
            and str(prim.GetPath()).startswith(root + "/")
        ]
        for path in sorted(paths, key=len, reverse=True):
            prims.delete_prim(path)
        if log:
            self._append_log(
                f"[fov:{key}] cleared {len(paths)} plane(s)"
                if paths else f"[fov:{key}] nothing to clear"
            )

    def _camera_spec_m(self, key):
        """This target's spec (fov_w, fov_h, wd) in meters, or None if invalid."""
        t = self._cam_targets[key]
        fov_w_m = float(t["fov_w"].get_value_as_float()) / 1000.0
        fov_h_m = float(t["fov_h"].get_value_as_float()) / 1000.0
        wd_m = float(t["wd"].get_value_as_float()) / 1000.0
        if fov_w_m <= 0.0 or fov_h_m <= 0.0 or wd_m <= 0.0:
            self._append_log(
                f"[cam:{key}] invalid spec: FOV={fov_w_m * 1000:.1f}x"
                f"{fov_h_m * 1000:.1f} mm, WD={wd_m * 1000:.1f} mm"
            )
            return None
        return fov_w_m, fov_h_m, wd_m

    def _apply_camera_spec_to_camera(self, key):
        """Push this target's spec to its camera so the VIEW matches the UI:
        focalLength=WD, horizontal/verticalAperture=FOV (footprint model —
        see scene.setup_inspection_camera). Silent (called on every edit)."""
        import omni.usd
        from pxr import Gf, UsdGeom

        from common import config as _config

        t = self._cam_targets[key]
        fov_w_mm = float(t["fov_w"].get_value_as_float())
        fov_h_mm = float(t["fov_h"].get_value_as_float())
        wd_mm = float(t["wd"].get_value_as_float())
        if fov_w_mm <= 0.0 or fov_h_mm <= 0.0 or wd_mm <= 0.0:
            return
        stage = omni.usd.get_context().get_stage()
        prim = self._find_camera_prim(stage, key)
        if prim is None:
            return
        cam = UsdGeom.Camera(prim)
        cam.GetFocalLengthAttr().Set(wd_mm)
        cam.GetHorizontalApertureAttr().Set(fov_w_mm)
        cam.GetVerticalApertureAttr().Set(fov_h_mm)
        cam.GetFocusDistanceAttr().Set(wd_mm * 1e-3)
        # 스펙을 다시 밀 때 clipping 도 같이 — 손으로 만졌거나 옛 스테이지에서 온 카메라의
        # near(0.01) 가 남아 있으면 화면이 자기 렌즈 배럴로 가득 찬다.
        cam.GetClippingRangeAttr().Set(
            Gf.Vec2f(float(_config.CAMERA_NEAR_CLIP_M), float(_config.CAMERA_FAR_CLIP_M)))

    def _tick_camera_ranges(self, dt):
        """Throttled per-frame re-cast so the range rays follow the camera as the
        robot moves. Uses the cached object mesh (Load Object / re-toggle to
        refresh it after moving the object)."""
        if not any(t["range_on"] for t in self._cam_targets.values()):
            self._range_accum = 0.0
            return
        self._range_accum += dt
        if self._range_accum < CAMERA_RANGE_UPDATE_DT:
            return
        self._range_accum = 0.0
        for key, t in self._cam_targets.items():
            if t["range_on"]:
                self._draw_camera_range_rays(key)

    def _draw_fov_rectangle(self, key) -> bool:
        """Draw the FOV as a rectangle at the working distance under this target's
        optical frame (footprint), from its editable spec. Returns True on draw."""
        import omni.usd
        from pxr import Gf, UsdGeom, Vt

        stage = omni.usd.get_context().get_stage()
        robot_root = self._cam_targets[key]["root"]
        if not stage.GetPrimAtPath(robot_root).IsValid():
            self._append_log(f"[fov:{key}] robot not found: {robot_root}")
            return False
        camera_frame = self._find_prim_by_name(
            stage, robot_root, urctl.CAMERA_OPTICAL_FRAME_NAME,
        )
        if camera_frame is None or not camera_frame.IsValid():
            self._append_log(
                f"[fov:{key}] {urctl.CAMERA_OPTICAL_FRAME_NAME} not found under {robot_root}"
            )
            return False
        spec = self._camera_spec_m(key)
        if spec is None:
            return False
        fov_w_m, fov_h_m, wd_m = spec

        self._delete_fov_plane(key, log=False)
        half_w = fov_w_m * 0.5
        half_h = fov_h_m * 0.5
        corners = [
            Gf.Vec3f(-half_w, -half_h, wd_m),
            Gf.Vec3f( half_w, -half_h, wd_m),
            Gf.Vec3f( half_w,  half_h, wd_m),
            Gf.Vec3f(-half_w,  half_h, wd_m),
        ]
        scope_path = f"{camera_frame.GetPath()}/{FOV_PLANE_SCOPE_NAME}"
        UsdGeom.Xform.Define(stage, scope_path)

        plane = UsdGeom.Mesh.Define(stage, f"{scope_path}/Plane")
        plane.CreatePointsAttr(Vt.Vec3fArray(corners))
        plane.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
        plane.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
        plane.CreateDoubleSidedAttr(True)
        plane.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(1.0, 0.78, 0.05)]))
        plane.CreateDisplayOpacityAttr(Vt.FloatArray([0.22]))

        outline_points = corners + [corners[0]]
        outline = UsdGeom.BasisCurves.Define(stage, f"{scope_path}/Outline")
        outline.CreateTypeAttr(UsdGeom.Tokens.linear)
        outline.CreateCurveVertexCountsAttr(Vt.IntArray([len(outline_points)]))
        outline.CreatePointsAttr(Vt.Vec3fArray(outline_points))
        outline.CreateWidthsAttr(Vt.FloatArray([FOV_PLANE_OUTLINE_WIDTH_M]))
        outline.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        outline.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(1.0, 0.38, 0.0)]))
        outline.CreateDisplayOpacityAttr(Vt.FloatArray([0.95]))

        center_line = UsdGeom.BasisCurves.Define(stage, f"{scope_path}/WorkingDistance")
        center_line.CreateTypeAttr(UsdGeom.Tokens.linear)
        center_line.CreateCurveVertexCountsAttr(Vt.IntArray([2]))
        center_line.CreatePointsAttr(Vt.Vec3fArray([
            Gf.Vec3f(0.0, 0.0, 0.0),
            Gf.Vec3f(0.0, 0.0, wd_m),
        ]))
        center_line.CreateWidthsAttr(Vt.FloatArray([FOV_PLANE_CENTERLINE_WIDTH_M]))
        center_line.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        center_line.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.0, 0.85, 1.0)]))
        center_line.CreateDisplayOpacityAttr(Vt.FloatArray([0.95]))

        self._append_log(
            f"[fov:{key}] rectangle {fov_w_m * 1000:.1f}x{fov_h_m * 1000:.1f} mm "
            f"@ WD={wd_m * 1000:.1f} mm under {camera_frame.GetPath()}"
        )
        return True

    def _on_toggle_fov(self, key):
        """Toggle this target's FOV rectangle on/off."""
        t = self._cam_targets[key]
        if t["fov_on"]:
            self._delete_fov_plane(key, log=True)
            t["fov_on"] = False
            if t["btn_fov"] is not None:
                t["btn_fov"].text = "Show FOV"
        elif self._draw_fov_rectangle(key):
            t["fov_on"] = True
            if t["btn_fov"] is not None:
                t["btn_fov"].text = "Hide FOV"

    def _object_world_trimesh(self, stage):
        """Build a trimesh of /World/target_object in WORLD coordinates (every
        UsdGeom.Mesh under it, polygons triangulated). None if no geometry."""
        import numpy as np
        import trimesh
        from pxr import Usd, UsdGeom

        root = stage.GetPrimAtPath(TARGET_OBJECT_PRIM)
        if not root or not root.IsValid():
            return None
        verts_all, faces_all, offset = [], [], 0
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            pts = mesh.GetPointsAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            idx = mesh.GetFaceVertexIndicesAttr().Get()
            if not pts or not counts or not idx:
                continue
            M = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            wp = np.array([list(M.Transform(p)) for p in pts], dtype=np.float64)
            tris, k = [], 0
            for c in counts:
                for t in range(1, c - 1):
                    tris.append((idx[k], idx[k + t], idx[k + t + 1]))
                k += c
            if not tris:
                continue
            faces_all.append(np.array(tris, dtype=np.int64) + offset)
            verts_all.append(wp)
            offset += len(wp)
        if not verts_all:
            return None
        return trimesh.Trimesh(
            vertices=np.vstack(verts_all), faces=np.vstack(faces_all), process=False,
        )

    def _delete_camera_range(self, key, log):
        import omni.usd
        from isaacsim.core.utils import prims

        scope = f"/World/{CAMERA_RANGE_SCOPE_NAME}_{key}"
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath(scope).IsValid():
            prims.delete_prim(scope)
            if log:
                self._append_log(f"[range:{key}] cleared")
        elif log:
            self._append_log(f"[range:{key}] nothing to clear")

    def _draw_camera_range_rays(self, key, verbose=False) -> bool:
        """Cast an N×N grid of rays across this camera's FOV from the optical
        origin and draw each one to where it hits /World/target_object (ray-mesh
        intersection), in world space under an identity /World scope. Rebuilt in
        place by the per-frame tick so the rays follow the camera. `verbose`
        logs (toggle only) — the tick stays silent."""
        import numpy as np
        import omni.usd
        from pxr import Gf, Usd, UsdGeom, Vt

        stage = omni.usd.get_context().get_stage()
        robot_root = self._cam_targets[key]["root"]
        # Gate on robot visibility so the rays behave like the FOV rectangle
        # (a child of the frame): the preview ghost is hidden until Load & Preview,
        # so its range must not show either. The real robot is always visible.
        root_prim = stage.GetPrimAtPath(robot_root)
        if (not root_prim or not root_prim.IsValid()
                or UsdGeom.Imageable(root_prim).ComputeVisibility(Usd.TimeCode.Default())
                == UsdGeom.Tokens.invisible):
            self._delete_camera_range(key, log=False)
            if verbose:
                self._append_log(f"[range:{key}] robot hidden - Load & Preview first.")
            return False
        camera_frame = self._find_prim_by_name(
            stage, robot_root, urctl.CAMERA_OPTICAL_FRAME_NAME,
        )
        if camera_frame is None or not camera_frame.IsValid():
            if verbose:
                self._append_log(
                    f"[range:{key}] {urctl.CAMERA_OPTICAL_FRAME_NAME} not found under {robot_root}"
                )
            return False
        spec = self._camera_spec_m(key)
        if spec is None:
            return False
        fov_w_m, fov_h_m, wd_m = spec

        if self._range_trimesh is None:
            self._range_trimesh = self._object_world_trimesh(stage)
        tm = self._range_trimesh
        if tm is None:
            if verbose:
                self._append_log("[range] no target object mesh - Load Object first.")
            return False

        M = UsdGeom.Xformable(camera_frame).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        origin = np.array(list(M.Transform(Gf.Vec3d(0.0, 0.0, 0.0))), dtype=np.float64)
        # Match the ray grid to the aspect of the viewport actually showing this
        # camera: a non-square window conforms the square aperture and crops the
        # top/bottom (or sides), so the full square would put rays outside the
        # real view. When no such viewport exists, use the full aperture.
        # Full square FOV. The inspection viewports are locked to a 1:1 render
        # aspect (_lock_inspection_viewports), so the camera view shows the whole
        # square (letterboxed) and matches these rays instead of cropping it.
        gx = np.linspace(-fov_w_m * 0.5, fov_w_m * 0.5, CAMERA_RANGE_GRID)
        gy = np.linspace(-fov_h_m * 0.5, fov_h_m * 0.5, CAMERA_RANGE_GRID)
        dirs = []
        for y in gy:
            for x in gx:
                d_world = M.TransformDir(Gf.Vec3d(float(x), float(y), float(wd_m)))
                v = np.array([d_world[0], d_world[1], d_world[2]], dtype=np.float64)
                nrm = np.linalg.norm(v)
                if nrm > 1e-12:
                    dirs.append(v / nrm)
        dirs = np.array(dirs, dtype=np.float64)

        try:
            locs, ray_idx, _tri = tm.ray.intersects_location(
                np.tile(origin, (len(dirs), 1)), dirs, multiple_hits=False,
            )
        except Exception as e:  # noqa: BLE001 — raycast backend / geometry issue
            if verbose:
                self._append_log(f"[range:{key}] raycast failed: {e}")
            return False

        origin_f = Gf.Vec3f(float(origin[0]), float(origin[1]), float(origin[2]))
        seg_points, seg_counts, hit_pts = [], [], []
        by_ray = {int(r): locs[i] for i, r in enumerate(ray_idx)}
        for i in range(len(dirs)):
            p = by_ray.get(i)
            if p is None:
                continue
            hp = Gf.Vec3f(float(p[0]), float(p[1]), float(p[2]))
            seg_points.extend([origin_f, hp])
            seg_counts.append(2)
            hit_pts.append(hp)

        # No delete: update the prims in place each tick so they follow the
        # camera. If nothing hits right now, clear (leaving the toggle on).
        if not seg_points:
            self._delete_camera_range(key, log=False)
            if verbose:
                self._append_log(f"[range:{key}] no rays hit the object (check pose / FOV).")
            return False

        # Identity /World scope: the ray endpoints are already world-space, so
        # they must NOT sit under a transformed parent (that double-transforms).
        scope_path = f"/World/{CAMERA_RANGE_SCOPE_NAME}_{key}"
        UsdGeom.Xform.Define(stage, scope_path)
        rays = UsdGeom.BasisCurves.Define(stage, f"{scope_path}/Rays")
        rays.CreateTypeAttr(UsdGeom.Tokens.linear)
        rays.CreateCurveVertexCountsAttr(Vt.IntArray(seg_counts))
        rays.CreatePointsAttr(Vt.Vec3fArray(seg_points))
        rays.CreateWidthsAttr(Vt.FloatArray([CAMERA_RANGE_RAY_WIDTH_M]))
        rays.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        rays.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.1, 1.0, 0.35)]))
        rays.CreateDisplayOpacityAttr(Vt.FloatArray([0.9]))

        hits = UsdGeom.Points.Define(stage, f"{scope_path}/Hits")
        hits.CreatePointsAttr(Vt.Vec3fArray(hit_pts))
        hits.CreateWidthsAttr(Vt.FloatArray([CAMERA_RANGE_HIT_WIDTH_M] * len(hit_pts)))
        hits.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(1.0, 0.2, 0.2)]))

        if verbose:
            self._append_log(
                f"[range:{key}] {len(hit_pts)}/{len(dirs)} rays hit object "
                f"(FOV {fov_w_m * 1000:.1f}x{fov_h_m * 1000:.1f} mm @ WD={wd_m * 1000:.1f} mm)"
            )
        return True

    def _on_toggle_range(self, key):
        """Toggle this target's camera-range rays. While on, the per-frame tick
        re-casts so the rays follow the camera as the robot moves."""
        t = self._cam_targets[key]
        if t["range_on"]:
            t["range_on"] = False
            self._delete_camera_range(key, log=True)
            if t["btn_range"] is not None:
                t["btn_range"].text = "Show Camera Range"
        else:
            t["range_on"] = True
            self._range_trimesh = None  # re-extract object geometry fresh
            self._lock_inspection_viewports(verbose=True)  # diag: what viewports/res
            self._draw_camera_range_rays(key, verbose=True)
            if t["btn_range"] is not None:
                t["btn_range"].text = "Hide Camera Range"

    def _on_camera_spec_changed(self, key):
        """이 카메라 스펙이 바뀌면 그 카메라의 intrinsic + 켜진 시각화만 갱신한다.
        UI 입력값은 카메라별 분리라 다른 카메라는 안 건드린다."""
        t = self._cam_targets[key]
        if t["updating"]:
            return
        self._apply_camera_spec_to_camera(key)
        if t["fov_on"]:
            self._draw_fov_rectangle(key)
        if t["range_on"]:
            self._draw_camera_range_rays(key)
        if key == "execute":
            self._apply_render_resolution()

    def _apply_render_resolution(self):
        """ROS 렌더 프로덕트 해상도를 실 카메라(execute)의 FOV 종횡비에 맞춘다.

        USD 카메라는 세로 화각을 렌더 해상도 비율에서 다시 계산하므로(verticalAperture 는
        사실상 무시), 해상도 비율이 FOV 비율과 다르면 퍼블리시 이미지가 FOV_H 를 덮지 않는다.
        렌더 프로덕트는 실 카메라(InspectionCamera) 것이므로 execute 스펙을 쓴다.
        """
        from common import config as _config

        t = self._cam_targets["execute"]
        fov_w_mm = float(t["fov_w"].get_value_as_float())
        fov_h_mm = float(t["fov_h"].get_value_as_float())
        if fov_w_mm <= 0.0 or fov_h_mm <= 0.0:
            return
        wh = _config.publish_resolution(fov_w_mm, fov_h_mm)
        if wh == self._render_resolution:
            return                      # 스펙 편집마다 그래프를 건드리지 않는다
        try:
            urctl.set_render_resolution(self._real_graph_path, wh[0], wh[1])
        except Exception as e:  # noqa: BLE001 — RP 노드가 없을 수 있다(카메라 없이 그래프 생성)
            self._append_log(f"[cam] render product resize skipped: {e}")
            return
        self._render_resolution = wh
        self._append_log(
            f"[cam] render product -> {wh[0]}x{wh[1]} "
            f"(FOV {fov_w_mm:.0f}x{fov_h_mm:.0f} mm)")

    def _set_camera_spec_mm(self, key, fov_w_mm, fov_h_mm, wd_mm):
        """한 카메라의 스펙 세 필드를 한 번에 설정(적용/재그리기 한 번)."""
        t = self._cam_targets[key]
        t["updating"] = True
        try:
            t["fov_w"].set_value(float(fov_w_mm))
            t["fov_h"].set_value(float(fov_h_mm))
            t["wd"].set_value(float(wd_mm))
        finally:
            t["updating"] = False
        self._apply_camera_spec_to_camera(key)
        if t["fov_on"]:
            self._draw_fov_rectangle(key)
        if t["range_on"]:
            self._draw_camera_range_rays(key)
        if key == "execute":
            self._apply_render_resolution()

    def _on_reset_camera_spec(self, key):
        """이 카메라 스펙을 공유 default(로드된 뷰포인트를 만들 때 쓴 값)로 되돌린다."""
        d = self._cam_spec_default
        self._set_camera_spec_mm(key, d["fov_w"], d["fov_h"], d["wd"])
        self._append_log(
            f"[cam:{key}] reset to viewpoint default "
            f"(FOV {d['fov_w']:.0f}x{d['fov_h']:.0f} mm, WD {d['wd']:.0f} mm)")

    def _sync_camera_spec_from_h5(self, h5_path: str):
        """공유 default 를 viewpoint h5 스냅샷으로 갱신하고 두 카메라를 그 값으로 맞춘다.
        (이후 사용자가 카메라별로 다르게 편집 가능.) Best-effort."""
        try:
            from core.viewpoint.storage import load_viewpoints_hdf5
            vp = load_viewpoints_hdf5(h5_path)
        except Exception as e:  # noqa: BLE001 — snapshot is best-effort
            self._append_log(f"[cam] could not read camera spec from h5: {e}")
            return
        self._cam_spec_default = {
            "fov_w": float(vp.fov_width_mm),
            "fov_h": float(vp.fov_height_mm),
            "wd": float(vp.working_distance_mm),
        }
        for key in self._cam_targets:
            self._set_camera_spec_mm(
                key, self._cam_spec_default["fov_w"],
                self._cam_spec_default["fov_h"], self._cam_spec_default["wd"])
        self._append_log(
            f"[cam] default <- snapshot {vp.fov_width_mm:.1f}x"
            f"{vp.fov_height_mm:.1f} mm @ WD={vp.working_distance_mm:.1f} mm (both cameras)"
        )

    @staticmethod
    def _load_collision_spheres():
        import yaml
        from common import config as _config

        robot_cfg_path = (
            _config.PROJECT_ROOT
            / "workcell"
            / "robot"
            / _config.DEFAULT_ROBOT_CONFIG
        )
        with open(robot_cfg_path) as f:
            cfg = yaml.safe_load(f)
        kin = cfg["robot_cfg"]["kinematics"]
        urdf_path = (
            _config.PROJECT_ROOT
            / "workcell"
            / "robot"
            / Path(kin["urdf_path"]).name
        )
        sphere_buffer = kin.get("collision_sphere_buffer", 0.0)
        if isinstance(sphere_buffer, dict):
            buffer_by_link = {
                link_name: float(value)
                for link_name, value in sphere_buffer.items()
            }
        else:
            buffer_by_link = {
                link_name: float(sphere_buffer or 0.0)
                for link_name in kin["collision_spheres"]
            }
        collision_spheres = {
            link_name: [
                {
                    **sphere_cfg,
                    "radius": float(sphere_cfg["radius"])
                    + float(buffer_by_link.get(link_name, 0.0)),
                }
                for sphere_cfg in link_spheres
            ]
            for link_name, link_spheres in kin["collision_spheres"].items()
        }
        max_sphere_buffer = max(buffer_by_link.values(), default=0.0)
        return (
            robot_cfg_path,
            urdf_path,
            kin["collision_link_names"],
            collision_spheres,
            max_sphere_buffer,
        )

    @staticmethod
    def _find_link_prim(stage, robot_root: str, link_name: str):
        return PipelineWindow._find_prim_by_name(stage, robot_root, link_name)

    @staticmethod
    def _rpy_xyz_to_np(rpy, xyz) -> np.ndarray:
        roll, pitch, yaw = (float(v) for v in rpy)
        x, y, z = (float(v) for v in xyz)
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
        T = np.eye(4)
        T[:3, :3] = rz @ ry @ rx
        T[:3, 3] = [x, y, z]
        return T

    @classmethod
    def _load_fixed_urdf_edges(cls, urdf_path: Path):
        import xml.etree.ElementTree as ET

        root = ET.parse(urdf_path).getroot()
        edges = {}
        for joint in root.findall("joint"):
            if joint.get("type") != "fixed":
                continue
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                continue
            origin = joint.find("origin")
            xyz = (origin.get("xyz", "0 0 0").split() if origin is not None else ["0", "0", "0"])
            rpy = (origin.get("rpy", "0 0 0").split() if origin is not None else ["0", "0", "0"])
            edges[child.get("link")] = (
                parent.get("link"),
                cls._rpy_xyz_to_np(rpy, xyz),
            )
        return edges

    def _link_prim_or_fixed_frame(self, stage, robot_root: str, link_name: str,
                                  fixed_edges: dict):
        from pxr import UsdGeom

        link_prim = self._find_link_prim(stage, robot_root, link_name)
        if link_prim is not None and link_prim.IsValid():
            return link_prim

        chain = []
        current = link_name
        anchor_prim = None
        while current in fixed_edges:
            parent, T_parent_current = fixed_edges[current]
            chain.append((current, T_parent_current))
            anchor_prim = self._find_link_prim(stage, robot_root, parent)
            if anchor_prim is not None and anchor_prim.IsValid():
                break
            current = parent
        if anchor_prim is None or not anchor_prim.IsValid():
            return None

        T_anchor_link = np.eye(4)
        for _, T_parent_child in reversed(chain):
            T_anchor_link = T_anchor_link @ T_parent_child

        frame_root = f"{anchor_prim.GetPath()}/{COLLISION_SPHERES_SCOPE_NAME}/frames"
        UsdGeom.Xform.Define(stage, frame_root)
        frame = UsdGeom.Xform.Define(stage, f"{frame_root}/{link_name}")
        xf = UsdGeom.Xformable(frame)
        xf.ClearXformOpOrder()
        xf.AddTransformOp(opSuffix="urdfFixedFrame").Set(_np_to_gf(T_anchor_link))
        return frame.GetPrim()

    def _delete_collision_spheres(self, log: bool):
        import omni.usd
        from isaacsim.core.utils import prims

        stage = omni.usd.get_context().get_stage()
        paths = [
            str(prim.GetPath())
            for prim in stage.Traverse()
            if prim.GetName() == COLLISION_SPHERES_SCOPE_NAME
        ]
        for path in sorted(paths, key=len, reverse=True):
            prims.delete_prim(path)
        if log:
            self._append_log(
                f"[spheres] cleared {len(paths)} collision sphere scope(s)"
                if paths else "[spheres] nothing to clear"
            )

    def _on_show_collision_spheres(self):
        import omni.usd
        from pxr import Gf, UsdGeom, Vt

        stage = omni.usd.get_context().get_stage()
        robot_root = GHOST_ROOT_PATH
        if not stage.GetPrimAtPath(robot_root).IsValid():
            self._append_log(f"[spheres] ghost robot not found: {robot_root}")
            return

        try:
            cfg_path, urdf_path, collision_link_names, collision_spheres, sphere_buffer = (
                self._load_collision_spheres()
            )
            fixed_edges = self._load_fixed_urdf_edges(urdf_path)
        except Exception as e:
            self._append_log(f"[spheres] load failed: {e}")
            return

        self._delete_collision_spheres(log=False)

        n_spheres = 0
        missing_links = []
        for link_name in collision_link_names:
            link_prim = self._link_prim_or_fixed_frame(
                stage, robot_root, link_name, fixed_edges,
            )
            if link_prim is None or not link_prim.IsValid():
                missing_links.append(link_name)
                continue

            scope_path = f"{link_prim.GetPath()}/{COLLISION_SPHERES_SCOPE_NAME}"
            UsdGeom.Xform.Define(stage, scope_path)
            is_camera = link_name in CAMERA_COLLISION_LINKS
            color = Gf.Vec3f(1.0, 0.42, 0.08) if is_camera else Gf.Vec3f(0.2, 1.0, 0.35)
            opacity = 0.38 if is_camera else 0.22

            for i, sphere_cfg in enumerate(collision_spheres[link_name]):
                center = sphere_cfg["center"]
                radius = float(sphere_cfg["radius"])
                sphere = UsdGeom.Sphere.Define(stage, f"{scope_path}/s_{i:03d}")
                sphere.CreateRadiusAttr(radius)
                sphere.CreateDisplayColorAttr(Vt.Vec3fArray([color]))
                sphere.CreateDisplayOpacityAttr(Vt.FloatArray([opacity]))
                xf = UsdGeom.Xformable(sphere)
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(Gf.Vec3d(
                    float(center[0]), float(center[1]), float(center[2])
                ))
                n_spheres += 1

        msg = (
            f"[spheres] displayed {n_spheres} cuRobo collision spheres from "
            f"{cfg_path.name} under {robot_root}"
        )
        if sphere_buffer > 0.0:
            msg += f" (+{sphere_buffer * 1000:.1f} mm YAML buffer)"
        if missing_links:
            msg += f" (missing links: {', '.join(missing_links)})"
        self._append_log(msg)

    def _on_clear_collision_spheres(self):
        self._delete_collision_spheres(log=True)

    def _on_generate(self):
        if self._gen_runner.running:
            self._append_log("[generate] already running")
            return

        # Single input: the viewpoints .h5. Object name + viewpoint count come
        # from its standard path (data/{object}/viewpoint/{N}/...); the object's
        # live pose comes from the scene gizmo.
        h5 = self._h5_path_model.get_value_as_string().strip()
        if not h5:
            self._append_log("[generate] pick a viewpoints .h5 first (Browse...).")
            return
        if not Path(h5).exists():
            self._append_log(f"[generate] h5 not found: {h5}")
            return

        obj, n_vp = self._parse_h5_meta(h5)
        if obj is None:
            obj = (self._current_object or "").strip()
            self._append_log(
                f"[generate] couldn't read object from h5 path; using loaded object '{obj}'.")
        if n_vp is None:
            n_vp = 124
            self._append_log(
                f"[generate] couldn't read viewpoint count from h5 path; defaulting "
                f"--num-viewpoints {n_vp} (affects output dir only).")
        if self._current_object and obj and obj != self._current_object:
            self._append_log(
                f"[generate] WARNING: h5 object '{obj}' != loaded scene object "
                f"'{self._current_object}'. Pose & collision mesh come from the SCENE "
                "object - load the matching object or pick the matching h5.")

        spacing = 0.01

        # Read the object's live world pose (gizmo-moved) and pass it to the
        # planner. No silent fallback: if there's no target prim, abort so we
        # never plan against a stale config pose.
        pose = self._read_object_world_pose()
        if pose is None:
            self._append_log(
                "[generate] no target object on stage - pick one in the Object "
                "dropdown and click 'Load Object' first.")
            return
        pos_robot, quat_wxyz = pose

        # GLNS solve → collision-aware verification/join. Both stages stream
        # stdout; verify prints the joined CSV path last for preview/publish capture.
        hops = max(1, int(self._get_field("glns_hops", int)))
        max_candidates = max(1, int(self._get_field("glns_max_candidates", int)))
        # 공유 IK 후보 옵션(Check and Save IK 와 동일) — 같으면 저장 IK 를 재사용한다.
        augment = (f" {' '.join(self._ik_candidate_tokens())}"
                   f" --ik-batch-size {self._ik_batch()}"
                   f" --max-candidates-per-viewpoint {max_candidates}")
        from common import config as _config

        # 해와 궤적은 한 폴더에 산다. 이름에 앱 이름을 넣지 않는다 — 어느 앱이 만들었든
        # 같은 물체/viewpoint 수면 같은 해다(재solve 는 덮어쓰기).
        det_h5 = str(_config.get_solution_path(obj, n_vp))
        trajectory_dir = str(Path(det_h5).parent)
        pos_s = " ".join(f"{v:.6f}" for v in pos_robot)
        quat_s = " ".join(f"{v:.6f}" for v in quat_wxyz)
        shell = (
            f"{self._uv} run --no-sync scripts/core/glns/solve.py "
            f"--object {obj!r} --viewpoints {h5!r} "
            f"--scene {self._scene!r} "
            f"--object-position {pos_s} --object-quat {quat_s} "
            f"--delaunay-expand-hops {hops}{augment} --output {det_h5!r} "
            f"&& {self._uv} run --no-sync scripts/core/glns/verify.py "
            f"--result {det_h5!r} --join --require-full-coverage --spacing {spacing} "
            f"--no-home-bracket --output-dir {trajectory_dir!r}"
        )
        cmd = ["bash", "-c", shell]

        self._set_busy(self._btn_cancel_gen)
        self._append_log("[generate] $ " + " ".join(cmd))
        generated_csv_path: list[str] = []

        def on_line(line: str):
            self._append_log(line)
            m = CSV_PATH_RE.search(line)
            if m:
                csv = m.group(1)
                if not Path(csv).is_absolute():
                    csv = str(PROJECT_ROOT / csv)
                self._csv_path_model.set_value(csv)
                generated_csv_path[:] = [csv]
                self._append_log(f"[generate] captured CSV: {csv}")

        def on_exit(rc: int):
            self._append_log(
                f"[generate] cancelled" if self._gen_runner.cancelled
                else f"[generate] exit code = {rc}")
            self._clear_busy()
            if rc == 0 and generated_csv_path:
                csv = generated_csv_path[0]
                if self._preview.load(csv):
                    self._update_slider_bounds()
                    self._refresh_status()
                    self._append_log(f"[preview] auto-loaded generated CSV: {csv}")

        self._gen_runner.start(cmd, cwd=PROJECT_ROOT, on_line=on_line, on_exit=on_exit)

    def _home_move_target(self, transition: str, obj: str):
        """(목표 관절값, 라벨, 출력 CSV 경로) — 실패 시 None.

        목표는 오늘과 같은 곳에서 읽는다: approach = Execute CSV 첫 행, return = HOME.
        GLNS 결과 h5 에 기대지 않으므로 DP 궤적 CSV 에서도 그대로 동작한다.
        """
        from common import config as _config

        label = HOME_TRANSITIONS[transition]
        csv = self._csv_path_model.get_value_as_string().strip()
        if transition == "approach":
            if not csv or not Path(csv).exists():
                self._append_log(f"[home] scan CSV not found: {csv!r}")
                return None
            try:
                solutions, _times = load_trajectory_csv(csv)
                target_q = np.asarray(solutions[0], dtype=np.float64)
            except Exception as exc:  # noqa: BLE001
                self._append_log(f"[home] scan CSV load failed: {exc}")
                return None
        else:
            target_q = np.asarray(_config.ROBOT_START_STATE, dtype=np.float64)

        # 계획 산출물은 자기가 브래킷하는 스캔 궤적 옆에 둔다. 스캔 CSV 가 없으면(return 만
        # 가능) h5 경로에서 물체/viewpoint 수를 읽어 같은 레이아웃 자리를 만들고, 그것도
        # 없으면 물체 폴더까지만. 어느 경우든 data/{object}/ 밖으로 나가지 않는다.
        name = f"home_move_{transition}.csv"
        if csv and Path(csv).parent.is_dir():
            return target_q, label, Path(csv).parent / name
        _, n_vp = self._parse_h5_meta(self._h5_path_model.get_value_as_string().strip())
        if n_vp is not None:
            return target_q, label, _config.get_trajectory_path(obj, n_vp, filename=name)
        return target_q, label, _config.DATA_ROOT / obj / "trajectory" / name

    def _on_plan_home_transition(self, transition: str):
        """HOME↔스캔시작을 **충돌-free 로 계획한 뒤** 실행한다.

        현재 자세에서 계획하는 게 핵심이다. 고정된 자세(HOME/스캔끝)에서 시작하는 궤적을
        실행하면, 두 실행기 모두 현재 자세→CSV 첫 행 사이에 **계획되지 않은 직선**을 덧붙인다
        (IsaacArticulationExecutor.start / publish.build_interpolated_points). 시작점이 곧
        현재 자세면 그 구간이 아예 생기지 않아 전 구간이 계획된 이동이 된다.

        2단계 체인이다: plan_move.py(_gen_runner) → 성공 시 _start_csv_execution.
        _gen_runner 를 쓰므로 계획 중 버튼 비활성화와 Cancel(Generate) 이 그대로 적용된다.
        """
        if transition not in HOME_TRANSITIONS:
            raise ValueError(f"unknown HOME transition: {transition}")
        context = self._move_context("home")
        if context is None:
            return
        obj, pos_robot, quat_wxyz = context

        target = self._home_move_target(transition, obj)
        if target is None:
            return
        target_q, label, out_csv = target
        self._plan_and_execute_move(
            tag="home", obj=obj, pos_robot=pos_robot, quat_wxyz=quat_wxyz,
            target_q=target_q, label=label, out_csv=out_csv)

    def _move_context(self, tag: str):
        """이동 계획의 공통 전제 → (object, pos_robot, quat_wxyz) 또는 None.

        충돌 세계는 스테이지의 살아있는 물체 pose 로 만든다(_on_generate 와 같은 관례).
        물체를 먼저 확정한다 — 계획의 입력이자 산출물이 놓일 자리를 정한다.
        """
        if (self._gen_runner.running or self._sim_executor.running
                or self._pub_runner.running):
            self._append_log(f"[{tag}] a plan or move is already running")
            return None
        obj = (self._current_object or "").strip()
        pose = self._read_object_world_pose()
        if not obj or pose is None:
            self._append_log(
                f"[{tag}] no target object on stage - cannot build the collision world. "
                "Load an object first, then retry.")
            return None
        return obj, pose[0], pose[1]

    def _plan_and_execute_move(self, *, tag, obj, pos_robot, quat_wxyz,
                               target_q, label, out_csv):
        """현재 자세 → target_q 를 계획해 실행한다. HOME 이동과 tilt 진입이 공유한다.

        2단계 체인이다: plan_move.py(_gen_runner) → 성공 시 _start_csv_execution.
        _gen_runner 를 쓰므로 계획 중 버튼 비활성화와 Cancel(Generate) 이 그대로 적용된다.
        """
        try:
            current_q = self._sim_executor.current_joints()
        except Exception as exc:  # noqa: BLE001 — 로봇/스테이지 미준비
            self._append_log(f"[{tag}] could not read the current joint state: {exc}")
            return

        if self._preview.loaded:
            self._preview.stop()
        # 계획과 실행을 한 덩어리로 취소할 수 있게, 두 단계 모두 Execute 패널의 Cancel 을
        # 살려둔다 — 사용자가 누른 버튼(Move to Start / Return to HOME)과 같은 패널이라
        # 어디를 눌러야 멈추는지 찾을 필요가 없다. 어느 러너를 멈출지는
        # _on_cancel_execute 가 단계를 보고 고른다.
        self._set_busy(self._btn_cancel_pub)

        def finished(rc: int):
            self._append_log(f"[{tag}] {label} exit code = {rc}")
            self._clear_busy()

        def on_planned(rc: int):
            if self._gen_runner.cancelled:
                self._append_log(f"[{tag}] {label}: planning cancelled - not moving.")
                self._clear_busy()
                return
            self._append_log(f"[{tag}] plan exit code = {rc}")
            if rc != 0 or not out_csv.exists():
                self._append_log(
                    f"[{tag}] {label}: no collision-free route - not moving.")
                self._clear_busy()
                return
            self._append_log(f"[{tag}] executing planned {label}: {out_csv}")
            if not self._start_csv_execution(str(out_csv), tag=tag, on_done=finished):
                self._clear_busy()

        # NB: --from/to-joints 는 =<v> (공백 아님). 관절 문자열은 첫 값이 음수면 '-1.5,...'
        # 로 시작하는데, argparse 의 음수 휴리스틱은 순수 숫자 하나("-1.5")만 값으로 봐주고
        # 쉼표가 붙으면 옵션으로 오인해 "expected one argument" 로 죽는다. 현재 자세는 임의
        # 값이라 언제든 걸린다. publish.py --joint-target 도 같은 이유로 = 형태다.
        # (--object-position/quat 은 값이 하나씩 떨어져 있어 휴리스틱에 걸린다 — 공백 유지.)
        shell = (
            f"{self._uv} run --no-sync scripts/core/trajectory/plan_move.py "
            f"--object {obj!r} "
            f"--scene {self._scene!r} "
            f"--object-position {' '.join(f'{v:.6f}' for v in pos_robot)} "
            f"--object-quat {' '.join(f'{v:.6f}' for v in quat_wxyz)} "
            f"--from-joints={','.join(f'{v:.6f}' for v in current_q)!r} "
            f"--to-joints={','.join(f'{v:.6f}' for v in target_q)!r} "
            f"--output {str(out_csv)!r}"
        )
        self._append_log(
            f"[{tag}] {label}: planning a collision-free route from the current pose...")
        self._append_log(f"[{tag}] $ " + shell)
        self._gen_runner.start(["bash", "-c", shell], cwd=PROJECT_ROOT,
                               on_line=self._append_log, on_exit=on_planned)

    # ------------------------------------------------------------------
    # Tilt panel callbacks
    # ------------------------------------------------------------------
    def _tilt_index(self) -> int:
        return max(0, int(self._get_field("tilt_index", int)))

    def _tilt_viewpoints(self):
        """Tilt 가 쓸 (h5 경로, object, viewpoint 수) — 없거나 못 읽으면 None.

        입력은 Generate 와 완전히 같다: 위 패널에서 고른 viewpoints .h5 하나. 물체 이름과
        viewpoint 수는 그 표준 경로에서 읽는다(data/{object}/viewpoint/{N}/...).
        """
        h5 = self._h5_path_model.get_value_as_string().strip()
        if not h5:
            self._append_log("[tilt] pick a viewpoints .h5 first (Browse...).")
            return None
        if not Path(h5).exists():
            self._append_log(f"[tilt] h5 not found: {h5}")
            return None
        obj, n_vp = self._parse_h5_meta(h5)
        if obj is None:
            obj = (self._current_object or "").strip()
        return h5, obj, n_vp

    def _on_highlight_tilt_viewpoint(self):
        """고른 중심 viewpoint 를 스테이지에서 크고 노랗게 표시한다(어느 점인지 확인용)."""
        picked = self._tilt_viewpoints()
        if picked is None:
            return
        h5 = picked[0]

        import omni.usd

        stage = omni.usd.get_context().get_stage()
        target_prim = stage.GetPrimAtPath(TARGET_OBJECT_PRIM)
        if not target_prim or not target_prim.IsValid():
            self._append_log("[tilt] no target object on stage - Load Object first.")
            return
        try:
            points_local, wd_m = self._load_camera_viewpoint_points(h5)
        except Exception as e:  # noqa: BLE001 — 파일/스키마 문제를 UI 로 보고
            self._append_log(f"[tilt] viewpoint load failed: {e}")
            return
        index = self._tilt_index()
        if index >= len(points_local):
            self._append_log(
                f"[tilt] center index {index} out of range - this file has "
                f"{len(points_local)} viewpoints.")
            return
        self._draw_camera_viewpoint_points(points_local, highlight=index)
        self._append_log(
            f"[tilt] center = viewpoint #{index} / {len(points_local)} "
            f"(yellow point, WD={wd_m * 1000:.1f} mm)")

    # ---- Tilt 부채꼴 시각화 -------------------------------------------------
    @staticmethod
    def _even_subset(n: int, k: int) -> list:
        """0..n-1 에서 k 개를 균등하게 고른다. 마지막(=최대각)은 반드시 포함."""
        if n <= 0:
            return []
        if n <= k:
            return list(range(n))
        step = (n - 1) / float(max(k - 1, 1))
        return sorted({int(round(i * step)) for i in range(k)} | {n - 1})

    def _tilt_fan(self):
        """지금 패널 설정대로의 부채꼴 → (target, legs, center_pose) — 물체 로컬 프레임.

        object_pose 를 단위행렬로 주면 포즈가 물체 로컬로 나온다. 그 좌표로 물체 프림 아래에
        그리면 USD 가 물체 변환을 적용해 주므로, 기즈모로 옮겨도 부채꼴이 따라온다.
        기하는 CLI(tilt_motion.py)와 같은 common.tilt_geometry 를 쓴다 — 화면의 부채꼴이
        실제로 생성될 포즈와 어긋날 수 없다.
        """
        from common.tilt_geometry import camera_pose, tilt_legs
        from core.viewpoint.storage import load_viewpoints_hdf5

        picked = self._tilt_viewpoints()
        if picked is None:
            return None
        viewpoint = load_viewpoints_hdf5(picked[0])
        index = self._tilt_index()
        if index >= viewpoint.count:
            self._append_log(
                f"[tilt-fan] center index {index} out of range - this file has "
                f"{viewpoint.count} viewpoints.")
            return None

        wd_m = viewpoint.working_distance_m
        center = camera_pose(viewpoint.positions[index], viewpoint.normals[index],
                             wd_m, np.eye(4))
        target, legs = tilt_legs(
            center, wd_m,
            pitch_min=float(self._get_field("tilt_pitch_min", float)),
            pitch_max=float(self._get_field("tilt_pitch_max", float)),
            pitch_n=max(2, int(self._get_field("tilt_pitch_n", int))),
            roll_min=float(self._get_field("tilt_roll_min", float)),
            roll_max=float(self._get_field("tilt_roll_max", float)),
            roll_n=max(2, int(self._get_field("tilt_roll_n", int))))
        if not legs:
            self._append_log("[tilt-fan] every leg angle is 0 deg - nothing to draw.")
            return None
        return target, legs, center

    def _delete_tilt_fan(self, log: bool):
        from isaacsim.core.utils import prims

        scope = f"{TARGET_OBJECT_PRIM}/{TILT_FAN_SCOPE_NAME}"
        if prims.is_prim_path_valid(scope):
            prims.delete_prim(scope)
            if log:
                self._append_log(f"[tilt-fan] cleared {scope}")
        elif log:
            self._append_log("[tilt-fan] nothing to clear")

    def _draw_tilt_fan(self) -> bool:
        """부채꼴을 그린다: leg 별 호(arc) + waypoint 점 + 주시점으로 모이는 시선."""
        import omni.usd
        from pxr import Gf, UsdGeom, Vt

        stage = omni.usd.get_context().get_stage()
        target_prim = stage.GetPrimAtPath(TARGET_OBJECT_PRIM)
        if not target_prim or not target_prim.IsValid():
            self._append_log("[tilt-fan] no target object on stage - Load Object first.")
            return False
        try:
            fan = self._tilt_fan()
        except Exception as e:  # noqa: BLE001 — 파일/스키마 문제를 UI 로 보고
            self._append_log(f"[tilt-fan] failed: {e}")
            return False
        if fan is None:
            return False
        target, legs, center = fan

        self._delete_tilt_fan(log=False)
        scope_path = f"{TARGET_OBJECT_PRIM}/{TILT_FAN_SCOPE_NAME}"
        UsdGeom.Xform.Define(stage, scope_path)

        def vec(p):
            return Gf.Vec3f(float(p[0]), float(p[1]), float(p[2]))

        centre_cam = vec(center[:3, 3])
        focus = vec(target)

        arc_pts, arc_counts, arc_colors = [], [], []
        wp_pts, wp_colors = [centre_cam], [Gf.Vec3f(*TILT_CENTER_COLOR)]
        ray_pts, ray_counts, ray_colors = [focus, centre_cam], [2], \
            [Gf.Vec3f(*TILT_CENTER_COLOR)]
        summary = []
        for label, poses, angles in legs:
            colour = Gf.Vec3f(*TILT_LEG_COLORS.get(label, TILT_CENTER_COLOR))
            cams = [vec(p[:3, 3]) for p in poses]
            # 호는 중심에서 시작해 그 leg 의 끝까지 — 카메라가 실제로 지나가는 자리다.
            arc_pts.extend([centre_cam] + cams)
            arc_counts.append(len(cams) + 1)
            arc_colors.append(colour)
            wp_pts.extend(cams)
            wp_colors.extend([colour] * len(cams))
            for i in self._even_subset(len(cams), TILT_FAN_RAYS_PER_LEG):
                ray_pts.extend([focus, cams[i]])
                ray_counts.append(2)
                ray_colors.append(colour)
            summary.append(f"{label} {float(angles[-1]):+.1f}")

        arcs = UsdGeom.BasisCurves.Define(stage, f"{scope_path}/Arcs")
        arcs.CreateTypeAttr(UsdGeom.Tokens.linear)
        arcs.CreateCurveVertexCountsAttr(Vt.IntArray(arc_counts))
        arcs.CreatePointsAttr(Vt.Vec3fArray(arc_pts))
        arcs.CreateWidthsAttr(Vt.FloatArray([TILT_FAN_ARC_WIDTH_M]))
        arcs.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        # uniform = 커브 하나당 색 하나 → 호가 어느 leg 인지 색으로 읽힌다.
        arcs.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(Vt.Vec3fArray(arc_colors))

        rays = UsdGeom.BasisCurves.Define(stage, f"{scope_path}/ViewRays")
        rays.CreateTypeAttr(UsdGeom.Tokens.linear)
        rays.CreateCurveVertexCountsAttr(Vt.IntArray(ray_counts))
        rays.CreatePointsAttr(Vt.Vec3fArray(ray_pts))
        rays.CreateWidthsAttr(Vt.FloatArray([TILT_FAN_RAY_WIDTH_M]))
        rays.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        rays.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(Vt.Vec3fArray(ray_colors))
        rays.CreateDisplayOpacityAttr(Vt.FloatArray([0.55]))

        waypoints = UsdGeom.Points.Define(stage, f"{scope_path}/Waypoints")
        waypoints.CreatePointsAttr(Vt.Vec3fArray(wp_pts))
        waypoints.CreateWidthsAttr(Vt.FloatArray([TILT_FAN_WAYPOINT_WIDTH_M] * len(wp_pts)))
        waypoints.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray(wp_colors))

        centre_marker = UsdGeom.Points.Define(stage, f"{scope_path}/SurfacePoint")
        centre_marker.CreatePointsAttr(Vt.Vec3fArray([focus]))
        centre_marker.CreateWidthsAttr(Vt.FloatArray([TILT_FAN_CENTER_WIDTH_M]))
        centre_marker.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(1.0, 0.15, 0.15)]))

        self._append_log(
            f"[tilt-fan] viewpoint #{self._tilt_index()}: {len(wp_pts)} waypoints, "
            f"{len(ray_counts)} view rays, deg [{', '.join(summary)}] under {scope_path}")
        return True

    def _refresh_tilt_fan(self, *_args):
        """설정이 바뀌면 켜져 있는 부채꼴만 다시 그린다(각도를 만지면 바로 보이게)."""
        if self._tilt_fan_on:
            self._draw_tilt_fan()

    def _on_clear_tilt_fan(self):
        self._delete_tilt_fan(log=True)
        self._tilt_fan_on = False
        if self._btn_tilt_fan is not None:
            self._btn_tilt_fan.text = "Show Tilt Fan"

    def _on_toggle_tilt_fan(self):
        if self._tilt_fan_on:
            self._delete_tilt_fan(log=True)
            self._tilt_fan_on = False
        elif self._draw_tilt_fan():
            self._tilt_fan_on = True
        if self._btn_tilt_fan is not None:
            self._btn_tilt_fan.text = (
                "Hide Tilt Fan" if self._tilt_fan_on else "Show Tilt Fan")

    def _tilt_output_path(self, obj: str, n_vp, index: int) -> Path:
        """Tilt CSV 자리 — 스캔 산출물과 같은 폴더, 같은 명명 규칙(trajectory_{역할}.csv)."""
        from common import config as _config

        role = f"tilt_vp{index:04d}"
        if n_vp is not None:
            return _config.get_trajectory_artifact_path(obj, n_vp, role=role)
        return _config.DATA_ROOT / obj / "trajectory" / f"trajectory_{role}.csv"

    def _on_generate_tilt(self):
        """중심 viewpoint 하나를 공전하는 tilt 궤적을 만든다 (tilt_motion.py 서브프로세스).

        _gen_runner 를 쓰므로 Generate 와 동시에 돌 수 없고, Cancel 버튼이 그대로 듣는다.
        성공하면 CSV 경로가 공용 칸에 들어가고 프리뷰가 자동 로드된다 — Generate 와 동일.
        """
        if self._gen_runner.running:
            self._append_log("[tilt] a generate/plan is already running")
            return
        picked = self._tilt_viewpoints()
        if picked is None:
            return
        h5, obj, n_vp = picked
        if not obj:
            self._append_log(
                "[tilt] couldn't read object from h5 path and no object is loaded.")
            return
        if self._current_object and obj != self._current_object:
            self._append_log(
                f"[tilt] WARNING: h5 object '{obj}' != loaded scene object "
                f"'{self._current_object}'. Pose & collision mesh come from the SCENE "
                "object - load the matching object or pick the matching h5.")

        pose = self._read_object_world_pose()
        if pose is None:
            self._append_log(
                "[tilt] no target object on stage - pick one in the Object dropdown "
                "and click 'Load Object' first.")
            return
        pos_robot, quat_wxyz = pose

        index = self._tilt_index()
        out_csv = self._tilt_output_path(obj, n_vp, index)

        # anchor = 로봇의 현재 자세. tilt 는 팔 분기를 자유롭게 고를 수 있어서, 지금 있는
        # 자리에서 가장 가까운 분기로 시작해야 진입 이동(Move to Tilt Start)이 짧아진다.
        anchor = ""
        try:
            current_q = self._sim_executor.current_joints()
            anchor = " --anchor-joints=" + repr(",".join(f"{v:.6f}" for v in current_q))
        except Exception as exc:  # noqa: BLE001 — 로봇 미준비면 anchor 없이 진행
            self._append_log(
                f"[tilt] no current joint state, continuing without an anchor: {exc}")

        pos_s = " ".join(f"{v:.6f}" for v in pos_robot)
        quat_s = " ".join(f"{v:.6f}" for v in quat_wxyz)
        clamp = self._get_field("tilt_clamp", bool)
        shell = (
            f"{self._uv} run --no-sync scripts/core/trajectory/tilt_motion.py "
            f"--object {obj!r} --viewpoints {h5!r} --viewpoint-index {index} "
            f"--object-position {pos_s} --object-quat {quat_s} "
            f"--pitch-min {float(self._get_field('tilt_pitch_min', float)):.3f} "
            f"--pitch-max {float(self._get_field('tilt_pitch_max', float)):.3f} "
            f"--pitch-n {max(2, int(self._get_field('tilt_pitch_n', int)))} "
            f"--roll-min {float(self._get_field('tilt_roll_min', float)):.3f} "
            f"--roll-max {float(self._get_field('tilt_roll_max', float)):.3f} "
            f"--roll-n {max(2, int(self._get_field('tilt_roll_n', int)))} "
            f"--num-seeds {max(1, int(self._get_field('tilt_num_seeds', int)))} "
            f"--batch-size {max(1, int(self._get_field('tilt_batch_size', int)))}"
            f"{'' if clamp else ' --no-clamp'}{anchor} "
            f"--output {str(out_csv)!r}"
        )

        self._set_busy(self._btn_tilt_cancel)
        self._append_log(f"[tilt] center viewpoint #{index} -> {out_csv}")
        self._append_log("[tilt] $ " + shell)
        generated_csv_path: list[str] = []

        def on_line(line: str):
            self._append_log(line)
            m = CSV_PATH_RE.search(line)
            if m:
                csv = m.group(1)
                if not Path(csv).is_absolute():
                    csv = str(PROJECT_ROOT / csv)
                self._csv_path_model.set_value(csv)
                generated_csv_path[:] = [csv]
                self._append_log(f"[tilt] captured CSV: {csv}")

        def on_exit(rc: int):
            self._append_log(
                f"[tilt] cancelled" if self._gen_runner.cancelled
                else f"[tilt] exit code = {rc}")
            self._clear_busy()
            if rc == 0 and generated_csv_path:
                csv = generated_csv_path[0]
                if self._preview.load(csv):
                    self._update_slider_bounds()
                    self._refresh_status()
                    self._append_log(f"[preview] auto-loaded tilt CSV: {csv}")

        self._gen_runner.start(["bash", "-c", shell], cwd=PROJECT_ROOT,
                               on_line=on_line, on_exit=on_exit)

    def _on_cancel_generate(self):
        if self._gen_runner.running:
            self._append_log("[generate] terminating...")
            self._gen_runner.terminate()

    # ------------------------------------------------------------------
    # File picker (shared by panels B and C)
    # ------------------------------------------------------------------
    def _open_file_picker(self, title: str, model, item_label: str, ext: str, start_dir: str,
                          on_selected=None):
        """Open the Omni file picker filtered to `ext`, writing the pick into `model`.

        ``on_selected(full_path)`` 는 선택이 확정된 뒤에만 불린다 — 필드는 손으로도 편집할 수
        있어서 ``model.add_value_changed_fn`` 을 쓰면 타자 한 글자마다 불려버린다.
        """
        def _on_apply(filename: str, dirname: str):
            full = os.path.join(dirname, filename) if filename else dirname
            model.set_value(full)
            self._append_log(f"[browse] selected: {full}")
            if on_selected is not None:
                try:
                    on_selected(full)
                except Exception as e:  # noqa: BLE001 — 부가 동작이 선택을 막지 않게
                    self._append_log(f"[browse] post-select hook failed: {e}")
            try:
                dialog.hide()
            except Exception:
                pass

        def _on_cancel(*_):
            try:
                dialog.hide()
            except Exception:
                pass

        try:
            from omni.kit.window.filepicker import FilePickerDialog
        except ImportError as e:
            self._append_log(f"[browse] file picker unavailable: {e}")
            return

        try:
            dialog = FilePickerDialog(
                title,
                apply_button_label="Select",
                click_apply_handler=_on_apply,
                click_cancel_handler=_on_cancel,
                item_filter_options=[item_label, "All Files (*.*)"],
                item_filter_fn=lambda item: item.is_folder or item.path.endswith(ext),
                current_directory=start_dir,
            )
            dialog.show()
        except Exception as e:
            self._append_log(f"[browse] picker open failed: {e}")

    def _start_dir_for(self, model, *subdirs: str) -> str:
        """Sensible picker start dir: the model's current parent if valid, else
        the first existing of data/{object}/<subdirs...>, data/, PROJECT_ROOT."""
        current = model.get_value_as_string().strip()
        if current and Path(current).parent.is_dir():
            return str(Path(current).parent)
        obj = (self._current_object or "sample").strip() or "sample"
        candidates = [PROJECT_ROOT / "data" / obj / sd for sd in subdirs]
        candidates += [PROJECT_ROOT / "data", PROJECT_ROOT]
        return str(next((p for p in candidates if p.is_dir()), PROJECT_ROOT))

    def _on_browse_csv(self):
        """Open Omni file picker pre-rooted at data/{object}/trajectory/."""
        start_dir = self._start_dir_for(self._csv_path_model, "trajectory")
        self._open_file_picker("Select trajectory CSV", self._csv_path_model,
                               "CSV (*.csv)", ".csv", start_dir)

    def _on_browse_h5(self):
        """Open Omni file picker pre-rooted at data/{object}/viewpoint/."""
        start_dir = self._start_dir_for(self._h5_path_model, "viewpoint")
        # 고른 즉시 카메라 스펙을 그 h5 스냅샷으로 맞춘다 — 스펙 입력칸이 바로 아래 있어서
        # 파일을 고르면 따라 바뀔 거라고 기대하게 된다(Show Viewpoints 를 눌러야만 바뀌면 놀란다).
        self._open_file_picker("Select viewpoints .h5", self._h5_path_model,
                               "HDF5 (*.h5)", ".h5", start_dir,
                               on_selected=self._sync_camera_spec_from_h5)

    # ------------------------------------------------------------------
    # Preview panel callbacks
    # ------------------------------------------------------------------
    def _on_load_preview(self):
        csv = self._csv_path_model.get_value_as_string().strip()
        if not csv:
            self._append_log("[preview] CSV path is empty")
            return
        if not Path(csv).exists():
            self._append_log(f"[preview] CSV not found: {csv}")
            return
        if self._preview.load(csv):
            self._update_slider_bounds()
            self._refresh_status()

    def _on_play(self):
        if not self._preview.loaded:
            self._append_log("[preview] load a CSV first")
            return
        self._preview.play()

    def _on_pause(self):
        self._preview.pause()

    def _on_stop(self):
        self._preview.stop()
        self._set_slider_value(0.0)
        self._refresh_status()

    def _on_slider(self, model):
        if self._updating_slider:
            return
        if not self._preview.loaded:
            return
        self._preview.seek(model.get_value_as_float())
        self._refresh_status()

    def _update_slider_bounds(self):
        if self._slider_model is None:
            return
        duration = max(float(self._preview.state.duration), 1e-6)
        if self._slider is not None:
            self._slider.min = 0.0
            self._slider.max = duration
        self._set_slider_value(0.0)

    def _set_slider_value(self, value: float):
        if self._slider_model is None:
            return
        self._updating_slider = True
        try:
            self._slider_model.set_value(float(value))
        finally:
            self._updating_slider = False

    def _refresh_status(self):
        if self._status_label is None:
            return
        s = self._preview.state
        if not self._preview.loaded:
            self._status_label.text = "t=0.00s / 0.00s  (no CSV)"
            return
        # Find the nearest waypoint index for display.
        i = int(np.searchsorted(s.times - s.times[0], s.t))
        i = max(0, min(i, len(s.times) - 1))
        self._status_label.text = (
            f"t={s.t:.2f}s / {s.duration:.2f}s  (wp {i}/{len(s.times)-1})"
        )

    def step_preview(self, dt: float):
        """Called from the simulation loop each frame."""
        if self._preview.state.playing:
            self._preview.step(dt)
            if self._slider_model is not None:
                self._set_slider_value(self._preview.state.t)
            self._refresh_status()

    # ------------------------------------------------------------------
    # Execute panel callbacks
    # ------------------------------------------------------------------
    def _on_execute(self):
        # Execute drives the CURRENT robot: SIM applies in-process articulation
        # targets without ROS; REAL sends FollowJointTrajectory through ROS2.
        if self._pub_runner.running or self._sim_executor.running:
            self._append_log("[execute] already running")
            return
        csv = self._csv_path_model.get_value_as_string().strip()
        if not csv or not Path(csv).exists():
            self._append_log(f"[execute] CSV not found: {csv!r}")
            return

        # Execution always drives the real Isaac articulation (or the real robot
        # mirrored into Isaac), never the preview ghost. Hide the ghost even when
        # preview is paused so two robot poses cannot overlap in the viewport.
        if self._preview.loaded:
            self._preview.stop()

        self._set_busy(self._btn_cancel_pub)

        def on_done(rc: int):
            self._append_log(f"[execute] exit code = {rc}")
            self._clear_busy()

        if not self._start_csv_execution(csv, tag="execute", on_done=on_done):
            self._clear_busy()

    def _start_csv_execution(self, csv: str, *, tag: str,
                             on_done: Callable[[int], None]) -> bool:
        """CSV 하나를 현재 대상 로봇에서 실행한다.

        sim → in-process articulation, real → publish.py 를 통한 FollowJointTrajectory.
        Execute 버튼과 HOME 이동이 공유하므로 `publish.py --csv` 셸 문자열과
        컨트롤러 활성화가 한 곳에만 존재한다.

        Returns: 시작에 동기적으로 실패하면 False (그때 on_done 은 불리지 않는다).
        """
        if self._mode == "sim":
            return self._sim_executor.start(csv, on_done=on_done)

        # REAL 은 ROS2 trajectory controller 경로. 보내기 전에 inspection 컨트롤러를
        # 활성화한다(헬퍼 주석 참고 — run-mode 전환이 이 스위치를 놓칠 수 있다).
        shell_cmd = (
            "source /opt/ros/jazzy/setup.bash && "
            f"{self._ensure_inspection_controller_cmd()} && "
            f"exec {self._uv} run --no-sync scripts/core/trajectory/publish.py "
            f"--csv {csv!r} --target controller"
        )
        self._append_log(f"[{tag}] target=real robot")
        self._append_log(f"[{tag}] $ " + shell_cmd)
        self._pub_runner.start(["bash", "-c", shell_cmd], cwd=PROJECT_ROOT,
                               on_line=self._append_log, on_exit=on_done)
        return True

    def _on_cancel_execute(self):
        # HOME/틸트 진입 이동은 plan -> execute 2단계다. 계획 단계면 그 러너를 멈춘다 —
        # 사용자에게는 "지금 하고 있는 그 이동"을 멈추는 버튼 하나로 보여야 한다.
        if self._gen_runner.running:
            self._append_log("[execute] cancelling the motion plan...")
            self._gen_runner.terminate()
            return
        if self._sim_executor.running:
            self._sim_executor.cancel()
            self._clear_busy()
            return
        if self._pub_runner.running:
            self._append_log("[execute] terminating trajectory sender...")
            self._pub_runner.terminate()
        if self._mode == "sim":
            self._append_log("[execute] no Isaac trajectory is running")
            return
        # real: the controller already holds the whole trajectory goal and keeps
        # executing it, so terminating the publisher is not enough — cancel the goal.
        shell_cmd = (
            "source /opt/ros/jazzy/setup.bash && "
            f"timeout 3 ros2 service call /{INSPECTION_CONTROLLER}/follow_joint_trajectory"
            "/_action/cancel_goal action_msgs/srv/CancelGoal '{}'"
        )
        self._append_log(f"[execute] cancelling goals on {INSPECTION_CONTROLLER}")
        self._ctrl_runner.start(
            ["bash", "-c", shell_cmd], cwd=PROJECT_ROOT,
            on_line=self._append_log,
            on_exit=lambda rc: self._append_log(f"[execute] cancel exit={rc}"))

    # ------------------------------------------------------------------
    # Per-frame pump
    # ------------------------------------------------------------------
    def pump(self, dt: float):
        self._gen_runner.pump()
        self._ik_runner.pump()
        self._pub_runner.pump()
        self._ctrl_runner.pump()
        self._relay_runner.pump()
        self._sim_executor.step(dt)
        # Live camera-range rays: re-cast so they follow the moving camera.
        try:
            self._tick_camera_ranges(dt)
        except Exception:  # noqa: BLE001
            pass
        # Keep inspection-camera viewports square so the full FOV is shown, and
        # keep the cameras mouse-locked (UI-only control).
        self._lock_inspection_viewports()
        try:
            self._lock_camera_prims()
        except Exception:  # noqa: BLE001
            pass
        # While anything long-running is going (execute / generate / IK check), lock
        # BOTH mode combos: switching pipeline mode would deactivate the inspection
        # controller and abort the trajectory, and switching run mode (sim/real) would
        # move the ground under a job that already picked its target.
        busy = (self._pub_runner.running or self._sim_executor.running
                or self._gen_runner.running or self._ik_runner.running)
        if self._pipeline_combo is not None:
            self._pipeline_combo.enabled = not busy
        if self._mode_combo is not None:
            self._mode_combo.enabled = not busy
        self.step_preview(dt)


# =============================================================================
# Main
# =============================================================================

def main():
    args = urctl.parse_args()
    if not args.usd_path.exists():
        sys.exit(f"Robot USD not found: {args.usd_path}")

    # 씬을 먼저 반영한다 — load_workcell 의 테이블 스케일과 장애물 스폰이 이 값을 읽는다.
    scene_config.apply_cli(args, _cfg_module)

    simulation_app = urctl.start_sim(headless=False)

    from isaacsim.core.api import SimulationContext
    simulation_context = SimulationContext(stage_units_in_meters=1.0)

    urctl.load_workcell(args.usd_path)
    simulation_app.update()
    urctl.load_target_object(args.object)
    simulation_app.update()

    articulation_root = urctl.find_articulation_root()
    simulation_app.update()
    inspection_cam = urctl.setup_inspection_camera()
    if inspection_cam is not None:
        simulation_app.update()

    real_graph_path = urctl.build_real_robot_graph(articulation_root, inspection_cam)
    simulation_app.update()

    # Separate MoveIt bridge graph (/isaac_joint_commands → robot, robot → /isaac_joint_states).
    # Gated independently from the inspection graph by the top-level pipeline mode.
    sim_graph_path = urctl.build_sim_robot_graph(articulation_root)
    simulation_app.update()

    # Physics-free ghost overlay for trajectory preview. Built once offline
    # by scripts/setup/build_ghost_usd.py — referencing it here
    # should add zero physics state and leave the real /World/UR20
    # articulation untouched.
    ghost_usd_path = args.usd_path.parent / GHOST_USD_NAME
    if not ghost_usd_path.exists():
        sys.exit(
            f"Ghost USD not found: {ghost_usd_path}\n"
            f"Build it first: uv run scripts/setup/build_ghost_usd.py"
        )
    base_link, chain = spawn_preview_ghost(
        usd_path=ghost_usd_path,
        ghost_root=GHOST_ROOT_PATH,
        position=np.array([0.0, 0.0, urctl.MOUNT_HEIGHT]),
        joint_order=JOINT_NAMES,
        log=print,
    )
    simulation_app.update()

    # Preview ghost gets its own viewport camera, named distinctly so it never
    # collides with the real robot's InspectionCamera (ROS render product binds
    # to that one). No render product / ROS publisher here — this is view-only,
    # and it follows the ghost's FK poses during playback.
    preview_cam = urctl.setup_inspection_camera(
        root_path=GHOST_ROOT_PATH, camera_name="InspectionCameraPreview",
    )
    if preview_cam is not None:
        simulation_app.update()

    window = PipelineWindow(
        ghost_root_prim=GHOST_ROOT_PATH,
        base_link_path=base_link,
        chain=chain,
        real_graph_path=real_graph_path,
        default_object=(args.object or "sample"),
        initial_mode=args.mode,
        sim_graph_path=sim_graph_path,
        initial_pipeline_mode=args.pipeline_mode,
        articulation_root=articulation_root,
        scene=_cfg_module.ACTIVE_SCENE,
    )

    simulation_context.initialize_physics()
    simulation_context.play()

    # Apply the initial mode now that the graph exists and playback has started:
    # default sim → graph tick OFF from frame 0 (no /joint_states, no publish).
    window.apply_mode(args.mode)
    # Then apply the top-level pipeline mode: inspection (default) leaves the
    # above in place + blocks MoveIt; moveit flips to the MoveIt graph and locks
    # the Inspection UI.
    window.apply_pipeline_mode(args.pipeline_mode)

    # Stand the robot at the configured start pose instead of the all-zero USD
    # default. Sim mode only — in real mode the action graph mirrors the live
    # /joint_states, which must win. One physics step first so the articulation
    # view is bound before set_start_pose initializes it.
    if args.mode == "sim":
        from common import config as _cfg
        simulation_context.step(render=False)
        try:
            urctl.set_start_pose(articulation_root, JOINT_NAMES, _cfg.ROBOT_START_STATE)
            window._append_log(
                "[start-pose] robot set to ROBOT_START_STATE "
                f"{np.rad2deg(_cfg.ROBOT_START_STATE).round(1).tolist()} deg")
        except Exception as e:  # noqa: BLE001 — pose is cosmetic, never fatal
            window._append_log(
                f"[start-pose] failed ({e}); robot stays at USD default")

    # Stop/Play handling: on each transition clear both graphs' ArticulationController
    # commands (so a stale retained command isn't re-applied → no snap), and on Play
    # restore the configured start pose (Isaac resets the articulation to its USD
    # default = zeros on Stop; we want it back at ROBOT_START_STATE, like boot).
    # With the relay no longer holding an idle setpoint, set_start_pose sticks.
    from common import config as _cfg

    last_t = None
    import time as _time
    was_playing = simulation_context.is_playing()
    restore_pending = False  # restore start pose after Play (once a step has run)
    while simulation_app.is_running():
        now = _time.time()
        dt = 0.0 if last_t is None else (now - last_t)
        last_t = now
        is_playing = simulation_context.is_playing()
        if is_playing != was_playing:
            # Clear stale commands before the next step so they aren't re-applied.
            clear_artic_commands(real_graph_path, sim_graph_path)
            if is_playing:
                # Restore start pose only in sim (Isaac is the robot). In real mode
                # the /RealRobotGraph mirror re-drives Isaac from the live /joint_states,
                # so forcing a start pose would just fight the twin.
                restore_pending = (window._mode == "sim")
                window._append_log(
                    "[playback] resumed; cleared commands"
                    + ("; restoring start pose." if restore_pending else " (real: twin mirrors robot)."))
            else:
                window._append_log("[playback] paused/stopped; cleared command inputs.")
        was_playing = is_playing
        window.pump(dt)
        simulation_context.step(render=True)
        # After a step has bound the physics view, restore the configured start pose
        # (Isaac reset the robot to USD default on Stop). Retry until it succeeds.
        if restore_pending and is_playing:
            try:
                urctl.set_start_pose(articulation_root, JOINT_NAMES, _cfg.ROBOT_START_STATE)
                restore_pending = False
                window._append_log(
                    "[playback] start pose restored "
                    f"{np.rad2deg(_cfg.ROBOT_START_STATE).round(1).tolist()} deg.")
            except Exception:  # noqa: BLE001 — not ready yet; retry next frame
                pass

    simulation_context.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
