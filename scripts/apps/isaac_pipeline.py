#!/usr/bin/env python3
"""
Omni UI panel for the trajectory pipeline inside Isaac Sim.

Boots Isaac Sim with the same workcell as scene.py, then opens
an Omni UI window with four panels:

    A) Load object (dropdown + native viewport gizmo move)
    B) plan_trajectory parameters + [Generate Trajectory]   (subprocess)
    C) Ghost preview with Play/Pause/Stop/Slider            (in-process; sim, ROS-free)
    D) publish_trajectory parameters + [Publish]            (subprocess; real mode only)

The pipeline scripts run as `uv run` subprocesses to keep Isaac Sim's bundled
Python isolated from cuRobo / rclpy. Stdout streams into a scrolling log.

Preview overlays a pre-built physics-free ghost UR20 with the camera attached
(built via scripts/isaac/usd/build_ghost_usd.py) at /World/UR20_preview and
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

# Reuse loaders from scene.py — same workcell, robot, camera.
from isaac import scene as urctl  # noqa: E402

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

CSV_PATH_RE = re.compile(r"CSV saved to (\S+)")

GHOST_ROOT_PATH = "/World/UR20_preview"
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
COLLISION_SPHERES_SCOPE_NAME = "CuRoboCollisionSpheres"
FOV_PLANE_SCOPE_NAME = "CameraFovPlane"
FOV_PLANE_OUTLINE_WIDTH_M = 0.003
FOV_PLANE_CENTERLINE_WIDTH_M = 0.002
CAMERA_COLLISION_LINKS = {
    "tool0",
    "camera_cable_frame",
    "camera_frame_1",
    "camera_frame_2",
    "camera_link",
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

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._queue: Queue = Queue()
        self._reader: Optional[threading.Thread] = None
        self._on_line: Optional[Callable[[str], None]] = None
        self._on_exit: Optional[Callable[[int], None]] = None
        self._done = True

    @property
    def running(self) -> bool:
        return not self._done

    def start(self, cmd, cwd, on_line, on_exit):
        if self.running:
            raise RuntimeError("SubprocessRunner already running")

        env = os.environ.copy()
        # Kit's PYTHONHOME/PATH leaks into children and confuses `uv run`.
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)
        env["PYTHONUNBUFFERED"] = "1"

        self._on_line = on_line
        self._on_exit = on_exit
        self._done = False
        self._proc = subprocess.Popen(
            cmd, cwd=str(cwd), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, universal_newlines=True,
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self):
        assert self._proc is not None
        try:
            for line in iter(self._proc.stdout.readline, ""):
                self._queue.put(line.rstrip("\n"))
        finally:
            self._proc.stdout.close()
            rc = self._proc.wait()
            self._queue.put(("__exit__", rc))

    def terminate(self):
        if not self.running or self._proc is None:
            return
        try:
            self._proc.terminate()
        except Exception:
            pass

    def pump(self):
        """Drain the queue, call on_line / on_exit on the UI thread."""
        if self._on_line is None:
            return
        try:
            while True:
                item = self._queue.get_nowait()
                if isinstance(item, tuple) and item and item[0] == "__exit__":
                    self._done = True
                    if self._on_exit is not None:
                        self._on_exit(int(item[1]))
                else:
                    self._on_line(str(item))
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
    return solutions, times


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
    ActionGraphSwitch and any external /joint_states publisher keep running
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

class ActionGraphSwitch:
    """Disable OnPlaybackTick while preview is active so the graph stops writing."""

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
    """Four-panel Omni UI window: Load / Generate / Preview / Publish + Log."""

    LOG_MAX_LINES = 500

    def __init__(self, ghost_root_prim: str, base_link_path: str,
                 chain: "list[GhostJoint]", graph_path: str,
                 default_object: str, initial_mode: str = "sim",
                 moveit_graph_path: str = "/MoveItGraph",
                 initial_pipeline_mode: str = "inspection",
                 articulation_root: str = ""):
        import omni.ui as ui

        self._ui = ui
        self._mode = initial_mode  # "sim" (no live ROS) | "real" (ROS robot)
        # Top-level mode: "inspection" (this whole UI) | "moveit" (MoveIt drives robot).
        self._pipeline_mode = initial_pipeline_mode
        self._log_lines: list[str] = []
        self._log_model = ui.SimpleStringModel("")
        self._csv_path_model = ui.SimpleStringModel("")
        self._h5_path_model = ui.SimpleStringModel("")

        self._gen_runner = SubprocessRunner()
        self._ik_runner = SubprocessRunner()
        self._pub_runner = SubprocessRunner()
        self._ctrl_runner = SubprocessRunner()   # ros2 control switch/cancel calls
        self._relay_runner = SubprocessRunner()  # ros2 param set on the relay (mode gate)
        # Keep ActionGraphSwitch around for the publish path. Preview no
        # longer needs it (ghost is a separate prim tree, not the real UR20),
        # so we leave the graph untouched during preview — the user-confirmed
        # stable original idle behavior is preserved.
        self._graph = ActionGraphSwitch(graph_path, self._append_log)
        # Separate switch for the MoveIt bridge graph (/isaac_joint_commands).
        # Only one of (_graph, _moveit_graph) ticks at a time — see apply_pipeline_mode.
        self._moveit_graph = ActionGraphSwitch(moveit_graph_path, self._append_log)
        self._graph_path = graph_path
        self._moveit_graph_path = moveit_graph_path
        self._articulation_root = articulation_root
        self._mode_applied: Optional[str] = None  # last run mode actually applied
        self._preview = PreviewPlayer(
            ghost_root_prim, base_link_path, chain, self._append_log,
        )

        self._uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
        if not Path(self._uv).exists() and shutil.which("uv") is None:
            self._append_log(f"[warn] uv binary not found on PATH; falling back to: {self._uv}")

        # Mutable field models (created in _build).
        self._fields: dict = {}
        self._btn_generate = None
        self._btn_cancel_gen = None
        self._btn_check_ik = None
        self._btn_cancel_ik = None
        self._btn_publish = None
        self._btn_cancel_pub = None
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
            return "● MoveIt (Inspection locked)"
        return "● Inspection"

    def _on_pipeline_mode_changed(self, *_):
        if self._pipeline_combo is None:
            return
        idx = self._pipeline_combo.model.get_item_value_model().get_value_as_int()
        self.apply_pipeline_mode("moveit" if idx == 1 else "inspection")

    def apply_pipeline_mode(self, mode: str):
        """Pipeline mode = command source + UI lock. It does NOT toggle any graph —
        the Run mode (sim/real) owns graph selection (see apply_mode). Both MoveIt
        (RViz move_group) and Inspection (Publish panel) ultimately command the same
        controller; pipeline mode only changes which tool the user drives with and
        locks the other.

        moveit     → Inspection panels greyed/locked (use RViz to drive the robot).
        inspection → Inspection panels active (Publish drives the current robot:
                     Isaac in sim, real robot in real).
        """
        self._pipeline_mode = mode
        self._set_inspection_ui_enabled(mode != "moveit")
        if mode == "inspection":
            self._sync_mode_ui()  # restore Publish-button state after unlock
        self._sync_pipeline_ui()
        # Activate exactly the controller for this mode so the OTHER source is
        # blocked at the controller level: MoveIt → scaled_joint_trajectory_controller,
        # Inspection → joint_trajectory_controller. In Inspection mode scaled is
        # deactivated, so MoveIt Execute is rejected ("controller not active").
        # (No-op in real mode if its stack uses the same controller names; best-effort
        # if the ROS stack isn't up yet.)
        if mode == "moveit":
            self._switch_controllers(MOVEIT_CONTROLLER, INSPECTION_CONTROLLER)
        else:
            self._switch_controllers(INSPECTION_CONTROLLER, MOVEIT_CONTROLLER)
        self._append_log(
            f"[pipeline] → {mode.upper()} :: "
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
        frame = ui.CollapsableFrame("Run Mode (sim / real)", height=0, collapsed=False)
        with frame:
            with ui.VStack(spacing=4):
                with ui.HStack(height=26, spacing=8):
                    ui.Label("Run mode", width=80)
                    idx = 0 if self._mode == "sim" else 1
                    self._mode_combo = ui.ComboBox(
                        idx, "sim (Isaac only)", "real (ROS robot)")
                    self._mode_combo.model.add_item_changed_fn(self._on_mode_changed)
                    self._mode_label = ui.Label(self._mode_text(), width=120)
                # ui.Label("sim = Isaac only, no ROS — A Load, B Generate, C Preview.  "
                #          "real = all of that + D Publish to the robot & mirror /joint_states "
                #          "(needs ur_robot_driver).",
                #          height=40, word_wrap=True)

    def _mode_text(self) -> str:
        return f"● {self._mode.upper()}"

    def _on_mode_changed(self, *_):
        if self._mode_combo is None:
            return
        idx = self._mode_combo.model.get_item_value_model().get_value_as_int()
        self.apply_mode("sim" if idx == 0 else "real")

    def apply_mode(self, mode: str):
        """Run mode = which robot drives the Isaac articulation:

          sim  → Isaac IS the robot: /MoveItGraph drives from /isaac_joint_commands
                 and publishes /isaac_joint_states + /clock; /ActionGraph mirror OFF.
          real → Isaac MIRRORS the real robot: /ActionGraph ON (drive from real
                 /joint_states + cameras); /MoveItGraph's driving + publishing nodes
                 OFF (so it neither moves Isaac nor feeds the twin loop).

        Cross-mode replay is stopped at the SOURCE: the relay (셸2) is the only thing
        that feeds /isaac_joint_commands, so the app sets its `forward_enabled`
        parameter — true in sim, false in real. In real mode the relay discards
        commands, so a MoveIt Execute done in real mode never reaches Isaac and there
        is nothing to replay when sim is re-entered (works even if the goal is still
        active). No rebuild / cancel / timing needed.

        sim↔real also requires relaunching the matching ROS stack (셸2).
        """
        self._mode = mode
        if mode == "sim":
            self._set_relay_forwarding(True)   # relay feeds Isaac
            self._set_moveit_driving(True)     # MoveIt drives Isaac + publishes state/clock
            clear_artic_commands(self._moveit_graph_path)
            self._graph.set_active(False)      # /ActionGraph mirror off
            which = "/MoveItGraph drives (live /isaac_joint_commands)"
        else:  # real
            self._set_relay_forwarding(False)  # relay discards → Isaac not driven by commands
            self._set_moveit_driving(False)    # MoveIt graph: no drive, no state/clock
            clear_artic_commands(self._graph_path)
            self._graph.set_active(True)       # /ActionGraph mirror on
            which = "/ActionGraph mirrors real /joint_states (twin)"
        self._mode_applied = mode
        self._sync_mode_ui()
        self._append_log(f"[run-mode] → {mode.upper()} :: {which}")

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
        self._append_log(f"[relay] forward_enabled → {val}")
        self._relay_runner.start(
            ["bash", "-c", cmd], cwd=PROJECT_ROOT, on_line=self._append_log,
            on_exit=lambda rc: self._append_log(f"[relay] param set exit={rc}"))

    # /MoveItGraph nodes that are "Isaac IS the robot" duties — enabled in sim,
    # disabled in real. SubscribeJointCommand is NOT here (harmless when idle). In
    # real mode, leaving PublishJointState on would feed /isaac_joint_states →
    # (셸2) /joint_states → /ActionGraph mirror → Isaac → /isaac_joint_states, a
    # self-driving loop that makes the robot shake; /clock belongs to sim time only.
    _MOVEIT_SIM_ONLY_NODES = ("ArticulationController", "PublishJointState", "PublishClock")

    def _set_moveit_driving(self, on: bool):
        """Enable (sim) / disable (real) /MoveItGraph's robot-driving + state/clock
        publishing nodes, without disabling the whole graph."""
        import omni.graph.core as og
        for name in self._MOVEIT_SIM_ONLY_NODES:
            try:
                node = og.Controller.node(f"{self._moveit_graph_path}/{name}")
                if node is not None and hasattr(node, "set_disabled"):
                    node.set_disabled(not on)
            except Exception as e:  # noqa: BLE001 — best effort
                self._append_log(f"[run-mode] toggle {name} failed: {e}")

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
            # Publish now works in both sim (→ Isaac) and real (→ real robot); it is
            # available in Inspection pipeline mode (locked in MoveIt mode).
            self._btn_publish.enabled = (
                self._pipeline_mode == "inspection") and not self._pub_runner.running

    def _build_panel_object(self):
        ui = self._ui
        frame = ui.CollapsableFrame("A. Load Object", height=0)
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
                # ui.Label("Pick an object and Load it, then move/rotate it with the viewport "
                #          "gizmo (W = move, E = rotate). Its live pose is read at Generate time.",
                #          height=28, word_wrap=True)

    def _build_panel_generate(self):
        ui = self._ui
        frame = ui.CollapsableFrame("B. Generate Trajectory (DP | GLNS)", height=0)
        self._inspection_frames.append(frame)
        with frame:
            with ui.VStack(spacing=4):
                # ui.Label("Pick the object's viewpoints .h5 and Generate. Object name + viewpoint "
                #          "count are read from the h5 path; the object's live pose comes from the "
                #          "scene — load & place it in panel A first.",
                #          height=40, word_wrap=True)
                with ui.HStack(height=26, spacing=8):
                    ui.Label("Planner backend", width=110)
                    # 0=DP (plan_trajectory), 1=GLNS (solve_glns_path → verify --join).
                    # Both emit the same trajectory CSV → preview/publish unchanged.
                    self._backend_combo = ui.ComboBox(
                        0, "DP (plan_trajectory)", "GLNS (solve + verify --join)")
                with ui.HStack(height=22, spacing=6):
                    ui.Label("Viewpoints (h5)", width=110)
                    self._lock(ui.StringField(model=self._h5_path_model))
                    self._lock(ui.Button("Browse...", width=80, clicked_fn=self._on_browse_h5))
                with ui.HStack(height=28, spacing=6):
                    self._lock(ui.Button("Show Viewpoints", clicked_fn=self._on_show_viewpoints))
                    self._lock(ui.Button("Clear Viewpoints", clicked_fn=self._on_clear_viewpoints))
                with ui.HStack(height=28, spacing=6):
                    self._btn_check_ik = self._lock(ui.Button(
                        "Check IK Reachability",
                        clicked_fn=self._on_check_ik_reachability,
                    ))
                    self._btn_cancel_ik = self._lock(ui.Button("Cancel IK Check", clicked_fn=self._on_cancel_ik))
                with ui.CollapsableFrame("Advanced", height=0, collapsed=True):
                    with ui.VStack(spacing=4):
                        self._fields["spacing"]       = self._row("--spacing",       0.01)
                        self._fields["output_suffix"] = self._row("--output-suffix", "dp")
                        self._fields["glns_hops"]     = self._row("--delaunay-expand-hops (GLNS)", 2)
                        self._fields["glns_roll_augment"] = self._row("--roll-augment (GLNS, 1/0)", 1)
                        self._fields["glns_tilt_augment"] = self._row("--tilt-augment (GLNS, 1/0)", 1)
                        self._fields["glns_tilt_angles"] = self._row("--tilt-angles-deg (GLNS)", "5 10")
                        self._fields["glns_tilt_azimuths"] = self._row("--tilt-azimuths (GLNS)", 8)
                        self._fields["glns_max_candidates"] = self._row(
                            "--max-candidates-per-viewpoint (GLNS)", 16)
                with ui.HStack(height=28, spacing=6):
                    self._btn_generate = self._lock(ui.Button("Generate Trajectory", clicked_fn=self._on_generate))
                    self._btn_cancel_gen = self._lock(ui.Button("Cancel", clicked_fn=self._on_cancel_generate))

    def _build_panel_preview(self):
        ui = self._ui
        frame = ui.CollapsableFrame("C. Preview in Simulation", height=0)
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
                with ui.HStack(height=28, spacing=6):
                    self._lock(ui.Button("Show FOV Plane", clicked_fn=self._on_show_fov_plane))
                    self._lock(ui.Button("Clear FOV Plane", clicked_fn=self._on_clear_fov_plane))
                with ui.HStack(height=22, spacing=6):
                    ui.Label("t", width=20)
                    self._slider_model = ui.SimpleFloatModel(0.0)
                    self._slider = self._lock(ui.FloatSlider(self._slider_model, min=0.0, max=1.0))
                    self._slider.model.add_value_changed_fn(self._on_slider)
                self._status_label = ui.Label("t=0.00s / 0.00s  (no CSV)")

    def _build_panel_publish(self):
        ui = self._ui
        frame = ui.CollapsableFrame("D. Publish to Real Robot (publish_trajectory.py)", height=0)
        self._inspection_frames.append(frame)
        with frame:
            with ui.VStack(spacing=4):
                # self._publish_hint_label = ui.Label(self._publish_hint_text(),
                #                                      height=28, word_wrap=True)
                with ui.HStack(height=22, spacing=6):
                    ui.Label("CSV path", width=80)
                    self._lock(ui.StringField(model=self._csv_path_model))
                    self._lock(ui.Button("Browse...", width=80, clicked_fn=self._on_browse_csv))
                with ui.HStack(height=28, spacing=6):
                    self._btn_publish = self._lock(ui.Button("Publish to Robot", clicked_fn=self._on_publish))
                    self._btn_cancel_pub = self._lock(ui.Button("Cancel Publish", clicked_fn=self._on_cancel_publish))

    def _publish_hint_text(self) -> str:
        if self._mode == "real":
            return "● REAL mode — sends the CSV to the live robot (ur_robot_driver required)."
        return ("⛔ Disabled in SIM mode — this is the only step that needs REAL. "
                "Switch Run mode to 'real' to publish to the robot.")

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
        if kind is str:
            return m.get_value_as_string()
        if kind is int:
            return m.get_value_as_int()
        if kind is float:
            return m.get_value_as_float()
        raise TypeError(kind)

    def _on_load_object(self):
        """Swap /World/target_object to the dropdown selection at its default pose."""
        idx = self._object_combo.model.get_item_value_model().get_value_as_int()
        obj = self._objects[idx]
        usd_path = PROJECT_ROOT / "data" / obj / "mesh" / "source.usd"
        if not usd_path.exists():
            self._append_log(
                f"[object] '{obj}' has no source.usd — build it once, then retry:\n"
                f"  uv run scripts/isaac/usd/build_object_usd.py --object {obj}")
            return
        self._append_log(f"[object] loading '{obj}' ...")
        try:
            urctl.load_target_object(obj)
        except Exception as e:
            self._append_log(f"[object] load failed: {e}")
            return
        self._current_object = obj
        self._append_log(
            f"[object] loaded '{obj}'. Move it with the viewport gizmo (W/E), then Generate.")

    def _on_log_object_pose(self):
        """Print the current object world orientation — feed it to reorient_mesh.py."""
        pose = self._read_object_world_pose()
        if pose is None:
            self._append_log("[object] no target prim on stage — Load Object first.")
            return
        (rx, ry, rz), (w, x, y, z) = pose
        obj = (self._current_object or "").strip() or "<name>"
        self._append_log(
            f"[object] world quat (w,x,y,z) = {w:.6f} {x:.6f} {y:.6f} {z:.6f}\n"
            f"[object] robot-frame pos = {rx:.4f} {ry:.4f} {rz:.4f}  "
            f"(world z = {rz + urctl.MOUNT_HEIGHT:.4f})\n"
            f"[object] bake upright: uv run scripts/prep/reorient_mesh.py --object {obj} "
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
        import h5py
        from common import config as _config

        with h5py.File(h5_path, "r") as f:
            if "viewpoints" not in f:
                raise ValueError("missing 'viewpoints' group")
            grp = f["viewpoints"]
            if "positions" not in grp or "normals" not in grp:
                raise ValueError("expected viewpoints/positions and viewpoints/normals")

            positions = np.array(grp["positions"], dtype=np.float64)
            normals = np.array(grp["normals"], dtype=np.float64)
            if positions.ndim != 2 or positions.shape[1] != 3:
                raise ValueError(f"positions must be shaped (N, 3), got {positions.shape}")
            if normals.shape != positions.shape:
                raise ValueError(
                    f"normals shape {normals.shape} does not match positions {positions.shape}")

            wd_m = float(_config.CAMERA_WORKING_DISTANCE_MM) / 1000.0
            if "metadata" in f and "camera_spec" in f["metadata"]:
                cs = f["metadata"]["camera_spec"]
                if "working_distance_mm" in cs.attrs:
                    wd_m = float(cs.attrs["working_distance_mm"]) / 1000.0

        n = np.linalg.norm(normals, axis=1, keepdims=True)
        safe_normals = np.divide(
            normals, n,
            out=np.zeros_like(normals),
            where=n > 1e-12,
        )
        return positions + safe_normals * wd_m, wd_m

    def _draw_camera_viewpoint_points(self, points_local, colors=None, opacity: float = 0.9):
        import omni.usd
        from pxr import Gf, UsdGeom, Vt

        stage = omni.usd.get_context().get_stage()
        self._delete_viewpoint_points(log=False)

        UsdGeom.Xform.Define(stage, VIEWPOINTS_ROOT_PRIM)
        points = UsdGeom.Points.Define(stage, VIEWPOINTS_POINTS_PRIM)
        points.CreatePointsAttr(Vt.Vec3fArray([
            Gf.Vec3f(float(p[0]), float(p[1]), float(p[2]))
            for p in points_local
        ]))
        points.CreateWidthsAttr(Vt.FloatArray(
            [VIEWPOINT_POINT_WIDTH_M] * len(points_local)
        ))

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
            self._append_log("[viewpoints] no target object on stage — Load Object first.")
            return

        try:
            points_local, wd_m = self._load_camera_viewpoint_points(h5)
        except Exception as e:
            self._append_log(f"[viewpoints] load failed: {e}")
            return
        from common import config as _config
        cfg_wd_m = float(_config.CAMERA_WORKING_DISTANCE_MM) / 1000.0
        if abs(wd_m - cfg_wd_m) > 1e-9:
            self._append_log(
                f"[viewpoints] WARNING: h5 working distance={wd_m * 1000:.1f} mm, "
                f"current config={cfg_wd_m * 1000:.1f} mm; using h5 metadata."
            )

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
                "[ik] no target object on stage — pick one in the Object dropdown "
                "and click 'Load Object' first."
            )
            return
        pos_robot, quat_wxyz = pose

        result_path = Path("/tmp") / (
            f"isaac_pipeline_ik_{os.getpid()}_{int(time.time() * 1000)}.json"
        )
        cmd = [
            self._uv, "run", "scripts/core/check_viewpoint_ik.py",
            "--object", obj,
            "--viewpoints", h5,
            "--output", str(result_path),
        ]
        if n_vp is not None:
            cmd += ["--num-viewpoints", str(n_vp)]
        cmd += ["--object-position", *(f"{v:.6f}" for v in pos_robot)]
        cmd += ["--object-quat", *(f"{v:.6f}" for v in quat_wxyz)]

        if self._btn_check_ik is not None:
            self._btn_check_ik.enabled = False
        self._append_log("[ik] $ " + " ".join(cmd))

        def on_line(line: str):
            self._append_log(line)

        def on_exit(rc: int):
            self._append_log(f"[ik] exit code = {rc}")
            if self._btn_check_ik is not None:
                self._btn_check_ik.enabled = True
            if rc == 0:
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
            if p.startswith(robot_root) and prim.GetName() == prim_name:
                return prim
        return None

    def _delete_fov_plane(self, log: bool):
        import omni.usd
        from isaacsim.core.utils import prims

        stage = omni.usd.get_context().get_stage()
        paths = [
            str(prim.GetPath())
            for prim in stage.Traverse()
            if prim.GetName() == FOV_PLANE_SCOPE_NAME
        ]
        for path in sorted(paths, key=len, reverse=True):
            prims.delete_prim(path)
        if log:
            self._append_log(
                f"[fov] cleared {len(paths)} FOV plane scope(s)"
                if paths else "[fov] nothing to clear"
            )

    def _on_show_fov_plane(self):
        import omni.usd
        from pxr import Gf, UsdGeom, Vt
        from common import config as _config

        stage = omni.usd.get_context().get_stage()
        robot_root = GHOST_ROOT_PATH
        if not stage.GetPrimAtPath(robot_root).IsValid():
            self._append_log(f"[fov] ghost robot not found: {robot_root}")
            return

        camera_frame = self._find_prim_by_name(
            stage, robot_root, urctl.CAMERA_OPTICAL_FRAME_NAME,
        )
        if camera_frame is None or not camera_frame.IsValid():
            self._append_log(
                f"[fov] {urctl.CAMERA_OPTICAL_FRAME_NAME} not found under {robot_root}"
            )
            return

        fov_w_m = float(_config.CAMERA_FOV_WIDTH_MM) / 1000.0
        fov_h_m = float(_config.CAMERA_FOV_HEIGHT_MM) / 1000.0
        wd_m = float(_config.CAMERA_WORKING_DISTANCE_MM) / 1000.0
        if fov_w_m <= 0.0 or fov_h_m <= 0.0 or wd_m <= 0.0:
            self._append_log(
                "[fov] invalid camera spec: "
                f"FOV={_config.CAMERA_FOV_WIDTH_MM}x{_config.CAMERA_FOV_HEIGHT_MM} mm, "
                f"WD={_config.CAMERA_WORKING_DISTANCE_MM} mm"
            )
            return

        self._delete_fov_plane(log=False)

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
            f"[fov] displayed {_config.CAMERA_FOV_WIDTH_MM:.1f}x"
            f"{_config.CAMERA_FOV_HEIGHT_MM:.1f} mm plane at "
            f"WD={_config.CAMERA_WORKING_DISTANCE_MM:.1f} mm under {camera_frame.GetPath()}"
        )

    def _on_clear_fov_plane(self):
        self._delete_fov_plane(log=True)

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
                "object — load the matching object or pick the matching h5.")

        spacing = self._get_field("spacing", float)
        suffix  = self._get_field("output_suffix", str).strip() or "dp"

        # Read the object's live world pose (gizmo-moved) and pass it to the
        # planner. No silent fallback: if there's no target prim, abort so we
        # never plan against a stale config pose.
        pose = self._read_object_world_pose()
        if pose is None:
            self._append_log(
                "[generate] no target object on stage — pick one in the Object "
                "dropdown and click 'Load Object' first.")
            return
        pos_robot, quat_wxyz = pose

        backend_idx = 0
        combo = getattr(self, "_backend_combo", None)
        if combo is not None:
            backend_idx = combo.model.get_item_value_model().get_value_as_int()

        if backend_idx == 1:
            # GLNS backend: solve_glns_path → verify_glns_trajectory --join, chained in
            # one shell (publish-style bash -c). Both stages stream stdout; verify prints
            # the joined "CSV saved to ..." LAST, so CSV_PATH_RE captures the joined
            # trajectory (same 14-col schema as DP → preview/publish need no change).
            hops = max(1, int(self._get_field("glns_hops", int)))
            augment = ""
            if int(self._get_field("glns_roll_augment", int)) != 0:
                augment += " --roll-augment"
            if int(self._get_field("glns_tilt_augment", int)) != 0:
                angles = " ".join(
                    str(float(x)) for x in self._get_field("glns_tilt_angles", str).split())
                azimuths = max(1, int(self._get_field("glns_tilt_azimuths", int)))
                augment += (f" --tilt-augment --tilt-angles-deg {angles}"
                            f" --tilt-azimuths {azimuths}")
            max_candidates = max(1, int(self._get_field("glns_max_candidates", int)))
            augment += f" --max-candidates-per-viewpoint {max_candidates}"
            det_h5 = f"data/{obj}/ik/{n_vp}/glns_result_gui.h5"
            pos_s = " ".join(f"{v:.6f}" for v in pos_robot)
            quat_s = " ".join(f"{v:.6f}" for v in quat_wxyz)
            shell = (
                f"{self._uv} run scripts/core/solve_glns_path.py "
                f"--object {obj!r} --viewpoints {h5!r} "
                f"--object-position {pos_s} --object-quat {quat_s} "
                f"--delaunay-expand-hops {hops}{augment} --output {det_h5!r} "
                f"&& {self._uv} run scripts/core/verify_glns_trajectory.py "
                f"--result {det_h5!r} --join --require-full-coverage --spacing {spacing}"
            )
            cmd = ["bash", "-c", shell]
        else:
            # DP backend (default): plan_trajectory.py end-to-end.
            cmd = [
                self._uv, "run", "scripts/core/plan_trajectory.py",
                "--object", obj,
                "--num-viewpoints", str(n_vp),
                "--viewpoints", h5,
                "--spacing", str(spacing),
                "--output-suffix", suffix,
                "--object-position", *(f"{v:.6f}" for v in pos_robot),
                "--object-quat", *(f"{v:.6f}" for v in quat_wxyz),
            ]

        self._btn_generate.enabled = False
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
            self._append_log(f"[generate] exit code = {rc}")
            self._btn_generate.enabled = True
            if rc == 0 and generated_csv_path:
                csv = generated_csv_path[0]
                if self._preview.load(csv):
                    self._update_slider_bounds()
                    self._refresh_status()
                    self._append_log(f"[preview] auto-loaded generated CSV: {csv}")

        self._gen_runner.start(cmd, cwd=PROJECT_ROOT, on_line=on_line, on_exit=on_exit)

    def _on_cancel_generate(self):
        if self._gen_runner.running:
            self._append_log("[generate] terminating...")
            self._gen_runner.terminate()

    # ------------------------------------------------------------------
    # File picker (shared by panels B and C)
    # ------------------------------------------------------------------
    def _open_file_picker(self, title: str, model, item_label: str, ext: str, start_dir: str):
        """Open the Omni file picker filtered to `ext`, writing the pick into `model`."""
        def _on_apply(filename: str, dirname: str):
            full = os.path.join(dirname, filename) if filename else dirname
            model.set_value(full)
            self._append_log(f"[browse] selected: {full}")
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
        self._open_file_picker("Select viewpoints .h5", self._h5_path_model,
                               "HDF5 (*.h5)", ".h5", start_dir)

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
    # Publish panel callbacks
    # ------------------------------------------------------------------
    def _on_publish(self):
        # Publish drives the CURRENT robot via the trajectory controller — in sim it
        # routes to Isaac (sim stack), in real to the real robot. Works in both Run
        # modes. The Run mode already selected the correct Isaac graph, so we do NOT
        # toggle graphs here. (The button is UI-locked in MoveIt pipeline mode.)
        if self._pub_runner.running:
            self._append_log("[publish] already running")
            return
        csv = self._csv_path_model.get_value_as_string().strip()
        if not csv or not Path(csv).exists():
            self._append_log(f"[publish] CSV not found: {csv!r}")
            return

        # Ghost preview and Publish both move things in the viewport; stop the ghost
        # so the two don't fight.
        if self._preview.state.playing:
            self._preview.stop()

        # Route by Run mode so Inspection is independent of which 셸2 stack is up:
        #   sim  → stream straight to Isaac (/isaac_joint_commands); no 셸2 needed.
        #   real → send to the real robot's trajectory controller (needs real 셸2).
        target = "isaac" if self._mode == "sim" else "controller"
        shell_cmd = (
            "source /opt/ros/jazzy/setup.bash && "
            f"exec {self._uv} run --no-sync scripts/core/publish_trajectory.py "
            f"--csv {csv!r} --target {target}"
        )
        cmd = ["bash", "-c", shell_cmd]

        self._btn_publish.enabled = False
        self._append_log("[publish] $ " + shell_cmd)

        def on_line(line: str):
            self._append_log(line)

        def on_exit(rc: int):
            self._append_log(f"[publish] exit code = {rc}")
            self._btn_publish.enabled = True

        self._pub_runner.start(cmd, cwd=PROJECT_ROOT, on_line=on_line, on_exit=on_exit)

    def _on_cancel_publish(self):
        if self._pub_runner.running:
            self._append_log("[publish] terminating publisher...")
            self._pub_runner.terminate()
        if self._mode == "sim":
            # sim streams directly to /isaac_joint_commands — killing the publisher
            # stops the stream immediately and Isaac holds the last commanded pose.
            # There is no controller goal to cancel.
            self._append_log("[publish] sim stream stopped (Isaac holds current pose)")
            return
        # real: the controller already holds the whole trajectory goal and keeps
        # executing it, so terminating the publisher is not enough — cancel the goal.
        shell_cmd = (
            "source /opt/ros/jazzy/setup.bash && "
            f"timeout 3 ros2 service call /{INSPECTION_CONTROLLER}/follow_joint_trajectory"
            "/_action/cancel_goal action_msgs/srv/CancelGoal '{}'"
        )
        self._append_log(f"[publish] cancelling goals on {INSPECTION_CONTROLLER}")
        self._ctrl_runner.start(
            ["bash", "-c", shell_cmd], cwd=PROJECT_ROOT,
            on_line=self._append_log,
            on_exit=lambda rc: self._append_log(f"[publish] cancel exit={rc}"))

    # ------------------------------------------------------------------
    # Per-frame pump
    # ------------------------------------------------------------------
    def pump(self, dt: float):
        self._gen_runner.pump()
        self._ik_runner.pump()
        self._pub_runner.pump()
        self._ctrl_runner.pump()
        self._relay_runner.pump()
        # While a Publish trajectory is executing, lock BOTH mode combos so the user
        # can't switch pipeline mode (would deactivate the inspection controller and
        # abort the trajectory) OR run mode (sim/real) mid-execution.
        executing = self._pub_runner.running
        if self._pipeline_combo is not None:
            self._pipeline_combo.enabled = not executing
        if self._mode_combo is not None:
            self._mode_combo.enabled = not executing
        self.step_preview(dt)


# =============================================================================
# Main
# =============================================================================

def main():
    args = urctl.parse_args()
    if not args.usd_path.exists():
        sys.exit(f"Robot USD not found: {args.usd_path}")

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

    graph_path = urctl.build_action_graph(articulation_root, inspection_cam)
    simulation_app.update()

    # Separate MoveIt bridge graph (/isaac_joint_commands → robot, robot → /isaac_joint_states).
    # Gated independently from the inspection graph by the top-level pipeline mode.
    moveit_graph_path = urctl.build_moveit_graph(articulation_root)
    simulation_app.update()

    # Physics-free ghost overlay for trajectory preview. Built once offline
    # by scripts/isaac/usd/build_ghost_usd.py — referencing it here
    # should add zero physics state and leave the real /World/UR20
    # articulation untouched.
    ghost_usd_path = args.usd_path.parent / GHOST_USD_NAME
    if not ghost_usd_path.exists():
        sys.exit(
            f"Ghost USD not found: {ghost_usd_path}\n"
            f"Build it first: uv run scripts/isaac/usd/build_ghost_usd.py"
        )
    base_link, chain = spawn_preview_ghost(
        usd_path=ghost_usd_path,
        ghost_root=GHOST_ROOT_PATH,
        position=np.array([0.0, 0.0, urctl.MOUNT_HEIGHT]),
        joint_order=JOINT_NAMES,
        log=print,
    )
    simulation_app.update()

    window = PipelineWindow(
        ghost_root_prim=GHOST_ROOT_PATH,
        base_link_path=base_link,
        chain=chain,
        graph_path=graph_path,
        default_object=(args.object or "sample"),
        initial_mode=args.mode,
        moveit_graph_path=moveit_graph_path,
        initial_pipeline_mode=args.pipeline_mode,
        articulation_root=articulation_root,
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
            clear_artic_commands(graph_path, moveit_graph_path)
            if is_playing:
                # Restore start pose only in sim (Isaac is the robot). In real mode
                # the /ActionGraph mirror re-drives Isaac from the live /joint_states,
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
