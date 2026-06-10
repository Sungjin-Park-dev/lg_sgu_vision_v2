#!/usr/bin/env python3
"""
Omni UI panel for the trajectory pipeline inside Isaac Sim.

Boots Isaac Sim with the same workcell as joint_control.py, then opens
an Omni UI window with three panels:

    A) plan_trajectory parameters + [Generate Trajectory]   (subprocess)
    B) CSV preview with Play/Pause/Stop/Slider           (in-process animation)
    C) publish_trajectory parameters + [Publish]         (subprocess)

The pipeline scripts run as `uv run` subprocesses to keep Isaac Sim's bundled
Python isolated from cuRobo / rclpy. Stdout streams into a scrolling log.

Preview overlays a pre-built physics-free "ghost" UR20 (built once via
scripts/isaac/usd/build_ghost_usd.py → ur20_description/ur20_ghost.usd,
cyan-tinted, no rigid bodies, no articulation) at /World/UR20_preview and
poses each link by writing one xformOp per frame via FK. The real
/World/UR20 articulation is never touched by preview, so the Action Graph
and any external /joint_states publisher keep running uninterrupted.
ActionGraphSwitch is kept for the publish path (ensure-graph-enabled) but
not toggled during preview.

Usage:
    uv run scripts/isaac/pipeline_ui.py --object sample
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Callable, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse loaders from joint_control.py — same workcell, robot, camera.
from isaac import joint_control as urctl  # noqa: E402

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
GHOST_USD_NAME = "ur20_ghost.usd"  # built by scripts/isaac/usd/build_ghost_usd.py


# =============================================================================
# Preview ghost — references a pre-built physics-free ghost USD (cyan-tinted,
# 30% opacity, all UsdPhysics/PhysxSchema APIs stripped, joints disabled).
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
# Preview player — drives ArticulationController.apply_action from the CSV.
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

class PipelineWindow:
    """Three-panel Omni UI window: Generate / Preview / Publish + Log."""

    LOG_MAX_LINES = 500

    def __init__(self, ghost_root_prim: str, base_link_path: str,
                 chain: "list[GhostJoint]", graph_path: str,
                 default_object: str):
        import omni.ui as ui

        self._ui = ui
        self._log_lines: list[str] = []
        self._log_model = ui.SimpleStringModel("")
        self._csv_path_model = ui.SimpleStringModel("")

        self._gen_runner = SubprocessRunner()
        self._pub_runner = SubprocessRunner()
        # Keep ActionGraphSwitch around for the publish path. Preview no
        # longer needs it (ghost is a separate prim tree, not the real UR20),
        # so we leave the graph untouched during preview — the user-confirmed
        # stable original idle behavior is preserved.
        self._graph = ActionGraphSwitch(graph_path, self._append_log)
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
        self._btn_publish = None
        self._btn_cancel_pub = None
        self._slider_model: Optional["ui.SimpleFloatModel"] = None
        self._status_label: Optional["ui.Label"] = None

        self._window = ui.Window("Pipeline UI", width=520, height=820)
        self._default_object = default_object
        self._build()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build(self):
        ui = self._ui
        with self._window.frame:
            with ui.VStack(spacing=6):
                self._build_panel_generate()
                self._build_panel_preview()
                self._build_panel_publish()
                self._build_log()

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

    def _build_panel_generate(self):
        ui = self._ui
        with ui.CollapsableFrame("A. Generate Trajectory (plan_trajectory.py)", height=0):
            with ui.VStack(spacing=4):
                self._fields["object"]            = self._row("--object",              self._default_object)
                self._fields["num_viewpoints"]    = self._row("--num-viewpoints",      124)
                self._fields["viewpoints"]        = self._row("--viewpoints (h5)",     "")
                self._fields["spacing"]           = self._row("--spacing",             0.01)
                self._fields["output_suffix"]     = self._row("--output-suffix",       "dp")
                with ui.HStack(height=28, spacing=6):
                    self._btn_generate = ui.Button("Generate Trajectory", clicked_fn=self._on_generate)
                    self._btn_cancel_gen = ui.Button("Cancel", clicked_fn=self._on_cancel_generate)

    def _build_panel_preview(self):
        ui = self._ui
        with ui.CollapsableFrame("B. Preview Trajectory", height=0):
            with ui.VStack(spacing=4):
                with ui.HStack(height=22, spacing=6):
                    ui.Label("CSV path", width=80)
                    ui.StringField(model=self._csv_path_model)
                    ui.Button("Browse...", width=80, clicked_fn=self._on_browse_csv)
                with ui.HStack(height=28, spacing=6):
                    ui.Button("Load & Preview", clicked_fn=self._on_load_preview)
                    ui.Button("Play", clicked_fn=self._on_play)
                    ui.Button("Pause", clicked_fn=self._on_pause)
                    ui.Button("Stop", clicked_fn=self._on_stop)
                with ui.HStack(height=22, spacing=6):
                    ui.Label("t", width=20)
                    self._slider_model = ui.SimpleFloatModel(0.0)
                    slider = ui.FloatSlider(self._slider_model, min=0.0, max=1.0)
                    slider.model.add_value_changed_fn(self._on_slider)
                self._status_label = ui.Label("t=0.00s / 0.00s  (no CSV)")

    def _build_panel_publish(self):
        ui = self._ui
        with ui.CollapsableFrame("C. Publish to Robot (publish_trajectory.py)", height=0):
            with ui.VStack(spacing=4):
                with ui.HStack(height=22, spacing=6):
                    ui.Label("CSV path", width=80)
                    ui.StringField(model=self._csv_path_model)
                    ui.Button("Browse...", width=80, clicked_fn=self._on_browse_csv)
                with ui.HStack(height=28, spacing=6):
                    self._btn_publish = ui.Button("Publish to Robot", clicked_fn=self._on_publish)
                    self._btn_cancel_pub = ui.Button("Cancel Publish", clicked_fn=self._on_cancel_publish)

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

    def _on_generate(self):
        if self._gen_runner.running:
            self._append_log("[generate] already running")
            return

        obj         = self._get_field("object", str).strip()
        n_vp        = self._get_field("num_viewpoints", int)
        viewpoints  = self._get_field("viewpoints", str).strip()
        spacing     = self._get_field("spacing", float)
        suffix      = self._get_field("output_suffix", str).strip() or "dp"

        cmd = [
            self._uv, "run", "scripts/pipeline/plan_trajectory.py",
            "--object", obj,
            "--num-viewpoints", str(n_vp),
            "--spacing", str(spacing),
            "--output-suffix", suffix,
        ]
        if viewpoints:
            cmd += ["--viewpoints", viewpoints]

        self._btn_generate.enabled = False
        self._append_log("[generate] $ " + " ".join(cmd))

        def on_line(line: str):
            self._append_log(line)
            m = CSV_PATH_RE.search(line)
            if m:
                csv = m.group(1)
                if not Path(csv).is_absolute():
                    csv = str(PROJECT_ROOT / csv)
                self._csv_path_model.set_value(csv)
                self._append_log(f"[generate] captured CSV: {csv}")

        def on_exit(rc: int):
            self._append_log(f"[generate] exit code = {rc}")
            self._btn_generate.enabled = True

        self._gen_runner.start(cmd, cwd=PROJECT_ROOT, on_line=on_line, on_exit=on_exit)

    def _on_cancel_generate(self):
        if self._gen_runner.running:
            self._append_log("[generate] terminating...")
            self._gen_runner.terminate()

    # ------------------------------------------------------------------
    # File picker (shared by panels B and C)
    # ------------------------------------------------------------------
    def _on_browse_csv(self):
        """Open Omni file picker pre-rooted at data/{object}/trajectory/."""
        # Pick a sensible starting directory: data/{object}/trajectory/{N}/ if it
        # exists, else data/{object}/trajectory/, else data/, else PROJECT_ROOT.
        current = self._csv_path_model.get_value_as_string().strip()
        if current and Path(current).parent.is_dir():
            start_dir = str(Path(current).parent)
        else:
            obj = self._get_field("object", str).strip() or "sample"
            n_vp = self._get_field("num_viewpoints", int)
            candidates = [
                PROJECT_ROOT / "data" / obj / "trajectory" / str(n_vp),
                PROJECT_ROOT / "data" / obj / "trajectory",
                PROJECT_ROOT / "data",
                PROJECT_ROOT,
            ]
            start_dir = str(next((p for p in candidates if p.is_dir()), PROJECT_ROOT))

        def _on_apply(filename: str, dirname: str):
            full = os.path.join(dirname, filename) if filename else dirname
            self._csv_path_model.set_value(full)
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
                "Select trajectory CSV",
                apply_button_label="Select",
                click_apply_handler=_on_apply,
                click_cancel_handler=_on_cancel,
                item_filter_options=["CSV (*.csv)", "All Files (*.*)"],
                item_filter_fn=lambda item: item.is_folder or item.path.endswith(".csv"),
                current_directory=start_dir,
            )
            dialog.show()
        except Exception as e:
            self._append_log(f"[browse] picker open failed: {e}")

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
        self._refresh_status()

    def _on_slider(self, model):
        if not self._preview.loaded:
            return
        self._preview.seek(model.get_value_as_float())
        self._refresh_status()

    def _update_slider_bounds(self):
        if self._slider_model is None:
            return
        # omni.ui.FloatSlider min/max are widget-level; we just clamp via seek().
        self._slider_model.set_value(0.0)

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
                self._slider_model.set_value(self._preview.state.t)
            self._refresh_status()

    # ------------------------------------------------------------------
    # Publish panel callbacks
    # ------------------------------------------------------------------
    def _on_publish(self):
        if self._pub_runner.running:
            self._append_log("[publish] already running")
            return
        csv = self._csv_path_model.get_value_as_string().strip()
        if not csv or not Path(csv).exists():
            self._append_log(f"[publish] CSV not found: {csv!r}")
            return

        # Re-enable the Action Graph so Isaac Sim mirrors /joint_states from the
        # real controller while the trajectory executes. If preview was left
        # running (or Stop wasn't pressed), the OnPlaybackTick is still disabled
        # and the UR20 in the viewport would appear frozen.
        if self._preview.state.playing:
            self._preview.stop()
        self._graph.set_active(True)
        self._append_log("[publish] Action Graph re-enabled for /joint_states mirroring")

        rd = os.environ.get("ROS_DISTRO")
        domain = os.environ.get("ROS_DOMAIN_ID", "(unset)")
        if rd:
            self._append_log(
                f"[publish] ROS_DISTRO={rd} ROS_DOMAIN_ID={domain} — "
                "Isaac was launched with ROS sourced; potential FastDDS conflict. Proceeding."
            )

        shell_cmd = (
            "source /opt/ros/jazzy/setup.bash && "
            f"exec {self._uv} run scripts/pipeline/publish_trajectory.py "
            f"--csv {csv!r}"
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
            self._append_log("[publish] terminating...")
            self._pub_runner.terminate()

    # ------------------------------------------------------------------
    # Per-frame pump
    # ------------------------------------------------------------------
    def pump(self, dt: float):
        self._gen_runner.pump()
        self._pub_runner.pump()
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
    )

    simulation_context.initialize_physics()
    simulation_context.play()

    last_t = None
    import time as _time
    while simulation_app.is_running():
        now = _time.time()
        dt = 0.0 if last_t is None else (now - last_t)
        last_t = now
        window.pump(dt)
        simulation_context.step(render=True)

    simulation_context.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
