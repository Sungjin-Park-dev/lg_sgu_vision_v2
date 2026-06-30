#!/usr/bin/env python3
"""Unified viser trajectory studio — place object, inspect live IK, generate (DP|GLNS), play back.

One browser tool that replicates isaac_pipeline.py Panels A/B/C without Isaac Sim:
  (A) Load an object + viewpoints, move the object with a gizmo.
  (B) See the LIVE in-process cuRobo IK distribution as you place it — per-viewpoint
      representative IK branches (green=collision-free / red=collision) and, on "Apply
      pose", an aggregate reachability sweep that recolors every viewpoint green/red.
  (C) Generate a collision-free trajectory via the GLNS backend (solve + verify --join)
      OR the DP backend (plan_trajectory.py), then play it back densely (transit=red /
      scan=green).

The live IK reuses plan_trajectory's robot_cfg / collision world / wrist_3 lock /
batch_collision_check (via ik_inspector.IKBackend), so a viewpoint that shows 0
collision-free reps here is exactly the one DP/GLNS will drop (cross-validation).

Generation runs the same headless core scripts as a subprocess (isolated cuRobo
process); GLNS reloads the rich glns_result h5 (Delaunay graph + components + reconfig),
DP reloads its npz sidecar. Both backends emit {joints, ee_positions, is_transit, times},
so dense playback is identical.

Publish to the real robot is intentionally NOT here — use isaac_pipeline.py /
publish_trajectory.py with the generated CSV.

사용법:
    uv run --no-sync scripts/apps/trajectory_studio.py --object sample
    uv run --no-sync scripts/apps/trajectory_studio.py --result data/sample/ik/74/glns_result_X.h5
"""

from __future__ import annotations

import argparse
import colorsys
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

import h5py
import numpy as np
import torch
import trimesh
import viser
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT / "core"))
sys.path.insert(0, str(SCRIPTS_ROOT / "apps"))

from common import config  # noqa: E402
from common.glns_utils import read_result_hdf5  # noqa: E402
import plan_trajectory as PT  # noqa: E402
from ik_inspector import (  # noqa: E402
    MAX_REP_SLIDER, RobotViz, IKBackend, discover_objects, discover_viewpoints,
)

DATA_ROOT = PROJECT_ROOT / "data"
OBJ_NODE = "/studio/object"          # 물체 이동 gizmo (mesh 는 자식 → 드래그하면 따라온다)

COLOR_OBJECT = (90, 200, 255)
COLOR_OBSTACLE = (120, 120, 130)
COLOR_DROPPED = (180, 70, 70)
COLOR_SELECTED = (255, 220, 50)
COLOR_DELAUNAY = (0, 180, 220)
COLOR_PATH = (70, 210, 120)
COLOR_RECONFIG = (240, 80, 70)
COLOR_VP_ALL = (110, 120, 140)
COLOR_VP_SEL = (255, 210, 60)
COLOR_REACH_FREE = (70, 210, 120)
COLOR_REACH_BLOCKED = (224, 96, 88)
COLOR_ROBOT_FREE = (170, 174, 184)
COLOR_ROBOT_COLLIDE = (224, 96, 88)


def _wxyz_from_matrix(R: np.ndarray) -> np.ndarray:
    """3x3 회전행렬 → (w, x, y, z) quaternion."""
    q = Rotation.from_matrix(R).as_quat()        # (x, y, z, w)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _decode_attr(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, np.generic):
        return value.item()
    return value


class SubprocessRunner:
    """Background subprocess → stdout 라인을 큐로. UI 루프가 매 프레임 pump() 로 비운다."""

    def __init__(self):
        self._proc = None
        self._queue: Queue = Queue()
        self._on_line = None
        self._on_exit = None
        self._done = True

    @property
    def running(self) -> bool:
        return not self._done

    def start(self, cmd, cwd, on_line, on_exit):
        if self.running:
            raise RuntimeError("SubprocessRunner already running")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self._on_line, self._on_exit, self._done = on_line, on_exit, False
        self._proc = subprocess.Popen(
            cmd, cwd=str(cwd), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, universal_newlines=True,
        )
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _read_loop(self):
        try:
            for line in iter(self._proc.stdout.readline, ""):
                self._queue.put(line.rstrip("\n"))
        finally:
            self._proc.stdout.close()
            rc = self._proc.wait()
            self._queue.put(("__exit__", rc))

    def pump(self):
        """UI 스레드에서 큐를 비워 on_line/on_exit 호출."""
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


def discover_results() -> dict[str, Path]:
    results = {}
    for path in sorted(PROJECT_ROOT.glob("data/*/ik/*/glns_result*.h5")):
        results[str(path.relative_to(PROJECT_ROOT))] = path.resolve()
    return results


def _component_color(component_id: int) -> tuple[int, int, int]:
    h = (component_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
    return int(255 * r), int(255 * g), int(255 * b)


class TrajectoryStudio:
    """Object 배치 + 라이브 IK 분포 + DP/GLNS 경로 생성 + dense 재생을 한 viser 앱으로."""

    def __init__(self, server, objects, initial_object, result_paths, initial_result=None):
        self.server = server
        self.objects = objects
        self.object_name = initial_object
        self.result_paths = result_paths

        self.backend = IKBackend(initial_object)             # 라이브 in-process cuRobo IK
        self.robot = RobotViz(server, self.backend.urdf_path)
        # 물체 기본 pose (리셋용) — 어떤 변경보다 먼저 캡처
        self._obj_pose0 = (
            np.asarray(config.TARGET_OBJECT["position"], dtype=float).copy(),
            np.asarray(config.TARGET_OBJECT["rotation"], dtype=float).copy(),
        )

        # ── 상태 ──────────────────────────────────────────────────────────
        self.result = None              # GLNS 결과 dict (rich h5) 또는 None
        self.result_kind = None         # "glns" | "dp" | None
        self.result_h5_path = None
        self.dense_dir = None           # glns_trajectory_*.npz 디렉토리
        self.world_poses = None         # (N,4,4) base_link 프레임 카메라 포즈
        self._vp_raw = None             # (positions, normals, wd_m) — 물체 이동 시 재계산
        self._vp_path = None            # 로드된 viewpoints h5 경로 (생성 입력)
        self._reach = None              # (N,) bool — aggregate reachability 결과
        self.reps = np.empty((0, 6))
        self.colliding = np.empty((0,), dtype=bool)
        self.current_component = None
        self.dense = None               # {joints, ee_pos, is_transit}
        self._dense_label = ""
        self.play_position = 0.0
        self.obj_gizmo = None
        self.component_by_label = {}
        self._suppress_object_update = False   # 프로그램적 dd_object.value 설정 시 on_update 무시
        self.runner = SubprocessRunner()
        self._pending_out = None        # GLNS 결과 h5
        self._pending_kind = None
        self._pending_npz = None        # DP npz
        self._pending_csv = None
        self._log_lines = []
        self._src = "target"
        self.layers = {
            "object": [], "obstacles": [], "points": [], "delaunay": [], "path": [],
        }
        self._vp_sel = None             # viewpoint 프로브 선택 강조
        self.highlight = None           # 재생 step 강조
        self.camera_frame = None
        self.step_slider = None
        self.sl_vp = None

        self._build_gui(initial_result)
        self._build_object_scene(self.object_name, *self._obj_pose0)
        self.robot.set_config(config.ROBOT_START_STATE)
        if initial_result is not None and initial_result in self.result_paths:
            self._load_result_path(self.result_paths[initial_result])

    # ── GUI ────────────────────────────────────────────────────────────────
    def _build_gui(self, initial_result):
        gui = self.server.gui
        with gui.add_folder("Scene"):
            self.dd_object = gui.add_dropdown(
                "Object", options=self.objects, initial_value=self.object_name,
            )
        self.folder_vp = gui.add_folder("Viewpoints")
        with self.folder_vp:
            vps = discover_viewpoints(DATA_ROOT, self.object_name)
            self.dd_vp = gui.add_dropdown(
                "h5", options=list(vps.keys()) or ["(none)"],
                initial_value=next(iter(vps), "(none)"),
            )
            self.btn_load_vp = gui.add_button("Load viewpoints")
        self._make_vp_slider(1)
        with gui.add_folder("Move object + live IK"):
            self.move_object = gui.add_checkbox("Show object gizmo", initial_value=True)
            self.btn_apply = gui.add_button("Apply pose → recompute IK")
            self.btn_reset = gui.add_button("Reset object pose")
            self.sl_sol = gui.add_slider(
                "Solution k", min=0, max=MAX_REP_SLIDER, step=1, initial_value=0,
            )
            self.reach_md = gui.add_markdown("reachability: (Apply pose)")
        with gui.add_folder("Generate (DP | GLNS)"):
            self.backend_dd = gui.add_dropdown(
                "Backend",
                options=["GLNS (solve + verify --join)", "DP (plan_trajectory)"],
                initial_value="GLNS (solve + verify --join)",
            )
            with gui.add_folder("GLNS Advanced"):
                self.expand_hops = gui.add_number(
                    "Delaunay expand hops", initial_value=2, min=1, max=4, step=1,
                )
                self.roll_augment = gui.add_checkbox("Roll augment", initial_value=True)
                self.tilt_augment = gui.add_checkbox("Tilt augment", initial_value=True)
                self.tilt_angles = gui.add_text("Tilt angles deg", initial_value="5 10")
                self.tilt_azimuths = gui.add_number(
                    "Tilt azimuths", initial_value=8, min=1, max=32, step=1,
                )
                self.max_candidates = gui.add_number(
                    "Max candidates/viewpoint", initial_value=16, min=1, max=64, step=1,
                )
                self.ik_num_seeds = gui.add_number(
                    "IK seeds/pose", initial_value=32, min=1, max=200, step=1,
                )
                self.ik_batch_size = gui.add_number(
                    "IK pose batch size", initial_value=16, min=1, max=128, step=1,
                )
            self.spacing = gui.add_number(
                "Spacing (m)", initial_value=0.01, min=0.002, max=0.05, step=0.001,
            )
            self.suffix = gui.add_text("Output suffix (DP)", initial_value="dp")
            self.btn_gen = gui.add_button("Generate trajectory")
            self.metrics_md = gui.add_markdown("metrics: (GLNS result)")
            self.gen_log = gui.add_markdown("")
        with gui.add_folder("GLNS result (load existing)"):
            opts = list(self.result_paths) or ["(none)"]
            init_r = initial_result if initial_result in self.result_paths else opts[0]
            self.result_dropdown = gui.add_dropdown(
                "GLNS HDF5", options=opts, initial_value=init_r,
            )
            self.load_button = gui.add_button("Load result")
        self.component_folder = gui.add_folder("Result / Playback")
        with self.component_folder:
            self.component_dropdown = gui.add_dropdown(
                "Run", options=["(none)"], initial_value="(none)",
            )
            self.play_checkbox = gui.add_checkbox("Play", initial_value=False)
            self.speed_slider = gui.add_slider(
                "Speed (steps/s)", min=0.25, max=20.0, step=0.25, initial_value=3.0,
            )
            self.show_dense = gui.add_checkbox(
                "Dense trajectory (NPZ)", initial_value=True,
            )
        self._make_step_slider(1)
        with gui.add_folder("Layers"):
            self.show_object = gui.add_checkbox("Object", initial_value=True)
            self.show_obstacles = gui.add_checkbox("Obstacles", initial_value=False)
            self.show_points = gui.add_checkbox("Viewpoints", initial_value=True)
            self.show_delaunay = gui.add_checkbox("Delaunay graph", initial_value=False)
            self.show_path = gui.add_checkbox("Path / trajectory", initial_value=True)
        self.status = gui.add_markdown("Load an object, then viewpoints.")

        self.dd_object.on_update(lambda _: self._on_object_change())
        self.btn_load_vp.on_click(lambda _: self._load_viewpoints())
        self.btn_apply.on_click(lambda _: self._apply_object_pose())
        self.btn_reset.on_click(lambda _: self._reset_object_pose())
        self.sl_sol.on_update(lambda _: self._show_solution())
        self.move_object.on_update(lambda _: self._toggle_gizmo())
        self.btn_gen.on_click(lambda _: self._on_generate())
        self.load_button.on_click(lambda _: self._load_selected_result_dropdown())
        self.component_dropdown.on_update(lambda _: self._on_component_change())
        self.show_dense.on_update(lambda _: self._on_component_change())
        for checkbox in (
            self.show_object, self.show_obstacles, self.show_points,
            self.show_delaunay, self.show_path,
        ):
            checkbox.on_update(lambda _: self._apply_visibility())

    def _make_vp_slider(self, count: int):
        if self.sl_vp is not None:
            self.sl_vp.remove()
        with self.folder_vp:
            self.sl_vp = self.server.gui.add_slider(
                "Viewpoint idx", min=0, max=max(int(count) - 1, 1), step=1, initial_value=0,
            )
        self.sl_vp.on_update(lambda _: self._on_vp_select())

    def _make_step_slider(self, count: int):
        if self.step_slider is not None:
            self.step_slider.remove()
        with self.component_folder:
            self.step_slider = self.server.gui.add_slider(
                "Path step", min=0, max=max(int(count) - 1, 1), step=1, initial_value=0,
            )
        self.step_slider.on_update(lambda _: self._show_step())

    def _apply_visibility(self):
        mapping = {
            "object": self.show_object,
            "obstacles": self.show_obstacles,
            "points": self.show_points,
            "delaunay": self.show_delaunay,
            "path": self.show_path,
        }
        for name, checkbox in mapping.items():
            for handle in self.layers[name]:
                handle.visible = checkbox.value

    def _log(self, line: str):
        self._log_lines.append(line)
        self._log_lines = self._log_lines[-18:]
        self.gen_log.content = "```\n" + "\n".join(self._log_lines) + "\n```"

    # ── 정적 씬 (물체 + 장애물) ───────────────────────────────────────────────
    def _build_object_scene(self, object_name, position, quat):
        position = np.asarray(position, dtype=float)
        quat = np.asarray(quat, dtype=float)
        if self.obj_gizmo is None:
            self.obj_gizmo = self.server.scene.add_transform_controls(
                OBJ_NODE, scale=0.2, position=position.copy(), wxyz=quat.copy(),
                visible=self.move_object.value,
            )
        else:
            self.obj_gizmo.position = position.copy()
            self.obj_gizmo.wxyz = quat.copy()
        # mesh (gizmo 자식 → 드래그하면 시각적으로 따라온다)
        for handle in self.layers["object"]:
            handle.remove()
        self.layers["object"].clear()
        mesh_path = config.get_mesh_path(object_name, mesh_type="source")
        if mesh_path.exists():
            loaded = trimesh.load(str(mesh_path), force="mesh")
            if isinstance(loaded, trimesh.Scene):
                loaded = trimesh.util.concatenate(list(loaded.geometry.values()))
            self.layers["object"].append(self.server.scene.add_mesh_simple(
                f"{OBJ_NODE}/mesh", np.asarray(loaded.vertices), np.asarray(loaded.faces),
                color=COLOR_OBJECT, opacity=0.4, side="double",
            ))
        for handle in self.layers["obstacles"]:
            handle.remove()
        self.layers["obstacles"].clear()
        config.sync_support_to_target()
        for obstacle in [config.TABLE, config.ROBOT_MOUNT] + list(config.WALLS):
            box = trimesh.creation.box(extents=np.asarray(obstacle["dimensions"], dtype=float))
            self.layers["obstacles"].append(self.server.scene.add_mesh_simple(
                f"/studio/obstacles/{obstacle['name']}",
                np.asarray(box.vertices), np.asarray(box.faces),
                color=COLOR_OBSTACLE, opacity=0.18, side="double",
                position=np.asarray(obstacle["position"], dtype=float),
            ))
        self._apply_visibility()

    def _draw_plain_viewpoints(self):
        """Object 모드 viewpoint 점군 — reachability 있으면 green/red, 없으면 neutral."""
        for handle in self.layers["points"]:
            handle.remove()
        self.layers["points"].clear()
        if self.world_poses is None:
            return
        cam = self.world_poses[:, :3, 3]
        if self._reach is None:
            colors = np.tile(np.asarray(COLOR_VP_ALL, dtype=np.uint8), (len(cam), 1))
        else:
            colors = np.empty((len(cam), 3), dtype=np.uint8)
            colors[self._reach] = COLOR_REACH_FREE
            colors[~self._reach] = COLOR_REACH_BLOCKED
        self.layers["points"].append(self.server.scene.add_point_cloud(
            "/studio/viewpoints", cam, colors, point_size=0.008, point_shape="circle",
        ))
        self._apply_visibility()

    def _draw_component_viewpoints(self):
        """GLNS 모드 viewpoint 점군 — component 별 색 + dropped 빨강."""
        for handle in self.layers["points"]:
            handle.remove()
        self.layers["points"].clear()
        cam = self.world_poses[:, :3, 3]
        labels = self.result["component_id"]
        for cid in sorted(int(x) for x in np.unique(labels) if x >= 0):
            indices = np.where(labels == cid)[0]
            color = _component_color(cid)
            self.layers["points"].append(self.server.scene.add_point_cloud(
                f"/studio/viewpoints/c{cid}", cam[indices],
                np.tile(np.asarray(color, dtype=np.uint8), (len(indices), 1)),
                point_size=0.007, point_shape="circle",
            ))
        dropped = np.where(~self.result["reachable_mask"])[0]
        if len(dropped):
            self.layers["points"].append(self.server.scene.add_point_cloud(
                "/studio/viewpoints/dropped", cam[dropped],
                np.tile(np.asarray(COLOR_DROPPED, dtype=np.uint8), (len(dropped), 1)),
                point_size=0.008, point_shape="circle",
            ))

    def _draw_delaunay(self):
        for handle in self.layers["delaunay"]:
            handle.remove()
        self.layers["delaunay"].clear()
        cam = self.world_poses[:, :3, 3]
        for edge_index, (a, b) in enumerate(self.result["induced_edges"]):
            self.layers["delaunay"].append(self.server.scene.add_spline_catmull_rom(
                f"/studio/delaunay/e{edge_index}",
                positions=np.stack([cam[a], cam[b]]),
                color=COLOR_DELAUNAY, line_width=1.0,
            ))

    # ── 물체 / 씬 전환 ───────────────────────────────────────────────────────
    def _toggle_gizmo(self):
        if self.obj_gizmo is not None:
            self.obj_gizmo.visible = self.move_object.value

    def _on_object_change(self):
        if self._suppress_object_update:
            return
        self.object_name = self.dd_object.value
        config.apply_object_placement(self.object_name)
        pos = np.asarray(config.TARGET_OBJECT["position"], dtype=float).copy()
        rot = np.asarray(config.TARGET_OBJECT["rotation"], dtype=float).copy()
        self.status.content = f"Object → {self.object_name}: cuRobo world 재구성…"
        self.backend.set_object(self.object_name)            # IK + world 재빌드
        self._build_object_scene(self.object_name, pos, rot)
        self.robot.set_config(config.ROBOT_START_STATE)
        vps = discover_viewpoints(DATA_ROOT, self.object_name)
        self.dd_vp.options = list(vps.keys()) or ["(none)"]
        self.dd_vp.value = next(iter(vps), "(none)")
        self._clear_viewpoints()
        self._clear_result()
        self.status.content = f"Object → {self.object_name}. Load viewpoints."

    def _clear_viewpoints(self):
        self.world_poses = None
        self._vp_raw = None
        self._vp_path = None
        self._reach = None
        for handle in self.layers["points"]:
            handle.remove()
        self.layers["points"].clear()
        for name in ("_vp_sel", "camera_frame"):
            handle = getattr(self, name)
            if handle is not None:
                handle.remove()
                setattr(self, name, None)
        self._make_vp_slider(1)

    def _clear_result(self):
        self.result = None
        self.result_kind = None
        self.dense = None
        self.current_component = None
        for name in ("delaunay", "path"):
            for handle in self.layers[name]:
                handle.remove()
            self.layers[name].clear()
        if self.highlight is not None:
            self.highlight.remove()
            self.highlight = None
        self.component_dropdown.options = ["(none)"]
        self.component_dropdown.value = "(none)"
        self._make_step_slider(1)

    # ── viewpoints ───────────────────────────────────────────────────────────
    def _load_viewpoints(self):
        vps = discover_viewpoints(DATA_ROOT, self.object_name)
        label = self.dd_vp.value
        if label not in vps:
            self.status.content = "선택된 viewpoints h5 가 없습니다."
            return
        self._vp_path = vps[label]
        positions, normals, path_order, _cluster, wd_m = PT.load_viewpoints(self._vp_path)
        if path_order is not None:                  # 방문 순서로 정렬 (파이프라인과 동일)
            order = np.argsort(path_order)
            positions, normals = positions[order], normals[order]
        self._vp_raw = (positions, normals, wd_m)
        self.world_poses = PT.build_camera_poses(positions, normals, wd_m)
        self._reach = None
        self._draw_plain_viewpoints()
        self._make_vp_slider(len(self.world_poses))
        self.status.content = (f"{len(self.world_poses)} viewpoints 로드 "
                               f"(wd={wd_m*1000:.0f}mm). Apply pose 로 reachability 확인.")
        self._on_vp_select()

    def _on_vp_select(self):
        if self.world_poses is None or len(self.world_poses) == 0:
            return
        i = int(min(self.sl_vp.value, len(self.world_poses) - 1))
        T = self.world_poses[i]
        pos = T[:3, 3]
        wxyz = _wxyz_from_matrix(T[:3, :3])
        if self._vp_sel is not None:
            self._vp_sel.remove()
        self._vp_sel = self.server.scene.add_point_cloud(
            "/studio/vp_sel", pos[None], np.asarray([COLOR_VP_SEL], dtype=np.uint8),
            point_size=0.016, point_shape="circle",
        )
        if self.camera_frame is not None:
            self.camera_frame.remove()
        self.camera_frame = self.server.scene.add_frame(
            "/studio/cam", position=pos, wxyz=wxyz, axes_length=0.06, axes_radius=0.003,
        )
        self._solve_and_show(pos, wxyz, src=f"viewpoint {i}")

    # ── live IK ──────────────────────────────────────────────────────────────
    def _solve_and_show(self, pos, wxyz, src="target"):
        self.reps, self.colliding = self.backend.solve_reps(pos, wxyz)
        self._src = src
        self._show_solution()

    def _show_solution(self):
        K = len(self.reps)
        n_free = int((~self.colliding).sum()) if K else 0
        if K == 0:
            self.robot.set_config(config.ROBOT_START_STATE)
            self.robot.set_color(COLOR_ROBOT_COLLIDE)
            self.status.content = (
                f"**{self._src}** — UNREACHABLE: 대표해 0개 (IK 실패 또는 전부 충돌). "
                "plan_trajectory/solve_glns_path 가 drop 하는 viewpoint 와 일치.")
            return
        k = int(min(self.sl_sol.value, K - 1))
        q = self.reps[k]
        is_col = bool(self.colliding[k])
        self.robot.set_color(COLOR_ROBOT_COLLIDE if is_col else COLOR_ROBOT_FREE)
        self.robot.set_config(q)
        state = "🔴 COLLISION" if is_col else "🟢 free"
        deg = ", ".join(f"{np.rad2deg(v):.0f}" for v in q)
        self.status.content = (
            f"**{self._src}** — reps={K}, collision-free={n_free}\n\n"
            f"solution k={k}/{K-1}: {state}\n\n`[{deg}]°`")

    def _sweep_reachability(self):
        """모든 viewpoint 에 IK 를 풀어 ≥1 collision-free 대표해가 있으면 reachable."""
        if self.world_poses is None:
            return None
        N = len(self.world_poses)
        free = np.zeros(N, dtype=bool)
        for i in range(N):
            T = self.world_poses[i]
            reps, colliding = self.backend.solve_reps(T[:3, 3], _wxyz_from_matrix(T[:3, :3]))
            free[i] = len(reps) > 0 and bool((~colliding).any())
        k = int(free.sum())
        self.reach_md.content = f"reachable (collision-free) **{k}/{N}**"
        self._log(f"sweep: free {k}/{N}")
        return free

    def _apply_object_pose(self):
        """gizmo 의 현재 물체 pose 를 확정 → 충돌월드 재구성 + viewpoint/IK 재계산."""
        if self.runner.running:
            self._log("⏳ 생성 중 — 끝난 뒤 Apply")
            return
        pos = np.asarray(self.obj_gizmo.position, dtype=float)
        wxyz = np.asarray(self.obj_gizmo.wxyz, dtype=float)
        config.TARGET_OBJECT["position"] = pos
        config.TARGET_OBJECT["rotation"] = wxyz
        self.status.content = "물체 pose 적용 — 충돌월드 재구성…"
        self.backend.rebuild_world()
        if self._vp_raw is not None:                # viewpoint 는 물체 로컬 → 따라 이동
            positions, normals, wd_m = self._vp_raw
            self.world_poses = PT.build_camera_poses(positions, normals, wd_m)
            self.status.content = f"reachability sweep ({len(self.world_poses)} vp)…"
            self._reach = self._sweep_reachability()
            self._draw_plain_viewpoints()
            self._on_vp_select()
        self._log(f"≡ pose applied pos={np.round(pos, 3).tolist()}")

    def _reset_object_pose(self):
        pos0, rot0 = self._obj_pose0
        self.obj_gizmo.position = pos0.copy()
        self.obj_gizmo.wxyz = rot0.copy()
        self._apply_object_pose()

    # ── 생성 (DP | GLNS) ─────────────────────────────────────────────────────
    def _set_buttons_enabled(self, enabled: bool):
        self.btn_gen.disabled = not enabled
        self.btn_apply.disabled = not enabled
        self.btn_reset.disabled = not enabled

    def _num_viewpoints(self) -> int:
        try:
            return int(Path(self._vp_path).parent.name)
        except (ValueError, AttributeError, TypeError):
            return len(self.world_poses) if self.world_poses is not None else 0

    def _on_generate(self):
        if self.runner.running:
            self._log("⏳ 이미 실행 중…")
            return
        if self._vp_path is None:
            self._log("Viewpoints 를 먼저 로드하세요.")
            return
        obj = self.object_name
        pos = np.asarray(self.obj_gizmo.position, dtype=float)
        wxyz = np.asarray(self.obj_gizmo.wxyz, dtype=float)
        spacing = float(self.spacing.value)
        n = self._num_viewpoints()
        vp = str(self._vp_path)
        pos_s = " ".join(f"{v:.6f}" for v in pos)
        quat_s = " ".join(f"{v:.6f}" for v in wxyz)
        # 생성 서브프로세스가 자체 cuRobo 를 띄우기 전에 캐시된 VRAM 반납(co-residency 완화).
        torch.cuda.empty_cache()
        self._pending_out = self._pending_npz = self._pending_csv = None

        if self.backend_dd.value.startswith("GLNS"):
            hops = max(1, int(round(self.expand_hops.value)))
            augment = ""
            if self.roll_augment.value:
                augment += " --roll-augment"
            if self.tilt_augment.value:
                angles = " ".join(str(float(x)) for x in self.tilt_angles.value.split())
                augment += (f" --tilt-augment --tilt-angles-deg {angles}"
                            f" --tilt-azimuths {int(round(self.tilt_azimuths.value))}")
            augment += f" --max-candidates-per-viewpoint {int(round(self.max_candidates.value))}"
            ik_num_seeds = max(1, int(round(self.ik_num_seeds.value)))
            ik_batch_size = max(1, int(round(self.ik_batch_size.value)))
            ik_options = f" --num-seeds {ik_num_seeds} --ik-batch-size {ik_batch_size}"
            det_h5 = PROJECT_ROOT / f"data/{obj}/ik/{n}/glns_result_studio.h5"
            det_h5.parent.mkdir(parents=True, exist_ok=True)
            shell = (
                f"uv run --no-sync scripts/core/solve_glns_path.py "
                f"--object {obj} --viewpoints '{vp}' "
                f"--object-position {pos_s} --object-quat {quat_s} "
                f"--delaunay-expand-hops {hops}{augment}{ik_options} --output '{det_h5}' "
                f"&& uv run --no-sync scripts/core/verify_glns_trajectory.py "
                f"--result '{det_h5}' --join --require-full-coverage --spacing {spacing}"
            )
            cmd = ["bash", "-c", shell]
            self._pending_out = det_h5
            self._pending_kind = "glns"
            self._log(f"▶ GLNS @ pos={np.round(pos, 3).tolist()}, hops={hops}, "
                      f"seeds={ik_num_seeds}, batch={ik_batch_size}, sp={spacing} …")
        else:
            suffix = self.suffix.value or "dp"
            cmd = [
                "uv", "run", "--no-sync", "scripts/core/plan_trajectory.py",
                "--object", obj, "--num-viewpoints", str(n),
                "--viewpoints", vp, "--spacing", str(spacing),
                "--output-suffix", suffix,
                "--object-position", *(f"{v:.6f}" for v in pos),
                "--object-quat", *(f"{v:.6f}" for v in wxyz),
            ]
            self._pending_kind = "dp"
            self._log(f"▶ DP @ pos={np.round(pos, 3).tolist()}, sp={spacing}, suffix={suffix} …")

        self._set_buttons_enabled(False)
        self.runner.start(
            cmd, cwd=PROJECT_ROOT,
            on_line=self._on_proc_line,
            on_exit=lambda rc: self._on_proc_exit(rc),
        )

    def _on_proc_line(self, line: str):
        self._log(line)
        m = re.search(r"GLNS_RESULT_H5\s+(\S+)", line)
        if m:
            self._pending_out = Path(m.group(1))
        m = re.search(r"NPZ saved to (\S+)", line)
        if m:
            self._pending_npz = Path(m.group(1))
        m = re.search(r"CSV saved to (\S+)", line)
        if m:
            self._pending_csv = Path(m.group(1))

    def _on_proc_exit(self, rc: int):
        self._log(f"■ exit {rc}")
        self._set_buttons_enabled(True)
        if rc != 0:
            self._log("✗ 실패 — 로그 확인")
            return
        if self._pending_kind == "glns" and self._pending_out is not None:
            p = Path(self._pending_out).resolve()
            key = str(p)
            self.result_paths[key] = p
            if key not in list(self.result_dropdown.options):
                self.result_dropdown.options = list(self.result_dropdown.options) + [key]
            self.result_dropdown.value = key
            self._load_result_path(p)
            self._log("✓ GLNS result 로드 — Component/Joined 재생")
        elif self._pending_kind == "dp":
            npz = self._pending_npz
            if npz is None and self._pending_csv is not None:
                npz = Path(self._pending_csv).with_suffix(".npz")
            if npz is None or not Path(npz).exists():
                self._log("✗ DP npz 를 찾지 못함")
                return
            dense = self._load_dense_npz(npz)
            if dense is None:
                self._log("✗ DP npz 로드 실패")
                return
            self._enter_dense_mode(dense, f"DP ({Path(npz).name})")
            self._log("✓ DP trajectory 로드 — 재생")

    # ── GLNS 결과 로드 ───────────────────────────────────────────────────────
    def _source_path(self, metadata: dict) -> Path:
        source = Path(str(_decode_attr(metadata["source_viewpoints"])))
        return source if source.is_absolute() else PROJECT_ROOT / source

    def _load_selected_result_dropdown(self):
        label = self.result_dropdown.value
        if label not in self.result_paths:
            self.status.content = "Selected result does not exist."
            return
        self._load_result_path(self.result_paths[label])

    def _load_result_path(self, path):
        path = Path(path).resolve()
        result = read_result_hdf5(path)
        metadata = result["metadata"]
        source_path = self._source_path(metadata)
        if not source_path.exists():
            self.status.content = f"Source viewpoint HDF5 not found: `{source_path}`"
            return
        with h5py.File(source_path, "r") as f:
            group = f["viewpoints"]
            positions = np.asarray(group["positions"], dtype=np.float64)
            normals = np.asarray(group["normals"], dtype=np.float64)

        object_name = str(_decode_attr(metadata["object"]))
        object_position = np.asarray(_decode_attr(metadata["object_position"]), dtype=np.float64)
        object_quat = np.asarray(_decode_attr(metadata["object_quat_wxyz"]), dtype=np.float64)
        config.TARGET_OBJECT["position"] = object_position
        config.TARGET_OBJECT["rotation"] = object_quat
        if object_name != self.backend.object_name:
            self.status.content = f"결과 object={object_name}: cuRobo world 재구성…"
            self.backend.set_object(object_name)
        self.object_name = object_name
        if object_name in self.objects:
            self._suppress_object_update = True
            self.dd_object.value = object_name
            self._suppress_object_update = False
        wd_m = float(_decode_attr(metadata["working_distance_m"]))

        self.result_h5_path = path
        self.dense_dir = path.parent
        self.world_poses = PT.build_camera_poses(positions, normals, wd_m)
        self._vp_raw = (positions, normals, wd_m)
        self._vp_path = source_path
        self._reach = None
        self.result = result
        self.result_kind = "glns"

        self._build_object_scene(object_name, object_position, object_quat)
        self._draw_component_viewpoints()
        self._draw_delaunay()

        options = []
        self.component_by_label = {}
        for component in result["components"]:
            name = component["name"]
            label = f"C{name} · {component['status']} · {len(component['members'])} vp"
            options.append(label)
            self.component_by_label[label] = component
        if (self.dense_dir / "glns_trajectory_joined.npz").exists():
            options.append("⨝ Joined (all components)")
        self.component_dropdown.options = options or ["(none)"]
        self.component_dropdown.value = options[0] if options else "(none)"
        self._on_component_change()
        self._apply_visibility()
        self._update_metrics()

    def _compute_metrics(self, result) -> dict:
        reach = np.asarray(result["reachable_mask"], dtype=bool)
        total_recfg = total_base = total_wrist = total_flip = 0
        for c in result["components"]:
            if c["status"] != "solved":
                continue
            a = c["attrs"]
            total_recfg += int(a.get("num_reconfigurations", 0))
            total_base += int(a.get("num_reconfigurations_base", 0))
            total_wrist += int(a.get("num_reconfigurations_wrist", 0))
            sel = c.get("selected_joints")
            if sel is None or len(sel) < 2:
                continue
            d = np.degrees(np.max(np.abs(np.diff(np.asarray(sel), axis=0)), axis=1))
            total_flip += int((d > 60.0).sum())
        return {"reach": (int(reach.sum()), len(reach)), "reconfigs": total_recfg,
                "base": total_base, "wrist": total_wrist, "flips": total_flip}

    def _update_metrics(self):
        if self.result is None or self.result_kind != "glns":
            return
        m = self._compute_metrics(self.result)
        meta = self.result["metadata"]
        pos = np.round(np.asarray(_decode_attr(meta["object_position"]), float), 3).tolist()
        self.metrics_md.content = (
            f"**placement** pos={pos}\n\n"
            f"reachable **{m['reach'][0]}/{m['reach'][1]}** · "
            f"reconfigs **{m['reconfigs']}** "
            f"(base **{m['base']}** / wrist **{m['wrist']}**) · "
            f"big-flips(>60°) **{m['flips']}**"
        )

    # ── 재생 (component path / dense) ────────────────────────────────────────
    def _load_dense_npz(self, path):
        """임의 npz {joints, ee_positions, is_transit, times} → dense dict. 없으면 None."""
        path = Path(path)
        if not path.exists():
            return None
        data = np.load(path)
        return {
            "joints": np.asarray(data["joints"], dtype=np.float64),
            "ee_pos": np.asarray(data["ee_positions"], dtype=np.float64),
            "is_transit": np.asarray(data["is_transit"], dtype=bool),
        }

    def _build_dense_path(self):
        """dense EE 경로를 transit(빨강)/scan(초록) run 단위 polyline 으로 그린다."""
        for handle in self.layers["path"]:
            handle.remove()
        self.layers["path"].clear()
        ee = self.dense["ee_pos"]
        it = self.dense["is_transit"]
        if len(ee) < 2:
            return
        edge_transit = it[:-1] | it[1:]
        start, run = 0, 0
        for i in range(1, len(edge_transit) + 1):
            if i == len(edge_transit) or edge_transit[i] != edge_transit[start]:
                seg = ee[start:i + 1]
                is_t = bool(edge_transit[start])
                self.layers["path"].append(self.server.scene.add_spline_catmull_rom(
                    f"/studio/path/dense{run}", positions=seg,
                    color=COLOR_RECONFIG if is_t else COLOR_PATH,
                    line_width=5.0 if is_t else 3.0,
                ))
                run += 1
                start = i

    def _enter_dense_mode(self, dense, label):
        """DP 결과(또는 임의 dense npz) 재생 모드. GLNS 그래프 컨트롤은 비활성."""
        self.result = None
        self.result_kind = "dp"
        self.current_component = None
        for name in ("delaunay", "path"):
            for handle in self.layers[name]:
                handle.remove()
            self.layers[name].clear()
        self.dense = dense
        self._dense_label = label
        self.play_position = 0.0
        self.metrics_md.content = f"metrics: DP dense `{len(dense['joints'])}` wp"
        self.component_dropdown.options = ["(DP dense)"]
        self.component_dropdown.value = "(DP dense)"
        self._build_dense_path()
        self._make_step_slider(len(dense["joints"]))
        self._show_step()
        self._apply_visibility()

    def _on_component_change(self):
        if self.result_kind == "dp":         # DP dense 는 dropdown 을 거치지 않는다
            return
        for handle in self.layers["path"]:
            handle.remove()
        self.layers["path"].clear()
        self.play_position = 0.0
        self.dense = None
        self.current_component = None
        if self.result is None or self.result_kind != "glns":
            self._make_step_slider(1)
            return

        label = self.component_dropdown.value
        if label.startswith("⨝ Joined"):
            self.dense = self._load_dense_npz(self.dense_dir / "glns_trajectory_joined.npz")
            self._dense_label = "Joined (all components)"
            if self.dense is not None:
                self._build_dense_path()
                self._make_step_slider(len(self.dense["joints"]))
                self._show_step()
            else:
                self.status.content = "joined npz 없음 — verify --join 먼저 실행"
            self._apply_visibility()
            return

        component = self.component_by_label.get(label)
        self.current_component = component
        if component is None:
            self._make_step_slider(1)
            return

        if self.show_dense.value:
            self.dense = self._load_dense_npz(
                self.dense_dir / f"glns_trajectory_comp{component['name']}.npz")
        if self.dense is not None:
            self._dense_label = f"Component {component['name']}"
            self._build_dense_path()
            self._make_step_slider(len(self.dense["joints"]))
            self._show_step()
            self._apply_visibility()
            return

        order = component.get("viewpoint_order")
        if order is None:
            self._make_step_slider(len(component["members"]))
            self.robot.set_config(config.ROBOT_START_STATE)
            note = "\n\n(no dense trajectory)" if self.show_dense.value else ""
            self.status.content = (
                f"**Component {component['name']} — {component['status']}**\n\n"
                f"{component['reason']}{note}"
            )
            self._apply_visibility()
            return

        cam = self.world_poses[:, :3, 3]
        reconfig = np.asarray(component["is_reconfiguration"], dtype=bool)
        for i, (a, b) in enumerate(zip(order[:-1], order[1:])):
            is_reconfig = bool(reconfig[i])
            self.layers["path"].append(self.server.scene.add_spline_catmull_rom(
                f"/studio/path/e{i}",
                positions=np.stack([cam[a], cam[b]]),
                color=COLOR_RECONFIG if is_reconfig else COLOR_PATH,
                line_width=5.0 if is_reconfig else 3.0,
            ))
        self._make_step_slider(len(order))
        self._show_step()
        self._apply_visibility()

    def _show_dense_step(self):
        joints = self.dense["joints"]
        n = len(joints)
        step = int(min(self.step_slider.value, n - 1))
        q = joints[step]
        self.robot.set_config(q)
        transit = bool(self.dense["is_transit"][step])
        self.robot.set_color(COLOR_RECONFIG if transit else COLOR_ROBOT_FREE)

        pos = self.dense["ee_pos"][step]
        if self.highlight is not None:
            self.highlight.remove()
        if self.camera_frame is not None:
            self.camera_frame.remove()
            self.camera_frame = None
        self.highlight = self.server.scene.add_point_cloud(
            "/studio/selected", pos[None],
            np.asarray([COLOR_SELECTED], dtype=np.uint8),
            point_size=0.016, point_shape="circle",
        )
        n_transit = int(self.dense["is_transit"].sum())
        deg = ", ".join(f"{v:.1f}" for v in np.rad2deg(q))
        self.status.content = (
            f"**{self._dense_label} — DENSE step {step}/{n - 1}**\n\n"
            f"segment: **{'TRANSIT' if transit else 'scan'}**\n\n"
            f"joints: `[{deg}]°`\n\n"
            f"dense waypoints: `{n}` (transit `{n_transit}`, scan `{n - n_transit}`)"
        )

    def _show_step(self):
        if self.dense is not None:
            self._show_dense_step()
            return
        component = self.current_component
        if component is None or component.get("viewpoint_order") is None:
            return
        order = component["viewpoint_order"]
        step = int(min(self.step_slider.value, len(order) - 1))
        viewpoint = int(order[step])
        q = np.asarray(component["selected_joints"][step], dtype=np.float64)
        self.robot.set_config(q)
        incoming_reconfig = step > 0 and bool(component["is_reconfiguration"][step - 1])
        self.robot.set_color(COLOR_RECONFIG if incoming_reconfig else COLOR_ROBOT_FREE)

        position = self.world_poses[viewpoint, :3, 3]
        rotation = self.world_poses[viewpoint, :3, :3]
        wxyz = _wxyz_from_matrix(rotation)
        if self.highlight is not None:
            self.highlight.remove()
        if self.camera_frame is not None:
            self.camera_frame.remove()
        self.highlight = self.server.scene.add_point_cloud(
            "/studio/selected", position[None],
            np.asarray([COLOR_SELECTED], dtype=np.uint8),
            point_size=0.016, point_shape="circle",
        )
        self.camera_frame = self.server.scene.add_frame(
            "/studio/cam", position=position, wxyz=wxyz,
            axes_length=0.06, axes_radius=0.003,
        )

        candidate = int(component["selected_candidate_index"][step])
        deg = ", ".join(f"{v:.1f}" for v in np.rad2deg(q))
        if step > 0:
            linf_deg = np.rad2deg(component["edge_linf_rad"][step - 1])
            l2 = component["edge_l2_rad"][step - 1]
            transition = (
                f"incoming: **{'RECONFIG' if incoming_reconfig else 'continuous'}**, "
                f"L∞={linf_deg:.1f}°, L2={l2:.3f} rad"
            )
        else:
            transition = "start of open path"
        attrs = component["attrs"]
        self.status.content = (
            f"**Component {component['name']} — step {step}/{len(order)-1}**\n\n"
            f"viewpoint raw index: `{viewpoint}` · IK candidate: `{candidate}`\n\n"
            f"{transition}\n\n"
            f"joints: `[{deg}]°`\n\n"
            f"component reconfigs: `{int(attrs.get('num_reconfigurations', 0))}`"
        )

    def tick(self, dt: float):
        if not self.play_checkbox.value:
            return
        if self.dense is not None:
            count = len(self.dense["joints"])
        elif self.current_component is not None:
            order = self.current_component.get("viewpoint_order")
            if order is None or len(order) < 2:
                return
            count = len(order)
        else:
            return
        if count < 2:
            return
        self.play_position = (
            self.play_position + dt * float(self.speed_slider.value)
        ) % count
        step = int(self.play_position)
        if step != self.step_slider.value:
            self.step_slider.value = step
            self._show_step()


def parse_args():
    parser = argparse.ArgumentParser(description="Unified viser trajectory studio (DP|GLNS)")
    parser.add_argument("--object", type=str, default=None, help="object name (data/{object}/...)")
    parser.add_argument("--result", type=Path, default=None, help="optional glns_result*.h5 to open")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    return parser.parse_args()


def main():
    args = parse_args()
    objects = discover_objects(DATA_ROOT)
    if not objects:
        raise SystemExit(f"No objects with mesh/source.obj under {DATA_ROOT}")
    initial_object = args.object if args.object in objects else objects[0]
    if args.object and args.object not in objects:
        print(f"  '{args.object}' 없음 → '{initial_object}' 사용. 가능: {objects}")

    # 물체 배치를 config 에 반영(IKBackend/gizmo 가 이 pose 를 캡처하므로 그 전에).
    if config.apply_object_placement(initial_object):
        print(f"  Per-object placement '{initial_object}': pos={config.TARGET_OBJECT['position']}, "
              f"quat={config.TARGET_OBJECT['rotation']}")

    results = discover_results()
    initial_result = None
    if args.result is not None:
        path = args.result.resolve()
        results[str(path)] = path
        initial_result = str(path)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.gui.configure_theme(control_layout="collapsible", dark_mode=True)
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/grid", width=2.0, height=2.0, plane="xy", cell_size=0.1, section_size=0.5,
    )

    print(f"[trajectory_studio] 초기화 중 (object={initial_object}) — cuRobo IK warmup…")
    app = TrajectoryStudio(server, objects, initial_object, results, initial_result)
    print(f"[trajectory_studio] 준비 완료. http://localhost:{args.port}")

    last = time.perf_counter()
    while True:
        now = time.perf_counter()
        app.tick(now - last)
        app.runner.pump()
        last = now
        time.sleep(0.05)


if __name__ == "__main__":
    main()
