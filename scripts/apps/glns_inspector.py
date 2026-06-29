#!/usr/bin/env python3
"""Viser playback for standalone Delaunay-constrained GLNS results.

The viewer never reruns IK or GLNS. It loads a ``glns_result_*.h5``, renders
the induced Delaunay graph and one component path, and drives the UR20 mesh
through the IK configuration selected by GLNS.
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
from ik_inspector import RobotViz  # noqa: E402


COLOR_OBJECT = (90, 200, 255)
COLOR_OBSTACLE = (120, 120, 130)
COLOR_DROPPED = (180, 70, 70)
COLOR_SELECTED = (255, 220, 50)
COLOR_DELAUNAY = (0, 180, 220)
COLOR_PATH = (70, 210, 120)
COLOR_RECONFIG = (240, 80, 70)


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


class GlnsInspector:
    def __init__(self, server: viser.ViserServer, result_paths: dict[str, Path], initial: str):
        self.server = server
        self.result_paths = result_paths
        robot_cfg = PT._resolve_robot_config(config.DEFAULT_ROBOT_CONFIG)
        self.robot = RobotViz(server, robot_cfg["robot_cfg"]["kinematics"]["urdf_path"])

        self.result = None
        self.positions = None
        self.normals = None
        self.world_poses = None
        self.current_component = None
        self.play_position = 0.0
        self.step_slider = None
        self.result_h5_path = None
        self.dense_dir = None      # glns_trajectory_comp{cid}.npz 가 있는 디렉토리
        self.dense = None          # 현재 성분의 dense 궤적(있을 때) {joints, ee_pos, is_transit}
        self.obj_gizmo = None      # 물체 이동 gizmo (translation + z-yaw)
        self.runner = SubprocessRunner()
        self._pending_out = None   # GLNS subprocess 가 쓸 결과 h5 경로
        self._log_lines = []
        self.layers = {
            "object": [], "obstacles": [], "points": [], "delaunay": [], "path": [],
        }
        self.highlight = None
        self.camera_frame = None

        self._build_gui(initial)
        self._load_selected_result()

    def _build_gui(self, initial: str):
        gui = self.server.gui
        with gui.add_folder("Result"):
            self.result_dropdown = gui.add_dropdown(
                "GLNS HDF5", options=list(self.result_paths), initial_value=initial,
            )
            self.load_button = gui.add_button("Load")
        with gui.add_folder("Layers"):
            self.show_object = gui.add_checkbox("Object", initial_value=True)
            self.show_obstacles = gui.add_checkbox("Obstacles", initial_value=False)
            self.show_points = gui.add_checkbox("Viewpoints", initial_value=True)
            self.show_delaunay = gui.add_checkbox("Delaunay graph", initial_value=False)
            self.show_path = gui.add_checkbox("Selected component path", initial_value=True)
        self.component_folder = gui.add_folder("Component")
        with self.component_folder:
            self.component_dropdown = gui.add_dropdown(
                "Run", options=["(none)"], initial_value="(none)",
            )
            self.play_checkbox = gui.add_checkbox("Play", initial_value=False)
            self.speed_slider = gui.add_slider(
                "Speed (steps/s)", min=0.25, max=20.0, step=0.25, initial_value=3.0,
            )
            self.show_dense = gui.add_checkbox(
                "Dense trajectory (CSV)", initial_value=False,
            )
        with gui.add_folder("Generate (move object → path)"):
            self.move_object = gui.add_checkbox("Move object (gizmo)", initial_value=False)
            self.expand_hops = gui.add_number(
                "Delaunay expand hops", initial_value=1, min=1, max=4, step=1,
            )
            self.btn_gen_path = gui.add_button("Generate path (GLNS)")
            self.btn_gen_traj = gui.add_button("Generate trajectory (collision)")
            self.metrics_md = gui.add_markdown("metrics: (load a result)")
            self.gen_log = gui.add_markdown("")
        self._make_step_slider(1)
        self.status = gui.add_markdown("Load a GLNS result.")

        self.load_button.on_click(lambda _: self._load_selected_result())
        self.component_dropdown.on_update(lambda _: self._on_component_change())
        self.show_dense.on_update(lambda _: self._on_component_change())
        self.move_object.on_update(lambda _: self._toggle_gizmo())
        self.btn_gen_path.on_click(lambda _: self._on_generate("path"))
        self.btn_gen_traj.on_click(lambda _: self._on_generate("traj"))
        for checkbox in (
            self.show_object, self.show_obstacles, self.show_points,
            self.show_delaunay, self.show_path,
        ):
            checkbox.on_update(lambda _: self._apply_visibility())

    def _make_step_slider(self, count: int):
        if self.step_slider is not None:
            self.step_slider.remove()
        with self.component_folder:
            self.step_slider = self.server.gui.add_slider(
                "Path step", min=0, max=max(int(count) - 1, 1), step=1, initial_value=0,
            )
        self.step_slider.on_update(lambda _: self._show_step())

    def _clear_handles(self, names=None):
        names = names or list(self.layers)
        for name in names:
            while self.layers[name]:
                self.layers[name].pop().remove()
        for handle_name in ("highlight", "camera_frame"):
            handle = getattr(self, handle_name)
            if handle is not None:
                handle.remove()
                setattr(self, handle_name, None)

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

    def _source_path(self, metadata: dict) -> Path:
        source = Path(str(_decode_attr(metadata["source_viewpoints"])))
        return source if source.is_absolute() else PROJECT_ROOT / source

    def _load_selected_result(self):
        label = self.result_dropdown.value
        if label not in self.result_paths:
            self.status.content = "Selected result does not exist."
            return
        self.result_h5_path = self.result_paths[label]
        self.dense_dir = self.result_h5_path.parent
        result = read_result_hdf5(self.result_h5_path)
        metadata = result["metadata"]
        source_path = self._source_path(metadata)
        if not source_path.exists():
            self.status.content = f"Source viewpoint HDF5 not found: `{source_path}`"
            return

        with h5py.File(source_path, "r") as f:
            group = f["viewpoints"]
            self.positions = np.asarray(group["positions"], dtype=np.float64)
            self.normals = np.asarray(group["normals"], dtype=np.float64)

        object_position = np.asarray(_decode_attr(metadata["object_position"]), dtype=np.float64)
        object_quat = np.asarray(_decode_attr(metadata["object_quat_wxyz"]), dtype=np.float64)
        config.TARGET_OBJECT["position"] = object_position
        config.TARGET_OBJECT["rotation"] = object_quat
        wd_m = float(_decode_attr(metadata["working_distance_m"]))
        self.world_poses = PT.build_camera_poses(self.positions, self.normals, wd_m)
        self.result = result
        self._clear_handles()
        self._build_static_scene(str(_decode_attr(metadata["object"])), object_position, object_quat)

        options = []
        self.component_by_label = {}
        for component in result["components"]:
            name = component["name"]
            label = f"C{name} · {component['status']} · {len(component['members'])} vp"
            options.append(label)
            self.component_by_label[label] = component
        self.component_dropdown.options = options or ["(none)"]
        self.component_dropdown.value = options[0] if options else "(none)"
        self._on_component_change()
        self._apply_visibility()
        self._update_metrics()

    def _build_static_scene(self, object_name: str, object_position, object_quat):
        mesh_path = config.get_mesh_path(object_name, mesh_type="source")
        if mesh_path.exists():
            loaded = trimesh.load(str(mesh_path), force="mesh")
            if isinstance(loaded, trimesh.Scene):
                loaded = trimesh.util.concatenate(list(loaded.geometry.values()))
            # 물체 gizmo: mesh 를 자식으로 두어 드래그하면 따라온다. x/y 회전 잠금 →
            # 이동 + z-yaw 만(viewpoint object-local 이라 재생성 불필요한 범위).
            if self.obj_gizmo is not None:
                self.obj_gizmo.remove()
            self.obj_gizmo = self.server.scene.add_transform_controls(
                "/glns/object", scale=0.15,
                position=np.asarray(object_position, dtype=float),
                wxyz=np.asarray(object_quat, dtype=float),
                rotation_limits=((0.0, 0.0), (0.0, 0.0), (-1000.0, 1000.0)),
                visible=self.move_object.value,
            )
            self.layers["object"].append(self.server.scene.add_mesh_simple(
                "/glns/object/mesh", np.asarray(loaded.vertices), np.asarray(loaded.faces),
                color=COLOR_OBJECT, opacity=0.4, side="double",
            ))

        for obstacle in [config.TABLE, config.ROBOT_MOUNT] + list(config.WALLS):
            box = trimesh.creation.box(extents=np.asarray(obstacle["dimensions"], dtype=float))
            self.layers["obstacles"].append(self.server.scene.add_mesh_simple(
                f"/glns/obstacles/{obstacle['name']}",
                np.asarray(box.vertices), np.asarray(box.faces),
                color=COLOR_OBSTACLE, opacity=0.18, side="double",
                position=np.asarray(obstacle["position"], dtype=float),
            ))

        camera_positions = self.world_poses[:, :3, 3]
        labels = self.result["component_id"]
        for cid in sorted(int(x) for x in np.unique(labels) if x >= 0):
            indices = np.where(labels == cid)[0]
            color = _component_color(cid)
            self.layers["points"].append(self.server.scene.add_point_cloud(
                f"/glns/viewpoints/c{cid}", camera_positions[indices],
                np.tile(np.asarray(color, dtype=np.uint8), (len(indices), 1)),
                point_size=0.007, point_shape="circle",
            ))
        dropped = np.where(~self.result["reachable_mask"])[0]
        if len(dropped):
            self.layers["points"].append(self.server.scene.add_point_cloud(
                "/glns/viewpoints/dropped", camera_positions[dropped],
                np.tile(np.asarray(COLOR_DROPPED, dtype=np.uint8), (len(dropped), 1)),
                point_size=0.008, point_shape="circle",
            ))

        for edge_index, (a, b) in enumerate(self.result["induced_edges"]):
            self.layers["delaunay"].append(self.server.scene.add_spline_catmull_rom(
                f"/glns/delaunay/e{edge_index}",
                positions=np.stack([camera_positions[a], camera_positions[b]]),
                color=COLOR_DELAUNAY, line_width=1.0,
            ))

    # ---- 물체 이동 → 경로 재생성 ----------------------------------------

    def _toggle_gizmo(self):
        if self.obj_gizmo is not None:
            self.obj_gizmo.visible = self.move_object.value

    def _log(self, line: str):
        self._log_lines.append(line)
        self._log_lines = self._log_lines[-18:]
        self.gen_log.content = "```\n" + "\n".join(self._log_lines) + "\n```"

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
            total_flip += int((d > 60.0).sum())   # via-roll 유발하는 큰 flip(6-DoF L∞>60°)
        return {"reach": (int(reach.sum()), len(reach)), "reconfigs": total_recfg,
                "base": total_base, "wrist": total_wrist, "flips": total_flip}

    def _update_metrics(self):
        if self.result is None:
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
        self._log(f"≡ pos={pos} → reach {m['reach'][0]}/{m['reach'][1]}, "
                  f"reconfig {m['reconfigs']} (b{m['base']}/w{m['wrist']}), flip {m['flips']}")

    def _set_buttons_enabled(self, enabled: bool):
        self.btn_gen_path.disabled = not enabled
        self.btn_gen_traj.disabled = not enabled

    def _on_generate(self, kind: str):
        if self.runner.running:
            self._log("⏳ 이미 실행 중…")
            return
        if self.result is None or self.obj_gizmo is None:
            self._log("결과를 먼저 로드하세요.")
            return
        meta = self.result["metadata"]
        obj = str(_decode_attr(meta["object"]))
        studio_out = self.result_h5_path.parent / "glns_result_studio.h5"

        if kind == "path":
            src = self._source_path(meta)
            pos = np.asarray(self.obj_gizmo.position, dtype=float)
            wxyz = np.asarray(self.obj_gizmo.wxyz, dtype=float)
            # 방어: gizmo 가 z-yaw 로 잠겨 있어도 비-z 성분이 남으면 경고(재생성 필요 범위).
            rv = Rotation.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]).as_rotvec()
            if abs(rv[0]) > 0.02 or abs(rv[1]) > 0.02:
                self._log(f"⚠ 비-z 회전 {np.round(np.degrees(rv), 1).tolist()}° — "
                          "viewpoint 는 z-yaw 가정(재생성 필요)")
            hops = max(1, int(round(self.expand_hops.value)))
            cmd = [
                "uv", "run", "--no-sync", "scripts/core/solve_glns_path.py",
                "--object", obj, "--viewpoints", str(src),
                "--object-position", *(f"{v:.6f}" for v in pos),
                "--object-quat", *(f"{v:.6f}" for v in wxyz),
                "--delaunay-expand-hops", str(hops),
                "--output", str(studio_out),
            ]
            self._pending_out = studio_out
            self._log(f"▶ GLNS @ pos={np.round(pos, 3).tolist()}, hops={hops} …")
        else:  # traj
            cmd = [
                "uv", "run", "--no-sync", "scripts/core/verify_glns_trajectory.py",
                "--result", str(self.result_h5_path),
            ]
            self._pending_out = None
            self._log("▶ trajectory (collision) …")

        self._set_buttons_enabled(False)
        self.runner.start(
            cmd, cwd=PROJECT_ROOT,
            on_line=self._on_proc_line,
            on_exit=lambda rc: self._on_proc_exit(rc, kind),
        )

    def _on_proc_line(self, line: str):
        self._log(line)
        m = re.search(r"GLNS_RESULT_H5\s+(\S+)", line)
        if m:
            self._pending_out = Path(m.group(1))

    def _on_proc_exit(self, rc: int, kind: str):
        self._log(f"■ exit {rc}")
        self._set_buttons_enabled(True)
        if rc != 0:
            self._log("✗ 실패 — 로그 확인")
            return
        if kind == "path" and self._pending_out is not None:
            p = Path(self._pending_out).resolve()
            key = str(p)
            self.result_paths[key] = p
            if key not in list(self.result_dropdown.options):
                self.result_dropdown.options = list(self.result_dropdown.options) + [key]
            self.result_dropdown.value = key
            self._load_selected_result()          # 새 배치/경로로 씬 재구성 + 메트릭
        elif kind == "traj":
            self._log("✓ trajectory npz 갱신 — Dense trajectory 토글로 재생")

    def _load_dense(self, cid):
        """결과 옆 glns_trajectory_comp{cid}.npz(검증 스크립트 산출)를 로드. 없으면 None."""
        if self.dense_dir is None:
            return None
        path = self.dense_dir / f"glns_trajectory_comp{cid}.npz"
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
        ee = self.dense["ee_pos"]
        it = self.dense["is_transit"]
        if len(ee) < 2:
            return
        edge_transit = it[:-1] | it[1:]   # edge i 가 transit 인가(양 끝 중 하나라도 transit)
        start, run = 0, 0
        for i in range(1, len(edge_transit) + 1):
            if i == len(edge_transit) or edge_transit[i] != edge_transit[start]:
                seg = ee[start:i + 1]     # 경계점 공유로 연속 표시
                is_t = bool(edge_transit[start])
                self.layers["path"].append(self.server.scene.add_spline_catmull_rom(
                    f"/glns/path/dense{run}", positions=seg,
                    color=COLOR_RECONFIG if is_t else COLOR_PATH,
                    line_width=5.0 if is_t else 3.0,
                ))
                run += 1
                start = i

    def _on_component_change(self):
        while self.layers["path"]:
            self.layers["path"].pop().remove()
        self.current_component = self.component_by_label.get(self.component_dropdown.value)
        self.play_position = 0.0
        self.dense = None
        if self.current_component is None:
            self._make_step_slider(1)
            return
        component = self.current_component

        # Dense 모드: 검증된 실제 motion(transit 포함) 재생
        if self.show_dense.value:
            self.dense = self._load_dense(component["name"])
        if self.dense is not None:
            self._build_dense_path()
            self._make_step_slider(len(self.dense["joints"]))
            self._show_step()
            self._apply_visibility()
            return

        order = component.get("viewpoint_order")
        if order is None:
            self._make_step_slider(len(component["members"]))
            self.robot.set_config(config.ROBOT_START_STATE)
            note = "\n\n(no dense trajectory for this component)" if self.show_dense.value else ""
            self.status.content = (
                f"**Component {component['name']} — {component['status']}**\n\n"
                f"{component['reason']}{note}"
            )
            self._apply_visibility()
            return

        camera_positions = self.world_poses[:, :3, 3]
        reconfig = np.asarray(component["is_reconfiguration"], dtype=bool)
        for i, (a, b) in enumerate(zip(order[:-1], order[1:])):
            is_reconfig = bool(reconfig[i])
            self.layers["path"].append(self.server.scene.add_spline_catmull_rom(
                f"/glns/path/e{i}",
                positions=np.stack([camera_positions[a], camera_positions[b]]),
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
        self.robot.set_color(COLOR_RECONFIG if transit else (170, 174, 184))

        pos = self.dense["ee_pos"][step]
        if self.highlight is not None:
            self.highlight.remove()
        if self.camera_frame is not None:
            self.camera_frame.remove()
            self.camera_frame = None
        self.highlight = self.server.scene.add_point_cloud(
            "/glns/selected", pos[None],
            np.asarray([COLOR_SELECTED], dtype=np.uint8),
            point_size=0.016, point_shape="circle",
        )
        n_transit = int(self.dense["is_transit"].sum())
        deg = ", ".join(f"{v:.1f}" for v in np.rad2deg(q))
        self.status.content = (
            f"**Component {self.current_component['name']} — DENSE step {step}/{n - 1}**\n\n"
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
        self.robot.set_color(COLOR_RECONFIG if incoming_reconfig else (170, 174, 184))

        position = self.world_poses[viewpoint, :3, 3]
        rotation = self.world_poses[viewpoint, :3, :3]
        q_xyzw = Rotation.from_matrix(rotation).as_quat()
        wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        if self.highlight is not None:
            self.highlight.remove()
        if self.camera_frame is not None:
            self.camera_frame.remove()
        self.highlight = self.server.scene.add_point_cloud(
            "/glns/selected", position[None],
            np.asarray([COLOR_SELECTED], dtype=np.uint8),
            point_size=0.016, point_shape="circle",
        )
        self.camera_frame = self.server.scene.add_frame(
            "/glns/camera_frame", position=position, wxyz=wxyz,
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
        component = self.current_component
        if not self.play_checkbox.value or component is None:
            return
        if self.dense is not None:
            count = len(self.dense["joints"])
        else:
            order = component.get("viewpoint_order")
            if order is None or len(order) < 2:
                return
            count = len(order)
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
    parser = argparse.ArgumentParser(description="Viser viewer for Delaunay GLNS results")
    parser.add_argument("--result", type=Path, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    return parser.parse_args()


def main():
    args = parse_args()
    results = discover_results()
    if args.result is not None:
        path = args.result.resolve()
        results[str(path)] = path
        initial = str(path)
    elif results:
        initial = next(iter(results))
    else:
        raise SystemExit("No GLNS result found. Run scripts/core/solve_glns_path.py first.")

    server = viser.ViserServer(host=args.host, port=args.port)
    server.gui.configure_theme(control_layout="collapsible", dark_mode=True)
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/grid", width=2.0, height=2.0, plane="xy", cell_size=0.1, section_size=0.5,
    )
    app = GlnsInspector(server, results, initial)
    print(f"[glns_inspector] http://localhost:{args.port}")
    last = time.perf_counter()
    while True:
        now = time.perf_counter()
        app.tick(now - last)
        app.runner.pump()
        last = now
        time.sleep(0.05)


if __name__ == "__main__":
    main()
