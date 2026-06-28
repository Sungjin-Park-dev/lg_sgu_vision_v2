#!/usr/bin/env python3
"""viser 기반 IK 확인 도구 — UR20 + 검사 물체를 불러와 cuRobo IK 해를 시각화한다.

Isaac Sim에서 모션을 돌려보지 않고도, 브라우저에서 가볍게:
  (1) Viewpoint 모드 — viewpoints*.h5 를 로드하고 슬라이더로 viewpoint 를 고르면 그
      카메라 포즈(camera_optical_frame)에 대한 IK 대표해(분기)들을 로봇에 표시.
  (2) Gizmo 모드   — transform gizmo 로 타겟을 자유롭게 옮기며 실시간으로 IK 를 푼다.
모든 대표해를 충돌 여부(초록=free / 빨강=collision)와 함께 보여준다. plan_trajectory 와
동일한 robot_cfg / collision world / wrist_3 잠금 / batch_collision_check 를 재사용하므로,
여기서 'collision-free 대표해 0개'로 나오는 viewpoint 는 plan_trajectory 가 drop 하는
viewpoint 와 일치한다(교차검증 용도).

좌표계: 전부 robot base_link 프레임(미터). 로봇 base = viser 원점. (Isaac 의 0.805 m
mount height 는 적용하지 않는다 — cuRobo IK/충돌이 base_link 프레임이기 때문.)

로봇 메쉬: cuRobo 가 충돌검사에 쓰는 collision STL 을 yourdfpy 로 직접 FK 구동해 렌더한다
(visual .dae 는 pycollada 미설치 환경에서 못 읽으므로 사용 안 함).

사용법:
    uv run --no-sync scripts/apps/ik_inspector.py --object sample
    uv run --no-sync scripts/apps/ik_inspector.py --object curved_structure --port 8081
"""

import argparse
import contextlib
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
import viser
import yourdfpy
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT / "core"))

from common import config                       # noqa: E402
import plan_trajectory as PT                    # noqa: E402  (헤비: curobo import)
from curobo.types import Pose, GoalToolPose     # noqa: E402
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg  # noqa: E402

# ── 상수 ────────────────────────────────────────────────────────────────────
NUM_SEEDS = 60                       # gizmo 실시간 응답 위해 파이프라인(100)보다 줄임
MAX_REP_SLIDER = NUM_SEEDS - 1       # 대표해 인덱스 슬라이더 상한(실제 K 로 clamp)
COLOR_ROBOT_FREE = (170, 174, 184)
COLOR_ROBOT_COLLIDE = (224, 96, 88)
COLOR_OBJECT = (90, 200, 255)
COLOR_OBSTACLE = (120, 120, 130)
COLOR_VP_ALL = (110, 120, 140)
COLOR_VP_SEL = (255, 210, 60)
DATA_ROOT = PROJECT_ROOT / "data"
OBJ_NODE = "/object_ctrl"            # 물체 이동 gizmo 노드 (mesh 는 그 자식)


def _wxyz_from_matrix(R: np.ndarray) -> np.ndarray:
    """3x3 회전행렬 → (w, x, y, z) quaternion."""
    q = Rotation.from_matrix(R).as_quat()        # (x, y, z, w)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


@contextlib.contextmanager
def _quiet():
    """cluster/solve 의 print 홍수를 콘솔에서 숨긴다."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ── 로봇 메쉬 경로 해석 (package:// → 실제 STL/OBJ) ─────────────────────────────
def _find_ur_description_parent() -> Path:
    """`<parent>/ur_description/meshes/ur20/collision/base.stl` 가 있는 parent 디렉토리."""
    candidates = [
        Path.home() / ".cache" / "robot_descriptions",
        PROJECT_ROOT / ".venv/lib/python3.12/site-packages/curobo/content/assets/robot",
    ]
    for parent in candidates:
        if (parent / "ur_description/meshes/ur20/collision/base.stl").exists():
            return parent
    raise FileNotFoundError(
        "ur_description collision meshes 를 못 찾음. 확인한 경로:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


def _make_package_handler():
    """yourdfpy filename_handler: package://ur_description/... → 실제 파일 경로.

    카메라 메쉬(camera_body.obj)는 robot_descriptions 캐시에 없고 ur20_description/camera/
    에만 있으므로 로컬로 매핑한다. (yourdfpy 는 핸들러를 handler(fname=...) 로 호출하므로
    인자 이름이 반드시 `fname` 이어야 한다.)
    """
    parent = _find_ur_description_parent()
    local_camera = PROJECT_ROOT / "ur20_description" / "camera"

    def handler(fname):
        if fname.startswith("package://"):
            rel = fname[len("package://"):]
            if "camera" in rel:
                return str(local_camera / os.path.basename(rel))
            return str(parent / rel)
        return fname

    return handler


class RobotViz:
    """collision STL 을 yourdfpy FK 로 구동해 viser 에 렌더하는 로봇 비주얼.

    viser scene-graph 이름 계층에 의존하지 않고, 매 config 마다 각 geom 의 base_link
    기준 절대 변환을 직접 setattr 한다(flat). 색 변경 시에만 메쉬를 재추가한다.
    """

    def __init__(self, server: viser.ViserServer, urdf_path: str, root: str = "/robot"):
        handler = _make_package_handler()
        self.urdf = yourdfpy.URDF.load(
            urdf_path, filename_handler=handler,
            load_meshes=False, build_scene_graph=True,
            load_collision_meshes=True, build_collision_scene_graph=True,
        )
        self.cs = self.urdf.collision_scene
        self.server = server
        self.root = root
        self.color = COLOR_ROBOT_FREE
        self._geom = {
            name: (np.asarray(m.vertices), np.asarray(m.faces))
            for name, m in self.cs.geometry.items()
        }
        self.handles: dict[str, object] = {}
        self._last_q = np.asarray(config.ROBOT_START_STATE, dtype=float)
        self._add(self.color)
        self.set_config(self._last_q)

    def _add(self, color):
        for name, (v, f) in self._geom.items():
            self.handles[name] = self.server.scene.add_mesh_simple(
                f"{self.root}/{name}", v, f, color=color,
                side="double", flat_shading=True,
            )

    def set_color(self, color):
        if tuple(color) == tuple(self.color):
            return
        self.color = tuple(color)
        for h in self.handles.values():
            h.remove()
        self.handles = {}
        self._add(self.color)
        self.set_config(self._last_q)

    def set_config(self, q):
        self._last_q = np.asarray(q, dtype=float)
        self.urdf.update_cfg(self._last_q)
        for name, h in self.handles.items():
            T = self.cs.graph.get(name)[0]
            h.position = T[:3, 3]
            h.wxyz = _wxyz_from_matrix(T[:3, :3])

    def camera_optical_pose(self, q) -> np.ndarray:
        """주어진 config 에서 camera_optical_frame 의 base_link 기준 4x4 변환."""
        self.urdf.update_cfg(np.asarray(q, dtype=float))
        T = self.urdf.get_transform("camera_optical_frame", "base_link")
        self.urdf.update_cfg(self._last_q)          # 복원
        return np.asarray(T)


class IKBackend:
    """object 마다 collision world + 영속 IK 솔버를 들고 IK 대표해/충돌을 푼다."""

    def __init__(self, object_name: str):
        self.robot_cfg = PT._resolve_robot_config(config.DEFAULT_ROBOT_CONFIG)
        self.urdf_path = self.robot_cfg["robot_cfg"]["kinematics"]["urdf_path"]
        self._world_ctr = 0
        self.set_object(object_name)

    def _build_world(self):
        """build_collision_world + target mesh 이름을 매번 유니크하게.

        cuRobo 는 mesh 를 '이름'으로 전역 캐시한다("Mesh already in cache, reusing…"). 이름이
        같으면 pose/geometry 변경이 무시돼 물체를 옮겨도/물체를 바꿔도 옛 캐시를 재사용한다.
        → 빌드마다 이름에 카운터를 붙여 새 pose/geometry 로 강제 재로드.
        """
        self._world_ctr += 1
        with _quiet():
            world = PT.build_collision_world(self.object_name)
        for m in (world.mesh or []):
            m.name = f"{m.name}_{self._world_ctr}"
        return world

    def set_object(self, object_name: str):
        self.object_name = object_name
        with _quiet():
            self.world = self._build_world()
            cache = {
                "obb": max(1, len(self.world.cuboid)),
                "mesh": max(1, len(self.world.mesh)),
            }
            cfg = InverseKinematicsCfg.create(
                robot=self.robot_cfg, scene_model={}, self_collision_check=True,
                num_seeds=NUM_SEEDS, max_batch_size=1,
                use_cuda_graph=False, collision_cache=cache,
            )
            self.ik = InverseKinematics(cfg)
            self.ik.update_world(self.world)
            self.tool = self.ik.tool_frames[0]

    def rebuild_world(self):
        """충돌 월드만 현재 config.TARGET_OBJECT pose 로 재구성(솔버는 유지, update_world 만)."""
        with _quiet():
            self.world = self._build_world()
            self.ik.update_world(self.world)

    def solve_reps(self, position_xyz, quat_wxyz):
        """타겟 카메라 포즈 → (reps (K,6), colliding (K,) bool). plan_trajectory 와 동일 로직."""
        with _quiet():
            bp = torch.tensor(np.asarray(position_xyz, dtype=np.float32)[None],
                              device="cuda:0", dtype=torch.float32)
            bq = torch.tensor(np.asarray(quat_wxyz, dtype=np.float32)[None],
                              device="cuda:0", dtype=torch.float32)
            goal = Pose(position=bp, quaternion=bq)
            result = self.ik.solve_pose(
                GoalToolPose.from_poses({self.tool: goal}, num_goalset=1),
                return_seeds=NUM_SEEDS,
            )
            sol = result.js_solution.position.cpu().numpy()[0]       # (S, dof)
            if sol.shape[-1] != 6:
                sol = sol[..., :6]
            succ = result.success.cpu().numpy()[0]                   # (S,)
            sol = PT.normalize_joints(sol)

            reps_list = PT.cluster_ik_solutions(
                sol[None], succ[None], eps=PT.DBSCAN_EPS_RAD,
            )
            reps = reps_list[0]
            if len(reps) == 0:
                return np.empty((0, 6)), np.empty((0,), dtype=bool)
            reps = np.asarray(reps, dtype=np.float64)
            reps[:, -1] = config.ROBOT_START_STATE[-1]               # wrist_3 잠금 (pre-DP 동일)
            colliding, _ = PT.batch_collision_check(reps, self.robot_cfg, self.world)
        return reps, np.asarray(colliding, dtype=bool)


# ── 데이터 탐색 ──────────────────────────────────────────────────────────────
def discover_objects(data_root: Path) -> list[str]:
    return [p.parent.parent.name for p in sorted(data_root.glob("*/mesh/source.obj"))]


def discover_viewpoints(data_root: Path, object_name: str) -> dict[str, Path]:
    base = data_root / object_name / "viewpoint"
    out: dict[str, Path] = {}
    for path in sorted(base.glob("*/viewpoints*.h5")):
        out[f"{path.parent.name}/{path.name}"] = path
    return out


# ── 메인 앱 ─────────────────────────────────────────────────────────────────
class Inspector:
    def __init__(self, server: viser.ViserServer, objects: list[str], initial_object: str):
        self.server = server
        self.objects = objects
        self.object_name = initial_object

        self.backend = IKBackend(initial_object)
        self.robot = RobotViz(server, self.backend.urdf_path)

        # 물체 pose 기본값(리셋용) — 어떤 변경보다 먼저 캡처
        self._obj_pose0 = (
            np.asarray(config.TARGET_OBJECT["position"], dtype=float).copy(),
            np.asarray(config.TARGET_OBJECT["rotation"], dtype=float).copy(),
        )

        # 현재 viewpoint set / IK 상태
        self.world_poses = None          # (N,4,4) base_link 프레임 카메라 포즈
        self._vp_raw = None              # (positions, normals, wd_m) — 물체 이동 시 재계산용
        self.reps = np.empty((0, 6))
        self.colliding = np.empty((0,), dtype=bool)
        self._suppress_gizmo = False
        self._obstacle_handles: list[object] = []
        self._object_handle = None
        self._vp_cloud = None
        self._vp_sel = None

        self._build_gui()
        # 물체 이동 gizmo (mesh 는 이 노드의 자식 → 드래그하면 시각적으로 따라온다)
        self.obj_gizmo = server.scene.add_transform_controls(
            OBJ_NODE, scale=0.2,
            position=self._obj_pose0[0].copy(), wxyz=self._obj_pose0[1].copy(),
            visible=self.cb_obj_gizmo.value,
        )
        self._load_object_scene()
        # 타겟 gizmo 를 start-state 카메라 위치에 둔다.
        T0 = self.robot.camera_optical_pose(config.ROBOT_START_STATE)
        self.gizmo = server.scene.add_transform_controls(
            "/target", scale=0.12,
            position=T0[:3, 3], wxyz=_wxyz_from_matrix(T0[:3, :3]),
        )
        self.gizmo.on_update(lambda _: self._on_gizmo())
        self.target_frame = server.scene.add_frame(
            "/target_frame", axes_length=0.08, axes_radius=0.004,
            position=T0[:3, 3], wxyz=_wxyz_from_matrix(T0[:3, :3]),
        )
        self._solve_and_show(T0[:3, 3], _wxyz_from_matrix(T0[:3, :3]))

    # ---- GUI ----
    def _build_gui(self):
        g = self.server.gui
        with g.add_folder("Scene"):
            self.dd_object = g.add_dropdown(
                "Object", options=self.objects, initial_value=self.object_name)
            self.cb_obstacles = g.add_checkbox("Show obstacles", initial_value=False)
        self.folder_vp = g.add_folder("Viewpoints")
        with self.folder_vp:
            vps = discover_viewpoints(DATA_ROOT, self.object_name)
            self.dd_vp = g.add_dropdown(
                "h5", options=list(vps.keys()) or ["(none)"],
                initial_value=(next(iter(vps), "(none)")))
            self.btn_load_vp = g.add_button("Load viewpoints")
        self.sl_vp = None
        self._make_vp_slider(1)        # slider.max 는 setter 가 없어 매번 재생성한다
        with g.add_folder("Move object"):
            self.cb_obj_gizmo = g.add_checkbox("Show object gizmo", initial_value=True)
            self.btn_apply_obj = g.add_button("Apply pose → recompute IK")
            self.btn_reset_obj = g.add_button("Reset object pose")
        with g.add_folder("IK"):
            self.cb_live = g.add_checkbox("Live IK (gizmo drag)", initial_value=True)
            self.btn_solve = g.add_button("Solve at gizmo")
            self.sl_sol = g.add_slider(
                "Solution k", min=0, max=MAX_REP_SLIDER, step=1, initial_value=0)
            self.md_status = g.add_markdown("…")

        self.dd_object.on_update(lambda _: self._on_object_change())
        self.cb_obstacles.on_update(lambda _: self._apply_obstacle_visibility())
        self.btn_load_vp.on_click(lambda _: self._load_viewpoints())
        self.cb_obj_gizmo.on_update(lambda _: self._toggle_obj_gizmo())
        self.btn_apply_obj.on_click(lambda _: self._apply_object_pose())
        self.btn_reset_obj.on_click(lambda _: self._reset_object_pose())
        self.btn_solve.on_click(lambda _: self._on_gizmo(force=True))
        self.sl_sol.on_update(lambda _: self._show_solution())

    def _make_vp_slider(self, n: int):
        if self.sl_vp is not None:
            self.sl_vp.remove()
        with self.folder_vp:
            self.sl_vp = self.server.gui.add_slider(
                "Viewpoint idx", min=0, max=max(int(n) - 1, 1), step=1, initial_value=0)
        self.sl_vp.on_update(lambda _: self._on_vp_select())

    # ---- object / scene ----
    def _on_object_change(self):
        self.object_name = self.dd_object.value
        # 물체 pose 를 canonical 기본값으로 리셋하고 gizmo 도 그 위치로
        pos0, rot0 = self._obj_pose0
        config.TARGET_OBJECT["position"] = pos0.copy()
        config.TARGET_OBJECT["rotation"] = rot0.copy()
        self.obj_gizmo.position = pos0.copy()
        self.obj_gizmo.wxyz = rot0.copy()
        with _quiet():
            self.backend.set_object(self.object_name)
        self._load_object_scene()
        # viewpoint 목록 갱신 + 기존 viewpoint 레이어 제거
        vps = discover_viewpoints(DATA_ROOT, self.object_name)
        self.dd_vp.options = list(vps.keys()) or ["(none)"]
        self.dd_vp.value = next(iter(vps), "(none)")
        self._clear_viewpoints()
        self._status(f"Object → {self.object_name}. Viewpoints h5 를 로드하세요.")

    def _load_object_scene(self):
        if self._object_handle is not None:
            self._object_handle.remove()
            self._object_handle = None
        mesh_path = config.get_mesh_path(self.object_name, mesh_type="source")
        if mesh_path.exists():
            loaded = trimesh.load(str(mesh_path), force="mesh")
            if isinstance(loaded, trimesh.Scene):
                loaded = trimesh.util.concatenate(list(loaded.geometry.values()))
            # mesh 를 gizmo 노드의 자식(로컬 identity)으로 추가 → gizmo 이동 시 자동으로 따라옴.
            # (vertices 는 object-local 프레임이고 object 원점 = gizmo 원점)
            self._object_handle = self.server.scene.add_mesh_simple(
                f"{OBJ_NODE}/mesh", np.asarray(loaded.vertices), np.asarray(loaded.faces),
                color=COLOR_OBJECT, opacity=0.45, side="double",
            )
        self._build_obstacles()

    def _build_obstacles(self):
        for h in self._obstacle_handles:
            h.remove()
        self._obstacle_handles = []
        obstacles = [config.TABLE, config.ROBOT_MOUNT] + list(config.WALLS)
        for obj in obstacles:
            try:
                box = trimesh.creation.box(extents=np.asarray(obj["dimensions"], dtype=float))
                h = self.server.scene.add_mesh_simple(
                    f"/obstacles/{obj['name']}",
                    np.asarray(box.vertices), np.asarray(box.faces),
                    color=COLOR_OBSTACLE, opacity=0.18, side="double",
                    position=np.asarray(obj["position"], dtype=float),
                    visible=self.cb_obstacles.value,
                )
                self._obstacle_handles.append(h)
            except Exception:
                continue

    def _apply_obstacle_visibility(self):
        for h in self._obstacle_handles:
            h.visible = self.cb_obstacles.value

    # ---- move object ----
    def _toggle_obj_gizmo(self):
        self.obj_gizmo.visible = self.cb_obj_gizmo.value

    def _apply_object_pose(self):
        """gizmo 의 현재 물체 pose 를 확정 → 충돌월드 재구성 + viewpoint/IK 재계산."""
        pos = np.asarray(self.obj_gizmo.position, dtype=float)
        wxyz = np.asarray(self.obj_gizmo.wxyz, dtype=float)
        config.TARGET_OBJECT["position"] = pos
        config.TARGET_OBJECT["rotation"] = wxyz
        self._status("물체 pose 적용 중 — 충돌월드 재구성…")
        self.backend.rebuild_world()
        if self._vp_raw is not None:                    # viewpoint 는 물체 로컬 → 따라 이동
            positions, normals, wd_m = self._vp_raw
            self.world_poses = PT.build_camera_poses(positions, normals, wd_m)
            self._refresh_vp_cloud()
            self._on_vp_select()                        # 현재 viewpoint 재타겟 + 재계산
        else:                                           # gizmo-target 모드: 충돌만 바뀜
            self._on_gizmo(force=True)
        p = ", ".join(f"{v:.3f}" for v in pos)
        self._status(f"물체 pose 적용됨 → `[{p}]m`\n\n{self._last_status}")

    def _reset_object_pose(self):
        pos0, rot0 = self._obj_pose0
        self.obj_gizmo.position = pos0.copy()
        self.obj_gizmo.wxyz = rot0.copy()
        self._apply_object_pose()

    # ---- viewpoints ----
    def _clear_viewpoints(self):
        self.world_poses = None
        self._vp_raw = None
        for h in (self._vp_cloud, self._vp_sel):
            if h is not None:
                h.remove()
        self._vp_cloud = self._vp_sel = None
        self._make_vp_slider(1)

    def _load_viewpoints(self):
        vps = discover_viewpoints(DATA_ROOT, self.object_name)
        label = self.dd_vp.value
        if label not in vps:
            self._status("선택된 viewpoints h5 가 없습니다.")
            return
        positions, normals, path_order, _cluster, wd_m = PT.load_viewpoints(vps[label])
        if path_order is not None:                     # 방문 순서로 정렬 (파이프라인과 동일)
            order = np.argsort(path_order)
            positions, normals = positions[order], normals[order]
        self._vp_raw = (positions, normals, wd_m)      # 물체 이동 시 재계산용 원본 보관
        self.world_poses = PT.build_camera_poses(positions, normals, wd_m)
        self._refresh_vp_cloud()
        self._make_vp_slider(len(self.world_poses))
        self._status(f"{len(self.world_poses)} viewpoints 로드 "
                     f"(wd={wd_m*1000:.0f}mm). 슬라이더로 선택.")
        self._on_vp_select()

    def _refresh_vp_cloud(self):
        """현재 world_poses 로 viewpoint 점군을 다시 그린다(물체 이동 후 위치 갱신)."""
        if self.world_poses is None:
            return
        cam = self.world_poses[:, :3, 3]
        if self._vp_cloud is not None:
            self._vp_cloud.remove()
        self._vp_cloud = self.server.scene.add_point_cloud(
            "/viewpoints", cam, COLOR_VP_ALL, point_size=0.008, point_shape="circle")

    def _on_vp_select(self):
        if self.world_poses is None or len(self.world_poses) == 0:
            return
        i = int(min(self.sl_vp.value, len(self.world_poses) - 1))
        T = self.world_poses[i]
        pos = T[:3, 3]
        wxyz = _wxyz_from_matrix(T[:3, :3])
        # 선택 viewpoint 강조 + gizmo/타겟 frame 을 그 포즈로 스냅
        if self._vp_sel is not None:
            self._vp_sel.remove()
        self._vp_sel = self.server.scene.add_point_cloud(
            "/viewpoint_sel", pos[None], np.array([COLOR_VP_SEL], dtype=np.uint8),
            point_size=0.016, point_shape="circle")
        self._set_target(pos, wxyz)
        self._solve_and_show(pos, wxyz, src=f"viewpoint {i}")

    # ---- gizmo ----
    def _set_target(self, pos, wxyz):
        """gizmo + target frame 을 프로그램적으로 이동(재귀 solve 방지 플래그 사용)."""
        self._suppress_gizmo = True
        self.gizmo.position = np.asarray(pos, dtype=float)
        self.gizmo.wxyz = np.asarray(wxyz, dtype=float)
        self._suppress_gizmo = False
        self.target_frame.position = np.asarray(pos, dtype=float)
        self.target_frame.wxyz = np.asarray(wxyz, dtype=float)

    def _on_gizmo(self, force=False):
        if self._suppress_gizmo:
            return
        if not force and not self.cb_live.value:
            return
        pos = np.asarray(self.gizmo.position, dtype=float)
        wxyz = np.asarray(self.gizmo.wxyz, dtype=float)
        self.target_frame.position = pos
        self.target_frame.wxyz = wxyz
        self._solve_and_show(pos, wxyz, src="gizmo")

    # ---- IK solve / display ----
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
            self._status(
                f"**{getattr(self, '_src', 'target')}** — UNREACHABLE: 대표해 0개 "
                f"(IK 실패 또는 전부 충돌). plan_trajectory 가 drop 하는 viewpoint 와 일치.")
            return
        k = int(min(self.sl_sol.value, K - 1))
        q = self.reps[k]
        is_col = bool(self.colliding[k])
        self.robot.set_color(COLOR_ROBOT_COLLIDE if is_col else COLOR_ROBOT_FREE)
        self.robot.set_config(q)
        state = "🔴 COLLISION" if is_col else "🟢 free"
        deg = ", ".join(f"{np.rad2deg(v):.0f}" for v in q)
        self._status(
            f"**{getattr(self, '_src', 'target')}** — reps={K}, collision-free={n_free}\n\n"
            f"solution k={k}/{K-1}: {state}\n\n`[{deg}]°`")

    def _status(self, text):
        self._last_status = text
        self.md_status.content = text


def parse_args():
    p = argparse.ArgumentParser(description="viser IK inspector for UR20 + object")
    p.add_argument("--object", type=str, default=None, help="object name (data/{object}/...)")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    return p.parse_args()


def main():
    args = parse_args()
    objects = discover_objects(DATA_ROOT)
    if not objects:
        raise SystemExit(f"No objects with mesh/source.obj under {DATA_ROOT}")
    initial = args.object if args.object in objects else objects[0]
    if args.object and args.object not in objects:
        print(f"  '{args.object}' 없음 → '{initial}' 사용. 가능: {objects}")

    server = viser.ViserServer(host=args.host, port=args.port)
    server.gui.configure_theme(control_layout="collapsible", dark_mode=True)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", width=2.0, height=2.0, plane="xy",
                          cell_size=0.1, section_size=0.5)

    print(f"[ik_inspector] 초기화 중 (object={initial}) — cuRobo IK 솔버 warmup…")
    Inspector(server, objects, initial)
    print(f"[ik_inspector] 준비 완료. 브라우저에서 http://localhost:{args.port} 접속.")

    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    main()
