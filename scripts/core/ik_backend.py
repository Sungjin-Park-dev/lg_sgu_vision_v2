"""cuRobo IK backend + robot visualization library (headless-friendly, GPU).

UR20 + 검사 물체에 대한 cuRobo IK 대표해/충돌을 푸는 재사용 컴포넌트 모음.
과거 `scripts/apps/ik_inspector.py` 의 GUI 가 쓰던 코어를 라이브러리로 분리한 것으로,
지금은 `apps/trajectory_studio.py` 와 `tools/optimize_placement.py` 가 import 한다.

  - `IKBackend`  — object 마다 collision world + 영속 IK 솔버를 들고 IK 대표해/충돌 계산.
                   plan_trajectory 와 동일한 robot_cfg / collision world / wrist_3 잠금 /
                   batch_collision_check 를 재사용하므로, 여기서 'collision-free 대표해
                   0개'로 나오는 viewpoint 는 plan_trajectory 가 drop 하는 것과 일치한다.
  - `RobotViz`   — cuRobo collision STL 을 yourdfpy FK 로 구동해 viser 에 렌더.
  - `discover_objects` / `discover_viewpoints` — data/ 아래 물체·viewpoint 탐색.

좌표계: 전부 robot base_link 프레임(미터). 로봇 base = viser 원점. (Isaac 의 0.805 m
mount height 는 적용하지 않는다 — cuRobo IK/충돌이 base_link 프레임이기 때문.)

로봇 메쉬: cuRobo 가 충돌검사에 쓰는 collision STL 을 yourdfpy 로 직접 FK 구동해 렌더한다
(visual .dae 는 pycollada 미설치 환경에서 못 읽으므로 사용 안 함).
"""

import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np
import torch
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
DATA_ROOT = PROJECT_ROOT / "data"


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

    카메라 메쉬(camera_body.obj)는 robot_descriptions 캐시에 없고 workcell/robot/camera/
    에만 있으므로 로컬로 매핑한다. (yourdfpy 는 핸들러를 handler(fname=...) 로 호출하므로
    인자 이름이 반드시 `fname` 이어야 한다.)
    """
    parent = _find_ur_description_parent()
    local_camera = PROJECT_ROOT / "workcell" / "robot" / "camera"

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

            reps_list = PT.cluster_ik_solutions(sol[None], succ[None])
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
