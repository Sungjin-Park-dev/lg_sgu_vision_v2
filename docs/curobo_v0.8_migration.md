# cuRobo v0.8.0 Migration — `plan_trajectory.py`

cuRobo v0.8.0 은 flat / inheritance‑free 아키텍처로 대대적으로 리팩토링되었고
CHANGELOG 에 *"Major refactor breaks most existing api."* 라고 명시되어 있다.
`scripts/core/plan_trajectory.py` 가 사용하던 옛 진입점 (`curobo.wrap.reacher.*`,
`curobo.wrap.model.*`, `curobo.geom.types.WorldConfig`,
`curobo.types.base.TensorDeviceType`, `curobo.types.robot.JointState`,
`curobo.util_file.*`) 은 모두 사라졌다. 본 문서는 그 마이그레이션의 결과를 정리한다.

> 알고리즘 (DBSCAN, DP, transit, resample) 과 시각화 / CSV 출력 포맷은 그대로 유지.
> 변경은 cuRobo 호출부에 한정.

## API 매핑

| 옛 (≤0.7.x) | 새 (v0.8.0) |
|---|---|
| `curobo.types.base.TensorDeviceType` | (제거) — 필요 시 `curobo.types.DeviceCfg` |
| `curobo.types.math.Pose` | `curobo.types.Pose` (wxyz 동일) |
| `curobo.types.robot.JointState` | `curobo.types.JointState` |
| `curobo.geom.types.WorldConfig(cuboid=, mesh=)` | `curobo.scene.Scene(cuboid=, mesh=)` (`SceneCfg` 별칭) |
| `curobo.geom.types.{Cuboid, Mesh}` | `curobo.scene.{Cuboid, Mesh}` (시그니처 동일) |
| `curobo.util_file.{get_robot_configs_path, join_path, load_yaml}` 후 dict 추출 | 제거 — `*Cfg.create(robot=...)` 가 직접 처리 |
| `IKSolver` / `IKSolverConfig.load_from_robot_config(...)` | `InverseKinematics` / `InverseKinematicsCfg.create(...)` |
| `ik_solver.solve_batch(goal, num_seeds=N, return_seeds=N)` | `ik.solve_pose(GoalToolPose.from_poses({tool: pose}, num_goalset=1), return_seeds=N)` |
| `result.solution` (shape `(B, S, dof)`) | `result.js_solution.position` (shape 동일) |
| `result.success` | 동일 |
| `RobotWorld` / `RobotWorldConfig` (FK 용) | `Kinematics(KinematicsCfg.from_robot_yaml_file(...))` |
| `state.ee_position`, `state.ee_quaternion` | `state.tool_poses.get_link_pose(tool).position`, `.quaternion` |
| `RobotWorld.get_world_self_collision_distance_from_joints(q)` | `RobotCollisionChecker` (단, 버그로 우회 필요 — 아래 참조) |
| `MotionGen` / `MotionGenConfig.load_from_robot_config(..., interpolation_dt=...)` | `MotionPlanner` / `MotionPlannerCfg.create(...)` |
| `motion_gen.plan_single_js(start, goal, MotionGenPlanConfig(max_attempts, enable_*, timeout))` | `planner.plan_cspace(goal_state, current_state, max_attempts=N, enable_graph_attempt=1)` — `enable_*`/`timeout` 은 plan‑time 인자에서 사라지고 cfg 레벨로 흡수 |
| `result.get_interpolated_plan().position` | 동일 |

**불변 사항**: 쿼터니언 wxyz, 충돌 거리 부호 (음수 = 충돌), `Cuboid(name=, pose=[x,y,z,qw,qx,qy,qz], dims=)` / `Mesh(name=, pose=, vertices=, faces=)` 시그니처.

## 주요 사용 패턴 변경

### 충돌 세계 주입 (프로그래매틱 Scene)

옛 API 는 `IKSolverConfig.load_from_robot_config(robot_cfg, world_config, ...)` 로 한 번에 넘김.
새 API 는 `*Cfg.create(scene_model=..., collision_cache=...)` 로 충돌 인프라를 미리 할당하고,
런타임에 `solver.update_world(scene)` 로 채워 넣는다. `scene_model={}` 로 빈 placeholder 를 넘기면
인프라만 만들어진다.

```python
cfg = InverseKinematicsCfg.create(
    robot=robot_cfg_dict,
    scene_model={},
    self_collision_check=True,
    num_seeds=num_seeds,
    max_batch_size=batch_size,
    use_cuda_graph=False,
    collision_cache={"obb": n_cuboids, "mesh": n_meshes},
)
ik = InverseKinematics(cfg)
ik.update_world(world_scene)   # 우리가 만든 Scene 주입
```

### Multi‑seed IK

```python
goal = Pose(position=positions_t, quaternion=quats_wxyz_t)   # (B,3), (B,4)
result = ik.solve_pose(
    GoalToolPose.from_poses({ik.tool_frames[0]: goal}, num_goalset=1),
    return_seeds=num_seeds,
)
solutions = result.js_solution.position   # (B, num_seeds, dof)
success   = result.success                # (B, num_seeds)
```

### Joint‑to‑joint motion planning (transit)

```python
start = JointState.from_position(start_q.unsqueeze(0), joint_names=planner.joint_names)
goal  = JointState.from_position(goal_q.unsqueeze(0),  joint_names=planner.joint_names)
result = planner.plan_cspace(goal, start, max_attempts=10)
if result is not None and result.success.any():
    waypoints = result.get_interpolated_plan().position.squeeze(0).cpu().numpy()
```

## 우회가 필요한 v0.8 이슈

### 1. `RobotCollisionChecker` 의 high‑level 메서드 버그

`get_scene_self_collision_distance_from_joints(q)` 가 내부적으로
`SceneCollisionCost.forward(state)` 를 호출하면서 `KinematicsState` 가 아닌 `Tensor` 를
넘겨서 `AttributeError: 'Tensor' object has no attribute 'robot_spheres'` 가 난다.

**우회**: low‑level 컴포넌트를 직접 호출.

```python
q3 = q_traj.unsqueeze(1)                                   # (B, 1, dof)
state = checker.get_kinematics(q3)
num_spheres = state.robot_spheres.shape[-2]
checker.collision_cost.update_num_spheres(num_spheres, batch_size=B, horizon=1)
checker.self_collision_cost.setup_batch_tensors(B, 1)
d_scene = checker.collision_cost.forward(state)
d_self  = checker.self_collision_cost.forward(state.robot_spheres)
# 음수 = 충돌; trailing dim 어떻게 나오든 view(B,-1).any(-1) 로 축소
```

### 2. Robot YAML 호환성

옛 YAML (`ur20_with_camera.yml`) 의 다음 필드를 새 `KinematicsLoaderCfg` /
`CSpaceParams` 가 거부한다:

- 최상위: `usd_path`, `usd_robot_root`, `isaac_usd_path`, `usd_flip_joints`,
  `usd_flip_joint_limits`, `link_names`
- `ee_link` → `tool_frames` (리스트) 로 이름 변경
- `cspace.retract_config` → `cspace.default_joint_position` 로 이름 변경

**우회**: `_resolve_robot_config()` 헬퍼가 (1) YAML 을 `ur20_description/` 또는
cuRobo content 경로에서 자동 탐색, (2) `urdf_path` / `asset_root_path` 를 절대경로로
패치, (3) legacy 키 변환·필터링, (4) 결과 dict 를 모든 `*Cfg.create(robot=...)` 에 전달.
이로써 cuRobo content 폴더에 robot config 를 별도로 옮길 필요가 사라졌다.

```python
robot_cfg = _resolve_robot_config(args.robot)              # dict
ik_cfg    = InverseKinematicsCfg.create(robot=robot_cfg, ...)
plan_cfg  = MotionPlannerCfg.create(robot=robot_cfg, ...)
kin       = Kinematics(KinematicsCfg.from_robot_yaml_file(robot_cfg))
coll_cfg  = RobotCollisionCheckerCfg.load_from_config(robot_config=robot_cfg, ...)
```

### 3. MotionPlanner transit 의 interpolation 길이

옛 `MotionGenConfig(interpolation_dt=0.02)` 와 새 `MotionPlannerCfg.create()` 의
trajopt 기본값이 다르다. 현재 sample/124 실행에서 transit 이 "1 waypoint" 로 반환되는
경우가 있음 — Phase 5 의 cumulative L2 resample 단계에서 dense path 로 채워지므로
최종 출력에는 영향 없으나, 더 매끄러운 transit 이 필요하면 `MotionPlannerCfg.create` 에
trajopt cfg 를 명시 전달해야 함.

### 4. CUDA 12 통일 (Isaac Sim 6.0.0 호환)

NVIDIA 가 Isaac Sim 6.0.0 의 PyPI 휠을 **CUDA 12 빌드만** 제공한다 — `isaacsim-core`
가 `nvidia-cublas-cu12==12.8.4.1` 를 고정 의존성으로 박아두고 런타임에
`libcublas.so.12` 를 dlopen 한다. cu13 의 `libcublas.so.13` 은 ABI 가 다른 별개 파일이라
대체 불가.

Isaac Sim 과 cuRobo 를 **같은 venv** 에 공존시키려면 cu12 로 통일한다:

| 항목 | 값 |
|---|---|
| Python | 3.12 (`requires-python = ">=3.12,<3.13"`) — Isaac Sim 6.0.0 요건 |
| PyTorch 인덱스 | `https://download.pytorch.org/whl/cu128` (cu130 → cu128 변경) |
| PyTorch | `2.10.0+cu128` |
| cuRobo extra | `cu12-torch` (`uv pip install "./curobo[cu12-torch]"`) |
| Isaac Sim | `isaacsim[extscache,ros2]==6.0.0` (NVIDIA 인덱스 추가 필요) |
| `[tool.uv]` | `prerelease = "allow"` (isaacsim 의 `tinyobjloader==2.0.0rc13` 때문) |
| `warp-lang` | `>=1.12,<1.13` 핀 (1.13 이 `wp.torch` 제거, cuRobo 와 충돌) |

`pyproject.toml` 인덱스 블록:

```toml
[tool.uv]
index-strategy = "unsafe-best-match"
prerelease = "allow"

[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu128"

[[tool.uv.index]]
name = "pypi"
url = "https://pypi.org/simple"

[[tool.uv.index]]
name = "nvidia"
url = "https://pypi.nvidia.com"
```

설치 순서:

```bash
uv sync                                    # Isaac Sim, torch+cu128, NVIDIA libs
uv pip install "./curobo[cu12-torch]"      # cuRobo (path-install, lock 외)
```

> **운영 주의**: cuRobo 는 path-install 이라 `uv.lock` 에 들어가지 않는다.
> `uv sync` 를 다시 돌리면 cuRobo 가 제거되니, 위 두 번째 줄을 다시 실행하거나
> `uv run --no-sync ...` 로 sync 단계를 건너뛰어 실행한다.

## 검증 (sample/124, num_seeds=32)

```
Phase 1: 26.2% IK success (1041/3968)
Phase 2: avg 3.7 representatives/viewpoint
Phase 3: 15 reconfigs, max_jump=270.1°
Phase 4: 6/8 transit OK
Phase 5: 287 waypoints, collisions=0
출력: trajectory_dp_s010.{csv,html,_anim.html}
```

## 영향받는 파일

- `scripts/core/plan_trajectory.py` — 본 마이그레이션의 유일한 코드 변경 대상
- `README.md` — cuRobo 설치 명령을 새 빌드 익스트라 (`./curobo[cu13-torch]`) 로 갱신
- `data/sample/trajectory/124/trajectory_dp_s010.{csv,html,_anim.html}` — 새 API 로
  재생성된 참고 출력

`scripts/prev/` 의 옛 스크립트들도 같은 옛 API 를 쓰지만 운영에서 호출되지 않으므로
이번 작업 범위에서 제외.
