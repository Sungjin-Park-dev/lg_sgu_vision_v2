# 로봇·카메라 에셋 정의 지도

"UR20 로봇과 카메라는 **어느 파일에** 정의돼 있는가"에 대한 답. 같은 로봇이 YAML·URDF·USD·XRDF에
나뉘어 있고 각각을 읽는 소비자가 달라서, 하나만 고치면 조용히 어긋난다.

기하 값·용어의 정의는 [camera-geometry.md](camera-geometry.md)가 단일 진실원이다.
이 문서는 **어디에 무엇이 있고 무엇을 같이 고쳐야 하는가**를 다룬다.

## 1. IK·모션플래닝이 실제로 읽는 것

```
config.DEFAULT_ROBOT_CONFIG = "ur20_with_camera.yml"        scripts/common/config.py
        ↓  core/trajectory/settings.py: ROBOT_CONFIG
        ↓  core/trajectory/robot.py: resolve_robot_config()
           탐색 ① workcell/robot/<name>  ② cuRobo content/configs/robot/<name>
           urdf_path 를 파일명만 취해 workcell/robot/ 절대경로로 재작성
        ↓
workcell/robot/ur20_with_camera.yml                          ← cuRobo 진입점
        urdf_path: ur20_with_camera_curobo.urdf
        base_link: base_link,  tool_frames: [camera_optical_frame]
        collision_spheres / self_collision_ignore / cspace
        ↓
workcell/robot/ur20_with_camera_curobo.urdf                  ← 운동학의 소유자
```

`resolve_robot_config()` 를 거치는 진입점: [pipeline.py](../../scripts/core/trajectory/pipeline.py),
[check_ik.py](../../scripts/core/trajectory/check_ik.py),
[glns/solve.py](../../scripts/core/glns/solve.py), [glns/verify.py](../../scripts/core/glns/verify.py),
[live_ik.py](../../scripts/core/trajectory/live_ik.py).
`isaac_pipeline.py` 는 궤적 생성을 `uv run` 서브프로세스로 던지므로 결국 같은 경로를 탄다.

## 2. 파일별 역할

| 파일 | 소유하는 것 | 읽는 쪽 |
|---|---|---|
| `workcell/robot/ur20_with_camera.yml` | cuRobo 로봇 정의(충돌 스피어, tool_frames, cspace) | cuRobo IK·모션플래닝 전부 |
| `workcell/robot/ur20_with_camera_curobo.urdf` | **링크·조인트 기하** (`camera_optical_joint` 포함) | cuRobo, yourdfpy(viser 렌더), isaac_pipeline 스피어 시각화 |
| `workcell/robot/ur20_with_camera.usd` | Isaac Sim 실로봇 (물리·아티큘레이션) | `core/isaac/scene.py` |
| `workcell/robot/ur20_with_camera_ghost.usd` | Isaac preview ghost (물리 제거) | `isaac_pipeline.py` |
| `workcell/robot/camera/camera.usdc`, `camera_body.obj` | 카메라 몸체 메시(mount=camera_link 프레임에 pre-bake) | 위 USD 2개, URDF visual/collision, MorphIt |
| `workcell/robot/ur20_with_camera.xrdf` | cuMotion(ROS2) 용 로봇 정의(카메라 포함) | ⚠️ **현재 아무도 안 읽는다** — 아래 참고 |
| `scripts/common/config.py` | WD·FOV 등 **운용 파라미터** | 파이프라인 전역 |

### URDF 는 `_curobo` 하나뿐이다

`workcell/robot/` 의 URDF 는 `ur20_with_camera_curobo.urdf` **하나**다. 헷갈릴 짝이 없다.

예전에는 `_curobo` 없는 `ur20_with_camera.urdf` 와 `config.DEFAULT_URDF_PATH` 가 함께
있었는데 **둘 다 삭제했다**(둘 다 참조하는 코드가 0곳). 그 파일은 카메라를 `tool0` 에
프리미티브 박스·실린더로 붙인 옛 표현이었고 `camera_optical_joint` 도 `xyz="-0.212 0.03 0"`
으로 현행 `0.141` 과 부호까지 달랐다 — 기준으로 삼으면 조용히 어긋나는 종류의 파일이다.
명목상 역할이던 MorphIt 원본 기능도 실제로는 못 했다: 이 파일을 만든 `.urdf.xacro` 도,
MorphIt 빌더도 리포에 없어서 재생성이 불가능했다.

내용이 필요하면 git 이력에 있다(`git log --follow -- workcell/robot/ur20_with_camera.urdf`).
**새 URDF 를 추가하지 말 것** — 운동학의 소유자는 `_curobo.urdf` 단 하나다.

### ⚠️ MoveIt 은 이 폴더의 로봇을 안 쓴다 (카메라가 없다)

`scripts/moveit/` 의 launch 둘은 `workcell/robot/` 를 전혀 참조하지 않는다:

| launch | robot_description | xrdf |
|---|---|---|
| `ur20_isaac_state_synced.launch.py` | `scripts/moveit/ur_config/ur_gated.urdf.xacro` | `scripts/moveit/ur20.xrdf` |
| `ur20_real_moveit.launch.py` | `isaac_ros_cumotion_examples/ur_config/ur.urdf.xacro` (외부 패키지) | 〃 |

`ur_gated.urdf.xacro` 는 `ur_description/urdf/ur_macro.xacro`(순정 UR20) + `world` +
ros2_control 이 전부고 **카메라 링크·조인트가 없다**. `scripts/moveit/ur20.xrdf` 도
`camera` 문자열 0 회에 `tool_frames: ["tool0"]` 이다. 즉 **MoveIt/cuMotion 은 카메라가
충돌 모델에 없는 맨 UR20 으로 계획한다.** 정작 카메라를 가진
`workcell/robot/ur20_with_camera.xrdf`(camera_link 스피어, `tool_frames:
camera_optical_frame`)는 어느 launch 의 기본값도 아니고 override 하는 코드도 없다.

그래서 카메라 기하 변경(clocking·optical_frame 이동)은 **MoveIt 에 아무 영향이 없다** —
반영할 카메라가 거기 없기 때문이다. 이 격차 자체는 따로 다뤄야 할 문제다.


## 3. 카메라 프레임 체인

```
wrist_3_link
  └─(wrist_3-flange, fixed, rpy 0,-90°,-90°)→ flange
       ├─(flange-tool0, fixed)→ tool0
       └─(camera_mount_joint, fixed, rpy -90°,0,0)→ camera_link  ← 메시가 이 프레임에 pre-bake
            └─(camera_optical_joint, xyz="0.141 0 0", rpy 90°,0,90°)→ camera_optical_frame
```

- `camera_mount_joint` 의 rpy 는 **카메라가 플랜지에 물린 clocking** 이다. roll 만 쓴다:
  회전축이 툴 축(flange **+X** = 광축)이라 **광축 방향도 `camera_optical_frame` 원점도
  움직이지 않는다**. 그래서 `0.141` 은 clocking 과 무관하게 flange 기준 광축 거리 그대로다.
  바뀌는 것은 몸체가 어느 쪽으로 뻗는지와 이미지 회전뿐이다 —
  flange 좌표 bbox 가 `y[-94.9,+50] z[-56,+56]` → `y[-56,+56] z[-50,+94.9] mm` 로 옮겨간다.
- rpy(90°,0,90°) 가 flange **+X** 광축을 optical_frame **+Z** 광축으로 돌린다
  (플래너·USD 카메라 공통 규약).

- `camera_optical_frame` 은 단순 표식이 아니라 **IK 목표 프레임 자체**다
  (`tool_frames[0]`, [ik.py](../../scripts/core/trajectory/ik.py) `solve_pose`,
  [robot.py](../../scripts/core/trajectory/robot.py) `compute_fk`).
  Isaac 의 `InspectionCamera` 도 이 prim 아래에 붙는다
  ([scene.py](../../scripts/core/isaac/scene.py) `setup_inspection_camera`).

### clocking 을 바꿀 때 (≠ optical_frame 이동, §5 와 다른 작업)

두 곳이 같은 값의 사본이고, 한쪽만 고치면 **계획(cuRobo/URDF)과 화면(USD)이 다른 로봇**이
된다 — 어느 쪽도 에러를 내지 않는다.

1. `ur20_with_camera_curobo.urdf` — `camera_mount_joint` 의 `origin rpy`
2. `ur20_with_camera.usd` — `camera_mount` 의 `xformOp:rotateXYZ` (deg)
3. `build_camera_mesh.py` — `CAMERA_MOUNT_RPY_DEG` (1·2 를 맞춰보는 기준값)
4. ghost 재생성 — `uv run --no-sync scripts/setup/build_ghost_usd.py`
   (ghost 는 로봇 USD 에서 만들어지는 산출물이다. 직접 고치지 말 것)
5. 검증 — `uv run --no-sync scripts/setup/build_camera_mesh.py --verify-only --ghost`
   가 URDF·USD·상수 셋을 대조한다.

**h5 재생성은 필요 없다** — viewpoint 위치도 WD 기하도 안 건드린다. 대신 카메라 몸체가
도는 만큼 **충돌 형상이 실제로 바뀌므로** 도달성/충돌은 다시 봐야 한다(`check_ik.py`).

## 4. 목표 자세가 계산되는 방식

```
flange 목표 = 표면점 + 법선 × (WD + mount_offset)
                              ↑          ↑
       CAMERA_WORKING_DISTANCE_MM   URDF camera_optical_joint (하드웨어 상수)
              = 0.250 m                    = 0.141 m
                              합 = 0.391 m = CAD VIEW_1 검사면
```

- **WD 만 튜닝 대상**이다. 이 값은 벤더 공칭 WD 와 같은 기준점(카메라 몸체 앞면)을 쓴다.
- **mount_offset 은 하드웨어 사실**이다. 바꾸려면 §5 체크리스트 전체를 밟아야 한다.
- 실제 적용 지점: [poses.py](../../scripts/core/trajectory/poses.py)
  `camera_positions = positions + normals * working_distance_m`.

### WD/FOV 는 실행별 값이고, h5 가 그것을 나른다

config 는 기본값이다. 실제 값은 **viewpoint 생성 시점**에 정해진다 —
스튜디오의 "Camera spec" 폴더 또는 `viewpoint/cli.py --working-distance/--fov-width/
--fov-height/--overlap`. 고른 값은 `ViewpointGenParams` 가 소유하고
([models.py](../../scripts/core/viewpoint/models.py), `__post_init__` 이 미지정분을 config 로 해소),
h5 `metadata/camera_spec` 에 저장된다.

```
ViewpointGenParams (생성 시 선택)
   → h5 metadata/camera_spec {fov_width_mm, fov_height_mm, working_distance_mm}
   → load_viewpoints_hdf5 → ViewpointData.working_distance_m / fov_width_m / fov_height_m
   → IK·궤적·GLNS (build_camera_poses)  +  Isaac 카메라 intrinsic·FOV 사각형
```

**h5 값이 config 를 이긴다.** 그래서 WD 를 바꾼 h5 를 만들면 하류가 자동으로 따라온다 —
다만 그 h5 를 실제로 새로 만들어야 한다. config 만 고치는 것으로는 기존 h5 가 바뀌지 않는다.
불가능한 WD(검사면이 렌즈 안쪽)는 `config.working_distance_error()` 가 CLI·스튜디오·h5 로드
세 지점에서 잡는다.

### 한 줄 규칙: config 는 출발값, h5 는 진실

> **config = 새 h5 를 만들 때의 출발값. h5 = 만들어진 뒤의 진실.**

config 를 없앨 수는 없다 — 첫 h5 를 만들 때 FOV/WD 가 어디선가는 와야 하는데 h5 는 그 결과물이라
닭-달걀이고, [inspect_camera_step.py](../../scripts/setup/inspect_camera_step.py) 가
`object_plane == body_face + CAMERA_WORKING_DISTANCE_MM` 로 기본값을 CAD 에 고정한다.

대신 "지금 어느 쪽이 적용 중인가"를 알 수 없는 구간을 없앴다:

- **h5 에 `camera_spec` 이 없으면 경고**한다(`storage.py`). 빠진 키를 이름으로 찍는다 —
  옛 파일이 조용히 config 값을 집어가는 유일한 경로였다.
- **trajectory_studio 는 아예 읽기 전용**으로 표시만 한다(`— from h5`). 그 앱에서 WD 는
  IK 대상 자세를 바꾸는데 생성 서브프로세스는 h5 를 읽으므로, 편집을 허용하면 화면과
  결과가 갈린다.

isaac_pipeline 의 카메라는 h5 를 고르기 전에 config 기본값으로 만들어지고,
h5 를 고르면(Browse 또는 Show Viewpoints) `_sync_camera_spec_from_h5` 가 그 값으로 맞춘다.
로그에 `[cam] spec <- snapshot ...` 로 남는다.

## 5. `camera_optical_frame` 을 옮길 때 동기화 체크리스트

한 곳만 고치면 조용히 어긋난다. 순서대로 전부.

1. `workcell/robot/ur20_with_camera_curobo.urdf` — `camera_optical_joint` 의 `origin xyz`
2. `scripts/setup/build_camera_mesh.py` — `OPTICAL_FRAME_X`
   (안 고치면 `--verify` 가 assert 로 막는다 — 의도된 가드)
3. `workcell/robot/ur20_with_camera.usd` 와 `..._ghost.usd` 의
   `/Root/UR20/wrist_3_link/flange/camera_mount/camera_optical_frame` 트랜스폼
4. `scripts/common/config.py` — `TOOL_TO_CAMERA_OPTICAL_OFFSET_M`,
   그리고 물체면을 유지하려면 `CAMERA_WORKING_DISTANCE_MM` 을 상보적으로
5. 문서 — [camera-geometry.md](camera-geometry.md), [configuration.md](configuration.md)
6. **viewpoint h5 재생성** — h5 의 `metadata/camera_spec/working_distance_mm` 가
   config 보다 우선하므로([storage.py](../../scripts/core/viewpoint/storage.py)),
   옛 파일을 그대로 읽으면 물체면이 틀린 자리에 잡힌다. 불일치 시 경고가 출력된다.
7. 검증 —
   ```bash
   uv run --no-sync scripts/setup/inspect_camera_step.py      # CAD 랜드마크가 그대로인지
   uv run --no-sync scripts/setup/build_camera_mesh.py --verify-only --ghost   # USD 2개
   uv run --no-sync scripts/core/trajectory/check_ik.py --object <obj> \
       --viewpoints <새 h5> --output /tmp/ik.h5                # 도달성 회귀 확인
   ```

`ur20_with_camera.yml` 과 `.xrdf` 는 좌표를 들고 있지 않다(링크 이름과 스피어만) —
이번 종류의 변경에서는 손댈 필요가 없다. 스피어는 `camera_link` 기준이라 optical_frame
이동과도, clocking 회전과도 무관하다(프레임째 같이 돈다 — 재피팅 불필요).

## 6. cuRobo config 재생성

`ur20_with_camera.yml` 은 MorphIt 이 생성한 산출물이다. 링크 하나의 스피어만 다시 맞출 때는
`--edit-config --refit-link` 로 해당 블록만 갈아끼운다. 자세한 절차는
[guides/prepare-object-assets.md](../guides/prepare-object-assets.md) 와
`scripts/setup/` 의 빌더들을 참고한다.
