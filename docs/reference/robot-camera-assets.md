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
| `workcell/robot/ur20_with_camera.xrdf` | cuMotion(ROS2) 용 로봇 정의(카메라 스피어, tool_frames) | MoveIt/cuMotion launch — `moveit_assets.prepare_xrdf` 가 파생 |
| `scripts/common/config.py` | WD·FOV 등 **운용 파라미터** | 파이프라인 전역 |

### URDF 는 `_curobo` 하나뿐이다

`workcell/robot/` 의 URDF 는 `ur20_with_camera_curobo.urdf` **하나**다. 헷갈릴 짝이 없다.

예전에는 `_curobo` 없는 `ur20_with_camera.urdf` 와 `config.DEFAULT_URDF_PATH` 가 함께
있었는데 **둘 다 삭제했다**(둘 다 참조하는 코드가 0곳). 그 파일은 카메라를 `tool0` 에
프리미티브 박스·실린더로 붙인 옛 표현이었고 `camera_optical_joint` 도 `xyz="-0.212 0.03 0"`
으로 현행 `0.21877` 과 부호까지 달랐다 — 기준으로 삼으면 조용히 어긋나는 종류의 파일이다.
명목상 역할이던 MorphIt 원본 기능도 실제로는 못 했다: 이 파일을 만든 `.urdf.xacro` 도,
MorphIt 빌더도 리포에 없어서 재생성이 불가능했다.

내용이 필요하면 git 이력에 있다(`git log --follow -- workcell/robot/ur20_with_camera.urdf`).
**새 URDF 를 추가하지 말 것** — 운동학의 소유자는 `_curobo.urdf` 단 하나다.

### MoveIt 도 같은 로봇을 쓴다 (2026-08-23~)

`scripts/moveit/` 의 launch 둘은 `workcell/` 의 원본에서 MoveIt 입력을 **파생시킨다**.
파생은 [moveit_assets.py](../../scripts/moveit/moveit_assets.py) 가 맡고, 산출물은
리포가 아니라 `/tmp` 에 쓴다 — 원본을 고친 뒤 재생성을 잊어 MoveIt 만 옛것을 보는 일이
구조적으로 불가능하게(기존 xacro → `/tmp/collated_ur20_urdf.urdf` 와 같은 패턴).

| MoveIt 입력 | 원본 | 파생 시 손보는 것 |
|---|---|---|
| robot_description | `ur20_with_camera_curobo.urdf` (+ `ur_config/ur_camera.urdf.xacro` 껍데기) | ros2_control 교체, 카메라 메시 `package://` → 절대경로 |
| xrdf | `ur20_with_camera.xrdf` | 월드충돌용 스피어 집합 파생(아래) |
| planning scene | `workcell/scenes/{name}.yaml` | `.scene` 생성 |
| SRDF | 벤더 `ur_moveit_config` + `ur_config/ur_camera.srdf.xacro` | 카메라 자기충돌 예외 1쌍 |

**기하값은 복사하지 않는다** — xacro 가 `_curobo.urdf` 를 `include` 한다. 카메라를 옮기면
MoveIt 이 자동으로 따라온다. 팔 기구학까지 한 파일에서 온다.

두 군데는 값을 그대로 못 쓴다:

- **`base_link_inertia` 스피어를 월드충돌에서 뺀다.** 로봇 베이스 바닥과 `robot_mount`
  상면이 정확히 같은 평면(볼트 체결면)이라, 평평한 면을 구로 덮는 한 반드시 튀어나온다
  (스피어 59개로도 ~16mm 남는다) → cuMotion 이 모든 시작 자세를 world collision 으로
  거부한다. Inspection 은 [settings.py](../../scripts/core/trajectory/settings.py)
  `COLLISION_EXCLUDE_LINKS` 로 같은 링크를 이미 빼고 있어 **그 목록을 읽어 공유한다.**
  `self_collision` 은 원래 집합을 계속 가리킨다 — 팔이 자기 베이스에 닿는 것은 실제로
  일어난다(무작위 자세의 4.2%). `robot_mount` 자체는 빼면 안 된다: 실제 작업 자세가
  전부 기둥 10cm 이내에서 움직이고 무작위 자세의 36.5%가 기둥을 침투한다.
- **SRDF 에 `camera_link` ↔ `wrist_3_link` 예외.** 벤더 SRDF 는 camera 를 모른다.
  `upper_arm_link` 는 **열지 않는다** — 팔을 접으면 카메라가 상완에 실제로 박는다.

미해결: planning group 의 tip 은 `tool0` 인데 xrdf `tool_frames` 는
`camera_optical_frame` 이라 원리적으로 정합하지 않는다(현재 계획은 된다).


## 3. 카메라 프레임 체인

```
wrist_3_link
  └─(wrist_3-flange, fixed, rpy 0,-90°,-90°)→ flange
       ├─(flange-tool0, fixed)→ tool0
       └─(camera_mount_joint, fixed, rpy -90°,0,0)→ camera_link  ← 메시가 이 프레임에 pre-bake
            └─(camera_optical_joint, xyz="0.21877 0 0", rpy 90°,0,90°)→ camera_optical_frame
```

- `camera_mount_joint` 의 rpy 는 **카메라가 플랜지에 물린 clocking** 이다. roll 만 쓴다:
  회전축이 툴 축(flange **+X** = 광축)이라 **광축 방향도 `camera_optical_frame` 원점도
  움직이지 않는다**. 그래서 `0.21877` 은 clocking 과 무관하게 flange 기준 광축 거리 그대로다.
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
             = 0.17223 m                 = 0.21877 m
                              합 = 0.391 m = CAD VIEW_1 검사면
```

- **WD 만 튜닝 대상**이다. 기준점은 **카메라의 끝**(렌즈 배럴 앞면)이라 현장에서 자로
  "카메라 끝 → 물체" 를 재면 그 값이다. ⚠️ 벤더 공칭 WD(250mm)는 `body_face` 기준이라
  다른 숫자다 — 환산은 `새 WD = 구 WD − 77.770` (camera-geometry.md §B).
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
`object_plane == lens_front + CAMERA_WORKING_DISTANCE_MM` 로 기본값을 CAD 에 고정한다.

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
3. `workcell/robot/ur20_with_camera.usd` 의
   `/Root/UR20/wrist_3_link/flange/camera_mount/camera_optical_frame` 트랜스폼 —
   `xformOp:transform` 의 **병진 행만** 바꾼다(회전 basis 는 flange +X → optical +Z 규약).
   그다음 ghost 재생성: `uv run --no-sync scripts/setup/build_ghost_usd.py`
   (ghost 는 산출물이다. 재생성하면 `Flattened_Prototype_NN` 번호가 갈려 바이너리 diff 가
   크게 잡히는데, 실제 의미 차이는 이 트랜스폼 하나뿐이다 — 2026-08-27 구조 diff 로 확인)
4. `scripts/common/config.py` — `TOOL_TO_CAMERA_OPTICAL_OFFSET_M`,
   그리고 물체면을 유지하려면 `CAMERA_WORKING_DISTANCE_MM` 을 상보적으로
5. 문서 — [camera-geometry.md](camera-geometry.md), [configuration.md](configuration.md)
6. **viewpoint h5 재생성** — h5 의 `metadata/camera_spec/working_distance_mm` 가
   config 보다 우선하므로([storage.py](../../scripts/core/viewpoint/storage.py)),
   옛 파일을 그대로 읽으면 물체면이 틀린 자리에 잡힌다.
   ⚠️ **경고에 기대지 말 것.** `working_distance_error()` 는 *물리적으로 불가능한* 값만
   잡는다. 프레임을 옮긴 뒤 옛 WD 숫자가 여전히 유효 범위면 조용히 통과한다 — 실제로
   2026-08-27 이전 h5(250·273 등)가 그 상태다(camera-geometry.md §미해결).
   `data/*/trajectory/*/solution.h5` 와 `data/*/ik/*` 의 `working_distance_m` 도 같다.
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
