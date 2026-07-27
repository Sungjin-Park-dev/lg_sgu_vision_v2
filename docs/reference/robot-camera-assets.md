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
| `workcell/robot/camera/camera.usdc`, `camera_body.obj` | 카메라 몸체 메시(flange 프레임에 pre-bake) | 위 USD 2개, URDF visual/collision, MorphIt |
| `workcell/robot/ur20_with_camera.xrdf` | cuMotion(ROS2) 용 로봇 정의 | MoveIt/cuMotion launch |
| `scripts/common/config.py` | WD·FOV 등 **운용 파라미터** | 파이프라인 전역 |

### 주의: 같은 이름의 낡은 파일

- **`workcell/robot/ur20_with_camera.urdf`** (`_curobo` 없는 쪽) — cuRobo 는 **읽지 않는다.**
  MorphIt 스피어 피팅용 원본이며 카메라 표현이 낡았다(프록시 박스 + `camera.stl`,
  `camera_optical_joint` 도 다른 값). 운동학 수정 시 기준으로 삼지 말 것.
- **`config.DEFAULT_URDF_PATH`** — 어디서도 참조되지 않는 죽은 상수. 컨테이너 내부 경로라
  현재 해석 경로와 무관하다.

## 3. 카메라 프레임 체인

```
wrist_3_link
  └─(wrist_3-flange, fixed, rpy 0,-90°,-90°)→ flange
       ├─(flange-tool0, fixed)→ tool0
       └─(camera_mount_joint, fixed, identity)→ camera_link   ← 메시가 flange 프레임에 pre-bake
            └─(camera_optical_joint, xyz="0.141 0 0", rpy 90°,0,90°)→ camera_optical_frame
```

- `camera_mount_joint` 가 identity 이므로 **`camera_link` 좌표계 = `flange` 좌표계**다.
  따라서 `camera_optical_joint` 의 `0.141` 은 그대로 flange 기준 광축 거리다.
- rpy(90°,0,90°) 가 flange **+X** 광축을 optical_frame **+Z** 광축으로 돌린다
  (플래너·USD 카메라 공통 규약).
- `camera_optical_frame` 은 단순 표식이 아니라 **IK 목표 프레임 자체**다
  (`tool_frames[0]`, [ik.py](../../scripts/core/trajectory/ik.py) `solve_pose`,
  [robot.py](../../scripts/core/trajectory/robot.py) `compute_fk`).
  Isaac 의 `InspectionCamera` 도 이 prim 아래에 붙는다
  ([scene.py](../../scripts/core/isaac/scene.py) `setup_inspection_camera`).

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
- **Isaac 카메라 스펙칸 아래에 출처를 표시**한다 — `source: h5 · <파일명>` /
  `config default` / `manual edit` / `viewport`. 카메라는 h5 를 고르기 전에 config 로
  만들어지므로, 부팅~h5 로드 사이 구간이 여기서 드러난다.
- **trajectory_studio 는 아예 읽기 전용**으로 표시만 한다(`— from h5`). 그 앱에서 WD 는
  IK 대상 자세를 바꾸는데 생성 서브프로세스는 h5 를 읽으므로, 편집을 허용하면 화면과
  결과가 갈린다.

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
이번 종류의 변경에서는 손댈 필요가 없다. 스피어는 `camera_link`(=flange) 기준이라
optical_frame 이동과 무관하다.

## 6. cuRobo config 재생성

`ur20_with_camera.yml` 은 MorphIt 이 생성한 산출물이다. 링크 하나의 스피어만 다시 맞출 때는
`--edit-config --refit-link` 로 해당 블록만 갈아끼운다. 자세한 절차는
[guides/prepare-object-assets.md](../guides/prepare-object-assets.md) 와
`scripts/setup/` 의 빌더들을 참고한다.
