# 주요 설정값

공유 설정은 `scripts/common/config.py`, 궤적 기본값은 `scripts/core/trajectory/settings.py`,
셀 기하(장애물·물체 배치)는 `workcell/scenes/{name}.yaml`에 있다. 문서보다 코드의 현재 값을 우선한다.

## 값을 어디에 둘 것인가

새 설정을 추가할 때 이 질문 하나로 정한다:

> **현장을 옮기면 바뀌는가?**

| | 예 → `workcell/scenes/{name}.yaml` | 아니오 → `scripts/common/config.py` |
|---|---|---|
| 예 | 테이블·벽·받침 위치와 치수, 물체 배치, **마운트 기둥 높이** | 카메라 WD·FOV·광축 오프셋, 로봇 config 파일명, 알고리즘 튜닝, 경로 헬퍼, 물체별 메시 예외 |

`config.py` 는 씬 값을 **소유하지 않는다.** `TARGET_OBJECT` / `OBSTACLES` / `TABLE` /
`ROBOT_MOUNT` / `WALLS` / `OBJECT_PLACEMENTS` / `MOUNT_HEIGHT` 는 빈 껍데기로 선언돼 있고
`load_scene()` 이 YAML 에서 읽어 채운다(façade). 소비자가 YAML 을 직접 파싱하지 않게 하려는
것이지 값이 두 곳에 있는 것이 아니다 — **같은 숫자를 두 파일에 적지 않는다.**

⚠️ `MOUNT_HEIGHT` 는 `robot_mount` 장애물의 `dimensions[2]` 에서 파생된다. 기둥 높이를 바꾸려면
YAML 의 그 상자를 고치고, **`position[2]` 도 같이 내려 상면이 z=0(로봇 베이스 판)에 오게** 한다.
어긋나면 `scene_config` 가 로드 시점에 거절한다.

⚠️ 씬을 전환할 수 있으므로(`--scene`) 씬 파생값은 **import 시점에 스냅샷하지 않는다.**
호출 시점에 `config.MOUNT_HEIGHT` 를 읽는다.

## 카메라

용어·기준점은 [camera-geometry.md](camera-geometry.md)를 단일 진실원으로 한다.

아래 FOV·WD·overlap 은 **기본값**이다. 실행별 값은 viewpoint 생성 시 고르고(스튜디오 "Camera spec"
폴더 또는 `viewpoint/cli.py --working-distance/--fov-width/--fov-height/--overlap`), h5
`metadata/camera_spec` 에 저장되어 **그 h5 를 읽는 쪽이 config 보다 우선**한다.

| 항목 | 값 | 코드 심볼 |
|---|---|---|
| FOV_footprint 가정 | 50 × 50 mm (트릭, 실광학 아님. CAD `VIEW_1` 과 일치) | `CAMERA_FOV_WIDTH/HEIGHT_MM` |
| WD (optical_frame=**카메라 끝**→object. 실사용 값 = 구 기준 273 − 77.770) | 195.23 mm | `CAMERA_WORKING_DISTANCE_MM` |
| WD 하한 (검사면이 카메라 끝보다 앞) | 0 mm | `CAMERA_MIN_WORKING_DISTANCE_MM` |
| mount_offset (flange→optical_frame=렌즈 앞면) | 0.21877 m (하드웨어 상수, URDF가 소유) | `TOOL_TO_CAMERA_OPTICAL_OFFSET_M` |
| lens_front (flange→렌즈 앞면) | 0.21877 m (CAD 실측) | `CAMERA_LENS_FRONT_OFFSET_M` |
| overlap | 0.5 | `CAMERA_OVERLAP_RATIO` |
| 렌더 near clip | 10 mm (기하 파생 아님 — 카메라 원점 앞을 가리는 부품이 없다) | `CAMERA_NEAR_CLIP_M` |
| 퍼블리시 해상도 | FOV 비율에서 유도 (50×50 → 880×880) | `CAMERA_PUBLISH_W/H` |
| 렌즈 / 센서 | MFA121-U50 f=50mm / AR0820 8.08×4.55mm | — |

## 로봇과 충돌

| 항목 | 기본값 |
|---|---|
| robot config | `ur20_with_camera.yml` |
| joint 순서 | shoulder pan, shoulder lift, elbow, wrist 1, wrist 2, wrist 3 |
| 시작 자세 | `ROBOT_START_STATE` |
| collision margin | 0 m |

## 씬 (장애물·물체 배치)

물체 위치, 테이블, 벽, support 형상은 **`workcell/scenes/{name}.yaml`** 이 소유한다. 스키마와
검증은 `scripts/common/scene_config.py`, 값은 `config.TABLE/WALLS/ROBOT_MOUNT/OBSTACLES/
OBJECT_PLACEMENTS` 로 노출된다(로드는 config import 시점에 1회). 설정을 바꾼 뒤에는 viewpoint
pose, IK와 충돌 검사를 다시 실행한다.

씬은 여러 개 둘 수 있고 모든 진입점이 `--scene {이름|경로}` 를 받는다(기본 `sim_default`).
시뮬 셀과 실측 셀을 나란히 두고 비교하는 것이 목적이다.

| 필드 | 설명 |
|---|---|
| `version` | 스키마 버전(현재 1) |
| `target_object` | `object_placements` 에 없는 물체의 기본 pose |
| `obstacles[]` | `name`, `type`, `position`, `rotation`(선택), 타입별 치수, `isaac_visual` |
| `object_placements` | 물체별 `position`/`rotation` override |

좌표는 전부 **robot base_link frame**(m), 회전은 쿼터니언 `[w,x,y,z]`.
Isaac world z = robot z + `MOUNT_HEIGHT`(0.805).

**예약 이름** `table` / `robot_mount` / `support` — `sync_support_to_target()` 과
`core/isaac/scene.py` 가 이름으로 찾으므로 바꾸면 안 된다. `table` 은 회전 불가(identity):
support 높이를 `position[2] + dimensions[2]/2` 로 구해 축정렬을 가정한다.

**⚠️ 프리미티브 타입**: cuRobo 0.8 은 `sphere`/`cylinder`/`capsule` 을 충돌 검사에 **넣지
않는다**(예외도 경고도 없이 무시). 그래서 `scene_config.obstacle_obb()` 가 전부 OBB 로 바꿔서
넘긴다 — 플래너·viser·Isaac 이 같은 함수를 쓰므로 화면에 보이는 것이 곧 충돌 월드다. 정확한
비-박스 형상이 필요하면 cuboid 여러 개로 나누는 편이 낫다.

`isaac_visual` 토큰(어휘 소유자는 `core/isaac/scene.py`): `usd_table`/`usd_mount`(전용 USD 자산이
이미 배치됨), `hidden`(프림은 만들되 invisible — Stage 트리에서 켜면 보인다), `primitive`(반투명 박스).

**Isaac 에서 잰 값을 넣는 법**: 뷰포트에서 prim 을 고르고 Scene 패널의
`Log Selected Prim as YAML` → 로그에 찍힌 조각을 씬 YAML 에 붙여넣는다.
기즈모로 옮기기만 하고 저장하지 않으면 **계획에 반영되지 않는다** — 스테이지는 플래너의
진실원이 아니다(타깃 물체 pose 만 예외적으로 실시간 전달된다).

**재현성**: `solve.py` 는 해결된 씬 스냅샷을 `solution.h5` 에 박제하고, `verify.py` 와
`trajectory_studio` 는 이름이 아니라 그 스냅샷으로 월드를 되살린다. 그래서 solve 이후 YAML 을
편집해도 검증은 같은 셀을 본다.

씬이 **아닌** 것(물체/로봇 속성이라 셀을 바꿔도 안 변한다): `OBJECT_COLLISION_SHAPE`,
`OBJECT_FILTER_INTERIOR`, `OBJECT_TARGET_MATERIAL`, `MOUNT_HEIGHT`, 카메라 스펙 — 계속
`config.py` 에 있다.

경로 생성에는 `get_mesh_path`, `resolve_viewpoint_path`, `get_solution_path`, `get_trajectory_artifact_path` 헬퍼를 사용한다. 파일명 규칙은 [데이터 형식](data-formats.md)에 있다.
