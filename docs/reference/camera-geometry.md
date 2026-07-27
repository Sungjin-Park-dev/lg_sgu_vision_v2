# 카메라 기하·용어 표준 (단일 진실원)

카메라 위치·거리·광학 용어의 **표준 정의**. 코드 주석·다른 문서·논의는 모두 이 용어를 따른다.
혼란의 근본 원인은 "거리"를 서로 다른 기준점에서 재고 같은 이름으로 부른 것이었다 —
그래서 모든 거리는 **"어디서 → 어디까지"를 이름에 못박는다.**

값은 CAD `workcell/robot/camera/source/camera_asm_wo_light.stp` **실측**이다
(2026-07-27, [§CAD 실측](#cad-실측-값의-출처) 참고). 미확정 항목은 §미해결에 명시한다.

## A. 기준점 (광축 위, flange 프레임 기준 미터)

```
flange ──── sensor ──── optical_frame ──── lens_front ──[?pupil]──── object_plane
 0.000      0.1243        0.141              0.2188                    0.391   [m]
                          └───────────── WD 0.250 (벤더 공칭) ────────────┘
```

| 표준어 | 뜻 | 위치 | 근거 |
|---|---|---|---|
| **flange** | EE 연결면 | 0.000 | `ZIVID_END_EFFECTOR_UR20_PT2` 뒷면 = ROS-I `flange` 링크 |
| **sensor** | 이미지 센서 기판 앞면 | 0.1243 | `PCB_2` (t1.6). 센서 다이/패키지는 CAD 미모델링 |
| **optical_frame** | `camera_optical_frame` = **body_face** | **0.141** | 커버 앞면. **코드 tool frame**, 벤더 WD 기준점 |
| **lens_front** | 렌즈 앞면(배럴 끝) | 0.2188 | `MFA121-U50` 배럴 Ø38.6 (실린더 반경 19.3 실측) |
| **pupil** | 광학중심(입사동공) | 렌즈 내부(미확정) | 핀홀 투영중심. CAD에 없음 |
| **object_plane** | 물체면 = 검사면 = 초점면 | 0.391 | `VIEW_1` — 50×50mm 판의 근접면 |

## B. 거리 — 이름에 기준점을 못박음

| 표준어 | 정의 (from → to) | 값 | 코드 심볼 |
|---|---|---|---|
| **mount_offset** | flange → optical_frame | 0.141 m | `TOOL_TO_CAMERA_OPTICAL_OFFSET_M` (URDF가 소유) |
| **WD** = **frame_standoff** | optical_frame(=body_face) → object_plane | **250 mm** | `CAMERA_WORKING_DISTANCE_MM` |
| **flange_to_object** | flange → object_plane | 391 mm | mount_offset + WD |
| **lens_WD** | lens_front → object_plane | 172.2 mm | (코드에 없음) |
| **sensor_to_object** | sensor → object_plane | 266.7 mm | (코드에 없음) |

> **`CAMERA_WORKING_DISTANCE_MM` 은 이제 벤더가 말하는 그 WD 다.** `optical_frame` 이 카메라
> 몸체 앞면에 놓여 있으므로 이 값을 바꾸면 물체면이 **벤더 스펙과 같은 의미로** 이동한다.

### WD 를 실제로 조절하는 법

config 값은 **기본값**일 뿐이고, 실행별 값은 viewpoint 생성 시점에 정한다:

```bash
uv run scripts/apps/viewpoint_studio.py --object sample     # Camera spec 폴더에서 조절
uv run scripts/core/viewpoint/cli.py --object sample --working-distance 150 \
    --fov-width 60 --fov-height 40 --overlap 0.5
```

고른 값은 h5 `metadata/camera_spec` 에 박히고, **그 h5 를 읽는 쪽이 config 보다 그 값을 우선**한다
(IK·궤적·GLNS·Isaac 카메라 intrinsic). 물체면이 실제로 움직이므로 h5 를 새로 만든 뒤
도달성·충돌을 다시 확인한다.

**하한**: `CAMERA_MIN_WORKING_DISTANCE_MM` = 77.8 mm (= lens_front − optical_frame).
이보다 작으면 검사면이 렌즈 배럴 안쪽이라 기하학적으로 불가능하다 —
`config.working_distance_error()` 가 CLI(하드 실패) / 스튜디오(에러 표시) /
h5 로드(경고)에서 각각 잡는다. 구 `optical_frame`(0.346) 시절 h5 는 WD 46mm 라 여기 걸린다.

## C. 광학 용어 — "거리"와 구분

| 표준어 | 뜻 | 값 | 코드/USD |
|---|---|---|---|
| **f** (초점거리) | 렌즈 고유 광학상수 | 50 mm | USD `focalLength` |
| **min_focus** | 이보다 가까우면 초점 불가한 최소 물체거리 | base 500 / 매크로변형 ~250 | — |
| **sensor** (센서크기) | 물리 센서칩 크기(mm) | AR0820 = 8.08 × 4.55 | USD `aperture` (입력) |
| **FOV_angle** (화각) | 센서 + f 로 나오는 각도 | 파생 | — |
| **FOV_footprint** | 특정 WD에서 보이는 물체 크기(mm) | 50 × 50 (CAD `VIEW_1`) | `CAMERA_FOV_*_MM` |

## 2대 혼동 주의

1. **f(50mm) ≠ WD(물체거리)** — 초점거리와 작업거리는 완전히 다른 값. 50mm 렌즈여도 물체는 min_focus 밖에 둬야 초점이 맞는다.
2. **FOV(결과) ≠ sensor(입력)** — FOV는 sensor + f + 거리에서 나오는 파생값. 현재 config는 FOV 50×50을 입력으로 넣는 **"footprint 트릭"**([scene.py](../../scripts/core/isaac/scene.py)의 `setup_inspection_camera`, `focalLength=WD, aperture=FOV`)으로 실제 광학이 아니다. 다만 그 50×50은 임의값이 아니라 CAD `VIEW_1` 판과 일치한다.

> ~~3. 코드 WD ≠ 벤더 WD~~ — **2026-07-27 해소.** optical_frame 을 body_face 로 옮겨
> 두 값의 기준점을 통일했다.

## CAD 실측 (값의 출처)

`camera_asm_wo_light.stp` (Creo, AP203, mm) 를 파싱해 광축(어셈블리 루트 −Y)에 투영한 값.
원점은 `ZIVID_END_EFFECTOR_UR20_PT2` 의 로봇쪽 면 = URDF `flange`.

```bash
uv run --no-sync scripts/setup/inspect_camera_step.py   # 아래 표를 재출력 + 3개 랜드마크 assert
```

| flange 기준 [mm] | 부품 |
|---|---|
| 0.0 – 18.0 | `ZIVID_END_EFFECTOR_UR20_PT2` (Ø100 어댑터 플레이트) |
| 63.0 – 138.0 | `PRAIVISION_BRACKET_7` |
| 122.7 – **124.3** | `PCB_2` (t1.6 기판) |
| 103.7 – **141.0** | 커버 → 앞면이 **body_face** |
| 140.0 – 162.0 | `PRAIVISION_LENS_HOLDER` |
| 157.5 – **218.770** | `MFA121-U50_STEP_ASM` (렌즈) |
| 141.0 – **346.0** | `LIGHT_1` — **제거된 조명박스** |
| **391.0** – 392.0 | `VIEW_1` — 50×50 mm 검사면 판 |

검증: 계산된 렌즈 배럴 끝 **218.770 mm** 가
[build_camera_mesh.py](../../scripts/setup/build_camera_mesh.py) `EXPECT_HI` 의 x_max
**0.21877 m** 와 일치 → STEP → flange 프레임 매핑이 확정.

**391.0 − 141.0 = 250.0** 이 벤더 공칭 WD 와 정확히 일치한다.

### 해소된 파킹 항목

- ~~**WD 기준점 78mm**~~ — 벤더 WD 250mm 는 **body_face 기준**임이 CAD로 확정.
  78mm 는 배럴 길이(218.77 − 141.0 = 77.77)였을 뿐 기준점 불일치가 아니었다.
- ~~**optical_frame 이 렌즈보다 127mm 앞 허공**~~ — 그 위치(0.346)는 어셈블리에서
  제거된 조명박스 `LIGHT_1` 의 앞면이었다. 2026-07-27 에 body_face 로 이전.

## 미해결 / 파킹

- **센서 실측** — config `4096×3000 @ 10µm`(=40.96×30mm)는 **placeholder**. AR0820 native는 3848×2168 @ 2.1µm(8.08×4.55mm)이며 렌즈 이미지써클(≤1.2″) 안. 카메라 실제 출력 해상도 확인 필요.
- **실광학 intrinsic 전환** — footprint 트릭 → 실 sensor/f 로 교체 시 `camera_info` K가 실제값이 되지만, FOV가 viewpoint 간격([pipeline.py](../../scripts/core/viewpoint/pipeline.py) `col_spacing`)에도 쓰여 **재생성**이 따른다.
- **pupil 위치** — 렌더 프러스텀 꼭짓점은 원칙적으로 입사동공에 있어야 한다. 현재는 body_face 로 근사. 벤더 광학 데이터 필요.
- **min_focus 확정** — base 500mm vs 매크로변형(`MFA2-230`, ~230~250) 확정 필요.

## 관련 파일

에셋 정의가 어느 파일에 있는지는 [robot-camera-assets.md](robot-camera-assets.md) 참고.

- [workcell/robot/ur20_with_camera_curobo.urdf](../../workcell/robot/ur20_with_camera_curobo.urdf) — `camera_optical_joint` (**mount_offset 의 소유자**)
- [scripts/common/config.py](../../scripts/common/config.py) — `CAMERA_*` 기본값, `TOOL_TO_CAMERA_OPTICAL_OFFSET_M`, `CAMERA_LENS_FRONT_OFFSET_M`, `working_distance_error()`
- [scripts/core/viewpoint/models.py](../../scripts/core/viewpoint/models.py) — `ViewpointGenParams` (실행별 카메라 스펙의 소유자), `ViewpointData` (h5 스냅샷)
- [scripts/core/isaac/scene.py](../../scripts/core/isaac/scene.py) — `setup_inspection_camera` (InspectionCamera intrinsic)
- [scripts/core/trajectory/poses.py](../../scripts/core/trajectory/poses.py) — `build_camera_poses` (WD 적용)
- [scripts/setup/build_camera_mesh.py](../../scripts/setup/build_camera_mesh.py) — USD optical_frame·배럴 형상 굽기
