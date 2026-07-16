# 카메라 기하·용어 표준 (단일 진실원)

카메라 위치·거리·광학 용어의 **표준 정의**. 코드 주석·다른 문서·논의는 모두 이 용어를 따른다.
혼란의 근본 원인은 "거리"를 서로 다른 기준점에서 재고 같은 이름으로 부른 것이었다 —
그래서 모든 거리는 **"어디서 → 어디까지"를 이름에 못박는다.**

값은 `ur20_with_camera` 에셋과 벤더 메일(2026-07) 기준이며, 미확정 항목은 §미해결에 명시한다.

## A. 기준점 (광축 위, flange 프레임 기준 미터)

```
flange ──── body_face ──── lens_front ──[pupil]── optical_frame ──── object_plane
 0.000       0.141          0.219        (렌즈안)     0.346             0.392   [m]
```

| 표준어 | 뜻 | 위치 | 비고 |
|---|---|---|---|
| **flange** | EE 연결면 | 0.000 | ROS-Industrial `flange` 링크 |
| **body_face** | 카메라 몸체 앞면 | 0.141 | CAD 몸체 끝 ([2026-07-09](../logs/2026-07-09.md)) |
| **lens_front** | 렌즈 앞면(배럴 끝) | 0.219 | 실제 MFA121-U50 배럴 (Ø38.4≈Φ38.6mm) |
| **pupil** | 광학중심(입사동공) | 렌즈 내부(미확정) | 핀홀 투영중심. 현재 lens_front 로 근사 |
| **optical_frame** | `camera_optical_frame` | 0.346 | ⚠️ 코드 tool frame. 렌즈보다 127mm 앞 허공(낡음) |
| **object_plane** | 물체면 = 검사면 = 초점면 | 0.392 | 검사 표면 위치 |

## B. 거리 — 이름에 기준점을 못박음

| 표준어 | 정의 (from → to) | 값 | 코드 심볼 |
|---|---|---|---|
| **mount_offset** | flange → optical_frame | 0.346 m | `TOOL_TO_CAMERA_OPTICAL_OFFSET_M` |
| **frame_standoff** | optical_frame → object_plane | 46 mm | `CAMERA_WORKING_DISTANCE_MM` ⚠️ |
| **WD** (작업거리) | **lens_front → object_plane** | 현 기하 173mm / 벤더공칭 250mm | (코드에 없음) |
| **body_WD** | body_face → object_plane | 251 mm ≈ 벤더 250 | (코드에 없음) |
| **flange_to_object** | flange → object_plane | 392 mm | (CAD 실측 391) |

> ⚠️ `CAMERA_WORKING_DISTANCE_MM`(46mm)은 이름이 "working distance"지만 **광학 WD가 아니다.**
> optical_frame 기준 standoff다. 벤더가 말하는 WD(250mm)와는 **기준점이 다른 별개 값**이며,
> 둘은 **같은 물체 배치를 다르게 잰 것**이다(46@optical_frame = 251@body_face = object_plane 0.392).
> 이 변수를 250 으로 바꾸면 [poses.py](../../scripts/core/trajectory/poses.py) 가 viewpoint 를
> 250mm 띄워 **기하가 통째로 이동 → 전체 재생성**이 된다. 바꾸지 말 것.

## C. 광학 용어 — "거리"와 구분

| 표준어 | 뜻 | 값 | 코드/USD |
|---|---|---|---|
| **f** (초점거리) | 렌즈 고유 광학상수 | 50 mm | USD `focalLength` |
| **min_focus** | 이보다 가까우면 초점 불가한 최소 물체거리 | base 500 / 매크로변형 ~250 | — |
| **sensor** (센서크기) | 물리 센서칩 크기(mm) | AR0820 = 8.08 × 4.55 | USD `aperture` (입력) |
| **FOV_angle** (화각) | 센서 + f 로 나오는 각도 | 파생 | — |
| **FOV_footprint** | 특정 WD에서 보이는 물체 크기(mm) | 파생, 거리 비례 | — |

## 3대 혼동 주의

1. **f(50mm) ≠ WD(물체거리)** — 초점거리와 작업거리는 완전히 다른 값. 50mm 렌즈여도 물체는 min_focus 밖에 둬야 초점이 맞는다.
2. **코드 WD(46, optical_frame) ≠ 벤더 WD(250, body/lens)** — 같은 배치를 다른 기준점에서 잰 것(§B ⚠️).
3. **FOV(결과) ≠ sensor(입력)** — FOV는 sensor + f + 거리에서 나오는 파생값. 현재 config는 FOV 50×50을 입력으로 넣는 **"footprint 트릭"**([scene.py](../../scripts/core/isaac/scene.py)의 `setup_inspection_camera`, `focalLength=frame_standoff, aperture=FOV`)으로 실제 광학이 아니다.

## 미해결 / 파킹

- **WD 기준점 78mm** — 벤더 250mm가 lens_front 기준이면 object_plane은 0.469m여야 하고 현 0.392는 78mm(=배럴 길이) 너무 가깝다. body_face 기준이면 현 배치가 맞다. → 벤더 확인 대기, 그동안 CAD(0.392) 유지.
- **센서 실측** — config `4096×3000 @ 10µm`(=40.96×30mm)는 **placeholder**. AR0820 native는 3848×2168 @ 2.1µm(8.08×4.55mm)이며 렌즈 이미지써클(≤1.2″) 안. 카메라 실제 출력 해상도 확인 필요.
- **실광학 intrinsic 전환** — footprint 트릭 → 실 sensor/f 로 교체 시 `camera_info` K가 실제값이 되지만, FOV가 viewpoint 간격([pipeline.py](../../scripts/core/viewpoint/pipeline.py) `col_spacing`)에도 쓰여 **재생성**이 따른다.
- **min_focus 확정** — base 500mm vs 매크로변형(`MFA2-230`, ~230~250) 확정 필요.

## 관련 파일

- [scripts/common/config.py](../../scripts/common/config.py) — `CAMERA_*`, `TOOL_TO_CAMERA_OPTICAL_OFFSET_M`
- [scripts/core/isaac/scene.py](../../scripts/core/isaac/scene.py) — `setup_inspection_camera` (InspectionCamera intrinsic)
- [scripts/core/trajectory/poses.py](../../scripts/core/trajectory/poses.py) — `build_camera_poses` (frame_standoff 적용)
- [scripts/setup/build_camera_mesh.py](../../scripts/setup/build_camera_mesh.py) — USD optical_frame·배럴 형상 굽기
