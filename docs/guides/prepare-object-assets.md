# 새 물체 자산 준비

앱의 Object 목록에 표시되려면 `data/{object}/mesh/source.obj`가 필요하다. Isaac Pipeline에서 불러오려면 `source.usd`도 준비한다.

## 1. OBJ 정규화

```bash
uv run scripts/setup/prepare_object_mesh.py normalize \
  --object my_object --input /path/to/raw.obj
```

기본적으로 mm를 m로 변환하고 바닥 중심을 원점에 맞춘다. 결과를 쓰지 않고 확인하려면 `--dry-run`을 사용한다.

## 2. 방향 보정

```bash
uv run scripts/setup/prepare_object_mesh.py reorient \
  --object my_object --euler 90 0 0
```

Euler 대신 `--quat W X Y Z` 또는 `--world-target-quat W X Y Z`를 사용할 수 있다. 같은 파일을 덮어쓸 때는 기본적으로 백업을 만든다.

## 3. USD 생성

```bash
uv run scripts/setup/build_object_usd.py --object my_object --force
```

카메라 CAD 또는 preview ghost를 교체할 때만 다음 도구를 사용한다.

```bash
uv run scripts/setup/build_camera_mesh.py --source /path/to/camera.obj --dry-run
uv run scripts/setup/build_ghost_usd.py
```

## 4. 예외 설정 등록

`scripts/common/config.py`의 세 표는 **물체 고유 속성**이다. 해당하는 물체만 등록하고, 대부분의 물체는 아무 데도 안 넣어도 된다.

| 표 | 언제 등록하나 | 등록 안 하면 |
|---|---|---|
| `OBJECT_TARGET_MATERIAL` | 메시에 **검사 대상이 아닌 면**이 섞여 있을 때. 초록 `"0,255,0"` = 대상 규약 | 뷰포인트가 전체 메시에 생긴다 — `sample`은 74개 대신 161개 |
| `OBJECT_FILTER_INTERIOR` | 물체가 **속이 비어** 안쪽 면에 뷰포인트가 생길 때 | 공동 안쪽에 뷰포인트가 생겨 도달할 수 없다 |
| `OBJECT_COLLISION_SHAPE` | 메시 최소 bbox가 **5cm 이하**일 때 | cuRobo가 모든 자세를 충돌로 오판한다 — "No reachable viewpoints" |

```python
OBJECT_TARGET_MATERIAL = {"sample": "0,255,0"}
OBJECT_FILTER_INTERIOR = {"square_structure": {"hull_align_min": 0.3}}
OBJECT_COLLISION_SHAPE = {"cylinder_sample": "box"}
```

⚠️ **첫 번째가 가장 위험하다.** 에러 없이 뷰포인트 개수만 조용히 틀린다. 대상 면을 초록으로 통일해뒀다면 반드시 등록한다.

## 5. 셀에 배치

물체를 놓을 자리는 `workcell/scenes/{name}.yaml`의 `object_placements`에 쓴다. 좌표는 **robot base_link 프레임**이고, `position`은 물체 **바닥면 중심**이다.

```yaml
object_placements:
  my_object:
    position: [-1.1228, -0.2489, 0.0018]
    rotation: [1.0, 0.0, 0.0, 0.0]        # z-yaw 만 쓴다
```

등록하지 않으면 씬의 `target_object` 기본값에 놓인다. 배치는 셀이 바뀌면 같이 바뀌므로 씬 YAML에 두고, 4단계의 물체 속성은 셀과 무관하므로 `config.py`에 둔다 — 기준은 [주요 설정값](../reference/configuration.md#값을-어디에-둘-것인가)에 있다.

아루코로 실측한 값을 쓸 때는 프레임 변환에 주의한다. 자세한 것은 [현장 시운전](../workflows/field-commissioning.md#아루코--씬-yaml-변환).

## 확인

Isaac Pipeline이나 Trajectory Studio의 Object 목록에 뜨면 자산은 준비된 것이다. 그다음 [뷰포인트 만들기](../workflows/create-viewpoints.md)로 넘어간다.

메시를 바꾸거나 크게 회전했다면 뷰포인트를 다시 생성한다.
