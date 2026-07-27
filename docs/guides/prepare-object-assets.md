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

메시를 바꾸거나 크게 회전했다면 뷰포인트를 다시 생성한다.
