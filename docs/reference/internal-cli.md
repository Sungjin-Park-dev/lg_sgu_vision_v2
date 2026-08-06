# 내부 CLI

일반 사용자는 세 앱을 실행하면 된다. 아래 CLI는 자동화, 디버깅과 개발 검증용이다.

| 역할 | 경로 |
|---|---|
| viewpoint 배치 생성 | `scripts/core/viewpoint/cli.py` |
| viewpoint IK 검사 | `scripts/core/trajectory/check_ik.py` |
| 두 자세 사이 충돌-free 이동 계획 | `scripts/core/trajectory/plan_move.py` |
| ROS2/Isaac 궤적 전송 | `scripts/core/trajectory/publish.py` |
| GLNS solve (1단계) | `scripts/core/glns/solve.py` |
| GLNS motion 계획·연결 (2단계) | `scripts/core/glns/verify.py` |

옵션의 현재 기본값은 각 명령의 `--help`를 기준으로 한다.

```bash
uv run --no-sync scripts/core/viewpoint/cli.py --help
uv run --no-sync scripts/core/glns/solve.py --help
uv run --no-sync scripts/core/glns/verify.py --help
```

직접 호출할 때도 입력·출력 형식은 앱과 동일하다. 호환용 이전 `scripts/core/*.py` 경로는 제공하지 않는다.

## `--scene` (공통)

셀 기하를 고르는 플래그. 위 CLI 전부와 세 앱, `setup/prepare_object_mesh.py reorient` 가 받는다.
값은 씬 이름(`workcell/scenes/{이름}.yaml`) 또는 YAML 경로이고, 기본값은 `sim_default`.
스키마는 [configuration.md](configuration.md#씬-장애물물체-배치) 참고.

```bash
uv run --no-sync scripts/core/viewpoint/cli.py --object sample --scene real_cell ...
uv run --no-sync scripts/core/glns/solve.py    --object sample --scene real_cell ...
```

물체 배치(`object_placements`)가 씬 소유라, `--scene` 은 항상 물체 배치 적용보다 먼저 반영된다.
`--object-position/--object-quat` override 는 그 뒤에 오므로 그대로 우선한다.

**`verify.py` 는 예외다.** 기본값이 `--scene` 이 아니라 `solution.h5` 에 박제된 **씬 스냅샷**이다
— solve 가 푼 바로 그 셀을 재현해야 검증이 성립하고, 실측 셀을 맞추는 동안 YAML 은 계속 편집되기
때문이다. `--scene` 을 명시하면 override 하되 "이 해는 그 씬으로 풀린 게 아니다" 경고를 찍는다.
