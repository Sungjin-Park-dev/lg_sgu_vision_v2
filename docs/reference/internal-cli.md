# 내부 CLI

일반 사용자는 세 앱을 실행하면 된다. 아래 CLI는 자동화, 디버깅과 개발 검증용이다.

| 역할 | 경로 |
|---|---|
| viewpoint 배치 생성 | `scripts/core/viewpoint/cli.py` |
| DP 궤적 생성 | `scripts/core/trajectory/cli.py` |
| viewpoint IK 검사 | `scripts/core/trajectory/check_ik.py` |
| ROS2/Isaac 궤적 전송 | `scripts/core/trajectory/publish.py` |
| GLNS solve | `scripts/core/glns/solve.py` |
| GLNS motion 검증·연결 | `scripts/core/glns/verify.py` |

옵션의 현재 기본값은 각 명령의 `--help`를 기준으로 한다.

```bash
uv run --no-sync scripts/core/viewpoint/cli.py --help
uv run --no-sync scripts/core/trajectory/cli.py --help
uv run --no-sync scripts/core/glns/solve.py --help
uv run --no-sync scripts/core/glns/verify.py --help
```

직접 호출할 때도 입력·출력 형식은 앱과 동일하다. 호환용 이전 `scripts/core/*.py` 경로는 제공하지 않는다.
