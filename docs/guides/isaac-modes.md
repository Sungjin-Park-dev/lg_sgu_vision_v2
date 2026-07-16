# Isaac 실행 모드

Isaac Pipeline은 로봇 대상과 명령 소스를 각각 선택한다.

|  | `sim` | `real` |
|---|---|---|
| `inspection` | 앱의 CSV를 Isaac UR20에서 실행 | 앱의 CSV를 robot controller로 전송하고 Isaac은 상태를 미러링 |
| `moveit` | RViz 명령으로 Isaac UR20 구동 | RViz 명령으로 mock/실로봇을 구동하고 Isaac은 상태를 미러링 |

`--mode`와 `--pipeline-mode`는 시작값이며 UI에서 바꿀 수 있다. 실행 중에는 모드를 전환하지 않는다.

## Inspection × sim

가장 안전한 기본 모드다. ROS 스택 없이 궤적 생성, preview와 Isaac 실행이 가능하다.

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync \
  scripts/apps/isaac_pipeline.py --object sample \
  --mode sim --pipeline-mode inspection
```

## MoveIt × sim

Isaac 앱을 먼저 Play한 뒤 별도 ROS 셸에서 MoveIt을 실행한다.

```bash
ros2 launch scripts/moveit/ur20_isaac_state_synced.launch.py
```

이 구성에는 `/workspace/ros2_overlay/install/setup.bash`가 source되어 있어야 한다.

## Real 또는 mock

Isaac 앱은 `--mode real`로 시작하고 ROS 셸에서 real 스택을 실행한다. 기본은 mock hardware다.

```bash
ros2 launch scripts/moveit/ur20_real_moveit.launch.py
```

실제 로봇 옵션은 [로봇 실행 워크플로](../workflows/execute-on-robot.md)의 안전 확인 후 적용한다. Run 모드를 바꾸면 대응하는 ROS 스택도 다시 시작한다.
