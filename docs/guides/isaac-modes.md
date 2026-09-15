# Isaac 실행 모드

Isaac Pipeline은 로봇 대상과 명령 소스를 각각 선택한다.

|  | `sim` | `real` |
|---|---|---|
| `inspection` | 앱의 CSV를 Isaac UR20에서 실행 | 앱의 CSV를 robot controller로 전송하고 Isaac은 상태를 미러링 |
| `moveit` | RViz 명령으로 Isaac UR20 구동 | RViz 명령으로 mock/실로봇을 구동하고 Isaac은 상태를 미러링 |

`--mode`와 `--pipeline-mode`는 시작값이며 UI에서 바꿀 수 있다. 실행 중에는 모드를 전환하지 않는다.

## 모드는 어떻게 집행되나

네 조합 모두 같은 두 동작이다. 앱은 컨트롤러나 ROS 파라미터를 직접 건드리지 않는다.

| 로봇 | 게이트 | 규칙 |
|---|---|---|
| Isaac | `/SimRobotGraph`의 관절 쓰는 노드 | 구독은 항상, **쓰기는 `sim/moveit`일 때만** |
| 실로봇 | 스택의 `mode_gate.py` | `/looksgood/mode`를 받아 활성 컨트롤러를 유지 (`moveit`→scaled, `inspection`→jtc) |

앱은 `/looksgood/mode`에 `"<run>/<pipeline>"`을 2 Hz로 **계속** 발행한다(`mode_publisher.py`).
스택이 늦게 뜨든 재시작되든 다음 주기에 맞는다. 발행이 **2초** 끊기면(앱 종료) 게이트는
궤적 컨트롤러를 전부 내린다 — 앱이 말하기 전엔 누구도 로봇을 못 움직인다. 두 launch 모두
컨트롤러를 비활성으로 올리는 이유다.

앱 없이 MoveIt만 쓰려면 게이트에 기본 모드를 준다:

```bash
ros2 launch scripts/moveit/ur20_isaac_state_synced.launch.py default_mode:=sim/moveit
ros2 launch scripts/moveit/ur20_real_moveit.launch.py       default_mode:=real/moveit
```

`/looksgood/gate_state`를 echo하면 게이트가 지금 무엇을 보고 무엇을 켰는지 나온다.

## Inspection × sim

가장 안전한 기본 모드다. ROS 스택 없이 궤적 생성, preview와 Isaac 실행이 가능하다.

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync \
  scripts/apps/isaac_pipeline.py --object sample \
  --mode sim --pipeline-mode inspection
```

## MoveIt × sim

Isaac이 로봇이고, RViz(cuMotion)로 계획·실행한다. 셸 두 개를 쓴다(`docker exec -it ros-jazzy bash`로 각각 접속).

**셸1 — Isaac 앱** (venv, 시스템 ROS는 source하지 않는다). 앱이 뜨면 뷰포트에서 **▶ Play**를 눌러 `/isaac_joint_states`가 흐르게 한다.

```bash
source /workspace/.venv/bin/activate
OMNI_KIT_ACCEPT_EULA=YES python scripts/apps/isaac_pipeline.py \
  --object sample --mode sim --pipeline-mode moveit
```

> MoveIt은 ROS2 브리지(`/isaac_joint_states`)가 필요하다. `uv run` 대신 `.venv`를 activate해 브리지용 `LD_LIBRARY_PATH`가 적용되게 한다.

**셸2 — SIM 스택** (셸1을 Play한 뒤 실행). `topic_based` 브리지 때문에 `ros2_overlay`까지 source한다.

```bash
source /opt/ros/jazzy/setup.bash
source /workspace/ros2_overlay/install/setup.bash
ros2 launch scripts/moveit/ur20_isaac_state_synced.launch.py
```

컨트롤러가 활성화되고 move_group·cuMotion·RViz가 뜨면 `You can start planning now!`가 출력된다.

**사용**: RViz에서 목표 자세 지정 → **Plan & Execute** → Isaac UR20이 그 경로로 움직인다. 평소엔 MoveIt이 Isaac 현재 상태를 반영하고, Execute할 때만 로봇이 움직인다.

**주의**
- 셸1을 먼저 Play한 뒤 셸2를 실행한다. 순서가 바뀌면 `/isaac_joint_states`가 없어 컨트롤러가 gate에서 타임아웃된다(`Switch controller timed out`).
- 셸2 스택은 한 번에 하나만 띄운다 — sim/real을 동시에 띄우면 `/robot_description` 충돌로 `ros2_control_node`가 segfault한다.

## Real 또는 mock

Isaac 앱은 `--mode real`로 시작하고 ROS 셸에서 real 스택을 실행한다. 기본은 mock hardware다.
mock 은 sim 과 같은 준비 자세(`config.ROBOT_START_STATE`)에서 시작한다 — launch 가 그 값을
드라이버의 `initial_positions` 로 넣는다. 실로봇에는 영향이 없다(실측 자세가 진실).

```bash
ros2 launch scripts/moveit/ur20_real_moveit.launch.py
```

실제 로봇 옵션은 [로봇 실행 워크플로](../workflows/execute-on-robot.md)의 안전 확인 후 적용한다. Run 모드를 바꾸면 대응하는 ROS 스택도 다시 시작한다.
