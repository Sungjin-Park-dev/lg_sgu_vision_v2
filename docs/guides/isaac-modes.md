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
- 앱도 한 번에 하나만. 두 개가 떠 있으면 `/looksgood/mode` 를 서로 다르게 발행해 게이트가 흔들린다(`pgrep -af isaac_pipeline` 로 확인).
- **같은 머신의 다른 ROS 노드와 섞이지 않게** `ROS_DOMAIN_ID` 를 쓴다. 컨테이너는 host 네트워크라 도메인 0 에는 다른 사용자의 스택(다른 `/controller_manager`, `/robot_description`, `/joint_states`)이 그대로 보이고, 그러면 launch 가 남의 매니저와 얘기하거나 `ros2_control_node` 가 남의 URDF 를 받아 죽는다. `docker/compose.yaml` 이 `ROS_DOMAIN_ID=30` 을 컨테이너 환경에 넣으므로 새 컨테이너의 셸은 자동으로 격리된다. 그 전에 만든 컨테이너면 **앱 셸·launch 셸·확인 셸 모두**에서 `export ROS_DOMAIN_ID=30` 을 먼저 친다(Isaac 의 ROS 2 브리지도 이 변수를 읽는다). `ros2 node list` 에 모르는 노드가 보이면 격리가 안 된 것이다.

## Real 또는 mock

Isaac 앱은 `--mode real`로 시작하고 ROS 셸에서 real 스택을 실행한다. 기본은 mock hardware다.
mock 은 sim 과 같은 준비 자세(`config.ROBOT_START_STATE`)에서 시작한다 — launch 가 그 값을
드라이버의 `initial_positions` 로 넣는다. 실로봇에는 영향이 없다(실측 자세가 진실).

```bash
ros2 launch scripts/moveit/ur20_real_moveit.launch.py
```

실제 로봇 옵션은 [로봇 실행 워크플로](../workflows/execute-on-robot.md)의 안전 확인 후 적용한다. Run 모드를 바꾸면 대응하는 ROS 스택도 다시 시작한다.

**`Overrun detected! The controller manager missed its desired rate of 500 Hz` 가 계속 찍히면** —
오류가 아니라 경고다. UR 드라이버는 controller_manager 를 실로봇 RTDE 주기(500 Hz = 2 ms)로
돌리며 FIFO 실시간 우선순위를 요구하는데, 기본 docker 권한에서는 못 얻고(`Could not enable FIFO
RT scheduling policy`) 머신이 바쁘면(Isaac Sim 렌더링, 같은 머신의 다른 스택) 2 ms 를 넘긴다.
mock 에서는 결과에 영향이 없고, launch 가 mock 일 때 125 Hz 로 내리므로 거의 안 보인다. 실로봇은
500 Hz 를 유지해야 하므로 컨테이너에 실시간 권한을 준다 — `docker/compose.yaml` 의 `cap_add:
SYS_NICE` / `ulimits: rtprio: 99` (그 줄이 들어가기 전에 만든 컨테이너는 다시 만들어야 적용된다).
`Maybe failed to update robot state, time diff: 0.003s` 는 RViz 의 무해한 잡음이다.

**로봇이 0 자세로 쭉 뻗은 채 RViz·Isaac 이 안 따라오면** — UR 드라이버 버전 문제다. NVIDIA
isaac-ros 저장소가 `ros-jazzy-ur-*` 를 `99.2.0` 으로 내보내고, 핀이 없으면 apt 가 ros.org 의
`3.8.0` 대신 그걸 고른다. 그 빌드는 Jazzy 의 `joint_trajectory_controller`/`realtime_tools` 와
맞지 않아 `io_and_status_controller` 가 첫 update 에서 segfault 를 내고 `ros2_control_node` 가
통째로 죽는다(`scaled_joint_trajectory_controller` 도 configure 에서 `map::at` 로 실패). mock 은
준비 자세로 뜨지만 몇 ms 뒤 `/joint_states` 가 끊겨 화면은 0 자세로 남는다. 실로봇도 같은
이유로 MoveIt 모드가 뜨지 못한다. real launch 는 이 경우 처방을 찍고 멈춘다. 처방
(`docker/Dockerfile` 2b 절과 같다):

```bash
docker exec ros-jazzy bash -c "printf '%s\n' 'Package: ros-jazzy-ur-*' 'Pin: origin packages.ros.org' 'Pin-Priority: 1001' > /etc/apt/preferences.d/ros-org-ur.pref \
  && apt-get update && apt-get install -y --allow-downgrades ros-jazzy-ur-robot-driver ros-jazzy-ur-controllers"
docker exec ros-jazzy bash /workspace/docker/verify_env.sh   # "ur_robot_driver is 3.8.x" 가 OK 여야 한다
```
