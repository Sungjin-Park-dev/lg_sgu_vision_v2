# MoveIt / Inspection 최상위 모드

`scripts/apps/isaac_pipeline.py` 상단에 **Pipeline Mode** 선택이 추가됐다.
두 모드는 같은 로봇(`/World/UR20`)을 구동하므로 **상호 배제**된다.

| 모드 | 동작 |
|---|---|
| **Inspection** (기본) | 기존 전체 워크플로(sim/real + Load/Generate/Preview/Publish). MoveIt 명령(`/isaac_joint_commands`) 수신 **차단**. |
| **MoveIt** | Inspection UI 전체 **비활성화(회색)**. Isaac이 `/isaac_joint_commands` 를 구독해 로봇 구동 + `/isaac_joint_states` 발행. |

전환은 즉시(액션그래프 tick 토글)이며, 두 그래프 중 **하나만** 틱한다:
- `/ActionGraph` — Inspection용 `/joint_states` 구독 + 카메라 발행 (sim/real 토글로 게이팅)
- `/MoveItGraph` — MoveIt용 `/isaac_joint_commands` 구독 + `/isaac_joint_states` 발행

> MoveIt 모드에서는 `/ActionGraph`가 꺼지므로 **검사 카메라 발행도 함께 멈춘다**(모션 전용).
> 카메라를 MoveIt 모드에서도 유지하려면 카메라 노드를 별도 상시 그래프로 분리해야 한다(후속 작업).

## 토픽 규약 (isaac_ros-dev 와 일치)

`config.py`:
```python
MOVEIT_JOINT_COMMANDS_TOPIC = "/isaac_joint_commands"  # ROS→Isaac
MOVEIT_JOINT_STATES_TOPIC   = "/isaac_joint_states"    # Isaac→ROS
```
참고: `isaac_ros-dev/.../ur_config/ur.ros2_control.xacro` 의 TopicBasedSystem 파라미터와 동일.

## 실행 (엔드투엔드)

cuMotion 4.4가 jazzy 네이티브로 설치되어 **모두 같은 `ros-jazzy` 컨테이너**에서 돈다.
MoveIt/cuMotion 실행과 state-synced launch는 [moveit_cumotion_ur20.md](moveit_cumotion_ur20.md) 참조.

```bash
# 셸 1 — Isaac 앱, MoveIt 모드 (venv)
docker exec -it ros-jazzy bash -c \
  'source /workspace/.venv/bin/activate && cd /workspace && \
   OMNI_KIT_ACCEPT_EULA=YES python scripts/apps/isaac_pipeline.py \
     --object sample --pipeline-mode moveit'
#   (UI Pipeline Mode 드롭다운으로 런타임 전환 가능)

# 셸 2 — MoveIt + cuMotion, state-synced (시스템 ROS)
docker exec -it ros-jazzy bash -c \
  'source /opt/ros/jazzy/setup.bash && cd /workspace && \
   ros2 launch scripts/moveit/ur20_isaac_state_synced.launch.py'
#   → RViz에서 Plan & Execute
```

## 검증 체크리스트

1. **빌드** (자동, 완료) — headless에서 `/MoveItGraph` 6개 노드 생성, SUB=`/isaac_joint_commands`,
   PUB=`/isaac_joint_states` 확인됨.
2. **UI 잠금** — Pipeline Mode→MoveIt 시 Inspection 패널 전체 회색/클릭 불가.
3. **구동** — RViz Plan&Execute → Isaac 로봇이 따라 움직임.
   - `ros2 topic hz /isaac_joint_states` (Isaac 발행)
   - `ros2 topic echo /isaac_joint_commands` (MoveIt 명령)
4. **차단** — Pipeline Mode→Inspection 후 다시 Execute → Isaac **안 움직임**(명령 게이팅).
5. **로그** — Log 패널에 `[pipeline] → MOVEIT/INSPECTION`, `[graph] disable mode = node|evaluator`.
   `noop`이면 게이팅 실패이므로 그래프 확인.

## 알려진 리스크

- cuMotion은 nvblox(ESDF) 없이 동작(`read_esdf_world=False`) — 정적 planning scene +
  xrdf self-collision 기반. 자세한 한계는 [moveit_cumotion_ur20.md](moveit_cumotion_ur20.md).
- RViz Plan&Execute → Isaac 실제 구동 및 RTX 5080 cuMotion GPU 런타임은 대화형 검증 필요.
