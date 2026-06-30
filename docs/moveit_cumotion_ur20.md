# UR20 + cuMotion(MoveIt) ↔ Isaac Sim — 한 컨테이너(jazzy)

`ros-jazzy` 컨테이너 한 곳에서 **MoveIt2 + cuMotion(GPU 플래너)** 으로 UR20을 플래닝·구동한다.
(별도 humble 컨테이너 불필요 — cuMotion 4.4가 jazzy 네이티브.)

## 2×2 모드 (Run × Pipeline)

| | **sim** (Isaac이 로봇) | **real** (실로봇, Isaac은 트윈) |
|---|---|---|
| **MoveIt** | RViz Execute → Isaac | RViz Execute → 실로봇, Isaac이 미러 |
| **Inspection** | Publish → Isaac 검사경로 | Publish → 실로봇 검사경로, Isaac이 미러 |

- **Run 모드(sim/real)** = 로봇 주체. Isaac UI의 Run mode 콤보가 Isaac 그래프를 결정:
  sim→`/MoveItGraph`(Isaac=하드웨어), real→`/ActionGraph` 미러(실로봇 `/joint_states` 추종=트윈).
  sim↔real 전환 시 **셸2 ROS 스택도 맞는 것으로 재실행**해야 함(ros2_control 하드웨어가 launch 고정).
- **Pipeline 모드(MoveIt/Inspection)** = 명령 소스 + UI 잠금(MoveIt 모드=Inspection 패널 회색).
  MoveIt→`scaled_joint_trajectory_controller`, Inspection→`joint_trajectory_controller`로
  **분리**해 모드별로 한쪽만 활성화(상호 차단). 최종 동작은 아래 **[최근 변경](#최근-변경)** 참조.
- 불변 원칙: 모드 전환/Stop·Play 시 MoveIt이 로봇 상태를 반영(로봇이 stale MoveIt을 안 따라감).

> ⚠️ 이 문서의 일부 초기 설명(특히 "State-synced 시작 동작"의 idle-latch·resume-cancel relay)은
> 이후 설계로 **대체**되었다. 현재 동작의 정본은 맨 아래 **[최근 변경](#최근-변경)** 절이다.

### 실행 (셸 구성)
```bash
# 공통 — 셸1: Isaac 앱 (Run mode를 UI에서 sim/real 선택; --mode 로 부팅 기본값 지정)
docker exec -it ros-jazzy bash -c \
  'source /workspace/.venv/bin/activate && cd /workspace && OMNI_KIT_ACCEPT_EULA=YES \
   python scripts/apps/isaac_pipeline.py --object sample --mode sim --pipeline-mode moveit'

# SIM 스택 — 셸2 (Run=sim일 때)
docker exec -it ros-jazzy bash -c \
  'source /opt/ros/jazzy/setup.bash && source /workspace/ros2_overlay/install/setup.bash && \
   cd /workspace && ros2 launch scripts/moveit/ur20_isaac_state_synced.launch.py'

# REAL 스택 — 셸2 (Run=real일 때; mock_hardware 기본)
docker exec -it ros-jazzy bash -c \
  'source /opt/ros/jazzy/setup.bash && cd /workspace && \
   ros2 launch scripts/moveit/ur20_real_moveit.launch.py use_mock_hardware:=true'
```
- sim↔real 전환: 셸1에서 Run mode 콤보 변경 + 셸2를 해당 스택으로 재실행.
- real은 ur_robot_driver를 launch가 포함하므로 별도 driver 셸 불필요(mock 기본; 실 UR은
  `use_mock_hardware:=false robot_ip:=<ip>`).

## 구성 요소

apt(jazzy)로 설치된 것:
- `ros-jazzy-isaac-ros-cumotion{,-moveit,-examples,-robot-description}` 4.4.0
- `ros-jazzy-ur-moveit-config`, `ros-jazzy-ur-description`
- **`ros-jazzy-topic-based-ros2-control`** ← Isaac↔ros2_control 브리지 하드웨어 플러그인
  (`topic_based_ros2_control/TopicBasedSystem`). cuMotion 설치에 자동으로 안 딸려오므로
  **별도 설치 필수**(없으면 `controller_manager`가 "TopicBasedSystem does not exist"로 죽음).
- cuMotion 4.4 구조: **C++ MoveIt 플래닝 플러그인**(`isaac_ros_cumotion_moveit/CumotionPlanner`)이
  move_group 안에서, **cuMotion 액션 서버**(composable: `CumotionPlanner` +
  `StaticPlanningSceneServer`, `isaac_ros_cumotion.launch.py`)로 요청을 전달. 별도 실행파일
  `cumotion_planner_node`는 없음(예전 방식).

프로젝트 repo(`scripts/moveit/`, git 관리 — 컨테이너 재생성에도 영속):
- `ur20_isaac_state_synced.launch.py` — **state-synced** 통합 launch (아래)
- `ur20.xrdf` — UR20용 cuMotion 로봇 기술(표준 tool0). apt robot-description엔 ur20이
  없어 공급. (apt `ur10e.xrdf`와 동일 포맷, spheres inline self-contained.)
- `ur_config/ur_gated.urdf.xacro`, `ur_config/ur_gated.ros2_control.xacro` — apt 4.4
  ur.urdf/ros2_control 기반, 명령 토픽만 `/isaac_joint_commands` → **`/isaac_joint_commands_raw`**
- `ur_config/ros2_controllers.yaml` — apt 4.4 컨트롤러 설정
- `isaac_joint_command_relay.py`, `wait_for_joint_state_gate.py` — state-sync 핵심(아래)

## State-synced 시작 동작 (핵심 요구사항)

기본 튜토리얼은 Isaac 재생 시 로봇이 MoveIt 기본자세로 **튀는** 문제가 있다. 이 launch은
그걸 막는다:

1. **gate** (`wait_for_joint_state_gate.py`): Isaac이 `/isaac_joint_states`를 발행하기 전엔
   컨트롤러·move_group을 띄우지 않는다. → move_group/RViz가 **Isaac의 실제 자세에서 시드**.
2. **gated ros2_control**: TopicBasedSystem이 명령을 `/isaac_joint_commands`로 직접 쏘지 않고
   **`/isaac_joint_commands_raw`** 로 보낸다.
3. **relay** (`isaac_joint_command_relay.py`): raw→`/isaac_joint_commands`를 **실제 trajectory
   실행 중 + Isaac 상태가 신선할 때만** 전달. → 평소/시작 시엔 안 튀고, **Plan&Execute 할 때만**
   Isaac 로봇이 움직인다. (relay가 보는 status 토픽 = `scaled_joint_trajectory_controller`.)
4. **idle hold (latch)**: 실행 중이 아닐 때 relay는 **idle 진입 시점에 캡처한 고정 자세**를
   명령으로 유지 → 로봇이 자기 자리에 안정적으로 고정(실시간 되먹임은 피드백 진동·중력 처짐을
   유발하므로 안 씀). MoveIt은 live /joint_states로 current state를 잡아 시뮬에 동기화.
5. **전환 시 보류 goal 취소**: Inspection 모드에선 `/clock`도 멈춰 trajectory 실행이 **얼어붙은
   채 대기**한다(컨트롤러 비활성화는 하드웨어가 stale이라 controller_manager가 거부함 →
   사용 불가). 그래서 relay가 **MoveIt 모드 진입(resume) 시 컨트롤러 액션에 cancel-all**을
   보내고(`/{controller}/follow_joint_trajectory/_action/cancel_goal`), cooldown 동안 forwarding을
   막아 취소가 먼저 적용되게 한다. → 얼었던 궤적이 전환 순간 재생되며 튀는 문제 해결.
   (CANCELING 상태 goal은 forwarding 대상에서 제외해 마진 확보.)

결과: **재생 순간 MoveIt이 Isaac 로봇 현재 상태로 동기화 → 그 뒤 MoveIt으로 제어.**

## 실행 (두 조각, 같은 컨테이너 / 같은 DDS)

```bash
# 셸 1 — Isaac 앱 (MoveIt 모드). venv, 시스템 ROS source 안 함
docker exec -it ros-jazzy bash -c \
  'source /workspace/.venv/bin/activate && cd /workspace && \
   OMNI_KIT_ACCEPT_EULA=YES python scripts/apps/isaac_pipeline.py \
     --object sample --pipeline-mode moveit'
#   → Isaac이 /isaac_joint_commands 구독, /isaac_joint_states 발행 (build_moveit_graph)

# 셸 2 — MoveIt + cuMotion. 시스템 ROS + topic_based overlay source
docker exec -it ros-jazzy bash -c \
  'source /opt/ros/jazzy/setup.bash && \
   source /workspace/ros2_overlay/install/setup.bash && \
   cd /workspace && \
   ros2 launch scripts/moveit/ur20_isaac_state_synced.launch.py'
#   → RViz에서 목표 설정 → Plan & Execute → Isaac UR20 구동
```

> **overlay 필수**: `topic_based_ros2_control`(Isaac↔ros2_control 브리지)는 apt(isaac-ros
> 99.99.1)판이 컨테이너의 ros2_control 4.45.2와 **ABI 불일치로 segfault**. 그래서 소스로
> 빌드해 `/workspace/ros2_overlay`에 둠(영속). 위처럼 overlay를 **반드시 source**해야
> ABI 일치 빌드본이 쓰임. (재빌드:
> `curl -fsSL https://codeload.github.com/PickNikRobotics/topic_based_ros2_control/tar.gz/refs/heads/main`
> → `colcon build --packages-select topic_based_ros2_control --cmake-args -DBUILD_TESTING=OFF`)

> **/clock 필수**: cuMotion/move_group/controller_manager가 `use_sim_time:=True`라 Isaac이
> `/clock`을 발행해야 컨트롤러가 활성화된다. `scene.py`의 `build_moveit_graph()`가
> `ROS2PublishClock`로 발행하므로, Isaac 앱을 **이 수정 이후 버전으로 재시작**해야 함.

> `ur_type`/`xrdf_path`는 인자로 override 가능(기본 ur20 + scripts/moveit/ur20.xrdf).

## 검증 상태
- ✅ launch 구성/문법, gated urdf의 ur20 xacro 처리, MoveItConfigsBuilder(ur20),
  ur20.xrdf 파싱, relay 기동(올바른 status 토픽) — 비대화형 검증 완료.
- ⏳ RViz Plan&Execute → Isaac 실제 구동, state-sync 시각 확인, RTX 5080(Blackwell)
  cuMotion GPU 런타임 — 대화형 검증 필요(사용자).

## 주의/한계
- **nvblox(ESDF) 없음** → `read_esdf_world=False`로 둠(동적 복셀 충돌월드 미사용; 정적
  planning scene + xrdf self-collision 기반). nvblox 쓰려면 별도 구성.
- ur20.xrdf는 **표준 tool0** 기준(launch가 ur_description ur20 URDF 사용). 카메라 포함
  로봇으로 플래닝하려면 `ur20_description/ur20_with_camera.xrdf`와 매칭되는 URDF 필요.
- apt 예제 `ur_isaac_sim.launch.py`(4.4)는 존재하지 않는 `cumotion_planner_node`를
  참조하는 stale 상태 → 그대로 쓰지 말 것. 이 프로젝트 launch 사용.

## Stop/Play
Isaac 타임라인 Stop/Pause↔Play를 매끄럽게 처리하기 위해 세 가지를 한다:
1. `isaac_pipeline.py` 메인 루프가 **전환 시마다 ArticulationController command 입력
   (jointNames/positionCommand)을 비움** → SubscribeJointState의 stale 명령이 Play 시 재적용돼
   로봇이 튀는 것 방지(참조 start_isaac_sim_ur20.py).
2. **Play 시 `set_start_pose(ROBOT_START_STATE)`** 재적용 → Stop이 로봇을 USD 기본(0)으로
   리셋하므로 시작 자세로 복귀(step 이후 성공할 때까지 재시도).
3. **`ReadSimTime.inputs:resetOnStop=False`** (build_moveit_graph) → sim time이 Stop 시 0으로
   리셋되지 않고 단조 증가. 안 그러면 /clock이 뒤로 점프해 MoveIt/controller_manager
   (use_sim_time)가 회복에 한참 걸림 → False면 Play 후 즉시 재연동.

## 트러블슈팅 (실제 겪은 것)
- cuMotion **"CUDA error: no CUDA-capable device is detected"** + `nvidia-smi`가
  **"Failed to initialize NVML: Unknown Error"**: 컨테이너가 GPU 접근을 잃은 것(Docker+NVIDIA
  cgroup 이슈, 호스트 systemctl daemon-reload/오랜 가동 후 발생). 코드 문제 아님 →
  **`docker restart ros-jazzy`** 로 GPU 접근 복구.
- `ros2_control_node` **segfault** (`HardwareComponentInterface::get_lifecycle_id`):
  apt topic_based(99.99.1) ABI 불일치 → overlay 소스빌드 + source (위).
- 컨트롤러 **"Switch controller timed out after 5s"** / 두 번째 gate `/joint_states` 타임아웃:
  `/clock` 미발행 → Isaac 앱을 ROS2PublishClock 포함 버전으로 재시작.
- cuMotion **"Invalid XRDF ... non-positive radius [-0.01] ... tool0"**: ur20.xrdf의 tool0
  placeholder 구(radius -0.01)를 양수(0.01)로 수정함(4.4는 음수 radius 거부).

## 최근 변경 — 모드 분리·게이팅 최종 동작 (2026-06)

여러 번의 실기 테스트로 모드 전환 시 재생/떨림/오작동을 잡으며 정리한 **현재 동작**.
앞 절들의 일부 초기 설명(idle-latch, resume-cancel relay, 그래프 set_active 토글로 모드 전환)은
아래로 대체됨.

### 1) Run 모드 그래프 — `/MoveItGraph`는 끄지 않는다
- `/MoveItGraph`는 **항상 틱**(`set_active(False)` 안 함). 끄면 노드가 compute를 멈춰 토픽/상태
  갱신·재구독이 안 돼 stale·loop 버그가 났다.
- 대신 **sim 전용 노드만** on/off (`_set_moveit_driving`): `ArticulationController`,
  `PublishJointState`, `PublishClock` → **sim=ON / real=OFF**. `SubscribeJointCommand`는 항상 ON.
- `/ActionGraph`(미러)는 종전대로 **real=ON / sim=OFF**.
- real에서 `PublishJointState`를 끄는 이유: 켜두면
  `/isaac_joint_states → (셸2) /joint_states → /ActionGraph 미러 → Isaac → /isaac_joint_states`
  **자기참조 루프로 로봇이 계속 떨림**.

### 2) 컨트롤러 분리 (Pipeline 모드)
- MoveIt → `scaled_joint_trajectory_controller`, Inspection → `joint_trajectory_controller`.
- `apply_pipeline_mode`가 `ros2 control switch_controllers`로 한쪽만 활성(다른 쪽 차단).
  전환 전후로 **양쪽 컨트롤러 goal을 cancel**(나가는 건 활성 중에, 들어오는 건 전환 후) →
  비활성화로 status가 EXECUTING으로 굳어 relay가 오인 forwarding 하던 **떨림** 방지.
- 실행 중(`pub_runner.running`)엔 Pipeline·Run 콤보 둘 다 잠금.

### 3) Cross-mode 재생 차단 = relay 모드 게이트
- `/isaac_joint_commands`를 먹이는 **유일한 지점인 relay**에 파라미터 `forward_enabled` 추가.
  relay는 `forward_enabled AND goal-active`일 때만 raw→`/isaac_joint_commands` 전달.
- 앱이 `apply_mode`에서 `ros2 param set /isaac_joint_command_relay forward_enabled true|false`
  (**sim=true / real=false**). real에선 relay가 명령을 안 흘려 real-mode 실행이 Isaac에 **도달
  자체가 안 됨** → sim 전환 시 재생 없음(goal이 살아있어도 차단). rebuild/topic-swap/타이머 방식은
  모두 실패해 폐기, 이 게이트가 정답.
- (참고) Isaac `ROS2SubscribeJointState`는 런타임 `topicName` 변경을 무시하고 구독을 유지함 —
  토픽 스왑으로는 못 막는다.

### 4) Inspection Publish 라우팅 — 셸2 독립 (run모드별 자동)
- `publish_trajectory.py --target {isaac|controller}` 추가, 앱이 run모드로 선택:
  - **sim** = `--target isaac`: `/isaac_joint_commands`로 JointState **100Hz 직접 스트리밍**
    (`STREAM_HZ`, 선형 재샘플링으로 매끄럽게). 현재자세는 `/isaac_joint_states`에서 읽음.
    **셸2 없이도** Isaac sim 로봇 구동.
  - **real** = `--target controller`: `FollowJointTrajectory` → `joint_trajectory_controller`.
- Cancel: sim=스트림 종료(퍼블리셔 kill→마지막 자세 유지), real=컨트롤러 cancel-goal.

### 5) real cuMotion Execute — 종료속도 0 아님 허용
- cuMotion 궤적은 마지막 점 속도가 정확히 0이 아니라(예: 0.006 rad/s) UR 컨트롤러가 기본
  거부("Velocity of last trajectory point ... is not zero"→ABORTED). `ur20_real_moveit.launch.py`가
  컨트롤러 기동 후 `scaled`/`joint_trajectory_controller`에
  `allow_nonzero_velocity_at_trajectory_end=true`를 **자동 param set**(런타임 적용 검증됨).

### 알려진 미해결 (TODO)
- **Inspection real**에서 Publish가 실로봇을 안 움직이고(=`joint_trajectory_controller` 비활성),
  MoveIt(scaled)로는 움직임 → 컨트롤러 전환이 real 스택에 적용 안 되는 케이스. inspection 모드
  진입 시 real 스택에서도 jtc 활성/scaled 비활성이 확실히 되도록 보완 필요.

## 관련
- Isaac 앱 모드 전환/그래프: [moveit_inspection_mode.md](moveit_inspection_mode.md)
