# URSim + ur_robot_driver 셋업

UR20 시뮬레이터(URSim) 또는 mock hardware로 ROS2 driver와 통신하는 환경 구성. `docs/setup_docker.md`의 컨테이너 셋업이 완료된 상태에서 진행.

## 두 가지 모드

| 모드 | URSim | 용도 |
|---|---|---|
| **Mock hardware** | 불필요 | 개발/디버깅 — 빠른 iteration, tolerance 위반 없음 |
| **URSim (full)** | 필요 | 검증 — 실제 PolyScope 환경, robot mode 시뮬, safety mode |

대부분의 개발은 **mock hardware**로 충분. URSim은 실로봇 전 마지막 검증 단계에서.

---

## Mode A: Mock Hardware (개발 추천)

URSim 없이 driver만 띄움. 명령된 joint 위치를 즉시 state로 반영 — 물리 시뮬 없음.

```bash
# 컨테이너, 셸 A
source /opt/ros/jazzy/setup.bash
ros2 launch ur_robot_driver ur_control.launch.py \
    ur_type:=ur20 \
    robot_ip:=192.168.56.101 \
    use_mock_hardware:=true \
    launch_rviz:=true
```

### 장점
- PolyScope 초기화/Play 단계 불필요
- External Control URCap 설정 불필요
- `Position Tolerance` 위반 같은 트래킹 이슈 없음
- 실행 즉시 `scaled_joint_trajectory_controller`가 active

### 검증
```bash
# 다른 셸에서
ros2 service call /controller_manager/list_controllers \
    controller_manager_msgs/srv/ListControllers \
    | grep -A1 scaled_joint_trajectory
# state='active' 보여야 함

ros2 topic echo /joint_states --once   # joint position 출력
```

이 상태에서 `move_to_start.py`, `publish_trajectory.py` 등 모두 동작.

---

## Mode B: URSim (full simulator)

PolyScope GUI + 실제 robot mode/safety mode 시뮬. URSim은 별도 Docker 컨테이너로 실행됨.

### 1) URSim 띄우기

`ros-jazzy` 컨테이너 안에서 (Docker socket 마운트 덕에 호스트 daemon 사용):

```bash
ros2 run ur_client_library start_ursim.sh -m UR20
```

처음 실행 시 `universalrobots/ursim_e-series` Docker 이미지 다운로드 (~몇 분).

지원 모델: `UR3`, `UR3e`, `UR5`, `UR5e`, `UR10`, `UR10e`, `UR16e`, `UR20`, `UR30`.

전체 옵션: `start_ursim.sh -h`

### 2) URSim 도달 확인

```bash
# 컨테이너 안
ping -c 2 192.168.56.101
nc -zv 192.168.56.101 29999    # dashboard
nc -zv 192.168.56.101 30001    # primary interface
```

응답 오면 OK. 안 오면 호스트 `vboxnet0` 인터페이스 확인:
```bash
# 호스트에서
ip -br addr | grep 192.168.56
```

### 3) PolyScope GUI 접속

웹 브라우저로 `http://192.168.56.101:6080/vnc.html` (VNC over web).

### 4) 로봇 초기화

PolyScope **좌하단** 빨간 점 클릭:
1. **ON** (전원)
2. **START** (브레이크 해제)
3. 좌하단이 **초록색 "Normal"**이 되면 완료

### 5) External Control 프로그램 만들기

External Control URCap이 driver와 연결을 담당. 이 프로그램이 Play 상태여야 trajectory 전송 가능.

**Program 탭 → 빈 프로그램에 노드 추가**:
- **URCaps → External Control** 노드 추가

**External Control URCap 노드 클릭 → Host IP 설정**:
- driver가 도달 가능한 호스트의 IP
- **권장**: `192.168.56.1` (호스트의 vboxnet 인터페이스, URSim과 같은 서브넷)
- Custom port: 기본값 `50002` 그대로

### 6) ur_robot_driver 실행

```bash
# 컨테이너, 셸 A
source /opt/ros/jazzy/setup.bash
ros2 launch ur_robot_driver ur_control.launch.py \
    ur_type:=ur20 \
    robot_ip:=192.168.56.101 \
    launch_rviz:=true
```

driver 로그에 `Connected: Universal Robots Dashboard Server` 떠야 함.

### 7) 프로그램 Play

PolyScope 우하단 ▶ **Play** 버튼 누름.

driver 로그에:
```
Robot connected to reverse interface
```

`scaled_joint_trajectory_controller`가 자동으로 `inactive` → `active`로 전환.

---

## Move to Start

URSim 또는 mock hardware 어느 쪽이든 사용 가능:

```bash
uv run scripts/ros2/move_to_start.py
```

목표 자세는 `scripts/common/config.py`의 `ROBOT_START_STATE` (radian 배열).

기본값:
- `--duration 5.0` — 최소 이동 시간 (초)
- `--max-vel 0.5` — joint 최대 각속도 (rad/s)

URSim 모드에서 `Position Tolerance` 위반(error_code=-4) 시:
```bash
uv run scripts/ros2/move_to_start.py --duration 10 --max-vel 0.3
```

천천히 이동시키면 controller가 트래킹 가능.

---

## 셸 분리 (URSim 모드)

URSim 사용 시 동시에 띄우는 셸 구성:

| 셸 | 명령 | 용도 |
|---|---|---|
| **A** | `start_ursim.sh -m UR20` | URSim 컨테이너 (호스트 daemon에 띄움) |
| **B** | `ur_control.launch.py ...` | ur_robot_driver |
| **C** | `move_to_start.py`, `publish_trajectory.py` | 명령 전송 |
| **D** | `joint_control.py` (Isaac Sim) | 시각화 — 옵션 |

각각 `docker exec -it ros-jazzy bash`로 별도 셸 진입.
