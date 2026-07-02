# UR20 2×2 모드 실행 가이드 (sim/real × MoveIt/Inspection)

Docker 컨테이너 시작부터 4개 모드 각각의 실행까지. **설치는 다루지 않음**
(설치·구성은 [moveit_cumotion_ur20.md](moveit_cumotion_ur20.md), [isaac_sim_full_setup.md](isaac_sim_full_setup.md)).

---

## 0. 개념 요약

로봇 동작은 **두 축**으로 정해진다.

| 축 | 값 | 의미 |
|---|---|---|
| **Run 모드** | `sim` / `real` | 로봇 주체. sim=Isaac이 로봇, real=실(또는 mock)로봇 + Isaac은 미러(트윈) |
| **Pipeline 모드** | `moveit` / `inspection` | 명령 소스. moveit=RViz Plan&Execute, inspection=Publish 패널(검사 경로) |

이 둘을 곱한 **2×2** 가 아래 4개 모드다.

|  | **sim** (Isaac이 로봇) | **real** (실/mock 로봇, Isaac 미러) |
|---|---|---|
| **MoveIt** | RViz Execute → Isaac | RViz Execute → 실로봇, Isaac이 따라감 |
| **Inspection** | Publish → Isaac | Publish → 실로봇, Isaac이 따라감 |

**셸 구성**
- **셸1** = Isaac 앱 (`isaac_pipeline.py`). 항상 필요.
- **셸2** = ROS 스택(MoveIt/컨트롤러). Run 모드에 맞는 것을 띄운다.

**핵심 규칙**
- **Run 모드 ↔ 셸2 스택은 반드시 짝**을 맞춘다. Run 모드를 바꾸면 **셸2도 해당 스택으로 재실행**.
- Pipeline 모드(MoveIt↔Inspection)는 UI 드롭다운으로 자유롭게 전환 가능(셸 재시작 불필요).
- `--mode` / `--pipeline-mode` 는 **부팅 기본값**일 뿐, UI에서 바꿀 수 있다.

---

## 1. Docker 컨테이너 시작 (모든 모드 공통)

```bash
# 컨테이너 실행 (이미 켜져 있으면 무시됨)
docker start ros-jazzy

# 실행 확인
docker ps | grep ros-jazzy
```

> **GPU 오류 시**: `nvidia-smi`가 "Failed to initialize NVML" 또는 cuMotion이 "no CUDA-capable
> device" 라면 컨테이너가 GPU 접근을 잃은 것 → `docker restart ros-jazzy` 후 다시 시작.

각 "셸"은 아래처럼 `docker exec`로 컨테이너에 새로 붙는 터미널이다(별도 터미널 창에서 실행).

---

## 2. 공통 실행 조각 (아래 모드에서 재사용)

### 셸1 — Isaac 앱 (venv, 시스템 ROS source 안 함)
```bash
docker exec -it ros-jazzy bash -c \
  'source /workspace/.venv/bin/activate && cd /workspace && OMNI_KIT_ACCEPT_EULA=YES \
   python scripts/apps/isaac_pipeline.py --object sample --mode <RUN> --pipeline-mode <PIPE>'
```
- `<RUN>` = `sim` | `real`, `<PIPE>` = `moveit` | `inspection`
- 앱이 뜨면 뷰포트에서 **▶ Play**(타임라인 재생)를 눌러야 로봇/그래프가 돈다.

### 셸2-A — SIM 스택 (Run=sim일 때)
```bash
docker exec -it ros-jazzy bash -c \
  'source /opt/ros/jazzy/setup.bash && source /workspace/ros2_overlay/install/setup.bash \
   && cd /workspace && ros2 launch scripts/moveit/ur20_isaac_state_synced.launch.py'
```
> SIM 스택은 `topic_based_ros2_control` 브리지 때문에 **`ros2_overlay`까지 source** 해야 한다.
> **셸1이 sim으로 Play된 뒤** 실행할 것(컨트롤러가 Isaac의 `/clock`·상태를 기다림).

### 셸2-B — REAL 스택 (Run=real일 때, 기본 mock_hardware)
```bash
docker exec -it ros-jazzy bash -c \
  'source /opt/ros/jazzy/setup.bash && cd /workspace \
   && ros2 launch scripts/moveit/ur20_real_moveit.launch.py'
```
- 실제 UR20 로봇을 쓸 때만: `... use_mock_hardware:=false robot_ip:=<로봇IP>`
- ur_robot_driver가 launch에 포함되어 별도 드라이버 셸 불필요. RViz 1개가 함께 뜬다.

---

## 3. 모드별 실행

### A. MoveIt × sim — RViz로 Isaac 로봇 구동
Isaac이 로봇이고, RViz(cuMotion)로 계획·실행한다.

```bash
# 셸1
docker exec -it ros-jazzy bash -c \
  'source /workspace/.venv/bin/activate && cd /workspace && OMNI_KIT_ACCEPT_EULA=YES \
   python scripts/apps/isaac_pipeline.py --object sample --mode sim --pipeline-mode moveit'
#   → 앱에서 ▶ Play

# 셸2 (SIM 스택)  ← 셸1 Play 후
docker exec -it ros-jazzy bash -c \
  'source /opt/ros/jazzy/setup.bash && source /workspace/ros2_overlay/install/setup.bash \
   && cd /workspace && ros2 launch scripts/moveit/ur20_isaac_state_synced.launch.py'
```
**사용**: RViz에서 목표 자세 지정 → **Plan & Execute** → Isaac UR20이 그 경로로 움직임.
평소엔 MoveIt이 Isaac 현재 상태를 반영하고, Execute할 때만 로봇이 움직인다.

---

### B. MoveIt × real — RViz로 실(mock)로봇 구동 + Isaac 미러
실(또는 mock)로봇이 움직이고 Isaac은 그 상태를 따라가는 디지털 트윈.

```bash
# 셸1
docker exec -it ros-jazzy bash -c \
  'source /workspace/.venv/bin/activate && cd /workspace && OMNI_KIT_ACCEPT_EULA=YES \
   python scripts/apps/isaac_pipeline.py --object sample --mode real --pipeline-mode moveit'
#   → 앱에서 ▶ Play

# 셸2 (REAL 스택, mock 기본)
docker exec -it ros-jazzy bash -c \
  'source /opt/ros/jazzy/setup.bash && cd /workspace \
   && ros2 launch scripts/moveit/ur20_real_moveit.launch.py'
```
**사용**: RViz에서 **Plan & Execute** → mock 로봇 이동 + Isaac이 미러로 따라감.
- cuMotion 궤적 종료속도가 0이 아니어도 실행되도록 launch가 컨트롤러 파라미터를 자동 설정한다.
- 확인: `ros2 control list_controllers` 에 `scaled_joint_trajectory_controller` = **active**,
  `ros2 topic hz /joint_states` 약 500Hz.

---

### C. Inspection × sim — Publish로 Isaac 로봇 구동 (**셸2 불필요**)
검사 경로(CSV)를 Isaac sim 로봇에 직접 스트리밍한다. **ROS 스택(셸2)이 없어도 동작**한다.

```bash
# 셸1 만 필요
docker exec -it ros-jazzy bash -c \
  'source /workspace/.venv/bin/activate && cd /workspace && OMNI_KIT_ACCEPT_EULA=YES \
   python scripts/apps/isaac_pipeline.py --object sample --mode sim --pipeline-mode inspection'
#   → 앱에서 ▶ Play
```
**사용**: 앱 패널에서 Load → Generate → Preview → **Publish to Robot**.
- sim에서는 Publish가 `/isaac_joint_commands`로 **100Hz 직접 스트리밍**(매끄러움) → Isaac 로봇이 움직임.
- 어떤 셸2가 떠 있든(혹은 없든) 항상 **sim 로봇만** 구동한다.
- **Cancel Publish** = 스트림 즉시 중단(로봇은 마지막 자세 유지).

---

### D. Inspection × real — Publish로 실로봇 구동 + Isaac 미러
검사 경로를 실(mock)로봇에 보내고 Isaac은 미러링.

```bash
# 셸1
docker exec -it ros-jazzy bash -c \
  'source /workspace/.venv/bin/activate && cd /workspace && OMNI_KIT_ACCEPT_EULA=YES \
   python scripts/apps/isaac_pipeline.py --object sample --mode real --pipeline-mode inspection'
#   → 앱에서 ▶ Play

# 셸2 (REAL 스택)
docker exec -it ros-jazzy bash -c \
  'source /opt/ros/jazzy/setup.bash && cd /workspace \
   && ros2 launch scripts/moveit/ur20_real_moveit.launch.py'
```
**사용**: Publish → 실(mock)로봇이 검사 경로로 이동 + Isaac 미러.
- Publish는 `joint_trajectory_controller`(실 스택)로 전송된다.

> ⚠️ **알려진 이슈(수정 예정)**: 현재 Inspection real에서 Publish가 실로봇을 안 움직이고
> (jtc가 비활성 상태) MoveIt(scaled)로만 움직이는 경우가 있다. inspection 진입 시 real 스택에서도
> jtc 활성/scaled 비활성 전환이 확실히 되도록 보완 필요. (자세한 내용은
> [moveit_cumotion_ur20.md](moveit_cumotion_ur20.md)의 "알려진 미해결" 참조.)

---

## 4. Inspection 모드 패널 & 버튼 설명

Inspection 모드의 좌측 패널은 검사 워크플로 **A → B → C → D** 순서다.
(MoveIt 모드에선 이 패널들이 **회색으로 잠긴다**.)

**상단 공통 선택**
- **Pipeline Mode** (콤보): `MoveIt` ↔ `Inspection` 전환.
- **Run Mode (sim / real)** (콤보): `sim` ↔ `real` 전환. 바꾸면 **셸2도 맞는 스택으로 재실행**해야 함.

### A. Load Object — 대상 객체 배치
| 항목 | 설명 |
|---|---|
| **Object** (콤보) | 검사할 객체 선택 |
| **Load Object** | 선택한 객체 USD를 씬에 로드. 이후 뷰포트 gizmo로 이동/회전(**W**=이동, **E**=회전) |
| **Log Pose** | 현재 객체의 위치/자세를 Log 패널에 출력(배치 확인·기록용) |

### B. Generate Trajectory (`plan_trajectory.py`) — 검사 경로 생성
| 항목 | 설명 |
|---|---|
| **Viewpoints (h5)** + **Browse...** | 뷰포인트 `.h5` 파일 경로 지정 |
| **Show Viewpoints** / **Clear Viewpoints** | h5의 뷰포인트를 씬에 표시 / 제거 |
| **Check IK Reachability** / **Cancel IK Check** | 각 뷰포인트가 IK로 도달 가능한지 검사 / 검사 취소 |
| **Advanced** | `--spacing`(경로 간격), `--output-suffix`(출력 파일 접미사) |
| **Generate Trajectory** / **Cancel** | 객체 자세 + 뷰포인트로 경로 계획 실행 → trajectory CSV 생성 / 취소 |

### C. Preview in Simulation — 고스트 미리보기 (시각 전용, 로봇/ROS 안 건드림)
| 항목 | 설명 |
|---|---|
| **CSV path** + **Browse...** | 미리볼 trajectory CSV 경로 |
| **Load & Preview** | CSV를 **고스트(반투명) 로봇**으로 로드 |
| **Play / Pause / Stop** | 고스트 재생 / 일시정지 / 정지 |
| **Show/Clear Collision Spheres** | cuRobo 충돌 구(collision sphere) 표시 / 제거 |
| **Show/Clear FOV Plane** | 카메라 시야(FOV) 평면 표시 / 제거 |
| **t 슬라이더** | 재생 시점 스크럽. 아래 라벨에 `t=현재/전체` 표시 |

> Preview는 **고스트만** 움직인다 — 실제 로봇/Isaac 로봇은 움직이지 않는다.

### D. Publish to Robot (`publish_trajectory.py`) — 실제 구동
| 항목 | 설명 |
|---|---|
| **CSV path** + **Browse...** | 전송할 trajectory CSV(패널 C와 공유) |
| **Publish to Robot** | CSV를 **실제로 로봇에 전송**. `sim`=Isaac에 100Hz 직접 스트리밍, `real`=실로봇 컨트롤러로 전송 |
| **Cancel Publish** | 전송 중단. `sim`=스트림 즉시 종료(마지막 자세 유지), `real`=컨트롤러 goal cancel |

> Preview(C)는 고스트 미리보기일 뿐이고, **로봇을 실제로 움직이는 건 이 Publish(D)** 다.
> 실행 중에는 Pipeline/Run 콤보가 잠긴다.

---

## 5. 모드 전환 빠른 참고

| 하고 싶은 것 | 방법 |
|---|---|
| MoveIt ↔ Inspection | 앱 UI의 **Pipeline Mode** 드롭다운 (셸 재시작 불필요) |
| sim ↔ real | 앱 UI의 **Run Mode** 드롭다운 변경 **＋ 셸2를 해당 스택으로 재실행** |
| 실행 중 전환 | Publish 실행 중에는 두 콤보 모두 잠김 (완료/취소 후 전환) |

## 6. 트러블슈팅 (요약)
- **GPU/NVML 오류·cuMotion CUDA 오류** → `docker restart ros-jazzy`.
- **sim에서 컨트롤러 활성 안 됨 / gate 타임아웃** → 셸1을 먼저 Play한 뒤 셸2를 실행했는지 확인.
- **real에서 Execute가 ABORTED** ("Velocity of last trajectory point ... not zero") → 최신
  `ur20_real_moveit.launch.py` 사용(컨트롤러 파라미터 자동 설정).
- **sim인데 실로봇이 움직임 / Isaac이 안 움직임** → 셸2가 Run 모드와 안 맞는 것. 스택을 맞춰 재실행.
