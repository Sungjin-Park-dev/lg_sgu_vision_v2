# 실행 방법

UR20 비전 검사 파이프라인 실행 흐름. 환경 셋업은 [setup_docker.md](setup_docker.md), [setup_ursim.md](setup_ursim.md), [setup_isaac_sim.md](setup_isaac_sim.md) 참조.

## 전체 흐름

```
[1. 뷰포인트 생성]   generate_viewpoints.py → viewpoints.h5
        ↓
[2. IK + 궤적 계획]  plan_trajectory.py        → trajectory_dp.csv
        ↓
[3. 로봇 실행]       publish_trajectory.py  → ROS2 action goal
        ↓                                      ↑
[4. 시각화 (선택)]   scene.py  └─ ur_robot_driver
                     (Isaac Sim)
```

1, 2단계는 **CSV 생성** (오프라인 계산), 3단계는 **로봇 제어** (ROS2 통신), 4단계는 **시각화**.

## 셸 구성

모든 작업은 `ros-jazzy` 컨테이너 안에서 진행. 추천 셸 분리:

| 셸 | 환경 | 용도 |
|---|---|---|
| **A** | `source /opt/ros/jazzy/setup.bash`, venv 비활성 | UR driver, RViz |
| **B** | venv 활성, 시스템 ROS2 sourcing X | Isaac Sim |
| **C** | 시스템 ROS2 + venv | 파이프라인 스크립트 (1, 2, 3단계) |
| **D** | (호스트 또는 컨테이너) | URSim 띄우기 — 실로봇 검증 시 |

진입:
```bash
# 호스트에서
docker exec -it ros-jazzy bash
```

이유: 시스템 ROS2와 Isaac Sim 번들 ROS2를 같은 셸에서 sourcing하면 FastDDS/FastCDR ABI 충돌 위험. 셸을 나누면 안전.

---

## 1. 뷰포인트 생성

**셸 C** (venv + ROS2 sourcing 무관, 순수 Python 작업):

```bash
# 기본 (dbscan 클러스터링)
uv run scripts/core/generate_viewpoints.py --object sample

# 재질 RGB 필터 + 옵션
uv run scripts/core/generate_viewpoints.py --object sample --material-rgb "0,255,0" \
    --cluster-method coacd+dbscan --normal-weight 0.05 --coacd-threshold 0.25
```

출력:
- `data/{object}/viewpoint/{N}/viewpoints_{method}.h5`
- `data/{object}/viewpoint/{N}/viewpoints_{method}.html` (시각화)

상세 옵션은 [generate_viewpoints.md](generate_viewpoints.md) 참조.

---

## 2. IK + 궤적 계획

**셸 C** (venv + GPU 사용 — cuRobo):

```bash
uv run scripts/core/plan_trajectory.py --object sample --num-viewpoints 124
```

옵션:
| 플래그 | 기본값 | 설명 |
|---|---|---|
| `--object` | (필수) | Object name |
| `--num-viewpoints` | (필수) | 뷰포인트 수 |
| `--viewpoints` | None | h5 파일 직접 지정 |
| `--spacing` | 0.01 | EE resample 간격 (m) |

**사전조건**: 1단계의 `viewpoints_*.h5` 필요.

출력:
- `data/{object}/trajectory/{N}/trajectory_dp_*.csv` — joint trajectory
- `data/{object}/trajectory/{N}/trajectory_dp_*.html` — 시각화
- `data/{object}/trajectory/{N}/trajectory_dp_*_anim.html` — 애니메이션

상세는 [plan_trajectory.md](plan_trajectory.md) 참조.

---

## 3. 로봇 실행 (ROS2)

### 3a. UR Driver

**셸 A** — 두 가지 모드 중 선택:

**Mock Hardware** (개발 추천 — URSim 없이 즉시 동작):
```bash
source /opt/ros/jazzy/setup.bash
ros2 launch ur_robot_driver ur_control.launch.py \
    ur_type:=ur20 \
    use_mock_hardware:=true \
    robot_ip:=192.168.56.101 \
    launch_rviz:=true
```

**URSim** (실제 PolyScope 환경):
```bash
ros2 launch ur_robot_driver ur_control.launch.py \
    ur_type:=ur20 \
    robot_ip:=192.168.56.101 \
    launch_rviz:=true
```
URSim은 별도 컨테이너로 띄움 + PolyScope에서 External Control 프로그램 Play 필요. [setup_ursim.md](setup_ursim.md) 참조.
```

### 3b. 시작 자세 복귀

**셸 C**:
```bash
uv run scripts/robot/move_to_start.py
```

목표 자세: `scripts/common/config.py`의 `ROBOT_START_STATE`.

기본값: `--duration 5.0`, `--max-vel 0.5`. tolerance 위반 시 더 천천히:
```bash
uv run scripts/robot/move_to_start.py --duration 10 --max-vel 0.3
```

### 3c. 궤적 전송

**셸 C**:

```bash
uv run scripts/core/publish_trajectory.py \
    --csv data/sample/trajectory/124/trajectory_dp_ee_s0010_eev50mms_av20dps_jv0p30_corner30d_x2p5.csv
```

스크립트가 자동 처리:
- 현재 위치를 t=0 포인트로 포함
- 포인트 간 보간 (`MAX_STEP_RAD=0.1`)
- CSV의 `time` 컬럼을 ROS trajectory `time_from_start`로 사용
- CSV 헤더 prefix(`ur20_*`) 자동 매칭 (suffix 기반)

### 3d. RViz 워크셀 마커 (선택)

```bash
uv run scripts/robot/publish_workcell_markers.py --object sample
```

`config.py`의 TABLE/WALLS/ROBOT_MOUNT/TARGET_OBJECT를 RViz에 마커로 표시.

---

## 4. Isaac Sim 시각화 (선택)

**셸 B**:
```bash
source /workspace/.venv/bin/activate
uv run scripts/isaac/scene.py --object sample
```

- `--no-sync` — `uv sync`가 cuRobo path-install을 건드리지 않도록
- `--object sample` — 워크셀(테이블/벽/타겟) 함께 로드
- `--usd-path PATH` — 다른 USD 사용 (기본: `ur20_description/ur20/ur20.usd`)

GUI에 UR20 + 워크셀 표시. ROS2 `/joint_states` 구독 → 시뮬 로봇이 driver 상태 미러링.

USD가 없으면 GUI URDF Importer로 먼저 변환. [setup_isaac_sim.md](setup_isaac_sim.md) 참조.
```

---

## ROS2 인터페이스

| 인터페이스 | 타입 | 용도 |
|---|---|---|
| `/scaled_joint_trajectory_controller/follow_joint_trajectory` | Action | 궤적 실행 |
| `/joint_states` | Topic | 로봇 현재 상태 (driver 발행) |
| `/io_and_status_controller/robot_mode` | Topic | 로봇 모드 (RUNNING=7) |
| `/io_and_status_controller/safety_mode` | Topic | 안전 모드 (NORMAL=1) |

`/joint_states`는 조인트를 알파벳 순서로 발행하지만 스크립트가 이름 매칭으로 처리하므로 순서 무관.

---

## 전체 워크플로우 예시 (sample 객체, 124 뷰포인트)

```bash
# 셸 C — 오프라인 계산
uv run scripts/core/generate_viewpoints.py --object sample \
    --material-rgb "0,255,0" --cluster-method coacd+dbscan \
    --normal-weight 0.05 --coacd-threshold 0.25
uv run scripts/core/plan_trajectory.py --object sample --num-viewpoints 124

# 셸 A — driver
source /opt/ros/jazzy/setup.bash
ros2 launch ur_robot_driver ur_control.launch.py \
    ur_type:=ur20 use_mock_hardware:=true launch_rviz:=true

# 셸 B — Isaac Sim 시각화 (선택)
source /workspace/.venv/bin/activate
OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python \
    scripts/isaac/scene.py --object sample

# 셸 C — 로봇 제어
source /opt/ros/jazzy/setup.bash
uv run scripts/robot/publish_workcell_markers.py --object sample &
uv run scripts/robot/move_to_start.py
uv run scripts/core/publish_trajectory.py \
    --csv data/sample/trajectory/124/trajectory_dp_ee_s0010_eev50mms_av20dps_jv0p30_corner30d_x2p5.csv
```

---

## 트러블슈팅

### `Goal rejected: Controller is not running`

URSim 모드에서 External Control 프로그램이 Play되지 않음. PolyScope에서 ▶ 누르기.
또는 mock hardware 모드 사용.

### `State tolerances failed: Position Error: X, Tolerance: 0.200000`

trajectory가 너무 빠름. controller가 트래킹 못 함:
- `move_to_start.py --duration 10 --max-vel 0.3`로 천천히
- 또는 mock hardware 모드 (tolerance 위반 없음)

### `KeyError: 'shoulder_pan_joint'` (publish_trajectory.py)

CSV 헤더가 표준 이름이거나 prefix(`ur20_`) 붙은 형식 — 둘 다 지원. ValueError 메시지에 실제 fieldnames 표시되니 확인.

### Isaac Sim `Available DOFs: []`

USD의 ArticulationRoot가 잘못 적용됨. GUI URDF Importer로 재변환. [setup_isaac_sim.md](setup_isaac_sim.md) 트러블슈팅 참조.

### 더 자세한 트러블슈팅

- 컨테이너/환경 문제: [setup_docker.md](setup_docker.md)
- URSim/driver 문제: [setup_ursim.md](setup_ursim.md)
- Isaac Sim 문제: [setup_isaac_sim.md](setup_isaac_sim.md)
