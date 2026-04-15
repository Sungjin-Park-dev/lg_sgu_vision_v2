# 실행 방법

## 1. UR 드라이버 실행

궤적을 전송하려면 먼저 UR 드라이버를 실행해야 한다.

### Mock Hardware (개발용, 추천)

URSim/실제 로봇 없이 가상 컨트롤러로 동작. 코드 변경 없이 실제 로봇 전환 가능.

```bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur5e \
  robot_ip:=yyy.yyy.yyy.yyy \
  use_mock_hardware:=true \
  launch_rviz:=true \
  initial_joint_controller:=scaled_joint_trajectory_controller
```
- `robot_ip`는 아무 값이나 넣어도 됨 (필수 파라미터이지만 연결 안 함)

### URSim 연결

```bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur5e \
  robot_ip:=192.168.56.101 \
  launch_rviz:=true
```

### RViz 마커 (작업 환경 시각화)

```bash
uv run scripts/publish_workcell_markers.py --object sample
```

## 2. 뷰포인트 생성

```bash
# 기본 (클러스터 없이)
uv run scripts/generate_viewpoints.py --object sample --visualize

# 클러스터링 포함 (경로 최적화)
uv run scripts/generate_viewpoints.py --object sample --cluster
```

클러스터 없이도 동작하지만, `--cluster`를 사용하면 MotionGen 하이브리드 경로를 사용한다.

## 3. 궤적 생성 및 전송

### IK + matplotlib 시각화 (ROS2 불필요)

```bash
uv run scripts/plan_motion.py --object sample --num-viewpoints 124
```

### ROS2로 궤적 전송

```bash
uv run scripts/plan_motion.py --object sample --num-viewpoints 124 --publish --dt 0.2
```

주요 옵션:
| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--publish` | off | ROS2 action 전송 (없으면 matplotlib 시각화) |
| `--dt` | 1.0 | 웨이포인트 간 시간 (초) |
| `--interp-spacing` | 2.0 | 밀집 보간 간격 (mm) |
| `--num-rotations` | 8 | z축 회전 수 (goalset 크기) |
| `--robot` | ur5e.yml | cuRobo 로봇 설정 파일 |

### 시작 자세로 복귀

```bash
uv run scripts/move_to_start.py
```

## ROS2 인터페이스

| 인터페이스 | 타입 | 용도 |
|-----------|------|------|
| `/scaled_joint_trajectory_controller/follow_joint_trajectory` | Action (FollowJointTrajectory) | 궤적 실행 |
| `/joint_states` | Topic (JointState) | 로봇 현재 상태 (드라이버가 발행) |

### /joint_states 주의사항

토픽이 조인트를 **알파벳 순서**로 발행하지만, 코드에서는 **이름 매칭**으로 처리하므로 순서 무관.

## 트러블슈팅

### Tolerance 에러

컨트롤러 기본 tolerance: 0.2 rad. 초과 시:
```
State tolerances failed for joint N: Position Error: X, Position Tolerance: 0.200000
```

`plan_motion.py`는 이미 아래 대책을 포함:
1. 현재 위치를 t=0 포인트로 포함
2. 포인트 간 보간 (max_step 0.1 rad < tolerance/2)
3. 시간 할당을 joint 변화량에 비례하게 설정
