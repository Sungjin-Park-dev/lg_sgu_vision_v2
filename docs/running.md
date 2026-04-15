# 실행 방법

## 1. 뷰포인트 생성

```bash
# 기본 (dbscan 클러스터링)
uv run scripts/generate_viewpoints.py --object sample

# 재질 RGB 필터 + 옵션
uv run scripts/generate_viewpoints.py --object sample --material-rgb "170,163,158" \
  --cluster-method coacd+dbscan --normal-weight 0.05 --coacd-threshold 0.25

# 파라미터 비교 HTML (여러 변형을 드롭다운으로)
uv run scripts/generate_viewpoints.py --object sample --cluster-method dbscan --compare

# 통계만 확인 (HDF5 저장 안 함)
uv run scripts/generate_viewpoints.py --object sample --dry-run
```

출력:
- `data/{object}/viewpoint/{num}/viewpoints_{method}.h5`
- `data/{object}/viewpoint/{num}/viewpoints_{method}.html`

상세 옵션은 [generate_viewpoints.md](generate_viewpoints.md) 참조.

## 2. 궤적 생성

```bash
uv run scripts/plan_motion.py --object sample --num-viewpoints 124

# 보간 간격 변경 (기본 2mm)
uv run scripts/plan_motion.py --object sample --num-viewpoints 124 --interp-spacing 5.0
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--object` | (필수) | Object name |
| `--num-viewpoints` | (필수) | 뷰포인트 수 |
| `--robot` | `ur20_with_camera.yml` | cuRobo 로봇 설정 파일 |
| `--interp-spacing` | 2.0 | Cartesian 보간 간격 (mm) |

**사전조건**: `viewpoints.h5`에 클러스터 데이터 필수. `generate_viewpoints.py`를 먼저 실행.

출력: `data/{object}/trajectory/{num}/trajectory.csv` + `trajectory.html`

## 3. 로봇 실행 (ROS2)

### UR 드라이버

Mock Hardware (개발용, 추천):

```bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur5e \
  robot_ip:=yyy.yyy.yyy.yyy \
  use_mock_hardware:=true \
  launch_rviz:=true \
  initial_joint_controller:=scaled_joint_trajectory_controller
```
- `robot_ip`은 필수 파라미터지만 mock 모드에서는 값 무관.

URSim 또는 실제 로봇:

```bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur5e \
  robot_ip:=192.168.56.101 \
  launch_rviz:=true
```

### 궤적 전송

```bash
# 저장된 trajectory.csv를 로봇으로 전송
uv run scripts/publish_trajectory.py --object sample --num-viewpoints 124

# CSV 경로 직접 지정
uv run scripts/publish_trajectory.py --csv data/sample/trajectory/124/trajectory.csv
```

### 시작 자세 복귀

```bash
uv run scripts/move_to_start.py
```

### RViz 마커 (작업 환경 시각화)

```bash
uv run scripts/publish_workcell_markers.py --object sample
```

## ROS2 인터페이스

| 인터페이스 | 타입 | 용도 |
|-----------|------|------|
| `/scaled_joint_trajectory_controller/follow_joint_trajectory` | Action | 궤적 실행 |
| `/joint_states` | Topic | 로봇 현재 상태 (드라이버가 발행) |

`/joint_states`는 조인트를 **알파벳 순서**로 발행하지만 코드는 **이름 매칭**으로 처리하므로 순서 무관.

## 트러블슈팅

### Tolerance 에러

컨트롤러 기본 tolerance: 0.2 rad. 초과 시:
```
State tolerances failed for joint N: Position Error: X, Position Tolerance: 0.200000
```

`publish_trajectory.py`는 이미 아래 대책을 포함:
1. 현재 위치를 t=0 포인트로 포함
2. 포인트 간 보간 (`MAX_STEP_RAD=0.1` < tolerance/2)
3. 시간 할당을 joint 변화량에 비례하게 설정 (`MAX_JOINT_VEL=2.0 rad/s`)
