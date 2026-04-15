# 비전 검사 파이프라인 실행 방법

로봇 비전 검사를 위한 4단계 파이프라인

## 파이프라인 개요

```
1단계: 뷰포인트 생성 → 2단계: 궤적 생성 → 3단계: 시뮬레이션 → 4단계: ROS 퍼블리시
```

## 실행 준비
Docker 컨테이너 실행
```bash
cd gtsp_trajectory
./execution.sh
```

컨테이너 안에서 ROS2 통신을 위한 환경 변수 설정
```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=<사용할_ID_숫자>
```

컨테이너 안에서 작업 디렉토리로 이동
```bash
cd /curobo/gtsp_trajectory
```

## 전체 실행 예시

```bash
# 1단계: 뷰포인트 생성
omni_python scripts/1_create_viewpoint.py \
    --object sample --material-rgb "170,163,158"

# 2단계: 궤적 생성
omni_python scripts/2_generate_trajectory.py \
    --object sample --num_viewpoints 163

# 3단계: 시뮬레이션
omni_python scripts/3_simulation.py \
    --object sample --num_viewpoints 163

# 4단계: 생성된 궤적으로 ROS로 퍼블리시
omni_python scripts/4_publish_trajectory.py \
    --object sample --num_viewpoints 163
```

## 스크립트 설명

### 1_create_viewpoint.py
멀티 머티리얼 메시에서 검사 대상 표면 추출 및 뷰포인트 생성

```bash
--object         # 객체 이름 (필수)
--material-rgb   # 대상 머티리얼 RGB "R,G,B" (필수)
--visualize      # Open3D 시각화 (옵션)
```

### 2_generate_trajectory.py
모든 뷰포인트를 최적 순서로 방문하는 충돌 없는 로봇 궤적 계산

```bash
--object          # 객체 이름 (필수)
--num_viewpoints  # 뷰포인트 개수 (필수)
```

### 2-1_tilt_trajectory.py
물체 검사 궤적에서 상하좌우로 움직이며 tilt 검사를 실시

```bash
--object          # 객체 이름 (필수)
--num_viewpoints  # 뷰포인트 개수 (필수)
```

### 3_simulation.py
Isaac Sim에서 궤적 실행

```bash
--object            # 객체 이름 (필수)
--num_viewpoints    # 뷰포인트 개수 (필수)
--visualize_spheres # 로봇 충돌 구체 표시 (옵션)
--tilt              # tilt 궤적 사용 (옵션)
```

### 4_publish_trajectory.py
ROS2 토픽으로 궤적 퍼블리시
이 스크립트는 궤적을 보내기만 하고, 
moveit 쪽에서 받은 궤적을 따라 로봇을 움직입니다.

```bash
--object          # 객체 이름 (필수)
--num_viewpoints  # 뷰포인트 개수 (필수)
--tilt              # tilt 궤적 사용 (옵션)
```

## 디렉토리 구조

```
data/{object}/
├── mesh/
│   ├── source.obj        # 입력 메시
│   └── source.mtl        # 머티리얼 정의
├── viewpoint/{num}/
│   └── viewpoints.h5     # 1단계 출력
└── trajectory/{num}/
    └── trajectory.csv    # 2단계 출력
```

## 설정

`common/config.py`에서 파라미터 설정

### 카메라 사양
| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `CAMERA_FOV_WIDTH_MM` | 41.0 | 카메라 시야각 너비 (mm) |
| `CAMERA_FOV_HEIGHT_MM` | 30.0 | 카메라 시야각 높이 (mm) |
| `CAMERA_WORKING_DISTANCE_MM` | 110.0 | 작업 거리 (mm) |
| `CAMERA_OVERLAP_RATIO` | 0.5 | 뷰포인트 유효 면적 비율 |

### 로봇 설정
| 파라미터 | 설명 |
|---------|------|
| `DEFAULT_ROBOT_CONFIG` | 로봇 YAML 설정 파일 |
| `TOOL_TO_CAMERA_OPTICAL_OFFSET_M` | 툴-카메라 오프셋 (m) |

### IK/충돌/재계획 파라미터
| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `COLLISION_MARGIN` | 0.0 | 충돌 마진 |
| `REPLAN_MAX_ATTEMPTS` | 60 | 재계획 최대 시도 횟수 |
| `REPLAN_TIMEOUT` | 10.0 | 재계획 타임아웃 (초) |

### 월드 설정
`TARGET_OBJECT`, `TABLE`, `WALLS`, `ROBOT_MOUNT` 등 환경 및 물체에 대한 정보

