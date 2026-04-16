# lg_sgu_vision_v2

UR20 로봇을 이용한 비전 검사 궤적 생성 시스템. cuRobo(IK/충돌검사) 기반.

## 디렉토리 구조

```
scripts/
  pipeline/          핵심 파이프라인
    generate_viewpoints.py   뷰포인트 생성 + 클러스터링 + GTSP 순서
    select_ik_dp.py          DBSCAN+DP 기반 IK 해 선택 + MotionGen transit
    publish_trajectory.py    ROS2 궤적 전송
  common/            config, math_utils
  ros2/              ROS2 유틸 (move_to_start, publish_workcell_markers)
  viz/               시각화 (visualize_viewpoints, visualize_coacd)
  isaac/             Isaac Sim 전용
  prev/              이전 버전 스크립트
data/{object}/       mesh/ viewpoint/ trajectory/ (gitignore)
```

## 환경 설정

```bash
# 1. cuRobo 클론
git clone https://github.com/NVlabs/curobo.git

# 2. Python 의존성 설치
uv sync

# 3. cuRobo 로컬 설치 (CUDA 빌드 필요)
cd curobo
uv pip install -e . --no-build-isolation
cd ..
```

## 실행 방법

### 1. 뷰포인트 생성

```bash
uv run scripts/pipeline/generate_viewpoints.py \
    --object sample \
    --material-rgb "170,163,158" \
    --cluster-method coacd+dbscan \
    --normal-weight 0.05 \
    --coacd-threshold 0.25 \
    --eps 25
```

`data/sample/viewpoint/124/viewpoints_coacd+dbscan.h5` 생성.

### 2. IK 선택 + 궤적 생성

```bash
uv run scripts/pipeline/select_ik_dp.py \
    --object sample \
    --num-viewpoints 124 \
    --viewpoints data/sample/viewpoint/124/viewpoints_coacd+dbscan.h5
```

출력:
- `data/sample/trajectory/124/trajectory_dp.csv` — joint angles + EE pose
- `data/sample/trajectory/124/trajectory_dp.html` — 정적 3D 시각화
- `data/sample/trajectory/124/trajectory_dp_anim.html` — 애니메이션 시각화

주요 옵션:
- `--viewpoints <path>` — h5 파일 직접 지정
- `--num-seeds 100` — viewpoint당 IK seed 수
- `--dbscan-eps 0.3` — DBSCAN eps (radians)
- `--reconfig-threshold 29.0` — reconfig 판정 임계값 (degrees)
- `--spacing 0.05` — uniform resample 간격 (radians)
