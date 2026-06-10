# lg_sgu_vision_v2

UR20 로봇을 이용한 비전 검사 궤적 생성 시스템. cuRobo(IK/충돌검사) 기반.

## 디렉토리 구조

```
scripts/
  pipeline/          핵심 파이프라인
    generate_viewpoints.py   뷰포인트 생성 + 클러스터링 + GTSP 순서
    plan_trajectory.py       DBSCAN+DP 기반 IK 해 선택 + MotionGen transit
    publish_trajectory.py    ROS2 궤적 전송
  prep/              mesh 전처리 (normalize_mesh, generate_normals)
  viser/             viser 웹 프론트엔드 (view_meshes)
  common/            config, math_utils, viewpoint_viz
  ros2/              ROS2 유틸 (move_to_start, publish_workcell_markers)
  isaac/             Isaac Sim 전용 (usd/ 하위: build_ghost_usd, inspect_usd)
  prev/              이전 버전 스크립트
data/{object}/       mesh/ viewpoint/ trajectory/ (gitignore)
```

## 환경 설정

CUDA 12 로 통일된 환경 (Isaac Sim 6.0.0 이 cu12 빌드만 제공하기 때문 — 자세한 배경은
[docs/curobo_v0.8_migration.md](docs/curobo_v0.8_migration.md) 참고).

```bash
# 1. Python 의존성 설치 (Isaac Sim, torch+cu128, NVIDIA libs 포함)
uv sync

# 2. cuRobo 클론 및 설치 (cu12-torch extra)
git clone https://github.com/NVlabs/curobo.git
uv pip install "./curobo[cu12-torch]"
```

> **주의**: cuRobo 는 path-install 이라 `uv.lock` 에 들어가지 않습니다.
> `uv sync` 를 다시 돌리면 cuRobo 가 제거되니, 위 `uv pip install` 한 줄을 다시
> 실행해야 합니다. 매번 sync 를 건너뛰려면 `uv run --no-sync ...` 사용.

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
uv run scripts/pipeline/plan_trajectory.py \
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

### 3. Isaac Sim 시뮬레이션 (선택)

```bash
# (1회성) URDF → USD 변환
uv run --no-sync python -m urdf_usd_converter \
    --package "ur_description=$(realpath ur20_description)" \
    ur20_description/ur20_with_camera.urdf ur20_description/

# 시뮬레이션 + ROS2 bridge
OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python \
    scripts/isaac/joint_control.py --object sample
```

자세한 절차·트러블슈팅: [docs/running.md](docs/running.md#isaac-sim-시뮬레이션).
