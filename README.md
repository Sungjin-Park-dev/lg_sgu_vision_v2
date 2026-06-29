# lg_sgu_vision_v2

UR20 로봇을 이용한 비전 검사 궤적 생성 시스템. cuRobo(IK/충돌검사) 기반.

## 디렉토리 구조

```
scripts/
  apps/              ★ 사용자 직접 실행 GUI
    viewpoint_studio.py      뷰포인트 생성/튜닝/시각화 (viser)
    isaac_pipeline.py        Isaac Sim 물체 선택+이동+궤적 생성/preview/publish
  core/              headless 엔진 (apps가 라이브러리/서브프로세스로 사용)
    generate_viewpoints.py   뷰포인트 생성 + 클러스터링 + GTSP 순서
    plan_trajectory.py       DBSCAN+DP 기반 IK 해 선택 + MotionGen transit
    publish_trajectory.py    ROS2 궤적 전송
  prep/              mesh 전처리 (normalize_mesh, generate_normals)
  isaac/             Isaac Sim 씬/런타임 (scene, launch_sim, load_workcell, usd/)
  robot/             실로봇 ROS2 유틸 (move_to_start, publish_workcell_markers)
  common/            config, math_utils, viewpoint_viz
  tools/             보조 뷰어 (view_meshes)
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
uv run scripts/core/generate_viewpoints.py \
    --object sample \
    --material-rgb "0,255,0" \
    --cluster-method coacd+dbscan \
    --normal-weight 0.05 \
    --coacd-threshold 0.25 \
    --eps 25
```

`data/sample/viewpoint/124/viewpoints_coacd+dbscan.h5` 생성.

> **배치 모드**: 기본 `grid`(PCA 평면 투영)는 평평한 판재에 적합하지만 곡면·측벽엔 점이 거의
> 안 생긴다. 곡면/입체물은 `--sampling-mode surface --ordering-mode graph`(표면 직접 균일
> 샘플링 + 탄젠트 그래프 순서)를 쓴다. 자세한 내용: [docs/generate_viewpoints.md](docs/generate_viewpoints.md#배치-모드---sampling-mode).

#### 뷰포인트 스튜디오 (viser, 선택)

브라우저에서 물체를 골라 **파라미터 튜닝으로 viewpoint를 실시간 재생성**하거나, 기존
`viewpoints*.h5`를 불러와 확인 (메시·클러스터·경로·CoACD 파트 + 경로 순서 재생, Save):

```bash
uv run scripts/apps/viewpoint_studio.py --object curved_structure
# http://localhost:8080 접속 → Generate / Save
```

자세한 사용법: [docs/viewpoint_studio.md](docs/viewpoint_studio.md).

### 2. IK 선택 + 궤적 생성

```bash
uv run scripts/core/plan_trajectory.py \
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

### 2-1. Delaunay + GLNS 경로 실험 (선택)

기존 trajectory pipeline과 별도로 Delaunay 인접 viewpoint만 허용하면서 viewpoint 순서와
IK branch를 GLNS로 함께 선택할 수 있다.

```bash
julia --project=scripts/julia/glns -e 'using Pkg; Pkg.instantiate()'

uv run --no-sync scripts/core/solve_glns_path.py \
  --object sample \
  --viewpoints data/sample/viewpoint/74/viewpoints_coacd+agglomerative.h5

uv run --no-sync scripts/apps/glns_inspector.py \
  --result data/sample/ik/74/glns_result_YYYYMMDD_HHMMSS.h5
```

자세한 모델과 결과 형식: [docs/glns_path.md](docs/glns_path.md).

### 3. Isaac Sim 시뮬레이션 (선택)

```bash
# (1회성) URDF → USD 변환
uv run --no-sync python -m urdf_usd_converter \
    --package "ur_description=$(realpath ur20_description)" \
    ur20_description/ur20_with_camera.urdf ur20_description/

# 시뮬레이션 + ROS2 bridge
OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python \
    scripts/isaac/scene.py --object sample
```

자세한 절차·트러블슈팅: [docs/running.md](docs/running.md#isaac-sim-시뮬레이션).
