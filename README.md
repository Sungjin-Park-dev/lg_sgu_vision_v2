# lg_sgu_vision_v2

UR20 로봇을 이용한 비전 검사 궤적 생성 시스템. cuRobo(IK/충돌검사) 기반.

## 디렉토리 구조

사용자가 직접 실행하는 것은 `scripts/apps/`의 GUI 3개다. 나머지 폴더
(`core`, `prep`, `isaac`, `moveit`, `robot`, `common`, `tools`, `julia`)는 이들이
라이브러리/서브프로세스로 쓰는 엔진·유틸이다.

```
scripts/apps/          ★ 사용자 직접 실행 GUI
  viewpoint_studio.py    뷰포인트 생성/튜닝/시각화 (viser)
  trajectory_studio.py   Isaac 없이 브라우저에서 배치+라이브 IK+궤적(DP|GLNS) 생성/재생 (viser)
  isaac_pipeline.py      Isaac Sim 물체 선택+기즈모 이동+궤적 생성/preview/publish
```

기타 최상위: `workcell/`(로봇·환경 URDF/USD/config), `data/{object}/`(mesh·viewpoint·궤적),
`docs/`(상세 문서·로그), `curobo/`(cuRobo 라이브러리, 별도 clone).

## 환경 설정

NVIDIA GPU + CUDA 12.x toolkit(`nvcc`), Python 3.12 필요.

```bash
# 1. uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Python 의존성 (Isaac Sim, torch, cuRobo 포함)
uv sync

# 3. Julia + GLNS (trajectory_studio 의 GLNS 경로용)
curl -fsSL https://install.julialang.org | sh
julia --project=scripts/julia/glns -e 'using Pkg; Pkg.instantiate()'
```

## 실행 방법

진입점은 `scripts/apps/` 의 GUI 4개다. 이들이 내부적으로 `core/` 엔진을 서브프로세스로
호출하므로, core CLI 를 직접 부를 필요는 없다 (core 세부는 각 항목의 문서 링크 참고).

viser 기반 2개(`viewpoint_studio`, `trajectory_studio`)는 브라우저 도구다 — 실행하면 뜨는
`http://localhost:<port>` 에 접속한다. 이미 sync 된 환경에서는 cuRobo 재빌드를 피하려고
`uv run --no-sync` 를 쓴다.

### 1. 뷰포인트 생성/튜닝 — `viewpoint_studio.py`

브라우저에서 물체를 골라 파라미터 튜닝으로 viewpoint 를 실시간 재생성하거나, 기존
`viewpoints*.h5` 를 불러와 확인·수정한다 (메시·클러스터·경로·CoACD 파트 시각화 + 순서
재생 + Save).

```bash
uv run --no-sync scripts/apps/viewpoint_studio.py --object curved_structure
# http://localhost:8080 → Generate / Save
```

출력: `data/{object}/viewpoint/{N}/viewpoints_*.h5`.
자세히: [docs/viewpoint_studio.md](docs/viewpoint_studio.md),
[docs/generate_viewpoints.md](docs/generate_viewpoints.md).

### 2. 궤적 생성/미리보기 (headless) — `trajectory_studio.py`

Isaac 없이 브라우저에서: 물체를 gizmo 로 배치 → 라이브 IK 도달성 확인 → 궤적 생성
(DP 또는 GLNS) → 재생. `plan_trajectory` / `solve_glns_path` 엔진을 감싼다.

```bash
uv run --no-sync scripts/apps/trajectory_studio.py --object sample
# http://localhost:8082

# 이미 만들어진 GLNS 결과 h5 를 바로 열기:
uv run --no-sync scripts/apps/trajectory_studio.py \
    --result data/sample/ik/74/glns_result_YYYYMMDD_HHMMSS.h5
```

GLNS 경로를 쓰려면 최초 1회 Julia 패키지를 설치한다:

```bash
julia --project=scripts/julia/glns -e 'using Pkg; Pkg.instantiate()'
```

출력: `data/{object}/trajectory/{N}/trajectory_dp.{csv,html}` 등.
자세히: [docs/plan_trajectory.md](docs/plan_trajectory.md),
[docs/glns_path.md](docs/glns_path.md).

### 3. Isaac Sim 통합 파이프라인 — `isaac_pipeline.py`

Isaac Sim 을 띄워 물체 선택 + 뷰포트 gizmo 이동 + 궤적 생성/ghost preview + 실행까지
한 창에서 처리한다. 실로봇 전송까지 포함하는 전체 워크플로.

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync scripts/apps/isaac_pipeline.py \
    --object sample --mode sim
```

- `--mode {sim,real}` (기본 `sim`) — 궤적을 Isaac UR20(sim) 또는 실로봇(real)으로 전송
- `--pipeline-mode {inspection,moveit}` (기본 `inspection`) — 궤적 생성 백엔드

자세히: [docs/running.md](docs/running.md), [docs/moveit_inspection_mode.md](docs/moveit_inspection_mode.md),
[docs/run_2x2_modes.md](docs/run_2x2_modes.md).
