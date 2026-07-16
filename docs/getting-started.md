# 시작 가이드

뷰포인트와 궤적 작업은 로컬 `uv` 환경을 기본으로 한다. Isaac Sim, ROS2, MoveIt이 필요한 작업만 Docker 환경을 사용한다.

## 1. 로컬 환경 준비

필수 조건은 Python 3.12와 NVIDIA GPU다. 프로젝트 루트에서 다음을 실행한다.

```bash
uv sync
```

GLNS를 사용할 경우 Julia 환경을 한 번 준비한다.

```bash
julia --project=scripts/julia/glns -e 'using Pkg; Pkg.instantiate()'
```

## 2. 앱 선택

```bash
# 뷰포인트 생성: http://localhost:8080
uv run --no-sync scripts/apps/viewpoint_studio.py

# 궤적 생성: http://localhost:8081
uv run --no-sync scripts/apps/trajectory_studio.py
```

Isaac Pipeline도 프로젝트 루트에서 `uv`로 실행한다.

```bash
uv run --no-sync scripts/apps/isaac_pipeline.py
```

## 3. 기본 데이터 흐름

```text
mesh/source.obj
  → viewpoints_*.h5
  → glns_result*.h5 또는 DP 결과
  → trajectory*.csv / trajectory*.npz
  → Isaac preview 또는 ROS2 실행
```

새 물체로 시작한다면 [자산 준비](guides/prepare-object-assets.md)를 먼저 진행한다. 환경별 상세 조건은 [환경 설정](guides/environment-setup.md)을 참고한다.
