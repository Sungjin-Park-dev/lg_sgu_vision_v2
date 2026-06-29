# Delaunay-constrained GLNS experiment

`solve_glns_path.py`는 기존 DP trajectory pipeline과 독립적인 실험 도구다.
원본 viewpoint HDF5의 순서를 변경하지 않고, Delaunay 연결 성분마다 viewpoint 순서와
collision-free IK branch를 GLNS로 함께 선택한다.

## 모델

- GTSP set: viewpoint
- GTSP vertex: 해당 viewpoint의 collision-free IK 대표해
- 허용 전이: `viewpoints/adjacency/edges`에 포함된 viewpoint 쌍만
- 목적함수: reconfiguration 개수 최소화 → joint L2 거리 최소화
- dummy singleton set: GLNS cycle을 open path로 변환
- 연결 성분은 서로 잇지 않고 별도 run으로 출력

reachable viewpoint 제거 후 induced graph를 다시 계산한다. 각 성분에 Delaunay-only
Hamiltonian open path가 없으면 `infeasible`로 기록하며 비-Delaunay edge로 대체하지 않는다.

## 준비

viewpoint HDF5를 현재 generator로 다시 생성해 `viewpoints/adjacency`가 있어야 한다.

```bash
uv run scripts/core/generate_viewpoints.py \
  --object sample --material-rgb "0,255,0" \
  --sampling-mode surface --surface-spacing 25 \
  --cluster-method coacd+agglomerative --coacd-threshold 0.2 \
  --max-span 100 --ordering-mode lawnmower
```

Julia GLNS 환경은 최초 한 번 설치한다.

```bash
julia --project=scripts/julia/glns -e 'using Pkg; Pkg.instantiate()'
```

## 실행

```bash
uv run --no-sync scripts/core/solve_glns_path.py \
  --object sample \
  --viewpoints data/sample/viewpoint/74/viewpoints_coacd+agglomerative.h5
```

기본값은 IK 100 seeds, reconfiguration threshold 29°, GLNS `fast`, 성분당 30초다.
결과는 기존 viewpoint 파일과 별개로 다음 위치에 저장된다.

```text
data/{object}/ik/{N}/glns_result_{timestamp}.h5
```

GLNS가 선택한 순서와 joints 외에도 unreachable mask, induced Delaunay graph,
성분별 solver 상태와 edge L∞/L2가 포함된다. 이 파일은 trajectory가 아니며 motion planner나
resampling을 거치지 않았다.

## Viser 확인

```bash
uv run --no-sync scripts/apps/glns_inspector.py \
  --result data/sample/ik/74/glns_result_YYYYMMDD_HHMMSS.h5
```

기본 포트는 8082다. 성분을 선택해 GLNS 순서를 재생할 수 있고 다음 레이어를 토글한다.

- induced Delaunay graph
- reachable/dropped viewpoints
- continuous path edge(초록)
- reconfiguration edge(빨강)
- GLNS가 선택한 joint configuration의 UR20 자세

`infeasible` 또는 `solver_failed` 성분도 목록에 남으며 실패 원인을 표시한다.

### Dense trajectory 재생 (실제 motion)

기본 재생은 GLNS가 고른 **이산 자세**를 viewpoint 단위로 순간이동시킬 뿐, viewpoint 사이의
실제 이동(transit via-roll 우회 등)은 보여주지 않는다. **Component 폴더의 `Dense trajectory (CSV)`
토글**을 켜면, `verify_glns_trajectory.py`가 같은 디렉토리에 저장한
`glns_trajectory_comp{cid}.npz`(검증된 dense 궤적)를 읽어 실제 motion을 재생한다.

- step slider가 viewpoint 대신 **dense waypoint 단위**로 바뀐다(예: 성분 0 → 836 waypoint).
- transit 구간은 로봇 본체·경로가 **빨강**, scan 구간은 **초록**으로 칠해진다 → via-roll이 물체를
  스치지 않고 어떻게 도는지 눈으로 확인 가능.
- npz가 없는(검증 미실행 또는 충돌 FAIL) 성분은 토글을 켜도 기존 이산 자세 재생으로 폴백한다.

먼저 `verify_glns_trajectory.py`를 돌려 npz를 만들어 두어야 한다.

## 충돌 고려 검증 (collision-aware verification)

GLNS는 각 viewpoint의 **정적 자세 충돌**만 검사하고 viewpoint 사이의 **이동(motion)** 은
계획·충돌검사하지 않는다. `verify_glns_trajectory.py`는 GLNS 결과를 받아 **성분마다 독립적으로**
GLNS가 고른 joint 순서를 `plan_trajectory.py`의 Phase 4-6
(reconfig transit 계획 → densify 충돌검증 → uniform resample → FK/시간 → CSV)에 그대로
흘려보내, "충돌을 고려하면 이 경로가 실제로 실행 가능한가"를 확인한다.

`plan_trajectory`는 일체 수정하지 않고 라이브러리로 재사용한다(`solve_glns_path`와 동일 패턴).
두 도구가 같은 collision world / robot config / wrist_3 lock 값을 쓰므로 GLNS에서 충돌-free였던
자세는 여기서도 충돌-free다 — 검증 대상은 **오직 자세 사이의 이동**이다. 결과에 박제된 물체 배치
(`object_position`/`object_quat_wxyz`)를 config에 주입해 GLNS IK가 풀린 바로 그 world를 재현한다.

```bash
uv run --no-sync scripts/core/verify_glns_trajectory.py \
  --result data/sample/ik/74/glns_result_YYYYMMDD_HHMMSS.h5
```

성분별 trajectory CSV는 결과 h5와 같은 디렉토리에 `glns_trajectory_comp{cid}.csv`로 저장된다
(DP의 `trajectory_*.csv`와 구분). reconfig 경계는 `selected`의 L∞로 재산출해 GLNS의
`is_reconfiguration`과 교차검증한다. scan 구간 resample 간격은 `--spacing`(기본 0.01 m).

성분별로 다음을 보고한다 — transit `k/total OK`, coverage(`covered/M`), 드롭된 viewpoint
(원본 인덱스), 충돌 dense waypoint 수, scan+transit 시간. **합격 기준 = 모든 solved 성분이
`collisions=0`.** 충돌이 검출된 성분은 CSV를 쓰지 않고 `FAIL`로 보고하며, 다른 성분 검증은 계속된다.

- **드롭된 viewpoint** = 연속 scan edge가 densify 충돌하거나 reconfig transit 계획이 실패해
  해당 viewpoint를 건너뛰었다는 뜻 → GLNS 경로가 충돌-aware 이동에서 완전히 보존되지 못함.
- **충돌-free + 0 드롭** = GLNS 경로가 이동까지 그대로 실행 가능 → 이 방식 채택 가능.
