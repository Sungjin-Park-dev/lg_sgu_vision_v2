# 궤적 계획하기

Trajectory Studio에서 물체 위치를 조절하고 IK 도달성을 확인한 뒤 DP 또는 GLNS 궤적을 생성한다. Isaac Sim 없이 브라우저에서 결과를 재생할 수 있다.

## 실행

```bash
uv run --no-sync scripts/apps/trajectory_studio.py --object sample
```

브라우저에서 `http://localhost:8081`에 접속한다. 기존 GLNS 결과는 `--result PATH`로 바로 열 수 있다.

## 작업 순서

1. `Scene`에서 물체를 선택하고 `Load viewpoints`를 누른다.
2. gizmo로 물체를 옮긴 뒤 `Apply pose → recompute IK`로 도달성을 확인한다.
3. `Backend`에서 DP 또는 GLNS를 선택한다.
4. `1. Solve GLNS / Generate DP`를 실행한다.
5. GLNS는 이어서 `2. Plan scan motion (no HOME)`을 실행한다.
6. `Result / Playback`에서 이산 경로와 dense trajectory를 확인한다.

## 백엔드 선택

| 백엔드 | 용도 |
|---|---|
| DP | 저장된 뷰포인트 순서를 유지하며 IK branch를 선택 |
| GLNS | Delaunay 그래프 안에서 뷰포인트 순서와 IK branch를 함께 최적화 |

결과는 `data/{object}/ik/{N}/`과 `data/{object}/trajectory/{N}/`에 저장된다. GLNS에는 Julia 환경이 필요하며, 실제 로봇 실행 전에는 [Isaac 미리보기](simulate-and-preview.md)를 권장한다.
