# 궤적 계획하기

Trajectory Studio에서 물체 위치를 조절하고 IK 도달성을 확인한 뒤 DP 또는 GLNS 궤적을 생성한다. Isaac Sim 없이 브라우저에서 결과를 재생할 수 있다.

## 실행

```bash
uv run scripts/apps/trajectory_studio.py
```

브라우저에서 `http://localhost:8081`에 접속한다.

## 작업 순서

1. `Load Object & Viewpoints`에서 물체와 `Viewpoints (h5)`를 고르고 `Load viewpoints`를 누른다.
2. gizmo로 물체를 옮긴 뒤 `Apply pose → recompute IK`로 도달성을 확인한다.
3. `Generate Trajectory`의 `Backend`에서 DP 또는 GLNS를 선택한다.
4. `1. Solve GLNS / Generate DP`를 실행한다.
5. GLNS는 이어서 `2. Plan scan motion (no HOME)`을 실행한다.
6. `Result / Playback`에서 결과를 열고 이산 경로와 dense trajectory를 확인한다.

## 백엔드 선택

| 백엔드 | 용도 |
|---|---|
| DP | 저장된 뷰포인트 순서를 유지하며 IK branch를 선택 |
| GLNS | Delaunay 그래프 안에서 뷰포인트 순서와 IK branch를 함께 최적화 |

## 카메라 스펙 (읽기 전용)

`Camera` 줄은 불러온 h5의 FOV와 working distance를 보여준다. 이 값이 뷰포인트 위치를 이미
결정했으므로 여기서는 바꿀 수 없다 — 다른 스펙으로 계획하려면
[뷰포인트 만들기](create-viewpoints.md)에서 다시 생성한다. GLNS 결과를 불러온 경우 solve
당시 쓰인 working distance를 보여준다.

궤적은 **화면의 현재 물체 위치**를 기준으로 생성된다. 배치는 도달 가능한 뷰포인트 수를 크게 좌우한다.

결과는 `data/{object}/ik/{N}/`과 `data/{object}/trajectory/{N}/`에 저장된다. GLNS에는 Julia 환경이 필요하며, 실제 로봇 실행 전에는 [Isaac 미리보기](simulate-and-preview.md)를 권장한다.
