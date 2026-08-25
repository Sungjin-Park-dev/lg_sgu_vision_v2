# 로봇에서 실행하기

검증된 CSV를 Isaac UR20, mock hardware, URSim 또는 실제 로봇에서 실행한다. 먼저 [시뮬레이션 미리보기](simulate-and-preview.md)를 완료한다.

## 안전한 기본 경로

1. `sim` 또는 mock hardware로 컨트롤러 연결을 확인한다.
2. CSV의 시작 자세와 작업 공간을 확인한다.
3. `Plan to Start` → 고스트로 접근 경로를 확인 → `Move to Start`.
4. `Execute Scan`으로 검사 구간을 실행한다.
5. 종료 후 `Plan to HOME` → 확인 → `Move to HOME`.

실행 중에는 Run/Pipeline 모드를 바꾸지 않는다(작업 중에는 자동으로 잠긴다). `Cancel`이 활성 상태이고 컨트롤러가 살아 있는지 확인한다.

## 버튼

이동은 leg 당 두 버튼이다 — **계획하고 눈으로 본 뒤에야 움직인다**. `Move` 는 그 leg 의 계획이 대기 중일 때만 활성화된다.

| 버튼 | 동작 |
|---|---|
| `Plan to Start` | 현재 자세 → CSV 첫 행의 충돌-free 경로를 계획해 **고스트에서 바로 재생한다**(실행 안 함). 스캔·틸트 CSV 공용 |
| `Move to Start` | 그 계획을 실행한다 |
| `Plan to HOME` | 현재 자세 → `ROBOT_START_STATE`. 〃 |
| `Move to HOME` | 그 계획을 실행한다 |
| `Execute Scan` | CSV path 칸의 궤적을 실행한다 |
| `Cancel` | 계획 단계와 실행 단계 모두 중단 |

계획이 나오면 고스트가 **자동으로 한 번 재생한다.** 다시 보려면 `Preview in Simulation` 의 `Play`/슬라이더를 쓴다(스캔 궤적은 길어서 자동 재생하지 않는다 — 지금처럼 로드만 된다).

### 두 패널이 서로 다른 파일을 가리킬 때

계획은 고스트에만 올라간다 — **CSV path 칸은 스캔 궤적을 가리킨 채로 남는다.** 그 칸 하나를 Preview 와 Execute 가 공유하고, 그것이 곧 `Execute Scan` 의 대상이자 `Plan to Start` 의 목표(첫 행)이기 때문이다. 계획 출력으로 덮어쓰면 `Execute Scan` 이 스캔 대신 이동을 실행하고, `Plan to Start` 가 자기 산출물을 목표로 삼는다.

그래서 계획을 재생하는 동안에는 **바가 트는 파일과 CSV path 칸이 다르다.** 숨기지 않는다 — Preview 바의 상태 줄이 지금 재생 중인 파일 이름을 적는다:

```
t=2.14s / 6.31s  (wp 140/411)  playing: home_move_approach.csv    ← 바가 트는 것
CSV path: .../trajectory/74/trajectory.csv                        ← Execute Scan 대상
```

`Load & Preview` 로 스캔을 다시 올리면 `playing:` 도 따라 바뀌므로 이 줄은 언제나 사실이다. 버튼 아래 한 줄은 지금 무엇이 실행 대기 중인지 따로 말한다(`planned: Move to Start - 412 wp, 6.31 s`).

이동은 **현재 자세에서** 계획하므로 로봇이 어디 있든 전 구간이 검사된다. 계획에 수 초 걸리며, 물체가 스테이지에 로드돼 있어야 한다.

**경로를 못 찾으면 계획이 남지 않는다**(`plan exit code = 2`). 물체를 옮기거나 자세를 바꿔 다시 시도하고, 임의 이동이 필요하면 Pipeline mode = MoveIt의 RViz Plan & Execute를 쓴다.

### 계획이 낡으면 거부한다

`Move` 는 실행 직전에 계획 당시의 입력을 다시 확인한다 — 로봇이 아직 계획의 시작점에 있는지, 목표(스캔 CSV 첫 행)와 물체 pose 가 그대로인지. 하나라도 어긋나면 **실행하지 않고** 이유를 로그에 찍은 뒤 계획을 버린다(`the robot moved since planning (max |Δ| = 42.31 deg) - re-plan from where it is now.`). 그대로 실행하면 실행기가 현재 자세와 CSV 첫 행 사이에 **계획되지 않은 직선**을 덧붙여, 미리 본 것과 다른 길을 가기 때문이다.

## 모드별 명령 소스

| Pipeline | 명령 소스 |
|---|---|
| Inspection | Isaac Pipeline의 `Execute Trajectory` |
| MoveIt | RViz의 Plan & Execute |

Run 모드와 ROS 스택은 반드시 같은 대상을 가리켜야 한다. 조합과 시작 명령은 [Isaac 모드](../guides/isaac-modes.md)를 참고한다.

## 실제 로봇 전 확인

- 로봇 주변과 케이블 이동 범위가 비어 있는지 확인한다.
- 비상 정지와 감속 수단을 준비한다.
- `/joint_states`와 trajectory controller 상태를 확인한다.
- 낮은 속도와 안전한 시작 자세로 별도 검증한다.

이 문서는 실제 로봇을 즉시 움직이는 복사·실행 명령을 제공하지 않는다. 현장 네트워크와 안전 설정을 확인한 뒤 `real` 모드를 사용한다.
