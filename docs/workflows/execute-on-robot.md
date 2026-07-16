# 로봇에서 실행하기

검증된 CSV를 Isaac UR20, mock hardware, URSim 또는 실제 로봇에서 실행한다. 먼저 [시뮬레이션 미리보기](simulate-and-preview.md)를 완료한다.

## 안전한 기본 경로

1. `sim` 또는 mock hardware로 컨트롤러 연결을 확인한다.
2. CSV의 시작 자세와 작업 공간을 확인한다.
3. `Move to Scan Start`로 시작점 접근을 별도로 검증한다.
4. `Execute Selected CSV`로 검사 구간을 실행한다.
5. 종료 후 `Return to HOME`을 사용한다.

실행 중에는 Run/Pipeline 모드를 바꾸지 않는다. `Cancel Execution`이 보이고 컨트롤러가 활성 상태인지 확인한다.

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
