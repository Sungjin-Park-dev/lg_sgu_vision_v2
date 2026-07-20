# 시뮬레이션과 미리보기

Isaac Pipeline에서 물체를 배치하고 검사 궤적을 생성·재생한다. 이 문서는 로봇 명령을 보내지 않는 `sim` 모드를 기준으로 한다.

## 실행

Isaac 환경이 준비된 컨테이너에서 실행한다.

```bash
uv run scripts/apps/isaac_pipeline.py
```

## 작업 순서

1. `Load Object`에서 물체를 불러오고 viewport gizmo로 배치한다.
2. `Generate Trajectory`에서 HDF5를 선택한다.
3. `Show Viewpoints`와 `Check IK Reachability`로 배치 상태를 확인한다.
4. `Generate Scan Motion`으로 GLNS 궤적을 만든다.
5. `Preview in Simulation`에서 CSV를 열고 `Load & Preview`를 누른다.
6. `Play`, 시간 슬라이더, 충돌 구와 FOV 평면으로 동작을 확인한다.

Ghost preview는 실제 UR20 articulation과 ROS2 컨트롤러를 움직이지 않는다. `Execute Trajectory`는 별도 동작이므로 미리보기 확인 전에는 사용하지 않는다.

실행 모드 차이는 [Isaac 모드](../guides/isaac-modes.md), 실행 문제가 있으면 [문제 해결](../guides/troubleshooting.md)을 참고한다.
