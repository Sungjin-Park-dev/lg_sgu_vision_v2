# 시뮬레이션과 미리보기

Isaac Pipeline에서 물체를 배치하고 검사 궤적을 생성·재생한다. 이 문서는 로봇 명령을 보내지 않는 `sim` 모드를 기준으로 한다.

## 실행

Isaac 환경이 준비된 컨테이너에서 실행한다.

```bash
uv run scripts/apps/isaac_pipeline.py
```

## 작업 순서

1. `Load Object & Viewpoints`에서 물체를 불러오고 viewport gizmo로 배치한다.
2. 같은 패널에서 `Viewpoints (h5)`를 고르고 `Show Viewpoints`로 배치 상태를 확인한다.
3. `Generate Trajectory`의 `Check IK Reachability`로 도달성을 확인한다.
4. `Generate Scan Motion`으로 궤적을 만든다.
5. `Preview in Simulation`에서 CSV를 열고 `Load & Preview`를 누른다.
6. `Play`, 시간 슬라이더, 충돌 구와 FOV 평면으로 동작을 확인한다.

Ghost preview는 실제 UR20 articulation과 ROS2 컨트롤러를 움직이지 않는다. `Execute Trajectory`는 별도 동작이므로 미리보기 확인 전에는 사용하지 않는다.

## 패널 구성

| 패널 | 역할 |
|---|---|
| `Load Object & Viewpoints` | 물체, 그 물체의 viewpoint h5, 카메라 스펙 |
| `Generate Trajectory` | IK 확인과 궤적 생성 |
| `Preview in Simulation` | ghost 로봇으로 재생 |
| `Execute Trajectory` | 실제 로봇 실행 ([로봇에서 실행하기](execute-on-robot.md)) |

## 카메라

`FOV W / FOV H / WD` 입력칸은 두 검사 카메라가 공유한다. h5를 고르거나 `Show Viewpoints`를
누르면 그 파일의 스펙으로 맞춰지고, `Reset`은 config 기본값으로 되돌린다.

| 카메라 | 위치 |
|---|---|
| `InspectionCameraPreview` | ghost 로봇 (Preview 패널) |
| `InspectionCamera` | 실제 로봇 (Execute 패널) |

`Show FOV`, `Show Camera Range`는 서로 다른 로봇 아래에 그리므로 패널별로 있다. 스펙 값은 하나다.

`InspectionCamera`를 viewport 카메라로 고르면 검사 중 실제로 보는 화면을 확인할 수 있다.
화각·초점거리·near clip과 ROS 퍼블리시 해상도는 현재 스펙에서 자동으로 맞춰진다 —
FOV를 바꾸면 로그에 `[cam] render product -> WxH`가 찍힌다.

실행 모드 차이는 [Isaac 모드](../guides/isaac-modes.md), 실행 문제가 있으면 [문제 해결](../guides/troubleshooting.md)을 참고한다.
