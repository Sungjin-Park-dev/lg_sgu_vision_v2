# lg_sgu_vision_v2

UR20 로봇의 비전 검사 지점과 충돌 회피 궤적을 만들고, 시뮬레이션 또는 로봇에서 실행하는 도구다.

## 주요 기능

| 기능 | 실행 파일 |
|---|---|
| 뷰포인트 생성·수정 | `scripts/apps/viewpoint_studio.py` |
| IK 확인 및 DP·GLNS 궤적 생성 | `scripts/apps/trajectory_studio.py` |
| Isaac Sim 미리보기와 로봇 실행 | `scripts/apps/isaac_pipeline.py` |

## 빠른 시작

Python 3.12 환경에서 의존성을 설치한다.

```bash
uv sync
```

첫 번째 앱을 실행하고 브라우저에서 `http://localhost:8080`에 접속한다.

```bash
uv run --no-sync scripts/apps/viewpoint_studio.py --object sample
```

전체 설치 방법, 기능별 작업 순서와 기술 참고자료는 [문서 홈](docs/README.md)에서 찾을 수 있다.
