# 시작 가이드

모든 작업은 `ros-jazzy` 컨테이너 하나에서 실행한다. 컨테이너 빌드·설치는 [환경 설정](guides/environment-setup.md)을 참고한다.

## 1. 환경 준비

프로젝트 루트에서 아래 4단계를 순서대로 진행한다. 1~2는 컨테이너를 띄우는 과정이고,
3은 머신당 최초 1회만 필요하다.

```bash
# 1) 컨테이너가 호스트 X 서버에 접근하도록 허용 (Isaac GUI 창)
xhost +local:root

# 2) 시스템 이미지 빌드 + 컨테이너 시작
docker compose -f docker/compose.yaml up -d --build

# 3) venv·ROS overlay·Julia depot 설치 (최초 1회)
docker exec -it ros-jazzy bash /workspace/docker/install_env.sh

# 4) 설치 결과 점검
docker exec -it ros-jazzy bash /workspace/docker/verify_env.sh
```

## 2. 앱 선택

컨테이너 셸(`docker exec -it ros-jazzy bash`)에서 실행한다.

```bash
# 뷰포인트 생성: http://localhost:8080
uv run scripts/apps/viewpoint_studio.py

# 궤적 생성: http://localhost:8081
uv run scripts/apps/trajectory_studio.py

# Isaac Pipeline
uv run scripts/apps/isaac_pipeline.py
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
