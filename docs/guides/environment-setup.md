# 환경 설정

## 로컬 환경

Viewpoint Studio와 Trajectory Studio의 기본 환경이다.

- Python 3.12
- NVIDIA GPU와 CUDA 12.x
- `uv`

```bash
uv sync
uv run --no-sync scripts/apps/viewpoint_studio.py --help
uv run --no-sync scripts/apps/trajectory_studio.py --help
```

GLNS를 사용한다면 Julia 패키지를 준비한다.

```bash
julia --project=scripts/julia/glns -e 'using Pkg; Pkg.instantiate()'
```

## Isaac·ROS 환경

Isaac Sim, ROS2 Jazzy, MoveIt은 GPU와 host network를 사용하는 `ros-jazzy` 컨테이너에서 실행한다. 호스트에는 NVIDIA 드라이버, Docker와 NVIDIA Container Toolkit이 필요하다.

```bash
# GPU 전달 확인
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# 최초 컨테이너 생성: 프로젝트 루트의 호스트 셸에서 실행
xhost +local:docker
docker run -d -it \
  --name ros-jazzy \
  --gpus all --network host --ipc host \
  -e DISPLAY="$DISPLAY" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$(pwd)":/workspace \
  -w /workspace \
  osrf/ros:jazzy-desktop-full bash

# 기존 컨테이너 시작과 진입
docker start ros-jazzy
docker exec -it ros-jazzy bash
```

컨테이너 안에서 ROS2 Jazzy와 UR driver 패키지를 설치하고 `uv sync`를 실행한다. 프로젝트는 `/workspace`에 마운트되어 있어야 한다.

```bash
apt update && apt upgrade -y
apt install -y curl git build-essential ros-jazzy-control-msgs \
  ros-jazzy-ur-robot-driver ros-jazzy-ros2controlcli
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv sync
```

## 셸 분리 원칙

| 셸 | 환경 | 용도 |
|---|---|---|
| Isaac | `.venv` 활성, 시스템 ROS source 안 함 | Isaac Pipeline |
| ROS | `/opt/ros/jazzy/setup.bash` source | driver, MoveIt, RViz |

Isaac Sim 번들 ROS와 시스템 ROS를 같은 셸에서 함께 source하지 않는다. FastDDS/FastCDR 라이브러리 충돌이 날 수 있다.
