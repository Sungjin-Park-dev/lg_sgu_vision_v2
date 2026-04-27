# Docker 환경 셋업

UR20 비전 검사 파이프라인을 단일 Docker 컨테이너에서 실행하기 위한 환경 구성.

## 구조 개요

호스트와 컨테이너의 책임 분리:

| 영역 | 위치 | 역할 |
|---|---|---|
| **호스트** | Ubuntu (NVIDIA 드라이버) | Docker 실행, 코드 편집, git |
| **컨테이너** (`ros-jazzy`) | `osrf/ros:jazzy-desktop-full` 기반 | ROS2 + ur_robot_driver + Isaac Sim + cuRobo 전부 |
| **URSim 컨테이너** | `universalrobots/ursim_e-series` | UR20 시뮬레이터 (필요 시) |

호스트는 NVIDIA 드라이버 + Docker만 깔린 상태로 깨끗하게 유지. **모든 파이썬/ROS 작업은 ros-jazzy 컨테이너 안에서**.

## 사전 준비 (호스트)

다음이 호스트에 설치되어 있어야 함:

1. **NVIDIA 드라이버** (CUDA 12.x 호환)
   ```bash
   nvidia-smi   # 동작 확인
   ```

2. **Docker** (Engine + Compose)
   ```bash
   docker --version
   ```

3. **NVIDIA Container Toolkit** — GPU 패스스루
   ```bash
   # 설치 확인
   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   ```

   안 깔렸으면:
   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
     | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
     | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt update && sudo apt install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

4. **X11 전달 허용** (Isaac Sim/RViz GUI용)
   ```bash
   xhost +local:docker
   ```

## 컨테이너 생성 (최초 1회)

호스트 셸에서 실행:

```bash
docker run -it \
    --name ros-jazzy \
    --gpus all \
    --network host \
    --ipc host \
    -e DISPLAY=$DISPLAY \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /home/sungjin/.Xauthority:/root/.Xauthority:ro \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /usr/bin/docker:/usr/bin/docker \
    -v /home/sungjin/Documents/lg_sgu_vision_v2:/workspace \
    -w /workspace \
    osrf/ros:jazzy-desktop-full \
    bash
```

### 옵션 설명

| 옵션 | 이유 |
|---|---|
| `--gpus all` | NVIDIA GPU 패스스루 (Isaac Sim, cuRobo, torch) |
| `--network host` | ROS2 DDS discovery (호스트와 같은 네트워크 네임스페이스) |
| `--ipc host` | FastDDS shared memory transport (성능) |
| `-e DISPLAY` + X11 mounts | Isaac Sim/RViz GUI 표시 |
| `-v docker.sock` + `-v docker` | 컨테이너 안에서 호스트 docker 호출 (URSim 띄우기용 — Docker-in-Docker 효과) |
| `-v /workspace` | 호스트 코드 + .venv를 컨테이너에서 직접 사용 |
| `-w /workspace` | 컨테이너 진입 시 자동으로 프로젝트 디렉토리에 위치 |

## 컨테이너 내부 셋업 (최초 1회)

컨테이너 안 (자동으로 `/workspace`):

### 1) 시스템 의존성

```bash
apt update && apt upgrade -y    # FastCDR/pal_statistics ABI 정렬 (필수)

apt install -y \
    curl git build-essential \
    libgl1 libglu1-mesa libxkbcommon0 libxkbcommon-x11-0 \
    libvulkan1 libxcb-cursor0 libxcb-xinerama0 \
    netcat-openbsd ca-certificates \
    ros-jazzy-control-msgs \
    ros-jazzy-ur-robot-driver \
    ros-jazzy-ur-client-library \
    ros-jazzy-ros2controlcli
```

`apt upgrade`는 **필수**. 안 하면 `controller_manager`가 `undefined symbol: _ZN8eprosima7fastcdr3Cdr9serializeEj` 에러로 startup 직후 죽음.

### 2) uv 설치

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 3) Python 의존성 설치

```bash
uv sync
```

`pyproject.toml` 기반으로 다음을 설치:
- `torch==2.10.0+cu128`, `torchvision`
- `isaacsim[all,extscache]==6.0.0` (~15GB 다운로드, 첫 실행 30분~1시간)
- `nvidia-curobo[cu12] @ git+...@v0.8.0` (C++/CUDA extension 빌드 ~5분)
- `nvidia-cuda-nvcc-cu12`, `nvidia-cudnn-cu12` 등 CUDA wheel
- numpy, scipy, scikit-learn, h5py, trimesh, open3d, coacd 등

처음 한 번만 오래 걸리고, 이후는 캐시 사용해서 빠름.

### 4) ROS2 환경 source

```bash
source /opt/ros/jazzy/setup.bash
```

매 셸마다 source 필요. 자동화하려면:
```bash
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
```

### 5) venv 활성화 (필요 시)

Isaac Sim, cuRobo, 파이프라인 스크립트 실행 시:
```bash
source /workspace/.venv/bin/activate
```

또는 매 명령마다 `uv run` 사용:
```bash
uv run scripts/pipeline/publish_trajectory.py --csv ...
```

## 동작 검증

```bash
# torch + CUDA
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 기대: 2.10.0+cu128 True

# cuRobo
python -c "from curobo.types import Pose, JointState; print('curobo OK')"

# ROS2 (시스템)
python -c "import rclpy; from control_msgs.action import FollowJointTrajectory; print('ros2 OK')"
```

세 줄 다 출력 나오면 셋업 완료.

## 컨테이너 상태 보존

`docker commit`으로 이미지화하면 다음에 컨테이너가 망가지거나 새로 만들 때 환경 재구축 불필요:

```bash
# 호스트에서
docker commit ros-jazzy lg_sgu_vision:setup-v1
```

이후 `docker run ... lg_sgu_vision:setup-v1`로 시작하면 위 셋업이 이미 들어 있음.

## 일상 사용

### 컨테이너 다시 들어가기

```bash
# 호스트에서
docker start ros-jazzy           # 멈춰 있으면 시작
docker exec -it ros-jazzy bash   # 새 셸 진입
```

여러 셸 동시 사용 — 각 `docker exec`로 별도 터미널 가능.

### 추천 셸 분리 (개발 워크플로우)

| 셸 | 환경 | 용도 |
|---|---|---|
| **A** | `source /opt/ros/jazzy/setup.bash`, **venv 비활성** | UR driver, RViz |
| **B** | venv 활성 (`source /workspace/.venv/bin/activate`), **시스템 ROS2 sourcing X** | Isaac Sim |
| **C** | 시스템 ROS2 + venv | 파이프라인 스크립트 (cuRobo + ROS2 동시 사용) |

이유: 시스템 ROS2(`/opt/ros/jazzy`)와 Isaac Sim 번들 ROS2가 같은 셸에서 sourcing되면 FastDDS/FastCDR ABI 충돌 위험. 셸을 나눠 쓰면 `LD_LIBRARY_PATH` 우선순위 다툼이 없음.

### 별칭 (호스트 `.bashrc`에)

```bash
alias dev='docker exec -it ros-jazzy bash'
alias dev-up='docker start ros-jazzy && docker exec -it ros-jazzy bash'
```

## 트러블슈팅

### `Could not load the dynamic library librmw_implementation.so`

ROS2 환경 변수 미설정. 컨테이너 안에서:
```bash
source /opt/ros/jazzy/setup.bash
```

### `ModuleNotFoundError: No module named 'rclpy'`

venv 안에 있는 상태에서 시스템 ROS2를 source 안 한 경우. 위와 동일하게 `source /opt/ros/jazzy/setup.bash`.

### `ModuleNotFoundError: No module named 'control_msgs'`

`apt install -y ros-jazzy-control-msgs` 누락. 셋업 1번 단계 다시 확인.

### `undefined symbol: _ZN8eprosima7fastcdr3Cdr9serializeEj`

apt 패키지 버전 불일치. `apt update && apt upgrade -y` 후 재시도.

### `cuRobo` 빌드 실패 (nvcc 못 찾음)

`pyproject.toml`의 `[tool.uv] no-build-isolation-package = ["nvidia-curobo"]` 누락. cuRobo는 격리되지 않은 환경에서 빌드해야 환경의 torch/CUDA wheel을 찾을 수 있음.

### Isaac Sim GUI 안 뜸

X11 전달 문제:
```bash
# 호스트에서
xhost +local:docker
echo $DISPLAY                 # 값 있어야 함 (예: :0 또는 :1)
```

### 호스트 `.venv` 활성화 후 segfault

호스트에서는 `.venv` 활성화하지 말 것. 컨테이너에서 빌드된 wheel은 호스트와 ABI가 다름. **모든 실행은 컨테이너에서**.

## 다음 단계

컨테이너 셋업이 끝나면:
- ROS2 + URSim 셋업: `docs/setup_ursim.md` (예정)
- 파이프라인 실행: `docs/running.md`
- 환경 변수 설정: `docs/config.md`
