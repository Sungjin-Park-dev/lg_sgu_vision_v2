# Isaac Sim 전체 셋업·실행 가이드

도커 설치부터 컨테이너 생성, Isaac Sim 실행까지 **처음부터 끝까지** 정리한 문서.
실제로 동작 확인한 명령과, 도중에 막혔던 부분의 해결법을 포함한다.

> 배경: 이 프로젝트는 **모든 파이썬/ROS/Isaac Sim 작업을 `ros-jazzy` 도커 컨테이너 안에서**
> 한다(원래 `docs/setup_docker.md`에 정해진 방침). 시스템 ROS2와 Isaac Sim 번들 ROS2의
> FastDDS/FastCDR ABI 충돌을 컨테이너로 격리해 피하기 위함이며, 호스트에서 직접 venv를
> 켜고 실행하면 segfault가 난다. 호스트에는 NVIDIA 드라이버 + Docker만 둔다.

---

## 변수 설정 (먼저 셸에 export)

아래 명령들은 이 변수를 사용한다. 본인 환경에 맞게 한 번만 설정하면 그대로 복붙 가능하다.

```bash
# 프로젝트 루트 (git clone 받은 경로)
export REPO="$(git -C . rev-parse --show-toplevel 2>/dev/null || pwd)"
# 또는 직접 지정:  export REPO="$HOME/lg/lg_sgu_vision_v2"

# 컨테이너 이름 / 저장 이미지 태그
export CTR=ros-jazzy
export IMG=lg_sgu_vision:setup-v1

echo "REPO=$REPO  /  DISPLAY=$DISPLAY"   # DISPLAY 값이 비어 있으면 GUI 안 뜸
```

> `$DISPLAY`는 로그인 세션이 자동으로 설정한다(예: `:0`, `:1`). 비어 있으면
> 데스크톱 세션에서 실행 중인지 확인한다.

---

## 0. 전체 흐름 한눈에

```
[호스트 준비]  NVIDIA 드라이버 + Docker + NVIDIA Container Toolkit + xhost
      ↓
[컨테이너 생성]  A) 저장해둔 이미지(lg_sgu_vision:setup-v1)로 즉시   ← 권장
                 B) 맨바닥(osrf/ros:jazzy-desktop-full)부터 직접 빌드
      ↓
[컨테이너 시작]  docker start / docker exec 로 진입
      ↓
[Isaac Sim 실행]  scene.py / launch_sim.py / isaac_pipeline.py
```

---

## 1. 호스트 사전 준비 (최초 1회)

### 1-1. NVIDIA 드라이버 확인
```bash
nvidia-smi
# GPU 이름 + 드라이버 버전이 보이면 OK (이 머신: RTX 5080 / 580.105.08)
```

### 1-2. Docker 설치 (이미 있으면 건너뜀)
```bash
# 설치 확인
docker --version

# 없으면 설치 (Ubuntu)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER   # sudo 없이 docker 쓰려면 (재로그인 필요)
```

### 1-3. NVIDIA Container Toolkit (GPU 패스스루)
```bash
# 동작 확인
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# 안 되면 설치
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 1-4. X11 GUI 허용 (재부팅 후 처음 한 번)
```bash
xhost +local:docker
echo $DISPLAY   # 값이 있어야 함 (예: :0, :1)
```

> ⚠️ `~/.Xauthority`가 root 소유 빈 디렉토리로 깨져 있는 경우가 있다(`ls -la ~/.Xauthority`로
> 파일이 아니라 `d`로 시작하면 깨진 것). 그럴 땐 `sudo rmdir ~/.Xauthority`로 지우고,
> `docker run`에 `.Xauthority`를 마운트하지 말고 `xhost`로만 처리한다.

---

## 2. 컨테이너 생성

### 방법 A — 저장해둔 이미지로 생성 (권장, 빠름)

지난번 셋업을 `$IMG`(`lg_sgu_vision:setup-v1`) 이미지로 커밋해뒀다. apt·Vulkan 설정이 다
들어 있으므로 이걸로 만들면 **추가 셋업 없이 바로 실행 가능**하다.

```bash
docker run -d -it \
    --name "$CTR" \
    --gpus all \
    --network host \
    --ipc host \
    -e DISPLAY=$DISPLAY \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /usr/bin/docker:/usr/bin/docker \
    -v "$REPO":/workspace \
    -w /workspace \
    "$IMG" \
    bash
```

확인:
```bash
docker ps --filter name="$CTR" --format '{{.Names}}\t{{.Status}}'
```

→ 바로 [4. Isaac Sim 실행](#4-isaac-sim-실행)으로.

> 이미지가 없으면(`docker images lg_sgu_vision`로 확인) 방법 B로 진행.

---

### 방법 B — 맨바닥부터 직접 빌드 (최초 셋업 / 이미지 없을 때)

#### B-1. 베이스 이미지 받기
```bash
docker pull osrf/ros:jazzy-desktop-full
```

#### B-2. 컨테이너 생성
방법 A와 동일하되 **맨 끝 이미지만** `osrf/ros:jazzy-desktop-full`로:
```bash
docker run -d -it \
    --name "$CTR" \
    --gpus all --network host --ipc host \
    -e DISPLAY=$DISPLAY \
    -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /usr/bin/docker:/usr/bin/docker \
    -v "$REPO":/workspace \
    -w /workspace \
    osrf/ros:jazzy-desktop-full \
    bash
```

#### B-3. apt 의존성 (컨테이너 안)
```bash
docker exec "$CTR" bash -c '
export DEBIAN_FRONTEND=noninteractive
apt update && apt upgrade -y          # apt upgrade 필수 (안 하면 controller_manager 죽음)
apt install -y \
    curl git build-essential \
    libgl1 libglu1-mesa libxkbcommon0 libxkbcommon-x11-0 \
    libvulkan1 libxcb-cursor0 libxcb-xinerama0 \
    netcat-openbsd ca-certificates \
    ros-jazzy-control-msgs ros-jazzy-ur-robot-driver \
    ros-jazzy-ur-client-library ros-jazzy-ros2controlcli
'
```

#### B-4. uv 설치 + Python 의존성
```bash
docker exec "$CTR" bash -c '
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd /workspace
UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=4 uv sync   # Isaac Sim ~15GB, 30분~1시간
'
```
- `UV_HTTP_TIMEOUT=600`이 **중요**: NVIDIA 대용량 wheel이 기본 30초 타임아웃에 자주
  실패한다(`isaacsim-test`, `nvidia-nvshmem-cu12` 등). 실패하면 같은 명령을 다시 실행하면
  캐시를 이어받아 통과한다.
- cuRobo(v0.8)는 `pyproject.toml`의 git 의존성이라 `uv sync`가 자동 설치한다.
  README의 별도 `git clone curobo`는 구버전 방식이라 **불필요**.

#### B-5. ⚠️ NVIDIA Vulkan ICD 주입 (이게 핵심)
컨테이너에 NVIDIA Vulkan ICD가 없으면 Isaac Sim 렌더러가
`VkResult: ERROR_INCOMPATIBLE_DRIVER` / `no suitable CUDA GPU found`로 죽는다
(**torch.cuda는 True인데도**). 호스트의 정상 파일을 복사한다:
```bash
docker exec "$CTR" mkdir -p /usr/share/vulkan/icd.d
docker cp /usr/share/vulkan/icd.d/nvidia_icd.json "$CTR":/usr/share/vulkan/icd.d/nvidia_icd.json
# 유효한 JSON인지 확인
docker exec "$CTR" python3 -c "import json; print(json.load(open('/usr/share/vulkan/icd.d/nvidia_icd.json')))"
```

#### B-6. venv activate에 ROS2 라이브러리 경로 추가
`uv sync`가 venv를 새로 만들면서 빠진 export를 다시 넣는다 (ROS2 bridge용, 1회):
```bash
docker exec "$CTR" bash -c '
ACT=/workspace/.venv/bin/activate
grep -q "isaacsim.ros2.core/jazzy/lib" "$ACT" || \
echo "export LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH:\$VIRTUAL_ENV/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core/jazzy/lib\"" >> "$ACT"
'
```

#### B-7. 동작 검증
```bash
docker exec "$CTR" bash -c '
export PATH="$HOME/.local/bin:$PATH"; cd /workspace
OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python -c "
import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())
import isaacsim; print(\"isaacsim OK\")
from curobo.types import Pose, JointState; print(\"curobo OK\")
"'
```
세 줄 다 나오면 셋업 완료.

#### B-8. 이미지로 보존 (강력 권장)
apt·Vulkan 설정은 컨테이너 레이어라 컨테이너를 지우면 사라진다. 지금 상태를 이미지로 굳혀
두면 다음엔 [방법 A](#방법-a--저장해둔-이미지로-생성-권장-빠름)로 바로 만들 수 있다:
```bash
docker commit "$CTR" "$IMG"
docker images lg_sgu_vision   # 확인 (약 56GB)
```

---

## 3. 컨테이너 시작 / 재진입 (일상 사용)

```bash
# 멈춰 있으면 시작
docker start "$CTR"

# 새 셸로 진입
docker exec -it "$CTR" bash

# 상태 확인
docker ps -a --filter name="$CTR"
```

별칭으로 편하게 (호스트 `~/.bashrc`, 컨테이너 이름이 `ros-jazzy`인 경우):
```bash
alias dev='docker exec -it ros-jazzy bash'
alias dev-up='docker start ros-jazzy && docker exec -it ros-jazzy bash'
```

---

## 4. Isaac Sim 실행

호스트에서 `xhost +local:docker` 한 뒤 (재부팅 후 1회), 아래 한 줄:

```bash
docker exec -it "$CTR" bash -c \
  'source /workspace/.venv/bin/activate && cd /workspace && \
   OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac/scene.py --object sample'
```

- GUI 창이 호스트 화면(`$DISPLAY`, 예: `:0`/`:1`)에 뜬다.
- 종료: `Ctrl+C` 또는 GUI 창 닫기.
- 첫 부팅은 셰이더 컴파일로 ~8분, 이후엔 캐시로 빨라진다.

### 실행 모드 (스크립트 경로만 교체)

| 스크립트 | 용도 |
|---|---|
| `scripts/isaac/scene.py --object sample` | UR20 + 워크셀 + ROS2 `/joint_states` 연동 |
| `scripts/isaac/launch_sim.py` | 빈 시뮬레이터 (URDF/USD 수동 import용) |
| `scripts/apps/isaac_pipeline.py --object sample` | 물체 선택·이동 + 궤적 생성/미리보기 GUI |

앞부분(`source ... && cd /workspace && OMNI_KIT_ACCEPT_EULA=YES python`)은 공통.

### 프로세스 수동 종료
```bash
docker exec "$CTR" pkill -f "scripts/isaac/scene.py"
```

---

## 5. 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `VkResult: ERROR_INCOMPATIBLE_DRIVER`, `no suitable CUDA GPU found` (torch는 cuda=True) | Vulkan ICD 누락 → [B-5](#b-5-️-nvidia-vulkan-icd-주입-이게-핵심) |
| `uv sync` network timeout (`isaacsim-test` 등) | `UV_HTTP_TIMEOUT=600`으로 재실행 (캐시 이어받음) |
| `ROS2 Bridge startup failed` | activate에 LD_LIBRARY_PATH 누락 → [B-6](#b-6-venv-activate에-ros2-라이브러리-경로-추가) |
| `Available DOFs: []` | USD articulation root 문제. GUI URDF Importer로 재변환 (docs/setup_isaac_sim.md) |
| GUI 안 뜸 | 호스트에서 `xhost +local:docker`, `echo $DISPLAY` 확인 |
| `controller_manager` undefined symbol | apt upgrade 안 함 → `apt update && apt upgrade -y` |
| 호스트에서 `.venv` 켜니 segfault | 호스트 직접 실행 금지. 항상 컨테이너 안에서. |

---

## 6. 컨테이너 재생성 시 주의

| 항목 | 위치 | `docker rm` 후 재생성하면 |
|---|---|---|
| 코드 · `.venv` · 데이터 | 호스트 `$REPO` (마운트) | **유지** ✅ |
| apt 설치분 · Vulkan ICD | 컨테이너 레이어 (`/var/lib/docker`) | **사라짐** ❌ |

→ 그래서 [방법 A](#방법-a--저장해둔-이미지로-생성-권장-빠름)(저장 이미지)로 만들거나, 맨바닥에서
만들었으면 반드시 [B-8 commit](#b-8-이미지로-보존-강력-권장)으로 굳혀둔다.

---

## 관련 문서
- 컨테이너 셋업 원본: `docs/setup_docker.md`
- Isaac Sim 상세/URDF Importer: `docs/setup_isaac_sim.md`
- 파이프라인 실행(viewpoint→IK→publish): `docs/running.md`
- cuRobo v0.8 마이그레이션: `docs/curobo_v0.8_migration.md`
