# 환경 설정

모든 작업(Viewpoint/Trajectory Studio, Isaac Pipeline, ROS2·MoveIt)은 `ros-jazzy` 컨테이너 하나에서 실행한다. 호스트에는 NVIDIA 드라이버, Docker, NVIDIA Container Toolkit이 필요하다.

## 컨테이너 준비

시스템 스택(ROS2 Jazzy, cuMotion, UR driver, CUDA)은 `docker/`로 이미지를 빌드하고, 무거운 Isaac venv와 ROS overlay는 컨테이너 안에서 한 번만 설치한다.

```bash
# GPU 전달 확인
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# 이미지 빌드 + 컨테이너 시작 (프로젝트 루트에서)
xhost +local:root
docker compose -f docker/compose.yaml up -d --build

# Isaac venv(uv sync)와 ros2_overlay를 /workspace에 생성 (최초 1회, 수십 GB 다운로드)
docker exec -it ros-jazzy bash /workspace/docker/install_env.sh

# 설치 검증
docker exec -it ros-jazzy bash /workspace/docker/verify_env.sh
```

`install_env.sh`는 Isaac Sim venv(`.venv`), ROS2 bridge용 `LD_LIBRARY_PATH` 패치, `topic_based_ros2_control` overlay(`ros2_overlay`)를 만든다. 이미지에는 굽지 않으므로 컨테이너를 다시 만들어도 `/workspace` 마운트에 남는다. 설치 세부와 gotcha는 [docker/README.md](../../docker/README.md)를 참고한다.

## 셸 분리 원칙

컨테이너에 `docker exec -it ros-jazzy bash`로 붙어, 용도에 따라 셸을 나눈다.

| 셸 | 환경 | 용도 |
|---|---|---|
| Isaac | `.venv` 활성, 시스템 ROS source 안 함 | Isaac Pipeline, Viewpoint/Trajectory Studio |
| ROS | `/opt/ros/jazzy/setup.bash` + `ros2_overlay/install/setup.bash` source | driver, MoveIt, RViz |

Isaac Sim 번들 ROS와 시스템 ROS를 같은 셸에서 함께 source하지 않는다. FastDDS/FastCDR 라이브러리 충돌이 날 수 있다.

GLNS를 사용한다면 Julia 패키지를 한 번 준비한다(GLNS 사용 시에만 필요).

```bash
julia --project=scripts/julia/glns -e 'using Pkg; Pkg.instantiate()'
```
