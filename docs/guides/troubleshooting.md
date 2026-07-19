# 문제 해결

## 앱과 Python

| 증상 | 확인할 내용 |
|---|---|
| 모듈을 찾지 못함 | 컨테이너에서 `install_env.sh`로 venv 설치 후 `uv run --no-sync ...`로 실행 |
| 브라우저 앱이 열리지 않음 | Viewpoint는 8080, Trajectory는 8081 포트와 터미널 로그 확인 |
| GLNS 실행 실패 | Julia 설치와 `scripts/julia/glns` 패키지 초기화 확인 |
| CUDA 또는 cuRobo 오류 | `nvidia-smi`, CUDA 12.x, `torch.cuda.is_available()` 확인 |

## 데이터와 궤적

| 증상 | 확인할 내용 |
|---|---|
| Object 목록에 물체가 없음 | `data/{object}/mesh/source.obj` 존재 여부 확인 |
| Isaac에서 물체 로드 실패 | `source.usd`를 생성했는지 확인 |
| IK 도달률이 낮음 | 물체 pose, working distance와 viewpoint 방향 확인 |
| GLNS 결과가 없음 | HDF5에 Delaunay adjacency가 있는지 확인 |
| 재생할 dense trajectory가 없음 | GLNS solve 후 scan motion planning까지 실행했는지 확인 |

## Isaac과 ROS2

| 증상 | 확인할 내용 |
|---|---|
| Isaac GUI가 뜨지 않음 | `DISPLAY`, X11 허용, GPU container 전달 확인 |
| `Authorization required` / `GLFW initialization failed` | 호스트에서 `xhost +local:root` 다시 실행(재부팅·`docker start`마다 초기화됨) |
| Isaac이 `ERROR_INCOMPATIBLE_DRIVER` / `no suitable CUDA GPU`로 종료 | `NVIDIA_DRIVER_CAPABILITIES=all` 확인, 필요하면 host의 `nvidia_icd.json`을 컨테이너로 복사 |
| `nvidia-smi: Failed to initialize NVML` | 컨테이너가 GPU 접근을 잃음 → `docker restart ros-jazzy` |
| `Available DOFs: []` | robot USD의 Articulation Root 확인 |
| ROS2 bridge 시작 실패 | Isaac 셸에서 시스템 ROS를 source하지 않았는지 확인 |
| 로봇이 움직이지 않음 | Run/Pipeline 조합과 활성 controller 확인 |
| 시작 순간 자세가 튐 | Isaac을 Play한 뒤 상태 동기화가 끝나고 실행했는지 확인 |
| SIM 스택이 gate에서 멈춤 / `Switch controller timed out` | Isaac 앱을 먼저 Play해 `/isaac_joint_states`가 흐른 뒤 ROS 스택을 실행 |
| ROS 스택 실행 직후 `ros2_control_node` segfault / controller 서비스 접속 실패 | ROS 스택은 한 번에 하나만 — 이전 스택을 `Ctrl+C`로 완전히 끈 뒤 다시 실행 |

원인을 찾을 때는 Isaac Pipeline의 `Log` 패널과 `ros2 control list_controllers` 결과를 함께 확인한다.
