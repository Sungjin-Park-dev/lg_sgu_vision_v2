# Docker: Isaac Sim + ROS 2 Jazzy + MoveIt/cuMotion

Reproducible install of the full runtime on a new machine. The **image** carries
only the machine-independent system stack; the heavy Isaac Sim venv (~24 GB) and
the ROS overlay are created once per machine against the bind-mounted repo, so
the image stays light (~10–15 GB) and portable.

## Host prerequisites

- NVIDIA GPU + recent driver
- Docker + **NVIDIA Container Toolkit** (`--gpus all` must work)
- An X server on the host (Isaac Sim opens a GUI window)

Verify GPU passthrough:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## Install (new machine)

```bash
git clone <repo> lg_sgu_vision_v2 && cd lg_sgu_vision_v2

# 1. allow the container to reach the host X server
xhost +local:root

# 2. build the system image and start the container
docker compose -f docker/compose.yaml up -d --build

# 3. create the venv + ROS overlay on the mounted /workspace (once; ~tens of GB download)
docker exec -it ros-jazzy bash /workspace/docker/install_env.sh

# 4. verify
docker exec -it ros-jazzy bash /workspace/docker/verify_env.sh
```

## Run (the two shells)

Isaac Sim and system ROS must **not** be sourced in the same shell (FastDDS
conflict). See `docs/guides/isaac-modes.md` for the 2×2 modes.

```bash
# shell 1 — Isaac app (venv, no system ROS)
docker exec -it ros-jazzy bash -c \
  'source /workspace/.venv/bin/activate && cd /workspace && OMNI_KIT_ACCEPT_EULA=YES \
   python scripts/apps/isaac_pipeline.py --object sample --mode sim --pipeline-mode moveit'
#   then press ▶ Play in the viewport

# shell 2 — ROS/MoveIt stack (system ROS + overlay). SIM stack shown; press Play first.
docker exec -it ros-jazzy bash -c \
  'source /opt/ros/jazzy/setup.bash && source /workspace/ros2_overlay/install/setup.bash \
   && cd /workspace && ros2 launch scripts/moveit/ur20_isaac_state_synced.launch.py'
```

Run **only one** shell-2 stack at a time — a leftover stack keeps publishing
`/robot_description` and the next `ros2_control_node` picks up the wrong hardware
and segfaults. `Ctrl+C` the previous stack fully before starting another.

## What lives where

| Layer | Where | Built by |
|---|---|---|
| ROS 2 Jazzy desktop | image | base `osrf/ros:jazzy-desktop-full` |
| cuMotion 4.4, UR driver, CUDA 13, VPI, uv | image | `docker/Dockerfile` |
| Isaac Sim venv (`.venv`, ~24 GB) | host mount | `install_env.sh` (`uv sync`) |
| `ros2_overlay` (topic_based 0.3.0) | host mount | `install_env.sh` (colcon) |
| repo code, `data/` | host mount | your clone |

## Notes / gotchas

- **Vulkan**: the container relies on the NVIDIA Container Toolkit injecting the
  driver's Vulkan ICD (enabled by `NVIDIA_DRIVER_CAPABILITIES=all`). If Isaac
  dies with `VkResult ERROR_INCOMPATIBLE_DRIVER`, copy the host ICD in as a
  fallback: `docker cp /usr/share/vulkan/icd.d/nvidia_icd.json ros-jazzy:/usr/share/vulkan/icd.d/`.
- **X auth resets** on every `docker start`/reboot — re-run `xhost +local:root`
  if the app fails with `Authorization required` / `GLFW initialization failed`.
- **GPU lost** (`nvidia-smi: Failed to initialize NVML`) → `docker restart ros-jazzy`.
  Persistent fix: set Docker cgroup driver to `cgroupfs` in `/etc/docker/daemon.json`.
- `install_env.sh` is idempotent; re-run it after changing `pyproject.toml`/`uv.lock`.
- Pin the source-built overlay with `TOPIC_BASED_REF=<branch>` if `master` drifts.
- **UR driver version**: the Isaac ROS repo now also ships UR packages at `99.2.0`,
  which apt prefers over the ROS repo's `3.8.0` (99 > 3). This machine was tested
  on the ROS `3.8.0` driver + `ur-moveit-config` `99.2.0`. `99.2.0` is NVIDIA's
  own UR build and is expected to work, but to reproduce the tested combo exactly,
  pin the driver/controllers to the ROS repo before the install step in the
  Dockerfile, e.g. add an apt preference:
  `Package: ros-jazzy-ur-robot-driver ros-jazzy-ur-controllers` /
  `Pin: origin packages.ros.org` / `Pin-Priority: 1001`.
