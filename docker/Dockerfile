# lg_sgu_vision_v2 — Isaac Sim + ROS 2 Jazzy + MoveIt/cuMotion system image.
#
# This image bakes only the MACHINE-INDEPENDENT SYSTEM stack (apt):
#   - ROS 2 Jazzy desktop (from the base image)
#   - NVIDIA Isaac ROS cuMotion 4.4 (C++ MoveIt planner backend)
#   - Universal Robots driver + ros2_control CLI
#   - CUDA 13 toolkit + VPI (cuMotion runtime deps)
#   - uv (for the Isaac Sim Python venv, created later)
#   - colcon + build tools (for the topic_based_ros2_control overlay, built later)
#
# It DOES NOT bake the repo, the Isaac Sim venv (~24 GB) or the ros2_overlay.
# Those are created once per machine by docker/install_env.sh against the
# bind-mounted /workspace. See docker/README.md.
#
# Build:  docker build -t lg_sgu_vision:system -f docker/Dockerfile .
# Run:    docker compose -f docker/compose.yaml up -d   (then install_env.sh once)
#
# Pinned/tested versions (2026-07, this machine): cuMotion 4.4.0,
# cuda-toolkit-13-0 13.0.3, libnvvpi4 4.0.5, ur-robot-driver 3.8.0,
# ros2controlcli 4.45.2. The NVIDIA apt repos below only serve one release
# line each, so metapackage names resolve to the tested line.

FROM osrf/ros:jazzy-desktop-full

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# 1. Base tooling: keys, https apt, git, build + colcon, Vulkan loader.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl wget gnupg2 lsb-release software-properties-common \
      git build-essential cmake \
      python3-colcon-common-extensions python3-vcstool \
      xauth libvulkan1 vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. NVIDIA apt repositories (exact set from the working ros-jazzy container).
#    - CUDA (ubuntu2404/x86_64)      -> cuMotion/VPI runtime deps
#    - Isaac ROS release-4.4 (noble) -> cuMotion + ur-moveit-config 99.x
#    - Jetson x86 r38.4 (noble)      -> libnvvpi4
# ---------------------------------------------------------------------------
RUN install -d /usr/share/keyrings && \
    # CUDA repo (cuda-keyring drops the keyring + .list under sources.list.d)
    wget -qO /tmp/cuda-keyring.deb \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i /tmp/cuda-keyring.deb && rm -f /tmp/cuda-keyring.deb && \
    # Isaac ROS repo
    curl -fsSL https://isaac.download.nvidia.com/isaac-ros/repos.key \
      | gpg --dearmor -o /usr/share/keyrings/nvidia-isaac-ros.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/nvidia-isaac-ros.gpg] https://isaac.download.nvidia.com/isaac-ros/release-4.4 noble main" \
      > /etc/apt/sources.list.d/nvidia-isaac-ros.list && \
    # Jetson x86 repo (for VPI)
    curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc \
      | gpg --dearmor -o /usr/share/keyrings/nvidia-jetson.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/nvidia-jetson.gpg] https://repo.download.nvidia.com/jetson/x86_64/noble r38.4 main" \
      > /etc/apt/sources.list.d/nvidia-jetson-x86.list

# ---------------------------------------------------------------------------
# 3. System stack: CUDA toolkit + VPI + cuMotion 4.4 + UR driver + ros2_control.
#    ros-jazzy-topic-based-ros2-control (apt 99.99.1) is pulled as a cuMotion
#    dependency but is ABI-incompatible with ros2_control 4.45 and SEGFAULTS at
#    runtime; install_env.sh source-builds 0.3.0 into ros2_overlay to shadow it.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      cuda-toolkit-13-0 \
      libnvvpi4 \
      ros-jazzy-isaac-ros-cumotion \
      ros-jazzy-isaac-ros-cumotion-moveit \
      ros-jazzy-isaac-ros-cumotion-examples \
      ros-jazzy-isaac-ros-cumotion-robot-description \
      ros-jazzy-ur-robot-driver \
      ros-jazzy-ur-moveit-config \
      ros-jazzy-ur-controllers \
      ros-jazzy-control-msgs \
      ros-jazzy-ros2controlcli \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 4. uv, installed system-wide so install_env.sh / any shell can use it.
# ---------------------------------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh \
      | env UV_UNMANAGED_INSTALL=/usr/local/bin sh \
    && uv --version

# Isaac Sim pip wheels require EULA acceptance to import; large NVIDIA wheels
# need a generous download timeout during `uv sync`.
ENV OMNI_KIT_ACCEPT_EULA=YES \
    UV_HTTP_TIMEOUT=600 \
    UV_CONCURRENT_DOWNLOADS=4

WORKDIR /workspace
# The repo is bind-mounted at /workspace at run time (not COPYed in).
CMD ["bash"]
