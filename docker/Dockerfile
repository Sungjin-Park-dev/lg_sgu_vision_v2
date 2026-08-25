# lg_sgu_vision_v2 — Isaac Sim + ROS 2 Jazzy + MoveIt/cuMotion system image.
#
# This image bakes only the MACHINE-INDEPENDENT SYSTEM stack (apt):
#   - ROS 2 Jazzy desktop (from the base image)
#   - NVIDIA Isaac ROS cuMotion 4.4 (C++ MoveIt planner backend)
#   - Universal Robots driver + ros2_control CLI
#   - CUDA 13 toolkit + VPI (cuMotion runtime deps)
#   - uv (for the Isaac Sim Python venv, created later)
#   - colcon + build tools (for the topic_based_ros2_control overlay, built later)
#   - Julia (runtime only, for the GLNS.jl GTSP solver; depot created later)
#
# It DOES NOT bake the repo, the Isaac Sim venv (~24 GB), the ros2_overlay or
# the Julia depot. Those are created once per machine by docker/install_env.sh
# against the bind-mounted /workspace. See docker/README.md.
#
# Build:  docker build -t lg_sgu_vision:system -f docker/Dockerfile .
# Run:    docker compose -f docker/compose.yaml up -d   (then install_env.sh once)
#
# REPRODUCIBILITY
# ---------------
# Every apt package, the base image, uv and Julia are pinned to the set verified
# working on 2026-08-25 (see section 3). Rebuild anywhere and you get that set.
#
# Why this matters: packages.ros.org and the NVIDIA repos are ROLLING — they keep
# only the newest version of each package. Without pins an unattended rebuild
# silently picks up whatever shipped that morning, and we have already been bitten
# twice (UR 99.2.0, diagnostic_updater ABI skew). Pinned, a moved repo makes the
# build FAIL LOUDLY instead of producing a robot stack that crashes at run time.
#
# When a pin 404s (the repo dropped that version), rebuild unpinned:
#     docker build --build-arg PIN_APT=0 -t lg_sgu_vision:system -f docker/Dockerfile .
# That takes the current versions but KEEPS the ros.org UR pin from section 2b,
# which is the one that actually breaks the robot. Re-verify before trusting it.

FROM osrf/ros:jazzy-desktop-full@sha256:922657e1da43e8f91df9f43e8218cc6ec6f75b19d2c15e20321ec970e966101a

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
# 2b. Keep the UR stack on packages.ros.org.
#     The Isaac ROS repo also serves ros-jazzy-ur-* — as version 99.2.0, which
#     outranks ros.org's 3.8.0 on any plain `apt-get install`. That build is not
#     the released UR driver and CRASHES controller_manager, so every real-robot
#     run dies at startup. Priority 1001 (> 1000) makes ros.org win even when it
#     means a downgrade, hence --allow-downgrades in section 3.
#
#     This file is load-bearing: without it a rebuild produces a broken image
#     that still LOOKS like it built fine.
# ---------------------------------------------------------------------------
RUN printf '%s\n' \
      'Package: ros-jazzy-ur-*' \
      'Pin: origin packages.ros.org' \
      'Pin-Priority: 1001' \
      > /etc/apt/preferences.d/ros-org-ur.pref

# ---------------------------------------------------------------------------
# 3. System stack: CUDA toolkit + VPI + cuMotion 4.4 + UR driver + ros2_control.
#    ros-jazzy-topic-based-ros2-control (apt 99.99.1) is pulled as a cuMotion
#    dependency but is ABI-incompatible with ros2_control 4.45 and SEGFAULTS at
#    runtime; install_env.sh source-builds 0.3.0 into ros2_overlay to shadow it.
#
#    `apt-get upgrade` first: the base image pins some ROS packages (e.g.
#    diagnostic_updater 4.2.6) older than the ros2_control 4.45 the repo now
#    serves. Without the upgrade, ros2_control_node dies with a symbol lookup
#    error (undefined diagnostic_updater::Updater 8-arg ctor).
#
#    Versions below are the set verified on 2026-08-25. PIN_APT=0 strips them
#    (see the REPRODUCIBILITY note at the top) — the section 2b UR pin still holds.
# ---------------------------------------------------------------------------
ARG PIN_APT=1
RUN apt-get update && apt-get upgrade -y \
    && PKGS="cuda-toolkit-13-0=13.0.3-1 \
             libnvvpi4=4.0.5 \
             ros-jazzy-isaac-ros-cumotion=4.4.0-0noble.20260428200047773 \
             ros-jazzy-isaac-ros-cumotion-moveit=4.4.0-0noble.20260428211026915 \
             ros-jazzy-isaac-ros-cumotion-examples=4.4.0-0noble.20260428214116449 \
             ros-jazzy-isaac-ros-cumotion-robot-description=4.4.0-0noble.20260428184406204 \
             ros-jazzy-ur-robot-driver=3.8.0-1noble.20260615.175905 \
             ros-jazzy-ur-moveit-config=3.8.0-1noble.20260617.155215 \
             ros-jazzy-ur-controllers=3.8.0-1noble.20260615.172009 \
             ros-jazzy-ur-client-library=2.12.0-1noble.20260519.172915 \
             ros-jazzy-ur-dashboard-msgs=3.8.0-1noble.20260615.103026 \
             ros-jazzy-control-msgs=5.9.0-1noble.20260615.113124 \
             ros-jazzy-ros2controlcli=4.45.2-1noble.20260615.165650" \
    && if [ "$PIN_APT" != "1" ]; then PKGS="$(echo "$PKGS" | sed 's/=[^ ]*//g')"; fi \
    && apt-get install -y --no-install-recommends --allow-downgrades $PKGS \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 4. uv, installed system-wide so install_env.sh / any shell can use it.
# ---------------------------------------------------------------------------
ARG UV_VERSION=0.12.5
RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" \
      | env UV_UNMANAGED_INSTALL=/usr/local/bin sh \
    && uv --version

# ---------------------------------------------------------------------------
# 5. Julia, for the GLNS.jl GTSP solver (scripts/julia/glns).
#    Version is pinned to match scripts/julia/glns/Manifest.toml's
#    julia_version — on a different minor, Julia re-resolves dependencies and
#    the committed Manifest stops pinning anything.
#    Only the RUNTIME lives in the image; the depot (downloaded packages, ~GB)
#    is machine-local and goes on the /workspace mount — see JULIA_DEPOT_PATH
#    below and install_env.sh step 4.
# ---------------------------------------------------------------------------
ARG JULIA_VERSION=1.12.6
RUN curl -fsSL "https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_VERSION%.*}/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" \
      | tar -xz -C /opt \
    && ln -s "/opt/julia-${JULIA_VERSION}/bin/julia" /usr/local/bin/julia \
    && julia --version

# Isaac Sim pip wheels require EULA acceptance to import; large NVIDIA wheels
# need a generous download timeout during `uv sync`.
#
# JULIA_DEPOT_PATH: keep the Julia depot on the bind-mounted /workspace so it
# survives container recreation (like .venv and ros2_overlay). Set as ENV, not
# in a shell rc — `docker exec ... bash script.sh` does not read rc files.
ENV OMNI_KIT_ACCEPT_EULA=YES \
    UV_HTTP_TIMEOUT=600 \
    UV_CONCURRENT_DOWNLOADS=4 \
    JULIA_DEPOT_PATH=/workspace/.julia

WORKDIR /workspace
# The repo is bind-mounted at /workspace at run time (not COPYed in).
CMD ["bash"]
