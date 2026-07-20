#!/usr/bin/env bash
# Per-machine environment setup, run ONCE inside the container after the image
# is built and /workspace is bind-mounted. Idempotent: safe to re-run.
#
#   docker exec -it ros-jazzy bash /workspace/docker/install_env.sh
#
# Creates, on the host-mounted /workspace (so it survives container recreation):
#   1. .venv          — Isaac Sim + curobo + torch, via `uv sync`
#   2. .venv patch    — LD_LIBRARY_PATH for the Isaac ROS 2 bridge
#   3. ros2_overlay   — topic_based_ros2_control 0.3.0, source-built (ABI-matched
#                       replacement for the segfaulting apt 99.99.1)
set -euo pipefail

WS=/workspace
TB_REF="${TOPIC_BASED_REF:-main}"   # PickNikRobotics/topic_based_ros2_control (default branch = main)

cd "$WS"

echo "=== [1/3] uv sync — Isaac Sim venv (this downloads ~tens of GB on a fresh machine) ==="
UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}" UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}" \
  uv sync

echo "=== [2/3] Patch .venv activate with the Isaac ROS 2 bridge LD_LIBRARY_PATH ==="
ACTIVATE="$WS/.venv/bin/activate"
BRIDGE_LINE='export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$VIRTUAL_ENV/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core/jazzy/lib"'
if ! grep -qF 'isaacsim.ros2.core/jazzy/lib' "$ACTIVATE"; then
  printf '\n# lg_sgu_vision: Isaac Sim ROS 2 bridge libraries\n%s\n' "$BRIDGE_LINE" >> "$ACTIVATE"
  echo "  patched $ACTIVATE"
else
  echo "  already patched"
fi

echo "=== [3/3] Build topic_based_ros2_control overlay (ABI-matched, shadows apt 99.99.1) ==="
OVERLAY="$WS/ros2_overlay"
SRC="$OVERLAY/src/topic_based_ros2_control"
if [ ! -f "$SRC/package.xml" ]; then
  mkdir -p "$OVERLAY/src"
  # git clone can hit an auth prompt inside the container; fetch the tarball.
  echo "  fetching topic_based_ros2_control@$TB_REF"
  curl -fsSL "https://codeload.github.com/PickNikRobotics/topic_based_ros2_control/tar.gz/refs/heads/$TB_REF" \
    | tar -xz -C "$OVERLAY/src"
  mv "$OVERLAY"/src/topic_based_ros2_control-* "$SRC"
fi
# BUILD_TESTING=OFF: ros_testing is not installed in the base image.
# set +u: ROS setup.bash references unset vars (AMENT_TRACE_SETUP_FILES) that
# trip nounset.
( set +u; cd "$OVERLAY" && source /opt/ros/jazzy/setup.bash && \
  colcon build --cmake-args -DBUILD_TESTING=OFF )

echo
echo "=== done. Verify with docker/verify_env.sh, then run shells per docs/guides/isaac-modes.md ==="
echo "  shell1 (Isaac): source .venv/bin/activate && python scripts/apps/isaac_pipeline.py ..."
echo "  shell2 (ROS):   source /opt/ros/jazzy/setup.bash && source ros2_overlay/install/setup.bash && ros2 launch ..."
