#!/usr/bin/env bash
# Sanity-check that the system image + install_env.sh produced a working stack.
# Run inside the container:  docker exec -it ros-jazzy bash /workspace/docker/verify_env.sh
set -uo pipefail
ok=0; bad=0
check() { if eval "$2" >/dev/null 2>&1; then echo "  OK   $1"; ok=$((ok+1)); else echo "  FAIL $1"; bad=$((bad+1)); fi; }

echo "=== system (apt) stack ==="
# ros2 pkg checks need ROS sourced; docker exec does not run the entrypoint.
check "cuMotion moveit plugin" "bash -c 'source /opt/ros/jazzy/setup.bash; ros2 pkg prefix isaac_ros_cumotion_moveit'"
check "ur_robot_driver"        "bash -c 'source /opt/ros/jazzy/setup.bash; ros2 pkg prefix ur_robot_driver'"
check "ros2controlcli"         "bash -c 'source /opt/ros/jazzy/setup.bash; ros2 pkg prefix ros2controlcli'"
check "nvcc (cuda-13)"         "test -x /usr/local/cuda-13.0/bin/nvcc || command -v nvcc"
check "VPI libnvvpi4"          "dpkg -s libnvvpi4"
check "uv"                     "uv --version"
check "julia 1.12.6"           "julia --version | grep -q '1\.12\.6'"

# These four are the ones a rebuild silently gets wrong. Everything above only
# proves a package is present; these prove it is the RIGHT one.
echo "=== pins (a rebuild can break these and still look fine) ==="
check "UR pinned to packages.ros.org" \
      "grep -q 'packages.ros.org' /etc/apt/preferences.d/ros-org-ur.pref"
# The Isaac ROS repo also ships ros-jazzy-ur-* as 99.2.0, which outranks ros.org
# on a plain install and crashes controller_manager at startup.
check "ur_robot_driver is 3.8.x, not NVIDIA 99.x" \
      "dpkg-query -W -f='\${Version}' ros-jazzy-ur-robot-driver | grep -q '^3[.]8[.]'"
# Newer topic_based drops off the ros2_control ABI this image pins -> segfault.
check "topic_based_ros2_control is 0.3.0" \
      "grep -q '<version>0[.]3[.]0</version>' /workspace/ros2_overlay/src/topic_based_ros2_control/package.xml"
# Not fatal, but on a fresh machine it decides whether an interrupted 25 GB
# `uv sync` resumes or starts over.
check "uv cache on /workspace (survives container recreation)" \
      "test \"\${UV_CACHE_DIR:-}\" = /workspace/.uv-cache"

echo "=== per-machine (install_env.sh) stack ==="
check ".venv present"          "test -f /workspace/.venv/bin/activate"
check ".venv bridge patch"     "grep -q 'isaacsim.ros2.core/jazzy/lib' /workspace/.venv/bin/activate"
check "ros2_overlay built"     "test -f /workspace/ros2_overlay/install/setup.bash"
check "topic_based in overlay" "test -d /workspace/ros2_overlay/install/topic_based_ros2_control"
check "julia depot on mount"   "test -d /workspace/.julia"
check "GLNS.jl usable"         "julia --project=/workspace/scripts/julia/glns --startup-file=no -e 'using GLNS'"

echo "=== venv imports (isaac / torch cuda / curobo v0.8) ==="
check "isaacsim import"        "bash -c 'source /workspace/.venv/bin/activate && OMNI_KIT_ACCEPT_EULA=YES python -c \"import isaacsim\"'"
check "torch cuda available"   "bash -c 'source /workspace/.venv/bin/activate && python -c \"import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)\"'"
check "curobo v0.8 import"     "bash -c 'source /workspace/.venv/bin/activate && python -c \"from curobo.types import Pose, JointState\"'"

echo "=== GPU runtime (needs --gpus all + toolkit) ==="
check "nvidia-smi"             "nvidia-smi -L"
check "vulkaninfo device"      "vulkaninfo --summary | grep -qiE 'deviceName|GPU'"

echo
echo "PASS=$ok FAIL=$bad"
test "$bad" -eq 0
