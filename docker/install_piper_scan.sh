#!/usr/bin/env bash
set -eo pipefail
SCAN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
if ! /usr/bin/python3 -c 'import ensurepip' >/dev/null 2>&1; then
  apt-get update
  apt-get install -y python3.12-venv
fi
/usr/bin/python3 -m venv --system-site-packages "$SCAN_ROOT/.venv-piper"
"$SCAN_ROOT/.venv-piper/bin/python" -m pip install -r "$SCAN_ROOT/docker/piper_scan_requirements.txt"
"$SCAN_ROOT/.venv-piper/bin/python" - <<'PY'
import cv2, numpy, rclpy, cv_bridge, tf2_ros, pyAgxArm, open3d, viser, yourdfpy
assert numpy.__version__.startswith('1.'), 'cv_bridge requires the tested NumPy 1.x environment'
assert hasattr(cv2, 'aruco') and hasattr(cv2, 'calibrateHandEye')
print('PiPER scan environment ready:', cv2.__version__, numpy.__version__, open3d.__version__)
PY
