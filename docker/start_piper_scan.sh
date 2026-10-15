#!/usr/bin/env bash
set -eo pipefail
SCAN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
if [[ ! -x "$SCAN_ROOT/.venv-piper/bin/python" ]]; then
  echo 'Run docker/install_piper_scan.sh inside ros2-jazzy-piper first.' >&2
  exit 1
fi
exec "$SCAN_ROOT/.venv-piper/bin/python" "$SCAN_ROOT/scripts/apps/piper_scan.py" "$@"
