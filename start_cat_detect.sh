#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Source the Hailo/TAPPAS environment exactly as run_local.sh expects.
source "$ROOT_DIR/setup_env.sh"

# Make hailo_apps available when launched outside an interactive shell.
HAILO_EXAMPLES_SITE="$HOME/repos/hailo-rpi5-examples/venv_hailo_rpi_examples/lib/python3.11/site-packages"
export PYTHONPATH="${HAILO_EXAMPLES_SITE}:${PYTHONPATH:-}"

# Headless service mode: stream enabled, local viewer disabled.
exec /bin/bash "$ROOT_DIR/run_local.sh" \
  --hef /usr/share/hailo-models/yolov8s_h8.hef \
  --stream \
  --no-viewer
