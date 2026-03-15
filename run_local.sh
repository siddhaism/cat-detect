#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

print_usage() {
  cat <<EOF
Usage: bash run_local.sh --hef /path/to/model.hef [--no-viewer] [--arch hailo8|hailo8l]

Starts camera producer, detection pipeline (AWS disabled), and viewer.

Options:
  --hef PATH       Path to HEF model file (required)
  --no-viewer      Do not launch the viewer sink
  --arch VALUE     Force Hailo architecture (e.g., hailo8 or hailo8l)
EOF
}

HEF_PATH=""
LAUNCH_VIEWER=1
ARCH_ARG=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hef)
      HEF_PATH="$2"; shift 2;;
    --no-viewer)
      LAUNCH_VIEWER=0; shift;;
    --arch)
      ARCH_ARG=("--arch" "$2"); shift 2;;
    -h|--help)
      print_usage; exit 0;;
    *)
      echo "Unknown argument: $1" >&2; print_usage; exit 1;;
  esac
done

if [[ -z "$HEF_PATH" ]]; then
  echo "Error: --hef /path/to/model.hef is required" >&2
  print_usage
  exit 1
fi

if ! command -v gst-launch-1.0 >/dev/null 2>&1; then
  echo "gst-launch-1.0 not found. Install GStreamer packages on the Pi." >&2
  exit 1
fi

# Source Hailo/TAPPAS environment (must be sourced, not executed)
if [[ -f "$ROOT_DIR/setup_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/setup_env.sh" || {
    echo "Failed to source setup_env.sh" >&2
    exit 1
  }
else
  echo "setup_env.sh not found in repo root." >&2
  exit 1
fi

# Basic checks for post-process libs (warn-only)
for so in libyolo_hailortpp_postprocess.so libwhole_buffer.so; do
  if [[ ! -f "$ROOT_DIR/$so" ]]; then
    echo "Warning: $so not found in repo root. Ensure paths in detection_pipeline.py are valid." >&2
  fi
done

cleanup() {
  echo "\nStopping processes..."
  [[ -n "${P_PRODUCER:-}" ]] && kill "$P_PRODUCER" 2>/dev/null || true
  [[ -n "${P_DETECT:-}" ]] && kill "$P_DETECT" 2>/dev/null || true
  [[ -n "${P_VIEWER:-}" ]] && kill "$P_VIEWER" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting camera producer..."
bash "$ROOT_DIR/producer-camera.sh" >/dev/null 2>&1 &
P_PRODUCER=$!
sleep 1

echo "Starting detection pipeline (local, AWS disabled)..."
DISABLE_AWS=1 python3 "$ROOT_DIR/detection.py" --hef-path "$HEF_PATH" "${ARCH_ARG[@]}" &
P_DETECT=$!
sleep 1

if [[ "$LAUNCH_VIEWER" -eq 1 ]]; then
  echo "Starting viewer..."
  bash "$ROOT_DIR/consumer-infered-fpsdisplaysink.sh" &
  P_VIEWER=$!
fi

echo "\nRunning. Press Ctrl+C to stop."
wait "$P_DETECT"

