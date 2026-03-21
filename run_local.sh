#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

print_usage() {
  cat <<EOF
Usage: bash run_local.sh --hef /path/to/model.hef [--no-viewer] [--stream] [--arch hailo8|hailo8l]
                         [--brownie-model PATH] [--brownie-embedding PATH]
                         [--brownie-threshold 0.70] [--no-brownie]
                         [--motion] [--motion-threshold NUM] [--motion-cooldown SECS]

Starts camera producer, detection pipeline (AWS disabled), and viewer/stream.

Options:
  --hef PATH                Path to HEF model file (required)
  --no-viewer               Do not launch the viewer sink
  --stream                  Stream to MediaMTX via RTSP (view at http://<pi-ip>:8889/catcam)
  --arch VALUE              Force Hailo architecture (e.g., hailo8 or hailo8l)
  --brownie-model PATH      Path to embedding model HEF (default: resnet_v1_50_h8l.hef)
  --brownie-embedding PATH  Path to brownie_embedding.npy
  --brownie-threshold NUM   Similarity threshold 0.0-1.0 (default: 0.70)
  --brownie-backend VALUE   "simple" or "hailo" (default: simple)
  --no-brownie              Disable Brownie identification
  --motion                  Enable motion detection (only process when motion detected)
  --motion-threshold NUM    Pixel change threshold 0-255 (default: 25)
  --motion-cooldown SECS    Keep processing for N seconds after motion stops (default: 3.0)
EOF
}

HEF_PATH=""
LAUNCH_VIEWER=1
ENABLE_STREAM=0
ARCH_ARG=()
BROWNIE_MODEL="${BROWNIE_MODEL:-}"
BROWNIE_EMBEDDING="${BROWNIE_EMBEDDING:-}"
BROWNIE_THRESHOLD="${BROWNIE_THRESHOLD:-}"
BROWNIE_BACKEND="${BROWNIE_BACKEND:-}"
DISABLE_BROWNIE="${DISABLE_BROWNIE:-0}"
ENABLE_MOTION="${ENABLE_MOTION:-0}"
MOTION_THRESHOLD="${MOTION_THRESHOLD:-}"
MOTION_COOLDOWN="${MOTION_COOLDOWN:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hef)
      HEF_PATH="$2"; shift 2;;
    --no-viewer)
      LAUNCH_VIEWER=0; shift;;
    --stream)
      ENABLE_STREAM=1; shift;;
    --arch)
      ARCH_ARG=("--arch" "$2"); shift 2;;
    --brownie-model)
      BROWNIE_MODEL="$2"; shift 2;;
    --brownie-embedding)
      BROWNIE_EMBEDDING="$2"; shift 2;;
    --brownie-threshold)
      BROWNIE_THRESHOLD="$2"; shift 2;;
    --brownie-backend)
      BROWNIE_BACKEND="$2"; shift 2;;
    --no-brownie)
      DISABLE_BROWNIE=1; shift;;
    --motion)
      ENABLE_MOTION=1; shift;;
    --motion-threshold)
      MOTION_THRESHOLD="$2"; shift 2;;
    --motion-cooldown)
      MOTION_COOLDOWN="$2"; shift 2;;
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

# Make hailo_apps package available (installed in hailo-rpi5-examples venv)
HAILO_EXAMPLES_SITE="$HOME/repos/hailo-rpi5-examples/venv_hailo_rpi_examples/lib/python3.11/site-packages"
if [[ -d "$HAILO_EXAMPLES_SITE" ]]; then
  export PYTHONPATH="${HAILO_EXAMPLES_SITE}:${PYTHONPATH:-}"
else
  echo "Warning: hailo_apps site-packages not found at $HAILO_EXAMPLES_SITE" >&2
fi

# Post-process .so files should be in the repo root
for so in libyolo_hailortpp_postprocess.so libwhole_buffer.so; do
  if [[ ! -f "$ROOT_DIR/$so" ]]; then
    echo "Warning: $so not found in $ROOT_DIR" >&2
  fi
done

P_MEDIAMTX=""
P_PRODUCER=""
P_DETECT=""
P_VIEWER=""
P_STREAM=""

cleanup() {
  echo ""
  echo "Stopping processes..."
  [[ -n "$P_STREAM" ]]   && kill "$P_STREAM"   2>/dev/null || true
  [[ -n "$P_VIEWER" ]]   && kill "$P_VIEWER"   2>/dev/null || true
  [[ -n "$P_DETECT" ]]   && kill "$P_DETECT"   2>/dev/null || true
  [[ -n "$P_PRODUCER" ]] && kill "$P_PRODUCER"  2>/dev/null || true
  [[ -n "$P_MEDIAMTX" ]] && kill "$P_MEDIAMTX" 2>/dev/null || true
  rm -f /tmp/feed.raw /tmp/infered.feed
}
trap cleanup EXIT INT TERM

# --- Start MediaMTX if streaming is enabled ---
if [[ "$ENABLE_STREAM" -eq 1 ]]; then
  if command -v mediamtx >/dev/null 2>&1; then
    echo "Starting MediaMTX..."
    mediamtx /etc/mediamtx/mediamtx.yml &>/dev/null &
    P_MEDIAMTX=$!
    sleep 1
  elif systemctl is-active --quiet mediamtx 2>/dev/null; then
    echo "MediaMTX is already running as a service."
  else
    echo "Error: --stream requires MediaMTX. Run: bash install-mediamtx.sh" >&2
    exit 1
  fi
fi

# --- Start camera producer ---
echo "Starting camera producer..."
bash "$ROOT_DIR/producer-camera.sh" >/dev/null 2>&1 &
P_PRODUCER=$!

# Wait for camera shared memory to be ready
echo "Waiting for camera feed..."
for i in $(seq 1 30); do
  if [[ -e /tmp/feed.raw ]]; then
    break
  fi
  sleep 1
done
if [[ ! -e /tmp/feed.raw ]]; then
  echo "Error: /tmp/feed.raw not created after 30s. Camera producer may have failed." >&2
  exit 1
fi

# --- Start detection pipeline ---
echo "Starting detection pipeline (local, AWS disabled)..."
export DISABLE_AWS=1
export DISABLE_BROWNIE="$DISABLE_BROWNIE"
[[ -n "$BROWNIE_MODEL" ]]     && export BROWNIE_MODEL="$BROWNIE_MODEL"
[[ -n "$BROWNIE_EMBEDDING" ]] && export BROWNIE_EMBEDDING="$BROWNIE_EMBEDDING"
[[ -n "$BROWNIE_THRESHOLD" ]] && export BROWNIE_THRESHOLD="$BROWNIE_THRESHOLD"
[[ -n "$BROWNIE_BACKEND" ]]   && export BROWNIE_BACKEND="$BROWNIE_BACKEND"
[[ "$ENABLE_MOTION" -eq 1 ]] && export ENABLE_MOTION=1
[[ -n "$MOTION_THRESHOLD" ]]  && export MOTION_THRESHOLD="$MOTION_THRESHOLD"
[[ -n "$MOTION_COOLDOWN" ]]   && export MOTION_COOLDOWN="$MOTION_COOLDOWN"
python3 -u "$ROOT_DIR/detection.py" --hef-path "$HEF_PATH" "${ARCH_ARG[@]}" &
P_DETECT=$!

# --- Wait for inference shared memory to be ready ---
echo "Waiting for inference pipeline to be ready..."
for i in $(seq 1 30); do
  if [[ -e /tmp/infered.feed ]]; then
    break
  fi
  sleep 1
done
if [[ ! -e /tmp/infered.feed ]]; then
  echo "Warning: /tmp/infered.feed not created after 30s. Starting consumers anyway..." >&2
fi

# --- Start RTSP stream consumer ---
if [[ "$ENABLE_STREAM" -eq 1 ]]; then
  echo "Starting RTSP stream to MediaMTX..."
  bash "$ROOT_DIR/stream-to-mediamtx.sh" &
  P_STREAM=$!
  echo "Stream available at: http://<pi-ip>:8889/catcam"
fi

# --- Start display viewer ---
if [[ "$LAUNCH_VIEWER" -eq 1 ]]; then
  echo "Starting viewer..."
  bash "$ROOT_DIR/consumer-infered-fpsdisplaysink.sh" &
  P_VIEWER=$!
fi

echo ""
echo "Running. Press Ctrl+C to stop."
wait "$P_DETECT"
