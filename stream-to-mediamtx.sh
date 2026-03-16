#!/usr/bin/env bash
# Reads post-overlay frames from shared memory and pushes H264 RTSP to MediaMTX.
# Usage: bash stream-to-mediamtx.sh [--rtsp-url rtsp://localhost:8554/catcam]
set -euo pipefail

RTSP_URL="${1:-rtsp://localhost:8554/catcam}"

exec gst-launch-1.0 \
  shmsrc socket-path=/tmp/infered.feed do-timestamp=true is-live=true ! \
  video/x-raw,format=RGB,width=1920,height=1080,framerate=30/1 ! \
  queue max-size-buffers=3 leaky=downstream ! \
  videoconvert ! \
  video/x-raw,format=I420 ! \
  x264enc tune=zerolatency bitrate=5000 speed-preset=superfast key-int-max=30 ! \
  h264parse ! \
  rtspclientsink location="$RTSP_URL" protocols=tcp latency=0
