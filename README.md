Cat Detect (Hailo + GStreamer) — Local Run

Overview
- Ingests camera frames via GStreamer, runs YOLOv8 on Hailo (8/8L), overlays detections, and optionally publishes events to AWS IoT.
- This guide focuses on running locally on a Raspberry Pi without AWS.

Prerequisites
- Raspberry Pi 4/5 with camera (CSI or USB) and Hailo-8/8L module installed.
- Raspberry Pi OS 64-bit (Bookworm recommended).
- HailoRT + TAPPAS (or TAPPAS Core) installed and working (`hailortcli fw-control identify`).
- GStreamer + plugins: `sudo apt install -y gstreamer1.0-tools gstreamer1.0-plugins-{base,good,bad,ugly} gstreamer1.0-gl gstreamer1.0-libav gstreamer1.0-libcamera python3-gi gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0`

Setup
1) Source Hailo/TAPPAS environment
- From the repo root: `. ./setup_env.sh`
  - This activates the appropriate venv and environment variables.

2) Ensure model and post-process libs
- Provide a HEF file (compiled model), e.g. `/path/to/yolov8m.hef` (Hailo-8) or `/path/to/yolov8s_h8l.hef` (Hailo-8L).
- Ensure these post-process shared libs are present (from TAPPAS). Either copy or symlink into the repo, or adjust paths in `detection_pipeline.py`:
  - `libyolo_hailortpp_postprocess.so`
  - `libwhole_buffer.so`

Quick Start (no AWS)
- One command launcher (starts camera, inference, and viewer):
  - `bash run_local.sh --hef /full/path/to/model.hef`

Manual Run (no AWS)
- Terminal A (camera → shared memory):
  - `bash ./producer-camera.sh`
- Terminal B (inference + overlay; skips AWS):
  - `DISABLE_AWS=1 python3 detection.py --hef-path /full/path/to/model.hef`
- Terminal C (view post-inference stream):
  - `bash ./consumer-infered-fpsdisplaysink.sh`

Notes
- If architecture auto-detection fails, add `--arch hailo8` or `--arch hailo8l` to the Python command.
- The Python pipeline will use the `--hef-path` you provide (recommended). If omitted, it tries to auto-pick based on Hailo arch.
- `DISABLE_AWS=1` ensures no AWS dependencies are needed.
- To view the raw camera feed instead, use `bash ./consumer-raw-fpsdisplaysink.sh`.

Troubleshooting
- `libcamerasrc` missing: `sudo apt install gstreamer1.0-libcamera`, enable camera, and reboot if needed.
- HEF not found: Pass a valid HEF path with `--hef-path`.
- Post-process `.so` not found: Point `detection_pipeline.py` to your TAPPAS libs or place them in the repo root.
- No video: Verify the raw feed first with `consumer-raw-fpsdisplaysink.sh`; then check HEF and `.so` paths.

