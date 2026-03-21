from pathlib import Path
import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import json

from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import get_caps_from_pad, get_numpy_from_buffer
from detection_pipeline import GStreamerDetectionApp, app_callback_class

# Make AWS optional for local runs
try:
    import boto3  # type: ignore
except Exception:  # boto3 is optional
    boto3 = None

# Set to keep track of emitted tracking IDs
emitted_ids = set()
MAX_TRACKED_IDS = 10000

# Define the target classes for detection
# this should align to the COCO list of objects
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
# note that this just filters which ones we'll call MQTT on, they'll still appear as bounding boxes because we haven't
# told hailo to exclude them.
target_classes = {'cat'}


# Setup the IoT client for sending MQTT (optional)
DISABLE_AWS = (
    os.getenv("DISABLE_AWS", "0") in ("1", "true", "True")
    or os.getenv("AWS_DISABLED", "0") in ("1", "true", "True")
)

topic = "detections"
client = None
if not DISABLE_AWS and boto3 is not None:
    access_key = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_REGION")
    if access_key and secret_key and region:
        try:
            client = boto3.client(
                "iot-data",
                region_name=region,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
            )
        except Exception:
            client = None
else:
    # Explicitly disabled or boto3 missing
    client = None

# -----------------------------------------------------------------------------------------------
# Brownie identification setup
# -----------------------------------------------------------------------------------------------
brownie_id = None
BROWNIE_EMBEDDING = os.environ.get(
    "BROWNIE_EMBEDDING",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "brownie_embedding.npy"),
)
BROWNIE_MODEL = os.environ.get(
    "BROWNIE_MODEL",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "resnet_v1_50_h8l.hef"),
)
BROWNIE_BACKEND = os.environ.get("BROWNIE_BACKEND", "simple")
BROWNIE_THRESHOLD = float(os.environ.get("BROWNIE_THRESHOLD", "0.30"))
DISABLE_BROWNIE = os.environ.get("DISABLE_BROWNIE", "0") in ("1", "true", "True")

if not DISABLE_BROWNIE:
    _missing = []
    if not os.path.exists(BROWNIE_EMBEDDING):
        _missing.append(f"embedding not found at {BROWNIE_EMBEDDING}")
    if BROWNIE_BACKEND != "simple" and not os.path.exists(BROWNIE_MODEL):
        _missing.append(f"model not found at {BROWNIE_MODEL}")

    if not _missing:
        try:
            from brownie_identifier import BrownieIdentifier
            brownie_id = BrownieIdentifier(
                embedding_path=BROWNIE_EMBEDDING,
                model_path=BROWNIE_MODEL,
                backend=BROWNIE_BACKEND,
                similarity_threshold=BROWNIE_THRESHOLD,
                standalone=True,
            )
            if brownie_id.reference_embedding is not None:
                print(f"\n{'='*60}", file=sys.stderr)
                print(f"  BROWNIE ID ACTIVE: backend={BROWNIE_BACKEND}, "
                      f"dim={brownie_id.reference_embedding.shape[0]}, "
                      f"threshold={BROWNIE_THRESHOLD}", file=sys.stderr)
                print(f"{'='*60}\n", file=sys.stderr)
            else:
                print(f"\n{'='*60}", file=sys.stderr)
                print(f"  BROWNIE ID DISABLED: no valid reference embedding", file=sys.stderr)
                print(f"  Run: python3 compute_embeddings.py "
                      f"--photos-dir ~/brownie_photos/ --backend simple", file=sys.stderr)
                print(f"{'='*60}\n", file=sys.stderr)
                brownie_id = None
        except Exception as e:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"  BROWNIE ID FAILED: {e}", file=sys.stderr)
            print(f"{'='*60}\n", file=sys.stderr)
            brownie_id = None
    else:
        for m in _missing:
            print(f"Brownie {m} — Brownie ID disabled.")

# -----------------------------------------------------------------------------------------------
# Motion detection configuration
# -----------------------------------------------------------------------------------------------
ENABLE_MOTION = os.environ.get("ENABLE_MOTION", "0") not in ("0", "false", "False")
MOTION_THRESHOLD = int(os.environ.get("MOTION_THRESHOLD", "25"))
MOTION_MIN_AREA = int(os.environ.get("MOTION_MIN_AREA", "1500"))
MOTION_COOLDOWN_SECS = float(os.environ.get("MOTION_COOLDOWN", "3.0"))
_motion_prev_gray = None
_motion_cooldown = 0
_motion_cooldown_frames = int(MOTION_COOLDOWN_SECS * 30)

if ENABLE_MOTION:
    print(f"Motion detection enabled (threshold={MOTION_THRESHOLD}, min_area={MOTION_MIN_AREA}, cooldown={MOTION_COOLDOWN_SECS}s)")

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        # Always enable frame access so we can draw bounding boxes via OpenCV
        self.use_frame = True

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

def _relabel_detection(roi, detection, new_label):
    """Replace a detection's label by removing and re-adding with the new label.
    Needed because HailoDetection only exposes get_label(), not set_label()."""
    bbox = detection.get_bbox()
    confidence = detection.get_confidence()
    # Preserve sub-objects (tracking IDs, classifications, etc.)
    sub_objects = detection.get_objects()

    # Remove original detection
    roi.remove_object(detection)

    # Create new detection with updated label
    new_det = hailo.HailoDetection(
        bbox, new_label, confidence
    )
    # Re-attach sub-objects
    for obj in sub_objects:
        new_det.add_object(obj)

    roi.add_object(new_det)
    return new_det


def _detect_motion(frame):
    """Detect motion using frame differencing. Returns True if motion detected."""
    global _motion_prev_gray, _motion_cooldown

    small = cv2.resize(frame, (320, 180))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if _motion_prev_gray is None:
        _motion_prev_gray = gray
        return True  # First frame, assume motion

    diff = cv2.absdiff(_motion_prev_gray, gray)
    thresh = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    changed = cv2.countNonZero(thresh)
    _motion_prev_gray = gray

    if changed > MOTION_MIN_AREA:
        _motion_cooldown = _motion_cooldown_frames
        return True

    if _motion_cooldown > 0:
        _motion_cooldown -= 1
        return True

    return False


def _draw_detection(frame, detection, width, height):
    """Draw a bounding box and label on the frame using OpenCV."""
    bbox = detection.get_bbox()
    x1 = max(0, int(bbox.xmin() * width))
    y1 = max(0, int(bbox.ymin() * height))
    x2 = min(width, int((bbox.xmin() + bbox.width()) * width))
    y2 = min(height, int((bbox.ymin() + bbox.height()) * height))

    label = detection.get_label()
    confidence = detection.get_confidence()
    text = f"{label} {confidence:.2f}"

    # Green box for Brownie, blue for other cats
    color = (0, 255, 0) if label == "Brownie" else (0, 128, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    # Draw label background
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
    cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Increment frame count
    user_data.increment()

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # Retrieve the video frame if required
    frame = None
    if user_data.use_frame and format and width and height:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Motion detection - skip processing and clear overlay if no motion
    if ENABLE_MOTION and frame is not None:
        if not _detect_motion(frame):
            roi = hailo.get_roi_from_buffer(buffer)
            for det in roi.get_objects_typed(hailo.HAILO_DETECTION):
                roi.remove_object(det)
            return Gst.PadProbeReturn.OK

    # Extract detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    hailo_detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Filter detections to include only target classes
    filtered_detections = [
        detection for detection in hailo_detections
        if detection.get_label() in target_classes
    ]

    # Remove non-target detections from ROI so overlay only draws cats
    for detection in hailo_detections:
        if detection.get_label() not in target_classes:
            roi.remove_object(detection)

    for detection in filtered_detections:
        label = detection.get_label()

        # --- Brownie identification ---
        if brownie_id is not None:
            if frame is None:
                print(f"Brownie skip: frame is None (format={format}, w={width}, h={height})", file=sys.stderr)
            else:
                track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
                track_id = track[0].get_id() if track else None
                print(f"Brownie debug: track_id={track_id}, should_check={brownie_id.should_check(track_id) if track_id else 'N/A'}", file=sys.stderr)
        if brownie_id is not None and frame is not None:
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            track_id = track[0].get_id() if track else None

            if track_id is not None and brownie_id.should_check(track_id):
                try:
                    bbox = detection.get_bbox()
                    x1 = max(0, int(bbox.xmin() * width))
                    y1 = max(0, int(bbox.ymin() * height))
                    x2 = min(width, int((bbox.xmin() + bbox.width()) * width))
                    y2 = min(height, int((bbox.ymin() + bbox.height()) * height))

                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2].copy()
                        is_brownie, similarity = brownie_id.is_brownie(crop)
                        brownie_id.cache_result(track_id, is_brownie)
                        print(f"Brownie check: track={track_id}, sim={similarity:.3f}, match={is_brownie}", file=sys.stderr)
                        if is_brownie:
                            detection = _relabel_detection(roi, detection, "Brownie")
                            label = "Brownie"
                except Exception as e:
                    print(f"Brownie ID error: {e}", file=sys.stderr)
            elif track_id is not None and brownie_id.get_cached_result(track_id):
                detection = _relabel_detection(roi, detection, "Brownie")
                label = "Brownie"

        # --- Tracking and MQTT ---
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 0:
            continue
        tracking_id = track[0].get_id()
        # Emit event only if the tracking ID hasn't been emitted before
        if tracking_id not in emitted_ids:
            print(f"Detection!: {tracking_id} {label} {detection.get_confidence():.2f}\n")
            emitted_ids.add(tracking_id)
            if len(emitted_ids) > MAX_TRACKED_IDS:
                emitted_ids.clear()
            if client is not None:
                payload = {
                    "tracking_id": tracking_id,
                    "label": label,
                }
                try:
                    response = client.publish(topic=topic, qos=1, payload=json.dumps(payload))
                    print(f"Message published: {response}")
                except Exception as e:
                    print(f"AWS publish failed (ignored): {e}")

    # Draw bounding boxes directly on the frame via OpenCV
    if frame is not None:
        detections_to_draw = roi.get_objects_typed(hailo.HAILO_DETECTION)
        if detections_to_draw:
            for detection in detections_to_draw:
                _draw_detection(frame, detection, width, height)
            # Write the modified frame back into the GStreamer buffer
            success, map_info = buffer.map(Gst.MapFlags.WRITE)
            if success:
                frame_bytes = frame.tobytes()
                map_info.data[:len(frame_bytes)] = frame_bytes
                buffer.unmap(map_info)

    return Gst.PadProbeReturn.OK


if __name__ == "__main__":
    # Point to the hailo-rpi5-examples .env file
    env_file = Path(os.path.expanduser("~/repos/hailo-rpi5-examples/.env"))
    if env_file.exists():
        os.environ["HAILO_ENV_FILE"] = str(env_file)

    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    if client is None:
        print("AWS disabled or not configured; running locally without IoT publish.")
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
