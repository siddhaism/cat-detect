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

from collections import OrderedDict

from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import get_caps_from_pad, get_numpy_from_buffer
from detection_pipeline import GStreamerDetectionApp, app_callback_class

# Make AWS optional for local runs
try:
    import boto3  # type: ignore
except Exception:  # boto3 is optional
    boto3 = None


def _env_bool(name, default=False):
    """Parse a boolean environment variable (accepts 1/true/True/0/false/False)."""
    val = os.environ.get(name, "")
    if not val:
        return default
    return val.lower() in ("1", "true", "yes")


# Set to keep track of emitted tracking IDs (OrderedDict for FIFO eviction)
emitted_ids = OrderedDict()
MAX_TRACKED_IDS = 10000

# Define the target classes for detection
target_classes = {'cat'}


# Setup the IoT client for sending MQTT (optional)
DISABLE_AWS = _env_bool("DISABLE_AWS") or _env_bool("AWS_DISABLED")

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
    client = None

# -----------------------------------------------------------------------------------------------
# Brownie identification setup (DNN backend — MobileNetV2 on CPU)
# -----------------------------------------------------------------------------------------------
brownie_id = None
BROWNIE_EMBEDDING = os.environ.get(
    "BROWNIE_EMBEDDING",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "brownie_embedding.npy"),
)
BROWNIE_THRESHOLD = float(os.environ.get("BROWNIE_THRESHOLD", "0.80"))
BROWNIE_MIN_CONFIDENCE = float(os.environ.get("BROWNIE_MIN_CONFIDENCE", "0.30"))
DISABLE_BROWNIE = _env_bool("DISABLE_BROWNIE")

BROWNIE_BACKEND = os.environ.get("BROWNIE_BACKEND", "hailo")  # hailo -> dnn -> simple

if not DISABLE_BROWNIE:
    if os.path.exists(BROWNIE_EMBEDDING):
        try:
            from brownie_identifier import BrownieIdentifier
            brownie_id = BrownieIdentifier(
                embedding_path=BROWNIE_EMBEDDING,
                backend=BROWNIE_BACKEND,
                similarity_threshold=BROWNIE_THRESHOLD,
            )
            if brownie_id.reference_embedding is not None:
                print(f"\n{'='*60}", file=sys.stderr)
                print(f"  BROWNIE ID ACTIVE: backend={brownie_id.backend}, "
                      f"dim={brownie_id.reference_embedding.shape[0]}, "
                      f"threshold={BROWNIE_THRESHOLD}", file=sys.stderr)
                print(f"{'='*60}\n", file=sys.stderr)
            else:
                brownie_id = None
        except Exception as e:
            print(f"BROWNIE ID FAILED: {e}", file=sys.stderr)
            brownie_id = None
    else:
        print(f"Brownie embedding not found at {BROWNIE_EMBEDDING} — Brownie ID disabled.")

# -----------------------------------------------------------------------------------------------
# Motion detection configuration
# -----------------------------------------------------------------------------------------------
ENABLE_MOTION = _env_bool("ENABLE_MOTION")
MOTION_THRESHOLD = int(os.environ.get("MOTION_THRESHOLD", "25"))
MOTION_MIN_AREA = int(os.environ.get("MOTION_MIN_AREA", "1500"))
MOTION_COOLDOWN_SECS = float(os.environ.get("MOTION_COOLDOWN", "3.0"))
_motion_prev_gray = None
_motion_cooldown = 0
_motion_cooldown_frames = int(MOTION_COOLDOWN_SECS * 30)

# Crop saving mode: set SAVE_CROPS=1 to save detected cat crops to disk
SAVE_CROPS = _env_bool("SAVE_CROPS")
SAVE_CROPS_DIR = os.environ.get("SAVE_CROPS_DIR", os.path.expanduser("~/brownie_crops_live"))
_save_crop_count = 0

if SAVE_CROPS:
    os.makedirs(SAVE_CROPS_DIR, exist_ok=True)
    print(f"CROP SAVING ENABLED: saving cat crops to {SAVE_CROPS_DIR}")
    print(f"  Hold up Brownie's printout, then recompute with:")
    print(f"  python3 compute_embeddings.py --photos-dir {SAVE_CROPS_DIR} --backend hailo")

if ENABLE_MOTION:
    print(f"Motion detection enabled (threshold={MOTION_THRESHOLD}, min_area={MOTION_MIN_AREA}, cooldown={MOTION_COOLDOWN_SECS}s)")

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        if brownie_id is not None or ENABLE_MOTION:
            self.use_frame = True

# -----------------------------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------------------------

def _relabel_detection(roi, detection, new_label):
    """Replace a detection's label by removing and re-adding with the new label."""
    bbox = detection.get_bbox()
    confidence = detection.get_confidence()
    sub_objects = detection.get_objects()
    roi.remove_object(detection)
    new_det = hailo.HailoDetection(bbox, new_label, confidence)
    for obj in sub_objects:
        new_det.add_object(obj)
    roi.add_object(new_det)
    return new_det


def _detect_motion(frame):
    """Detect motion using frame differencing."""
    global _motion_prev_gray, _motion_cooldown

    small = cv2.resize(frame, (320, 180))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if _motion_prev_gray is None:
        _motion_prev_gray = gray
        return True

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


# -----------------------------------------------------------------------------------------------
# Main callback
# -----------------------------------------------------------------------------------------------

def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    format, width, height = get_caps_from_pad(pad)

    # Map buffer for read access
    frame = None
    if format and width and height:
        channels = 3
        expected_size = height * width * channels
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if success and len(map_info.data) >= expected_size:
            frame = np.ndarray(
                shape=(height, width, channels),
                dtype=np.uint8,
                buffer=map_info.data,
            ).copy()
            buffer.unmap(map_info)
        elif success:
            buffer.unmap(map_info)

    # Motion detection
    if ENABLE_MOTION and frame is not None:
        if not _detect_motion(frame):
            roi = hailo.get_roi_from_buffer(buffer)
            for det in roi.get_objects_typed(hailo.HAILO_DETECTION):
                roi.remove_object(det)
            return Gst.PadProbeReturn.OK

    roi = hailo.get_roi_from_buffer(buffer)
    hailo_detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Remove non-cat detections
    for detection in hailo_detections:
        if detection.get_label() not in target_classes:
            roi.remove_object(detection)

    # Brownie identification on remaining cat detections
    if brownie_id is not None and frame is not None:
        for detection in roi.get_objects_typed(hailo.HAILO_DETECTION):
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            track_id = track[0].get_id() if track else None
            if track_id is None:
                continue
            # Skip low-confidence detections — crops are too noisy for ID
            if detection.get_confidence() < BROWNIE_MIN_CONFIDENCE:
                continue
            if brownie_id.should_check(track_id):
                try:
                    bbox = detection.get_bbox()
                    x1 = max(0, int(bbox.xmin() * width))
                    y1 = max(0, int(bbox.ymin() * height))
                    x2 = min(width, int((bbox.xmin() + bbox.width()) * width))
                    y2 = min(height, int((bbox.ymin() + bbox.height()) * height))
                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2].copy()
                        # Save crops when in capture mode
                        if SAVE_CROPS:
                            global _save_crop_count
                            crop_path = os.path.join(SAVE_CROPS_DIR, f"crop_{_save_crop_count:04d}.jpg")
                            cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                            _save_crop_count += 1
                            if _save_crop_count % 5 == 0:
                                print(f"Saved {_save_crop_count} crops to {SAVE_CROPS_DIR}")
                        # Submit to background thread (non-blocking)
                        brownie_id.submit_async(track_id, crop)
                except Exception as e:
                    print(f"Brownie ID error: {e}", file=sys.stderr)
            elif brownie_id.get_cached_result(track_id):
                _relabel_detection(roi, detection, "Brownie")

    # Tracking and MQTT
    for detection in roi.get_objects_typed(hailo.HAILO_DETECTION):
        label = detection.get_label()
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 0:
            continue
        tracking_id = track[0].get_id()
        if tracking_id not in emitted_ids:
            print(f"Detection!: {tracking_id} {label} {detection.get_confidence():.2f}\n")
            emitted_ids[tracking_id] = True
            # Evict oldest entries when limit reached
            while len(emitted_ids) > MAX_TRACKED_IDS:
                emitted_ids.popitem(last=False)
            if client is not None:
                payload = {"tracking_id": tracking_id, "label": label}
                try:
                    response = client.publish(topic=topic, qos=1, payload=json.dumps(payload))
                    print(f"Message published: {response}")
                except Exception as e:
                    print(f"AWS publish failed (ignored): {e}")

    return Gst.PadProbeReturn.OK


if __name__ == "__main__":
    env_file = Path(os.path.expanduser("~/repos/hailo-rpi5-examples/.env"))
    if env_file.exists():
        os.environ["HAILO_ENV_FILE"] = str(env_file)

    user_data = user_app_callback_class()
    if client is None:
        print("AWS disabled or not configured; running locally without IoT publish.")
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
