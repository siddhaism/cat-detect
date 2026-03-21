import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import setproctitle
import hailo

from hailo_apps.hailo_app_python.core.common.installation_utils import detect_hailo_arch
from hailo_apps.hailo_app_python.core.common.core import get_default_parser
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import (
    GStreamerApp,
    app_callback_class,
    dummy_callback,
)


# -----------------------------------------------------------------------------------------------
# User Gstreamer Application
# -----------------------------------------------------------------------------------------------

# Custom detection pipeline: reads from shared memory, outputs to shared memory with overlays
class GStreamerDetectionApp(GStreamerApp):
    def __init__(self, app_callback, user_data, parser=None):
        if parser is None:
            parser = get_default_parser()
        parser.add_argument(
            "--labels-json",
            default=None,
            help="Path to custom labels JSON file",
        )
        parser.add_argument(
            "--embedding-hef",
            default=None,
            help="Path to embedding model HEF (e.g., resnet_v1_50_h8l.hef) for Brownie identification",
        )
        # Call the parent class constructor (pass parser, it calls parse_args internally)
        super().__init__(parser, user_data)

        # Set Hailo parameters based on the model used
        self.batch_size = 1
        self.network_width = 640
        self.network_height = 640
        self.network_format = "RGB"
        nms_score_threshold = 0.15
        nms_iou_threshold = 0.45

        # Determine the architecture if not specified
        if self.options_menu.arch is None:
            detected_arch = detect_hailo_arch()
            if detected_arch is None:
                raise ValueError("Could not auto-detect Hailo architecture. Please specify --arch manually.")
            self.arch = detected_arch
            print(f"Auto-detected Hailo architecture: {self.arch}")
        else:
            self.arch = self.options_menu.arch

        if self.options_menu.hef_path is not None:
            self.hef_path = self.options_menu.hef_path
        # Set the HEF file path based on the arch
        elif self.arch == "hailo8":
            self.hef_path = os.path.join(self.current_path, '../resources/yolov8m.hef')
        else:  # hailo8l
            self.hef_path = os.path.join(self.current_path, '../resources/yolov8s_h8l.hef')

        # Embedding model HEF (optional, for Brownie identification in-pipeline)
        self.embedding_hef = self.options_menu.embedding_hef

        # Post-processing shared objects (check repo dir, then system paths)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.post_process_so = os.path.join(script_dir, 'libyolo_hailortpp_postprocess.so')
        self.cropper_process_so = os.path.join(script_dir, 'libwhole_buffer.so')

        # Detection-based cropper SO (from TAPPAS post-process dir)
        tappas_post_proc = os.environ.get(
            "TAPPAS_POST_PROC_DIR",
            "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes",
        )
        self.detection_crop_so = None
        for candidate in [
            os.path.join(tappas_post_proc, "cropping_algorithms", "libdetection_croppers.so"),
            os.path.join(tappas_post_proc, "cropping", "libcroppers.so"),
            os.path.join(tappas_post_proc, "libcroppers.so"),
        ]:
            if os.path.exists(candidate):
                self.detection_crop_so = candidate
                break

        # User-defined label JSON file
        self.labels_json = self.options_menu.labels_json

        self.app_callback = app_callback

        self.thresholds_str = (
            f"nms-score-threshold={nms_score_threshold} "
            f"nms-iou-threshold={nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )

        # Set the process title
        setproctitle.setproctitle("Hailo Detection App")

        self.create_pipeline()

    def get_pipeline_string(self):
        # Stage 1: Detection (YOLO)
        pipeline = (
            "shmsrc socket-path=/tmp/feed.raw do-timestamp=true is-live=true ! "
            + "video/x-raw, format=NV12, width=1920, height=1080, framerate=30/1 ! "
            + "videoconvert ! "
            + "video/x-raw, format=RGB, width=1920, height=1080, framerate=30/1 ! "
            + f"hailocropper so-path={self.cropper_process_so} function-name=create_crops use-letterbox=true resize-method=inter-area internal-offset=true name=cropper1 "
            + "hailoaggregator name=agg1 "
            + "cropper1. ! queue name=bypass1_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! agg1. "
            + "cropper1. ! "
            + "queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! "
            + "videoscale qos=false n-threads=2 ! "
            + "video/x-raw, pixel-aspect-ratio=1/1 ! "
            + "queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! "
            + f"hailonet hef-path={self.hef_path} batch-size=1 scheduling-algorithm=1 {self.thresholds_str} ! "
            + "queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! "
            + f"hailofilter so-path={self.post_process_so} qos=false ! "
            + "queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! agg1. "
            + "agg1. ! hailotracker name=hailo_tracker keep-tracked-frames=3 keep-new-frames=3 keep-lost-frames=3 ! "
        )

        # Stage 2: Embedding (ResNet) — cascade on detected regions
        if self.embedding_hef and os.path.exists(self.embedding_hef):
            if self.detection_crop_so:
                print(f"Cascade pipeline: embedding model={self.embedding_hef}")
                print(f"  Detection cropper SO: {self.detection_crop_so}")
                pipeline += (
                    f"hailocropper so-path={self.detection_crop_so} function-name=all_detections use-letterbox=true resize-method=inter-area internal-offset=true name=cropper2 "
                    + "hailoaggregator name=agg2 "
                    + "cropper2. ! queue name=bypass2_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! agg2. "
                    + "cropper2. ! "
                    + "queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! "
                    + "videoscale qos=false n-threads=2 ! "
                    + "video/x-raw, pixel-aspect-ratio=1/1 ! "
                    + "queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! "
                    + f"hailonet hef-path={self.embedding_hef} batch-size=1 scheduling-algorithm=1 ! "
                    + "queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! agg2. "
                    + "agg2. ! "
                )
            else:
                print(f"WARNING: Detection cropper SO not found")
                print("  Embedding cascade disabled. Install TAPPAS cropper libraries.")
        elif self.embedding_hef:
            print(f"WARNING: Embedding HEF not found at {self.embedding_hef}")

        # Stage 3: Callback + overlay + output
        pipeline += (
            "queue name=queue_user_callback leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! "
            + "identity name=identity_callback signal-handoffs=true ! "
            + "queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! "
            + "hailooverlay qos=false line-thickness=4 font-thickness=4 ! "
            + "queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! "
            + "videoconvert n-threads=2 qos=false ! "
            + "shmsink socket-path=/tmp/infered.feed sync=false wait-for-connection=false"
        )
        print(pipeline)
        return pipeline

if __name__ == "__main__":
    user_data = app_callback_class()
    app_callback = dummy_callback
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
