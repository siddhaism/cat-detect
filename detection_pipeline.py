import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import argparse
import multiprocessing
import numpy as np
import setproctitle
import cv2
import time
import hailo
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import (
    get_default_parser,
    GStreamerApp,
    app_callback_class,
    dummy_callback,
)
from hailo_apps.hailo_app_python.core.common.utils import detect_hailo_arch



# -----------------------------------------------------------------------------------------------
# User Gstreamer Application
# -----------------------------------------------------------------------------------------------

# This class inherits from the hailo_rpi_common.GStreamerApp class
class GStreamerDetectionApp(GStreamerApp):
    def __init__(self, app_callback, user_data):
        parser = get_default_parser()
        parser.add_argument(
            "--labels-json",
            default=None,
            help="Path to custom labels JSON file",
        )
        args = parser.parse_args()
        # Call the parent class constructor
        super().__init__(args, user_data)
        # Additional initialization code can be added here
        # Set Hailo parameters these parameters should be set based on the model used
        self.batch_size = 1
        self.network_width = 640
        self.network_height = 640
        self.network_format = "RGB"
        nms_score_threshold = 0.3
        nms_iou_threshold = 0.45


        # Determine the architecture if not specified
        if args.arch is None:
            detected_arch = detect_hailo_arch()
            if detected_arch is None:
                raise ValueError("Could not auto-detect Hailo architecture. Please specify --arch manually.")
            self.arch = detected_arch
            print(f"Auto-detected Hailo architecture: {self.arch}")
        else:
            self.arch = args.arch


        if args.hef_path is not None:
            self.hef_path = args.hef_path
        # Set the HEF file path based on the arch
        elif self.arch == "hailo8":
            self.hef_path = os.path.join(self.current_path, '../resources/yolov8m.hef')
        else:  # hailo8l
            self.hef_path = os.path.join(self.current_path, '../resources/yolov8s_h8l.hef')

        # Set the post-processing shared object file
        self.post_process_so = os.path.join(self.current_path, 'libyolo_hailortpp_postprocess.so')

        self.cropper_process_so = os.path.join(self.current_path, 'libwhole_buffer.so')

        # User-defined label JSON file
        self.labels_json = args.labels_json

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
        pipeline = (
            "shmsrc socket-path=/tmp/feed.raw do-timestamp=true is-live=true ! "
            + "video/x-raw, format=NV12, width=1920, height=1080, framerate=30/1 ! "
            + "videoconvert ! "
            + "video/x-raw, format=RGB, width=1920, height=1080, framerate=30/1 ! "
            + f"hailocropper so-path={self.cropper_process_so} function-name=create_crops use-letterbox=true resize-method=inter-area internal-offset=true name=cropper1 "
            + "hailoaggregator name=agg1 "
            + "cropper1. ! queue name=bypess1_q leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! agg1. "
            + "cropper1. ! "
            + "queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! "
            + "videoscale qos=false n-threads=2 ! "
            + "video/x-raw, pixel-aspect-ratio=1/1 ! "
            + "queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! "
            + f"hailonet hef-path={self.hef_path} batch-size=1 ! "
            + "queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! "
            + f"hailofilter so-path={self.post_process_so} qos=false ! "
            + "queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! agg1. "
            + "agg1. ! hailotracker name=hailo_tracker keep-tracked-frames=3 keep-new-frames=3 keep-lost-frames=3 ! "
            + "queue name=queue_user_callback leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! "
            + "identity name=identity_callback signal-handoffs=true ! "
            + "queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! "
            + "hailooverlay qos=false line-thickness=4 font-thickness=4 ! "
            + "queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! "
            + "videoconvert n-threads=2 qos=false ! "

            # fps displaysink
            #+ "fpsdisplaysink video-sink=xvimagesink name=hailo_display sync=false"

            # OR try the shared memory sink
            + "shmsink socket-path=/tmp/infered.feed sync=false wait-for-connection=false"

        )
        print(pipeline)
        return pipeline

if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = app_callback_class()

    app_callback = dummy_callback
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
