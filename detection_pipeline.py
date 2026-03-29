import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
from gi.repository import Gst, GLib, GObject, GstBase
import os
import sys
import setproctitle
import numpy as np
import cv2
import hailo

from hailo_apps.hailo_app_python.core.common.installation_utils import detect_hailo_arch
from hailo_apps.hailo_app_python.core.common.core import get_default_parser
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import (
    GStreamerApp,
    app_callback_class,
    dummy_callback,
)


# -----------------------------------------------------------------------------------------------
# Custom overlay element: GREEN for Brownie, RED for other cats
# GstBaseTransform gets writable buffers natively (no pad probe issues)
# -----------------------------------------------------------------------------------------------

class ColorOverlay(GstBase.BaseTransform):
    __gstmetadata__ = (
        'ColorOverlay',
        'Filter/Effect/Video',
        'Draw colored detection bounding boxes (green=Brownie, red=cat)',
        'cat-detect',
    )

    __gsttemplates__ = (
        Gst.PadTemplate.new(
            "sink",
            Gst.PadDirection.SINK,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.from_string("video/x-raw,format=RGB"),
        ),
        Gst.PadTemplate.new(
            "src",
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.from_string("video/x-raw,format=RGB"),
        ),
    )

    def __init__(self):
        super().__init__()
        self.width = 0
        self.height = 0
        self.set_in_place(True)
        self.set_passthrough(False)

    def do_set_caps(self, incaps, outcaps):
        struct = incaps.get_structure(0)
        self.width = struct.get_value("width")
        self.height = struct.get_value("height")
        return True

    def do_transform_ip(self, buffer):
        if self.width == 0 or self.height == 0:
            return Gst.FlowReturn.OK

        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
        if not detections:
            return Gst.FlowReturn.OK

        success, map_info = buffer.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE)
        if not success:
            return Gst.FlowReturn.OK

        try:
            frame = np.ndarray(
                shape=(self.height, self.width, 3),
                dtype=np.uint8,
                buffer=map_info.data,
            )

            for detection in detections:
                label = detection.get_label()
                confidence = detection.get_confidence()
                bbox = detection.get_bbox()

                x1 = max(0, int(bbox.xmin() * self.width))
                y1 = max(0, int(bbox.ymin() * self.height))
                x2 = min(self.width, int((bbox.xmin() + bbox.width()) * self.width))
                y2 = min(self.height, int((bbox.ymin() + bbox.height()) * self.height))

                # GREEN for Brownie, RED for other cats (RGB order)
                if label == "Brownie":
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)

                text = f"{label} {confidence:.0%}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                thickness = 3
                (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                cv2.rectangle(
                    frame,
                    (x1, y1 - th - baseline - 6),
                    (x1 + tw + 4, y1),
                    color,
                    -1,
                )
                cv2.putText(frame, text, (x1 + 2, y1 - baseline - 3),
                            font, font_scale, (0, 0, 0), thickness + 1)
                cv2.putText(frame, text, (x1 + 2, y1 - baseline - 3),
                            font, font_scale, (255, 255, 255), thickness)
        finally:
            buffer.unmap(map_info)

        return Gst.FlowReturn.OK


GObject.type_register(ColorOverlay)
Gst.Element.register(None, "coloroverlay", Gst.Rank.NONE, ColorOverlay)


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

        # Post-processing shared objects (check repo dir, then system paths)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.post_process_so = os.path.join(script_dir, 'libyolo_hailortpp_postprocess.so')
        self.cropper_process_so = os.path.join(script_dir, 'libwhole_buffer.so')

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
            + f"hailonet hef-path={self.hef_path} batch-size=1 scheduling-algorithm=1 vdevice-group-id=shared {self.thresholds_str} ! "
            + "queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! "
            + f"hailofilter so-path={self.post_process_so} qos=false ! "
            + "queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! agg1. "
            + "agg1. ! hailotracker name=hailo_tracker keep-tracked-frames=3 keep-new-frames=3 keep-lost-frames=3 ! "
            + "queue name=queue_user_callback leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! "
            + "identity name=identity_callback signal-handoffs=true ! "
            + "queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! "
            + "coloroverlay ! "
            + "queue leaky=no max-size-buffers=30 max-size-bytes=0 max-size-time=0 ! "
            + "videoconvert n-threads=2 qos=false ! "
            + "shmsink socket-path=/tmp/infered.feed sync=false wait-for-connection=false"
        )
        print(pipeline, file=sys.stderr)
        return pipeline

if __name__ == "__main__":
    user_data = app_callback_class()
    app_callback = dummy_callback
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
