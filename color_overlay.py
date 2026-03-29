"""Custom GStreamer element that draws colored bounding boxes.

GREEN for "Brownie", RED for other cats.

GstBaseTransform elements get writable buffers natively, avoiding the
"non-writable buffer" issue that pad probes have.

Usage in pipeline string:
    "coloroverlay ! ..."

Must be registered before pipeline creation by importing this module.
"""

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
from gi.repository import Gst, GObject, GstBase
import numpy as np
import cv2
import hailo


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

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)

                # Label text with background
                text = f"{label} {confidence:.0%}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                thickness = 3
                (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                # Colored background bar
                cv2.rectangle(
                    frame,
                    (x1, y1 - th - baseline - 6),
                    (x1 + tw + 4, y1),
                    color,
                    -1,
                )
                # White text with dark outline for readability
                cv2.putText(frame, text, (x1 + 2, y1 - baseline - 3),
                            font, font_scale, (0, 0, 0), thickness + 1)
                cv2.putText(frame, text, (x1 + 2, y1 - baseline - 3),
                            font, font_scale, (255, 255, 255), thickness)
        finally:
            buffer.unmap(map_info)

        return Gst.FlowReturn.OK


# Register the element so it can be used in pipeline strings
GObject.type_register(ColorOverlay)
Gst.Element.register(None, "coloroverlay", Gst.Rank.NONE, ColorOverlay)
