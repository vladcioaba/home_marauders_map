"""Detector backend: Hailo-10H on Raspberry Pi 5 + AI HAT+2.

STATUS: SCAFFOLDED but UNTESTED. I do not have access to Hailo hardware.
The inference path follows the public HailoRT Python API; expect to adjust
layer names, input format, and YOLO output decoding for your specific
compiled .hef. Everything downstream (tracker, floor plan, logger) uses
portable code and works today.

Working without Hailo hardware:
  - `IoUTracker` (src/marauders_map/iou_tracker.py) — simple standalone tracker.
  - YOLO output decoder (`_decode_yolo_output`) for the two most common
    post-compile layouts (post-NMS detections and raw YOLOv8-style heads).
  - TrackedFrame assembly and box drawing.

To enable on the Pi:
  1. On the Pi: `pip install hailo-platform` inside the project venv.
  2. Compile a YOLO model to `.hef` via https://github.com/hailo-ai/hailo_model_zoo
     (YOLO26 is the target per the original LinkedIn announcement).
  3. Dump the HEF's layer names (`hailo parse <file>.hef`) and update
     `_INPUT_NAME` / `_OUTPUT_NAME` below to match.
  4. Inspect the raw output shape once with `np.save`; if it's neither
     [N, 6] (post-NMS) nor [N, 85] (YOLOv8 head), extend `_decode_yolo_output`.
  5. Run: `PI_HOST=pi@host BACKEND=hailo ./scripts/deploy_to_pi.sh`.
"""
from __future__ import annotations

import cv2
import numpy as np

from marauders_map.detector import Detection, TrackedFrame
from marauders_map.iou_tracker import IoUTracker

# Tune these to your compiled .hef.
_INPUT_NAME = "input_layer"
_OUTPUT_NAME = "output_layer"
_INPUT_SIZE = (640, 640)
_PERSON_CLASS = 0


class HailoDetector:
    def __init__(
        self, *, model: str, tracker: str, conf: float,
        device: str | None, imgsz: int = 640,
    ):
        try:
            from hailo_platform import (  # type: ignore
                HEF, VDevice, HailoStreamInterface, ConfigureParams,
                InputVStreamParams, OutputVStreamParams, FormatType,
            )
        except ImportError as e:
            raise RuntimeError(
                "hailo-platform not installed. On the Pi, inside the project venv:\n"
                "  pip install hailo-platform\n"
                "See src/marauders_map/backends/hailo_backend.py docstring for the full setup."
            ) from e

        self._conf = conf
        self._input_size = (imgsz, imgsz) if imgsz else _INPUT_SIZE
        self._tracker = IoUTracker()

        self._hef = HEF(model)
        self._target = VDevice()
        params = ConfigureParams.create_from_hef(
            hef=self._hef, interface=HailoStreamInterface.PCIe,
        )
        self._network_group = self._target.configure(self._hef, params)[0]
        self._input_params = InputVStreamParams.make(
            self._network_group, format_type=FormatType.FLOAT32,
        )
        self._output_params = OutputVStreamParams.make(
            self._network_group, format_type=FormatType.FLOAT32,
        )
        self._activation_ctx = self._network_group.activate()
        self._activation_ctx.__enter__()

    def track(self, frame: np.ndarray) -> TrackedFrame:
        from hailo_platform import InferVStreams  # type: ignore

        preproc, scale, pad_x, pad_y = _letterbox(frame, self._input_size)
        with InferVStreams(
            self._network_group, self._input_params, self._output_params,
        ) as streams:
            raw = streams.infer({_INPUT_NAME: preproc})
        output = np.asarray(raw[_OUTPUT_NAME])

        boxes, classes, confs = _decode_yolo_output(
            output, self._conf, scale, pad_x, pad_y, frame.shape[:2],
        )
        assignments = self._tracker.update(boxes, classes, confs)

        annotated = frame.copy()
        detections: list[Detection] = []
        for i, tid in assignments:
            x1, y1, x2, y2 = boxes[i]
            detections.append(Detection(
                track_id=tid, cls=classes[i], conf=confs[i],
                bbox=(x1, y1, x2, y2),
            ))
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 2)
            cv2.putText(annotated, f"id{tid}", (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return TrackedFrame(annotated=annotated, detections=detections)

    def close(self) -> None:
        try:
            self._activation_ctx.__exit__(None, None, None)
        except Exception:
            pass


def _letterbox(
    frame: np.ndarray, size: tuple[int, int],
) -> tuple[np.ndarray, float, int, int]:
    iw, ih = size
    h, w = frame.shape[:2]
    scale = min(iw / w, ih / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.full((ih, iw, 3), 114, dtype=np.uint8)
    pad_x = (iw - new_w) // 2
    pad_y = (ih - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.expand_dims(img, 0), scale, pad_x, pad_y


def _decode_yolo_output(
    output: np.ndarray,
    conf_threshold: float,
    scale: float,
    pad_x: int,
    pad_y: int,
    frame_shape: tuple[int, int],
) -> tuple[list[tuple[float, float, float, float]], list[int], list[float]]:
    """Decode two common Hailo YOLO output layouts:
       - [N, 6]  post-NMS: [x1, y1, x2, y2, conf, cls]
       - [N, 85] raw YOLOv8 head: [cx, cy, w, h, obj, cls0..79]
    If your .hef produces something else, extend this.
    """
    if output.ndim == 3:
        output = output[0]

    boxes: list[tuple[float, float, float, float]] = []
    classes: list[int] = []
    confs: list[float] = []
    h_f, w_f = frame_shape
    for row in output:
        if row.shape[0] == 6:
            x1, y1, x2, y2, c, cls = row
            cls_i = int(cls)
        elif row.shape[0] >= 85:
            cx, cy, w, h, obj = row[:5]
            cls_scores = row[5:]
            cls_i = int(np.argmax(cls_scores))
            c = obj * cls_scores[cls_i]
            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = cx + w / 2, cy + h / 2
        else:
            continue
        if c < conf_threshold:
            continue
        x1 = float((x1 - pad_x) / scale)
        y1 = float((y1 - pad_y) / scale)
        x2 = float((x2 - pad_x) / scale)
        y2 = float((y2 - pad_y) / scale)
        x1 = max(0.0, min(w_f - 1.0, x1))
        y1 = max(0.0, min(h_f - 1.0, y1))
        x2 = max(0.0, min(w_f - 1.0, x2))
        y2 = max(0.0, min(h_f - 1.0, y2))
        boxes.append((x1, y1, x2, y2))
        classes.append(cls_i)
        confs.append(float(c))
    return boxes, classes, confs
