"""Detector backend: Ultralytics YOLO + built-in tracker (ByteTrack/BoT-SORT).

Used on Mac and any host with PyTorch available.
"""
from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from marauders_map.detector import Detection, TrackedFrame
from marauders_map.reid import histogram_embedding


class UltralyticsDetector:
    def __init__(
        self, *, model: str, tracker: str, conf: float,
        device: str | None, imgsz: int = 640,
    ):
        self._model = YOLO(model)
        self._tracker = tracker
        self._conf = conf
        self._device = device
        self._imgsz = imgsz

    def track(self, frame: np.ndarray) -> TrackedFrame:
        results = self._model.track(
            frame, persist=True, tracker=self._tracker, conf=self._conf,
            device=self._device, imgsz=self._imgsz, verbose=False,
        )
        r = results[0]
        annotated = r.plot()

        detections: list[Detection] = []
        if r.boxes is not None and r.boxes.id is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            classes = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for bbox, tid, cls, c in zip(xyxy, ids, classes, confs):
                bb = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
                detections.append(Detection(
                    track_id=int(tid),
                    cls=int(cls),
                    conf=float(c),
                    bbox=bb,
                    embedding=histogram_embedding(frame, bb),
                ))
        return TrackedFrame(annotated=annotated, detections=detections)

    def close(self) -> None:
        pass
