"""Per-camera processing — shared by the cv2 viewer and the headless web loop.

Inputs: a `CameraConfig` (with calibration), a raw frame, a `Detector`, and the
set of COCO class ids we care about.

Outputs: an annotated frame (for display) plus a list of `Detection`s already
projected onto the floor plan via the camera's homography.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from marauders_map.detector import Detector
from marauders_map.geometry import foot_point, project_point
from marauders_map.house import CameraConfig


@dataclass
class Detection:
    cam_id: str
    local_id: int
    cls: int
    bbox: tuple[float, float, float, float]
    floor_xy: tuple[float, float]
    embedding: np.ndarray | None = None


def process_camera(
    cam: CameraConfig,
    frame: np.ndarray,
    detector: Detector,
    target_classes: set[int],
) -> tuple[np.ndarray, list[Detection]]:
    tracked = detector.track(frame)
    annotated = tracked.annotated
    cv2.putText(
        annotated, cam.name, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
    )

    detections: list[Detection] = []
    H = cam.calibration.homography
    if H is not None:
        for d in tracked.detections:
            if d.cls not in target_classes:
                continue
            floor_xy = project_point(H, foot_point(d.bbox))
            detections.append(Detection(
                cam.id, d.track_id, d.cls, d.bbox, floor_xy, d.embedding,
            ))
    return annotated, detections
