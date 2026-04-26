"""Backend-agnostic detection + tracking interface.

Each backend produces a `TrackedFrame`: the pre-annotated image plus a list of
`Detection`s. Downstream code (foot-point projection, global tracker, logger,
floor-plan renderer) is backend-independent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Detection:
    track_id: int
    cls: int
    conf: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    embedding: np.ndarray | None = None  # appearance descriptor (optional)


@dataclass
class TrackedFrame:
    annotated: np.ndarray
    detections: list[Detection]


class Detector(Protocol):
    def track(self, frame: np.ndarray) -> TrackedFrame: ...
    def close(self) -> None: ...


def make_detector(
    backend: str,
    *,
    model: str,
    tracker: str,
    conf: float,
    device: str | None,
    imgsz: int = 640,
) -> Detector:
    if backend == "ultralytics":
        from marauders_map.backends.ultralytics_backend import UltralyticsDetector
        return UltralyticsDetector(
            model=model, tracker=tracker, conf=conf, device=device, imgsz=imgsz,
        )
    if backend == "hailo":
        from marauders_map.backends.hailo_backend import HailoDetector
        return HailoDetector(
            model=model, tracker=tracker, conf=conf, device=device, imgsz=imgsz,
        )
    raise ValueError(f"unknown backend: {backend!r} (expected 'ultralytics' or 'hailo')")
