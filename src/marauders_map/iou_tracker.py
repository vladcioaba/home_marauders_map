"""Tiny IoU-based tracker for backends that don't ship their own.

Simple enough to not need external deps. Good enough for the Hailo backend
scaffolding — swap in a real ByteTrack/BoT-SORT implementation when you
deploy and measure.

Algorithm:
  For each new detection, greedy-match to the existing track with the
  highest IoU above `iou_threshold`. Unmatched detections spawn new tracks.
  Tracks missed for `max_missed` consecutive frames are dropped.
"""
from __future__ import annotations

from dataclasses import dataclass


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


@dataclass
class _Track:
    id: int
    bbox: tuple[float, float, float, float]
    cls: int
    conf: float
    missed: int = 0


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 5):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(
        self,
        boxes: list[tuple[float, float, float, float]],
        classes: list[int],
        confs: list[float],
    ) -> list[tuple[int, int]]:
        """Returns list of (detection_index, track_id) in input order."""
        assignments: list[tuple[int, int]] = []
        updated_ids: set[int] = set()

        for i, bbox in enumerate(boxes):
            best, best_iou = None, self.iou_threshold
            for tr in self._tracks:
                if tr.id in updated_ids:
                    continue
                iou = _iou(bbox, tr.bbox)
                if iou > best_iou:
                    best_iou, best = iou, tr
            if best is not None:
                best.bbox = bbox
                best.cls = classes[i]
                best.conf = confs[i]
                best.missed = 0
                updated_ids.add(best.id)
                assignments.append((i, best.id))
            else:
                new = _Track(id=self._next_id, bbox=bbox, cls=classes[i], conf=confs[i])
                self._next_id += 1
                self._tracks.append(new)
                updated_ids.add(new.id)
                assignments.append((i, new.id))

        for tr in self._tracks:
            if tr.id not in updated_ids:
                tr.missed += 1
        self._tracks = [tr for tr in self._tracks if tr.missed <= self.max_missed]
        return assignments
