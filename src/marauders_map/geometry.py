"""Homography + foot-point helpers for mapping detections to the floor plan."""
from __future__ import annotations

import cv2
import numpy as np


def compute_homography(image_pts: np.ndarray, floor_pts: np.ndarray) -> np.ndarray | None:
    """Homography from image pixel coords to floor-plan meter coords.

    Needs ≥4 non-collinear correspondences. Returns None if OpenCV fails.
    """
    if image_pts.shape[0] < 4 or image_pts.shape != floor_pts.shape:
        return None
    H, _ = cv2.findHomography(
        image_pts.astype(np.float32),
        floor_pts.astype(np.float32),
    )
    return H


def project_point(H: np.ndarray, pt: tuple[float, float]) -> tuple[float, float]:
    v = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
    out = H @ v
    w = out[2] if out[2] != 0 else 1e-9
    return float(out[0] / w), float(out[1] / w)


def foot_point(bbox_xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    """Bottom-center of a bounding box — the ground contact assumed for a person."""
    x1, _, x2, y2 = bbox_xyxy
    return ((x1 + x2) * 0.5, y2)
