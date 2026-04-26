"""Lightweight appearance embedding via HSV color histograms.

Not as strong as a learned ReID network (OSNet), but:
  - no extra dependencies (just numpy + cv2)
  - cheap enough to run on every detection at 30 fps
  - good enough to disambiguate differently-clothed people in the same house

Focus is on the torso region (upper middle of the bbox), which avoids face
bias and is usually the most color-distinctive part of a person.
"""
from __future__ import annotations

import cv2
import numpy as np


def histogram_embedding(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    bins: int = 16,
) -> np.ndarray:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return np.zeros(bins * 3, dtype=np.float32)
    patch = frame[y1:y2, x1:x2]
    ph = patch.shape[0]
    torso = patch[ph // 4 : ph * 3 // 4] if ph >= 8 else patch
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    chans: list[np.ndarray] = []
    for ch, rng in zip(range(3), ((0, 180), (0, 256), (0, 256))):
        hist = cv2.calcHist([hsv], [ch], None, [bins], rng)
        cv2.normalize(hist, hist)
        chans.append(hist.flatten())
    return np.concatenate(chans).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
