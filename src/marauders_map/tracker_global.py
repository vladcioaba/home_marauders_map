"""Cross-camera identity linker.

Inputs are per-camera detections already projected onto the floor plan (via
each camera's homography). Outputs are stable global IDs that persist as a
person moves between cameras.

Strategy (single loop):
  1. Sticky mapping — once (cam_id, local_id) is linked to a global_id, keep it.
  2. Otherwise spatial filter: candidate tracks within `max_distance_m` and
     seen within `max_gap_s`. If >1 candidate and we have appearance
     embeddings, pick by highest cosine similarity (ReID tiebreaker);
     otherwise pick spatial nearest.
  3. Spawn a new global_id if nothing matches.
  4. Expire tracks silent longer than `expire_s`.

This handles both overlapping-camera fusion (spatial agreement in the same
frame) and door handoffs (brief silence then reappearance nearby on another
camera). The appearance tiebreaker reduces ID swaps when two people pass
close to each other.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PersonMarker:
    cam_id: str
    track_id: int                      # per-camera (ByteTrack) id
    floor_xy: tuple[float, float]      # meters
    embedding: np.ndarray | None = None  # appearance descriptor (optional)


@dataclass
class GlobalMarker:
    global_id: int
    cam_id: str
    track_id: int
    floor_xy: tuple[float, float]


@dataclass
class _GlobalTrack:
    id: int
    last_xy: tuple[float, float]
    last_seen: float
    history: deque                     # of (t_seconds, (x, y))
    last_emb: np.ndarray | None = None


class GlobalTracker:
    def __init__(
        self,
        *,
        max_distance_m: float = 1.5,
        max_gap_s: float = 3.0,
        trail_s: float = 3.0,
        expire_s: float = 10.0,
        use_appearance: bool = True,
    ) -> None:
        self.max_distance_m = max_distance_m
        self.max_gap_s = max_gap_s
        self.trail_s = trail_s
        self.expire_s = expire_s
        self.use_appearance = use_appearance
        self._tracks: dict[int, _GlobalTrack] = {}
        self._local_to_global: dict[tuple[str, int], int] = {}
        self._next_id = 1

    def update(self, now: float, markers: list[PersonMarker]) -> list[GlobalMarker]:
        self._expire(now)

        taken: set[int] = set()
        out: list[GlobalMarker] = []

        for m in markers:
            key = (m.cam_id, m.track_id)
            gid = self._local_to_global.get(key)
            if gid is not None and gid in self._tracks and gid not in taken:
                self._update_track(gid, m.floor_xy, m.embedding, now)
            else:
                gid = self._nearest(m.floor_xy, m.embedding, now, taken)
                if gid is None:
                    gid = self._spawn(m.floor_xy, m.embedding, now)
                else:
                    self._update_track(gid, m.floor_xy, m.embedding, now)
                self._local_to_global[key] = gid
            taken.add(gid)
            out.append(GlobalMarker(gid, m.cam_id, m.track_id, m.floor_xy))

        return out

    def trails(self, now: float) -> dict[int, list[tuple[float, float]]]:
        cutoff = now - self.trail_s
        result: dict[int, list[tuple[float, float]]] = {}
        for gid, t in self._tracks.items():
            if now - t.last_seen > self.max_gap_s:
                continue
            pts = [xy for ts, xy in t.history if ts >= cutoff]
            if len(pts) >= 2:
                result[gid] = pts
        return result

    def _expire(self, now: float) -> None:
        dead = [gid for gid, t in self._tracks.items() if now - t.last_seen > self.expire_s]
        for gid in dead:
            del self._tracks[gid]
        if dead:
            dead_set = set(dead)
            self._local_to_global = {
                k: v for k, v in self._local_to_global.items() if v not in dead_set
            }

    def _nearest(
        self,
        xy: tuple[float, float],
        emb: np.ndarray | None,
        now: float,
        taken: set[int],
    ) -> int | None:
        candidates: list[tuple[int, _GlobalTrack, float]] = []
        for gid, t in self._tracks.items():
            if gid in taken or now - t.last_seen > self.max_gap_s:
                continue
            d = math.hypot(t.last_xy[0] - xy[0], t.last_xy[1] - xy[1])
            if d < self.max_distance_m:
                candidates.append((gid, t, d))
        if not candidates:
            return None
        if len(candidates) == 1 or emb is None or not self.use_appearance:
            return min(candidates, key=lambda c: c[2])[0]
        best_gid, best_sim = None, -1.0
        for gid, t, _ in candidates:
            if t.last_emb is None:
                continue
            sim = _cos(emb, t.last_emb)
            if sim > best_sim:
                best_sim, best_gid = sim, gid
        return best_gid if best_gid is not None else min(candidates, key=lambda c: c[2])[0]

    def _spawn(
        self, xy: tuple[float, float], emb: np.ndarray | None, now: float,
    ) -> int:
        gid = self._next_id
        self._next_id += 1
        self._tracks[gid] = _GlobalTrack(
            id=gid, last_xy=xy, last_seen=now,
            history=deque([(now, xy)], maxlen=512),
            last_emb=emb.copy() if emb is not None else None,
        )
        return gid

    def _update_track(
        self,
        gid: int,
        xy: tuple[float, float],
        emb: np.ndarray | None,
        now: float,
    ) -> None:
        t = self._tracks[gid]
        t.last_xy = xy
        t.last_seen = now
        t.history.append((now, xy))
        if emb is not None:
            # EMA on the embedding to smooth over lighting/pose variation
            if t.last_emb is None:
                t.last_emb = emb.copy()
            else:
                t.last_emb = 0.8 * t.last_emb + 0.2 * emb
        cutoff = now - self.trail_s
        while t.history and t.history[0][0] < cutoff:
            t.history.popleft()


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
