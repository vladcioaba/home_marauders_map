"""Floor-plan rendering — Marauder's Map skin.

Parchment background, sepia ink for rooms/doors/cameras, footprint trails
for each tracked person. Names override the default `P{gid}` label.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from marauders_map.house import House
from marauders_map.tracker_global import GlobalMarker

# BGR colors. Tuned so the floor plan reads as old parchment with brown ink.
PARCHMENT = (180, 215, 235)
INK_DARK = (40, 70, 110)
INK_MED = (70, 110, 150)
INK_LIGHT = (110, 160, 200)
INK_FAINT = (150, 195, 220)
CAM_INK = (60, 130, 180)

# Per-person hue rotation around a sepia base so multiple people stay readable.
_FOOTPRINT_PALETTE = [
    (40, 70, 110),     # base sepia
    (40, 80, 150),     # warmer brown
    (60, 100, 90),     # forest moss
    (90, 60, 130),     # mulberry
    (120, 100, 60),    # teal-ink
    (50, 130, 130),    # olive
]


def _color(global_id: int) -> tuple[int, int, int]:
    return _FOOTPRINT_PALETTE[(global_id - 1) % len(_FOOTPRINT_PALETTE)]


def _name(global_id: int, names: dict[int, str] | None) -> str:
    if names is not None and global_id in names:
        return names[global_id]
    return f"P{global_id}"


def draw_floorplan(house: House) -> np.ndarray:
    w_m, h_m = house.floorplan.size
    s = house.floorplan.scale
    img = np.full(
        (int(round(h_m * s)), int(round(w_m * s)), 3), PARCHMENT, dtype=np.uint8,
    )

    for room in house.rooms:
        x, y, w, h = room.rect
        p1 = (int(x * s), int(y * s))
        p2 = (int((x + w) * s), int((y + h) * s))
        cv2.rectangle(img, p1, p2, INK_MED, 2, cv2.LINE_AA)
        cv2.putText(
            img, room.name, (p1[0] + 8, p1[1] + 22),
            cv2.FONT_HERSHEY_TRIPLEX, 0.5, INK_DARK, 1, cv2.LINE_AA,
        )

    for door in house.doors:
        (x1, y1), (x2, y2) = door.segment
        cv2.line(
            img, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)),
            PARCHMENT, 6,
        )
        cv2.line(
            img, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)),
            INK_FAINT, 1, cv2.LINE_AA,
        )

    for cam in house.cameras:
        cx, cy = cam.position
        cp = (int(cx * s), int(cy * s))
        cv2.circle(img, cp, 6, CAM_INK, -1, cv2.LINE_AA)
        cv2.putText(
            img, cam.id, (cp[0] + 10, cp[1] - 8),
            cv2.FONT_HERSHEY_TRIPLEX, 0.4, CAM_INK, 1, cv2.LINE_AA,
        )
        half = math.radians(cam.fov / 2)
        theta = math.radians(cam.heading)
        length = 1.5 * s
        for sign in (-1, 1):
            a = theta + sign * half
            ex = int(cp[0] + math.cos(a) * length)
            ey = int(cp[1] + math.sin(a) * length)
            cv2.line(img, cp, (ex, ey), CAM_INK, 1, cv2.LINE_AA)
        if cam.calibration.homography is None:
            cv2.putText(
                img, "uncalibrated", (cp[0] + 10, cp[1] + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1, cv2.LINE_AA,
            )

    _draw_title(img)
    return img


def _draw_title(img: np.ndarray) -> None:
    title = "I solemnly swear that I am up to no good"
    cv2.putText(
        img, title, (12, img.shape[0] - 14),
        cv2.FONT_HERSHEY_TRIPLEX, 0.5, INK_DARK, 1, cv2.LINE_AA,
    )


def draw_mischief_managed(img: np.ndarray) -> None:
    """Overlay 'Mischief Managed' centered when people are hidden."""
    text = "Mischief Managed"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 1.2, 2)
    org = ((img.shape[1] - tw) // 2, (img.shape[0] + th) // 2)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_TRIPLEX, 1.2, INK_DARK, 2, cv2.LINE_AA)


def _draw_footprint(
    img: np.ndarray,
    x_px: float,
    y_px: float,
    angle_rad: float,
    side: int,
    color: tuple[int, int, int],
    size: float = 1.0,
) -> None:
    """One small foot at (x, y) oriented along angle_rad. side=+1 right, -1 left."""
    perp = angle_rad + math.pi / 2
    offset = 4.0 * size
    ox = x_px + side * offset * math.cos(perp)
    oy = y_px + side * offset * math.sin(perp)
    axes = (max(2, int(round(5 * size))), max(1, int(round(3 * size))))
    angle_deg = math.degrees(angle_rad)
    cv2.ellipse(
        img, (int(round(ox)), int(round(oy))), axes,
        angle_deg, 0, 360, color, -1, cv2.LINE_AA,
    )


def draw_trails(
    img: np.ndarray,
    house: House,
    trails: dict[int, list[tuple[float, float]]],
) -> None:
    """Stamp footprint pairs along each person's recent path."""
    s = house.floorplan.scale
    spacing_px = max(8.0, 0.35 * s)  # ~35cm between footprints
    for gid, pts in trails.items():
        if len(pts) < 2:
            continue
        color = _color(gid)
        pixel_pts = [(x * s, y * s) for x, y in pts]
        side = 1
        accum = 0.0
        for i in range(1, len(pixel_pts)):
            x1, y1 = pixel_pts[i - 1]
            x2, y2 = pixel_pts[i]
            dx, dy = x2 - x1, y2 - y1
            seg_len = math.hypot(dx, dy)
            if seg_len < 1.0:
                continue
            angle = math.atan2(dy, dx)
            accum += seg_len
            while accum >= spacing_px:
                d_from_start = seg_len - (accum - spacing_px)
                if 0 <= d_from_start <= seg_len:
                    frac = d_from_start / seg_len
                    px = x1 + dx * frac
                    py = y1 + dy * frac
                    # fade older prints (earlier segments) by darkening less
                    _draw_footprint(img, px, py, angle, side, color, size=0.85)
                    side = -side
                accum -= spacing_px


def draw_markers(
    img: np.ndarray,
    house: House,
    markers: list[GlobalMarker],
    *,
    names: dict[int, str] | None = None,
    headings: dict[int, float] | None = None,
) -> None:
    """Current-position footprint pair + name label per person."""
    s = house.floorplan.scale
    for m in markers:
        fx, fy = m.floor_xy
        px, py = fx * s, fy * s
        c = _color(m.global_id)
        angle = (headings or {}).get(m.global_id, 0.0)
        # current-position pair, slightly larger than trail prints
        _draw_footprint(img, px, py, angle, +1, c, size=1.2)
        _draw_footprint(img, px, py, angle, -1, c, size=1.2)

        room = house.room_at(m.floor_xy)
        label = _name(m.global_id, names)
        if room:
            label += f" @ {room}"
        cv2.putText(
            img, label, (int(px) + 12, int(py) + 5),
            cv2.FONT_HERSHEY_TRIPLEX, 0.45, c, 1, cv2.LINE_AA,
        )


def headings_from_trails(
    trails: dict[int, list[tuple[float, float]]],
) -> dict[int, float]:
    """Movement direction per gid from the last two trail points (radians)."""
    out: dict[int, float] = {}
    for gid, pts in trails.items():
        if len(pts) < 2:
            continue
        x1, y1 = pts[-2]
        x2, y2 = pts[-1]
        dx, dy = x2 - x1, y2 - y1
        if dx * dx + dy * dy < 1e-8:
            continue
        out[gid] = math.atan2(dy, dx)
    return out
