"""Loader (and writer) for house layout: rooms, doors, cameras, calibration."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from marauders_map.geometry import compute_homography


@dataclass
class Floorplan:
    size: tuple[float, float]  # (width, height) in meters
    scale: float               # pixels per meter for rendering


@dataclass
class Room:
    id: str
    name: str
    rect: tuple[float, float, float, float]  # (x, y, w, h) in meters


@dataclass
class Door:
    between: tuple[str, str]
    segment: tuple[tuple[float, float], tuple[float, float]]


@dataclass
class Calibration:
    image_points: np.ndarray | None = None  # (N, 2) pixel coords
    floor_points: np.ndarray | None = None  # (N, 2) meter coords
    homography: np.ndarray | None = None    # (3, 3)


@dataclass
class CameraConfig:
    id: str
    name: str
    source: int | str
    position: tuple[float, float] = (0.0, 0.0)
    heading: float = 0.0
    fov: float = 90.0
    calibration: Calibration = field(default_factory=Calibration)


@dataclass
class House:
    floorplan: Floorplan
    rooms: list[Room]
    doors: list[Door]
    cameras: list[CameraConfig]

    def room_at(self, xy: tuple[float, float]) -> str | None:
        """Point-in-rectangle lookup — returns room id or None."""
        x, y = xy
        for room in self.rooms:
            rx, ry, rw, rh = room.rect
            if rx <= x < rx + rw and ry <= y < ry + rh:
                return room.id
        return None

    @classmethod
    def load(
        cls,
        house_path: str | Path,
        calibration_path: str | Path | None = None,
    ) -> "House":
        data = yaml.safe_load(Path(house_path).read_text()) or {}

        fp = data.get("floorplan") or {}
        floorplan = Floorplan(
            size=tuple(fp.get("size", [10.0, 6.0])),
            scale=float(fp.get("scale", 60)),
        )
        rooms = [
            Room(id=r["id"], name=r.get("name", r["id"]), rect=tuple(r["rect"]))
            for r in (data.get("rooms") or [])
        ]
        doors = [
            Door(
                between=tuple(d["between"]),
                segment=(tuple(d["segment"][0]), tuple(d["segment"][1])),
            )
            for d in (data.get("doors") or [])
        ]
        cameras = []
        for c in (data.get("cameras") or []):
            cameras.append(CameraConfig(
                id=c["id"],
                name=c.get("name", c["id"]),
                source=c["source"],
                position=tuple(c.get("position", [0.0, 0.0])),
                heading=float(c.get("heading", 0.0)),
                fov=float(c.get("fov", 90.0)),
            ))
        if not cameras:
            raise ValueError(f"{house_path}: no cameras defined")

        house = cls(floorplan=floorplan, rooms=rooms, doors=doors, cameras=cameras)

        if calibration_path is not None:
            cal_path = Path(calibration_path)
            if cal_path.exists():
                house._apply_calibration(cal_path)

        return house

    def _apply_calibration(self, path: Path) -> None:
        data = yaml.safe_load(path.read_text()) or {}
        by_id = {c.id: c for c in self.cameras}
        for cam_id, entry in (data.get("cameras") or {}).items():
            cam = by_id.get(cam_id)
            if cam is None:
                continue
            img = np.asarray(entry.get("image") or [], dtype=np.float32)
            flr = np.asarray(entry.get("floor") or [], dtype=np.float32)
            if img.ndim != 2 or img.shape != flr.shape or img.shape[0] < 4:
                continue
            cam.calibration = Calibration(
                image_points=img,
                floor_points=flr,
                homography=compute_homography(img, flr),
            )


def house_to_dict(house: House) -> dict[str, Any]:
    """Serializable view of a House — same shape that House.load() consumes."""
    return {
        "floorplan": {
            "size": list(house.floorplan.size),
            "scale": house.floorplan.scale,
        },
        "rooms": [
            {"id": r.id, "name": r.name, "rect": list(r.rect)}
            for r in house.rooms
        ],
        "doors": [
            {
                "between": list(d.between),
                "segment": [list(d.segment[0]), list(d.segment[1])],
            }
            for d in house.doors
        ],
        "cameras": [
            {
                "id": c.id,
                "name": c.name,
                "source": c.source,
                "position": list(c.position),
                "heading": c.heading,
                "fov": c.fov,
            }
            for c in house.cameras
        ],
    }


def dump_house(house: House, path: str | Path) -> None:
    """Write `house` as YAML to `path` in the shape House.load() expects."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(house_to_dict(house), sort_keys=False))
