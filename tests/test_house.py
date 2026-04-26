from __future__ import annotations

from pathlib import Path

import pytest

from marauders_map.house import (
    CameraConfig, Door, Floorplan, House, Room,
)


@pytest.fixture
def small_house():
    return House(
        floorplan=Floorplan(size=(10.0, 6.0), scale=60.0),
        rooms=[
            Room(id="kitchen", name="Kitchen", rect=(0.0, 0.0, 4.0, 3.0)),
            Room(id="hall", name="Hall", rect=(4.0, 0.0, 6.0, 3.0)),
        ],
        doors=[Door(between=("kitchen", "hall"),
                    segment=((4.0, 1.0), (4.0, 2.0)))],
        cameras=[CameraConfig(id="cam0", name="webcam", source=0)],
    )


def test_room_at_inside(small_house):
    assert small_house.room_at((1.0, 1.0)) == "kitchen"
    assert small_house.room_at((5.0, 1.0)) == "hall"


def test_room_at_outside(small_house):
    assert small_house.room_at((100.0, 100.0)) is None


def test_room_at_boundary_uses_half_open_rect(small_house):
    # Right edge of kitchen is x=4 → not inside kitchen, inside hall.
    assert small_house.room_at((4.0, 1.0)) == "hall"


def test_house_load_from_yaml(tmp_path: Path):
    yaml_text = """
floorplan:
  size: [8.0, 5.0]
  scale: 50
rooms:
  - id: living
    name: Living Room
    rect: [0, 0, 8, 5]
cameras:
  - id: cam0
    name: webcam
    source: 0
    position: [4, 2.5]
    heading: 90
    fov: 80
"""
    p = tmp_path / "house.yaml"
    p.write_text(yaml_text)
    h = House.load(p)
    assert h.floorplan.size == (8.0, 5.0)
    assert len(h.rooms) == 1 and h.rooms[0].id == "living"
    assert h.cameras[0].fov == 80.0
    assert h.cameras[0].calibration.homography is None


def test_house_load_no_cameras_raises(tmp_path: Path):
    yaml_text = "floorplan:\n  size: [4.0, 3.0]\nrooms: []\ncameras: []\n"
    p = tmp_path / "empty.yaml"
    p.write_text(yaml_text)
    with pytest.raises(ValueError, match="no cameras"):
        House.load(p)
