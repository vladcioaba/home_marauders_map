from __future__ import annotations

from pathlib import Path

from marauders_map.house import (
    CameraConfig,
    Door,
    Floorplan,
    House,
    Room,
    dump_house,
    house_to_dict,
)


def _sample_house() -> House:
    return House(
        floorplan=Floorplan(size=(8.0, 5.0), scale=60.0),
        rooms=[
            Room(id="kitchen", name="Kitchen", rect=(0.0, 0.0, 4.0, 3.0)),
            Room(id="hall", name="Hall", rect=(4.0, 0.0, 4.0, 3.0)),
        ],
        doors=[Door(between=("kitchen", "hall"),
                    segment=((4.0, 1.0), (4.0, 2.0)))],
        cameras=[CameraConfig(
            id="cam0", name="entry-cam", source="rtsp://10.0.0.5/stream",
            position=(2.0, 1.5), heading=45.0, fov=80.0,
        )],
    )


def test_house_to_dict_shape():
    d = house_to_dict(_sample_house())
    assert d["floorplan"]["size"] == [8.0, 5.0]
    assert d["rooms"][0]["id"] == "kitchen"
    assert d["doors"][0]["between"] == ["kitchen", "hall"]
    assert d["cameras"][0]["source"] == "rtsp://10.0.0.5/stream"


def test_dump_house_round_trip(tmp_path: Path):
    p = tmp_path / "out.yaml"
    h = _sample_house()
    dump_house(h, p)

    loaded = House.load(p)
    assert loaded.floorplan.size == h.floorplan.size
    assert loaded.floorplan.scale == h.floorplan.scale
    assert [r.rect for r in loaded.rooms] == [r.rect for r in h.rooms]
    assert [r.id for r in loaded.rooms] == [r.id for r in h.rooms]
    assert loaded.doors[0].segment == h.doors[0].segment
    assert loaded.cameras[0].source == "rtsp://10.0.0.5/stream"
    assert loaded.cameras[0].heading == 45.0
    assert loaded.cameras[0].fov == 80.0


def test_dump_house_creates_parent_dir(tmp_path: Path):
    p = tmp_path / "nested" / "deep" / "out.yaml"
    dump_house(_sample_house(), p)
    assert p.exists()
