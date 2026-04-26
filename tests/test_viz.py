from __future__ import annotations

import numpy as np

from marauders_map.house import CameraConfig, Floorplan, House, Room
from marauders_map.tracker_global import GlobalMarker
from marauders_map.viz import (
    PARCHMENT,
    draw_floorplan,
    draw_markers,
    draw_mischief_managed,
    draw_trails,
    headings_from_trails,
)


def _house():
    return House(
        floorplan=Floorplan(size=(6.0, 4.0), scale=60.0),
        # Inset the room so (0,0) is empty parchment — not on any drawn line.
        rooms=[Room(id="r1", name="r1", rect=(1.0, 1.0, 4.0, 2.0))],
        doors=[],
        cameras=[CameraConfig(id="cam0", name="cam0", source=0,
                              position=(3.0, 2.0), heading=0.0, fov=90.0)],
    )


def test_draw_floorplan_returns_parchment_image():
    img = draw_floorplan(_house())
    h, w = img.shape[:2]
    assert (h, w) == (240, 360)  # 4m * 60, 6m * 60
    # Top-left pixel sits outside any drawn room/door — should be parchment.
    np.testing.assert_array_equal(img[0, 0], np.asarray(PARCHMENT, dtype=np.uint8))


def test_draw_trails_no_crash_on_empty():
    h = _house()
    img = draw_floorplan(h)
    draw_trails(img, h, {})  # nothing to draw — should not raise


def test_draw_trails_short_trail_skipped():
    h = _house()
    img = draw_floorplan(h)
    draw_trails(img, h, {1: [(1.0, 1.0)]})  # only one point


def test_draw_trails_longer_trail_modifies_image():
    h = _house()
    base = draw_floorplan(h)
    drawn = base.copy()
    pts = [(1.5 + 0.3 * i, 1.5) for i in range(8)]
    draw_trails(drawn, h, {1: pts})
    assert not np.array_equal(base, drawn)


def test_draw_markers_writes_pixels():
    h = _house()
    base = draw_floorplan(h)
    drawn = base.copy()
    draw_markers(drawn, h, [GlobalMarker(1, "cam0", 1, (2.5, 2.0))],
                 names={1: "Vlad"})
    assert not np.array_equal(base, drawn)


def test_headings_from_trails_basic():
    out = headings_from_trails({1: [(0.0, 0.0), (1.0, 0.0)]})
    assert 1 in out
    assert abs(out[1]) < 1e-6  # heading along +x is 0 rad


def test_headings_skips_short_or_static():
    assert headings_from_trails({1: [(0.0, 0.0)]}) == {}
    assert headings_from_trails({1: [(1.0, 1.0), (1.0, 1.0)]}) == {}


def test_mischief_managed_overlays_text():
    h = _house()
    img = draw_floorplan(h)
    base = img.copy()
    draw_mischief_managed(img)
    assert not np.array_equal(base, img)
