from __future__ import annotations

import numpy as np

from marauders_map.tracker_global import GlobalTracker, PersonMarker


def test_first_marker_spawns_global_id():
    gt = GlobalTracker()
    out = gt.update(0.0, [PersonMarker("cam0", 1, (1.0, 1.0))])
    assert len(out) == 1
    assert out[0].global_id == 1


def test_sticky_mapping_keeps_same_gid():
    gt = GlobalTracker()
    a = gt.update(0.0, [PersonMarker("cam0", 1, (1.0, 1.0))])
    b = gt.update(0.1, [PersonMarker("cam0", 1, (1.05, 1.05))])
    assert a[0].global_id == b[0].global_id == 1


def test_two_distinct_people_get_distinct_ids():
    gt = GlobalTracker()
    out = gt.update(0.0, [
        PersonMarker("cam0", 1, (0.0, 0.0)),
        PersonMarker("cam0", 2, (5.0, 5.0)),
    ])
    ids = sorted(m.global_id for m in out)
    assert ids == [1, 2]


def test_handoff_across_cameras_preserves_gid():
    """cam0 sees a person, then cam1 picks it up at the same floor location."""
    gt = GlobalTracker(max_distance_m=1.0, max_gap_s=2.0)
    a = gt.update(0.0, [PersonMarker("cam0", 1, (3.0, 4.0))])
    # Brief gap; reappear on a different camera near the same point.
    b = gt.update(0.5, [PersonMarker("cam1", 9, (3.2, 4.1))])
    assert a[0].global_id == b[0].global_id


def test_far_marker_spawns_new_gid():
    gt = GlobalTracker(max_distance_m=1.0)
    a = gt.update(0.0, [PersonMarker("cam0", 1, (0.0, 0.0))])
    b = gt.update(0.5, [PersonMarker("cam1", 1, (10.0, 10.0))])
    assert a[0].global_id != b[0].global_id


def test_expire_drops_old_tracks():
    gt = GlobalTracker(max_distance_m=1.0, max_gap_s=0.5, expire_s=1.0)
    gt.update(0.0, [PersonMarker("cam0", 1, (0.0, 0.0))])
    # Long gap — old track expires; reappearance gets a fresh gid.
    later = gt.update(5.0, [PersonMarker("cam1", 2, (0.0, 0.0))])
    assert later[0].global_id == 2  # not 1


def test_appearance_breaks_spatial_tie():
    """When two existing tracks are equidistant, ReID picks the closer color."""
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    blue = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    gt = GlobalTracker(max_distance_m=2.0, use_appearance=True)
    a = gt.update(0.0, [
        PersonMarker("cam0", 1, (0.0, 0.0), red),
        PersonMarker("cam0", 2, (1.0, 0.0), blue),
    ])
    red_gid = a[0].global_id
    blue_gid = a[1].global_id
    assert red_gid != blue_gid

    # New marker midway between them, but visually red — should match the red track.
    b = gt.update(0.2, [PersonMarker("cam1", 99, (0.5, 0.0), red)])
    assert b[0].global_id == red_gid


def test_trails_returns_recent_history():
    gt = GlobalTracker(trail_s=10.0, max_gap_s=10.0)
    for i in range(5):
        gt.update(float(i) * 0.1, [PersonMarker("cam0", 1, (float(i), 0.0))])
    trails = gt.trails(0.5)
    assert 1 in trails
    assert len(trails[1]) >= 2
